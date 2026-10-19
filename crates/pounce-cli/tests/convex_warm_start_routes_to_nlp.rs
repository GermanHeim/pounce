//! A requested warm start is not silently discarded by the convex arm.
//!
//! `solver_selection=auto` sends an LP / convex QP to `pounce-convex`, whose
//! dispatch re-parses the `.nl` into a `QpProblem` and starts cold: it reads
//! neither `x0`, the `d` segment, nor the `ipopt_zL_in` / `ipopt_zU_in`
//! suffixes. With `warm_start_init_point=yes max_iter=0` it therefore wrote
//! `x = 0` and zero duals to the `.sol` instead of the supplied point. Under
//! `auto` the model now goes to the NLP interior point, which honors the
//! request; a forced convex engine keeps its route and says it ignores it.
//!
//! Model: the convex QP `min (x0-2)^2 + (x1-2)^2 + (x2+1)^2  s.t.
//! x0 + x1 <= 2,  x2 >= 0`, started at `x = (0.3, 0.7, 0.4)` with an AMPL
//! dual of -1.5 and `ipopt_zL_in(x2) = 0.25`.

use std::path::PathBuf;
use std::process::{Command, Output};

const NL: &str = "g3 1 1 0
 3 1 1 0 0
 0 1
 0 0
 0 3 0
 0 0 0 1
 0 0 0 0 0
 2 3
 0 0
 0 0 0 0 0
C0
n0
O0 0
o54
3
o5
o0
v0
n-2
n2
o5
o0
v1
n-2
n2
o5
o0
v2
n1
n2
d1
0 -1.5
x3
0 0.3
1 0.7
2 0.4
r
1 2
b
3
3
2 0
k2
1
2
J0 2
0 1
1 1
G0 3
0 0
1 0
2 0
S4 1 ipopt_zL_in
2 0.25
";

fn run(tag: &str, opts: &[&str]) -> (Output, String) {
    let dir = std::env::temp_dir().join(format!("pounce_convex_warm_{tag}"));
    std::fs::create_dir_all(&dir).expect("mkdir");
    let nl = dir.join("q.nl");
    std::fs::write(&nl, NL).expect("write nl");
    let sol = dir.join("q.sol");
    let _ = std::fs::remove_file(&sol);
    let out = Command::new(PathBuf::from(env!("CARGO_BIN_EXE_pounce")))
        .arg(&nl)
        .arg("-AMPL")
        .args(opts)
        .output()
        .expect("spawn pounce");
    let text = std::fs::read_to_string(&sol).unwrap_or_else(|e| {
        panic!(
            "{tag}: no .sol ({e}); stderr {}",
            String::from_utf8_lossy(&out.stderr)
        )
    });
    (out, text)
}

/// `(dual, x, zL_out(x2))` from a `.sol`.
fn parse(text: &str) -> (f64, Vec<f64>, f64) {
    let lines: Vec<&str> = text.lines().collect();
    let o = lines.iter().position(|l| l.trim() == "Options").unwrap();
    let k: usize = lines[o + 1].trim().parse().unwrap();
    let c = o + 2 + k;
    let num = |i: usize| -> f64 { lines[i].trim().parse().unwrap() };
    let zl = lines
        .iter()
        .position(|l| l.trim() == "ipopt_zL_out")
        .and_then(|at| {
            lines[at + 1..]
                .iter()
                .take_while(|l| !l.starts_with("suffix"))
                .find_map(|l| {
                    let mut it = l.split_whitespace();
                    (it.next() == Some("2")).then(|| it.next().unwrap().parse().unwrap())
                })
        })
        .unwrap_or(0.0);
    (num(c + 4), (0..3).map(|i| num(c + 5 + i)).collect(), zl)
}

const WARM: [&str; 3] = [
    "warm_start_init_point=yes",
    "warm_start_recentering=none",
    "max_iter=0",
];

#[test]
fn auto_routes_a_warm_start_to_the_nlp_arm_and_honors_it() {
    let (out, text) = run("auto", &WARM);
    let (dual, x, zl) = parse(&text);
    for (got, want) in x.iter().zip([0.3, 0.7, 0.4]) {
        assert!(
            (got - want).abs() < 1e-6,
            "x {x:?}, supplied (0.3, 0.7, 0.4)"
        );
    }
    assert!((dual + 1.5).abs() < 1e-12, "dual {dual}, supplied -1.5");
    assert!((zl - 0.25).abs() < 1e-12, "zL_out(x2) {zl}, supplied 0.25");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("routing to the general NLP interior-point path so the warm start"),
        "the reroute is announced: {stderr}"
    );
}

#[test]
fn a_cold_start_keeps_the_convex_arm() {
    let (out, _) = run("cold", &[]);
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("Selected solver: convex QP interior-point"),
        "no warm start, no reroute: {stdout}"
    );
}

#[test]
fn a_forced_convex_engine_says_it_ignores_the_warm_start() {
    for sel in ["qp-ipm", "qp-active-set", "socp"] {
        let opt = format!("solver_selection={sel}");
        let (out, _) = run(sel, &[&opt, WARM[0]]);
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(
            stderr.contains("the warm start will be ignored"),
            "{sel}: {stderr}"
        );
    }
}
