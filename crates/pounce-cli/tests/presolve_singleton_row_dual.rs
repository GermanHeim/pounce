//! Presolve reports a singleton row's dual on the row, not on the bound it
//! made from it (M24).
//!
//! Phase 1 turns `C1: x0 <= 0.5` into the variable bound `x0 <= 0.5`, Phase 2
//! then drops the row as redundant, and the solver's multiplier for the bound
//! used to come out as `ipopt_zU_out` on `x0` — a variable the model gives no
//! upper bound — while the row's dual read 0. `presolve=no` and the convex
//! arms report the row dual (-2 in the `.sol`) and nothing on `x0`.
//!
//! Model: `min (x0-2)^2 + (x1-2)^2 + (x2+1)^2  s.t.  C0: x0 + x1 <= 2,
//! C1: <row on x0>,  x2 >= 0`, solved with `presolve=yes` and `presolve=no` on
//! the NLP interior point and the active-set SQP arm, and in both senses.

use std::path::PathBuf;
use std::process::Command;

/// `r1` is C1's `r` line, `a` its coefficient on `x0`, `b0` x0's `b` line.
fn nl(r1: &str, a: f64, b0: &str, maximize: bool, extra: &str) -> String {
    let (sense, neg) = if maximize { (1, "o16\n") } else { (0, "") };
    format!(
        "g3 1 1 0
 3 2 1 0 0
 0 1
 0 0
 0 3 0
 0 0 0 1
 0 0 0 0 0
 3 3
 0 0
 0 0 0 0 0
C0
n0
C1
n0
O0 {sense}
{neg}o54
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
x3
0 0.1
1 0.1
2 0.1
r
1 2
{r1}
b
{b0}
3
2 0
k2
2
3
J0 2
0 1
1 1
J1 1
0 {a}
G0 3
0 0
1 0
2 0
{extra}"
    )
}

#[derive(Debug)]
struct Sol {
    duals: Vec<f64>,
    x: Vec<f64>,
    zl: Vec<(usize, f64)>,
    zu: Vec<(usize, f64)>,
}

fn solve(tag: &str, body: &str, opts: &[&str]) -> Sol {
    let dir = std::env::temp_dir().join(format!("pounce_presolve_singleton_dual_{tag}"));
    std::fs::create_dir_all(&dir).expect("mkdir");
    let nl = dir.join("m.nl");
    std::fs::write(&nl, body).expect("write nl");
    let sol = dir.join("m.sol");
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
    let lines: Vec<&str> = text.lines().collect();
    let o = lines.iter().position(|l| l.trim() == "Options").unwrap();
    let k: usize = lines[o + 1].trim().parse().unwrap();
    let c = o + 2 + k;
    let num = |i: usize| -> f64 { lines[i].trim().parse().unwrap() };
    let (m, n): (usize, usize) = (
        lines[c].trim().parse().unwrap(),
        lines[c + 2].trim().parse().unwrap(),
    );
    let block = |name: &str| -> Vec<(usize, f64)> {
        let Some(at) = lines.iter().position(|l| l.trim() == name) else {
            return Vec::new();
        };
        lines[at + 1..]
            .iter()
            .take_while(|l| !l.starts_with("suffix"))
            .map(|l| {
                let mut it = l.split_whitespace();
                (
                    it.next().unwrap().parse().unwrap(),
                    it.next().unwrap().parse().unwrap(),
                )
            })
            .collect()
    };
    Sol {
        duals: (0..m).map(|i| num(c + 4 + i)).collect(),
        x: (0..n).map(|i| num(c + 4 + m + i)).collect(),
        zl: block("ipopt_zL_out"),
        zu: block("ipopt_zU_out"),
    }
}

fn entry(v: &[(usize, f64)], j: usize) -> f64 {
    v.iter().find(|(i, _)| *i == j).map_or(0.0, |&(_, z)| z)
}

fn assert_same(tag: &str, a: &Sol, b: &Sol, tol: f64) {
    let close = |p: f64, q: f64| (p - q).abs() <= tol;
    let ok = a.duals.len() == b.duals.len()
        && a.duals.iter().zip(&b.duals).all(|(p, q)| close(*p, *q))
        && a.x.iter().zip(&b.x).all(|(p, q)| close(*p, *q))
        && (0..a.x.len()).all(|j| {
            close(entry(&a.zl, j), entry(&b.zl, j)) && close(entry(&a.zu, j), entry(&b.zu, j))
        });
    assert!(ok, "{tag}: presolve=yes {a:?}\n   vs presolve=no {b:?}");
}

const ARMS: [(&str, &[&str]); 2] = [
    ("nlp", &["solver_selection=nlp"]),
    ("sqp", &["solver_selection=nlp", "algorithm=active-set-sqp"]),
];

#[test]
fn a_singleton_rows_dual_stays_on_the_row() {
    // (name, C1's r line, coefficient, x0's b line, expected C1 .sol dual)
    let cases = [
        ("le", "1 0.5", 1.0, "3", -2.0),
        ("ge_neg", "2 -0.5", -1.0, "3", 2.0),
        ("le_scaled", "1 1.5", 3.0, "3", -2.0 / 3.0),
        ("ge_neg_scaled", "2 -1.0", -2.0, "3", 1.0),
        ("declared_looser", "1 0.5", 1.0, "1 1.0", -2.0),
        ("declared_tighter", "1 0.5", 1.0, "1 0.3", 0.0),
        ("range", "0 -5 0.5", 1.0, "3", -2.0),
    ];
    for (name, r1, a, b0, want) in cases {
        for maximize in [false, true] {
            let sign = if maximize { -1.0 } else { 1.0 };
            let body = nl(r1, a, b0, maximize, "");
            for (arm, opts) in ARMS {
                let tag = format!("{name}_{}_{arm}", if maximize { "max" } else { "min" });
                let on = solve(
                    &format!("{tag}_on"),
                    &body,
                    &[opts, &["presolve=yes"]].concat(),
                );
                let off = solve(
                    &format!("{tag}_off"),
                    &body,
                    &[opts, &["presolve=no"]].concat(),
                );
                assert!(
                    (on.duals[1] - sign * want).abs() < 1e-6,
                    "{tag}: C1 dual {} with presolve=yes, expected {}",
                    on.duals[1],
                    sign * want
                );
                assert_same(&tag, &on, &off, 1e-6);
            }
        }
    }
}

/// The inverse: a `.sol` written with presolve on, fed back as a warm start
/// with zero iterations, hands the row dual back unchanged — through the
/// bound presolve made from the row and out again.
#[test]
fn a_singleton_rows_dual_round_trips_through_a_warm_start() {
    for (name, r1, a) in [("le", "1 0.5", 1.0), ("ge_neg", "2 -0.5", -1.0)] {
        let opts = ["solver_selection=nlp", "presolve=yes"];
        let full = solve(
            &format!("rt_{name}_full"),
            &nl(r1, a, "3", false, ""),
            &opts,
        );
        let mut extra = format!("d2\n0 {:?}\n1 {:?}\n", full.duals[0], full.duals[1]);
        for (sfx, blk) in [("ipopt_zL_in", &full.zl), ("ipopt_zU_in", &full.zu)] {
            if !blk.is_empty() {
                extra += &format!("S4 {} {sfx}\n", blk.len());
                for (j, z) in blk.iter() {
                    extra += &format!("{j} {z:?}\n");
                }
            }
        }
        let body = nl(r1, a, "3", false, &extra).replace(
            "x3\n0 0.1\n1 0.1\n2 0.1\n",
            &format!(
                "x3\n0 {:?}\n1 {:?}\n2 {:?}\n",
                full.x[0], full.x[1], full.x[2]
            ),
        );
        let warm = solve(
            &format!("rt_{name}_warm"),
            &body,
            &[
                &opts[..],
                &[
                    "warm_start_init_point=yes",
                    "warm_start_recentering=none",
                    "max_iter=0",
                ],
            ]
            .concat(),
        );
        assert!(
            (warm.duals[1] - full.duals[1]).abs() < 1e-9,
            "{name}: C1 dual {} after zero warm iterations, supplied {}",
            warm.duals[1],
            full.duals[1]
        );
        assert_eq!(entry(&warm.zu, 0), 0.0, "{name}: {:?}", warm.zu);
        assert_eq!(entry(&warm.zl, 0), 0.0, "{name}: {:?}", warm.zl);
    }
}
