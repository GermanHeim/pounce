//! Regression test for gh#1033: the `.nl` `d` segment's initial duals are in
//! AMPL's sign (`d obj / d b`, the sign a `.sol` dual block carries), so the
//! reader must negate them into pounce's `L = f + lambda' g` multiplier just
//! as `sol_writer` negates on the way out. Before the fix a warm start with
//! `max_iter=0` handed a `d` entry of 5 back as -5; Ipopt's AMPL driver
//! gives back 5.
//!
//! The model is `min (x - 1)^2  s.t.  exp(x) >= 8,  0 <= x <= 10`, started
//! at x = 3. A full solve's AMPL dual is `2 (x* - 1) / exp(x*) ≈ 0.27 > 0`,
//! which pins the sign convention of the `.sol` side independently.

use std::path::PathBuf;
use std::process::Command;

fn pounce_exe() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_pounce"))
}

fn nl_text(dual: &str) -> String {
    format!(
        "g3 1 1 0
 1 1 1 0 0
 1 1
 0 0
 1 1 1
 0 0 0 1
 0 0 0 0 0
 1 1
 0 0
 0 0 0 0 0
C0
o44
v0
O0 0
o5
o0
v0
n-1
n2
d1
0 {dual}
x1
0 3.0
r
2 8
b
0 0 10
k0
J0 1
0 0
G0 1
0 0
"
    )
}

/// Solve with `-AMPL` and return the single constraint dual from the `.sol`.
fn sol_dual(tag: &str, dual: &str, opts: &[&str]) -> f64 {
    let dir = std::env::temp_dir().join(format!("pounce_issue_1033_{tag}"));
    std::fs::create_dir_all(&dir).expect("mkdir");
    let nl = dir.join("d.nl");
    std::fs::write(&nl, nl_text(dual)).expect("write nl");
    let sol = dir.join("d.sol");
    let _ = std::fs::remove_file(&sol);
    let out = Command::new(pounce_exe())
        .arg(&nl)
        .arg("-AMPL")
        .args(opts)
        .output()
        .expect("spawn pounce");
    let text = std::fs::read_to_string(&sol).unwrap_or_else(|e| {
        panic!(
            "no .sol written ({e}); stderr: {}",
            String::from_utf8_lossy(&out.stderr)
        )
    });
    // Layout: message, blank, "Options", option count k, k option lines,
    // then four counts (m, m, n, n), then the m duals.
    let lines: Vec<&str> = text.lines().collect();
    let o = lines
        .iter()
        .position(|l| l.trim() == "Options")
        .expect("Options line");
    let k: usize = lines[o + 1].trim().parse().expect("option count");
    let counts = o + 2 + k;
    let m: usize = lines[counts].trim().parse().expect("m");
    assert_eq!(m, 1, "one constraint dual expected");
    lines[counts + 4].trim().parse().expect("dual value")
}

#[test]
fn warm_start_round_trips_the_nl_initial_dual() {
    for (tag, d, want) in [("pos", "5.0", 5.0), ("neg", "-5.0", -5.0)] {
        let got = sol_dual(tag, d, &["warm_start_init_point=yes", "max_iter=0"]);
        assert!(
            (got - want).abs() < 1e-12,
            "d entry {d}: .sol dual after zero iterations is {got}, expected {want} \
             (a .nl initial dual and a .sol dual must share one sign)"
        );
    }
}

#[test]
fn full_solve_dual_is_in_ampl_sign() {
    // Independent anchor on the `.sol` side: d obj / d b > 0 here.
    let got = sol_dual("full", "0.0", &[]);
    let xs = 8f64.ln();
    let want = 2.0 * (xs - 1.0) / 8.0;
    assert!(
        (got - want).abs() < 1e-6,
        "full-solve AMPL dual {got}, expected {want}"
    );
}
