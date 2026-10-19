//! Maximize-model sign checks on two CLI reports that read multipliers back
//! in a fixed frame, found auditing the boundaries next to gh#1033.
//!
//! * `pounce verify` compared the `.sol` bound blocks — `sigma*z_L` /
//!   `-sigma*z_U` since gh#959 — against the gradient of the objective as
//!   `NlTnlp` evaluates it, negated for a maximize model, so a correct
//!   maximize `.sol` read as dual infeasibility `2 (z_L - z_U)` = 4.0.
//! * `--minima` wrote the best minimum's JSON `lambda` without the sense
//!   factor its own `.sol` and a plain solve's report apply, and left
//!   `problem.minimize` at `true`.
//!
//! The model is `min/max ±((x0-2)^2 + (x1-2)^2 + (x2+1)^2)` subject to
//! `x0 + x1 <= 2`, `0 <= x0 <= 0.5`, `x2 >= 0`: an active row, an active
//! upper bound and an active lower bound, all with nonzero multipliers.

use std::path::{Path, PathBuf};
use std::process::Command;

fn pounce_exe() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_pounce"))
}

fn nl_text(maximize: bool) -> String {
    let head = if maximize { "O0 1\no16\n" } else { "O0 0\n" };
    format!(
        "g3 1 1 0
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
{head}o54
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
b
0 0 0.5
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
"
    )
}

fn setup(tag: &str, maximize: bool) -> (PathBuf, PathBuf) {
    let dir = std::env::temp_dir().join(format!("pounce_max_sign_{tag}"));
    std::fs::create_dir_all(&dir).expect("mkdir");
    let nl = dir.join("q.nl");
    std::fs::write(&nl, nl_text(maximize)).expect("write nl");
    (dir, nl)
}

fn json(path: &Path) -> serde_json::Value {
    serde_json::from_str(&std::fs::read_to_string(path).expect("read json")).expect("parse json")
}

#[test]
fn verify_accepts_a_correct_maximize_sol() {
    for (tag, maximize) in [("verify_min", false), ("verify_max", true)] {
        let (dir, nl) = setup(tag, maximize);
        let sol = dir.join("q.sol");
        let out = Command::new(pounce_exe())
            .arg(&nl)
            .arg(&sol)
            .arg("solver_selection=nlp")
            .output()
            .expect("spawn solve");
        assert!(out.status.success(), "{tag}: solve failed");
        let receipt = dir.join("receipt.json");
        let out = Command::new(pounce_exe())
            .args(["verify", "--require-optimal", "--json-output"])
            .arg(&receipt)
            .arg(&nl)
            .arg(&sol)
            .output()
            .expect("spawn verify");
        assert!(
            out.status.success(),
            "{tag}: verify rejected pounce's own optimal .sol:\n{}",
            String::from_utf8_lossy(&out.stdout)
        );
    }
}

#[test]
fn minima_json_lambda_and_sense_match_a_plain_solve() {
    for (tag, maximize) in [("minima_min", false), ("minima_max", true)] {
        let (dir, nl) = setup(tag, maximize);
        let plain = dir.join("plain.json");
        let st = Command::new(pounce_exe())
            .arg(&nl)
            .arg(dir.join("plain.sol"))
            .arg("--json-output")
            .arg(&plain)
            .arg("solver_selection=nlp")
            .status()
            .expect("spawn plain");
        assert!(st.success(), "{tag}: plain solve failed");
        let mm = dir.join("minima.json");
        let st = Command::new(pounce_exe())
            .arg(&nl)
            .arg(dir.join("minima.sol"))
            .args([
                "--minima",
                "multistart",
                "--n-minima",
                "2",
                "--max-solves",
                "4",
            ])
            .arg("--json-output")
            .arg(&mm)
            .status()
            .expect("spawn minima");
        assert!(st.success(), "{tag}: --minima failed");
        let (a, b) = (json(&plain), json(&mm));
        assert_eq!(
            b["problem"]["minimize"], a["problem"]["minimize"],
            "{tag}: --minima reports a different objective sense"
        );
        assert_eq!(a["problem"]["minimize"], serde_json::json!(!maximize));
        let la = a["solution"]["lambda"][0].as_f64().expect("plain lambda");
        let lb = b["solution"]["lambda"][0].as_f64().expect("minima lambda");
        assert!(
            la.abs() > 0.5,
            "{tag}: the row must carry a real multiplier"
        );
        assert!(
            (la - lb).abs() < 1e-5,
            "{tag}: --minima JSON lambda {lb} vs plain solve {la}"
        );
    }
}
