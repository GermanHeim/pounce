//! gh#1000 (II.22) — the gh#983 item-5 downgrade does not fire on a
//! stationarity residual at the rounding floor of its own terms.
//!
//! Both fixtures are run the way discopt runs an incumbent NLP: the NLP arm,
//! `bound_relax_factor = 0`, `constr_viol_tol = 1e-8`. Without the relaxation
//! an active bound and a row become linearly dependent, the multipliers drift
//! along the unbounded ray of valid ones, and the run carries the gh#884
//! runaway-multiplier signature.
//!
//! * `pump_network_h148_brf0` (the optimization book's pump network, NLP(y)
//!   at `H0_max = 148`): residual `5.5e-3` against terms of `1.57e13` — 1.6
//!   ulps — with the objective right to `5e-9` relative (the convex QCQP arm
//!   finds `130209.66232452`). It was downgraded on the `5.5e-3` alone.
//! * `haverly_multistart_t3_brf0` (Haverly pooling, multistart trial 3): the
//!   control. Residual `9.4e-3` against `6.6e10` (140 ulps), objective `7e-8`
//!   off and complementarity `1.8e-6`; the downgrade is right and stays.

use std::path::PathBuf;
use std::process::Command;

fn run(name: &str) -> String {
    let mut fx = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    fx.push("tests");
    fx.push("fixtures");
    fx.push(name);
    let out = Command::new(env!("CARGO_BIN_EXE_pounce"))
        .arg(fx)
        .args([
            "solver_selection=nlp",
            "bound_relax_factor=0",
            "constr_viol_tol=1e-8",
        ])
        .output()
        .expect("run pounce");
    String::from_utf8_lossy(&out.stdout).into_owned()
}

fn exit_line(out: &str) -> &str {
    out.lines()
        .rfind(|l| l.starts_with("EXIT:"))
        .unwrap_or_else(|| panic!("no EXIT line:\n{out}"))
}

#[test]
fn a_residual_at_the_roundoff_floor_keeps_its_success_verdict() {
    let out = run("pump_network_h148_brf0.nl");
    assert_eq!(exit_line(&out), "EXIT: Optimal Solution Found.", "{out}");
    assert!(!out.contains("status is downgraded"), "{out}");
    // The scale caveat still tells the user what the multipliers did.
    assert!(out.contains("WARNING: large_dual_scale:"), "{out}");
}

#[test]
fn a_residual_above_the_floor_is_still_downgraded() {
    let out = run("haverly_multistart_t3_brf0.nl");
    assert_eq!(
        exit_line(&out),
        "EXIT: Solved To Acceptable Level.",
        "{out}"
    );
    assert!(out.contains("status is downgraded"), "{out}");
}
