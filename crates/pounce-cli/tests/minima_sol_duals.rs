//! Regression test for issue #196 (related, Finding 2): the `--minima`
//! multistart writer must emit the *real* base-problem constraint duals at each
//! reported minimum, not a zero placeholder.
//!
//! `convex_qp.nl` is `min x0² + x1²  s.t.  x0 + x1 = 2`, optimum (1, 1) with the
//! equality multiplier λ = −2. A plain solve writes that dual; before the fix,
//! `--minima` wrote 0.0. The multistart archive can also accept points from
//! penalty/tunnel solves on an augmented objective, so the fix recovers duals
//! via a clean re-solve at each minimum — this test pins that the recovered
//! dual matches the true value (and is not the old zero placeholder).

use std::path::PathBuf;
use std::process::Command;

fn pounce_exe() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_pounce"))
}

fn fixture() -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("tests");
    p.push("fixtures");
    p.push("convex_qp.nl");
    p
}

fn best_lambda(args: &[&str]) -> Vec<f64> {
    // Sanitize the tag: an `=` (or other punctuation) in the path would make
    // the CLI misparse the .nl path token as a `key=value` option.
    let tag: String = args
        .join("_")
        .chars()
        .map(|c| if c.is_alphanumeric() { c } else { '_' })
        .collect();
    let dir = std::env::temp_dir().join(format!("pounce_minima_duals_{tag}"));
    std::fs::create_dir_all(&dir).expect("mkdir");
    let nl = dir.join("m.nl");
    std::fs::copy(fixture(), &nl).expect("copy fixture");
    let json = dir.join("out.json");
    let out = Command::new(pounce_exe())
        .arg(&nl)
        .args(args)
        .arg("--json-output")
        .arg(&json)
        .arg("--no-sol")
        .output()
        .expect("spawn pounce");
    assert_eq!(out.status.code(), Some(0), "solve should succeed");
    let report: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&json).expect("read json")).expect("parse");
    report["solution"]["lambda"]
        .as_array()
        .expect("lambda array")
        .iter()
        .map(|v| v.as_f64().expect("f64"))
        .collect()
}

#[test]
fn minima_sol_writes_real_duals_not_zeros() {
    // The general NLP path's dual is the reference value (λ = −2 for the
    // single equality).
    let reference = best_lambda(&["solver_selection=nlp"]);
    assert_eq!(reference.len(), 1);
    assert!(
        (reference[0] - (-2.0)).abs() < 1e-6,
        "reference equality dual should be ≈ −2; got {}",
        reference[0]
    );

    // --minima must now recover that same dual (was 0.0 before the fix).
    let minima = best_lambda(&["--minima", "multistart"]);
    assert_eq!(minima.len(), 1);
    assert!(
        minima[0].abs() > 1e-3,
        "the dual must not be the old zero placeholder; got {}",
        minima[0]
    );
    assert!(
        (minima[0] - reference[0]).abs() < 1e-5,
        "--minima dual {} should match the NLP-path dual {}",
        minima[0],
        reference[0]
    );
}
