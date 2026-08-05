//! End-to-end guard for gh#483: the `.nl` `scaling_factor` suffixes are
//! the AMPL/Pyomo channel for `nlp_scaling_method=user-scaling`, and
//! pounce used to read none of them.
//!
//! Before this, `NlTnlp` did not implement `get_scaling_parameters` at
//! all, so the default `TNLP` impl answered "no scaling supplied": a
//! Pyomo user who tagged `model.scaling_factor` and set
//! `nlp_scaling_method=user-scaling` — the workflow that works with
//! Ipopt through ASL — got no scaling and no message saying so. The
//! option was accepted and meant "none".
//!
//! Fixtures (both written by Pyomo's NL writer, so the suffix layout is
//! the real one, not a hand-rolled approximation):
//!
//! ```text
//! min (x1 - 2)^4 + (x2 - 3)^2   s.t.  x1*x2 >= 1,  x1 - x2 == 0.5
//! scaling_factor[obj] = 100,  scaling_factor[c1] = 10
//! ```
//!
//! * `user_scaling_suffix.nl` — objective + constraint factors only.
//! * `user_scaling_var_suffix.nl` — the same, plus
//!   `scaling_factor[x1] = 3`, a per-variable factor pounce does not
//!   model.

use std::path::PathBuf;
use std::process::Command;

fn pounce_exe() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_pounce"))
}

fn fixture(name: &str) -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("tests");
    p.push("fixtures");
    p.push(name);
    p
}

/// Run pounce on a fixture with extra key=value options; returns
/// `(success, stdout, stderr)`. The `.nl` is copied into a per-test
/// scratch directory first so the `.sol` the solver writes next to it
/// cannot collide between concurrently running tests.
fn run(fixture_name: &str, tag: &str, opts: &[&str]) -> (bool, String, String) {
    let dir = std::env::temp_dir().join(format!("pounce_gh483_{tag}"));
    std::fs::create_dir_all(&dir).expect("scratch dir");
    let nl = dir.join(fixture_name);
    std::fs::copy(fixture(fixture_name), &nl).expect("copy fixture");

    let out = Command::new(pounce_exe())
        .arg(&nl)
        .args(opts)
        .output()
        .expect("run pounce");
    let _ = std::fs::remove_dir_all(&dir);
    (
        out.status.success(),
        String::from_utf8_lossy(&out.stdout).into_owned(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
    )
}

/// The `(scaled, unscaled)` objective pair from the end-of-run summary
/// block, which is exactly where "was the scaling applied" is visible.
fn objective_columns(stdout: &str) -> (f64, f64) {
    let line = stdout
        .lines()
        .find(|l| l.trim_start().starts_with("Objective."))
        .unwrap_or_else(|| panic!("no Objective summary line in:\n{stdout}"));
    let nums: Vec<f64> = line
        .split_whitespace()
        .filter_map(|t| t.parse::<f64>().ok())
        .collect();
    assert_eq!(nums.len(), 2, "expected scaled + unscaled columns: {line}");
    (nums[0], nums[1])
}

/// With `user-scaling`, the objective the IPM sees is the user's
/// factor times the model's — and the value handed back is still in
/// model units. Pre-fix both columns read the same, because nothing
/// ever read the suffix.
#[test]
fn scaling_factor_suffix_is_applied() {
    let (ok, stdout, stderr) = run(
        "user_scaling_suffix.nl",
        "applied",
        &["nlp_scaling_method=user-scaling"],
    );
    assert!(ok, "solve failed\nstdout:\n{stdout}\nstderr:\n{stderr}");
    let (scaled, unscaled) = objective_columns(&stdout);
    assert!(
        (scaled / unscaled - 100.0).abs() < 1e-6,
        "scaling_factor[obj]=100 should scale the IPM's objective; \
         got scaled={scaled}, unscaled={unscaled}",
    );
}

/// The same file without the option is untouched: the suffix alone
/// changes nothing, matching Ipopt (and every model that carries a
/// `scaling_factor` Suffix for `core.scale_model` rather than for the
/// solver).
#[test]
fn scaling_factor_suffix_is_inert_without_the_option() {
    let (ok, stdout, stderr) = run("user_scaling_suffix.nl", "inert", &[]);
    assert!(ok, "solve failed\nstdout:\n{stdout}\nstderr:\n{stderr}");
    let (scaled, unscaled) = objective_columns(&stdout);
    assert!(
        (scaled - unscaled).abs() < 1e-9,
        "default scaling should leave this objective alone; \
         got scaled={scaled}, unscaled={unscaled}",
    );
}

/// A per-variable factor is refused with an explanation, not applied
/// in part. Solving with the objective/constraint factors and dropping
/// the variable one answers a question nobody asked.
#[test]
fn variable_scaling_factor_is_refused_with_a_message() {
    let (ok, stdout, stderr) = run(
        "user_scaling_var_suffix.nl",
        "varfactor",
        &["nlp_scaling_method=user-scaling"],
    );
    assert!(!ok, "expected a failing exit\nstdout:\n{stdout}");
    assert!(
        stderr.contains("per-variable scaling factors"),
        "the refusal must say what was wrong; stderr:\n{stderr}",
    );
    assert!(
        stderr.contains("483"),
        "the refusal should point at the tracking issue; stderr:\n{stderr}",
    );
}

/// Without `user-scaling` the variable entry is not a request pounce
/// is dropping, so the solve runs normally.
#[test]
fn variable_scaling_factor_is_inert_without_the_option() {
    let (ok, stdout, stderr) = run("user_scaling_var_suffix.nl", "varinert", &[]);
    assert!(ok, "solve failed\nstdout:\n{stdout}\nstderr:\n{stderr}");
}
