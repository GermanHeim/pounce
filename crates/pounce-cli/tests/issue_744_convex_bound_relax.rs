//! The convex path must apply `bound_relax_factor`, like the NLP path does
//! (gh #744, gh #745).
//!
//! `OrigIpoptNlp::relax_bounds` widens the variable box and the
//! inequality-row bounds by `min(bound_relax_factor·…, constr_viol_tol)`
//! before the NLP algorithm ever sees them — Ipopt's default `1e-8`, capped
//! at `1e-4`. The LP/QP/SOCP extractors read the `.nl` bounds verbatim, so
//! the *same binary* on the *same file* solved two different models
//! depending on `solver_selection`.
//!
//! That is not a hairline difference on a constraint-degenerate model. On
//! `LISWET1` every one of the 10 000 monotonicity rows is active at the
//! optimum and the multipliers sum to `1.6e9`, so widening the rows by `1e-8`
//! buys `9.0` of objective: the convex arm returned the exact optimum
//! `36.1224` and the NLP arm (and the Ipopt-MA57 reference) the relaxed one,
//! `27.1221`. Nine Maros-Meszaros QPs and 68 of 371 LPs disagreed that way,
//! always one-signed, and it read as a wrong answer from `pounce-convex`.
//! It was not: both points are what their model asks for.
//!
//! The two fixtures here isolate the two widenings. Each is an LP whose
//! optimum sits exactly on a bound of magnitude `1e4`, where the relaxation
//! is `min(1e-8, 1e-4)·1e4 = 1e-4` — far above solver noise and far above
//! the ulp at `1e4`, so "did the widening happen" is unambiguous.
//!
//! * `bound_relax_row.nl` — `min x0` s.t. `x0 − x1 >= 1e4`, `x0, x1 >= 0`.
//!   The binding bound is an **inequality row**.
//! * `bound_relax_var.nl` — `min x0 + x1` s.t. `x0 + x1 >= 0`, `x0 >= 1e4`,
//!   `x1` fixed at `0`. The binding bound is a **variable bound**, and the
//!   fixed `x1` pins the rule that a fixed variable is never widened
//!   (upstream's default `fixed_variable_treatment=make_parameter` lifts it
//!   out of the problem before `relax_bounds` runs).

use std::path::PathBuf;
use std::process::Command;

/// Solve `fixture` and return the reported objective.
fn objective(tag: &str, fixture: &str, opts: &[&str]) -> f64 {
    let dir = std::env::temp_dir().join(format!("pounce_issue_744_{tag}"));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("scratch dir");
    let mut src = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    src.push("tests/fixtures");
    src.push(fixture);
    std::fs::copy(&src, dir.join("m.nl")).expect("copy fixture");

    let mut cmd = Command::new(PathBuf::from(env!("CARGO_BIN_EXE_pounce")));
    cmd.current_dir(&dir).arg("m.nl").arg("--no-sol");
    for o in opts {
        cmd.arg(o);
    }
    let out = cmd.output().expect("run pounce");
    let stdout = String::from_utf8_lossy(&out.stdout);
    let line = stdout
        .lines()
        .find(|l| l.starts_with("Objective..."))
        .unwrap_or_else(|| panic!("no objective line in:\n{stdout}"));
    line.split_whitespace()
        .nth(1)
        .expect("objective value")
        .parse()
        .expect("parse objective")
}

/// The convex arm must route here — if a fixture stopped classifying as an
/// LP the test would silently compare the NLP path against itself.
fn assert_routed_to_convex(fixture: &str) {
    // Per-fixture: the two callers run concurrently under the test harness.
    let stem = fixture.trim_end_matches(".nl");
    let dir = std::env::temp_dir().join(format!("pounce_issue_744_route_{stem}"));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("scratch dir");
    let mut src = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    src.push("tests/fixtures");
    src.push(fixture);
    std::fs::copy(&src, dir.join("m.nl")).expect("copy fixture");
    let out = Command::new(PathBuf::from(env!("CARGO_BIN_EXE_pounce")))
        .current_dir(&dir)
        .arg("m.nl")
        .arg("--no-sol")
        .output()
        .expect("run pounce");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("pounce-convex"),
        "{fixture} no longer routes to the convex solver:\n{stdout}"
    );
}

/// The widening is `min(1e-8, 1e-4)·1e4 = 1e-4` on the row bound, plus
/// `min(1e-8·max(0,1), 1e-4) = 1e-8` on `x1`'s declared-zero lower bound,
/// which the optimum also rides. Assert against the total with room for
/// interior-point noise but none for the un-relaxed answer.
const RELAXED_ROW_OBJ: f64 = 1e4 - 1e-4 - 1e-8;
const RELAXED_VAR_OBJ: f64 = 1e4 - 1e-4;
const NOISE: f64 = 1e-6;

#[test]
fn convex_arm_relaxes_an_inequality_row_bound_like_the_nlp_arm() {
    assert_routed_to_convex("bound_relax_row.nl");
    let convex = objective("row_convex", "bound_relax_row.nl", &[]);
    let nlp = objective("row_nlp", "bound_relax_row.nl", &["solver_selection=nlp"]);
    assert!(
        (convex - RELAXED_ROW_OBJ).abs() < NOISE,
        "convex arm did not relax the row bound: {convex} vs {RELAXED_ROW_OBJ}"
    );
    assert!(
        (convex - nlp).abs() < NOISE,
        "the two arms disagree: convex {convex} vs nlp {nlp}"
    );
}

#[test]
fn convex_arm_relaxes_a_variable_bound_and_leaves_a_fixed_variable_alone() {
    assert_routed_to_convex("bound_relax_var.nl");
    let convex = objective("var_convex", "bound_relax_var.nl", &[]);
    let nlp = objective("var_nlp", "bound_relax_var.nl", &["solver_selection=nlp"]);
    // `x1` is fixed at 0 and must not contribute a further `-1e-8`; the
    // objective is `x0 + x1`, so a widened `x1` would show up here.
    assert!(
        (convex - RELAXED_VAR_OBJ).abs() < NOISE,
        "convex arm did not relax the variable bound (or relaxed the fixed \
         variable): {convex} vs {RELAXED_VAR_OBJ}"
    );
    assert!(
        (convex - nlp).abs() < NOISE,
        "the two arms disagree: convex {convex} vs nlp {nlp}"
    );
}

/// `bound_relax_factor=0` is the escape hatch back to the model exactly as
/// declared — the answer the convex arm used to give unconditionally.
#[test]
fn bound_relax_factor_zero_restores_the_declared_model_on_both_arms() {
    for fixture in ["bound_relax_row.nl", "bound_relax_var.nl"] {
        let tag = fixture.trim_end_matches(".nl");
        let convex = objective(
            &format!("{tag}_convex0"),
            fixture,
            &["bound_relax_factor=0"],
        );
        let nlp = objective(
            &format!("{tag}_nlp0"),
            fixture,
            &["solver_selection=nlp", "bound_relax_factor=0"],
        );
        assert!(
            (convex - 1e4).abs() < NOISE,
            "{fixture}: convex arm at bound_relax_factor=0 is {convex}, want 1e4"
        );
        assert!(
            (nlp - 1e4).abs() < NOISE,
            "{fixture}: nlp arm at bound_relax_factor=0 is {nlp}, want 1e4"
        );
    }
}
