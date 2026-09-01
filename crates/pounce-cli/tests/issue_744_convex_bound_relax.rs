//! The convex path solves the model **as declared**; `bound_relax_factor`
//! buys the NLP path's widening on request (gh #744, gh #745, reversed).
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
//!
//! gh #744 originally closed that gap by widening on the convex arm too. That
//! made both arms report `27.1221` — and `27.1221` is wrong. HiGHS returns
//! `36.1224020850`, the Maros-Meszaros DOC 97/6 ground truth is `36.1224`, and
//! `benchmarks/qp_four_way.md` — generated before that change — scored the
//! unrelaxed convex arm at 137/138 correct with 0 solved-but-wrong while
//! listing `LISWET1(re=2.5e-01)` among Ipopt-MA57's *wrong* objectives. The
//! widening does not converge the arms on the model; it converges them on one
//! arm's internal perturbation, and the error it introduces is `delta` times the
//! bound's multiplier, which nothing bounds, and always one-signed.
//!
//! So the default reversed: this arm solves what the caller declared, and the
//! widening is opt-in. The NLP arm keeps it — a feasible-iterate log-barrier
//! needs `x` strictly inside its bounds, and matching Ipopt is that arm's
//! contract. The two therefore differ on constraint-degenerate models by
//! design, which `final_declared_constr_viol` reports rather than hides.
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

/// Asking for the widening buys `min(1e-8, 1e-4)·1e4 = 1e-4` on the row
/// bound, plus `min(1e-8·max(0,1), 1e-4) = 1e-8` on `x1`'s declared-zero
/// lower bound, which the optimum also rides. Far above interior-point noise
/// and far above the ulp at `1e4`, so "did the widening happen" is
/// unambiguous in either direction.
const RELAXED_ROW_OBJ: f64 = 1e4 - 1e-4 - 1e-8;
const RELAXED_VAR_OBJ: f64 = 1e4 - 1e-4;
/// The declared optimum: both fixtures sit exactly on a bound of magnitude
/// `1e4`.
const DECLARED_OBJ: f64 = 1e4;
const NOISE: f64 = 1e-6;
/// Opt in to the NLP arm's model.
const RELAX: &str = "bound_relax_factor=1e-8";

#[test]
fn convex_arm_takes_an_inequality_row_bound_as_declared() {
    assert_routed_to_convex("bound_relax_row.nl");
    let convex = objective("row_convex", "bound_relax_row.nl", &[]);
    assert!(
        (convex - DECLARED_OBJ).abs() < NOISE,
        "convex arm should solve the row bound as declared: {convex} vs \
         {DECLARED_OBJ} (the widened answer is {RELAXED_ROW_OBJ})"
    );
}

/// ... and asking for the widening by name still reproduces the NLP arm's
/// model exactly, so the escape hatch to Ipopt parity is real.
#[test]
fn asking_for_the_widening_reproduces_the_nlp_arm_on_a_row_bound() {
    let convex = objective("row_convex_relax", "bound_relax_row.nl", &[RELAX]);
    let nlp = objective(
        "row_nlp",
        "bound_relax_row.nl",
        &["solver_selection=nlp", RELAX],
    );
    assert!(
        (convex - RELAXED_ROW_OBJ).abs() < NOISE,
        "opt-in widening did not move the row bound: {convex} vs {RELAXED_ROW_OBJ}"
    );
    assert!(
        (convex - nlp).abs() < NOISE,
        "the two arms disagree under an explicit widening: convex {convex} \
         vs nlp {nlp}"
    );
}

#[test]
fn convex_arm_takes_a_variable_bound_as_declared() {
    assert_routed_to_convex("bound_relax_var.nl");
    let convex = objective("var_convex", "bound_relax_var.nl", &[]);
    assert!(
        (convex - DECLARED_OBJ).abs() < NOISE,
        "convex arm should solve the variable bound as declared: {convex} \
         vs {DECLARED_OBJ} (the widened answer is {RELAXED_VAR_OBJ})"
    );
}

/// The opt-in path still leaves a FIXED variable alone. `x1` is fixed at 0
/// and must not contribute a further `-1e-8`; the objective is `x0 + x1`, so
/// a widened `x1` would show up here. Upstream's default
/// `fixed_variable_treatment=make_parameter` lifts it out before
/// `relax_bounds` runs, and the extractor mirrors that.
#[test]
fn asking_for_the_widening_still_leaves_a_fixed_variable_alone() {
    let convex = objective("var_convex_relax", "bound_relax_var.nl", &[RELAX]);
    let nlp = objective(
        "var_nlp",
        "bound_relax_var.nl",
        &["solver_selection=nlp", RELAX],
    );
    assert!(
        (convex - RELAXED_VAR_OBJ).abs() < NOISE,
        "opt-in widening did not move the variable bound (or it widened the \
         fixed variable): {convex} vs {RELAXED_VAR_OBJ}"
    );
    assert!(
        (convex - nlp).abs() < NOISE,
        "the two arms disagree under an explicit widening: convex {convex} \
         vs nlp {nlp}"
    );
}

/// `bound_relax_factor=0` still names the declared model explicitly — the
/// convex arm's default now, and the NLP arm's escape hatch to it.
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
