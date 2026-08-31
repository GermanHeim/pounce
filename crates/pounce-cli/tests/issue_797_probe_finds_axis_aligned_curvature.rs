//! gh #797 follow-up — the negative-curvature *probe* must find the witness,
//! not just be asked for one.
//!
//! `issue_797_neg_curvature_escape.rs` pins that the escape fires on
//! `nonconvex_qp.nl`. It cannot see the failure this file pins, because that
//! model's negative-curvature direction happens to overlap the probe's fixed
//! seed well. When it does not, the probe declines and the solve reports
//! `Solve_Succeeded` at the saddle — gh#797's own defect, reached through the
//! search rather than through the gate.
//!
//! The fixtures are `min ½(a·x₀² + b·x₁²)` on `[-2, 2]²` from the origin, which
//! is a strict saddle with every KKT residual zero. The minimum is `-2.1`, at
//! the bound of whichever coordinate carries the negative curvature. The two
//! differ ONLY by which coordinate that is:
//!
//! * `saddle_axis_second.nl` — `a = 1, b = -1.05`
//! * `saddle_axis_first.nl`  — `a = -1.05, b = 1`
//!
//! They are the same model with the variables renamed, so any answer that
//! differs between them is reporting a property of the solver's seed rather
//! than of the problem. Before the fix `saddle_axis_second` returned `0` and
//! `saddle_axis_first` returned `-2.1`.
//!
//! Two things caused it, and both are pinned below.
//!
//! 1. The shift ladder climbs by ×10 and stops at the first rung that factors,
//!    so it can overshoot `|λ_min|` by a decade. Here it rejects `δ = 1` and
//!    takes `δ = 10`, leaving eigenvalues `(11, 8.95)` — a ratio of 0.81, at
//!    which inverse iteration barely separates anything.
//! 2. Only three inverse iterations ran, amplifying the negative direction by
//!    `1.9×`, which is not enough from a seed with a small component along it.
//!
//! The fix bisects the bracket the ladder leaves and raises the iteration
//! budget, mirroring `crates/pounce-qp/src/negcurv.rs` (gh#848), which had
//! already been given both on the QP arm.
//!
//! MUTATION TABLE — measured, not assumed. **The two halves of the fix are
//! individually sufficient on this fixture**, so reverting either one alone
//! leaves the suite green; only reverting both reproduces the defect. That is
//! recorded here rather than tidied away, because a table claiming each
//! constant is separately load-bearing would be false, and the next reader
//! deleting "the redundant one" needs to know it is redundant *on this
//! fixture* and not in general — a wider spectrum starves 20 iterations too,
//! and a shift landing exactly on `|λ_min|` still needs iterations to converge.
//!
//! | change                                             | result                                    |
//! |----------------------------------------------------|-------------------------------------------|
//! | `NEG_CURV_SHIFT_REFINEMENTS = 0` alone             | green — 20 iterations carry it            |
//! | `NEG_CURV_INVERSE_ITERS = 3` alone                 | green — the tightened shift carries it    |
//! | **both** reverted (`0` and `3`)                    | `both_axis_orientations_leave_the_saddle` and `the_answer_does_not_depend_on_which_coordinate_is_concave` fail, `saddle_axis_second.nl` reporting `0` |
//! | make the escape unconditional (ignore the option)  | `the_kill_switch_still_reports_the_saddle` fails |

use pounce_solve_report::SolveReport;
use std::path::PathBuf;
use std::process::Command;

/// The saddle's objective, and the value a declining probe reports.
const SADDLE_OBJ: f64 = 0.0;
/// The true minimum of both fixtures: `-2 * 1.05`.
const TRUE_MIN: f64 = -2.1;

fn pounce_exe() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_pounce"))
}

fn fixture_named(name: &str) -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("tests");
    p.push("fixtures");
    p.push(name);
    p
}

fn solve_named(fixture: &str, tag: &str, opts: &[&str]) -> SolveReport {
    let json = std::env::temp_dir().join(format!("pounce_797_probe_{tag}.json"));
    let _ = std::fs::remove_file(&json);
    let out = Command::new(pounce_exe())
        .arg(fixture_named(fixture))
        .arg("--no-sol")
        .arg("--json-output")
        .arg(&json)
        .args(opts)
        .output()
        .expect("spawn pounce");
    let combined = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    assert_eq!(out.status.code(), Some(0), "must solve:\n{combined}");
    let text = std::fs::read_to_string(&json).expect("JSON report should be written");
    let _ = std::fs::remove_file(&json);
    serde_json::from_str(&text).expect("deserialize report")
}

const ORIENTATIONS: [&str; 2] = ["saddle_axis_second.nl", "saddle_axis_first.nl"];

/// The defect, stated as the property it violated: renaming the variables must
/// not change the answer.
#[test]
fn both_axis_orientations_leave_the_saddle() {
    for fixture in ORIENTATIONS {
        let tag = fixture.trim_end_matches(".nl");
        let r = solve_named(fixture, tag, &[]);
        let obj = r.solution.objective;
        assert!(
            (obj - TRUE_MIN).abs() < 1e-4,
            "{fixture}: expected the minimum {TRUE_MIN}, got {obj}. \
             {SADDLE_OBJ} is the saddle the probe is supposed to leave — a value \
             there means the negative-curvature search declined, which is the \
             gh#797 defect reached through the search instead of the gate."
        );
    }
}

/// The two fixtures are one model under a permutation, so they must agree to
/// the solver's own tolerance — not merely both be "close to the minimum".
#[test]
fn the_answer_does_not_depend_on_which_coordinate_is_concave() {
    let a = solve_named(ORIENTATIONS[0], "perm_a", &[])
        .solution
        .objective;
    let b = solve_named(ORIENTATIONS[1], "perm_b", &[])
        .solution
        .objective;
    assert!(
        (a - b).abs() < 1e-6,
        "the same model with its variables renamed gave {a} and {b}; the probe's \
         fixed seed is leaking into the answer"
    );
}

/// The escape stays opt-out, and turning it off restores the pre-gh#797
/// behaviour on exactly these models. Without this, a change that made the
/// escape unconditional would pass everything above.
#[test]
fn the_kill_switch_still_reports_the_saddle() {
    for (i, fixture) in ORIENTATIONS.iter().enumerate() {
        let r = solve_named(fixture, &format!("off_{i}"), &["neg_curv_escapes=0"]);
        let obj = r.solution.objective;
        assert!(
            (obj - SADDLE_OBJ).abs() < 1e-6,
            "{fixture}: with neg_curv_escapes=0 the first-order certificate at \
             the saddle is what should be reported, got {obj}"
        );
    }
}

/// The bisection must not cost the model the original gh#797 test covers: the
/// escape still fires on `nonconvex_qp.nl`, whose minimum on the feasible
/// segment is `0` against the constrained maximum `1`.
#[test]
fn the_original_797_model_still_escapes() {
    let r = solve_named("nonconvex_qp.nl", "orig", &[]);
    let obj = r.solution.objective;
    assert!(
        obj.abs() < 1e-4,
        "nonconvex_qp.nl should still reach the minimum 0, got {obj}"
    );
}
