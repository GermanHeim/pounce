//! gh #871 at the CLI: the second-order refutation must survive an equality row.
//!
//! `crates/pounce-cli/tests/fixtures/nonconvex_qp_eq.nl` is
//! `min −x₀² s.t. x₀ + x₁ + x₂ = 0` over `[0,1] × [−1,1]²`. The origin is
//! first-order clean (the gradient vanishes, so `x₀`'s bound is weakly active
//! with multiplier zero) and the true minimum is `−1` at `x₀ = 1`.
//!
//! `nonconvex_qp_ineq.nl` — gh #848's own CLI fixture — has no equality rows,
//! and neither does any `QpProblem` in either #848 unit-test file. That is the
//! dimension the corpus was uniform in, and it is the dimension the guard acts
//! on: `max_feasible_step` rejects outright any direction with an equality
//! component, and the curvature search runs on `P` rather than on `P`
//! restricted to `null(A)`, so it produced `e₀` and then threw it away. Before
//! the fix this file came back `EXIT: Optimal Solution Found.` at `f = 0`.
//!
//! The refusal is `NumericalFailure`, which the shared console vocabulary
//! renders as an internal error — see `issue848_nonconvex_qp_verdict.rs`, which
//! documents that rendering and the stderr note that compensates for it.

use std::path::PathBuf;
use std::process::Command;

fn fixture(name: &str) -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("tests");
    p.push("fixtures");
    p.push(name);
    p
}

fn run(name: &str, extra: &[&str]) -> (String, String) {
    let sol = std::env::temp_dir().join(format!("pounce_871_{name}.sol"));
    let out = Command::new(PathBuf::from(env!("CARGO_BIN_EXE_pounce")))
        .arg(fixture(name))
        .arg(&sol)
        .args(extra)
        .output()
        .expect("run pounce");
    (
        String::from_utf8_lossy(&out.stdout).into_owned(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
    )
}

fn objective(stdout: &str) -> Option<f64> {
    stdout
        .lines()
        .find(|l| l.trim_start().starts_with("Objective."))
        .and_then(|l| l.split_whitespace().nth(1))
        .and_then(|v| v.parse().ok())
}

/// The independent oracle, in the same binary: `auto` routes this class to the
/// NLP filter line-search interior-point arm, which returns `−1`. `ipopt`
/// returns `−1.0000000174963595` on the same file.
#[test]
fn the_nlp_arm_puts_the_optimum_at_minus_one() {
    let (out, _) = run("nonconvex_qp_eq.nl", &[]);
    assert!(
        out.contains("EXIT: Optimal Solution Found."),
        "the NLP arm should solve this cleanly:\n{out}"
    );
    let obj = objective(&out).expect("an objective line");
    assert!(
        (obj - (-1.0)).abs() < 1e-6,
        "the NLP arm should reach f = −1, got {obj:.6e}"
    );
}

/// The defect: the active-set engine must not certify the origin.
#[test]
fn the_active_set_engine_does_not_certify_the_saddle_behind_the_equality_row() {
    let (out, err) = run(
        "nonconvex_qp_eq.nl",
        &["solver_selection=qp-active-set", "bound_relax_factor=0"],
    );
    let obj = objective(&out).expect("an objective line");
    let certified = out.contains("EXIT: Optimal Solution Found.");
    assert!(
        !(certified && obj > -1.0 + 1e-6),
        "certified f = {obj:.6e} as optimal, but (1, −1, 0) is feasible at \
         f = −1:\n{out}"
    );
    if !certified {
        assert!(
            err.contains("not a local minimum") && err.contains("solver_selection=nlp"),
            "the refusal should say what happened and where the answer is:\n{err}"
        );
    }
}

/// The mutation guard lives one layer down, in
/// `pounce-convex/tests/issue871_refutation_with_equality_rows.rs`: a fix of
/// the shape "refuse every indefinite QP carrying an equality row" passes both
/// tests above and fails `a_convex_qp_with_equality_rows_is_untouched` and
/// `the_negative_direction_is_orthogonal_to_the_null_space` there. It is not
/// restated here because the CLI corpus has no convex QP with an equality row
/// to state it on, and a fixture added only to duplicate a unit test is a
/// fixture that rots.
mod _mutation_guard_lives_in_pounce_convex {}
