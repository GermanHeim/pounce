//! gh #797 — the NLP filter line-search IPM must not certify a constrained
//! *maximum* as `Solve_Succeeded`.
//!
//! The fixture is `nonconvex_qp.nl`: `min x₀·x₁ s.t. x₀ + x₁ = 2, 0 ≤ x ≤ 4`.
//! Restricted to the feasible segment the objective is `f(x₀) = x₀(2 - x₀)`,
//! which is **concave** — *maximized* at the interior stationary point `(1,1)`
//! with `f = 1`, and minimized at the two endpoints `(0,2)` and `(2,0)` with
//! `f = 0`. So the objective is a real discriminator on this model and not a
//! smoke check: `1` is the wrong extremum, reported under a confident status.
//!
//! From the bound-pushed start `(0.01, 0.01)` the first Newton step lands
//! exactly on `(1,1)`, where every KKT residual is zero. The convergence test
//! is first-order and has nothing to object to; inertia correction engages and
//! cannot help, because `δ_x I` is symmetric and cannot break the symmetry the
//! model and the iterate share. gh #797 adds the missing second-order question
//! — see `neg_curv_escapes` in the option help, and
//! `IpoptAlgorithm::try_neg_curv_escape`.

use pounce_solve_report::SolveReport;
use std::path::PathBuf;
use std::process::Command;

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

/// Run the fixture with the given extra options and return its JSON report.
fn solve(tag: &str, opts: &[&str]) -> SolveReport {
    solve_named("nonconvex_qp.nl", tag, opts)
}

fn solve_named(fixture: &str, tag: &str, opts: &[&str]) -> SolveReport {
    let json = std::env::temp_dir().join(format!("pounce_issue_797_{tag}.json"));
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

/// The headline case. Stock options, the default `solver_selection=auto`
/// routing (which sends this model to the NLP arm), and the answer is the
/// constrained *minimum*.
#[test]
fn the_nlp_arm_no_longer_certifies_the_constrained_maximum() {
    let report = solve("default", &[]);
    assert_eq!(report.solution.solve_result_num, 0, "AMPL srn 0 = solved");
    assert!(
        report.solution.objective.abs() < 1e-6,
        "expected the constrained minimum 0, not the maximum 1; got {}",
        report.solution.objective
    );
}

/// The same run with the escape switched off returns `1` — the pre-#797
/// answer. This is what makes the test above a test: without it, a change that
/// merely reroutes the model to a different engine would pass the first
/// assertion while leaving the reported defect in place.
#[test]
fn neg_curv_escapes_zero_reproduces_the_reported_defect() {
    let report = solve("off", &["solver_selection=nlp", "neg_curv_escapes=0"]);
    assert_eq!(report.solution.solve_result_num, 0);
    assert!(
        (report.solution.objective - 1.0).abs() < 1e-6,
        "with the escape disabled the NLP arm should still stop at the interior \
         stationary point (obj 1); got {}",
        report.solution.objective
    );
}

/// The escape is a bet placed *from* a strict certificate, and this is the net
/// under it: cut the continuation off before it can converge and the certified
/// stationary point comes back, under the status it always had.
///
/// `max_iter = 8` is one iteration past the escape (which happens at iter 6),
/// so the continuation is guaranteed to run out of budget. Without the floor
/// this run reports `Maximum_Iterations_Exceeded` at whatever point the escape
/// last touched; with it, `Solve_Succeeded` at `obj = 1`.
///
/// `mu_strategy_fallback=no` is what makes that an actual test of the floor,
/// and it was added because the first version of this test was not one.
/// pounce#748 turned the retry on by default and it triggers on exactly this
/// status: with the floor deleted the run still came back `Optimal Solution
/// Found` at `obj = 1`, because the retry re-solved under the other μ schedule
/// and *that* run converged to the stationary point in seven iterations. The
/// assertions below were satisfied by a mechanism that has nothing to do with
/// gh #797. Pinning the retry off leaves the floor as the only thing that can
/// produce this outcome — verified by deleting the floor and watching this
/// test, and only this test, go red.
#[test]
fn a_lost_bet_hands_back_the_certified_stationary_point() {
    let report = solve(
        "lost_bet",
        &[
            "solver_selection=nlp",
            "max_iter=8",
            "mu_strategy_fallback=no",
        ],
    );
    assert_eq!(
        report.solution.solve_result_num, 0,
        "the floor is a strict certificate, so the status stays Solve_Succeeded"
    );
    assert!(
        (report.solution.objective - 1.0).abs() < 1e-6,
        "a cut-off continuation must hand back the certified point, not \
         wherever it happened to stop; got {}",
        report.solution.objective
    );
}

/// The escape reads curvature off `W`, and under
/// `hessian_approximation=limited-memory` there is no `W` to read — BFGS
/// maintains its `B` positive definite by construction, so the probe's inertia
/// test passes at `δ_x = 0` and the escape correctly declines. This documents
/// that limit rather than asserting it away: the L-BFGS leg of
/// `scripts/sweep-fixtures.sh` still reports `obj = 1` on this fixture, and a
/// change that makes it report `0` is a change worth noticing, not a silent
/// improvement.
#[test]
fn the_limited_memory_arm_declines_for_want_of_curvature() {
    let report = solve(
        "lbfgs",
        &[
            "solver_selection=nlp",
            "hessian_approximation=limited-memory",
        ],
    );
    assert_eq!(report.solution.solve_result_num, 0);
    assert!(
        (report.solution.objective - 1.0).abs() < 1e-6,
        "the quasi-Newton arm has no negative curvature to find; got {}",
        report.solution.objective
    );
}

/// The equality row is not what makes this happen, and the `s` block is not
/// dead weight in the probe. `nonconvex_qp_ineq.nl` is the same model with the
/// row relaxed to `x₀ + x₁ ≥ 2`, so the solve carries one inequality row, a
/// slack, and a `Σ_s` diagonal — and lands on the same interior stationary
/// point `(1, 1)` from the same start, by the same mechanism, thirteen
/// iterations in. Without a fixture on this side, every vector the probe
/// touches in the slack space would have length zero and the code that handles
/// it would be untested by construction.
#[test]
fn the_escape_reaches_the_inequality_shape_too() {
    let report = solve_named("nonconvex_qp_ineq.nl", "ineq", &[]);
    assert_eq!(report.solution.solve_result_num, 0);
    assert!(
        report.solution.objective.abs() < 1e-6,
        "expected the constrained minimum 0, not the maximum 1; got {}",
        report.solution.objective
    );
}

/// …and the same discriminator, so the test above cannot pass by accident.
#[test]
fn the_inequality_shape_reproduces_the_defect_with_the_escape_off() {
    let report = solve_named(
        "nonconvex_qp_ineq.nl",
        "ineq_off",
        &["solver_selection=nlp", "neg_curv_escapes=0"],
    );
    assert_eq!(report.solution.solve_result_num, 0);
    assert!(
        (report.solution.objective - 1.0).abs() < 1e-6,
        "with the escape disabled the inequality shape should still stop at the \
         interior stationary point (obj 1); got {}",
        report.solution.objective
    );
}

/// gh#823 review finding 1 (@srikanth-gm): the escape must run against `W` at
/// the iterate it is judging under `hessian_approximation=finite-difference`
/// too, not one iterate stale.
///
/// The refresh used to be gated on `provides_exact_hessian`, which conflated
/// "is exact" with "can be re-evaluated here". The FD updater answers `false`
/// to the first and `true` to the second, so it took the stale path; it is now
/// asked the second question directly, via `HessianUpdater::hessian_at_current`.
///
/// **What this test does and does not establish.** It establishes that the
/// escape still reaches the constrained minimum under FD — i.e. that adding a
/// current-iterate rebuild did not break the mechanism, and that FD does not
/// certify the maximum on this model. It does **not** reach the stale-`W`
/// branch, and is therefore not evidence that the stale-`W` defect is fixed:
/// this fixture's objective is `x₀·x₁`, whose Hessian is the constant
/// `[[0,1],[1,0]]`, so the previous iterate's matrix and the current one's are
/// the same matrix and the two paths cannot diverge here. Reproducing the
/// symptom needs a model whose curvature varies between consecutive iterates
/// AND whose earlier iterate looks positive-definite; the reporter has such a
/// reproducer, and this repository does not yet. Recorded rather than papered
/// over, per the "which branch does the fixture take" rule in CLAUDE.md.
#[test]
fn the_escape_still_reaches_the_minimum_under_a_finite_difference_hessian() {
    let report = solve("fd", &["hessian_approximation=finite-difference"]);
    assert_eq!(report.solution.solve_result_num, 0, "AMPL srn 0 = solved");
    assert!(
        report.solution.objective.abs() < 1e-6,
        "expected the constrained minimum 0, not the maximum 1; got {}",
        report.solution.objective
    );
}
