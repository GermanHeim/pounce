//! Issue #689 regression: the direct convex driver (`qp_hsde=no`) must solve
//! `scaled_feasible_a` / `scaled_feasible_b` instead of diverging on them.
//!
//! Both fixtures are `min Σ(xᵢ − aᵢ)²` with the objective's own centre `a`
//! *inside* the feasible set — three constraints of it exactly active there —
//! so the optimum is `x* = a` at objective `0` — checkable by hand against the
//! `.nl` file, and independently what the NLP route returns.
//!
//! Before the fix, at `qp_hsde=no`, `a` ran to the 199-iteration cap at
//! `final_kkt_error 8.4e45`, `final_dual_inf 3.7e41`, objective `1.14e11`; `b`
//! broke down at 99 iterations with objective `5.0e11`. Neither was slow
//! convergence — the driver was diverging. The cause was the cold start: on
//! these models the Ruiz-equilibrated feasible set sits at `‖ĥ‖ ≈ 5e9`, and
//! `s = z = e` put the starting slacks nine orders below it, so
//! fraction-to-boundary cut the first (good) Newton step to `α ≈ 8e-9` and the
//! corrector's `σμ / s` then returned directions of `1e18`.
//!
//! This is not a debug-only path. `QpWarmStart`, the build-once
//! `QpFactorization` handle and the dual-infeasibility reverify guard all use
//! the direct driver, and none of them can fall back to HSDE the way the
//! one-shot path can.
//!
//! The objective assertion is the sharper half of the test, and it is asserted
//! on **both** routes. HSDE used to stop on these two at `236.85` and `456.33`:
//! its scale-relative gap test normalizes by the objective magnitude, and
//! `QpProblem` carries the quadratic form only, so on a least-squares objective
//! that magnitude is `Σaᵢ² = 5e11` of constant — a blanket `5e3` slack on the
//! gap. That is the "objective moves under any trajectory perturbation" half of
//! #689: the fixtures are not degenerate, the stopping test was admitting
//! non-optimal iterates. Telling the solver its objective constant
//! (`QpOptions::obj_constant`) makes the same test normalize by the objective
//! the user reads, and both routes now land on `0`.

use std::path::PathBuf;
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

use pounce_cli::solve_report::SolveReport;

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

fn tmp_path(suffix: &str) -> PathBuf {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let mut p = std::env::temp_dir();
    p.push(format!(
        "pounce_issue689_{}_{}_{suffix}",
        std::process::id(),
        n
    ));
    p
}

fn solve(fixture_name: &str, extra_opts: &[&str]) -> SolveReport {
    let json_path = tmp_path(&format!("{fixture_name}.json"));
    let sol_path = tmp_path(&format!("{fixture_name}.sol"));
    let mut cmd = Command::new(pounce_exe());
    cmd.arg(fixture(fixture_name))
        .arg(&sol_path)
        .arg("--json-output")
        .arg(&json_path);
    for opt in extra_opts {
        cmd.arg(opt);
    }
    let _ = cmd.status().expect("spawn pounce");
    let text = std::fs::read_to_string(&json_path).expect("read json report");
    let _ = std::fs::remove_file(&json_path);
    let _ = std::fs::remove_file(&sol_path);
    serde_json::from_str(&text).expect("deserialize SolveReport")
}

/// The iteration cap the pre-fix runs hit (`max_iter` 200, reported as 199) and
/// the neighbourhood a healthy solve of these models lands in (27–28). Anything
/// past this is the divergence coming back, whatever status it wears.
const ITER_BUDGET: i32 = 60;

/// Both fixtures' true optimum. The objective is a sum of squares whose centre
/// is feasible, so it is exactly `0`; the tolerance is what the cancellation
/// against the `5e11` constant offset leaves resolvable.
const OPTIMUM_TOL: f64 = 1e-3;

#[test]
fn direct_driver_solves_the_scaled_feasible_pair() {
    for model in ["scaled_feasible_a.nl", "scaled_feasible_b.nl"] {
        let report = solve(model, &["qp_hsde=no"]);
        let code = report.solution.solve_result_num;
        assert!(
            (0..100).contains(&code),
            "{model} at qp_hsde=no: solve_result_num={code} (status {:?}) after \
             {} iterations, objective {:e}, kkt_error {:e}. This model is \
             feasible with optimum 0 and the direct driver must reach it \
             (gh #689).",
            report.solution.status,
            report.statistics.iteration_count,
            report.solution.objective,
            report.statistics.final_kkt_error,
        );
        assert!(
            report.statistics.iteration_count <= ITER_BUDGET,
            "{model} at qp_hsde=no: {} iterations (budget {ITER_BUDGET}); the \
             pre-fix divergence ran to the 199-iteration cap",
            report.statistics.iteration_count,
        );
        assert!(
            report.solution.objective.abs() <= OPTIMUM_TOL,
            "{model} at qp_hsde=no: objective {:e}, want 0 ± {OPTIMUM_TOL:e}. \
             The objective's own centre is feasible on this model, so 0 is the \
             optimum, not a tolerance artifact.",
            report.solution.objective,
        );
    }
}

/// The other half of #689: the **default** route must not stop at the false
/// optimum either — and, gh #712, must not dress a point it never reached in a
/// success verdict.
///
/// HSDE returned `Optimal` here at `236.85` / `456.33` — `4.7e-10` relative in
/// its own metric, and 100% wrong in the caller's, because the metric was the
/// `5e11` constant the solver had never been told about. With
/// `QpOptions::obj_constant` supplied it normalizes by the caller's objective
/// and runs on toward `0`.
///
/// `scaled_feasible_a` does not get there inside the default 200-iteration
/// budget: it provably needs ~3596 (measured — the cold HSDE solve reaches
/// `final_kkt_error 1.9e-10` there). What it used to do instead was accept the
/// gh #293 equilibrated retry's `Optimal`, reported at 123 iterations on a
/// point whose absolute KKT error is `2.3e3` — certified because the *second*
/// objective-magnitude normalizer, in `equilibrated_kkt_rel_parts`, had not had
/// the same correction (gh #712). So the default budget is asserted here for
/// what it is: an honest non-convergence, with the optimum reachable as soon as
/// the budget allows it.
///
/// Kept separate from the `qp_hsde=no` test above because the two are fixed by
/// different changes and can regress independently.
#[test]
fn the_default_route_reaches_the_same_optimum() {
    // Whatever the verdict, the *point* must not be the #689 false optimum.
    for model in ["scaled_feasible_a.nl", "scaled_feasible_b.nl"] {
        let report = solve(model, &[]);
        assert!(
            report.solution.objective.abs() <= OPTIMUM_TOL,
            "{model} at defaults: objective {:e}, want 0 ± {OPTIMUM_TOL:e}. This \
             is the gh #689 false optimum: the scale-relative gap test \
             normalizing by an objective magnitude that is `5e11` of constant \
             offset rather than by the objective the caller reads.",
            report.solution.objective,
        );
    }

    // `b` converges inside the default budget, so it also reports success.
    let report = solve("scaled_feasible_b.nl", &[]);
    let code = report.solution.solve_result_num;
    assert!(
        (0..100).contains(&code),
        "scaled_feasible_b.nl at defaults: solve_result_num={code} (status {:?})",
        report.solution.status,
    );

    // `a` does not, and must say so rather than certify the retry's point.
    let report = solve("scaled_feasible_a.nl", &[]);
    let code = report.solution.solve_result_num;
    assert!(
        !(0..100).contains(&code),
        "scaled_feasible_a.nl at defaults: solve_result_num={code} (status {:?}) \
         after {} iterations at final_kkt_error {:e}. This model needs ~3596 \
         iterations; a success verdict inside a 200-iteration budget is the \
         gh #712 false optimum, whose absolute KKT error is 2.3e3.",
        report.solution.status,
        report.statistics.iteration_count,
        report.statistics.final_kkt_error,
    );

    // Given the budget it needs, it reaches the same optimum as the direct
    // driver does — the verdict above is about the budget, not the model.
    let report = solve("scaled_feasible_a.nl", &["max_iter=4000"]);
    let code = report.solution.solve_result_num;
    assert!(
        (0..100).contains(&code),
        "scaled_feasible_a.nl at max_iter=4000: solve_result_num={code} (status \
         {:?}) after {} iterations",
        report.solution.status,
        report.statistics.iteration_count,
    );
    assert!(
        report.solution.objective.abs() <= OPTIMUM_TOL,
        "scaled_feasible_a.nl at max_iter=4000: objective {:e}, want 0 ± {OPTIMUM_TOL:e}",
        report.solution.objective,
    );
}

/// gh #712, stated as the property the issue opened on: **a budget cap must not
/// change the accuracy of a solve that reports success.**
///
/// The symptom was a sharp, deterministic flip at `max_iter = 2610/2611` —
/// below it `SolveSucceeded` at 123 iterations and `final_kkt_error 4.6e-3`
/// (a point whose absolute KKT error is `2.3e3`), above it `SolveSucceeded` at
/// a genuine `5.2e-11`. Same model, same options, same verdict, six orders of
/// accuracy apart. The rule this asserts does not care *which* budgets
/// converge — the trajectory oscillates on this model and that is its own
/// business — only that every budget which claims success hands back a point
/// that is one.
#[test]
fn a_success_verdict_carries_the_same_accuracy_at_every_budget() {
    // Spans the historical flip (2610/2611) and the budget at which the cold
    // solve converges outright (3596).
    for cap in [200, 400, 2610, 2611, 3000, 3595, 3596, 4000] {
        let report = solve("scaled_feasible_a.nl", &[&format!("max_iter={cap}")]);
        let code = report.solution.solve_result_num;
        if !(0..100).contains(&code) {
            continue;
        }
        assert!(
            report.statistics.final_kkt_error <= 1e-6,
            "scaled_feasible_a.nl at max_iter={cap}: reports success (status \
             {:?}) at final_kkt_error {:e} after {} iterations. A success \
             verdict has to mean the same thing at every budget (gh #712).",
            report.solution.status,
            report.statistics.final_kkt_error,
            report.statistics.iteration_count,
        );
    }
}
