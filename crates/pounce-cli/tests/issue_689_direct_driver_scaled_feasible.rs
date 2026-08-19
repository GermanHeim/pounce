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
//! The objective assertion is the sharper half of the test. HSDE stops on these
//! two at `236.85` and `456.33` — its scale-relative gap test normalizes by the
//! objective magnitude, which here is `5e11` of constant offset, so it certifies
//! `Optimal` a few hundred short of the real optimum. That is the "objective
//! moves under any trajectory perturbation" half of #689: the fixtures are not
//! degenerate, the relative gap test is simply admitting non-optimal iterates.
//! The direct driver lands on `0`.

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

/// A canary, not a fix: the default route never enters the direct driver, so
/// #689 leaves it exactly as it was — reporting `Optimal` a few hundred above
/// the optimum the direct driver reaches (`236.85` / `456.33` as measured).
/// The assertion is deliberately one-sided and loose: it records *that* HSDE
/// stops short, not where. If it ever fails, HSDE has started reaching the real
/// optimum on these models — which is the outcome we want. Delete this test
/// then, and with it the caveat in
/// `dev-notes/issue-689-direct-driver-cold-start.md` §3 about reading these two
/// fixtures as objective sentinels in the sweep.
#[test]
fn hsde_still_stops_short_of_the_optimum_on_the_same_pair() {
    for model in ["scaled_feasible_a.nl", "scaled_feasible_b.nl"] {
        let report = solve(model, &[]);
        assert!(
            (0..100).contains(&report.solution.solve_result_num),
            "{model} at defaults: HSDE must still report solved",
        );
        assert!(
            report.solution.objective.abs() > 1.0,
            "{model} at defaults: HSDE returned objective {:e}, at or near the \
             true optimum 0 — better than the `236.85`/`456.33` this canary was \
             written against. Nothing is broken: delete this test and the \
             sweep caveat it guards (dev-notes/issue-689-direct-driver-cold-\
             start.md §3).",
            report.solution.objective,
        );
    }
}
