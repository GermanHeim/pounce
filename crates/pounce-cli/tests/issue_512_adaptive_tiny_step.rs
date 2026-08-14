//! gh #512 — the adaptive μ update must terminate on a tiny step.
//!
//! When the line search takes two consecutive steps so small that any
//! nonzero α is floating-point noise, `IpoptData::tiny_step_flag` goes up
//! and the barrier update gets one chance to move μ. If it cannot, the
//! iterate is frozen: every later iteration recomputes the same point and
//! the solve burns its budget standing still. Upstream stops there and
//! says so — `TINY_STEP_DETECTED` → `STOP_AT_TINY_STEP`, "problem solved
//! to best possible numerical accuracy".
//!
//! `IpMonotoneMuUpdate.cpp:158-161` throws once, and pounce reconstructed
//! it from "flag was set, μ came back the same". `IpAdaptiveMuUpdate.cpp`
//! throws in *two* places — `:330-333` after the fixed-mode
//! Fiacco-McCormick decrease, `:377-380` on the free→fixed switch — and a
//! comment in `ipopt_alg.rs` asserted the opposite, that the adaptive
//! update only ever routes the flag through `force_no_progress` and keeps
//! iterating. So `mu_strategy=adaptive` never stopped.
//!
//! The issue was filed from a source comparison, without a model. There
//! are several already in this directory. `airport.nl` at `tol=1e-12`:
//!
//! ```text
//!   before   300 iterations   Maximum_Iterations_Exceeded
//!   after     16 iterations   Search_Direction_Becomes_Too_Small
//! ```
//!
//! and the two runs' final objective, dual infeasibility and constraint
//! violation agree to every digit printed — the 284 extra iterations moved
//! nothing. `hs71_obj1e8.nl` shows the same at *default* tolerances (70 →
//! 11), where iterations 11..70 are all tagged `T` at one frozen point.
//!
//! The tests below pin the invariant rather than those counts: **an
//! adaptive-μ solve that reaches a flagged tiny step must stop there**,
//! and must stop at a point no worse than the one it would have ground out
//! at the iteration cap.
//!
//! Restoration is on the same hook — see
//! `a_square_infeasible_model_survives_a_tiny_step_exit_from_restoration`,
//! which guards the interaction with gh #508's verdict machinery.

use std::path::PathBuf;
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

use pounce_cli::solve_report::SolveReport;
use pounce_nlp::ApplicationReturnStatus;

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
        "pounce_issue512_{}_{}_{suffix}",
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
        .arg(&json_path)
        .arg("print_level=0");
    for opt in extra_opts {
        cmd.arg(opt);
    }
    let _ = cmd.status().expect("spawn pounce");
    let text = std::fs::read_to_string(&json_path).expect("read json report");
    let _ = std::fs::remove_file(&json_path);
    let _ = std::fs::remove_file(&sol_path);
    serde_json::from_str(&text).expect("deserialize SolveReport")
}

/// The iteration cap the tight-tolerance runs are given. Small enough to
/// keep the tests quick, large enough that hitting it means the solve was
/// genuinely stuck rather than under-resourced — the fix stops these
/// models inside 20 iterations.
const MAX_ITER: &str = "max_iter=300";

/// Tight-`tol` option set. `acceptable_tol` is pushed below anything
/// reachable so the acceptable-level exit cannot pre-empt the tiny-step
/// one and mask the behaviour under test.
fn tight_adaptive() -> Vec<&'static str> {
    vec![
        "mu_strategy=adaptive",
        "tol=1e-12",
        "acceptable_tol=1e-15",
        MAX_ITER,
    ]
}

/// gh #512 proper. `airport.nl` settles onto a tiny step at iteration 16
/// and cannot move μ again; before the fix it recomputed that same point
/// until `max_iter`.
///
/// Asserted as "did not burn the budget", not as "returned status X": what
/// the issue is about is the solve continuing past the point where nothing
/// can change. Which of the honest terminal statuses it lands on is the
/// certificate machinery's call, and both `StopAtTinyStep` and an upgrade
/// to a solved status are correct answers to "you stopped in time".
#[test]
fn an_adaptive_solve_stops_at_a_flagged_tiny_step() {
    let r = solve("airport.nl", &tight_adaptive());
    assert_ne!(
        r.solution.status,
        ApplicationReturnStatus::MaximumIterationsExceeded,
        "adaptive μ ran out the iteration budget at a frozen iterate \
         instead of stopping at the tiny step (gh #512); \
         iteration_count={}",
        r.statistics.iteration_count,
    );
    assert!(
        r.statistics.iteration_count < 300,
        "iteration_count={} means the solve reached the cap; the tiny step \
         is detected around iteration 16 and nothing after it moves",
        r.statistics.iteration_count,
    );
}

/// Direction guard, and the reason stopping early is not giving up: the
/// point the solve stops at is converged. Passes before the fix too — the
/// pre-fix run reaches the *same* point and then keeps recomputing it —
/// which is exactly what makes it the right guard. If a future change
/// makes the tiny-step exit fire somewhere genuinely unconverged, this
/// fails while the test above still passes.
#[test]
fn the_tiny_step_exit_lands_on_a_converged_point() {
    let r = solve("airport.nl", &tight_adaptive());
    let s = &r.statistics;
    assert!(
        s.final_constr_viol < 1e-8,
        "stopped at constraint violation {} — a tiny step is only a valid \
         reason to stop if the iterate is feasible",
        s.final_constr_viol,
    );
    assert!(
        s.final_dual_inf < 1e-6,
        "stopped at dual infeasibility {} — that is a stall, not \
         convergence to best available accuracy",
        s.final_dual_inf,
    );
}

/// The same defect at **default** tolerances, where it costs iterations
/// rather than a status. `hs71_obj1e8.nl` (HS71 × 1e8, the gh #266
/// fixture) converges at iteration 11 and then, before the fix, spent
/// iterations 11..70 re-deriving one frozen iterate — every row tagged `T`
/// with an unchanged objective, `inf_pr` and `lg(mu)`.
///
/// The bound is deliberately loose: 11 after the fix, 70 before it.
#[test]
fn adaptive_does_not_grind_at_a_frozen_iterate_at_default_tolerances() {
    let r = solve("hs71_obj1e8.nl", &["mu_strategy=adaptive"]);
    // `0` exactly — the strict certificate. Since gh #591 the `0..=99` solved
    // band also holds `1` (`Solved_To_Acceptable_Level`), which is the
    // fallback this fixture must not need.
    assert!(
        r.solution.solve_result_num == 0,
        "expected the known optimum to be certified, got \
         solve_result_num={} ({:?})",
        r.solution.solve_result_num,
        r.solution.status,
    );
    assert!(
        r.statistics.iteration_count <= 35,
        "iteration_count={} — the solve reaches its optimum at ~11 \
         iterations and the rest are tiny steps at a point that no longer \
         moves (gh #512)",
        r.statistics.iteration_count,
    );
}

/// The default `mu_strategy` is monotone, which already terminated on a
/// tiny step through the main loop's μ-comparison. Nothing in gh #512
/// touches that route; this pins the default path against a regression
/// from the adaptive-side plumbing.
#[test]
fn the_monotone_route_is_untouched() {
    let r = solve("hs71_obj1e8.nl", &[]);
    assert!(
        r.solution.solve_result_num == 0,
        "default (monotone) solve regressed: solve_result_num={} ({:?})",
        r.solution.solve_result_num,
        r.solution.status,
    );
}

/// Restoration runs the same inner IPM, so it acquired the same new exit —
/// and gh #508's verdict machinery had a hole underneath it.
///
/// `issue_508_infeasible_gap_1em2.nl` is `min (x-5)² s.t. x²+1e-2 = 0`:
/// square, and infeasible by a full percent. At `tol=1e-12` the inner
/// restoration solve reaches its stationary point by tiny step rather than
/// by a certified convergence, and `resto_inner_solver`'s tiny-step gate
/// carried a `!is_square_problem` exclusion — so the verdict fell through
/// to `Restoration_Failed`, AMPL 500, Pyomo `internalSolverError`. Upstream
/// throws `LOCALLY_INFEASIBLE` on that branch (`IpRestoMinC_1Nrm.cpp:278-291`)
/// with no test on problem shape.
///
/// Bites on the parent through the `monotone` arm, which could already
/// reach the inner tiny-step exit; the `adaptive` arm guards the
/// interaction gh #512 newly creates.
#[test]
fn a_square_infeasible_model_survives_a_tiny_step_exit_from_restoration() {
    for strategy in ["mu_strategy=monotone", "mu_strategy=adaptive"] {
        let r = solve(
            "issue_508_infeasible_gap_1em2.nl",
            &[strategy, "tol=1e-12", "acceptable_tol=1e-15", MAX_ITER],
        );
        assert_eq!(
            r.solution.solve_result_num, 200,
            "{strategy}: a model infeasible by 1e-2 must report local \
             infeasibility (AMPL 200), got {} ({:?}) — 500 tells the user \
             their solver broke (gh #508, #512)",
            r.solution.solve_result_num, r.solution.status,
        );
    }
}
