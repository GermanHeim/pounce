//! gh#592 — a `Solve_Succeeded` point that a restart from it improves.
//!
//! Reported against `pounce-solver` 0.10.0 on a LyoPRONTO Problem 2 GDP
//! fixed-policy NLP: pounce returned `Solve_Succeeded`, and re-solving
//! the identical NLP from the returned primal point improved the
//! objective by 25.10 s (0.079%), landing on the point Ipopt 3.14.16
//! reaches in a single solve.
//!
//! ## What it was
//!
//! Two compounding faults in how a `Singular` factorization report is
//! produced and answered.
//!
//! 1. `feral_inertia_pivot_floor` (gh#540) reports `Singular` when a
//!    mismatching inertia count was read off a pivot at the noise floor.
//!    Its floor was the constant `1e-12`, which corresponds to `n ≈ 4500`
//!    on the `n · eps` scale the option's own rationale names — more than
//!    an order of magnitude too generous for the few-hundred-order KKTs
//!    an IPM actually factors. It is now `n · eps`.
//! 2. `Singular` means "the constraint Jacobian may be rank-deficient",
//!    so the handler answers it with `δ_c`. On this model the Jacobian
//!    has full rank, `δ_c` could not help, and because it stays switched
//!    on for the rest of the augmented system the `δ_x` ladder had to
//!    climb against a matrix `δ_c` had made *harder* to hit the requested
//!    inertia on: five rungs, ending at `δ_w = 1e2` where Ipopt accepted
//!    the step at `1e-4`. The over-damped step froze the objective for
//!    eight iterations and the solver exited on the loose tolerance.
//!    `perturb_delta_c_max_rungs` now withdraws `δ_c` once the ladder has
//!    demonstrated it is not helping.
//!
//! ## What this file pins
//!
//! The reported model is GPL-3.0 (LyoPRONTO) and pounce is EPL-2.0, so
//! the captured `.nl` is not vendored here. The mechanism is pinned
//! directly and deterministically by unit tests —
//! `pounce_common::pd_perturbation` for the walk-back state machine and
//! `pounce_feral` for the floor — and this file pins the end-to-end
//! consequence on a fixture the repository already carries.
//!
//! `pooling_rt2stp` walks the same detour. gh#544 took it from 206 to 812
//! iterations, recorded at the time as a known cost of the
//! `feral_inertia_pivot_floor` fix (see
//! `issue_250_dual_guard_never_worse.rs`). It is the same `δ_c` spent on
//! the same evidence, and withdrawing it returns the model to 298.

use std::path::PathBuf;
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

use pounce_cli::solve_report::SolveReport;
use pounce_nlp::return_codes::ApplicationReturnStatus;

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
        "pounce_issue592_{}_{}_{suffix}",
        std::process::id(),
        n
    ));
    p
}

/// Sized against `cargo test` contention, not against this model — see
/// the note in `issue_250_dual_guard_never_worse.rs`.
const HANG_GUARD: &str = "max_wall_time=300";

fn solve(extra: &[&str]) -> SolveReport {
    let json_path = tmp_path("pooling.json");
    let sol_path = tmp_path("pooling.sol");
    let mut cmd = Command::new(pounce_exe());
    cmd.arg(fixture("pooling_rt2stp.nl"))
        .arg(&sol_path)
        .arg("--json-output")
        .arg(&json_path)
        .arg(HANG_GUARD);
    for o in extra {
        cmd.arg(o);
    }
    let _ = cmd.status().expect("spawn pounce");
    let text = std::fs::read_to_string(&json_path).expect("read json report");
    let _ = std::fs::remove_file(&json_path);
    let _ = std::fs::remove_file(&sol_path);
    serde_json::from_str(&text).expect("deserialize SolveReport")
}

/// The pre-#592 escalation: `δ_c` stays on however far the `δ_x` ladder
/// has to climb against it.
const NO_WALKBACK: [&str; 1] = ["perturb_delta_c_max_rungs=0"];

const POOLING_OPTIMUM: f64 = -3273.9549;

/// The headline. 812 iterations is the number gh#544 left behind and
/// `issue_250` recorded; 298 is what withdrawing the unhelpful `δ_c`
/// gives back. The bound is set between the two rather than at either,
/// so ordinary trajectory drift does not fail it but the detour coming
/// back does.
#[test]
fn withdrawing_an_unhelpful_delta_c_removes_the_gh544_detour() {
    let r = solve(&[]);
    assert_eq!(
        r.solution.status,
        ApplicationReturnStatus::SolveSucceeded,
        "pooling_rt2stp lost its certificate (dual inf {:e})",
        r.statistics.final_dual_inf,
    );
    assert!(
        (r.solution.objective - POOLING_OPTIMUM).abs() < 1e-3,
        "pooling_rt2stp did not reach its known optimum: {}",
        r.solution.objective,
    );
    assert!(
        r.statistics.iteration_count < 500,
        "pooling_rt2stp took {} iterations; the delta_c detour gh#592 \
         removes is back (it was 812 before, 298 after)",
        r.statistics.iteration_count,
    );
}

/// The guard against a vacuous pass: with the walk-back off this build
/// must still reproduce the long run. If a later change shortens
/// `pooling_rt2stp` by some other route, this fails and says so rather
/// than letting the test above pass for a reason it does not describe.
#[test]
fn disabling_the_walkback_restores_the_long_run() {
    let r = solve(&NO_WALKBACK);
    assert!(
        r.statistics.iteration_count > 600,
        "the pre-#592 escalation no longer reproduces the 812-iteration \
         run (got {}), so the test above is no longer pinning the fix it \
         describes",
        r.statistics.iteration_count,
    );
}

/// The walk-back must not cost the certificate it is meant to reach
/// sooner: same point, either way.
#[test]
fn the_walkback_reaches_the_same_point_it_used_to() {
    let with = solve(&[]).solution.objective;
    let without = solve(&NO_WALKBACK).solution.objective;
    assert!(
        (with - without).abs() < 1e-4,
        "withdrawing delta_c moved the answer: {with} vs {without}",
    );
}
