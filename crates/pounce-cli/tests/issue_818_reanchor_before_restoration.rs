//! gh #818 — a line-search failure at an *already feasible* point is not a
//! restoration problem.
//!
//! When the backtracking line search cannot accept any trial step, either
//! the **point** is bad — infeasible, and the restoration phase is exactly
//! the right tool — or the **direction** is, because `W` is a quasi-Newton
//! model carrying curvature the iterate has left behind. Upstream has one
//! answer for both, because restoration is the only fallback it has.
//!
//! At a feasible point that answer is a no-op. The restoration NLP minimizes
//! the constraint violation and there is none to minimize, so it wanders at
//! `theta ~ 1e-13` and reports `Restoration_Failed`. Measured on `deb7` under
//! `limited-memory`: the solve stalls with `inf_pr ~ 1e-12` and `inf_du ~ 1e5`,
//! enters restoration at a point feasible to 8e-13, and spends 340 of its 1242
//! iterations there. On an unconstrained model `theta` is identically zero, so
//! restoration cannot move at all.
//!
//! `IpoptAlgorithm::try_reanchor_before_restoration` puts one rung in front of
//! the hand-off: drop every curvature pair but the newest and retry. It is not
//! a feasibility gate on restoration — that was tried and rejected before (see
//! the `constr_viol_tol` paragraph in `IpoptAlgorithm::invoke_restoration`) —
//! and it runs *after* the acceptable-point decline, so a reportable point is
//! still reported.
//!
//! **What this file is not evidence about.** It pins that the rung is wired,
//! fires, and is switchable. It does not pin the corpus: that is
//! `scripts/sweep-fixtures.sh`, where this change moves five lbfgs-leg lines
//! (`deb7` 2295 -> 1695, `eigena2` 131 -> 93, `infeasible_square_scaled_1em4`
//! 28 -> 19, `issue_508_infeasible_gap_1em4` 79 -> 76, and one objective
//! digit) and leaves the exact leg byte-identical.

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

/// Solve `fixture` on the limited-memory arm with `restarts` re-anchors
/// allowed, returning its JSON report. The CLI's exit code is not asserted:
/// `eigena2` fails on this arm either way, and the point of the test is the
/// iteration count, not the status.
fn solve(fixture: &str, tag: &str, restarts: u32) -> SolveReport {
    let json = std::env::temp_dir().join(format!("pounce_issue_818_{tag}.json"));
    let _ = std::fs::remove_file(&json);
    let out = Command::new(pounce_exe())
        .arg(fixture_named(fixture))
        .arg("--no-sol")
        .arg("--json-output")
        .arg(&json)
        .arg("hessian_approximation=limited-memory")
        .arg(format!("limited_memory_ls_failure_restarts={restarts}"))
        .output()
        .expect("spawn pounce");
    let text = std::fs::read_to_string(&json).unwrap_or_else(|e| {
        panic!(
            "JSON report should be written for {fixture} ({tag}): {e}\n{}{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        )
    });
    let _ = std::fs::remove_file(&json);
    serde_json::from_str(&text).expect("deserialize report")
}

fn iters(r: &SolveReport) -> i32 {
    r.statistics.iteration_count
}

/// `infeasible_square_scaled_1em4` is the cheap, unambiguous case: the rung
/// fires, the solve reaches the *same* verdict — the problem really is
/// infeasible — and gets there in fewer iterations.
///
/// The verdict assertion is the load-bearing half. A rung that reached the
/// answer faster by giving up on a genuine restoration would show here as a
/// changed status, which is precisely the failure mode the ordering against
/// the acceptable-point decline is meant to prevent.
#[test]
fn reanchor_shortens_the_infeasibility_certificate_without_changing_it() {
    let off = solve("infeasible_square_scaled_1em4.nl", "infeas_off", 0);
    let on = solve("infeasible_square_scaled_1em4.nl", "infeas_on", 1);

    assert_eq!(
        off.solution.status, on.solution.status,
        "the rung must not change the verdict, only how long it takes"
    );
    assert!(
        matches!(
            off.solution.status,
            pounce_nlp::return_codes::ApplicationReturnStatus::InfeasibleProblemDetected
        ),
        "expected an infeasibility certificate, got {:?}",
        off.solution.status
    );
    assert!(
        iters(&on) < iters(&off),
        "the rung fired but cost iterations: {} with, {} without",
        iters(&on),
        iters(&off)
    );
}

/// `eigena2` is the stall the rung exists for: the solve fails on this arm
/// whichever way the option is set, so the only thing that can move is how
/// long it spends failing. Measured 131 iterations without the rung and 93
/// with — a third of the trajectory was a restoration phase that had a
/// constraint violation of ~1e-10 to work with.
///
/// Asserted as a strict inequality rather than a pinned count: the number is
/// a property of a stalling solve and will drift, but "re-anchoring is not
/// worse than walking into a restoration that has nothing to reduce" is the
/// claim, and it is falsifiable.
#[test]
fn reanchor_shortens_a_stall_that_ends_in_restoration() {
    let off = solve("eigena2.nl", "eigena2_off", 0);
    let on = solve("eigena2.nl", "eigena2_on", 1);
    assert!(
        iters(&on) < iters(&off),
        "expected the rung to shorten the stall; {} with, {} without",
        iters(&on),
        iters(&off)
    );
}

/// `0` restores the pre-#818 hand-off. Without this the option could be
/// registered and ignored — the gh #677 failure mode, where
/// `limited_memory_initialization` was parsed, validated, and read nowhere.
#[test]
fn zero_restores_the_unconditional_restoration_handoff() {
    let off = solve("eigena2.nl", "eigena2_zero", 0);
    let on = solve("eigena2.nl", "eigena2_one", 1);
    assert_ne!(
        iters(&off),
        iters(&on),
        "limited_memory_ls_failure_restarts=0 took the same {} iterations as \
         the default 1 — the option is parsed but not reaching the algorithm",
        iters(&off)
    );
}
