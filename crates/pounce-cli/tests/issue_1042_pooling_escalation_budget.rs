//! gh#1042 R2b: `pooling_rt2stp` (exact Hessian) gets its iteration-4
//! refinement escalation back once the inertia ladder and iterative
//! refinement each have their own quality-escalation budget per matrix.
//!
//! On 0.13.0 (FERAL 0.21) the first factor of the iteration-4 ladder is a
//! matrix singular to working precision whose count reads 71 of 72
//! negatives. The too-few-negatives branch spent the matrix's single
//! escalation flag (upstream's `augsys_improved_`) on it; the gh#592
//! walk-back then accepted a `δ_c`-free, numerically singular factor whose
//! step has a dual part of 6.4e16, refinement stalled on it, could not
//! escalate, and kept the step through the `S` exit. `W` was indefinite by
//! 1e13 one iteration later and the solve took 492 iterations with 24
//! restoration calls (0.12.0: 162 / 3). With the per-consumer budget
//! (`EscalationBudget` in `pd_full_space_solver.rs`) refinement escalates
//! and refactors, and on x86_64 the solve takes 109 iterations with 3
//! restoration calls, landing on the global optimum.
//!
//! The bounds are the plan's acceptance bounds (≤ 300 iterations, < 24
//! restoration calls, either certified optimum), not the measured
//! 109 / 3, so an honest cross-architecture trajectory difference does not
//! fail it; restoring the shared flag does (492 / 24). The `< 500` pin in
//! `issue_592_delta_c_walkback.rs` guards a different régime (gh#544's 812)
//! and is left where it is.

use std::path::PathBuf;
use std::process::Command;

fn pounce_exe() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_pounce"))
}

fn fixture() -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("tests");
    p.push("fixtures");
    p.push("pooling_rt2stp.nl");
    p
}

/// MINLPLib's global optimum, and the local one Ipopt reports. Which one a
/// run lands in is a rounding fact on this nonconvex model (see
/// `issue_592_delta_c_walkback.rs`), so either is accepted.
const POOLING_OPTIMUM: f64 = -4391.8260;
const POOLING_LOCAL_OPTIMUM: f64 = -3273.9549;

#[test]
fn pooling_refinement_keeps_its_escalation_after_the_inertia_branch_spends_one() {
    let json = std::env::temp_dir().join(format!(
        "pounce_issue_1042_pooling_r2b_{}.json",
        std::process::id()
    ));
    let _ = std::fs::remove_file(&json);

    let out = Command::new(pounce_exe())
        .arg(fixture())
        .arg("--no-sol")
        .arg("--json-output")
        .arg(&json)
        .arg("max_wall_time=300")
        .output()
        .expect("spawn pounce");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("EXIT: Optimal Solution Found."),
        "pooling_rt2stp did not solve:\n{stdout}"
    );
    assert!(
        !stdout.contains("second opinion"),
        "pooling_rt2stp needed the second-opinion ladder:\n{stdout}"
    );

    let text = std::fs::read_to_string(&json).expect("read JSON report");
    let _ = std::fs::remove_file(&json);
    let report: serde_json::Value = serde_json::from_str(&text).expect("parse JSON report");
    let obj = report["statistics"]["final_objective"]
        .as_f64()
        .expect("final_objective");
    assert!(
        (obj - POOLING_OPTIMUM).abs() < 1e-3 || (obj - POOLING_LOCAL_OPTIMUM).abs() < 1e-3,
        "objective {obj} is neither certified optimum"
    );
    let iters = report["statistics"]["iteration_count"]
        .as_u64()
        .expect("iteration_count");
    assert!(
        iters <= 300,
        "pooling_rt2stp took {iters} iterations (0.12.0: 162; 0.13.0 with one \
         escalation flag shared by the inertia ladder and refinement: 492)"
    );

    // `Number of restoration iterations = N (in K calls)`; absent when
    // restoration never ran.
    let calls = stdout
        .lines()
        .find(|l| l.starts_with("Number of restoration iterations"))
        .and_then(|l| l.split("(in ").nth(1))
        .and_then(|r| r.split_whitespace().next())
        .map(|k| k.parse::<u64>().expect("restoration call count"))
        .unwrap_or(0);
    assert!(
        calls < 24,
        "pooling_rt2stp entered restoration {calls} times (0.13.0 with the \
         shared flag: 24; 0.12.0: 3)"
    );
}
