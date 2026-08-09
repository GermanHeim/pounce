//! Issue #557 end-to-end coverage: a model with a CSE (`.nl` defined
//! variable) shared across many constraints, solved through the full
//! CLI — no repository fixture had one, so the shared-CSE tape paths
//! (`eval_g` always; `eval_jac_g` / `eval_h` above their op-ratio
//! gates) were exercised by unit tests only.
//!
//! The generated model is
//!
//! ```text
//! min  x0 + x1 + Σ_i x_{i+2}
//! s.t. S · x_{i+2} >= 1        i = 0..m,   S = (log ∘ exp)^p(2 + 0.01 (x0 + x1))
//!      x0, x1 ∈ [0.5, 5],  x_{i+2} ∈ [1e-3, 1e4]
//! ```
//!
//! The body is mathematically the identity chain, so `S = 2 + 0.01 (x0 +
//! x1)` and the optimum is analytic — `x0 = x1 = 0.5` (their objective
//! coefficient dominates the constraint relief they buy), every
//! constraint active at `x_{i+2} = 1 / 2.01` — while the AD still walks
//! `2p` transcendental ops with nonzero curvature per body reference.
//! With `p = 20` the flat/shared op ratio is ≈ 8, past both the Jacobian
//! (4) and Hessian (2.5) gates, so the default run drives all three
//! hybrid evaluators; the `POUNCE_DBG_NO_HYBRID=1` run pins the flat
//! tapes as the reference. Both must reach the same optimum.

use std::path::PathBuf;
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

use pounce_cli::solve_report::SolveReport;

fn pounce_exe() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_pounce"))
}

/// Unique temp path per call (tests run in parallel in one process).
fn tmp_path(suffix: &str) -> PathBuf {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let mut p = std::env::temp_dir();
    p.push(format!(
        "pounce_issue557_{}_{}_{suffix}",
        std::process::id(),
        n
    ));
    p
}

const M: usize = 16;
const PAIRS: usize = 20;

/// `.nl` text for the model above: one `V` segment referenced by all
/// `m` constraint rows.
fn shared_cse_model_nl(m: usize, pairs: usize) -> String {
    let n = m + 2;
    let nzc = 3 * m;
    let mut s = String::new();
    s.push_str("g3 1 1 0\n");
    s.push_str(&format!(" {n} {m} 1 0 0 0\n"));
    s.push_str(&format!(" {m} 0\n 0 0\n"));
    s.push_str(&format!(" {n} 0 0\n"));
    s.push_str(" 0 0 0 1\n 0 0 0 0 0\n");
    s.push_str(&format!(" {nzc} {n}\n"));
    s.push_str(" 0 0\n 0 1 0 0 0\n");
    // Shared body: (log ∘ exp)^pairs(2 + 0.01 (x0 + x1)).
    s.push_str(&format!("V{n} 0 0\n"));
    for _ in 0..pairs {
        s.push_str("o43\no44\n");
    }
    s.push_str("o0\nn2\no2\nn0.01\no0\nv0\nv1\n");
    // Rows: S * x_{i+2} >= 1.
    for i in 0..m {
        s.push_str(&format!("C{i}\no2\nv{n}\nv{}\n", i + 2));
    }
    s.push_str("O0 0\nn0\n");
    s.push_str(&format!("x{n}\n"));
    for j in 0..n {
        s.push_str(&format!("{j} 1.0\n"));
    }
    s.push_str("r\n");
    for _ in 0..m {
        s.push_str("2 1\n");
    }
    s.push_str("b\n");
    s.push_str("0 0.5 5\n0 0.5 5\n");
    for _ in 0..m {
        s.push_str("0 0.001 10000\n");
    }
    // Cumulative Jacobian column counts: cols 0 and 1 in every row,
    // col i+2 in one.
    s.push_str(&format!("k{}\n", n - 1));
    let mut acc = 0;
    for j in 0..n - 1 {
        acc += if j < 2 { m } else { 1 };
        s.push_str(&format!("{acc}\n"));
    }
    for i in 0..m {
        s.push_str(&format!("J{i} 3\n0 0.0\n1 0.0\n{} 0.0\n", i + 2));
    }
    // Linear objective: every variable with coefficient 1.
    s.push_str(&format!("G0 {n}\n"));
    for j in 0..n {
        s.push_str(&format!("{j} 1.0\n"));
    }
    s
}

/// Solve the generated model, optionally with `POUNCE_DBG_NO_HYBRID=1`,
/// returning the parsed report.
fn solve(no_hybrid: bool) -> SolveReport {
    let nl_path = tmp_path("model.nl");
    std::fs::write(&nl_path, shared_cse_model_nl(M, PAIRS)).expect("write model");
    let json_path = tmp_path("report.json");
    let sol_path = tmp_path("model.sol");
    let mut cmd = Command::new(pounce_exe());
    cmd.arg(&nl_path)
        .arg(&sol_path)
        .arg("--json-output")
        .arg(&json_path);
    if no_hybrid {
        cmd.env("POUNCE_DBG_NO_HYBRID", "1");
    }
    let status = cmd.status().expect("spawn pounce");
    assert!(
        status.success(),
        "solve exited nonzero (no_hybrid={no_hybrid})"
    );
    let text = std::fs::read_to_string(&json_path).expect("read json report");
    let _ = std::fs::remove_file(&nl_path);
    let _ = std::fs::remove_file(&json_path);
    let _ = std::fs::remove_file(&sol_path);
    serde_json::from_str(&text).expect("deserialize SolveReport")
}

/// Analytic optimum: x0 = x1 = 0.5, x_{i+2} = 1 / 2.01.
fn expected_objective() -> f64 {
    1.0 + M as f64 / 2.01
}

fn assert_solved_at_optimum(report: &SolveReport, ctx: &str) {
    let code = report.solution.solve_result_num;
    assert!(
        (0..100).contains(&code),
        "{ctx}: not solved (solve_result_num={code}, status={:?})",
        report.solution.status,
    );
    let obj = report.solution.objective;
    let want = expected_objective();
    assert!(
        (obj - want).abs() < 1e-4 * want,
        "{ctx}: objective {obj} is not the analytic optimum {want}",
    );
}

/// The default run takes the shared-CSE paths (the op ratio is past both
/// derivative gates) and must reach the analytic optimum.
#[test]
fn shared_cse_model_solves_on_the_hybrid_paths() {
    let report = solve(false);
    assert_solved_at_optimum(&report, "shared-CSE model (hybrid)");
}

/// Same binary, hybrid disabled: the flat-tape reference solve, and the
/// two answers must agree — a wrong hybrid Hessian would surface here as
/// a different (or failed) solution, not a panic.
#[test]
fn shared_cse_model_matches_the_flat_tape_solve() {
    let hybrid = solve(false);
    let flat = solve(true);
    assert_solved_at_optimum(&flat, "shared-CSE model (flat reference)");
    let (oh, of) = (hybrid.solution.objective, flat.solution.objective);
    assert!(
        (oh - of).abs() <= 1e-6 * of.abs().max(1.0),
        "hybrid ({oh}) and flat ({of}) solves disagree"
    );
}
