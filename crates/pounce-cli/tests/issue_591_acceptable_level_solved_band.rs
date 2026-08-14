//! Issue #591 regression: an accepted (reduced-accuracy) solve must leave
//! AMPL's *solved* band in the `.sol`, with Ipopt's own code.
//!
//! `Solved_To_Acceptable_Level` used to be written as `solve_result_num =
//! 100`. Nothing reads that number in isolation — consumers key on the band —
//! and the two bands are not equivalent: Pyomo's legacy `.sol` reader turns
//! `0..=99` into `SolverStatus.ok` and `100..=199` into `SolverStatus.warning`,
//! both with `TerminationCondition.optimal`. So an accepted POUNCE solve loaded
//! as
//!
//! ```text
//! solver.status          warning
//! termination_condition  optimal
//! message                POUNCE 0.10.0: SolvedToAcceptableLevel
//! ```
//!
//! and Pyomo logged "Loading a SolverResults object with a warning status",
//! while the equivalent Ipopt solve ("Solved To Acceptable Level.") loaded as
//! `status=ok`. Ipopt's ASL driver emits `1` for `STOP_AT_ACCEPTABLE_POINT`
//! (`Ipopt/src/Apps/AmplSolver/AmplTNLP.cpp`), so POUNCE does too — which fixes
//! the plugin route and the route where the POUNCE binary is driven through
//! Pyomo's generic `ipopt` ASL interface alike, since both read this file.
//!
//! The scientific distinction is not lost: it stays in the status name on the
//! `.sol` message line, in `solve_result_num` itself (`1`, not `0`), and in the
//! JSON report's status field.

use std::path::PathBuf;
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

use pounce_cli::solve_report::SolveReport;
use pounce_nlp::return_codes::ApplicationReturnStatus;

fn pounce_exe() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_pounce"))
}

fn tmp_path(suffix: &str) -> PathBuf {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let mut p = std::env::temp_dir();
    p.push(format!(
        "pounce_issue591_{}_{}_{suffix}",
        std::process::id(),
        n
    ));
    p
}

/// Solve a built-in problem and return `(report, .sol text)`.
fn solve(problem: &str, extra_opts: &[&str]) -> (SolveReport, String) {
    let json_path = tmp_path(&format!("{problem}.json"));
    let sol_path = tmp_path(&format!("{problem}.sol"));
    let mut cmd = Command::new(pounce_exe());
    cmd.arg("--problem")
        .arg(problem)
        .arg("--sol-output")
        .arg(&sol_path)
        .arg("--json-output")
        .arg(&json_path)
        .arg("print_level=0");
    for opt in extra_opts {
        cmd.arg(opt);
    }
    let _ = cmd.status().expect("spawn pounce");
    let json = std::fs::read_to_string(&json_path).expect("read json report");
    let sol = std::fs::read_to_string(&sol_path).expect("read .sol");
    let _ = std::fs::remove_file(&json_path);
    let _ = std::fs::remove_file(&sol_path);
    (
        serde_json::from_str(&json).expect("deserialize SolveReport"),
        sol,
    )
}

/// The `objno <objno> <solve_result_num>` line every `.sol` ends with.
fn objno_code(sol: &str) -> i32 {
    let line = sol
        .lines()
        .find(|l| l.starts_with("objno "))
        .unwrap_or_else(|| panic!("no objno line in .sol:\n{sol}"));
    line.split_whitespace()
        .nth(2)
        .and_then(|t| t.parse().ok())
        .unwrap_or_else(|| panic!("unparsable objno line {line:?}"))
}

/// `tol` below anything reachable plus a generous `acceptable_tol` routes the
/// solve through the acceptable-level fallback — the same recipe as the
/// `optimize_hs71` and `pounce.minimize` (#119) acceptable-level tests.
const FORCE_ACCEPTABLE: [&str; 3] = ["tol=1e-30", "acceptable_tol=1e-4", "acceptable_iter=1"];

#[test]
fn acceptable_level_writes_ipopts_code_in_the_solved_band() {
    let (report, sol) = solve("rosenbrock", &FORCE_ACCEPTABLE);
    assert_eq!(
        report.solution.status,
        ApplicationReturnStatus::SolvedToAcceptableLevel,
        "fixture no longer reaches the acceptable-level fallback, so this test \
         is not exercising #591 — .sol was:\n{sol}",
    );

    let code = objno_code(&sol);
    assert_eq!(code, 1, "Ipopt's STOP_AT_ACCEPTABLE_POINT code");
    assert!(
        (0..=99).contains(&code),
        "an accepted solve must be in AMPL's solved band, which Pyomo's legacy \
         .sol reader loads as status=ok; {code} is not (gh #591)",
    );
    // The `.sol` and the JSON report cannot disagree about the same run.
    assert_eq!(report.solution.solve_result_num, code);

    // Reduced accuracy is still legible to the caller: the status name reaches
    // the message line verbatim, exactly as before the code change.
    let message = sol.lines().next().unwrap_or_default();
    assert!(
        message.contains("SolvedToAcceptableLevel"),
        "the acceptable-level distinction must stay visible in the message, \
         got {message:?}",
    );
}

/// The control: a full-accuracy solve still reports `0`, so the two outcomes
/// remain distinguishable by code and not merely by message.
#[test]
fn a_strict_solve_still_writes_zero() {
    let (report, sol) = solve("rosenbrock", &[]);
    assert_eq!(
        report.solution.status,
        ApplicationReturnStatus::SolveSucceeded,
    );
    assert_eq!(objno_code(&sol), 0);
}
