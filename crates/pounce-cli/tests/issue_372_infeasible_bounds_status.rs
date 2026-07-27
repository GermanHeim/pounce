//! gh #372 — a trivially infeasible bounded NLP must be diagnosed as locally
//! infeasible (AMPL 200 range) at *any* user `tol`, not just the default.
//!
//! Fixture `issue_372_infeasible_bounds.nl` is the reporter's Pyomo model:
//!
//! ```text
//! min  x^3 + x^2   s.t.  x >= 0.7,   0 <= x <= 0.6
//! ```
//!
//! The contradiction is visible directly in the model — the bound caps `x` at
//! `0.6` while the constraint demands `x >= 0.7`. Ipopt 3.14.16 reports
//! `Converged to a locally infeasible point`; POUNCE reported
//! `Restoration_Failed`, which lands in the AMPL 500 *failure* range and
//! surfaces through Pyomo as `SolverStatus.error` /
//! `TerminationCondition.internalSolverError` — indistinguishable from a
//! solver implementation bug.
//!
//! The trigger was the reporter's `tol=1e-10`, not the model. At the default
//! `tol=1e-8` the restoration inner sub-IPM converges (`Success`) and the
//! `strict` gate in `resto_inner_solver.rs` correctly declares local
//! infeasibility. At `tol <= 1e-10` the inner reaches the same stationary
//! point of the feasibility problem (`inner_kkt_err ~ 1.2e-10`,
//! `orig_inf_pr = 1.0e-1`) but can no longer certify it against its own,
//! equally tightened, convergence test — every remaining step is below the
//! tiny-step threshold, so it exits `StopAtTinyStep`. No gate admitted that
//! status, so the diagnosis was thrown away and the solve fell through to
//! `Restoration_Failed`. The `tiny_step_locally_infeasible` gate closes it.
//!
//! Regression coverage requested in the issue:
//!   1. this one-variable `.nl` shape in the restoration tests,
//!   2. an infeasible (non-internal) status, and
//!   3. no candidate solution advertised as optimal.

use std::path::PathBuf;
use std::process::Command;

fn pounce_exe() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_pounce"))
}

fn fixture() -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("tests");
    p.push("fixtures");
    p.push("issue_372_infeasible_bounds.nl");
    p
}

fn solve_result_num(text: &str) -> i32 {
    for line in text.lines() {
        if let Some(rest) = line.trim().strip_prefix("objno ") {
            if let Some(code) = rest.split_whitespace().nth(1) {
                return code.parse().expect("objno code parses");
            }
        }
    }
    panic!("no `objno` line in .sol:\n{text}");
}

/// Solve the fixture under `-AMPL` with `extra` options appended, returning
/// the `.sol` text. `tag` keeps concurrent test cases off each other's files.
fn solve(tag: &str, extra: &[&str]) -> String {
    let sol = std::env::temp_dir().join(format!("pounce_issue372_{tag}.sol"));
    let _ = std::fs::remove_file(&sol);

    let out = Command::new(pounce_exe())
        .arg(fixture())
        .arg("-AMPL")
        .arg("--sol-output")
        .arg(&sol)
        .args(extra)
        .output()
        .expect("spawn pounce");

    // Under `-AMPL` the termination travels in the file, so exit stays 0.
    assert_eq!(out.status.code(), Some(0), "-AMPL must exit 0");

    std::fs::read_to_string(&sol).expect("read .sol")
}

/// The core regression: the reporter's exact option set must produce the
/// AMPL *infeasible* range, not the *failure* range.
#[test]
fn reporter_option_set_is_infeasible_not_failure() {
    let text = solve(
        "reporter",
        &[
            "print_level=0",
            "tol=1.0e-10",
            "bound_relax_factor=0.0",
            "honor_original_bounds=yes",
            "max_cpu_time=30.0",
        ],
    );
    let srn = solve_result_num(&text);

    assert!(
        (200..300).contains(&srn),
        "expected the AMPL infeasible range (200..299) for `0 <= x <= 0.6, \
         x >= 0.7`, got solve_result_num={srn}. The 500 failure range makes \
         Pyomo report TerminationCondition.internalSolverError, which client \
         code cannot distinguish from a solver bug:\n{text}"
    );
    assert!(
        !text.contains("RestorationFailed"),
        "a one-inequality contradiction must not be reported as a restoration \
         failure:\n{text}"
    );
    assert!(
        text.contains("InfeasibleProblemDetected"),
        "expected the InfeasibleProblemDetected verdict:\n{text}"
    );
}

/// The diagnosis must not depend on `tol`. `1e-8` is the default (which
/// already worked); `1e-10` and `1e-12` are past the point where the
/// restoration inner sub-IPM can still exit `Success`.
#[test]
fn diagnosis_is_stable_across_tolerances() {
    for (tag, tol) in [
        ("t6", "1e-6"),
        ("t8", "1e-8"),
        ("t10", "1e-10"),
        ("t12", "1e-12"),
    ] {
        let text = solve(tag, &["print_level=0", &format!("tol={tol}")]);
        let srn = solve_result_num(&text);
        assert!(
            (200..300).contains(&srn),
            "tol={tol}: expected the AMPL infeasible range (200..299), got \
             solve_result_num={srn}. A tighter user tolerance must not turn a \
             certified infeasibility into a restoration failure:\n{text}"
        );
    }
}

/// Deliverable 3 from the issue: the nonoptimal result must never be
/// advertised in the AMPL *solved* family, which is what makes Pyomo load the
/// returned point as `optimal`.
#[test]
fn never_reported_as_solved() {
    let text = solve("solved_family", &["print_level=0", "tol=1.0e-10"]);
    let srn = solve_result_num(&text);

    assert!(
        !(0..200).contains(&srn),
        "an infeasible model reported in the AMPL solved family \
         (solve_result_num={srn}); 0..99 = solved and 100..199 = \
         solved-with-warning both make Pyomo report \
         TerminationCondition.optimal:\n{text}"
    );
}

/// Outside `-AMPL` the console verdict must say local infeasibility and the
/// process must exit non-zero, matching what the library reports.
#[test]
fn cli_reports_local_infeasibility_and_exits_nonzero() {
    let out = Command::new(pounce_exe())
        .arg(fixture())
        .arg("--no-sol")
        .arg("tol=1.0e-10")
        .output()
        .expect("spawn pounce");

    let combined = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );

    assert!(
        combined.contains("point of local infeasibility"),
        "expected the local-infeasibility verdict:\n{combined}"
    );
    assert!(
        !combined.contains("Restoration Failed"),
        "must not report a restoration failure for a visible \
         contradiction:\n{combined}"
    );
    assert_ne!(
        out.status.code(),
        Some(0),
        "an infeasible solve must exit non-zero outside -AMPL mode"
    );
}
