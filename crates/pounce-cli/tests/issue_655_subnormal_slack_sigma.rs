//! gh#655 end to end: a solve must not report `SolveSucceeded` at a point
//! whose slack has underflowed far enough that `Σ = z/s` is `inf`.
//!
//! The reported settings — `mu_strategy=adaptive`, `tol = compl_inf_tol =
//! 1e-306`, `mu_min = 5e-324` — reach exactly that on `linear_eq_collapsed_box`,
//! which carries an active bound with a nonzero multiplier. Before the fix the
//! solve stopped at iteration 20 with
//!
//! ```text
//!   z          = 84.375
//!   slack      = 4.209e-310      (subnormal)
//!   Sigma      = z/slack         = inf
//!   final_compl = z*slack        = 3.551e-308   ->  under compl_inf_tol
//!   status     = SolveSucceeded
//! ```
//!
//! The complementarity that cleared the tolerance was the product of a slack
//! nothing downstream can divide by: the same `s` puts `inf` on the KKT
//! diagonal, and from there into every backsolve the sensitivity path makes.
//!
//! `calculate_safe_slack` now floors `s` at `max_i z_i / (f64::MAX/4)`, so the
//! slack at that point is `1.877e-306`, `Σ` is `4.494e307`, and the honest
//! complementarity is `1.584e-304` — which does *not* clear a `compl_inf_tol`
//! of `1e-306`. The solve reports the stall it actually reached rather than a
//! success, matching the issue's own table at `tol = 1e-308`.
//!
//! The trajectory is untouched: same 20 iterations, same objective to the last
//! bit, same factorization counts. Only the final verdict moves, and only under
//! tolerances this far into the subnormal range — `scripts/sweep-fixtures.sh`
//! at default settings is byte-identical across the whole corpus.

use std::path::PathBuf;
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

use pounce_cli::solve_report::SolveReport;
use pounce_nlp::ApplicationReturnStatus;

/// The issue's option set, verbatim.
const PATHOLOGICAL: &[&str] = &[
    "mu_strategy=adaptive",
    "tol=1e-306",
    "compl_inf_tol=1e-306",
    "mu_min=5e-324",
];

fn solve(extra: &[&str]) -> SolveReport {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!("pounce_issue655_{}_{n}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("scratch dir");
    let json_path = dir.join("m.json");
    let sol_path = dir.join("m.sol");

    let mut fixture = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    fixture.push("tests/fixtures/linear_eq_collapsed_box.nl");

    let mut cmd = Command::new(PathBuf::from(env!("CARGO_BIN_EXE_pounce")));
    cmd.arg(&fixture)
        .arg(&sol_path)
        .arg("--json-output")
        .arg(&json_path)
        // Hang guard only; nothing below asserts an iteration count.
        .arg("max_iter=200");
    for o in extra {
        cmd.arg(o);
    }
    let _ = cmd.status().expect("spawn pounce");
    let text = std::fs::read_to_string(&json_path).expect("read json report");
    let _ = std::fs::remove_file(&json_path);
    let _ = std::fs::remove_file(&sol_path);
    serde_json::from_str(&text).expect("deserialize SolveReport")
}

#[test]
fn a_solve_that_bottoms_out_on_a_subnormal_slack_does_not_report_success() {
    let r = solve(PATHOLOGICAL);

    // The answer is the same one the pre-fix run reported; the floor moves a
    // slack, not the iterate.
    assert!(
        (r.statistics.final_objective - 161.999_997_84).abs() < 1e-6,
        "objective moved: {}",
        r.statistics.final_objective
    );

    // The point of the fix. `final_compl` is now `z·s` on a slack that can be
    // divided by, and at `1.6e-304` it does not meet the `1e-306` the caller
    // asked for — so the solve may not call itself converged.
    assert!(
        r.statistics.final_compl > 1e-306,
        "complementarity {} clears compl_inf_tol=1e-306, which at this point \
         is only reachable on a slack too small to divide z by (gh#655)",
        r.statistics.final_compl
    );
    assert_ne!(
        r.solution.status,
        ApplicationReturnStatus::SolveSucceeded,
        "reported success with a complementarity of {} against \
         compl_inf_tol=1e-306 (gh#655)",
        r.statistics.final_compl
    );
}

#[test]
fn the_same_fixture_still_converges_at_default_tolerances() {
    // The guard above must not have cost the ordinary solve anything: nothing
    // in this fixture is near the overflow edge at default `tol`.
    let r = solve(&[]);
    assert_eq!(r.solution.status, ApplicationReturnStatus::SolveSucceeded);
    assert!(
        (r.statistics.final_objective - 161.999_997_84).abs() < 1e-6,
        "objective moved: {}",
        r.statistics.final_objective
    );
}
