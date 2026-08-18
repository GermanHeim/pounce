//! The inertia test must not be run against a factorization that cannot
//! answer it (gh #540).
//!
//! WHAT WENT WRONG. CUTE `eigena2` stopped at `Solved To Acceptable Level`
//! with the dual infeasibility stuck at 3.3e-7, where Ipopt certifies
//! `Optimal` at 5.2e-9. The objective was already right to twelve digits;
//! only the last two orders of the dual residual were missing. The reported
//! symptom was that `δ_w` re-escalated from `10^-0.8` to `10^1.4` at the
//! second-to-last iteration and damped the Newton step from `1.2e-7` to
//! `8.2e-9`, which is where the superlinear tail went.
//!
//! WHY. Not the `δ_w` update rule — that is an exact port of upstream's
//! `get_deltas_for_wrong_inertia` and it did what it is told. The input was
//! wrong. `eigena2`'s constraint Jacobian degenerates as the iterate
//! converges (45 of its 55 singular values fall to ~1e-8 by iteration 27, on
//! a `‖A‖ ≈ 240` KKT), so every factorization taken with `δ_c = 0` down that
//! tail is singular to working precision, with a smallest pivot at ~1e-16.
//! The negative-eigenvalue count read off such a factor is noise: the same
//! iterate returned 64, 58 and 62 against an expected 55, and a LAPACK
//! eigendecomposition of the dumped matrices agrees with none of those. The
//! `δ_w` ladder then escalated ×8 per retry against a reading that does not
//! respond to `δ_w` at all. The perturbation that *does* repair a
//! rank-deficient constraint block is `δ_c` — and `δ_c` is exactly what the
//! `Singular` verdict reaches for. The fix routes there:
//! `feral_inertia_pivot_floor` reports `Singular` instead of `WrongInertia`
//! when the count is contradicted by a working-precision pivot. Applying
//! `δ_c` lifts the smallest pivot from ~1e-16 to 5.8e-9, and from that point
//! the counts feral reports agree with LAPACK exactly.
//!
//! The trigger only ever fires on a factorization the caller was already
//! going to reject, so it cannot turn a usable factor into a failure; that
//! property is pinned as a unit test in `pounce-feral`
//! (`well_conditioned_inertia_mismatch_still_reports_wrong_inertia`).
//!
//! FIXTURE PROVENANCE. `fixtures/eigena2.nl` is CUTE `EIGENA2` (the
//! symmetric eigenvalue problem posed as an equality-constrained NLP; 110
//! variables, 55 equality constraints) from Vanderbei's CUTE-in-AMPL
//! collection, the same `.nl` the reporter ran, attached to gh #540 because
//! the benchmark archive itself is gitignored and not in the checkout. On
//! this build the pre-fix behaviour reproduces the issue's numbers exactly:
//! 29 iterations, dual infeasibility 3.3144912958031115e-07, constraint
//! violation 2.4123147923660326e-11, `Solved To Acceptable Level`.

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
        "pounce_issue540_{}_{}_{suffix}",
        std::process::id(),
        n
    ));
    p
}

fn solve(extra: &[&str]) -> SolveReport {
    solve_with_env(extra, &[])
}

fn solve_with_env(extra: &[&str], env: &[(&str, &str)]) -> SolveReport {
    let json_path = tmp_path("eigena2.json");
    let sol_path = tmp_path("eigena2.sol");
    let mut cmd = Command::new(pounce_exe());
    cmd.arg(fixture("eigena2.nl"))
        .arg(&sol_path)
        .arg("--json-output")
        .arg(&json_path);
    for o in extra {
        cmd.arg(o);
    }
    for (k, v) in env {
        cmd.env(k, v);
    }
    let _ = cmd.status().expect("spawn pounce");
    let text = std::fs::read_to_string(&json_path).expect("read json report");
    let _ = std::fs::remove_file(&json_path);
    let _ = std::fs::remove_file(&sol_path);
    serde_json::from_str(&text).expect("deserialize SolveReport")
}

/// The pre-#540 behaviour: every mismatching count is taken at face value and
/// answered with `δ_w`, however small the pivot it was read off.
const NO_TRIGGER: [&str; 1] = ["feral_inertia_pivot_floor=0"];

/// The headline: `eigena2` now earns a strict certificate rather than the
/// acceptable-level fallback.
#[test]
fn eigena2_converges_to_optimal() {
    let r = solve(&[]);
    assert_eq!(
        r.solution.status,
        ApplicationReturnStatus::SolveSucceeded,
        "eigena2 did not reach a strict certificate (status {:?}, dual inf {:e})",
        r.solution.status,
        r.statistics.final_dual_inf,
    );
}

/// ...and it is the certificate that improved, not the tolerance that moved:
/// the dual residual has to actually clear `tol = 1e-8`, which is the two
/// orders of magnitude the issue is about. Ipopt reaches 9.3e-9 here.
#[test]
fn eigena2_dual_infeasibility_clears_the_strict_tolerance() {
    let dual = solve(&[]).statistics.final_dual_inf;
    assert!(
        dual < 1e-8,
        "dual infeasibility {dual:e} did not clear tol = 1e-8",
    );
}

/// The guard against a vacuous pass: with the trigger disabled this build
/// must still reproduce the reported failure. If a later change fixes
/// `eigena2` by some other route, this test fails and says so rather than
/// letting the two tests above pass for a reason they do not describe.
///
/// **`POUNCE_DBG_NO_QUAD=1` is part of the reproduction, as of gh #588's Q4.**
/// The pre-#540 route is a knife edge — it turns on an inertia count read off
/// a factor whose smallest pivot is ~1e-16 — and Q4 moved `∇²L[8, 8]` by
/// **one ulp** on this model. `eigena2` has 55 quadratic rows and a
/// non-quadratic objective, so that entry is now assembled as a constant
/// scatter plus a decoded tape share rather than as one sum, and the two
/// associate differently. With the fast path on, `feral_inertia_pivot_floor=0`
/// converges (27 iterations, dual 6.0e-10) instead of stalling, and this
/// guard would pass vacuously in the other direction — by having nothing left
/// to reproduce.
///
/// The env var restores the pre-Q4 arithmetic exactly: 29 iterations, dual
/// 1.841e-7, `SolvedToAcceptableLevel`, the same numbers as before that
/// commit. **The shipped route is untouched** — with the trigger on,
/// `eigena2` takes 26 iterations to `SolveSucceeded` at dual 2.211e-9 before
/// Q4 and 2.208e-9 after, on the same objective. So what changed is the
/// reproducibility of a failure, not the fix and not the answer.
#[test]
fn disabling_the_trigger_reproduces_the_reported_failure() {
    let r = solve_with_env(&NO_TRIGGER, &[("POUNCE_DBG_NO_QUAD", "1")]);
    assert_eq!(
        r.solution.status,
        ApplicationReturnStatus::SolvedToAcceptableLevel,
        "the pre-#540 route no longer reproduces the issue, so the tests \
         above are no longer pinning the fix they describe",
    );
    let dual = r.statistics.final_dual_inf;
    assert!(
        dual > 1e-7,
        "expected the stuck ~3.3e-7 dual residual from the issue, got {dual:e}",
    );
}
