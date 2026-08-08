//! Issue #541 regression: `eigenb2` (Vanderbei) is a degenerate NLP whose
//! reduced Hessian `Zᵀ W Z` collapses to singular along the run while the
//! KKT inertia stays correct, so the inertia test never asks for a Hessian
//! perturbation and the Newton step blows up along a direction of
//! numerically-zero curvature.
//!
//! The full diagnosis — including the measurement that POUNCE's inertia is
//! the *correct* one at the iteration the issue points at, and why the
//! `feral_singular_pivot_floor` default cannot simply be raised — is in
//! `dev-notes/issue-541-eigenb2-degenerate-reduced-hessian.md`.
//!
//! Two things are pinned here:
//!
//! * the default solve still lands on the right answer (`obj = 1.6`), so a
//!   future change to the perturbation handler or the FERAL inertia path
//!   cannot quietly turn a slow-but-correct solve into a wrong one;
//! * `feral_singular_pivot_floor=1e-8` — the documented recipe in
//!   `docs/src/troubleshooting.md` — certifies `Optimal` in materially
//!   fewer iterations than the default. A step-curvature guard that would
//!   fix this without a per-problem knob was prototyped and rejected —
//!   it regresses `jit1_node` from 24 to 246 iterations and pushes
//!   `cresc4` and `pooling_rt2stp` past the iteration cap (dev-note §7) —
//!   so the recipe is the only answer this problem has today, and it
//!   needs a test that fails if it stops working.

use std::path::PathBuf;
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

use pounce_cli::solve_report::SolveReport;
use pounce_nlp::ApplicationReturnStatus;

/// Exact optimum of `eigenb2`. Both POUNCE and the committed Ipopt-MA57
/// reference agree to 12+ digits.
const EIGENB2_OPTIMUM: f64 = 1.6;

/// Iteration count of the unmodified default solve at the time this test
/// was written (67). The recipe must stay comfortably under it — the point
/// of the recipe is that it removes the stall, not that it shaves a couple
/// of iterations off.
const DEFAULT_ITER_CEILING: i32 = 55;

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

/// Unique temp path per call — tests run in parallel in one process and
/// both cases below drive the same fixture.
fn tmp_path(suffix: &str) -> PathBuf {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let mut p = std::env::temp_dir();
    p.push(format!(
        "pounce_issue541_{}_{}_{suffix}",
        std::process::id(),
        n
    ));
    p
}

fn solve(extra_opts: &[&str]) -> SolveReport {
    let json_path = tmp_path("eigenb2.json");
    // Give the `.sol` an explicit destination; without one pounce writes
    // `eigenb2.sol` next to the input and pollutes the fixtures dir.
    let sol_path = tmp_path("eigenb2.sol");
    let mut cmd = Command::new(pounce_exe());
    cmd.arg(fixture("eigenb2.nl"))
        .arg(&sol_path)
        .arg("--json-output")
        .arg(&json_path);
    for opt in extra_opts {
        cmd.arg(opt);
    }
    // A "Solved To Acceptable Level" exit is non-zero; assert on the report.
    let _ = cmd.status().expect("spawn pounce");
    let text = std::fs::read_to_string(&json_path).expect("read json report");
    let _ = std::fs::remove_file(&json_path);
    let _ = std::fs::remove_file(&sol_path);
    serde_json::from_str(&text).expect("deserialize SolveReport")
}

fn assert_at_optimum(report: &SolveReport, ctx: &str) {
    let obj = report.solution.objective;
    assert!(
        (obj - EIGENB2_OPTIMUM).abs() < 1e-6,
        "{ctx}: objective {obj} is not the known optimum {EIGENB2_OPTIMUM} \
         (status={:?})",
        report.solution.status,
    );
}

/// The default solve is slow (67 iterations, `Solved To Acceptable Level`)
/// but it does converge to the right point. Nothing about the degeneracy
/// may turn that into a wrong answer or a hard failure.
#[test]
fn eigenb2_default_reaches_the_optimum() {
    let report = solve(&[]);
    assert_at_optimum(&report, "eigenb2 (defaults)");
}

/// The documented recipe: flagging the numerically rank-deficient KKT as
/// singular routes the IPM into `PerturbForSingularity` → `δ_x`, which caps
/// the null-direction step and removes the line-search stall. This must
/// certify `Optimal`, not merely "acceptable".
#[test]
fn eigenb2_singular_pivot_floor_recipe_certifies_optimal() {
    let report = solve(&["feral_singular_pivot_floor=1e-8"]);
    assert_at_optimum(&report, "eigenb2 (feral_singular_pivot_floor=1e-8)");

    // `SolveSucceeded` is "Optimal Solution Found"; the default run returns
    // `SolvedToAcceptableLevel` instead, which is exactly what the recipe is
    // supposed to lift.
    assert_eq!(
        report.solution.status,
        ApplicationReturnStatus::SolveSucceeded,
        "eigenb2 with feral_singular_pivot_floor=1e-8 should certify optimality, \
         got status={:?}",
        report.solution.status,
    );

    let iters = report.statistics.iteration_count;
    assert!(
        iters <= DEFAULT_ITER_CEILING,
        "eigenb2 with feral_singular_pivot_floor=1e-8 took {iters} iterations; \
         the recipe is supposed to remove the stall (was 39 vs 67 at the time \
         issue #541 was diagnosed, ceiling {DEFAULT_ITER_CEILING})",
    );
}
