//! Issue #541 regression: `eigenb2` (Vanderbei) is a degenerate NLP whose
//! reduced Hessian `Zᵀ W Z` collapses to singular along the run — its
//! smallest eigenvalue falls from `1.4e+02` to `7.2e-13` — so the Newton
//! step aligns almost entirely with a direction of numerically-zero
//! curvature. Because the constraints are quadratic that step grows the
//! constraint violation sixfold, the filter cuts `alpha` to `1/8 … 1/128`,
//! and the second-order correction diverges rather than repairing it.
//!
//! **Fixed by #544** (`feral_inertia_pivot_floor`), which is why the
//! default solve below certifies `Optimal`. The KKT is singular to working
//! precision down the whole tail, so its negative-eigenvalue count is
//! noise: FERAL reports 43…64 against an expected 55 across 20
//! factorizations of the pre-fix run, and the old code answered each of
//! those with `δ_w` — 11 iterations of the failing run carried a nonzero
//! `lg(rg)`, escalating again over iterations 61-67. #544 routes an
//! unmeasurable inertia test to `δ_c` instead; its trigger fires 15 times
//! on this model, more than on `eigena2`, the model it was written for.
//!
//! The full diagnosis — including why the `feral_singular_pivot_floor`
//! default cannot simply be raised — is in
//! `dev-notes/issue-541-eigenb2-degenerate-reduced-hessian.md`.
//!
//! Two things are pinned here:
//!
//! * the default solve certifies `Optimal` at `obj = 1.6`. This is the
//!   regression test for #544's second issue: #544 itself pins `eigena2`,
//!   and found this model only through a corpus sweep. If the inertia
//!   routing regresses, this fails.
//! * `feral_singular_pivot_floor=1e-8` — the tuning note in
//!   `docs/src/troubleshooting.md` — still reaches the optimum in
//!   materially fewer iterations (39 against 68). It is no longer needed
//!   for correctness, but it remains the fastest route through this
//!   model's degeneracy and should not silently stop working. A
//!   step-curvature guard that would have fixed this without any knob was
//!   prototyped and rejected — it regresses `jit1_node` from 24 to 246
//!   iterations and pushes `cresc4` and `pooling_rt2stp` past the
//!   iteration cap (dev-note §7).

use std::path::PathBuf;
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

use pounce_cli::solve_report::SolveReport;
use pounce_nlp::ApplicationReturnStatus;

/// Exact optimum of `eigenb2`. Both POUNCE and the committed Ipopt-MA57
/// reference agree to 12+ digits.
const EIGENB2_OPTIMUM: f64 = 1.6;

/// The default solve takes 68 iterations post-#544. The tuning knob must
/// stay comfortably under that (it takes 39) — the point of it is that it
/// removes the stall, not that it shaves a couple of iterations off.
const RECIPE_ITER_CEILING: i32 = 55;

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

/// Under defaults this certifies `Optimal` in 68 iterations. Before #544 it
/// returned `Solved To Acceptable Level` in 67 — the right point, without a
/// certificate. This is the regression test for the `eigenb2` half of #544:
/// if the inertia-noise routing regresses, the status drops back and this
/// fails.
#[test]
fn eigenb2_default_certifies_optimal() {
    let report = solve(&[]);
    assert_at_optimum(&report, "eigenb2 (defaults)");

    assert_eq!(
        report.solution.status,
        ApplicationReturnStatus::SolveSucceeded,
        "eigenb2 should certify optimality under defaults since #544 \
         (feral_inertia_pivot_floor); got status={:?}. A drop back to \
         SolvedToAcceptableLevel means the unmeasurable-inertia test is \
         being answered with delta_w again.",
        report.solution.status,
    );
}

/// The tuning knob: flagging the numerically rank-deficient KKT as singular
/// routes the IPM into `PerturbForSingularity` → `δ_x`, which caps the
/// null-direction step and removes the line-search stall outright rather
/// than correcting it per-factorization the way #544 does. Since #544 this
/// is a speedup (39 against 68), not a fix — but it is the same degeneracy
/// being addressed, and it should not silently stop working.
#[test]
fn eigenb2_singular_pivot_floor_reaches_the_optimum_faster() {
    let report = solve(&["feral_singular_pivot_floor=1e-8"]);
    assert_at_optimum(&report, "eigenb2 (feral_singular_pivot_floor=1e-8)");

    assert_eq!(
        report.solution.status,
        ApplicationReturnStatus::SolveSucceeded,
        "eigenb2 with feral_singular_pivot_floor=1e-8 should certify optimality, \
         got status={:?}",
        report.solution.status,
    );

    let iters = report.statistics.iteration_count;
    assert!(
        iters <= RECIPE_ITER_CEILING,
        "eigenb2 with feral_singular_pivot_floor=1e-8 took {iters} iterations; \
         it is supposed to remove the stall, not shave a few iterations \
         (39 against the default's 68, ceiling {RECIPE_ITER_CEILING})",
    );
}
