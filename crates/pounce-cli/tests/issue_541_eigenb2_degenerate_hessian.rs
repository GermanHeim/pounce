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
//!   `docs/src/troubleshooting.md` — still reaches the optimum *point*.
//!   It is no longer the fast route: since gh#693 the default reaches the
//!   optimum in 21 iterations and the knob takes 72, losing the strict
//!   certificate on the way. That is not special to this model: across
//!   the 110 hardest benchmark-corpus problems the same knob is a coin
//!   flip (89 unchanged, 10 better, 11 worse, and five of the seven
//!   regressions are `Optimal -> Solved To Acceptable Level`), which is
//!   why `docs/src/troubleshooting.md` now frames it as a gamble to
//!   measure rather than a recipe to apply. A step-curvature guard that would
//!   have fixed this without any knob was prototyped and rejected — it
//!   regresses `jit1_node` from 24 to 246 iterations and pushes `cresc4`
//!   and `pooling_rt2stp` past the iteration cap (dev-note §7).

use std::path::PathBuf;
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

use pounce_cli::solve_report::SolveReport;
use pounce_nlp::ApplicationReturnStatus;

/// Exact optimum of `eigenb2`. Both POUNCE and the committed Ipopt-MA57
/// reference agree to 12+ digits.
const EIGENB2_OPTIMUM: f64 = 1.6;

/// Since gh#693 the default reaches the optimum in 21 iterations, well
/// inside the 39 the `feral_singular_pivot_floor` recipe used to take. The
/// ceiling below is now a bound on the *default*, not on the recipe — see
/// `eigenb2_default_is_now_the_fast_route`.
const DEFAULT_ITER_CEILING: i32 = 35;

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

/// Under defaults this certifies `Optimal` in 21 iterations (68 between #544
/// and gh#693). Before #544 it returned `Solved To Acceptable Level` in 67 —
/// the right point, without a certificate. This is the regression test for
/// the `eigenb2` half of #544: if the inertia-noise routing regresses, the
/// status drops back and this fails.
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
/// than correcting it per-factorization the way #544 does.
///
/// **This test asserted the opposite until gh#693, and it failed.** It
/// pinned the knob as a speedup — `SolveSucceeded` in at most 55 iterations
/// against the default's 68. Removing the Tikhonov perturbation from the
/// equality-multiplier initializer inverted the comparison outright:
///
/// ```text
///   options                                0.10.0                with gh#693
///   (defaults)                    67 it, 3.504e-09, Optimal   21 it, 2.712e-09, Optimal
///   feral_singular_pivot_floor=1e-8   39 it, 7.806e-10, Optimal   72 it, 2.394e-08, Acceptable
///   ...  + mu_strategy=adaptive     30 it, 3.11e-09,  Optimal   86 it, 1.768e-08, Acceptable
///   mu_strategy=adaptive           63 it, 7.763e-10, Optimal   21 it, 2.712e-09, Optimal
/// ```
///
/// So the default is now faster than every recipe 0.10.0 had, and the knob
/// is worse than doing nothing on this model — it costs 51 extra iterations
/// and drops the certificate, ending at a dual residual of 2.394e-08 against
/// `tol = 1e-8`.
///
/// What is asserted here is therefore reduced to what survives: the knob
/// still reaches the right *point*. The lost certificate is a real cost, it
/// is not "fixed" by the default having improved, and it is tracked in
/// `docs/src/troubleshooting.md` — including the corpus measurement of whether the troubleshooting
/// recipe is still correct advice for the symptom it is written for, which
/// cannot be answered from this one fixture.
#[test]
fn eigenb2_singular_pivot_floor_still_reaches_the_optimum_point() {
    let report = solve(&["feral_singular_pivot_floor=1e-8"]);
    assert_at_optimum(&report, "eigenb2 (feral_singular_pivot_floor=1e-8)");

    // Deliberately NOT asserting SolveSucceeded: since gh#693 this returns
    // SolvedToAcceptableLevel at dual 2.394e-08, deterministically: 25
    // values of `mu_init` at 0.1 +/- k*1e-12 give the identical result at
    // every point, so this is not trajectory noise. What must not
    // happen is the model failing outright or landing somewhere else.
    assert!(
        matches!(
            report.solution.status,
            ApplicationReturnStatus::SolveSucceeded
                | ApplicationReturnStatus::SolvedToAcceptableLevel
        ),
        "eigenb2 with feral_singular_pivot_floor=1e-8 no longer converges at \
         all (status={:?}); the troubleshooting doc records it dropping to acceptable-level, \
         which is already a cost — an outright failure is a different and \
         worse regression",
        report.solution.status,
    );
}

/// The other half of the story, pinned so the improvement cannot silently
/// evaporate: the default path is what got fast here, and it is what a user
/// following `docs/src/troubleshooting.md` should now be told to try first.
#[test]
fn eigenb2_default_is_now_the_fast_route() {
    let default_iters = solve(&[]).statistics.iteration_count;
    assert!(
        default_iters <= DEFAULT_ITER_CEILING,
        "eigenb2 on defaults took {default_iters} iterations; since gh#693 it \
         reaches the optimum in 21, which is fewer than the 39 the \
         feral_singular_pivot_floor recipe took at its best (ceiling \
         {DEFAULT_ITER_CEILING}). If this regresses, the troubleshooting \
         advice in docs/src/troubleshooting.md needs revisiting again",
    );

    let recipe_iters = solve(&["feral_singular_pivot_floor=1e-8"])
        .statistics
        .iteration_count;
    assert!(
        default_iters < recipe_iters,
        "the feral_singular_pivot_floor recipe ({recipe_iters} iterations) is \
         supposed to be the slower route on this model since gh#693, against \
         the default's {default_iters}. If it is faster again, the \
         premise has changed",
    );
}
