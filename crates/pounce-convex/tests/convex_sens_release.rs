//! The release half of fix-relax on the convex arm, against a re-solve oracle.
//!
//! # Why an oracle, when the adapter tests exist
//!
//! `convex_sens_backsolver.rs` compares numbers this layer produced against
//! other numbers this layer produced. That catches a step that is not
//! self-consistent. It cannot catch a step that is *self-consistently wrong* —
//! and release is exactly where that risk lives, because a released step
//! changes which system is solved, not just the right-hand side. So every claim
//! here is checked against **an independent solve of the perturbed problem**,
//! at a tolerance two orders tighter than the base solve.
//!
//! This is also where the debt from the pin-only phase is paid. That phase
//! recorded, as a measured result, that `natural_units_factor = None` was
//! correct but **unguarded**: its only consumer is the release path, and
//! release was off, so returning garbage from it turned nothing red. Release is
//! on now, so the value is load-bearing, and
//! `a_released_step_reaches_the_resolve` is what holds it — a mis-scaled `F`
//! shifts the released multiplier by the wrong amount and the oracle rejects
//! the answer.
//!
//! # The fixture, and why its arithmetic is written out
//!
//! `min ½‖x‖² − 2x₀ + x₁  s.t.  x₀ + x₁ = b,  x ≥ 0`.
//!
//! At `b = 1` the optimum is `x = (1, 0)`: x₁ sits on its lower bound carrying
//! multiplier `z₁ = 2`. Raising `b` drives that multiplier down — `z₁ = 3 − b` —
//! so at `b = 3` it hits zero and beyond that the bound must **release**. For
//! `b ≥ 3` both coordinates are interior and `x = ((3+b)/2, (b−3)/2)`.
//!
//! So a perturbation of `+3` (to `b = 4`) has a known closed-form answer,
//! `x = (3.5, 0.5)`, and crosses exactly one breakpoint at two-thirds of the
//! way. Holding the active set instead gives `x = (4, 0)` — the whole released
//! distance wrong, which is the error mode being guarded against.
//!
//! # What this file is NOT evidence about
//!
//! - **Multiple simultaneous releases.** One bound releases here. A model where
//!   two release at the same breakpoint takes a different branch of the
//!   refinement and is not covered.
//! - **Releases on inequality rows.** Only *variable bounds* are releasable;
//!   `G` rows are not `BoundRow`s on either arm.
//! - **Scaling.** Both fixtures run unscaled. The convex arm still has no
//!   scale-invariance leg — that lands with the classification phase.
//! - **Anything at benchmark scale.** Two and three variables.
//!
//! # Mutation evidence
//!
//! Each row was **run**, not predicted:
//!
//! | mutation | red here | note |
//! |---|---|---|
//! | `natural_units_factor` returns a non-identity vector | all three oracle legs (`a_released_step_reaches_the_resolve`, `the_path_reports_the_release_breakpoint…`, `a_refinement_can_release_one_bound_and_pin_another`) | the debt from the pin-only phase, now genuinely discharged |
//! | `solve_released_step` skips the RHS shift | `a_released_step_reaches_the_resolve`, `a_refinement_can_release_one_bound_and_pin_another` | the shift is what moves the released multiplier onto its variable; the path leg survives because the walk re-solves per segment |
//! | the shift uses `+z` for a lower bound instead of `−z` | the same two | the sign the pin-only phase fixed, now load-bearing rather than latent |
//! | `supports_release()` returns `false` | all three, plus `the_refinement_can_both_pin_and_release` in `convex_sens_backsolver.rs` | reverts to pin-only |
//!
//! Two procedural notes, both learned the hard way while producing this table.
//! A mutation that fails to **compile** produces no `FAILED` lines and reads as
//! "nothing went red" — so each row above was taken only after checking the
//! mutated tree compiles clean. And `cargo test` stops at the first failing
//! binary, so the rows were taken with `--no-fail-fast`; without it, mutation 4
//! looked like it broke one test when it breaks four.

use pounce_convex::QpOptions;
use pounce_convex::ipm::solve_qp_ipm;
use pounce_convex::qp::{QpProblem, QpStatus, Triplet};
use pounce_convex::sensitivity::QpSensitivity;
use pounce_feral::FeralSolverInterface;
use pounce_linsol::SparseSymLinearSolverInterface;

fn backend() -> Box<dyn SparseSymLinearSolverInterface> {
    Box::new(FeralSolverInterface::new())
}

fn tri(row: usize, col: usize, val: f64) -> Triplet {
    Triplet { row, col, val }
}

/// The fixture above, at right-hand side `b`.
fn releasing_qp(b: f64) -> QpProblem {
    QpProblem {
        n: 2,
        p_lower: vec![tri(0, 0, 1.0), tri(1, 1, 1.0)],
        c: vec![-2.0, 1.0],
        a: vec![tri(0, 0, 1.0), tri(0, 1, 1.0)],
        b: vec![b],
        g: vec![],
        h: vec![],
        lb: vec![0.0, 0.0],
        ub: vec![f64::INFINITY, f64::INFINITY],
    }
}

/// The independent answer: solve the perturbed problem outright, at a tolerance
/// two orders tighter than the base solve, and return its primal.
fn oracle(b: f64) -> Vec<f64> {
    let opts = QpOptions {
        tol: 1e-11,
        ..Default::default()
    };
    let sol = solve_qp_ipm(&releasing_qp(b), &opts, backend);
    assert_eq!(
        sol.status,
        QpStatus::Optimal,
        "the oracle solve must converge"
    );
    sol.x
}

fn base_sens(prob: &QpProblem) -> (QpSensitivity, Vec<f64>) {
    let opts = QpOptions::default();
    let sol = solve_qp_ipm(prob, &opts, backend);
    assert_eq!(sol.status, QpStatus::Optimal);
    let x = sol.x.clone();
    match QpSensitivity::build(prob, &sol, &opts, 1e-7, backend) {
        Ok(s) => (s, x),
        Err(e) => panic!("the fixture must build a sensitivity, got {e:?}"),
    }
}

// ---------------------------------------------------------------------------
// Preconditions — without these every leg below is vacuous.
// ---------------------------------------------------------------------------

/// The base point really does hold x₁ on its lower bound with a strictly
/// positive multiplier, and the closed form really does predict a release.
#[test]
fn the_fixture_holds_a_bound_that_the_perturbation_must_release() {
    let sol = solve_qp_ipm(&releasing_qp(1.0), &QpOptions::default(), backend);
    assert_eq!(sol.status, QpStatus::Optimal);
    assert!(
        (sol.x[0] - 1.0).abs() < 1e-6 && sol.x[1].abs() < 1e-6,
        "base optimum must be (1, 0), got {:?}",
        sol.x
    );
    assert!(
        (sol.z_lb[1] - 2.0).abs() < 1e-5,
        "x1's lower bound must carry multiplier 2, got {}",
        sol.z_lb[1]
    );

    // And the perturbed problem must genuinely leave the bound.
    let x4 = oracle(4.0);
    assert!(
        x4[1] > 1e-3,
        "at b = 4 the bound must have released (x1 > 0), got {x4:?} — if this \
         holds at zero the fixture no longer exercises a release at all"
    );
    assert!(
        (x4[0] - 3.5).abs() < 1e-5 && (x4[1] - 0.5).abs() < 1e-5,
        "closed form says (3.5, 0.5); got {x4:?}"
    );
}

/// Holding the active set — the plain step — misses the release entirely. This
/// is the error the release machinery exists to avoid, and stating its size
/// here is what makes the oracle test below meaningful rather than a tautology.
#[test]
fn the_plain_step_misses_the_release_by_the_full_released_distance() {
    let prob = releasing_qp(1.0);
    let (mut sens, base) = base_sens(&prob);
    let plain = sens.parametric_step(&[0], &[3.0]);
    let predicted: Vec<f64> = base.iter().zip(plain.iter()).map(|(a, b)| a + b).collect();
    let truth = oracle(4.0);

    assert!(
        predicted[1].abs() < 1e-6,
        "the plain step holds x1 at its bound, got {predicted:?}"
    );
    let miss = (predicted[1] - truth[1]).abs();
    assert!(
        miss > 0.4,
        "the plain step should be wrong by about the released distance (0.5); \
         got {miss}. If this shrinks, the fixture stopped exercising a release."
    );
}

// ---------------------------------------------------------------------------
// The oracle.
// ---------------------------------------------------------------------------

/// The headline: with release on, the refined step reaches the independently
/// re-solved answer.
///
/// This is the guard the pin-only phase said it owed. A mis-scaled
/// `natural_units_factor`, a missing right-hand-side shift, or the wrong sign
/// on that shift all move the release point, and all three are rejected here.
#[test]
fn a_released_step_reaches_the_resolve() {
    let prob = releasing_qp(1.0);
    let (mut sens, base) = base_sens(&prob);

    let Ok((dx, _pinned, _stop)) = sens.parametric_step_bounded(&[0], &[3.0], 1e-3, 32) else {
        panic!("the refinement must run");
    };
    let predicted: Vec<f64> = base.iter().zip(dx.iter()).map(|(a, b)| a + b).collect();
    let truth = oracle(4.0);

    for (j, (p, t)) in predicted.iter().zip(truth.iter()).enumerate() {
        assert!(
            (p - t).abs() < 1e-5,
            "released step must reach the re-solve at coordinate {j}: \
             predicted {p}, re-solve {t} (full predicted {predicted:?} vs {truth:?})"
        );
    }
}

// ---------------------------------------------------------------------------
// The path.
// ---------------------------------------------------------------------------

/// Walking the perturbation reports the breakpoint where the bound releases,
/// and lands on the same answer.
///
/// The closed form puts the release at `b = 3`, which is two-thirds of the way
/// from `b = 1` to `b = 4` — so the segment's `at` is a number with a known
/// value, not merely "somewhere in (0, 1)".
#[test]
fn the_path_reports_the_release_breakpoint_and_lands_on_the_resolve() {
    let prob = releasing_qp(1.0);
    let (mut sens, base) = base_sens(&prob);

    let Ok((dx, segments)) = sens.parametric_step_path(&[0], &[3.0], 32) else {
        panic!("the path walk must run");
    };
    assert!(
        !segments.is_empty(),
        "the walk must report the breakpoint it crossed"
    );
    let release = segments
        .iter()
        .find(|s| s.var_row == 1)
        .unwrap_or_else(|| panic!("x1's bound is what changes status; got {segments:?}"));
    assert!(
        (release.at - 2.0 / 3.0).abs() < 1e-3,
        "the bound releases at b = 3, two-thirds of the way from 1 to 4; \
         got at = {}",
        release.at
    );

    let predicted: Vec<f64> = base.iter().zip(dx.iter()).map(|(a, b)| a + b).collect();
    let truth = oracle(4.0);
    for (j, (p, t)) in predicted.iter().zip(truth.iter()).enumerate() {
        assert!(
            (p - t).abs() < 1e-5,
            "the walked path must land on the re-solve at coordinate {j}: \
             {p} vs {t}"
        );
    }
}

/// A perturbation that crosses nothing produces no segments and the plain step.
/// Without this, the test above could pass on a walk that always reports a
/// breakpoint.
#[test]
fn a_path_that_crosses_nothing_reports_no_breakpoint() {
    let prob = releasing_qp(1.0);
    let (mut sens, _base) = base_sens(&prob);
    let Ok((dx, segments)) = sens.parametric_step_path(&[0], &[1e-4], 32) else {
        panic!("the path walk must run");
    };
    assert!(
        segments.is_empty(),
        "a tiny perturbation crosses no breakpoint, got {segments:?}"
    );
    let plain = sens.parametric_step(&[0], &[1e-4]);
    for (a, b) in dx.iter().zip(plain.iter()) {
        assert!(
            (a - b).abs() < 1e-9,
            "inside one segment the walk is the plain step: {a} vs {b}"
        );
    }
}

// ---------------------------------------------------------------------------
// Release and pin in the same refinement.
// ---------------------------------------------------------------------------

/// Releasing must not cost the pinning half. A model where one bound releases
/// while another is reached keeps both behaviours in one answer.
#[test]
fn a_refinement_can_release_one_bound_and_pin_another() {
    // `min ½‖x‖² − 2x₀ + x₁  s.t.  x₀ + x₁ + x₂ = b,  x₁ ≥ 0,  x₂ ≥ 0`,
    // with x₀ free. As b rises x₁ leaves its bound (as in the two-variable
    // fixture); as it rises further x₂ is driven onto its own.
    let prob = QpProblem {
        n: 3,
        p_lower: vec![tri(0, 0, 1.0), tri(1, 1, 1.0), tri(2, 2, 1.0)],
        c: vec![-2.0, 1.0, 2.0],
        a: vec![tri(0, 0, 1.0), tri(0, 1, 1.0), tri(0, 2, 1.0)],
        b: vec![1.0],
        g: vec![],
        h: vec![],
        lb: vec![f64::NEG_INFINITY, 0.0, 0.0],
        ub: vec![f64::INFINITY; 3],
    };
    let (mut sens, base) = base_sens(&prob);
    let Ok((dx, _pinned, _stop)) = sens.parametric_step_bounded(&[0], &[3.0], 1e-3, 32) else {
        panic!("the refinement must run");
    };

    // Whatever the active set does, the answer must be feasible and must match
    // an independent solve of the perturbed problem.
    let mut perturbed = prob.clone();
    perturbed.b = vec![4.0];
    let opts = QpOptions {
        tol: 1e-11,
        ..Default::default()
    };
    let truth = solve_qp_ipm(&perturbed, &opts, backend);
    assert_eq!(truth.status, QpStatus::Optimal);

    let predicted: Vec<f64> = base.iter().zip(dx.iter()).map(|(a, b)| a + b).collect();
    for (j, (p, t)) in predicted.iter().zip(truth.x.iter()).enumerate() {
        assert!(
            (p - t).abs() < 1e-5,
            "coordinate {j}: refined {p} vs re-solve {t} \
             (full {predicted:?} vs {:?})",
            truth.x
        );
    }
}
