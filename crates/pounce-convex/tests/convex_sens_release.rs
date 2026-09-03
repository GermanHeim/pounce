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
use pounce_convex::cones::ConeSpec;
use pounce_convex::ipm::{solve_qp_ipm, solve_socp_ipm};
use pounce_convex::qp::{QpProblem, QpStatus, Triplet};
use pounce_convex::sensitivity::{ConeBlockKind, QpSensitivity};
use pounce_sens_core::boundcheck::RefineStop;

/// The partition used by the conic release fixture at the bottom of this file.
const CONIC_SOC3: [ConeSpec; 1] = [ConeSpec::SecondOrder(3)];
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

// ---------------------------------------------------------------------------
// Release on a CONIC build. Round 5 of #889 listed this under "not covered":
// `QpKktBacksolver` against `boundcheck`'s index expectations, and the bound
// refinement on conic builds with combined face rows. Every fixture above is
// orthant, so the release path had never run against an active block holding
// anything but a `G` row.
// ---------------------------------------------------------------------------

/// The releasing fixture above, with a **second-order block on its boundary**
/// bolted alongside.
///
/// Variables `(x₀, x₁, t, v, w)`. The first two are the orthant release pair,
/// unchanged: `min ½‖x‖² − 2x₀ + x₁ s.t. x₀ + x₁ = b, x ≥ 0`, so `x₁` sits on
/// its bound with `z₁ = 3 − b` and must release past `b = 3`. The last three
/// carry `(t, v, w) ∈ Q₃` under `min ½‖·‖² − v − 0.2w`, which puts the block on
/// its **boundary**, and are otherwise **decoupled** from the equality.
///
/// # The coupling is the point, and the first draft got this backwards
///
/// `v` is a **coordinate of the cone** and it sits in the equality, so the
/// perturbation's answer has to travel through the block's face row and its
/// curvature. That is what makes the fixture evidence about a conic release
/// rather than about a release that happens to have a cone nearby.
///
/// The first version of this fixture left the block **decoupled**, on the
/// reasoning that "what is under test is not an interaction — it is whether the
/// release path indexes correctly around a face row". Measured, that fixture is
/// nearly worthless: with `assemble_kkt` mutated to drop the face curvature
/// entirely it stays **green**, because a decoupled block cannot influence the
/// released coordinates. Eight other tests catch that mutation and the one
/// written for the conic release path does not.
///
/// The index-space mutation that draft's doc named as its guard is worse — it
/// is **vacuous**. `release_slots` is keyed off `active_rows.len()`, and a face
/// contributes exactly one entry to `active_rows` and one to `active_ineq`, so
/// the two lengths are equal on every build and substituting one for the other
/// changes nothing. A doc naming a guard the code does not have is the failure
/// this PR has paid for more than any other.
///
/// An **apex** block would not test this: there the block contributes every one
/// of its `G` rows, so provenance and the active rows coincide. That is exactly
/// the coincidence which hid round 5's Finding 1 in `reduced_hessian` — the
/// apex fixtures were accidentally correct while the boundary ones were wrong —
/// so the fixture here is a boundary block on purpose.
fn conic_releasing_qp(b: f64) -> QpProblem {
    QpProblem {
        n: 5,
        p_lower: (0..5).map(|j| tri(j, j, 1.0)).collect(),
        c: vec![-2.0, -0.9, 0.0, -1.0, -0.2],
        // x₀ + x₁ + v = b. `v` is a COORDINATE OF THE CONE, and that coupling
        // is the whole point — see the note above.
        a: vec![tri(0, 0, 1.0), tri(0, 1, 1.0), tri(0, 3, 1.0)],
        b: vec![b],
        // s = (t, v, w) = h − Gx with h = 0
        g: vec![tri(0, 2, -1.0), tri(1, 3, -1.0), tri(2, 4, -1.0)],
        h: vec![0.0, 0.0, 0.0],
        lb: vec![0.0, 0.0, -1e19, -1e19, -1e19],
        ub: vec![1e19; 5],
    }
}

/// **A release on a conic build reaches the re-solve**, judged the way the
/// orthant case is: against an independent solve of the perturbed problem.
///
/// # Why `δ = 0.2` and not the orthant fixture's `+3`
///
/// The perturbation has to cross the release breakpoint, and it has to be small
/// enough that a *first-order* step along a **curved** face can still be
/// compared to a re-solve. `c₁ = −0.9` puts the breakpoint next door — `z₁ =
/// 0.0999998` at `b = 1`, released by `b = 1.2` — so `δ = 0.2` does both. At
/// the orthant fixture's `δ = 3` the comparison is meaningless: measured, the
/// step misses the re-solve by `0.9`, essentially all of it in `t`, which is
/// the curvature being *correct* rather than wrong.
///
/// # What the numbers look like, and what they pin
///
/// ```text
///              x₀         x₁          t          v          w
///   predicted  1.1200012  0.0200012   0.0999988  0.0599976  0.1000000
///   oracle     1.1199999  0.0199999   0.1166190  0.0600003  0.0999998
/// ```
///
/// The released pair matches to `~1.4e-6`; `t`, the curved coordinate, carries
/// `1.7e-2`, which is `O(δ²)` at `δ = 0.2`. That split is the face being used.
///
/// # Mutation evidence — run, and one of them negative
///
/// | mutation | result |
/// |---|---|
/// | `assemble_kkt` drops the face curvature | **red here.** `x₁` comes back `1.4e-6` instead of `0.0200012` — the bound does not release at all |
/// | `release_slots` keyed off `active_ineq.len()` instead of `active_rows.len()` | **vacuous, nothing red anywhere.** The two vectors have equal length on every build; recorded so nobody re-derives it as a guard |
///
/// The first row is what the decoupled first draft could not do.
#[test]
fn a_release_on_a_conic_build_reaches_the_resolve() {
    let opts = QpOptions {
        tol: 1e-11,
        ..Default::default()
    };
    let prob = conic_releasing_qp(1.0);
    let sol = solve_socp_ipm(&prob, &CONIC_SOC3, &opts, backend);
    assert_eq!(sol.status, QpStatus::Optimal);

    let mut sens = match QpSensitivity::build_conic(&prob, &CONIC_SOC3, &sol, &opts, 1e-7, backend)
    {
        Ok(s) => s,
        Err(e) => panic!("the fixture must build a conic sensitivity, got {e:?}"),
    };

    // The two properties that make this fixture the one described.
    assert_eq!(
        sens.cone_block_kinds(),
        [(0, ConeBlockKind::Boundary)],
        "the block must be on its BOUNDARY — an apex contributes its own G rows \
         and would not exercise the combined-row index space"
    );
    assert_eq!(
        sens.active_bound_vars(),
        [1],
        "and x₁ must be on its bound, or there is nothing to release"
    );

    const DELTA: f64 = 0.2;
    let Ok((dx, _pinned, stop)) = sens.parametric_step_bounded(&[0], &[DELTA], 1e-3, 32) else {
        panic!("the refinement must run on a conic build");
    };
    assert_eq!(stop, RefineStop::Settled);

    let predicted: Vec<f64> = sol.x.iter().zip(dx.iter()).map(|(a, b)| a + b).collect();
    let truth = solve_socp_ipm(
        &conic_releasing_qp(1.0 + DELTA),
        &CONIC_SOC3,
        &opts,
        backend,
    );
    assert_eq!(
        truth.status,
        QpStatus::Optimal,
        "the oracle solve must converge"
    );

    // The released pair sits on the flat part of the geometry, so it must reach
    // the re-solve outright.
    for j in [0usize, 1] {
        assert!(
            (predicted[j] - truth.x[j]).abs() < 1e-5,
            "released coordinate {j} must reach the re-solve: predicted {}, \
             oracle {} (full {predicted:?} vs {:?})",
            predicted[j],
            truth.x[j],
            truth.x
        );
    }

    // x₁ must genuinely leave its bound. Holding the active set leaves it at 0;
    // dropping the face curvature leaves it at ~1.4e-6, which is the same thing
    // by a different route. Either is orders away from the 0.02 that is right.
    assert!(
        predicted[1] > 0.015 && (predicted[1] - 0.02).abs() < 1e-4,
        "x₁ must actually release (≈0.02): got {}",
        predicted[1]
    );

    // The curved coordinate is a different claim: a first-order step along a
    // curved face carries O(δ²), so assert the ORDER rather than the value —
    // asserting a match would be asserting the face is flat.
    let err = |d: f64| -> f64 {
        let mut s2 =
            match QpSensitivity::build_conic(&prob, &CONIC_SOC3, &sol, &opts, 1e-7, backend) {
                Ok(v) => v,
                Err(e) => panic!("rebuild must succeed, got {e:?}"),
            };
        let (dxd, _, _) = s2
            .parametric_step_bounded(&[0], &[d], 1e-3, 32)
            .expect("the refinement must run");
        let t = solve_socp_ipm(&conic_releasing_qp(1.0 + d), &CONIC_SOC3, &opts, backend);
        ((sol.x[2] + dxd[2]) - t.x[2]).abs()
    };
    let (e_full, e_half) = (err(DELTA), err(DELTA / 2.0));
    let ratio = e_full / e_half;
    assert!(
        (3.0..=5.0).contains(&ratio),
        "the curved coordinate's residual must be second order (halving δ \
         should quarter it): {e_full:e} / {e_half:e} = {ratio}"
    );
}
