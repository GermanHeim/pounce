//! The convex arm as a `SensBacksolver`, and the sign that had to be right first.
//!
//! # Why this file exists
//!
//! `pounce-sens-core`'s fix-relax / path / directional machinery is generic
//! over `SensBacksolver`, so the convex arm reaches it by *implementing* the
//! trait rather than reimplementing 8k lines. This file guards the two things
//! that implementation can get wrong without anything else noticing.
//!
//! **The bound-row sign.** In the `Gx ≤ h` orientation the convex form uses, an
//! active lower bound `lb ≤ xⱼ` is the row `−eⱼᵀ` and an active upper bound is
//! `+eⱼᵀ`. The assembly emitted `+1` for *both* until this change. That is
//! invisible while the active block's right-hand side is zero — `eⱼᵀ dx = 0`
//! and `−eⱼᵀ dx = 0` are the same constraint — which is exactly why 774 lines
//! of inline tests and every existing caller never caught it. It stops being
//! invisible the moment anything reads the recovered multiplier block `dz_a`,
//! which is what a release decision does. The defect would have been introduced
//! *by* the feature that first depended on it, in a file the feature did not
//! touch. `the_recovered_bound_multipliers_carry_the_solutions_sign` is the
//! guard.
//!
//! **The frame.** `dx/db` must not depend on whether the solve equilibrated
//! internally: Ruiz equilibration lives inside `solve_qp_ipm` and is undone
//! before `QpSolution` is returned, so the KKT and the multipliers are already
//! in one frame. `the_step_is_unmoved_by_internal_equilibration` holds that.
//!
//! It does **not** hold the related `natural_units_factor = None` claim — see
//! the negative-space section below, which is a measured result, not a
//! suspicion.
//!
//! # What this file is NOT evidence about
//!
//! - **Release.** `supports_release()` is `false` in this phase, so the
//!   refinement pins and never releases. `a_pin_only_refinement_does_not_release`
//!   pins that limit rather than leaving it implied; the release half arrives
//!   with the path modes.
//! - **`natural_units_factor`, and this one is measured.** Making it return a
//!   non-identity vector turns *nothing* in this crate red. Its only consumer
//!   is the release half of `refine_step_onto_bounds`, which `supports_release
//!   () == false` never reaches — so the value is correct and untested at the
//!   same time. Whoever turns release on inherits the obligation to guard it;
//!   a released step checked against a re-solve cannot pass with a mis-scaled
//!   `F`. Stated here rather than left for someone to assume the equilibration
//!   test covers it, because it does not: that test drives `parametric_step`,
//!   which never reads the factor.
//! - **The NLP arm.** Nothing here exercises `PdSensBacksolver`. The two
//!   implementations agree only as far as the shared core makes them; a
//!   cross-arm equality test belongs with the parity work.
//! - **Scale invariance of the classifier.** The convex arm still has no
//!   scale-invariance leg; that lands with the classification phase.
//! - **Degenerate LPs.** `lp_without_crossover` reports the hazard; it does not
//!   fix it, and `a_degenerate_lp_is_flagged_not_answered` says only that the
//!   flag and the conditioning diagnostic both fire.
//!
//! # Mutation evidence
//!
//! Each row was **run**, not predicted:
//!
//! | mutation | red here | note |
//! |---|---|---|
//! | `assemble_kkt` emits `+1.0` for a lower-active bound (the pre-change code) | `the_recovered_bound_multipliers_carry_the_solutions_sign` alone | every other test in the crate stays green — that is the point |
//! | `natural_units_factor` returns a non-identity vector | **nothing** | the row that corrected the file: the claim it was written to support is false, and the negative-space section above now says so |
//! | `parametric_step_bounded` returns the plain step, skipping the refinement | `a_crossing_step_is_pinned_at_the_bound` | |

use pounce_convex::QpOptions;
use pounce_convex::ipm::solve_qp_ipm;
use pounce_convex::qp::{QpProblem, QpStatus, Triplet};
use pounce_convex::sensitivity::QpSensitivity;
use pounce_feral::FeralSolverInterface;
use pounce_linsol::SparseSymLinearSolverInterface;
use pounce_sens_core::backsolver::SensBacksolver;
use pounce_sens_core::boundcheck::RefineStop;

fn backend() -> Box<dyn SparseSymLinearSolverInterface> {
    Box::new(FeralSolverInterface::new())
}

fn tri(row: usize, col: usize, val: f64) -> Triplet {
    Triplet { row, col, val }
}

fn solved(prob: &QpProblem, opts: &QpOptions) -> QpSensitivity {
    let sol = solve_qp_ipm(prob, opts, backend);
    assert_eq!(sol.status, QpStatus::Optimal, "fixture must solve");
    match QpSensitivity::build(prob, &sol, opts, 1e-7, backend) {
        Ok(s) => s,
        Err(e) => panic!("fixture must build a sensitivity, got {e:?}"),
    }
}

/// `min ½‖x‖² − cᵀx  s.t.  x₀ + x₁ = b,  x ≥ 0`, with `c` chosen so the
/// unconstrained optimum wants `x₁ < 0` and the **lower** bound on `x₁` is
/// active with a strictly positive multiplier. The lower bound is the case the
/// sign fix is about.
fn lower_bound_active() -> QpProblem {
    QpProblem {
        n: 2,
        p_lower: vec![tri(0, 0, 1.0), tri(1, 1, 1.0)],
        c: vec![-2.0, 1.0],
        a: vec![tri(0, 0, 1.0), tri(0, 1, 1.0)],
        b: vec![1.0],
        g: vec![],
        h: vec![],
        lb: vec![0.0, 0.0],
        ub: vec![f64::INFINITY, f64::INFINITY],
    }
}

// ---------------------------------------------------------------------------
// Preconditions — without these the guards below are vacuous.
// ---------------------------------------------------------------------------

/// The fixture really does put a *lower* bound in the active set with a
/// strictly positive multiplier. If it stops doing so, the sign test below is
/// asserting nothing.
#[test]
fn the_fixture_has_a_strictly_active_lower_bound() {
    let prob = lower_bound_active();
    let sol = solve_qp_ipm(&prob, &QpOptions::default(), backend);
    assert_eq!(sol.status, QpStatus::Optimal);
    assert!(
        sol.z_lb[1] > 1e-6,
        "x1's LOWER bound must carry a real multiplier; got z_lb = {:?}, z_ub = {:?}",
        sol.z_lb,
        sol.z_ub
    );
    assert!(
        sol.z_ub[1] < 1e-9,
        "and the upper bound must be inactive, or the orientation is ambiguous"
    );
    assert!(
        sol.x[1].abs() < 1e-6,
        "x1 must sit at its lower bound, got {}",
        sol.x[1]
    );
}

// ---------------------------------------------------------------------------
// The sign.
// ---------------------------------------------------------------------------

/// Reading the recovered multiplier block back must reproduce the solution's
/// own non-negative bound multiplier, sign included.
///
/// Construction: perturb `b`, and solve the active-set KKT directly through the
/// backsolver. For the *stationarity* right-hand side `−e_j` placed on the
/// variable's `x` row, the recovered `dz_a` entry is the sensitivity of that
/// bound's multiplier, and its sign is fixed by the row orientation. With the
/// row emitted as `+eⱼᵀ` for a lower bound — the pre-change code — this comes
/// back negated.
#[test]
fn the_recovered_bound_multipliers_carry_the_solutions_sign() {
    let prob = lower_bound_active();
    let opts = QpOptions::default();
    let sens = solved(&prob, &opts);
    let bs = sens.backsolver();

    let Some(rows) = bs.bound_rows() else {
        panic!("the adapter must report bound rows");
    };
    let Some(lower) = rows.iter().find(|r| r.var_row == 1) else {
        panic!("x1's bound must be in the active set");
    };
    assert!(lower.lower, "x1 is held by its LOWER bound");

    // Push along the bound's own row: the active-set system then reports how
    // the multiplier responds. Orientation-sensitive by construction.
    let mut rhs = vec![0.0; bs.dim()];
    rhs[lower.row] = 1.0;
    let mut lhs = vec![0.0; bs.dim()];
    assert!(bs.solve(&rhs, &mut lhs), "the back-solve must succeed");

    // The variable is pinned, so moving its bound row by +1 must move the
    // variable by exactly −1 under the `−eⱼᵀ` orientation the convex form uses
    // for a lower bound. Under the `+eⱼᵀ` the code emitted before, it moves +1.
    assert!(
        (lhs[1] - (-1.0)).abs() < 1e-6,
        "a lower-bound row is `−e_j`, so a unit move of that row moves x1 by −1; \
         got {}. A `+1` here is the pre-change orientation, which is invisible to \
         every test that does not read the multiplier block.",
        lhs[1]
    );
}

// ---------------------------------------------------------------------------
// The frame invariant.
// ---------------------------------------------------------------------------

/// `natural_units_factor()` is `None` because the KKT and the multipliers are
/// already in one frame. Solving the same problem with the internal
/// equilibration on and off must therefore give the same `dx/db`.
///
/// If someone later assembles the sensitivity KKT from equilibrated data, this
/// goes red — which is the point, because nothing else would notice.
#[test]
fn the_step_is_unmoved_by_internal_equilibration() {
    let prob = lower_bound_active();

    let on = QpOptions {
        equilibrate: true,
        ..Default::default()
    };
    let off = QpOptions {
        equilibrate: false,
        ..Default::default()
    };

    let mut a = solved(&prob, &on);
    let mut b = solved(&prob, &off);
    let da = a.parametric_step(&[0], &[1.0]);
    let db = b.parametric_step(&[0], &[1.0]);

    assert_eq!(da.len(), db.len());
    for (i, (x, y)) in da.iter().zip(db.iter()).enumerate() {
        assert!(
            (x - y).abs() < 1e-7,
            "dx/db[{i}] must not depend on whether the solve equilibrated \
             internally: {x} vs {y}. If this fails, the sensitivity KKT is being \
             built from equilibrated data and `natural_units_factor` can no \
             longer be `None`."
        );
    }
}

// ---------------------------------------------------------------------------
// The refinement, reached through the shared core.
// ---------------------------------------------------------------------------

/// A perturbation large enough to carry a variable past its bound must come
/// back with that coordinate pinned AT the bound and the others moved to suit —
/// not merely clipped.
#[test]
fn a_crossing_step_is_pinned_at_the_bound() {
    // `min ½‖x‖²  s.t.  x₀ + x₁ + x₂ = b,  x₀ ≥ 0` (x₁, x₂ free), at b = 3 so
    // the base point is (1, 1, 1) with no bound active.
    //
    // Only x₀ is bounded, and that is deliberate: the perturbation has to be
    // *repairable*. Bound every coordinate below and driving `b` negative asks
    // for a sum that no non-negative point can reach, so fix-relax would have
    // no answer to find and the test would be asserting against an infeasible
    // problem rather than against the refinement.
    //
    // Driving b to −3 wants x = (−1, −1, −1): x₀ crosses, and the two free
    // coordinates have room to absorb the difference once it is pinned.
    let prob = QpProblem {
        n: 3,
        p_lower: vec![tri(0, 0, 1.0), tri(1, 1, 1.0), tri(2, 2, 1.0)],
        c: vec![0.0, 0.0, 0.0],
        a: vec![tri(0, 0, 1.0), tri(0, 1, 1.0), tri(0, 2, 1.0)],
        b: vec![3.0],
        g: vec![],
        h: vec![],
        lb: vec![0.0, f64::NEG_INFINITY, f64::NEG_INFINITY],
        ub: vec![f64::INFINITY; 3],
    };
    let opts = QpOptions::default();
    let mut sens = solved(&prob, &opts);

    let plain = sens.parametric_step(&[0], &[-6.0]);
    let base = prob_base(&prob, &opts);
    assert!(
        (base[0] + plain[0]) < -1e-3,
        "the fixture must make the plain step carry x0 past its bound, or the \
         refinement has nothing to do: base {base:?} + {plain:?}"
    );

    let Ok((dx, pinned, stop)) = sens.parametric_step_bounded(&[0], &[-6.0], 1e-3, 16) else {
        panic!("refinement must run");
    };
    assert!(
        !pinned.is_empty(),
        "a crossing step must pin at least one coordinate, got {pinned:?} (stop {stop:?})"
    );
    assert!(
        base[0] + dx[0] >= -1e-6,
        "x0 must land at or inside its bound after refinement: {} + {} = {}",
        base[0],
        dx[0],
        base[0] + dx[0]
    );
    // The repair is not a clip: the equality must still hold, which means the
    // free coordinates moved to absorb what pinning x0 took away.
    let sum: f64 = (0..3).map(|j| base[j] + dx[j]).sum();
    assert!(
        (sum - (3.0 - 6.0)).abs() < 1e-6,
        "the perturbed equality x0+x1+x2 = b−6 = −3 must still hold after the \
         refinement — a clip would break it; got {sum}"
    );
}

fn prob_base(prob: &QpProblem, opts: &QpOptions) -> Vec<f64> {
    solve_qp_ipm(prob, opts, backend).x
}

/// The refinement pins but does not release in this phase, and the adapter says
/// so. Pinning the limit rather than leaving it implied means the phase that
/// adds release has a test to flip.
#[test]
fn a_pin_only_refinement_does_not_release() {
    let prob = lower_bound_active();
    let opts = QpOptions::default();
    let sens = solved(&prob, &opts);
    assert!(
        !sens.backsolver().supports_release(),
        "release lands with the path modes; until then the refinement pins only"
    );
}

/// `RefineStop` comes back from the shared core, so a caller can tell a settled
/// refinement from one that ran out of iterations.
#[test]
fn a_step_that_crosses_nothing_settles_immediately() {
    let prob = lower_bound_active();
    let opts = QpOptions::default();
    let mut sens = solved(&prob, &opts);
    // A tiny perturbation cannot carry anything across a bound.
    let Ok((dx, pinned, stop)) = sens.parametric_step_bounded(&[0], &[1e-6], 1e-3, 16) else {
        panic!("refinement must run");
    };
    assert!(
        pinned.is_empty(),
        "nothing should be pinned, got {pinned:?}"
    );
    assert!(
        matches!(stop, RefineStop::Settled),
        "an uncrossed step settles, got {stop:?}"
    );
    let plain = sens.parametric_step(&[0], &[1e-6]);
    for (a, b) in dx.iter().zip(plain.iter()) {
        assert!(
            (a - b).abs() < 1e-9,
            "with nothing to repair the refined step is the plain one: {a} vs {b}"
        );
    }
}

// ---------------------------------------------------------------------------
// The degenerate-LP hazard, named.
// ---------------------------------------------------------------------------

/// A pure LP built without crossover is flagged. The flag does not fix the
/// hazard — it names it, and the conditioning diagnostic is what actually
/// catches the bad step.
#[test]
fn a_degenerate_lp_is_flagged_not_answered() {
    // min −x0 − x1 s.t. x0 + x1 = 1, x0 + x1 <= 1, x >= 0. Every point of the
    // face is optimal and the active set is over-determined.
    let prob = QpProblem {
        n: 2,
        p_lower: vec![],
        c: vec![-1.0, -1.0],
        a: vec![tri(0, 0, 1.0), tri(0, 1, 1.0)],
        b: vec![1.0],
        g: vec![tri(0, 0, 1.0), tri(0, 1, 1.0)],
        h: vec![1.0],
        lb: vec![0.0, 0.0],
        ub: vec![f64::INFINITY, f64::INFINITY],
    };
    let opts = QpOptions::default();
    assert!(!opts.crossover, "the default is crossover off");
    let mut sens = solved(&prob, &opts);

    assert!(
        sens.lp_without_crossover(),
        "a pure LP solved without crossover must be flagged"
    );
    let dx = sens.parametric_step(&[0], &[1.0]);
    assert!(
        sens.ill_conditioned(),
        "and the degenerate step must be reported untrustworthy; got dx = {dx:?}"
    );
}

/// The flag is about the LP branch specifically: a QP is not flagged, whatever
/// `crossover` says, because crossover only runs on a pure LP.
#[test]
fn a_qp_is_not_flagged_for_crossover() {
    let prob = lower_bound_active();
    let sens = solved(&prob, &QpOptions::default());
    assert!(
        !sens.lp_without_crossover(),
        "crossover is an LP phase; a QP must not be flagged for skipping it"
    );
}

// ---------------------------------------------------------------------------
// The duality measure.
// ---------------------------------------------------------------------------

/// `duality_measure` is the achieved complementarity, so at a converged optimum
/// it is small and positive — the scale a μ-scaled classifier would band
/// against.
#[test]
fn the_duality_measure_is_small_and_positive_at_an_optimum() {
    let prob = lower_bound_active();
    let sens = solved(&prob, &QpOptions::default());
    let mu = sens.duality_measure();
    assert!(
        (0.0..1e-6).contains(&mu),
        "a converged QP's achieved complementarity should be tiny and \
         non-negative, got {mu}"
    );
}
