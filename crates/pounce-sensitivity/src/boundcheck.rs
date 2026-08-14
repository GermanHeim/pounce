//! Holding the parametric sensitivity step inside the variable bounds.
//!
//! Mirrors upstream
//! [`SensStdStepCalculator::BoundCheck`](https://github.com/coin-or/Ipopt/blob/master/contrib/sIPOPT/src/SensStdStepCalc.cpp),
//! which is what `sens_boundcheck` turns on.
//!
//! A step can point outside the box. Clipping the offending coordinate
//! back to its bound is cheap, but it leaves every other coordinate at
//! its linear-predictor value, so the result satisfies the bounds and
//! no longer satisfies the constraints. On upstream's own parametric
//! example that costs an order of magnitude against a full re-solve.
//!
//! [`refine_step_onto_bounds`] instead adds a row pinning the offending
//! coordinate at its bound and re-solves, so the others move with it.
//! [`worst_violation`] picks which coordinate that is and
//! [`expand_bounds`] puts the bounds in a form both can read.
//!
//! # Why the pin alone is enough
//!
//! Upstream describes this as fix-relax: pin the variable at its bound
//! AND relax the complementarity condition that went with the bound,
//! giving it a new multiplier. Only the pin is done here, and it gives
//! the same primal step.
//!
//! The barrier contributes `Σ = z_l/s_l + z_u/s_u` to the Hessian
//! block, and Σ is diagonal, so `Σ_ii` appears only in row `i`. Adding
//! the pin row fixes `Δx_i`, which turns row `i` from a constraint on
//! `Δx` into the equation determining the pin's own multiplier. `Σ_ii`
//! therefore shifts that multiplier and leaves `Δx` unchanged, so
//! removing it, which is what relaxing the complementarity does, cannot
//! move the primal step.
//!
//! That equivalence covers the primal step only. A caller wanting the
//! bound multiplier sensitivities at a pinned coordinate would need the
//! relaxation, since those are exactly what it changes.

use crate::schur_data::IndexSchurData;
use pounce_common::types::{Index, Number};
use pounce_linalg::Vector;
use pounce_linalg::expansion_matrix::ExpansionMatrix;
use std::rc::Rc;

/// Expand the compressed bound vectors into full var-x arrays, with
/// infinities where a variable has no bound on that side.
///
/// The compressed form pairs an [`ExpansionMatrix`] with a dense vector
/// holding only the bounded slots. Reading it repeatedly means holding
/// a borrow of the NLP, which a caller that also re-solves cannot do,
/// so this copies once.
pub fn expand_bounds(
    n_x: usize,
    px_l: &Rc<dyn pounce_linalg::Matrix>,
    px_u: &Rc<dyn pounce_linalg::Matrix>,
    x_l: &dyn Vector,
    x_u: &dyn Vector,
) -> (Vec<Number>, Vec<Number>) {
    let mut lo = vec![Number::NEG_INFINITY; n_x];
    let mut hi = vec![Number::INFINITY; n_x];
    for (pm, src, dst) in [(px_l, x_l, &mut lo), (px_u, x_u, &mut hi)] {
        let Some(em) = pm.as_any().downcast_ref::<ExpansionMatrix>() else {
            continue;
        };
        let vals = compressed_values(src);
        for (ci, &full_pos) in em.expanded_pos_indices().iter().enumerate() {
            let i = full_pos as usize;
            if let (true, Some(&v)) = (i < n_x, vals.get(ci)) {
                dst[i] = v;
            }
        }
    }
    (lo, hi)
}

/// The coordinate whose predicted value leaves its bound by the most,
/// as `(index, the bound it leaves)`.
///
/// This is the half of the bound check that fix-relax keeps. The clamp
/// above answers "put it back", which loses the other coordinates; the
/// refinement needs "which one, and where does it belong", and then
/// re-solves with that coordinate pinned so the rest respond.
///
/// The worst violator is chosen rather than the first by index, so the
/// order of the pins does not depend on how the model was written.
/// `skip` names coordinates already pinned by an earlier pass, which
/// sit ON their bound and would otherwise be picked again.
pub fn worst_violation(
    x_curr: &[Number],
    dx: &[Number],
    lo: &[Number],
    hi: &[Number],
    eps: Number,
    skip: &[usize],
) -> Option<(usize, Number)> {
    let mut worst: Option<(usize, Number, Number)> = None;
    for i in 0..x_curr.len().min(dx.len()) {
        if skip.contains(&i) {
            continue;
        }
        let trial = x_curr[i] + dx[i];
        let (bound, over) = if trial < lo[i] {
            (lo[i], lo[i] - trial)
        } else if trial > hi[i] {
            (hi[i], trial - hi[i])
        } else {
            continue;
        };
        if over > eps && worst.is_none_or(|(_, _, w)| over > w) {
            worst = Some((i, bound, over));
        }
    }
    worst.map(|(i, bound, _)| (i, bound))
}

/// Extract dense values from a `dyn Vector` that wraps a `DenseVector`.
/// Returns an empty vector when the downcast fails (and the bound
/// vector is just treated as having no entries — the boundcheck then
/// silently no-ops, matching upstream's behavior when bounds aren't
/// represented as DenseVectors).
fn compressed_values(v: &dyn Vector) -> Vec<Number> {
    use pounce_linalg::dense_vector::DenseVector;
    match v.as_any().downcast_ref::<DenseVector>() {
        // `expanded_values` (not `values`) so a homogeneous bound
        // vector — e.g. every lower bound 0 — materializes its scalar
        // instead of tripping `DenseVector::values`'s
        // `!homogeneous` debug_assert (L16).
        Some(dv) => dv.expanded_values(),
        None => Vec::new(),
    }
}

// Quieter index-typed signature helper for callers that pass usize-
// dimensioned slices but receive Index-counted bound dimensions.
#[doc(hidden)]
pub fn _index_to_usize(i: Index) -> usize {
    i as usize
}

#[cfg(test)]
mod tests {
    use super::*;
    use pounce_linalg::Vector;
    use pounce_linalg::compound_vector::{CompoundVector, CompoundVectorSpace};
    use pounce_linalg::dense_vector::{DenseVector, DenseVectorSpace};
    use pounce_linalg::expansion_matrix::{ExpansionMatrix, ExpansionMatrixSpace};

    fn make_dv(values: &[Number]) -> DenseVector {
        let space = DenseVectorSpace::new(values.len() as Index);
        let mut dv = DenseVector::new(space);
        dv.values_mut().copy_from_slice(values);
        dv
    }

    /// A homogeneous DenseVector of length `dim`, every entry `scalar`.
    /// Built via `Vector::set`, which puts the vector in homogeneous
    /// representation (no materialized storage) — the state under which
    /// `DenseVector::values()` debug_asserts.
    fn make_homogeneous_dv(dim: Index, scalar: Number) -> DenseVector {
        let space = DenseVectorSpace::new(dim);
        let mut dv = DenseVector::new(space);
        dv.set(scalar);
        assert!(dv.is_homogeneous());
        dv
    }

    /// `(px, compressed)` for a bound present on the given positions.
    fn expansion(n: Index, positions: &[Index]) -> Rc<dyn pounce_linalg::Matrix> {
        let space = ExpansionMatrixSpace::new(n, positions.len() as Index, positions, 0);
        Rc::new(ExpansionMatrix::new(space)) as Rc<dyn pounce_linalg::Matrix>
    }

    #[test]
    fn expand_bounds_puts_infinity_where_a_bound_is_absent() {
        // only x1 has a lower bound, only x2 an upper one
        let (lo, hi) = expand_bounds(
            3,
            &expansion(3, &[1]),
            &expansion(3, &[2]),
            &make_dv(&[-2.0]),
            &make_dv(&[7.0]),
        );
        assert_eq!(lo, vec![Number::NEG_INFINITY, -2.0, Number::NEG_INFINITY]);
        assert_eq!(hi, vec![Number::INFINITY, Number::INFINITY, 7.0]);
    }

    #[test]
    fn expand_bounds_materializes_a_homogeneous_vector() {
        // every lower bound 0, stored as a scalar rather than an array
        let (lo, _) = expand_bounds(
            2,
            &expansion(2, &[0, 1]),
            &expansion(2, &[]),
            &make_homogeneous_dv(2, 0.0),
            &make_dv(&[]),
        );
        assert_eq!(lo, vec![0.0, 0.0]);
    }

    #[test]
    fn worst_violation_takes_the_largest_overshoot_not_the_first() {
        let x = [0.5, 0.5, 0.5];
        let dx = [-0.6, -2.0, -0.7];
        let lo = [0.0, 0.0, 0.0];
        let hi = [10.0, 10.0, 10.0];
        // x1 is out by 1.5, x0 by 0.1, x2 by 0.2
        let (i, bound) = worst_violation(&x, &dx, &lo, &hi, 1e-9, &[]).unwrap();
        assert_eq!(i, 1);
        assert_eq!(bound, 0.0);
    }

    #[test]
    fn worst_violation_skips_what_is_already_pinned() {
        let x = [0.5, 0.5];
        let dx = [-0.6, -2.0];
        let lo = [0.0, 0.0];
        let hi = [10.0, 10.0];
        let (i, _) = worst_violation(&x, &dx, &lo, &hi, 1e-9, &[1]).unwrap();
        assert_eq!(i, 0, "the worst one is pinned, so the next is taken");
    }

    #[test]
    fn worst_violation_reports_an_upper_bound_too() {
        let x = [0.5];
        let dx = [3.0];
        let (i, bound) = worst_violation(&x, &dx, &[0.0], &[1.0], 1e-9, &[]).unwrap();
        assert_eq!((i, bound), (0, 1.0));
    }

    #[test]
    fn worst_violation_is_none_inside_the_bounds_and_within_eps() {
        let x = [0.5];
        assert!(worst_violation(&x, &[0.1], &[0.0], &[1.0], 1e-9, &[]).is_none());
        // just outside, but under the tolerance
        assert!(worst_violation(&x, &[0.5 + 1e-12], &[0.0], &[1.0], 1e-9, &[]).is_none());
    }
}

/// Hold every coordinate the step takes past a bound AT that bound, by
/// pinning and re-solving rather than by clipping.
///
/// Returns the refined step and the coordinates pinned to reach it,
/// worst violator first. This is the loop upstream runs under
/// `sens_boundcheck`, and the reason it is not a clamp: clipping a
/// coordinate leaves every other one at its linear-predictor value, so
/// the result satisfies the bounds and no longer satisfies the
/// constraints.
///
/// `dx_plain` is the unrefined step over the whole compound KKT vector,
/// `x_curr` and the bounds cover its primal block, and all four are in
/// the same units. Each pass augments the held factorization with every
/// pin so far and takes the Schur complement over them, so the cost is
/// one dense `k × k` solve and a backsolve per pass. The factorization
/// is never rebuilt.
///
/// Passes stop when nothing is outside its bound by more than `eps`, at
/// `max_passes`, or when a pin cannot be achieved. That last case is
/// how an over-determined pin set is caught: once the pins exhaust the
/// problem's degrees of freedom no step can hold them all, the
/// augmented system is singular, and a dense LU returns a large
/// solution instead of reporting it. The check is whether the pass
/// actually moved the pinned coordinates where it asked, and the step
/// returned is the last one that did.
pub fn refine_step_onto_bounds<B>(
    backsolver: &B,
    dx_plain: &[Number],
    x_curr: &[Number],
    lo: &[Number],
    hi: &[Number],
    eps: Number,
    max_passes: usize,
) -> Result<(Vec<Number>, Vec<usize>), String>
where
    B: crate::backsolver::SensBacksolver + Clone,
{
    use crate::sens_app::{SensApplication, SensOptions};

    let n_full = dx_plain.len();
    let mut dx = dx_plain.to_vec();
    let mut pins: Vec<(usize, Number)> = Vec::new();

    for _ in 0..max_passes {
        let Some((i, bound)) = worst_violation(
            x_curr,
            &dx,
            lo,
            hi,
            eps,
            &pins.iter().map(|&(p, _)| p).collect::<Vec<_>>(),
        ) else {
            break;
        };
        pins.push((i, bound));

        // A and B are both the pin rows, so the Schur complement is
        // square, which the dense driver requires. The right-hand side
        // is negated: the step returned at a pinned coordinate is the
        // negative of the displacement asked for.
        let rows: Vec<Index> = pins.iter().map(|&(p, _)| p as Index).collect();
        let signs = vec![1; rows.len()];
        let mk = |r: Vec<Index>| {
            IndexSchurData::from_parts(r, signs.clone()).map_err(|e| format!("{e:?}"))
        };
        let opts = SensOptions {
            run_sens: true,
            ..SensOptions::default()
        };
        let mut pin_app = SensApplication::new(mk(rows.clone())?, backsolver.clone(), opts);

        // measured from the PLAIN step, since this pass carries every
        // pin at once; adding successive corrections counts the earlier
        // pins twice
        let rhs: Vec<Number> = pins
            .iter()
            .map(|&(p, b)| (x_curr[p] + dx_plain[p]) - b)
            .collect();
        let mut du = vec![0.0; rows.len()];
        let mut corr = vec![0.0; n_full];
        if !pin_app.run_sens_step(&mk(rows)?, &rhs, &mut du, &mut corr) {
            return Err("SensApplication::run_sens_step failed".into());
        }

        let achieved = pins
            .iter()
            .zip(rhs.iter())
            .all(|(&(p, _), r)| (corr[p] + r).abs() <= 1e-6 * r.abs().max(1.0));
        if !achieved {
            pins.pop();
            break;
        }
        for (k, d) in dx.iter_mut().enumerate() {
            *d = dx_plain[k] + corr[k];
        }
    }
    Ok((dx, pins.into_iter().map(|(p, _)| p).collect()))
}
