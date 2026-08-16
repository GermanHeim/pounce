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
//! # Both halves, and why each matters
//!
//! Upstream's fix-relax is two cases (Pirnay, Lopez-Negrete and Biegler
//! 2012, section 2.5), and the name refers to both. Its equation 17
//! pins a variable the step carries past a bound, activating it. Its
//! equation 18 sets a bound multiplier to zero when the step drives it
//! negative, deactivating that bound so the variable can move.
//!
//! They fail differently. Without the pin, a crossing variable is
//! clamped and every other one keeps a value computed as though it had
//! not been. Without the release, a variable sitting on a bound stays
//! there however hard the perturbation pulls it off, because the linear
//! step preserves complementarity. Measured against sIPOPT on a model
//! whose bound wants to release, that second case is the difference
//! between returning 0.0 and 1.667.
//!
//! Both are solved the same way: add the row, re-solve the augmented
//! system through the Schur complement over the added rows, which is
//! what the paper's equations 19 through 22 describe.

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

/// A bound multiplier the step can drive negative: where it sits in the
/// compound KKT vector, and its value at the base point.
///
/// A negative multiplier means the bound should no longer be active,
/// which is the second half of upstream's fix-relax (its equation 18).
pub struct BoundMultiplier {
    /// Row of the compound KKT vector holding this multiplier.
    pub row: usize,
    /// Its value at the converged point, read raw off `curr.z_l` /
    /// `curr.z_u`, so in the coordinates the solve ran in rather than
    /// the model's own. [`refine_step_onto_bounds`] converts it with
    /// the backsolver's own `F`, so every caller hands over the same
    /// raw value and none of them needs to know the convention.
    pub base: Number,
}

/// Repair the active set the step implies, by pinning and releasing.
///
/// Returns the refined step and the compound rows it constrained.
/// This is upstream's fix-relax, both cases:
///
/// * a variable the step carries past a bound is pinned AT that bound,
///   which activates it (their equation 17);
/// * a bound multiplier the step drives negative is set to zero, which
///   deactivates that bound and lets the variable move (equation 18).
///
/// Without the second, a variable sitting on a bound at the base point
/// stays there however hard the perturbation pulls it off, because the
/// step holds complementarity. Measured on a model whose bound wants to
/// release, that is the difference between 0.0 and 1.667.
///
/// Each pass adds one condition and re-solves the augmented system
/// carrying all of them, against the original factorization, so its
/// correction is measured from the plain step rather than the previous
/// pass. Adding successive corrections counts the earlier ones twice.
/// The Schur complement over those rows is what upstream's equations 19
/// through 22 describe. The factorization is never rebuilt, which is
/// what makes this cheaper than a re-solve, but the Schur complement is
/// rebuilt from scratch each pass: pass `k` costs one dense `k × k`
/// solve and `k + 1` back-solves, so the work grows quadratically in
/// the number of conditions, and the default `max_iter` of 16 is 136
/// back-solves rather than 16.
///
/// `multipliers` carry their base values in the solve's own
/// coordinates. They are converted here, once, with the backsolver's
/// [`SensBacksolver::natural_units_factor`], so they agree with the `z`
/// rows of `dx_plain` before either is used.
///
/// Passes stop when nothing is violated, at `max_iter`, or when a
/// condition cannot be achieved, which is how an over-determined set is
/// caught: once the conditions exhaust the problem's degrees of freedom
/// no step satisfies them all, the augmented system is singular, and a
/// dense LU returns a large solution rather than reporting it.
pub fn refine_step_onto_bounds<B>(
    backsolver: &B,
    dx_plain: &[Number],
    x_curr: &[Number],
    lo: &[Number],
    hi: &[Number],
    multipliers: &[BoundMultiplier],
    rhs_plain: &[Number],
    eps: Number,
    max_iter: usize,
) -> Result<(Vec<Number>, Vec<usize>), String>
where
    B: crate::backsolver::SensBacksolver + Clone,
{
    use crate::sens_app::{SensApplication, SensOptions};

    let n_full = dx_plain.len();
    let mut dx = dx_plain.to_vec();
    // Into the units the step is in, before either is read. `F` is
    // indexed by compound row, the same space `BoundMultiplier::row`
    // lives in.
    let multipliers: Vec<BoundMultiplier> = match backsolver.natural_units_factor() {
        None => multipliers
            .iter()
            .map(|m| BoundMultiplier {
                row: m.row,
                base: m.base,
            })
            .collect(),
        Some(f) => multipliers
            .iter()
            .map(|m| BoundMultiplier {
                row: m.row,
                base: m.base * f[m.row],
            })
            .collect(),
    };
    let multipliers = &multipliers[..];
    let bound_rows = backsolver.bound_rows();
    let can_release = backsolver.supports_release() && rhs_plain.len() == n_full;

    // (compound row, right-hand side), the rhs fixed at creation since
    // it is measured from the base step
    let mut pins: Vec<(usize, Number)> = Vec::new();
    // Multiplier rows taken out of the active set. Unlike a pin these
    // never become a Schur condition: they change the operator, so the
    // step is re-solved against a factorization that does not carry
    // their `sigma` at all.
    let mut released: Vec<usize> = Vec::new();
    // The step corrections are measured from. It moves whenever the
    // released set does, since that is a different system.
    let mut dx_base = dx_plain.to_vec();

    for _ in 0..max_iter {
        let taken: Vec<usize> = pins.iter().map(|&(r, _)| r).collect();

        // one condition per pass, primal first: a variable outside its
        // bound is the more direct violation, and releasing a bound can
        // only matter once the variables it constrains have settled
        let next = match worst_violation(x_curr, &dx, lo, hi, eps, &taken) {
            Some((i, bound)) => Some((i, (x_curr[i] + dx_base[i]) - bound)),
            None => None,
        };

        // Pins first. Only when nothing is outside its bound do we look
        // at releasing, so the primal violations settle before a bound
        // leaves the active set.
        let Some((row, rhs_row)) = next else {
            // A release is not a condition on the step, it is a
            // different system: re-solve with that bound's `sigma`
            // gone, then start over from the step that produced.
            let worst = if can_release {
                multipliers
                    .iter()
                    .filter(|m| !released.contains(&m.row))
                    .filter(|m| bound_rows.is_some_and(|br| br.iter().any(|b| b.row == m.row)))
                    .map(|m| (m.row, m.base + dx[m.row]))
                    .filter(|&(_, v)| v < -eps)
                    // most negative first
                    .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(r, _)| r)
            } else {
                None
            };
            let Some(row) = worst else { break };
            released.push(row);
            let mut base = vec![0.0; n_full];
            if !backsolver.solve_released_step(&released, rhs_plain, &mut base) {
                // Could not factor with that bound out. Keep the step
                // reached without it, which is what every other stop
                // here does.
                released.pop();
                break;
            }
            // A released bound's multiplier is zero by construction; its
            // own row of the re-solved step is a by-product of the
            // complementarity row the factor still carries.
            for &r in &released {
                if let Some(m) = multipliers.iter().find(|m| m.row == r) {
                    base[r] = -m.base;
                }
            }
            dx_base = base;
            dx.copy_from_slice(&dx_base);
            // pins are measured from the base, which just moved
            pins.clear();
            continue;
        };
        pins.push((row, rhs_row));

        let rows: Vec<Index> = pins.iter().map(|&(r, _)| r as Index).collect();
        let signs = vec![1; rows.len()];
        let mk = |r: Vec<Index>| {
            IndexSchurData::from_parts(r, signs.clone()).map_err(|e| format!("{e:?}"))
        };
        let opts = SensOptions {
            run_sens: true,
            ..SensOptions::default()
        };
        // Against the released operator, not the converged one: once a
        // bound is out of the active set, every later condition has to
        // be solved in the system that reflects that.
        let view = ReleasedView {
            base: backsolver.clone(),
            rows: released.clone(),
        };
        let mut pin_app = SensApplication::new(mk(rows.clone())?, view, opts);
        let rhs: Vec<Number> = pins.iter().map(|&(_, r)| r).collect();
        let mut du = vec![0.0; rows.len()];
        let mut corr = vec![0.0; n_full];
        if !pin_app.run_sens_step(&mk(rows)?, &rhs, &mut du, &mut corr) {
            // An exactly singular augmented system, where the
            // near-singular case is what the achievement check below
            // catches. Drop the condition that could not be solved and
            // stop, rather than discarding the refinement and the plain
            // step with it: the caller is told how far it got by the
            // rows returned, which is what every other stop does.
            pins.pop();
            break;
        }

        // The guard is for the singular case, where the conditions
        // have exhausted the degrees of freedom and a dense LU returns a
        // solution around 1e15 rather than reporting it. It is not an
        // accuracy check: a healthy pass lands within a few parts per
        // million, so demanding more than that rejects working pins.
        let achieved = pins
            .iter()
            .all(|&(r, want)| (corr[r] + want).abs() <= 1e-3 * want.abs().max(1.0));
        if !achieved {
            pins.pop();
            break;
        }
        for (k, d) in dx.iter_mut().enumerate() {
            *d = dx_base[k] + corr[k];
        }
    }
    let mut out = released.clone();
    out.extend(pins.into_iter().map(|(r, _)| r));
    Ok((dx, out))
}

/// The converged backsolver with a set of bounds out of the active set,
/// so the pin machinery can run against the released system without
/// knowing that is what it is doing.
#[derive(Clone)]
struct ReleasedView<B: crate::backsolver::SensBacksolver + Clone> {
    base: B,
    rows: Vec<usize>,
}

impl<B: crate::backsolver::SensBacksolver + Clone> crate::backsolver::SensBacksolver
    for ReleasedView<B>
{
    fn dim(&self) -> usize {
        self.base.dim()
    }
    fn solve(&self, rhs: &[Number], lhs: &mut [Number]) -> bool {
        self.base.solve_released(&self.rows, rhs, lhs)
    }
    fn natural_units_factor(&self) -> Option<&[Number]> {
        self.base.natural_units_factor()
    }
    fn bound_rows(&self) -> Option<&[crate::backsolver::BoundRow]> {
        self.base.bound_rows()
    }
    fn supports_release(&self) -> bool {
        self.base.supports_release()
    }
    fn solve_released(&self, released: &[usize], rhs: &[Number], lhs: &mut [Number]) -> bool {
        self.base.solve_released(released, rhs, lhs)
    }
    fn solve_released_step(&self, released: &[usize], rhs: &[Number], lhs: &mut [Number]) -> bool {
        self.base.solve_released_step(released, rhs, lhs)
    }
}
