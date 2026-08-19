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
    fn combinations_advance_in_lexicographic_order_and_terminate() {
        // The enumeration order is the determinism promise of the
        // directional search: size first, least index within a size.
        let n = 4;
        let mut seen: Vec<Vec<usize>> = Vec::new();
        for size in 0..=n {
            let mut combo: Vec<usize> = (0..size).collect();
            loop {
                seen.push(combo.clone());
                if !next_combination(&mut combo, n) {
                    break;
                }
            }
        }
        assert_eq!(seen.len(), 16, "2^4 subsets in total");
        let expected_size_2: Vec<Vec<usize>> = vec![
            vec![0, 1],
            vec![0, 2],
            vec![0, 3],
            vec![1, 2],
            vec![1, 3],
            vec![2, 3],
        ];
        let got_size_2: Vec<Vec<usize>> = seen.iter().filter(|c| c.len() == 2).cloned().collect();
        assert_eq!(got_size_2, expected_size_2);
        for w in seen.windows(2) {
            assert!(w[0].len() <= w[1].len(), "sizes never decrease");
        }
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

/// A bound this far out is the reader's absent-bound sentinel rather
/// than a bound, and a step cannot cross it.
const NO_BOUND_LO: Number = -1e19;
/// Mirror of [`NO_BOUND_LO`].
const NO_BOUND_HI: Number = 1e19;
/// A segment shorter than this has not advanced the path, so the
/// rows changed at its start stay barred from changing back.
const PATH_MIN_SEGMENT: Number = 1e-12;

/// One breakpoint the path stopped at.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PathSegment {
    /// Fraction of the perturbation applied when this segment ended,
    /// measured from the base point.
    pub at: Number,
    /// Var-x row of the variable whose bound status changed, whatever
    /// the kind of change. A release is detected on the bound's
    /// multiplier row, but it is recorded here by the variable it
    /// frees, so a caller never needs the multiplier layout to read
    /// the record.
    pub var_row: usize,
    /// `true` when the bound involved is the variable's lower bound.
    pub lower: bool,
    /// `true` when the variable reached the bound and is held there
    /// from this fraction on, `false` when it left it: either a bound
    /// active at the base whose multiplier reached zero, or a hold
    /// this path added earlier whose multiplier crossed zero.
    pub pinned: bool,
}

/// A variable the path holds at a bound it reached, with the
/// accumulated multiplier on its Schur row. The multiplier starts at
/// zero where the hold is added, exactly the crossing, takes a sign on
/// the segment after, and the hold drops where it crosses zero again,
/// which is the "drop" half of add-and-drop.
#[derive(Clone, Copy, Debug)]
struct PathHold {
    /// Var-x row held.
    row: usize,
    /// `true` when the bound held is the variable's lower bound. Only
    /// the record reads this: the drop test does not care which side
    /// the hold is on.
    lower: bool,
    /// Accumulated Schur-row multiplier, in whatever sign convention
    /// the augmented system uses: the drop test only asks when it
    /// crosses zero, so the convention never needs to be named.
    mult: Number,
}

/// Apply the perturbation a little at a time, stopping wherever the
/// active set changes.
///
/// [`refine_step_onto_bounds`] decides every condition at the base
/// point. This advances instead: it takes the fraction of the
/// perturbation that reaches the first breakpoint, applies that one
/// change, and continues from there with the remainder under the new
/// active set. The result is piecewise linear in the parameter, which
/// is the exact solution for a QP, whose solution is piecewise affine
/// in the parameter. For an NLP it stays a predictor, because nothing
/// is re-linearized between breakpoints.
///
/// Three kinds of breakpoint end a segment, all ratio tests on
/// quantities the step already carries. A variable strictly inside its
/// bounds reaches one, and is held there. A bound active at the base
/// has its multiplier reach zero, and the variable leaves it. A hold
/// this path added earlier has its multiplier cross zero, and the
/// variable leaves that bound too: the direction changes at every
/// breakpoint, so a bound reached under one direction may stop binding
/// under a later one.
///
/// Releasing a base-active bound needs no right-hand-side shift,
/// unlike the base-point refinement. The path stops exactly where the
/// multiplier reaches zero, so there is nothing left to drive to zero.
/// Dropping a hold needs no re-factorization at all, since the hold is
/// a Schur row rather than a term in the held factor.
///
/// Returns the accumulated step and the breakpoints crossed. When
/// `max_iter` segments are used before the target is reached, the
/// remainder is taken in one step under the active set reached, since
/// stopping short would answer a perturbation the caller did not ask
/// for. A returned segment count equal to `max_iter` is what says that
/// happened.
#[allow(clippy::too_many_arguments)]
pub fn step_along_path<B>(
    backsolver: &B,
    rhs_plain: &[Number],
    x_curr: &[Number],
    lo: &[Number],
    hi: &[Number],
    multipliers: &[BoundMultiplier],
    max_iter: usize,
    forced_active: &[usize],
    initial_holds: &[(usize, bool)],
) -> Result<(Vec<Number>, Vec<PathSegment>), String>
where
    B: crate::backsolver::SensBacksolver + Clone,
{
    let n_full = backsolver.dim();
    let n_x = x_curr.len().min(lo.len()).min(hi.len());
    if rhs_plain.len() != n_full {
        return Err("step_along_path: rhs length is not the KKT dimension".into());
    }
    // The same conversion the refinement makes, for the same reason:
    // these arrive in the solve's coordinates and get compared against
    // the z rows of a step, which are in the model's.
    let mult_nat: Vec<BoundMultiplier> = match backsolver.natural_units_factor() {
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
    let bound_rows: Option<Vec<crate::backsolver::BoundRow>> =
        backsolver.bound_rows().map(|b| b.to_vec());
    let can_release = backsolver.supports_release();

    // Which bounds the factorization enforces, decided once. Active
    // means the multiplier dominates the slack. A converged interior
    // point never sits ON a bound: an active bound's slack is order mu
    // over the multiplier, so testing slack against `eps` calls every
    // active bound inactive and the path never releases anything.
    // Complementarity splits the two sides cleanly, z of order one
    // against slack of order mu on the active side and the reverse on
    // the inactive, which is the same split the activity classifier
    // draws.
    //
    // The split is evaluated at the BASE point, which is what makes
    // deciding it here, before the loop, correct rather than a cache:
    // activity of a multiplier row is a property of the factorization,
    // whose sigma for this bound was frozen at the base, and a bound
    // inactive there is represented by a Schur-row hold if the path
    // reaches it, never by its multiplier row. Testing accumulated
    // values instead let a near-bound inactive multiplier drift past
    // its shrinking slack mid-path and "release" a bound that was
    // never held, putting a departure in the record for a variable
    // that was not on that bound.
    //
    // What stays live at every consumer is the released list: a
    // base-active bound whose row has been released is no longer in
    // the factorization, from that fraction on.
    let mut base_active_row: Vec<[Option<usize>; 2]> = vec![[None, None]; n_x];
    if let Some(rows) = bound_rows.as_ref() {
        for br in rows {
            if br.var_row >= n_x {
                continue;
            }
            let slack_base = if br.lower {
                x_curr[br.var_row] - lo[br.var_row]
            } else {
                hi[br.var_row] - x_curr[br.var_row]
            };
            if !slack_base.is_finite() {
                continue;
            }
            if forced_active.contains(&br.row)
                || mult_nat
                    .iter()
                    .any(|m| m.row == br.row && m.base > slack_base)
            {
                let side = if br.lower { 0 } else { 1 };
                base_active_row[br.var_row][side] = Some(br.row);
            }
        }
    }
    let base_active_rows: Vec<usize> = base_active_row
        .iter()
        .flatten()
        .filter_map(|slot| *slot)
        .collect();

    let mut acc = vec![0.0; n_full];
    let mut t = 0.0_f64;
    // Seeded state from the directional-derivative decision at a
    // degenerate base point. A weakly active row the direction holds
    // arrives released, since its order-one sigma is wrong once the
    // direction later changes, and pinned through a Schur hold with
    // zero accumulated multiplier, exactly as a hold added at fraction
    // zero would, so the drop test can end it later like any other. A
    // weakly active row the direction leaves goes into the
    // base-activity table below instead, so the release scan frees it
    // at the fraction where its multiplier actually reaches zero:
    // essentially zero at an exact kink, and partway along the step
    // when the held solve sits inside the ambiguous band, where the
    // bound is genuinely active for the first stretch. Deciding those
    // rows at fraction zero released them a sixth of a step early on
    // the CSTR held at 75% of the breakpoint fraction, and overshot
    // tenfold against the walk's own release.
    let mut holds: Vec<PathHold> = initial_holds
        .iter()
        .map(|&(row, lower)| PathHold {
            row,
            lower,
            mult: 0.0,
        })
        .collect();
    let mut released: Vec<usize> = initial_holds
        .iter()
        .filter_map(|&(var_row, lower)| {
            bound_rows.as_ref().and_then(|rows| {
                rows.iter()
                    .find(|b| b.var_row == var_row && b.lower == lower)
                    .map(|b| b.row)
            })
        })
        .collect();
    let mut segments: Vec<PathSegment> = Vec::new();
    // Rows already changed at the fraction the path currently ends at.
    // A zero-length segment is where cycling comes from, so a row that
    // just changed cannot change back at the same fraction. The list
    // clears as soon as the path advances: barring a row any longer
    // makes it miss real breakpoints in the following segment,
    // which showed up as a released variable whose next bound crossing
    // went unrecorded.
    let mut changed_here: Vec<usize> = Vec::new();
    let mut last_beta = 1.0_f64;

    /// What the earliest breakpoint found so far does.
    #[derive(Clone, Copy, PartialEq)]
    enum Event {
        ReachLower,
        ReachUpper,
        ReleaseBase,
        DropHold,
    }

    for _ in 0..max_iter {
        if last_beta > PATH_MIN_SEGMENT {
            changed_here.clear();
        }
        let held: Vec<usize> = holds.iter().map(|h| h.row).collect();
        let (d, du) = path_direction(backsolver, rhs_plain, &released, &held)?;
        let remaining = 1.0 - t;
        if remaining <= 0.0 {
            break;
        }

        let mut best: Option<(Number, usize, Event)> = None;
        let mut offer = |beta: Number, row: usize, ev: Event| {
            if !beta.is_finite() || beta < 0.0 || beta > remaining {
                return;
            }
            match best {
                Some((b, _, _)) if b <= beta => {}
                _ => best = Some((beta, row, ev)),
            }
        };

        // A free variable reaching a bound. A bound the held
        // factorization still enforces is not reachable this way: its
        // variable sits essentially on it already, and holding it AGAIN
        // through a Schur row would enforce the same bound twice. Such
        // a bound leaves the active set only through its own
        // multiplier's release below.
        for i in 0..n_x {
            if holds.iter().any(|h| h.row == i) || changed_here.contains(&i) {
                continue;
            }
            // Base activity was decided once, at the table above; only
            // the released exclusion is live, since a released bound
            // left the factorization mid-path.
            let factor_holds = |lower_side: bool| -> bool {
                let side = if lower_side { 0 } else { 1 };
                base_active_row[i][side].is_some_and(|r| !released.contains(&r))
            };
            let v = x_curr[i] + acc[i];
            if d[i] < 0.0 && lo[i] > NO_BOUND_LO && !factor_holds(true) {
                offer((lo[i] - v) / d[i], i, Event::ReachLower);
            }
            if d[i] > 0.0 && hi[i] < NO_BOUND_HI && !factor_holds(false) {
                offer((hi[i] - v) / d[i], i, Event::ReachUpper);
            }
        }
        // A bound active at the base whose multiplier reaches zero.
        // Base activity comes from the table above; which rows have
        // since been released stays a live check.
        if can_release {
            for m in &mult_nat {
                if released.contains(&m.row)
                    || changed_here.contains(&m.row)
                    || !base_active_rows.contains(&m.row)
                {
                    continue;
                }
                let z_curr = m.base + acc[m.row];
                if d[m.row] < 0.0 {
                    offer(-z_curr / d[m.row], m.row, Event::ReleaseBase);
                }
            }
        }
        // A hold this path added whose multiplier crosses zero. The
        // rate is the row's `du` under the current direction. Which
        // sign is the valid side depends on conventions three layers
        // deep, so the test does not choose one: the multiplier took
        // some sign on the segment after the hold was added, and
        // crossing zero from that side is what ends the hold's
        // validity. At creation the multiplier is exactly zero and the
        // product below is zero, so a fresh hold cannot drop before it
        // has accumulated a sign.
        for (k, h) in holds.iter().enumerate() {
            if changed_here.contains(&h.row) {
                continue;
            }
            let rate = du[k];
            if h.mult * rate < 0.0 {
                offer(-h.mult / rate, h.row, Event::DropHold);
            }
        }

        let Some((beta, row, ev)) = best else {
            // Nothing changes before the target, so the rest is one step.
            for (a, dv) in acc.iter_mut().zip(d.iter()) {
                *a += remaining * dv;
            }
            t = 1.0;
            break;
        };

        for (a, dv) in acc.iter_mut().zip(d.iter()) {
            *a += beta * dv;
        }
        for (k, h) in holds.iter_mut().enumerate() {
            h.mult += beta * du[k];
        }
        last_beta = beta;
        t += beta;
        changed_here.push(row);
        let (var_row, lower) = match ev {
            Event::ReachLower | Event::ReachUpper => {
                let lower = ev == Event::ReachLower;
                holds.push(PathHold {
                    row,
                    lower,
                    mult: 0.0,
                });
                (row, lower)
            }
            Event::ReleaseBase => {
                // The release scan only offers rows it found bound
                // metadata for, so this lookup cannot miss.
                let Some(br) = bound_rows
                    .as_ref()
                    .and_then(|rows| rows.iter().find(|b| b.row == row))
                else {
                    return Err("step_along_path: released a row with no bound metadata".into());
                };
                // Bar the released variable's own row too: the reach
                // scan works in var rows while the release recorded the
                // multiplier row, and without this the variable can be
                // re-held at the same fraction it was just released.
                changed_here.push(br.var_row);
                released.push(row);
                (br.var_row, br.lower)
            }
            Event::DropHold => {
                // The drop event came from iterating the holds, so the
                // hold is present.
                let Some(h) = holds.iter().find(|h| h.row == row).copied() else {
                    return Err("step_along_path: dropped a hold that does not exist".into());
                };
                holds.retain(|h| h.row != row);
                (row, h.lower)
            }
        };
        segments.push(PathSegment {
            at: t,
            var_row,
            lower,
            pinned: matches!(ev, Event::ReachLower | Event::ReachUpper),
        });
    }

    // The cap bound before the target was reached, so take what is left
    // under the active set reached.
    if t < 1.0 {
        let held: Vec<usize> = holds.iter().map(|h| h.row).collect();
        let (d, _) = path_direction(backsolver, rhs_plain, &released, &held)?;
        for (a, dv) in acc.iter_mut().zip(d.iter()) {
            *a += (1.0 - t) * dv;
        }
    }
    Ok((acc, segments))
}

/// The step for the whole perturbation under the active set the path
/// has reached: released bounds out of the operator with their
/// multipliers constrained to stay at zero, and held variables kept
/// where they are.
///
/// The multiplier constraint is not optional. The re-factored released
/// operator drops the bound's diagonal term, but the factor's
/// complementarity row for that bound still couples the direction
/// through the base slack and multiplier it was built from, and
/// without the constraint the released direction is measurably wrong:
/// on a two-variable QP the free direction after a release came back
/// [1.154, 0.194] against the analytic [1.227, 0.454].
/// A bound the classifier could not call active or inactive at the
/// base point: variable on the bound with a multiplier of the same
/// order as the slack, both order sqrt(mu). The solution map has a
/// kink there, and no single linear step is right for both sides.
#[derive(Clone, Copy, Debug)]
pub struct WeakBound {
    /// Bound-multiplier row in the compound KKT vector.
    pub row: usize,
    /// Var-x row of the variable the bound covers.
    pub var_row: usize,
    /// `true` when the bound is the variable's lower bound.
    pub lower: bool,
}

/// The directional derivative at a degenerate base point: the QP of
/// Pirnay, Lopez-Negrete and Biegler 2012, eq. 14, solved as an
/// active-set search over the weakly active rows on the held
/// factorization.
///
/// Every weakly active row is released in every trial, because its
/// sigma is `z / s` with both of order sqrt(mu), an order-one term
/// that half-enforces a bound the direction may need to leave, and it
/// is wrong for both sides of the kink. A candidate working set then
/// pins its variable rows to zero movement through Schur rows. The
/// candidate is accepted when
///
/// 1. every out variable moves into its feasible side, and
/// 2. every pin is necessary: removing it alone makes its variable
///    move into violation.
///
/// The necessity probe stands in for the dual-feasibility sign test
/// on the pin's Schur multiplier, whose sign convention the rest of
/// this file deliberately never names. The two tests agree wherever
/// the returned direction differs: a pin whose multiplier is exactly
/// zero at a doubly degenerate QP can be in or out, and both answers
/// give the same direction.
///
/// Candidates are enumerated by size and then by least index, so the
/// result is deterministic, and every trial counts against
/// `max_iter`, the shared budget. Returns the direction, the var rows
/// pinned in the accepted set, and the trials spent. `Err` when no
/// candidate fits the budget or none is sign-consistent, and the
/// caller falls back to the plain direction and says so.
pub fn directional_step<B>(
    backsolver: &B,
    rhs_plain: &[Number],
    weak: &[WeakBound],
    max_iter: usize,
) -> Result<(Vec<Number>, Vec<usize>, usize), String>
where
    B: crate::backsolver::SensBacksolver + Clone,
{
    let released: Vec<usize> = weak.iter().map(|w| w.row).collect();
    let mut trials = 0usize;
    // dx tolerance, relative to the direction's own magnitude so the
    // decision is invariant to the perturbation's scale: an absolute
    // tolerance accepted the all-released set at a perturbation of
    // 1e-10, reading the holding side's derivative as -1 instead of 0.
    // Zero movement of a pinned variable is roundoff relative to the
    // direction's norm, so pinned rows still clear it.
    const EPS_REL: Number = 1e-9;
    let scale_of =
        |d: &[Number]| -> Number { d.iter().fold(0.0_f64, |a, &b| a.max(b.abs())).max(1e-300) };
    let feasible = |w: &WeakBound, di: Number, tol: Number| -> bool {
        if w.lower { di >= -tol } else { di <= tol }
    };
    let violates = |w: &WeakBound, di: Number, tol: Number| -> bool {
        if w.lower { di < -tol } else { di > tol }
    };

    let n = weak.len();
    let budget_err = || {
        format!(
            "directional derivative: the budget of {max_iter} trials \
             ran out over {n} weakly active bound(s)"
        )
    };
    // Working sets are generated lazily in size-then-least-index
    // order: the first accepted candidate is deterministic, the budget
    // bounds the work, and nothing is materialized. Building all 2^n
    // masks up front allocated gigabytes near n = 30 to try at most
    // max_iter of them, and overflowed the shift at n = 32.
    for size in 0..=n {
        let mut combo: Vec<usize> = (0..size).collect();
        loop {
            if trials >= max_iter {
                return Err(budget_err());
            }
            let pinned: Vec<usize> = combo.iter().map(|&k| weak[k].var_row).collect();
            let (d, _) = path_direction(backsolver, rhs_plain, &released, &pinned)?;
            trials += 1;
            let tol = EPS_REL * scale_of(&d);
            let out_ok = (0..n)
                .filter(|k| !combo.contains(k))
                .all(|k| feasible(&weak[k], d[weak[k].var_row], tol));
            if out_ok {
                // Necessity of each pin, one removal probe per member.
                let mut all_needed = true;
                for &k in &combo {
                    if trials >= max_iter {
                        return Err(budget_err());
                    }
                    let probe: Vec<usize> = combo
                        .iter()
                        .filter(|&&j| j != k)
                        .map(|&j| weak[j].var_row)
                        .collect();
                    let (dp, _) = path_direction(backsolver, rhs_plain, &released, &probe)?;
                    trials += 1;
                    let ptol = EPS_REL * scale_of(&dp);
                    if !violates(&weak[k], dp[weak[k].var_row], ptol) {
                        all_needed = false;
                        break;
                    }
                }
                if all_needed {
                    return Ok((d, pinned, trials));
                }
            }
            if !next_combination(&mut combo, n) {
                break;
            }
        }
    }
    Err(format!(
        "directional derivative: no sign-consistent working set over \
         {n} weakly active bound(s)"
    ))
}

/// Advance `combo` to the next lexicographic combination of its size
/// over `0..n`, returning `false` when it was the last one. The empty
/// combination has no successor.
fn next_combination(combo: &mut [usize], n: usize) -> bool {
    let k = combo.len();
    let mut i = k;
    while i > 0 {
        i -= 1;
        if combo[i] != i + n - k {
            combo[i] += 1;
            for j in i + 1..k {
                combo[j] = combo[j - 1] + 1;
            }
            return true;
        }
    }
    false
}

pub(crate) fn path_direction<B>(
    backsolver: &B,
    rhs_plain: &[Number],
    released: &[usize],
    pinned: &[usize],
) -> Result<(Vec<Number>, Vec<Number>), String>
where
    B: crate::backsolver::SensBacksolver + Clone,
{
    use crate::sens_app::{SensApplication, SensOptions};

    let n_full = backsolver.dim();
    let mut d = vec![0.0; n_full];
    let ok = if released.is_empty() {
        backsolver.solve(rhs_plain, &mut d)
    } else {
        backsolver.solve_released(released, rhs_plain, &mut d)
    };
    if !ok {
        return Err("step_along_path: back-solve failed".into());
    }
    if pinned.is_empty() {
        return Ok((d, Vec::new()));
    }
    // Hold each variable where the path left it, on its bound, by
    // asking the augmented system for the correction that takes its
    // further movement to zero.
    let rows: Vec<Index> = pinned.iter().map(|&r| r as Index).collect();
    let signs = vec![1; rows.len()];
    let mk =
        |r: Vec<Index>| IndexSchurData::from_parts(r, signs.clone()).map_err(|e| format!("{e:?}"));
    let opts = SensOptions {
        run_sens: true,
        ..SensOptions::default()
    };
    let view = ReleasedView {
        base: backsolver.clone(),
        rows: released.to_vec(),
    };
    let mut app = SensApplication::new(mk(rows.clone())?, view, opts);
    let rhs: Vec<Number> = pinned.iter().map(|&i| d[i]).collect();
    let mut du = vec![0.0; rows.len()];
    let mut corr = vec![0.0; n_full];
    if !app.run_sens_step(&mk(rows)?, &rhs, &mut du, &mut corr) {
        return Err(format!(
            "step_along_path: augmented solve failed (holds {pinned:?}, released {released:?})"
        ));
    }
    for (k, v) in d.iter_mut().enumerate() {
        *v += corr[k];
    }
    Ok((d, du))
}
