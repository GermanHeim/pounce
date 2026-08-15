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
//! clipped and every other one keeps a value computed as though it had
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
/// the number of conditions, and the default `max_passes` of 16 is 136
/// back-solves rather than 16.
///
/// `multipliers` carry their base values in the solve's own
/// coordinates. They are converted here, once, with the backsolver's
/// [`SensBacksolver::natural_units_factor`], so they agree with the `z`
/// rows of `dx_plain` before either is used.
///
/// Passes stop when nothing is violated, at `max_passes`, or when a
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
    eps: Number,
    max_passes: usize,
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

    // Where the bound-multiplier blocks start, and which variable each
    // multiplier row constrains. Without that map a bound cannot be
    // released, since the release acts on the variable's own row.
    let bound_rows = backsolver.bound_rows();

    // One condition per pass. `x_row` is the row it acts on -- always
    // an x row, for a release as much as for a pin -- `diag` is the
    // bordered system's (2,2) entry, and `report_row` is what the
    // caller is told, which stays the multiplier's own row for a
    // release so the returned list keeps its documented meaning.
    struct Cond {
        x_row: usize,
        rhs: Number,
        diag: Number,
        report_row: usize,
        /// `(multiplier row, base value)` when this is a release.
        release: Option<(usize, Number)>,
    }
    let mut conds: Vec<Cond> = Vec::new();

    for _ in 0..max_passes {
        let taken: Vec<usize> = conds.iter().map(|c| c.x_row).collect();
        let taken_z: Vec<usize> = conds
            .iter()
            .filter_map(|c| c.release.map(|(r, _)| r))
            .collect();

        // one condition per pass, primal first: a variable outside its
        // bound is the more direct violation, and releasing a bound can
        // only matter once the variables it constrains have settled
        let next = match worst_violation(x_curr, &dx, lo, hi, eps, &taken) {
            Some((i, bound)) => Some(Cond {
                x_row: i,
                rhs: (x_curr[i] + dx_plain[i]) - bound,
                diag: 0.0,
                report_row: i,
                release: None,
            }),
            None => bound_rows.and_then(|br| {
                multipliers
                    .iter()
                    .filter(|m| !taken_z.contains(&m.row))
                    .filter_map(|m| br.iter().find(|b| b.row == m.row).map(|b| (m, b)))
                    .filter(|(_, b)| !taken.contains(&b.var_row))
                    .map(|(m, b)| (m, b, m.base + dx[m.row]))
                    .filter(|&(_, _, v)| v < -eps)
                    // most negative first
                    .min_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal))
                    .and_then(|(m, b, _)| {
                        let i = b.var_row;
                        let lower = b.lower;
                        // The slack the bound carries at the base point.
                        // Its reciprocal is the `sigma` this release
                        // takes back off the x diagonal, so the tiny
                        // number is only ever formed as `s / z`.
                        let s = if lower {
                            x_curr[i] - lo[i]
                        } else {
                            hi[i] - x_curr[i]
                        };
                        if !s.is_finite() || m.base == 0.0 {
                            return None;
                        }
                        // Rank-1 downdate of `sigma = z / s` at row `i`,
                        // which is what takes this bound back out of
                        // the active set.
                        //
                        // The released x row also wants the multiplier
                        // moved to the right-hand side. Most of that
                        // shift is already carried by the parametric
                        // right-hand side -- the barrier correction
                        // leaves `-mu` on the multiplier's row, and the
                        // elimination divides it by `s` -- but only most
                        // of it, since `mu = s * z` holds at the base
                        // point to the solve's tolerance and no better.
                        // Taking the shift as zero on that identity
                        // costs three orders of magnitude at a default
                        // tolerance, so it is recovered exactly from the
                        // step instead: the multiplier's own row gives
                        // `r_z / s = sigma * dx_i + dz`, and what is
                        // left is the multiplier the plain step
                        // predicts, scaled by `s / z`.
                        // The x row carries a lower bound's multiplier
                        // with the opposite sign to an upper one, so
                        // the shift is antisymmetric between the two
                        // sides even though `sigma` is not.
                        let shift = (s / m.base) * (m.base + dx_plain[m.row]);
                        Some(Cond {
                            x_row: i,
                            rhs: if lower { -shift } else { shift },
                            diag: s / m.base,
                            report_row: m.row,
                            release: Some((m.row, m.base)),
                        })
                    })
            }),
        };
        let Some(cond) = next else { break };
        conds.push(cond);

        let rows: Vec<Index> = conds.iter().map(|c| c.x_row as Index).collect();
        let signs = vec![1; rows.len()];
        let mk = |r: Vec<Index>| {
            IndexSchurData::from_parts(r, signs.clone()).map_err(|e| format!("{e:?}"))
        };
        let opts = SensOptions {
            run_sens: true,
            ..SensOptions::default()
        };
        let mut pin_app = SensApplication::new(mk(rows.clone())?, backsolver.clone(), opts);
        let rhs: Vec<Number> = conds.iter().map(|c| c.rhs).collect();
        let diag: Vec<Number> = conds.iter().map(|c| c.diag).collect();
        let mut du = vec![0.0; rows.len()];
        let mut corr = vec![0.0; n_full];
        if !pin_app.run_sens_step_with_diag(&mk(rows)?, &rhs, &diag, &mut du, &mut corr) {
            // An exactly singular augmented system, where the
            // near-singular case is what the achievement check below
            // catches. Drop the condition that could not be solved and
            // stop, rather than discarding the refinement and the plain
            // step with it: the caller is told how far it got by the
            // rows returned, which is what every other stop does.
            conds.pop();
            break;
        }

        // The guard is for the singular case, where the conditions
        // have exhausted the degrees of freedom and a dense LU returns a
        // solution around 1e15 rather than reporting it. It is not an
        // accuracy check: a healthy pass lands within a few parts per
        // million, so demanding more than that rejects working ones.
        //
        // A pin is checked on the displacement it asked for. A release
        // is checked on what it was actually for -- the multiplier
        // reaching zero -- since its own row carries the bordered
        // system's diagonal and so does not simply return `-rhs`.
        let achieved = conds.iter().enumerate().all(|(k, c)| match c.release {
            None => (corr[c.x_row] + c.rhs).abs() <= 1e-3 * c.rhs.abs().max(1.0),
            // A release carries the bordered system's diagonal, so its
            // row does not simply return `-rhs`. What it does hold is
            // that row itself, `x_i = -D * u`, and a singular solve
            // breaks it by orders of magnitude. Compared on magnitude
            // so the check does not depend on the driver's sign
            // convention for `u`.
            Some(_) => {
                let vi = dx_plain[c.x_row] + corr[c.x_row];
                let pred = (c.diag * du[k]).abs();
                vi.is_finite() && (vi.abs() - pred).abs() <= 1e-3 * vi.abs().max(1.0)
            }
        });
        if !achieved {
            conds.pop();
            break;
        }
        for (k, d) in dx.iter_mut().enumerate() {
            *d = dx_plain[k] + corr[k];
        }
        // The released bound's own row still carries the value the
        // retained complementarity row produced, which is a by-product
        // of the downdate and not the released multiplier. The released
        // multiplier is zero by construction, so write the step that
        // says so rather than leaving `z / s` noise in the block.
        for c in &conds {
            if let Some((z_row, base)) = c.release {
                dx[z_row] = -base;
            }
        }
    }
    Ok((dx, conds.into_iter().map(|c| c.report_row).collect()))
}
