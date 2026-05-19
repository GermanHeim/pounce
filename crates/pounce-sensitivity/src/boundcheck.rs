//! Single-pass bound clamping for the parametric sensitivity step.
//!
//! Mirrors the **role** of upstream
//! [`SensStdStepCalculator::BoundCheck`](../../../ref/Ipopt/contrib/sIPOPT/src/SensStdStepCalc.cpp)
//! — keep the perturbed primal feasible — but uses a simpler single-pass
//! clamp rather than upstream's iterative Schur-refinement loop.
//!
//! # What this does
//!
//! Given a converged iterate `x_curr` (length `n_x`), the algorithm's
//! bound expansion matrices `Px_L` / `Px_U` and the compressed bounds
//! `x_l` / `x_u`, project the proposed step `dx` so that
//! `x_curr + dx ∈ [x_l, x_u]` within `eps`:
//!
//! * For each lower-bounded variable `i` with `x_curr[i] + dx[i] <
//!   x_l[i] - eps`, clip `dx[i] := x_l[i] - x_curr[i]`.
//! * For each upper-bounded variable `i` with `x_curr[i] + dx[i] >
//!   x_u[i] + eps`, clip `dx[i] := x_u[i] - x_curr[i]`.
//!
//! # Difference from upstream
//!
//! Upstream's `BoundCheck` couples back into the Schur driver: each
//! violation triggers a new row in `A` and `B`, followed by a
//! re-factorize + re-solve, so the unviolated coordinates of `dx` shift
//! to absorb the clamp under the IFT constraints. The single-pass
//! clamp here keeps the non-violating coordinates frozen, which is
//! cheaper but less accurate when violations are deep. For most user
//! workflows the simple clamp is enough to keep the perturbed primal
//! inside the feasible set without changing problem topology; users
//! who need the full refinement should track
//! [pounce#7](https://github.com/jkitchin/pounce/issues/7).
//!
//! Returns the number of clamped indices.

use pounce_common::types::{Index, Number};
use pounce_linalg::expansion_matrix::ExpansionMatrix;
use pounce_linalg::Vector;
use std::rc::Rc;

/// Clamp `dx` (length `n_x`) so that `x_curr + dx ∈ [x_l, x_u]`
/// within `eps`. See module docs.
///
/// `px_l`, `px_u` are the algorithm-side bound expansion matrices
/// (mapping compressed-bound slots into full var-x). `x_l`, `x_u`
/// are the compressed bound vectors (length `n_lb`, `n_ub`).
///
/// Returns the count of clamped entries (0 means the step was
/// already feasible).
pub fn clamp_step_to_bounds(
    x_curr: &[Number],
    dx: &mut [Number],
    px_l: &Rc<dyn pounce_linalg::Matrix>,
    px_u: &Rc<dyn pounce_linalg::Matrix>,
    x_l: &dyn Vector,
    x_u: &dyn Vector,
    eps: Number,
) -> usize {
    let n_x = x_curr.len();
    if dx.len() != n_x {
        return 0;
    }
    let mut clamped = 0;

    if let Some(em) = px_l.as_any().downcast_ref::<ExpansionMatrix>() {
        let bounds = compressed_values(x_l);
        let expanded = em.expanded_pos_indices();
        for (compressed_i, &full_pos) in expanded.iter().enumerate() {
            let i = full_pos as usize;
            if i >= n_x {
                continue;
            }
            let trial = x_curr[i] + dx[i];
            let lo = bounds[compressed_i];
            if trial < lo - eps {
                dx[i] = lo - x_curr[i];
                clamped += 1;
            }
        }
    }

    if let Some(em) = px_u.as_any().downcast_ref::<ExpansionMatrix>() {
        let bounds = compressed_values(x_u);
        let expanded = em.expanded_pos_indices();
        for (compressed_i, &full_pos) in expanded.iter().enumerate() {
            let i = full_pos as usize;
            if i >= n_x {
                continue;
            }
            let trial = x_curr[i] + dx[i];
            let hi = bounds[compressed_i];
            if trial > hi + eps {
                dx[i] = hi - x_curr[i];
                clamped += 1;
            }
        }
    }

    clamped
}

/// Extract dense values from a `dyn Vector` that wraps a `DenseVector`.
/// Returns an empty vector when the downcast fails (and the bound
/// vector is just treated as having no entries — the boundcheck then
/// silently no-ops, matching upstream's behavior when bounds aren't
/// represented as DenseVectors).
fn compressed_values(v: &dyn Vector) -> Vec<Number> {
    use pounce_linalg::dense_vector::DenseVector;
    match v.as_any().downcast_ref::<DenseVector>() {
        Some(dv) => dv.values().to_vec(),
        None => Vec::new(),
    }
}

/// Convenience: walk an `IpoptNlp` + iterates handle to call
/// [`clamp_step_to_bounds`]. Returns the count of clamped entries.
pub fn clamp_with_nlp(
    nlp: &dyn pounce_nlp::ipopt_nlp::IpoptNlp,
    x_curr: &[Number],
    dx: &mut [Number],
    eps: Number,
) -> usize {
    let px_l = nlp.px_l();
    let px_u = nlp.px_u();
    let x_l = nlp.x_l();
    let x_u = nlp.x_u();
    clamp_step_to_bounds(x_curr, dx, &px_l, &px_u, x_l, x_u, eps)
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
    use pounce_linalg::dense_vector::{DenseVector, DenseVectorSpace};
    use pounce_linalg::expansion_matrix::{ExpansionMatrix, ExpansionMatrixSpace};

    fn make_dv(values: &[Number]) -> DenseVector {
        let space = DenseVectorSpace::new(values.len() as Index);
        let mut dv = DenseVector::new(space);
        dv.values_mut().copy_from_slice(values);
        dv
    }

    #[test]
    fn clamp_lowers_violating_step() {
        // n_x = 3, lower bounds at slots {0, 2} with values {0.0, 0.5}.
        let n_x = 3;
        let x_curr = [0.1, 0.5, 0.6];
        let mut dx = [-0.3, 0.0, -0.5]; // trial = [-0.2, 0.5, 0.1]
        let px_l_space = ExpansionMatrixSpace::new(n_x as Index, 2, &[0, 2], 0);
        let px_l: Rc<dyn pounce_linalg::Matrix> = Rc::new(ExpansionMatrix::new(px_l_space));
        // No upper bounds.
        let px_u_space = ExpansionMatrixSpace::new(n_x as Index, 0, &[], 0);
        let px_u: Rc<dyn pounce_linalg::Matrix> = Rc::new(ExpansionMatrix::new(px_u_space));
        let x_l = make_dv(&[0.0, 0.5]);
        let x_u = make_dv(&[]);

        let n = clamp_step_to_bounds(&x_curr, &mut dx, &px_l, &px_u, &x_l, &x_u, 1e-9);
        assert_eq!(n, 2);
        // Slot 0: trial -0.2 < 0 → dx clamps to (0 - 0.1) = -0.1.
        assert!((dx[0] - (-0.1)).abs() < 1e-12, "dx[0] = {}", dx[0]);
        // Slot 1: not lower-bounded, untouched.
        assert!((dx[1] - 0.0).abs() < 1e-12);
        // Slot 2: trial 0.1 < 0.5 → dx clamps to (0.5 - 0.6) = -0.1.
        assert!((dx[2] - (-0.1)).abs() < 1e-12, "dx[2] = {}", dx[2]);
    }

    #[test]
    fn clamp_uppers_violating_step() {
        let n_x = 2;
        let x_curr = [0.9, 1.0];
        let mut dx = [0.5, 0.0]; // trial = [1.4, 1.0]
                                 // No lower bounds.
        let px_l_space = ExpansionMatrixSpace::new(n_x as Index, 0, &[], 0);
        let px_l: Rc<dyn pounce_linalg::Matrix> = Rc::new(ExpansionMatrix::new(px_l_space));
        let px_u_space = ExpansionMatrixSpace::new(n_x as Index, 1, &[0], 0);
        let px_u: Rc<dyn pounce_linalg::Matrix> = Rc::new(ExpansionMatrix::new(px_u_space));
        let x_l = make_dv(&[]);
        let x_u = make_dv(&[1.0]);

        let n = clamp_step_to_bounds(&x_curr, &mut dx, &px_l, &px_u, &x_l, &x_u, 1e-9);
        assert_eq!(n, 1);
        // Slot 0: trial 1.4 > 1.0 → dx clamps to (1.0 - 0.9) = 0.1.
        assert!((dx[0] - 0.1).abs() < 1e-12, "dx[0] = {}", dx[0]);
        assert!((dx[1] - 0.0).abs() < 1e-12);
    }

    #[test]
    fn clamp_is_noop_when_step_is_feasible() {
        let n_x = 2;
        let x_curr = [0.5, 0.5];
        let mut dx = [0.1, -0.1]; // both inside [0, 1]
        let px_l_space = ExpansionMatrixSpace::new(n_x as Index, 2, &[0, 1], 0);
        let px_l: Rc<dyn pounce_linalg::Matrix> = Rc::new(ExpansionMatrix::new(px_l_space));
        let px_u_space = ExpansionMatrixSpace::new(n_x as Index, 2, &[0, 1], 0);
        let px_u: Rc<dyn pounce_linalg::Matrix> = Rc::new(ExpansionMatrix::new(px_u_space));
        let x_l = make_dv(&[0.0, 0.0]);
        let x_u = make_dv(&[1.0, 1.0]);

        let n = clamp_step_to_bounds(&x_curr, &mut dx, &px_l, &px_u, &x_l, &x_u, 1e-9);
        assert_eq!(n, 0);
        assert!((dx[0] - 0.1).abs() < 1e-12);
        assert!((dx[1] - (-0.1)).abs() < 1e-12);
    }

    #[test]
    fn clamp_respects_epsilon_tolerance() {
        let n_x = 1;
        let x_curr = [0.0];
        // trial = -5e-4. With eps = 1e-3, this is within tolerance →
        // no clamp.
        let mut dx = [-5e-4];
        let px_l_space = ExpansionMatrixSpace::new(n_x as Index, 1, &[0], 0);
        let px_l: Rc<dyn pounce_linalg::Matrix> = Rc::new(ExpansionMatrix::new(px_l_space));
        let px_u_space = ExpansionMatrixSpace::new(n_x as Index, 0, &[], 0);
        let px_u: Rc<dyn pounce_linalg::Matrix> = Rc::new(ExpansionMatrix::new(px_u_space));
        let x_l = make_dv(&[0.0]);
        let x_u = make_dv(&[]);

        let n = clamp_step_to_bounds(&x_curr, &mut dx, &px_l, &px_u, &x_l, &x_u, 1e-3);
        assert_eq!(n, 0);
        assert!((dx[0] - (-5e-4)).abs() < 1e-12);
    }
}
