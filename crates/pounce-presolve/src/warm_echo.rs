//! The warm start a presolve layer served, kept so that its postsolve can be
//! the exact inverse of its inbound map (M24, warm round trip).
//!
//! A layer's postsolve re-derives the multipliers its reduced problem does
//! not carry — a consumed row's `λ`, a derived bound's `z` moved back to the
//! rows and declared bounds that imply it — from stationarity at the `x` the
//! solver returns. On a converged solve that is the only information there
//! is and it is exact. On a warm start the solver hands back without taking
//! a step (`max_iter=0`, or a start already inside the tolerance) it is not:
//! the interior-point start moved `x` off its bounds by
//! `warm_start_bound_push` (`1e-3` by default), so `∇f` there is not the `∇f`
//! the supplied multipliers balance, and every re-derived multiplier is off
//! by `O(push)`. `presolve=no` hands the supplied duals back unchanged.
//!
//! So a layer records what it was handed and what it served, and when the
//! solver returns the served multipliers unchanged it returns what it was
//! handed. That is the inbound map's inverse on its image, needs no reading
//! of `x`, and is reachable only from a warm start that supplied duals: a
//! cold solve never records one, so its postsolve is untouched.

use pounce_common::types::Number;

/// Relative agreement within which a returned multiplier is the served one.
/// The solver's own round trip through `nlp_scaling` multiplies and divides
/// by the same factor (a few ulps); any accepted dual step moves it by far
/// more.
const ECHO_RTOL: Number = 1e-12;

/// A bound multiplier the solver keeps as given: strictly positive and
/// finite. Anything else (`0`, negative, NaN = "unseeded") the solver
/// replaces with `warm_start_mult_bound_push` or `bound_mult_init_val`, so it
/// carries no information about whether a step was taken.
pub(crate) fn is_seed(z: Number) -> bool {
    z.is_finite() && z > 0.0
}

/// A supplied bound multiplier as a reported value: the seed, or 0.
pub(crate) fn as_reported_z(z: Number) -> Number {
    if is_seed(z) { z } else { 0.0 }
}

/// A supplied row multiplier as a reported value: finite, or 0.
pub(crate) fn as_reported_lambda(l: Number) -> Number {
    if l.is_finite() { l } else { 0.0 }
}

/// The served bound multipliers as recorded: NaN ("not a seed") where the
/// solver does not use them (`init_z` off) and on a variable the served box
/// fixes (`x_l == x_u`), which the solver removes and whose multiplier it
/// re-derives at the end rather than iterating on.
pub(crate) fn served_bound_multipliers(
    init_z: bool,
    z: &[Number],
    x_l: &[Number],
    x_u: &[Number],
) -> Vec<Number> {
    z.iter()
        .enumerate()
        .map(|(j, &v)| {
            let fixed = matches!((x_l.get(j), x_u.get(j)), (Some(l), Some(u)) if l == u);
            if init_z && !fixed { v } else { Number::NAN }
        })
        .collect()
}

/// The served row multipliers as recorded: NaN where `init_lambda` is off.
pub(crate) fn served_row_multipliers(init_lambda: bool, lambda: &[Number]) -> Vec<Number> {
    if init_lambda {
        lambda.to_vec()
    } else {
        vec![Number::NAN; lambda.len()]
    }
}

/// One served warm start: the duals the layer was handed (`*_in`, in the
/// space below it) and the ones it served the solver (`*_out`).
#[derive(Debug, Clone, Default)]
pub(crate) struct ServedWarmStart {
    pub z_l_in: Vec<Number>,
    pub z_u_in: Vec<Number>,
    pub lambda_in: Vec<Number>,
    pub z_l_out: Vec<Number>,
    pub z_u_out: Vec<Number>,
    pub lambda_out: Vec<Number>,
}

impl ServedWarmStart {
    /// `true` when `(z_l, z_u, lambda)` — what the solver returned — is the
    /// served warm start: every served bound-multiplier seed and every
    /// served finite row multiplier comes back unchanged, and at least one
    /// of them is non-zero (an all-unseeded start says nothing about
    /// whether the solver moved).
    pub(crate) fn echoed_by(&self, z_l: &[Number], z_u: &[Number], lambda: &[Number]) -> bool {
        if z_l.len() != self.z_l_out.len()
            || z_u.len() != self.z_u_out.len()
            || lambda.len() != self.lambda_out.len()
        {
            return false;
        }
        let same = |got: Number, served: Number| {
            (got - served).abs() <= ECHO_RTOL * served.abs() || got.to_bits() == served.to_bits()
        };
        let mut informative = 0usize;
        for (got, &served) in z_l
            .iter()
            .chain(z_u)
            .zip(self.z_l_out.iter().chain(&self.z_u_out))
        {
            if is_seed(served) {
                if !same(*got, served) {
                    return false;
                }
                informative += 1;
            }
        }
        for (&got, &served) in lambda.iter().zip(&self.lambda_out) {
            if served.is_finite() {
                if !same(got, served) {
                    return false;
                }
                if served != 0.0 {
                    informative += 1;
                }
            }
        }
        informative > 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn served() -> ServedWarmStart {
        ServedWarmStart {
            z_l_out: vec![20.0, 0.0, Number::NAN],
            z_u_out: vec![0.0, 0.0, 0.0],
            lambda_out: vec![-2.0, 0.0],
            ..Default::default()
        }
    }

    #[test]
    fn an_unchanged_return_is_an_echo_whatever_the_solver_filled() {
        // The solver floors the 0 and fills the NaN; neither is a seed.
        let s = served();
        assert!(s.echoed_by(&[20.0, 1e-3, 1.0], &[1e-3; 3], &[-2.0, 0.0]));
        // A scaling round trip's few ulps are still the same value.
        assert!(s.echoed_by(&[20.0 * (1.0 + 4e-16), 0.0, 1.0], &[0.0; 3], &[-2.0, 0.0]));
    }

    #[test]
    fn any_moved_seed_or_row_multiplier_is_not_an_echo() {
        let s = served();
        assert!(!s.echoed_by(&[20.0 + 1e-9, 0.0, 1.0], &[0.0; 3], &[-2.0, 0.0]));
        assert!(!s.echoed_by(&[20.0, 0.0, 1.0], &[0.0; 3], &[-2.0, 1e-300]));
        assert!(!s.echoed_by(&[20.0, 0.0], &[0.0; 3], &[-2.0, 0.0]));
    }

    #[test]
    fn a_start_with_nothing_seeded_is_never_an_echo() {
        let s = ServedWarmStart {
            z_l_out: vec![0.0, Number::NAN],
            z_u_out: vec![0.0, 0.0],
            lambda_out: vec![0.0],
            ..Default::default()
        };
        assert!(!s.echoed_by(&[1e-3, 1.0], &[1e-3, 1e-3], &[0.0]));
    }
}
