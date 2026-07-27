//! Scale-aware significance tests for feasibility and optimality decisions.
//!
//! # Why this exists
//!
//! Multiplying a constraint row by a positive constant leaves the feasible set
//! *exactly* unchanged — it is the same problem written differently. So a
//! solver's verdict must not depend on it. Comparing a scale-*dependent*
//! quantity (a constraint residual) against an *absolute* threshold breaks that
//! invariant, and has produced defects in the restoration gates, presolve
//! certification, and the solution verifier.
//!
//! The sharpest example: `x >= 2` over `x ∈ [0, 1]` reports
//! `Infeasible_Problem_Detected` as written, and `Solve_Succeeded` when every
//! row is multiplied by `1e-12` — because at that scale the residual falls
//! under an absolute tolerance. Same empty feasible set, opposite verdicts.
//!
//! # The rule
//!
//! Compare a residual against `tol * scale`, where `scale` is the quantity's
//! own natural magnitude. Both sides then move together under row scaling, so
//! the test is invariant.
//!
//! A clamped form — `tol * max(scale, 1)` — looks safer and is **wrong**: the
//! clamp reinstates the absolute floor for `scale < 1`, which is exactly the
//! down-scaled direction that fails. This was measured, not assumed:
//!
//! ```text
//!   k    residual   scale     tol*max(s,1)  fires?  |  tol*s     fires?
//! -12    1.00e-12   1.00e-12  1.00e-08      false   |  1.00e-20  true
//!  -8    1.00e-08   1.00e-08  1.00e-08      false   |  1.00e-16  true
//!   0    1.00e+00   1.00e+00  1.00e-08      true    |  1.00e-08  true
//!  12    1.00e+12   1.00e+12  1.00e+04      true    |  1.00e+04  true
//! ```
//!
//! # Direction of failure
//!
//! Both functions **fail closed**: a residual that cannot be judged (`NaN`, or
//! non-finite) is reported *not* significant. For an infeasibility test that is
//! the safe direction — it withholds a verdict rather than fabricating one. A
//! caller that needs the opposite polarity must handle non-finite values itself
//! rather than inverting the result.

use crate::types::Number;

/// Whether `value` is large enough, relative to its own natural magnitude, to
/// be treated as a real quantity rather than numerical noise.
///
/// The threshold is `tol * |scale|`. When `scale` is zero or non-finite the
/// relative test is undefined, so it degrades to the absolute `tol` — that case
/// is a degenerate row with no magnitude, where there is nothing to be relative
/// to.
///
/// Returns `false` for a non-finite `value` (see the module note on failing
/// closed).
///
/// ```
/// use pounce_common::tolerance::is_significant;
/// // Same model at three row scalings: the verdict must not move.
/// assert!(is_significant(1.0e-12, 1.0e-12, 1e-8));
/// assert!(is_significant(1.0, 1.0, 1e-8));
/// assert!(is_significant(1.0e12, 1.0e12, 1e-8));
/// // Noise at any scale is still noise.
/// assert!(!is_significant(1.0e-20, 1.0, 1e-8));
/// ```
pub fn is_significant(value: Number, scale: Number, tol: Number) -> bool {
    if !value.is_finite() {
        return false;
    }
    let s = scale.abs();
    let threshold = if s.is_finite() && s > 0.0 {
        tol * s
    } else {
        tol
    };
    value.abs() > threshold
}

/// The natural magnitude of a row, from the NLP scaling factor applied to it.
///
/// POUNCE's scaling picks `dc_i` so that `dc_i * c_i` is O(1); the row's own
/// magnitude is therefore `1 / dc_i`. Using the factor the solver already
/// computed avoids inventing a second, possibly disagreeing, notion of scale —
/// `c_scale_vec` / `d_scale_vec` are the authority.
///
/// A missing, zero, or non-finite factor means "no scaling applied", which is
/// magnitude `1.0`.
///
/// ```
/// use pounce_common::tolerance::row_scale_from_factor;
/// assert_eq!(row_scale_from_factor(1.0), 1.0);
/// assert_eq!(row_scale_from_factor(1e-6), 1e6);   // row shrunk by 1e-6 => magnitude 1e6
/// assert_eq!(row_scale_from_factor(0.0), 1.0);    // degenerate => unscaled
/// ```
pub fn row_scale_from_factor(factor: Number) -> Number {
    if factor.is_finite() && factor > 0.0 {
        1.0 / factor
    } else {
        1.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The property the whole module exists for: scaling a residual and its
    /// magnitude together must not change the answer.
    #[test]
    fn verdict_is_invariant_under_row_scaling() {
        let tol = 1e-8;
        for k in -12..=12 {
            let s = 10f64.powi(k);
            assert!(
                is_significant(1.0 * s, s, tol),
                "a unit violation at scale 10^{k} must stay significant"
            );
            assert!(
                !is_significant(1e-12 * s, s, tol),
                "noise at scale 10^{k} must stay insignificant"
            );
        }
    }

    /// Pins the bug in the clamped form, so nobody reintroduces it.
    #[test]
    fn clamped_form_would_lose_the_down_scaled_direction() {
        let tol = 1e-8;
        let (value, scale) = (1e-12, 1e-12); // a full unit violation at 1e-12 scale
        assert!(is_significant(value, scale, tol));
        // What `tol * max(scale, 1)` would have concluded:
        assert!(
            !(value.abs() > tol * scale.abs().max(1.0)),
            "the clamped form misses this — that is why it is not used"
        );
    }

    #[test]
    fn degenerate_scale_falls_back_to_absolute() {
        let tol = 1e-8;
        assert!(is_significant(1e-6, 0.0, tol));
        assert!(!is_significant(1e-10, 0.0, tol));
        assert!(is_significant(1e-6, f64::INFINITY, tol));
        assert!(is_significant(1e-6, f64::NAN, tol));
    }

    #[test]
    fn non_finite_value_is_not_evidence() {
        let tol = 1e-8;
        assert!(!is_significant(f64::NAN, 1.0, tol));
        assert!(!is_significant(f64::INFINITY, 1.0, tol));
        assert!(!is_significant(f64::NEG_INFINITY, 1.0, tol));
    }

    #[test]
    fn exactly_at_threshold_is_not_significant() {
        // Strict `>` keeps the boundary on the conservative side.
        assert!(!is_significant(1e-8, 1.0, 1e-8));
        assert!(is_significant(1.0000001e-8, 1.0, 1e-8));
    }

    #[test]
    fn row_scale_inverts_the_factor() {
        assert_eq!(row_scale_from_factor(1.0), 1.0);
        assert_eq!(row_scale_from_factor(1e-6), 1e6);
        assert_eq!(row_scale_from_factor(1e6), 1e-6);
        // Degenerate factors mean "unscaled".
        assert_eq!(row_scale_from_factor(0.0), 1.0);
        assert_eq!(row_scale_from_factor(-1.0), 1.0);
        assert_eq!(row_scale_from_factor(f64::NAN), 1.0);
        assert_eq!(row_scale_from_factor(f64::INFINITY), 1.0);
    }

    /// End-to-end: a factor from `c_scale_vec` feeding the significance test.
    #[test]
    fn factor_and_significance_compose() {
        let tol = 1e-8;
        // Solver shrank this row by 1e-6, so its natural magnitude is 1e6.
        let scale = row_scale_from_factor(1e-6);
        assert_eq!(scale, 1e6);
        // Threshold is tol * scale = 1e-8 * 1e6 = 1e-2.
        assert!(
            !is_significant(1e-3, scale, tol),
            "1e-3 is below the 1e-2 threshold"
        );
        assert!(is_significant(1e-1, scale, tol), "1e-1 is above it");
    }
}
