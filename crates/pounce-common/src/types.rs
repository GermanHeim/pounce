//! Fundamental scalar types.
//!
//! Mirrors `Common/IpTypes.h` and `Common/IpTypes.hpp`. We commit to
//! `f64` and `i32` for v1.0 because Ipopt's MUMPS/HSL ABI is built
//! around those widths; widening here would force us off the
//! bit-equivalence path.

/// Floating-point scalar — `Number` in Ipopt.
pub type Number = f64;

/// Signed index — `Index` in Ipopt. Held at 32 bits for ABI parity
/// with MUMPS, MA27, etc.
pub type Index = i32;

/// Sentinel used by Ipopt for "no bound" in TNLP get_bounds_info.
/// Value `1e19` is hard-coded throughout upstream; we match it.
pub const NLP_LOWER_BOUND_INF: Number = -1e19;
pub const NLP_UPPER_BOUND_INF: Number = 1e19;

/// Is this *lower* bound a real bound, or the absent-bound sentinel?
///
/// The sentinel convention is **directional**, and reading it as a magnitude
/// is a bug this codebase has now hit four separate times (#396, #398, #401,
/// #402). A lower bound is absent only at or below [`NLP_LOWER_BOUND_INF`];
/// `-5e20` is an ordinary finite lower bound, not "beyond infinity", and a
/// symmetric `|b| < 1e19` test silently discards it.
///
/// Pair with [`upper_bound_present`] and decide presence *before* comparing a
/// pair: `lo > hi` means nothing until both sides are known to be real, and
/// neither does `lo == hi` (an "equality" at the sentinel is a one-sided row).
///
/// Callers that override the thresholds via `nlp_lower_bound_inf` /
/// `nlp_upper_bound_inf` must test against their own values instead — these
/// helpers hard-code the defaults.
#[inline]
pub fn lower_bound_present(lo: Number) -> bool {
    lo.is_finite() && lo > NLP_LOWER_BOUND_INF
}

/// Is this *upper* bound a real bound, or the absent-bound sentinel?
/// See [`lower_bound_present`] — an upper bound is absent only at or above
/// [`NLP_UPPER_BOUND_INF`].
#[inline]
pub fn upper_bound_present(hi: Number) -> bool {
    hi.is_finite() && hi < NLP_UPPER_BOUND_INF
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn presence_is_directional_not_symmetric() {
        // Sentinels: absent on their own side.
        assert!(!lower_bound_present(NLP_LOWER_BOUND_INF));
        assert!(!upper_bound_present(NLP_UPPER_BOUND_INF));
        assert!(!lower_bound_present(-2.0e19));
        assert!(!upper_bound_present(2.0e19));

        // Past the *opposite* sentinel: an ordinary bound, not an absent one.
        // A symmetric `|b| < 1e19` test gets both of these wrong.
        assert!(upper_bound_present(-5.0e20));
        assert!(lower_bound_present(5.0e20));

        // True infinities are absent on either side.
        assert!(!lower_bound_present(Number::NEG_INFINITY));
        assert!(!upper_bound_present(Number::INFINITY));
        assert!(!lower_bound_present(Number::NAN));
        assert!(!upper_bound_present(Number::NAN));

        // Ordinary bounds.
        assert!(lower_bound_present(0.0));
        assert!(upper_bound_present(0.0));
    }
}
