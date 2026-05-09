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
