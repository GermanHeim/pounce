//! NLP scaling object trait — port of `IpNLPScalingObject.hpp`.
//!
//! Concrete impls scale the NLP's objective, gradient, constraints
//! and Jacobian by per-row / per-column factors so that the KKT
//! linear system is well-conditioned. The default in upstream Ipopt
//! 3.14 is `nlp_scaling_method = gradient-based`.

use pounce_common::types::Number;

pub trait NlpScalingObject {
    /// Multiplicative scale factor applied to the objective `f(x)`.
    fn obj_scaling(&self) -> Number {
        1.0
    }

    /// Whether the scaling object actually scales anything (returns
    /// `false` for [`super::none::NoNlpScalingObject`] so callers can
    /// shortcut). Mirrors upstream `HaveScales()`.
    fn have_scales(&self) -> bool {
        false
    }
}
