//! Gradient-based scaling — port of
//! `Algorithm/IpGradientScaling.{hpp,cpp}`. Default in Ipopt 3.14.x.
//!
//! Applies `1 / max(1, ||grad_f||_inf)` to the objective and
//! `1 / max(1, ||row_i J_g||_inf)` to each constraint row. Phase 7
//! ships the structural placeholder; the arithmetic ports alongside
//! `OrigIpoptNLP::initialize_strategies`.

use crate::scaling::r#trait::NlpScalingObject;
use pounce_common::types::Number;

pub struct GradientScaling {
    pub obj_scale: Number,
    pub max_gradient: Number,
}

impl Default for GradientScaling {
    fn default() -> Self {
        Self {
            obj_scale: 1.0,
            // Default from `IpGradientScaling.cpp:RegisterOptions`.
            max_gradient: 100.0,
        }
    }
}

impl GradientScaling {
    pub fn new() -> Self {
        Self::default()
    }
}

impl NlpScalingObject for GradientScaling {
    fn obj_scaling(&self) -> Number {
        self.obj_scale
    }

    fn have_scales(&self) -> bool {
        true
    }
}
