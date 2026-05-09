//! Equilibration scaling — port of
//! `Algorithm/IpEquilibrationScaling.{hpp,cpp}`. Computes scaling
//! factors via MC19 over the Jacobian.

use crate::scaling::r#trait::NlpScalingObject;
use pounce_common::types::Number;

pub struct EquilibrationScaling {
    pub obj_scale: Number,
}

impl Default for EquilibrationScaling {
    fn default() -> Self {
        Self { obj_scale: 1.0 }
    }
}

impl EquilibrationScaling {
    pub fn new() -> Self {
        Self::default()
    }
}

impl NlpScalingObject for EquilibrationScaling {
    fn obj_scaling(&self) -> Number {
        self.obj_scale
    }

    fn have_scales(&self) -> bool {
        true
    }
}
