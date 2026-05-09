//! User-supplied scaling — port of `Algorithm/IpUserScaling.{hpp,cpp}`.
//! Reads `obj_scaling`, `x_scaling`, `g_scaling` from the TNLP via
//! `get_scaling_parameters`.

use crate::scaling::r#trait::NlpScalingObject;
use pounce_common::types::Number;

#[derive(Debug, Default, Clone)]
pub struct UserScaling {
    pub obj_scale: Number,
}

impl UserScaling {
    pub fn new() -> Self {
        Self { obj_scale: 1.0 }
    }
}

impl NlpScalingObject for UserScaling {
    fn obj_scaling(&self) -> Number {
        self.obj_scale
    }

    fn have_scales(&self) -> bool {
        true
    }
}
