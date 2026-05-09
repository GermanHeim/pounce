//! No-op scaling — port of `Algorithm/IpNoNLPScalingObject.{hpp,cpp}`.

use crate::scaling::r#trait::NlpScalingObject;

#[derive(Debug, Default, Clone, Copy)]
pub struct NoNlpScalingObject;

impl NoNlpScalingObject {
    pub fn new() -> Self {
        Self
    }
}

impl NlpScalingObject for NoNlpScalingObject {}
