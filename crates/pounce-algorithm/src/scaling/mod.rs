//! NLP scaling strategies — port of `Algorithm/IpNLPScaling.{hpp,cpp}`,
//! `IpStandardScalingBase.{hpp,cpp}`,
//! `IpUserScaling.{hpp,cpp}`,
//! `IpGradientScaling.{hpp,cpp}`,
//! `IpEquilibrationScaling.{hpp,cpp}`,
//! `IpNoNLPScalingObject.{hpp,cpp}`.
//!
//! See `ref/Ipopt/AGENT_REFERENCE/SCALING.md`.

pub mod equilibration;
pub mod gradient;
pub mod none;
pub mod r#trait;
pub mod user;

pub use equilibration::EquilibrationScaling;
pub use gradient::GradientScaling;
pub use none::NoNlpScalingObject;
pub use r#trait::NlpScalingObject;
pub use user::UserScaling;
