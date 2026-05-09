//! Convergence-check strategies — port of `Algorithm/IpConvCheck.hpp`,
//! `IpOptErrorConvCheck.{hpp,cpp}`. Restoration-phase variants live
//! in `pounce-restoration`.

pub mod opt_error;
pub mod r#trait;

pub use r#trait::{ConvCheck, ConvergenceStatus};
