//! Iteration output strategies — port of
//! `Algorithm/IpIterationOutput.hpp`,
//! `IpOrigIterationOutput.{hpp,cpp}`. Restoration-phase iteration
//! output (`IpRestoIterationOutput`) lives in `pounce-restoration`.

pub mod orig;
pub mod r#trait;

pub use orig::OrigIterationOutput;
pub use r#trait::IterationOutput;
