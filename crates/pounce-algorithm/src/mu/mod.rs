//! Barrier-parameter update strategies — port of
//! `Algorithm/IpMuUpdate.hpp`, `IpMonotoneMuUpdate.{hpp,cpp}`,
//! `IpAdaptiveMuUpdate.{hpp,cpp}`, and the four oracle files
//! (`IpMuOracle.hpp`, `IpLoqoMuOracle.cpp`, `IpProbingMuOracle.cpp`,
//! `IpQualityFunctionMuOracle.cpp`).
//!
//! Phase 7 ships [`monotone::MonotoneMuUpdate`] (Fiacco-McCormick).
//! Phase 10 adds the adaptive path and all four oracles.

pub mod adaptive;
pub mod monotone;
pub mod oracle;
pub mod r#trait;

pub use r#trait::MuUpdate;
