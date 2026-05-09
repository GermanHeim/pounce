//! Equality-multiplier estimation — port of
//! `IpEqMultCalculator.hpp`, `IpLeastSquareMults.{hpp,cpp}`.

pub mod least_square;
pub mod r#trait;

pub use r#trait::EqMultCalculator;
