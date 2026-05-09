//! Iterate-initialization strategies — port of
//! `Algorithm/IpIterateInitializer.hpp`,
//! `IpDefaultIterateInitializer.{hpp,cpp}`,
//! `IpWarmStartIterateInitializer.{hpp,cpp}`.

pub mod default;
pub mod r#trait;
pub mod warm_start;

pub use r#trait::IterateInitializer;
