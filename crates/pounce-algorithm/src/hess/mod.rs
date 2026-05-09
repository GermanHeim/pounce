//! Hessian-update strategies — port of `IpHessianUpdater.hpp`,
//! `IpExactHessianUpdater.{hpp,cpp}`,
//! `IpLimMemQuasiNewtonUpdater.{hpp,cpp}` (Phase 8).

pub mod exact;
pub mod lim_mem_quasi_newton;
pub mod r#trait;

pub use r#trait::HessianUpdater;
