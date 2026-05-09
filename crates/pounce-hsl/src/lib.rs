//! HSL backend crate for POUNCE.
//!
//! Wraps `libcoinhsl.dylib` to provide an MA57 implementation of
//! [`pounce_linsol::SparseSymLinearSolverInterface`]. Port of
//! `Algorithm/LinearSolvers/IpMa57TSolverInterface.{hpp,cpp}` from
//! Ipopt 3.14.x.
//!
//! v1.0 ships only MA57 since this is the only HSL solver on the
//! bit-equivalence path that POUNCE targets first. MA27/77/86/97
//! sit behind the same trait and can be added without touching
//! callers.

#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]

pub mod ffi;
pub mod ma57;
pub mod mc19;
pub mod reg_op;

pub use ma57::{Ma57SolverInterface, Options as Ma57Options};
pub use mc19::Mc19TSymScalingMethod;
pub use reg_op::register_options_ma57;
