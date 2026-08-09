//! Active-set SQP warm starting — the working-set contract, re-exported
//! (feature `qp`).
//!
//! Switching the driver to the SQP path needs nothing from this module and
//! no feature at all: it is one option string on the default build.
//!
//! ```
//! use pounce_rs::prelude::*;
//!
//! struct Quad; // min (x0-1)^2 + (x1-2)^2  s.t. x0 + x1 == 3
//! impl Problem for Quad {
//!     fn objective(&self, x: &[f64]) -> f64 {
//!         (x[0] - 1.0).powi(2) + (x[1] - 2.0).powi(2)
//!     }
//!     fn n_constraints(&self) -> usize { 1 }
//!     fn constraints(&self, x: &[f64], g: &mut [f64]) { g[0] = x[0] + x[1]; }
//! }
//!
//! let sol = Nlp::new(Quad)
//!     .var_bounds(&[0.0, 0.0], &[5.0, 5.0])
//!     .constraint_bounds(&[3.0], &[3.0])
//!     .option_str("algorithm", "active-set-sqp")
//!     .solve();
//! assert!(sol.success);
//! ```
//!
//! What *does* need this module is the payoff: carrying the discrete
//! **working set** — which bounds and constraints are active — from one
//! solve into the next. The floating-point part of a warm start (`x`, `λ_g`,
//! `λ_x`) is available from any solve, but the active set is what lets the
//! next QP build its KKT block correctly from iteration zero instead of
//! rediscovering it. That is the whole reason to prefer SQP over the IPM on
//! a sequence of nearby problems (MPC, continuation, homotopy).
//!
//! The round trip is [`IpoptApplication::last_sqp_working_set`] out,
//! [`SqpIterates`] in, [`IpoptApplication::set_sqp_warm_start`] to install:
//!
//! ```no_run
//! use pounce_rs::prelude::*;
//! use pounce_rs::sqp::SqpIterates;
//! use std::cell::RefCell;
//! use std::rc::Rc;
//!
//! # fn demo(tnlp: Rc<RefCell<dyn TNLP>>, x_star: Vec<f64>) {
//! let mut app = IpoptApplication::new();
//! app.initialize().unwrap();
//! app.initialize_with_options_str("algorithm active-set-sqp\n").unwrap();
//!
//! // Cold solve: converges and leaves a working set behind.
//! let status = app.optimize_tnlp(Rc::clone(&tnlp));
//! assert_eq!(status, ApplicationReturnStatus::SolveSucceeded);
//! let working = app.last_sqp_working_set().cloned();
//!
//! // ... the caller updates the parameter inside `tnlp` here ...
//!
//! // Warm solve: same shape, seeded from the previous active set.
//! app.set_sqp_warm_start(SqpIterates {
//!     x: x_star,
//!     lambda_g: vec![0.0; 1],
//!     lambda_x: vec![0.0; 2],
//!     working,
//! });
//! let status = app.optimize_tnlp(tnlp);
//! # }
//! ```
//!
//! The warm start is consumed by the solve that follows it, so each
//! iteration of a sequence installs a fresh one;
//! [`IpoptApplication::clear_sqp_warm_start`] drops an unused one.
//!
//! When the seed comes from a *sensitivity predictor* rather than a previous
//! solve — the `SensSolve` → SQP-corrector playbook in [`crate::sensitivity`]
//! — there is no previous working set to carry, so
//! [`classify_working_set`] derives one from a predicted point and its
//! multipliers. Note its `lambda_x` argument is the **packed** bound
//! multiplier `z_l − z_u`, not the pair.
//!
//! `λ_g` and `λ_x` are unconstrained in sign only where the corresponding
//! row is inactive; a warm start whose multipliers disagree with its working
//! set is repaired rather than trusted, so an approximate seed is safe.

pub use pounce_algorithm::sqp::{
    SqpGlobalization, SqpHessianSource, SqpIterates, SqpOptions, warm_start::classify_working_set,
};

/// The working set itself, and its per-row status codes. Re-exported from
/// [`crate::qp`], where the QP engine that consumes it lives.
pub use pounce_qp::{BoundStatus, ConsStatus, WorkingSet};
