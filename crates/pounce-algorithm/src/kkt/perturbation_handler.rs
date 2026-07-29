//! Primal-dual perturbation handler — the shared implementation now lives in
//! [`pounce_common::pd_perturbation`].
//!
//! It moved there so the active-set QP (`pounce-qp`) can use the same tested
//! state machine. The dependency runs `pounce-algorithm -> pounce-qp`, so a
//! shared lower crate was the only way to reach it without duplicating 674
//! lines of IPOPT-ported logic. This module keeps the names other
//! `pounce-algorithm` code imports, and supplies the `PerturbationSink`
//! implementation over `IpoptDataHandle`.

use crate::ipopt_data::IpoptDataHandle;
use pounce_common::types::{Index, Number};

pub use pounce_common::pd_perturbation::{
    DegenType, Deltas, PdPerturbationHandler, PerturbationSink, TrialStatus,
};

/// Adapts `IpoptDataHandle` to the sink the shared handler writes through.
pub struct IpoptDataSink<'a>(pub &'a IpoptDataHandle);

impl PerturbationSink for IpoptDataSink<'_> {
    fn append_info(&self, s: &str) {
        self.0.borrow_mut().append_info_string(s);
    }
    fn set_regu_x(&self, v: Number) {
        self.0.borrow_mut().info_regu_x = v;
    }
    fn iter_count(&self) -> Index {
        self.0.borrow().iter_count
    }
    fn debug(&self, msg: &str) {
        tracing::debug!(target: "pounce::linsol", "{msg}");
    }
}
