//! Warm-start iterate initializer — port of
//! `IpWarmStartIterateInitializer.{hpp,cpp}`. Used when a previous
//! solve has left a trial point that should be reused.
//!
//! Phase-7 stub: trusts that the caller has already populated
//! `data.curr` with the warm-start iterate (which is what upstream's
//! `IpoptApplication::ReOptimizeTNLP` does before invoking the
//! algorithm). The full version invalidates `iter_count` and copies
//! the previous solution's bound multipliers.

use crate::init::r#trait::IterateInitializer;
use crate::ipopt_cq::IpoptCqHandle;
use crate::ipopt_data::IpoptDataHandle;
use crate::ipopt_nlp::IpoptNlp;
use crate::kkt::aug_system_solver::AugSystemSolver;
use std::cell::RefCell;
use std::rc::Rc;

pub struct WarmStartIterateInitializer;

impl WarmStartIterateInitializer {
    pub fn new() -> Self {
        Self
    }
}

impl Default for WarmStartIterateInitializer {
    fn default() -> Self {
        Self::new()
    }
}

impl IterateInitializer for WarmStartIterateInitializer {
    fn set_initial_iterates(
        &mut self,
        data: &IpoptDataHandle,
        _cq: &IpoptCqHandle,
        _nlp: &Rc<RefCell<dyn IpoptNlp>>,
        _aug_solver: &mut dyn AugSystemSolver,
    ) -> bool {
        // Warm-start path leaves whatever was placed on `data.curr`
        // by the application driver alone. We only signal success.
        data.borrow().curr.is_some()
    }
}
