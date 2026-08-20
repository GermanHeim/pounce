//! `EqMultCalculator` trait — port of `IpEqMultCalculator.hpp`.

use crate::ipopt_cq::IpoptCqHandle;
use crate::ipopt_data::IpoptDataHandle;
use crate::ipopt_nlp::IpoptNlp;
use crate::kkt::aug_system_solver::AugSystemSolver;
use pounce_linalg::Vector;
use std::cell::RefCell;
use std::rc::Rc;

pub trait EqMultCalculator {
    /// Compute initial equality multipliers `y_c`, `y_d`. Mirrors
    /// `Ipopt::EqMultiplierCalculator::CalculateMultipliers`. Returns
    /// `false` if the underlying linear solve fails.
    fn calculate_y_eq(
        &mut self,
        data: &IpoptDataHandle,
        cq: &IpoptCqHandle,
        nlp: &Rc<RefCell<dyn IpoptNlp>>,
        aug_solver: &mut dyn AugSystemSolver,
        y_c: &mut dyn Vector,
        y_d: &mut dyn Vector,
    ) -> bool;
}
