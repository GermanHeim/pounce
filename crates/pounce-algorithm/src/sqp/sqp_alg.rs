//! `SqpAlgorithm` — active-set SQP outer loop. Sits parallel to
//! [`crate::ipopt_alg::IpoptAlgorithm`]; uses the same
//! [`crate::ipopt_nlp::IpoptNlp`] handle and delegates the QP
//! subproblem solve to `pounce_qp`.
//!
//! Phase 5b commit 1 ships only the struct skeleton. The main
//! loop, KKT-error check, l1-merit / filter globalization, and
//! QP assembly land in subsequent commits.

use crate::ipopt_nlp::IpoptNlp;
use crate::sqp::iterates::SqpIterates;
use crate::sqp::options::SqpOptions;
use pounce_qp::ParametricActiveSetSolver;
use std::cell::RefCell;
use std::rc::Rc;

/// SQP-side algorithm driver. Holds the NLP handle, the QP
/// subproblem solver, and the per-call iterate state.
pub struct SqpAlgorithm {
    nlp: Rc<RefCell<dyn IpoptNlp>>,
    qp_solver: ParametricActiveSetSolver,
    opts: SqpOptions,
    iterates: Option<SqpIterates>,
}

impl SqpAlgorithm {
    pub fn new(
        nlp: Rc<RefCell<dyn IpoptNlp>>,
        qp_solver: ParametricActiveSetSolver,
        opts: SqpOptions,
    ) -> Self {
        Self {
            nlp,
            qp_solver,
            opts,
            iterates: None,
        }
    }

    pub fn options(&self) -> &SqpOptions {
        &self.opts
    }

    pub fn nlp(&self) -> &Rc<RefCell<dyn IpoptNlp>> {
        &self.nlp
    }

    pub fn qp_solver_mut(&mut self) -> &mut ParametricActiveSetSolver {
        &mut self.qp_solver
    }

    pub fn iterates(&self) -> Option<&SqpIterates> {
        self.iterates.as_ref()
    }
}
