//! TNLP wrapper that hides the inner problem's second derivatives, so a
//! model that *has* a Hessian can be benchmarked as one that does not.
//!
//! # Why this exists
//!
//! `hessian_approximation=finite-difference` and `limited-memory` both
//! exist for models that supply first derivatives and no second ones — a
//! collocation transcription built from an FMU, say. Every such model in
//! this repository's corpus is an AMPL `.nl`, which *does* carry an exact
//! Hessian through AMPL's AD. Measuring those paths by setting the option
//! on a model that has a Hessian therefore simulates the constraint
//! rather than reproducing it, and the simulation is weaker than it looks:
//! it changes which updater runs, but `nnz_h_lag` is still positive,
//! `h_space` is still built, and `uninitialized_h()` still returns the
//! real pattern.
//!
//! A model that genuinely has no Hessian differs in all three, and the
//! difference reaches real code: the finite-difference updater's
//! `declared` pattern source finds nothing and must fall back to the
//! Jacobian derivation, and the augmented-system solver sees a `W` whose
//! nonzero count changes from 0 to the assembled pattern rather than
//! staying put.
//!
//! This wrapper reproduces the real thing. It reports `nnz_h_lag = 0` and
//! declines `eval_h` — exactly what `pounce-py` does for a Python problem
//! object with no `hessian` method (`problem.rs`, the `has_hessian`
//! branch), which is the actual shape of the models this targets.
//!
//! Enabled with `POUNCE_DROP_HESSIAN=1`. It is a **benchmarking
//! facility**, deliberately an environment variable rather than a
//! registered option: nothing about a real solve should ever want to
//! discard information the model was willing to provide.

use pounce_common::types::{Index, Number};
use pounce_nlp::tnlp::{
    BoundsInfo, InfeasibilityProof, IpoptCq, IpoptData, IterStats, Linearity, MetaData, NlpInfo,
    ScalingRequest, Solution, SparsityRequest, StartingPoint, TNLP,
};
use std::cell::RefCell;
use std::rc::Rc;

pub struct NoHessianTnlp {
    inner: Rc<RefCell<dyn TNLP>>,
}

impl NoHessianTnlp {
    pub fn new(inner: Rc<RefCell<dyn TNLP>>) -> Self {
        Self { inner }
    }

    /// Whether `POUNCE_DROP_HESSIAN` asks for the wrapper.
    pub fn requested() -> bool {
        std::env::var("POUNCE_DROP_HESSIAN")
            .map(|v| !matches!(v.trim(), "" | "0" | "no" | "false"))
            .unwrap_or(false)
    }
}

impl TNLP for NoHessianTnlp {
    /// The one field that is not forwarded: a model with no second
    /// derivatives declares no Hessian nonzeros.
    fn get_nlp_info(&mut self) -> Option<NlpInfo> {
        self.inner.borrow_mut().get_nlp_info().map(|mut i| {
            i.nnz_h_lag = 0;
            i
        })
    }

    /// Decline, for the structure call and the values call alike. The
    /// trait's own default does this; it is spelled out because declining
    /// is the entire purpose of the wrapper.
    fn eval_h(
        &mut self,
        _x: Option<&[Number]>,
        _new_x: bool,
        _obj_factor: Number,
        _lambda: Option<&[Number]>,
        _new_lambda: bool,
        _mode: SparsityRequest<'_>,
    ) -> bool {
        false
    }

    // ---- everything else forwards unchanged ----------------------------

    fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
        self.inner.borrow_mut().get_bounds_info(b)
    }
    fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
        self.inner.borrow_mut().get_starting_point(sp)
    }
    fn eval_f(&mut self, x: &[Number], new_x: bool) -> Option<Number> {
        self.inner.borrow_mut().eval_f(x, new_x)
    }
    fn eval_grad_f(&mut self, x: &[Number], new_x: bool, grad_f: &mut [Number]) -> bool {
        self.inner.borrow_mut().eval_grad_f(x, new_x, grad_f)
    }
    fn eval_g(&mut self, x: &[Number], new_x: bool, g: &mut [Number]) -> bool {
        self.inner.borrow_mut().eval_g(x, new_x, g)
    }
    fn eval_jac_g(
        &mut self,
        x: Option<&[Number]>,
        new_x: bool,
        mode: SparsityRequest<'_>,
    ) -> bool {
        self.inner.borrow_mut().eval_jac_g(x, new_x, mode)
    }
    fn finalize_solution(&mut self, sol: Solution<'_>, d: &IpoptData, cq: &IpoptCq) {
        self.inner.borrow_mut().finalize_solution(sol, d, cq)
    }
    fn get_var_con_metadata(&mut self, var: &mut MetaData, con: &mut MetaData) -> bool {
        self.inner.borrow_mut().get_var_con_metadata(var, con)
    }
    fn get_scaling_parameters(&mut self, req: ScalingRequest<'_>) -> bool {
        self.inner.borrow_mut().get_scaling_parameters(req)
    }
    fn get_variables_linearity(&mut self, types: &mut [Linearity]) -> bool {
        self.inner.borrow_mut().get_variables_linearity(types)
    }
    fn get_objective_variables_linearity(&mut self, types: &mut [Linearity]) -> bool {
        self.inner
            .borrow_mut()
            .get_objective_variables_linearity(types)
    }
    fn get_constraints_linearity(&mut self, types: &mut [Linearity]) -> bool {
        self.inner.borrow_mut().get_constraints_linearity(types)
    }
    fn get_number_of_nonlinear_variables(&mut self) -> Index {
        self.inner.borrow_mut().get_number_of_nonlinear_variables()
    }
    fn derivative_proofs(&mut self) -> pounce_nlp::constant_derivatives::DerivativeProofs {
        self.inner.borrow_mut().derivative_proofs()
    }
    fn get_list_of_nonlinear_variables(&mut self, pos: &mut [Index]) -> bool {
        self.inner.borrow_mut().get_list_of_nonlinear_variables(pos)
    }
    fn intermediate_callback(&mut self, s: IterStats, d: &IpoptData, cq: &IpoptCq) -> bool {
        self.inner.borrow_mut().intermediate_callback(s, d, cq)
    }
    fn finalize_metadata(&mut self, var: &MetaData, con: &MetaData) {
        self.inner.borrow_mut().finalize_metadata(var, con)
    }
    /// Transparent decorator: forward the proof, or the application never
    /// sees it and the solve runs on an infeasible model anyway.
    fn presolve_infeasibility_proof(&self) -> Option<InfeasibilityProof> {
        self.inner.borrow().presolve_infeasibility_proof()
    }
}
