//! Exact Hessian — port of `IpExactHessianUpdater.{hpp,cpp}`.
//! Pulls `eval_h` from the NLP at every iterate via
//! `IpoptCalculatedQuantities::curr_exact_hessian` and stashes the
//! result into `IpoptData::w`.

use crate::hess::r#trait::HessianUpdater;
use crate::ipopt_cq::IpoptCqHandle;
use crate::ipopt_data::IpoptDataHandle;

pub struct ExactHessianUpdater;

impl ExactHessianUpdater {
    pub fn new() -> Self {
        Self
    }
}

impl Default for ExactHessianUpdater {
    fn default() -> Self {
        Self::new()
    }
}

impl HessianUpdater for ExactHessianUpdater {
    fn update_hessian(&mut self, data: &IpoptDataHandle, cq: &IpoptCqHandle) -> bool {
        let w = cq.borrow().curr_exact_hessian();
        data.borrow_mut().w = Some(w);
        true
    }

    fn provides_exact_hessian(&self) -> bool {
        true
    }

    /// A pure evaluation at the current iterate; `data.w` is untouched.
    fn hessian_at_current(
        &mut self,
        _data: &IpoptDataHandle,
        cq: &IpoptCqHandle,
    ) -> Option<std::rc::Rc<dyn pounce_linalg::SymMatrix>> {
        Some(cq.borrow().curr_exact_hessian())
    }
}
