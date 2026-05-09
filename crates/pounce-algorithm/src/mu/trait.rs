//! `MuUpdate` trait — port of `IpMuUpdate.hpp`.

use crate::ipopt_cq::IpoptCqHandle;
use crate::ipopt_data::IpoptDataHandle;
use crate::ipopt_nlp::IpoptNlp;
use crate::kkt::pd_search_dir_calc::PdSearchDirCalc;
use pounce_common::types::Number;
use std::cell::RefCell;
use std::rc::Rc;

pub trait MuUpdate {
    /// Initialize `data.curr_mu` and `data.curr_tau` before the first
    /// iteration. Mirrors upstream's `MuUpdate::InitializeImpl`.
    /// Default is no-op so existing implementors don't have to change.
    fn initialize(&mut self, _data: &IpoptDataHandle) {}

    /// Compute the next mu after a successful iteration. Mirrors
    /// upstream's `MuUpdate::UpdateBarrierParameter`. Implementations
    /// that need the iterate state (adaptive mu, oracles) read it via
    /// the supplied handles; pure scalar reductions like
    /// Fiacco-McCormick consult only `data.curr_mu`.
    ///
    /// `nlp` and `pd_search_dir` are optional handles needed by the
    /// adaptive μ oracles that drive an affine-step / centring solve
    /// (probing, quality-function). When either is `None` the adaptive
    /// path silently falls back to the LOQO closed form — matching
    /// upstream's "oracle returned no candidate" branch
    /// (`IpAdaptiveMuUpdate.cpp:CalculateMuFromOracle:330-340`).
    fn update_barrier_parameter(
        &mut self,
        data: &IpoptDataHandle,
        cq: &IpoptCqHandle,
        nlp: Option<&Rc<RefCell<dyn IpoptNlp>>>,
        pd_search_dir: Option<&mut PdSearchDirCalc>,
    ) -> Number;
}
