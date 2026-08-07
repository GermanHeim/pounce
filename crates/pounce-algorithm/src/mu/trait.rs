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

    /// Whether the main loop may infer `STOP_AT_TINY_STEP` from
    /// "tiny-step flag set and μ came back unchanged".
    ///
    /// Upstream `IpMonotoneMuUpdate.cpp:158-161` throws
    /// `TINY_STEP_DETECTED` in exactly that case, and it is the update's
    /// only throw site, so the inference is exact — `MonotoneMuUpdate`
    /// overrides to `true`.
    ///
    /// `IpAdaptiveMuUpdate.cpp` also terminates on a tiny step (`:330-333`
    /// and `:377-380`), but only at those two sites; elsewhere it routes
    /// the flag through `force_no_progress`, fixing μ and continuing. The
    /// μ comparison cannot tell the two apart, so the adaptive update
    /// signals its own termination through
    /// [`IpoptData::request_tiny_step_stop`](crate::ipopt_data::IpoptData::request_tiny_step_stop)
    /// and leaves this `false` (pounce#512). A `false` here means "does
    /// not use the inference", not "never terminates".
    fn terminates_on_tiny_step(&self) -> bool {
        false
    }
}
