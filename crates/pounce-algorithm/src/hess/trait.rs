//! `HessianUpdater` trait — port of `IpHessianUpdater.hpp`.

use crate::ipopt_cq::IpoptCqHandle;
use crate::ipopt_data::IpoptDataHandle;
use std::rc::Rc;

pub trait HessianUpdater {
    /// Refresh `data.w` for the current iterate. Returns `true` on
    /// success. Mirrors `IpHessianUpdater::UpdateHessian` (which is
    /// pure-virtual; implementations write into `IpData().Set_W(...)`).
    fn update_hessian(&mut self, data: &IpoptDataHandle, cq: &IpoptCqHandle) -> bool;

    /// Whether `data.w` is the *exact* Lagrangian Hessian at the iterate it was
    /// built from, rather than a quasi-Newton approximation (gh #797).
    ///
    /// The negative-curvature probe needs `W` at the iterate it is judging, and
    /// `data.w` is always one iterate behind at the point the convergence check
    /// runs. When this is `true` the caller re-evaluates
    /// [`crate::ipopt_cq::IpoptCalculatedQuantities::curr_exact_hessian`]
    /// instead — a pure evaluation, where calling `update_hessian` a second
    /// time at the same iterate would feed the limited-memory updater a
    /// zero-length curvature pair and count it against
    /// `limited_memory_max_skipping`.
    fn provides_exact_hessian(&self) -> bool {
        false
    }

    /// `W` at the *current* iterate, for an updater that can produce it as a
    /// pure function of `(x, y)` with no dependence on the step history.
    ///
    /// This exists because `provides_exact_hessian` conflated two different
    /// questions (gh#823 review, finding 1, reported by @srikanth-gm): "is
    /// this the exact Hessian?" and "can this be re-evaluated at the current
    /// iterate?". The negative-curvature probe only ever needed the second.
    /// Answering it with the first is safe for the limited-memory updater —
    /// BFGS keeps `B` positive definite, so the probe declines at `δ_x = 0`
    /// and declining is correct — but it is *not* safe for a finite-difference
    /// Hessian, which carries genuine negative curvature and was being judged
    /// one iterate stale. The measured symptom is a stationary maximum
    /// reported as optimal, where the exact path escapes it.
    ///
    /// Returning `None` keeps the previous behaviour: the probe runs against
    /// whatever `data.w` holds. That is the right answer for a history-carrying
    /// quasi-Newton `B`, where there is nothing to re-evaluate — recomputing
    /// would hand the updater a zero-length curvature pair to skip and count
    /// against `limited_memory_max_skipping`.
    ///
    /// An implementation MUST NOT leave `data.w` describing a different
    /// iterate than it found it describing; the post-optimal sensitivity hook
    /// reads it.
    fn hessian_at_current(
        &mut self,
        _data: &IpoptDataHandle,
        _cq: &IpoptCqHandle,
    ) -> Option<Rc<dyn pounce_linalg::SymMatrix>> {
        None
    }
}
