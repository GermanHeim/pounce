//! `HessianUpdater` trait — port of `IpHessianUpdater.hpp`.

use crate::ipopt_cq::IpoptCqHandle;
use crate::ipopt_data::IpoptDataHandle;

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

    /// Discard the accumulated quasi-Newton curvature and re-anchor the
    /// approximation at the current iterate, returning `true` if there
    /// was anything to discard (gh#818).
    ///
    /// This is the recovery for "the *direction* is unusable", as
    /// distinct from restoration's "the *point* is infeasible". A
    /// limited-memory model can carry curvature the iterate has left
    /// behind, and when the resulting step cannot be accepted at any
    /// step length the fix is to forget it, not to walk somewhere else.
    /// L-BFGS-B does exactly this on a line-search failure (`col = 0`
    /// in `mainlb`); Ipopt has no equivalent, because it always has
    /// restoration to fall back on — which is fine while the point is
    /// infeasible and a no-op when it is not.
    ///
    /// The default is `false`: an exact Hessian has no history, so
    /// there is nothing to re-anchor and the caller must fall straight
    /// through to its existing hand-off.
    fn reanchor(&mut self) -> bool {
        false
    }
}
