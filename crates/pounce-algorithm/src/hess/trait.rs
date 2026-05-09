//! `HessianUpdater` trait — port of `IpHessianUpdater.hpp`.

use crate::ipopt_cq::IpoptCqHandle;
use crate::ipopt_data::IpoptDataHandle;

pub trait HessianUpdater {
    /// Refresh `data.w` for the current iterate. Returns `true` on
    /// success. Mirrors `IpHessianUpdater::UpdateHessian` (which is
    /// pure-virtual; implementations write into `IpData().Set_W(...)`).
    fn update_hessian(&mut self, data: &IpoptDataHandle, cq: &IpoptCqHandle) -> bool;
}
