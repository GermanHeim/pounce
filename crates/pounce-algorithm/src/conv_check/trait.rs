//! `ConvCheck` trait — port of `IpConvCheck.hpp`.
//!
//! Upstream `ConvergenceCheck::CheckConvergence` reads the NLP error
//! and iter count off `IpData()`/`IpCq()`. The default trait method
//! pushes that read to the caller (the main loop in `ipopt_alg.rs`)
//! so simple convergence policies stay pure scalar state machines
//! over `(nlp_err, iter_count)`. The richer
//! [`ConvCheck::check_convergence_with_state`] entry point exposes
//! the live `(IpoptData, IpoptCq)` so policies that need iterate
//! components — notably the restoration-side
//! `RestoFilterConvergenceCheck::TestOrigProgress` — can read them.
//! Default impl just delegates to the scalar method, preserving
//! backwards compatibility for every existing impl.

use crate::ipopt_cq::IpoptCqHandle;
use crate::ipopt_data::IpoptDataHandle;
use pounce_common::types::{Index, Number};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConvergenceStatus {
    Continue,
    Converged,
    MaxIterExceeded,
    Failed,
}

pub trait ConvCheck {
    fn check_convergence(&mut self, nlp_err: Number, iter_count: Index) -> ConvergenceStatus;

    /// State-aware convergence check. The main loop calls this on
    /// every iteration so policies that need access to the iterate
    /// (e.g. `RestoConvCheckAdapter`'s orig-NLP `inf_pr` evaluation
    /// for the kappa-reduction early-exit) can read `data.curr` and
    /// the cq layer. Default impl delegates to
    /// [`Self::check_convergence`], so scalar-only policies don't
    /// need to override.
    fn check_convergence_with_state(
        &mut self,
        nlp_err: Number,
        iter_count: Index,
        _data: &IpoptDataHandle,
        _cq: &IpoptCqHandle,
    ) -> ConvergenceStatus {
        self.check_convergence(nlp_err, iter_count)
    }
}
