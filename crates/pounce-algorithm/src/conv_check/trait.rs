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

    /// Whether the supplied `nlp_err` is at or below the acceptable
    /// tolerance — port of upstream
    /// `OptimalityErrorConvergenceCheck::CurrentIsAcceptable`. Used by
    /// the main loop to gate `StoreAcceptablePoint` /
    /// `RestoreAcceptablePoint`. Default returns `false` so policies
    /// that don't track an acceptable level (e.g. resto-of-resto inner
    /// adapters) silently skip the rollback machinery.
    fn current_is_acceptable(&self, _nlp_err: Number) -> bool {
        false
    }

    /// Outer NLP convergence tolerance, as used by the main loop's
    /// almost-feasible bypass guard (port of
    /// `IpBacktrackingLineSearch.cpp:580`). Default `1e-8` matches
    /// upstream's default `tol`.
    fn tol_or_default(&self) -> Number {
        1e-8
    }
}
