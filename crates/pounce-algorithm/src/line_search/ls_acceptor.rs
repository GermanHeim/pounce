//! Line-search acceptor trait — port of `IpBacktrackingLSAcceptor.hpp`
//! and `IpLineSearch.hpp`.

use crate::ipopt_cq::IpoptCqHandle;
use crate::ipopt_data::IpoptDataHandle;
use crate::iterates_vector::IteratesVector;
use crate::line_search::filter_acceptor::AcceptDecision;
use crate::restoration::OrigProgressCallback;
use pounce_common::types::Number;

/// Acceptor side of the backtracking line search. Concrete impls:
/// [`super::filter_acceptor::FilterLsAcceptor`] (Phase 7),
/// `PenaltyLsAcceptor` (Phase 10), `CGPenaltyLsAcceptor` (Phase 10).
///
/// The driver calls `check_trial_point` on each backtracking step.
/// Acceptors that need the trial-iterate components (rather than
/// scalar `(theta, phi)`) can extend this surface in later phases —
/// the filter acceptor only needs the four scalars upstream feeds in
/// at line `IpFilterLSAcceptor.cpp:CheckAcceptabilityOfTrialPoint`.
pub trait BacktrackingLsAcceptor {
    /// Reset acceptor state for a new outer iteration.
    fn reset(&mut self);

    /// Hook called once per outer iteration, after the search direction
    /// `delta` has been computed and before the α-loop. Mirrors
    /// `IpPenaltyLSAcceptor.cpp:InitThisLineSearch` — the penalty
    /// acceptor uses it to snapshot reference (θ, φ, ∇φᵀδ, δᵀWδ) and to
    /// bump the penalty parameter ν. Default: no-op (filter acceptor
    /// has nothing to cache between α-loop iterations).
    fn init_this_line_search(
        &mut self,
        _data: &IpoptDataHandle,
        _cq: &IpoptCqHandle,
        _delta: &IteratesVector,
    ) {
    }

    /// Compute the minimum primal step length below which the
    /// driver should declare a tiny step / hand off to restoration.
    /// Mirrors `IpFilterLSAcceptor.cpp:CalculateAlphaMin` — the value
    /// depends on the current `(theta, d_phi)` pair (the directional
    /// derivative of the barrier objective along the search step) and
    /// on the acceptor's lazily-initialised `theta_min`. Default impl
    /// returns 0.0 so non-filter acceptors degenerate to the driver's
    /// own absolute `alpha_min` floor.
    fn calc_alpha_min(&mut self, _d_phi: Number, _theta: Number) -> Number {
        0.0
    }

    /// Decide whether the trial `(theta_trial, phi_trial)` at primal
    /// step `alpha_primal` is acceptable, given the current iterate's
    /// `(theta, phi)` and the directional derivative `d_phi`.
    /// Default: always accept (lets stub acceptors compose without
    /// interfering with the driver's α-loop).
    fn check_trial_point(
        &self,
        _alpha_primal: Number,
        _theta: Number,
        _phi: Number,
        _d_phi: Number,
        _theta_trial: Number,
        _phi_trial: Number,
    ) -> AcceptDecision {
        AcceptDecision::Accept
    }

    /// Post-accept hook — port of
    /// `IpFilterLSAcceptor::UpdateForNextIteration`. Both decides the
    /// `info_alpha_primal_char` tag *and* augments the filter when
    /// upstream would. Returns:
    ///
    /// * `'f'` — F-type Armijo step (`IsFtype && ArmijoHolds`); filter
    ///   is **not** augmented.
    /// * `'h'` — anything else (`!IsFtype || !ArmijoHolds`); filter
    ///   **is** augmented with `(theta_add, phi_add) = ((1 - γ_θ)·θ_ref,
    ///   φ_ref - γ_φ·θ_ref)`.
    ///
    /// The driver calls this once per accepted step, after
    /// `check_trial_point` returns Accept and before
    /// `accept_trial_point` promotes `trial → curr`. Default impl
    /// returns `'h'` (no filter), so non-filter acceptors remain valid.
    fn update_for_next_iteration(
        &mut self,
        _alpha_primal: Number,
        _theta: Number,
        _phi: Number,
        _d_phi: Number,
        _phi_trial: Number,
    ) -> char {
        'h'
    }

    /// Build the orig-progress callback the inner restoration IPM
    /// should consult to decide whether the recovered iterate is
    /// acceptable to *this* (outer) acceptor's filter and reference
    /// iterate. Mirrors upstream
    /// `IpRestoFilterConvCheck::SetOrigLSAcceptor` /
    /// `TestOrigProgress`. Default returns `None` — penalty / CG-penalty
    /// acceptors do not gate restoration on a filter, so they fall
    /// through to the kappa-reduction-only guard.
    ///
    /// `reference_theta` and `reference_barr` are the outer iterate's
    /// `(curr_constraint_violation, curr_barrier_obj)` at restoration
    /// entry; `obj_max_inc` is the upstream `obj_max_inc` option
    /// (default 5.0).
    fn make_orig_progress_check(
        &self,
        _reference_theta: Number,
        _reference_barr: Number,
        _obj_max_inc: Number,
    ) -> Option<OrigProgressCallback> {
        None
    }
}
