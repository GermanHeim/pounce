//! Per-iteration "intermediate" context shared with downstream
//! callers (notably the C-API inspector functions
//! `GetIpoptCurrentIterate` / `GetIpoptCurrentViolations`).
//!
//! Mirrors upstream Ipopt's `OrigIpoptNLP::GetIpoptCurrent*` flow: the
//! main loop installs a snapshot of the algorithm-side state into
//! thread-local storage immediately before invoking the user's
//! intermediate callback, and clears it on return. Inspector functions
//! consult the TLS slot; outside the callback window every accessor
//! reports "not available".
//!
//! The snapshot is intentionally cheap to assemble — we stash `Rc`
//! handles to `IpoptData`, `IpoptCq`, and the algorithm-side `IpoptNlp`
//! rather than precomputing every field, so callers that read just one
//! quantity pay only for what they look at.

use crate::ipopt_cq::IpoptCqHandle;
use crate::ipopt_data::IpoptDataHandle;
use crate::ipopt_nlp::IpoptNlp;
use std::cell::RefCell;
use std::rc::Rc;

/// Snapshot stashed in TLS for the duration of one
/// `TNLP::intermediate_callback` invocation.
#[derive(Clone)]
pub struct IntermediateContext {
    pub data: IpoptDataHandle,
    pub cq: IpoptCqHandle,
    pub nlp: Rc<RefCell<dyn IpoptNlp>>,
}

/// The materialised iterate of the user's problem (full `n`, full `m`,
/// original `g` order), and its window. Defined in
/// [`pounce_nlp::live_iterate`] so the decorators that lift it one layer
/// per `intermediate_callback` forward — presolve's above all — can reach
/// it; re-exported here, where callers have always found it.
///
/// The active-set SQP arm installs one at every fire. The interior-point
/// arm installs one when the TNLP asks for it
/// ([`pounce_nlp::tnlp::TNLP::wants_current_iterate`]), built by
/// [`ipm_user_iterate`].
pub use pounce_nlp::live_iterate::{UserIterate, UserIterateGuard, with_current_user_iterate};

thread_local! {
    static CURRENT_CTX: RefCell<Option<IntermediateContext>> = const { RefCell::new(None) };
}

/// The interior-point iterate in the space of the TNLP the NLP wraps, by
/// the conversions `finalize_via_orig_nlp` applies to the final one, so the
/// iterate read at the last fire is the solution `finalize_solution`
/// reports:
///
/// * `x` through `finalize_solution_x` (fixed variables spliced back in);
/// * `z_L`, `z_U`, `λ` through the `finalize_solution_*` lifts, which
///   divide out `obj_scale_factor` (the unscaled-Lagrangian convention);
/// * the bound multipliers of variables `make_parameter` removed recovered
///   from `z_L − z_U = ∇f + Jᵀλ` when `fixed_var_duals` — upstream
///   `TNLP::get_curr_iterate` does the same through
///   `TNLPAdapter::ResortBoundMultipliers` for every treatment but
///   `make_parameter_nodual`. Left at `0` they were not a zero multiplier
///   but a missing one;
/// * per-variable `user-scaling` (`x̃ = d ⊙ x` in a `ScalingTnlp` below the
///   NLP) undone, `x = x̃ ⊘ d`, `z = d ⊙ z̃`;
/// * `g` from the iterate's own `c` / `d` values, unscaled.
///
/// The fixed-variable recovery evaluates the TNLP, so this must run before
/// the callback borrows it, never from inside the callback.
pub fn ipm_user_iterate(
    nlp: &Rc<RefCell<dyn IpoptNlp>>,
    data: &IpoptDataHandle,
    cq: &IpoptCqHandle,
    fixed_var_duals: bool,
) -> Option<UserIterate> {
    let curr = data.borrow().curr.clone()?;
    // `curr_c` / `curr_d` evaluate through the NLP when not cached; no NLP
    // borrow may be alive across them.
    let (c, d) = {
        let cq = cq.borrow();
        (cq.curr_c(), cq.curr_d())
    };
    let nlp = nlp.borrow();
    let n = nlp.n_full_x() as usize;
    let m = nlp.n_full_g() as usize;
    let mut x = nlp.finalize_solution_x(&*curr.x);
    let mut z_l = nlp.finalize_solution_z_l(&*curr.z_l);
    if z_l.is_empty() {
        z_l = vec![0.0; n];
    }
    let mut z_u = nlp.finalize_solution_z_u(&*curr.z_u);
    if z_u.is_empty() {
        z_u = vec![0.0; n];
    }
    let mut lambda = nlp.finalize_solution_lambda(&*curr.y_c, &*curr.y_d);
    if lambda.is_empty() {
        lambda = vec![0.0; m];
    }
    if fixed_var_duals {
        nlp.complete_fixed_var_bound_multipliers(&x, &lambda, &mut z_l, &mut z_u);
    }
    pounce_nlp::scaling_tnlp::unscale_iterate(
        nlp.variable_scaling().as_deref(),
        &mut x,
        &mut z_l,
        &mut z_u,
    );
    let g = nlp.pack_g_for_user(&*c, &*d);
    Some(UserIterate {
        x,
        z_l,
        z_u,
        g,
        lambda,
    })
}

/// RAII guard — installs `ctx` on construction, clears on drop. Used
/// by the algorithm to scope visibility of live iterate state to one
/// callback fire.
pub struct CtxGuard {
    _private: (),
}

impl CtxGuard {
    pub fn install(ctx: IntermediateContext) -> Self {
        CURRENT_CTX.with(|c| *c.borrow_mut() = Some(ctx));
        Self { _private: () }
    }
}

impl Drop for CtxGuard {
    fn drop(&mut self) {
        CURRENT_CTX.with(|c| *c.borrow_mut() = None);
    }
}

/// Read access to the currently installed context. Returns `None`
/// outside the intermediate-callback window.
pub fn with_current<F, R>(f: F) -> Option<R>
where
    F: FnOnce(&IntermediateContext) -> R,
{
    CURRENT_CTX.with(|c| c.borrow().as_ref().map(f))
}

/// Whether a context is currently installed.
pub fn is_active() -> bool {
    CURRENT_CTX.with(|c| c.borrow().is_some()) || pounce_nlp::live_iterate::is_installed()
}
