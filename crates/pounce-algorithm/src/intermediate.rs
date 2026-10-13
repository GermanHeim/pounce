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
use pounce_common::types::Number;
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

/// Iterate of the user's problem, already in the user's index spaces
/// (full `n`, full `m`, original `g` order).
///
/// The active-set SQP arm has no `IpoptData` / `IpoptCq` to stash handles
/// to, so it hands the callback window a materialised copy instead. The
/// interior-point arm never installs one.
#[derive(Clone, Debug)]
pub struct UserIterate {
    pub x: Vec<Number>,
    pub z_l: Vec<Number>,
    pub z_u: Vec<Number>,
    pub g: Vec<Number>,
    pub lambda: Vec<Number>,
}

thread_local! {
    static CURRENT_CTX: RefCell<Option<IntermediateContext>> = const { RefCell::new(None) };
    static CURRENT_USER_ITERATE: RefCell<Option<UserIterate>> = const { RefCell::new(None) };
}

/// RAII guard for a [`UserIterate`] — the SQP counterpart of [`CtxGuard`].
pub struct UserIterateGuard {
    _private: (),
}

impl UserIterateGuard {
    pub fn install(it: UserIterate) -> Self {
        CURRENT_USER_ITERATE.with(|c| *c.borrow_mut() = Some(it));
        Self { _private: () }
    }
}

impl Drop for UserIterateGuard {
    fn drop(&mut self) {
        CURRENT_USER_ITERATE.with(|c| *c.borrow_mut() = None);
    }
}

/// Read access to the SQP arm's current iterate; `None` outside its
/// callback window.
pub fn with_current_user_iterate<F, R>(f: F) -> Option<R>
where
    F: FnOnce(&UserIterate) -> R,
{
    CURRENT_USER_ITERATE.with(|c| c.borrow().as_ref().map(f))
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
    CURRENT_CTX.with(|c| c.borrow().is_some())
        || CURRENT_USER_ITERATE.with(|c| c.borrow().is_some())
}
