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

thread_local! {
    static CURRENT_CTX: RefCell<Option<IntermediateContext>> = const { RefCell::new(None) };
}

/// RAII guard — installs `ctx` on construction, clears on drop. Used
/// by the algorithm to scope visibility of live iterate state to one
/// callback fire.
pub struct CtxGuard {
    _private: (),
}

impl CtxGuard {
    pub fn install(ctx: IntermediateContext) -> Self {
        // wasm32-wasip1 browser/Node hosts are single-threaded here, and
        // touching this destructor-bearing TLS slot can block indefinitely.
        // The context only serves optional C-API inspector calls during the
        // callback.
        #[cfg(not(target_arch = "wasm32"))]
        CURRENT_CTX.with(|c| *c.borrow_mut() = Some(ctx));
        #[cfg(target_arch = "wasm32")]
        let _ = ctx;
        Self { _private: () }
    }
}

impl Drop for CtxGuard {
    fn drop(&mut self) {
        #[cfg(not(target_arch = "wasm32"))]
        CURRENT_CTX.with(|c| *c.borrow_mut() = None);
    }
}

/// Read access to the currently installed context. Returns `None`
/// outside the intermediate-callback window.
pub fn with_current<F, R>(f: F) -> Option<R>
where
    F: FnOnce(&IntermediateContext) -> R,
{
    #[cfg(target_arch = "wasm32")]
    {
        // Live-iterate inspection is unavailable on the Wasm build because
        // its callback context is deliberately not installed above.
        let _ = f;
        None
    }
    #[cfg(not(target_arch = "wasm32"))]
    CURRENT_CTX.with(|c| c.borrow().as_ref().map(f))
}

/// Whether a context is currently installed.
pub fn is_active() -> bool {
    #[cfg(target_arch = "wasm32")]
    {
        false
    }
    #[cfg(not(target_arch = "wasm32"))]
    CURRENT_CTX.with(|c| c.borrow().is_some())
}
