//! "The debugger asked us to stop" — a solve-wide latch, on the same rails as
//! [`crate::deadline`].
//!
//! A [`DebugAction::Stop`](pounce_common::debug::DebugAction::Stop) breaks the
//! interior-point loop where it stands and leaves the solution carrying
//! whatever non-converged status the loop had reached. That is the whole point
//! of the debugger's `quit`: you get the run you were watching, stopped where
//! you stopped it.
//!
//! It becomes a problem the moment the debug entry points stop being parallel
//! implementations and enter the ordinary solve path instead (gh #892). The
//! ordinary path reads a non-converged status as *something to recover from*
//! and re-solves — the gh #293 equilibrated retry, the gh #414 verify, the
//! gh #226 PSD HSDE retry, the LP crossover. Those re-solves run unhooked, by
//! design, so they would run to completion after you quit and hand back a
//! clean `Optimal` for a solve you deliberately halted: `quit` would report
//! success. Measured on `debug.rs::stop_action_halts_the_solve`, which is what
//! caught it.
//!
//! So a stop is latched here when it happens and every recovery gate consults
//! it, exactly as each already consults [`crate::deadline::expired`]. Unlike a
//! deadline this never restamps a verdict — it only declines to start another
//! solve, which leaves the halted run's own status standing.
//!
//! The latch is per-solve (scoped by [`with_scope`], which the public debug
//! entry points open) and per-thread, so a stopped solve cannot affect the
//! next one or a concurrent one.

use std::cell::Cell;

thread_local! {
    /// `Some(false)` inside a debug scope that has not been stopped;
    /// `Some(true)` once a hook has asked to stop; `None` outside any scope
    /// (no debugger attached, so there is nothing to latch).
    static STOPPED: Cell<Option<bool>> = const { Cell::new(None) };
}

/// Run `f` with a fresh stop latch. Nested scopes reuse the outer one, so a
/// stop inside a sub-solve is still visible to the entry point that owns it.
pub(crate) fn with_scope<T>(f: impl FnOnce() -> T) -> T {
    STOPPED.with(|slot| {
        if slot.get().is_some() {
            return f();
        }
        slot.set(Some(false));
        struct Reset<'a>(&'a Cell<Option<bool>>);
        impl Drop for Reset<'_> {
            fn drop(&mut self) {
                self.0.set(None);
            }
        }
        let _reset = Reset(slot);
        f()
    })
}

/// Latch a stop. Called from [`crate::debug::fire`], the one place a
/// `DebugAction` is read, so no driver can forget to report one.
#[inline]
pub(crate) fn mark() {
    STOPPED.with(|slot| {
        if slot.get().is_some() {
            slot.set(Some(true));
        }
    });
}

/// Has a hook asked this solve to stop? `false` when no debugger is attached,
/// which is what makes every gate below a no-op on the ordinary path.
#[inline]
pub(crate) fn requested() -> bool {
    STOPPED.with(|slot| slot.get() == Some(true))
}
