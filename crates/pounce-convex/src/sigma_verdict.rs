//! Whether the `σ` cascade certified the answer it returned.
//!
//! gh #880. When cost normalization engages, [`crate::ipm::solve_qp_core`]
//! may try three drivers, ask `normalized_optimum_is_genuine` of each, reject
//! all three, and still return the best of them — which is the right point to
//! keep, but not a certified one. The caller has to be told, because a
//! `QpStatus::Optimal` on an uncertified point is exactly the wrong answer
//! that issue reports.
//!
//! Carried out-of-band rather than on [`crate::QpSolution`]: the struct has
//! 145 construction sites and no `Default`, so a field is a crate-wide edit
//! for one bit. This mirrors [`crate::deadline`], including its
//! outermost-frame-owns-the-slot discipline and RAII reset.
//!
//! The discipline that makes it correct is that [`record`] is called on
//! **every** exit from the cascade, not only the uncertified one, and
//! [`clear`] runs at the top of every solve attempt. A retry that re-enters
//! the solver overwrites the verdict with its own, so the slot always
//! describes the attempt whose answer is actually being returned — never a
//! stale `true` from an attempt that was superseded.

use std::cell::Cell;

thread_local! {
    /// `None` outside a tracked solve; `Some(true)` once an attempt has
    /// returned an answer the `σ` cascade declined to certify.
    static SLOT: Cell<Option<bool>> = const { Cell::new(None) };
}

/// Run `f` as the outermost convex solve, reporting alongside its result
/// whether the `σ` cascade declined to certify the answer being returned.
///
/// Nested calls (the un-normalized re-solve and the direct-driver fallback
/// both re-enter the solver) see the slot already installed and leave
/// ownership to the outer frame, which is what makes the outer cascade's
/// verdict — recorded last, after every inner solve has finished — the one
/// that survives.
pub(crate) fn tracking<T>(f: impl FnOnce() -> T) -> (T, bool) {
    SLOT.with(|slot| {
        if slot.get().is_some() {
            return (f(), false);
        }
        slot.set(Some(false));
        struct Reset<'a>(&'a Cell<Option<bool>>);
        impl Drop for Reset<'_> {
            fn drop(&mut self) {
                self.0.set(None);
            }
        }
        let _reset = Reset(slot);
        let out = f();
        (out, slot.get() == Some(true))
    })
}

/// Record this attempt's verdict, overwriting any earlier one.
pub(crate) fn record(uncertified: bool) {
    SLOT.with(|slot| {
        if slot.get().is_some() {
            slot.set(Some(uncertified));
        }
    });
}

/// Reset at the top of a solve attempt, so a retry cannot inherit a verdict
/// about an answer it is replacing.
pub(crate) fn clear() {
    record(false);
}
