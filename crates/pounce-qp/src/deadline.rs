//! Solve-wide monotonic wall-clock deadline for the active-set solver.
//!
//! One deadline per *top-level* solve: the nested stages a solve can reach —
//! homotopy, elastic phase-1, feasibility/recovery solves, seeded retries —
//! share the outer budget instead of each restarting the duration.

use std::cell::Cell;
use std::time::{Duration, Instant};

thread_local! {
    static DEADLINE: Cell<Option<Instant>> = const { Cell::new(None) };
    /// When the owning scope started. Used only to report how long a
    /// cancelled solve actually ran (`QpStats::time`), so a timeout is not
    /// recorded as an instantaneous solve.
    static SCOPE_START: Cell<Option<Instant>> = const { Cell::new(None) };
}

pub(crate) struct Guard {
    owner: bool,
}

pub(crate) fn enter(limit: Option<Duration>) -> Guard {
    let owner = DEADLINE.with(|slot| {
        if slot.get().is_some() {
            false
        } else {
            let now = Instant::now();
            let deadline = limit.and_then(|d| now.checked_add(d));
            slot.set(deadline);
            SCOPE_START.with(|s| s.set(Some(now)));
            true
        }
    });
    Guard { owner }
}

impl Drop for Guard {
    fn drop(&mut self) {
        if self.owner {
            DEADLINE.with(|slot| slot.set(None));
            SCOPE_START.with(|s| s.set(None));
        }
    }
}

#[inline]
pub(crate) fn expired() -> bool {
    DEADLINE.with(|slot| {
        slot.get()
            .is_some_and(|deadline| Instant::now() >= deadline)
    })
}

/// Wall-clock time spent inside the current top-level solve, or
/// [`Duration::ZERO`] when there is no active scope.
pub(crate) fn scope_elapsed() -> Duration {
    SCOPE_START.with(|s| s.get().map_or(Duration::ZERO, |start| start.elapsed()))
}
