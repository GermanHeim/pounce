use std::cell::Cell;
use std::time::{Duration, Instant};

thread_local! {
    static DEADLINE: Cell<Option<Instant>> = const { Cell::new(None) };
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
            true
        }
    });
    Guard { owner }
}

impl Drop for Guard {
    fn drop(&mut self) {
        if self.owner {
            DEADLINE.with(|slot| slot.set(None));
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
