//! The live iterate of the user's problem, for the intermediate-callback
//! window (`GetIpoptCurrentIterate` / `GetIpoptCurrentViolations`,
//! `pounce_algorithm::intermediate::with_current_user_iterate`).
//!
//! Upstream Ipopt's `TNLP::get_curr_iterate` reads the algorithm's iterate
//! and converts it on the spot, through `TNLPAdapter::ResortX` /
//! `ResortBoundMultipliers` / `ResortG`, into the space the user's TNLP
//! declared. pounce cannot do that conversion at read time: the reader runs
//! inside the user's callback, which holds the TNLP mutably, and part of the
//! conversion — the bound multipliers of fixed variables, the rows and
//! columns a presolve layer removed — has to *evaluate* that TNLP. So the
//! algorithm materialises the iterate before the callback, in the space of
//! the TNLP it solves, and each decorator between that TNLP and the user's
//! lifts it one layer on its way through `intermediate_callback`, by the
//! rule its own `finalize_solution` applies. What the user reads at the last
//! fire is then the point `finalize_solution` reports.
//!
//! Lives here rather than in `pounce-algorithm` so the decorators that lift
//! it (`pounce-presolve`'s, which do not depend on the algorithm crate) can
//! reach it.

use pounce_common::types::Number;
use std::cell::RefCell;

/// Iterate of a TNLP in that TNLP's own index spaces (full `n`, full `m`,
/// its `g` order), in the units `finalize_solution` reports: the
/// unscaled-Lagrangian multipliers `∇f + Jᵀλ − z_L + z_U = 0`.
#[derive(Clone, Debug, Default)]
pub struct UserIterate {
    pub x: Vec<Number>,
    pub z_l: Vec<Number>,
    pub z_u: Vec<Number>,
    pub g: Vec<Number>,
    pub lambda: Vec<Number>,
}

thread_local! {
    static CURRENT: RefCell<Option<UserIterate>> = const { RefCell::new(None) };
}

/// Installs a [`UserIterate`] for the duration of one callback fire and
/// restores whatever was installed before on drop, so a decorator can
/// shadow the iterate with its lifted one while it forwards the callback.
pub struct UserIterateGuard {
    prev: Option<UserIterate>,
}

impl UserIterateGuard {
    pub fn install(it: UserIterate) -> Self {
        let prev = CURRENT.with(|c| c.borrow_mut().replace(it));
        Self { prev }
    }
}

impl Drop for UserIterateGuard {
    fn drop(&mut self) {
        let prev = self.prev.take();
        CURRENT.with(|c| *c.borrow_mut() = prev);
    }
}

/// Read access to the installed iterate; `None` outside a callback window.
pub fn with_current_user_iterate<F, R>(f: F) -> Option<R>
where
    F: FnOnce(&UserIterate) -> R,
{
    CURRENT.with(|c| c.borrow().as_ref().map(f))
}

/// A copy of the installed iterate, for a decorator about to lift it.
pub fn current_user_iterate() -> Option<UserIterate> {
    CURRENT.with(|c| c.borrow().clone())
}

/// Whether an iterate is installed.
pub fn is_installed() -> bool {
    CURRENT.with(|c| c.borrow().is_some())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn it(x: Number) -> UserIterate {
        UserIterate {
            x: vec![x],
            ..Default::default()
        }
    }

    #[test]
    fn a_nested_guard_restores_the_outer_iterate() {
        assert!(!is_installed());
        {
            let _outer = UserIterateGuard::install(it(1.0));
            {
                let _inner = UserIterateGuard::install(it(2.0));
                assert_eq!(with_current_user_iterate(|u| u.x[0]), Some(2.0));
            }
            assert_eq!(with_current_user_iterate(|u| u.x[0]), Some(1.0));
        }
        assert!(!is_installed());
    }
}
