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

/// Per-entry violations and complementarities at an iterate, in the shape
/// of upstream `TNLP::get_curr_violations`: full `n` for the first five,
/// full `m` for the last two, zero where a bound or row does not apply.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Violations {
    pub x_l_violation: Vec<Number>,
    pub x_u_violation: Vec<Number>,
    pub compl_x_l: Vec<Number>,
    pub compl_x_u: Vec<Number>,
    pub grad_lag_x: Vec<Number>,
    pub constraint_violation: Vec<Number>,
    pub compl_g: Vec<Number>,
}

/// The declared problem the violations are measured against: the user's
/// own box and row bounds, with the infinity thresholds that mark a side
/// absent.
#[derive(Clone, Copy, Debug)]
pub struct DeclaredBounds<'a> {
    pub x_l: &'a [Number],
    pub x_u: &'a [Number],
    pub g_l: &'a [Number],
    pub g_u: &'a [Number],
    pub lower_inf: Number,
    pub upper_inf: Number,
}

/// `∇f + Jᵀλ − z_L + z_U` at `it`, from the gradient and the Jacobian
/// triplets (0-based) evaluated at `it.x`.
pub fn lagrangian_gradient(
    it: &UserIterate,
    grad_f: &[Number],
    jac_rows: &[usize],
    jac_cols: &[usize],
    jac_vals: &[Number],
) -> Vec<Number> {
    let mut s = grad_f.to_vec();
    for ((&r, &c), &v) in jac_rows.iter().zip(jac_cols).zip(jac_vals) {
        if let (Some(&l), Some(sc)) = (it.lambda.get(r), s.get_mut(c)) {
            *sc += v * l;
        }
    }
    for (j, sj) in s.iter_mut().enumerate() {
        *sj += it.z_u.get(j).copied().unwrap_or(0.0) - it.z_l.get(j).copied().unwrap_or(0.0);
    }
    s
}

/// The unscaled violations of upstream `get_curr_violations(scaled=false)`,
/// computed from the user-space iterate against the declared problem:
///
/// * `x_L_violation = max(0, x_L − x)` (upstream's
///   `unscaled_curr_orig_x_L_violation`: the *original*, unrelaxed bound);
/// * `compl_x_L = (x − x_L) · z_L`, the bound's slack times its multiplier.
///   Upstream multiplies the algorithm's slack, which is measured to the
///   bound `bound_relax_factor` widened, by `z_L / obj_scal`; the two differ
///   by at most `bound_relax_factor · max(1, |x_L|) · z_L`;
/// * `grad_lag_x = ∇f + Jᵀλ − z_L + z_U` (`grad_lag`; the caller evaluates
///   it), zeroed at fixed variables when `zero_fixed` — upstream's
///   `ResortX(…, usefixedvals=false)` under `make_parameter`;
/// * equality rows: violation `|g − g_L|`, `compl_g = −(g − g_L) · λ`;
///   inequality rows: violation `max(0, g_L − g, g − g_U)`,
///   `compl_g = min(λ, 0)(g_L − g) − max(λ, 0)(g − g_U)` over the finite
///   sides — upstream's signs, taken from `ResortBoundMultipliers`.
///
/// A side at or beyond the infinity threshold contributes `0`.
pub fn user_violations(
    it: &UserIterate,
    b: &DeclaredBounds<'_>,
    grad_lag: Option<Vec<Number>>,
    zero_fixed: bool,
) -> Violations {
    let n = it.x.len();
    let m = it.g.len();
    let mut v = Violations {
        x_l_violation: vec![0.0; n],
        x_u_violation: vec![0.0; n],
        compl_x_l: vec![0.0; n],
        compl_x_u: vec![0.0; n],
        grad_lag_x: grad_lag.unwrap_or_else(|| vec![0.0; n]),
        constraint_violation: vec![0.0; m],
        compl_g: vec![0.0; m],
    };
    for j in 0..n {
        let x = it.x[j];
        let (lo, up) = (b.x_l[j], b.x_u[j]);
        if lo > b.lower_inf {
            v.x_l_violation[j] = (lo - x).max(0.0);
            v.compl_x_l[j] = (x - lo) * it.z_l[j];
        }
        if up < b.upper_inf {
            v.x_u_violation[j] = (x - up).max(0.0);
            v.compl_x_u[j] = (up - x) * it.z_u[j];
        }
        if zero_fixed
            && lo == up
            && let Some(gj) = v.grad_lag_x.get_mut(j)
        {
            *gj = 0.0;
        }
    }
    for i in 0..m {
        let (g, lam) = (it.g[i], it.lambda[i]);
        let (lo, up) = (b.g_l[i], b.g_u[i]);
        if lo == up {
            v.constraint_violation[i] = (g - lo).abs();
            v.compl_g[i] = -(g - lo) * lam;
            continue;
        }
        let mut viol: Number = 0.0;
        let mut compl = 0.0;
        if lo > b.lower_inf {
            viol = viol.max(lo - g);
            compl += lam.min(0.0) * (lo - g);
        }
        if up < b.upper_inf {
            viol = viol.max(g - up);
            compl -= lam.max(0.0) * (g - up);
        }
        v.constraint_violation[i] = viol;
        v.compl_g[i] = compl;
    }
    v
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
