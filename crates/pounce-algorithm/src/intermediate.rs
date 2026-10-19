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
use pounce_common::types::{Index, Number};
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

fn dense_values(v: &dyn pounce_linalg::Vector) -> Vec<Number> {
    use pounce_linalg::dense_vector::DenseVector;
    match v.as_any().downcast_ref::<DenseVector>() {
        Some(d) => d.expanded_values(),
        None => vec![0.0; v.dim().max(0) as usize],
    }
}

fn dense(vals: &[Number]) -> pounce_linalg::dense_vector::DenseVector {
    let mut d =
        pounce_linalg::dense_vector::DenseVectorSpace::new(vals.len() as Index).make_new_dense();
    if !vals.is_empty() {
        d.set_values(vals);
    }
    d
}

/// `P · v` for an expansion matrix `P` (compressed bound slots → `dim`).
fn expand(p: &dyn pounce_linalg::Matrix, v: &[Number], dim: usize) -> Vec<Number> {
    let mut out = dense(&vec![0.0; dim]);
    if !v.is_empty() && dim > 0 {
        p.mult_vector(1.0, &dense(v), 0.0, &mut out);
    }
    dense_values(&out)
}

/// The **scaled** violations of upstream `get_curr_violations(scaled=true)`:
/// the algorithm's own quantities, in its own (objective-, row- and
/// variable-scaled) space, scattered to the full `x` / `g` of the TNLP the
/// NLP wraps.
///
/// * bound violations against the *declared* box (`curr_orig_x_L_violation`
///   upstream), in the algorithm's coordinates;
/// * `compl_x_*` the algorithm's own `slack · z` (`curr_compl_x_L`);
/// * `grad_lag_x` the algorithm's `∇L`, `0` at fixed variables
///   (`ResortX(…, usefixedvals=false)`);
/// * constraint violation `|c|` and `max(0, d_L − d, d − d_U)` against the
///   declared, scaled row bounds; `compl_g = −c · y_c` and
///   `min(y_d, 0)(d_L − d) − max(y_d, 0)(d − d_U)`.
///
/// `None` with no live iterate. The callback holds the TNLP, so nothing
/// here may evaluate it — every quantity is the iterate's cached one.
pub fn ipm_scaled_violations(
    ctx: &IntermediateContext,
) -> Option<pounce_nlp::live_iterate::Violations> {
    if ctx.data.borrow().curr.is_none() {
        return None;
    }
    // Evaluate everything off `cq` before any `nlp` borrow.
    let (slack_l, slack_u, compl_l, compl_u, glx, c, d, y_c, y_d) = {
        let cq = ctx.cq.borrow();
        let curr = ctx.data.borrow().curr.clone()?;
        (
            dense_values(&*cq.curr_slack_x_l()),
            dense_values(&*cq.curr_slack_x_u()),
            cq.curr_compl_x_l(),
            cq.curr_compl_x_u(),
            cq.curr_grad_lag_x(),
            dense_values(&*cq.curr_c()),
            dense_values(&*cq.curr_d()),
            dense_values(&*curr.y_c),
            dense_values(&*curr.y_d),
        )
    };
    let nlp = ctx.nlp.borrow();
    let n = nlp.n_full_x() as usize;
    let m = nlp.n_full_g() as usize;
    // x = live bound + slack; the violation is measured to the declared one.
    let live_l = dense_values(nlp.x_l());
    let live_u = dense_values(nlp.x_u());
    let (decl_l, decl_u) = nlp
        .declared_x_bounds()
        .unwrap_or_else(|| (live_l.clone(), live_u.clone()));
    let viol_l: Vec<Number> = (0..slack_l.len())
        .map(|i| {
            let b = decl_l.get(i).copied().unwrap_or(live_l[i]);
            (b - (live_l[i] + slack_l[i])).max(0.0)
        })
        .collect();
    let viol_u: Vec<Number> = (0..slack_u.len())
        .map(|i| {
            let b = decl_u.get(i).copied().unwrap_or(live_u[i]);
            ((live_u[i] - slack_u[i]) - b).max(0.0)
        })
        .collect();
    let mut grad_lag_x = nlp.lift_x_to_full(&*glx);
    for (j, g) in grad_lag_x.iter_mut().enumerate() {
        if nlp.full_x_to_var_x(j as Index).is_none() {
            *g = 0.0;
        }
    }
    // Row bounds, expanded from their compressed slots to the d block, with
    // a 0/1 mask for which rows have that side at all.
    let m_d = d.len();
    let live_dl = dense_values(nlp.d_l());
    let live_du = dense_values(nlp.d_u());
    let (decl_dl, decl_du) = nlp
        .declared_d_bounds()
        .unwrap_or_else(|| (live_dl.clone(), live_du.clone()));
    let pd_l = nlp.pd_l();
    let pd_u = nlp.pd_u();
    let d_lo = expand(&*pd_l, &decl_dl, m_d);
    let d_up = expand(&*pd_u, &decl_du, m_d);
    let has_lo = expand(&*pd_l, &vec![1.0; decl_dl.len()], m_d);
    let has_up = expand(&*pd_u, &vec![1.0; decl_du.len()], m_d);
    let mut constraint_violation = vec![0.0; m];
    let mut compl_g = vec![0.0; m];
    for i in 0..m {
        if let Some(ci) = nlp.full_g_to_c_block(i as Index).map(|k| k as usize) {
            if let (Some(&cv), Some(&yv)) = (c.get(ci), y_c.get(ci)) {
                constraint_violation[i] = cv.abs();
                compl_g[i] = -cv * yv;
            }
        } else if let Some(di) = nlp.full_g_to_d_block(i as Index).map(|k| k as usize) {
            let (Some(&dv), Some(&yv)) = (d.get(di), y_d.get(di)) else {
                continue;
            };
            let mut viol: Number = 0.0;
            let mut compl = 0.0;
            if has_lo[di] != 0.0 {
                viol = viol.max(d_lo[di] - dv);
                compl += yv.min(0.0) * (d_lo[di] - dv);
            }
            if has_up[di] != 0.0 {
                viol = viol.max(dv - d_up[di]);
                compl -= yv.max(0.0) * (dv - d_up[di]);
            }
            constraint_violation[i] = viol;
            compl_g[i] = compl;
        }
    }
    let v = pounce_nlp::live_iterate::Violations {
        x_l_violation: nlp.pack_z_l_for_user(&dense(&viol_l)),
        x_u_violation: nlp.pack_z_u_for_user(&dense(&viol_u)),
        compl_x_l: nlp.pack_z_l_for_user(&*compl_l),
        compl_x_u: nlp.pack_z_u_for_user(&*compl_u),
        grad_lag_x,
        constraint_violation,
        compl_g,
    };
    (v.grad_lag_x.len() == n).then_some(v)
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
