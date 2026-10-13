//! gh#1015 — `TNLP::intermediate_callback` and `GetIpoptCurrentIterate` on
//! the active-set SQP arm (`algorithm = active-set-sqp`).
//!
//! The interior-point arm's version of this lives in
//! `current_iterate_inspectors.rs`. The SQP arm has no `IpoptData` /
//! `IpoptCq`, so it installs a materialised user-space iterate instead; this
//! file pins that the two behave alike from C: one fire per iteration
//! (iteration 0 and the converged iterate included), a live iterate in the
//! user's index spaces, and `false` ending the solve.
//!
//! `GetIpoptCurrentViolations` is not served on this arm and returns FALSE.

use pounce_cinterface::*;
use pounce_nlp::return_codes::ApplicationReturnStatus;
use std::cell::RefCell;
use std::ffi::{CString, c_void};

const N: usize = 2;
const M: usize = 1;

#[derive(Clone, Debug)]
struct Fire {
    iter: Index,
    ok: bool,
    x: [Number; N],
    z_l: [Number; N],
    z_u: [Number; N],
    g: [Number; M],
    lambda: [Number; M],
}

thread_local! {
    static FIRES: RefCell<Vec<Fire>> = const { RefCell::new(Vec::new()) };
    /// `(fires seen so far, x handed to eval_h)` at each Hessian evaluation.
    static HESS_CALLS: RefCell<Vec<(usize, [Number; N])>> = const { RefCell::new(Vec::new()) };
    static STOP_AT: RefCell<Option<Index>> = const { RefCell::new(None) };
}

unsafe extern "C" fn ev_f(
    _n: Index,
    x: *const Number,
    _new_x: Bool,
    obj: *mut Number,
    _u: *mut c_void,
) -> Bool {
    unsafe {
        let x = std::slice::from_raw_parts(x, N);
        *obj = (1.0 - x[0]).powi(2) + 100.0 * (x[1] - x[0] * x[0]).powi(2);
        1
    }
}

unsafe extern "C" fn ev_grad_f(
    _n: Index,
    x: *const Number,
    _new_x: Bool,
    grad: *mut Number,
    _u: *mut c_void,
) -> Bool {
    unsafe {
        let x = std::slice::from_raw_parts(x, N);
        let g = std::slice::from_raw_parts_mut(grad, N);
        g[0] = -2.0 * (1.0 - x[0]) - 400.0 * x[0] * (x[1] - x[0] * x[0]);
        g[1] = 200.0 * (x[1] - x[0] * x[0]);
        1
    }
}

fn con(x: &[Number]) -> Number {
    x[0] * x[0] + x[1] * x[1] - 1.5
}

unsafe extern "C" fn ev_g(
    _n: Index,
    x: *const Number,
    _new_x: Bool,
    _m: Index,
    gout: *mut Number,
    _u: *mut c_void,
) -> Bool {
    unsafe {
        let x = std::slice::from_raw_parts(x, N);
        *gout = con(x);
        1
    }
}

unsafe extern "C" fn ev_jac_g(
    _n: Index,
    x: *const Number,
    _new_x: Bool,
    _m: Index,
    _nele_jac: Index,
    i_row: *mut Index,
    j_col: *mut Index,
    values: *mut Number,
    _u: *mut c_void,
) -> Bool {
    unsafe {
        if values.is_null() {
            let rows = std::slice::from_raw_parts_mut(i_row, N);
            let cols = std::slice::from_raw_parts_mut(j_col, N);
            for j in 0..N {
                rows[j] = 0;
                cols[j] = j as Index;
            }
            return 1;
        }
        let x = std::slice::from_raw_parts(x, N);
        let v = std::slice::from_raw_parts_mut(values, N);
        v[0] = 2.0 * x[0];
        v[1] = 2.0 * x[1];
        1
    }
}

/// Exact Lagrangian Hessian of the Rosenbrock + circle problem; records the
/// callback state it was called under.
#[allow(clippy::too_many_arguments)]
unsafe extern "C" fn ev_h(
    _n: Index,
    x: *const Number,
    _new_x: Bool,
    obj_factor: Number,
    _m: Index,
    lambda: *const Number,
    _new_lambda: Bool,
    _nele_hess: Index,
    i_row: *mut Index,
    j_col: *mut Index,
    values: *mut Number,
    _u: *mut c_void,
) -> Bool {
    unsafe {
        if values.is_null() {
            let r = std::slice::from_raw_parts_mut(i_row, 3);
            let c = std::slice::from_raw_parts_mut(j_col, 3);
            (r[0], c[0]) = (0, 0);
            (r[1], c[1]) = (1, 0);
            (r[2], c[2]) = (1, 1);
            return 1;
        }
        let x = std::slice::from_raw_parts(x, N);
        let lam = *lambda;
        let v = std::slice::from_raw_parts_mut(values, 3);
        v[0] = obj_factor * (2.0 - 400.0 * (x[1] - 3.0 * x[0] * x[0])) + 2.0 * lam;
        v[1] = obj_factor * (-400.0 * x[0]);
        v[2] = obj_factor * 200.0 + 2.0 * lam;
        let fires = FIRES.with(|f| f.borrow().len());
        HESS_CALLS.with(|h| h.borrow_mut().push((fires, [x[0], x[1]])));
        1
    }
}

#[allow(clippy::too_many_arguments)]
unsafe extern "C" fn on_iter(
    _alg_mod: Index,
    iter: Index,
    _obj: Number,
    _inf_pr: Number,
    _inf_du: Number,
    _mu: Number,
    _d_norm: Number,
    _regu: Number,
    _alpha_du: Number,
    _alpha_pr: Number,
    _ls_trials: Index,
    user_data: *mut c_void,
) -> Bool {
    unsafe {
        let prob = user_data as IpoptProblem;
        let mut f = Fire {
            iter,
            ok: false,
            x: [0.0; N],
            z_l: [0.0; N],
            z_u: [0.0; N],
            g: [0.0; M],
            lambda: [0.0; M],
        };
        f.ok = GetIpoptCurrentIterate(
            prob,
            0,
            N as Index,
            f.x.as_mut_ptr(),
            f.z_l.as_mut_ptr(),
            f.z_u.as_mut_ptr(),
            M as Index,
            f.g.as_mut_ptr(),
            f.lambda.as_mut_ptr(),
        ) != 0;
        FIRES.with(|v| v.borrow_mut().push(f));
        let stop = STOP_AT.with(|s| *s.borrow());
        if stop == Some(iter) { 0 } else { 1 }
    }
}

struct Outcome {
    status: i32,
    x: [Number; N],
    mult_g: [Number; M],
    z_l: [Number; N],
    z_u: [Number; N],
    fires: Vec<Fire>,
}

fn solve(x_l: [Number; N], x_u: [Number; N], stop_at: Option<Index>) -> Outcome {
    solve_with(x_l, x_u, stop_at, false)
}

fn solve_with(x_l: [Number; N], x_u: [Number; N], stop_at: Option<Index>, exact: bool) -> Outcome {
    FIRES.with(|v| v.borrow_mut().clear());
    HESS_CALLS.with(|v| v.borrow_mut().clear());
    STOP_AT.with(|s| *s.borrow_mut() = stop_at);
    unsafe {
        let g_l = [-2.0e19 as Number];
        let g_u = [0.0 as Number];
        let prob = CreateIpoptProblem(
            N as Index,
            x_l.as_ptr(),
            x_u.as_ptr(),
            M as Index,
            g_l.as_ptr(),
            g_u.as_ptr(),
            N as Index,
            if exact { 3 } else { 0 },
            0,
            Some(ev_f),
            Some(ev_g),
            Some(ev_grad_f),
            Some(ev_jac_g),
            if exact { Some(ev_h) } else { None },
        );
        assert!(!prob.is_null(), "CreateIpoptProblem returned NULL");
        let s = |k: &str, v: &str| {
            let k = CString::new(k).unwrap();
            let v = CString::new(v).unwrap();
            assert_ne!(
                AddIpoptStrOption(prob, k.as_ptr() as *mut _, v.as_ptr() as *mut _),
                0
            );
        };
        s("algorithm", "active-set-sqp");
        if !exact {
            s("hessian_approximation", "limited-memory");
        }
        let k = CString::new("print_level").unwrap();
        assert_ne!(AddIpoptIntOption(prob, k.as_ptr() as *mut _, 0), 0);
        assert_ne!(SetIntermediateCallback(prob, Some(on_iter)), 0);

        let mut o = Outcome {
            status: 0,
            x: [-1.0, 2.0],
            mult_g: [0.0; M],
            z_l: [0.0; N],
            z_u: [0.0; N],
            fires: Vec::new(),
        };
        let mut g = [0.0 as Number; M];
        let mut obj = 0.0 as Number;
        let st = IpoptSolve(
            prob,
            o.x.as_mut_ptr(),
            g.as_mut_ptr(),
            &mut obj,
            o.mult_g.as_mut_ptr(),
            o.z_l.as_mut_ptr(),
            o.z_u.as_mut_ptr(),
            prob as *mut c_void,
        );
        FreeIpoptProblem(prob);
        o.status = st as i32;
        o.fires = FIRES.with(|v| v.borrow().clone());
        o
    }
}

/// gh#1015: the callback fires on the active-set SQP arm, once per
/// iteration, and `GetIpoptCurrentIterate` answers from the SQP iterate.
/// `x0 <= 0.5` binds at the solution, so `z_U` is non-zero and the
/// signed-`λ_x` split is exercised.
#[test]
fn the_callback_fires_on_the_sqp_arm_with_a_live_iterate() {
    let o = solve([-5.0; N], [0.5, 5.0], None);
    assert_eq!(o.status, 0, "expected Solve_Succeeded");
    assert!(o.fires.len() >= 2, "callback fired {} times", o.fires.len());
    for (k, f) in o.fires.iter().enumerate() {
        assert_eq!(f.iter, k as Index, "one fire per iteration, in order");
        assert!(f.ok, "GetIpoptCurrentIterate refused at iteration {k}");
        assert!(
            (f.g[0] - con(&f.x)).abs() < 1e-9,
            "g disagrees with g(x) at iteration {k}"
        );
    }
    assert_ne!(
        o.fires[0].x,
        o.fires.last().unwrap().x,
        "iterate never moved"
    );
    // The last fire is the converged iterate: it must be what IpoptSolve returns.
    let last = o.fires.last().unwrap();
    for i in 0..N {
        assert!((last.x[i] - o.x[i]).abs() < 1e-9, "x[{i}]");
        assert!((last.z_l[i] - o.z_l[i]).abs() < 1e-9, "z_l[{i}]");
        assert!((last.z_u[i] - o.z_u[i]).abs() < 1e-9, "z_u[{i}]");
    }
    assert!((last.lambda[0] - o.mult_g[0]).abs() < 1e-9, "lambda");
    assert!(
        o.z_u[0] > 1e-3,
        "the x0 upper bound should bind: {:?}",
        o.z_u
    );
    assert!(last.z_u[0] > 1e-3 && last.z_l[0].abs() < 1e-9);
}

/// A fixed variable sits ahead of a free one: the reported `x` is full-space
/// and carries the fixed value, not a neighbour's (the gh#450 shape).
#[test]
fn a_fixed_variable_is_reported_in_full_space() {
    let o = solve([0.3, -5.0], [0.3, 5.0], None);
    assert_eq!(o.status, 0, "expected Solve_Succeeded");
    assert!(!o.fires.is_empty());
    for f in &o.fires {
        assert!(f.ok);
        assert_eq!(f.x[0], 0.3, "fixed variable moved or was mis-indexed");
        assert!((f.g[0] - con(&f.x)).abs() < 1e-9);
    }
}

/// Returning `false` from the callback ends the solve with
/// `User_Requested_Stop`, after exactly that many fires.
#[test]
fn false_from_the_callback_stops_the_sqp_solve() {
    let full = solve([-5.0; N], [0.5, 5.0], None);
    assert!(
        full.fires.len() > 2,
        "need a solve longer than 2 iterations"
    );
    let o = solve([-5.0; N], [0.5, 5.0], Some(1));
    assert_eq!(o.fires.len(), 2, "stopped at iteration 1");
    assert_eq!(o.status, ApplicationReturnStatus::UserRequestedStop as i32);
}

/// Outside the callback window there is no live iterate.
#[test]
fn no_iterate_outside_the_callback() {
    let _ = solve([-5.0; N], [0.5, 5.0], None);
    assert!(!pounce_algorithm::intermediate::is_active());
}

/// The point of the feature (gh#1015): a caller builds the Lagrangian Hessian
/// from the iterate it read in the callback. So the callback for iteration `k`
/// must have fired BEFORE `eval_h` is asked for iteration `k` — otherwise the
/// stashed duals are a step stale. Every `eval_h` at the iterate the most
/// recent fire reported must see that fire already recorded.
#[test]
fn the_callback_fires_before_the_hessian_of_the_same_iterate() {
    let o = solve_with([-5.0; N], [0.5, 5.0], None, true);
    assert_eq!(o.status, 0, "expected Solve_Succeeded");
    let calls = HESS_CALLS.with(|h| h.borrow().clone());
    assert!(!calls.is_empty(), "eval_h never called");
    for (fires_before, x) in calls {
        assert!(fires_before > 0, "eval_h ran before the first callback");
        let last = &o.fires[fires_before - 1];
        assert_eq!(
            last.x, x,
            "eval_h at {x:?} but the latest callback reported {:?}",
            last.x
        );
    }
}
