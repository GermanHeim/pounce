//! `GetIpoptCurrentIterate` at the last intermediate-callback fire reads
//! the solution `IpoptSolve` writes back, on both arms, under every layer
//! between the algorithm and the user's problem: gradient-based scaling
//! with an objective factor, per-variable user scaling, a fixed variable.
//!
//! Upstream (Ipopt 3.14 `TNLP::get_curr_iterate`) reinserts fixed
//! variables (`TNLPAdapter::ResortX`) and recovers their bound multipliers
//! (`ResortBoundMultipliers`, `z_L − z_U = ∇f + Jᵀλ` under
//! `make_parameter`). The interior-point arm here left them at `0`.
//!
//! Model: `min K·[(x0-3)² + (x1-3)² + (x2-1)² + (x3-4)²]`
//! s.t. `g0 = x0 + x1 + x3 ≤ 3.5`, `g1 = x0 − x2 = 0`,
//! `g2 = x1 + x3 ≤ 10` (implied by the box: `x1 ≤ 1`, `x3 = 1`),
//! `x0` free, `x1 ∈ [-10, 1]` (upper active), `x2 ≥ 0`, `x3 = 1` fixed.

use pounce_cinterface::*;
use std::cell::{Cell, RefCell};
use std::ffi::{CString, c_void};

const N: usize = 4;
const M: usize = 3;
const NNZ: usize = 7;

#[derive(Clone, Debug, Default)]
struct It {
    ok: bool,
    x: [Number; N],
    z_l: [Number; N],
    z_u: [Number; N],
    g: [Number; M],
    lambda: [Number; M],
}

thread_local! {
    static FIRES: RefCell<Vec<It>> = const { RefCell::new(Vec::new()) };
    static K: Cell<Number> = const { Cell::new(1.0) };
}

fn k() -> Number {
    K.with(|k| k.get())
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
        *obj = k()
            * ((x[0] - 3.0).powi(2)
                + (x[1] - 3.0).powi(2)
                + (x[2] - 1.0).powi(2)
                + (x[3] - 4.0).powi(2));
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
        g[0] = 2.0 * k() * (x[0] - 3.0);
        g[1] = 2.0 * k() * (x[1] - 3.0);
        g[2] = 2.0 * k() * (x[2] - 1.0);
        g[3] = 2.0 * k() * (x[3] - 4.0);
        1
    }
}

unsafe extern "C" fn ev_g(
    _n: Index,
    x: *const Number,
    _new_x: Bool,
    _m: Index,
    g: *mut Number,
    _u: *mut c_void,
) -> Bool {
    unsafe {
        let x = std::slice::from_raw_parts(x, N);
        let g = std::slice::from_raw_parts_mut(g, M);
        g[0] = x[0] + x[1] + x[3];
        g[1] = x[0] - x[2];
        g[2] = x[1] + x[3];
        1
    }
}

const JAC_ROW: [Index; NNZ] = [0, 0, 0, 1, 1, 2, 2];
const JAC_COL: [Index; NNZ] = [0, 1, 3, 0, 2, 1, 3];
const JAC_VAL: [Number; NNZ] = [1.0, 1.0, 1.0, 1.0, -1.0, 1.0, 1.0];

unsafe extern "C" fn ev_jac_g(
    _n: Index,
    _x: *const Number,
    _new_x: Bool,
    _m: Index,
    _nele: Index,
    i_row: *mut Index,
    j_col: *mut Index,
    values: *mut Number,
    _u: *mut c_void,
) -> Bool {
    unsafe {
        if values.is_null() {
            std::slice::from_raw_parts_mut(i_row, NNZ).copy_from_slice(&JAC_ROW);
            std::slice::from_raw_parts_mut(j_col, NNZ).copy_from_slice(&JAC_COL);
        } else {
            std::slice::from_raw_parts_mut(values, NNZ).copy_from_slice(&JAC_VAL);
        }
        1
    }
}

#[allow(clippy::too_many_arguments)]
unsafe extern "C" fn ev_h(
    _n: Index,
    _x: *const Number,
    _new_x: Bool,
    of: Number,
    _m: Index,
    _lambda: *const Number,
    _new_lambda: Bool,
    _nele: Index,
    i_row: *mut Index,
    j_col: *mut Index,
    values: *mut Number,
    _u: *mut c_void,
) -> Bool {
    unsafe {
        if values.is_null() {
            std::slice::from_raw_parts_mut(i_row, N).copy_from_slice(&[0, 1, 2, 3]);
            std::slice::from_raw_parts_mut(j_col, N).copy_from_slice(&[0, 1, 2, 3]);
        } else {
            std::slice::from_raw_parts_mut(values, N).copy_from_slice(&[2.0 * k() * of; N]);
        }
        1
    }
}

#[allow(clippy::too_many_arguments)]
unsafe extern "C" fn on_iter(
    _alg_mod: Index,
    _iter: Index,
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
        let mut it = It::default();
        it.ok = GetIpoptCurrentIterate(
            user_data as IpoptProblem,
            0,
            N as Index,
            it.x.as_mut_ptr(),
            it.z_l.as_mut_ptr(),
            it.z_u.as_mut_ptr(),
            M as Index,
            it.g.as_mut_ptr(),
            it.lambda.as_mut_ptr(),
        ) != 0;
        FIRES.with(|f| f.borrow_mut().push(it));
        1
    }
}

#[derive(Clone, Copy, Debug)]
enum Cfg {
    /// `gradient-based` (engaged: `K = 1000` puts `‖∇f‖∞` far above
    /// `nlp_scaling_max_gradient`) times `obj_scaling_factor = 3`.
    GradientBased,
    /// `user-scaling`: obj 3, x `[2, 0.5, 4, 3]`, g `[0.01, 5, 1]`.
    UserX,
}

/// `(IpoptSolve's output, every fire's iterate)`.
fn solve(algorithm: &str, cfg: Cfg) -> (It, Vec<It>) {
    FIRES.with(|f| f.borrow_mut().clear());
    K.with(|k| {
        k.set(match cfg {
            Cfg::GradientBased => 1000.0,
            _ => 1.0,
        })
    });
    let x_l = [-2e19, -10.0, 0.0, 1.0];
    let x_u = [2e19, 1.0, 2e19, 1.0];
    let g_l = [-2e19, 0.0, -2e19];
    let g_u = [3.5, 0.0, 10.0];
    unsafe {
        let p = CreateIpoptProblem(
            N as Index,
            x_l.as_ptr(),
            x_u.as_ptr(),
            M as Index,
            g_l.as_ptr(),
            g_u.as_ptr(),
            NNZ as Index,
            N as Index,
            0,
            Some(ev_f),
            Some(ev_g),
            Some(ev_grad_f),
            Some(ev_jac_g),
            Some(ev_h),
        );
        assert!(!p.is_null());
        let opt = |k: &str, v: &str| {
            let (k, v) = (CString::new(k).unwrap(), CString::new(v).unwrap());
            assert_ne!(AddIpoptStrOption(p, k.as_ptr(), v.as_ptr()), 0);
        };
        opt("algorithm", algorithm);
        let kp = CString::new("print_level").unwrap();
        assert_ne!(AddIpoptIntOption(p, kp.as_ptr(), 0), 0);
        match cfg {
            Cfg::GradientBased => {
                opt("nlp_scaling_method", "gradient-based");
                let k = CString::new("obj_scaling_factor").unwrap();
                assert_ne!(AddIpoptNumOption(p, k.as_ptr(), 3.0), 0);
            }
            Cfg::UserX => {
                opt("nlp_scaling_method", "user-scaling");
                let xs = [2.0, 0.5, 4.0, 3.0];
                let gs = [0.01, 5.0, 1.0];
                assert_ne!(SetIpoptProblemScaling(p, 3.0, xs.as_ptr(), gs.as_ptr()), 0);
            }
        }
        assert_ne!(SetIntermediateCallback(p, Some(on_iter)), 0);
        let mut out = It {
            ok: true,
            x: [0.0, 0.0, 0.5, 1.0],
            ..It::default()
        };
        let mut obj = 0.0;
        let st = IpoptSolve(
            p,
            out.x.as_mut_ptr(),
            out.g.as_mut_ptr(),
            &mut obj,
            out.lambda.as_mut_ptr(),
            out.z_l.as_mut_ptr(),
            out.z_u.as_mut_ptr(),
            p as *mut c_void,
        );
        FreeIpoptProblem(p);
        assert_eq!(st, 0, "{algorithm} {cfg:?}: expected Solve_Succeeded");
        (out, FIRES.with(|f| f.borrow().clone()))
    }
}

fn close(what: &str, got: &[Number], want: &[Number]) {
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!(
            (g - w).abs() <= 1e-9 * w.abs() + 1e-300,
            "{what}[{i}]: callback {g:e}, IpoptSolve {w:e} ({got:?} vs {want:?})"
        );
    }
}

fn check(algorithm: &str, cfg: Cfg) {
    let (out, fires) = solve(algorithm, cfg);
    assert!(
        fires.len() > 1,
        "{algorithm} {cfg:?}: {} fires",
        fires.len()
    );
    assert!(
        fires.iter().all(|f| f.ok),
        "{algorithm} {cfg:?}: GetIpoptCurrentIterate refused"
    );
    // Informative: the fixed x3 carries a multiplier (`2K(1-4) + λ0 < 0`,
    // so z_U), and x1's upper bound binds.
    assert!(out.z_u[3] > 1e-3, "{algorithm} {cfg:?}: {out:?}");
    assert!(out.z_u[1] > 1e-3, "{algorithm} {cfg:?}: {out:?}");
    let last = fires.last().unwrap();
    close("x", &last.x, &out.x);
    close("z_l", &last.z_l, &out.z_l);
    close("z_u", &last.z_u, &out.z_u);
    close("g", &last.g, &out.g);
    close("lambda", &last.lambda, &out.lambda);
}

#[test]
fn ipm_iterate_is_the_solution_under_gradient_scaling_and_a_fixed_variable() {
    check("interior-point", Cfg::GradientBased);
}

#[test]
fn ipm_iterate_is_the_solution_under_user_x_scaling_and_a_fixed_variable() {
    check("interior-point", Cfg::UserX);
}

#[test]
fn sqp_iterate_is_the_solution_under_gradient_scaling_and_a_fixed_variable() {
    check("active-set-sqp", Cfg::GradientBased);
}

#[test]
fn sqp_iterate_is_the_solution_under_user_x_scaling_and_a_fixed_variable() {
    check("active-set-sqp", Cfg::UserX);
}
