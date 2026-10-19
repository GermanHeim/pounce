//! `SetIpoptProblemScaling` with per-variable factors, the intermediate
//! callback, and `GetIpoptCurrentIterate`, on both arms.
//!
//! Non-unit `x` factors put a `ScalingTnlp` (`x̃ = d ⊙ x`) under the NLP.
//! It did not forward `intermediate_callback`, so the C callback never
//! fired; and neither arm's `GetIpoptCurrentIterate` undid the
//! substitution, so a callback that did run would have read `x̃` / `z̃`.
//! The last fire is the converged iterate, so what it reads must equal what
//! `IpoptSolve` writes back.
//!
//! Model: `min (x0-3)² + (x1-3)² + (x2-1)²` s.t. `g0 = x0 + x1 ≤ 2.5`,
//! `g1 = x0 − x2 = 0`, `x0` free, `x1 ∈ [-10, 1]` (upper active),
//! `x2 ≥ 0`; scaling obj 3, x `[2, 0.5, 4]`, g `[0.01, 5]`.

use pounce_cinterface::*;
use std::cell::RefCell;
use std::ffi::{CString, c_void};

const N: usize = 3;
const M: usize = 2;

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
        *obj = (x[0] - 3.0).powi(2) + (x[1] - 3.0).powi(2) + (x[2] - 1.0).powi(2);
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
        g[0] = 2.0 * (x[0] - 3.0);
        g[1] = 2.0 * (x[1] - 3.0);
        g[2] = 2.0 * (x[2] - 1.0);
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
        g[0] = x[0] + x[1];
        g[1] = x[0] - x[2];
        1
    }
}

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
            std::slice::from_raw_parts_mut(i_row, 4).copy_from_slice(&[0, 0, 1, 1]);
            std::slice::from_raw_parts_mut(j_col, 4).copy_from_slice(&[0, 1, 0, 2]);
        } else {
            std::slice::from_raw_parts_mut(values, 4).copy_from_slice(&[1.0, 1.0, 1.0, -1.0]);
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
            std::slice::from_raw_parts_mut(i_row, 3).copy_from_slice(&[0, 1, 2]);
            std::slice::from_raw_parts_mut(j_col, 3).copy_from_slice(&[0, 1, 2]);
        } else {
            std::slice::from_raw_parts_mut(values, 3).copy_from_slice(&[2.0 * of; 3]);
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

/// `(IpoptSolve's output, every fire's iterate)`.
fn solve(algorithm: &str) -> (It, Vec<It>) {
    FIRES.with(|f| f.borrow_mut().clear());
    let x_l = [-2e19, -10.0, 0.0];
    let x_u = [2e19, 1.0, 2e19];
    let g_l = [-2e19, 0.0];
    let g_u = [2.5, 0.0];
    unsafe {
        let p = CreateIpoptProblem(
            N as Index,
            x_l.as_ptr(),
            x_u.as_ptr(),
            M as Index,
            g_l.as_ptr(),
            g_u.as_ptr(),
            4,
            3,
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
        opt("nlp_scaling_method", "user-scaling");
        let k = CString::new("print_level").unwrap();
        assert_ne!(AddIpoptIntOption(p, k.as_ptr(), 0), 0);
        let xs = [2.0, 0.5, 4.0];
        let gs = [0.01, 5.0];
        assert_ne!(SetIpoptProblemScaling(p, 3.0, xs.as_ptr(), gs.as_ptr()), 0);
        assert_ne!(SetIntermediateCallback(p, Some(on_iter)), 0);
        let mut out = It {
            ok: true,
            x: [0.0, 0.0, 0.5],
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
        assert_eq!(st, 0, "{algorithm}: expected Solve_Succeeded");
        (out, FIRES.with(|f| f.borrow().clone()))
    }
}

/// Relative, with only a denormal-scale floor: the last fire IS the final
/// iterate, so the two agree to rounding. An absolute `1 + |w|` floor would
/// hide the inactive bound multipliers (`~1e-10` on the IPM arm), which are
/// the only `z_L` entries this model has and still carry the factor `d`.
fn close(what: &str, got: &[Number], want: &[Number], tol: Number) {
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!(
            (g - w).abs() <= tol * w.abs() + 1e-300,
            "{what}[{i}]: callback {g:e}, IpoptSolve {w:e} ({got:?} vs {want:?})"
        );
    }
}

fn check(algorithm: &str, tol: Number) {
    let (out, fires) = solve(algorithm);
    assert!(
        fires.len() > 1,
        "{algorithm}: callback fired {} times",
        fires.len()
    );
    assert!(fires.iter().all(|f| f.ok), "GetIpoptCurrentIterate refused");
    // Informative: x1's upper bound binds and no variable sits at 0.
    assert!(out.z_u[1] > 1e-3, "{out:?}");
    assert!(out.x.iter().all(|v| v.abs() > 1e-3), "{out:?}");
    let last = fires.last().unwrap();
    close("x", &last.x, &out.x, tol);
    close("z_l", &last.z_l, &out.z_l, tol);
    close("z_u", &last.z_u, &out.z_u, tol);
    close("g", &last.g, &out.g, tol);
    close("lambda", &last.lambda, &out.lambda, tol);
}

#[test]
fn sqp_arm_callback_reads_the_unscaled_iterate() {
    check("active-set-sqp", 1e-9);
}

/// The interior-point arm's last fire is the iterate `finalize_solution`
/// reports (no fixed variables, `honor_original_bounds` off, so finalize
/// adds nothing of its own). The objective factor 3 is what pins the
/// multipliers: `GetIpoptCurrentIterate` used `pack_*_for_user` there,
/// which leaves `obj_scale_factor` in, and read `z` and `λ` 3× too large.
#[test]
fn interior_point_arm_callback_reads_the_unscaled_iterate() {
    check("interior-point", 1e-9);
}
