//! gh#698 (Observation 4) — under `hessian_approximation = limited-memory`
//! the user's `eval_h` must never be invoked.
//!
//! The zero-W block that `LeastSquareMults` and the default iterate
//! initializer hand the augmented-system solver exists only to pin the W
//! triplet structure; they pass `w_factor = 0.0` and never read the
//! values. Building it from `curr_exact_hessian()` — an unmemoized
//! `eval_h` — called into the TNLP once per `calculate_y_eq`, in the one
//! mode whose premise is that the caller is not supplying a Hessian.
//!
//! The model declares a genuine Hessian pattern and a working `eval_h`,
//! so the only thing separating the two runs is the option: the exact
//! run must use the callback, the limited-memory run must not touch it.

use pounce_cinterface::*;
use std::ffi::{CString, c_void};
use std::sync::atomic::{AtomicUsize, Ordering};

const N: usize = 2;
const M: usize = 1;
/// Lower triangle of a dense 2x2 Hessian.
const NELE_HESS: usize = 3;

static H_CALLS: AtomicUsize = AtomicUsize::new(0);

// f(x) = (1 - x0)^2 + 100 (x1 - x0^2)^2
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

// g(x) = x0 + x1 == 2
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
        *std::slice::from_raw_parts_mut(gout, M).get_unchecked_mut(0) = x[0] + x[1];
        1
    }
}

unsafe extern "C" fn ev_jac_g(
    _n: Index,
    _x: *const Number,
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
            rows.copy_from_slice(&[0, 0]);
            cols.copy_from_slice(&[0, 1]);
            return 1;
        }
        std::slice::from_raw_parts_mut(values, N).copy_from_slice(&[1.0, 1.0]);
        1
    }
}

#[allow(clippy::too_many_arguments)]
unsafe extern "C" fn ev_h(
    _n: Index,
    x: *const Number,
    _new_x: Bool,
    obj_factor: Number,
    _m: Index,
    _lambda: *const Number,
    _new_lambda: Bool,
    _nele_hess: Index,
    i_row: *mut Index,
    j_col: *mut Index,
    values: *mut Number,
    _u: *mut c_void,
) -> Bool {
    unsafe {
        if values.is_null() {
            let rows = std::slice::from_raw_parts_mut(i_row, NELE_HESS);
            let cols = std::slice::from_raw_parts_mut(j_col, NELE_HESS);
            rows.copy_from_slice(&[0, 1, 1]);
            cols.copy_from_slice(&[0, 0, 1]);
            return 1;
        }
        // Structure requests are bookkeeping; only a values request is a
        // real evaluation of the user's Hessian.
        H_CALLS.fetch_add(1, Ordering::SeqCst);
        let x = std::slice::from_raw_parts(x, N);
        let v = std::slice::from_raw_parts_mut(values, NELE_HESS);
        v[0] = obj_factor * (2.0 - 400.0 * (x[1] - 3.0 * x[0] * x[0]));
        v[1] = obj_factor * (-400.0 * x[0]);
        v[2] = obj_factor * 200.0;
        1
    }
}

/// Solve, returning `(status, eval_h value-request count)`.
fn solve(limited_memory: bool) -> (i32, usize) {
    unsafe {
        H_CALLS.store(0, Ordering::SeqCst);
        let x_l = [-5.0 as Number; N];
        let x_u = [5.0 as Number; N];
        let g_l = [2.0 as Number];
        let g_u = [2.0 as Number];

        let prob = CreateIpoptProblem(
            N as Index,
            x_l.as_ptr(),
            x_u.as_ptr(),
            M as Index,
            g_l.as_ptr(),
            g_u.as_ptr(),
            N as Index,
            NELE_HESS as Index,
            0,
            Some(ev_f),
            Some(ev_g),
            Some(ev_grad_f),
            Some(ev_jac_g),
            Some(ev_h),
        );
        assert!(!prob.is_null());

        if limited_memory {
            let key = CString::new("hessian_approximation").unwrap();
            let val = CString::new("limited-memory").unwrap();
            assert_ne!(
                AddIpoptStrOption(prob, key.as_ptr() as *mut _, val.as_ptr() as *mut _),
                0
            );
        }
        let key = CString::new("print_level").unwrap();
        assert_ne!(AddIpoptIntOption(prob, key.as_ptr() as *mut _, 0), 0);

        let mut x = [0.0 as Number, 0.0];
        let mut g = [0.0 as Number; M];
        let mut obj = 0.0 as Number;
        let mut mult_g = [0.0 as Number; M];
        let mut z_l = [0.0 as Number; N];
        let mut z_u = [0.0 as Number; N];
        let status = IpoptSolve(
            prob,
            x.as_mut_ptr(),
            g.as_mut_ptr(),
            &mut obj,
            mult_g.as_mut_ptr(),
            z_l.as_mut_ptr(),
            z_u.as_mut_ptr(),
            std::ptr::null_mut(),
        );
        FreeIpoptProblem(prob);
        (status as i32, H_CALLS.load(Ordering::SeqCst))
    }
}

/// Both directions live in one test on purpose: `H_CALLS` is process
/// global (a C callback has nowhere else to put it), and cargo runs test
/// functions concurrently in one process, so two tests sharing the
/// counter would race.
#[test]
fn limited_memory_never_evaluates_the_user_hessian() {
    // The exact path first — it is the control. If this stopped calling
    // eval_h the limited-memory assertion below would pass for the wrong
    // reason.
    let (status, exact_calls) = solve(false);
    assert_eq!(status, 0, "exact-Hessian solve did not succeed");
    assert!(exact_calls > 0, "exact-Hessian solve never called eval_h");

    let (status, lbfgs_calls) = solve(true);
    assert_eq!(status, 0, "limited-memory solve did not succeed");
    assert_eq!(
        lbfgs_calls, 0,
        "eval_h was called {lbfgs_calls} times under \
         hessian_approximation=limited-memory (exact path: {exact_calls})"
    );
}
