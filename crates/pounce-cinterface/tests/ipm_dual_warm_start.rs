//! The C API's multiplier buffers are in/out, as upstream Ipopt's are.
//!
//! `IpoptSolve(problem, x, g, obj, mult_g, mult_x_L, mult_x_U, ud)`:
//! under `warm_start_init_point=yes` upstream's `StdInterfaceTNLP::
//! get_starting_point` copies `start_lam` / `start_z_L` / `start_z_U` —
//! the caller's `mult_*` buffers — into the starting duals. The bridge
//! used to copy only `x`, so a C caller asking for a dual warm start got
//! a primal-only one, silently: the solve still converged, from the
//! wrong place.
//!
//! The round-trip test pins the contract with a solve that takes no
//! steps: re-solve from the returned `x` / `mult_*` with the warm start
//! on, recentering off, the pushes made negligible and `max_iter=0`, and
//! the multipliers written back must be the ones read in. Under the bug
//! they come back as the warm-start initializer's fills for an unseeded
//! block instead. The NULL case pins the other half: NULL means "not
//! supplied", not a crash and not an error.

use pounce_cinterface::*;
use std::ffi::{CString, c_void};

const N: usize = 4;
const M: usize = 2;
/// `Maximum_Iterations_Exceeded`, what a `max_iter=0` solve returns.
const MAX_ITER_EXCEEDED: Index = -1;

unsafe extern "C" fn ev_f(
    _n: Index,
    x: *const Number,
    _new_x: Bool,
    obj: *mut Number,
    _u: *mut c_void,
) -> Bool {
    unsafe {
        let x = std::slice::from_raw_parts(x, N);
        *obj = x[0] * x[3] * (x[0] + x[1] + x[2]) + x[2];
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
        g[0] = x[3] * (2.0 * x[0] + x[1] + x[2]);
        g[1] = x[0] * x[3];
        g[2] = x[0] * x[3] + 1.0;
        g[3] = x[0] * (x[0] + x[1] + x[2]);
        1
    }
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
        let g = std::slice::from_raw_parts_mut(gout, M);
        g[0] = x[0] * x[1] * x[2] * x[3];
        g[1] = x[0] * x[0] + x[1] * x[1] + x[2] * x[2] + x[3] * x[3];
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
            let rows = std::slice::from_raw_parts_mut(i_row, M * N);
            let cols = std::slice::from_raw_parts_mut(j_col, M * N);
            let mut k = 0;
            for i in 0..M {
                for j in 0..N {
                    rows[k] = i as Index;
                    cols[k] = j as Index;
                    k += 1;
                }
            }
            return 1;
        }
        let x = std::slice::from_raw_parts(x, N);
        let v = std::slice::from_raw_parts_mut(values, M * N);
        v[0] = x[1] * x[2] * x[3];
        v[1] = x[0] * x[2] * x[3];
        v[2] = x[0] * x[1] * x[3];
        v[3] = x[0] * x[1] * x[2];
        v[4] = 2.0 * x[0];
        v[5] = 2.0 * x[1];
        v[6] = 2.0 * x[2];
        v[7] = 2.0 * x[3];
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
    lambda: *const Number,
    _new_lambda: Bool,
    _nele_hess: Index,
    i_row: *mut Index,
    j_col: *mut Index,
    values: *mut Number,
    _u: *mut c_void,
) -> Bool {
    const NNZ: usize = N * (N + 1) / 2;
    unsafe {
        if values.is_null() {
            let rows = std::slice::from_raw_parts_mut(i_row, NNZ);
            let cols = std::slice::from_raw_parts_mut(j_col, NNZ);
            let mut k = 0;
            for i in 0..N {
                for j in 0..=i {
                    rows[k] = i as Index;
                    cols[k] = j as Index;
                    k += 1;
                }
            }
            return 1;
        }
        let x = std::slice::from_raw_parts(x, N);
        let lam = std::slice::from_raw_parts(lambda, M);
        let v = std::slice::from_raw_parts_mut(values, NNZ);
        v[0] = obj_factor * (2.0 * x[3]);
        v[1] = obj_factor * x[3];
        v[2] = 0.0;
        v[3] = obj_factor * x[3];
        v[4] = 0.0;
        v[5] = 0.0;
        v[6] = obj_factor * (2.0 * x[0] + x[1] + x[2]);
        v[7] = obj_factor * x[0];
        v[8] = obj_factor * x[0];
        v[9] = 0.0;
        v[1] += lam[0] * (x[2] * x[3]);
        v[3] += lam[0] * (x[1] * x[3]);
        v[4] += lam[0] * (x[0] * x[3]);
        v[6] += lam[0] * (x[1] * x[2]);
        v[7] += lam[0] * (x[0] * x[2]);
        v[8] += lam[0] * (x[0] * x[1]);
        v[0] += lam[1] * 2.0;
        v[2] += lam[1] * 2.0;
        v[5] += lam[1] * 2.0;
        v[9] += lam[1] * 2.0;
        1
    }
}

fn str_opt(p: IpoptProblem, k: &str, v: &str) {
    let (k, v) = (CString::new(k).unwrap(), CString::new(v).unwrap());
    assert_eq!(unsafe { AddIpoptStrOption(p, k.as_ptr(), v.as_ptr()) }, 1);
}

fn int_opt(p: IpoptProblem, k: &str, v: Index) {
    let k = CString::new(k).unwrap();
    assert_eq!(unsafe { AddIpoptIntOption(p, k.as_ptr(), v) }, 1);
}

fn num_opt(p: IpoptProblem, k: &str, v: Number) {
    let k = CString::new(k).unwrap();
    assert_eq!(unsafe { AddIpoptNumOption(p, k.as_ptr(), v) }, 1);
}

/// HS071 on the default (interior-point) path.
fn make_problem() -> IpoptProblem {
    let x_l = [1.0; N];
    let x_u = [5.0; N];
    let g_l = [25.0, 40.0];
    let g_u = [2.0e19, 40.0];
    let p = unsafe {
        CreateIpoptProblem(
            N as Index,
            x_l.as_ptr(),
            x_u.as_ptr(),
            M as Index,
            g_l.as_ptr(),
            g_u.as_ptr(),
            (M * N) as Index,
            (N * (N + 1) / 2) as Index,
            0,
            Some(ev_f),
            Some(ev_g),
            Some(ev_grad_f),
            Some(ev_jac_g),
            Some(ev_h),
        )
    };
    assert!(!p.is_null(), "CreateIpoptProblem failed");
    int_opt(p, "print_level", 0);
    p
}

/// A warm start that takes no step and moves nothing it is handed: the
/// initializer's pushes and floors are made negligible, so whatever duals
/// the TNLP supplied are the duals `finalize_solution` reports.
fn make_zero_step_warm_problem() -> IpoptProblem {
    let p = make_problem();
    str_opt(p, "warm_start_init_point", "yes");
    str_opt(p, "warm_start_recentering", "none");
    int_opt(p, "max_iter", 0);
    for k in [
        "warm_start_bound_push",
        "warm_start_bound_frac",
        "warm_start_slack_bound_push",
        "warm_start_slack_bound_frac",
        "warm_start_mult_bound_push",
    ] {
        num_opt(p, k, 1e-300);
    }
    p
}

struct Out {
    status: Index,
    x: [Number; N],
    lam: [Number; M],
    zl: [Number; N],
    zu: [Number; N],
}

/// Solve through the C ABI with all three multiplier buffers non-NULL.
fn solve(
    p: IpoptProblem,
    x0: [Number; N],
    lam0: [Number; M],
    z0: ([Number; N], [Number; N]),
) -> Out {
    let mut o = Out {
        status: -1,
        x: x0,
        lam: lam0,
        zl: z0.0,
        zu: z0.1,
    };
    let mut obj = 0.0;
    o.status = unsafe {
        IpoptSolve(
            p,
            o.x.as_mut_ptr(),
            std::ptr::null_mut(),
            &mut obj,
            o.lam.as_mut_ptr(),
            o.zl.as_mut_ptr(),
            o.zu.as_mut_ptr(),
            std::ptr::null_mut(),
        )
    };
    unsafe { FreeIpoptProblem(p) };
    o
}

/// Converged HS071 primal-dual solution, through the C API.
fn cold_solve() -> Out {
    let o = solve(
        make_problem(),
        [1.0, 5.0, 5.0, 1.0],
        [0.0; M],
        ([0.0; N], [0.0; N]),
    );
    assert_eq!(o.status, 0, "cold HS071 solve must succeed");
    // Ipopt sign convention (L = f + lambda'g, z >= 0): HS071's `>= 25`
    // row is active at its lower bound, so its lambda is negative; x0
    // sits on its lower bound, so z_L[0] > 0.
    assert!(o.lam[0] < 0.0, "lambda = {:?}", o.lam);
    assert!(o.zl[0] > 0.1, "z_L = {:?}", o.zl);
    assert!(o.zl.iter().chain(&o.zu).all(|&z| z >= 0.0));
    o
}

#[test]
fn hs071_multiplier_buffers_seed_the_warm_start_and_round_trip() {
    let cold = cold_solve();
    let warm = solve(
        make_zero_step_warm_problem(),
        cold.x,
        cold.lam,
        (cold.zl, cold.zu),
    );
    assert_eq!(warm.status, MAX_ITER_EXCEEDED, "max_iter=0 takes no step");
    assert_eq!(warm.x, cold.x, "a zero-step warm solve must not move x");
    assert_eq!(
        warm.lam, cold.lam,
        "mult_g must round-trip: it is the starting lambda"
    );
    assert_eq!(
        warm.zl, cold.zl,
        "mult_x_L must round-trip: it is the starting z_L"
    );
    assert_eq!(
        warm.zu, cold.zu,
        "mult_x_U must round-trip: it is the starting z_U"
    );
}

#[test]
fn hs071_null_multiplier_buffers_mean_a_primal_only_warm_start() {
    let cold = cold_solve();
    let p = make_zero_step_warm_problem();
    let mut x = cold.x;
    let mut obj = 0.0;
    let status = unsafe {
        IpoptSolve(
            p,
            x.as_mut_ptr(),
            std::ptr::null_mut(),
            &mut obj,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
        )
    };
    unsafe { FreeIpoptProblem(p) };
    // NULL is "not supplied", not an error: the solve runs (and stops at
    // max_iter=0) exactly as a primal-only warm start does.
    assert_eq!(
        status, MAX_ITER_EXCEEDED,
        "NULL multiplier buffers must not fail the solve"
    );
    assert_eq!(x, cold.x, "the primal half of the warm start still applies");
}

/// With `warm_start_init_point` off (the default) the buffers are
/// out-only and may hold garbage: a sentinel planted in them must never
/// reach the solve.
#[test]
fn hs071_multiplier_buffers_are_not_read_without_the_warm_start_option() {
    const JUNK: Number = 1.0e300;
    let p = make_problem();
    int_opt(p, "max_iter", 0);
    let o = solve(p, [1.0, 5.0, 5.0, 1.0], [JUNK; M], ([JUNK; N], [JUNK; N]));
    assert_eq!(o.status, MAX_ITER_EXCEEDED);
    for v in o.lam.iter().chain(&o.zl).chain(&o.zu) {
        assert!(
            v.is_finite() && v.abs() < 1e6,
            "garbage reached the cold start: {v}"
        );
    }
}
