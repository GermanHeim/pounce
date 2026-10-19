//! `IpoptGetWorkingSet` / `IpoptSetWarmStartWorkingSet` index by the
//! caller's variables and rows, both ways, including when a variable is
//! fixed.
//!
//! The two entry points used to rebuild the SQP's internal order from the
//! bounds (equalities first, fixed variables dropped) and permute against
//! it. Once the application started lifting bound statuses to full `n`
//! (gh#986) that reconstruction no longer matched what it read: on any
//! problem with a fixed variable `IpoptGetWorkingSet` returned FALSE after a
//! successful solve, so the GAMS link's state file and every C round trip
//! silently lost the working set. The application now publishes and takes
//! user order itself, and the C layer copies positionally.
//!
//! Model (also `pounce-algorithm/tests/sqp_working_set_user_order.rs`):
//! `min (x1-3)² + (x2-3)² + (x3-1)² + x0·x2` s.t. `g0 = x1 + x2 ≤ 2.5`
//! (active), `g1 = x2 − x3 = 0`, `x0 ∈ [0.5, X0_U]`, `x1 ∈ [-10, 1]`
//! (upper active), `x2` free, `x3 ≥ 0`. With `X0_U = 0.5` the first
//! variable is fixed; with `X0_U = 0.6` it sits on its lower bound.

use pounce_cinterface::*;
use std::ffi::{CString, c_void};

const N: usize = 4;
const M: usize = 2;
const INACTIVE: IpoptBoundStatus = 0;
const AT_LOWER: IpoptBoundStatus = 1;
const AT_UPPER: IpoptBoundStatus = 2;
const FIXED_OR_EQ: IpoptBoundStatus = 3;

unsafe extern "C" fn ev_f(
    _n: Index,
    x: *const Number,
    _new_x: Bool,
    obj: *mut Number,
    _u: *mut c_void,
) -> Bool {
    unsafe {
        let x = std::slice::from_raw_parts(x, N);
        *obj = (x[1] - 3.0).powi(2) + (x[2] - 3.0).powi(2) + (x[3] - 1.0).powi(2) + x[0] * x[2];
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
        g[0] = x[2];
        g[1] = 2.0 * (x[1] - 3.0);
        g[2] = 2.0 * (x[2] - 3.0) + x[0];
        g[3] = 2.0 * (x[3] - 1.0);
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
        g[0] = x[1] + x[2];
        g[1] = x[2] - x[3];
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
            let r = std::slice::from_raw_parts_mut(i_row, 4);
            let c = std::slice::from_raw_parts_mut(j_col, 4);
            r.copy_from_slice(&[0, 0, 1, 1]);
            c.copy_from_slice(&[1, 2, 2, 3]);
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
            let r = std::slice::from_raw_parts_mut(i_row, 4);
            let c = std::slice::from_raw_parts_mut(j_col, 4);
            r.copy_from_slice(&[1, 2, 3, 2]);
            c.copy_from_slice(&[1, 2, 3, 0]);
        } else {
            std::slice::from_raw_parts_mut(values, 4).copy_from_slice(&[
                2.0 * of,
                2.0 * of,
                2.0 * of,
                of,
            ]);
        }
        1
    }
}

fn create(x0_u: Number) -> IpoptProblem {
    let x_l = [0.5, -10.0, -2e19, 0.0];
    let x_u = [x0_u, 1.0, 2e19, 2e19];
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
            4,
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
        opt("algorithm", "active-set-sqp");
        let k = CString::new("print_level").unwrap();
        assert_ne!(AddIpoptIntOption(p, k.as_ptr(), 0), 0);
        p
    }
}

struct Sol {
    status: Index,
    x: [Number; N],
    mult_g: [Number; M],
    z_l: [Number; N],
    z_u: [Number; N],
}

fn solve(p: IpoptProblem, x0: [Number; N], duals: Option<&Sol>) -> Sol {
    let mut s = Sol {
        status: -1,
        x: x0,
        mult_g: duals.map_or([0.0; M], |d| d.mult_g),
        z_l: duals.map_or([0.0; N], |d| d.z_l),
        z_u: duals.map_or([0.0; N], |d| d.z_u),
    };
    let mut obj = 0.0;
    s.status = unsafe {
        IpoptSolve(
            p,
            s.x.as_mut_ptr(),
            std::ptr::null_mut(),
            &mut obj,
            s.mult_g.as_mut_ptr(),
            s.z_l.as_mut_ptr(),
            s.z_u.as_mut_ptr(),
            std::ptr::null_mut(),
        )
    };
    s
}

fn get_ws(p: IpoptProblem) -> Option<([IpoptBoundStatus; N], [IpoptConsStatus; M])> {
    let mut b = [-1; N];
    let mut c = [-1; M];
    let ok = unsafe { IpoptGetWorkingSet(p, b.as_mut_ptr(), c.as_mut_ptr()) };
    (ok != 0).then_some((b, c))
}

/// (a) after a cold SQP solve, every status sits on its own user row and
/// variable; (b) staging that set (with the reported point and duals under
/// `warm_start_init_point`) re-solves in 0 iterations and reads back equal.
fn check(x0_u: Number, x0_status: IpoptBoundStatus) {
    let p = create(x0_u);
    let cold = solve(p, [0.5, 0.0, 0.0, 0.5], None);
    assert_eq!(cold.status, 0, "cold solve");
    let want_x = [0.5, 1.0, 1.5, 1.5];
    for j in 0..N {
        assert!((cold.x[j] - want_x[j]).abs() < 1e-7, "x = {:?}", cold.x);
    }
    assert!((cold.mult_g[0] - 1.5).abs() < 1e-6 && (cold.mult_g[1] - 1.0).abs() < 1e-6);

    let want = (
        [x0_status, AT_UPPER, INACTIVE, INACTIVE],
        [AT_UPPER, FIXED_OR_EQ],
    );
    let got = get_ws(p).expect("IpoptGetWorkingSet after a successful SQP solve");
    assert_eq!(got, want, "published working set (bounds, rows)");

    let set = unsafe { IpoptSetWarmStartWorkingSet(p, got.0.as_ptr(), got.1.as_ptr()) };
    assert_eq!(set, 1, "the published set is accepted back");
    let k = CString::new("warm_start_init_point").unwrap();
    let v = CString::new("yes").unwrap();
    assert_ne!(unsafe { AddIpoptStrOption(p, k.as_ptr(), v.as_ptr()) }, 0);
    let warm = solve(p, cold.x, Some(&cold));
    assert_eq!(warm.status, 0, "warm solve");
    assert_eq!(
        unsafe { GetIpoptIterCount(p) },
        0,
        "warm re-solve iterations"
    );
    assert_eq!(get_ws(p), Some(want), "re-published working set");
    unsafe { FreeIpoptProblem(p) };
}

#[test]
fn working_set_is_user_ordered_with_a_fixed_variable_ahead() {
    check(0.5, FIXED_OR_EQ);
}

#[test]
fn working_set_is_user_ordered_with_no_fixed_variable() {
    check(0.6, AT_LOWER);
}
