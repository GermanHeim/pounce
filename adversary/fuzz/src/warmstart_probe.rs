//! Metamorphic probe for the C warm-start ABI.
//!
//! The contract `pounce.h` states is that
//! `IpoptSetWarmStartWorkingSet` supplies *a working set*. A working set
//! is a hint about which constraints are active — it selects the path
//! the active-set method takes, never the answer it arrives at. So:
//!
//! > For any problem and any starting point, staging any *valid* working
//! > set before `IpoptSolve` must not change where the solve ends up.
//!
//! That is the property gh#484 violated. The call also reset the primal
//! iterate to zero, so on a box excluding the origin the "hint" decided
//! the answer — it returned `x = 0` and `Infeasible_Problem_Detected`.
//!
//! The probe is metamorphic rather than oracle-based on purpose: it
//! needs no reference optimum, so it can hammer randomized problems
//! where no published solution exists. It compares pounce against
//! *itself under a transformation that must be answer-preserving*, which
//! is a different and stronger thing than comparing two solvers.
//!
//! Deliberately adversarial choices:
//!
//! * boxes translated far from the origin, so the discarded-iterate bug
//!   is catastrophic rather than cosmetic;
//! * **wrong** working sets — all-inactive, all-at-lower, and random —
//!   because a bad hint must still be answer-preserving. Case D of the
//!   reporter's driver showed the contents never mattered; making the
//!   call at all was the defect;
//! * starting points at, near, and far from the solution.

use pounce_cinterface::*;
use std::ffi::{CString, c_void};

/// `min ½Σ(x_j − t_j)² + ½ Σ_j c_j x_j x_{j+1}`
/// s.t. `Σ x = s`, `x_0 · x_1 >= p`, `lo ≤ x ≤ hi`
///
/// One equality, one nonconvex inequality, a box. The cross term makes
/// ∇²L indefinite, which is what puts the step QP on the elastic path.
/// The parameters live in a thread-local so the `extern "C"` callbacks
/// can reach them without a user-data round trip.
#[derive(Clone)]
pub struct Params {
    /// When set, the objective drops its cross terms and the second row
    /// is linear, so the whole program is convex and its minimizer is
    /// unique. Answer-transparency is only a *theorem* under convexity:
    /// on a nonconvex program an SQP is a local method, and a different
    /// working set may legitimately steer it to a different local
    /// minimum. Testing the strict property there would be testing a
    /// claim that is not true.
    pub convex: bool,
    pub n: usize,
    pub t: Vec<f64>,
    pub c: Vec<f64>,
    pub s: f64,
    pub p: f64,
    pub lo: f64,
    pub hi: f64,
}

thread_local! {
    static PARAMS: std::cell::RefCell<Params> = std::cell::RefCell::new(Params {
        convex: false, n: 3, t: vec![0.0; 3], c: vec![0.0; 2], s: 1.0, p: 0.0, lo: 0.0, hi: 1.0,
    });
}

fn with<R>(f: impl FnOnce(&Params) -> R) -> R {
    PARAMS.with(|p| f(&p.borrow()))
}

unsafe extern "C" fn ev_f(
    _n: Index,
    x: *const Number,
    _new_x: Bool,
    obj: *mut Number,
    _u: *mut c_void,
) -> Bool {
    with(|p| unsafe {
        let x = std::slice::from_raw_parts(x, p.n);
        let mut v = 0.0;
        for j in 0..p.n {
            v += 0.5 * (x[j] - p.t[j]).powi(2);
        }
        if !p.convex {
            for j in 0..p.n - 1 {
                v += 0.5 * p.c[j] * x[j] * x[j + 1];
            }
        }
        *obj = v;
        1
    })
}

unsafe extern "C" fn ev_grad_f(
    _n: Index,
    x: *const Number,
    _new_x: Bool,
    grad: *mut Number,
    _u: *mut c_void,
) -> Bool {
    with(|p| unsafe {
        let x = std::slice::from_raw_parts(x, p.n);
        let g = std::slice::from_raw_parts_mut(grad, p.n);
        for j in 0..p.n {
            g[j] = x[j] - p.t[j];
        }
        if !p.convex {
            for j in 0..p.n - 1 {
                g[j] += 0.5 * p.c[j] * x[j + 1];
                g[j + 1] += 0.5 * p.c[j] * x[j];
            }
        }
        1
    })
}

unsafe extern "C" fn ev_g(
    _n: Index,
    x: *const Number,
    _new_x: Bool,
    _m: Index,
    gout: *mut Number,
    _u: *mut c_void,
) -> Bool {
    with(|p| unsafe {
        let x = std::slice::from_raw_parts(x, p.n);
        let g = std::slice::from_raw_parts_mut(gout, 2);
        g[0] = x.iter().sum();
        g[1] = if p.convex { x[0] + 2.0 * x[1] } else { x[0] * x[1] };
        1
    })
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
    with(|p| unsafe {
        let n = p.n;
        if values.is_null() {
            let rows = std::slice::from_raw_parts_mut(i_row, 2 * n);
            let cols = std::slice::from_raw_parts_mut(j_col, 2 * n);
            let mut k = 0;
            for i in 0..2 {
                for j in 0..n {
                    rows[k] = i as Index;
                    cols[k] = j as Index;
                    k += 1;
                }
            }
            return 1;
        }
        let x = std::slice::from_raw_parts(x, n);
        let v = std::slice::from_raw_parts_mut(values, 2 * n);
        for j in 0..n {
            v[j] = 1.0;
        }
        for j in 0..n {
            v[n + j] = 0.0;
        }
        if p.convex {
            v[n] = 1.0;
            v[n + 1] = 2.0;
        } else {
            v[n] = x[1];
            v[n + 1] = x[0];
        }
        1
    })
}

#[allow(clippy::too_many_arguments)]
unsafe extern "C" fn ev_h(
    _n: Index,
    _x: *const Number,
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
    with(|p| unsafe {
        let n = p.n;
        let nnz = n * (n + 1) / 2;
        if values.is_null() {
            let rows = std::slice::from_raw_parts_mut(i_row, nnz);
            let cols = std::slice::from_raw_parts_mut(j_col, nnz);
            let mut k = 0;
            for i in 0..n {
                for j in 0..=i {
                    rows[k] = i as Index;
                    cols[k] = j as Index;
                    k += 1;
                }
            }
            return 1;
        }
        let lam = std::slice::from_raw_parts(lambda, 2);
        let v = std::slice::from_raw_parts_mut(values, nnz);
        let at = |i: usize, j: usize| i * (i + 1) / 2 + j;
        for e in v.iter_mut() {
            *e = 0.0;
        }
        for i in 0..n {
            v[at(i, i)] = obj_factor;
        }
        if !p.convex {
            for j in 0..n - 1 {
                v[at(j + 1, j)] += obj_factor * 0.5 * p.c[j];
            }
            // ∇²(x0·x1) contributes 1 to the (1,0) entry; the convex
            // variant's second row is linear, so it contributes nothing.
            v[at(1, 0)] += lam[1];
        }
        1
    })
}

fn make_problem(p: &Params) -> IpoptProblem {
    let n = p.n;
    let x_l = vec![p.lo; n];
    let x_u = vec![p.hi; n];
    let g_l = [p.s, p.p];
    let g_u = [p.s, 2.0e19];
    unsafe {
        let prob = CreateIpoptProblem(
            n as Index,
            x_l.as_ptr(),
            x_u.as_ptr(),
            2,
            g_l.as_ptr(),
            g_u.as_ptr(),
            (2 * n) as Index,
            (n * (n + 1) / 2) as Index,
            0,
            Some(ev_f),
            Some(ev_g),
            Some(ev_grad_f),
            Some(ev_jac_g),
            Some(ev_h),
        );
        assert!(!prob.is_null());
        let k = CString::new("algorithm").unwrap();
        let v = CString::new("active-set-sqp").unwrap();
        assert_eq!(AddIpoptStrOption(prob, k.as_ptr(), v.as_ptr()), 1);
        let k = CString::new("print_level").unwrap();
        assert_eq!(AddIpoptIntOption(prob, k.as_ptr(), 0), 1);
        prob
    }
}

pub struct SolveOut {
    pub status: Index,
    pub x: Vec<f64>,
    pub obj: f64,
}

/// Solve from `x0`, optionally staging `ws` first.
pub fn solve(p: &Params, x0: &[f64], ws: Option<(&[i32], &[i32])>) -> SolveOut {
    PARAMS.with(|c| *c.borrow_mut() = p.clone());
    let prob = make_problem(p);
    if let Some((b, c)) = ws {
        let rc = unsafe { IpoptSetWarmStartWorkingSet(prob, b.as_ptr(), c.as_ptr()) };
        assert_eq!(rc, 1, "IpoptSetWarmStartWorkingSet rejected a valid set");
    }
    let mut x = x0.to_vec();
    let mut g = [0.0; 2];
    let mut mult_g = [0.0; 2];
    let mut zl = vec![0.0; p.n];
    let mut zu = vec![0.0; p.n];
    let mut obj = 0.0;
    let status = unsafe {
        IpoptSolve(
            prob,
            x.as_mut_ptr(),
            g.as_mut_ptr(),
            &mut obj,
            mult_g.as_mut_ptr(),
            zl.as_mut_ptr(),
            zu.as_mut_ptr(),
            std::ptr::null_mut(),
        )
    };
    unsafe { FreeIpoptProblem(prob) };
    SolveOut { status, x, obj }
}

/// Working set read back from a converged solve, or `None`.
pub fn working_set_after(p: &Params, x0: &[f64]) -> Option<(Vec<i32>, Vec<i32>)> {
    PARAMS.with(|c| *c.borrow_mut() = p.clone());
    let prob = make_problem(p);
    let mut x = x0.to_vec();
    let mut g = [0.0; 2];
    let mut obj = 0.0;
    let status = unsafe {
        IpoptSolve(
            prob,
            x.as_mut_ptr(),
            g.as_mut_ptr(),
            &mut obj,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
        )
    };
    let mut b = vec![-1i32; p.n];
    let mut c = vec![-1i32; 2];
    let got = unsafe { IpoptGetWorkingSet(prob, b.as_mut_ptr(), c.as_mut_ptr()) };
    unsafe { FreeIpoptProblem(prob) };
    (status == 0 && got == 1).then_some((b, c))
}

/// Violation of the program's own constraints at `x`. The invariant that
/// survives nonconvexity: whatever local solution a warm solve reaches,
/// a converged answer must satisfy the constraints. The gh#484 bug
/// returned `x = 0` on a box excluding the origin, so this alone catches
/// it — no reference optimum and no convexity assumption needed.
pub fn violation(p: &Params, x: &[f64]) -> f64 {
    let mut worst: f64 = 0.0;
    for &v in x {
        worst = worst.max(p.lo - v).max(v - p.hi);
    }
    let sum: f64 = x.iter().sum();
    worst = worst.max((sum - p.s).abs());
    let g1 = if p.convex { x[0] + 2.0 * x[1] } else { x[0] * x[1] };
    worst = worst.max(p.p - g1);
    worst.max(0.0)
}
