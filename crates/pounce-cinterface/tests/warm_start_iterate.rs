//! Regression test for gh#484 — `IpoptSetWarmStartWorkingSet` used to
//! discard the caller's iterate.
//!
//! The set-then-solve path eagerly built a full `SqpIterates` with
//! `x = 0`. `SqpAlgorithm::optimize_with_warm_start` treats a supplied
//! iterate as *the* starting point (it only consults the NLP's
//! `get_starting_x` on the cold branch), so those zeros overrode the `x`
//! buffer passed to `IpoptSolve`. On HS071 — whose bounds are
//! `1 <= x <= 5`, so the origin is not even bound-feasible — the warm
//! solve returned `Infeasible_Problem_Detected` at iteration 0 and wrote
//! zeros back into `x`.
//!
//! This mirrors cases B/C/D of the reporter's C driver: the *only*
//! difference between the control and the warm solves is the single
//! `IpoptSetWarmStartWorkingSet` call, so any failure here is
//! attributable to that call alone.

use pounce_cinterface::*;
use pounce_nlp::ApplicationReturnStatus;
use std::ffi::{CString, c_void};

const N: usize = 4;
const M: usize = 2;

// Status codes are pinned by `pounce.h` (`POUNCE_WS_*`); the crate keeps
// them private, so spell out the one we need the way a C caller would.
const WS_INACTIVE: IpoptBoundStatus = 0;

/// HS071 solution, to ~1e-6.
const X_STAR: [Number; N] = [1.0, 4.742999, 3.821150, 1.379408];

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
            // Dense 2x4 structure, row-major, 0-based.
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
            // Dense lower triangle, 0-based.
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

/// Fresh HS071 problem on the active-set-SQP path.
fn make_problem() -> IpoptProblem {
    let x_l = [1.0; N];
    let x_u = [5.0; N];
    let g_l = [25.0, 40.0];
    let g_u = [2.0e19, 40.0];
    unsafe {
        let p = CreateIpoptProblem(
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
        );
        assert!(!p.is_null(), "CreateIpoptProblem failed");
        let key = CString::new("algorithm").unwrap();
        let val = CString::new("active-set-sqp").unwrap();
        assert_eq!(AddIpoptStrOption(p, key.as_ptr(), val.as_ptr()), 1);
        let key = CString::new("print_level").unwrap();
        assert_eq!(AddIpoptIntOption(p, key.as_ptr(), 0), 1);
        p
    }
}

/// Solve starting from `x0`, optionally staging `working_set` first.
/// Returns `(status, x, obj)`.
fn solve_from(
    x0: [Number; N],
    working_set: Option<(&[IpoptBoundStatus; N], &[IpoptConsStatus; M])>,
) -> (Index, [Number; N], Number) {
    let p = make_problem();
    if let Some((bounds, cons)) = working_set {
        let rc = unsafe { IpoptSetWarmStartWorkingSet(p, bounds.as_ptr(), cons.as_ptr()) };
        assert_eq!(rc, 1, "IpoptSetWarmStartWorkingSet returned FALSE");
    }
    let mut x = x0;
    let mut g = [0.0; M];
    let mut mult_g = [0.0; M];
    let mut zl = [0.0; N];
    let mut zu = [0.0; N];
    let mut obj = 0.0;
    let status = unsafe {
        IpoptSolve(
            p,
            x.as_mut_ptr(),
            g.as_mut_ptr(),
            &mut obj,
            mult_g.as_mut_ptr(),
            zl.as_mut_ptr(),
            zu.as_mut_ptr(),
            std::ptr::null_mut(),
        )
    };
    unsafe { FreeIpoptProblem(p) };
    (status, x, obj)
}

fn assert_at_optimum(tag: &str, status: Index, x: [Number; N], obj: Number) {
    assert_eq!(
        status,
        ApplicationReturnStatus::SolveSucceeded as Index,
        "{tag}: expected Solve_Succeeded, got status {status} with x = {x:?}"
    );
    for (i, (got, want)) in x.iter().zip(X_STAR.iter()).enumerate() {
        assert!(
            (got - want).abs() < 1e-4,
            "{tag}: x[{i}] = {got}, expected ~{want} (full x = {x:?})"
        );
    }
    assert!(
        (obj - 17.014_017_3).abs() < 1e-5,
        "{tag}: obj = {obj}, expected ~17.0140173"
    );
}

/// Case A in the report: cold solve from `(1, 5, 5, 1)`, then read the
/// converged iterate and working set back out. Everything downstream
/// warm-starts from exactly this pair, as the reporter's driver does.
fn cold_solve_and_working_set() -> ([Number; N], [IpoptBoundStatus; N], [IpoptConsStatus; M]) {
    let p = make_problem();
    let mut x = [1.0, 5.0, 5.0, 1.0];
    let mut g = [0.0; M];
    let mut obj = 0.0;
    let status = unsafe {
        IpoptSolve(
            p,
            x.as_mut_ptr(),
            g.as_mut_ptr(),
            &mut obj,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
        )
    };
    assert_at_optimum("A cold solve", status, x, obj);

    let mut bounds = [-1; N];
    let mut cons = [-1; M];
    let got = unsafe { IpoptGetWorkingSet(p, bounds.as_mut_ptr(), cons.as_mut_ptr()) };
    assert_eq!(
        got, 1,
        "IpoptGetWorkingSet returned FALSE after a cold solve"
    );
    // The statuses must be indexed by the *caller's* rows and variables,
    // not by the SQP's internal equalities-first ordering. HS071's rows
    // are `[x₀x₁x₂x₃ >= 25, Σxᵢ² = 40]`, so at the solution row 0 is an
    // inequality sitting on its lower bound and row 1 is the equality:
    // `[AtLower, Equality]`. Reported the other way round — `[3, 1]`, as
    // in the gh#484 reproducer's output — the documented get/set
    // round-trip feeds the next solve a working set with the two rows'
    // statuses swapped.
    assert_eq!(
        cons,
        [1, 3],
        "constraint statuses must be in the caller's row order: row 0 is \
         the `>= 25` inequality (AtLower = 1), row 1 the equality (3)"
    );
    // x₀ = 1 sits on its lower bound; the rest are interior.
    assert_eq!(
        bounds,
        [1, 0, 0, 0],
        "bound statuses must be in the caller's variable order"
    );
    unsafe { FreeIpoptProblem(p) };
    for s in bounds.iter().chain(cons.iter()) {
        assert!((0..=3).contains(s), "status code {s} out of range");
    }
    (x, bounds, cons)
}

/// Case B in the report: the control. No warm-start call at all, so this
/// establishes that the problem and the starting point are both fine.
#[test]
fn hs071_control_starting_at_the_solution_converges() {
    let (x_star, _, _) = cold_solve_and_working_set();
    let (status, x, obj) = solve_from(x_star, None);
    assert_at_optimum("B control (no warm call)", status, x, obj);
}

/// Case C: identical to B plus one `IpoptSetWarmStartWorkingSet` call
/// carrying the true converged working set. Before gh#484 this returned
/// `Infeasible_Problem_Detected` with `x = (0,0,0,0)`.
#[test]
fn hs071_warm_start_preserves_the_callers_iterate() {
    let (x_star, bounds, cons) = cold_solve_and_working_set();
    let (status, x, obj) = solve_from(x_star, Some((&bounds, &cons)));
    assert_at_optimum("C warm (true working set)", status, x, obj);
}

/// Case D: an all-inactive working set is semantically a cold start, so
/// it must behave like B too. This pins down that the *contents* of the
/// working set were never the problem — making the call at all was.
#[test]
fn hs071_warm_start_with_inactive_working_set_preserves_the_iterate() {
    let (x_star, _, _) = cold_solve_and_working_set();
    let bounds = [WS_INACTIVE; N];
    let cons = [WS_INACTIVE; M];
    let (status, x, obj) = solve_from(x_star, Some((&bounds, &cons)));
    assert_at_optimum("D warm (all-inactive WS)", status, x, obj);
}

/// The one-shot convenience wrapper delegates to the same staging path,
/// so it was affected identically.
#[test]
fn hs071_solve_warm_start_one_shot_preserves_the_callers_iterate() {
    let (x_star, bounds, cons) = cold_solve_and_working_set();
    let p = make_problem();
    let mut x = x_star;
    let mut g = [0.0; M];
    let mut obj = 0.0;
    let mut bounds_out = [-1; N];
    let mut cons_out = [-1; M];
    let status = unsafe {
        IpoptSolveWarmStart(
            p,
            x.as_mut_ptr(),
            g.as_mut_ptr(),
            &mut obj,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            bounds.as_ptr(),
            cons.as_ptr(),
            bounds_out.as_mut_ptr(),
            cons_out.as_mut_ptr(),
            std::ptr::null_mut(),
        )
    };
    unsafe { FreeIpoptProblem(p) };
    assert_at_optimum("IpoptSolveWarmStart", status, x, obj);
    for s in bounds_out.iter().chain(cons_out.iter()) {
        assert!((0..=3).contains(s), "output status code {s} out of range");
    }
}

/// A staged working set is consumed by exactly one solve, and
/// `IpoptClearWarmStartWorkingSet` drops it. Neither may leave the
/// pending state wired to a stale iterate: a second solve on the same
/// handle must start from the `x` buffer it is given.
#[test]
fn hs071_pending_working_set_is_consumed_once() {
    let (x_star, bounds, cons) = cold_solve_and_working_set();
    let p = make_problem();
    let rc = unsafe { IpoptSetWarmStartWorkingSet(p, bounds.as_ptr(), cons.as_ptr()) };
    assert_eq!(rc, 1);

    let mut x = x_star;
    let mut g = [0.0; M];
    let mut obj = 0.0;
    let mut run = |x: &mut [Number; N], obj: &mut Number| -> Index {
        unsafe {
            IpoptSolve(
                p,
                x.as_mut_ptr(),
                g.as_mut_ptr(),
                obj,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
            )
        }
    };
    let s1 = run(&mut x, &mut obj);
    assert_at_optimum("warm solve", s1, x, obj);

    // Second solve, no fresh staging: cold, from a different point.
    let mut x2 = [1.0, 5.0, 5.0, 1.0];
    let mut obj2 = 0.0;
    let s2 = run(&mut x2, &mut obj2);
    assert_at_optimum("follow-up cold solve", s2, x2, obj2);

    // Clearing a staged working set is a no-op on the next solve.
    assert_eq!(
        unsafe { IpoptSetWarmStartWorkingSet(p, bounds.as_ptr(), cons.as_ptr()) },
        1
    );
    assert_eq!(unsafe { IpoptClearWarmStartWorkingSet(p) }, 1);
    let mut x3 = [1.0, 5.0, 5.0, 1.0];
    let mut obj3 = 0.0;
    let s3 = run(&mut x3, &mut obj3);
    assert_at_optimum("solve after clear", s3, x3, obj3);

    unsafe { FreeIpoptProblem(p) };
}

/// Under `warm_start_init_point=yes` the multiplier buffers are inputs
/// too (upstream Ipopt's `IpoptSolve` contract). Seeding them with the
/// previous solve's duals must not perturb the answer.
#[test]
fn hs071_warm_start_seeds_duals_when_opted_in() {
    // Cold solve first, keeping the duals.
    let p = make_problem();
    let mut x = [1.0, 5.0, 5.0, 1.0];
    let mut g = [0.0; M];
    let mut mult_g = [0.0; M];
    let mut zl = [0.0; N];
    let mut zu = [0.0; N];
    let mut obj = 0.0;
    let status = unsafe {
        IpoptSolve(
            p,
            x.as_mut_ptr(),
            g.as_mut_ptr(),
            &mut obj,
            mult_g.as_mut_ptr(),
            zl.as_mut_ptr(),
            zu.as_mut_ptr(),
            std::ptr::null_mut(),
        )
    };
    assert_at_optimum("cold solve", status, x, obj);
    let mut bounds = [-1; N];
    let mut cons = [-1; M];
    assert_eq!(
        unsafe { IpoptGetWorkingSet(p, bounds.as_mut_ptr(), cons.as_mut_ptr()) },
        1
    );
    unsafe { FreeIpoptProblem(p) };

    let p = make_problem();
    let key = CString::new("warm_start_init_point").unwrap();
    let val = CString::new("yes").unwrap();
    assert_eq!(
        unsafe { AddIpoptStrOption(p, key.as_ptr(), val.as_ptr()) },
        1
    );
    assert_eq!(
        unsafe { IpoptSetWarmStartWorkingSet(p, bounds.as_ptr(), cons.as_ptr()) },
        1
    );
    let mut x2 = x;
    let mut obj2 = 0.0;
    let status = unsafe {
        IpoptSolve(
            p,
            x2.as_mut_ptr(),
            g.as_mut_ptr(),
            &mut obj2,
            mult_g.as_mut_ptr(),
            zl.as_mut_ptr(),
            zu.as_mut_ptr(),
            std::ptr::null_mut(),
        )
    };
    unsafe { FreeIpoptProblem(p) };
    assert_at_optimum("warm solve with dual seeds", status, x2, obj2);
}
