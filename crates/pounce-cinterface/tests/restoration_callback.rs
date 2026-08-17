//! gh#645 — the intermediate callback fires from the feasibility
//! restoration phase, labelled, and can stop the solve from there.
//!
//! Before this, `fire_intermediate` had two call sites, both in the
//! outer loop, so `alg_mod` was not merely untracked but unreachable:
//! no fire could ever legitimately report `RestorationPhaseMode`. A
//! caller was therefore blind for the whole of restoration — the phase
//! a real-time caller most needs to be able to abort in, because it is
//! the one that overruns a control period.
//!
//! Three properties, one per test:
//!
//! 1. restoration fires happen and are labelled `alg_mod == 1`; a solve
//!    that never restores never produces one;
//! 2. the `GetIpoptCurrent*` inspectors report no data during those
//!    fires — the iterate belongs to the min-‖c‖₁ subproblem and does
//!    not have this problem's dimensions;
//! 3. returning `false` from a restoration fire ends the solve at the
//!    last iterate accepted for the *original* NLP, not at the
//!    subproblem's iterate. A controller that aborts on a deadline has
//!    to apply something, so which point comes back is the part that
//!    matters to it — more than the status code.

use pounce_cinterface::*;
use std::cell::RefCell;
use std::ffi::{CString, c_void};

/// `ApplicationReturnStatus::User_Requested_Stop`. Spelled out rather
/// than imported because an integration test sees only the crate under
/// test and its dev-dependencies, and the enum lives in `pounce-nlp`.
/// `return_codes.rs` pins the discriminant.
const USER_REQUESTED_STOP: Index = 5;

const N: usize = 1;
const M: usize = 1;

/// Whether the callback should ask to stop, and what it saw.
#[derive(Default)]
struct Seen {
    regular_fires: usize,
    restoration_fires: usize,
    /// `x` at the last fire with `alg_mod == 0`, read through the
    /// inspector — the last iterate of the user's NLP the solver
    /// reported.
    last_regular_x: Option<Number>,
    /// Inspector verdicts, split by mode. `(calls, ok)`.
    regular_inspector: (usize, usize),
    restoration_inspector: (usize, usize),
    /// Ask for a stop on the first restoration fire.
    stop_on_restoration: bool,
}

thread_local! {
    static SEEN: RefCell<Seen> = RefCell::new(Seen::default());
}

// ---------------------------------------------------------------
// A problem whose equality constraint cannot be satisfied: with
// `g = x² + 1` pinned to zero there is no feasible point, so the line
// search fails to find an acceptable step and the solver enters
// restoration. Same shape the CasADi parity suite uses to exercise the
// solve-level restoration counters.
// ---------------------------------------------------------------

unsafe extern "C" fn ev_f(
    _n: Index,
    x: *const Number,
    _new_x: Bool,
    obj: *mut Number,
    _u: *mut c_void,
) -> Bool {
    unsafe {
        *obj = (*x) * (*x);
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
        *grad = 2.0 * (*x);
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
        *gout = (*x) * (*x) + 1.0;
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
            *i_row = 0;
            *j_col = 0;
            return 1;
        }
        *values = 2.0 * (*x);
        1
    }
}

/// Records the mode of every fire, asks both inspectors, and — when
/// armed — requests a stop from the first restoration fire.
#[allow(clippy::too_many_arguments)]
unsafe extern "C" fn on_iter(
    alg_mod: Index,
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
        let prob = user_data as IpoptProblem;

        let mut x = [0.0 as Number; N];
        let mut z_l = [0.0 as Number; N];
        let mut z_u = [0.0 as Number; N];
        let mut g = [0.0 as Number; M];
        let mut lambda = [0.0 as Number; M];
        let ok = GetIpoptCurrentIterate(
            prob,
            0,
            N as Index,
            x.as_mut_ptr(),
            z_l.as_mut_ptr(),
            z_u.as_mut_ptr(),
            M as Index,
            g.as_mut_ptr(),
            lambda.as_mut_ptr(),
        );

        SEEN.with(|s| {
            let mut s = s.borrow_mut();
            if alg_mod == 0 {
                s.regular_fires += 1;
                s.regular_inspector.0 += 1;
                if ok != 0 {
                    s.regular_inspector.1 += 1;
                    s.last_regular_x = Some(x[0]);
                }
                1
            } else {
                s.restoration_fires += 1;
                s.restoration_inspector.0 += 1;
                if ok != 0 {
                    s.restoration_inspector.1 += 1;
                }
                if s.stop_on_restoration { 0 } else { 1 }
            }
        })
    }
}

/// Build the infeasible-equality problem. `feasible = true` relaxes the
/// constraint to a satisfiable one, giving the control case.
unsafe fn make_problem(feasible: bool) -> IpoptProblem {
    unsafe {
        let x_l = [-5.0 as Number; N];
        let x_u = [5.0 as Number; N];
        // `x² + 1 == 0` is unsatisfiable; `x² + 1 == 2` is not.
        let (g_l, g_u) = if feasible {
            ([2.0 as Number], [2.0 as Number])
        } else {
            ([0.0 as Number], [0.0 as Number])
        };

        let prob = CreateIpoptProblem(
            N as Index,
            x_l.as_ptr(),
            x_u.as_ptr(),
            M as Index,
            g_l.as_ptr(),
            g_u.as_ptr(),
            N as Index,
            0,
            0,
            Some(ev_f),
            Some(ev_g),
            Some(ev_grad_f),
            Some(ev_jac_g),
            None,
        );
        assert!(!prob.is_null(), "CreateIpoptProblem returned NULL");

        let key = CString::new("hessian_approximation").unwrap();
        let val = CString::new("limited-memory").unwrap();
        assert_ne!(
            AddIpoptStrOption(prob, key.as_ptr() as *mut _, val.as_ptr() as *mut _),
            0
        );
        let key = CString::new("print_level").unwrap();
        assert_ne!(AddIpoptIntOption(prob, key.as_ptr() as *mut _, 0), 0);
        assert_ne!(SetIntermediateCallback(prob, Some(on_iter)), 0);
        prob
    }
}

/// Solve, returning `(status, x_returned)`. Resets the observation
/// state first, and arms the stop-on-restoration behaviour per `stop`.
unsafe fn solve(feasible: bool, stop: bool) -> (Index, Number) {
    unsafe {
        SEEN.with(|s| {
            let mut s = s.borrow_mut();
            *s = Seen::default();
            s.stop_on_restoration = stop;
        });
        let prob = make_problem(feasible);
        let mut x = [0.5 as Number; N];
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
            prob as *mut c_void,
        );
        FreeIpoptProblem(prob);
        (status as Index, x[0])
    }
}

#[test]
fn restoration_iterations_fire_the_callback_and_are_labelled() {
    unsafe {
        let (_status, _x) = solve(false, false);
        SEEN.with(|s| {
            let s = s.borrow();
            assert!(s.regular_fires > 0, "no outer-loop fires at all");
            assert!(
                s.restoration_fires > 0,
                "restoration ran but never fired the callback — {} regular fires only",
                s.regular_fires
            );
        });

        // Control: a solve that never restores must never produce a
        // restoration fire. Without this the test above would pass on an
        // implementation that labelled every fire as restoration.
        let (status, _x) = solve(true, false);
        assert_eq!(status, 0, "control problem did not solve");
        SEEN.with(|s| {
            let s = s.borrow();
            assert!(s.regular_fires > 0, "no fires on the control solve");
            assert_eq!(
                s.restoration_fires, 0,
                "control solve reported {} restoration fires without restoring",
                s.restoration_fires
            );
        });
    }
}

#[test]
fn inspectors_report_no_data_during_a_restoration_fire() {
    unsafe {
        solve(false, false);
        SEEN.with(|s| {
            let s = s.borrow();
            assert!(s.restoration_inspector.0 > 0, "no restoration fires");
            // The restoration iterate is a compound `(x_orig, n, p)`
            // vector of the subproblem, so there is no current iterate
            // of *this* problem to report. Answering anyway would mean
            // filling the caller's `n`-sized buffers from a differently
            // shaped `cq`.
            assert_eq!(
                s.restoration_inspector.1, 0,
                "GetIpoptCurrentIterate answered {} of {} restoration fires",
                s.restoration_inspector.1, s.restoration_inspector.0
            );
            // ...and still answers normally outside them, which is the
            // property `current_iterate_inspectors.rs` pins in full.
            assert_eq!(
                s.regular_inspector.1, s.regular_inspector.0,
                "inspector refused an outer-loop fire"
            );
        });
    }
}

#[test]
fn stopping_from_restoration_returns_the_last_original_nlp_iterate() {
    unsafe {
        let (status, x_returned) = solve(false, true);
        assert_eq!(
            status, USER_REQUESTED_STOP,
            "callback returned false from restoration but the solve did not stop"
        );

        let (fires, last_regular_x) =
            SEEN.with(|s| (s.borrow().restoration_fires, s.borrow().last_regular_x));
        // Exactly one: the stop must take effect on the fire that asked
        // for it, not after the sub-solve runs to completion.
        assert_eq!(
            fires, 1,
            "the sub-solve kept iterating after the callback asked to stop"
        );

        let expected = last_regular_x.expect("no outer-loop fire recorded an iterate");
        assert_eq!(
            x_returned, expected,
            "aborting inside restoration handed back a point that is not the last \
             iterate accepted for the original NLP (subproblem iterate leaked?)"
        );
    }
}
