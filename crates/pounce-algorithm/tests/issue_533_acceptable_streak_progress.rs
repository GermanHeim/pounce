//! gh #533 — the acceptable-level streak had no progress test.
//!
//! Acceptable-level termination fires after `acceptable_iter` consecutive
//! iterates under `acceptable_tol`, and asked nothing about whether the solve
//! was still moving. On the reported corpus models that streak completed at a
//! point near-stationary for the *barrier subproblem* while the NLP solve was
//! still descending, and POUNCE returned a worse answer under a weaker status
//! than continuing would have reached: `kissing` (Vanderbei) stopped at
//! iteration 103 with objective `1.00000108` where continuing reaches
//! `0.84544259` with a strict certificate at 550 (18% high, and Ipopt's own
//! answer to eight figures is the lower one); `NARX_CFy` (Mittelmann) stopped
//! at 565 where 60 more iterations collapse both residuals by five orders.
//!
//! Neither model is in-tree (the benchmark corpus is regenerated locally), and
//! the streak's window arithmetic is covered against the reported traces by the
//! unit tests in `conv_check::opt_error`. What this file covers is the
//! end-to-end behaviour on a self-contained problem: the mechanism engages
//! where the solve is still descending, stands aside where it is genuinely
//! stalled, is inert on an ordinary solve, and never returns a worse outcome
//! than the `acceptable_progress_kappa = 0` baseline.
//!
//! The trick that makes a two-line objective exhibit the shape is an
//! unreachable `tol`: with `tol = 1e-30` the strict gate can never fire, so the
//! *only* way out is the acceptable-level streak — which is exactly the regime
//! the reported models were in (both had already bottomed out `mu` and could not
//! reach `tol` at the point they stopped).

use pounce_algorithm::application::IpoptApplication;
use pounce_common::types::Number;
use pounce_nlp::return_codes::ApplicationReturnStatus;
use pounce_nlp::tnlp::{
    BoundsInfo, IndexStyle, IpoptCq, IpoptData, NlpInfo, Solution, SparsityRequest, StartingPoint,
    TNLP,
};
use std::cell::RefCell;
use std::rc::Rc;

/// `min x⁴`, unconstrained, from `x = 1`.
///
/// A Newton step on `x⁴` is `x → (2/3)x`, so the objective falls by `(2/3)⁴ ≈
/// 0.198` **per iteration, forever** while the gradient `4x³` falls by `(2/3)³ ≈
/// 0.296`. That is the shape the streak criterion cannot see: within a couple of
/// iterations of entering the acceptable band the KKT error is small enough to
/// qualify on every subsequent iterate, and the count runs out while the
/// objective is still dropping by a factor of five per iteration.
#[derive(Default)]
struct Quartic;

impl TNLP for Quartic {
    fn get_nlp_info(&mut self) -> Option<NlpInfo> {
        Some(NlpInfo {
            n: 1,
            m: 0,
            nnz_jac_g: 0,
            nnz_h_lag: 1,
            index_style: IndexStyle::C,
        })
    }

    fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
        b.x_l[0] = -2.0e19;
        b.x_u[0] = 2.0e19;
        true
    }

    fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
        sp.x[0] = 1.0;
        true
    }

    fn eval_f(&mut self, x: &[Number], _new_x: bool) -> Option<Number> {
        Some(x[0].powi(4))
    }

    fn eval_grad_f(&mut self, x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
        g[0] = 4.0 * x[0].powi(3);
        true
    }

    fn eval_g(&mut self, _x: &[Number], _new_x: bool, _g: &mut [Number]) -> bool {
        true
    }

    fn eval_jac_g(
        &mut self,
        _x: Option<&[Number]>,
        _new_x: bool,
        _mode: SparsityRequest<'_>,
    ) -> bool {
        true
    }

    fn eval_h(
        &mut self,
        x: Option<&[Number]>,
        _new_x: bool,
        obj_factor: Number,
        _lambda: Option<&[Number]>,
        _new_lambda: bool,
        mode: SparsityRequest<'_>,
    ) -> bool {
        match mode {
            SparsityRequest::Structure { irow, jcol } => {
                irow[0] = 0;
                jcol[0] = 0;
            }
            SparsityRequest::Values { values } => {
                let x = x.expect("eval_h(Values) without x");
                values[0] = obj_factor * 12.0 * x[0] * x[0];
            }
        }
        true
    }

    fn finalize_solution(&mut self, _sol: Solution<'_>, _d: &IpoptData, _q: &IpoptCq) {}
}

/// `min x` subject to `x >= 0`, from `x = 1`.
///
/// The other side of the coin: a linear objective against a single bound. The
/// iterate tracks `x ≈ mu`, so once `mu` bottoms out at the monotone floor
/// *nothing* moves — objective, error and step all pinned. That is a genuine
/// stall, the case acceptable-level termination exists to cut short, and the
/// progress test must stand aside there.
#[derive(Default)]
struct LinearAgainstBound;

impl TNLP for LinearAgainstBound {
    fn get_nlp_info(&mut self) -> Option<NlpInfo> {
        Some(NlpInfo {
            n: 1,
            m: 0,
            nnz_jac_g: 0,
            nnz_h_lag: 1,
            index_style: IndexStyle::C,
        })
    }

    fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
        b.x_l[0] = 0.0;
        b.x_u[0] = 2.0e19;
        true
    }

    fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
        sp.x[0] = 1.0;
        true
    }

    fn eval_f(&mut self, x: &[Number], _new_x: bool) -> Option<Number> {
        Some(x[0])
    }

    fn eval_grad_f(&mut self, _x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
        g[0] = 1.0;
        true
    }

    fn eval_g(&mut self, _x: &[Number], _new_x: bool, _g: &mut [Number]) -> bool {
        true
    }

    fn eval_jac_g(
        &mut self,
        _x: Option<&[Number]>,
        _new_x: bool,
        _mode: SparsityRequest<'_>,
    ) -> bool {
        true
    }

    fn eval_h(
        &mut self,
        _x: Option<&[Number]>,
        _new_x: bool,
        _obj_factor: Number,
        _lambda: Option<&[Number]>,
        _new_lambda: bool,
        mode: SparsityRequest<'_>,
    ) -> bool {
        match mode {
            SparsityRequest::Structure { irow, jcol } => {
                irow[0] = 0;
                jcol[0] = 0;
            }
            SparsityRequest::Values { values } => values[0] = 0.0,
        }
        true
    }

    fn finalize_solution(&mut self, _sol: Solution<'_>, _d: &IpoptData, _q: &IpoptCq) {}
}

struct Outcome {
    status: ApplicationReturnStatus,
    obj: Number,
    iters: usize,
}

fn solve(
    make: fn() -> Rc<RefCell<dyn TNLP>>,
    opts: &[(&str, Number)],
    ints: &[(&str, i32)],
) -> Outcome {
    let mut app = IpoptApplication::new();
    // Keep the objective scaling out of it: this file is about the streak, and
    // gradient-based scaling would drag the gh #200 masked-certificate veto into
    // the picture on a quartic (see `masked_certificate_veto.rs`).
    app.options_mut()
        .set_string_value("nlp_scaling_method", "none", true, false)
        .unwrap();
    for (k, v) in opts {
        app.options_mut()
            .set_numeric_value(k, *v, true, false)
            .unwrap();
    }
    for (k, v) in ints {
        app.options_mut()
            .set_integer_value(k, *v, true, false)
            .unwrap();
    }
    app.initialize().unwrap();
    let status = app.optimize_tnlp(make());
    let s = app.statistics();
    Outcome {
        status,
        obj: s.final_objective,
        iters: s.iteration_count as usize,
    }
}

fn quartic() -> Rc<RefCell<dyn TNLP>> {
    Rc::new(RefCell::new(Quartic))
}

fn linear_against_bound() -> Rc<RefCell<dyn TNLP>> {
    Rc::new(RefCell::new(LinearAgainstBound))
}

/// `tol` low enough that the strict gate can never fire, so the acceptable-level
/// streak is the only way out — the regime both reported models were in.
const UNREACHABLE_TOL: (&str, Number) = ("tol", 1e-30);
const PROGRESS_OFF: (&str, Number) = ("acceptable_progress_kappa", 0.0);

/// The bug, and the fix. With the progress test off, the streak's count runs out
/// while the objective is still falling by a factor of five per iteration and
/// the solve stops there. With it on, the streak is refused until the window
/// flattens — and the objective it returns is strictly better.
///
/// Measured on this branch: the opt-out stops at iteration 27 with objective
/// `9.5971884e-20`; the default stops at 29 with `3.7446734e-21`, 25× lower for
/// two extra iterations.
#[test]
fn a_still_descending_solve_is_not_stopped_by_the_streak() {
    let off = solve(quartic, &[UNREACHABLE_TOL, PROGRESS_OFF], &[]);
    let on = solve(quartic, &[UNREACHABLE_TOL], &[]);

    eprintln!(
        "progress test off: {:?} obj={:.6e} iters={}\n\
         progress test on : {:?} obj={:.6e} iters={}",
        off.status, off.obj, off.iters, on.status, on.obj, on.iters
    );

    // Guard the premise: without the progress test this must actually exhibit
    // the reported shape — an acceptable-level exit taken while the solve was
    // still descending — or the assertions below prove nothing.
    assert!(
        matches!(off.status, ApplicationReturnStatus::SolvedToAcceptableLevel),
        "premise: the opt-out run should exit through the streak, got {:?}",
        off.status
    );

    assert!(
        matches!(on.status, ApplicationReturnStatus::SolvedToAcceptableLevel),
        "the status must not regress, got {:?}",
        on.status
    );
    assert!(
        on.obj < off.obj,
        "the whole point: refusing a streak that had not flattened must return a \
         better objective (on {:.6e} vs off {:.6e})",
        on.obj,
        off.obj
    );
    // Continuing costs iterations — that is the trade, and it is bounded.
    assert!(
        on.iters > off.iters,
        "the refusal should have bought extra iterations (on {} vs off {})",
        on.iters,
        off.iters
    );
}

/// The bare count is recovered at *both* ends of `acceptable_progress_kappa`,
/// and recovered exactly — same iteration, same objective to the last bit.
///
/// `0` is the documented opt-out. A very large kappa is the other end of the
/// same statement: it makes every window count as flat, so the flat branch has
/// to reproduce upstream's criterion bit for bit. The two together pin the
/// mechanism as a pure *addition* to the count rather than a rewrite of it.
#[test]
fn the_bare_count_is_recovered_at_both_ends_of_kappa() {
    let off = solve(quartic, &[UNREACHABLE_TOL, PROGRESS_OFF], &[]);
    let everything_is_flat = solve(
        quartic,
        &[UNREACHABLE_TOL, ("acceptable_progress_kappa", 1e6)],
        &[],
    );
    assert!(matches!(
        off.status,
        ApplicationReturnStatus::SolvedToAcceptableLevel
    ));
    assert_eq!(
        std::mem::discriminant(&everything_is_flat.status),
        std::mem::discriminant(&off.status),
    );
    assert_eq!(
        everything_is_flat.iters, off.iters,
        "a flat-by-construction window must stop where the bare count stops"
    );
    assert_eq!(
        everything_is_flat.obj, off.obj,
        "…and at the same point (off {:.17e}, wide {:.17e})",
        off.obj, everything_is_flat.obj
    );
}

/// A solve that stalls to a genuine standstill must be untouched.
///
/// `x` tracks `mu` here, so once `mu` bottoms out at the monotone floor nothing
/// moves — objective, error and step all pinned — and the run leaves through the
/// tiny-step exit rather than the streak. This is the shape the issue reports for
/// `eigena2` / `eigenb2`, where `acceptable_iter = 0` changes nothing because
/// they stop for a different reason: the progress test must be equally invisible
/// to them.
#[test]
fn a_solve_that_stalls_to_a_standstill_is_untouched() {
    let off = solve(
        linear_against_bound,
        &[UNREACHABLE_TOL, PROGRESS_OFF],
        &[("max_iter", 200)],
    );
    let on = solve(
        linear_against_bound,
        &[UNREACHABLE_TOL],
        &[("max_iter", 200)],
    );

    eprintln!(
        "stalled, progress test off: {:?} obj={:.6e} iters={}\n\
         stalled, progress test on : {:?} obj={:.6e} iters={}",
        off.status, off.obj, off.iters, on.status, on.obj, on.iters
    );

    assert!(
        !matches!(
            off.status,
            ApplicationReturnStatus::MaximumIterationsExceeded
        ),
        "premise: this model should reach a standstill well inside the cap, got {:?}",
        off.status
    );
    assert_eq!(
        std::mem::discriminant(&on.status),
        std::mem::discriminant(&off.status),
        "a stalled solve must not have its status changed by the progress test \
         (off {:?}, on {:?})",
        off.status,
        on.status
    );
    assert_eq!(
        on.iters, off.iters,
        "nor its iteration count (off {}, on {})",
        off.iters, on.iters
    );
    assert_eq!(
        on.obj, off.obj,
        "nor the point it returns (off {:.17e}, on {:.17e})",
        off.obj, on.obj
    );
}

/// The safety property that lets the test run without predicting whether a
/// streak is really premature: if continuing does not pan out, the refused
/// iterate comes back with the status it would originally have had. Capping
/// `max_iter` just past the refusal forces exactly that path.
#[test]
fn a_refusal_that_does_not_pan_out_restores_the_refused_point() {
    let off = solve(quartic, &[UNREACHABLE_TOL, PROGRESS_OFF], &[]);
    let cap = (off.iters + 1) as i32;
    let capped = solve(quartic, &[UNREACHABLE_TOL], &[("max_iter", cap)]);

    eprintln!(
        "capped at {cap} iters: {:?} obj={:.6e} (refused point was {:.6e})",
        capped.status, capped.obj, off.obj
    );

    assert!(
        !matches!(
            capped.status,
            ApplicationReturnStatus::MaximumIterationsExceeded
        ),
        "a refusal cut short by max_iter must fall back to the refused iterate, \
         not surface a bare failure"
    );
    assert!(
        matches!(
            capped.status,
            ApplicationReturnStatus::SolvedToAcceptableLevel
                | ApplicationReturnStatus::SolveSucceeded
        ),
        "got {:?}",
        capped.status
    );
    assert!(
        capped.obj <= off.obj,
        "never worse than the point the streak would have returned (capped {:.6e} \
         vs off {:.6e})",
        capped.obj,
        off.obj
    );
}

/// An ordinary solve — one that reaches `tol` — must be untouched. The streak
/// never completes there, so the mechanism has nothing to look at, and this is
/// what bounds its blast radius on the corpus.
#[test]
fn the_progress_test_is_inert_on_an_ordinary_solve() {
    for model in [quartic, linear_against_bound] {
        let off = solve(model, &[PROGRESS_OFF], &[]);
        let on = solve(model, &[], &[]);
        assert_eq!(
            std::mem::discriminant(&on.status),
            std::mem::discriminant(&off.status),
            "status moved on a default solve (off {:?}, on {:?})",
            off.status,
            on.status
        );
        assert_eq!(
            on.iters, off.iters,
            "iteration count moved on a default solve (off {}, on {})",
            off.iters, on.iters
        );
        assert_eq!(
            on.obj, off.obj,
            "objective moved on a default solve (off {:.17e}, on {:.17e})",
            off.obj, on.obj
        );
    }
}
