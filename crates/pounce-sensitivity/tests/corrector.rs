//! The corrector on a QP whose exact answer is known.
//!
//! ```text
//! min 0.5 x1^2 + 0.5 x2^2 + G x1 x2 - a(p) x1 - b(p) x2
//! s.t. x3 = p,   a(p) = 0.18 + 1.10 x3,   b(p) = -0.29 + 0.11 x3
//!      0 <= x1 <= 1,   0 <= x2 <= 10
//! ```
//!
//! The objective is quadratic and the constraint linear, so the
//! solution is exactly linear in `p` while the active set holds, and
//! the parametric step is already exact. That makes the fixture a
//! check that the corrector does no harm where there is nothing to
//! correct: the residual it reports must be at the barrier's own
//! floor, and the step must come back unchanged.
//!
//! Moving `p` far enough to change the active set gives the opposite
//! case. At `p = 1` the true solution has `x1` on its upper bound,
//! which the base point's factor holds nothing against, so the step
//! needs correcting and the corrector has room to work.

use std::cell::RefCell;
use std::rc::Rc;

use pounce_algorithm::application::IpoptApplication;
use pounce_common::types::{Index, Number};
use pounce_nlp::TNLP;
use pounce_nlp::return_codes::ApplicationReturnStatus;
use pounce_nlp::tnlp::{
    BoundsInfo, IndexStyle, IpoptCq, IpoptData, NlpInfo, Solution, SparsityRequest, StartingPoint,
};
use pounce_sensitivity::Solver;

const G: Number = -0.28;
const A0: Number = 0.18;
const A1: Number = 1.10;
const B0: Number = -0.29;
const B1: Number = 0.11;

struct ParamQp;

impl TNLP for ParamQp {
    fn get_nlp_info(&mut self) -> Option<NlpInfo> {
        Some(NlpInfo {
            n: 3,
            m: 1,
            nnz_jac_g: 1,
            nnz_h_lag: 5,
            index_style: IndexStyle::C,
        })
    }

    fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
        b.x_l[0] = 0.0;
        b.x_u[0] = 1.0;
        b.x_l[1] = 0.0;
        b.x_u[1] = 10.0;
        b.x_l[2] = -1.0e19;
        b.x_u[2] = 1.0e19;
        b.g_l[0] = 0.0;
        b.g_u[0] = 0.0;
        true
    }

    fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
        sp.x[0] = 0.3;
        sp.x[1] = 0.3;
        sp.x[2] = 0.0;
        true
    }

    fn eval_f(&mut self, x: &[Number], _new_x: bool) -> Option<Number> {
        let (x1, x2, p) = (x[0], x[1], x[2]);
        let a = A0 + A1 * p;
        let b = B0 + B1 * p;
        Some(0.5 * x1 * x1 + 0.5 * x2 * x2 + G * x1 * x2 - a * x1 - b * x2)
    }

    fn eval_grad_f(&mut self, x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
        let (x1, x2, p) = (x[0], x[1], x[2]);
        g[0] = x1 + G * x2 - (A0 + A1 * p);
        g[1] = x2 + G * x1 - (B0 + B1 * p);
        g[2] = -A1 * x1 - B1 * x2;
        true
    }

    fn eval_g(&mut self, x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
        g[0] = x[2];
        true
    }

    fn eval_jac_g(&mut self, _x: Option<&[Number]>, _nx: bool, mode: SparsityRequest<'_>) -> bool {
        match mode {
            SparsityRequest::Structure { irow, jcol } => {
                irow[0] = 0;
                jcol[0] = 2;
            }
            SparsityRequest::Values { values } => values[0] = 1.0,
        }
        true
    }

    fn eval_h(
        &mut self,
        _x: Option<&[Number]>,
        _new_x: bool,
        obj_factor: Number,
        _lambda: Option<&[Number]>,
        _new_lambda: bool,
        mode: SparsityRequest<'_>,
    ) -> bool {
        match mode {
            SparsityRequest::Structure { irow, jcol } => {
                let rs: [Index; 5] = [0, 1, 1, 2, 2];
                let cs: [Index; 5] = [0, 1, 0, 0, 1];
                irow.copy_from_slice(&rs);
                jcol.copy_from_slice(&cs);
            }
            SparsityRequest::Values { values } => {
                values[0] = obj_factor;
                values[1] = obj_factor;
                values[2] = obj_factor * G;
                values[3] = -obj_factor * A1;
                values[4] = -obj_factor * B1;
            }
        }
        true
    }

    fn finalize_solution(&mut self, _s: Solution<'_>, _d: &IpoptData, _q: &IpoptCq) {}
}

fn solved() -> Solver {
    let mut app = IpoptApplication::new();
    app.options_mut()
        .set_integer_value("print_level", 0, true, false)
        .unwrap();
    app.options_mut()
        .set_string_value("sb", "yes", true, false)
        .unwrap();
    app.options_mut()
        .set_numeric_value("tol", 1e-10, true, false)
        .unwrap();
    app.options_mut()
        .set_numeric_value("bound_relax_factor", 0.0, true, false)
        .unwrap();
    app.initialize().unwrap();
    let tnlp: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(ParamQp));
    let mut solver = Solver::new(app, tnlp);
    let status = solver.solve();
    assert!(
        matches!(status, ApplicationReturnStatus::SolveSucceeded),
        "base solve failed: {status:?}",
    );
    solver
}

#[test]
fn the_corrector_leaves_an_exact_step_alone() {
    // A small move keeps the active set, where the quadratic model
    // makes the parametric step exact. There is nothing to correct, so
    // the residual must start at the barrier floor and the returned
    // step must match the one handed in.
    let solver = solved();
    let step = solver
        .parametric_step_full(&[0], &[0.05])
        .expect("parametric step");
    let (out, report) = solver
        .correct_step(&[0], &[0.05], &step, 8)
        .expect("corrector");
    assert_eq!(out.len(), step.len());
    assert!(
        report.initial_residual < 1e-6,
        "an exact step should start at the barrier floor, got {}",
        report.initial_residual,
    );
    assert!(
        report.residual <= report.initial_residual,
        "the corrector must not make the residual worse: {} -> {}",
        report.initial_residual,
        report.residual,
    );
    let moved = out
        .iter()
        .zip(&step)
        .fold(0.0_f64, |a, (&o, &s)| a.max((o - s).abs()));
    assert!(
        moved < 1e-6,
        "an exact step should come back unchanged, moved by {moved}",
    );
}

#[test]
fn the_corrector_reduces_the_residual_where_the_step_is_wrong() {
    // p = 1 carries x1 to its upper bound, so the linear step is well
    // off and leaves a residual the held factor can work on.
    let solver = solved();
    let step = solver
        .parametric_step_full(&[0], &[1.0])
        .expect("parametric step");
    let (out, report) = solver
        .correct_step(&[0], &[1.0], &step, 12)
        .expect("corrector");
    assert!(
        report.initial_residual > 1e-6,
        "this step should leave a residual to work on, got {}",
        report.initial_residual,
    );
    assert!(
        report.improved(),
        "the corrector should reduce the residual: {} -> {} in {} iteration(s)",
        report.initial_residual,
        report.residual,
        report.iterations,
    );
    assert!(report.iterations >= 1);
    // every corrected point stays inside the variable bounds
    assert!(
        out[0] + solver.converged().expect("converged").x[0] <= 1.0 + 1e-9,
        "x1 left its upper bound",
    );
}

#[test]
fn a_zero_budget_measures_the_step_without_iterating() {
    // Zero iterations still puts the point inside the bounds, since
    // that is what the returned point guarantees and what the residual
    // is defined at. So this costs one evaluation, no back-solve, and
    // reports how far the step is from satisfying the barrier system.
    let solver = solved();
    let base = solver.converged().expect("converged").x.clone();
    let step = solver
        .parametric_step_full(&[0], &[1.0])
        .expect("parametric step");
    assert!(
        base[0] + step[0] > 1.0,
        "this step should carry x1 past its upper bound, to {}",
        base[0] + step[0],
    );
    let (out, report) = solver
        .correct_step(&[0], &[1.0], &step, 0)
        .expect("corrector");
    assert_eq!(report.iterations, 0);
    assert_eq!(report.residual, report.initial_residual);
    assert!(!report.improved());
    assert!(
        report.residual > 0.0,
        "a zero budget should still report the step's residual",
    );
    assert!(
        base[0] + out[0] <= 1.0,
        "the returned point must satisfy the bounds, x1 at {}",
        base[0] + out[0],
    );
}

#[test]
fn the_corrector_rejects_a_step_of_the_wrong_length() {
    let solver = solved();
    let err = solver
        .correct_step(&[0], &[1.0], &[0.0; 3], 4)
        .expect_err("a short step should be refused");
    let msg = format!("{err:?}");
    assert!(msg.contains("step"), "unhelpful error: {msg}");
}
