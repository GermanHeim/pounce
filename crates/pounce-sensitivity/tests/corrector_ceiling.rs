//! The corrector on a base solve the gh#737 sigma ceiling touched, or
//! that crossed over into the gh#654 declared frame.
//!
//! In both of those cases the sensitivity layer stores a base-point
//! diagonal for the back-solves through the held factor, and
//! `barrier_sigma_x()` returns the stored copy. The corrector must
//! not use it: it factors its own operator at the predicted point,
//! so `corrector_sigma` rebuilds both diagonal blocks there, with
//! the frame rule and the ceiling re-derived at that point, and this
//! file pins that a correction on such solves acts instead of
//! stalling.
//!
//! The fixture is gh#737's cap trigger (a variable held between a
//! binding equality and its own bound, the multiplier split not
//! unique) welded to a curvature workload on a second parameter: the
//! perturbation moves `y` through `exp(y)` curvature, which the plain
//! step misses at second order and a correction closes.
//!
//! # Measured (2026-08-28), error against a tol 1e-10 re-solve
//!
//! ```text
//!                                  plain err   c1        c8
//!  ceiling engaged, frozen copy    1.48e-1     1.48e-1   1.48e-1   improved: false
//!  ceiling engaged, rebuilt        1.48e-1     1.05e-2   1.05e-2   improved: true
//!  bound removed (control)         1.48e-1     1.04e-2   5.0e-8    improved: true
//! ```
//!
//! The frozen row is the pre-`corrector_sigma` behaviour, the double
//! column's no-progress signature: the first iteration fails to
//! reduce the residual against the mixed operator and the
//! no-improvement rule keeps the handed step at every budget. The
//! rebuilt row's first iteration matches the control's, and the
//! plateau past it belongs to the degenerate rows themselves: the
//! trial residual at the non-unique multiplier split stops falling,
//! measured identically under a forced raw uncapped diagonal
//! (`6.9e27` on this geometry), so no diagonal choice moves it. The
//! control, with nothing degenerate, continues to the floor.

use std::cell::RefCell;
use std::rc::Rc;

use pounce_algorithm::application::IpoptApplication;
use pounce_common::types::{Index, Number};
use pounce_nlp::return_codes::ApplicationReturnStatus;
use pounce_nlp::tnlp::{
    BoundsInfo, IndexStyle, IpoptCq, IpoptData, Linearity, NlpInfo, Solution, SparsityRequest,
    StartingPoint, TNLP,
};
use pounce_sensitivity::Solver;

/// Where the parameter row holds `pv`, and equally `x`'s upper bound.
const PIN: Number = 1.0;
/// The second parameter row's base value.
const Q0: Number = 1.0;
/// The perturbation asked of the second parameter row.
const DELTA: Number = -0.45;

/// ```text
/// min  ½(x − t)² + ½(w − 1)² + exp(y) − q·y + ½(b − ½ − (q − Q0))²
/// s.t. g₀:  x − pv = 0
///      g₁:      pv = PIN      ← holds x on its own upper bound
///      g₂:  w − x  = 0
///      g₃:      q  = q0       ← the perturbed parameter row
///      0 ≤ x ≤ PIN,  0 ≤ b,  pv, w, q, y free
/// ```
///
/// The `x`/`pv`/`w` block is gh#737's cap trigger verbatim and is not
/// perturbed. The `q` block is the correction workload: at the base,
/// `y* = ln(Q0) = 0` and `b* = ½`; at `q0 + DELTA` the re-solve has
/// `y* = ln(0.55)` while the linear step reaches `ln'(1)·DELTA`, and
/// `b* = 0.05`, one tenth of its base slack.
struct CeilingWithWorkload {
    /// Objective target for `x`, above `PIN`. `2.0` trips the
    /// ceiling; `1e3` converges through a different multiplier split
    /// and lands two hundred times short of it (measured on gh#737's
    /// fixture, unchanged here).
    t: Number,
    /// The second parameter row's right-hand side.
    q0: Number,
    /// Whether `x` carries its upper bound. `false` removes only the
    /// bound, so the equality still holds `x` at `PIN` but nothing is
    /// degenerate and no ceiling can engage.
    bounded: bool,
    /// Starting point override, for warm-starting the truth solve.
    start: Option<Vec<Number>>,
}

impl TNLP for CeilingWithWorkload {
    fn get_nlp_info(&mut self) -> Option<NlpInfo> {
        Some(NlpInfo {
            n: 6,
            m: 4,
            nnz_jac_g: 6,
            nnz_h_lag: 7,
            index_style: IndexStyle::C,
        })
    }

    fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
        // x, pv, w, q, y, b
        b.x_l[0] = 0.0;
        b.x_u[0] = if self.bounded { PIN } else { 1.0e19 };
        for i in 1..5 {
            b.x_l[i] = -1.0e19;
            b.x_u[i] = 1.0e19;
        }
        b.x_l[5] = 0.0;
        b.x_u[5] = 1.0e19;
        b.g_l[0] = 0.0;
        b.g_u[0] = 0.0;
        b.g_l[1] = PIN;
        b.g_u[1] = PIN;
        b.g_l[2] = 0.0;
        b.g_u[2] = 0.0;
        b.g_l[3] = self.q0;
        b.g_u[3] = self.q0;
        true
    }

    fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
        match &self.start {
            Some(x0) => sp.x.copy_from_slice(x0),
            None => {
                sp.x[..3].fill(0.5);
                sp.x[3] = Q0;
                sp.x[4] = 0.0;
                sp.x[5] = 0.5;
            }
        }
        true
    }

    fn get_constraints_linearity(&mut self, types: &mut [Linearity]) -> bool {
        types.fill(Linearity::Linear);
        true
    }

    fn eval_f(&mut self, x: &[Number], _new_x: bool) -> Option<Number> {
        let d = x[5] - 0.5 - (x[3] - Q0);
        Some(
            0.5 * (x[0] - self.t).powi(2) + 0.5 * (x[2] - 1.0).powi(2) + x[4].exp() - x[3] * x[4]
                + 0.5 * d * d,
        )
    }

    fn eval_grad_f(&mut self, x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
        let d = x[5] - 0.5 - (x[3] - Q0);
        g[0] = x[0] - self.t;
        g[1] = 0.0;
        g[2] = x[2] - 1.0;
        g[3] = -x[4] - d;
        g[4] = x[4].exp() - x[3];
        g[5] = d;
        true
    }

    fn eval_g(&mut self, x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
        g[0] = x[0] - x[1];
        g[1] = x[1];
        g[2] = x[2] - x[0];
        g[3] = x[3];
        true
    }

    fn eval_jac_g(
        &mut self,
        _x: Option<&[Number]>,
        _new_x: bool,
        mode: SparsityRequest<'_>,
    ) -> bool {
        match mode {
            SparsityRequest::Structure { irow, jcol } => {
                for (k, &(r, c)) in [(0, 0), (0, 1), (1, 1), (2, 2), (2, 0), (3, 3)]
                    .iter()
                    .enumerate()
                {
                    irow[k] = r as Index;
                    jcol[k] = c as Index;
                }
            }
            SparsityRequest::Values { values } => {
                values.copy_from_slice(&[1.0, -1.0, 1.0, 1.0, -1.0, 1.0]);
            }
        }
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
        // lower triangle: (0,0) (2,2) (3,3) (4,4) (5,5) (4,3) (5,3)
        match mode {
            SparsityRequest::Structure { irow, jcol } => {
                for (k, &(r, c)) in [(0, 0), (2, 2), (3, 3), (4, 4), (5, 5), (4, 3), (5, 3)]
                    .iter()
                    .enumerate()
                {
                    irow[k] = r as Index;
                    jcol[k] = c as Index;
                }
            }
            SparsityRequest::Values { values } => {
                let y = x.map_or(0.0, |x| x[4]);
                values[0] = obj_factor;
                values[1] = obj_factor;
                values[2] = obj_factor;
                values[3] = obj_factor * y.exp();
                values[4] = obj_factor;
                values[5] = -obj_factor;
                values[6] = -obj_factor;
            }
        }
        true
    }

    fn finalize_solution(&mut self, _s: Solution<'_>, _d: &IpoptData, _c: &IpoptCq) {}
}

fn solved(
    t: Number,
    q0: Number,
    bounded: bool,
    crossover: bool,
    start: Option<Vec<Number>>,
    tol: Option<Number>,
) -> Solver {
    let mut app = IpoptApplication::new();
    {
        let o = app.options_mut();
        o.set_integer_value("print_level", 0, true, false).unwrap();
        o.set_string_value("sb", "yes", true, false).unwrap();
        o.set_numeric_value("bound_relax_factor", 0.0, true, false)
            .unwrap();
        o.set_string_value(
            "crossover",
            if crossover { "yes" } else { "no" },
            true,
            false,
        )
        .unwrap();
        if let Some(tol) = tol {
            o.set_numeric_value("tol", tol, true, false).unwrap();
        }
    }
    app.initialize().unwrap();
    let tnlp: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(CeilingWithWorkload {
        t,
        q0,
        bounded,
        start,
    }));
    let mut solver = Solver::new(app, tnlp);
    let status = solver.solve();
    assert!(
        matches!(
            status,
            ApplicationReturnStatus::SolveSucceeded
                | ApplicationReturnStatus::SolvedToAcceptableLevel
        ),
        "fixture must converge (t={t:e}, q0={q0}); got {status:?}",
    );
    solver
}

/// The perturbed problem's solution at tol 1e-10, warm-started from
/// the base point, the same truth the re-solve oracle uses.
fn truth(t: Number, bounded: bool, base_x: &[Number]) -> Vec<Number> {
    let s = solved(
        t,
        Q0 + DELTA,
        bounded,
        false,
        Some(base_x.to_vec()),
        Some(1e-10),
    );
    s.converged().expect("truth converged").x.clone()
}

fn dist(a: &[Number], b: &[Number]) -> Number {
    a.iter()
        .zip(b)
        .fold(0.0_f64, |m, (&x, &y)| m.max((x - y).abs()))
}

fn add(base: &[Number], step: &[Number]) -> Vec<Number> {
    base.iter().zip(step).map(|(&b, &s)| b + s).collect()
}

/// A base solve the ceiling touched corrects like any other: the
/// diagonal is rebuilt at the predicted point with the ceiling
/// re-derived there, never the stored base-point copy.
///
/// The first iteration takes the error down an order and the residual
/// fifteenfold, matching the no-ceiling control's first iteration.
/// The iterations then stop: the trial residual at the degenerate
/// rows, where the multiplier split between the bound and the
/// equality is not unique, does not fall further, and the measured
/// plateau is identical under a forced raw uncapped diagonal, so it
/// is the degeneracy's own property and no diagonal choice moves it.
/// The budget-8 error equals the budget-1 error here, pinned as a
/// measured fact rather than a target.
#[test]
fn a_ceiling_solve_corrects_at_the_predicted_point() {
    let solver = solved(2.0, Q0, true, false, None, None);
    let report = solver.classify_activity().expect("activity report");
    assert!(
        report.var_sigma[0] > 1e10 && report.var_sigma[0] < 1e20,
        "the fixture only says anything with the ceiling engaged: \
         sigma on x is {:e}",
        report.var_sigma[0],
    );
    let base = solver.converged().expect("converged").x.clone();
    let want = truth(2.0, true, &base);
    let step = solver
        .parametric_step_full(&[3], &[DELTA])
        .expect("full step");
    let n = base.len();
    let plain = dist(&add(&base, &step[..n]), &want);
    assert!(
        plain > 1e-1,
        "the workload must leave the plain step something to close: {plain:e}",
    );
    for budget in [1_usize, 8] {
        let (out, rep) = solver
            .correct_step(&[3], &[DELTA], &step, budget)
            .expect("corrector");
        let corrected = dist(&add(&base, &out[..n]), &want);
        assert!(
            rep.improved() && corrected < plain / 10.0,
            "budget {budget}: the predicted-point diagonal must let the \
             correction act: {plain:e} -> {corrected:e} (residual {:e} \
             -> {:e})",
            rep.initial_residual,
            rep.residual,
        );
    }
}

/// The identical workload with only the degenerate bound removed has
/// no ceiling and nothing degenerate, and converges to the floor: the
/// control separating the workload from the degenerate rows' plateau
/// above.
#[test]
fn the_same_workload_without_the_ceiling_corrects() {
    let solver = solved(2.0, Q0, false, false, None, None);
    let base = solver.converged().expect("converged").x.clone();
    let want = truth(2.0, false, &base);
    let step = solver
        .parametric_step_full(&[3], &[DELTA])
        .expect("full step");
    let n = base.len();
    let plain = dist(&add(&base, &step[..n]), &want);
    let (out, rep) = solver
        .correct_step(&[3], &[DELTA], &step, 8)
        .expect("corrector");
    let corrected = dist(&add(&base, &out[..n]), &want);
    assert!(
        rep.improved() && corrected < 1e-6 && corrected < plain * 1e-5,
        "the live operator closes the curvature: {plain:e} -> {corrected:e} \
         (residual {:e} -> {:e})",
        rep.initial_residual,
        rep.residual,
    );
}

/// The other way the stored diagonal freezes: a solve that crossed
/// over into the declared frame. `corrector_sigma` re-derives that
/// frame at the predicted point (the declared bounds are constants,
/// so the frame follows the iterate), and the correction acts the
/// same way it does on the interior ceiling solve above.
#[test]
fn a_crossover_solve_corrects_at_the_predicted_point() {
    let solver = solved(2.0, Q0, true, true, None, None);
    let base = solver.converged().expect("converged").x.clone();
    let want = truth(2.0, true, &base);
    let step = solver
        .parametric_step_full(&[3], &[DELTA])
        .expect("full step");
    let n = base.len();
    let plain = dist(&add(&base, &step[..n]), &want);
    assert!(
        plain > 1e-1,
        "the workload must leave the plain step something to close: {plain:e}",
    );
    let (out, rep) = solver
        .correct_step(&[3], &[DELTA], &step, 8)
        .expect("corrector");
    let corrected = dist(&add(&base, &out[..n]), &want);
    assert!(
        rep.improved() && corrected < plain / 10.0,
        "the declared frame must follow the iterate: {plain:e} ->          {corrected:e} (residual {:e} -> {:e})",
        rep.initial_residual,
        rep.residual,
    );
}
