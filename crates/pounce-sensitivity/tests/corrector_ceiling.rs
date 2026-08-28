//! The corrector on a base solve that trips the gh#737 sigma ceiling.
//!
//! When the ceiling caps at least one entry, the sensitivity layer
//! stores the whole capped diagonal, and `barrier_sigma_x()` returns
//! that stored copy instead of reading the current iterate. During a
//! correction the current iterate is the predicted point, so on such
//! a solve the corrector factors the base point's diagonal against
//! predicted-point derivatives: the mixed operator the change that
//! moved the corrector to the predicted point measured making no
//! progress on the double column. This file pins what that mixture
//! does to a correction here, against the same workload with the
//! degenerate bound removed, which takes the live path.
//!
//! The fixture is gh#737's cap trigger (a variable held between a
//! binding equality and its own bound, the multiplier split not
//! unique) welded to a curvature workload on a second parameter: the
//! perturbation moves `y` through `exp(y)` curvature, which the plain
//! step misses at second order and a correction closes.
//!
//! # Measured (2026-08-28, instrumented `barrier_sigma_x`)
//!
//! ```text
//!                       branch   plain err   c1        c2        c8
//!  ceiling engaged      frozen   1.48e-1     1.48e-1   1.48e-1   1.48e-1   improved: false
//!  bound removed        live     1.48e-1     1.04e-2   1.38e-3   5.0e-8    improved: true
//! ```
//!
//! On the frozen branch the first iteration fails to reduce the
//! residual and the no-improvement rule keeps the handed step, at
//! every budget: the double column's signature. Forcing the live
//! branch on the same capped fixture recovers the correction in
//! proportion to what the raw diagonal permits: fully on a geometry
//! whose uncapped entry is `7.1e15` (to `1.8e-7` at budget 8), and
//! only the first iteration (to `1.0e-2`, then flat) on this one,
//! whose uncapped entry is `6.9e27`, past what the factorization can
//! carry, which is why the ceiling exists. A fix therefore needs the
//! live rebuild plus the ceiling re-derived at the predicted point,
//! not raw `z/s`, and the first test below is pinned deliberately:
//! that fix flips it, and should update it on purpose.

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
    let s = solved(t, Q0 + DELTA, bounded, Some(base_x.to_vec()), Some(1e-10));
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

/// A base solve the ceiling touched hands the corrector the frozen
/// base-point diagonal, and the correction achieves nothing: the
/// first iteration fails to reduce the residual against the mixed
/// operator, the no-improvement rule keeps the handed step, and
/// `improved()` says so at every budget.
///
/// Pinned deliberately: a corrector that closes this workload on a
/// ceiling-engaged solve has gained a predicted-point diagonal with
/// the ceiling re-derived there, and this test should then be
/// updated on purpose. The module doc carries the forced-live
/// measurement saying what to expect from such a fix.
#[test]
fn a_ceiling_solve_stalls_the_correction() {
    let solver = solved(2.0, Q0, true, None, None);
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
        assert!(
            !rep.improved(),
            "budget {budget}: the mixed operator finds no reduction and \
             must say so: residual {:e} -> {:e}",
            rep.initial_residual,
            rep.residual,
        );
        let corrected = dist(&add(&base, &out[..n]), &want);
        assert!(
            (corrected - plain).abs() < 1e-8,
            "budget {budget}: a correction that achieves nothing leaves \
             the estimate where it was: {plain:e} -> {corrected:e}",
        );
    }
}

/// The identical workload with only the degenerate bound removed
/// takes the live branch and converges, which pins the stall above on
/// the frozen diagonal rather than on the workload or the equality
/// block.
#[test]
fn the_same_workload_without_the_ceiling_corrects() {
    let solver = solved(2.0, Q0, false, None, None);
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
