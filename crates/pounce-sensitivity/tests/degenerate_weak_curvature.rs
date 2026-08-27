//! Two identical kinks that differ only in curvature must both be in
//! the weak set, and the directional step must move both exactly.
//!
//! At a kink the multiplier comes from the curvature along the
//! coordinate. Stationarity gives `z = H s`, and `s z = mu` then puts
//! `sigma = z / s` at `H`, so any membership band on `sigma` alone is
//! a band on curvature and misses a kink whose curvature lies outside
//! it. The fixture holds two coordinates at the same kink with
//! curvature `1` and `1e4`. The exact directional step moves both by
//! `A1 * dp`.

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

const A1: Number = 1.10;
/// The stiff coordinate's curvature. Its `sigma` at the kink is `C`.
const C: Number = 1.0e4;

/// `x0` and `x1` both at the kink of their lower bound, differing
/// only in curvature:
///
/// ```text
/// min 0.5 x0^2 - A1 p x0 + 0.5 C x1^2 - A1 C p x1
/// s.t. p = 0,   0 <= x0 <= 10,   0 <= x1 <= 10
/// ```
///
/// At `p = 0` both sit at zero with slack and multiplier vanishing
/// together. For `dp = +1e-3` both leave the bound and the exact step
/// is `dx0 = dx1 = A1 dp`.
struct TwoCurvatureKinks;

impl TNLP for TwoCurvatureKinks {
    fn get_nlp_info(&mut self) -> Option<NlpInfo> {
        Some(NlpInfo {
            n: 3,
            m: 1,
            nnz_jac_g: 1,
            nnz_h_lag: 4,
            index_style: IndexStyle::C,
        })
    }

    fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
        b.x_l[0] = 0.0;
        b.x_u[0] = 10.0;
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
        let (x0, x1, p) = (x[0], x[1], x[2]);
        Some(0.5 * x0 * x0 - A1 * p * x0 + 0.5 * C * x1 * x1 - A1 * C * p * x1)
    }

    fn eval_grad_f(&mut self, x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
        let (x0, x1, p) = (x[0], x[1], x[2]);
        g[0] = x0 - A1 * p;
        g[1] = C * x1 - A1 * C * p;
        g[2] = -A1 * x0 - A1 * C * x1;
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
                let rs: [Index; 4] = [0, 1, 2, 2];
                let cs: [Index; 4] = [0, 1, 0, 1];
                irow.copy_from_slice(&rs);
                jcol.copy_from_slice(&cs);
            }
            SparsityRequest::Values { values } => {
                values[0] = obj_factor;
                values[1] = obj_factor * C;
                values[2] = -obj_factor * A1;
                values[3] = -obj_factor * A1 * C;
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
        .set_numeric_value("tol", 1e-8, true, false)
        .unwrap();
    app.options_mut()
        .set_numeric_value("bound_relax_factor", 0.0, true, false)
        .unwrap();
    app.initialize().unwrap();
    let mut solver = Solver::new(app, Rc::new(RefCell::new(TwoCurvatureKinks)));
    let status = solver.solve();
    assert!(
        matches!(status, ApplicationReturnStatus::SolveSucceeded),
        "base solve failed: {status:?}",
    );
    solver
}

#[test]
fn a_kink_is_weak_at_any_curvature() {
    let solver = solved();
    let weak: Vec<usize> = solver
        .weakly_active_bounds()
        .expect("classification")
        .iter()
        .map(|w| w.var_row)
        .collect();
    assert!(
        weak.contains(&0) && weak.contains(&1),
        "both kinks are weak whatever their curvature: {weak:?}"
    );

    let dp = 1.0e-3;
    let (d, _held, _spent) = solver
        .parametric_step_directional(&[0], &[dp], 16)
        .expect("the decision completes");
    let exact = A1 * dp;
    assert!(
        (d[0] - exact).abs() < 1e-8,
        "x0 leaves its bound by A1 dp, got {}",
        d[0]
    );
    assert!(
        (d[1] - exact).abs() < 1e-8,
        "x1 leaves its bound by A1 dp whatever its curvature, got {}",
        d[1]
    );
}

/// A kink's decision is a directional derivative, linear in the
/// step, so the holding side holds at a step far below the barrier
/// width, not only at steps that cover the remaining slack.
#[test]
fn a_kink_decides_by_direction_at_any_scale() {
    let solver = solved();
    let dp = -1.0e-10;
    let (d, held, _spent) = solver
        .parametric_step_directional(&[0], &[dp], 16)
        .expect("the decision completes");
    assert!(
        held.contains(&0) && held.contains(&1),
        "both kinks hold on the holding side: held {held:?}"
    );
    assert!(d[0].abs() < 1e-12, "x0 holds at a tiny step, got {}", d[0]);
    assert!(d[1].abs() < 1e-12, "x1 holds at a tiny step, got {}", d[1]);
}
