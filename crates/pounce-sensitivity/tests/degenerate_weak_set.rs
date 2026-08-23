//! Membership of the degeneracy weak set, on the two rows that must
//! stay out of it.
//!
//! An interior coordinate with low curvature is not a kink, however
//! its barrier-weight-to-curvature ratio classifies: its own barrier
//! diagonal is `O(mu)`, orders of magnitude under the weak band. And a
//! coordinate an equality pins cannot be decided by a pin force: its
//! own diagonal entry of the reduced `S` is zero, so the row is
//! dropped from the engaged set rather than handed to the QP, which
//! would be unbounded along it.

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
/// Soft curvature: above the classifier's unidentified floor, small
/// enough that the interior coordinate's ratio lands in the gap the
/// old membership admitted.
const EPS_SOFT: Number = 1.0e-7;
/// The pinned coordinate's distance from its bound: `mu / S0^2` sits
/// inside the weak band for the final `mu` a tol of 1e-8 reaches.
const S0: Number = 1.0e-5;

/// `x1` weakly active at its lower bound, `xs` interior at 5 with
/// curvature `EPS_SOFT`:
///
/// ```text
/// min 0.5 x1^2 - A1 x3 x1 + 0.5 EPS_SOFT (xs - 5)^2
/// s.t. x3 = p,   0 <= x1 <= 10,   0 <= xs <= 10
/// ```
struct SoftInteriorQp;

impl TNLP for SoftInteriorQp {
    fn get_nlp_info(&mut self) -> Option<NlpInfo> {
        Some(NlpInfo {
            n: 3,
            m: 1,
            nnz_jac_g: 1,
            nnz_h_lag: 3,
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
        sp.x[1] = 5.0;
        sp.x[2] = 0.0;
        true
    }

    fn eval_f(&mut self, x: &[Number], _new_x: bool) -> Option<Number> {
        let (x1, xs, p) = (x[0], x[1], x[2]);
        Some(0.5 * x1 * x1 - A1 * p * x1 + 0.5 * EPS_SOFT * (xs - 5.0) * (xs - 5.0))
    }

    fn eval_grad_f(&mut self, x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
        let (x1, xs, p) = (x[0], x[1], x[2]);
        g[0] = x1 - A1 * p;
        g[1] = EPS_SOFT * (xs - 5.0);
        g[2] = -A1 * x1;
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
                let rs: [Index; 3] = [0, 1, 2];
                let cs: [Index; 3] = [0, 1, 0];
                irow.copy_from_slice(&rs);
                jcol.copy_from_slice(&cs);
            }
            SparsityRequest::Values { values } => {
                values[0] = obj_factor;
                values[1] = obj_factor * EPS_SOFT;
                values[2] = -obj_factor * A1;
            }
        }
        true
    }

    fn finalize_solution(&mut self, _s: Solution<'_>, _d: &IpoptData, _q: &IpoptCq) {}
}

/// `x1` weakly active at its lower bound, `xp` held at `S0` above its
/// lower bound by an equality of its own:
///
/// ```text
/// min 0.5 x1^2 - A1 x3 x1
/// s.t. x3 = p,   xp = S0,   0 <= x1 <= 10,   0 <= xp <= 10
/// ```
struct PinnedNearBoundQp;

impl TNLP for PinnedNearBoundQp {
    fn get_nlp_info(&mut self) -> Option<NlpInfo> {
        Some(NlpInfo {
            n: 3,
            m: 2,
            nnz_jac_g: 2,
            nnz_h_lag: 2,
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
        b.g_l[1] = S0;
        b.g_u[1] = S0;
        true
    }

    fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
        sp.x[0] = 0.3;
        sp.x[1] = 0.1;
        sp.x[2] = 0.0;
        true
    }

    fn eval_f(&mut self, x: &[Number], _new_x: bool) -> Option<Number> {
        let (x1, p) = (x[0], x[2]);
        Some(0.5 * x1 * x1 - A1 * p * x1)
    }

    fn eval_grad_f(&mut self, x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
        let (x1, p) = (x[0], x[2]);
        g[0] = x1 - A1 * p;
        g[1] = 0.0;
        g[2] = -A1 * x1;
        true
    }

    fn eval_g(&mut self, x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
        g[0] = x[2];
        g[1] = x[1];
        true
    }

    fn eval_jac_g(&mut self, _x: Option<&[Number]>, _nx: bool, mode: SparsityRequest<'_>) -> bool {
        match mode {
            SparsityRequest::Structure { irow, jcol } => {
                irow[0] = 0;
                jcol[0] = 2;
                irow[1] = 1;
                jcol[1] = 1;
            }
            SparsityRequest::Values { values } => {
                values[0] = 1.0;
                values[1] = 1.0;
            }
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
                let rs: [Index; 2] = [0, 2];
                let cs: [Index; 2] = [0, 0];
                irow.copy_from_slice(&rs);
                jcol.copy_from_slice(&cs);
            }
            SparsityRequest::Values { values } => {
                values[0] = obj_factor;
                values[1] = -obj_factor * A1;
            }
        }
        true
    }

    fn finalize_solution(&mut self, _s: Solution<'_>, _d: &IpoptData, _q: &IpoptCq) {}
}

fn solved(tnlp: Rc<RefCell<dyn TNLP>>) -> Solver {
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
    let mut solver = Solver::new(app, tnlp);
    let status = solver.solve();
    assert!(
        matches!(status, ApplicationReturnStatus::SolveSucceeded),
        "base solve failed: {status:?}",
    );
    solver
}

#[test]
fn an_interior_low_curvature_coordinate_is_not_weak() {
    let solver = solved(Rc::new(RefCell::new(SoftInteriorQp)));
    let x = solver.converged().expect("converged").x.clone();
    assert!(x[0].abs() < 1e-3, "x1 sits on its bound, got {}", x[0]);
    assert!((x[1] - 5.0).abs() < 1e-3, "xs sits mid-range, got {}", x[1]);
    let weak: Vec<usize> = solver
        .weakly_active_bounds()
        .expect("classification")
        .iter()
        .map(|w| w.var_row)
        .collect();
    assert!(weak.contains(&0), "the kink at x1 is weak: {weak:?}");
    assert!(
        !weak.contains(&1),
        "xs is 5.0 from either bound and must not be weak: {weak:?}"
    );
}

#[test]
fn a_row_an_equality_pins_is_dropped_not_decided() {
    let solver = solved(Rc::new(RefCell::new(PinnedNearBoundQp)));
    let x = solver.converged().expect("converged").x.clone();
    assert!((x[1] - S0).abs() < 1e-9, "xp pinned at S0, got {}", x[1]);

    // The setup must still exercise the path: xp's bound is inside the
    // weak band, so it is in the weak set and the perturbation below
    // engages it.
    let weak: Vec<usize> = solver
        .weakly_active_bounds()
        .expect("classification")
        .iter()
        .map(|w| w.var_row)
        .collect();
    assert!(
        weak.contains(&1),
        "xp must be in the weak set for this test to test anything: {weak:?}"
    );

    // Pin row 0 moves toward x1's hold side, pin row 1 moves xp toward
    // its bound. The QP cannot decide xp, no force moves a coordinate
    // an equality owns, so the row is dropped and the rest is decided.
    let (d, held, _spent) = solver
        .parametric_step_directional(&[0, 1], &[-1.0, -1.0e-3], 16)
        .expect("the decision completes with the pinned row dropped");
    assert!(
        held.contains(&0) && !held.contains(&1),
        "x1 held, xp not decided: held {held:?}"
    );
    assert!(d[0].abs() < 1e-8, "x1 holds at its bound, got {}", d[0]);
    assert!(
        (d[1] + 1.0e-3).abs() < 1e-8,
        "xp moves by its pin's shift, got {}",
        d[1]
    );
}
