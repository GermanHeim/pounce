//! The released solve, measured directly against hand algebra.
//!
//! A two-variable QP with a parameter, small enough that the direction
//! of the solution in the parameter is computable by hand:
//!
//! ```text
//! min 0.5 x1^2 + 0.5 x2^2 + g x1 x2 - a(p) x1 - b(p) x2
//! s.t. x3 = p,   a(p) = 0.18 + 1.10 x3,   b(p) = -0.29 + 0.11 x3
//!      0 <= x1 <= 1,   0 <= x2 <= 10
//! ```
//!
//! At `p = 0` the solve puts `x2` on its lower bound and leaves `x1`
//! interior. With `x2` held there, `dx/dp = [a'(p), 0]`. With `x2`
//! released, the free system `H dx = [a', b']` gives
//! `dx = [1.2266, 0.4536]` at `g = -0.28`.
//!
//! `solve()` against the held factor must give the first direction,
//! and `solve_released` on `x2`'s lower-bound row must give the
//! second. The pinned direction has always been right. The released
//! one is what this file measures.

use std::cell::RefCell;
use std::rc::Rc;

use pounce_algorithm::application::IpoptApplication;
use pounce_common::types::{Index, Number};
use pounce_nlp::TNLP;
use pounce_nlp::return_codes::ApplicationReturnStatus;
use pounce_nlp::tnlp::{
    BoundsInfo, IndexStyle, IpoptCq, IpoptData, NlpInfo, Solution, SparsityRequest, StartingPoint,
};
use pounce_sensitivity::{PdSensBacksolver, SensBacksolver, Solver};

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

fn converged_backsolver() -> PdSensBacksolver {
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
    app.initialize().unwrap();

    let captured: Rc<RefCell<Option<PdSensBacksolver>>> = Rc::new(RefCell::new(None));
    let cap = Rc::clone(&captured);
    app.set_on_converged(Box::new(move |data, cq, nlp, pd| {
        *cap.borrow_mut() = Some(
            PdSensBacksolver::new(data, cq, nlp, pd).expect("backsolver from converged state"),
        );
    }));
    let tnlp: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(ParamQp));
    let status = app.optimize_tnlp(Rc::clone(&tnlp));
    assert!(
        matches!(
            status,
            ApplicationReturnStatus::SolveSucceeded
                | ApplicationReturnStatus::SolvedToAcceptableLevel
        ),
        "base solve failed: {status:?}",
    );
    let bs = captured.borrow_mut().take().expect("callback ran");
    bs
}

/// The parametric right-hand side: one on the pin equality's `y_c` row.
fn pin_rhs(bs: &PdSensBacksolver) -> Vec<Number> {
    let dims = bs.block_dims();
    let mut rhs = vec![0.0; bs.dim()];
    rhs[dims[0] + dims[1]] = 1.0;
    rhs
}

#[test]
fn the_held_direction_matches_hand_algebra() {
    let bs = converged_backsolver();
    let rhs = pin_rhs(&bs);
    let mut d = vec![0.0; bs.dim()];
    assert!(bs.solve(&rhs, &mut d), "plain solve failed");
    // with x2 held at its bound: dx = [a', 0]
    assert!(
        (d[0] - A1).abs() < 1e-6 && d[1].abs() < 1e-6,
        "held direction [{}, {}] should be [{A1}, 0]",
        d[0],
        d[1],
    );
}

#[test]
fn the_released_direction_matches_hand_algebra() {
    let bs = converged_backsolver();
    assert!(bs.supports_release(), "release path unavailable");
    let dims = bs.block_dims();
    // x2's lower-bound multiplier row: second entry of the z_l block
    let z_row = dims[0] + dims[1] + dims[2] + dims[3] + 1;
    let rows = bs.bound_rows().expect("bound rows");
    assert!(
        rows.iter()
            .any(|b| b.row == z_row && b.var_row == 1 && b.lower),
        "row {z_row} is not x2's lower bound in {rows:?}",
    );

    let rhs = pin_rhs(&bs);
    let mut d = vec![0.0; bs.dim()];
    assert!(
        bs.solve_released(&[z_row], &rhs, &mut d),
        "released solve failed"
    );
    // free system: H dx = [a', b']
    let det = 1.0 - G * G;
    let want = [(A1 - G * B1) / det, (B1 - G * A1) / det];
    assert!(
        (d[0] - want[0]).abs() < 1e-6 && (d[1] - want[1]).abs() < 1e-6,
        "released direction [{}, {}] should be [{}, {}]",
        d[0],
        d[1],
        want[0],
        want[1],
    );
}

#[test]
fn the_released_direction_survives_a_prior_plain_solve() {
    // The sequence every path-following call makes: a plain solve
    // first, which factors and caches with the original diagonal, then
    // the released solve, which must re-factor rather than reuse.
    let bs = converged_backsolver();
    let dims = bs.block_dims();
    let z_row = dims[0] + dims[1] + dims[2] + dims[3] + 1;
    let rhs = pin_rhs(&bs);

    let mut d0 = vec![0.0; bs.dim()];
    assert!(bs.solve(&rhs, &mut d0), "plain solve failed");
    assert!(d0[1].abs() < 1e-6, "held direction should keep x2 still");

    let mut d = vec![0.0; bs.dim()];
    assert!(
        bs.solve_released(&[z_row], &rhs, &mut d),
        "released solve failed"
    );
    let det = 1.0 - G * G;
    let want = [(A1 - G * B1) / det, (B1 - G * A1) / det];
    assert!(
        (d[0] - want[0]).abs() < 1e-6 && (d[1] - want[1]).abs() < 1e-6,
        "released direction after a plain solve [{}, {}] should be [{}, {}]",
        d[0],
        d[1],
        want[0],
        want[1],
    );
}

#[test]
fn the_walk_releases_and_reaches_the_target_on_this_qp() {
    // The same problem through the walk itself. From p = 0 to p = 1
    // the true path releases x2 near 0.57 and holds x1 at its upper
    // bound near 0.75, ending at x = [1, 0.1].
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
    app.initialize().unwrap();
    let tnlp: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(ParamQp));
    let mut solver = Solver::new(app, tnlp);
    let status = solver.solve();
    assert!(
        matches!(status, ApplicationReturnStatus::SolveSucceeded),
        "base solve failed: {status:?}",
    );
    let base = solver.converged().expect("converged").x.clone();

    let (dx, segments) = solver
        .parametric_step_path(&[0], &[1.0], 20)
        .expect("parametric_step_path");
    let endpoint = [base[0] + dx[0], base[1] + dx[1]];
    assert!(
        (endpoint[0] - 1.0).abs() < 1e-6 && (endpoint[1] - 0.1).abs() < 1e-6,
        "endpoint [{}, {}] should be [1.0, 0.1]; segments {segments:?}",
        endpoint[0],
        endpoint[1],
    );
}
