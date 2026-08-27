//! The same two kinks as `degenerate_weak_curvature.rs`, at an
//! **upper** bound.
//!
//! Every other fixture that reaches `parametric_step_directional` has
//! its weak rows at lower bounds -- `degenerate_expansion.rs` and
//! `degenerate_reentry.rs` assert as much outright, and
//! `sens_invariance_legs.rs` pins its kink as `(0, true)`. So
//! `sign(k)` is `+1` everywhere the drop rule has ever run, and the
//! `-1` half of it is a branch no test executes.
//!
//! That branch is load-bearing rather than cosmetic. The rule reads
//! `own = sign(k) * col[k]`, and the column's back-solve already
//! carries the sign in its right-hand side, so by linearity
//! `col[k] = sign(k) * S_kk` and the second multiply is what recovers
//! the bare diagonal. Applied once instead of twice, an upper-bound
//! row lands at `kappa <= 0` for any geometry whatsoever, is dropped
//! as though an equality owned it, and its bound goes undecided --
//! while every fixture in the crate stays green, because none of them
//! has an upper-bound row to drop.
//!
//! The holding-side test below is the mutation guard: delete one
//! `sign(k)` and it fails, alone.

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
/// The stiff coordinate's curvature, as in the lower-bound fixture.
const C: Number = 1.0e4;

/// `x0` and `x1` both at the kink of their *upper* bound, differing
/// only in curvature:
///
/// ```text
/// min 0.5 x0^2 + A1 p x0 + 0.5 C x1^2 + A1 C p x1
/// s.t. p = 0,   -10 <= x0 <= 0,   -10 <= x1 <= 0
/// ```
///
/// The sign on the cross term is flipped against the lower-bound
/// fixture, which reflects the whole problem through `x = 0`: at
/// `p = 0` both coordinates sit at zero, now their *upper* bound,
/// with slack and multiplier vanishing together. For `dp = +1e-3`
/// both leave the bound downward and the exact step is
/// `dx0 = dx1 = -A1 dp`; a negative `dp` pushes them through the
/// bound, so that is the holding side.
struct TwoCurvatureKinksAtUpperBound;

impl TNLP for TwoCurvatureKinksAtUpperBound {
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
        b.x_l[0] = -10.0;
        b.x_u[0] = 0.0;
        b.x_l[1] = -10.0;
        b.x_u[1] = 0.0;
        b.x_l[2] = -1.0e19;
        b.x_u[2] = 1.0e19;
        b.g_l[0] = 0.0;
        b.g_u[0] = 0.0;
        true
    }

    fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
        sp.x[0] = -0.3;
        sp.x[1] = -0.3;
        sp.x[2] = 0.0;
        true
    }

    fn eval_f(&mut self, x: &[Number], _new_x: bool) -> Option<Number> {
        let (x0, x1, p) = (x[0], x[1], x[2]);
        Some(0.5 * x0 * x0 + A1 * p * x0 + 0.5 * C * x1 * x1 + A1 * C * p * x1)
    }

    fn eval_grad_f(&mut self, x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
        let (x0, x1, p) = (x[0], x[1], x[2]);
        g[0] = x0 + A1 * p;
        g[1] = C * x1 + A1 * C * p;
        g[2] = A1 * x0 + A1 * C * x1;
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
                values[2] = obj_factor * A1;
                values[3] = obj_factor * A1 * C;
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
    let mut solver = Solver::new(app, Rc::new(RefCell::new(TwoCurvatureKinksAtUpperBound)));
    let status = solver.solve();
    assert!(
        matches!(status, ApplicationReturnStatus::SolveSucceeded),
        "base solve failed: {status:?}",
    );
    solver
}

/// The setup check, and the one that names the branch: both rows must
/// be weak *and* both must be upper-bound rows, or the rest of this
/// file is a second copy of the lower-bound fixture.
#[test]
fn the_weak_rows_here_are_upper_bound_rows() {
    let solver = solved();
    let weak = solver.weakly_active_bounds().expect("classification");
    let rows: Vec<(usize, bool)> = weak.iter().map(|w| (w.var_row, w.lower)).collect();
    assert!(
        rows.contains(&(0, false)) && rows.contains(&(1, false)),
        "both kinks are weak at their upper bound: {rows:?}"
    );
    assert!(
        weak.iter().all(|w| !w.lower),
        "no lower-bound row may sneak in, or the branch is not the one under test: {rows:?}"
    );
}

/// The leaving side: an upper-bound kink released by the perturbation
/// moves by the exact step, at either curvature.
#[test]
fn an_upper_bound_kink_leaves_its_bound_by_the_exact_step() {
    let solver = solved();
    let dp = 1.0e-3;
    let (d, _held, _spent) = solver
        .parametric_step_directional(&[0], &[dp], 16)
        .expect("the decision completes");
    let exact = -A1 * dp;
    assert!(
        (d[0] - exact).abs() < 1e-8,
        "x0 leaves its upper bound by -A1 dp, got {}",
        d[0]
    );
    assert!(
        (d[1] - exact).abs() < 1e-8,
        "x1 leaves its upper bound by -A1 dp whatever its curvature, got {}",
        d[1]
    );
}

/// The holding side, and the mutation guard for `sign(k)`.
///
/// A row the drop rule wrongly calls inert keeps its plain movement,
/// and here the plain movement is *through* the bound: `-A1 dp` with
/// `dp < 0` is positive, and the bound is above. So dropping these
/// rows does not merely lose precision, it returns a step that leaves
/// the feasible set. Applying `sign(k)` once rather than twice drops
/// exactly these rows and fails exactly this test.
#[test]
fn an_upper_bound_kink_holds_on_the_holding_side() {
    let solver = solved();
    let dp = -1.0e-10;
    let (d, held, _spent) = solver
        .parametric_step_directional(&[0], &[dp], 16)
        .expect("the decision completes");
    assert!(
        held.contains(&0) && held.contains(&1),
        "both upper-bound kinks hold on the holding side: held {held:?}"
    );
    assert!(d[0].abs() < 1e-12, "x0 holds at its bound, got {}", d[0]);
    assert!(d[1].abs() < 1e-12, "x1 holds at its bound, got {}", d[1]);
}
