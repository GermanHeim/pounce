//! gh #695: a non-finite **objective value** must never be reported as a
//! successful solve.
//!
//! `min f(x) s.t. x0 + x1 = 1`, where `f` returns `NaN` (or `inf`) and every
//! derivative is finite and exact. The NLP path returned
//! `Solve_Succeeded` / `obj_val = nan` — `status = 0` asserts the convergence
//! test passed, while the objective it reports is not a number, which is
//! self-contradictory on its face.
//!
//! The returned point is not wrong: `x = (0.5, 0.5)` is the minimizer of
//! `x·x` subject to `x0 + x1 = 1`, and with finite derivatives and a satisfied
//! equality the KKT residuals are genuinely small (`2.5e-9`). That is exactly
//! why the guard is needed — the convergence test never reads the objective
//! *value*, so nothing in it can notice.
//!
//! Specific to the equality-constrained shape, which is what made it survive
//! gh #292. That issue closed the NaN-*gradient* hole and explicitly recorded
//! `f`-returns-NaN as the safe contrast case — true for the shapes it
//! exercised (unconstrained, bounds-only, inequality-constrained, which return
//! `-3` or `-13`), and not once an equality constraint is present. The shape
//! matrix below is asserted in full so the next change cannot close one column
//! and reopen another.
//!
//! Oracle: Ipopt's `Eval_f` rejects a non-finite objective and terminates
//! `Invalid_Number_Detected`, and POUNCE documents itself as drop-in
//! compatible. POUNCE's own adjacent shapes agree.

use pounce_algorithm::application::IpoptApplication;
use pounce_common::types::Number;
use pounce_nlp::return_codes::ApplicationReturnStatus;
use pounce_nlp::tnlp::{
    BoundsInfo, IndexStyle, IpoptCq, IpoptData, NlpInfo, Solution, SparsityRequest, StartingPoint,
    TNLP,
};
use std::cell::RefCell;
use std::rc::Rc;

/// The starting point that reproduces. It is feasible for `x0 + x1 = 1` and is
/// already the minimizer of `x·x` there, so the solve converges directly and
/// reaches the convergence test with a `NaN` objective in hand — which is the
/// defect. From an *infeasible* start (`[1, 1]`) this driver instead enters
/// restoration and fails honestly (`Restoration_Failed`), because restoration's
/// line search cannot make progress on a `NaN`. The reporter's Python run took
/// that second route and its restoration *did* converge, after which the same
/// convergence test declared success — same defect, reached two ways. This is
/// the one that reproduces deterministically here.
const AT_OPTIMUM: [Number; 2] = [0.5, 0.5];

/// Which constraint the model carries — the axis gh #695 turns on.
#[derive(Clone, Copy, PartialEq, Debug)]
enum Shape {
    /// No constraints at all.
    None,
    /// `x0 + x1 = 1`.
    Equality,
    /// `x0 + x1 <= 1`.
    Inequality,
}

/// `min f(x)` with `f` non-finite by construction and `∇f = 2x`, `∇²f = 2I`
/// exact — so every quantity the convergence test *does* read is finite and
/// the solve converges on them.
struct NonFiniteObjective {
    value: Number,
    shape: Shape,
    bounded: bool,
    x0: [Number; 2],
}

impl TNLP for NonFiniteObjective {
    fn get_nlp_info(&mut self) -> Option<NlpInfo> {
        let m = pounce_common::types::Index::from(self.shape != Shape::None);
        Some(NlpInfo {
            n: 2,
            m,
            nnz_jac_g: 2 * m,
            nnz_h_lag: 2,
            index_style: IndexStyle::C,
        })
    }

    fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
        let (lo, hi) = if self.bounded {
            (-10.0, 10.0)
        } else {
            (-2.0e19, 2.0e19)
        };
        b.x_l.copy_from_slice(&[lo; 2]);
        b.x_u.copy_from_slice(&[hi; 2]);
        match self.shape {
            Shape::None => {}
            Shape::Equality => {
                b.g_l[0] = 1.0;
                b.g_u[0] = 1.0;
            }
            Shape::Inequality => {
                b.g_l[0] = -2.0e19;
                b.g_u[0] = 1.0;
            }
        }
        true
    }

    fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
        sp.x.copy_from_slice(&self.x0);
        true
    }

    fn eval_f(&mut self, _x: &[Number], _new_x: bool) -> Option<Number> {
        Some(self.value)
    }

    fn eval_grad_f(&mut self, x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
        g[0] = 2.0 * x[0];
        g[1] = 2.0 * x[1];
        true
    }

    fn eval_g(&mut self, x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
        if self.shape != Shape::None {
            g[0] = x[0] + x[1];
        }
        true
    }

    fn eval_jac_g(
        &mut self,
        _x: Option<&[Number]>,
        _new_x: bool,
        mode: SparsityRequest<'_>,
    ) -> bool {
        if self.shape == Shape::None {
            return true;
        }
        match mode {
            SparsityRequest::Structure { irow, jcol } => {
                irow.copy_from_slice(&[0, 0]);
                jcol.copy_from_slice(&[0, 1]);
            }
            SparsityRequest::Values { values } => {
                values.copy_from_slice(&[1.0, 1.0]);
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
                irow.copy_from_slice(&[0, 1]);
                jcol.copy_from_slice(&[0, 1]);
            }
            SparsityRequest::Values { values } => {
                // The constraint is linear, so `lambda` contributes nothing.
                values.copy_from_slice(&[obj_factor * 2.0, obj_factor * 2.0]);
            }
        }
        true
    }

    fn finalize_solution(&mut self, _sol: Solution<'_>, _d: &IpoptData, _q: &IpoptCq) {}
}

fn solve(
    value: Number,
    shape: Shape,
    bounded: bool,
    x0: [Number; 2],
) -> (ApplicationReturnStatus, Number) {
    let mut app = IpoptApplication::new();
    app.options_mut()
        .set_integer_value("print_level", 0, true, false)
        .unwrap();
    app.initialize().unwrap();
    let tnlp: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(NonFiniteObjective {
        value,
        shape,
        bounded,
        x0,
    }));
    let status = app.optimize_tnlp(tnlp);
    (status, app.statistics().final_objective)
}

/// A successful status and a non-finite objective is a contradiction, whatever
/// the shape. This is the property that matters; the shape matrix below only
/// records *which* column was broken.
#[test]
fn a_non_finite_objective_is_never_a_successful_solve() {
    for value in [Number::NAN, Number::INFINITY, Number::NEG_INFINITY] {
        for shape in [Shape::None, Shape::Equality, Shape::Inequality] {
            for bounded in [false, true] {
                let (status, obj) = solve(value, shape, bounded, AT_OPTIMUM);
                let succeeded = matches!(
                    status,
                    ApplicationReturnStatus::SolveSucceeded
                        | ApplicationReturnStatus::SolvedToAcceptableLevel
                );
                assert!(
                    !succeeded || obj.is_finite(),
                    "f = {value}, {shape:?}, bounded = {bounded}: reported {status:?} with \
                     final_objective = {obj}. `status` asserts the convergence test passed \
                     and the objective it reports is not a number (gh #695).",
                );
            }
        }
    }
}

/// The equality column specifically — the one that regressed — pinned against
/// the status the issue's three oracles agree on.
#[test]
fn the_equality_constrained_shape_reports_invalid_number() {
    for bounded in [false, true] {
        let (status, _) = solve(Number::NAN, Shape::Equality, bounded, AT_OPTIMUM);
        assert_eq!(
            status,
            ApplicationReturnStatus::InvalidNumberDetected,
            "NaN objective with an equality constraint (bounded = {bounded}) must report \
             Invalid_Number_Detected, as Ipopt's Eval_f does and as POUNCE's own \
             inequality-constrained shape already did",
        );
    }
}

/// The shapes that were already honest must stay honest — the guard must not
/// be bought by making a *finite* objective fail, and must not shift the
/// adjacent columns onto a different failure mode.
#[test]
fn the_shapes_that_already_failed_honestly_still_do() {
    for shape in [Shape::None, Shape::Inequality] {
        for bounded in [false, true] {
            let (status, _) = solve(Number::NAN, shape, bounded, AT_OPTIMUM);
            assert!(
                !matches!(
                    status,
                    ApplicationReturnStatus::SolveSucceeded
                        | ApplicationReturnStatus::SolvedToAcceptableLevel
                ),
                "{shape:?} (bounded = {bounded}) reported {status:?} on a NaN objective",
            );
        }
    }
}

/// The control: the same model with a *finite* objective still solves, to the
/// analytic optimum. Without this the guard could pass by rejecting everything.
#[test]
fn a_finite_objective_on_the_same_shape_still_solves() {
    let mut app = IpoptApplication::new();
    app.options_mut()
        .set_integer_value("print_level", 0, true, false)
        .unwrap();
    app.initialize().unwrap();
    // `f = x·x` rather than a constant, so the objective genuinely drives the
    // solve; minimized over `x0 + x1 = 1` at `x = (0.5, 0.5)`, `f* = 0.5`.
    let tnlp: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(SumOfSquares));
    let status = app.optimize_tnlp(tnlp);
    let obj = app.statistics().final_objective;
    assert!(
        matches!(
            status,
            ApplicationReturnStatus::SolveSucceeded
                | ApplicationReturnStatus::SolvedToAcceptableLevel
        ),
        "finite control did not solve: {status:?}",
    );
    assert!(
        (obj - 0.5).abs() < 1e-6,
        "finite control objective {obj}, expected 0.5",
    );
}

/// `min x·x s.t. x0 + x1 = 1` — [`NonFiniteObjective`] with a real objective.
struct SumOfSquares;

impl TNLP for SumOfSquares {
    fn get_nlp_info(&mut self) -> Option<NlpInfo> {
        Some(NlpInfo {
            n: 2,
            m: 1,
            nnz_jac_g: 2,
            nnz_h_lag: 2,
            index_style: IndexStyle::C,
        })
    }

    fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
        b.x_l.copy_from_slice(&[-10.0; 2]);
        b.x_u.copy_from_slice(&[10.0; 2]);
        b.g_l[0] = 1.0;
        b.g_u[0] = 1.0;
        true
    }

    fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
        sp.x.copy_from_slice(&[1.0, 1.0]);
        true
    }

    fn eval_f(&mut self, x: &[Number], _new_x: bool) -> Option<Number> {
        Some(x[0] * x[0] + x[1] * x[1])
    }

    fn eval_grad_f(&mut self, x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
        g[0] = 2.0 * x[0];
        g[1] = 2.0 * x[1];
        true
    }

    fn eval_g(&mut self, x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
        g[0] = x[0] + x[1];
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
                irow.copy_from_slice(&[0, 0]);
                jcol.copy_from_slice(&[0, 1]);
            }
            SparsityRequest::Values { values } => values.copy_from_slice(&[1.0, 1.0]),
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
                irow.copy_from_slice(&[0, 1]);
                jcol.copy_from_slice(&[0, 1]);
            }
            SparsityRequest::Values { values } => {
                values.copy_from_slice(&[obj_factor * 2.0, obj_factor * 2.0])
            }
        }
        true
    }

    fn finalize_solution(&mut self, _sol: Solution<'_>, _d: &IpoptData, _q: &IpoptCq) {}
}
