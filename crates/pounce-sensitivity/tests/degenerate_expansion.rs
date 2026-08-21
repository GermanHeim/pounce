//! The directional decision on fixtures with two and three weakly
//! active bounds, measured against hand algebra.
//!
//! Zeroing both linear terms of the released_direction.rs QP at
//! `p = 0` puts the unconstrained minimizer at the origin, so `x1`
//! and `x2` each sit on a lower bound with a vanishing multiplier:
//! two weakly active bounds. A third variable coupled through
//! `H = I + G*(offdiag)` gives three. Both fixtures are
//! hand-computable on each side of the kink.
//!
//! Toward positive `p` every bound releases and the direction is the
//! free system `H dx = c'` with `c'` the parametric gradient. Toward
//! negative `p` every bound holds: the pin forces are the parametric
//! gradient entries, all strictly positive, so every weak row passes
//! the held filter and the direction is exactly zero.
//!
//! The back-solve counts are the complexity regression: the decision
//! spends one solve for the all-released direction, one basis column
//! per engaged row, and one more to recover the direction from the pin
//! forces, so the hold side costs `ke + 2` where enumeration over the
//! same weak sets spent 6 (`ke = 2`) and 11 (`ke = 3`) trials. The
//! release side engages nothing and costs one.

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
const A1: Number = 1.10;
const B1: Number = 0.11;
const C1: Number = 0.55;

/// Two decision variables, both weakly active at `p = 0`:
///
/// ```text
/// min 0.5 x1^2 + 0.5 x2^2 + G x1 x2 - A1 x3 x1 - B1 x3 x2
/// s.t. x3 = p,   0 <= x1 <= 10,   0 <= x2 <= 10
/// ```
struct Ke2Qp;

impl TNLP for Ke2Qp {
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
        let (x1, x2, p) = (x[0], x[1], x[2]);
        Some(0.5 * x1 * x1 + 0.5 * x2 * x2 + G * x1 * x2 - A1 * p * x1 - B1 * p * x2)
    }

    fn eval_grad_f(&mut self, x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
        let (x1, x2, p) = (x[0], x[1], x[2]);
        g[0] = x1 + G * x2 - A1 * p;
        g[1] = x2 + G * x1 - B1 * p;
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

/// Three decision variables coupled through `H = I + G*(offdiag)`,
/// all weakly active at `p = 0`:
///
/// ```text
/// min 0.5 (x1^2 + x2^2 + x3^2) + G (x1 x2 + x1 x3 + x2 x3)
///     - A1 x4 x1 - B1 x4 x2 - C1 x4 x3
/// s.t. x4 = p,   0 <= x1, x2, x3 <= 10
/// ```
struct Ke3Qp;

impl TNLP for Ke3Qp {
    fn get_nlp_info(&mut self) -> Option<NlpInfo> {
        Some(NlpInfo {
            n: 4,
            m: 1,
            nnz_jac_g: 1,
            nnz_h_lag: 9,
            index_style: IndexStyle::C,
        })
    }

    fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
        for i in 0..3 {
            b.x_l[i] = 0.0;
            b.x_u[i] = 10.0;
        }
        b.x_l[3] = -1.0e19;
        b.x_u[3] = 1.0e19;
        b.g_l[0] = 0.0;
        b.g_u[0] = 0.0;
        true
    }

    fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
        sp.x[0] = 0.3;
        sp.x[1] = 0.3;
        sp.x[2] = 0.3;
        sp.x[3] = 0.0;
        true
    }

    fn eval_f(&mut self, x: &[Number], _new_x: bool) -> Option<Number> {
        let (x1, x2, x3, p) = (x[0], x[1], x[2], x[3]);
        Some(
            0.5 * (x1 * x1 + x2 * x2 + x3 * x3) + G * (x1 * x2 + x1 * x3 + x2 * x3)
                - A1 * p * x1
                - B1 * p * x2
                - C1 * p * x3,
        )
    }

    fn eval_grad_f(&mut self, x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
        let (x1, x2, x3, p) = (x[0], x[1], x[2], x[3]);
        g[0] = x1 + G * (x2 + x3) - A1 * p;
        g[1] = x2 + G * (x1 + x3) - B1 * p;
        g[2] = x3 + G * (x1 + x2) - C1 * p;
        g[3] = -A1 * x1 - B1 * x2 - C1 * x3;
        true
    }

    fn eval_g(&mut self, x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
        g[0] = x[3];
        true
    }

    fn eval_jac_g(&mut self, _x: Option<&[Number]>, _nx: bool, mode: SparsityRequest<'_>) -> bool {
        match mode {
            SparsityRequest::Structure { irow, jcol } => {
                irow[0] = 0;
                jcol[0] = 3;
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
                let rs: [Index; 9] = [0, 1, 1, 2, 2, 2, 3, 3, 3];
                let cs: [Index; 9] = [0, 0, 1, 0, 1, 2, 0, 1, 2];
                irow.copy_from_slice(&rs);
                jcol.copy_from_slice(&cs);
            }
            SparsityRequest::Values { values } => {
                values[0] = obj_factor;
                values[1] = obj_factor * G;
                values[2] = obj_factor;
                values[3] = obj_factor * G;
                values[4] = obj_factor * G;
                values[5] = obj_factor;
                values[6] = -obj_factor * A1;
                values[7] = -obj_factor * B1;
                values[8] = -obj_factor * C1;
            }
        }
        true
    }

    fn finalize_solution(&mut self, _s: Solution<'_>, _d: &IpoptData, _q: &IpoptCq) {}
}

/// Solve a fixture through `Solver` and return it, with bounds kept
/// exact so the classifier reads slacks against the model's own
/// bounds.
fn solved(tnlp: Rc<RefCell<dyn TNLP>>) -> Solver {
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
    let mut solver = Solver::new(app, tnlp);
    let status = solver.solve();
    assert!(
        matches!(status, ApplicationReturnStatus::SolveSucceeded),
        "base solve failed: {status:?}",
    );
    solver
}

#[test]
fn two_weak_bounds_are_detected() {
    let solver = solved(Rc::new(RefCell::new(Ke2Qp)));
    let x = solver.converged().expect("converged").x.clone();
    assert!(
        x[0].abs() < 1e-3,
        "x1 should sit on its bound, got {}",
        x[0]
    );
    assert!(
        x[1].abs() < 1e-3,
        "x2 should sit on its bound, got {}",
        x[1]
    );
    let mut weak: Vec<usize> = solver
        .weakly_active_bounds()
        .expect("classification")
        .iter()
        .inspect(|w| assert!(w.lower, "weak bounds here are all lower: {w:?}"))
        .map(|w| w.var_row)
        .collect();
    weak.sort_unstable();
    assert_eq!(weak, vec![0, 1], "both variables weakly active");
}

#[test]
fn the_two_weak_directional_step_matches_hand_algebra_on_both_sides() {
    let solver = solved(Rc::new(RefCell::new(Ke2Qp)));
    let det = 1.0 - G * G;

    // release side: the free system H dx = [A1, B1]
    let (d, pinned, trials) = solver
        .parametric_step_directional(&[0], &[1.0], 16)
        .expect("release side");
    let want = [(A1 - G * B1) / det, (B1 - G * A1) / det];
    assert!(
        (d[0] - want[0]).abs() < 1e-6 && (d[1] - want[1]).abs() < 1e-6,
        "release side [{}, {}] should be [{}, {}]",
        d[0],
        d[1],
        want[0],
        want[1],
    );
    assert!(pinned.is_empty(), "nothing pinned on the release side");
    assert_eq!(trials, 1, "the all-released solve is the whole decision");

    // hold side: both pin forces strictly positive, direction zero
    let (d, mut pinned, trials) = solver
        .parametric_step_directional(&[0], &[-1.0], 16)
        .expect("hold side");
    assert!(
        d[0].abs() < 1e-8 && d[1].abs() < 1e-8,
        "hold side [{}, {}] should be [0, 0]",
        d[0],
        d[1],
    );
    pinned.sort_unstable();
    assert_eq!(pinned, vec![0, 1], "both variables pinned on the hold side");
    assert_eq!(
        trials, 4,
        "the all-released solve, ke = 2 basis columns, and the combined          solve that recovers the direction"
    );
}

#[test]
fn three_weak_bounds_are_detected() {
    let solver = solved(Rc::new(RefCell::new(Ke3Qp)));
    let x = solver.converged().expect("converged").x.clone();
    for i in 0..3 {
        assert!(
            x[i].abs() < 1e-3,
            "x{} should sit on its bound, got {}",
            i + 1,
            x[i]
        );
    }
    let mut weak: Vec<usize> = solver
        .weakly_active_bounds()
        .expect("classification")
        .iter()
        .inspect(|w| assert!(w.lower, "weak bounds here are all lower: {w:?}"))
        .map(|w| w.var_row)
        .collect();
    weak.sort_unstable();
    assert_eq!(weak, vec![0, 1, 2], "all three variables weakly active");
}

#[test]
fn the_three_weak_directional_step_matches_hand_algebra_on_both_sides() {
    let solver = solved(Rc::new(RefCell::new(Ke3Qp)));
    // H = (1 - G) I + G J with J the all-ones matrix, so
    // H^{-1} = alpha (I - beta J) with the constants below
    let alpha = 1.0 / (1.0 - G);
    let beta = G / (1.0 + 2.0 * G);
    let s = A1 + B1 + C1;
    let want = [
        alpha * (A1 - beta * s),
        alpha * (B1 - beta * s),
        alpha * (C1 - beta * s),
    ];

    // release side: the free system H dx = [A1, B1, C1]
    let (d, pinned, trials) = solver
        .parametric_step_directional(&[0], &[1.0], 16)
        .expect("release side");
    for i in 0..3 {
        assert!(
            (d[i] - want[i]).abs() < 1e-6,
            "release side d[{i}] = {} should be {}",
            d[i],
            want[i],
        );
    }
    assert!(pinned.is_empty(), "nothing pinned on the release side");
    assert_eq!(trials, 1, "the all-released solve is the whole decision");

    // hold side: three simultaneous positive pin forces through the
    // held filter, dense off-diagonal S, direction zero
    let (d, mut pinned, trials) = solver
        .parametric_step_directional(&[0], &[-1.0], 16)
        .expect("hold side");
    for i in 0..3 {
        assert!(d[i].abs() < 1e-8, "hold side d[{i}] = {} should be 0", d[i]);
    }
    pinned.sort_unstable();
    assert_eq!(
        pinned,
        vec![0, 1, 2],
        "all three variables pinned on the hold side"
    );
    assert_eq!(
        trials, 5,
        "the all-released solve, ke = 3 basis columns, and the combined          solve that recovers the direction"
    );
}

#[test]
fn a_tight_budget_reports_what_to_raise_it_to() {
    // The ke = 3 hold side needs five back-solves: the all-released
    // solve, three basis columns, and the combined solve that recovers
    // the direction. A budget of three cannot finish, and what the
    // caller needs back is the number to raise degeneracy_iter to,
    // since raising it one at a time is a retry per engaged row.
    let solver = solved(Rc::new(RefCell::new(Ke3Qp)));
    let err = solver
        .parametric_step_directional(&[0], &[-1.0], 3)
        .expect_err("budget of 3 cannot fit 4 back-solves");
    let msg = format!("{err:?}");
    assert!(
        msg.contains("directional derivative"),
        "recoverable prefix missing: {msg}"
    );
}

#[test]
fn the_budget_message_names_the_knob_and_the_number() {
    // What a caller reads when the budget runs out has to be actionable:
    // the engaged count, which is the retry price, and the number to
    // raise degeneracy_iter to. Naming the weak-set size instead sends
    // a caller on one retry per engaged row.
    let solver = solved(Rc::new(RefCell::new(Ke3Qp)));
    for budget in [0usize, 1, 3] {
        let err = solver
            .parametric_step_directional(&[0], &[-1.0], budget)
            .expect_err("a budget below five cannot finish the ke = 3 hold side");
        let msg = match err {
            pounce_sensitivity::SolverError::SensComputationFailed(m) => m,
            other => panic!("wrong error at budget {budget}: {other:?}"),
        };
        assert!(!msg.is_empty(), "empty message at budget {budget}");
        assert!(
            msg.contains("degeneracy_iter"),
            "budget {budget} does not name the knob: {msg}",
        );
        println!("budget {budget}: {msg}");
    }
}
