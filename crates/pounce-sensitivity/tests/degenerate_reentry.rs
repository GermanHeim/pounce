//! The directional decision on a fixture whose engaged set grows on a
//! second pass, and the budget message that fixture's retries read.
//!
//! `degenerate_expansion.rs` covers the decision that finishes on one
//! pass: with uniform negative coupling, holding a weak row only ever
//! raises the other rows' movement, so every row that engages does so
//! against the all-released direction and the expansion loop never
//! re-enters. That leaves the loop's second pass, and a decision that
//! holds some rows while releasing others, reachable only on the
//! dynamic column model, which CI cannot run.
//!
//! This fixture reaches both. Over
//!
//! ```text
//!     H = [[ 1.0, -1.2, -0.5],      c = [1.6, 1.4, 0.8]
//!          [-1.2,  5.0,  3.5],
//!          [-0.5,  3.5,  3.0]]
//! ```
//!
//! `H` is positive definite (eigenvalues 0.186, 0.943, 7.87), so at
//! `p = 0` the unconstrained minimizer is the origin and all three
//! variables sit on a lower bound with a vanishing multiplier: three
//! weakly active bounds, as in `Ke3Qp`. The coupling signs are what
//! differ. With `M = H^-1`, `Mc` has a negative third entry, so on
//! one side of the kink the third row does not engage against the
//! all-released direction, and `M`'s third row is negative enough on
//! its first two entries that holding rows 1 and 2 drives it into
//! violation. The decision has to come back for it.
//!
//! Both sides are hand-computable:
//!
//! * `dp = -1` engages rows 1 and 2 against `d0 = -Mc`, holds them
//!   with `lambda = [26/15, 7/15]`, and the resulting direction moves
//!   row 3 to `-4/15`. Row 3 then engages, and the second pass returns
//!   `lambda = c`, all three strictly positive, direction exactly zero.
//! * `dp = +1` engages only row 3. Holding it and releasing the other
//!   two leaves the reduced system `H[1..2, 1..2] dx = c[1..2]`, whose
//!   solution is `[9.68/3.56, 3.32/3.56]`.
//!
//! The back-solve counts are what prove the loop re-entered. A
//! decision that finishes on one pass costs `ke + 2`; the `dp = -1`
//! side costs 6 over a final `ke` of 3, which one pass cannot reach.

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

const H11: Number = 1.0;
const H12: Number = -1.2;
const H13: Number = -0.5;
const H22: Number = 5.0;
const H23: Number = 3.5;
const H33: Number = 3.0;
const C1: Number = 1.6;
const C2: Number = 1.4;
const C3: Number = 0.8;

/// Three decision variables, all weakly active at `p = 0`, coupled so
/// that one of them engages only after the other two are held:
///
/// ```text
/// min 0.5 x^T H x - p c^T x
/// s.t. x4 = p,   0 <= x1, x2, x3 <= 10
/// ```
struct ReentryQp;

impl TNLP for ReentryQp {
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
            0.5 * (H11 * x1 * x1 + H22 * x2 * x2 + H33 * x3 * x3)
                + H12 * x1 * x2
                + H13 * x1 * x3
                + H23 * x2 * x3
                - p * (C1 * x1 + C2 * x2 + C3 * x3),
        )
    }

    fn eval_grad_f(&mut self, x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
        let (x1, x2, x3, p) = (x[0], x[1], x[2], x[3]);
        g[0] = H11 * x1 + H12 * x2 + H13 * x3 - C1 * p;
        g[1] = H12 * x1 + H22 * x2 + H23 * x3 - C2 * p;
        g[2] = H13 * x1 + H23 * x2 + H33 * x3 - C3 * p;
        g[3] = -(C1 * x1 + C2 * x2 + C3 * x3);
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
                values[0] = obj_factor * H11;
                values[1] = obj_factor * H12;
                values[2] = obj_factor * H22;
                values[3] = obj_factor * H13;
                values[4] = obj_factor * H23;
                values[5] = obj_factor * H33;
                values[6] = -obj_factor * C1;
                values[7] = -obj_factor * C2;
                values[8] = -obj_factor * C3;
            }
        }
        true
    }

    fn finalize_solution(&mut self, _s: Solution<'_>, _d: &IpoptData, _q: &IpoptCq) {}
}

/// Solve to a tight tolerance with the bounds unrelaxed, so the
/// classifier can read the slacks and call the three vanishing
/// multipliers weakly active.
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
fn all_three_bounds_are_weakly_active() {
    let solver = solved(Rc::new(RefCell::new(ReentryQp)));
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
fn a_row_that_engages_only_after_the_others_are_held_still_enters() {
    let solver = solved(Rc::new(RefCell::new(ReentryQp)));
    let (d, mut held, work) = solver
        .parametric_step_directional(&[0], &[-1.0], 16)
        .expect("hold side");

    // Every row ends up held, so the direction is zero on the primal
    // block. Row 3 gets there only on the second pass: against the
    // all-released direction it moves to +2.849, the feasible side of
    // its lower bound, and it is holding rows 1 and 2 that pushes it
    // to -4/15.
    for i in 0..3 {
        assert!(d[i].abs() < 1e-8, "hold side d[{i}] = {} should be 0", d[i]);
    }
    held.sort_unstable();
    assert_eq!(held, vec![0, 1, 2], "all three rows held");

    // The count is the evidence of re-entry, so it is asserted
    // exactly. One pass over a final engaged set of three costs
    // `ke + 2 = 5`: the all-released solve, three basis columns, and
    // one combined solve. Six is that plus the extra combined solve
    // the first pass spent before row 3 entered, and no arrangement
    // of a single pass reaches it.
    assert_eq!(
        work, 6,
        "the all-released solve, two basis columns and a combined solve on \
         the first pass, then row 3's column and a second combined solve",
    );
}

#[test]
fn the_other_side_holds_one_row_and_releases_two() {
    let solver = solved(Rc::new(RefCell::new(ReentryQp)));
    let (d, held, work) = solver
        .parametric_step_directional(&[0], &[1.0], 16)
        .expect("release side");

    // Only row 3 engages here. With x3 pinned, x1 and x2 solve the
    // reduced system H[1..2, 1..2] dx = c[1..2], whose determinant is
    // 5 - 1.44 = 3.56.
    let det = H11 * H22 - H12 * H12;
    let want = [(C1 * H22 - H12 * C2) / det, (H11 * C2 - H12 * C1) / det];
    for i in 0..2 {
        assert!(
            (d[i] - want[i]).abs() < 1e-6,
            "release side d[{i}] = {} should be {}",
            d[i],
            want[i],
        );
    }
    assert!(
        d[2].abs() < 1e-8,
        "the held row does not move, got {}",
        d[2]
    );
    assert_eq!(held, vec![2], "only the third row is held");
    assert_eq!(
        work, 3,
        "the all-released solve, one basis column, and the combined solve",
    );
}

#[test]
fn every_budget_retry_the_message_asks_for_buys_progress() {
    // A caller raises degeneracy_iter to whatever the message names
    // and calls again. On a decision that expands, the engaged count
    // alone prices only a single pass, so the floor has to stay above
    // what has already been spent or a retry asks for a budget the
    // caller is already at. Walking the budgets here must produce a
    // strictly increasing sequence that terminates.
    let solver = solved(Rc::new(RefCell::new(ReentryQp)));
    let mut budget = 1usize;
    let mut asked = Vec::new();
    let spent = loop {
        assert!(
            asked.len() < 16,
            "the retry sequence did not terminate: {asked:?}",
        );
        match solver.parametric_step_directional(&[0], &[-1.0], budget) {
            Ok((_, _, work)) => break work,
            Err(pounce_sensitivity::SolverError::SensComputationFailed(msg)) => {
                let need: usize = msg
                    .split("Raise degeneracy_iter to at least ")
                    .nth(1)
                    .and_then(|t| t.split(';').next())
                    .and_then(|t| t.trim().parse().ok())
                    .unwrap_or_else(|| panic!("no number to raise to in: {msg}"));
                assert!(
                    need > budget,
                    "budget {budget} was told to raise to {need}, which is not a raise: {msg}",
                );
                asked.push(need);
                budget = need;
            }
            Err(other) => panic!("wrong error at budget {budget}: {other:?}"),
        }
    };
    assert_eq!(
        asked,
        vec![4, 5, 6],
        "the sequence a caller walks, each entry the previous call's answer",
    );
    assert_eq!(spent, 6, "the budget the last answer bought");
}
