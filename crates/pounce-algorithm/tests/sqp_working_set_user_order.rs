//! The published SQP working set, and the one a warm start takes, are in the
//! caller's index spaces: `bounds[j]` is variable `j` of the full `n`,
//! `constraints[i]` is `g` row `i` in the model's own order.
//!
//! The SQP stacks its rows equalities first (`[c; d]`) and drops variables
//! that `fixed_variable_treatment = make_parameter` removes. The bound side
//! was lifted to full `n` by gh#986; the row side was published positionally
//! in `[c; d]` order, so on a model whose first row is an inequality and
//! second an equality `last_sqp_working_set` read `[Equality, AtUpper]` —
//! row 0 claimed to be an equality it is not — while every frontend (the C
//! API, the GAMS link, pounce-py, a hand-built Rust set) reads and writes
//! one status per user row.
//!
//! The model puts each half of the map under load: `g0` is an active
//! inequality ahead of the equality `g1` (so `[c; d]` is the reverse of `g`
//! order), and the fixed `x0` sits ahead of `x1`, whose upper bound is
//! active (so a reduced-space vector shifts every status by one).

use pounce_algorithm::application::IpoptApplication;
use pounce_algorithm::sqp::SqpIterates;
use pounce_common::types::Number;
use pounce_nlp::return_codes::ApplicationReturnStatus;
use pounce_nlp::tnlp::{
    BoundsInfo, IndexStyle, IpoptCq, IpoptData, NlpInfo, Solution, SparsityRequest, StartingPoint,
    TNLP,
};
use pounce_qp::{BoundStatus, ConsStatus, WorkingSet};
use std::cell::RefCell;
use std::rc::Rc;

#[derive(Clone, Debug, Default)]
struct Point {
    x: Vec<Number>,
    z_l: Vec<Number>,
    z_u: Vec<Number>,
    lambda: Vec<Number>,
}

/// `min (x1-3)² + (x2-3)² + (x3-1)² + x0·x2`
/// s.t. `g0 = x1 + x2 ≤ 2.5`, `g1 = x2 − x3 = 0`,
/// `x0 = 0.5` (fixed), `x1 ∈ [-10, 1]`, `x2` free, `x3 ≥ 0`.
///
/// Solution `x = (0.5, 1, 1.5, 1.5)`, `λ = (1.5, 1)`, `z_U[1] = 2.5`.
#[derive(Default)]
struct Model {
    out: Option<Point>,
}

impl TNLP for Model {
    fn get_nlp_info(&mut self) -> Option<NlpInfo> {
        Some(NlpInfo {
            n: 4,
            m: 2,
            nnz_jac_g: 4,
            nnz_h_lag: 4,
            index_style: IndexStyle::C,
        })
    }
    fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
        b.x_l.copy_from_slice(&[0.5, -10.0, -1e19, 0.0]);
        b.x_u.copy_from_slice(&[0.5, 1.0, 1e19, 1e19]);
        b.g_l.copy_from_slice(&[-1e19, 0.0]);
        b.g_u.copy_from_slice(&[2.5, 0.0]);
        true
    }
    fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
        sp.x.copy_from_slice(&[0.5, 0.0, 0.0, 0.5]);
        true
    }
    fn eval_f(&mut self, x: &[Number], _n: bool) -> Option<Number> {
        Some((x[1] - 3.0).powi(2) + (x[2] - 3.0).powi(2) + (x[3] - 1.0).powi(2) + x[0] * x[2])
    }
    fn eval_grad_f(&mut self, x: &[Number], _n: bool, g: &mut [Number]) -> bool {
        g[0] = x[2];
        g[1] = 2.0 * (x[1] - 3.0);
        g[2] = 2.0 * (x[2] - 3.0) + x[0];
        g[3] = 2.0 * (x[3] - 1.0);
        true
    }
    fn eval_g(&mut self, x: &[Number], _n: bool, g: &mut [Number]) -> bool {
        g[0] = x[1] + x[2];
        g[1] = x[2] - x[3];
        true
    }
    fn eval_jac_g(&mut self, _x: Option<&[Number]>, _n: bool, mode: SparsityRequest<'_>) -> bool {
        match mode {
            SparsityRequest::Structure { irow, jcol } => {
                irow.copy_from_slice(&[0, 0, 1, 1]);
                jcol.copy_from_slice(&[1, 2, 2, 3]);
            }
            SparsityRequest::Values { values } => {
                values.copy_from_slice(&[1.0, 1.0, 1.0, -1.0]);
            }
        }
        true
    }
    fn eval_h(
        &mut self,
        _x: Option<&[Number]>,
        _n: bool,
        of: Number,
        _l: Option<&[Number]>,
        _nl: bool,
        mode: SparsityRequest<'_>,
    ) -> bool {
        match mode {
            SparsityRequest::Structure { irow, jcol } => {
                irow.copy_from_slice(&[1, 2, 3, 2]);
                jcol.copy_from_slice(&[1, 2, 3, 0]);
            }
            SparsityRequest::Values { values } => {
                values.copy_from_slice(&[2.0 * of, 2.0 * of, 2.0 * of, of]);
            }
        }
        true
    }
    fn finalize_solution(&mut self, s: Solution<'_>, _d: &IpoptData, _q: &IpoptCq) {
        self.out = Some(Point {
            x: s.x.to_vec(),
            z_l: s.z_l.to_vec(),
            z_u: s.z_u.to_vec(),
            lambda: s.lambda.to_vec(),
        });
    }
}

/// What the converged solve's working set says in user indices.
fn expected() -> WorkingSet {
    WorkingSet {
        bounds: vec![
            BoundStatus::Fixed,
            BoundStatus::AtUpper,
            BoundStatus::Inactive,
            BoundStatus::Inactive,
        ],
        constraints: vec![ConsStatus::AtUpper, ConsStatus::Equality],
    }
}

fn app() -> IpoptApplication {
    let mut a = IpoptApplication::new();
    a.initialize().unwrap();
    let o = a.options_mut();
    o.set_integer_value("print_level", 0, true, false).unwrap();
    o.set_string_value("algorithm", "active-set-sqp", true, false)
        .unwrap();
    a
}

fn cold() -> (IpoptApplication, Rc<RefCell<Model>>, Point) {
    let p = Rc::new(RefCell::new(Model::default()));
    let mut a = app();
    assert_eq!(
        a.optimize_tnlp(p.clone()),
        ApplicationReturnStatus::SolveSucceeded
    );
    let sol = p.borrow_mut().out.take().unwrap();
    // The fixture is informative only if both rows and x1's bound bind.
    let want_x = [0.5, 1.0, 1.5, 1.5];
    for (j, (&g, &w)) in sol.x.iter().zip(&want_x).enumerate() {
        assert!((g - w).abs() < 1e-7, "x[{j}] = {g}, want {w}");
    }
    assert!((sol.lambda[0] - 1.5).abs() < 1e-6, "λ = {:?}", sol.lambda);
    assert!((sol.lambda[1] - 1.0).abs() < 1e-6, "λ = {:?}", sol.lambda);
    assert!((sol.z_u[1] - 2.5).abs() < 1e-6, "z_U = {:?}", sol.z_u);
    (a, p, sol)
}

/// (a) The published set has each status on its own user row / variable.
#[test]
fn the_published_working_set_is_in_user_row_and_variable_order() {
    let (a, _p, _sol) = cold();
    let ws = a
        .last_sqp_working_set()
        .expect("SQP publishes a working set");
    assert_eq!(ws, &expected());
}

/// (b) Feeding the published set (and the reported point) back in is a
/// 0-iteration re-solve that publishes the same set.
///
/// This is what pins the *inbound* map. With zero iterations no QP runs,
/// so the published set is the staged one carried through the solve, and
/// any order mismatch between the two directions shows. A warm solve that
/// does run a QP is weak evidence here: the QP repairs a status that
/// contradicts its row's bounds (`Equality` on an inequality, anything but
/// `Equality` on an equality), which on this model is exactly what a
/// swapped pair of rows looks like.
#[test]
fn the_published_working_set_round_trips_as_a_warm_start() {
    let (mut a, p, sol) = cold();
    let ws = a.last_sqp_working_set().cloned();
    a.set_sqp_warm_start(SqpIterates {
        x: sol.x.clone(),
        lambda_g: sol.lambda.clone(),
        lambda_x: sol.z_l.iter().zip(&sol.z_u).map(|(l, u)| l - u).collect(),
        working: ws,
    });
    assert_eq!(
        a.optimize_tnlp(p.clone()),
        ApplicationReturnStatus::SolveSucceeded
    );
    let st = a.statistics();
    assert!(st.sqp_warm_working_set_applied);
    assert_eq!(st.iteration_count, 0, "warm re-solve from the optimum");
    assert_eq!(a.last_sqp_working_set(), Some(&expected()));
}
