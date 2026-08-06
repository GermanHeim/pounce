//! Phase 6 acceptance (issue #487): eliminating variables determined by
//! linear equality rows must not change the answer.
//!
//! The fixture is deliberately shaped to exercise every arm of the
//! transform at once:
//!
//! * `x0 − 2·x1 = 0` — a free/free two-variable row, the shape the
//!   determined-block pipeline (#53) cannot reach, so `x0` folds onto `x1`
//!   with a coefficient of 2.
//! * `x3 = 1` — a singleton row, so `x3` leaves as a constant.
//! * `x1² + x2² = 2` — a nonlinear row, which survives and must be
//!   re-presented over the reduced columns.
//!
//! The objective carries an `x0·x1` cross term on purpose: both of its
//! columns collapse onto the *same* reduced column, which is the one
//! Hessian case that needs the symmetric pair counted twice, and an
//! `x3·x1` term whose second derivative must vanish once `x3` is a
//! constant.

use pounce_algorithm::application::IpoptApplication;
use pounce_common::types::Number;
use pounce_nlp::tnlp::{
    BoundsInfo, IndexStyle, IpoptCq, IpoptData, Linearity, NlpInfo, Solution, SparsityRequest,
    StartingPoint, TNLP,
};
use pounce_presolve::{LinearEqElimTnlp, PresolveOptions, PresolveTnlp};
use std::cell::RefCell;
use std::rc::Rc;

#[derive(Debug, Clone, Default)]
struct Captured {
    x: Vec<Number>,
    z_l: Vec<Number>,
    z_u: Vec<Number>,
    lambda: Vec<Number>,
    g: Vec<Number>,
    obj: Number,
}

#[derive(Default)]
struct Fixture {
    captured: Option<Captured>,
}

impl Fixture {
    fn grad_f(x: &[Number], out: &mut [Number]) {
        // f = (x0-1)² + (x2-3)² + x3·x1 + x0·x1
        out[0] = 2.0 * (x[0] - 1.0) + x[1];
        out[1] = x[3] + x[0];
        out[2] = 2.0 * (x[2] - 3.0);
        out[3] = x[1];
    }
}

/// Row-major dense Jacobian of the fixture, for the stationarity check.
fn jac_dense(x: &[Number]) -> [[Number; 4]; 3] {
    [
        [1.0, -2.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 2.0 * x[1], 2.0 * x[2], 0.0],
    ]
}

impl TNLP for Fixture {
    fn get_nlp_info(&mut self) -> Option<NlpInfo> {
        Some(NlpInfo {
            n: 4,
            m: 3,
            nnz_jac_g: 5,
            nnz_h_lag: 5,
            index_style: IndexStyle::C,
        })
    }

    fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
        b.x_l.copy_from_slice(&[-10.0, -10.0, 0.0, -10.0]);
        b.x_u.copy_from_slice(&[10.0, 10.0, 10.0, 10.0]);
        b.g_l.copy_from_slice(&[0.0, 1.0, 2.0]);
        b.g_u.copy_from_slice(&[0.0, 1.0, 2.0]);
        true
    }

    fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
        sp.x.copy_from_slice(&[1.0, 0.5, 1.0, 1.0]);
        true
    }

    fn get_constraints_linearity(&mut self, types: &mut [Linearity]) -> bool {
        types.copy_from_slice(&[Linearity::Linear, Linearity::Linear, Linearity::NonLinear]);
        true
    }

    fn eval_f(&mut self, x: &[Number], _new_x: bool) -> Option<Number> {
        Some((x[0] - 1.0).powi(2) + (x[2] - 3.0).powi(2) + x[3] * x[1] + x[0] * x[1])
    }

    fn eval_grad_f(&mut self, x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
        Self::grad_f(x, g);
        true
    }

    fn eval_g(&mut self, x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
        g[0] = x[0] - 2.0 * x[1];
        g[1] = x[3];
        g[2] = x[1] * x[1] + x[2] * x[2];
        true
    }

    fn eval_jac_g(
        &mut self,
        x: Option<&[Number]>,
        _new_x: bool,
        mode: SparsityRequest<'_>,
    ) -> bool {
        match mode {
            SparsityRequest::Structure { irow, jcol } => {
                irow.copy_from_slice(&[0, 0, 1, 2, 2]);
                jcol.copy_from_slice(&[0, 1, 3, 1, 2]);
            }
            SparsityRequest::Values { values } => {
                let Some(x) = x else { return false };
                values[0] = 1.0;
                values[1] = -2.0;
                values[2] = 1.0;
                values[3] = 2.0 * x[1];
                values[4] = 2.0 * x[2];
            }
        }
        true
    }

    fn eval_h(
        &mut self,
        _x: Option<&[Number]>,
        _new_x: bool,
        obj_factor: Number,
        lambda: Option<&[Number]>,
        _new_lambda: bool,
        mode: SparsityRequest<'_>,
    ) -> bool {
        match mode {
            SparsityRequest::Structure { irow, jcol } => {
                irow.copy_from_slice(&[0, 1, 1, 2, 3]);
                jcol.copy_from_slice(&[0, 0, 1, 2, 1]);
            }
            SparsityRequest::Values { values } => {
                let lam = lambda.map(|l| l[2]).unwrap_or(0.0);
                values[0] = obj_factor * 2.0; // ∂²/∂x0²
                values[1] = obj_factor; // ∂²/∂x1∂x0 (the x0·x1 term)
                values[2] = lam * 2.0; // ∂²/∂x1² (from the nonlinear row)
                values[3] = obj_factor * 2.0 + lam * 2.0; // ∂²/∂x2²
                values[4] = obj_factor; // ∂²/∂x1∂x3 (the x3·x1 term)
            }
        }
        true
    }

    fn finalize_solution(&mut self, sol: Solution<'_>, _d: &IpoptData, _q: &IpoptCq) {
        self.captured = Some(Captured {
            x: sol.x.to_vec(),
            z_l: sol.z_l.to_vec(),
            z_u: sol.z_u.to_vec(),
            lambda: sol.lambda.to_vec(),
            g: sol.g.to_vec(),
            obj: sol.obj_value,
        });
    }
}

fn opts(linear_eq_reduction: bool) -> PresolveOptions {
    PresolveOptions {
        enabled: true,
        linear_eq_reduction,
        // Keep the other phases out of the comparison: this test is about
        // the column reduction, and a dropped row would move the duals for
        // reasons of its own (the M24 attribution caveat).
        bound_tightening: false,
        redundant_constraint_removal: false,
        licq_check: false,
        warm_z_bounds: false,
        ..PresolveOptions::defaults()
    }
}

struct Outcome {
    captured: Captured,
    stats_obj: Number,
    reduced_n: i32,
    reduced_m: i32,
}

fn solve(linear_eq_reduction: bool) -> Outcome {
    let mut app = IpoptApplication::new();
    app.initialize().unwrap();
    // Unscaled, so the duals handed to `finalize_solution` are directly
    // comparable against the fixture's own analytic gradient.
    app.options_mut()
        .set_string_value("nlp_scaling_method", "none", true, false)
        .unwrap();

    let concrete = Rc::new(RefCell::new(Fixture::default()));
    let presolve = Rc::new(RefCell::new(PresolveTnlp::new(
        Rc::clone(&concrete) as Rc<RefCell<dyn TNLP>>,
        opts(linear_eq_reduction),
    )));
    let elim = Rc::new(RefCell::new(LinearEqElimTnlp::new(
        Rc::clone(&presolve) as Rc<RefCell<dyn TNLP>>,
        opts(linear_eq_reduction),
    )));
    let info = elim.borrow_mut().get_nlp_info().expect("dims");
    let _ = app.optimize_tnlp(Rc::clone(&elim) as Rc<RefCell<dyn TNLP>>);

    Outcome {
        captured: concrete.borrow().captured.clone().expect("finalized"),
        stats_obj: app.statistics().final_objective,
        reduced_n: info.n,
        reduced_m: info.m,
    }
}

#[test]
fn the_reduction_removes_the_determined_columns_and_their_rows() {
    let on = solve(true);
    assert_eq!(on.reduced_n, 2, "x0 and x3 should be gone");
    assert_eq!(on.reduced_m, 1, "only the nonlinear row should survive");

    let off = solve(false);
    assert_eq!(off.reduced_n, 4);
    assert_eq!(off.reduced_m, 3);
}

#[test]
fn the_solution_is_reported_in_full_space_and_matches_the_bare_solve() {
    let on = solve(true);
    let off = solve(false);

    assert_eq!(
        on.captured.x.len(),
        4,
        "finalize_solution must see all 4 columns"
    );
    assert_eq!(on.captured.z_l.len(), 4);
    assert_eq!(on.captured.lambda.len(), 3, "and all 3 rows");
    assert_eq!(on.captured.g.len(), 3);

    assert!(
        (on.captured.obj - off.captured.obj).abs() < 1e-7,
        "objective diverged: reduced={} full={}",
        on.captured.obj,
        off.captured.obj
    );
    assert!((on.stats_obj - on.captured.obj).abs() < 1e-9);
    for j in 0..4 {
        assert!(
            (on.captured.x[j] - off.captured.x[j]).abs() < 1e-6,
            "x[{j}] diverged: reduced={} full={}",
            on.captured.x[j],
            off.captured.x[j]
        );
    }
}

#[test]
fn the_eliminated_rows_are_satisfied_and_reported() {
    let on = solve(true);
    let x = &on.captured.x;
    assert!(
        (x[0] - 2.0 * x[1]).abs() < 1e-9,
        "x0 = 2·x1 violated: {x:?}"
    );
    assert!((x[3] - 1.0).abs() < 1e-9, "x3 = 1 violated: {x:?}");
    // `g` for the consumed rows is re-evaluated, not left at zero.
    assert!((on.captured.g[0] - 0.0).abs() < 1e-9);
    assert!((on.captured.g[1] - 1.0).abs() < 1e-9);
    assert!((on.captured.g[2] - 2.0).abs() < 1e-7);
}

#[test]
fn recovered_multipliers_satisfy_full_space_stationarity() {
    // The whole point of the reverse sweep: the consumed rows come back
    // with multipliers that close `∇f + Jᵀλ − z_l + z_u = 0` — POUNCE's
    // `finalize_solution` convention — in the *original* variable space,
    // not with zeros.
    let on = solve(true);
    let c = &on.captured;
    let mut grad = [0.0; 4];
    Fixture::grad_f(&c.x, &mut grad);
    let jac = jac_dense(&c.x);
    for j in 0..4 {
        let mut resid = grad[j] - c.z_l[j] + c.z_u[j];
        for (r, row) in jac.iter().enumerate() {
            resid += row[j] * c.lambda[r];
        }
        assert!(
            resid.abs() < 1e-6,
            "stationarity at column {j} = {resid}; λ = {:?}",
            c.lambda
        );
    }
    assert!(
        c.lambda[0].abs() > 1e-8 || c.lambda[1].abs() > 1e-8,
        "the consumed rows came back with zero multipliers: {:?}",
        c.lambda
    );
}

#[test]
fn duals_match_the_bare_solve() {
    let on = solve(true);
    let off = solve(false);
    for r in 0..3 {
        assert!(
            (on.captured.lambda[r] - off.captured.lambda[r]).abs() < 1e-5,
            "λ[{r}] diverged: reduced={} full={}",
            on.captured.lambda[r],
            off.captured.lambda[r]
        );
    }
}
