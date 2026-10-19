//! An active-set SQP warm start is staged in the caller's space and must be
//! mapped into the algorithm's, as the exact inverse of what the solve
//! reports on the way out.
//!
//! `set_sqp_warm_start` used to take `x` / `lambda_g` / `lambda_x` as the
//! algorithm's own iterate, while every frontend (C API `IpoptSolve`,
//! `pounce-py`'s `solve(working_set=...)`) staged the user's `x`, `mult_g`
//! and `z_L − z_U`. The algorithm's `lambda_g` is `[y_c; y_d]` — equality
//! rows first — times `obj_scaling_factor`, so on a model whose equality is
//! not the first row, the converged multipliers fed straight back came out
//! at iteration 0 with the rows swapped: `[6.667, 3.667]` became
//! `[3.667, 6.667]`, and under `obj_scaling_factor = 10` additionally off by
//! 10.
//!
//! The model is chosen so each half of the transform has something to move:
//! `g0` is an active inequality and `g1` an equality (so `c`/`d` order is
//! the reverse of `g` order), `x0` is free and sits before the bounded `x1`
//! (so the compressed bound spaces do not line up with variable indices),
//! `x1`'s upper bound is active, and `g0` carries a factor `K` so
//! gradient-based scaling has a row to scale. The per-variable leg runs
//! under `user-scaling` with non-unit `x` factors.

use pounce_algorithm::application::IpoptApplication;
use pounce_algorithm::intermediate::with_current_user_iterate;
use pounce_algorithm::sqp::SqpIterates;
use pounce_common::types::Number;
use pounce_nlp::return_codes::ApplicationReturnStatus;
use pounce_nlp::tnlp::{
    BoundsInfo, IndexStyle, IpoptCq, IpoptData, IterStats, NlpInfo, ScalingRequest, Solution,
    SparsityRequest, StartingPoint, TNLP,
};
use std::cell::RefCell;
use std::rc::Rc;

#[derive(Clone, Debug, Default)]
struct Point {
    x: Vec<Number>,
    z_l: Vec<Number>,
    z_u: Vec<Number>,
    lambda: Vec<Number>,
}

/// `min (x0-3)² + (x1-3)² + (x2-1)²`
/// s.t. `g0 = K(x0 + x1) ≤ 2.5K`, `g1 = x0 − x2 = 0`,
/// `x0` free, `x1 ∈ [-10, 1]`, `x2 ≥ 0`.
struct Model {
    k: Number,
    /// `(obj, g, x)` factors for `nlp_scaling_method = user-scaling`.
    user_scaling: Option<(Number, [Number; 2], [Number; 3])>,
    out: Option<Point>,
    iter0: Option<Point>,
}

impl Model {
    fn new(k: Number, user_scaling: Option<(Number, [Number; 2], [Number; 3])>) -> Self {
        Self {
            k,
            user_scaling,
            out: None,
            iter0: None,
        }
    }
}

impl TNLP for Model {
    fn get_nlp_info(&mut self) -> Option<NlpInfo> {
        Some(NlpInfo {
            n: 3,
            m: 2,
            nnz_jac_g: 4,
            nnz_h_lag: 3,
            index_style: IndexStyle::C,
        })
    }
    fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
        b.x_l.copy_from_slice(&[-1e19, -10.0, 0.0]);
        b.x_u.copy_from_slice(&[1e19, 1.0, 1e19]);
        b.g_l.copy_from_slice(&[-1e19, 0.0]);
        b.g_u.copy_from_slice(&[2.5 * self.k, 0.0]);
        true
    }
    fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
        sp.x.copy_from_slice(&[0.0, 0.0, 0.5]);
        true
    }
    fn get_scaling_parameters(&mut self, r: ScalingRequest<'_>) -> bool {
        let Some((o, g, x)) = self.user_scaling else {
            return false;
        };
        *r.obj_scaling = o;
        *r.use_g_scaling = true;
        r.g_scaling.copy_from_slice(&g);
        *r.use_x_scaling = true;
        r.x_scaling.copy_from_slice(&x);
        true
    }
    fn eval_f(&mut self, x: &[Number], _n: bool) -> Option<Number> {
        Some((x[0] - 3.0).powi(2) + (x[1] - 3.0).powi(2) + (x[2] - 1.0).powi(2))
    }
    fn eval_grad_f(&mut self, x: &[Number], _n: bool, g: &mut [Number]) -> bool {
        g[0] = 2.0 * (x[0] - 3.0);
        g[1] = 2.0 * (x[1] - 3.0);
        g[2] = 2.0 * (x[2] - 1.0);
        true
    }
    fn eval_g(&mut self, x: &[Number], _n: bool, g: &mut [Number]) -> bool {
        g[0] = self.k * (x[0] + x[1]);
        g[1] = x[0] - x[2];
        true
    }
    fn eval_jac_g(&mut self, _x: Option<&[Number]>, _n: bool, mode: SparsityRequest<'_>) -> bool {
        match mode {
            SparsityRequest::Structure { irow, jcol } => {
                irow.copy_from_slice(&[0, 0, 1, 1]);
                jcol.copy_from_slice(&[0, 1, 0, 2]);
            }
            SparsityRequest::Values { values } => {
                values.copy_from_slice(&[self.k, self.k, 1.0, -1.0]);
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
                irow.copy_from_slice(&[0, 1, 2]);
                jcol.copy_from_slice(&[0, 1, 2]);
            }
            SparsityRequest::Values { values } => values.copy_from_slice(&[2.0 * of; 3]),
        }
        true
    }
    fn intermediate_callback(&mut self, st: IterStats, _d: &IpoptData, _q: &IpoptCq) -> bool {
        if st.iter == 0 {
            self.iter0 = with_current_user_iterate(|u| Point {
                x: u.x.clone(),
                z_l: u.z_l.clone(),
                z_u: u.z_u.clone(),
                lambda: u.lambda.clone(),
            });
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

fn app(opts: &[(&str, &str)]) -> IpoptApplication {
    let mut a = IpoptApplication::new();
    a.initialize().unwrap();
    let o = a.options_mut();
    o.set_integer_value("print_level", 0, true, false).unwrap();
    o.set_string_value("algorithm", "active-set-sqp", true, false)
        .unwrap();
    for (k, v) in opts {
        match v.parse::<Number>() {
            Ok(n) => o.set_numeric_value(k, n, true, false).unwrap(),
            Err(_) => o.set_string_value(k, v, true, false).unwrap(),
        };
    }
    a
}

/// Cold solve, then a warm solve seeded with exactly what the cold one
/// reported. Returns the cold solution, the warm solve's iteration-0 user
/// iterate (`None` when the callback did not fire), and the warm solve's
/// iteration count.
fn round_trip(model: Model, opts: &[(&str, &str)]) -> (Point, Option<Point>, i32) {
    let p = Rc::new(RefCell::new(model));
    let mut a = app(opts);
    let st = a.optimize_tnlp(p.clone());
    assert_eq!(st, ApplicationReturnStatus::SolveSucceeded, "cold solve");
    let cold = p.borrow_mut().out.take().expect("cold finalize");
    let ws = a.last_sqp_working_set().cloned();
    assert!(ws.is_some(), "cold solve must publish a working set");
    p.borrow_mut().iter0 = None;

    a.set_sqp_warm_start(SqpIterates {
        x: cold.x.clone(),
        lambda_g: cold.lambda.clone(),
        lambda_x: cold.z_l.iter().zip(&cold.z_u).map(|(l, u)| l - u).collect(),
        working: ws,
    });
    let st = a.optimize_tnlp(p.clone());
    assert_eq!(st, ApplicationReturnStatus::SolveSucceeded, "warm solve");
    let iters = a.statistics().iteration_count;
    let iter0 = p.borrow_mut().iter0.take();
    (cold, iter0, iters)
}

fn assert_close(what: &str, got: &[Number], want: &[Number]) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!(
            (g - w).abs() <= 1e-9 * (1.0 + w.abs()),
            "{what}[{i}]: iteration 0 has {g:e}, the seed was {w:e} (got {got:?}, seed {want:?})"
        );
    }
}

fn assert_iter0_is_seed(seed: &Point, iter0: &Point) {
    assert_close("lambda", &iter0.lambda, &seed.lambda);
    assert_close("x", &iter0.x, &seed.x);
    assert_close("z_l", &iter0.z_l, &seed.z_l);
    assert_close("z_u", &iter0.z_u, &seed.z_u);
}

/// The fixture has to make every part of the transform matter, or the
/// tests below pass on an identity map: both constraint multipliers and
/// the `x1` upper-bound multiplier must be nonzero, and unequal.
fn assert_fixture_is_informative(cold: &Point, k: Number) {
    let (l0, l1) = (cold.lambda[0] * k, cold.lambda[1]);
    assert!(l0.abs() > 1e-3 && l1.abs() > 1e-3, "λ = {:?}", cold.lambda);
    assert!((l0.abs() - l1.abs()).abs() > 1e-3, "λ = {:?}", cold.lambda);
    assert!(cold.z_u[1] > 1e-3, "x1's upper bound inactive: {cold:?}");
}

#[test]
fn unscaled_seed_is_the_iteration_zero_iterate() {
    let (cold, iter0, iters) = round_trip(Model::new(1.0, None), &[("nlp_scaling_method", "none")]);
    assert_fixture_is_informative(&cold, 1.0);
    let iter0 = iter0.expect("intermediate callback at iteration 0");
    assert_iter0_is_seed(&cold, &iter0);
    assert!(
        iters <= 1,
        "warm solve from the optimum took {iters} iterations"
    );
}

#[test]
fn objective_and_gradient_scaled_seed_is_the_iteration_zero_iterate() {
    let k = 1e4;
    let (cold, iter0, iters) = round_trip(
        Model::new(k, None),
        &[
            ("nlp_scaling_method", "gradient-based"),
            ("obj_scaling_factor", "10"),
        ],
    );
    assert_fixture_is_informative(&cold, k);
    let iter0 = iter0.expect("intermediate callback at iteration 0");
    assert_iter0_is_seed(&cold, &iter0);
    assert!(
        iters <= 1,
        "warm solve from the optimum took {iters} iterations"
    );
}

/// Per-variable (`x`) scaling sits below the NLP in a `ScalingTnlp`, which
/// does not forward `intermediate_callback`, so this leg reads the seed back
/// through `finalize_solution` instead: with `sqp_max_iter = 0` the loop
/// never runs and the reported point is the staged one, mapped in and back
/// out. (A full warm solve is no evidence here: on this QP one exact-Hessian
/// step from a wrong multiplier still lands on the optimum.)
#[test]
fn per_variable_scaled_seed_round_trips_through_a_zero_iteration_solve() {
    let model = Model::new(1.0, Some((3.0, [0.01, 5.0], [2.0, 0.5, 4.0])));
    let p = Rc::new(RefCell::new(model));
    let mut a = app(&[("nlp_scaling_method", "user-scaling")]);
    assert_eq!(
        a.optimize_tnlp(p.clone()),
        ApplicationReturnStatus::SolveSucceeded
    );
    let cold = p.borrow_mut().out.take().unwrap();
    assert_fixture_is_informative(&cold, 1.0);
    a.set_sqp_warm_start(SqpIterates {
        x: cold.x.clone(),
        lambda_g: cold.lambda.clone(),
        lambda_x: cold.z_l.iter().zip(&cold.z_u).map(|(l, u)| l - u).collect(),
        working: a.last_sqp_working_set().cloned(),
    });
    a.options_mut()
        .set_integer_value("sqp_max_iter", 0, true, false)
        .unwrap();
    let _ = a.optimize_tnlp(p.clone());
    let warm = p
        .borrow_mut()
        .out
        .take()
        .expect("finalize after 0 iterations");
    assert_iter0_is_seed(&cold, &warm);
}
