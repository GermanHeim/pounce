//! The SQP working-set contract through the facade only — the round trip
//! `last_sqp_working_set` → [`SqpIterates`] → `set_sqp_warm_start` that
//! `docs/src/active-set-sqp.md` §2–§3 documents for Rust. If the facade ever
//! stops exporting enough to write that sequence, this stops compiling.
#![cfg(feature = "qp")]

use std::cell::RefCell;
use std::rc::Rc;

use pounce_rs::prelude::*;
use pounce_rs::sqp::{BoundStatus, ConsStatus, SqpIterates, WorkingSet, classify_working_set};

/// min (x₀ − 1)² + (x₁ − 2)²  s.t.  x₀ + x₁ == 3,  0 ≤ x ≤ 5.
/// KKT optimum is (1, 2) — the equality is already satisfied there, so the
/// active set is the equality row alone and no bound is active.
struct Quad {
    x_star: Rc<RefCell<Option<Vec<Number>>>>,
}

impl TNLP for Quad {
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
        b.x_l.copy_from_slice(&[0.0, 0.0]);
        b.x_u.copy_from_slice(&[5.0, 5.0]);
        b.g_l[0] = 3.0;
        b.g_u[0] = 3.0;
        true
    }

    fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
        sp.x.copy_from_slice(&[0.5, 0.5]);
        true
    }

    fn eval_f(&mut self, x: &[Number], _new_x: bool) -> Option<Number> {
        Some((x[0] - 1.0).powi(2) + (x[1] - 2.0).powi(2))
    }

    fn eval_grad_f(&mut self, x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
        g[0] = 2.0 * (x[0] - 1.0);
        g[1] = 2.0 * (x[1] - 2.0);
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
                values.copy_from_slice(&[2.0 * obj_factor, 2.0 * obj_factor]);
            }
        }
        true
    }

    fn finalize_solution(&mut self, sol: Solution<'_>, _d: &IpoptData, _q: &IpoptCq) {
        *self.x_star.borrow_mut() = Some(sol.x.to_vec());
    }
}

fn sqp_app() -> IpoptApplication {
    let mut app = IpoptApplication::new();
    app.options_mut()
        .set_integer_value("print_level", 0, true, false)
        .unwrap();
    app.options_mut()
        .set_string_value("sb", "yes", true, false)
        .unwrap();
    app.initialize().unwrap();
    app.initialize_with_options_str("algorithm active-set-sqp\n")
        .unwrap();
    app
}

#[test]
fn working_set_round_trips_from_one_solve_into_the_next() {
    let x_star = Rc::new(RefCell::new(None));
    let tnlp: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(Quad {
        x_star: Rc::clone(&x_star),
    }));
    let mut app = sqp_app();

    // Cold solve leaves a working set behind.
    let status = app.optimize_tnlp(Rc::clone(&tnlp));
    assert_eq!(status, ApplicationReturnStatus::SolveSucceeded);
    let working: Option<WorkingSet> = app.last_sqp_working_set().cloned();
    assert!(working.is_some(), "cold SQP solve must yield a working set");

    let x = x_star.borrow().clone().expect("finalize_solution ran");
    assert!(
        (x[0] - 1.0).abs() < 1e-6 && (x[1] - 2.0).abs() < 1e-6,
        "x = {x:?}"
    );

    // Warm solve seeded from it — same answer, and a working set again.
    app.set_sqp_warm_start(SqpIterates {
        x: x.clone(),
        lambda_g: vec![0.0],
        lambda_x: vec![0.0, 0.0],
        working,
    });
    let status = app.optimize_tnlp(Rc::clone(&tnlp));
    assert_eq!(status, ApplicationReturnStatus::SolveSucceeded);
    assert!(app.last_sqp_working_set().is_some());

    let x_warm = x_star
        .borrow()
        .clone()
        .expect("finalize_solution ran again");
    for k in 0..2 {
        assert!(
            (x_warm[k] - x[k]).abs() < 1e-8,
            "warm start changed the answer: {x_warm:?} vs {x:?}"
        );
    }
}

#[test]
fn classify_working_set_derives_a_seed_from_a_predicted_point() {
    // The sensitivity-predictor path: no previous solve to carry, so the
    // working set is classified from a point and its multipliers.
    // At (1, 2) the equality binds and neither bound is active.
    let ws = classify_working_set(
        &[0.0, 0.0], // lambda_x, packed z_l − z_u
        &[0.0],      // lambda_g
        1,           // m_eq — row 0 is the equality
        &[1.0, 2.0], // x
        &[0.0, 0.0], // x_l
        &[5.0, 5.0], // x_u
        &[3.0],      // g(x)
        &[3.0],      // g_l
        &[3.0],      // g_u
        1e-8,
        1e-6,
    );

    assert_eq!(ws.constraints[0], ConsStatus::Equality);
    assert_eq!(ws.bounds[0], BoundStatus::Inactive);
    assert_eq!(ws.bounds[1], BoundStatus::Inactive);
}

#[test]
fn switching_to_the_sqp_path_needs_no_working_set_types() {
    // The §2 claim: the algorithm flip alone is one option on the builder.
    struct P;
    impl pounce_rs::Problem for P {
        fn objective(&self, x: &[f64]) -> f64 {
            (x[0] - 1.0).powi(2) + (x[1] - 2.0).powi(2)
        }
        fn n_constraints(&self) -> usize {
            1
        }
        fn constraints(&self, x: &[f64], g: &mut [f64]) {
            g[0] = x[0] + x[1];
        }
    }

    let sol = Nlp::new(P)
        .var_bounds(&[0.0, 0.0], &[5.0, 5.0])
        .constraint_bounds(&[3.0], &[3.0])
        .option_str("algorithm", "active-set-sqp")
        .solve();

    assert!(sol.success, "status = {:?}", sol.status);
    assert!(
        (sol.x[0] - 1.0).abs() < 1e-5 && (sol.x[1] - 2.0).abs() < 1e-5,
        "x = {:?}",
        sol.x
    );
}
