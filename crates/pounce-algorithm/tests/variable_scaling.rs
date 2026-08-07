//! End-to-end acceptance for per-variable scaling (issue #486 stage 2).
//!
//! The wrapper installed by `optimize_tnlp` under
//! `nlp_scaling_method=user-scaling` must change how the problem is
//! CONDITIONED without changing what the problem IS. These tests solve
//! the same model with and without factors and require the answers to
//! agree in the user's own units, which is the parity criterion issue
//! #483 asked for, stated on the solution rather than iterate for
//! iterate.

use std::cell::RefCell;
use std::rc::Rc;

use pounce_algorithm::application::IpoptApplication;
use pounce_common::types::Number;
use pounce_nlp::tnlp::{
    BoundsInfo, IndexStyle, IpoptCq, IpoptData, NlpInfo, ScalingRequest, Solution, SparsityRequest,
    StartingPoint, TNLP,
};

/// `min (x0 - 3)^2 + (x1 - 2e6)^2  s.t.  x0 + x1/1e6 = 7`.
///
/// The second variable lives six orders of magnitude away from the
/// first, which is what a factor of 1e-6 on it is meant to repair.
/// The right-hand side is 7 rather than 5 so the equality actually
/// binds: at the unconstrained minimum (3, 2e6) the row already reads
/// 5, and a constraint that is slack at the optimum would leave the
/// multipliers untested. Moving the row to 7 costs 4 through x0 and
/// 4e12 through x1, so the optimum is x0 = 5, x1 = 2e6.
struct Skewed {
    factors: Option<Vec<Number>>,
    solution: Rc<RefCell<Option<Vec<Number>>>>,
}

impl Skewed {
    fn new(factors: Option<Vec<Number>>) -> Self {
        Self {
            factors,
            solution: Rc::new(RefCell::new(None)),
        }
    }
}

impl TNLP for Skewed {
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
        b.x_l.copy_from_slice(&[-1.0e20, -1.0e20]);
        b.x_u.copy_from_slice(&[1.0e20, 1.0e20]);
        b.g_l[0] = 7.0;
        b.g_u[0] = 7.0;
        true
    }
    fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
        if sp.init_x {
            sp.x.copy_from_slice(&[0.0, 0.0]);
        }
        true
    }
    fn eval_f(&mut self, x: &[Number], _new_x: bool) -> Option<Number> {
        Some((x[0] - 3.0).powi(2) + (x[1] - 2.0e6).powi(2))
    }
    fn eval_grad_f(&mut self, x: &[Number], _new_x: bool, grad_f: &mut [Number]) -> bool {
        grad_f[0] = 2.0 * (x[0] - 3.0);
        grad_f[1] = 2.0 * (x[1] - 2.0e6);
        true
    }
    fn eval_g(&mut self, x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
        g[0] = x[0] + x[1] / 1.0e6;
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
                true
            }
            SparsityRequest::Values { values } => {
                values[0] = 1.0;
                values[1] = 1.0e-6;
                true
            }
        }
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
                true
            }
            SparsityRequest::Values { values } => {
                values[0] = 2.0 * obj_factor;
                values[1] = 2.0 * obj_factor;
                true
            }
        }
    }
    fn finalize_solution(&mut self, sol: Solution<'_>, _d: &IpoptData, _c: &IpoptCq) {
        *self.solution.borrow_mut() = Some(sol.x.to_vec());
    }
    fn get_scaling_parameters(&mut self, req: ScalingRequest<'_>) -> bool {
        match &self.factors {
            Some(f) => {
                *req.obj_scaling = 1.0;
                *req.use_x_scaling = true;
                req.x_scaling.copy_from_slice(f);
                *req.use_g_scaling = false;
                true
            }
            None => false,
        }
    }
}

/// Solve once, returning the solution in the user's own units.
fn solve(factors: Option<Vec<Number>>, user_scaling: bool) -> Option<Vec<Number>> {
    let mut app = IpoptApplication::new();
    app.initialize().unwrap();
    app.options_mut()
        .set_integer_value("print_level", 0, true, false);
    if user_scaling {
        app.options_mut()
            .set_string_value("nlp_scaling_method", "user-scaling", true, false);
    }
    let concrete = Rc::new(RefCell::new(Skewed::new(factors)));
    let seen = concrete.borrow().solution.clone();
    let tnlp: Rc<RefCell<dyn TNLP>> = concrete;
    let _ = app.optimize_tnlp(tnlp);
    let got = seen.borrow().clone();
    got
}

/// The whole point: factors change conditioning, not the answer. The
/// solution comes back in the user's units either way.
#[test]
fn variable_factors_do_not_move_the_solution() {
    let plain = solve(None, false).expect("unscaled solve finalizes");
    let scaled = solve(Some(vec![1.0, 1.0e-6]), true).expect("scaled solve finalizes");

    // Analytic optimum of the model as written.
    assert!(
        (plain[0] - 5.0).abs() < 1e-4 && (plain[1] - 2.0e6).abs() < 1e2,
        "unscaled solve landed at {plain:?}"
    );
    assert!(
        (scaled[0] - plain[0]).abs() < 1e-4,
        "x0: scaled {} vs unscaled {}",
        scaled[0],
        plain[0]
    );
    assert!(
        (scaled[1] - plain[1]).abs() < 1e2,
        "x1: scaled {} vs unscaled {}",
        scaled[1],
        plain[1]
    );
}

/// Stage 1 refused this. It must now run rather than return
/// `InvalidOption`, which is the user-visible half of stage 2.
#[test]
fn a_scaled_solve_is_no_longer_refused() {
    assert!(
        solve(Some(vec![1.0, 1.0e-6]), true).is_some(),
        "the solve was refused before reaching finalize_solution"
    );
}

/// Factors reach the wrapper only under `user-scaling`; every other
/// method leaves the problem alone, so an unrelated model carrying a
/// suffix is unaffected.
#[test]
fn factors_are_ignored_under_other_scaling_methods() {
    let with_factors_but_gradient_based =
        solve(Some(vec![1.0, 1.0e-6]), false).expect("solve finalizes");
    let plain = solve(None, false).expect("solve finalizes");
    assert!(
        (with_factors_but_gradient_based[0] - plain[0]).abs() < 1e-6,
        "the default scaling method must not consult the factors"
    );
}
