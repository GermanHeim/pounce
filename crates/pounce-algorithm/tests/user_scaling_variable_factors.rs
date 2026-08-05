//! Regression test for gh#483: `nlp_scaling_method=user-scaling` with
//! per-variable scaling factors must fail the solve, not run it with
//! the factors quietly thrown away.
//!
//! pounce models objective and constraint scaling only. Until this
//! test, `OrigIpoptNlp::scale_user_supplied` ended in
//!
//! ```text
//! // `use_x_scaling`: silently ignored (not modeled — see doc).
//! let _ = use_x_scaling;
//! ```
//!
//! so a TNLP that asked for variable scaling got a converged answer to
//! a differently-conditioned problem than the one it described, with
//! nothing in the log to say the request had been dropped. The
//! objective and constraint factors on the same request *were* applied,
//! which makes the discard worse than a plain no-op: the conditioning
//! that came back was neither what was asked for nor the unscaled
//! problem.
//!
//! Problem (n = 2, m = 1): `min (x0 - 3)^2 + (x1 - 2)^2` s.t.
//! `x0 + x1 = 1`. Small, convex, and solvable — so a failure here is
//! the refusal, never the model.

use pounce_algorithm::application::IpoptApplication;
use pounce_common::types::{Index, Number};
use pounce_nlp::return_codes::ApplicationReturnStatus;
use pounce_nlp::tnlp::{
    BoundsInfo, IndexStyle, IpoptCq, IpoptData, NlpInfo, ScalingRequest, Solution, SparsityRequest,
    StartingPoint, TNLP,
};
use std::cell::RefCell;
use std::rc::Rc;

/// `min (x0 - 3)^2 + (x1 - 2)^2` s.t. `x0 + x1 = 1`, reporting
/// whatever `x_scaling` it is constructed with.
struct UserScaled {
    x_scaling: Vec<Number>,
    solved: bool,
}

impl TNLP for UserScaled {
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
        b.x_l.copy_from_slice(&[-1e20, -1e20]);
        b.x_u.copy_from_slice(&[1e20, 1e20]);
        b.g_l[0] = 1.0;
        b.g_u[0] = 1.0;
        true
    }

    fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
        sp.x.copy_from_slice(&[0.0, 0.0]);
        true
    }

    fn get_scaling_parameters(&mut self, req: ScalingRequest<'_>) -> bool {
        *req.obj_scaling = 2.0;
        *req.use_g_scaling = true;
        req.g_scaling[0] = 4.0;
        *req.use_x_scaling = true;
        req.x_scaling.copy_from_slice(&self.x_scaling);
        true
    }

    fn eval_f(&mut self, x: &[Number], _new_x: bool) -> Option<Number> {
        Some((x[0] - 3.0).powi(2) + (x[1] - 2.0).powi(2))
    }

    fn eval_grad_f(&mut self, x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
        g[0] = 2.0 * (x[0] - 3.0);
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
                irow.copy_from_slice(&[0 as Index, 0]);
                jcol.copy_from_slice(&[0 as Index, 1]);
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
                irow.copy_from_slice(&[0 as Index, 1]);
                jcol.copy_from_slice(&[0 as Index, 1]);
            }
            SparsityRequest::Values { values } => {
                values.copy_from_slice(&[2.0 * obj_factor, 2.0 * obj_factor])
            }
        }
        true
    }

    fn finalize_solution(&mut self, _sol: Solution<'_>, _d: &IpoptData, _q: &IpoptCq) {
        self.solved = true;
    }
}

/// Run the problem under `user-scaling` with the given variable
/// factors; returns `(status, reached_finalize_solution)`.
fn solve_with_x_scaling(x_scaling: &[Number]) -> (ApplicationReturnStatus, bool) {
    let mut app = IpoptApplication::new();
    app.options_mut()
        .set_string_value("nlp_scaling_method", "user-scaling", true, false)
        .unwrap();
    app.options_mut()
        .set_integer_value("print_level", 0, true, false)
        .unwrap();
    app.initialize().unwrap();

    let concrete = Rc::new(RefCell::new(UserScaled {
        x_scaling: x_scaling.to_vec(),
        solved: false,
    }));
    let tnlp: Rc<RefCell<dyn TNLP>> = Rc::clone(&concrete) as _;
    let status = app.optimize_tnlp(tnlp);
    let solved = concrete.borrow().solved;
    (status, solved)
}

/// Non-unit variable factors are refused up front. Pre-fix this
/// returned `SolveSucceeded` after solving with only the objective and
/// constraint factors applied.
#[test]
fn variable_scaling_factors_are_refused() {
    let (status, solved) = solve_with_x_scaling(&[1.0, 1e4]);
    assert!(
        matches!(status, ApplicationReturnStatus::InvalidOption),
        "expected InvalidOption for an unmodelable x_scaling, got {status:?}",
    );
    assert!(
        !solved,
        "the solve must not run at all — a converged answer here is an \
         answer to a problem the caller did not describe",
    );
}

/// An all-ones `x_scaling` asks for nothing, so it is a no-op and the
/// solve proceeds normally with the objective/constraint factors. The
/// refusal must be about factors that change the problem, not about
/// the request channel being used at all.
#[test]
fn unit_variable_scaling_still_solves() {
    let (status, solved) = solve_with_x_scaling(&[1.0, 1.0]);
    assert!(
        matches!(
            status,
            ApplicationReturnStatus::SolveSucceeded
                | ApplicationReturnStatus::SolvedToAcceptableLevel
        ),
        "unit x_scaling must not block the solve, got {status:?}",
    );
    assert!(solved, "finalize_solution should have run");
}
