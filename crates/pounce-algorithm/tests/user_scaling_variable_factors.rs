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
    /// The finalized `x`, in the user's own units.
    x: Vec<Number>,
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

    fn finalize_solution(&mut self, sol: Solution<'_>, _d: &IpoptData, _q: &IpoptCq) {
        self.solved = true;
        self.x = sol.x.to_vec();
    }
}

/// Run the problem under `user-scaling` with the given variable
/// factors; returns `(status, reached_finalize_solution, x)`,
/// where `x` is the finalized point in the user's own units.
fn solve_with_x_scaling(x_scaling: &[Number]) -> (ApplicationReturnStatus, bool, Vec<Number>) {
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
        x: Vec::new(),
    }));
    let tnlp: Rc<RefCell<dyn TNLP>> = Rc::clone(&concrete) as _;
    let status = app.optimize_tnlp(tnlp);
    let solved = concrete.borrow().solved;
    let x = concrete.borrow().x.clone();
    (status, solved, x)
}

/// Non-unit variable factors are APPLIED, as of gh#486 stage 2.
///
/// This asserted the opposite through stage 1, when the core had no
/// representation for a change of variables and refusing was the only
/// honest answer. The wrapper supplies that representation, so the
/// solve runs and reports its answer in the user's own units.
#[test]
fn variable_scaling_factors_are_applied() {
    let (status, solved, scaled_x) = solve_with_x_scaling(&[1.0, 1e4]);
    assert!(
        matches!(
            status,
            ApplicationReturnStatus::SolveSucceeded
                | ApplicationReturnStatus::SolvedToAcceptableLevel
        ),
        "a variable-scaled solve must run, got {status:?}",
    );
    assert!(solved, "finalize_solution should have run");

    // The same problem without factors: the answer must not move,
    // because scaling changes conditioning and nothing else.
    let (_, _, plain_x) = solve_with_x_scaling(&[1.0, 1.0]);
    assert_eq!(scaled_x.len(), plain_x.len());
    for (i, (a, b)) in scaled_x.iter().zip(plain_x.iter()).enumerate() {
        assert!(
            (a - b).abs() <= 1e-6 * b.abs().max(1.0),
            "x[{i}]: scaled solve reported {a}, unscaled {b}",
        );
    }
}

/// An all-ones `x_scaling` asks for nothing, so it is a no-op and the
/// solve proceeds normally with the objective/constraint factors. The
/// refusal must be about factors that change the problem, not about
/// the request channel being used at all.
#[test]
fn unit_variable_scaling_still_solves() {
    let (status, solved, _) = solve_with_x_scaling(&[1.0, 1.0]);
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
