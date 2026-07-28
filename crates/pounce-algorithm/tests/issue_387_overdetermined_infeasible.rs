//! gh #387 — an over-determined system that is *provably* infeasible must
//! report `InfeasibleProblemDetected`, not `NotEnoughDegreesOfFreedom`.
//!
//! `x == 0.2` with `x == 0.8` over `x in [0, 1]` has an empty feasible set —
//! about as provable as infeasibility gets — yet it exited through the DOF
//! gate (1 variable < 2 equality rows) with the structural 5xx error before
//! anything could look at the constraints. The gate fires before a single
//! iteration runs, so no downstream mechanism ever had the chance to detect
//! the contradiction.
//!
//! The fix consults presolve's bound-propagation certification on the DOF
//! failure path (independent of the `presolve` master switch — nothing is
//! transformed, no solve runs through the probe). The certification's
//! fail-closed safety net is inherited wholesale:
//!
//! * a *consistent* over-determined system still reports the DOF error, and
//! * at row scales so small that the solver's own acceptance test would wave
//!   any point through (`|s*x - 0.2*s| <= tol` for every `x` in the box), the
//!   witness-refutation rule withholds the proof and the DOF error stands —
//!   "proved infeasible" is never claimed for a model the solver would accept
//!   as feasible.

use pounce_algorithm::application::IpoptApplication;
use pounce_common::types::Number;
use pounce_nlp::return_codes::ApplicationReturnStatus;
use pounce_nlp::tnlp::{
    BoundsInfo, IndexStyle, IpoptCq, IpoptData, Linearity, NlpInfo, Solution, SparsityRequest,
    StartingPoint, TNLP,
};
use std::cell::RefCell;
use std::rc::Rc;

/// `min x^2  s.t.  s*a1*x == s*r1,  s*a2*x == s*r2,  x in [0, 1]`.
///
/// Every row is scaled by the same `s > 0`, which leaves the feasible set
/// exactly unchanged; the contradiction (or consistency) lives in
/// `(a1, r1, a2, r2)`.
struct TwoEqualities {
    scale: Number,
    a: [Number; 2],
    r: [Number; 2],
}

impl TwoEqualities {
    fn contradictory(scale: Number) -> Self {
        Self {
            scale,
            a: [1.0, 1.0],
            r: [0.2, 0.8],
        }
    }

    fn consistent(scale: Number) -> Self {
        Self {
            scale,
            a: [1.0, 2.0],
            r: [0.2, 0.4],
        }
    }
}

impl TNLP for TwoEqualities {
    fn get_nlp_info(&mut self) -> Option<NlpInfo> {
        Some(NlpInfo {
            n: 1,
            m: 2,
            nnz_jac_g: 2,
            nnz_h_lag: 1,
            index_style: IndexStyle::C,
        })
    }

    fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
        b.x_l[0] = 0.0;
        b.x_u[0] = 1.0;
        for i in 0..2 {
            b.g_l[i] = self.scale * self.r[i];
            b.g_u[i] = self.scale * self.r[i];
        }
        true
    }

    fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
        sp.x[0] = 0.5;
        true
    }

    fn get_constraints_linearity(&mut self, types: &mut [Linearity]) -> bool {
        types.fill(Linearity::Linear);
        true
    }

    fn eval_f(&mut self, x: &[Number], _new_x: bool) -> Option<Number> {
        Some(x[0] * x[0])
    }

    fn eval_grad_f(&mut self, x: &[Number], _new_x: bool, grad: &mut [Number]) -> bool {
        grad[0] = 2.0 * x[0];
        true
    }

    fn eval_g(&mut self, x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
        g[0] = self.scale * self.a[0] * x[0];
        g[1] = self.scale * self.a[1] * x[0];
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
                irow.copy_from_slice(&[0, 1]);
                jcol.copy_from_slice(&[0, 0]);
            }
            SparsityRequest::Values { values } => {
                values[0] = self.scale * self.a[0];
                values[1] = self.scale * self.a[1];
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
                irow[0] = 0;
                jcol[0] = 0;
            }
            SparsityRequest::Values { values } => {
                values[0] = 2.0 * obj_factor;
            }
        }
        true
    }

    fn finalize_solution(&mut self, _sol: Solution<'_>, _d: &IpoptData, _q: &IpoptCq) {}
}

/// Solve with all-default options — in particular `presolve` stays at its
/// default "no", which is the configuration the issue was filed against.
fn solve(problem: TwoEqualities) -> ApplicationReturnStatus {
    let mut app = IpoptApplication::new();
    app.options_mut()
        .set_integer_value("print_level", 0, true, false)
        .unwrap();
    app.initialize().unwrap();
    let tnlp: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(problem));
    app.optimize_tnlp(tnlp)
}

#[test]
fn contradictory_equalities_are_infeasible_not_dof_error() {
    let status = solve(TwoEqualities::contradictory(1.0));
    assert_eq!(
        status,
        ApplicationReturnStatus::InfeasibleProblemDetected,
        "x == 0.2 with x == 0.8 is provably infeasible; the structural DOF \
         error is the strictly weaker answer"
    );
}

/// Multiplying every row by `s > 0` leaves the feasible set unchanged, so the
/// verdict must not depend on `s` wherever the certification is willing to
/// speak at all.
#[test]
fn verdict_is_scale_invariant_where_certifiable() {
    for k in [-6, -4, -2, 0, 2, 4, 6, 8, 10, 12] {
        let status = solve(TwoEqualities::contradictory(10.0_f64.powi(k)));
        assert_eq!(
            status,
            ApplicationReturnStatus::InfeasibleProblemDetected,
            "row scale 1e{k}: same empty feasible set, different verdict"
        );
    }
}

/// Below the certification floor the proof is *withheld*, not manufactured:
/// at `s = 1e-12` every point in `[0, 1]` satisfies both rows within the
/// solver's own acceptance tolerance, so claiming "proved infeasible" would
/// contradict what the solver itself would report if it could run. The
/// fail-closed answer is the structural DOF error, unchanged from before.
#[test]
fn sub_tolerance_scales_keep_the_dof_error() {
    let status = solve(TwoEqualities::contradictory(1e-12));
    assert_eq!(status, ApplicationReturnStatus::NotEnoughDegreesOfFreedom);
}

/// A consistent over-determined system (`x == 0.2`, `2x == 0.4`) is not
/// infeasible; the structural DOF error must survive the fix.
#[test]
fn consistent_equalities_still_report_dof_error() {
    let status = solve(TwoEqualities::consistent(1.0));
    assert_eq!(status, ApplicationReturnStatus::NotEnoughDegreesOfFreedom);
}
