//! The `sensitivity` feature's surface, reached through the facade only
//! (gh #561): solve an NLP and get `∂x*/∂p` from the converged KKT factor,
//! without depending on `pounce-sensitivity` directly.
//!
//! The fixture is upstream sIPOPT's `parametric_cpp` problem, so the numbers
//! asserted here are the same golden values `pounce-sensitivity`'s own tests
//! pin against upstream 3.14.19 — a facade re-export that silently changed
//! them would fail.
#![cfg(feature = "sensitivity")]

use std::cell::RefCell;
use std::rc::Rc;

use pounce_rs::prelude::*;
use pounce_rs::sensitivity::SensSolve;

/// ```text
/// min  x₁² + x₂² + x₃²
/// s.t. 6x₁ + 3x₂ + 2x₃ − η₁ = 0
///      η₂x₁ + x₂ − x₃ − 1  = 0
///      η₁ = nominal_eta1        <- pinned parameter, row 2
///      η₂ = nominal_eta2        <- pinned parameter, row 3
/// ```
/// The parameters are lifted into variables `x₄`, `x₅` and pinned by the
/// two trailing equalities; perturbing their right-hand sides is what
/// `SensSolve::with_deltas` differentiates against.
struct ParametricTnlp {
    nominal_eta1: Number,
    nominal_eta2: Number,
}

impl TNLP for ParametricTnlp {
    fn get_nlp_info(&mut self) -> Option<NlpInfo> {
        Some(NlpInfo {
            n: 5,
            m: 4,
            nnz_jac_g: 10,
            nnz_h_lag: 5,
            index_style: IndexStyle::C,
        })
    }

    fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
        for k in 0..3 {
            b.x_l[k] = 0.0;
            b.x_u[k] = 1.0e19;
        }
        for k in 3..5 {
            b.x_l[k] = -1.0e19;
            b.x_u[k] = 1.0e19;
        }
        b.g_l[0] = 0.0;
        b.g_u[0] = 0.0;
        b.g_l[1] = 0.0;
        b.g_u[1] = 0.0;
        b.g_l[2] = self.nominal_eta1;
        b.g_u[2] = self.nominal_eta1;
        b.g_l[3] = self.nominal_eta2;
        b.g_u[3] = self.nominal_eta2;
        true
    }

    fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
        sp.x.copy_from_slice(&[0.15, 0.15, 0.0, 0.0, 0.0]);
        true
    }

    fn eval_f(&mut self, x: &[Number], _new_x: bool) -> Option<Number> {
        Some(x[0] * x[0] + x[1] * x[1] + x[2] * x[2])
    }

    fn eval_grad_f(&mut self, x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
        g.copy_from_slice(&[2.0 * x[0], 2.0 * x[1], 2.0 * x[2], 0.0, 0.0]);
        true
    }

    fn eval_g(&mut self, x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
        let (x1, x2, x3, eta1, eta2) = (x[0], x[1], x[2], x[3], x[4]);
        g[0] = 6.0 * x1 + 3.0 * x2 + 2.0 * x3 - eta1;
        g[1] = eta2 * x1 + x2 - x3 - 1.0;
        g[2] = eta1;
        g[3] = eta2;
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
                irow.copy_from_slice(&[0, 0, 0, 0, 1, 1, 1, 1, 2, 3]);
                jcol.copy_from_slice(&[0, 1, 2, 3, 0, 1, 2, 4, 3, 4]);
            }
            SparsityRequest::Values { values } => {
                let x = x.expect("eval_jac_g(Values) without x");
                values.copy_from_slice(&[6.0, 3.0, 2.0, -1.0, x[4], 1.0, -1.0, x[0], 1.0, 1.0]);
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
                irow.copy_from_slice(&[0, 1, 2, 4, 0]);
                jcol.copy_from_slice(&[0, 1, 2, 0, 0]);
            }
            SparsityRequest::Values { values } => {
                let lam = lambda.expect("eval_h(Values) without lambda");
                values.copy_from_slice(&[
                    2.0 * obj_factor,
                    2.0 * obj_factor,
                    2.0 * obj_factor,
                    lam[1],
                    0.0,
                ]);
            }
        }
        true
    }

    fn finalize_solution(&mut self, _sol: Solution<'_>, _d: &IpoptData, _q: &IpoptCq) {}
}

fn quiet_app() -> IpoptApplication {
    let mut app = IpoptApplication::new();
    app.options_mut()
        .set_integer_value("print_level", 0, true, false)
        .unwrap();
    app.options_mut()
        .set_string_value("sb", "yes", true, false)
        .unwrap();
    app.initialize().unwrap();
    app
}

fn tnlp() -> Rc<RefCell<dyn TNLP>> {
    Rc::new(RefCell::new(ParametricTnlp {
        nominal_eta1: 5.0,
        nominal_eta2: 1.0,
    }))
}

/// Upstream sIPOPT's Δx for Δη = (−0.5, 0) from the nominal (5, 1).
const UPSTREAM_DX: [Number; 5] = [
    0.576_530_601_168_321_9 - 0.632_653_057_519_998_2,
    0.377_551_038_130_684_8 - 0.387_755_107_968_002_7,
    -0.045_918_360_700_993_31 - 0.020_408_165_488_001_08,
    -0.5,
    0.0,
];

#[test]
fn parametric_step_through_the_facade_matches_upstream_sipopt() {
    let mut app = quiet_app();

    let result = SensSolve::new(vec![2, 3])
        .with_deltas(vec![-0.5, 0.0])
        .run(&mut app, tnlp());

    assert!(
        matches!(
            result.status,
            ApplicationReturnStatus::SolveSucceeded
                | ApplicationReturnStatus::SolvedToAcceptableLevel
        ),
        "solve failed: {:?}",
        result.status
    );
    assert!(
        result.error.is_none(),
        "sensitivity error: {:?}",
        result.error
    );

    let dx = result.dx.expect("dx populated when with_deltas was set");
    assert_eq!(dx.len(), 5);
    for k in 0..5 {
        assert!(
            (dx[k] - UPSTREAM_DX[k]).abs() < 1e-8,
            "dx[{k}] = {}, upstream = {}",
            dx[k],
            UPSTREAM_DX[k]
        );
    }
}

#[test]
fn reduced_hessian_is_reachable_through_the_facade() {
    let mut app = quiet_app();

    let result = SensSolve::new(vec![2, 3])
        .with_reduced_hessian()
        .run(&mut app, tnlp());

    assert!(
        result.error.is_none(),
        "sensitivity error: {:?}",
        result.error
    );
    let rh = result
        .reduced_hessian
        .expect("reduced Hessian populated when requested");
    // Two pinned parameters ⇒ a 2×2 reduced Hessian, symmetric.
    assert_eq!(rh.len(), 4, "expected a 2x2 reduced Hessian, got {rh:?}");
    assert!(
        (rh[1] - rh[2]).abs() < 1e-8,
        "reduced Hessian must be symmetric: {rh:?}"
    );
}
