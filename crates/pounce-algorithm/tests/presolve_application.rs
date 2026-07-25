//! Public-library presolve integration coverage. These tests configure
//! `IpoptApplication` only.

use pounce_algorithm::application::IpoptApplication;
use pounce_common::types::Number;
use pounce_nlp::SolverReturn;
use pounce_nlp::return_codes::ApplicationReturnStatus;
use pounce_nlp::tnlp::{
    BoundsInfo, IndexStyle, IpoptCq, IpoptData, Linearity, NlpInfo, Solution, SparsityRequest,
    StartingPoint, TNLP,
};
use std::cell::RefCell;
use std::rc::Rc;

#[derive(Debug, Clone)]
struct FinalPayload {
    status: SolverReturn,
    x: Vec<Number>,
    z_l: Vec<Number>,
    g: Vec<Number>,
    lambda: Vec<Number>,
    obj: Number,
}

/// `x >= 2` tightens the original `x >= 0` lower bound, after which that
/// linear row is redundant.
#[derive(Default)]
struct TightenedRow {
    final_payload: Option<FinalPayload>,
    warm_z_seen: Option<(Vec<Number>, Vec<Number>)>,
    warm_lambda_seen: Option<Vec<Number>>,
}

impl TNLP for TightenedRow {
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
        b.x_u[0] = 10.0;
        b.g_l.copy_from_slice(&[2.0, -1.0e19]);
        b.g_u.copy_from_slice(&[1.0e19, 10.0]);
        true
    }

    fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
        sp.x[0] = 3.0;
        if sp.init_z {
            sp.z_l[0] = 3.0;
            sp.z_u[0] = 0.0;
        }
        if sp.init_lambda {
            sp.lambda.copy_from_slice(&[7.0, 11.0]);
        }
        if sp.init_z {
            self.warm_z_seen = Some((sp.z_l.to_vec(), sp.z_u.to_vec()));
        }
        if sp.init_lambda {
            self.warm_lambda_seen = Some(sp.lambda.to_vec());
        }
        true
    }

    fn get_constraints_linearity(&mut self, types: &mut [Linearity]) -> bool {
        types[0] = Linearity::Linear;
        types[1] = Linearity::NonLinear;
        true
    }

    fn eval_f(&mut self, x: &[Number], _new_x: bool) -> Option<Number> {
        Some((x[0] - 1.0).powi(2))
    }

    fn eval_grad_f(&mut self, x: &[Number], _new_x: bool, grad: &mut [Number]) -> bool {
        grad[0] = 2.0 * (x[0] - 1.0);
        true
    }

    fn eval_g(&mut self, x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
        g.copy_from_slice(&[x[0], (x[0] - 2.0).powi(2)]);
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
                irow.copy_from_slice(&[0, 1]);
                jcol.copy_from_slice(&[0, 0]);
            }
            SparsityRequest::Values { values } => {
                let x = x.expect("Jacobian values need x");
                values.copy_from_slice(&[1.0, 2.0 * (x[0] - 2.0)]);
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
                irow[0] = 0;
                jcol[0] = 0;
            }
            SparsityRequest::Values { values } => {
                values[0] = 2.0 * obj_factor + 2.0 * lambda.map_or(0.0, |v| v[1]);
            }
        }
        true
    }

    fn finalize_solution(&mut self, sol: Solution<'_>, _d: &IpoptData, _q: &IpoptCq) {
        self.final_payload = Some(FinalPayload {
            status: sol.status,
            x: sol.x.to_vec(),
            z_l: sol.z_l.to_vec(),
            g: sol.g.to_vec(),
            lambda: sol.lambda.to_vec(),
            obj: sol.obj_value,
        });
    }
}

fn solve(
    presolve: bool,
    warm_start: bool,
    max_iter: Option<i32>,
) -> (ApplicationReturnStatus, TightenedRow) {
    let mut app = IpoptApplication::new();
    app.options_mut()
        .set_string_value("presolve", if presolve { "yes" } else { "no" }, true, false)
        .unwrap();
    if warm_start {
        app.options_mut()
            .set_string_value("warm_start_init_point", "yes", true, false)
            .unwrap();
    }
    if let Some(value) = max_iter {
        app.options_mut()
            .set_integer_value("max_iter", value, true, false)
            .unwrap();
    }
    app.initialize().unwrap();

    let problem = Rc::new(RefCell::new(TightenedRow::default()));
    let tnlp: Rc<RefCell<dyn TNLP>> = Rc::clone(&problem) as Rc<RefCell<dyn TNLP>>;
    let status = app.optimize_tnlp(tnlp);
    let result = std::mem::take(&mut *problem.borrow_mut());
    (status, result)
}

#[test]
fn library_presolve_off_preserves_unwrapped_solution_and_callback_shape() {
    let (off_status, off) = solve(false, false, None);
    let (on_status, on) = solve(true, false, None);

    assert!(
        matches!(
            off_status,
            ApplicationReturnStatus::SolveSucceeded
                | ApplicationReturnStatus::SolvedToAcceptableLevel
        ),
        "unexpected presolve=no status: {off_status:?}"
    );
    assert!(
        matches!(
            on_status,
            ApplicationReturnStatus::SolveSucceeded
                | ApplicationReturnStatus::SolvedToAcceptableLevel
        ),
        "unexpected presolve=yes status: {on_status:?}"
    );
    let off_final = off.final_payload.expect("presolve=no finalize_solution");
    let on_final = on.final_payload.expect("presolve=yes finalize_solution");
    assert!((off_final.x[0] - 2.0).abs() < 1e-5);
    assert!((on_final.x[0] - off_final.x[0]).abs() < 1e-5);
    assert!((on_final.obj - off_final.obj).abs() < 1e-5);
    assert_eq!(off_final.g.len(), 2);
    assert_eq!(off_final.lambda.len(), 2);
    assert_eq!(on_final.g.len(), 2, "postsolve restores the dropped row");
    assert_eq!(
        on_final.lambda.len(),
        2,
        "postsolve restores the dropped dual"
    );
    assert!((on_final.g[0] - 2.0).abs() < 1e-5);
    assert!(
        on_final.lambda[0].abs() < 1e-10,
        "dropped row dual = {}",
        on_final.lambda[0]
    );
    assert!(
        on_final.z_l[0] > 1.0,
        "tightened lower-bound dual = {}",
        on_final.z_l[0]
    );
}

#[test]
fn library_presolve_projects_warm_start_and_restores_original_payload() {
    let (status, problem) = solve(true, true, None);
    assert!(matches!(
        status,
        ApplicationReturnStatus::SolveSucceeded | ApplicationReturnStatus::SolvedToAcceptableLevel
    ));
    assert_eq!(problem.warm_z_seen, Some((vec![3.0], vec![0.0])));
    assert_eq!(problem.warm_lambda_seen, Some(vec![7.0, 11.0]));
    let final_payload = problem.final_payload.expect("finalize_solution");
    assert_eq!(final_payload.g.len(), 2);
    assert_eq!(final_payload.lambda.len(), 2);
    assert!(final_payload.lambda[0].abs() < 1e-10);
}

#[test]
fn library_presolve_preserves_max_iter_failure_and_callback_shape() {
    let (status, problem) = solve(true, false, Some(0));
    assert_eq!(status, ApplicationReturnStatus::MaximumIterationsExceeded);
    let final_payload = problem
        .final_payload
        .expect("finalize_solution on early termination");
    assert_eq!(final_payload.status, SolverReturn::MaxiterExceeded);
    assert_eq!(final_payload.g.len(), 2);
    assert_eq!(final_payload.lambda.len(), 2);
}
