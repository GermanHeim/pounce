use std::rc::Rc;
use std::time::Duration;

use pounce_linalg::triplet::{GenTMatrix, GenTMatrixSpace, SymTMatrix, SymTMatrixSpace};
use pounce_qp::{
    HessianInertia, ParametricActiveSetSolver, QpOptions, QpProblem, QpSolver, QpStatus,
};

#[test]
fn zero_duration_returns_time_limit_before_factorization() {
    let h_space = SymTMatrixSpace::new(1, vec![1], vec![1]);
    let mut h = SymTMatrix::new(Rc::clone(&h_space));
    h.set_values(&[2.0]);
    let a = GenTMatrix::new(GenTMatrixSpace::new(0, 1, vec![], vec![]));
    let g = [-2.0];
    let xl = [0.0];
    let xu = [2.0];
    let qp = QpProblem {
        n: 1,
        m: 0,
        h: &h,
        g: &g,
        a: &a,
        bl: &[],
        bu: &[],
        xl: &xl,
        xu: &xu,
        hessian_inertia: HessianInertia::Psd,
    };
    let mut solver =
        ParametricActiveSetSolver::new(Box::new(pounce_feral::FeralSolverInterface::new()));
    let opts = QpOptions {
        time_limit: Some(Duration::ZERO),
        ..QpOptions::default()
    };
    let sol = solver
        .solve(&qp, None, &opts)
        .expect("timeout is a soft status");
    assert_eq!(sol.status, QpStatus::TimeLimit);
    assert_eq!(sol.x.len(), 1);
    assert!(sol.x[0].is_finite());
}
