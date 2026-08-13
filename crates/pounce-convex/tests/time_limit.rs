use std::time::Duration;

use pounce_convex::{
    ActiveSetOverrides, ConeSpec, QpFactorization, QpOptions, QpProblem, QpStatus, Triplet,
    solve_qp_active_set, solve_qp_batch, solve_qp_ipm, solve_socp_ipm,
};
use pounce_linsol::SparseSymLinearSolverInterface;

fn backend() -> Box<dyn SparseSymLinearSolverInterface> {
    Box::new(pounce_feral::FeralSolverInterface::new())
}

fn qp() -> QpProblem {
    QpProblem {
        n: 1,
        p_lower: vec![Triplet::new(0, 0, 2.0)],
        c: vec![-2.0],
        a: vec![],
        b: vec![],
        g: vec![],
        h: vec![],
        lb: vec![0.0],
        ub: vec![2.0],
    }
}

fn zero(use_hsde: bool) -> QpOptions {
    QpOptions {
        time_limit: Some(Duration::ZERO),
        use_hsde,
        ..QpOptions::default()
    }
}

#[test]
fn zero_duration_stops_direct_and_symmetric_hsde() {
    assert_eq!(
        solve_qp_ipm(&qp(), &zero(false), backend).status,
        QpStatus::TimeLimit
    );
    assert_eq!(
        solve_qp_ipm(&qp(), &zero(true), backend).status,
        QpStatus::TimeLimit
    );
}

#[test]
fn zero_duration_stops_symmetric_and_nonsymmetric_conic_routes() {
    let symmetric = QpProblem {
        n: 1,
        p_lower: vec![],
        c: vec![0.0],
        a: vec![],
        b: vec![],
        g: vec![Triplet::new(0, 0, -1.0)],
        h: vec![1.0, 0.0],
        lb: vec![],
        ub: vec![],
    };
    assert_eq!(
        solve_socp_ipm(
            &symmetric,
            &[ConeSpec::SecondOrder(2)],
            &zero(true),
            backend
        )
        .status,
        QpStatus::TimeLimit
    );

    let nonsymmetric = QpProblem {
        n: 1,
        p_lower: vec![],
        c: vec![0.0],
        a: vec![],
        b: vec![],
        g: vec![],
        h: vec![0.0; 3],
        lb: vec![],
        ub: vec![],
    };
    assert_eq!(
        solve_socp_ipm(
            &nonsymmetric,
            &[ConeSpec::Exponential],
            &zero(true),
            backend
        )
        .status,
        QpStatus::TimeLimit
    );
}

#[test]
fn zero_duration_stops_active_set_and_each_batch_item_independently() {
    let mut mk = backend;
    let active = solve_qp_active_set(&qp(), &zero(true), &ActiveSetOverrides::default(), &mut mk);
    assert_eq!(active.status, QpStatus::TimeLimit);

    let problems = vec![qp(), qp(), qp()];
    let batch = solve_qp_batch(&problems, &zero(true), backend);
    assert_eq!(batch.len(), problems.len());
    assert!(batch.iter().all(|sol| sol.status == QpStatus::TimeLimit));
}

#[test]
fn no_limit_preserves_normal_convergence() {
    let sol = solve_qp_ipm(&qp(), &QpOptions::default(), backend);
    assert_eq!(sol.status, QpStatus::Optimal);
    assert!((sol.x[0] - 1.0).abs() < 1e-7);
}

#[test]
fn no_limit_preserves_iteration_limit_status() {
    let opts = QpOptions {
        max_iter: 0,
        time_limit: None,
        ..QpOptions::default()
    };
    assert_eq!(
        solve_qp_ipm(&qp(), &opts, backend).status,
        QpStatus::IterationLimit
    );
}

#[test]
fn reusable_factorization_gives_each_solve_its_own_deadline() {
    let opts = zero(false);
    let mut factor = QpFactorization::build(&qp(), &opts, backend).expect("build factorization");
    assert_eq!(factor.solve(&qp()).status, QpStatus::TimeLimit);
    assert_eq!(factor.solve(&qp()).status, QpStatus::TimeLimit);
}
