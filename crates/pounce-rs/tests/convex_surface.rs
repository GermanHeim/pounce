//! The `convex` feature's surface, exercised end to end through the facade
//! only — no `pounce-convex`, `pounce-linsol`, or `pounce-feral` in this
//! file's imports. That is the point of gh #561: a downstream crate doing
//! batched or parametric convex solves should not have to name the internal
//! crates.
#![cfg(feature = "convex")]

use pounce_rs::convex::{
    QpFactorization, QpOptions, QpProblem, QpStatus, Triplet, solve_qp_batch,
    solve_qp_batch_parallel, solve_qp_ipm,
};
use pounce_rs::linsol::{backend, serial_backend};

/// `min ‖x − t‖²` over the box `[0, 1]ⁿ`, written as `½ xᵀ(2I)x − 2tᵀx`.
/// The unconstrained optimum is `t`, clamped componentwise to `[0, 1]`.
fn boxed_qp(t: &[f64]) -> QpProblem {
    let n = t.len();
    QpProblem {
        n,
        p_lower: (0..n).map(|i| Triplet::new(i, i, 2.0)).collect(),
        c: t.iter().map(|v| -2.0 * v).collect(),
        a: vec![],
        b: vec![],
        g: vec![],
        h: vec![],
        lb: vec![0.0; n],
        ub: vec![1.0; n],
    }
}

#[test]
fn single_solve_hits_the_clamped_optimum() {
    let sol = solve_qp_ipm(
        &boxed_qp(&[0.25, 2.0, -1.0]),
        &QpOptions::default(),
        backend,
    );

    assert_eq!(sol.status, QpStatus::Optimal);
    for (got, want) in sol.x.iter().zip([0.25, 1.0, 0.0]) {
        assert!((got - want).abs() < 1e-6, "x = {:?}", sol.x);
    }
}

#[test]
fn equality_constrained_solve_respects_the_constraint() {
    // min ‖x‖² − 0.5·x0 − 1.5·x1  s.t.  x0 + x1 == 1, 0 ≤ x ≤ 5.
    // KKT: 2xᵢ = cᵢ − λ with λ = 0 ⇒ x = (0.25, 0.75), strictly inside the box.
    let prob = QpProblem {
        n: 2,
        p_lower: vec![Triplet::new(0, 0, 2.0), Triplet::new(1, 1, 2.0)],
        c: vec![-0.5, -1.5],
        a: vec![Triplet::new(0, 0, 1.0), Triplet::new(0, 1, 1.0)],
        b: vec![1.0],
        g: vec![],
        h: vec![],
        lb: vec![0.0, 0.0],
        ub: vec![5.0, 5.0],
    };

    let sol = solve_qp_ipm(&prob, &QpOptions::default(), backend);

    assert_eq!(sol.status, QpStatus::Optimal);
    assert!((sol.x[0] + sol.x[1] - 1.0).abs() < 1e-6, "x = {:?}", sol.x);
    assert!((sol.x[0] - 0.25).abs() < 1e-5, "x = {:?}", sol.x);
    assert!((sol.x[1] - 0.75).abs() < 1e-5, "x = {:?}", sol.x);
    assert_eq!(sol.y.len(), 1, "one equality multiplier expected");
}

#[test]
fn batched_solves_match_the_single_solves() {
    let targets = [[0.25, 0.75], [2.0, -1.0], [0.5, 0.5]];
    let probs: Vec<QpProblem> = targets.iter().map(|t| boxed_qp(t)).collect();
    let opts = QpOptions::default();

    let serial = solve_qp_batch(&probs, &opts, backend);
    let parallel = solve_qp_batch_parallel(&probs, &opts, serial_backend);

    assert_eq!(serial.len(), probs.len());
    assert_eq!(parallel.len(), probs.len());
    for (k, prob) in probs.iter().enumerate() {
        let single = solve_qp_ipm(prob, &opts, backend);
        assert_eq!(single.status, QpStatus::Optimal);
        for j in 0..prob.n {
            assert!(
                (serial[k].x[j] - single.x[j]).abs() < 1e-8,
                "serial batch differs at ({k}, {j})"
            );
            assert!(
                (parallel[k].x[j] - single.x[j]).abs() < 1e-8,
                "parallel batch differs at ({k}, {j})"
            );
        }
    }
}

#[test]
fn symbolic_factorization_is_reusable_across_instances() {
    // Fixed sparsity, varying data — the training-data / MPC shape #561 was
    // filed for. The reused symbolic factor must not change the answers.
    let opts = QpOptions::default();
    let targets = [[0.25, 0.75], [2.0, -1.0]];
    let probs: Vec<QpProblem> = targets.iter().map(|t| boxed_qp(t)).collect();

    let mut fact = QpFactorization::build(&probs[0], &opts, backend)
        .expect("symbolic analysis of a well-posed box QP");
    for (k, prob) in probs.iter().enumerate() {
        let reused = fact.solve(prob);
        let cold = solve_qp_ipm(prob, &opts, backend);
        assert_eq!(reused.status, QpStatus::Optimal, "instance {k}");
        for j in 0..prob.n {
            assert!(
                (reused.x[j] - cold.x[j]).abs() < 1e-6,
                "reused factor differs from a cold solve at ({k}, {j})"
            );
        }
    }
}
