//! The `qp` feature's surface — the sparse parametric active-set engine —
//! reached through the facade only (gh #561). Including the triplet storage
//! `QpProblem` borrows: without it the re-exported solver would still not be
//! usable from a crate that depends on `pounce-rs` alone.
#![cfg(feature = "qp")]

use pounce_rs::linsol::backend;
use pounce_rs::qp::{
    GenTMatrix, GenTMatrixSpace, HessianInertia, ParametricActiveSetSolver, QpOptions, QpProblem,
    QpSolver, QpWarmStart, SymTMatrix, SymTMatrixSpace,
};

/// `min ½(4x₀² + 4x₁²) − 2x₀ − 2x₁  s.t.  x₀ + x₁ == rhs`.
/// KKT: `4xᵢ − 2 + λ = 0` with `x₀ = x₁ = rhs/2`.
fn equality_qp() -> (SymTMatrix, GenTMatrix) {
    let mut h = SymTMatrix::new(SymTMatrixSpace::new(2, vec![1, 2], vec![1, 2]));
    h.set_values(&[4.0, 4.0]);
    let mut a = GenTMatrix::new(GenTMatrixSpace::new(1, 2, vec![1, 1], vec![1, 2]));
    a.set_values(&[1.0, 1.0]);
    (h, a)
}

fn problem<'a>(
    h: &'a SymTMatrix,
    a: &'a GenTMatrix,
    g: &'a [f64],
    bl: &'a [f64],
    bu: &'a [f64],
    xl: &'a [f64],
    xu: &'a [f64],
) -> QpProblem<'a> {
    QpProblem {
        n: 2,
        m: 1,
        h,
        g,
        a,
        bl,
        bu,
        xl,
        xu,
        hessian_inertia: HessianInertia::Psd,
    }
}

#[test]
fn cold_active_set_solve_hits_the_closed_form() {
    let (h, a) = equality_qp();
    let (g, bl, bu) = ([-2.0, -2.0], [1.0], [1.0]);
    let (xl, xu) = ([-1e20, -1e20], [1e20, 1e20]);
    let qp = problem(&h, &a, &g, &bl, &bu, &xl, &xu);

    let mut solver = ParametricActiveSetSolver::new(backend());
    let sol = solver
        .solve(&qp, None, &QpOptions::default())
        .expect("well-posed strictly convex QP");

    assert!((sol.x[0] - 0.5).abs() < 1e-10, "x = {:?}", sol.x);
    assert!((sol.x[1] - 0.5).abs() < 1e-10, "x = {:?}", sol.x);
    assert!((sol.obj + 1.0).abs() < 1e-10, "obj = {}", sol.obj);
}

#[test]
fn parametric_solve_tracks_a_shifted_rhs() {
    let (h, a) = equality_qp();
    let (g, xl, xu) = ([-2.0, -2.0], [-1e20, -1e20], [1e20, 1e20]);
    let opts = QpOptions::default();
    let mut solver = ParametricActiveSetSolver::new(backend());

    let (bl0, bu0) = ([1.0], [1.0]);
    let qp0 = problem(&h, &a, &g, &bl0, &bu0, &xl, &xu);
    let sol0 = solver.solve(&qp0, None, &opts).expect("base solve");

    // Perturb the equality right-hand side; the optimum moves to (rhs/2, rhs/2).
    let (bl1, bu1) = ([1.4], [1.4]);
    let qp1 = problem(&h, &a, &g, &bl1, &bu1, &xl, &xu);
    let tracked = solver
        .solve_parametric(&qp0, &sol0, &qp1, &opts)
        .expect("homotopy from the base solution");

    assert!((tracked.x[0] - 0.7).abs() < 1e-8, "x = {:?}", tracked.x);
    assert!((tracked.x[1] - 0.7).abs() < 1e-8, "x = {:?}", tracked.x);
}

#[test]
fn working_set_from_a_solve_warm_starts_the_next() {
    let (h, a) = equality_qp();
    let (g, bl, bu) = ([-2.0, -2.0], [1.0], [1.0]);
    let (xl, xu) = ([-1e20, -1e20], [1e20, 1e20]);
    let qp = problem(&h, &a, &g, &bl, &bu, &xl, &xu);
    let opts = QpOptions::default();

    let mut solver = ParametricActiveSetSolver::new(backend());
    let cold = solver.solve(&qp, None, &opts).expect("cold solve");

    let warm_seed = QpWarmStart {
        x: cold.x.clone(),
        lambda_g: cold.lambda_g.clone(),
        lambda_x: cold.lambda_x.clone(),
        working: cold.working.clone(),
    };
    let warm = solver
        .solve(&qp, Some(&warm_seed), &opts)
        .expect("warm solve from the previous working set");

    // A warm start changes the iteration count, never the answer.
    for j in 0..2 {
        assert!(
            (warm.x[j] - cold.x[j]).abs() < 1e-10,
            "warm x[{j}] = {}, cold = {}",
            warm.x[j],
            cold.x[j]
        );
    }
}
