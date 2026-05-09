//! End-to-end: drive [`TSymLinearSolver`] over the real MA57 backend.
//!
//! Validates the wiring between the Phase-4 wrapper, the triplet
//! converter, and the FFI-driven Fortran factorization. Uses a 4×4
//! KKT-shaped indefinite matrix (2×2 SPD H block + 2×2 zero block, with
//! Jacobian rows tying them together) so the inertia path matters.

use pounce_common::types::{Index, Number};
use pounce_hsl::Ma57SolverInterface;
use pounce_linsol::{ESymSolverStatus, IdentityScalingMethod, SymLinearSolver, TSymLinearSolver};

/// 4×4 saddle-point matrix, lower-triangle triplet (1-based):
/// ```text
///   [  4   1   1   0 ]
///   [  1   3   1   1 ]
///   [  1   1   0   0 ]   ← these "0"s are the (2,2) block of the saddle point
///   [  0   1   0   0 ]
/// ```
/// Inertia: the H block (top-left 2×2) is SPD; the J^T J Schur
/// complement gives a saddle-point matrix with 2 positive and 2 negative
/// eigenvalues. We test that MA57 reports `negevals=2`.
#[test]
fn solve_4x4_saddle_point_via_wrapper() {
    let backend = Ma57SolverInterface::new();
    let mut solver = TSymLinearSolver::new(Box::new(backend), None, false);

    // Lower-triangle triplet, 1-based (MA57 expects this).
    let irn: [Index; 8] = [1, 2, 2, 3, 3, 3, 4, 4];
    let jcn: [Index; 8] = [1, 1, 2, 1, 2, 3, 2, 4];
    //                    a11 a21 a22 a31 a32 a33 a42 a44
    let vals: [Number; 8] = [4.0, 1.0, 3.0, 1.0, 1.0, 0.0, 1.0, 0.0];

    assert_eq!(
        solver.initialize_structure(4, &irn, &jcn),
        ESymSolverStatus::Success
    );

    // Pick x = (1, 1, 1, 1) → b = A x.
    // Row sums:
    // r1: 4*1 + 1*1 + 1*1 + 0*1 = 6
    // r2: 1*1 + 3*1 + 1*1 + 1*1 = 6
    // r3: 1*1 + 1*1 + 0*1 + 0*1 = 2
    // r4: 0*1 + 1*1 + 0*1 + 0*1 = 1
    let mut rhs: [Number; 4] = [6.0, 6.0, 2.0, 1.0];

    assert_eq!(
        solver.multi_solve(&vals, true, 1, &mut rhs, true, 2),
        ESymSolverStatus::Success
    );

    for (i, &x) in rhs.iter().enumerate() {
        assert!(
            (x - 1.0).abs() < 1e-10,
            "x[{i}] = {x}, expected 1.0 (residual {})",
            (x - 1.0).abs()
        );
    }
    assert_eq!(solver.number_of_neg_evals(), 2);
}

/// Same solve with identity scaling enabled — values should be
/// unchanged at the bit level (identity scaling × identity scaling).
#[test]
fn solve_with_identity_scaling_matches_unscaled() {
    let backend = Ma57SolverInterface::new();
    let mut solver = TSymLinearSolver::new(
        Box::new(backend),
        Some(Box::new(IdentityScalingMethod)),
        false, // scale every refactor
    );
    let irn: [Index; 8] = [1, 2, 2, 3, 3, 3, 4, 4];
    let jcn: [Index; 8] = [1, 1, 2, 1, 2, 3, 2, 4];
    let vals: [Number; 8] = [4.0, 1.0, 3.0, 1.0, 1.0, 0.0, 1.0, 0.0];

    assert_eq!(
        solver.initialize_structure(4, &irn, &jcn),
        ESymSolverStatus::Success
    );
    let mut rhs: [Number; 4] = [6.0, 6.0, 2.0, 1.0];
    assert_eq!(
        solver.multi_solve(&vals, true, 1, &mut rhs, true, 2),
        ESymSolverStatus::Success
    );
    for &x in rhs.iter() {
        assert!((x - 1.0).abs() < 1e-10);
    }
}

/// Multiple right-hand sides packed column-major.
#[test]
fn multi_rhs_solve() {
    let backend = Ma57SolverInterface::new();
    let mut solver = TSymLinearSolver::new(Box::new(backend), None, false);
    // Trivial 2×2 SPD: A = [[2,1],[1,3]], lower-triangle triplet.
    let irn = [1, 2, 2];
    let jcn = [1, 1, 2];
    let vals = [2.0, 1.0, 3.0];
    solver.initialize_structure(2, &irn, &jcn);

    // Two RHS columns, packed (b1; b2) = ((3,4); (5,8)).
    // x1 = (1, 1) since 2+1=3, 1+3=4.
    // x2 = (?, ?): solve A x = (5, 8). Determinant=5; x=(7/5, 11/5).
    let mut rhs: [Number; 4] = [3.0, 4.0, 5.0, 8.0];
    assert_eq!(
        solver.multi_solve(&vals, true, 2, &mut rhs, false, 0),
        ESymSolverStatus::Success
    );
    assert!((rhs[0] - 1.0).abs() < 1e-12);
    assert!((rhs[1] - 1.0).abs() < 1e-12);
    assert!((rhs[2] - 7.0 / 5.0).abs() < 1e-12);
    assert!((rhs[3] - 11.0 / 5.0).abs() < 1e-12);
}
