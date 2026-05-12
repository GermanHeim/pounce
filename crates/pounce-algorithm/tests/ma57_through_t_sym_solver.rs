//! Integration test: feed a real triplet matrix through
//! [`pounce_linsol::TSymLinearSolver`] backed by MA57 (via
//! `pounce-hsl`'s `Ma57SolverInterface`), confirm the solve hits
//! machine-precision. Gated on the `ma57` cargo feature; without
//! HSL linked in, MA57 is not available.
//!
//! Lives in `pounce-algorithm` because the algorithm crate is the
//! consumer that wires Phase-4 (linsol + hsl) into Phase-6 (KKT
//! solvers). The wrapper itself is canonical at
//! `pounce_linsol::TSymLinearSolver`; the duplicate local copy under
//! `pounce-algorithm/src/kkt/` was removed when `StdAugSystemSolver`
//! was rewired in Phase 6.

#![cfg(feature = "ma57")]

use pounce_common::types::{Index, Number};
use pounce_hsl::Ma57SolverInterface;
use pounce_linsol::{ESymSolverStatus, TSymLinearSolver};

/// Solve `A x = b` for the symmetric indefinite 3×3:
///
/// ```text
///   A = [ 4  -1   0 ]      b = [ 4 ]      x = ...
///       [-1   4  -1 ]          [ 4 ]
///       [ 0  -1   4 ]          [ 6 ]
/// ```
///
/// Lower-triangular triplet (1-based for MA57):
///   (1,1)=4, (2,1)=-1, (2,2)=4, (3,2)=-1, (3,3)=4
#[test]
fn ma57_solves_3x3_spd_via_t_sym_wrapper() {
    let irn: [Index; 5] = [1, 2, 2, 3, 3];
    let jcn: [Index; 5] = [1, 1, 2, 2, 3];
    let vals: [Number; 5] = [4.0, -1.0, 4.0, -1.0, 4.0];

    let mut tsym = TSymLinearSolver::new(Box::new(Ma57SolverInterface::new()), None, false);
    assert_eq!(
        tsym.initialize_structure(3, &irn, &jcn),
        ESymSolverStatus::Success
    );

    // Hand-solved x:
    //   x3 = 110/56 = 55/28
    //   x2 = 4 x3 - 6 = 13/7
    //   x1 = (4 + x2)/4 = 41/28
    let x1: Number = 41.0 / 28.0;
    let x2: Number = 13.0 / 7.0;
    let x3: Number = 55.0 / 28.0;

    let mut rhs: [Number; 3] = [4.0, 4.0, 6.0];
    let status = tsym.multi_solve(&vals, true, 1, &mut rhs, true, 0);
    assert_eq!(status, ESymSolverStatus::Success);

    assert!((rhs[0] - x1).abs() < 1e-12, "x1: got {} want {}", rhs[0], x1);
    assert!((rhs[1] - x2).abs() < 1e-12, "x2: got {} want {}", rhs[1], x2);
    assert!((rhs[2] - x3).abs() < 1e-12, "x3: got {} want {}", rhs[2], x3);

    use pounce_linsol::SymLinearSolver;
    assert_eq!(
        <TSymLinearSolver as SymLinearSolver>::number_of_neg_evals(&tsym),
        0
    );
}

/// Indefinite 2×2: `[[1, 2], [2, 1]]` has eigenvalues `3, -1`, i.e. one
/// negative.
#[test]
fn ma57_reports_correct_inertia_on_indefinite_matrix() {
    let irn: [Index; 3] = [1, 2, 2];
    let jcn: [Index; 3] = [1, 1, 2];
    let vals: [Number; 3] = [1.0, 2.0, 1.0];

    let mut tsym = TSymLinearSolver::new(Box::new(Ma57SolverInterface::new()), None, false);
    assert_eq!(
        tsym.initialize_structure(2, &irn, &jcn),
        ESymSolverStatus::Success
    );

    let mut rhs: [Number; 2] = [3.0, 3.0];
    let status = tsym.multi_solve(&vals, true, 1, &mut rhs, false, 0);
    assert_eq!(status, ESymSolverStatus::Success);

    use pounce_linsol::SymLinearSolver;
    assert_eq!(
        <TSymLinearSolver as SymLinearSolver>::number_of_neg_evals(&tsym),
        1
    );

    // [[1,2],[2,1]] · [1,1]^T = [3,3] → x = [1,1].
    assert!((rhs[0] - 1.0_f64).abs() < 1e-12);
    assert!((rhs[1] - 1.0_f64).abs() < 1e-12);
}
