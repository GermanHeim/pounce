//! A cancelled factorization must never be mistaken for a solved one.
//!
//! `factorize_with_inertia_control` signals success by writing the *solved*
//! right-hand side back in place, so an early `Ok` return on a deadline hands
//! the caller an `rhs` that still holds `[-g; targets]` while claiming it is
//! `[x*; λ*]`. The active-set loop then reads `delta == 0`, skips its
//! rank-deficiency and recession guards, and reports `x = -g` as `Optimal` —
//! a fabricated answer, not a timeout. Cancellation is therefore an *error*
//! (`QpError::DeadlineExpired`) that `?` forces every caller to handle, and
//! only the public entry points turn it into the soft `QpStatus::TimeLimit`.
//!
//! Reproducing that window needs the deadline to cross *inside* a
//! factorization — after the entry-point check has already passed — which no
//! choice of `time_limit` alone can arrange. The backend below creates it on
//! demand: its first solve stalls past the budget and reports the singular
//! factor that sends the solver into the inertia-shift loop, which is exactly
//! where the check lives.

use std::rc::Rc;
use std::time::Duration;

use pounce_common::types::{Index, NLP_LOWER_BOUND_INF, NLP_UPPER_BOUND_INF, Number};
use pounce_linalg::triplet::{GenTMatrix, GenTMatrixSpace, SymTMatrix, SymTMatrixSpace};
use pounce_linsol::{EMatrixFormat, ESymSolverStatus, SparseSymLinearSolverInterface};
use pounce_qp::{
    HessianInertia, ParametricActiveSetSolver, QpOptions, QpProblem, QpSolver, QpStatus,
};

/// Budget the solve is given. Anything the first factorization overruns works;
/// small enough that the test costs one stall.
const BUDGET: Duration = Duration::from_millis(5);
/// How long the first factorization takes. An order of magnitude over `BUDGET`
/// so the crossing does not depend on scheduling noise.
const STALL: Duration = Duration::from_millis(50);

/// A backend whose first factorization outlives the deadline and then reports
/// a singular factor — the recoverable failure that routes the solver into the
/// inertia-shift loop.
struct StallThenSingular {
    values: Vec<Number>,
    stalled: bool,
}

impl SparseSymLinearSolverInterface for StallThenSingular {
    fn initialize_structure(
        &mut self,
        _dim: Index,
        nonzeros: Index,
        _ia: &[Index],
        _ja: &[Index],
    ) -> ESymSolverStatus {
        self.values = vec![0.0; nonzeros as usize];
        ESymSolverStatus::Success
    }

    fn values_array_mut(&mut self) -> &mut [Number] {
        &mut self.values
    }

    fn multi_solve(
        &mut self,
        _new_matrix: bool,
        _ia: &[Index],
        _ja: &[Index],
        _nrhs: Index,
        _rhs_vals: &mut [Number],
        _check_neg_evals: bool,
        _number_of_neg_evals: Index,
    ) -> ESymSolverStatus {
        // Only the first call stalls: the point is to be *inside* a
        // factorization when the budget runs out, not to make the test slow.
        // `rhs_vals` is deliberately left untouched — a backend that fails
        // wrote no solution, which is the whole hazard being tested.
        if !self.stalled {
            self.stalled = true;
            std::thread::sleep(STALL);
        }
        ESymSolverStatus::Singular
    }

    fn number_of_neg_evals(&self) -> Index {
        0
    }

    fn increase_quality(&mut self) -> bool {
        false
    }

    fn provides_inertia(&self) -> bool {
        true
    }

    fn matrix_format(&self) -> EMatrixFormat {
        EMatrixFormat::TripletFormat
    }
}

#[test]
fn a_deadline_crossed_inside_a_factorization_is_not_an_optimum() {
    // `min x₁² + 4x₂² − 2x₁ − 2x₂  s.t.  x₁ + x₂ = 4`, free variables — a pure
    // equality QP with no bounds, which takes the `solve_equality_only` fast
    // path: the one that reads the factorization's right-hand side back as
    // `[x*; λ*]` with no further check of its own.
    //
    // The data is chosen so the fabricated point is *feasible but wrong*. The
    // un-solved RHS is `[-g; b] = [2, 2; 4]`, so `x = (2, 2)` satisfies
    // `x₁ + x₂ = 4` — the primal-feasibility audit, the only downstream guard,
    // waves it through. The true optimum is `(3.2, 0.8)`.
    let h_space = SymTMatrixSpace::new(2, vec![1, 2], vec![1, 2]);
    let mut h = SymTMatrix::new(Rc::clone(&h_space));
    h.set_values(&[2.0, 8.0]);
    let a_space = GenTMatrixSpace::new(1, 2, vec![1, 1], vec![1, 2]);
    let mut a = GenTMatrix::new(a_space);
    a.set_values(&[1.0, 1.0]);
    let g = [-2.0, -2.0];
    let bl = [4.0];
    let bu = [4.0];
    let xl = [NLP_LOWER_BOUND_INF; 2];
    let xu = [NLP_UPPER_BOUND_INF; 2];
    let qp = QpProblem {
        n: 2,
        m: 1,
        h: &h,
        g: &g,
        a: &a,
        bl: &bl,
        bu: &bu,
        xl: &xl,
        xu: &xu,
        hessian_inertia: HessianInertia::Psd,
    };
    let mut solver = ParametricActiveSetSolver::new(Box::new(StallThenSingular {
        values: Vec::new(),
        stalled: false,
    }));
    let opts = QpOptions {
        time_limit: Some(BUDGET),
        ..QpOptions::default()
    };

    let sol = solver
        .solve(&qp, None, &opts)
        .expect("a timeout is a soft status, never an error out of the crate");

    // The load-bearing assertion. Before the typed cancellation this returned
    // `Optimal` at `x = -g = (2, 2)` — the un-solved right-hand side read back
    // as a solution. It is feasible, so nothing downstream objected, and it is
    // not the optimum.
    assert_ne!(
        sol.status,
        QpStatus::Optimal,
        "a cancelled factorization was reported as a solved QP (x = {:?})",
        sol.x
    );
    assert_eq!(sol.status, QpStatus::TimeLimit);
    assert!(sol.x.iter().all(|v| v.is_finite()));
    // The timeout is not an instantaneous solve: the stall is on the clock.
    assert!(
        sol.stats.time >= BUDGET,
        "stats.time = {:?} does not account for the budget actually spent",
        sol.stats.time
    );
}
