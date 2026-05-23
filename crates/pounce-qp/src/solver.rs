//! The [`QpSolver`] trait and its concrete implementation
//! [`ParametricActiveSetSolver`].
//!
//! Phase 5a commit 2 ships the cold-start equality-only path: KKT
//! assembly via [`crate::kkt`] + one factor-and-solve through a
//! caller-provided linear-solver backend. Working-set machinery,
//! Schur-complement updates, EXPAND anti-cycling, l1-elastic
//! phase-1, and the parametric homotopy land in subsequent commits.

use std::time::Instant;

use crate::error::{QpError, QpStatus};
use crate::factor::LinearSolver;
use crate::kkt::{is_pure_equality_no_bounds, rhs_equality_only, KktTriplet};
use crate::options::QpOptions;
use crate::problem::{HessianInertia, QpProblem, QpSolution, QpStats, QpWarmStart};
use crate::working_set::{ConsStatus, WorkingSet};
use pounce_common::Number;
use pounce_linsol::SparseSymLinearSolverInterface;

/// QP subproblem solver.
///
/// Two entry points: [`solve`](Self::solve) for a single QP with an
/// optional warm-start seed, and [`solve_parametric`](Self::solve_parametric)
/// for the SQP outer-loop case where the new QP is a perturbation of
/// the previous one and the parametric homotopy of §4.2 can reuse
/// the cached factorization across consecutive QPs without
/// rebuilding it.
pub trait QpSolver {
    /// Solve a single QP. `ws == None` ⇒ cold start (phase-1
    /// elastic mode infers the initial working set when the
    /// machinery lands).
    fn solve(
        &mut self,
        qp: &QpProblem,
        ws: Option<&QpWarmStart>,
        opts: &QpOptions,
    ) -> Result<QpSolution, QpError>;

    /// Parametric solve: trace the homotopy from `(qp_prev,
    /// sol_prev)` to `qp_new`. Falls back to
    /// [`solve`](Self::solve) when the parametric path detects a
    /// structural change that requires a fresh refactor.
    fn solve_parametric(
        &mut self,
        qp_prev: &QpProblem,
        sol_prev: &QpSolution,
        qp_new: &QpProblem,
        opts: &QpOptions,
    ) -> Result<QpSolution, QpError>;
}

/// The sparse parametric active-set QP solver (§4.2 of the design
/// note). Owns a single linear-solver backend; future Schur-
/// complement state lives here too.
pub struct ParametricActiveSetSolver {
    linsol: LinearSolver,
}

impl ParametricActiveSetSolver {
    pub fn new(backend: Box<dyn SparseSymLinearSolverInterface>) -> Self {
        Self {
            linsol: LinearSolver::new(backend),
        }
    }

    /// Cold-start path for QPs that have only equality constraints
    /// and no variable bounds. Builds the saddle-point KKT and
    /// hands it to the linear solver in one shot. The remaining
    /// problem classes route through the working-set machinery in
    /// later commits.
    fn solve_equality_only(
        &mut self,
        qp: &QpProblem,
        opts: &QpOptions,
    ) -> Result<QpSolution, QpError> {
        let started = Instant::now();
        let kkt = KktTriplet::assemble_equality_only(qp);
        let mut rhs = rhs_equality_only(qp);

        // Inertia expectation for [H Aᵀ; A 0] with full-rank A and
        // reduced Hessian PD on null(A): exactly m negative
        // eigenvalues (Gould-Hribar-Nocedal 2001 §3.2). Skip the
        // check when the caller declared H indefinite — the
        // §4.5 inertia-control path is required, and Phase 5a
        // commit 2 doesn't ship it yet.
        let expected_neg = match qp.hessian_inertia {
            HessianInertia::Psd | HessianInertia::Unknown => Some(qp.m as i32),
            HessianInertia::Indefinite => None,
        };
        self.linsol
            .factorize_and_solve(&kkt, &mut rhs, expected_neg)?;

        // RHS now holds [x*; λ*].
        let mut x = vec![0.0; qp.n];
        x.copy_from_slice(&rhs[..qp.n]);
        let mut lambda_g = vec![0.0; qp.m];
        lambda_g.copy_from_slice(&rhs[qp.n..]);

        let obj = quad_objective(qp, &x);

        // All general constraints are equalities (precondition of
        // this entry point) — mark them as such in the working set.
        let mut working = WorkingSet::cold(qp.n, qp.m);
        for c in working.constraints.iter_mut() {
            *c = ConsStatus::Equality;
        }

        let _ = opts; // QpOptions reserved for the working-set path.

        Ok(QpSolution {
            x,
            lambda_g,
            lambda_x: vec![0.0; qp.n],
            working,
            obj,
            status: QpStatus::Optimal,
            stats: QpStats {
                n_working_set_changes: 0,
                n_refactor: 1,
                n_schur_updates: 0,
                used_phase1: false,
                time: started.elapsed(),
            },
        })
    }
}

impl QpSolver for ParametricActiveSetSolver {
    fn solve(
        &mut self,
        qp: &QpProblem,
        _ws: Option<&QpWarmStart>,
        opts: &QpOptions,
    ) -> Result<QpSolution, QpError> {
        qp.validate()?;

        if is_pure_equality_no_bounds(qp) {
            return self.solve_equality_only(qp, opts);
        }

        Err(QpError::UnsupportedFeature(
            "QPs with variable bounds or one-sided inequality constraints \
             require the working-set + Schur-update machinery, which lands \
             in subsequent Phase 5a commits"
                .into(),
        ))
    }

    fn solve_parametric(
        &mut self,
        _qp_prev: &QpProblem,
        _sol_prev: &QpSolution,
        qp_new: &QpProblem,
        opts: &QpOptions,
    ) -> Result<QpSolution, QpError> {
        // No parametric path yet — fall back to a fresh cold solve.
        self.solve(qp_new, None, opts)
    }
}

/// Evaluate `½ xᵀ H x + gᵀ x`, walking the symmetric Hessian once
/// and fanning each off-diagonal entry into both halves.
fn quad_objective(qp: &QpProblem, x: &[Number]) -> Number {
    let mut quad = 0.0;
    let irows = qp.h.irows();
    let jcols = qp.h.jcols();
    let vals = qp.h.values();
    for k in 0..irows.len() {
        let i = (irows[k] - 1) as usize;
        let j = (jcols[k] - 1) as usize;
        let v = vals[k];
        if i == j {
            quad += 0.5 * v * x[i] * x[i];
        } else {
            quad += v * x[i] * x[j]; // each off-diag pair contributes once
        }
    }
    let lin: Number = qp.g.iter().zip(x.iter()).map(|(&gi, &xi)| gi * xi).sum();
    quad + lin
}
