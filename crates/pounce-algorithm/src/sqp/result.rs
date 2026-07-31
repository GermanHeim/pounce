//! `SqpResult` / `SqpStatus` / `SqpError` — return types for
//! `SqpAlgorithm::optimize`.

use pounce_common::Number;
use pounce_qp::{QpError, WorkingSet};
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SqpStatus {
    /// KKT residuals all below their tolerances.
    Optimal,
    /// `max_iter` reached without convergence.
    MaxIter,
    /// QP subproblem returned an `Infeasible` status (elastic
    /// mode certified the QP infeasible).
    InfeasibleSubproblem,
    /// Line search failed to find an acceptable step (Phase 5b
    /// commit 5+; not produced by the c3 always-full-step loop).
    LineSearchFailed,
    /// The QP subproblem solver neither produced a usable step nor
    /// certified infeasibility — it hit its own iteration limit or a
    /// numerical breakdown (e.g. the extreme m/n ≫ 1 degenerate phase-1
    /// of #282). This is an HONEST non-committal failure: the SQP could
    /// not compute a search direction, but — unlike `InfeasibleSubproblem`
    /// — it makes no infeasibility claim it cannot back with a
    /// certificate. Maps to `Search_Direction_Becomes_Too_Small`.
    QpStepFailed,
    /// The QP subproblem exhausted its own iteration budget
    /// ([`QpOptions::max_iter`], the `sqp_qp_max_iter` option) without
    /// converging or certifying anything.
    ///
    /// Split out from [`QpStepFailed`](Self::QpStepFailed) because the two
    /// call for opposite remedies and the merged status actively misled.
    /// A budget exhaustion is *actionable* — raise the limit — whereas
    /// `Search_Direction_Becomes_Too_Small` reads as a numerical stall with
    /// nothing to turn. On the Maros-Mészáros set the merged mapping hid the
    /// single largest failure class: the cold-start active-set method needs
    /// roughly one iteration per active-set change, so the flat default of
    /// 200 is below what a few hundred constraints require, and dozens of
    /// problems reported a step-size failure when they had simply run out of
    /// budget. `DUALC1` (n=9, m=215) is the type case — it exits here at the
    /// default and solves exactly, in one outer iteration, at a larger limit.
    ///
    /// Maps to `Maximum_Iterations_Exceeded`.
    QpIterationLimit,
    /// The problem is unbounded below: the step QP returned a certified
    /// recession ray (zero curvature, feasible for every step length,
    /// strict descent) *and* that ray was re-verified against the true
    /// NLP — feasible points along it drive `f` toward `−∞` at (at
    /// least) half the linear rate out to `1e12·‖d‖`. Maps to
    /// `Diverging_Iterates`, POUNCE's (Ipopt's) unboundedness verdict,
    /// the same status the IPM paths report on an unbounded model
    /// (gh #388). An *unverified* unbounded step QP is a statement about
    /// the local model only and falls back to
    /// [`QpStepFailed`](Self::QpStepFailed).
    Unbounded,
}

impl fmt::Display for SqpStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SqpStatus::Optimal => write!(f, "optimal"),
            SqpStatus::MaxIter => write!(f, "max-iter"),
            SqpStatus::InfeasibleSubproblem => write!(f, "infeasible-subproblem"),
            SqpStatus::LineSearchFailed => write!(f, "line-search-failed"),
            SqpStatus::QpStepFailed => write!(f, "qp-step-failed"),
            SqpStatus::QpIterationLimit => write!(f, "qp-iteration-limit"),
            SqpStatus::Unbounded => write!(f, "unbounded"),
        }
    }
}

#[derive(Debug)]
pub enum SqpError {
    /// Hard QP-solver failure (singular, dimension mismatch, etc.).
    QpFailure(QpError),
    /// Caller-supplied dimensions disagree.
    DimensionMismatch(String),
}

impl fmt::Display for SqpError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SqpError::QpFailure(e) => write!(f, "QP subproblem failure: {e}"),
            SqpError::DimensionMismatch(s) => write!(f, "dimension mismatch: {s}"),
        }
    }
}

impl From<QpError> for SqpError {
    fn from(e: QpError) -> Self {
        SqpError::QpFailure(e)
    }
}

#[derive(Debug, Clone)]
pub struct SqpResult {
    pub x: Vec<Number>,
    pub lambda_g: Vec<Number>,
    pub lambda_x: Vec<Number>,
    pub obj: Number,
    pub status: SqpStatus,
    pub n_iter: u32,
    pub n_qp_solves: u32,
    /// Active-set changes (adds + drops) summed over every step QP
    /// solved during this call — the inner work a working-set warm
    /// start exists to avoid. Reported separately from `n_iter`
    /// because the two move independently: on a QP-shaped NLP the
    /// outer loop always terminates in one iteration, so the entire
    /// warm-start effect shows up here and nowhere else.
    ///
    /// Excludes second-order-correction QPs, whose stats the line
    /// search does not surface.
    pub n_qp_working_set_changes: u32,
    /// Final stationarity residual (max-norm of `∇f + Jᵀ λ_g + λ_x`).
    pub final_stationarity: Number,
    /// Final constraint violation (max-norm of `c(x*)` for
    /// equalities plus bound-violation slack).
    pub final_constr_viol: Number,
    /// Final QP working set, suitable for warm-starting the next
    /// `optimize_with_warm_start` call (§6 design-note contract).
    /// `None` only when no QP was solved (e.g. cold-start declared
    /// the iterate optimal at the very first KKT check).
    pub working_set: Option<WorkingSet>,
}
