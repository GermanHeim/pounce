//! Drives generated instances through `pounce_qp` and checks three
//! invariants that hold regardless of what the solver decides.
//!
//! 1. **Soundness.** `Infeasible` is a Farkas claim. On an instance
//!    carrying a feasibility witness it is simply false, whatever the
//!    numerics did.
//! 2. **Consistency.** `Optimal` asserts the returned point solves the
//!    problem. A point that violates a row or a bound is not a solution,
//!    so `Optimal` at an infeasible point is a wrong answer even when
//!    the *problem* is feasible. This is the invariant that catches
//!    "converged to Optimal while violating the equality by 7.8".
//! 3. **Completeness (quality, not correctness).** On instances that are
//!    infeasible by exact arithmetic, `Optimal` is a wrong answer; a
//!    non-committal status is merely a weaker result. Tracked separately
//!    so that making the certificate harder to issue shows up as a
//!    measured cost rather than a silent one.

use crate::instances::{Instance, Truth};
use pounce_common::types::Index;
use pounce_linalg::triplet::{GenTMatrix, GenTMatrixSpace, SymTMatrix, SymTMatrixSpace};
use pounce_qp::solver::{ParametricActiveSetSolver, QpSolver};
use pounce_qp::{HessianInertia, QpOptions, QpProblem, QpStatus};

pub struct Outcome {
    pub status: String,
    /// Violation at the returned point; `None` when the solve errored.
    pub violation: Option<f64>,
    pub verdict: Verdict,
    pub detail: String,
}

#[derive(PartialEq, Eq, Debug, Clone, Copy)]
pub enum Verdict {
    /// Every invariant held.
    Ok,
    /// `Infeasible` on an instance with a feasibility witness.
    FalseInfeasible,
    /// `Optimal` at a point that violates the problem.
    OptimalButInfeasiblePoint,
    /// `Optimal` on an instance that is infeasible by exact arithmetic.
    OptimalOnInfeasible,
    /// Correct but non-committal where a verdict was available.
    Weak,
    /// The solver returned `Err`.
    Errored,
}

pub fn run(inst: &Instance) -> Outcome {
    run_with(inst, QpOptions::default().feas_tol)
}

pub fn run_with(inst: &Instance, feas_tol: f64) -> Outcome {
    let n = inst.n;
    let m = inst.m;

    let mut h_irows = Vec::new();
    let mut h_jcols = Vec::new();
    let mut h_vals = Vec::new();
    for &(i, j, v) in &inst.h {
        h_irows.push(i as Index + 1);
        h_jcols.push(j as Index + 1);
        h_vals.push(v);
    }
    let h_space = SymTMatrixSpace::new(n as Index, h_irows, h_jcols);
    let mut h = SymTMatrix::new(h_space);
    h.set_values(&h_vals);

    let mut a_irows = Vec::new();
    let mut a_jcols = Vec::new();
    for i in 0..m {
        for j in 0..n {
            a_irows.push(i as Index + 1);
            a_jcols.push(j as Index + 1);
        }
    }
    let a_space = GenTMatrixSpace::new(m as Index, n as Index, a_irows, a_jcols);
    let mut a = GenTMatrix::new(a_space);
    a.set_values(&inst.a);

    let qp = QpProblem {
        n,
        m,
        h: &h,
        g: &inst.g,
        a: &a,
        bl: &inst.bl,
        bu: &inst.bu,
        xl: &inst.xl,
        xu: &inst.xu,
        // The exact-∇²L case: the caller declares nothing about
        // definiteness, which is what the SQP does by default.
        hessian_inertia: HessianInertia::Indefinite,
    };

    let mut opts = QpOptions::default();
    opts.feas_tol = feas_tol;
    opts.opt_tol = feas_tol;
    let mut solver =
        ParametricActiveSetSolver::new(Box::new(pounce_feral::FeralSolverInterface::new()));
    let sol = match solver.solve(&qp, None, &opts) {
        Ok(s) => s,
        Err(e) => {
            return Outcome {
                status: "Err".into(),
                violation: None,
                verdict: Verdict::Errored,
                detail: format!("{e:?}"),
            };
        }
    };

    let viol = inst.violation(&sol.x);
    // The solver's own feasibility tolerance, loosened by the row scale.
    // A row scaled by 1e6 cannot be expected to hold to an absolute
    // 1e-9; judging it so would manufacture failures the solver never
    // claimed. Scale-relative is the honest comparison.
    let row_scale = inst
        .a
        .iter()
        .map(|v| v.abs())
        .fold(1.0_f64, f64::max)
        .max(inst.bl.iter().chain(inst.bu.iter()).filter(|v| v.is_finite() && v.abs() < 1e18).map(|v| v.abs()).fold(1.0, f64::max));
    let tol = opts.feas_tol * row_scale * 1e3;

    let status = format!("{:?}", sol.status);
    let (verdict, detail) = match (inst.truth, sol.status) {
        (Truth::Feasible, QpStatus::Infeasible) => (
            Verdict::FalseInfeasible,
            format!(
                "certified Infeasible; supplied witness violates by {:.3e}",
                inst.violation(inst.witness.as_ref().unwrap())
            ),
        ),
        (Truth::Feasible, QpStatus::Optimal) if viol > tol => (
            Verdict::OptimalButInfeasiblePoint,
            format!("Optimal at a point violating by {viol:.3e} (tol {tol:.3e})"),
        ),
        (Truth::Feasible, QpStatus::Optimal) => (Verdict::Ok, String::new()),
        (Truth::Feasible, _) => (
            Verdict::Weak,
            format!("no verdict on a feasible instance: {status}"),
        ),
        (Truth::Infeasible, QpStatus::Optimal) => (
            Verdict::OptimalOnInfeasible,
            format!(
                "claimed Optimal on a provably infeasible instance ({}); returned point violates by {viol:.3e}",
                inst.proof
            ),
        ),
        (Truth::Infeasible, QpStatus::Infeasible) => (Verdict::Ok, String::new()),
        (Truth::Infeasible, _) => (
            Verdict::Weak,
            format!("infeasible instance not certified: {status}"),
        ),
    };

    Outcome {
        status,
        violation: Some(viol),
        verdict,
        detail,
    }
}
