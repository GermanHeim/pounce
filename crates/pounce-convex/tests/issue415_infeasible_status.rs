//! gh #415 — an infeasible QP must come back as `PrimalInfeasible` from the
//! active-set driver, not `NumericalFailure`.
//!
//! The two statuses land in different AMPL `solve_result_num` families and
//! callers branch on that family: `200` tells AMPL / Pyomo / the GAMS links
//! "the model is infeasible — fix the model", while `500` tells them "the
//! solver broke — retry, or switch solvers". Reporting a diagnosed-infeasible
//! model as a solver crash is the actual user-visible harm here, so these tests
//! assert the *status*, and cross-check it against the IPM on the same data.
//!
//! This is the infeasible analogue of the unbounded case (#388) and the
//! rank-deficient equality case (#313).
//!
//! The three variants below are not redundant — they take three different
//! routes to the certificate:
//!
//! * **boxed** — the elastic multipliers carry a residual `Aᵀy + Gᵀz = −c`, and
//!   minimizing `qᵀx` over the finite box absorbs it exactly;
//! * **free** — no box to absorb anything, so the driver falls back on the
//!   objective-free feasibility twin, whose multipliers have no residual;
//! * **one-sided box** — a lower bound exists but the residual points at the
//!   *missing* upper bound, so this also takes the twin route. It is the one a
//!   `x ≥ 0`-style model actually hits.

use pounce_convex::{
    ActiveSetOverrides, QpOptions, QpProblem, QpStatus, Triplet, solve_qp_active_set, solve_qp_ipm,
};
use pounce_feral::FeralSolverInterface;
use pounce_linsol::SparseSymLinearSolverInterface;

fn backend() -> Box<dyn SparseSymLinearSolverInterface> {
    Box::new(FeralSolverInterface::new())
}

fn active_set(prob: &QpProblem) -> QpStatus {
    let mut mk = backend;
    solve_qp_active_set(
        prob,
        &QpOptions::default(),
        &ActiveSetOverrides::default(),
        &mut mk,
    )
    .status
}

/// The issue's LP: `min x₀ + x₁` s.t. `x₀ + x₁ ≤ 1` **and** `x₀ + x₁ ≥ 3`.
///
/// The second row is written `−x₀ − x₁ ≤ −3`. The two are directly
/// contradictory, so the feasible set is empty by inspection — no solver
/// tolerance or conditioning question enters into it. `scipy`'s HiGHS oracle
/// agrees (`status = 2`, "The problem is infeasible").
fn contradictory_rows() -> QpProblem {
    QpProblem {
        n: 2,
        p_lower: vec![],
        c: vec![1.0, 1.0],
        a: vec![],
        b: vec![],
        g: vec![
            Triplet::new(0, 0, 1.0),
            Triplet::new(0, 1, 1.0),
            Triplet::new(1, 0, -1.0),
            Triplet::new(1, 1, -1.0),
        ],
        h: vec![1.0, -3.0],
        lb: vec![0.0, 0.0],
        ub: vec![10.0, 10.0],
    }
}

#[test]
fn boxed_infeasible_lp_is_primal_infeasible() {
    let prob = contradictory_rows();
    assert_eq!(
        active_set(&prob),
        QpStatus::PrimalInfeasible,
        "active-set must report the model infeasible, not blame the solver"
    );
    // The engines must not disagree about the user's model.
    assert_eq!(
        solve_qp_ipm(&prob, &QpOptions::default(), backend).status,
        QpStatus::PrimalInfeasible,
        "IPM oracle"
    );
}

#[test]
fn free_variable_infeasible_lp_is_primal_infeasible() {
    let prob = QpProblem {
        lb: vec![],
        ub: vec![],
        ..contradictory_rows()
    };
    assert_eq!(active_set(&prob), QpStatus::PrimalInfeasible);
    assert_eq!(
        solve_qp_ipm(&prob, &QpOptions::default(), backend).status,
        QpStatus::PrimalInfeasible,
        "IPM oracle"
    );
}

#[test]
fn lower_bounded_infeasible_lp_is_primal_infeasible() {
    let prob = QpProblem {
        ub: vec![],
        ..contradictory_rows()
    };
    assert_eq!(active_set(&prob), QpStatus::PrimalInfeasible);
    assert_eq!(
        solve_qp_ipm(&prob, &QpOptions::default(), backend).status,
        QpStatus::PrimalInfeasible,
        "IPM oracle"
    );
}

/// A genuinely infeasible **QP** (nonzero Hessian), not just an LP: the
/// residual the box has to absorb is `−(Px + c)` and depends on the returned
/// point, so a curved objective exercises a different residual than the LP's
/// constant `−c`.
#[test]
fn infeasible_qp_with_hessian_is_primal_infeasible() {
    let prob = QpProblem {
        p_lower: vec![Triplet::new(0, 0, 2.0), Triplet::new(1, 1, 3.0)],
        c: vec![-4.0, 1.0],
        ..contradictory_rows()
    };
    assert_eq!(active_set(&prob), QpStatus::PrimalInfeasible);
    assert_eq!(
        solve_qp_ipm(&prob, &QpOptions::default(), backend).status,
        QpStatus::PrimalInfeasible,
        "IPM oracle"
    );
}

/// Infeasibility that lives in the **equality** block rather than the
/// inequalities: `x₀ + x₁ = 1` together with `x₀ + x₁ = 3`. This is the path
/// through `Aᵀy` and the `bᵀy` term of the certificate, which the inequality
/// tests above never touch.
#[test]
fn contradictory_equalities_are_primal_infeasible() {
    let prob = QpProblem {
        n: 2,
        p_lower: vec![],
        c: vec![1.0, 1.0],
        a: vec![
            Triplet::new(0, 0, 1.0),
            Triplet::new(0, 1, 1.0),
            Triplet::new(1, 0, 1.0),
            Triplet::new(1, 1, 1.0),
        ],
        b: vec![1.0, 3.0],
        g: vec![],
        h: vec![],
        lb: vec![0.0, 0.0],
        ub: vec![10.0, 10.0],
    };
    assert_eq!(active_set(&prob), QpStatus::PrimalInfeasible);
    assert_eq!(
        solve_qp_ipm(&prob, &QpOptions::default(), backend).status,
        QpStatus::PrimalInfeasible,
        "IPM oracle"
    );
}

/// The other half of the contract, and the more important one: a **feasible**
/// QP must never acquire an infeasibility verdict.
///
/// A wrong `PrimalInfeasible` is a false statement about the user's model —
/// strictly worse than any failure status, which only says the solver gave up.
/// The geometry here is the #282 hazard in miniature: the rows `±x₀ ≤ 0`,
/// `±x₁ ≤ 0` collapse the feasible set to exactly `{0}`, so every row is active
/// at the unique feasible point, there is no interior (Slater fails), and the
/// multipliers are wildly non-unique — which is precisely the situation in
/// which a phase-1 stall used to be mistaken for a certificate. `x = 0` is
/// plainly feasible, so no Farkas certificate can exist.
#[test]
fn feasible_qp_with_empty_interior_is_never_infeasible() {
    let prob = QpProblem {
        n: 2,
        p_lower: vec![Triplet::new(0, 0, 1.0), Triplet::new(1, 1, 1.0)],
        c: vec![-1.0, -1.0],
        a: vec![],
        b: vec![],
        g: vec![
            Triplet::new(0, 0, 1.0),
            Triplet::new(1, 0, -1.0),
            Triplet::new(2, 1, 1.0),
            Triplet::new(3, 1, -1.0),
        ],
        h: vec![0.0, 0.0, 0.0, 0.0],
        lb: vec![],
        ub: vec![],
    };
    let status = active_set(&prob);
    assert_ne!(
        status,
        QpStatus::PrimalInfeasible,
        "feasible set is exactly {{0}} — x = 0 satisfies every row, so there is \
         no Farkas certificate to find"
    );
    assert_eq!(
        solve_qp_ipm(&prob, &QpOptions::default(), backend).status,
        QpStatus::Optimal,
        "IPM oracle: this problem is solvable"
    );
}
