//! Presolve round-trip exactness (the Phase 3.5 correctness contract):
//! solving with presolve must reproduce the no-presolve `(x, y, z)` to
//! tolerance — primal *and* dual. Also covers presolve-detected
//! infeasibility.
//!
//! Tolerance note: each assertion compares *two independent* IPM solves
//! (direct vs presolved), so the bar is the solvers' own convergence
//! tolerance, not exact equality. We use 1e-5.

use pounce_convex::presolve::{presolve, solve_with_presolve, PresolveOutcome};
use pounce_convex::{solve_qp_ipm, QpOptions, QpProblem, QpStatus, Triplet};
use pounce_feral::FeralSolverInterface;
use pounce_linsol::SparseSymLinearSolverInterface;

const TOL: f64 = 1e-5;

fn backend() -> Box<dyn SparseSymLinearSolverInterface> {
    Box::new(FeralSolverInterface::new())
}

fn direct(prob: &QpProblem) -> pounce_convex::QpSolution {
    solve_qp_ipm(prob, &QpOptions::default(), backend)
}

fn with_presolve(prob: &QpProblem) -> pounce_convex::QpSolution {
    solve_with_presolve(prob, |reduced| {
        solve_qp_ipm(reduced, &QpOptions::default(), backend)
    })
}

fn assert_close(a: &[f64], b: &[f64], what: &str) {
    assert_eq!(a.len(), b.len(), "{what}: length mismatch");
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        assert!((x - y).abs() < TOL, "{what}[{i}]: {x} vs {y}");
    }
}

/// Fixed-variable elimination: `min x0²+x1²+x2² s.t. x0+x1+x2=3, x2=2`.
/// The singleton row `x2=2` fixes x2; presolve substitutes it out.
#[test]
fn fixed_variable_roundtrip_matches_direct() {
    let prob = QpProblem {
        n: 3,
        p_lower: vec![
            Triplet::new(0, 0, 2.0),
            Triplet::new(1, 1, 2.0),
            Triplet::new(2, 2, 2.0),
        ],
        c: vec![0.0, 0.0, 0.0],
        a: vec![
            Triplet::new(0, 0, 1.0),
            Triplet::new(0, 1, 1.0),
            Triplet::new(0, 2, 1.0),
            Triplet::new(1, 2, 1.0), // singleton → fixes x2 = 2
        ],
        b: vec![3.0, 2.0],
        g: vec![],
        h: vec![],
    };
    let d = direct(&prob);
    let p = with_presolve(&prob);
    assert_eq!(d.status, QpStatus::Optimal);
    assert_eq!(p.status, QpStatus::Optimal);
    assert_close(&p.x, &d.x, "x");
    assert_close(&p.y, &d.y, "y");
    assert!((p.obj - d.obj).abs() < TOL, "obj {} vs {}", p.obj, d.obj);
    assert!((p.x[2] - 2.0).abs() < 1e-9, "x2={}", p.x[2]);
}

/// Fixed variable coupling through an off-diagonal Hessian term, so the
/// substitution must move `P` coupling into the linear term:
/// `min x0² + x0 x1 + x1² s.t. x1 = 1`.
#[test]
fn fixed_variable_with_hessian_coupling_roundtrip() {
    let prob = QpProblem {
        n: 2,
        p_lower: vec![
            Triplet::new(0, 0, 2.0),
            Triplet::new(1, 0, 1.0), // x0 x1 coupling
            Triplet::new(1, 1, 2.0),
        ],
        c: vec![0.0, 0.0],
        a: vec![Triplet::new(0, 1, 1.0)], // x1 = 1
        b: vec![1.0],
        g: vec![],
        h: vec![],
    };
    let d = direct(&prob);
    let p = with_presolve(&prob);
    assert_eq!(p.status, QpStatus::Optimal);
    assert_close(&p.x, &d.x, "x");
    assert_close(&p.y, &d.y, "y");
    assert!((p.obj - d.obj).abs() < TOL, "obj {} vs {}", p.obj, d.obj);
}

/// Fixed variable plus an inequality whose RHS must be adjusted by the
/// substitution: `min x0²-6x0 s.t. x1=1, x0+x1 ≤ 3`. After fixing x1=1
/// the inequality becomes `x0 ≤ 2`, which binds (unconstrained x0=3).
#[test]
fn fixed_variable_adjusts_inequality_rhs() {
    let prob = QpProblem {
        n: 2,
        p_lower: vec![Triplet::new(0, 0, 2.0), Triplet::new(1, 1, 2.0)],
        c: vec![-6.0, 0.0],
        a: vec![Triplet::new(0, 1, 1.0)],
        b: vec![1.0],
        g: vec![Triplet::new(0, 0, 1.0), Triplet::new(0, 1, 1.0)], // x0+x1≤3
        h: vec![3.0],
    };
    let d = direct(&prob);
    let p = with_presolve(&prob);
    assert_eq!(p.status, QpStatus::Optimal);
    assert_close(&p.x, &d.x, "x");
    assert_close(&p.y, &d.y, "y");
    assert_close(&p.z, &d.z, "z");
    assert!((p.obj - d.obj).abs() < TOL, "obj {} vs {}", p.obj, d.obj);
    // The inequality binds with a clearly nonzero multiplier (~2).
    assert!(p.z[0] > 1.0, "inequality should bind, z={}", p.z[0]);
}

/// Empty-row removal must not change the solution and the empty row's
/// dual is 0. (Non-degenerate: the kept constraint is a strict equality.)
#[test]
fn empty_row_roundtrip() {
    let prob = QpProblem {
        n: 2,
        p_lower: vec![Triplet::new(0, 0, 2.0), Triplet::new(1, 1, 2.0)],
        c: vec![0.0, 0.0],
        a: vec![
            Triplet::new(0, 0, 0.0), // empty row, b=0 → feasible, dropped
            Triplet::new(1, 0, 1.0), // x0 + x1 = 2
            Triplet::new(1, 1, 1.0),
        ],
        b: vec![0.0, 2.0],
        g: vec![],
        h: vec![],
    };
    let d = direct(&prob);
    let p = with_presolve(&prob);
    assert_eq!(p.status, QpStatus::Optimal);
    assert_close(&p.x, &d.x, "x");
    assert!(p.y[0].abs() < 1e-9, "empty-row dual={}", p.y[0]);
}

/// Presolve detects trivial primal infeasibility from `0 = 5`.
#[test]
fn empty_row_infeasible_detected() {
    let prob = QpProblem {
        n: 1,
        p_lower: vec![Triplet::new(0, 0, 2.0)],
        c: vec![0.0],
        a: vec![Triplet::new(0, 0, 0.0)], // 0·x0 = 5
        b: vec![5.0],
        g: vec![],
        h: vec![],
    };
    assert!(matches!(presolve(&prob), PresolveOutcome::Infeasible));
    assert_eq!(with_presolve(&prob).status, QpStatus::PrimalInfeasible);
}

/// Nothing to presolve → identity round-trip. Non-degenerate: the bound
/// that binds (x0 ≤ 1, with unconstrained optimum x0 = 3) has a clearly
/// nonzero multiplier, so the two solves agree well within tolerance.
#[test]
fn noop_presolve_roundtrip() {
    let prob = QpProblem {
        n: 2,
        p_lower: vec![Triplet::new(0, 0, 2.0), Triplet::new(1, 1, 2.0)],
        c: vec![-6.0, -4.0], // unconstrained opt (3, 2)
        a: vec![],
        b: vec![],
        g: vec![
            Triplet::new(0, 0, 1.0),  // x0 ≤ 1 (binds, mult ~4)
            Triplet::new(1, 1, 1.0),  // x1 ≤ 5 (inactive)
            Triplet::new(2, 0, -1.0), // x0 ≥ 0
            Triplet::new(3, 1, -1.0), // x1 ≥ 0
        ],
        h: vec![1.0, 5.0, 0.0, 0.0],
    };
    let d = direct(&prob);
    let p = with_presolve(&prob);
    assert_close(&p.x, &d.x, "x");
    assert_close(&p.z, &d.z, "z");
}
