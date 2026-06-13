//! R1 de-risk for the LP-crossover bridge: prove the active-set engine
//! solves a *pure LP* (`H = 0`) whose optimum is pinned by **general
//! inequality rows** that a cold working set does NOT start with — so the
//! solver must drive its add/drop pivot loop through the §4.5 inertia-shift
//! δ path (`solver.rs:104-153`) to reach the vertex.
//!
//! Existing coverage (`analytical.rs::zero_hessian`, the box-constrained
//! tests) exercises `H = 0` only with variable *bounds*. Crossover will hand
//! pounce-qp an LP where every convex row — including expanded bound rows —
//! arrives as a *general constraint* and bounds are left free. This test
//! reproduces exactly that shape. If it fails, crossover is blocked at the
//! source and pounce-qp must be fixed first.
//!
//! Fixture:  min −2x₁ − x₂
//!   s.t.  x₁ + x₂ ≤ 6   (row 1)
//!         x₁      ≤ 4   (row 2)
//!              x₂ ≤ 4   (row 3)
//!        −x₁      ≤ 0   (row 4, i.e. x₁ ≥ 0)
//!             −x₂ ≤ 0   (row 5, i.e. x₂ ≥ 0)
//!   variable bounds all free.
//!
//! The objective pushes x₁ to its cap (row 2 binding), then x₂ as far as the
//! budget allows: x₁ + x₂ ≤ 6 ⇒ x₂ = 2 (row 1 binding). Unique optimal vertex
//! x* = (4, 2), f* = −10. Rows 1 and 2 are active; the cold working set holds
//! none of them, so the engine must add both via its pivot loop.

use pounce_linalg::triplet::{GenTMatrix, GenTMatrixSpace, SymTMatrix, SymTMatrixSpace};
use pounce_qp::working_set::WorkingSet;
use pounce_qp::{
    AntiCyclingChoice, HessianInertia, ParametricActiveSetSolver, QpOptions, QpProblem, QpSolver,
    QpStatus,
};
use std::rc::Rc;

#[test]
fn pure_lp_with_general_inequalities_from_cold_working_set() {
    // H = 0: empty symmetric triplet of dimension 2.
    let h_space = SymTMatrixSpace::new(2, Vec::new(), Vec::new());
    let h = SymTMatrix::new(Rc::clone(&h_space));

    // 5 general inequality rows over 2 variables.
    let a_space = GenTMatrixSpace::new(5, 2, vec![1, 1, 2, 3, 4, 5], vec![1, 2, 1, 2, 1, 2]);
    let mut a = GenTMatrix::new(Rc::clone(&a_space));
    a.set_values(&[1.0, 1.0, 1.0, 1.0, -1.0, -1.0]);

    let g = [-2.0, -1.0];
    let neg_inf = -1e20;
    let bl = [neg_inf; 5];
    let bu = [6.0, 4.0, 4.0, 0.0, 0.0];
    let xl = [neg_inf, neg_inf];
    let xu = [1e20, 1e20];

    let qp = QpProblem {
        n: 2,
        m: 5,
        h: &h,
        g: &g,
        a: &a,
        bl: &bl,
        bu: &bu,
        xl: &xl,
        xu: &xu,
        hessian_inertia: HessianInertia::Psd,
    };

    let mut solver =
        ParametricActiveSetSolver::new(Box::new(pounce_feral::FeralSolverInterface::new()));
    // Bland keeps the pivot loop guaranteed-finite for the degenerate LP
    // path; matches the rule crossover will use.
    let opts = QpOptions {
        anti_cycling: AntiCyclingChoice::Bland,
        max_iter: 1000,
        ..QpOptions::default()
    };

    let working = WorkingSet::cold(2, 5);
    let sol = solver
        .solve_with_working_set(&qp, &working, &opts)
        .expect("pure-LP solve_with_working_set must succeed");

    assert_eq!(sol.status, QpStatus::Optimal, "status = {:?}", sol.status);
    assert!((sol.x[0] - 4.0).abs() < 1e-8, "x[0] = {}", sol.x[0]);
    assert!((sol.x[1] - 2.0).abs() < 1e-8, "x[1] = {}", sol.x[1]);
    assert!((sol.obj - (-10.0)).abs() < 1e-8, "obj = {}", sol.obj);
}

/// Degenerate variant: the same optimal vertex x* = (4, 2) is pinned by
/// THREE general inequality rows (rows 1, 2, and a redundant 2x₁ + x₂ ≤ 10),
/// i.e. more binding rows than variables. Loss of a unique active basis at a
/// degenerate vertex is precisely what defeats the pure IPM on the NETLIB GEN
/// family; crossover must still drive the active-set engine to the exact
/// vertex here. Bland anti-cycling guarantees termination through the
/// degeneracy.
#[test]
fn degenerate_pure_lp_more_binding_rows_than_variables() {
    let h_space = SymTMatrixSpace::new(2, Vec::new(), Vec::new());
    let h = SymTMatrix::new(Rc::clone(&h_space));

    // 6 rows: the 5 above plus 2x₁ + x₂ ≤ 10 (binding & redundant at (4,2)).
    let a_space = GenTMatrixSpace::new(
        6,
        2,
        vec![1, 1, 2, 3, 4, 5, 6, 6],
        vec![1, 2, 1, 2, 1, 2, 1, 2],
    );
    let mut a = GenTMatrix::new(Rc::clone(&a_space));
    a.set_values(&[1.0, 1.0, 1.0, 1.0, -1.0, -1.0, 2.0, 1.0]);

    let g = [-2.0, -1.0];
    let neg_inf = -1e20;
    let bl = [neg_inf; 6];
    let bu = [6.0, 4.0, 4.0, 0.0, 0.0, 10.0];
    let xl = [neg_inf, neg_inf];
    let xu = [1e20, 1e20];

    let qp = QpProblem {
        n: 2,
        m: 6,
        h: &h,
        g: &g,
        a: &a,
        bl: &bl,
        bu: &bu,
        xl: &xl,
        xu: &xu,
        hessian_inertia: HessianInertia::Psd,
    };

    let mut solver =
        ParametricActiveSetSolver::new(Box::new(pounce_feral::FeralSolverInterface::new()));
    let opts = QpOptions {
        anti_cycling: AntiCyclingChoice::Bland,
        max_iter: 1000,
        ..QpOptions::default()
    };

    let working = WorkingSet::cold(2, 6);
    let sol = solver
        .solve_with_working_set(&qp, &working, &opts)
        .expect("degenerate pure-LP solve_with_working_set must succeed");

    assert_eq!(sol.status, QpStatus::Optimal, "status = {:?}", sol.status);
    assert!((sol.x[0] - 4.0).abs() < 1e-8, "x[0] = {}", sol.x[0]);
    assert!((sol.x[1] - 2.0).abs() < 1e-8, "x[1] = {}", sol.x[1]);
    assert!((sol.obj - (-10.0)).abs() < 1e-8, "obj = {}", sol.obj);
}
