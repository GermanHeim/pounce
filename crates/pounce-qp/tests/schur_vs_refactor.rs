//! Schur-update path vs refactor path — they must agree.
//!
//! `QpOptions::use_schur_updates` is documented as a pure *performance* switch:
//! when set, `solve_general` absorbs working-set changes as rank-2
//! Sherman-Morrison-Woodbury updates against a cached factor of the fixed-dim
//! `K_max` instead of assembling and factoring a fresh active-set KKT each
//! iteration. The crate README states the path's "correctness verified by
//! Schur- vs-refactor cross-checks".
//!
//! It is not equivalent in practice. These tests pin the discrepancy on small
//! problems with closed-form optima, so the failing side is unambiguous rather
//! than a benchmark-scale mystery. Found while enabling the Schur path on the
//! convex active-set QP driver (`pounce_convex::active_set`) to cut its
//! per-iteration refactorization cost: enabling it dropped the Maros-Mészáros
//! spot check from 22/24 to 20/24.

use pounce_common::types::{NLP_LOWER_BOUND_INF, NLP_UPPER_BOUND_INF};
use pounce_linalg::triplet::{GenTMatrix, GenTMatrixSpace, SymTMatrix, SymTMatrixSpace};
use pounce_qp::{
    HessianInertia, ParametricActiveSetSolver, QpOptions, QpProblem, QpSolver, QpStatus,
};
use std::rc::Rc;

fn new_solver() -> ParametricActiveSetSolver {
    ParametricActiveSetSolver::new(Box::new(pounce_feral::FeralSolverInterface::new()))
}

struct Case {
    n: usize,
    m: usize,
    /// 1-based lower-triangle Hessian triplets.
    h: (Vec<i32>, Vec<i32>, Vec<f64>),
    g: Vec<f64>,
    /// 1-based Jacobian triplets.
    a: (Vec<i32>, Vec<i32>, Vec<f64>),
    bl: Vec<f64>,
    bu: Vec<f64>,
    xl: Vec<f64>,
    xu: Vec<f64>,
}

/// Solve `case` with `use_schur_updates = schur`, returning `(status, x, obj)`.
fn solve(case: &Case, schur: bool) -> (QpStatus, Vec<f64>, f64) {
    solve_with(
        case,
        schur,
        QpOptions::default().max_schur_updates_before_refactor,
    )
}

/// As [`solve`], but with an explicit Schur-block refactor interval, so a test
/// can probe how far the answer drifts as rank-2 updates accumulate.
fn solve_with(case: &Case, schur: bool, refactor_every: u32) -> (QpStatus, Vec<f64>, f64) {
    let h_space = SymTMatrixSpace::new(case.n as i32, case.h.0.clone(), case.h.1.clone());
    let mut h = SymTMatrix::new(Rc::clone(&h_space));
    h.set_values(&case.h.2);

    let a_space = GenTMatrixSpace::new(
        case.m as i32,
        case.n as i32,
        case.a.0.clone(),
        case.a.1.clone(),
    );
    let mut a = GenTMatrix::new(Rc::clone(&a_space));
    a.set_values(&case.a.2);

    let qp = QpProblem {
        n: case.n,
        m: case.m,
        h: &h,
        g: &case.g,
        a: &a,
        bl: &case.bl,
        bu: &case.bu,
        xl: &case.xl,
        xu: &case.xu,
        hessian_inertia: HessianInertia::Psd,
    };
    let opts = QpOptions {
        use_schur_updates: schur,
        max_schur_updates_before_refactor: refactor_every,
        ..QpOptions::default()
    };
    let sol = new_solver().solve(&qp, None, &opts).expect("qp solve");
    (sol.status, sol.x.clone(), sol.obj)
}

/// `min ½x₁² − 2x₀ − x₁  s.t.  x₀ + x₁ ≤ 2,  x₀ ≤ 1.5,  x ≥ 0`.
///
/// `P = diag(0, 1)` — no curvature along `x₀`, so the reduced Hessian is
/// singular unless the working set pins that direction, which is exactly the
/// geometry where a stale/mis-signed Schur block shows up. Both rows bind at
/// the optimum: stationarity `(−2, x₁−1) + z₀(1,1) + z₁(1,0) = 0` gives
/// `z₀ = 0.5`, `z₁ = 1.5`, both `≥ 0`, so
/// `x* = (1.5, 0.5)` and `f* = 0.125 − 3 − 0.5 = −3.375`.
fn singular_hessian_case() -> Case {
    Case {
        n: 2,
        m: 2,
        h: (vec![2], vec![2], vec![1.0]), // H[1,1] = 1 (1-based), i.e. diag(0,1)
        g: vec![-2.0, -1.0],
        a: (vec![1, 1, 2], vec![1, 2, 1], vec![1.0, 1.0, 1.0]),
        bl: vec![NLP_LOWER_BOUND_INF, NLP_LOWER_BOUND_INF],
        bu: vec![2.0, 1.5],
        xl: vec![0.0, 0.0],
        xu: vec![NLP_UPPER_BOUND_INF, NLP_UPPER_BOUND_INF],
    }
}

#[test]
fn refactor_path_solves_singular_hessian() {
    let (status, x, obj) = solve(&singular_hessian_case(), false);
    assert_eq!(status, QpStatus::Optimal, "refactor path status");
    assert!((x[0] - 1.5).abs() < 1e-7, "x0 = {}", x[0]);
    assert!((x[1] - 0.5).abs() < 1e-7, "x1 = {}", x[1]);
    assert!((obj + 3.375).abs() < 1e-7, "obj = {obj}");
}

/// The Schur path must reach the *same* answer — it is a performance switch,
/// not an algorithm change.
#[test]
fn schur_path_matches_refactor_on_singular_hessian() {
    let case = singular_hessian_case();
    let (rs, rx, robj) = solve(&case, false);
    let (ss, sx, sobj) = solve(&case, true);

    assert_eq!(
        ss, rs,
        "status differs: schur {ss:?} vs refactor {rs:?} (obj {sobj} vs {robj})"
    );
    for i in 0..case.n {
        assert!(
            (sx[i] - rx[i]).abs() < 1e-7,
            "x[{i}] differs: schur {} vs refactor {}",
            sx[i],
            rx[i]
        );
    }
    assert!(
        (sobj - robj).abs() < 1e-7,
        "obj differs: schur {sobj} vs refactor {robj}"
    );
}

/// Is the Schur path's error *accumulation* in the rank-2 updates, or a
/// systematic bug? Forcing a refactor after every single update makes the
/// update layer never grow past one column; if the answer sharpens toward the
/// refactor path's, the discrepancy is drift, and the fix belongs in the solve
/// (refinement) rather than in the update algebra.
#[test]
fn schur_accuracy_vs_refactor_interval() {
    let case = singular_hessian_case();
    let (_, rx, _) = solve_with(&case, false, 50);
    for every in [1u32, 5, 50] {
        let (st, sx, _) = solve_with(&case, true, every);
        let err = (sx[0] - rx[0]).abs().max((sx[1] - rx[1]).abs());
        println!("refactor_every = {every:>2}: status {st:?}  max|dx| = {err:.3e}");
    }
}

/// **LICQ-violating redundant equality** — the rank-detection case the crate
/// README lists as a known limitation (analytical-ladder problem #4).
///
/// `min ½(x₀² + x₁²)  s.t.  x₀ + x₁ = 1,  2x₀ + 2x₁ = 2`
///
/// The second equality is exactly twice the first, so the active-set Jacobian
/// has rank 1 with 2 rows and the active-set KKT is singular. No H-block
/// inertia shift repairs a rank-deficient *constraint* block, so both paths
/// must instead detect the dependence, prune to a maximal independent subset,
/// and continue. The optimum is `x* = (0.5, 0.5)`, `f* = 0.25`.
///
/// `solve_general` has had this guard for a long time; `solve_general_schur`
/// had none, so enabling `use_schur_updates` turned this shape into a hard
/// `LinearSolverFailure("KKT matrix is singular (LICQ violation …)")`. Both
/// paths are asserted here so the two cannot drift apart again.
fn licq_redundant_equality_case() -> Case {
    Case {
        n: 2,
        m: 2,
        h: (vec![1, 2], vec![1, 2], vec![1.0, 1.0]), // P = I
        g: vec![0.0, 0.0],
        a: (vec![1, 1, 2, 2], vec![1, 2, 1, 2], vec![1.0, 1.0, 2.0, 2.0]),
        bl: vec![1.0, 2.0],
        bu: vec![1.0, 2.0],
        xl: vec![NLP_LOWER_BOUND_INF, NLP_LOWER_BOUND_INF],
        xu: vec![NLP_UPPER_BOUND_INF, NLP_UPPER_BOUND_INF],
    }
}

#[test]
fn licq_violating_equality_solves_on_both_paths() {
    let case = licq_redundant_equality_case();
    for schur in [false, true] {
        let (status, x, obj) = solve(&case, schur);
        let path = if schur { "schur" } else { "refactor" };
        assert_eq!(status, QpStatus::Optimal, "{path} path status");
        assert!((x[0] - 0.5).abs() < 1e-7, "{path}: x0 = {}", x[0]);
        assert!((x[1] - 0.5).abs() < 1e-7, "{path}: x1 = {}", x[1]);
        assert!((obj - 0.25).abs() < 1e-7, "{path}: obj = {obj}");
    }
}
