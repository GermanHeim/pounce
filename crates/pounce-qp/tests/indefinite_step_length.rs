//! gh #416 — an indefinite QP must take the step its model asks for.
//!
//! §4.5 inertia control factors `H + δI` when the reduced Hessian is not PD
//! on the active null space. The direction that comes back is a descent
//! direction for the *true* model, but the unit step no longer minimizes
//! along it: `α* = 1 + δ‖p‖²/pᵀHp`, which is `+∞` when the true curvature
//! along `p` is non-positive. Capping at 1 anyway turns the inner loop into
//! proximal-point iteration with parameter δ — and δ is chosen by
//! multiplying `inertia_shift_initial` by `inertia_shift_factor` (100) until
//! the shifted system is PD, so it usually *dominates* the spectrum and each
//! "full step" is a δ-sized crawl. The reported symptom was a QP burning its
//! whole 200-iteration budget with **zero** working-set changes, on a
//! Rosenbrock SQP subproblem whose only obstacle was one negative eigenvalue.
//!
//! Each fixture below is a 2-variable indefinite QP whose answer sits at a
//! bound reached along the negative-curvature direction. Pre-fix they all
//! exit `MaxIter`; the iteration-count assertions are what pins the fix
//! (the answers themselves are also wrong pre-fix, but only because the
//! crawl was interrupted).

use pounce_linalg::triplet::{GenTMatrix, GenTMatrixSpace, SymTMatrix, SymTMatrixSpace};
use pounce_qp::{
    HessianInertia, ParametricActiveSetSolver, QpOptions, QpProblem, QpSolver, QpStatus,
};
use std::rc::Rc;

const NEG_INF: f64 = -1e20;
const POS_INF: f64 = 1e20;

/// `H = diag(1, −1)` — PD in `x₁`, negative curvature in `x₂`.
fn saddle_hessian() -> SymTMatrix {
    let space = SymTMatrixSpace::new(2, vec![1, 2], vec![1, 2]);
    let mut h = SymTMatrix::new(Rc::clone(&space));
    h.set_values(&[1.0, -1.0]);
    h
}

fn no_rows() -> GenTMatrix {
    GenTMatrix::new(GenTMatrixSpace::new(0, 2, Vec::new(), Vec::new()))
}

fn new_solver() -> ParametricActiveSetSolver {
    ParametricActiveSetSolver::new(Box::new(pounce_feral::FeralSolverInterface::new()))
}

// ─────────────────────────────────────────────────────────────────
// Box path (`solve_box_constrained`).
//
//     min ½x₁² − ½x₂² − ½x₂   s.t. −1 ≤ x ≤ 1
//
// x₁ is convex with zero gradient ⇒ x₁* = 0. x₂ is concave, so its
// optimum is at a bound: −½ − ½ = −1 at x₂ = 1 versus −½ + ½ = 0 at
// x₂ = −1. Unique optimum (0, 1), obj −1.
//
// The cold iterate is the origin, where the reduced Hessian is
// indefinite ⇒ δ = 100 and the shifted step is `p₂ = 0.5/99 ≈ 5.05e−3`.
// Capped at α = 1 that needs ~198 iterations to walk x₂ out to its
// bound — the whole default budget, for one working-set change.
// Uncapped, the ratio test takes it there in one.
// ─────────────────────────────────────────────────────────────────
#[test]
fn box_negative_curvature_reaches_its_bound_in_one_step() {
    let h = saddle_hessian();
    let a = no_rows();
    let g = [0.0, -0.5];
    let bl: [f64; 0] = [];
    let bu: [f64; 0] = [];
    let xl = [-1.0, -1.0];
    let xu = [1.0, 1.0];

    let qp = QpProblem {
        n: 2,
        m: 0,
        h: &h,
        g: &g,
        a: &a,
        bl: &bl,
        bu: &bu,
        xl: &xl,
        xu: &xu,
        hessian_inertia: HessianInertia::Indefinite,
    };

    let sol = new_solver()
        .solve(&qp, None, &QpOptions::default())
        .expect("indefinite box QP must solve");

    assert_eq!(sol.status, QpStatus::Optimal, "x = {:?}", sol.x);
    assert!(
        (sol.x[0]).abs() < 1e-9 && (sol.x[1] - 1.0).abs() < 1e-9,
        "expected x* = (0, 1), got {:?}",
        sol.x
    );
    assert!((sol.obj + 1.0).abs() < 1e-9, "obj = {}", sol.obj);
    // One add for the x₂ bound, and the refactor count is the iteration
    // count: pre-fix this was 200 refactors and a `MaxIter` exit.
    assert_eq!(sol.stats.n_working_set_changes, 1);
    assert!(
        sol.stats.n_refactor <= 5,
        "took {} refactorizations for a single active-set change",
        sol.stats.n_refactor
    );
}

// ─────────────────────────────────────────────────────────────────
// Same fixture with `x₂` unbounded above: the model now falls forever
// along the negative-curvature direction and nothing blocks it, which
// is exactly the recession certificate — with `pᵀHp < 0` in place of
// the `Hp = 0` that `ray_is_unbounded_descent` looks for.
// ─────────────────────────────────────────────────────────────────
#[test]
fn box_negative_curvature_with_no_blocker_is_unbounded() {
    let h = saddle_hessian();
    let a = no_rows();
    let g = [0.0, -0.5];
    let bl: [f64; 0] = [];
    let bu: [f64; 0] = [];
    let xl = [-1.0, -1.0];
    let xu = [1.0, POS_INF];

    let qp = QpProblem {
        n: 2,
        m: 0,
        h: &h,
        g: &g,
        a: &a,
        bl: &bl,
        bu: &bu,
        xl: &xl,
        xu: &xu,
        hessian_inertia: HessianInertia::Indefinite,
    };

    let sol = new_solver()
        .solve(&qp, None, &QpOptions::default())
        .expect("indefinite box QP must solve");

    assert_eq!(sol.status, QpStatus::Unbounded, "x = {:?}", sol.x);
    let ray = sol.unbounded_ray.expect("unbounded verdict carries a ray");
    assert!(
        ray[1] > 0.0 && ray[1].abs() > ray[0].abs(),
        "ray {ray:?} should point along +x₂, the negative-curvature direction"
    );
}

// ─────────────────────────────────────────────────────────────────
// General path (`solve_general`): same QP plus one general inequality
// row that never binds, which is enough to route the dispatcher away
// from the box fast path.
//
//     min ½x₁² − ½x₂² − ½x₂
//     s.t. x₁ + x₂ ≤ 10,  −1 ≤ x ≤ 1
//
// The row is slack at every feasible point, so the optimum is the box
// problem's: (0, 1), obj −1.
// ─────────────────────────────────────────────────────────────────
#[test]
fn general_path_negative_curvature_reaches_its_bound() {
    let h = saddle_hessian();
    let a_space = GenTMatrixSpace::new(1, 2, vec![1, 1], vec![1, 2]);
    let mut a = GenTMatrix::new(Rc::clone(&a_space));
    a.set_values(&[1.0, 1.0]);

    let g = [0.0, -0.5];
    let bl = [NEG_INF];
    let bu = [10.0];
    let xl = [-1.0, -1.0];
    let xu = [1.0, 1.0];

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
        hessian_inertia: HessianInertia::Indefinite,
    };

    for schur in [false, true] {
        let opts = QpOptions {
            use_schur_updates: schur,
            ..QpOptions::default()
        };
        let sol = new_solver()
            .solve(&qp, None, &opts)
            .expect("indefinite general QP must solve");

        assert_eq!(sol.status, QpStatus::Optimal, "schur={schur} x={:?}", sol.x);
        assert!(
            (sol.x[0]).abs() < 1e-9 && (sol.x[1] - 1.0).abs() < 1e-9,
            "schur={schur}: expected x* = (0, 1), got {:?}",
            sol.x
        );
        assert!(
            sol.stats.n_refactor + sol.stats.n_schur_updates <= 6,
            "schur={schur}: {} refactors + {} updates for one active-set change",
            sol.stats.n_refactor,
            sol.stats.n_schur_updates,
        );
    }
}

// ─────────────────────────────────────────────────────────────────
// Convex control: with a PD Hessian no shift fires (δ = 0), the unit
// step IS the model minimizer, and behaviour must be untouched.
//
//     min ½x₁² + ½x₂² − x₁ − 3x₂,  −1 ≤ x ≤ 1
//
// Unconstrained minimizer (1, 3) ⇒ x* = (1, 1) at the box corner.
// ─────────────────────────────────────────────────────────────────
#[test]
fn convex_box_qp_is_unaffected() {
    let space = SymTMatrixSpace::new(2, vec![1, 2], vec![1, 2]);
    let mut h = SymTMatrix::new(Rc::clone(&space));
    h.set_values(&[1.0, 1.0]);
    let a = no_rows();

    let g = [-1.0, -3.0];
    let bl: [f64; 0] = [];
    let bu: [f64; 0] = [];
    let xl = [-1.0, -1.0];
    let xu = [1.0, 1.0];

    let qp = QpProblem {
        n: 2,
        m: 0,
        h: &h,
        g: &g,
        a: &a,
        bl: &bl,
        bu: &bu,
        xl: &xl,
        xu: &xu,
        hessian_inertia: HessianInertia::Psd,
    };

    let sol = new_solver()
        .solve(&qp, None, &QpOptions::default())
        .expect("convex box QP must solve");

    assert_eq!(sol.status, QpStatus::Optimal);
    assert!(
        (sol.x[0] - 1.0).abs() < 1e-9 && (sol.x[1] - 1.0).abs() < 1e-9,
        "expected x* = (1, 1), got {:?}",
        sol.x
    );
}
