//! gh #423 — `QpOptions::certify_recession_ray = false`: a point, not a
//! verdict.
//!
//! gh #416 taught the inner loop to run a δ-shifted negative-curvature
//! direction out to the model's own minimizer along it, and gh #419's
//! companion change made such a direction with *nothing to block it* a
//! recession certificate (`QpStatus::Unbounded`). That is the right answer
//! for a standalone QP. It is not always a usable one for the SQP outer
//! loop, whose step QP is unbounded below at every indefinite iterate of a
//! nonconvex NLP that has no blocking bound — with `m = 0` and no finite
//! bounds, at *every* indefinite iterate there is. The driver re-tests the
//! ray against the true NLP and, when the NLP turns out to be bounded, asks
//! the same subproblem again with the certificate declined: the shift stays,
//! the unblocked direction takes the δ-shifted proximal step (`α = 1`), and
//! the solve returns a point.
//!
//! The fixtures below are the `indefinite_step_length.rs` unbounded cases
//! with the flag flipped. Each asserts the same two things: the solve does
//! not claim unboundedness, and it moves — a certificate replaced by a
//! zero step would be no fix at all.

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

fn declining() -> QpOptions {
    QpOptions {
        certify_recession_ray: false,
        ..QpOptions::default()
    }
}

/// The certificate is what a standalone `solve_qp` is for, so it stays on
/// unless a caller deliberately turns it off.
#[test]
fn certification_is_on_by_default() {
    assert!(QpOptions::default().certify_recession_ray);
}

// ─────────────────────────────────────────────────────────────────
// Box path (`solve_box_constrained`).
//
//     min ½x₁² − ½x₂² − ½x₂   s.t. −1 ≤ x₁ ≤ 1,  −1 ≤ x₂
//
// `x₂` is concave and unbounded above, so the model falls forever along
// `+x₂` and no bound blocks it: `box_negative_curvature_with_no_blocker_is_
// unbounded` in `indefinite_step_length.rs` is this exact QP under the
// default options, and gets `Unbounded`.
// ─────────────────────────────────────────────────────────────────
#[test]
fn box_path_declines_the_certificate_and_steps() {
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

    // Control: the same QP does certify under the defaults.
    let certified = new_solver()
        .solve(&qp, None, &QpOptions::default())
        .expect("indefinite box QP must solve");
    assert_eq!(certified.status, QpStatus::Unbounded);

    let sol = new_solver()
        .solve(&qp, None, &declining())
        .expect("indefinite box QP must solve");

    assert_ne!(
        sol.status,
        QpStatus::Unbounded,
        "the caller declined the certificate"
    );
    assert!(sol.unbounded_ray.is_none());
    // Proximal-point iteration on a model that really does fall forever
    // never reaches a stationary point, so it walks until the budget runs
    // out — but it *walks*, along the negative-curvature direction. (The
    // SQP driver only adopts an `Optimal` re-solve, so this case still
    // reports honestly upstream.)
    assert!(
        sol.x[1] > 0.0,
        "expected progress along +x₂, got x = {:?}",
        sol.x
    );
}

// ─────────────────────────────────────────────────────────────────
// General path (`solve_general`): the same QP plus one general row
// `x₁ ≤ 10`. It never binds, and being an *inequality* it routes the
// dispatcher to `solve_general`; being flat in `x₂` it cannot block the
// negative-curvature direction either.
// ─────────────────────────────────────────────────────────────────
#[test]
fn general_path_declines_the_certificate_and_steps() {
    let h = saddle_hessian();
    let a_space = GenTMatrixSpace::new(1, 2, vec![1], vec![1]);
    let mut a = GenTMatrix::new(a_space);
    a.set_values(&[1.0]);
    let g = [0.0, -0.5];
    let bl = [NEG_INF];
    let bu = [10.0];
    let xl = [-1.0, -1.0];
    let xu = [1.0, POS_INF];

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

    let certified = new_solver()
        .solve(&qp, None, &QpOptions::default())
        .expect("indefinite general QP must solve");
    assert_eq!(certified.status, QpStatus::Unbounded);

    let sol = new_solver()
        .solve(&qp, None, &declining())
        .expect("indefinite general QP must solve");

    assert_ne!(sol.status, QpStatus::Unbounded);
    assert!(sol.unbounded_ray.is_none());
    assert!(
        sol.x[1] > 0.0,
        "expected progress along +x₂, got x = {:?}",
        sol.x
    );
}

// ─────────────────────────────────────────────────────────────────
// Equality-only path (`solve_equality_only`), which is where an
// unconstrained NLP's step QP lands: `m = 0` and no finite bound is
// exactly the "pure equality, no bounds" predicate.
//
//     min ½x₁² − x₂      (H = diag(1, 0), unconstrained)
//
// Flat and strictly descending in `x₂` — the N1 zero-curvature recession
// certificate, which this path detects from the blow-up of the δ-shifted
// saddle solution. Declining it returns that saddle solution as the
// answer, which is precisely the proximal point
// `argmin q(y) + ½δ‖y‖²` — a real descent step, not a certificate.
// ─────────────────────────────────────────────────────────────────
#[test]
fn equality_only_path_declines_the_certificate_and_returns_the_proximal_point() {
    let space = SymTMatrixSpace::new(2, vec![1, 2], vec![1, 2]);
    let mut h = SymTMatrix::new(Rc::clone(&space));
    h.set_values(&[1.0, 0.0]);
    let a = no_rows();
    let g = [0.0, -1.0];
    let bl: [f64; 0] = [];
    let bu: [f64; 0] = [];
    let xl = [NEG_INF, NEG_INF];
    let xu = [POS_INF, POS_INF];

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

    let certified = new_solver()
        .solve(&qp, None, &QpOptions::default())
        .expect("flat unbounded QP must solve");
    assert_eq!(certified.status, QpStatus::Unbounded);

    let sol = new_solver()
        .solve(&qp, None, &declining())
        .expect("flat unbounded QP must solve");

    assert_eq!(sol.status, QpStatus::Optimal);
    assert!(sol.unbounded_ray.is_none());
    assert!(
        sol.x[1] > 0.0 && sol.x[1].is_finite(),
        "expected a finite proximal step along +x₂, got x = {:?}",
        sol.x
    );
    // A descent step on the model, which is all the outer loop needs from
    // it: q(0) = 0 here.
    assert!(sol.obj < 0.0, "obj = {} is not descent", sol.obj);
}
