//! gh #871: `refute_indefinite_optimum` must survive an equality row.
//!
//! Every `QpProblem` in both gh #848 test files has `a: vec![]`. That is the
//! gh #756 pattern — a green guard proving nothing about a branch its fixture
//! never reaches — and the branch in question is the first statement of
//! `max_feasible_step`, a hard `return None` on any direction carrying an
//! equality component. The curvature search runs on `P`, not on `P` restricted
//! to `null(A)`, so on a model whose negative direction is not already in
//! `null(A)` the screen produced a direction it then rejected, every time.
//!
//! The corpus below is that dimension: identical models, `m_eq = 1`.
//!
//! ## Which branch each test reaches
//!
//! | test | what it catches |
//! |---|---|
//! | `a_saddle_behind_an_equality_row_is_refuted` | the projection missing entirely: the direction is `e₀`, `A e₀ = 1` |
//! | `an_unbounded_qp_behind_an_equality_row_certifies_dual_infeasible` | the projected direction not reaching `ray_certifies_unbounded`, which applies the same equality test |
//! | `the_same_model_without_the_equality_row_was_already_refuted` | the mutation guard: it names the equality row as the discriminator, so a regression here cannot be read as the fixture simply being hard |
//! | `the_negative_direction_is_orthogonal_to_the_null_space` | the reduced search: `P`'s own most-negative eigenvector projects to *zero*, so projecting a direction found on `P` cannot rescue this one and only the `ΠPΠ` iteration can |
//! | `a_convex_qp_with_equality_rows_is_untouched` | the new projection leaking a demotion onto a PSD model |

use pounce_convex::{
    ActiveSetOverrides, HessianInertia, QpOptions, QpProblem, QpSolution, QpStatus, Triplet,
    solve_qp_active_set_inertia,
};
use pounce_feral::FeralSolverInterface;
use pounce_linsol::SparseSymLinearSolverInterface;

fn backend() -> Box<dyn SparseSymLinearSolverInterface> {
    Box::new(FeralSolverInterface::new())
}

fn solve(prob: &QpProblem, inertia: HessianInertia) -> QpSolution {
    let mut mk = backend;
    solve_qp_active_set_inertia(
        prob,
        &QpOptions::default(),
        &ActiveSetOverrides::default(),
        inertia,
        &mut mk,
    )
}

/// `min −x₀²  s.t.  x₀ + x₁ + x₂ = 0`, `x₀ ∈ [0, 1]`, `x₁, x₂ ∈ [−1, 1]`.
///
/// The origin is first-order clean — `Px + c = 0`, `x₀` weakly at its lower
/// bound — and the true minimum is `−1` at `x₀ = 1` (feasible with, say,
/// `x₁ = −1, x₂ = 0`). `ipopt` returns `−1` on this model, and so does POUNCE's
/// own NLP arm, which is what `solver_selection=auto` routes to; the engine
/// under test here is the opt-in one.
fn saddle_behind_an_equality_row() -> QpProblem {
    QpProblem {
        n: 3,
        p_lower: vec![Triplet::new(0, 0, -2.0)],
        c: vec![0.0, 0.0, 0.0],
        a: vec![
            Triplet::new(0, 0, 1.0),
            Triplet::new(0, 1, 1.0),
            Triplet::new(0, 2, 1.0),
        ],
        b: vec![0.0],
        g: vec![],
        h: vec![],
        lb: vec![0.0, -1.0, -1.0],
        ub: vec![1.0, 1.0, 1.0],
    }
}

#[test]
fn a_saddle_behind_an_equality_row_is_refuted() {
    let sol = solve(&saddle_behind_an_equality_row(), HessianInertia::Indefinite);
    assert_ne!(
        sol.status,
        QpStatus::Optimal,
        "the origin is a saddle worth −1 less than the optimum; got Optimal at \
         {:?} (obj {})",
        sol.x,
        sol.obj
    );
}

#[test]
fn the_same_model_without_the_equality_row_was_already_refuted() {
    // The discriminator, stated as a test rather than as a comment: drop the
    // row and the screen has always worked. A failure *here* is a different
    // defect from a failure above, and the two must not be confused.
    let mut prob = saddle_behind_an_equality_row();
    prob.a.clear();
    prob.b.clear();
    let sol = solve(&prob, HessianInertia::Indefinite);
    assert_ne!(sol.status, QpStatus::Optimal, "obj = {}", sol.obj);
}

/// The same objective and row with the box opened: `x₀ ≥ 0` and nothing else.
///
/// `min −x₀²` along the feasible ray `(2, −1, −1)` falls without bound, so this
/// is `DualInfeasible`, not a merely suboptimal point. Reported `Optimal` at
/// `f = 0` before the fix.
fn unbounded_behind_an_equality_row() -> QpProblem {
    QpProblem {
        n: 3,
        p_lower: vec![Triplet::new(0, 0, -2.0)],
        c: vec![0.0, 0.0, 0.0],
        a: vec![
            Triplet::new(0, 0, 1.0),
            Triplet::new(0, 1, 1.0),
            Triplet::new(0, 2, 1.0),
        ],
        b: vec![0.0],
        g: vec![],
        h: vec![],
        lb: vec![0.0, f64::NEG_INFINITY, f64::NEG_INFINITY],
        ub: vec![],
    }
}

#[test]
fn an_unbounded_qp_behind_an_equality_row_certifies_dual_infeasible() {
    let sol = solve(
        &unbounded_behind_an_equality_row(),
        HessianInertia::Indefinite,
    );
    assert_eq!(
        sol.status,
        QpStatus::DualInfeasible,
        "unbounded below along (2, −1, −1); got {:?} at {:?} (obj {})",
        sol.status,
        sol.x,
        sol.obj
    );
}

/// `min ½xᵀPx  s.t.  x₀ + x₁ = 0` over `[−1, 1]²`, with
/// `P = [[1, 5], [5, 1]]`.
///
/// `P`'s most negative eigenvector is `(1, −1)/√2`, which is *exactly* the
/// null space of the row — so here the unprojected search does find a usable
/// direction, and the model is the control for the next one.
fn saddle_whose_negative_direction_lies_in_the_null_space() -> QpProblem {
    QpProblem {
        n: 2,
        p_lower: vec![
            Triplet::new(0, 0, 1.0),
            Triplet::new(1, 0, 5.0),
            Triplet::new(1, 1, 1.0),
        ],
        c: vec![0.0, 0.0],
        a: vec![Triplet::new(0, 0, 1.0), Triplet::new(0, 1, 1.0)],
        b: vec![0.0],
        g: vec![],
        h: vec![],
        lb: vec![-1.0, -1.0],
        ub: vec![1.0, 1.0],
    }
}

/// The same `P` with the row turned ninety degrees: `x₀ − x₁ = 0`.
///
/// Now `null(A) = span{(1, 1)}`, along which the curvature is `+6`, while the
/// negative eigenvector `(1, −1)` projects to the zero vector. So a direction
/// found on `P` and then projected is useless, and the model is only refutable
/// through the reduced operator `ΠPΠ` — whose curvature here is `+6`, i.e.
/// the origin genuinely *is* the constrained minimum.
///
/// That is the point: this is the false-demotion test. The screen must leave
/// it alone.
#[test]
fn the_negative_direction_is_orthogonal_to_the_null_space() {
    let control = solve(
        &saddle_whose_negative_direction_lies_in_the_null_space(),
        HessianInertia::Indefinite,
    );
    assert!(
        (control.obj - (-4.0)).abs() < 1e-7,
        "control: the negative direction (1, −1) is feasible here and the \
         corner ±(1, −1) is worth −4; got obj {} at {:?}",
        control.obj,
        control.x
    );

    let mut prob = saddle_whose_negative_direction_lies_in_the_null_space();
    prob.a = vec![Triplet::new(0, 0, 1.0), Triplet::new(0, 1, -1.0)];
    let sol = solve(&prob, HessianInertia::Indefinite);
    assert_eq!(
        sol.status,
        QpStatus::Optimal,
        "on null(A) = span{{(1, 1)}} the curvature is +6 and the origin is the \
         constrained minimum — refuting it would be a false demotion; got \
         {:?} at {:?}",
        sol.status,
        sol.x
    );
    assert!(sol.obj.abs() < 1e-8, "obj = {}", sol.obj);
}

/// The PSD leak guard. The projection and the reduced search cost nothing on a
/// convex model because they never run there, and this says so with a model
/// carrying the equality rows they key off.
#[test]
fn a_convex_qp_with_equality_rows_is_untouched() {
    let prob = QpProblem {
        n: 3,
        p_lower: vec![
            Triplet::new(0, 0, 2.0),
            Triplet::new(1, 1, 2.0),
            Triplet::new(2, 2, 2.0),
        ],
        c: vec![-6.0, -4.0, 0.0],
        a: vec![
            Triplet::new(0, 0, 1.0),
            Triplet::new(0, 1, 1.0),
            Triplet::new(0, 2, 1.0),
        ],
        b: vec![1.0],
        g: vec![],
        h: vec![],
        lb: vec![],
        ub: vec![],
    };
    let sol = solve(&prob, HessianInertia::Psd);
    assert_eq!(sol.status, QpStatus::Optimal, "x = {:?}", sol.x);
    // ½·2·‖x‖² − 6x₀ − 4x₁ subject to Σx = 1. Stationarity gives
    // x = ((6+μ)/2, (4+μ)/2, μ/2) with Σx = 1 ⇒ μ = −8/3.
    let mu = -8.0 / 3.0;
    let want = [(6.0 + mu) / 2.0, (4.0 + mu) / 2.0, mu / 2.0];
    for (i, w) in want.iter().enumerate() {
        assert!(
            (sol.x[i] - w).abs() < 1e-7,
            "x[{i}] = {} want {w} (x = {:?})",
            sol.x[i],
            sol.x
        );
    }
}
