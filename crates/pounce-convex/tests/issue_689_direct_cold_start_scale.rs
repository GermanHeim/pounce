//! gh #689: the direct convex driver must start from a point sized to the
//! problem's data, not from `s = z = e`.
//!
//! The failure the fixed cold start removes is not slow convergence — it is
//! divergence. On a QP whose feasible set lives far from the origin, `s = e`
//! makes the fraction-to-boundary rule cut the *first* Newton step (a perfectly
//! good direction, pointed at the optimum) by the ratio `1 / ‖ds‖`. The iterate
//! cannot move; the corrector then divides `σμ` by slacks still pinned at `1`,
//! returns a direction many orders larger, and `z` runs away. On
//! `crates/pounce-cli/tests/fixtures/scaled_feasible_a.nl` that took the driver
//! to `kkt_error 8.4e45` at the iteration cap while HSDE solved the same model
//! in 16 iterations.
//!
//! These are the same shape, in the small: the box sits at `x ≈ 1e9`, so the
//! implied slacks at the origin are `~1e9` and a unit-scaled start is nine
//! orders too small. Both drivers must land on the same optimum, and the direct
//! one must do it in an ordinary iteration count.

use pounce_convex::{QpOptions, QpProblem, QpStatus, Triplet, solve_qp_ipm};
use pounce_feral::FeralSolverInterface;
use pounce_linsol::SparseSymLinearSolverInterface;

fn backend() -> Box<dyn SparseSymLinearSolverInterface> {
    Box::new(FeralSolverInterface::new())
}

/// `min ½‖x − t‖²` over the box `0 ≤ xᵢ ≤ 2·shift`, with `t = (shift, shift/2)`
/// strictly inside it. The unconstrained minimizer is feasible, so the optimum
/// is `x* = t` at objective `−½‖t‖²` whatever `shift` is — only the *scale* of
/// the feasible set moves with it.
fn shifted_box_qp(shift: f64) -> QpProblem {
    let target = [shift, 0.5 * shift];
    QpProblem {
        n: 2,
        p_lower: vec![Triplet::new(0, 0, 1.0), Triplet::new(1, 1, 1.0)],
        c: vec![-target[0], -target[1]],
        a: vec![],
        b: vec![],
        g: vec![
            Triplet::new(0, 0, 1.0),
            Triplet::new(1, 1, 1.0),
            Triplet::new(2, 0, -1.0),
            Triplet::new(3, 1, -1.0),
        ],
        h: vec![2.0 * shift, 2.0 * shift, 0.0, 0.0],
        lb: vec![],
        ub: vec![],
    }
}

fn direct_opts() -> QpOptions {
    QpOptions {
        // The direct infeasible-start driver — what `QpWarmStart`, the
        // build-once `QpFactorization` handle and the dual-infeasibility
        // reverify guard all use, and what `qp_hsde=no` selects.
        use_hsde: false,
        ..QpOptions::default()
    }
}

#[test]
fn direct_driver_converges_when_the_feasible_set_is_far_from_the_origin() {
    for &shift in &[1.0, 1e3, 1e6, 1e9] {
        let prob = shifted_box_qp(shift);
        let sol = solve_qp_ipm(&prob, &direct_opts(), backend);
        assert_eq!(
            sol.status,
            QpStatus::Optimal,
            "shift={shift:e}: direct driver did not converge ({:?} after {} iters)",
            sol.status,
            sol.iters
        );
        let want = [shift, 0.5 * shift];
        for (i, (&got, &w)) in sol.x.iter().zip(&want).enumerate() {
            assert!(
                (got - w).abs() <= 1e-6 * w.max(1.0),
                "shift={shift:e}: x[{i}] = {got:e}, want {w:e}"
            );
        }
    }
}

#[test]
fn iteration_count_does_not_grow_with_the_distance_to_the_feasible_set() {
    // With a unit-scaled cold start the count grew without bound with `shift`
    // (and past `~1e8` the solve diverged instead of converging), because every
    // iteration could only close a fixed *fraction* of a gap that starts nine
    // orders wide. Sizing the start to the data makes the count flat.
    let counts: Vec<usize> = [1.0, 1e3, 1e6, 1e9]
        .iter()
        .map(|&shift| {
            let sol = solve_qp_ipm(&shifted_box_qp(shift), &direct_opts(), backend);
            assert_eq!(sol.status, QpStatus::Optimal, "shift={shift:e}");
            sol.iters
        })
        .collect();
    assert!(
        counts.iter().all(|&c| c <= 20),
        "iteration count is not flat across nine orders of problem scale: {counts:?}"
    );
}
