//! Bound multipliers must survive the *scale* of the bound they sit on.
//!
//! Postsolve decides whether a variable is on a box bound by comparing `x` to
//! it. An interior-point solve stops a **relative** ~1e-8 short of a bound, so
//! at `x ≤ 5e5` it lands ~5e-3 away — and an absolute `1e-6` window reads that
//! as interior. Everything keyed on that verdict then concludes the bound is
//! slack and reports its multiplier as **zero**, while the status stays
//! `Optimal`: a silently wrong dual on a completely ordinary model.
//!
//! It reaches users two ways. `ipopt_zL_out` / `ipopt_zU_out` in the `.sol`
//! are these multipliers directly. And since the `.nl` extractor hands the
//! solver a native box, a constraint row that is really a bound is folded into
//! that box by presolve and gets its dual back *through* them — so the row's
//! own `.sol` multiplier vanishes too.
//!
//! Both cases are pinned here across four decades of bound magnitude, against
//! the no-presolve solve as the reference.

use pounce_convex::presolve::solve_with_presolve;
use pounce_convex::{QpOptions, QpProblem, QpSolution, QpStatus, Triplet, solve_qp_ipm};
use pounce_feral::FeralSolverInterface;
use pounce_linsol::SparseSymLinearSolverInterface;

fn backend() -> Box<dyn SparseSymLinearSolverInterface> {
    Box::new(FeralSolverInterface::new())
}

fn direct(prob: &QpProblem) -> QpSolution {
    solve_qp_ipm(prob, &QpOptions::default(), backend)
}

fn with_presolve(prob: &QpProblem) -> QpSolution {
    solve_with_presolve(prob, |r| solve_qp_ipm(r, &QpOptions::default(), backend))
}

/// Bound magnitudes spanning the range where an absolute window fails: the
/// first two pass either way, the rest only with a scaled one.
const SCALES: [f64; 5] = [3.0, 1.0e3, 1.0e4, 5.0e5, 1.0e7];

/// `min x² − 4u·x` over the **box** `0 ≤ x ≤ u` pins `x` at `u`, where the
/// upper bound carries multiplier `4u − 2u = 2u`. Presolve has nothing to
/// reduce here — this is the plain boxed-QP path every `.nl` model takes — so
/// its postsolve must hand back exactly what the direct solve reports.
#[test]
fn a_native_box_keeps_its_bound_multiplier_at_every_scale() {
    for u in SCALES {
        let prob = QpProblem {
            n: 1,
            p_lower: vec![Triplet::new(0, 0, 2.0)],
            c: vec![-4.0 * u],
            a: vec![],
            b: vec![],
            g: vec![],
            h: vec![],
            lb: vec![0.0],
            ub: vec![u],
        };
        let d = direct(&prob);
        let p = with_presolve(&prob);
        assert_eq!(p.status, QpStatus::Optimal, "u={u}");
        assert!((p.x[0] - u).abs() <= 1e-6 * u, "u={u}: x={}", p.x[0]);
        assert!(
            (p.z_ub[0] - 2.0 * u).abs() <= 1e-6 * u,
            "u={u}: presolve lost the bound multiplier: {} (expected {}, \
             direct solve reports {})",
            p.z_ub[0],
            2.0 * u,
            d.z_ub[0]
        );
        // Stationarity in the original problem: 2x − 4u + z_ub − z_lb = 0.
        let stat = 2.0 * p.x[0] - 4.0 * u + p.z_ub[0] - p.z_lb[0];
        assert!(stat.abs() <= 1e-6 * u, "u={u}: stationarity {stat}");
    }
}

/// The same bound written as a **row** — the shape a constraint that is really
/// a bound arrives in. Presolve reads it as a bound and folds it into the box,
/// so the row's multiplier comes back out of the box duals; if those are lost
/// to the scale, so is the row's `.sol` dual.
#[test]
fn a_bound_written_as_a_row_keeps_its_multiplier_at_every_scale() {
    for u in SCALES {
        let prob = QpProblem {
            n: 1,
            p_lower: vec![Triplet::new(0, 0, 2.0)],
            c: vec![-4.0 * u],
            a: vec![],
            b: vec![],
            g: vec![Triplet::new(0, 0, 1.0)],
            h: vec![u],
            lb: vec![],
            ub: vec![],
        };
        let d = direct(&prob);
        let p = with_presolve(&prob);
        assert_eq!(p.status, QpStatus::Optimal, "u={u}");
        assert!((p.x[0] - u).abs() <= 1e-6 * u, "u={u}: x={}", p.x[0]);
        assert!(
            (p.z[0] - 2.0 * u).abs() <= 1e-6 * u,
            "u={u}: the active row lost its multiplier: {} (expected {}, \
             direct solve reports {})",
            p.z[0],
            2.0 * u,
            d.z[0]
        );
        // The problem declares no box, so the whole force must be on the row.
        let stat = 2.0 * p.x[0] - 4.0 * u + p.z[0];
        assert!(stat.abs() <= 1e-6 * u, "u={u}: stationarity {stat}");
    }
}

/// A scaled bound must not *invent* a multiplier either: the widened window is
/// still far tighter than the distance an interior variable sits from its
/// bounds, and the reduced cost at an interior optimum is zero regardless.
#[test]
fn an_interior_optimum_reports_no_bound_multiplier_at_any_scale() {
    for u in SCALES {
        // min (x − u/2)² over 0 ≤ x ≤ u: the optimum is strictly interior.
        let prob = QpProblem {
            n: 1,
            p_lower: vec![Triplet::new(0, 0, 2.0)],
            c: vec![-u],
            a: vec![],
            b: vec![],
            g: vec![],
            h: vec![],
            lb: vec![0.0],
            ub: vec![u],
        };
        let p = with_presolve(&prob);
        assert_eq!(p.status, QpStatus::Optimal, "u={u}");
        assert!((p.x[0] - u / 2.0).abs() <= 1e-6 * u, "u={u}: x={}", p.x[0]);
        assert!(p.z_lb[0].abs() <= 1e-9 * u, "u={u}: z_lb={}", p.z_lb[0]);
        assert!(p.z_ub[0].abs() <= 1e-9 * u, "u={u}: z_ub={}", p.z_ub[0]);
    }
}
