//! gh #641 regression: the parametric active-set QP driver reported
//! `OptimalInaccurate` / `NumericalFailure` on points that are *exactly*
//! optimal, once the problem data was merely large.
//!
//! Its post-loop adjudication compared the raw, unnormalized KKT error —
//! `max(primal, dual, complementarity)` — against the absolute `tol`. The
//! complementarity term `max|zᵢsᵢ|` carries the data magnitude twice over, so
//! its finite-precision floor is `≈ ‖z‖·‖data‖·ε`; on a QP with `‖data‖ ≳ 1e9`
//! that floor sits *above* the default `tol = 1e-8` and no iterate, however
//! exact, can reach the tolerance. A machine-precision-exact answer was
//! therefore labelled a non-convergence, while the convex IPM — the less
//! accurate engine on the same instance — reported a clean `Optimal`.
//!
//! This is the active-set analogue of gh #336: the same mechanism, on the
//! non-symmetric HSDE driver, fixed by #337 making *its* post-loop adjudication
//! scale-relative. This path was not covered by that fix.
//!
//! The fix must not go further than that. These tests pin both halves: the
//! large-data answers are certified, and the well-scaled end of every sweep
//! still has to earn `Optimal` on the tight absolute test.

use pounce_convex::{
    ActiveSetOverrides, QpOptions, QpProblem, QpSolution, QpStatus, Triplet, solve_qp_active_set,
    solve_qp_ipm,
};
use pounce_feral::FeralSolverInterface;
use pounce_linsol::SparseSymLinearSolverInterface;

fn mk() -> Box<dyn SparseSymLinearSolverInterface> {
    Box::new(FeralSolverInterface::new())
}

fn active_set(prob: &QpProblem, opts: &QpOptions) -> QpSolution {
    solve_qp_active_set(prob, opts, &ActiveSetOverrides::default(), &mut mk)
}

/// The issue's minimal case: `min ½K‖x‖² − K(x₀+x₁)  s.t.  x₀ + x₁ ≤ 1`.
///
/// `cond(P) = 1` — this is not an ill-conditioned problem, only a large one.
/// The row binds and by symmetry `x* = (½, ½)`, `obj* = −0.75K`.
fn scaled_projection_qp(k: f64) -> QpProblem {
    QpProblem {
        n: 2,
        p_lower: vec![Triplet::new(0, 0, k), Triplet::new(1, 1, k)],
        c: vec![-k, -k],
        a: vec![],
        b: vec![],
        g: vec![Triplet::new(0, 0, 1.0), Triplet::new(0, 1, 1.0)],
        h: vec![1.0],
        lb: vec![],
        ub: vec![],
    }
}

/// The reported instance, `K = 1e9`: `success=False` / `optimal_inaccurate` at
/// the exact optimum. The answer was never in question — assert it is still the
/// closed-form optimum to machine precision, and that the *label* now says so.
#[test]
fn exact_optimum_at_large_scale_is_optimal() {
    let k = 1e9;
    let prob = scaled_projection_qp(k);
    let sol = active_set(&prob, &QpOptions::default());

    assert_eq!(
        sol.status,
        QpStatus::Optimal,
        "kkt_error is not the measure"
    );
    assert!(
        (sol.x[0] - 0.5).abs() <= 1e-12 && (sol.x[1] - 0.5).abs() <= 1e-12,
        "x = {:?} must be the closed-form optimum (0.5, 0.5)",
        sol.x
    );
    let obj_star = -0.75 * k;
    assert!(
        (sol.obj - obj_star).abs() / obj_star.abs() <= 1e-12,
        "obj {} must be {obj_star}",
        sol.obj
    );
    // The two engines must not disagree about the same instance — least of all
    // with the failure verdict coming from the more accurate one.
    assert_eq!(
        solve_qp_ipm(&prob, &QpOptions::default(), mk).status,
        sol.status,
        "the IPM and active-set drivers must agree on this instance"
    );
}

/// The scale sweep from the issue. The answer is exact throughout; only the
/// label used to break, at `K = 1e9`. The small-`K` end is pinned to `Optimal`
/// on its own merits: there the absolute test is reachable and must still
/// govern, so this guards against the fix being over-applied.
#[test]
fn scale_sweep_never_reports_a_non_solve() {
    for kexp in [4, 6, 7, 8, 9, 10, 11, 12] {
        let k = 10f64.powi(kexp);
        let prob = scaled_projection_qp(k);
        let sol = active_set(&prob, &QpOptions::default());
        assert_eq!(
            sol.status,
            QpStatus::Optimal,
            "K=1e{kexp}: exact optimum reported {:?}",
            sol.status
        );
        assert!(
            (sol.x[0] - 0.5).abs() <= 1e-12 && (sol.x[1] - 0.5).abs() <= 1e-12,
            "K=1e{kexp}: x = {:?}",
            sol.x
        );
    }
}

/// Tightening `tol` used to make the label *worse* — `numerical_failure` —
/// while `kkt_error` did not move at all, which is the clearest statement that
/// the label was a threshold artifact rather than a measurement. A user asking
/// for more accuracy on a problem solved to machine precision must not be told
/// the solver broke.
#[test]
fn tight_tolerance_does_not_manufacture_a_numerical_failure() {
    let prob = scaled_projection_qp(1e10);
    for tol in [1e-8, 1e-9, 1e-10] {
        let opts = QpOptions {
            tol,
            ..QpOptions::default()
        };
        let sol = active_set(&prob, &opts);
        assert_ne!(
            sol.status,
            QpStatus::NumericalFailure,
            "tol={tol:.0e}: an exact answer must never be a numerical failure"
        );
        assert!(
            (sol.x[0] - 0.5).abs() <= 1e-12,
            "tol={tol:.0e}: x = {:?}",
            sol.x
        );
    }
}

/// A well-scaled QP whose engine really does stop short must still be demoted.
/// The scale-relative arm is gated on the absolute tolerance being unreachable,
/// so below that crossover nothing changes — pinned here end-to-end rather than
/// only on the internal adjudicator, since the gate is what keeps this fix from
/// becoming a blanket relaxation.
#[test]
fn well_scaled_solves_are_unchanged() {
    // `min (x₀−3)² + (x₁−2)² s.t. x₀ + x₁ ≤ 4`; x* = (2.5, 1.5), obj* = −12.5.
    let prob = QpProblem {
        n: 2,
        p_lower: vec![Triplet::new(0, 0, 2.0), Triplet::new(1, 1, 2.0)],
        c: vec![-6.0, -4.0],
        a: vec![],
        b: vec![],
        g: vec![Triplet::new(0, 0, 1.0), Triplet::new(0, 1, 1.0)],
        h: vec![4.0],
        lb: vec![],
        ub: vec![],
    };
    let sol = active_set(&prob, &QpOptions::default());
    assert_eq!(sol.status, QpStatus::Optimal);
    assert!((sol.x[0] - 2.5).abs() < 1e-8 && (sol.x[1] - 1.5).abs() < 1e-8);
    assert!(sol.kkt_residuals(&prob).kkt_error() <= QpOptions::default().tol);
}

/// A deterministic stand-in for the randomized instance the adversary job found
/// this on: `n = 10`, 2 equalities, 15 inequalities, ill-conditioned `P`, and
/// constraint rows scaled across six orders. Both engines are run on it and
/// must agree — on the answer *and* on the verdict.
#[test]
fn randomized_large_scale_instance_agrees_with_the_ipm() {
    let prob = random_qp();
    let sol = active_set(&prob, &QpOptions::default());
    let ipm = solve_qp_ipm(&prob, &QpOptions::default(), mk);

    assert_eq!(
        ipm.status,
        QpStatus::Optimal,
        "test setup: the IPM must solve this instance"
    );
    assert_eq!(
        sol.status,
        QpStatus::Optimal,
        "active-set reported {:?} on a point the IPM certifies",
        sol.status
    );
    let scale = ipm.obj.abs().max(1.0);
    assert!(
        (sol.obj - ipm.obj).abs() / scale <= 1e-8,
        "objectives disagree: active-set {} vs IPM {}",
        sol.obj,
        ipm.obj
    );
}

/// A 64-bit LCG, so the instance below is byte-identical on every platform.
struct Lcg(u64);

impl Lcg {
    /// Uniform on `[−1, 1)`.
    fn signed(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((self.0 >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
    }
}

fn random_qp() -> QpProblem {
    const N: usize = 10;
    const M_EQ: usize = 2;
    const M_INEQ: usize = 15;
    // Objective magnitude ~1e9 — the regime where the absolute complementarity
    // floor overtakes `tol`.
    const OBJ_SCALE: f64 = 1e9;

    let mut rng = Lcg(1_786_954_343);

    // `P = OBJ_SCALE · (BᵀB + εI)` with `B` dense random: symmetric positive
    // definite, and ill-conditioned (a Wishart-style spectrum) without being
    // singular.
    let b: Vec<Vec<f64>> = (0..N)
        .map(|_| (0..N).map(|_| rng.signed()).collect())
        .collect();
    let mut p_lower = Vec::new();
    for i in 0..N {
        for j in 0..=i {
            let mut v: f64 = (0..N).map(|r| b[r][i] * b[r][j]).sum();
            if i == j {
                v += 1e-3;
            }
            p_lower.push(Triplet::new(i, j, OBJ_SCALE * v));
        }
    }
    let c: Vec<f64> = (0..N).map(|_| OBJ_SCALE * rng.signed()).collect();

    // A reference interior point, so the constraint system is feasible by
    // construction and some inequalities bind at the optimum.
    let x0: Vec<f64> = (0..N).map(|_| rng.signed()).collect();

    let mut a = Vec::new();
    let mut bvec = vec![0.0; M_EQ];
    for i in 0..M_EQ {
        let row_scale = 10f64.powf(3.0 * rng.signed());
        for j in 0..N {
            let v = row_scale * rng.signed();
            a.push(Triplet::new(i, j, v));
            bvec[i] += v * x0[j];
        }
    }

    let mut g = Vec::new();
    let mut h = vec![0.0; M_INEQ];
    for i in 0..M_INEQ {
        let row_scale = 10f64.powf(3.0 * rng.signed());
        for j in 0..N {
            let v = row_scale * rng.signed();
            g.push(Triplet::new(i, j, v));
            h[i] += v * x0[j];
        }
        // Slack proportional to the row's own scale, so no row is trivially
        // slack and none is trivially binding.
        h[i] += row_scale * rng.signed().abs();
    }

    QpProblem {
        n: N,
        p_lower,
        c,
        a,
        b: bvec,
        g,
        h,
        lb: vec![],
        ub: vec![],
    }
}
