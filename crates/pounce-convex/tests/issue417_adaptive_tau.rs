//! gh #417 — the warm-start payoff was capped by a *static* fraction-to-
//! boundary parameter, not by the warm start itself.
//!
//! With `τ = 0.95` fixed, every accepted step covers at most 95% of the
//! distance to the cone boundary, so μ and the residuals fall by a fixed ~20×
//! per iteration once the direction stops being the limit. The iteration count
//! is then `log₂₀(μ₀/tol)` *no matter how good the starting point is*: a warm
//! start can only lower μ₀, buying a logarithm of the perturbation instead of
//! the one or two Newton steps a nearby problem deserves. Warm-started traces
//! showed `α_p = α_d = 0.950` exactly, from the second iteration to the last.
//!
//! The direct driver now takes the standard Mehrotra tail on its corrector
//! step — `τ = clamp(1 − μ, tau, tau_max)` — but **only on orthant blocks**:
//! driving τ → 1 on a second-order or PSD block puts the iterate on a curved
//! boundary its Nesterov–Todd scaling cannot survive, and costs the direct
//! driver most of the SOC instances it solves. `QpOptions::tau_max == tau`
//! restores the old static behaviour, which is what these tests compare
//! against.

use pounce_convex::{
    QpOptions, QpProblem, QpStatus, QpWarmStart, Triplet, solve_qp_ipm, solve_qp_ipm_warm,
};
use pounce_feral::FeralSolverInterface;
use pounce_linsol::SparseSymLinearSolverInterface;

fn backend() -> Box<dyn SparseSymLinearSolverInterface> {
    Box::new(FeralSolverInterface::new())
}

/// The pre-#417 behaviour: a flat τ for every step and every cone.
fn static_tau() -> QpOptions {
    QpOptions {
        tau_max: QpOptions::default().tau,
        ..QpOptions::default()
    }
}

/// `min ½‖x − θ‖² s.t. Σx = 1, x ≥ 0` — projection onto the simplex, the
/// family the issue traced.
fn simplex_proj(theta: f64, n: usize) -> QpProblem {
    QpProblem {
        n,
        p_lower: (0..n).map(|i| Triplet::new(i, i, 1.0)).collect(),
        c: (0..n)
            .map(|i| -(0.3 + 0.1 * (i as f64 + theta).sin()))
            .collect(),
        a: (0..n).map(|i| Triplet::new(0, i, 1.0)).collect(),
        b: vec![1.0],
        g: vec![],
        h: vec![],
        lb: vec![0.0; n],
        ub: vec![],
    }
}

/// `min ½·2‖x‖² + c(θ)ᵀx s.t. Σx ≤ cap(θ), 0 ≤ x ≤ u(θ)` — a QP whose active
/// set shifts as the bound moves.
fn moving_bound_qp(theta: f64, n: usize) -> QpProblem {
    QpProblem {
        n,
        p_lower: (0..n).map(|i| Triplet::new(i, i, 2.0)).collect(),
        c: (0..n)
            .map(|i| -1.0 - 0.1 * i as f64 + 0.3 * theta * (i as f64).cos())
            .collect(),
        a: vec![],
        b: vec![],
        g: (0..n).map(|i| Triplet::new(0, i, 1.0)).collect(),
        h: vec![5.0 + theta],
        lb: vec![0.0; n],
        ub: (0..n)
            .map(|i| 1.0 + 0.05 * i as f64 + 0.1 * theta)
            .collect(),
    }
}

/// A corner where more constraints hold with equality than the dimension
/// needs — nested partial sums `Σ_{i≤k} xᵢ ≤ cap_k(θ)`, all tight at the
/// optimum. Strict complementarity fails here, the classic hard case for a
/// fraction-to-boundary step.
fn degenerate_corner(theta: f64, n: usize) -> QpProblem {
    let mut g = Vec::new();
    let mut h = Vec::new();
    for k in 0..n {
        for i in 0..=k {
            g.push(Triplet::new(k, i, 1.0));
        }
        h.push(0.1 * (k + 1) as f64 + 0.01 * theta);
    }
    QpProblem {
        n,
        p_lower: (0..n).map(|i| Triplet::new(i, i, 1.0)).collect(),
        c: (0..n)
            .map(|i| -1.0 - 0.05 * i as f64 - 0.02 * theta)
            .collect(),
        a: vec![],
        b: vec![],
        g,
        h,
        lb: vec![0.0; n],
        ub: vec![],
    }
}

/// Walk `theta` in `steps` increments of `scale`, warm-starting each solve
/// from the previous one. Returns the total warm iteration count, having
/// checked every warm solve against the cold solve of the same problem.
fn warm_sequence<F>(build: F, scale: f64, steps: usize, opts: &QpOptions) -> usize
where
    F: Fn(f64) -> QpProblem,
{
    let mut prev = solve_qp_ipm(&build(0.0), opts, backend);
    assert_eq!(prev.status, QpStatus::Optimal, "cold seed solve");
    let mut total = 0;
    for k in 1..=steps {
        let prob = build(scale * k as f64);
        let cold = solve_qp_ipm(&prob, opts, backend);
        let warm = solve_qp_ipm_warm(&prob, opts, &QpWarmStart::from_solution(&prev), backend);
        assert_eq!(cold.status, QpStatus::Optimal, "cold solve at step {k}");
        assert_eq!(warm.status, QpStatus::Optimal, "warm solve at step {k}");
        // The start cannot change the KKT point: a faster warm solve is only
        // worth having if it is the *same* answer.
        assert!(
            (warm.obj - cold.obj).abs() / (1.0 + cold.obj.abs()) < 1e-6,
            "step {k}: warm obj {} vs cold {}",
            warm.obj,
            cold.obj
        );
        // Both solves stop at an *interior* point within `tol` of the same
        // optimum, reached along different central paths, so a component
        // pinned at a bound differs between them at exactly the scale the
        // barrier leaves it (~1e-4 for a nearly-degenerate one). Pointwise
        // primal agreement is therefore the wrong thing to assert; what the
        // solve actually promises is a small KKT error, so check that.
        let kkt = warm.kkt_residuals(&prob).kkt_error();
        assert!(
            kkt < 1e-6,
            "step {k}: warm KKT error {kkt:e} — the faster step must not cost \
             optimality"
        );
        total += warm.iters;
        prev = warm;
    }
    total
}

/// The headline: over a path of nearby QPs, the Mehrotra tail cuts warm
/// iterations substantially versus the static τ, on every perturbation size.
#[test]
fn adaptive_tau_cuts_warm_iterations_on_nearby_qps() {
    let steps = 20;
    for scale in [0.01, 0.05, 0.2] {
        for (name, build) in [
            (
                "simplex_proj",
                &(|t| simplex_proj(t, 12)) as &dyn Fn(f64) -> QpProblem,
            ),
            (
                "moving_bound_qp",
                &(|t| moving_bound_qp(t, 15)) as &dyn Fn(f64) -> QpProblem,
            ),
            (
                "degenerate_corner",
                &(|t| degenerate_corner(t, 10)) as &dyn Fn(f64) -> QpProblem,
            ),
        ] {
            let old = warm_sequence(build, scale, steps, &static_tau());
            let new = warm_sequence(build, scale, steps, &QpOptions::default());
            // Measured at ~50–65% fewer; asserted at 20% to leave room for
            // linear-solver and platform noise while still failing outright
            // if the tail is ever silently disabled.
            assert!(
                (new as f64) < 0.8 * old as f64,
                "{name} (scale {scale}): adaptive τ took {new} warm iterations \
                 over {steps} steps, static τ took {old} — expected a clear cut"
            );
        }
    }
}

/// Setting `tau_max = tau` is the documented escape hatch back to the static
/// rule, and it must reach the same optimum — the knob trades iterations for
/// conservatism, never correctness.
#[test]
fn tau_max_equal_to_tau_reproduces_the_static_solve() {
    let prob = moving_bound_qp(0.7, 15);
    let adaptive = solve_qp_ipm(&prob, &QpOptions::default(), backend);
    let static_ = solve_qp_ipm(&prob, &static_tau(), backend);
    assert_eq!(adaptive.status, QpStatus::Optimal);
    assert_eq!(static_.status, QpStatus::Optimal);
    assert!((adaptive.obj - static_.obj).abs() / (1.0 + static_.obj.abs()) < 1e-8);
    for i in 0..prob.n {
        assert!(
            (adaptive.x[i] - static_.x[i]).abs() < 1e-6,
            "x[{i}]: adaptive {} vs static {}",
            adaptive.x[i],
            static_.x[i]
        );
    }
}

/// The trap the issue documents: an *unrestricted* τ → 1 breaks second-order
/// cones — the direct driver loses ~60% of the SOC instances it solves. The
/// rule is scoped to orthant blocks, so a **mixed** problem (one SOC block
/// plus orthant rows, where both τ's are live in the same solve) still lands
/// on the cold answer. `conic_hsde_vs_direct::second_order_cones_agree_across_drivers`
/// is the broad sweep this pins a warm, mixed-cone case of.
#[test]
fn mixed_soc_and_orthant_warm_solves_are_unaffected() {
    use pounce_convex::{ConeSpec, solve_socp_ipm, solve_socp_ipm_warm};

    // min ½‖x‖² + c(θ)ᵀx  s.t.  x ∈ SOC₃,  x₁ ≤ 1.2,  x₂ ≥ −0.4.
    let socp = |theta: f64| QpProblem {
        n: 3,
        p_lower: (0..3).map(|i| Triplet::new(i, i, 1.0)).collect(),
        c: vec![-1.0 - theta, -2.0 + 0.5 * theta, 0.1 * theta],
        a: vec![],
        b: vec![],
        g: vec![
            // Rows 0–2: s = −Gx = x must lie in SOC₃.
            Triplet::new(0, 0, -1.0),
            Triplet::new(1, 1, -1.0),
            Triplet::new(2, 2, -1.0),
            // Rows 3–4: an orthant block, which is where the tail applies.
            Triplet::new(3, 1, 1.0),
            Triplet::new(4, 2, -1.0),
        ],
        h: vec![0.0, 0.0, 0.0, 1.2, 0.4],
        lb: vec![],
        ub: vec![],
    };
    let cones = [ConeSpec::SecondOrder(3), ConeSpec::Nonneg(2)];
    let opts = QpOptions::default();

    let base = solve_socp_ipm(&socp(0.0), &cones, &opts, backend);
    assert_eq!(base.status, QpStatus::Optimal);
    for theta in [0.05, 0.2] {
        let prob = socp(theta);
        let cold = solve_socp_ipm(&prob, &cones, &opts, backend);
        let warm = solve_socp_ipm_warm(
            &prob,
            &cones,
            &QpWarmStart::from_solution(&base),
            &opts,
            backend,
        );
        assert_eq!(cold.status, QpStatus::Optimal, "cold SOC solve @ {theta}");
        assert!(
            matches!(warm.status, QpStatus::Optimal | QpStatus::OptimalInaccurate),
            "warm SOC solve @ {theta}: {:?}",
            warm.status
        );
        assert!(
            (warm.obj - cold.obj).abs() / (1.0 + cold.obj.abs()) < 1e-6,
            "SOC @ {theta}: warm obj {} vs cold {}",
            warm.obj,
            cold.obj
        );
        for i in 0..prob.n {
            assert!(
                (warm.x[i] - cold.x[i]).abs() < 1e-3,
                "SOC @ {theta}, x[{i}]: warm {} vs cold {}",
                warm.x[i],
                cold.x[i]
            );
        }
    }
}
