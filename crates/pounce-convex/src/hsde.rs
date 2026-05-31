//! Homogeneous self-dual embedding (HSDE) driver for the convex IPM.
//!
//! This is the foundation for Clarabel cone parity (see
//! `dev-notes/hsde.md` and `dev-notes/clarabel-parity.md`). It reformulates
//! the interior-point iteration as a *single self-dual system* in the
//! embedded variables `(x, y, z, s, τ, κ)`, so that
//!
//! - a self-starting iterate handles primal- and dual-infeasible problems
//!   uniformly (no infeasible start), and
//! - infeasibility/unboundedness falls out of the embedding (`τ → 0`,
//!   `κ > 0`) rather than from a bolt-on certificate watch.
//!
//! **The per-cone math and the KKT factorization are reused verbatim.** The
//! embedding borders the existing symmetric `(x, y, z)` block `M`
//! (assembled by [`crate::ipm::KktStructure`], with each cone's NT scaling
//! `W²` from [`Cone::kkt_block`]) by the scalar `τ`, and solves it with
//! **two** back-solves through the *same* factorization plus a scalar Schur
//! complement (the SCS/ECOS scheme): `M p = (−c, b, h)` (the constant
//! direction) and `M q = residual`, combined with `Δτ` from the τ/κ row.
//!
//! ## Scope (Phase H2)
//!
//! This driver implements the **linear-conic** embedding (`P = 0`) over a
//! product of nonnegative-orthant and second-order cones — it solves LPs
//! and SOCPs. The quadratic-objective τ-row (the `xᵀPx/τ` coupling, Phase
//! H3) and the switch-over to make HSDE the default (Phase H4) follow; for
//! now `solve_qp_ipm`/`solve_socp_ipm` remain the production path and this
//! module is validated to reproduce their optima and certificates.

use crate::cones::{CompositeCone, Cone};
use crate::ipm::{
    build_factorization, build_rhs, detect_infeasibility, dot, inf_norm, split_step, QpOptions,
};
use crate::qp::{QpProblem, QpSolution, QpStatus};
use pounce_linsol::SparseSymLinearSolverInterface;

/// Fraction-to-boundary step for a positive scalar ray `v + α dv > 0`,
/// scaled by `tau` and capped at 1 (the scalar analogue of `Cone::max_step`
/// for the homogenizing variables `τ`, `κ`).
// Phase H2: the embedding is validated against the direct driver (see the
// test module) but not yet wired into a public entry point — that is Phase
// H4 (make HSDE the default). Until then these are exercised only by tests.
#[cfg_attr(not(test), allow(dead_code))]
fn ray_step(v: f64, dv: f64, tau: f64) -> f64 {
    if dv < 0.0 {
        (tau * (-v / dv)).min(1.0)
    } else {
        1.0
    }
}

/// Solve `min cᵀx s.t. Ax = b, Gx ⪯_K h` (linear objective, `P = 0`) via the
/// homogeneous self-dual embedding, returning the un-homogenized solution.
///
/// `cone` is the product cone `K` over the `m_ineq` inequality rows (built
/// by the caller exactly as for [`crate::ipm::solve_socp_ipm`]). Variable
/// bounds must already be expanded into `cone` rows by the caller.
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn solve_conic_hsde<F>(
    prob: &QpProblem,
    cone: &CompositeCone,
    opts: &QpOptions,
    mut make_backend: F,
) -> QpSolution
where
    F: FnMut() -> Box<dyn SparseSymLinearSolverInterface>,
{
    debug_assert!(
        prob.p_lower.is_empty(),
        "solve_conic_hsde is the linear-conic (P=0) embedding; quadratic is Phase H3"
    );

    let n = prob.n;
    let m_eq = prob.m_eq();
    let m_ineq = prob.m_ineq();
    let degree = cone.degree();

    let (kkt, mut fact) = match build_factorization(prob, cone, opts, &mut make_backend) {
        Ok(pair) => pair,
        Err(()) => return failed(prob),
    };

    // Constant border data: −b, −h (so `build_rhs` yields the `(−c, b, h)`
    // right-hand side of the constant direction `p`).
    let neg_b: Vec<f64> = prob.b.iter().map(|v| -v).collect();
    let neg_h: Vec<f64> = prob.h.iter().map(|v| -v).collect();
    let zeros_m = vec![0.0; m_ineq];

    // Self-dual start: x = y = 0, s = z = e (cone identity), τ = κ = 1.
    let mut x = vec![0.0; n];
    let mut y = vec![0.0; m_eq];
    let mut e = vec![0.0; m_ineq];
    cone.identity(&mut e);
    let mut s = e.clone();
    let mut z = e;
    let mut tau = 1.0_f64;
    let mut kappa = 1.0_f64;

    // Residual + work buffers.
    let mut rho_x = vec![0.0; n];
    let mut rho_y = vec![0.0; m_eq];
    let mut rho_z = vec![0.0; m_ineq];
    let mut r_c = vec![0.0; m_ineq];
    let mut comp = vec![0.0; m_ineq];
    let mut kkt_vals = kkt.values.clone();
    let mut rhs = vec![0.0; kkt.dim];

    // Direction buffers: p = constant direction, (dx,dy,dz) = the running
    // step, with affine slack/dual kept for the Mehrotra corrector.
    let mut p_x = vec![0.0; n];
    let mut p_y = vec![0.0; m_eq];
    let mut p_z = vec![0.0; m_ineq];
    let mut dx = vec![0.0; n];
    let mut dy = vec![0.0; m_eq];
    let mut dz = vec![0.0; m_ineq];
    let mut ds = vec![0.0; m_ineq];
    let mut dz_aff = vec![0.0; m_ineq];
    let mut ds_aff = vec![0.0; m_ineq];

    let mut status = QpStatus::IterationLimit;
    let mut iters = 0;

    for it in 0..opts.max_iter {
        iters = it;

        // --- homogeneous residuals ---
        // ρ_x = Aᵀy + Gᵀz + c·τ
        for (r, &ci) in rho_x.iter_mut().zip(&prob.c) {
            *r = ci * tau;
        }
        prob.at_mul(&y, &mut rho_x);
        prob.gt_mul(&z, &mut rho_x);
        // ρ_y = A x − b·τ
        for (r, &bi) in rho_y.iter_mut().zip(&prob.b) {
            *r = -bi * tau;
        }
        prob.a_mul(&x, &mut rho_y);
        // ρ_z = G x + s − h·τ
        for i in 0..m_ineq {
            rho_z[i] = s[i] - prob.h[i] * tau;
        }
        prob.g_mul(&x, &mut rho_z);
        // ρ_τ = κ + cᵀx + bᵀy + hᵀz
        let ctx = dot(&prob.c, &x);
        let bty = dot(&prob.b, &y);
        let htz = dot(&prob.h, &z);
        let rho_tau = kappa + ctx + bty + htz;

        let sz = dot(&s, &z);
        let mu = (sz + tau * kappa) / (degree as f64 + 1.0);

        // --- convergence (un-homogenized residuals; divide out τ) ---
        let pres = inf_norm(&rho_y).max(inf_norm(&rho_z)) / tau;
        let dres = inf_norm(&rho_x) / tau;
        let gap = (ctx + bty + htz).abs() / tau;
        if pres < opts.tol && dres < opts.tol && gap < opts.tol {
            status = QpStatus::Optimal;
            break;
        }

        // --- infeasibility (the embedding drives the iterate onto the
        // Farkas/recession ray as τ → 0; the same verified relative checks
        // as the direct driver apply to the homogeneous (x, y, z)). ---
        if tau < 1e-2 * kappa.max(1.0) {
            if let Some(st) = detect_infeasibility(prob, &x, &y, &z, opts) {
                status = st;
                break;
            }
        }

        // --- refactor M with the current cone scaling ---
        kkt.update_blocks(cone, &s, &z, opts.reg, &mut kkt_vals);
        if fact.refactor(&kkt_vals).is_err() {
            status = QpStatus::NumericalFailure;
            break;
        }

        // --- constant direction p: M p = (−c, b, h) ---
        build_rhs(&prob.c, &neg_b, &neg_h, &zeros_m, n, m_eq, m_ineq, &mut rhs);
        if fact.solve_one(&mut rhs).is_err() {
            status = QpStatus::NumericalFailure;
            break;
        }
        split_step(&rhs, n, m_eq, m_ineq, &mut p_x, &mut p_y, &mut p_z);
        // gᵀp with g = (c, b, h), and the scalar Schur denominator.
        let gtp = dot(&prob.c, &p_x) + dot(&prob.b, &p_y) + dot(&prob.h, &p_z);
        let denom = gtp - kappa / tau;

        // === Predictor (affine, σ = 0) ===
        cone.comp_residual(&s, &z, 0.0, &mut r_c);
        cone.rhs_comp_term(&s, &z, &r_c, &mut comp);
        build_rhs(&rho_x, &rho_y, &rho_z, &comp, n, m_eq, m_ineq, &mut rhs);
        if fact.solve_one(&mut rhs).is_err() {
            status = QpStatus::NumericalFailure;
            break;
        }
        split_step(&rhs, n, m_eq, m_ineq, &mut dx, &mut dy, &mut dz);
        let gtq = dot(&prob.c, &dx) + dot(&prob.b, &dy) + dot(&prob.h, &dz);
        // Δτ = [−ρ_τ − gᵀq − (σμ − τκ)/τ] / (gᵀp − κ/τ); predictor σμ = 0,
        // so −(0 − τκ)/τ = +κ.
        let dtau_aff = (-rho_tau - gtq + kappa) / denom;
        // Full affine directions dw = q + Δτ·p (only dz needed downstream).
        for i in 0..m_ineq {
            dz_aff[i] = dz[i] + dtau_aff * p_z[i];
        }
        let dkappa_aff = (-tau * kappa - kappa * dtau_aff) / tau;
        cone.recover_ds(&s, &z, &r_c, &dz_aff, &mut ds_aff);

        // Affine step length over the cone and the τ/κ rays.
        let mut alpha_aff = ray_step(tau, dtau_aff, opts.tau).min(ray_step(kappa, dkappa_aff, opts.tau));
        if m_ineq > 0 {
            alpha_aff = alpha_aff
                .min(cone.max_step(&s, &ds_aff, opts.tau))
                .min(cone.max_step(&z, &dz_aff, opts.tau));
        }
        // μ_aff and Mehrotra centering σ = (μ_aff/μ)³.
        let mut dot_aff = (tau + alpha_aff * dtau_aff) * (kappa + alpha_aff * dkappa_aff);
        for i in 0..m_ineq {
            dot_aff += (s[i] + alpha_aff * ds_aff[i]) * (z[i] + alpha_aff * dz_aff[i]);
        }
        let mu_aff = dot_aff / (degree as f64 + 1.0);
        let sigma = if mu > 0.0 { (mu_aff / mu).powi(3) } else { 0.0 };
        let sigma_mu = sigma * mu;

        // === Corrector (centered target + second-order term) ===
        cone.comp_residual_corrector(&s, &z, &ds_aff, &dz_aff, sigma_mu, &mut r_c);
        cone.rhs_comp_term(&s, &z, &r_c, &mut comp);
        build_rhs(&rho_x, &rho_y, &rho_z, &comp, n, m_eq, m_ineq, &mut rhs);
        if fact.solve_one(&mut rhs).is_err() {
            status = QpStatus::NumericalFailure;
            break;
        }
        split_step(&rhs, n, m_eq, m_ineq, &mut dx, &mut dy, &mut dz);
        let gtq = dot(&prob.c, &dx) + dot(&prob.b, &dy) + dot(&prob.h, &dz);
        // τκ corrector residual: τκ + Δτ_aff·Δκ_aff (target σμ).
        let r_tk = tau * kappa + dtau_aff * dkappa_aff;
        let dtau = (-rho_tau - gtq - (sigma_mu - r_tk) / tau) / denom;
        // Combine: dw = q + Δτ·p.
        for i in 0..n {
            dx[i] += dtau * p_x[i];
        }
        for i in 0..m_eq {
            dy[i] += dtau * p_y[i];
        }
        for i in 0..m_ineq {
            dz[i] += dtau * p_z[i];
        }
        let dkappa = (sigma_mu - r_tk - kappa * dtau) / tau;
        cone.recover_ds(&s, &z, &r_c, &dz, &mut ds);

        // Single fraction-to-boundary step (HSDE is primal/dual-symmetric).
        let mut alpha = ray_step(tau, dtau, opts.tau).min(ray_step(kappa, dkappa, opts.tau));
        if m_ineq > 0 {
            alpha = alpha
                .min(cone.max_step(&s, &ds, opts.tau))
                .min(cone.max_step(&z, &dz, opts.tau));
        }

        for i in 0..n {
            x[i] += alpha * dx[i];
        }
        for i in 0..m_eq {
            y[i] += alpha * dy[i];
        }
        for i in 0..m_ineq {
            s[i] += alpha * ds[i];
            z[i] += alpha * dz[i];
        }
        tau += alpha * dtau;
        kappa += alpha * dkappa;
    }

    // Un-homogenize: divide by τ to recover the original-space solution.
    let inv = if tau.abs() > 0.0 { 1.0 / tau } else { 1.0 };
    let x: Vec<f64> = x.iter().map(|v| v * inv).collect();
    let y: Vec<f64> = y.iter().map(|v| v * inv).collect();
    let z: Vec<f64> = z.iter().map(|v| v * inv).collect();
    let obj = dot(&prob.c, &x); // P = 0.

    QpSolution {
        status,
        x,
        y,
        z,
        z_lb: vec![0.0; n],
        z_ub: vec![0.0; n],
        obj,
        iters,
    }
}

#[cfg_attr(not(test), allow(dead_code))]
fn failed(prob: &QpProblem) -> QpSolution {
    QpSolution {
        status: QpStatus::NumericalFailure,
        x: vec![0.0; prob.n],
        y: vec![0.0; prob.m_eq()],
        z: vec![1.0; prob.m_ineq()],
        z_lb: vec![0.0; prob.n],
        z_ub: vec![0.0; prob.n],
        obj: 0.0,
        iters: 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cones::ConeSpec;
    use crate::ipm::solve_socp_ipm;
    use crate::qp::{QpProblem, Triplet};
    use pounce_feral::FeralSolverInterface;
    use pounce_linsol::SparseSymLinearSolverInterface;

    fn backend() -> Box<dyn SparseSymLinearSolverInterface> {
        Box::new(FeralSolverInterface::new())
    }

    fn opts() -> QpOptions {
        QpOptions {
            max_iter: 200,
            ..QpOptions::default()
        }
    }

    /// Solve the same (P=0) problem with the HSDE driver and the direct
    /// driver; assert both converge and agree on the primal.
    fn assert_agrees(prob: &QpProblem, specs: &[ConeSpec], tol: f64) -> QpSolution {
        let cone = CompositeCone::from_specs(specs);
        let hsde = solve_conic_hsde(prob, &cone, &opts(), backend);
        let direct = solve_socp_ipm(prob, specs, &opts(), backend);
        assert_eq!(hsde.status, QpStatus::Optimal, "HSDE not optimal");
        assert_eq!(direct.status, QpStatus::Optimal, "direct not optimal");
        assert_eq!(hsde.x.len(), direct.x.len());
        for i in 0..hsde.x.len() {
            assert!(
                (hsde.x[i] - direct.x[i]).abs() < tol,
                "x[{i}] HSDE {} vs direct {}",
                hsde.x[i],
                direct.x[i]
            );
        }
        hsde
    }

    /// LP with one inequality and a known vertex optimum.
    /// min −x0 − x1 s.t. x0+x1 ≤ 1, x ≥ 0  → obj −1 on the face x0+x1=1.
    #[test]
    fn lp_inequality_matches_direct() {
        // rows: x0+x1 ≤ 1 ; −x0 ≤ 0 ; −x1 ≤ 0  (all nonneg slacks)
        let prob = QpProblem {
            n: 2,
            p_lower: vec![],
            c: vec![-1.0, -1.0],
            a: vec![],
            b: vec![],
            g: vec![
                Triplet::new(0, 0, 1.0),
                Triplet::new(0, 1, 1.0),
                Triplet::new(1, 0, -1.0),
                Triplet::new(2, 1, -1.0),
            ],
            h: vec![1.0, 0.0, 0.0],
            lb: vec![],
            ub: vec![],
        };
        let sol = assert_agrees(&prob, &[ConeSpec::Nonneg(3)], 1e-6);
        assert!((sol.obj + 1.0).abs() < 1e-6, "obj {}", sol.obj);
        assert!((sol.x[0] + sol.x[1] - 1.0).abs() < 1e-6);
    }

    /// LP with an equality constraint: min cᵀx s.t. 1ᵀx = 1, x ≥ 0.
    /// min x0 + 2x1 s.t. x0+x1=1, x≥0  → x=(1,0), obj 1.
    #[test]
    fn lp_equality_matches_direct() {
        let prob = QpProblem {
            n: 2,
            p_lower: vec![],
            c: vec![1.0, 2.0],
            a: vec![Triplet::new(0, 0, 1.0), Triplet::new(0, 1, 1.0)],
            b: vec![1.0],
            g: vec![Triplet::new(0, 0, -1.0), Triplet::new(1, 1, -1.0)],
            h: vec![0.0, 0.0],
            lb: vec![],
            ub: vec![],
        };
        let sol = assert_agrees(&prob, &[ConeSpec::Nonneg(2)], 1e-6);
        assert!((sol.obj - 1.0).abs() < 1e-5, "obj {}", sol.obj);
        assert!(sol.x[0] > 0.99 && sol.x[1] < 1e-4, "x {:?}", sol.x);
    }

    /// SOCP norm minimization: min t s.t. (t, x−a) ∈ SOC(3).
    /// With G=−I, h=(0,−a0,−a1): optimum t=0, x=a.
    #[test]
    fn socp_norm_min_matches_direct() {
        let a = [2.0_f64, -1.0];
        let prob = QpProblem {
            n: 3,
            p_lower: vec![],
            c: vec![1.0, 0.0, 0.0],
            a: vec![],
            b: vec![],
            g: vec![
                Triplet::new(0, 0, -1.0),
                Triplet::new(1, 1, -1.0),
                Triplet::new(2, 2, -1.0),
            ],
            h: vec![0.0, -a[0], -a[1]],
            lb: vec![],
            ub: vec![],
        };
        let sol = assert_agrees(&prob, &[ConeSpec::SecondOrder(3)], 1e-5);
        assert!(sol.x[0].abs() < 1e-5, "t {}", sol.x[0]);
        assert!((sol.x[1] - a[0]).abs() < 1e-5 && (sol.x[2] - a[1]).abs() < 1e-5);
    }

    /// Mixed cone: a nonneg row and a second-order block together.
    /// min −x1 s.t. x1 ≤ 0.5 (nonneg), ‖x‖ ≤ 1 (soc (1,x0,x1)).
    #[test]
    fn socp_mixed_matches_direct() {
        let prob = QpProblem {
            n: 2,
            p_lower: vec![],
            c: vec![0.0, -1.0],
            a: vec![],
            b: vec![],
            g: vec![
                Triplet::new(0, 1, 1.0),  // nonneg: 0.5 − x1 ≥ 0
                Triplet::new(2, 0, -1.0), // soc s1 = x0
                Triplet::new(3, 1, -1.0), // soc s2 = x1
            ],
            h: vec![0.5, 1.0, 0.0, 0.0],
            lb: vec![],
            ub: vec![],
        };
        assert_agrees(&prob, &[ConeSpec::Nonneg(1), ConeSpec::SecondOrder(3)], 1e-5);
    }

    /// Primal-infeasible LP: x ≥ 2 and x ≤ 1.
    #[test]
    fn detects_primal_infeasible() {
        let prob = QpProblem {
            n: 1,
            p_lower: vec![],
            c: vec![1.0],
            a: vec![],
            b: vec![],
            g: vec![Triplet::new(0, 0, -1.0), Triplet::new(1, 0, 1.0)],
            h: vec![-2.0, 1.0], // −x ≤ −2 (x≥2) ; x ≤ 1
            lb: vec![],
            ub: vec![],
        };
        let cone = CompositeCone::from_specs(&[ConeSpec::Nonneg(2)]);
        let sol = solve_conic_hsde(&prob, &cone, &opts(), backend);
        assert_eq!(sol.status, QpStatus::PrimalInfeasible);
    }

    /// Dual-infeasible / unbounded LP: min −x s.t. x ≥ 0 (no upper bound).
    #[test]
    fn detects_dual_infeasible() {
        let prob = QpProblem {
            n: 1,
            p_lower: vec![],
            c: vec![-1.0],
            a: vec![],
            b: vec![],
            g: vec![Triplet::new(0, 0, -1.0)],
            h: vec![0.0],
            lb: vec![],
            ub: vec![],
        };
        let cone = CompositeCone::from_specs(&[ConeSpec::Nonneg(1)]);
        let sol = solve_conic_hsde(&prob, &cone, &opts(), backend);
        assert_eq!(sol.status, QpStatus::DualInfeasible);
    }
}
