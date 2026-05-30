//! Primal-dual interior-point driver for convex QP.
//!
//! Infeasible-start primal-dual path-following with **Mehrotra
//! predictor-corrector** (adaptive centering σ = (μ_aff/μ)³ plus the
//! second-order `Δs∘Δz` term) and fraction-to-boundary step control.
//! Predictor and corrector share one factorization per iteration. The
//! homogeneous self-dual embedding (for clean infeasibility detection
//! and a self-starting iterate) is the remaining Phase 3 piece and slots
//! into this same scaffolding.
//!
//! On bound/inequality-constrained convex QPs this reaches the solution
//! in materially fewer interior-point iterations than routing the same
//! problem through the NLP filter-IPM — see
//! `crates/pounce-cli/tests/qp_vs_nlp_iterations.rs` (≈41% fewer at
//! n=50), the check behind the plan's 30–50% claim.
//!
//! ## Method
//!
//! For the standard-form QP (see [`crate::qp`]) with slacks `s ≥ 0` on
//! the inequalities (`Gx + s = h`) and multipliers `y` (equality),
//! `z ≥ 0` (inequality), the KKT conditions are
//!
//! ```text
//!   P x + c + Aᵀ y + Gᵀ z = 0      (stationarity, r_d)
//!   A x − b              = 0       (r_p)
//!   G x + s − h          = 0       (r_g)
//!   s ∘ z                = 0       (complementarity)
//! ```
//!
//! Each iteration solves the symmetric indefinite Newton system
//!
//! ```text
//!   ⎡ P+δI   Aᵀ      Gᵀ        ⎤ ⎡dx⎤   ⎡ −r_d            ⎤
//!   ⎢ A      −δI     0         ⎥ ⎢dy⎥ = ⎢ −r_p            ⎥
//!   ⎣ G      0    −(S⊘Z)−δI    ⎦ ⎣dz⎦   ⎣ −r_g + r_c ⊘ z  ⎦
//! ```
//!
//! (with `ds` recovered from `dz`) through the shared
//! [`pounce_linsol::Factorization`]. The tiny static regularization `δ`
//! makes the system quasi-definite so the LDLᵀ has a well-defined
//! inertia; because convergence is tested on the *unregularized*
//! residuals, the fixed point is the true QP solution — `δ` only
//! perturbs the search direction.
//!
//! The cone-specific pieces (`μ`, the `S⊘Z` scaling diagonal, the
//! complementarity residual, `ds` recovery, and the fraction-to-boundary
//! step) all route through the [`Cone`](crate::cones::Cone) trait so
//! that Phases 4–6 extend rather than rewrite this driver.

use crate::cones::{Cone, NonnegCone};
use crate::qp::{QpProblem, QpSolution, QpStatus};
use pounce_common::types::{Index, Number};
use pounce_linsol::{Factorization, SparseSymLinearSolverInterface};
use std::collections::BTreeMap;

/// Options for the QP interior-point solve.
#[derive(Debug, Clone, Copy)]
pub struct QpOptions {
    /// Convergence tolerance on the max KKT residual and duality measure.
    pub tol: f64,
    /// Maximum iterations.
    pub max_iter: usize,
    /// Fraction-to-boundary parameter τ ∈ (0, 1). (The centering
    /// parameter σ is computed adaptively by the Mehrotra predictor;
    /// it is not an option.)
    pub tau: f64,
    /// Static KKT regularization δ.
    pub reg: f64,
}

impl Default for QpOptions {
    fn default() -> Self {
        QpOptions {
            tol: 1e-8,
            max_iter: 200,
            tau: 0.95,
            reg: 1e-8,
        }
    }
}

/// Solve a convex QP with the bare primal-dual IPM, using `backend` for
/// the augmented-system factorization. `make_backend` is called once per
/// iteration (the KKT pattern is rebuilt each step in this first
/// increment; constant-pattern symbolic reuse is a documented follow-up,
/// see `dev-notes/performance-engineering.md`).
pub fn solve_qp_ipm<F>(prob: &QpProblem, opts: &QpOptions, mut make_backend: F) -> QpSolution
where
    F: FnMut() -> Box<dyn SparseSymLinearSolverInterface>,
{
    let n = prob.n;
    let m_eq = prob.m_eq();
    let m_ineq = prob.m_ineq();
    let dim = n + m_eq + m_ineq;

    let cone = NonnegCone::new(m_ineq);

    // Infeasible-start iterate: x = 0, y = 0, s = z = 1. Strictly
    // interior for (s, z); primal/dual residuals are driven to zero.
    let mut x = vec![0.0; n];
    let mut y = vec![0.0; m_eq];
    let mut z = vec![1.0; m_ineq];
    let mut s = vec![1.0; m_ineq];

    // Scratch.
    let mut r_d = vec![0.0; n];
    let mut r_p = vec![0.0; m_eq];
    let mut r_g = vec![0.0; m_ineq];
    let mut r_c = vec![0.0; m_ineq];
    let mut scaling = vec![0.0; m_ineq];
    let mut rhs = vec![0.0; dim];
    let mut dx = vec![0.0; n];
    let mut dy = vec![0.0; m_eq];
    let mut dz = vec![0.0; m_ineq];
    let mut ds = vec![0.0; m_ineq];
    let mut ds_aff = vec![0.0; m_ineq];
    let mut dz_aff = vec![0.0; m_ineq];

    let mut iters = 0;
    let mut status = QpStatus::IterationLimit;

    for it in 0..opts.max_iter {
        iters = it;

        // --- residuals (unregularized; this is the convergence test) ---
        // r_d = P x + c + Aᵀ y + Gᵀ z
        r_d.iter_mut().zip(&prob.c).for_each(|(r, c)| *r = *c);
        prob.p_mul_add(&x, &mut r_d);
        prob.at_mul_add(&y, &mut r_d);
        prob.gt_mul_add(&z, &mut r_d);
        // r_p = A x − b
        r_p.iter_mut().zip(&prob.b).for_each(|(r, b)| *r = -*b);
        prob.a_mul_add(&x, &mut r_p);
        // r_g = G x + s − h
        for i in 0..m_ineq {
            r_g[i] = s[i] - prob.h[i];
        }
        prob.g_mul_add(&x, &mut r_g);

        let mu = cone.mu(&s, &z);
        let res = inf_norm(&r_d)
            .max(inf_norm(&r_p))
            .max(inf_norm(&r_g))
            .max(mu);
        if res < opts.tol {
            status = QpStatus::Optimal;
            break;
        }

        // --- assemble the symmetric KKT lower triangle and factor once.
        // The same factor backs both the predictor and corrector solves
        // (this single-factor / two-solve reuse is what makes Mehrotra
        // cheaper per iteration than two independent steps). ---
        cone.scaling_diag(&s, &z, &mut scaling);
        let (airn, ajcn, vals) = assemble_kkt(prob, &scaling, opts.reg, dim);
        let mut fact = match Factorization::new(dim as Index, airn, ajcn, vals, make_backend()) {
            Ok(f) => f,
            Err(_) => {
                status = QpStatus::NumericalFailure;
                break;
            }
        };

        // === Predictor (affine-scaling) step: σ = 0 ===
        // r_c = s∘z (affine target).
        cone.comp_residual(&s, &z, 0.0, &mut r_c);
        build_rhs(&r_d, &r_p, &r_g, &r_c, &z, n, m_eq, m_ineq, &mut rhs);
        if fact.solve_one(&mut rhs).is_err() {
            status = QpStatus::NumericalFailure;
            break;
        }
        split_step(&rhs, n, m_eq, m_ineq, &mut dx, &mut dy, &mut dz);
        cone.recover_ds(&s, &z, &r_c, &dz, &mut ds_aff);
        dz_aff.copy_from_slice(&dz);

        // Affine step lengths and the predicted duality measure μ_aff.
        let (alpha_p_aff, alpha_d_aff) =
            step_lengths(&cone, &s, &ds_aff, &z, &dz_aff, opts.tau, m_ineq);
        let sigma = if m_ineq == 0 {
            0.0
        } else {
            // μ_aff = ⟨s + αp ds_aff, z + αd dz_aff⟩ / m
            let mut dot = 0.0;
            for i in 0..m_ineq {
                dot += (s[i] + alpha_p_aff * ds_aff[i]) * (z[i] + alpha_d_aff * dz_aff[i]);
            }
            let mu_aff = dot / m_ineq as f64;
            // Mehrotra's heuristic centering parameter σ = (μ_aff/μ)³.
            (mu_aff / mu).powi(3)
        };

        // === Corrector step: centered target + second-order term ===
        if m_ineq == 0 {
            // No cone: predictor is already the full Newton step.
            for i in 0..n {
                x[i] += dx[i];
            }
            for i in 0..m_eq {
                y[i] += dy[i];
            }
        } else {
            let sigma_mu = sigma * mu;
            cone.comp_residual_corrector(&s, &z, &ds_aff, &dz_aff, sigma_mu, &mut r_c);
            build_rhs(&r_d, &r_p, &r_g, &r_c, &z, n, m_eq, m_ineq, &mut rhs);
            if fact.solve_one(&mut rhs).is_err() {
                status = QpStatus::NumericalFailure;
                break;
            }
            split_step(&rhs, n, m_eq, m_ineq, &mut dx, &mut dy, &mut dz);
            cone.recover_ds(&s, &z, &r_c, &dz, &mut ds);

            let (alpha_p, alpha_d) = step_lengths(&cone, &s, &ds, &z, &dz, opts.tau, m_ineq);
            for i in 0..n {
                x[i] += alpha_p * dx[i];
            }
            for i in 0..m_eq {
                y[i] += alpha_d * dy[i];
            }
            for i in 0..m_ineq {
                s[i] += alpha_p * ds[i];
                z[i] += alpha_d * dz[i];
            }
        }
    }

    // Objective ½ xᵀP x + cᵀx.
    let mut px = vec![0.0; n];
    prob.p_mul_add(&x, &mut px);
    let mut obj = 0.0;
    for i in 0..n {
        obj += 0.5 * x[i] * px[i] + prob.c[i] * x[i];
    }

    QpSolution {
        status,
        x,
        y,
        z,
        obj,
        iters,
    }
}

/// Build the Newton RHS `[−r_d; −r_p; −r_g + r_c ⊘ z]` for a given
/// complementarity residual `r_c` (predictor or corrector).
#[allow(clippy::too_many_arguments)]
fn build_rhs(
    r_d: &[f64],
    r_p: &[f64],
    r_g: &[f64],
    r_c: &[f64],
    z: &[f64],
    n: usize,
    m_eq: usize,
    m_ineq: usize,
    rhs: &mut [f64],
) {
    for i in 0..n {
        rhs[i] = -r_d[i];
    }
    for i in 0..m_eq {
        rhs[n + i] = -r_p[i];
    }
    for i in 0..m_ineq {
        rhs[n + m_eq + i] = -r_g[i] + r_c[i] / z[i];
    }
}

/// Copy the solved RHS into the (dx, dy, dz) step components.
fn split_step(
    rhs: &[f64],
    n: usize,
    m_eq: usize,
    m_ineq: usize,
    dx: &mut [f64],
    dy: &mut [f64],
    dz: &mut [f64],
) {
    dx.copy_from_slice(&rhs[0..n]);
    dy.copy_from_slice(&rhs[n..n + m_eq]);
    dz.copy_from_slice(&rhs[n + m_eq..n + m_eq + m_ineq]);
}

/// Separate fraction-to-boundary step lengths for the primal slack `s`
/// (via `ds`) and dual `z` (via `dz`). Returns `(alpha_primal,
/// alpha_dual)`; both are 1 when there is no cone.
fn step_lengths(
    cone: &NonnegCone,
    s: &[f64],
    ds: &[f64],
    z: &[f64],
    dz: &[f64],
    tau: f64,
    m_ineq: usize,
) -> (f64, f64) {
    if m_ineq == 0 {
        return (1.0, 1.0);
    }
    (cone.max_step(s, ds, tau), cone.max_step(z, dz, tau))
}

/// Assemble the lower triangle of the symmetric KKT matrix in 1-based
/// triplet form, summing duplicates via a `BTreeMap` so the caller never
/// relies on backend duplicate-summing. Variable layout: `x` then `y`
/// (equality) then `z` (inequality).
fn assemble_kkt(
    prob: &QpProblem,
    scaling: &[f64],
    reg: f64,
    dim: usize,
) -> (Vec<Index>, Vec<Index>, Vec<Number>) {
    let n = prob.n;
    let m_eq = prob.m_eq();
    let mut entries: BTreeMap<(usize, usize), f64> = BTreeMap::new();
    let mut add = |r: usize, c: usize, v: f64| {
        // lower triangle only (r ≥ c)
        let (r, c) = if r >= c { (r, c) } else { (c, r) };
        *entries.entry((r, c)).or_insert(0.0) += v;
    };

    // (x,x): P + δI (lower triangle of P as given).
    for t in &prob.p_lower {
        add(t.row, t.col, t.val);
    }
    for i in 0..n {
        add(i, i, reg);
    }

    // (y,x): A  (rows n.., cols 0..n) — these are lower triangle.
    for t in &prob.a {
        add(n + t.row, t.col, t.val);
    }
    // (y,y): −δI.
    for i in 0..m_eq {
        add(n + i, n + i, -reg);
    }

    // (z,x): G.
    for t in &prob.g {
        add(n + m_eq + t.row, t.col, t.val);
    }
    // (z,z): −(S⊘Z) − δI.
    for i in 0..prob.m_ineq() {
        add(n + m_eq + i, n + m_eq + i, -scaling[i] - reg);
    }

    debug_assert!(entries.keys().all(|(r, _)| *r < dim));
    let _ = dim;

    let mut airn = Vec::with_capacity(entries.len());
    let mut ajcn = Vec::with_capacity(entries.len());
    let mut vals = Vec::with_capacity(entries.len());
    for ((r, c), v) in entries {
        airn.push((r + 1) as Index); // 1-based
        ajcn.push((c + 1) as Index);
        vals.push(v);
    }
    (airn, ajcn, vals)
}

fn inf_norm(v: &[f64]) -> f64 {
    v.iter().fold(0.0_f64, |m, &x| m.max(x.abs()))
}
