//! Bare primal-dual interior-point driver for convex QP (Phase 2).
//!
//! This is the *bare* path-follower the plan calls for: a correct,
//! infeasible-start primal-dual method with a fixed centering parameter
//! and fraction-to-boundary step control. Mehrotra predictor-corrector
//! and the homogeneous self-dual embedding are Phase 3 — this iteration
//! is the scaffolding they slot into.
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
    /// Fixed centering parameter σ ∈ (0, 1) (bare method; Mehrotra in
    /// Phase 3 computes this adaptively).
    pub sigma: f64,
    /// Fraction-to-boundary parameter τ ∈ (0, 1).
    pub tau: f64,
    /// Static KKT regularization δ.
    pub reg: f64,
}

impl Default for QpOptions {
    fn default() -> Self {
        QpOptions {
            tol: 1e-8,
            max_iter: 200,
            sigma: 0.1,
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
    let mut ds = vec![0.0; m_ineq];

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

        // --- assemble the symmetric KKT lower triangle ---
        cone.scaling_diag(&s, &z, &mut scaling);
        let (airn, ajcn, vals) = assemble_kkt(prob, &scaling, opts.reg, dim);

        // --- right-hand side ---
        // complementarity residual r_c = s∘z − σμ e
        let sigma_mu = opts.sigma * mu;
        cone.comp_residual(&s, &z, sigma_mu, &mut r_c);

        let mut rhs = vec![0.0; dim];
        for i in 0..n {
            rhs[i] = -r_d[i];
        }
        for i in 0..m_eq {
            rhs[n + i] = -r_p[i];
        }
        for i in 0..m_ineq {
            // −r_g + r_c ⊘ z
            rhs[n + m_eq + i] = -r_g[i] + r_c[i] / z[i];
        }

        // --- factor & solve ---
        let mut fact = match Factorization::new(dim as Index, airn, ajcn, vals, make_backend()) {
            Ok(f) => f,
            Err(_) => {
                status = QpStatus::NumericalFailure;
                break;
            }
        };
        if fact.solve_one(&mut rhs).is_err() {
            status = QpStatus::NumericalFailure;
            break;
        }

        let dx = &rhs[0..n];
        let dy = &rhs[n..n + m_eq];
        let dz = &rhs[n + m_eq..n + m_eq + m_ineq];

        // recover ds = −(r_c ⊘ z) − (s ⊘ z) ∘ dz
        cone.recover_ds(&s, &z, &r_c, dz, &mut ds);

        // --- step length (single α keeps s, z interior) ---
        let alpha = if m_ineq == 0 {
            1.0
        } else {
            let a_s = cone.max_step(&s, &ds, opts.tau);
            let a_z = cone.max_step(&z, dz, opts.tau);
            a_s.min(a_z)
        };

        // --- update ---
        for i in 0..n {
            x[i] += alpha * dx[i];
        }
        for i in 0..m_eq {
            y[i] += alpha * dy[i];
        }
        for i in 0..m_ineq {
            z[i] += alpha * dz[i];
            s[i] += alpha * ds[i];
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
