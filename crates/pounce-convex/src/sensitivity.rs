//! Post-optimal sensitivity for the convex QP — the sIPOPT analog.
//!
//! Given a converged [`QpSolution`] to
//!
//! ```text
//!   min ½xᵀPx + cᵀx  s.t.  Ax = b,  Gx ≤ h,  lb ≤ x ≤ ub,
//! ```
//!
//! the first-order change of the primal–dual solution under a small
//! perturbation of the problem data — *holding the active set fixed* — is
//! the solution of the **active-set KKT system**
//!
//! ```text
//!   ⎡ P    Aᵀ   B_aᵀ ⎤ ⎡ dx  ⎤   ⎡ −dc                  ⎤
//!   ⎢ A    0    0    ⎥ ⎢ dy  ⎥ = ⎢  db                  ⎥
//!   ⎣ B_a  0    0    ⎦ ⎣ dz_a⎦   ⎣  dr_a                ⎦
//! ```
//!
//! where `B_a` stacks the **active** inequality rows of `G` and the active
//! variable-bound rows (`eⱼᵀ`), and the right-hand side is the parameter
//! derivative of the KKT residual. This is exactly the predictor used by
//! Ipopt's sIPOPT (Pirnay, López-Negrete & Biegler 2012) specialized to a
//! quadratic program, where the Lagrangian Hessian is the constant `P`.
//!
//! [`QpSensitivity`] assembles and factors this symmetric, indefinite
//! system **once** at the optimum; each [`QpSensitivity::parametric_step`]
//! is then a single back-substitution, so a parametric sweep costs one
//! solve per query (the build-once / solve-many idiom of the NLP
//! `Solver`). A tiny static regularization `δ` (the QP solver's own `reg`,
//! default `1e-8`) is placed on the diagonal so the indefinite factor is
//! stable; the induced error in the step is `O(δ)`.

use crate::ipm::QpOptions;
use crate::qp::{QpProblem, QpSolution, QpStatus};
use pounce_common::types::{Index, Number};
use pounce_linsol::{Factorization, SparseSymLinearSolverInterface};
use std::collections::BTreeMap;

/// A reason a [`QpSensitivity`] could not be built.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SensError {
    /// The solution was not optimal, so the active set is undefined.
    NotOptimal,
    /// The active-set KKT factorization failed (e.g. the active constraint
    /// gradients are rank-deficient, so the parametric step is not unique).
    FactorizationFailed,
}

/// Post-optimal sensitivity for a solved convex QP.
///
/// Holds the factored active-set KKT system at the optimum. Build it once
/// from a [`QpProblem`] and its [`QpSolution`], then call
/// [`parametric_step`](Self::parametric_step) for each parameter
/// perturbation — the factorization is reused across queries.
pub struct QpSensitivity {
    n: usize,
    m_eq: usize,
    /// KKT dimension `n + m_eq + n_active`.
    dim: usize,
    fact: Factorization,
}

impl QpSensitivity {
    /// Build the active-set sensitivity for `sol` (a solution of `prob`).
    ///
    /// The active set is read from the dual certificate: an inequality row
    /// `i` is active when `zᵢ > active_tol`, a lower bound on `xⱼ` when
    /// `z_lbⱼ > active_tol`, an upper bound when `z_ubⱼ > active_tol`. A
    /// good default for `active_tol` is `1e-7` (see
    /// [`build_default`](Self::build_default)).
    ///
    /// Returns [`SensError::NotOptimal`] if `sol` is not optimal, or
    /// [`SensError::FactorizationFailed`] if the active-set KKT is singular.
    pub fn build<F>(
        prob: &QpProblem,
        sol: &QpSolution,
        opts: &QpOptions,
        active_tol: f64,
        mut make_backend: F,
    ) -> Result<Self, SensError>
    where
        F: FnMut() -> Box<dyn SparseSymLinearSolverInterface>,
    {
        if sol.status != QpStatus::Optimal {
            return Err(SensError::NotOptimal);
        }
        let n = prob.n;
        let m_eq = prob.m_eq();
        let reg = opts.reg;

        // Active set: which inequality rows and which variable bounds bind.
        let active_ineq: Vec<usize> = (0..prob.m_ineq())
            .filter(|&i| sol.z[i] > active_tol)
            .collect();
        // A bound contributes one row `eⱼᵀ` (the gradient of `xⱼ = const` is
        // `eⱼ` whether the lower or upper bound is the active one).
        let active_bound_vars: Vec<usize> = (0..n)
            .filter(|&j| sol.z_lb[j] > active_tol || sol.z_ub[j] > active_tol)
            .collect();
        let n_active = active_ineq.len() + active_bound_vars.len();
        let dim = n + m_eq + n_active;

        // Assemble the lower triangle of the symmetric KKT matrix.
        let mut entries: BTreeMap<(usize, usize), f64> = BTreeMap::new();
        let mut add = |r: usize, c: usize, v: f64| {
            let (r, c) = if r >= c { (r, c) } else { (c, r) };
            *entries.entry((r, c)).or_insert(0.0) += v;
        };

        // (x,x): P + δI.
        for t in &prob.p_lower {
            add(t.row, t.col, t.val);
        }
        for i in 0..n {
            add(i, i, reg);
        }
        // (y,x): A; (y,y): −δI.
        for t in &prob.a {
            add(n + t.row, t.col, t.val);
        }
        for i in 0..m_eq {
            add(n + i, n + i, -reg);
        }
        // Active-row block `B_a` after the equality rows, in order:
        // active inequality rows, then active bound rows. (·,·): −δI diagonal.
        let abase = n + m_eq;
        for (k, &i) in active_ineq.iter().enumerate() {
            // The k-th active row holds G's row i.
            for t in prob.g.iter().filter(|t| t.row == i) {
                add(abase + k, t.col, t.val);
            }
        }
        for (k, &j) in active_bound_vars.iter().enumerate() {
            add(abase + active_ineq.len() + k, j, 1.0);
        }
        for k in 0..n_active {
            add(abase + k, abase + k, -reg);
        }

        // Triplets → 1-based lower-triangle arrays for the factorization.
        let nnz = entries.len();
        let mut airn = Vec::with_capacity(nnz);
        let mut ajcn = Vec::with_capacity(nnz);
        let mut values = Vec::with_capacity(nnz);
        for ((r, c), v) in entries {
            airn.push((r + 1) as Index);
            ajcn.push((c + 1) as Index);
            values.push(v);
        }

        let fact = Factorization::new(dim as Index, airn, ajcn, values, make_backend())
            .map_err(|_| SensError::FactorizationFailed)?;

        Ok(QpSensitivity { n, m_eq, dim, fact })
    }

    /// [`build`](Self::build) with the QP's default options and an active-set
    /// tolerance of `1e-7`.
    pub fn build_default<F>(
        prob: &QpProblem,
        sol: &QpSolution,
        make_backend: F,
    ) -> Result<Self, SensError>
    where
        F: FnMut() -> Box<dyn SparseSymLinearSolverInterface>,
    {
        Self::build(prob, sol, &QpOptions::default(), 1e-7, make_backend)
    }

    /// First-order primal step `dx ≈ x*(b + Δb) − x*(b)` for a perturbation
    /// of the **equality right-hand side** `b`, the direct QP analog of
    /// sIPOPT's "pin a constraint, perturb its value". Constraint
    /// `pin_constraint_indices[k]` (an index into `b`) is perturbed by
    /// `deltas[k]`; all others are held fixed.
    ///
    /// Returns the length-`n` primal sensitivity, so `x* + dx` predicts the
    /// solution of the perturbed QP (exact to first order while the active
    /// set is unchanged). The factorization is reused, so repeated calls
    /// (e.g. a continuation sweep) cost one back-substitution each.
    ///
    /// # Panics
    ///
    /// Panics if `pin_constraint_indices` and `deltas` differ in length, or
    /// if any pin index is `≥ m_eq`.
    pub fn parametric_step(
        &mut self,
        pin_constraint_indices: &[usize],
        deltas: &[f64],
    ) -> Vec<f64> {
        assert_eq!(
            pin_constraint_indices.len(),
            deltas.len(),
            "pin_constraint_indices and deltas must have equal length"
        );
        let mut db = vec![0.0; self.m_eq];
        for (&i, &d) in pin_constraint_indices.iter().zip(deltas) {
            assert!(
                i < self.m_eq,
                "pin constraint index {i} out of range (m_eq = {})",
                self.m_eq
            );
            db[i] += d;
        }
        self.step_from_db(&db)
    }

    /// Primal sensitivity for a full equality-RHS perturbation `db` (length
    /// `m_eq`): solves the active-set KKT with right-hand side `[0; db; 0]`
    /// and returns `dx = step[0..n]`.
    pub fn step_from_db(&mut self, db: &[f64]) -> Vec<f64> {
        assert_eq!(db.len(), self.m_eq, "db must have length m_eq");
        let mut rhs = vec![0.0 as Number; self.dim];
        rhs[self.n..self.n + self.m_eq].copy_from_slice(db);
        // A singular factor would have been caught at build; a back-solve
        // failure here is not recoverable, so surface a zero step.
        if self.fact.solve_one(&mut rhs).is_err() {
            return vec![0.0; self.n];
        }
        rhs.truncate(self.n);
        rhs
    }

    /// The active-set KKT dimension `n + m_eq + n_active`.
    pub fn kkt_dim(&self) -> usize {
        self.dim
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ipm::solve_qp_ipm;
    use crate::qp::Triplet;
    use pounce_feral::FeralSolverInterface;

    fn backend() -> Box<dyn SparseSymLinearSolverInterface> {
        Box::new(FeralSolverInterface::new())
    }

    /// `min ½‖x‖²  s.t.  x₀ + x₁ = b` (b = 2). The optimum is the projection
    /// of the origin onto the line: `x = (b/2, b/2)`, so `dx/db = (½, ½)`
    /// exactly. The parametric step for `Δb` must reproduce that.
    #[test]
    fn parametric_step_matches_closed_form_equality() {
        let prob = QpProblem {
            n: 2,
            p_lower: vec![Triplet::new(0, 0, 1.0), Triplet::new(1, 1, 1.0)],
            c: vec![0.0, 0.0],
            a: vec![Triplet::new(0, 0, 1.0), Triplet::new(0, 1, 1.0)],
            b: vec![2.0],
            g: vec![],
            h: vec![],
            lb: vec![],
            ub: vec![],
        };
        let sol = solve_qp_ipm(&prob, &QpOptions::default(), backend);
        assert_eq!(sol.status, QpStatus::Optimal);
        assert!((sol.x[0] - 1.0).abs() < 1e-7 && (sol.x[1] - 1.0).abs() < 1e-7);

        let mut sens = QpSensitivity::build_default(&prob, &sol, backend).unwrap();
        let dx = sens.parametric_step(&[0], &[1.0]); // Δb = +1
        assert!((dx[0] - 0.5).abs() < 1e-6, "dx0 = {}", dx[0]);
        assert!((dx[1] - 0.5).abs() < 1e-6, "dx1 = {}", dx[1]);

        // Predictor lands on the exact re-solve for the perturbed b.
        let mut prob2 = prob.clone();
        prob2.b = vec![3.0];
        let sol2 = solve_qp_ipm(&prob2, &QpOptions::default(), backend);
        assert!((sol.x[0] + dx[0] - sol2.x[0]).abs() < 1e-6);
        assert!((sol.x[1] + dx[1] - sol2.x[1]).abs() < 1e-6);
    }

    /// With an **active inequality** in the active set, the predictor must
    /// still match the re-solve. `min ½‖x‖² s.t. x₀+x₁ = b, x₀ ≥ 1`. At
    /// b = 1 the unconstrained projection would be (0.5, 0.5) but `x₀ ≥ 1`
    /// binds, giving `x = (1, 0)`. Perturbing b shifts along the active
    /// face: `x = (1, b−1)`, so `dx/db = (0, 1)`.
    #[test]
    fn parametric_step_with_active_inequality() {
        let prob = QpProblem {
            n: 2,
            p_lower: vec![Triplet::new(0, 0, 1.0), Triplet::new(1, 1, 1.0)],
            c: vec![0.0, 0.0],
            a: vec![Triplet::new(0, 0, 1.0), Triplet::new(0, 1, 1.0)],
            b: vec![1.0],
            g: vec![Triplet::new(0, 0, -1.0)], // −x₀ ≤ −1  ⇔  x₀ ≥ 1
            h: vec![-1.0],
            lb: vec![],
            ub: vec![],
        };
        let sol = solve_qp_ipm(&prob, &QpOptions::default(), backend);
        assert_eq!(sol.status, QpStatus::Optimal);
        assert!((sol.x[0] - 1.0).abs() < 1e-6 && sol.x[1].abs() < 1e-6);
        assert!(sol.z[0] > 1e-6, "inequality should be active");

        let mut sens = QpSensitivity::build_default(&prob, &sol, backend).unwrap();
        let dx = sens.parametric_step(&[0], &[0.5]);
        assert!(dx[0].abs() < 1e-6, "dx0 = {} (should stay on x₀=1)", dx[0]);
        assert!((dx[1] - 0.5).abs() < 1e-6, "dx1 = {}", dx[1]);
    }

    /// A non-optimal solution has no well-defined active set.
    #[test]
    fn build_rejects_non_optimal() {
        let prob = QpProblem {
            n: 1,
            p_lower: vec![],
            c: vec![-1.0],
            a: vec![],
            b: vec![],
            g: vec![Triplet::new(0, 0, -1.0)],
            h: vec![0.0], // x ≥ 0, min −x ⇒ unbounded
            lb: vec![],
            ub: vec![],
        };
        let sol = solve_qp_ipm(&prob, &QpOptions::default(), backend);
        assert_ne!(sol.status, QpStatus::Optimal);
        assert!(matches!(
            QpSensitivity::build_default(&prob, &sol, backend),
            Err(SensError::NotOptimal)
        ));
    }
}
