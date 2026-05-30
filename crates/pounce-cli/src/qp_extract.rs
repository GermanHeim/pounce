//! Extract a `pounce_convex::QpProblem` (standard form) from a parsed
//! `.nl` problem, for the LP/QP dispatch path (Phase 2).
//!
//! The classifier (`crate::dispatch`) has already decided the problem is
//! an LP or convex QP; this module marshals the parsed `NlProblem` into
//! the standard form the convex IPM consumes:
//!
//! ```text
//! minimize    ½ xᵀP x + cᵀx
//! subject to  A x = b          (equalities)
//!             G x ≤ h          (inequalities, incl. finite var bounds)
//! ```
//!
//! Mapping from the `.nl` representation:
//! - **Objective.** `P` is the Hessian of the (degree-≤2) objective —
//!   recovered with the same `analyze_quadratic` the classifier uses, so
//!   `P` here is exactly the matrix whose definiteness was tested. `c`
//!   is the objective's linear part. A `maximize` objective is negated
//!   into a minimization.
//! - **Constraints.** Each row has a linear part and bounds `g_l ≤ row ≤
//!   g_u`. An equality (`g_l == g_u`) becomes a row of `A`; a one- or
//!   two-sided inequality becomes one or two rows of `G` (`row ≤ g_u`
//!   and/or `−row ≤ −g_l`).
//! - **Variable bounds.** Finite `x_l`/`x_u` become `G` rows
//!   (`−x_i ≤ −x_l`, `x_i ≤ x_u`); the `.nl` "infinity" sentinel
//!   (`|v| ≥ 1e19`) is treated as no bound.

use crate::dispatch::analyze_quadratic;
use crate::nl_reader::NlProblem;
use pounce_convex::{QpProblem, Triplet};

/// The `.nl` infinity sentinel: AMPL writes ±1e20-ish for "no bound";
/// upstream Ipopt treats anything with magnitude ≥ 1e19 as infinite.
const NL_INF: f64 = 1e19;

fn is_finite_bound(v: f64) -> bool {
    v.abs() < NL_INF
}

/// Convert a classified LP/convex-QP `NlProblem` into `QpProblem`
/// standard form. Returns `None` if the objective is not actually a
/// degree-≤2 polynomial (should not happen for a problem the classifier
/// routed here, but the conversion is total and falls back gracefully).
pub fn extract_qp(prob: &NlProblem) -> Option<QpProblem> {
    let n = prob.n;
    let sign = if prob.minimize { 1.0 } else { -1.0 };

    // --- objective Hessian P (lower triangle) ---
    let hess = analyze_quadratic(&prob.obj_nonlinear, n)?;
    let mut p_lower: Vec<Triplet> = Vec::with_capacity(hess.len());
    for ((i, j), v) in &hess {
        // analyze_quadratic returns (i ≤ j) upper-ish keys; store as
        // lower triangle (row ≥ col) for the solver.
        let (row, col) = if i >= j { (*i, *j) } else { (*j, *i) };
        p_lower.push(Triplet::new(row, col, sign * v));
    }

    // --- objective linear term c ---
    let mut c = vec![0.0; n];
    for (var, coef) in &prob.obj_linear {
        c[*var] += sign * coef;
    }

    // --- constraints: equalities → A x = b, inequalities → G x ≤ h ---
    let mut a: Vec<Triplet> = Vec::new();
    let mut b: Vec<f64> = Vec::new();
    let mut g: Vec<Triplet> = Vec::new();
    let mut h: Vec<f64> = Vec::new();

    for (row, lin) in prob.con_linear.iter().enumerate() {
        let lo = prob.g_l[row];
        let hi = prob.g_u[row];
        if lo == hi && is_finite_bound(lo) {
            // Equality row.
            let eq_row = next_row(&b);
            for (var, coef) in lin {
                a.push(Triplet::new(eq_row, *var, *coef));
            }
            b.push(lo);
        } else {
            // Upper bound: row ≤ hi.
            if is_finite_bound(hi) {
                let gr = next_row(&h);
                for (var, coef) in lin {
                    g.push(Triplet::new(gr, *var, *coef));
                }
                h.push(hi);
            }
            // Lower bound: row ≥ lo  ⇔  −row ≤ −lo.
            if is_finite_bound(lo) {
                let gr = next_row(&h);
                for (var, coef) in lin {
                    g.push(Triplet::new(gr, *var, -*coef));
                }
                h.push(-lo);
            }
        }
    }

    // --- variable bounds as G rows ---
    for i in 0..n {
        let xl = prob.x_l[i];
        let xu = prob.x_u[i];
        if is_finite_bound(xu) {
            let gr = next_row(&h);
            g.push(Triplet::new(gr, i, 1.0)); // x_i ≤ xu
            h.push(xu);
        }
        if is_finite_bound(xl) {
            let gr = next_row(&h);
            g.push(Triplet::new(gr, i, -1.0)); // −x_i ≤ −xl
            h.push(-xl);
        }
    }

    Some(QpProblem {
        n,
        p_lower,
        c,
        a,
        b,
        g,
        h,
    })
}

/// The next 0-based row index for a constraint block keyed by its RHS
/// vector's current length.
fn next_row(rhs: &[f64]) -> usize {
    rhs.len()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nl_reader::{BinOp, Expr};
    use pounce_convex::{solve_qp_ipm, QpOptions, QpStatus};
    use pounce_feral::FeralSolverInterface;
    use pounce_linsol::SparseSymLinearSolverInterface;

    fn backend() -> Box<dyn SparseSymLinearSolverInterface> {
        Box::new(FeralSolverInterface::new())
    }

    fn pow2(var: usize) -> Expr {
        Expr::Binary(
            BinOp::Pow,
            Box::new(Expr::Var(var)),
            Box::new(Expr::Const(2.0)),
        )
    }

    /// min (x0)^2 + (x1)^2 s.t. x0 + x1 = 2, no var bounds → (1,1), f*=2.
    #[test]
    fn extract_and_solve_equality_qp() {
        let prob = NlProblem {
            n: 2,
            m: 1,
            num_obj: 1,
            minimize: true,
            obj_nonlinear: Expr::Binary(BinOp::Add, Box::new(pow2(0)), Box::new(pow2(1))),
            obj_linear: vec![],
            obj_constant: 0.0,
            con_nonlinear: vec![Expr::Const(0.0)],
            con_linear: vec![vec![(0, 1.0), (1, 1.0)]],
            x_l: vec![-2e19, -2e19],
            x_u: vec![2e19, 2e19],
            g_l: vec![2.0],
            g_u: vec![2.0],
            x0: vec![0.0, 0.0],
            lambda0: vec![0.0],
            suffixes: Default::default(),
            imported_funcs: Vec::new(),
        };
        let qp = extract_qp(&prob).expect("extract");
        // P = 2I → two diagonal entries.
        assert_eq!(qp.p_lower.len(), 2);
        assert_eq!(qp.m_eq(), 1);
        assert_eq!(qp.m_ineq(), 0);

        let sol = solve_qp_ipm(&qp, &QpOptions::default(), backend);
        assert_eq!(sol.status, QpStatus::Optimal);
        assert!((sol.x[0] - 1.0).abs() < 1e-6, "x0={}", sol.x[0]);
        assert!((sol.x[1] - 1.0).abs() < 1e-6, "x1={}", sol.x[1]);
        assert!((sol.obj - 2.0).abs() < 1e-6, "obj={}", sol.obj);
    }

    /// Bound-constrained: min (x0-3)^2 = x0^2 - 6 x0 + 9, 0 ≤ x0 ≤ 1.
    /// Optimum x0 = 1 (upper bound binds). (Constant 9 dropped from c.)
    #[test]
    fn extract_and_solve_bounded_qp() {
        let prob = NlProblem {
            n: 1,
            m: 0,
            num_obj: 1,
            minimize: true,
            obj_nonlinear: pow2(0),
            obj_linear: vec![(0, -6.0)],
            obj_constant: 9.0,
            con_nonlinear: vec![],
            con_linear: vec![],
            x_l: vec![0.0],
            x_u: vec![1.0],
            g_l: vec![],
            g_u: vec![],
            x0: vec![0.0],
            lambda0: vec![],
            suffixes: Default::default(),
            imported_funcs: Vec::new(),
        };
        let qp = extract_qp(&prob).expect("extract");
        // Two var-bound rows (x0 ≤ 1, −x0 ≤ 0).
        assert_eq!(qp.m_ineq(), 2);
        let sol = solve_qp_ipm(&qp, &QpOptions::default(), backend);
        assert_eq!(sol.status, QpStatus::Optimal);
        assert!((sol.x[0] - 1.0).abs() < 1e-6, "x0={}", sol.x[0]);
    }

    /// LP: min −x0 − x1, 0 ≤ x ≤ 1 → (1,1).
    #[test]
    fn extract_and_solve_lp() {
        let prob = NlProblem {
            n: 2,
            m: 0,
            num_obj: 1,
            minimize: true,
            obj_nonlinear: Expr::Const(0.0),
            obj_linear: vec![(0, -1.0), (1, -1.0)],
            obj_constant: 0.0,
            con_nonlinear: vec![],
            con_linear: vec![],
            x_l: vec![0.0, 0.0],
            x_u: vec![1.0, 1.0],
            g_l: vec![],
            g_u: vec![],
            x0: vec![0.0, 0.0],
            lambda0: vec![],
            suffixes: Default::default(),
            imported_funcs: Vec::new(),
        };
        let qp = extract_qp(&prob).expect("extract");
        assert!(qp.p_lower.is_empty(), "LP has no Hessian");
        assert_eq!(qp.m_ineq(), 4); // 2 vars × (upper + lower)
        let sol = solve_qp_ipm(&qp, &QpOptions::default(), backend);
        assert_eq!(sol.status, QpStatus::Optimal);
        assert!((sol.x[0] - 1.0).abs() < 1e-6);
        assert!((sol.x[1] - 1.0).abs() < 1e-6);
    }

    /// maximize x0 s.t. 0 ≤ x0 ≤ 5 → x0 = 5. Tests sign flip on a
    /// maximize objective.
    #[test]
    fn extract_maximize_negates() {
        let prob = NlProblem {
            n: 1,
            m: 0,
            num_obj: 1,
            minimize: false,
            obj_nonlinear: Expr::Const(0.0),
            obj_linear: vec![(0, 1.0)],
            obj_constant: 0.0,
            con_nonlinear: vec![],
            con_linear: vec![],
            x_l: vec![0.0],
            x_u: vec![5.0],
            g_l: vec![],
            g_u: vec![],
            x0: vec![0.0],
            lambda0: vec![],
            suffixes: Default::default(),
            imported_funcs: Vec::new(),
        };
        let qp = extract_qp(&prob).expect("extract");
        // minimize −x0.
        assert_eq!(qp.c[0], -1.0);
        let sol = solve_qp_ipm(&qp, &QpOptions::default(), backend);
        assert_eq!(sol.status, QpStatus::Optimal);
        assert!((sol.x[0] - 5.0).abs() < 1e-6, "x0={}", sol.x[0]);
    }
}
