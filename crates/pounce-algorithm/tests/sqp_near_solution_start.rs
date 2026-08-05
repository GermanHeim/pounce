//! Regression: the active-set SQP reported `InfeasibleSubproblem` when
//! started *near* a solution, on a problem it solves without trouble
//! from far away.
//!
//! Found while fixing gh#484 (the C warm-start path discarding the
//! caller's iterate). Warm-starting is precisely the case of starting
//! near a solution, so this was the defect standing behind that one:
//! HS071 cold-started from `(1,5,5,1)` converges, but nudge the start to
//! `x* + 1e-6·e₁` and iteration 0 died with an infeasibility verdict on
//! a problem whose feasible set is unchanged.
//!
//! Two independent defects in the step-QP path produced it, both in
//! `pounce_qp::solver::solve_elastic`:
//!
//! 1. The l1-elastic phase-1's residual slacks were read as an
//!    infeasibility certificate even when the elastic problem was
//!    nonconvex. The SQP's default Hessian is the exact ∇²L, which is
//!    indefinite here, so the active-set solve returns a *local* KKT
//!    point whose slacks are not the global minimal-l1 violation. With
//!    γ = 1e6 amplifying a ~1e-7 slack into ~0.1 of apparent objective,
//!    it settled at a far box vertex missing `feas_tol` by a factor of
//!    two — 1.95e-9 against 1e-9 — on a QP with points feasible to
//!    slack 1.66.
//! 2. The phase-2 recovery re-solve seeded a *cold* working set, which
//!    marks equality rows `Inactive`. The warm inner loop cannot pull an
//!    Inactive equality into the working set, so the equality went
//!    unenforced: recovery converged to `Optimal` at a point violating
//!    it by 7.8, failed the feasibility check, and fell through to the
//!    certificate it was supposed to prevent.

use pounce_algorithm::sqp::{
    SqpAlgorithm, SqpHessianSource, SqpOptions, SqpProblemSpec, SqpStatus, Triplet,
};
use pounce_common::types::{Index, Number};
use pounce_qp::solver::ParametricActiveSetSolver;

/// HS071 — the canonical Ipopt example.
///
/// ```text
///     min  x0·x3·(x0+x1+x2) + x2
///     s.t. x0·x1·x2·x3 >= 25
///          x0²+x1²+x2²+x3² == 40
///          1 <= x <= 5
/// ```
struct Hs071 {
    x0: Vec<Number>,
}

/// Converged solution, to full double precision.
const X_STAR: [Number; 4] = [
    1.0,
    4.742999637927625,
    3.8211499836197276,
    1.3794082930783524,
];
const F_STAR: Number = 17.014_017_289_2;

impl SqpProblemSpec for Hs071 {
    fn n(&self) -> usize {
        4
    }
    fn m(&self) -> usize {
        2
    }
    fn x_init(&self) -> Vec<Number> {
        self.x0.clone()
    }
    fn variable_bounds(&self) -> (Vec<Number>, Vec<Number>) {
        (vec![1.0; 4], vec![5.0; 4])
    }
    fn constraint_bounds(&self) -> (Vec<Number>, Vec<Number>) {
        (vec![25.0, 40.0], vec![2.0e19, 40.0])
    }
    fn eval_f(&mut self, x: &[Number]) -> Number {
        x[0] * x[3] * (x[0] + x[1] + x[2]) + x[2]
    }
    fn eval_grad_f(&mut self, x: &[Number]) -> Vec<Number> {
        vec![
            x[3] * (2.0 * x[0] + x[1] + x[2]),
            x[0] * x[3],
            x[0] * x[3] + 1.0,
            x[0] * (x[0] + x[1] + x[2]),
        ]
    }
    fn eval_c(&mut self, x: &[Number]) -> Vec<Number> {
        vec![
            x[0] * x[1] * x[2] * x[3],
            x[0] * x[0] + x[1] * x[1] + x[2] * x[2] + x[3] * x[3],
        ]
    }
    fn eval_jac_c(&mut self, x: &[Number]) -> Triplet {
        let mut irow = Vec::new();
        let mut jcol = Vec::new();
        let mut vals = Vec::new();
        for i in 0..2 {
            for j in 0..4 {
                irow.push(i as Index + 1);
                jcol.push(j as Index + 1);
                vals.push(if i == 0 {
                    x.iter()
                        .enumerate()
                        .filter(|(k, _)| *k != j)
                        .map(|(_, v)| *v)
                        .product()
                } else {
                    2.0 * x[j]
                });
            }
        }
        Triplet {
            n_rows: 2,
            n_cols: 4,
            irow,
            jcol,
            vals,
        }
    }
    fn eval_hess_lag(&mut self, x: &[Number], lam: &[Number]) -> Triplet {
        let mut h = [[0.0 as Number; 4]; 4];
        h[0][0] = 2.0 * x[3];
        h[1][0] = x[3];
        h[2][0] = x[3];
        h[3][0] = 2.0 * x[0] + x[1] + x[2];
        h[3][1] = x[0];
        h[3][2] = x[0];

        h[1][0] += lam[0] * (x[2] * x[3]);
        h[2][0] += lam[0] * (x[1] * x[3]);
        h[2][1] += lam[0] * (x[0] * x[3]);
        h[3][0] += lam[0] * (x[1] * x[2]);
        h[3][1] += lam[0] * (x[0] * x[2]);
        h[3][2] += lam[0] * (x[0] * x[1]);

        for (i, row) in h.iter_mut().enumerate() {
            row[i] += lam[1] * 2.0;
        }

        let mut irow = Vec::new();
        let mut jcol = Vec::new();
        let mut vals = Vec::new();
        for i in 0..4 {
            for j in 0..=i {
                irow.push(i as Index + 1);
                jcol.push(j as Index + 1);
                vals.push(h[i][j]);
            }
        }
        Triplet {
            n_rows: 4,
            n_cols: 4,
            irow,
            jcol,
            vals,
        }
    }
}

fn solve_from(x0: Vec<Number>, hessian: SqpHessianSource) -> pounce_algorithm::sqp::SqpResult {
    let mut nlp = Hs071 { x0 };
    let opts = SqpOptions {
        hessian,
        ..SqpOptions::default()
    };
    let qp = ParametricActiveSetSolver::new(Box::new(pounce_feral::FeralSolverInterface::new()));
    SqpAlgorithm::new(qp, opts)
        .optimize(&mut nlp)
        .expect("optimize must not error")
}

fn assert_converged(tag: &str, res: &pounce_algorithm::sqp::SqpResult) {
    assert_eq!(
        res.status,
        SqpStatus::Optimal,
        "{tag}: expected Optimal, got {:?} after {} iters at obj {}",
        res.status,
        res.n_iter,
        res.obj
    );
    assert!(
        (res.obj - F_STAR).abs() < 1e-6,
        "{tag}: obj = {}, expected ~{F_STAR}",
        res.obj
    );
    for (i, (got, want)) in res.x.iter().zip(X_STAR.iter()).enumerate() {
        assert!(
            (got - want).abs() < 1e-5,
            "{tag}: x[{i}] = {got}, expected ~{want} (full x = {:?})",
            res.x
        );
    }
}

/// The far cold start, which always worked — the control that shows the
/// problem, bounds and derivatives are fine.
#[test]
fn hs071_converges_from_the_far_cold_start() {
    let res = solve_from(vec![1.0, 5.0, 5.0, 1.0], SqpHessianSource::Exact);
    assert_converged("cold (1,5,5,1)", &res);
}

/// The regression proper: starts within `eps` of the solution — the
/// regime every warm start lands in. `1e-6` and `1e-3` both returned
/// `InfeasibleSubproblem` at iteration 0 before the fix, while `1e-8`
/// and `1e-1` happened to survive, so a single perturbation size is not
/// enough to pin this down.
#[test]
fn hs071_converges_from_starts_near_the_solution() {
    for eps in [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1] {
        for coord in 0..4 {
            let mut x0 = X_STAR.to_vec();
            x0[coord] += eps;
            // Stay inside `1 <= x <= 5`; x0[0] sits on its lower bound.
            x0[coord] = x0[coord].clamp(1.0, 5.0);
            let res = solve_from(x0, SqpHessianSource::Exact);
            assert_ne!(
                res.status,
                SqpStatus::InfeasibleSubproblem,
                "eps={eps:e} coord={coord}: HS071 declared infeasible from a \
                 point {eps:e} away from its own solution"
            );
            assert_converged(&format!("eps={eps:e} coord={coord}"), &res);
        }
    }
}

/// Starting exactly at the solution must be recognised immediately —
/// the ideal warm start, and the shape of the C-API case in gh#484.
#[test]
fn hs071_starting_at_the_solution_converges_immediately() {
    let res = solve_from(X_STAR.to_vec(), SqpHessianSource::Exact);
    assert_converged("at x*", &res);
    assert!(
        res.n_iter <= 1,
        "expected ~0-1 iterations from x*, took {}",
        res.n_iter
    );
}

/// The quasi-Newton Hessians are PSD by construction, so their step QPs
/// were convex and never tripped the nonconvex-certificate path. Pinned
/// so a future change to the elastic path cannot regress them either.
#[test]
fn hs071_converges_near_the_solution_with_quasi_newton_hessians() {
    for hessian in [SqpHessianSource::DampedBfgs, SqpHessianSource::Lbfgs] {
        for eps in [1e-6, 1e-3] {
            let mut x0 = X_STAR.to_vec();
            x0[1] += eps;
            let res = solve_from(x0, hessian);
            assert_converged(&format!("{hessian:?} eps={eps:e}"), &res);
        }
    }
}
