//! Sparse parametric active-set QP — the `pounce-qp` engine behind the
//! active-set SQP path, re-exported (feature `qp`).
//!
//! ```toml
//! [dependencies]
//! pounce-rs = { version = "0.9", features = ["qp"] }
//! ```
//!
//! This is a different solver family from [`crate::convex`], not a second
//! interface to it. The interior-point path is the right default for a cold
//! solve; the active-set path exists for the *parametric* case — an SQP outer
//! loop, an MPC receding horizon, a continuation sweep — where each QP is a
//! perturbation of the last and
//! [`QpSolver::solve_parametric`] traces the homotopy from the previous
//! solution, reusing the Schur-complement factor instead of refactorizing.
//! It also accepts an indefinite Hessian, which the convex path by
//! construction does not.
//!
//! The problem is
//!
//! ```text
//! min  ½ xᵀH x + gᵀx    s.t.  bl ≤ A x ≤ bu,  xl ≤ x ≤ xu
//! ```
//!
//! with two-sided bounds throughout (equality is `bl == bu`, a fixed variable
//! is `xl == xu`, free is `±1e20`). [`QpProblem`] *borrows* `H` and `A` as
//! [`SymTMatrix`] / [`GenTMatrix`] triplets — the solver never copies them —
//! so those types are re-exported here too.
//!
//! ```
//! use pounce_rs::linsol::backend;
//! use pounce_rs::qp::{
//!     GenTMatrix, GenTMatrixSpace, HessianInertia, ParametricActiveSetSolver, QpOptions,
//!     QpProblem, QpSolver, SymTMatrix, SymTMatrixSpace,
//! };
//! use std::rc::Rc;
//!
//! // min ½(4x₀² + 4x₁²) − 2x₀ − 2x₁  s.t.  x₀ + x₁ == 1   ⇒  x* = (½, ½)
//! // Triplet row/column indices are 1-based.
//! let mut h = SymTMatrix::new(SymTMatrixSpace::new(2, vec![1, 2], vec![1, 2]));
//! h.set_values(&[4.0, 4.0]);
//! let mut a = GenTMatrix::new(GenTMatrixSpace::new(1, 2, vec![1, 1], vec![1, 2]));
//! a.set_values(&[1.0, 1.0]);
//!
//! let qp = QpProblem {
//!     n: 2,
//!     m: 1,
//!     h: &h,
//!     g: &[-2.0, -2.0],
//!     a: &a,
//!     bl: &[1.0],
//!     bu: &[1.0],
//!     xl: &[-1e20, -1e20],
//!     xu: &[1e20, 1e20],
//!     hessian_inertia: HessianInertia::Psd,
//! };
//!
//! let mut solver = ParametricActiveSetSolver::new(backend());
//! let sol = solver.solve(&qp, None, &QpOptions::default()).unwrap();
//! assert!((sol.x[0] - 0.5).abs() < 1e-8 && (sol.x[1] - 0.5).abs() < 1e-8);
//! assert!((sol.obj + 1.0).abs() < 1e-8);
//! ```
//!
//! A solve returns its final [`WorkingSet`], which is what a subsequent
//! [`QpSolver::solve`] takes as a [`QpWarmStart`] seed; [`QpSolver::solve_parametric`]
//! goes further and carries the factor across.
//!
//! [`pounce_qp`] itself is re-exported for the internals — Schur factor
//! maintenance, the l1-elastic phase-1 reformulation, the KKT assembly
//! helpers, and the `.qps` reader.

pub use pounce_qp::{
    AntiCyclingChoice, BoundStatus, ConsStatus, ElasticReformulation, HessianInertia, KktTriplet,
    LinearSolver, ParametricActiveSetSolver, QpAlgorithm, QpError, QpOptions, QpProblem,
    QpSolution, QpSolver, QpStats, QpStatus, QpWarmStart, QpsModel, WorkingSet, parse_qps,
};

/// The borrowed triplet storage [`QpProblem`] is built from.
pub use pounce_linalg::triplet::{GenTMatrix, GenTMatrixSpace, SymTMatrix, SymTMatrixSpace};

/// The underlying crate, for anything not surfaced above.
pub use pounce_qp;
