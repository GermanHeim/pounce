//! The sparse symmetric linear-solver backend the QP entry points need.
//!
//! Both [`crate::convex`] and [`crate::qp`] are parameterized over the
//! factorization backend rather than hard-wiring one: the interior-point
//! driver takes a `FnMut() -> Box<dyn SparseSymLinearSolverInterface>`
//! factory, and `ParametricActiveSetSolver::new` takes a boxed backend
//! directly. That is deliberate — it is how POUNCE swaps FERAL for
//! HSL MA27/MA57 — but it means re-exporting the solvers alone would still
//! leave a caller depending on `pounce-linsol` + `pounce-feral` to produce
//! the argument. This module closes that gap with the same two factories
//! every in-tree caller writes by hand.
//!
//! Available whenever the `convex` or `qp` feature is on.

pub use pounce_feral::FeralSolverInterface;
pub use pounce_linsol::{Factorization, FactorizationError, SparseSymLinearSolverInterface};

/// The default backend: FERAL's sparse symmetric indefinite (LDLᵀ)
/// factorization, parallel inside a single factor.
///
/// Pass it as the factory argument — `solve_qp_ipm(&prob, &opts, backend)` —
/// or call it to build one boxed backend for
/// `qp::ParametricActiveSetSolver::new`.
///
/// ```
/// # #[cfg(feature = "convex")] {
/// use pounce_rs::convex::{QpOptions, QpProblem, QpStatus, Triplet, solve_qp_ipm};
/// use pounce_rs::linsol::backend;
///
/// // min ½·2·(x-1)² over 0 ≤ x ≤ 5  ⇒  x* = 1
/// let prob = QpProblem {
///     n: 1,
///     p_lower: vec![Triplet::new(0, 0, 2.0)],
///     c: vec![-2.0],
///     a: vec![],
///     b: vec![],
///     g: vec![],
///     h: vec![],
///     lb: vec![0.0],
///     ub: vec![5.0],
/// };
/// let sol = solve_qp_ipm(&prob, &QpOptions::default(), backend);
/// assert_eq!(sol.status, QpStatus::Optimal);
/// assert!((sol.x[0] - 1.0).abs() < 1e-6);
/// # }
/// ```
pub fn backend() -> Box<dyn SparseSymLinearSolverInterface> {
    Box::new(FeralSolverInterface::new())
}

/// The inner-serial backend, for outer-parallel batch solving.
///
/// `convex::solve_qp_batch_parallel` runs one instance per rayon worker;
/// handing each worker a backend that also parallelizes internally
/// oversubscribes the machine and is typically *slower* than the serial
/// factor. The serial and parallel FERAL drivers are bit-identical, so this
/// changes throughput only, never the answer.
pub fn serial_backend() -> Box<dyn SparseSymLinearSolverInterface> {
    Box::new(FeralSolverInterface::serial())
}
