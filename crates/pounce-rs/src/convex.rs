//! LP, convex QP, and conic programming — the `pounce-convex` interior-point
//! path, re-exported (feature `convex`).
//!
//! ```toml
//! [dependencies]
//! pounce-rs = { version = "0.9", features = ["convex"] }
//! ```
//!
//! The problem is [`QpProblem`] in the standard form
//!
//! ```text
//! minimize    ½ xᵀP x + cᵀx
//! subject to  A x = b          (equality)
//!             G x ≤ h          (inequality)
//!             lb ≤ x ≤ ub      (first-class variable box)
//! ```
//!
//! with `P` supplied as its lower triangle in [`Triplet`] form (an empty `P`
//! is an LP). Cone blocks beyond the nonnegative orthant — second-order,
//! exponential, power, PSD — are declared with [`ConeSpec`] and solved by
//! [`solve_socp_ipm`].
//!
//! Every entry point takes a linear-solver factory; [`crate::linsol::backend`]
//! is the default one.
//!
//! ```
//! use pounce_rs::convex::{QpOptions, QpProblem, QpStatus, Triplet, solve_qp_ipm};
//! use pounce_rs::linsol::backend;
//!
//! // min ‖x‖² − 0.5·x0 − 1.5·x1  s.t.  x0 + x1 == 1,  0 ≤ x ≤ 5
//! let prob = QpProblem {
//!     n: 2,
//!     p_lower: vec![Triplet::new(0, 0, 2.0), Triplet::new(1, 1, 2.0)],
//!     c: vec![-0.5, -1.5],
//!     a: vec![Triplet::new(0, 0, 1.0), Triplet::new(0, 1, 1.0)],
//!     b: vec![1.0],
//!     g: vec![],
//!     h: vec![],
//!     lb: vec![0.0, 0.0],
//!     ub: vec![5.0, 5.0],
//! };
//!
//! let sol = solve_qp_ipm(&prob, &QpOptions::default(), backend);
//! assert_eq!(sol.status, QpStatus::Optimal);
//! assert!((sol.x[0] - 0.25).abs() < 1e-5 && (sol.x[1] - 0.75).abs() < 1e-5);
//! ```
//!
//! ## Batched solves
//!
//! [`solve_qp_batch`] solves a slice of independent instances serially (each
//! factor parallel internally); [`solve_qp_batch_parallel`] runs one instance
//! per rayon worker, which is the faster arrangement for many small QPs —
//! pass [`crate::linsol::serial_backend`] there so the workers don't
//! oversubscribe. [`solve_qp_batch_parallel_warm`] seeds each instance from a
//! [`QpWarmStart`], and [`solve_qp_multi_rhs`] handles one problem under many
//! right-hand sides.
//!
//! When the instances share a *fixed* sparsity pattern and differ only in
//! their numbers, [`QpFactorization`] performs the AMD ordering and symbolic
//! analysis once and reuses it across solves.
//!
//! ```
//! use pounce_rs::convex::{QpOptions, QpProblem, QpStatus, Triplet, solve_qp_batch_parallel};
//! use pounce_rs::linsol::serial_backend;
//!
//! // The same box QP under many linear terms: min ‖x‖² − 2·tᵀx, 0 ≤ x ≤ 1.
//! let probs: Vec<QpProblem> = [0.25, 0.5, 2.0]
//!     .iter()
//!     .map(|t| QpProblem {
//!         n: 1,
//!         p_lower: vec![Triplet::new(0, 0, 2.0)],
//!         c: vec![-2.0 * t],
//!         a: vec![],
//!         b: vec![],
//!         g: vec![],
//!         h: vec![],
//!         lb: vec![0.0],
//!         ub: vec![1.0],
//!     })
//!     .collect();
//!
//! let sols = solve_qp_batch_parallel(&probs, &QpOptions::default(), serial_backend);
//! assert!(sols.iter().all(|s| s.status == QpStatus::Optimal));
//! assert!((sols[0].x[0] - 0.25).abs() < 1e-6);
//! assert!((sols[2].x[0] - 1.0).abs() < 1e-6);      // clamped by the box
//! ```
//!
//! ## Parametric families — [`ActiveSetSession`]
//!
//! The batch and warm-start entry points above are the *interior-point*
//! answer to a family of nearby QPs. The **active-set** answer is
//! [`ActiveSetSession`]: a persistent handle that keeps the previous solve in
//! the engine's own coordinates and traces a homotopy to the next problem
//! instead of starting over, falling back to the full cold driver whenever
//! reuse is not valid. It also owns the convex → `pounce-qp` translation and
//! the presolve/postsolve wrapper, so a frontend reaching the active-set
//! engine no longer restates either (gh #769).
//!
//! Vary `c`, `b` or `h` and keep the structure fixed — that is the family the
//! homotopy interpolates.
//!
//! ```
//! use pounce_rs::convex::{ActiveSetSession, QpProblem, QpStatus, Reuse, Triplet};
//! use pounce_rs::linsol::backend;
//!
//! // min ‖x‖² − 2·tᵀx  s.t.  x0 + x1 ≤ 1,  0 ≤ x ≤ 5, swept over t.
//! let qp = |t: f64| QpProblem {
//!     n: 2,
//!     p_lower: vec![Triplet::new(0, 0, 2.0), Triplet::new(1, 1, 2.0)],
//!     c: vec![-2.0 * t, -2.0 * t],
//!     a: vec![],
//!     b: vec![],
//!     g: vec![Triplet::new(0, 0, 1.0), Triplet::new(0, 1, 1.0)],
//!     h: vec![1.0],
//!     lb: vec![0.0, 0.0],
//!     ub: vec![5.0, 5.0],
//! };
//!
//! let mut session = ActiveSetSession::new(backend);
//! for t in [0.2, 0.3, 0.4, 0.9] {
//!     let sol = session.solve(&qp(t));
//!     assert_eq!(sol.status, QpStatus::Optimal);
//! }
//! assert_eq!(session.last_reuse(), Reuse::Parametric);
//! ```
//!
//! ## Sensitivity
//!
//! [`QpSensitivity`] differentiates a solved QP with respect to its data and
//! [`ReducedHessian`] gives the curvature on the null space of the active
//! constraints — the QP counterpart of [`crate::sensitivity`] on the NLP
//! path.
//!
//! ## Beyond the curated surface
//!
//! [`pounce_convex`] itself is re-exported, so the modules not listed here
//! (`crossover`, `presolve`, `hsde`, `equilibrate`, …) stay reachable
//! without adding a dependency.

pub use pounce_convex::{
    ActiveSetOverrides, ActiveSetQp, ActiveSetSession, ConeSpec, NEG_INF, POS_INF, PolyProblem,
    Polynomial, PresolveNote, PsdCertificateError, QpFactorization, QpIterate, QpOptions,
    QpProblem, QpResiduals, QpSensitivity, QpSolution, QpStatus, QpWarmStart, ReducedHessian,
    Reuse, SensError, SessionStats, SosBound, SosSolution, Triplet, certify_psd_lower_triangle,
    solve_qp_active_set, solve_qp_batch, solve_qp_batch_parallel, solve_qp_batch_parallel_warm,
    solve_qp_ipm, solve_qp_ipm_debug, solve_qp_ipm_warm, solve_qp_multi_rhs,
    solve_qp_multi_rhs_parallel, solve_socp_ipm, solve_socp_ipm_debug, solve_socp_ipm_warm,
    sos_constrained_lower_bound, sos_constrained_lower_bound_opts, sos_lower_bound,
    sos_lower_bound_opts, sos_minimize, sos_minimize_opts, sos_opts,
};

/// The underlying crate, for anything not surfaced above.
pub use pounce_convex;
