//! Engine-agnostic core of POUNCE's sensitivity layer.
//!
//! This crate holds the parts of the sIPOPT port that do not know which
//! solver produced the KKT system they are reading. Everything here is
//! written against one trait — [`backsolver::SensBacksolver`], whose whole
//! required surface is `dim()` and `solve(rhs, lhs)` — so any engine that can
//! back-solve against its converged factor gets the parametric machinery for
//! free:
//!
//! * [`boundcheck`] — fix-relax refinement, path following through
//!   active-set breakpoints, and the directional derivative at a kink.
//! * [`rowlimit`] — an observer block that gives a limit written as a
//!   *constraint row* the primal coordinate `boundcheck` decides in,
//!   for engines whose KKT has no slack block (gh#929).
//! * [`sens_app`] — the sIPOPT `SensApplication` driver, the reduced-Hessian
//!   entry point, and the option registrations.
//! * [`p_calculator`], [`schur_data`], [`schur_driver`], [`step_calc`],
//!   [`reduced_hessian`] — the `P = K⁻¹A` / Schur-complement stack.
//!
//! Two consumers exist in tree. `pounce-sensitivity` implements the trait over
//! the NLP filter-IPM's KKT factor (`PdSensBacksolver`); `pounce-convex`
//! implements it over the convex active-set KKT. Neither depends on the other,
//! and this crate depends on neither — it needs only `pounce-common` and
//! `pounce-linalg`.
//!
//! # Barrier iterates
//!
//! Two parts of the NLP arm need more of the engine than a back-solve: the
//! corrector, which iterates on the barrier KKT system against an operator
//! assembled at the predicted point, and the activity classification, which
//! reads the bound multipliers, slacks and barrier diagonal of the converged
//! point. Both are written against a second trait,
//! [`barrier::BarrierIterate`], which an engine implements beside
//! `SensBacksolver`:
//!
//! * [`corrector`]: Newton iterations on the barrier system from a
//!   parametric step.
//! * [`activity`]: the per-bound classification, and its refinement by the
//!   reduced curvature.
//!
//! `pounce-sensitivity` implements `BarrierIterate` over pounce's own
//! converged iterate. An interior-point engine outside pounce, such as Ipopt,
//! implements it from its own final iterate, its own model callbacks and its
//! own factorization of the KKT matrix, and gets the same two analyses over
//! its result.
//!
//! # Provenance
//!
//! Port of upstream Ipopt's `contrib/sIPOPT/` (Pirnay, López-Negrete &
//! Biegler 2012, DOI [10.1007/s12532-012-0043-2]). The module names mirror
//! upstream's file names so the two can be read side by side.
//!
//! [10.1007/s12532-012-0043-2]: https://doi.org/10.1007/s12532-012-0043-2

#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]

pub mod activity;
pub mod activity_kernel;
pub mod backsolver;
pub mod barrier;
pub mod boundcheck;
pub mod converged;
pub mod corrector;
pub mod dense;
pub mod index;
pub mod model;
pub mod p_calculator;
pub mod reduced_hessian;
pub mod rowlimit;
pub mod schur_data;
pub mod schur_driver;
pub mod sens_app;
pub mod stats;
pub mod step;
pub mod step_calc;

// Root re-exports. These are not a convenience: several moved modules refer to
// each other through the crate root (`&dyn crate::SchurData` appears in
// `schur_driver`'s public trait signature), so removing one of these breaks
// compilation rather than merely lengthening a path. `pounce-sensitivity`
// re-exports the same names, which is what keeps its published API unchanged
// across this extraction.
pub use backsolver::{DenseLuBacksolver, SensBacksolver};
pub use barrier::{ActivityData, BarrierIterate, BoundSide, EngineScaling, KktOperator, ModelMaps};
pub use p_calculator::{IndexPCalculator, PCalculator};
pub use reduced_hessian::compute_reduced_hessian;
pub use schur_data::{IndexSchurData, SchurData};
pub use schur_driver::{DenseGenSchurDriver, SchurDriver};
pub use sens_app::{SensApplication, SensOptions, register_options};
pub use step_calc::{SensStepCalc, StdStepCalc, WithBacksolver};
