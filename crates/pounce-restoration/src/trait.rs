//! Re-export of the [`RestorationPhase`] trait that lives in
//! `pounce-algorithm`. The trait is defined there (rather than here) so
//! that [`pounce_algorithm::ipopt_alg::IpoptAlgorithm`] can call into
//! it without forming a circular crate dependency
//! (`pounce-restoration` already depends on `pounce-algorithm`).
//!
//! Concrete restoration drivers (`MinC1NormRestoration`,
//! `RestoRestorationPhase`) `impl RestorationPhase for ...` against the
//! re-exported trait below.

pub use pounce_algorithm::restoration::{RestorationOutcome, RestorationPhase};
