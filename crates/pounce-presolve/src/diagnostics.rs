//! Diagnostics for the auxiliary-equality preprocessing pass (Phase 0).
//!
//! Populated incrementally across PRs 2–9 of the auxiliary-presolve
//! port (issue #53). PR 1 lands the struct with `Default`-only fields
//! so the orchestrator can return a zero-valued instance from the
//! no-op path.

use pounce_common::types::{Index, Number};

/// Reasons the orchestrator may decline to eliminate a candidate block.
///
/// PR 1 wires the enum so it can live in the diagnostics struct; the
/// populating logic lands with PR 5 (coupling classification) and PR 6
/// (block solve).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AuxiliaryRejectionReason {
    /// Block too large for the lightweight Newton solver and no IPM
    /// fallback installed (PR 11).
    BlockTooLarge,
    /// Block is coupled to inequality rows or the objective in a way
    /// the current coupling policy disallows.
    CouplingDisallowed,
    /// Newton diverged or hit `presolve_auxiliary_max_iter`.
    BlockSolveDiverged,
    /// Full-space KKT residual after the candidate reduction exceeded
    /// `presolve_auxiliary_tol`.
    ResidualCheckFailed,
}

/// Per-run summary of what the auxiliary-equality preprocessing pass
/// did. All counters are zeroed by [`Default::default`].
///
/// # Example
///
/// ```
/// use pounce_presolve::diagnostics::AuxiliaryPreprocessingDiagnostics;
///
/// let d = AuxiliaryPreprocessingDiagnostics::default();
/// assert_eq!(d.blocks_eliminated, 0);
/// assert_eq!(d.vars_eliminated, 0);
/// assert!(d.rejection_reasons.is_empty());
/// ```
#[derive(Debug, Clone, Default)]
pub struct AuxiliaryPreprocessingDiagnostics {
    /// Number of blocks the orchestrator successfully eliminated.
    pub blocks_eliminated: Index,
    /// Variables fixed by the eliminated blocks (sum of block dims).
    pub vars_eliminated: Index,
    /// Equality rows dropped from the reduced problem.
    pub rows_eliminated: Index,
    /// Total wall time spent inside Phase 0, in milliseconds.
    pub total_time_ms: u128,
    /// Largest block-solve residual accepted under
    /// `presolve_auxiliary_tol`. `0.0` when nothing was eliminated.
    pub max_block_residual: Number,
    /// One entry per rejected candidate; populated by PR 5/6/8.
    pub rejection_reasons: Vec<AuxiliaryRejectionReason>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn diagnostics_default_is_empty() {
        let d = AuxiliaryPreprocessingDiagnostics::default();
        assert_eq!(d.blocks_eliminated, 0);
        assert_eq!(d.vars_eliminated, 0);
        assert_eq!(d.rows_eliminated, 0);
        assert_eq!(d.total_time_ms, 0);
        assert_eq!(d.max_block_residual, 0.0);
        assert!(d.rejection_reasons.is_empty());
    }
}
