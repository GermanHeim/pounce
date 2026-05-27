//! Phase-0 orchestrator for auxiliary-equality preprocessing.
//!
//! PR 1 — no-op shell. Even with `presolve_auxiliary=yes` this returns
//! [`AuxiliaryPreprocessingDiagnostics::default`] without touching the
//! row mask or the reduction stack. PRs 2–8 fill in the pipeline:
//! incidence → matching → Dulmage-Mendelsohn → BTF → coupling →
//! block solve → reduction frame → full-space residual check.
//!
//! Tracking issue: <https://github.com/jkitchin/pounce/issues/53>.

use crate::diagnostics::AuxiliaryPreprocessingDiagnostics;
use crate::options::PresolveOptions;
use crate::reduction_frame::ReductionStack;

/// Run the Phase-0 pass and return its diagnostics.
///
/// PR 1 always early-returns the default diagnostics. `row_kept_inner`
/// and `reduction_stack` are left untouched so the existing Phases 1–5
/// in [`crate::PresolveTnlp::ensure_init`] see the same row mask they
/// always have.
///
/// # Example
///
/// ```
/// use pounce_presolve::auxiliary::run_auxiliary_phase0;
/// use pounce_presolve::options::PresolveOptions;
/// use pounce_presolve::reduction_frame::ReductionStack;
///
/// let opts = PresolveOptions::defaults();
/// let mut row_kept = vec![true, true, true];
/// let mut stack = ReductionStack::default();
/// let d = run_auxiliary_phase0(&opts, &mut row_kept, &mut stack);
/// assert_eq!(d.blocks_eliminated, 0);
/// assert!(stack.is_empty());
/// assert!(row_kept.iter().all(|&k| k));
/// ```
pub fn run_auxiliary_phase0(
    _opts: &PresolveOptions,
    _row_kept_inner: &mut [bool],
    _reduction_stack: &mut ReductionStack,
) -> AuxiliaryPreprocessingDiagnostics {
    AuxiliaryPreprocessingDiagnostics::default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn noop_when_disabled() {
        let opts = PresolveOptions::defaults();
        let mut row_kept = vec![true, true];
        let mut stack = ReductionStack::default();
        let d = run_auxiliary_phase0(&opts, &mut row_kept, &mut stack);
        assert_eq!(d.blocks_eliminated, 0);
        assert_eq!(d.vars_eliminated, 0);
        assert_eq!(d.rows_eliminated, 0);
        assert!(stack.is_empty());
        assert!(row_kept.iter().all(|&k| k));
    }

    #[test]
    fn noop_when_enabled_pre_pr2() {
        // Even with the master switch on, PR 1 returns the default
        // diagnostics and leaves the row mask alone.
        let mut opts = PresolveOptions::defaults();
        opts.auxiliary = true;
        let mut row_kept = vec![true; 4];
        let mut stack = ReductionStack::default();
        let d = run_auxiliary_phase0(&opts, &mut row_kept, &mut stack);
        assert_eq!(d.blocks_eliminated, 0);
        assert!(stack.is_empty());
        assert_eq!(row_kept, vec![true; 4]);
    }
}
