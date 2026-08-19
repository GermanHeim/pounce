//! Symmetric-matrix scaling for triplet inputs.
//!
//! Port of `Algorithm/LinearSolvers/IpTSymScalingMethod.hpp`. A scaling
//! method takes the matrix in triplet form `(airn, ajcn, a)` and writes
//! a per-row scaling factor `s[i]` to `scaling_factors`. The
//! `TSymLinearSolver` wrapper then applies the symmetric scaling
//! `A' = diag(s) · A · diag(s)` (and the inverse to the RHS / solution)
//! before / after delegating to the backend.
//!
//! Variants registered upstream:
//!
//! * `none` — no scaling, default in many problem classes
//!   ([`IdentityScalingMethod`]).
//! * `mc19` — HSL MC19 row/column scaling. Bit-equivalence-default;
//!   implemented as `pounce_hsl::Mc19TSymScalingMethod` (FFI to
//!   `libcoinhsl.dylib`'s `mc19ad_`).
//! * `slack-based` — slack-aware scaling driven by the current
//!   barrier slacks. Implemented as
//!   `pounce_algorithm::kkt::SlackBasedTSymScalingMethod`; lives in
//!   the algorithm crate because it reads `IpoptData::curr` /
//!   `IpoptCq::curr_slack_*`, which would otherwise create a
//!   circular dependency.

use pounce_common::types::{Index, Number};

/// Backend-agnostic scaling method.
///
/// Returns `true` on success. On `false` the caller must skip scaling
/// (mirrors upstream's `ComputeSymTScalingFactors` contract).
pub trait TSymScalingMethod {
    fn compute_sym_t_scaling_factors(
        &mut self,
        n: Index,
        nnz: Index,
        airn: &[Index],
        ajcn: &[Index],
        a: &[Number],
        scaling_factors: &mut [Number],
    ) -> bool;

    /// Hand the method the per-iterate data it needs, ahead of the
    /// factorization. Default no-op: matrix-only methods (Ruiz, MC19,
    /// identity) derive everything from the triplets they are given and
    /// ignore this.
    ///
    /// Exists because upstream's slack-based method is an algorithm
    /// strategy object that reads `IpCq()` and `IpNLP()` directly, and
    /// this crate is below the algorithm and cannot. The iterate-derived
    /// part is computed by the caller and pushed in here; the block
    /// layout and the constant blocks stay in the method, where the
    /// upstream algorithm keeps them.
    ///
    /// `nx` is the primal dimension and `s_scale` the `s`-block factors;
    /// the remaining blocks are 1. Called at most once per iteration,
    /// not once per solve — the quantity is a function of the iterate,
    /// and several augmented solves share one iterate.
    fn set_slack_scaling(&mut self, _nx: Index, _s_scale: &[Number]) {}
}

/// `linear_system_scaling=none` — write identity scaling factors. The
/// `TSymLinearSolver` wrapper detects this case and skips the symmetric
/// scaling pass entirely; this implementation exists so that callers
/// who hand a scaling method unconditionally get a sensible default.
#[derive(Debug, Default, Clone, Copy)]
pub struct IdentityScalingMethod;

impl TSymScalingMethod for IdentityScalingMethod {
    fn compute_sym_t_scaling_factors(
        &mut self,
        n: Index,
        _nnz: Index,
        _airn: &[Index],
        _ajcn: &[Index],
        _a: &[Number],
        scaling_factors: &mut [Number],
    ) -> bool {
        debug_assert_eq!(scaling_factors.len(), n as usize);
        for s in scaling_factors.iter_mut() {
            *s = 1.0;
        }
        true
    }
}

/// `linear_system_scaling=slack-based` — port of
/// `IpSlackBasedTSymScalingMethod.cpp:ComputeSymTScalingFactors`.
///
/// The augmented system is ordered `[x | s | y_c | y_d]`, and upstream
/// writes
///
/// ```text
///   x block          1
///   s block          min(Pd_L · slack_s_L + Pd_U · slack_s_U, 1)
///   y_c, y_d blocks  1
/// ```
///
/// Only the `s` block depends on the iterate. It arrives through
/// [`TSymScalingMethod::set_slack_scaling`], computed by
/// `IpoptCq::curr_slack_based_s_scaling` — see that method for why the
/// split lands here rather than inside this type.
///
/// Until the first `set_slack_scaling` this behaves as identity, so a
/// factorization that happens before the algorithm has an iterate (the
/// least-square multiplier estimate at initialization, for instance) is
/// scaled the way `none` would scale it rather than by a stale or empty
/// vector.
#[derive(Debug, Default, Clone)]
pub struct SlackBasedTSymScalingMethod {
    nx: Index,
    s_scale: Vec<Number>,
}

impl SlackBasedTSymScalingMethod {
    pub fn new() -> Self {
        Self::default()
    }
}

impl TSymScalingMethod for SlackBasedTSymScalingMethod {
    fn set_slack_scaling(&mut self, nx: Index, s_scale: &[Number]) {
        self.nx = nx;
        self.s_scale.clear();
        self.s_scale.extend_from_slice(s_scale);
    }

    fn compute_sym_t_scaling_factors(
        &mut self,
        n: Index,
        _nnz: Index,
        _airn: &[Index],
        _ajcn: &[Index],
        _a: &[Number],
        scaling_factors: &mut [Number],
    ) -> bool {
        debug_assert_eq!(scaling_factors.len(), n as usize);
        for s in scaling_factors.iter_mut() {
            *s = 1.0;
        }
        // No iterate yet, or an augmented system this method was not
        // built for: identity is the honest answer, and it is what
        // `none` would have produced anyway.
        let nx = self.nx as usize;
        let ns = self.s_scale.len();
        if ns == 0 || nx + ns > n as usize {
            return true;
        }
        scaling_factors[nx..nx + ns].copy_from_slice(&self.s_scale);
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slack_based_is_identity_before_the_first_iterate() {
        // The initialization-time least-square solve factorizes before
        // any iterate exists. Scaling it with an empty vector would
        // silently leave the s block at whatever `scaling_factors`
        // happened to hold.
        let mut m = SlackBasedTSymScalingMethod::new();
        let mut f = vec![0.0; 6];
        assert!(m.compute_sym_t_scaling_factors(6, 0, &[], &[], &[], &mut f));
        assert_eq!(f, vec![1.0; 6]);
    }

    #[test]
    fn slack_based_writes_only_the_s_block() {
        // n = 6 laid out as [x x | s s | y y].
        let mut m = SlackBasedTSymScalingMethod::new();
        m.set_slack_scaling(2, &[0.25, 0.5]);
        let mut f = vec![0.0; 6];
        assert!(m.compute_sym_t_scaling_factors(6, 0, &[], &[], &[], &mut f));
        assert_eq!(f, vec![1.0, 1.0, 0.25, 0.5, 1.0, 1.0]);
    }

    #[test]
    fn slack_based_declines_a_system_it_does_not_fit() {
        // A shorter system than the stored blocks describe means this
        // method is being asked about something else (the restoration
        // sub-IPM's augmented system, say). Writing the s block anyway
        // would scale unrelated rows.
        let mut m = SlackBasedTSymScalingMethod::new();
        m.set_slack_scaling(4, &[0.25, 0.5]);
        let mut f = vec![0.0; 5];
        assert!(m.compute_sym_t_scaling_factors(5, 0, &[], &[], &[], &mut f));
        assert_eq!(f, vec![1.0; 5], "must fall back to identity, not misplace");
    }

    #[test]
    fn identity_writes_unit_factors() {
        let mut method = IdentityScalingMethod;
        let irn = [1, 2, 2];
        let jcn = [1, 1, 2];
        let vals = [2.0, 1.0, 3.0];
        let mut s = vec![0.0; 2];
        assert!(method.compute_sym_t_scaling_factors(2, 3, &irn, &jcn, &vals, &mut s));
        assert_eq!(s, &[1.0, 1.0]);
    }
}
