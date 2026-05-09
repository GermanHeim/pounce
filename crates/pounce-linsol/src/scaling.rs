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

#[cfg(test)]
mod tests {
    use super::*;

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
