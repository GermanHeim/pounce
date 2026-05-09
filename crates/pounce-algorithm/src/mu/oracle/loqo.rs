//! LOQO mu oracle — port of `IpLoqoMuOracle.{hpp,cpp}`. Phase 10.
//!
//! The LOQO rule chooses
//!
//! ```text
//!   sigma = 0.1 * min(factor * (1 - xi) / xi, 2)^3
//!   mu_new = clamp(sigma * avrg_compl, mu_min, mu_max)
//! ```
//!
//! where `xi = curr_centrality_measure() = min_compl / avrg_compl` is
//! a measure of how far the current iterate is from uniform
//! complementarity, and `factor = 0.05` per upstream's hard-coded
//! choice (`IpLoqoMuOracle.cpp:52`).

use crate::mu::oracle::r#trait::MuOracle;
use pounce_common::types::Number;

pub struct LoqoMuOracle {
    pub mu_min: Number,
    pub mu_max: Number,
    /// Latest cached `(avrg_compl, centrality_xi)` from
    /// `IpoptCalculatedQuantities`. The full plumbing wires this in
    /// after the CQ port lands; here we expose a setter so the
    /// arithmetic can be unit-tested in isolation.
    pub avrg_compl: Number,
    pub centrality_xi: Number,
}

impl Default for LoqoMuOracle {
    fn default() -> Self {
        Self {
            mu_min: 1e-11,
            mu_max: 1e5,
            avrg_compl: 1.0,
            centrality_xi: 1.0,
        }
    }
}

impl LoqoMuOracle {
    pub fn new() -> Self {
        Self::default()
    }

    /// Pure-arithmetic LOQO formula. Exposed standalone so it can be
    /// validated against upstream's `IpLoqoMuOracle.cpp:43-65`
    /// captures.
    pub fn loqo_mu(avrg_compl: Number, centrality_xi: Number) -> Number {
        let factor: Number = 0.05;
        let xi = centrality_xi.max(Number::MIN_POSITIVE);
        let bracket = (factor * (1.0 - xi) / xi).min(2.0);
        let sigma = 0.1 * bracket.powi(3);
        sigma * avrg_compl
    }
}

impl MuOracle for LoqoMuOracle {
    fn calculate_mu(&mut self) -> Option<Number> {
        let raw = Self::loqo_mu(self.avrg_compl, self.centrality_xi);
        Some(raw.clamp(self.mu_min, self.mu_max))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loqo_at_uniform_complementarity_is_zero() {
        // xi = 1 (uniform) → bracket = 0 → sigma = 0 → mu = 0.
        assert_eq!(LoqoMuOracle::loqo_mu(1.0, 1.0), 0.0);
    }

    #[test]
    fn loqo_caps_bracket_at_two() {
        // xi very small → bracket = min(0.05*(1-eps)/eps, 2) = 2.
        // sigma = 0.1 * 8 = 0.8. mu = 0.8 * avrg_compl.
        let m = LoqoMuOracle::loqo_mu(0.5, 1e-10);
        assert!((m - 0.4).abs() < 1e-13);
    }

    #[test]
    fn loqo_intermediate_xi() {
        // xi = 0.5 → factor*(1-0.5)/0.5 = 0.05; bracket = 0.05.
        // sigma = 0.1 * 1.25e-4 = 1.25e-5; mu = 1.25e-5 * 1 = 1.25e-5.
        let m = LoqoMuOracle::loqo_mu(1.0, 0.5);
        assert!((m - 1.25e-5).abs() < 1e-15);
    }

    #[test]
    fn calculate_mu_clamps_to_band() {
        let mut o = LoqoMuOracle {
            mu_min: 1.0,
            mu_max: 2.0,
            avrg_compl: 1.0,
            centrality_xi: 1.0, // raw = 0
        };
        assert_eq!(o.calculate_mu(), Some(1.0));
    }
}
