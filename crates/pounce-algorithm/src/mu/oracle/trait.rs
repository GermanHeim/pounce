//! `MuOracle` trait — port of `IpMuOracle.hpp`.

use pounce_common::types::Number;

pub trait MuOracle {
    /// Probe the next mu given the current iterate state. Phase 10
    /// fills in the actual signatures; trait surface here keeps the
    /// option matrix matchable in `AlgBuilder`.
    fn calculate_mu(&mut self) -> Option<Number>;
}
