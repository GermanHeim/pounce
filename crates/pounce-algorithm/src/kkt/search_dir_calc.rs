//! Trait — port of `IpSearchDirCalculator.hpp`.

pub trait SearchDirCalculator {
    /// Compute the next search direction. Phase 7 wires the full
    /// signature once `IpoptData` carries `delta`/`delta_aff`.
    fn computed_normal_step(&self) -> bool {
        false
    }
}
