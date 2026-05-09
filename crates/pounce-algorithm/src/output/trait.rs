//! Iteration-output trait — port of `IpIterationOutput.hpp`.

use crate::ipopt_cq::IpoptCqHandle;
use crate::ipopt_data::IpoptDataHandle;

/// Strategy that emits one row of the iter-by-iter table. The default
/// `write_output` is a no-op so structural unit tests can drive the
/// algorithm without an output sink. Phase 7 ports `OrigIterationOutput`
/// with the full upstream column format.
pub trait IterationOutput {
    fn write_output(&mut self) {}

    /// Format the next iteration row into a fresh `String`, given the
    /// current data + CQ snapshots. Default returns an empty string.
    /// Mirrors `IpOrigIterationOutput::WriteOutput` minus the
    /// journalist write — callers route the returned line wherever
    /// they want (stdout, file, log buffer for tests).
    fn format_row(&mut self, _data: &IpoptDataHandle, _cq: &IpoptCqHandle) -> String {
        String::new()
    }
}
