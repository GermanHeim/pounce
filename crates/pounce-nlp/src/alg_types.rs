//! Algorithm-side termination codes.
//!
//! Mirrors `Interfaces/IpAlgTypes.hpp`. `SolverReturn` is what
//! `IpoptAlg::Optimize()` returns; `IpoptApplication` then translates
//! it (per the table in `MAIN_LOOP.md`) into a public-facing
//! [`crate::return_codes::ApplicationReturnStatus`].

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SolverReturn {
    Success,
    MaxiterExceeded,
    CpuTimeExceeded,
    WallTimeExceeded,
    StopAtTinyStep,
    StopAtAcceptablePoint,
    LocalInfeasibility,
    UserRequestedStop,
    FeasiblePointFound,
    DivergingIterates,
    RestorationFailure,
    ErrorInStepComputation,
    InvalidNumberDetected,
    TooFewDegreesOfFreedom,
    InvalidOption,
    OutOfMemory,
    InternalError,
    Unassigned,
}
