//! Application-level return codes.
//!
//! Mirrors `Interfaces/IpReturnCodes.{h,hpp}` and `IpReturnCodes_inc.h`.
//! The integer values **must** match upstream — `pounce-cinterface`
//! uses `#[repr(i32)]` so the C ABI emits identical numeric codes for
//! drop-in compatibility with PyIpopt / cyipopt / JuMP.

use pounce_common::types::Index;

/// Mirrors `enum ApplicationReturnStatus`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[repr(i32)]
pub enum ApplicationReturnStatus {
    SolveSucceeded = 0,
    SolvedToAcceptableLevel = 1,
    InfeasibleProblemDetected = 2,
    SearchDirectionBecomesTooSmall = 3,
    DivergingIterates = 4,
    UserRequestedStop = 5,
    FeasiblePointFound = 6,

    MaximumIterationsExceeded = -1,
    RestorationFailed = -2,
    ErrorInStepComputation = -3,
    MaximumCpuTimeExceeded = -4,
    MaximumWallTimeExceeded = -5,

    NotEnoughDegreesOfFreedom = -10,
    InvalidProblemDefinition = -11,
    InvalidOption = -12,
    InvalidNumberDetected = -13,

    UnrecoverableException = -100,
    NonIpoptExceptionThrown = -101,
    InsufficientMemory = -102,
    InternalError = -199,
}

impl ApplicationReturnStatus {
    pub fn as_int(self) -> Index {
        self as Index
    }

    /// The upstream C enumerator spelling (`Solve_Succeeded`,
    /// `Infeasible_Problem_Detected`, …) from `IpReturnCodes_inc.h`.
    ///
    /// This is the name every consumer of Ipopt already keys off — CUTEst
    /// status tables, `benchmarks/scripts/run_nl_bench.sh`, the reference
    /// JSONs under `benchmarks/*/ipopt_ma57.json` — so it is what the CLI
    /// prints on its machine-readable `Status:` line. The Rust `Debug` name
    /// is *not* interchangeable with it: `Debug` gives `SolveSucceeded`, and
    /// anything comparing against upstream's tables would silently never
    /// match.
    pub fn upstream_name(self) -> &'static str {
        match self {
            Self::SolveSucceeded => "Solve_Succeeded",
            Self::SolvedToAcceptableLevel => "Solved_To_Acceptable_Level",
            Self::InfeasibleProblemDetected => "Infeasible_Problem_Detected",
            Self::SearchDirectionBecomesTooSmall => "Search_Direction_Becomes_Too_Small",
            Self::DivergingIterates => "Diverging_Iterates",
            Self::UserRequestedStop => "User_Requested_Stop",
            Self::FeasiblePointFound => "Feasible_Point_Found",
            Self::MaximumIterationsExceeded => "Maximum_Iterations_Exceeded",
            Self::RestorationFailed => "Restoration_Failed",
            Self::ErrorInStepComputation => "Error_In_Step_Computation",
            Self::MaximumCpuTimeExceeded => "Maximum_CpuTime_Exceeded",
            Self::MaximumWallTimeExceeded => "Maximum_WallTime_Exceeded",
            Self::NotEnoughDegreesOfFreedom => "Not_Enough_Degrees_Of_Freedom",
            Self::InvalidProblemDefinition => "Invalid_Problem_Definition",
            Self::InvalidOption => "Invalid_Option",
            Self::InvalidNumberDetected => "Invalid_Number_Detected",
            Self::UnrecoverableException => "Unrecoverable_Exception",
            Self::NonIpoptExceptionThrown => "NonIpopt_Exception_Thrown",
            Self::InsufficientMemory => "Insufficient_Memory",
            Self::InternalError => "Internal_Error",
        }
    }
}

/// Mirrors `enum AlgorithmMode`. Exposed in `intermediate_callback`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(i32)]
pub enum AlgorithmMode {
    RegularMode = 0,
    RestorationPhaseMode = 1,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// From `IpReturnCodes_inc.h` — these values are the C ABI
    /// contract for `pounce-cinterface`. Never change them.
    #[test]
    fn integer_values_match_upstream() {
        assert_eq!(ApplicationReturnStatus::SolveSucceeded.as_int(), 0);
        assert_eq!(ApplicationReturnStatus::SolvedToAcceptableLevel.as_int(), 1);
        assert_eq!(
            ApplicationReturnStatus::InfeasibleProblemDetected.as_int(),
            2
        );
        assert_eq!(
            ApplicationReturnStatus::SearchDirectionBecomesTooSmall.as_int(),
            3
        );
        assert_eq!(ApplicationReturnStatus::DivergingIterates.as_int(), 4);
        assert_eq!(ApplicationReturnStatus::UserRequestedStop.as_int(), 5);
        assert_eq!(ApplicationReturnStatus::FeasiblePointFound.as_int(), 6);

        assert_eq!(
            ApplicationReturnStatus::MaximumIterationsExceeded.as_int(),
            -1
        );
        assert_eq!(ApplicationReturnStatus::RestorationFailed.as_int(), -2);
        assert_eq!(ApplicationReturnStatus::ErrorInStepComputation.as_int(), -3);
        assert_eq!(ApplicationReturnStatus::MaximumCpuTimeExceeded.as_int(), -4);
        assert_eq!(
            ApplicationReturnStatus::MaximumWallTimeExceeded.as_int(),
            -5
        );

        assert_eq!(
            ApplicationReturnStatus::NotEnoughDegreesOfFreedom.as_int(),
            -10
        );
        assert_eq!(
            ApplicationReturnStatus::InvalidProblemDefinition.as_int(),
            -11
        );
        assert_eq!(ApplicationReturnStatus::InvalidOption.as_int(), -12);
        assert_eq!(ApplicationReturnStatus::InvalidNumberDetected.as_int(), -13);

        assert_eq!(
            ApplicationReturnStatus::UnrecoverableException.as_int(),
            -100
        );
        assert_eq!(
            ApplicationReturnStatus::NonIpoptExceptionThrown.as_int(),
            -101
        );
        assert_eq!(ApplicationReturnStatus::InsufficientMemory.as_int(), -102);
        assert_eq!(ApplicationReturnStatus::InternalError.as_int(), -199);

        assert_eq!(AlgorithmMode::RegularMode as i32, 0);
        assert_eq!(AlgorithmMode::RestorationPhaseMode as i32, 1);
    }

    const ALL_STATUSES: [ApplicationReturnStatus; 20] = [
        ApplicationReturnStatus::SolveSucceeded,
        ApplicationReturnStatus::SolvedToAcceptableLevel,
        ApplicationReturnStatus::InfeasibleProblemDetected,
        ApplicationReturnStatus::SearchDirectionBecomesTooSmall,
        ApplicationReturnStatus::DivergingIterates,
        ApplicationReturnStatus::UserRequestedStop,
        ApplicationReturnStatus::FeasiblePointFound,
        ApplicationReturnStatus::MaximumIterationsExceeded,
        ApplicationReturnStatus::RestorationFailed,
        ApplicationReturnStatus::ErrorInStepComputation,
        ApplicationReturnStatus::MaximumCpuTimeExceeded,
        ApplicationReturnStatus::MaximumWallTimeExceeded,
        ApplicationReturnStatus::NotEnoughDegreesOfFreedom,
        ApplicationReturnStatus::InvalidProblemDefinition,
        ApplicationReturnStatus::InvalidOption,
        ApplicationReturnStatus::InvalidNumberDetected,
        ApplicationReturnStatus::UnrecoverableException,
        ApplicationReturnStatus::NonIpoptExceptionThrown,
        ApplicationReturnStatus::InsufficientMemory,
        ApplicationReturnStatus::InternalError,
    ];

    /// The names consumers actually key off — CUTEst status tables, the
    /// benchmark driver's `Status:` scrape, `benchmarks/*/ipopt_ma57.json`.
    /// A typo here is silent: the label just never matches and the run is
    /// scored as something it was not.
    #[test]
    fn upstream_names_match_upstream_spelling() {
        assert_eq!(
            ApplicationReturnStatus::SolveSucceeded.upstream_name(),
            "Solve_Succeeded"
        );
        assert_eq!(
            ApplicationReturnStatus::InfeasibleProblemDetected.upstream_name(),
            "Infeasible_Problem_Detected"
        );
        assert_eq!(
            ApplicationReturnStatus::MaximumIterationsExceeded.upstream_name(),
            "Maximum_Iterations_Exceeded"
        );
        // Upstream's own inconsistent casing, preserved verbatim: `CpuTime`
        // and `WallTime` are one word each, `NonIpopt` likewise.
        assert_eq!(
            ApplicationReturnStatus::MaximumCpuTimeExceeded.upstream_name(),
            "Maximum_CpuTime_Exceeded"
        );
        assert_eq!(
            ApplicationReturnStatus::NonIpoptExceptionThrown.upstream_name(),
            "NonIpopt_Exception_Thrown"
        );
    }

    /// Every upstream name is its `Debug` name with underscores inserted —
    /// true of all twenty, including the odd-cased ones above. Checking the
    /// invariant over the whole enum catches a typo or a missed variant in
    /// the arm that the spot checks above do not cover.
    #[test]
    fn upstream_names_are_debug_names_with_separators() {
        for status in ALL_STATUSES {
            assert_eq!(
                status.upstream_name().replace('_', ""),
                format!("{status:?}"),
                "upstream name for {status:?} is not its Debug name with separators",
            );
        }
    }

    /// Distinct statuses must not collapse onto one label — that would make
    /// a scrape read a different outcome than the one that shipped.
    #[test]
    fn upstream_names_are_unique() {
        let mut names: Vec<&str> = ALL_STATUSES.iter().map(|s| s.upstream_name()).collect();
        names.sort_unstable();
        let before = names.len();
        names.dedup();
        assert_eq!(before, names.len(), "duplicate upstream status name");
    }
}
