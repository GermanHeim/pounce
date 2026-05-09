//! Linear-solver return status — port of `ESymSolverStatus`
//! in `IpSymLinearSolver.hpp`.

/// Outcome of a single call into a [`crate::SymLinearSolver`] or
/// [`crate::SparseSymLinearSolverInterface`].
///
/// Variant order and semantics match upstream Ipopt's
/// `enum ESymSolverStatus` so the algorithm-side state machines port
/// 1:1.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ESymSolverStatus {
    /// Successful solve. Mirrors `SYMSOLVER_SUCCESS`.
    Success,
    /// Matrix is singular; solve was aborted. Mirrors
    /// `SYMSOLVER_SINGULAR`.
    Singular,
    /// Inertia mismatch — `numberOfNegEVals` did not match.
    /// Mirrors `SYMSOLVER_WRONG_INERTIA`.
    WrongInertia,
    /// Backend asks the caller to refill the values array and call
    /// again with the same RHS (e.g. MA57 after growing factor
    /// arrays). Mirrors `SYMSOLVER_CALL_AGAIN`.
    CallAgain,
    /// Unrecoverable error — the optimization should abort.
    /// Mirrors `SYMSOLVER_FATAL_ERROR`.
    FatalError,
}
