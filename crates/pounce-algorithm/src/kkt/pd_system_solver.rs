//! Trait — port of `IpPDSystemSolver.hpp`. The 8-block primal-dual
//! system solver. Implementations: [`super::pd_full_space_solver`].

use pounce_linsol::ESymSolverStatus;

pub trait PdSystemSolver {
    /// Run the configured iterative-refinement loop on the 8-block
    /// system. Phase 6 placeholder for the full-signature method;
    /// `super::pd_full_space_solver::PdFullSpaceSolver::solve` will
    /// implement it once `IteratesVector` is wired through.
    fn solve_status(&self) -> ESymSolverStatus;
}
