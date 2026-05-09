//! Low-level sparse-symmetric backend interface — port of
//! `IpSparseSymLinearSolverInterface.hpp`.
//!
//! Concrete implementors:
//! * `pounce_hsl::Ma57SolverInterface` (v1.0).
//! * Future: MUMPS, FERAL.

use crate::status::ESymSolverStatus;
use pounce_common::types::{Index, Number};

/// Sparse matrix format that a backend wants its triplet/CSR data in.
/// Mirrors `SparseSymLinearSolverInterface::EMatrixFormat`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EMatrixFormat {
    /// Triplet (COO) of the lower triangle, 1-based indices
    /// (MA27 / MA57 / MUMPS convention).
    TripletFormat,
    /// CSR of the upper triangle, 0-based indices.
    CsrFormat0Offset,
    /// CSR of the upper triangle, 1-based indices.
    CsrFormat1Offset,
    /// Full CSR (lower + upper), 0-based indices.
    CsrFullFormat0Offset,
    /// Full CSR (lower + upper), 1-based indices.
    CsrFullFormat1Offset,
}

/// Backend-side trait. The lifecycle mirrors upstream's narrative
/// comment in `IpSparseSymLinearSolverInterface.hpp`:
///
/// 1. caller asks [`Self::matrix_format`].
/// 2. caller calls [`Self::initialize_structure`] once with `(ia, ja)`.
/// 3. caller takes the values pointer from
///    [`Self::values_array_mut`], fills it.
/// 4. caller calls [`Self::multi_solve`] with `new_matrix=true` for
///    each new value pattern.
/// 5. caller may query [`Self::number_of_neg_evals`] /
///    [`Self::increase_quality`] between solves.
///
/// `new_matrix=false` requests a back-substitution against the
/// existing factorization.
pub trait SparseSymLinearSolverInterface {
    /// Initialize backend internal structures for a matrix of given
    /// dimension and pattern.
    fn initialize_structure(
        &mut self,
        dim: Index,
        nonzeros: Index,
        ia: &[Index],
        ja: &[Index],
    ) -> ESymSolverStatus;

    /// Slice into which the caller writes the matrix nonzeros (in the
    /// same order as `ja` from [`Self::initialize_structure`]).
    fn values_array_mut(&mut self) -> &mut [Number];

    /// Factor (if `new_matrix`) and back-substitute against `nrhs`
    /// right-hand sides packed in `rhs_vals` (length `nrhs * dim`).
    /// Solutions overwrite `rhs_vals`.
    #[allow(clippy::too_many_arguments)]
    fn multi_solve(
        &mut self,
        new_matrix: bool,
        ia: &[Index],
        ja: &[Index],
        nrhs: Index,
        rhs_vals: &mut [Number],
        check_neg_evals: bool,
        number_of_neg_evals: Index,
    ) -> ESymSolverStatus;

    /// Number of negative eigenvalues found in the most recent
    /// factorization. Caller must check [`Self::provides_inertia`]
    /// first.
    fn number_of_neg_evals(&self) -> Index;

    /// Ask the backend to use a more accurate (but slower) pivot
    /// strategy on the next solve. Returns `false` if the maximum
    /// quality is already reached.
    fn increase_quality(&mut self) -> bool;

    /// Whether this backend reports the number of negative
    /// eigenvalues post-factor.
    fn provides_inertia(&self) -> bool;

    /// Required matrix layout. Caller marshals data into this format.
    fn matrix_format(&self) -> EMatrixFormat;

    /// Whether [`Self::determine_dependent_rows`] is supported.
    fn provides_degeneracy_detection(&self) -> bool {
        false
    }

    /// Find linearly dependent rows — used by Ipopt's degeneracy
    /// probe. Default is `FatalError` matching upstream's default
    /// implementation.
    fn determine_dependent_rows(
        &mut self,
        _ia: &[Index],
        _ja: &[Index],
        _c_deps: &mut Vec<Index>,
    ) -> ESymSolverStatus {
        ESymSolverStatus::FatalError
    }
}
