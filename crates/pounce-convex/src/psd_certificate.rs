//! Conservative positive-semidefiniteness certificates for sparse matrices.

use std::collections::{BTreeMap, HashMap};

use pounce_common::types::Index;
use pounce_linsol::{Factorization, SparseSymLinearSolverInterface};

use crate::Triplet;

/// Certify that a sparse symmetric matrix is positive semidefinite within an
/// absolute tolerance.
///
/// `triplets` supplies the matrix's lower triangle in zero-based
/// coordinates (`row >= col`). Duplicate coordinates are summed. The
/// diagonal fast path accepts an eigenvalue down to `-tolerance`, while coupled
/// matrices are tested by factoring `A + tolerance * I` and inspecting its
/// inertia. A coupled matrix exactly on that shifted singular boundary may be
/// rejected conservatively when the backend cannot produce an inertia.
///
/// Empty and diagonal matrices are settled without constructing `make_backend`.
/// A coupled matrix is compressed to the variables present in its nonzero
/// pattern before it is factored, so structurally zero rows do not inflate the
/// factorization.
pub fn certify_psd_lower_triangle<F>(
    n: usize,
    triplets: &[Triplet],
    tolerance: f64,
    make_backend: F,
) -> bool
where
    F: FnOnce() -> Box<dyn SparseSymLinearSolverInterface>,
{
    if !tolerance.is_finite() || tolerance < 0.0 {
        return false;
    }

    let mut entries = HashMap::<(usize, usize), f64>::with_capacity(triplets.len());
    for &Triplet { row, col, val } in triplets {
        if row >= n || col >= n || row < col || !val.is_finite() {
            return false;
        }
        let sum = entries.entry((row, col)).or_default();
        *sum += val;
        if !sum.is_finite() {
            return false;
        }
    }
    entries.retain(|_, value| *value != 0.0);

    if entries.is_empty() {
        return true;
    }

    if entries.keys().all(|(row, col)| row == col) {
        return entries.values().all(|value| *value >= -tolerance);
    }

    let mut active = Vec::with_capacity(2 * entries.len());
    for &(row, col) in entries.keys() {
        active.push(row);
        active.push(col);
    }
    active.sort_unstable();
    active.dedup();

    let Ok(dim) = Index::try_from(active.len()) else {
        return false;
    };
    let mut shifted = BTreeMap::<(usize, usize), f64>::new();
    for ((row, col), value) in entries {
        let Ok(compressed_row) = active.binary_search(&row) else {
            return false;
        };
        let Ok(compressed_col) = active.binary_search(&col) else {
            return false;
        };
        shifted.insert((compressed_row, compressed_col), value);
    }

    for diagonal in 0..active.len() {
        let value = shifted.entry((diagonal, diagonal)).or_default();
        *value += tolerance;
        if !value.is_finite() {
            return false;
        }
    }
    if Index::try_from(shifted.len()).is_err() {
        return false;
    }

    let mut rows = Vec::with_capacity(shifted.len());
    let mut cols = Vec::with_capacity(shifted.len());
    let mut values = Vec::with_capacity(shifted.len());
    for ((row, col), value) in shifted {
        let Ok(row_1) = Index::try_from(row + 1) else {
            return false;
        };
        let Ok(col_1) = Index::try_from(col + 1) else {
            return false;
        };
        rows.push(row_1);
        cols.push(col_1);
        values.push(value);
    }

    let backend = make_backend();
    if !backend.provides_inertia() {
        return false;
    }
    let Ok(factor) = Factorization::new(dim, rows, cols, values, backend) else {
        return false;
    };
    factor.number_of_neg_evals() == Some(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use pounce_linsol::{EMatrixFormat, ESymSolverStatus};
    use std::cell::RefCell;
    use std::rc::Rc;

    fn feral_backend() -> Box<dyn SparseSymLinearSolverInterface> {
        Box::new(pounce_feral::FeralSolverInterface::serial())
    }

    #[test]
    fn empty_and_diagonal_matrices_do_not_construct_a_backend() {
        let no_backend = || -> Box<dyn SparseSymLinearSolverInterface> {
            panic!("the diagonal fast path must not construct a backend")
        };
        assert!(certify_psd_lower_triangle(4, &[], 1e-9, no_backend));

        let diagonal = [
            Triplet::new(0, 0, 3.0),
            Triplet::new(1, 1, -2.0),
            Triplet::new(1, 1, 2.0 - 5e-10),
        ];
        assert!(certify_psd_lower_triangle(2, &diagonal, 1e-9, || {
            panic!("the diagonal fast path must not construct a backend")
        }));
        assert!(!certify_psd_lower_triangle(2, &diagonal, 1e-11, || {
            panic!("the diagonal fast path must not construct a backend")
        }));
    }

    #[test]
    fn coupled_psd_and_indefinite_matrices_are_distinguished() {
        let singular_psd = [
            Triplet::new(0, 0, 1.0),
            Triplet::new(1, 0, 1.0),
            Triplet::new(1, 1, 1.0),
        ];
        assert!(certify_psd_lower_triangle(
            2,
            &singular_psd,
            1e-9,
            feral_backend
        ));

        let indefinite = [
            Triplet::new(0, 0, 2.0),
            Triplet::new(1, 0, 0.25),
            Triplet::new(1, 1, -1e-4),
        ];
        assert!(!certify_psd_lower_triangle(
            2,
            &indefinite,
            1e-9,
            feral_backend
        ));
    }

    #[test]
    fn coupled_tolerance_boundary_is_applied_to_the_spectrum() {
        const TOLERANCE: f64 = 1e-6;

        // [[1, 1 + delta], [1 + delta, 1]] has eigenvalues
        // -delta and 2 + delta.
        let matrix = |delta| {
            [
                Triplet::new(0, 0, 1.0),
                Triplet::new(1, 0, 1.0 + delta),
                Triplet::new(1, 1, 1.0),
            ]
        };
        assert!(certify_psd_lower_triangle(
            2,
            &matrix(0.5 * TOLERANCE),
            TOLERANCE,
            feral_backend
        ));
        assert!(!certify_psd_lower_triangle(
            2,
            &matrix(2.0 * TOLERANCE),
            TOLERANCE,
            feral_backend
        ));
    }

    #[test]
    fn duplicate_off_diagonals_are_summed_before_factorization() {
        // The off-diagonal duplicates cancel, leaving diag(1, 1).
        let matrix = [
            Triplet::new(0, 0, 1.0),
            Triplet::new(1, 0, 4.0),
            Triplet::new(1, 0, -4.0),
            Triplet::new(1, 1, 1.0),
        ];
        assert!(certify_psd_lower_triangle(2, &matrix, 0.0, || {
            panic!("cancelled coupling should use the diagonal fast path")
        }));
    }

    #[test]
    fn malformed_inputs_are_not_certified() {
        let cases = [
            (vec![Triplet::new(0, 1, 1.0)], 1e-9),
            (vec![Triplet::new(2, 0, 1.0)], 1e-9),
            (vec![Triplet::new(0, 0, f64::NAN)], 1e-9),
            (vec![Triplet::new(0, 0, 1.0)], -1.0),
            (vec![Triplet::new(0, 0, 1.0)], f64::INFINITY),
        ];
        for (matrix, tolerance) in cases {
            assert!(!certify_psd_lower_triangle(2, &matrix, tolerance, || {
                panic!("invalid input must not construct a backend")
            }));
        }
    }

    struct MockBackend {
        values: Vec<f64>,
        solve_status: ESymSolverStatus,
        inertia: Option<Index>,
        panic_on_initialize: bool,
    }

    impl SparseSymLinearSolverInterface for MockBackend {
        fn initialize_structure(
            &mut self,
            _dim: Index,
            nonzeros: Index,
            _ia: &[Index],
            _ja: &[Index],
        ) -> ESymSolverStatus {
            assert!(
                !self.panic_on_initialize,
                "a backend without inertia must not be initialized"
            );
            self.values.resize(nonzeros as usize, 0.0);
            ESymSolverStatus::Success
        }

        fn values_array_mut(&mut self) -> &mut [f64] {
            &mut self.values
        }

        fn multi_solve(
            &mut self,
            _new_matrix: bool,
            _ia: &[Index],
            _ja: &[Index],
            _nrhs: Index,
            _rhs_vals: &mut [f64],
            _check_neg_evals: bool,
            _number_of_neg_evals: Index,
        ) -> ESymSolverStatus {
            self.solve_status
        }

        fn number_of_neg_evals(&self) -> Index {
            self.inertia.unwrap_or_default()
        }

        fn increase_quality(&mut self) -> bool {
            false
        }

        fn provides_inertia(&self) -> bool {
            self.inertia.is_some()
        }

        fn matrix_format(&self) -> EMatrixFormat {
            EMatrixFormat::TripletFormat
        }
    }

    #[derive(Debug, Default)]
    struct BackendRecord {
        dim: Option<Index>,
        rows: Vec<Index>,
        cols: Vec<Index>,
        values: Vec<f64>,
    }

    struct RecordingBackend {
        values: Vec<f64>,
        record: Rc<RefCell<BackendRecord>>,
    }

    impl SparseSymLinearSolverInterface for RecordingBackend {
        fn initialize_structure(
            &mut self,
            dim: Index,
            nonzeros: Index,
            ia: &[Index],
            ja: &[Index],
        ) -> ESymSolverStatus {
            self.values.resize(nonzeros as usize, 0.0);
            let mut record = self.record.borrow_mut();
            record.dim = Some(dim);
            record.rows = ia.to_vec();
            record.cols = ja.to_vec();
            ESymSolverStatus::Success
        }

        fn values_array_mut(&mut self) -> &mut [f64] {
            &mut self.values
        }

        fn multi_solve(
            &mut self,
            _new_matrix: bool,
            _ia: &[Index],
            _ja: &[Index],
            _nrhs: Index,
            _rhs_vals: &mut [f64],
            _check_neg_evals: bool,
            _number_of_neg_evals: Index,
        ) -> ESymSolverStatus {
            self.record.borrow_mut().values = self.values.clone();
            ESymSolverStatus::Success
        }

        fn number_of_neg_evals(&self) -> Index {
            0
        }

        fn increase_quality(&mut self) -> bool {
            false
        }

        fn provides_inertia(&self) -> bool {
            true
        }

        fn matrix_format(&self) -> EMatrixFormat {
            EMatrixFormat::TripletFormat
        }
    }

    #[test]
    fn coupled_matrix_is_normalized_compressed_shifted_and_one_based() {
        let matrix = [
            Triplet::new(4, 4, 3.0),
            Triplet::new(4, 1, 0.75),
            Triplet::new(7, 7, 4.0),
            Triplet::new(1, 1, 2.0),
            Triplet::new(4, 1, 0.25),
        ];
        let record = Rc::new(RefCell::new(BackendRecord::default()));
        let backend_record = Rc::clone(&record);

        assert!(certify_psd_lower_triangle(8, &matrix, 0.5, move || {
            Box::new(RecordingBackend {
                values: Vec::new(),
                record: backend_record,
            })
        }));

        let record = record.borrow();
        assert_eq!(record.dim, Some(3));
        assert_eq!(record.rows, [1, 2, 2, 3]);
        assert_eq!(record.cols, [1, 1, 2, 3]);
        assert_eq!(record.values, [2.5, 1.0, 3.5, 4.5]);
    }

    fn coupled_matrix() -> [Triplet; 3] {
        [
            Triplet::new(0, 0, 2.0),
            Triplet::new(1, 0, 0.1),
            Triplet::new(1, 1, 2.0),
        ]
    }

    #[test]
    fn missing_inertia_and_factorization_failures_are_inconclusive() {
        let make_mock = |solve_status, inertia| {
            move || -> Box<dyn SparseSymLinearSolverInterface> {
                Box::new(MockBackend {
                    values: vec![],
                    solve_status,
                    inertia,
                    panic_on_initialize: inertia.is_none(),
                })
            }
        };
        assert!(!certify_psd_lower_triangle(
            2,
            &coupled_matrix(),
            1e-9,
            make_mock(ESymSolverStatus::Success, None),
        ));
        assert!(!certify_psd_lower_triangle(
            2,
            &coupled_matrix(),
            1e-9,
            make_mock(ESymSolverStatus::Singular, Some(0)),
        ));
    }
}
