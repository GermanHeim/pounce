//! Bipartite incidence graph between equality rows and variables.
//!
//! PR 2 of the auxiliary-presolve port (issue #53). Driven by the
//! inner TNLP's Jacobian sparsity plus the equality filter on
//! `(g_l, g_u)`. ripopt anchor:
//! `src/auxiliary_preprocessing.rs:2282-2318`.

use pounce_common::types::{Index, Number};
use pounce_nlp::tnlp::Linearity;

/// View into the data we need to build an [`EqualityIncidence`].
/// Decoupled from `TNLP` itself so this module stays unit-testable
/// without spinning up a full inner problem.
#[derive(Debug, Clone, Copy)]
pub struct ProbeView<'a> {
    /// Number of variables.
    pub n_vars: usize,
    /// Number of inner constraint rows.
    pub m_rows: usize,
    /// Inner Jacobian sparsity (one entry per structural nonzero).
    pub jac_irow: &'a [Index],
    pub jac_jcol: &'a [Index],
    /// Optional Jacobian values at a probe point. When provided,
    /// entries that evaluate to exactly `0.0` are dropped — this
    /// removes structural zeros that linearity hasn't already
    /// excluded.
    pub jac_values: Option<&'a [Number]>,
    pub g_l: &'a [Number],
    pub g_u: &'a [Number],
    /// Optional per-row linearity tags; unused by PR 2 but plumbed
    /// for PR 5's coupling classifier.
    pub linearity: Option<&'a [Linearity]>,
    /// `true` when the inner TNLP uses Fortran (1-based) indexing.
    pub one_based: bool,
    /// Tolerance for `g_l[i] == g_u[i]`. Rows tighter than this are
    /// treated as equalities.
    pub eq_tol: Number,
}

/// CSR-style bipartite adjacency: equality rows ↔ variables.
///
/// # Example
///
/// ```
/// use pounce_presolve::incidence::{EqualityIncidence, ProbeView};
///
/// // 2 rows × 2 vars, one equality (row 0), one inequality (row 1).
/// // Jacobian touches (0,0), (0,1), (1,0).
/// let irow = [0, 0, 1];
/// let jcol = [0, 1, 0];
/// let g_l = [1.0, 0.0];
/// let g_u = [1.0, 5.0]; // row 1 is g(x) ∈ [0, 5], not an equality.
/// let p = ProbeView {
///     n_vars: 2,
///     m_rows: 2,
///     jac_irow: &irow,
///     jac_jcol: &jcol,
///     jac_values: None,
///     g_l: &g_l,
///     g_u: &g_u,
///     linearity: None,
///     one_based: false,
///     eq_tol: 1e-12,
/// };
/// let inc = EqualityIncidence::from_probe(&p);
/// assert_eq!(inc.n_eq_rows(), 1);
/// assert_eq!(inc.neighbors(0), &[0, 1]);
/// ```
#[derive(Debug, Clone, Default)]
pub struct EqualityIncidence {
    /// Number of variables (columns) in the original problem.
    pub n_vars: usize,
    /// Inner-row indices of the equality rows, in ascending order.
    /// Length = `self.n_eq_rows()`.
    pub eq_row_inner_idx: Vec<usize>,
    /// CSR row pointers (length `n_eq_rows + 1`).
    pub adj_ptr: Vec<usize>,
    /// Sorted, deduped column indices per row.
    pub vars: Vec<usize>,
}

impl EqualityIncidence {
    /// Build an incidence graph from a probe.
    pub fn from_probe(p: &ProbeView<'_>) -> Self {
        // 1. Identify equality rows in inner-row index order.
        let mut eq_row_inner_idx: Vec<usize> = Vec::new();
        let mut inner_to_eq: Vec<Option<usize>> = vec![None; p.m_rows];
        for (i, slot) in inner_to_eq.iter_mut().enumerate() {
            if (p.g_u[i] - p.g_l[i]).abs() <= p.eq_tol {
                *slot = Some(eq_row_inner_idx.len());
                eq_row_inner_idx.push(i);
            }
        }
        let n_eq = eq_row_inner_idx.len();

        // 2. Bucket Jacobian entries by equality-row index.
        let mut per_row: Vec<Vec<usize>> = vec![Vec::new(); n_eq];
        let nnz = p.jac_irow.len();
        for k in 0..nnz {
            // Skip exact structural zeros when values are available.
            if let Some(vals) = p.jac_values {
                if vals[k] == 0.0 {
                    continue;
                }
            }
            let i = if p.one_based {
                (p.jac_irow[k] as isize - 1) as usize
            } else {
                p.jac_irow[k] as usize
            };
            if i >= p.m_rows {
                continue;
            }
            let Some(eq_k) = inner_to_eq[i] else { continue };
            let j = if p.one_based {
                (p.jac_jcol[k] as isize - 1) as usize
            } else {
                p.jac_jcol[k] as usize
            };
            if j >= p.n_vars {
                continue;
            }
            per_row[eq_k].push(j);
        }

        // 3. Sort + dedupe each row; pack into CSR.
        let mut adj_ptr: Vec<usize> = Vec::with_capacity(n_eq + 1);
        let mut vars: Vec<usize> = Vec::new();
        adj_ptr.push(0);
        for row in per_row.iter_mut() {
            row.sort_unstable();
            row.dedup();
            vars.extend_from_slice(row);
            adj_ptr.push(vars.len());
        }

        Self {
            n_vars: p.n_vars,
            eq_row_inner_idx,
            adj_ptr,
            vars,
        }
    }

    /// Number of equality rows in the incidence graph.
    pub fn n_eq_rows(&self) -> usize {
        self.eq_row_inner_idx.len()
    }

    /// Sorted column neighbours of equality row `k` (0-based into
    /// `eq_row_inner_idx`). Panics if `k >= self.n_eq_rows()`.
    pub fn neighbors(&self, k: usize) -> &[usize] {
        let lo = self.adj_ptr[k];
        let hi = self.adj_ptr[k + 1];
        &self.vars[lo..hi]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn probe<'a>(
        n_vars: usize,
        m_rows: usize,
        irow: &'a [Index],
        jcol: &'a [Index],
        g_l: &'a [Number],
        g_u: &'a [Number],
    ) -> ProbeView<'a> {
        ProbeView {
            n_vars,
            m_rows,
            jac_irow: irow,
            jac_jcol: jcol,
            jac_values: None,
            g_l,
            g_u,
            linearity: None,
            one_based: false,
            eq_tol: 1e-12,
        }
    }

    #[test]
    fn incidence_empty_problem_is_empty() {
        let p = probe(0, 0, &[], &[], &[], &[]);
        let inc = EqualityIncidence::from_probe(&p);
        assert_eq!(inc.n_eq_rows(), 0);
        assert_eq!(inc.vars.len(), 0);
        assert_eq!(inc.adj_ptr, vec![0]);
    }

    #[test]
    fn incidence_filters_inequalities() {
        // Row 0 is g = 1 (equality). Row 1 is g ∈ [0, 5] (range).
        let p = probe(2, 2, &[0, 0, 1], &[0, 1, 0], &[1.0, 0.0], &[1.0, 5.0]);
        let inc = EqualityIncidence::from_probe(&p);
        assert_eq!(inc.n_eq_rows(), 1);
        assert_eq!(inc.eq_row_inner_idx, vec![0]);
        assert_eq!(inc.neighbors(0), &[0, 1]);
    }

    #[test]
    fn incidence_dedupes_and_sorts_columns() {
        // Same equality row mentions column 1 twice and column 0
        // after column 1 — output must be sorted [0, 1] without dupes.
        let p = probe(2, 1, &[0, 0, 0, 0], &[1, 1, 0, 1], &[2.5], &[2.5]);
        let inc = EqualityIncidence::from_probe(&p);
        assert_eq!(inc.neighbors(0), &[0, 1]);
    }

    #[test]
    fn incidence_respects_fortran_indexing() {
        let mut p = probe(
            2,
            1,
            &[1, 1], // Fortran rows 1..=1
            &[1, 2], // Fortran cols 1..=2
            &[0.0],
            &[0.0],
        );
        p.one_based = true;
        let inc = EqualityIncidence::from_probe(&p);
        assert_eq!(inc.n_eq_rows(), 1);
        assert_eq!(inc.neighbors(0), &[0, 1]);
    }

    #[test]
    fn incidence_drops_structural_zeros_when_values_provided() {
        // Row 0 touches columns 0 and 1, but the (0, 1) entry has
        // value 0.0 at the probe point.
        let vals = [3.5, 0.0];
        let p = ProbeView {
            n_vars: 2,
            m_rows: 1,
            jac_irow: &[0, 0],
            jac_jcol: &[0, 1],
            jac_values: Some(&vals),
            g_l: &[1.0],
            g_u: &[1.0],
            linearity: None,
            one_based: false,
            eq_tol: 1e-12,
        };
        let inc = EqualityIncidence::from_probe(&p);
        assert_eq!(inc.neighbors(0), &[0]);
    }
}
