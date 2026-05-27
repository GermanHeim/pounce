//! Dulmage-Mendelsohn partition into under- / square- / overdetermined
//! parts.
//!
//! PR 3 of the auxiliary-presolve port (issue #53). Takes the maximum
//! matching produced by PR 2's Hopcroft-Karp and slices both the row
//! set and the column set into three coarse pieces:
//!
//! - **Over** — rows reachable from unmatched rows via alternating
//!   paths (non-matching edge into a column, then matching edge back
//!   to a row). These rows form the overdetermined block: more
//!   equations than variables in the slice.
//! - **Under** — columns reachable from unmatched columns via the
//!   symmetric alternating walk. Underdetermined block.
//! - **Square** — everything else. Has a perfect matching restricted
//!   to it; PR 3's [`crate::components`] further breaks this into
//!   independent connected components that become candidate blocks
//!   in PR 8.
//!
//! ripopt anchor: `src/auxiliary_preprocessing.rs:2320-2413`.

use std::collections::VecDeque;

use crate::incidence::EqualityIncidence;
use crate::matching::BipartiteMatching;

/// Which DM part a single row or column lives in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DMPart {
    Over,
    Square,
    Under,
}

/// Coarse Dulmage-Mendelsohn partition of an equality-row × variable
/// bipartite graph.
#[derive(Debug, Clone, Default)]
pub struct DulmageMendelsohnPartition {
    /// Per-row part assignment; `row_part.len() == inc.n_eq_rows()`.
    pub row_part: Vec<DMPart>,
    /// Per-column part assignment; `col_part.len() == inc.n_vars`.
    pub col_part: Vec<DMPart>,
    pub over_rows: Vec<usize>,
    pub over_cols: Vec<usize>,
    pub square_rows: Vec<usize>,
    pub square_cols: Vec<usize>,
    pub under_rows: Vec<usize>,
    pub under_cols: Vec<usize>,
}

impl DulmageMendelsohnPartition {
    /// Build the partition from an incidence graph and a maximum
    /// matching.
    ///
    /// # Example
    ///
    /// ```
    /// use pounce_presolve::incidence::{EqualityIncidence, ProbeView};
    /// use pounce_presolve::matching::hopcroft_karp;
    /// use pounce_presolve::dulmage_mendelsohn::{DulmageMendelsohnPartition, DMPart};
    ///
    /// // 3 equality rows × 3 vars, each row touching one distinct
    /// // var → perfect matching, all square.
    /// let p = ProbeView {
    ///     n_vars: 3,
    ///     m_rows: 3,
    ///     jac_irow: &[0, 1, 2],
    ///     jac_jcol: &[0, 1, 2],
    ///     jac_values: None,
    ///     g_l: &[0.0, 0.0, 0.0],
    ///     g_u: &[0.0, 0.0, 0.0],
    ///     linearity: None,
    ///     one_based: false,
    ///     eq_tol: 1e-12,
    /// };
    /// let inc = EqualityIncidence::from_probe(&p);
    /// let m = hopcroft_karp(&inc);
    /// let dm = DulmageMendelsohnPartition::from_matching(&inc, &m);
    /// assert_eq!(dm.square_rows.len(), 3);
    /// assert!(dm.over_rows.is_empty() && dm.under_cols.is_empty());
    /// assert!(dm.row_part.iter().all(|p| *p == DMPart::Square));
    /// ```
    pub fn from_matching(inc: &EqualityIncidence, m: &BipartiteMatching) -> Self {
        let n_rows = inc.n_eq_rows();
        let n_vars = inc.n_vars;

        let mut row_part = vec![DMPart::Square; n_rows];
        let mut col_part = vec![DMPart::Square; n_vars];

        // --- Over: alternating BFS from unmatched rows ---------------
        // Walk: row → col via any incidence edge, col → row via the
        // matching edge. Mark every visited row and col as Over.
        let mut row_seen = vec![false; n_rows];
        let mut col_seen = vec![false; n_vars];
        let mut queue: VecDeque<usize> = VecDeque::new();
        for (r, m_to) in m.row_to_var.iter().enumerate() {
            if m_to.is_none() {
                row_seen[r] = true;
                row_part[r] = DMPart::Over;
                queue.push_back(r);
            }
        }
        while let Some(r) = queue.pop_front() {
            for &v in inc.neighbors(r) {
                if col_seen[v] {
                    continue;
                }
                col_seen[v] = true;
                col_part[v] = DMPart::Over;
                if let Some(r2) = m.var_to_row[v] {
                    if !row_seen[r2] {
                        row_seen[r2] = true;
                        row_part[r2] = DMPart::Over;
                        queue.push_back(r2);
                    }
                }
                // If the column is unmatched, it can't extend the
                // walk: there is no matching edge back to a row.
                // (And an unmatched col reachable from an unmatched
                // row would be an augmenting path, which a max
                // matching forbids — so this branch is unreachable
                // when `m` is maximum.)
            }
        }

        // --- Under: symmetric BFS from unmatched columns -------------
        // Walk: col → row via any incidence edge, row → col via the
        // matching edge. Need the reverse adjacency (col → rows).
        let mut col_adj_ptr = vec![0usize; n_vars + 1];
        for k in 0..n_rows {
            for &v in inc.neighbors(k) {
                col_adj_ptr[v + 1] += 1;
            }
        }
        for i in 1..=n_vars {
            col_adj_ptr[i] += col_adj_ptr[i - 1];
        }
        let mut col_adj = vec![0usize; col_adj_ptr[n_vars]];
        let mut cursor = col_adj_ptr[..n_vars].to_vec();
        for k in 0..n_rows {
            for &v in inc.neighbors(k) {
                col_adj[cursor[v]] = k;
                cursor[v] += 1;
            }
        }

        let mut row_seen_u = vec![false; n_rows];
        let mut col_seen_u = vec![false; n_vars];
        let mut q2: VecDeque<usize> = VecDeque::new();
        for (v, m_to) in m.var_to_row.iter().enumerate() {
            if m_to.is_none() {
                col_seen_u[v] = true;
                // An unmatched col could already have been claimed by
                // Over; in a max matching that won't happen, but stay
                // defensive.
                if col_part[v] != DMPart::Over {
                    col_part[v] = DMPart::Under;
                }
                q2.push_back(v);
            }
        }
        while let Some(v) = q2.pop_front() {
            let lo = col_adj_ptr[v];
            let hi = col_adj_ptr[v + 1];
            for &r in &col_adj[lo..hi] {
                if row_seen_u[r] {
                    continue;
                }
                row_seen_u[r] = true;
                if row_part[r] != DMPart::Over {
                    row_part[r] = DMPart::Under;
                }
                if let Some(v2) = m.row_to_var[r] {
                    if !col_seen_u[v2] {
                        col_seen_u[v2] = true;
                        if col_part[v2] != DMPart::Over {
                            col_part[v2] = DMPart::Under;
                        }
                        q2.push_back(v2);
                    }
                }
            }
        }

        // --- Collect lists -------------------------------------------
        let mut over_rows = Vec::new();
        let mut square_rows = Vec::new();
        let mut under_rows = Vec::new();
        for (r, part) in row_part.iter().enumerate() {
            match part {
                DMPart::Over => over_rows.push(r),
                DMPart::Square => square_rows.push(r),
                DMPart::Under => under_rows.push(r),
            }
        }
        let mut over_cols = Vec::new();
        let mut square_cols = Vec::new();
        let mut under_cols = Vec::new();
        for (v, part) in col_part.iter().enumerate() {
            match part {
                DMPart::Over => over_cols.push(v),
                DMPart::Square => square_cols.push(v),
                DMPart::Under => under_cols.push(v),
            }
        }

        Self {
            row_part,
            col_part,
            over_rows,
            over_cols,
            square_rows,
            square_cols,
            under_rows,
            under_cols,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matching::hopcroft_karp;

    /// Build an `EqualityIncidence` directly from an edge list.
    fn eq_inc(n_vars: usize, n_rows: usize, edges: &[(usize, usize)]) -> EqualityIncidence {
        let mut per_row: Vec<Vec<usize>> = vec![Vec::new(); n_rows];
        for &(r, v) in edges {
            per_row[r].push(v);
        }
        let mut adj_ptr = Vec::with_capacity(n_rows + 1);
        let mut vars = Vec::new();
        adj_ptr.push(0);
        for row in per_row.iter_mut() {
            row.sort_unstable();
            row.dedup();
            vars.extend_from_slice(row);
            adj_ptr.push(vars.len());
        }
        EqualityIncidence {
            n_vars,
            eq_row_inner_idx: (0..n_rows).collect(),
            adj_ptr,
            vars,
        }
    }

    #[test]
    fn dm_empty_graph() {
        let inc = eq_inc(0, 0, &[]);
        let m = hopcroft_karp(&inc);
        let dm = DulmageMendelsohnPartition::from_matching(&inc, &m);
        assert!(dm.over_rows.is_empty());
        assert!(dm.square_rows.is_empty());
        assert!(dm.under_rows.is_empty());
        assert!(dm.over_cols.is_empty());
        assert!(dm.square_cols.is_empty());
        assert!(dm.under_cols.is_empty());
    }

    #[test]
    fn dm_all_square_3x3() {
        let inc = eq_inc(3, 3, &[(0, 0), (1, 1), (2, 2)]);
        let m = hopcroft_karp(&inc);
        let dm = DulmageMendelsohnPartition::from_matching(&inc, &m);
        assert_eq!(dm.square_rows, vec![0, 1, 2]);
        assert_eq!(dm.square_cols, vec![0, 1, 2]);
        assert!(dm.over_rows.is_empty());
        assert!(dm.under_cols.is_empty());
        assert!(dm.row_part.iter().all(|p| *p == DMPart::Square));
        assert!(dm.col_part.iter().all(|p| *p == DMPart::Square));
    }

    #[test]
    fn dm_pure_overdetermined_3x2() {
        // 3 rows × 2 cols, fully connected. Matching size = 2.
        let inc = eq_inc(2, 3, &[(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)]);
        let m = hopcroft_karp(&inc);
        assert_eq!(m.size, 2);
        let dm = DulmageMendelsohnPartition::from_matching(&inc, &m);
        // One row stays unmatched; alternating BFS reaches every row
        // and every col → all over.
        assert_eq!(dm.over_rows.len(), 3);
        assert_eq!(dm.over_cols.len(), 2);
        assert!(dm.square_rows.is_empty());
        assert!(dm.under_rows.is_empty());
    }

    #[test]
    fn dm_pure_underdetermined_2x3() {
        // 2 rows × 3 cols, fully connected. Matching size = 2.
        let inc = eq_inc(3, 2, &[(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]);
        let m = hopcroft_karp(&inc);
        assert_eq!(m.size, 2);
        let dm = DulmageMendelsohnPartition::from_matching(&inc, &m);
        // One col stays unmatched; alternating BFS reaches every col
        // and every row → all under.
        assert_eq!(dm.under_cols.len(), 3);
        assert_eq!(dm.under_rows.len(), 2);
        assert!(dm.square_cols.is_empty());
        assert!(dm.over_cols.is_empty());
    }

    #[test]
    fn dm_mixed_2x2_block_plus_singleton() {
        // 3 rows × 3 cols. Edges form a 1-row block and a 2-row
        // block that share variables only within themselves.
        let inc = eq_inc(3, 3, &[(0, 0), (1, 1), (1, 2), (2, 1), (2, 2)]);
        let m = hopcroft_karp(&inc);
        assert_eq!(m.size, 3);
        let dm = DulmageMendelsohnPartition::from_matching(&inc, &m);
        // All 3 rows and 3 cols are square (matched, fully covered).
        assert_eq!(dm.square_rows.len(), 3);
        assert_eq!(dm.square_cols.len(), 3);
        assert!(dm.over_rows.is_empty());
        assert!(dm.under_rows.is_empty());
    }

    #[test]
    fn dm_row_part_and_col_part_agree_with_lists() {
        let inc = eq_inc(3, 3, &[(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)]);
        let m = hopcroft_karp(&inc);
        let dm = DulmageMendelsohnPartition::from_matching(&inc, &m);
        for (r, part) in dm.row_part.iter().enumerate() {
            match part {
                DMPart::Over => assert!(dm.over_rows.contains(&r)),
                DMPart::Square => assert!(dm.square_rows.contains(&r)),
                DMPart::Under => assert!(dm.under_rows.contains(&r)),
            }
        }
        for (v, part) in dm.col_part.iter().enumerate() {
            match part {
                DMPart::Over => assert!(dm.over_cols.contains(&v)),
                DMPart::Square => assert!(dm.square_cols.contains(&v)),
                DMPart::Under => assert!(dm.under_cols.contains(&v)),
            }
        }
    }

    #[test]
    fn dm_square_has_perfect_matching_restricted() {
        let inc = eq_inc(4, 4, &[(0, 0), (1, 1), (2, 2), (3, 3)]);
        let m = hopcroft_karp(&inc);
        let dm = DulmageMendelsohnPartition::from_matching(&inc, &m);
        // Every square row is matched, and matched to a square col.
        for &r in &dm.square_rows {
            let v = m.row_to_var[r].expect("square row must be matched");
            assert_eq!(dm.col_part[v], DMPart::Square);
        }
    }

    #[test]
    fn dm_overdetermined_with_isolated_unmatched_row() {
        // 2 rows × 1 col. Row 0 ↔ col 0, row 1 has no edges. The
        // unmatched row 1 cannot reach row 0, so row 0 stays square.
        let inc = eq_inc(1, 2, &[(0, 0)]);
        let m = hopcroft_karp(&inc);
        assert_eq!(m.size, 1);
        let dm = DulmageMendelsohnPartition::from_matching(&inc, &m);
        assert_eq!(dm.over_rows, vec![1]);
        assert_eq!(dm.square_rows, vec![0]);
        assert_eq!(dm.square_cols, vec![0]);
        assert!(dm.under_cols.is_empty());
    }
}
