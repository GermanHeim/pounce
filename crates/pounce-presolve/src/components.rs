//! Weakly-connected component extraction on the square-matched part
//! of a Dulmage-Mendelsohn partition.
//!
//! PR 3 of the auxiliary-presolve port (issue #53). Each component
//! becomes one independent candidate block for elimination in PR 8.
//! ripopt anchor: `src/auxiliary_preprocessing.rs:2416-2469`.

use crate::dulmage_mendelsohn::{DMPart, DulmageMendelsohnPartition};
use crate::incidence::EqualityIncidence;
use crate::matching::BipartiteMatching;

/// One connected component of the square sub-graph.
#[derive(Debug, Clone, Default)]
pub struct SquareComponent {
    /// Equality-row indices (positions in `inc.eq_row_inner_idx`).
    pub eq_rows: Vec<usize>,
    /// Variable indices.
    pub cols: Vec<usize>,
}

/// Decomposition of the square sub-graph into connected components.
#[derive(Debug, Clone, Default)]
pub struct SquareComponents {
    /// One entry per component, sorted by smallest contained
    /// equality-row index for determinism.
    pub components: Vec<SquareComponent>,
}

/// Disjoint-set / union-find over `(rows ∪ cols)` with rows numbered
/// `0..n_rows` and cols offset by `n_rows`.
struct UnionFind {
    parent: Vec<usize>,
    rank: Vec<u8>,
}

impl UnionFind {
    fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
            rank: vec![0; n],
        }
    }

    fn find(&mut self, x: usize) -> usize {
        let mut root = x;
        while self.parent[root] != root {
            root = self.parent[root];
        }
        // Path compression.
        let mut cur = x;
        while self.parent[cur] != root {
            let nxt = self.parent[cur];
            self.parent[cur] = root;
            cur = nxt;
        }
        root
    }

    fn union(&mut self, x: usize, y: usize) {
        let rx = self.find(x);
        let ry = self.find(y);
        if rx == ry {
            return;
        }
        match self.rank[rx].cmp(&self.rank[ry]) {
            std::cmp::Ordering::Less => self.parent[rx] = ry,
            std::cmp::Ordering::Greater => self.parent[ry] = rx,
            std::cmp::Ordering::Equal => {
                self.parent[ry] = rx;
                self.rank[rx] += 1;
            }
        }
    }
}

impl SquareComponents {
    /// Decompose the square sub-graph of `part` into connected
    /// components. Edges considered are exactly those of `inc` whose
    /// row AND column are both in `DMPart::Square`.
    ///
    /// # Example
    ///
    /// ```
    /// use pounce_presolve::incidence::{EqualityIncidence, ProbeView};
    /// use pounce_presolve::matching::hopcroft_karp;
    /// use pounce_presolve::dulmage_mendelsohn::DulmageMendelsohnPartition;
    /// use pounce_presolve::components::SquareComponents;
    ///
    /// // 4×4 with a 2-block and two singletons.
    /// let p = ProbeView {
    ///     n_vars: 4,
    ///     m_rows: 4,
    ///     jac_irow: &[0, 0, 1, 1, 2, 3],
    ///     jac_jcol: &[0, 1, 0, 1, 2, 3],
    ///     jac_values: None,
    ///     g_l: &[0.0; 4],
    ///     g_u: &[0.0; 4],
    ///     linearity: None,
    ///     one_based: false,
    ///     eq_tol: 1e-12,
    /// };
    /// let inc = EqualityIncidence::from_probe(&p);
    /// let m = hopcroft_karp(&inc);
    /// let dm = DulmageMendelsohnPartition::from_matching(&inc, &m);
    /// let c = SquareComponents::of_square_part(&inc, &m, &dm);
    /// assert_eq!(c.components.len(), 3);
    /// ```
    pub fn of_square_part(
        inc: &EqualityIncidence,
        _m: &BipartiteMatching,
        part: &DulmageMendelsohnPartition,
    ) -> Self {
        let n_rows = inc.n_eq_rows();
        let n_vars = inc.n_vars;

        // Union-find IDs: rows are 0..n_rows, cols are n_rows..n_rows+n_vars.
        let mut uf = UnionFind::new(n_rows + n_vars);
        let col_off = n_rows;

        for r in 0..n_rows {
            if part.row_part[r] != DMPart::Square {
                continue;
            }
            for &v in inc.neighbors(r) {
                if part.col_part[v] != DMPart::Square {
                    continue;
                }
                uf.union(r, col_off + v);
            }
        }

        // Bucket members by component root.
        use std::collections::BTreeMap;
        let mut buckets: BTreeMap<usize, (Vec<usize>, Vec<usize>)> = BTreeMap::new();
        for r in 0..n_rows {
            if part.row_part[r] != DMPart::Square {
                continue;
            }
            let root = uf.find(r);
            buckets.entry(root).or_default().0.push(r);
        }
        for v in 0..n_vars {
            if part.col_part[v] != DMPart::Square {
                continue;
            }
            let root = uf.find(col_off + v);
            buckets.entry(root).or_default().1.push(v);
        }

        // Sort components by the smallest contained equality-row
        // index for deterministic output.
        let mut comps: Vec<SquareComponent> = buckets
            .into_values()
            .map(|(mut rows, mut cols)| {
                rows.sort_unstable();
                cols.sort_unstable();
                SquareComponent {
                    eq_rows: rows,
                    cols,
                }
            })
            .filter(|c| !c.eq_rows.is_empty() || !c.cols.is_empty())
            .collect();
        comps.sort_by_key(|c| c.eq_rows.first().copied().unwrap_or(usize::MAX));

        Self { components: comps }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matching::hopcroft_karp;

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

    fn decompose(n_vars: usize, n_rows: usize, edges: &[(usize, usize)]) -> SquareComponents {
        let inc = eq_inc(n_vars, n_rows, edges);
        let m = hopcroft_karp(&inc);
        let dm = DulmageMendelsohnPartition::from_matching(&inc, &m);
        SquareComponents::of_square_part(&inc, &m, &dm)
    }

    #[test]
    fn components_empty_square() {
        let c = decompose(0, 0, &[]);
        assert!(c.components.is_empty());
    }

    #[test]
    fn components_disjoint_3x3() {
        let c = decompose(3, 3, &[(0, 0), (1, 1), (2, 2)]);
        assert_eq!(c.components.len(), 3);
        for (i, comp) in c.components.iter().enumerate() {
            assert_eq!(comp.eq_rows, vec![i]);
            assert_eq!(comp.cols, vec![i]);
        }
    }

    #[test]
    fn components_two_blocks_5x5() {
        // Block A: rows {0,1,2} share cols {0,1,2}.
        // Block B: rows {3,4} share cols {3,4}.
        let edges = [
            (0, 0),
            (0, 1),
            (1, 1),
            (1, 2),
            (2, 0),
            (2, 2),
            (3, 3),
            (3, 4),
            (4, 4),
        ];
        let c = decompose(5, 5, &edges);
        assert_eq!(c.components.len(), 2);
        assert_eq!(c.components[0].eq_rows, vec![0, 1, 2]);
        assert_eq!(c.components[0].cols, vec![0, 1, 2]);
        assert_eq!(c.components[1].eq_rows, vec![3, 4]);
        assert_eq!(c.components[1].cols, vec![3, 4]);
    }

    #[test]
    fn components_star_inside_square() {
        // Row 0 acts as a hub connecting cols 0, 1, 2. Rows 1, 2
        // pick up cols 1, 2 respectively to keep things square.
        let c = decompose(3, 3, &[(0, 0), (0, 1), (0, 2), (1, 1), (2, 2)]);
        assert_eq!(c.components.len(), 1);
        let only = &c.components[0];
        assert_eq!(only.eq_rows, vec![0, 1, 2]);
        assert_eq!(only.cols, vec![0, 1, 2]);
    }

    #[test]
    fn components_order_is_deterministic() {
        // Build two disjoint 2-blocks but order edges so the second
        // block's rows come first in the edge list.
        let c = decompose(
            4,
            4,
            &[
                (2, 2),
                (2, 3),
                (3, 2),
                (3, 3),
                (0, 0),
                (0, 1),
                (1, 0),
                (1, 1),
            ],
        );
        assert_eq!(c.components.len(), 2);
        // Lowest row index in component[0] must be smaller.
        assert_eq!(c.components[0].eq_rows.first(), Some(&0));
        assert_eq!(c.components[1].eq_rows.first(), Some(&2));
    }

    #[test]
    fn components_skip_overdetermined_and_underdetermined() {
        // 3 rows × 2 cols, fully connected → all over. Square is
        // empty, so zero components.
        let c = decompose(2, 3, &[(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)]);
        assert!(c.components.is_empty());
    }
}
