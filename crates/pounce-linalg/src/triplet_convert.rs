//! Triplet → compressed-row converter.
//!
//! Port of `src/Algorithm/LinearSolvers/IpTripletToCSRConverter.{hpp,cpp}`
//! from Ipopt 3.14.x.
//!
//! Symmetric matrices arrive in triplet (COO) form with possibly
//! repeated `(row, col)` entries; this converter produces compressed
//! storage (CSR of the upper triangle, equivalently CSC of the lower)
//! with repeated entries summed. Two output layouts are supported:
//!
//! * [`TriFull::Triangular`] — store the upper triangle only.
//! * [`TriFull::Full`] — store both the upper and the lower triangles
//!   (mirror the off-diagonal entries).
//!
//! The output index base is selected by `offset` (0 for C-style,
//! 1 for Fortran-style).
//!
//! Inputs are interpreted as **1-based** triplet indices, matching
//! upstream's contract. The converter stores the cross-position
//! mapping needed to refill values cheaply when only the matrix
//! values change between calls.

use pounce_common::types::{Index, Number};

/// Half-vs-full storage selector.
///
/// Mirrors `TripletToCSRConverter::ETriFull`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TriFull {
    /// Store the upper (or, equivalently, lower) triangle only.
    Triangular,
    /// Store both upper and lower triangles, mirroring off-diagonals.
    Full,
}

/// Converter handle. After [`Self::initialize`], read [`Self::ia`],
/// [`Self::ja`], and call [`Self::convert_values`] each time the
/// numerical values change.
#[derive(Debug, Clone)]
pub struct TripletToCsrConverter {
    offset: Index,
    hf: TriFull,
    initialized: bool,
    dim: Index,
    nonzeros_triplet: Index,
    nonzeros_compressed: Index,
    num_doubles: Index,
    ia: Vec<Index>,
    ja: Vec<Index>,
    ipos_first: Vec<Index>,
    ipos_double_triplet: Vec<Index>,
    ipos_double_compressed: Vec<Index>,
}

#[derive(Debug, Clone, Copy)]
struct TripletEntry {
    i_row: Index,
    j_col: Index,
    i_pos: Index,
}

impl TripletEntry {
    fn new(i_row: Index, j_col: Index, i_pos: Index) -> Self {
        // Upper-triangle normalization (`TripletEntry::Set`):
        // swap so that i_row <= j_col regardless of input.
        if i_row > j_col {
            Self {
                i_row: j_col,
                j_col: i_row,
                i_pos,
            }
        } else {
            Self {
                i_row,
                j_col,
                i_pos,
            }
        }
    }
}

impl TripletToCsrConverter {
    /// New converter. `offset` must be `0` (C-style) or `1`
    /// (Fortran-style).
    pub fn new(offset: Index, hf: TriFull) -> Self {
        assert!(offset == 0 || offset == 1, "offset must be 0 or 1");
        Self {
            offset,
            hf,
            initialized: false,
            dim: 0,
            nonzeros_triplet: 0,
            nonzeros_compressed: 0,
            num_doubles: 0,
            ia: Vec::new(),
            ja: Vec::new(),
            ipos_first: Vec::new(),
            ipos_double_triplet: Vec::new(),
            ipos_double_compressed: Vec::new(),
        }
    }

    /// Build the CSR sparsity pattern from the triplet pattern
    /// `(airn, ajcn)`. Returns the number of distinct non-zeros in the
    /// compressed format (≤ `airn.len()` because duplicates are
    /// summed).
    ///
    /// `airn` and `ajcn` use **1-based** indices.
    pub fn initialize(&mut self, dim: Index, airn: &[Index], ajcn: &[Index]) -> Index {
        assert_eq!(airn.len(), ajcn.len());
        let nonzeros = airn.len() as Index;
        self.dim = dim;
        self.nonzeros_triplet = nonzeros;

        if nonzeros == 0 {
            self.ia = vec![self.offset; (dim + 1) as usize];
            self.ja = Vec::new();
            self.ipos_first = Vec::new();
            self.ipos_double_triplet = Vec::new();
            self.ipos_double_compressed = Vec::new();
            self.nonzeros_compressed = 0;
            self.num_doubles = 0;
            self.initialized = true;
            return 0;
        }
        assert!(dim > 0);

        // Build and sort the entry list (upper-triangle normalized).
        let mut entries: Vec<TripletEntry> = (0..nonzeros as usize)
            .map(|i| TripletEntry::new(airn[i], ajcn[i], i as Index))
            .collect();
        entries.sort_by(|a, b| (a.i_row, a.j_col).cmp(&(b.i_row, b.j_col)));

        // Working buffers — sized to the upper bound `nonzeros`.
        let mut ja_tmp: Vec<Index> = vec![0; nonzeros as usize];
        let mut ipos_first_tmp: Vec<Index> = vec![0; nonzeros as usize];
        let mut ipos_double_triplet_tmp: Vec<Index> = vec![0; nonzeros as usize];
        let mut ipos_double_compressed_tmp: Vec<Index> = vec![0; nonzeros as usize];
        let mut rc_tmp: Vec<Index> = if self.hf == TriFull::Full {
            vec![0; (dim + 1) as usize]
        } else {
            Vec::new()
        };
        // ia_ uses 0-based row pointers internally; offset is folded in
        // at the end (cpp:222-225 / cpp:307).
        self.ia = vec![0; (dim + 1) as usize];

        let mut nonzeros_compressed: Index = 0;
        let mut nonzeros_compressed_full: Index = 0;
        let mut idouble: Index = 0;
        let mut idouble_full: Index = 0;
        let mut cur_row: Index = 1;

        // Pad empty leading rows: ia[k] = 0 for every row strictly
        // before the first non-empty one (cpp:131-135).
        let first = entries[0];
        while cur_row < first.i_row {
            self.ia[(cur_row - 1) as usize] = 0;
            cur_row += 1;
        }
        self.ia[(cur_row - 1) as usize] = 0;
        ja_tmp[0] = first.j_col;
        ipos_first_tmp[0] = first.i_pos;
        if self.hf == TriFull::Full {
            nonzeros_compressed_full += 1;
            rc_tmp[(cur_row - 1) as usize] += 1;
            if cur_row != first.j_col {
                nonzeros_compressed_full += 1;
                rc_tmp[(first.j_col - 1) as usize] += 1;
            }
        }

        for entry in entries.iter().skip(1) {
            let irow = entry.i_row;
            let jcol = entry.j_col;
            if cur_row == irow && ja_tmp[nonzeros_compressed as usize] == jcol {
                // Repeated (row, col): record as duplicate to be summed
                // into the same compressed slot at value-fill time.
                ipos_double_triplet_tmp[idouble as usize] = entry.i_pos;
                ipos_double_compressed_tmp[idouble as usize] = nonzeros_compressed;
                idouble += 1;
                idouble_full += 1;
                if self.hf == TriFull::Full && irow != jcol {
                    idouble_full += 1;
                }
            } else {
                if self.hf == TriFull::Full {
                    nonzeros_compressed_full += 1;
                    rc_tmp[(jcol - 1) as usize] += 1;
                    if irow != jcol {
                        nonzeros_compressed_full += 1;
                        rc_tmp[(irow - 1) as usize] += 1;
                    }
                }
                nonzeros_compressed += 1;
                ja_tmp[nonzeros_compressed as usize] = jcol;
                ipos_first_tmp[nonzeros_compressed as usize] = entry.i_pos;
                if cur_row != irow {
                    // Crossed into a new (1-based) row: close the
                    // previous one (cpp:189-192).
                    self.ia[cur_row as usize] = nonzeros_compressed;
                    cur_row += 1;
                    // Pad any empty rows between the previous and
                    // current non-empty row.
                    while cur_row < irow {
                        self.ia[cur_row as usize] = nonzeros_compressed;
                        cur_row += 1;
                    }
                }
            }
        }
        nonzeros_compressed += 1;
        // Trailing rows past the last entry are empty.
        for i in cur_row..=self.dim {
            self.ia[i as usize] = nonzeros_compressed;
        }
        self.nonzeros_compressed = nonzeros_compressed;

        debug_assert_eq!(idouble, self.nonzeros_triplet - self.nonzeros_compressed);

        match self.hf {
            TriFull::Triangular => {
                self.ja = vec![0; nonzeros_compressed as usize];
                if self.offset == 0 {
                    for (dst, &src) in self
                        .ja
                        .iter_mut()
                        .zip(ja_tmp.iter())
                        .take(nonzeros_compressed as usize)
                    {
                        *dst = src - 1;
                    }
                } else {
                    self.ja[..nonzeros_compressed as usize]
                        .copy_from_slice(&ja_tmp[..nonzeros_compressed as usize]);
                    for ip in self.ia.iter_mut() {
                        *ip += 1;
                    }
                }
                self.ipos_first = ipos_first_tmp[..nonzeros_compressed as usize].to_vec();
                self.ipos_double_triplet = ipos_double_triplet_tmp[..idouble as usize].to_vec();
                self.ipos_double_compressed =
                    ipos_double_compressed_tmp[..idouble as usize].to_vec();
                self.num_doubles = self.nonzeros_triplet - self.nonzeros_compressed;
            }
            TriFull::Full => {
                // Per-row insert positions; ia_tmp[i+1] is the running
                // write cursor for row i (cpp:253-260).
                let mut ia_tmp: Vec<Index> = vec![0; (self.dim + 1) as usize];
                ia_tmp[0] = 0;
                ia_tmp[1] = 0;
                for i in 1..self.dim as usize {
                    ia_tmp[i + 1] = ia_tmp[i] + rc_tmp[i - 1];
                }

                self.ja = vec![0; nonzeros_compressed_full as usize];
                self.ipos_first = vec![0; nonzeros_compressed_full as usize];
                self.ipos_double_triplet = vec![0; idouble_full as usize];
                self.ipos_double_compressed = vec![0; idouble_full as usize];

                let mut jd1: Index = 0;
                let mut jd2: Index = 0;
                for i in 0..self.dim as usize {
                    let row_start = self.ia[i];
                    let row_end = self.ia[i + 1];
                    for j in row_start..row_end {
                        let jrow = ja_tmp[j as usize] - 1;
                        let slot_upper = ia_tmp[i + 1];
                        self.ja[slot_upper as usize] = jrow + self.offset;
                        self.ipos_first[slot_upper as usize] = ipos_first_tmp[j as usize];
                        while jd1 < idouble && j == ipos_double_compressed_tmp[jd1 as usize] {
                            self.ipos_double_triplet[jd2 as usize] =
                                ipos_double_triplet_tmp[jd1 as usize];
                            self.ipos_double_compressed[jd2 as usize] = ia_tmp[i + 1];
                            jd2 += 1;
                            if jrow != i as Index {
                                self.ipos_double_triplet[jd2 as usize] =
                                    ipos_double_triplet_tmp[jd1 as usize];
                                self.ipos_double_compressed[jd2 as usize] =
                                    ia_tmp[(jrow + 1) as usize];
                                jd2 += 1;
                            }
                            jd1 += 1;
                        }
                        ia_tmp[i + 1] += 1;
                        if jrow != i as Index {
                            let slot_lower = ia_tmp[(jrow + 1) as usize];
                            self.ja[slot_lower as usize] = i as Index + self.offset;
                            self.ipos_first[slot_lower as usize] = ipos_first_tmp[j as usize];
                            ia_tmp[(jrow + 1) as usize] += 1;
                        }
                    }
                }
                for (dst, &src) in self
                    .ia
                    .iter_mut()
                    .zip(ia_tmp.iter())
                    .take((self.dim + 1) as usize)
                {
                    *dst = src + self.offset;
                }
                self.nonzeros_compressed = nonzeros_compressed_full;
                self.num_doubles = idouble_full;
            }
        }

        self.initialized = true;
        self.nonzeros_compressed
    }

    /// Convert numerical values from triplet to compressed format.
    /// Duplicates summed into the same `(i, j)` slot are collapsed via
    /// the `ipos_double_*` cross-tables built by [`Self::initialize`].
    pub fn convert_values(&self, a_triplet: &[Number], a_compressed: &mut [Number]) {
        debug_assert!(self.initialized);
        debug_assert_eq!(a_triplet.len(), self.nonzeros_triplet as usize);
        debug_assert_eq!(a_compressed.len(), self.nonzeros_compressed as usize);
        for i in 0..self.nonzeros_compressed as usize {
            a_compressed[i] = a_triplet[self.ipos_first[i] as usize];
        }
        for i in 0..self.num_doubles as usize {
            a_compressed[self.ipos_double_compressed[i] as usize] +=
                a_triplet[self.ipos_double_triplet[i] as usize];
        }
    }

    /// Row-pointer array of the compressed format
    /// (length `dim + 1`, base `offset`).
    pub fn ia(&self) -> &[Index] {
        debug_assert!(self.initialized);
        &self.ia
    }

    /// Column-index array of the compressed format
    /// (length `nonzeros_compressed`, base `offset`).
    pub fn ja(&self) -> &[Index] {
        debug_assert!(self.initialized);
        &self.ja
    }

    /// Number of non-zero entries in the compressed format
    /// (≤ the input triplet count after duplicates collapse).
    pub fn nonzeros_compressed(&self) -> Index {
        self.nonzeros_compressed
    }

    /// For each compressed entry, the position of its representative in
    /// the original triplet array.
    pub fn ipos_first(&self) -> &[Index] {
        debug_assert!(self.initialized);
        &self.ipos_first
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 3×3 symmetric matrix delivered as a lower-triangle triplet:
    /// ```text
    ///   [10  2  3]
    ///   [ 2  4  .]
    ///   [ 3  .  5]
    /// ```
    /// Triplet (1-based): (1,1)=10, (2,1)=2, (2,2)=4, (3,1)=3, (3,3)=5.
    /// After upper-normalization: (1,1), (1,2)=2, (1,3)=3, (2,2),
    /// (3,3). CSR upper, 0-based: ia=[0,3,4,5], ja=[0,1,2, 1, 2],
    /// a=[10,2,3, 4, 5].
    #[test]
    fn triangular_zero_offset_dedup_free() {
        let irn = [1, 2, 2, 3, 3];
        let jcn = [1, 1, 2, 1, 3];
        let vals = [10.0, 2.0, 4.0, 3.0, 5.0];
        let mut conv = TripletToCsrConverter::new(0, TriFull::Triangular);
        let nz = conv.initialize(3, &irn, &jcn);
        assert_eq!(nz, 5);
        assert_eq!(conv.ia(), &[0, 3, 4, 5]);
        assert_eq!(conv.ja(), &[0, 1, 2, 1, 2]);
        let mut a = vec![0.0; nz as usize];
        conv.convert_values(&vals, &mut a);
        assert_eq!(a, &[10.0, 2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn triangular_one_offset_dedup_free() {
        let irn = [1, 2, 2, 3, 3];
        let jcn = [1, 1, 2, 1, 3];
        let mut conv = TripletToCsrConverter::new(1, TriFull::Triangular);
        conv.initialize(3, &irn, &jcn);
        assert_eq!(conv.ia(), &[1, 4, 5, 6]);
        assert_eq!(conv.ja(), &[1, 2, 3, 2, 3]);
    }

    /// Duplicates: two triplet entries land in the same compressed
    /// slot and must be summed.
    #[test]
    fn dedup_sums_repeated_entries() {
        // (1,1) appears 3× with values 1, 2, 3 → compressed value = 6.
        // (2,2) once with 7. (2,1) appears 2× with 4, 5 → compressed
        // value = 9.
        let irn = [1, 1, 1, 2, 2, 2];
        let jcn = [1, 1, 1, 1, 2, 1];
        let vals = [1.0, 2.0, 3.0, 4.0, 7.0, 5.0];
        let mut conv = TripletToCsrConverter::new(0, TriFull::Triangular);
        let nz = conv.initialize(2, &irn, &jcn);
        assert_eq!(nz, 3);
        assert_eq!(conv.ia(), &[0, 2, 3]);
        assert_eq!(conv.ja(), &[0, 1, 1]);
        let mut a = vec![0.0; nz as usize];
        conv.convert_values(&vals, &mut a);
        assert_eq!(a, &[6.0, 9.0, 7.0]);
    }

    /// `Full_Format` mirrors off-diagonals into the lower triangle.
    /// For matrix `[[1,2],[2,3]]` with triplet (1,1)=1, (2,1)=2,
    /// (2,2)=3 the full CSR has ia=[0,2,4], ja=[0,1, 0,1],
    /// a=[1,2, 2,3].
    #[test]
    fn full_format_mirrors_off_diagonals() {
        let irn = [1, 2, 2];
        let jcn = [1, 1, 2];
        let vals = [1.0, 2.0, 3.0];
        let mut conv = TripletToCsrConverter::new(0, TriFull::Full);
        let nz = conv.initialize(2, &irn, &jcn);
        assert_eq!(nz, 4);
        assert_eq!(conv.ia(), &[0, 2, 4]);
        assert_eq!(conv.ja(), &[0, 1, 0, 1]);
        let mut a = vec![0.0; nz as usize];
        conv.convert_values(&vals, &mut a);
        assert_eq!(a, &[1.0, 2.0, 2.0, 3.0]);
    }

    /// Empty leading and trailing rows produce a flat plateau in `ia`.
    #[test]
    fn empty_rows_handled() {
        // 4×4 matrix with only entries in row 3 (1-based).
        let irn = [3, 3];
        let jcn = [1, 3];
        let mut conv = TripletToCsrConverter::new(0, TriFull::Triangular);
        let nz = conv.initialize(4, &irn, &jcn);
        // Upper-normalized: (1,3), (3,3) → sorted: (1,3) then (3,3).
        // Row 1 has 1 entry, row 2 has 0, row 3 has 1, row 4 has 0.
        assert_eq!(nz, 2);
        assert_eq!(conv.ia(), &[0, 1, 1, 2, 2]);
        assert_eq!(conv.ja(), &[2, 2]);
    }

    /// Zero-nonzero matrix is a degenerate but legal input.
    #[test]
    fn zero_nonzeros_yields_flat_ia() {
        let irn: [Index; 0] = [];
        let jcn: [Index; 0] = [];
        let mut conv = TripletToCsrConverter::new(0, TriFull::Triangular);
        assert_eq!(conv.initialize(3, &irn, &jcn), 0);
        assert_eq!(conv.ia(), &[0, 0, 0, 0]);
        assert!(conv.ja().is_empty());
    }
}
