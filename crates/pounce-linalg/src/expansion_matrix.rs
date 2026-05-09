//! Expansion / projection matrices.
//!
//! Mirrors `LinAlg/IpExpansionMatrix.{hpp,cpp}`. The matrix is fully
//! described by a list of `n_small` row indices into the larger
//! `n_large`-dimensional space. `MultVector` lifts a small vector into
//! the large space (with zero-fill); `TransMultVector` filters a large
//! vector down to the small space.
//!
//! `ExpansionMatrixSpace` owns the index permutation; many
//! `ExpansionMatrix` instances can share the same space (matching
//! upstream's `MatrixSpace` reuse pattern).

use crate::dense_vector::DenseVector;
use crate::matrix::{Matrix, MatrixCache};
use crate::vector::Vector;
use pounce_common::tagged::{Tag, TaggedObject};
use pounce_common::types::{Index, Number};
use std::any::Any;
use std::rc::Rc;

/// Index data backing one or more `ExpansionMatrix` instances.
#[derive(Debug)]
pub struct ExpansionMatrixSpace {
    n_large: Index,
    n_small: Index,
    /// `expanded_pos[i]` = row in the large vector that small position
    /// `i` maps to (0-based).
    expanded_pos: Vec<Index>,
    /// `compressed_pos[j]` = small index for large row `j`, or -1 if
    /// row `j` is not in the small set.
    compressed_pos: Vec<Index>,
}

impl ExpansionMatrixSpace {
    /// `exp_pos[i] - offset` gives the large-vector row that the i-th
    /// small entry maps to. Upstream offsets `1` for Fortran-style
    /// inputs, `0` otherwise; pass `0` for natural Rust-side use.
    pub fn new(n_large: Index, n_small: Index, exp_pos: &[Index], offset: Index) -> Rc<Self> {
        assert_eq!(exp_pos.len(), n_small as usize);
        let mut expanded = Vec::with_capacity(n_small as usize);
        let mut compressed = vec![-1; n_large.max(0) as usize];
        for (i, &raw) in exp_pos.iter().enumerate() {
            let pos = raw - offset;
            debug_assert!(pos >= 0 && pos < n_large);
            expanded.push(pos);
            if !compressed.is_empty() {
                compressed[pos as usize] = i as Index;
            }
        }
        Rc::new(Self {
            n_large,
            n_small,
            expanded_pos: expanded,
            compressed_pos: compressed,
        })
    }

    pub fn n_large(&self) -> Index {
        self.n_large
    }
    pub fn n_small(&self) -> Index {
        self.n_small
    }
    pub fn expanded_pos_indices(&self) -> &[Index] {
        &self.expanded_pos
    }
    pub fn compressed_pos_indices(&self) -> &[Index] {
        &self.compressed_pos
    }
}

/// Sparse 0/1 expansion matrix with shape `n_large × n_small`.
#[derive(Debug)]
pub struct ExpansionMatrix {
    space: Rc<ExpansionMatrixSpace>,
    cache: MatrixCache,
}

impl ExpansionMatrix {
    pub fn new(space: Rc<ExpansionMatrixSpace>) -> Self {
        Self {
            space,
            cache: MatrixCache::new(),
        }
    }

    pub fn space(&self) -> &Rc<ExpansionMatrixSpace> {
        &self.space
    }
    pub fn expanded_pos_indices(&self) -> &[Index] {
        self.space.expanded_pos_indices()
    }
    pub fn compressed_pos_indices(&self) -> &[Index] {
        self.space.compressed_pos_indices()
    }
}

impl TaggedObject for ExpansionMatrix {
    fn get_tag(&self) -> Tag {
        self.cache.tag()
    }
}

fn dense(v: &dyn Vector) -> &DenseVector {
    match v.as_any().downcast_ref::<DenseVector>() {
        Some(dv) => dv,
        None => panic!("ExpansionMatrix only supports DenseVector inputs/outputs"),
    }
}

fn dense_mut(v: &mut dyn Vector) -> &mut DenseVector {
    match v.as_any_mut().downcast_mut::<DenseVector>() {
        Some(dv) => dv,
        None => panic!("ExpansionMatrix only supports DenseVector inputs/outputs"),
    }
}

impl Matrix for ExpansionMatrix {
    fn n_rows(&self) -> Index {
        self.space.n_large
    }
    fn n_cols(&self) -> Index {
        self.space.n_small
    }
    fn cache(&self) -> &MatrixCache {
        &self.cache
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
    fn as_tagged(&self) -> &dyn TaggedObject {
        self
    }
    fn as_dyn_matrix(&self) -> &dyn Matrix {
        self
    }

    fn mult_vector_impl(
        &self,
        alpha: Number,
        x: &dyn Vector,
        beta: Number,
        y: &mut dyn Vector,
    ) {
        // Order matches upstream IpExpansionMatrix.cpp:38-94: y is
        // scaled or zeroed first, then the scatter pass is applied.
        if beta != 0.0 {
            y.scal(beta);
        } else {
            y.set(0.0);
        }
        let exp_pos = self.expanded_pos_indices();
        let n_cols = self.space.n_small;

        let dx = dense(x);
        let dy = dense_mut(y);
        // Ensure y is materialised since we will index by row.
        dy.ensure_storage();
        let yvals = dy.values_mut();

        if dx.is_homogeneous() {
            let val = alpha * dx.scalar();
            if val != 0.0 {
                for &p in &exp_pos[..n_cols as usize] {
                    yvals[p as usize] += val;
                }
            }
        } else {
            let xvals = dx.values();
            if alpha == 1.0 {
                for i in 0..n_cols as usize {
                    yvals[exp_pos[i] as usize] += xvals[i];
                }
            } else if alpha == -1.0 {
                for i in 0..n_cols as usize {
                    yvals[exp_pos[i] as usize] -= xvals[i];
                }
            } else {
                for i in 0..n_cols as usize {
                    yvals[exp_pos[i] as usize] += alpha * xvals[i];
                }
            }
        }
    }

    fn trans_mult_vector_impl(
        &self,
        alpha: Number,
        x: &dyn Vector,
        beta: Number,
        y: &mut dyn Vector,
    ) {
        if beta != 0.0 {
            y.scal(beta);
        } else {
            y.set(0.0);
        }
        let exp_pos = self.expanded_pos_indices();
        let n_cols = self.space.n_small;

        let dx = dense(x);
        let dy = dense_mut(y);
        dy.ensure_storage();
        let yvals = dy.values_mut();

        if dx.is_homogeneous() {
            let val = alpha * dx.scalar();
            if val != 0.0 {
                for v in yvals.iter_mut().take(n_cols as usize) {
                    *v += val;
                }
            }
        } else {
            let xvals = dx.values();
            if alpha == 1.0 {
                for i in 0..n_cols as usize {
                    yvals[i] += xvals[exp_pos[i] as usize];
                }
            } else if alpha == -1.0 {
                for i in 0..n_cols as usize {
                    yvals[i] -= xvals[exp_pos[i] as usize];
                }
            } else {
                for i in 0..n_cols as usize {
                    yvals[i] += alpha * xvals[exp_pos[i] as usize];
                }
            }
        }
    }

    fn add_m_sinv_z_impl(
        &self,
        alpha: Number,
        s: &dyn Vector,
        z: &dyn Vector,
        x: &mut dyn Vector,
    ) {
        let ds = dense(s);
        let dz = dense(z);

        // Homogeneous-S falls back to the generic path.
        if ds.is_homogeneous() {
            self.add_m_sinv_z_default(alpha, s, z, x);
            return;
        }

        let exp_pos = self.expanded_pos_indices();
        let n_cols = self.space.n_small;
        let vals_s = ds.values();
        let dx = dense_mut(x);
        dx.ensure_storage();
        let vals_x = dx.values_mut();

        if dz.is_homogeneous() {
            let val = alpha * dz.scalar();
            if val != 0.0 {
                for i in 0..n_cols as usize {
                    vals_x[exp_pos[i] as usize] += val / vals_s[i];
                }
            }
        } else {
            let vals_z = dz.values();
            if alpha == 1.0 {
                for i in 0..n_cols as usize {
                    vals_x[exp_pos[i] as usize] += vals_z[i] / vals_s[i];
                }
            } else if alpha == -1.0 {
                for i in 0..n_cols as usize {
                    vals_x[exp_pos[i] as usize] -= vals_z[i] / vals_s[i];
                }
            } else {
                for i in 0..n_cols as usize {
                    vals_x[exp_pos[i] as usize] += alpha * vals_z[i] / vals_s[i];
                }
            }
        }
    }

    fn sinv_blrm_zmt_dbr_impl(
        &self,
        alpha: Number,
        s: &dyn Vector,
        r: &dyn Vector,
        z: &dyn Vector,
        d: &dyn Vector,
        x: &mut dyn Vector,
    ) {
        let ds = dense(s);
        let dr = dense(r);
        let dz = dense(z);
        let dd = dense(d);

        // Fall back to default when S or D is homogeneous.
        if ds.is_homogeneous() || dd.is_homogeneous() {
            self.sinv_blrm_zmt_dbr_default(alpha, s, r, z, d, x);
            return;
        }

        let exp_pos = self.expanded_pos_indices();
        let n_cols = self.space.n_small;
        let vals_s = ds.values();
        let vals_d = dd.values();
        let dx = dense_mut(x);
        dx.ensure_storage();
        let vals_x = dx.values_mut();

        if dr.is_homogeneous() {
            let scalar_r = dr.scalar();
            if dz.is_homogeneous() {
                let val = alpha * dz.scalar();
                if val == 0.0 {
                    for i in 0..n_cols as usize {
                        vals_x[i] = scalar_r / vals_s[i];
                    }
                } else {
                    for i in 0..n_cols as usize {
                        vals_x[i] = (scalar_r + val * vals_d[exp_pos[i] as usize]) / vals_s[i];
                    }
                }
            } else {
                let vals_z = dz.values();
                if alpha == 1.0 {
                    for i in 0..n_cols as usize {
                        vals_x[i] =
                            (scalar_r + vals_z[i] * vals_d[exp_pos[i] as usize]) / vals_s[i];
                    }
                } else if alpha == -1.0 {
                    for i in 0..n_cols as usize {
                        vals_x[i] =
                            (scalar_r - vals_z[i] * vals_d[exp_pos[i] as usize]) / vals_s[i];
                    }
                } else {
                    for i in 0..n_cols as usize {
                        vals_x[i] = (scalar_r
                            + alpha * vals_z[i] * vals_d[exp_pos[i] as usize])
                            / vals_s[i];
                    }
                }
            }
        } else {
            let vals_r = dr.values();
            if dz.is_homogeneous() {
                let val = alpha * dz.scalar();
                for i in 0..n_cols as usize {
                    vals_x[i] = (vals_r[i] + val * vals_d[exp_pos[i] as usize]) / vals_s[i];
                }
            } else {
                let vals_z = dz.values();
                if alpha == 1.0 {
                    for i in 0..n_cols as usize {
                        vals_x[i] =
                            (vals_r[i] + vals_z[i] * vals_d[exp_pos[i] as usize]) / vals_s[i];
                    }
                } else if alpha == -1.0 {
                    for i in 0..n_cols as usize {
                        vals_x[i] =
                            (vals_r[i] - vals_z[i] * vals_d[exp_pos[i] as usize]) / vals_s[i];
                    }
                } else {
                    for i in 0..n_cols as usize {
                        vals_x[i] = (vals_r[i]
                            + alpha * vals_z[i] * vals_d[exp_pos[i] as usize])
                            / vals_s[i];
                    }
                }
            }
        }
    }

    fn compute_row_amax_impl(&self, rows_norms: &mut dyn Vector, _init: bool) {
        // Upstream comment (`IpExpansionMatrix.cpp:374-389`): expects
        // the caller to have already initialised rows_norms (the
        // `init` flag argument is *unused* for this matrix — its only
        // job is to set selected rows to max(current, 1)).
        let exp_pos = self.expanded_pos_indices();
        let dy = dense_mut(rows_norms);
        dy.ensure_storage();
        let vec_vals = dy.values_mut();
        for &p in &exp_pos[..self.space.n_small as usize] {
            let row = p as usize;
            vec_vals[row] = vec_vals[row].max(1.0);
        }
    }

    fn compute_col_amax_impl(&self, cols_norms: &mut dyn Vector, init: bool) {
        // All columns of an expansion matrix have a single 1 → col-norm
        // is uniformly 1.
        if init {
            cols_norms.set(1.0);
        } else {
            let mut v = cols_norms.make_new();
            v.set(1.0);
            cols_norms.element_wise_max(v.as_dyn_vector());
        }
    }
}

impl ExpansionMatrix {
    /// Inlined copy of [`Matrix::add_m_sinv_z_impl`]'s default body —
    /// needed because we can't call the trait default from inside an
    /// override.
    fn add_m_sinv_z_default(
        &self,
        alpha: Number,
        s: &dyn Vector,
        z: &dyn Vector,
        x: &mut dyn Vector,
    ) {
        let mut tmp = s.make_new_copy();
        tmp.set(0.0);
        tmp.add_vector_quotient(1.0, z, s, 0.0);
        self.mult_vector(alpha, tmp.as_dyn_vector(), 1.0, x);
    }

    fn sinv_blrm_zmt_dbr_default(
        &self,
        alpha: Number,
        s: &dyn Vector,
        r: &dyn Vector,
        z: &dyn Vector,
        d: &dyn Vector,
        x: &mut dyn Vector,
    ) {
        self.trans_mult_vector(alpha, d, 0.0, x);
        x.element_wise_multiply(z);
        x.axpy(1.0, r);
        x.element_wise_divide(s);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dense_vector::DenseVectorSpace;

    fn dvec_box(values: &[Number]) -> Box<dyn Vector> {
        let space = DenseVectorSpace::new(values.len() as Index);
        let mut v = space.make_new_dense();
        v.set_values(values);
        Box::new(v)
    }

    #[test]
    fn mult_vector_lifts_small_to_large() {
        // small dim 2, large dim 5, mapping small[0]→row 1, small[1]→row 3.
        let space = ExpansionMatrixSpace::new(5, 2, &[1, 3], 0);
        let m = ExpansionMatrix::new(space);
        let x = dvec_box(&[7.0, -2.0]);
        let mut y = dvec_box(&[0.0; 5]);
        m.mult_vector(1.0, x.as_dyn_vector(), 0.0, y.as_mut());
        let dv = y.as_any().downcast_ref::<DenseVector>().unwrap();
        assert_eq!(dv.expanded_values().to_vec(), vec![0.0, 7.0, 0.0, -2.0, 0.0]);
    }

    #[test]
    fn trans_mult_vector_filters_large_to_small() {
        let space = ExpansionMatrixSpace::new(5, 2, &[1, 3], 0);
        let m = ExpansionMatrix::new(space);
        let large = dvec_box(&[10.0, 20.0, 30.0, 40.0, 50.0]);
        let mut small = dvec_box(&[0.0, 0.0]);
        m.trans_mult_vector(1.0, large.as_dyn_vector(), 0.0, small.as_mut());
        let dv = small.as_any().downcast_ref::<DenseVector>().unwrap();
        assert_eq!(dv.expanded_values().to_vec(), vec![20.0, 40.0]);
    }

    #[test]
    fn mult_vector_with_alpha_and_beta() {
        let space = ExpansionMatrixSpace::new(4, 2, &[0, 2], 0);
        let m = ExpansionMatrix::new(space);
        let x = dvec_box(&[3.0, 4.0]);
        let mut y = dvec_box(&[1.0, 2.0, 3.0, 4.0]);
        // y ← -2*Mx + 0.5 y
        // Mx = [3,0,4,0], -2 Mx = [-6,0,-8,0], 0.5 y = [0.5,1,1.5,2]
        // sum = [-5.5, 1, -6.5, 2]
        m.mult_vector(-2.0, x.as_dyn_vector(), 0.5, y.as_mut());
        let dv = y.as_any().downcast_ref::<DenseVector>().unwrap();
        assert_eq!(dv.expanded_values().to_vec(), vec![-5.5, 1.0, -6.5, 2.0]);
    }

    #[test]
    fn p_transpose_p_is_identity_on_small() {
        // For a full-rank expansion P, PᵀPx = x (each small entry is
        // selected exactly once).
        let space = ExpansionMatrixSpace::new(6, 3, &[5, 1, 2], 0);
        let m = ExpansionMatrix::new(space);
        let x = dvec_box(&[1.5, -2.5, 3.5]);
        let mut large = dvec_box(&[0.0; 6]);
        m.mult_vector(1.0, x.as_dyn_vector(), 0.0, large.as_mut());
        let mut roundtrip = dvec_box(&[0.0; 3]);
        m.trans_mult_vector(1.0, large.as_dyn_vector(), 0.0, roundtrip.as_mut());
        let dv = roundtrip.as_any().downcast_ref::<DenseVector>().unwrap();
        assert_eq!(dv.expanded_values().to_vec(), vec![1.5, -2.5, 3.5]);
    }

    #[test]
    fn col_amax_is_uniformly_one() {
        let space = ExpansionMatrixSpace::new(5, 2, &[1, 3], 0);
        let m = ExpansionMatrix::new(space);
        let mut norms = dvec_box(&[0.0, 0.0]);
        m.compute_col_amax(norms.as_mut(), true);
        let dv = norms.as_any().downcast_ref::<DenseVector>().unwrap();
        assert_eq!(dv.expanded_values().to_vec(), vec![1.0, 1.0]);
    }

    #[test]
    fn row_amax_marks_selected_rows() {
        let space = ExpansionMatrixSpace::new(5, 2, &[1, 3], 0);
        let m = ExpansionMatrix::new(space);
        let mut norms = dvec_box(&[0.0; 5]);
        // The Matrix::compute_row_amax wrapper zeros norms when init=true,
        // then the impl raises positions 1 and 3 to 1.0.
        m.compute_row_amax(norms.as_mut(), true);
        let dv = norms.as_any().downcast_ref::<DenseVector>().unwrap();
        assert_eq!(dv.expanded_values().to_vec(), vec![0.0, 1.0, 0.0, 1.0, 0.0]);
    }
}
