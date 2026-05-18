//! Matrix stored as a list of column vectors — port of
//! `LinAlg/IpMultiVectorMatrix.{hpp,cpp}`.
//!
//! `MultiVectorMatrix` holds `n_cols` column vectors in a single
//! `VectorSpace`. The primary consumer is the L-BFGS / SR1 limited-
//! memory quasi-Newton path, which stores its rolling curvature-pair
//! window as a `MultiVectorMatrix` and feeds it into
//! `LowRankUpdateSymMatrix` (Phase 8 follow-up).
//!
//! Upstream distinguishes const and non-const stored Vectors so the
//! columns can be aliased read-only references when callers don't need
//! to mutate them. Rust's borrow rules already prevent aliasing
//! mutation, so we collapse to a single slot per column holding an
//! `Rc<dyn Vector>` — `set_vector` replaces the slot, dropping any
//! previous occupant. Upstream's `ScaleRows` / `ScaleColumns` /
//! `AddOneMultiVectorMatrix` / `FillWithNewVectors` / `AddRightMultMatrix`
//! all need to *mutate* the column vectors, so for those we require
//! that the column slot holds a uniquely-owned `Rc` and use
//! `Rc::make_mut`-style cloning to obtain `&mut dyn Vector` access; in
//! practice the caller sets the column via `fill_with_new_vectors` or
//! `set_vector_owned` first.
//!
//! Reduction order matches upstream column-by-column traversal so
//! BLAS-1 accumulation order is preserved bit-for-bit.

use crate::dense_gen_matrix::DenseGenMatrix;
use crate::dense_vector::{DenseVector, DenseVectorSpace};
use crate::matrix::{Matrix, MatrixCache};
use crate::vector::Vector;
use pounce_common::tagged::{Tag, TaggedObject};
use pounce_common::types::{Index, Number};
use std::any::Any;
use std::rc::Rc;

#[derive(Debug)]
pub struct MultiVectorMatrixSpace {
    n_rows: Index,
    n_cols: Index,
    col_space: Rc<DenseVectorSpace>,
}

impl MultiVectorMatrixSpace {
    pub fn new(n_cols: Index, col_space: Rc<DenseVectorSpace>) -> Rc<Self> {
        Rc::new(Self {
            n_rows: col_space.dim(),
            n_cols,
            col_space,
        })
    }

    pub fn n_rows(&self) -> Index {
        self.n_rows
    }
    pub fn n_cols(&self) -> Index {
        self.n_cols
    }
    pub fn col_vector_space(&self) -> &Rc<DenseVectorSpace> {
        &self.col_space
    }

    pub fn make_new_multi_vector(self: &Rc<Self>) -> MultiVectorMatrix {
        MultiVectorMatrix::new(Rc::clone(self))
    }
}

#[derive(Debug)]
pub struct MultiVectorMatrix {
    space: Rc<MultiVectorMatrixSpace>,
    cache: MatrixCache,
    /// One column per slot. `None` means "not yet set" — calling any
    /// arithmetic on an unset column panics in debug, mirroring
    /// upstream's `DBG_ASSERT(IsValid(...))`.
    cols: Vec<Option<Rc<dyn Vector>>>,
}

impl MultiVectorMatrix {
    pub fn new(space: Rc<MultiVectorMatrixSpace>) -> Self {
        let n = space.n_cols.max(0) as usize;
        Self {
            space,
            cache: MatrixCache::new(),
            cols: (0..n).map(|_| None).collect(),
        }
    }

    pub fn space(&self) -> &Rc<MultiVectorMatrixSpace> {
        &self.space
    }

    pub fn col_vector_space(&self) -> &Rc<DenseVectorSpace> {
        self.space.col_vector_space()
    }

    /// Sets column `i` to share `vec`. Replaces any previous occupant.
    /// Mirrors `MultiVectorMatrix::SetVector` (the const overload —
    /// since Rust enforces borrow rules statically, the non-const
    /// overload collapses into the same operation).
    pub fn set_vector(&mut self, i: Index, vec: Rc<dyn Vector>) {
        let idx = i as usize;
        debug_assert!(idx < self.cols.len());
        debug_assert_eq!(vec.dim(), self.space.n_rows);
        self.cols[idx] = Some(vec);
        self.cache.bump();
    }

    pub fn get_vector(&self, i: Index) -> &Rc<dyn Vector> {
        let idx = i as usize;
        debug_assert!(idx < self.cols.len());
        self.cols[idx]
            .as_ref()
            .expect("MultiVectorMatrix column is unset")
    }

    /// Like upstream `FillWithNewVectors`: replaces every column with
    /// a freshly allocated, uninitialised dense vector from the
    /// column space.
    pub fn fill_with_new_vectors(&mut self) {
        for slot in self.cols.iter_mut() {
            let v = self.space.col_space.make_new_dense();
            *slot = Some(Rc::new(v) as Rc<dyn Vector>);
        }
        self.cache.bump();
    }

    fn col(&self, i: usize) -> &dyn Vector {
        self.cols[i]
            .as_ref()
            .expect("MultiVectorMatrix column is unset")
            .as_ref()
    }

    /// Helper: take exclusive `&mut dyn Vector` access to column `i`.
    /// Panics if the column is shared (Rc strong_count > 1) — that
    /// would indicate the caller stored the same column elsewhere and
    /// then asked us to mutate it, which upstream's non-const path
    /// rules out by construction.
    fn col_mut(&mut self, i: usize) -> &mut dyn Vector {
        let slot = self.cols[i]
            .as_mut()
            .expect("MultiVectorMatrix column is unset");
        let inner: &mut dyn Vector = Rc::get_mut(slot)
            .expect("MultiVectorMatrix column is shared; cannot mutate (clone first)");
        inner
    }

    /// `y ← α V Vᵀ x + β y`. Port of `LRMultVector`.
    pub fn lr_mult_vector(&self, alpha: Number, x: &dyn Vector, beta: Number, y: &mut dyn Vector) {
        debug_assert_eq!(self.space.n_rows, x.dim());
        debug_assert_eq!(self.space.n_rows, y.dim());
        if beta != 0.0 {
            y.scal(beta);
        } else {
            y.set(0.0);
        }
        for i in 0..self.cols.len() {
            let ci = self.col(i);
            let coef = alpha * ci.dot(x);
            y.add_one_vector(coef, ci, 1.0);
        }
    }

    /// `V[:, i] ← a · V[:, i] (no-op if c==1) + scaled by ScalarVec`
    /// helpers. Port of `ScaleColumns` for the dense, non-homogeneous
    /// case (homogeneous fast path is preserved).
    pub fn scale_columns(&mut self, scal: &DenseVector) {
        debug_assert_eq!(scal.dim(), self.space.n_cols);
        let nc = self.cols.len();
        if scal.is_homogeneous() {
            let s = scal.scalar();
            for i in 0..nc {
                self.col_mut(i).scal(s);
            }
        } else {
            let vals = scal.values().to_vec();
            for i in 0..nc {
                self.col_mut(i).scal(vals[i]);
            }
        }
        self.cache.bump();
    }

    /// `V[:, i] ← V[:, i] .* scal_vec` for every column. Port of
    /// `ScaleRows`.
    pub fn scale_rows(&mut self, scal: &dyn Vector) {
        debug_assert_eq!(scal.dim(), self.space.n_rows);
        let nc = self.cols.len();
        for i in 0..nc {
            self.col_mut(i).element_wise_multiply(scal);
        }
        self.cache.bump();
    }

    /// `V ← a · V1 + c · V` (column-wise). When `c == 0`, replaces
    /// every column with a fresh allocation first, mirroring upstream.
    pub fn add_one_multi_vector_matrix(&mut self, a: Number, mv1: &MultiVectorMatrix, c: Number) {
        debug_assert_eq!(self.space.n_rows, mv1.space.n_rows);
        debug_assert_eq!(self.space.n_cols, mv1.space.n_cols);
        if c == 0.0 {
            self.fill_with_new_vectors();
        }
        let nc = self.cols.len();
        for i in 0..nc {
            // Borrow trick: clone the source Rc out before taking a
            // mutable borrow of self.
            let src = Rc::clone(&mv1.cols[i].as_ref().expect("source column unset").clone());
            self.col_mut(i).add_one_vector(a, src.as_ref(), c);
        }
        self.cache.bump();
    }

    /// `V ← a · U · C + b · V`. `C` must be a `DenseGenMatrix` of
    /// shape `(U.n_cols, V.n_cols)`. Port of `AddRightMultMatrix`.
    /// When `b == 0`, columns of `V` are reallocated first.
    pub fn add_right_mult_matrix(
        &mut self,
        a: Number,
        u: &MultiVectorMatrix,
        c_mat: &DenseGenMatrix,
        b: Number,
    ) {
        debug_assert_eq!(self.space.n_rows, u.space.n_rows);
        debug_assert_eq!(u.space.n_cols, c_mat.n_rows());
        debug_assert_eq!(self.space.n_cols, c_mat.n_cols());

        if b == 0.0 {
            self.fill_with_new_vectors();
        }

        let c_n_rows = c_mat.n_rows() as usize;
        let c_values = c_mat.values().to_vec();
        let temp_space = DenseVectorSpace::new(c_mat.n_rows());
        let mut tmp_dv = temp_space.make_new_dense();
        let nc = self.cols.len();
        for i in 0..nc {
            // Extract column i of C (column-major: C[j, i] = values[j + i * n_rows])
            let base = i * c_n_rows;
            let col_slice: Vec<Number> = c_values[base..base + c_n_rows].to_vec();
            tmp_dv.set_values(&col_slice);
            // Materialize a fresh tmp Rc for u.mult_vector — but mult_vector
            // takes &dyn Vector, so we can pass &tmp_dv directly.
            // However, u.mult_vector needs to write into self's column i.
            // Use col_mut for the destination.
            //
            // The temp dense vector is used only as input here.
            u.mult_vector(a, &tmp_dv, b, self.col_mut(i));
        }
        self.cache.bump();
    }
}

impl TaggedObject for MultiVectorMatrix {
    fn get_tag(&self) -> Tag {
        self.cache.tag()
    }
}

impl Matrix for MultiVectorMatrix {
    fn n_rows(&self) -> Index {
        self.space.n_rows
    }
    fn n_cols(&self) -> Index {
        self.space.n_cols
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

    /// `y ← α · V · x + β · y`, where `V·x = Σⱼ xⱼ · V[:, j]`.
    /// Port of `MultVectorImpl`. Reduction order matches upstream:
    /// scal/set y first, then iterate columns left-to-right and
    /// accumulate via `AddOneVector`.
    fn mult_vector_impl(&self, alpha: Number, x: &dyn Vector, beta: Number, y: &mut dyn Vector) {
        debug_assert_eq!(self.space.n_cols, x.dim());
        debug_assert_eq!(self.space.n_rows, y.dim());

        if beta != 0.0 {
            y.scal(beta);
        } else {
            y.set(0.0);
        }

        let dx = x
            .as_any()
            .downcast_ref::<DenseVector>()
            .expect("MultiVectorMatrix expects DenseVector input");

        if dx.is_homogeneous() {
            let val = dx.scalar();
            for i in 0..self.cols.len() {
                y.add_one_vector(alpha * val, self.col(i), 1.0);
            }
        } else {
            let values = dx.values();
            for i in 0..self.cols.len() {
                y.add_one_vector(alpha * values[i], self.col(i), 1.0);
            }
        }
    }

    /// `y[i] ← α · V[:, i]ᵀ · x + β · y[i]`. Port of
    /// `TransMultVectorImpl`. `y` must be a DenseVector (matching
    /// upstream's static_cast).
    fn trans_mult_vector_impl(
        &self,
        alpha: Number,
        x: &dyn Vector,
        beta: Number,
        y: &mut dyn Vector,
    ) {
        debug_assert_eq!(self.space.n_cols, y.dim());
        debug_assert_eq!(self.space.n_rows, x.dim());

        // Compute the dot products into a scratch vector first so we
        // don't borrow `y` immutably and mutably at the same time.
        let nc = self.cols.len();
        let mut dots = Vec::with_capacity(nc);
        for i in 0..nc {
            dots.push(self.col(i).dot(x));
        }

        let dy = y
            .as_any_mut()
            .downcast_mut::<DenseVector>()
            .expect("MultiVectorMatrix expects DenseVector output");
        // Force materialization (homogeneous storage can't take per-index writes).
        let yvals = dy.values_mut();
        if beta != 0.0 {
            for i in 0..nc {
                yvals[i] = alpha * dots[i] + beta * yvals[i];
            }
        } else {
            for i in 0..nc {
                yvals[i] = alpha * dots[i];
            }
        }
    }

    fn has_valid_numbers_impl(&self) -> bool {
        for slot in &self.cols {
            match slot {
                Some(v) => {
                    if !v.has_valid_numbers() {
                        return false;
                    }
                }
                None => return false,
            }
        }
        true
    }

    fn compute_row_amax_impl(&self, _rows_norms: &mut dyn Vector, _init: bool) {
        unimplemented!("MultiVectorMatrix::compute_row_amax — upstream throws UNIMPLEMENTED");
    }

    fn compute_col_amax_impl(&self, _cols_norms: &mut dyn Vector, _init: bool) {
        unimplemented!("MultiVectorMatrix::compute_col_amax — upstream throws UNIMPLEMENTED");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dense_gen_matrix::DenseGenMatrixSpace;

    fn dvec(values: &[Number]) -> Rc<DenseVector> {
        let space = DenseVectorSpace::new(values.len() as Index);
        let mut v = space.make_new_dense();
        v.set_values(values);
        Rc::new(v)
    }

    fn dvec_box(values: &[Number]) -> Box<DenseVector> {
        let space = DenseVectorSpace::new(values.len() as Index);
        let mut v = space.make_new_dense();
        v.set_values(values);
        Box::new(v)
    }

    fn build_mv(cols: &[&[Number]]) -> MultiVectorMatrix {
        let n_rows = cols[0].len() as Index;
        let n_cols = cols.len() as Index;
        let cs = DenseVectorSpace::new(n_rows);
        let space = MultiVectorMatrixSpace::new(n_cols, cs);
        let mut mv = space.make_new_multi_vector();
        for (i, c) in cols.iter().enumerate() {
            mv.set_vector(i as Index, dvec(c) as Rc<dyn Vector>);
        }
        mv
    }

    #[test]
    fn dimensions_match_space() {
        let mv = build_mv(&[&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]]);
        assert_eq!(mv.n_rows(), 3);
        assert_eq!(mv.n_cols(), 2);
    }

    #[test]
    fn mult_vector_combines_columns() {
        // V = [[1,4],[2,5],[3,6]], x = [10, 100]
        // V·x = [1*10+4*100, 2*10+5*100, 3*10+6*100] = [410, 520, 630]
        let mv = build_mv(&[&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]]);
        let x = dvec_box(&[10.0, 100.0]);
        let mut y = dvec_box(&[0.0, 0.0, 0.0]);
        mv.mult_vector(1.0, x.as_dyn_vector(), 0.0, y.as_mut());
        assert_eq!(y.expanded_values(), vec![410.0, 520.0, 630.0]);
    }

    #[test]
    fn mult_vector_alpha_beta_reduction_order() {
        // Same matrix; verify y ← 2 * V * x + 0.5 * y where y starts [10, 20, 30]
        // expected: 2*[410, 520, 630] + 0.5*[10, 20, 30] = [825, 1050, 1275]
        let mv = build_mv(&[&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]]);
        let x = dvec_box(&[10.0, 100.0]);
        let mut y = dvec_box(&[10.0, 20.0, 30.0]);
        mv.mult_vector(2.0, x.as_dyn_vector(), 0.5, y.as_mut());
        assert_eq!(y.expanded_values(), vec![825.0, 1050.0, 1275.0]);
    }

    #[test]
    fn trans_mult_vector_dot_products() {
        // V = [[1,4],[2,5],[3,6]], x = [1, 1, 1]
        // Vᵀ·x = [1+2+3, 4+5+6] = [6, 15]
        let mv = build_mv(&[&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]]);
        let x = dvec_box(&[1.0, 1.0, 1.0]);
        let mut y = dvec_box(&[0.0, 0.0]);
        mv.trans_mult_vector(1.0, x.as_dyn_vector(), 0.0, y.as_mut());
        assert_eq!(y.expanded_values(), vec![6.0, 15.0]);
    }

    #[test]
    fn trans_mult_vector_with_beta() {
        let mv = build_mv(&[&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]]);
        let x = dvec_box(&[1.0, 1.0, 1.0]);
        let mut y = dvec_box(&[100.0, 200.0]);
        // y ← 2*Vᵀx + 0.5*y = 2*[6,15] + [50,100] = [62, 130]
        mv.trans_mult_vector(2.0, x.as_dyn_vector(), 0.5, y.as_mut());
        assert_eq!(y.expanded_values(), vec![62.0, 130.0]);
    }

    #[test]
    fn lr_mult_vector_yields_v_v_t_x() {
        // V = [[1,0],[0,1],[0,0]] → V Vᵀ = diag(1,1,0)
        // For x = [3, 5, 7], y = [3, 5, 0].
        let mv = build_mv(&[&[1.0, 0.0, 0.0], &[0.0, 1.0, 0.0]]);
        let x = dvec_box(&[3.0, 5.0, 7.0]);
        let mut y = dvec_box(&[0.0, 0.0, 0.0]);
        mv.lr_mult_vector(1.0, x.as_dyn_vector(), 0.0, y.as_mut());
        assert_eq!(y.expanded_values(), vec![3.0, 5.0, 0.0]);
    }

    #[test]
    fn lr_mult_vector_alpha_beta() {
        let mv = build_mv(&[&[1.0, 0.0], &[0.0, 2.0]]);
        // V Vᵀ = [[1,0],[0,4]]; x = [10, 10]; V Vᵀ x = [10, 40]
        let x = dvec_box(&[10.0, 10.0]);
        let mut y = dvec_box(&[1.0, 1.0]);
        // y ← 2*[10,40] + 3*[1,1] = [23, 83]
        mv.lr_mult_vector(2.0, x.as_dyn_vector(), 3.0, y.as_mut());
        assert_eq!(y.expanded_values(), vec![23.0, 83.0]);
    }

    #[test]
    fn fill_with_new_vectors_initializes_columns() {
        let cs = DenseVectorSpace::new(3);
        let space = MultiVectorMatrixSpace::new(2, cs);
        let mut mv = space.make_new_multi_vector();
        mv.fill_with_new_vectors();
        // Both columns should now be set; mutating them should work.
        assert!(mv.cols[0].is_some());
        assert!(mv.cols[1].is_some());
    }

    #[test]
    fn scale_columns_per_index() {
        let mv0 = build_mv(&[&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]]);
        // We need owned (unique-rc) columns to mutate; rebuild with fresh vectors.
        let cs = DenseVectorSpace::new(3);
        let space = MultiVectorMatrixSpace::new(2, cs);
        let mut mv = space.make_new_multi_vector();
        mv.fill_with_new_vectors();
        // Manually set each column's values using col_mut.
        mv.col_mut(0).copy(mv0.get_vector(0).as_ref());
        mv.col_mut(1).copy(mv0.get_vector(1).as_ref());

        let scal = {
            let s = DenseVectorSpace::new(2);
            let mut v = s.make_new_dense();
            v.set_values(&[10.0, 100.0]);
            v
        };
        mv.scale_columns(&scal);
        // col0 ← col0 * 10 = [10, 20, 30]; col1 ← col1 * 100 = [400, 500, 600]
        let mut probe = dvec_box(&[1.0, 0.0]); // y = V * [1, 0] = col0
        let mut y = dvec_box(&[0.0, 0.0, 0.0]);
        mv.mult_vector(1.0, probe.as_dyn_vector(), 0.0, y.as_mut());
        assert_eq!(y.expanded_values(), vec![10.0, 20.0, 30.0]);
        probe.set_values(&[0.0, 1.0]);
        mv.mult_vector(1.0, probe.as_dyn_vector(), 0.0, y.as_mut());
        assert_eq!(y.expanded_values(), vec![400.0, 500.0, 600.0]);
    }

    #[test]
    fn scale_rows_multiplies_each_column() {
        let cs = DenseVectorSpace::new(3);
        let space = MultiVectorMatrixSpace::new(2, cs);
        let mut mv = space.make_new_multi_vector();
        mv.fill_with_new_vectors();
        let v0 = dvec(&[1.0, 2.0, 3.0]);
        let v1 = dvec(&[4.0, 5.0, 6.0]);
        mv.col_mut(0).copy(v0.as_ref());
        mv.col_mut(1).copy(v1.as_ref());

        let scal = dvec(&[10.0, 1.0, 1.0]);
        mv.scale_rows(scal.as_ref());
        // col0 = [10, 2, 3], col1 = [40, 5, 6]
        let mut x = dvec_box(&[1.0, 0.0]);
        let mut y = dvec_box(&[0.0, 0.0, 0.0]);
        mv.mult_vector(1.0, x.as_dyn_vector(), 0.0, y.as_mut());
        assert_eq!(y.expanded_values(), vec![10.0, 2.0, 3.0]);
        x.set_values(&[0.0, 1.0]);
        mv.mult_vector(1.0, x.as_dyn_vector(), 0.0, y.as_mut());
        assert_eq!(y.expanded_values(), vec![40.0, 5.0, 6.0]);
    }

    #[test]
    fn add_right_mult_matrix_v_eq_u_times_c() {
        // U = [[1, 0], [0, 1]] (2x2 identity-ish, two columns)
        // C = [[2, 3], [4, 5]] (2x2)
        // V = U·C = C itself.
        let cs = DenseVectorSpace::new(2);
        let u_space = MultiVectorMatrixSpace::new(2, Rc::clone(&cs));
        let mut u = u_space.make_new_multi_vector();
        u.set_vector(0, dvec(&[1.0, 0.0]) as Rc<dyn Vector>);
        u.set_vector(1, dvec(&[0.0, 1.0]) as Rc<dyn Vector>);

        let c_space = DenseGenMatrixSpace::new(2, 2);
        let mut c_mat = c_space.make_new_dense_gen();
        // Column-major: col0 = [2, 4]; col1 = [3, 5]
        c_mat.values_mut().copy_from_slice(&[2.0, 4.0, 3.0, 5.0]);

        let v_space = MultiVectorMatrixSpace::new(2, cs);
        let mut v = v_space.make_new_multi_vector();
        // b = 0 → V is freshly allocated, then V[:,i] = a · U · C[:,i]
        v.add_right_mult_matrix(1.0, &u, &c_mat, 0.0);

        // Probe column 0: y = V * [1, 0] = V[:, 0] = U·C[:,0] = [2, 4]
        let probe = dvec_box(&[1.0, 0.0]);
        let mut y = dvec_box(&[0.0, 0.0]);
        v.mult_vector(1.0, probe.as_dyn_vector(), 0.0, y.as_mut());
        assert_eq!(y.expanded_values(), vec![2.0, 4.0]);

        let probe1 = dvec_box(&[0.0, 1.0]);
        v.mult_vector(1.0, probe1.as_dyn_vector(), 0.0, y.as_mut());
        assert_eq!(y.expanded_values(), vec![3.0, 5.0]);
    }

    #[test]
    fn has_valid_numbers_detects_nan_in_column() {
        let cs = DenseVectorSpace::new(2);
        let space = MultiVectorMatrixSpace::new(2, cs);
        let mut mv = space.make_new_multi_vector();
        mv.set_vector(0, dvec(&[1.0, 2.0]) as Rc<dyn Vector>);
        mv.set_vector(1, dvec(&[f64::NAN, 0.0]) as Rc<dyn Vector>);
        assert!(!mv.has_valid_numbers());
    }
}
