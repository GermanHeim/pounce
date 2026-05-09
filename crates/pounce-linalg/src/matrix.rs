//! Matrix trait + cache machinery.
//!
//! Mirrors `LinAlg/IpMatrix.{hpp,cpp}` and `LinAlg/IpSymMatrix.hpp`.
//! Like [`crate::vector::Vector`], the trait splits public BLAS-2 /
//! reduction routines (which manage the change tag and `valid` cache)
//! from `*_impl` methods that subclasses override.
//!
//! `SymMatrix` is a refinement of `Matrix` — concrete symmetric types
//! must impl both. Upstream gives `SymMatrix` a default
//! `TransMultVectorImpl` that calls `MultVector`; we provide the same
//! pattern via [`SymMatrix`] supplying default impls for the derived
//! quantities.
//!
//! Print is deferred to the iteration-output phase; the public method
//! is intentionally absent here so we don't drag the Journalist into
//! every concrete matrix in Phase 2.

use crate::vector::Vector;
use pounce_common::tagged::{Tag, TaggedCell, TaggedObject};
use pounce_common::types::{Index, Number};
use std::any::Any;
use std::cell::Cell;
use std::fmt::Debug;

/// Cached `valid_numbers` bit + change tag, embedded by every concrete
/// matrix type. Mirrors `Matrix::valid_cache_tag_` / `cached_valid_`.
#[derive(Debug)]
pub struct MatrixCache {
    tag: TaggedCell,
    valid: Cell<Option<(Tag, bool)>>,
}

impl Default for MatrixCache {
    fn default() -> Self {
        Self::new()
    }
}

impl MatrixCache {
    pub fn new() -> Self {
        Self {
            tag: TaggedCell::new(),
            valid: Cell::new(None),
        }
    }

    pub fn tag(&self) -> Tag {
        self.tag.tag()
    }

    /// Equivalent to `TaggedObject::ObjectChanged()`.
    pub fn bump(&self) {
        self.tag.bump();
    }
}

/// Matrix trait — full Ipopt `Matrix` API minus printing. Object-safe.
pub trait Matrix: TaggedObject + Debug + 'static {
    fn n_rows(&self) -> Index;
    fn n_cols(&self) -> Index;
    fn cache(&self) -> &MatrixCache;

    fn as_any(&self) -> &dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any;
    fn as_tagged(&self) -> &dyn TaggedObject;
    fn as_dyn_matrix(&self) -> &dyn Matrix;

    // ---- pure-virtual implementations ----

    /// `y ← α · M · x + β · y`.
    fn mult_vector_impl(&self, alpha: Number, x: &dyn Vector, beta: Number, y: &mut dyn Vector);

    /// `y ← α · Mᵀ · x + β · y`.
    fn trans_mult_vector_impl(
        &self,
        alpha: Number,
        x: &dyn Vector,
        beta: Number,
        y: &mut dyn Vector,
    );

    /// `rows_norms[i] ← max(rows_norms[i], maxⱼ |M[i,j]|)`. Caller has
    /// already zeroed `rows_norms` if `init`.
    fn compute_row_amax_impl(&self, rows_norms: &mut dyn Vector, init: bool);

    /// `cols_norms[j] ← max(cols_norms[j], maxᵢ |M[i,j]|)`. Caller has
    /// already zeroed `cols_norms` if `init`.
    fn compute_col_amax_impl(&self, cols_norms: &mut dyn Vector, init: bool);

    // ---- defaultable implementations ----

    /// Default returns true. Concrete matrices override when they hold
    /// floating-point storage that may go NaN/Inf.
    fn has_valid_numbers_impl(&self) -> bool {
        true
    }

    /// `X = X + α · M · S⁻¹ · Z`. Default: build `tmp = Z./S`, then
    /// `MultVector(α, tmp, 1, X)`. Override for ExpansionMatrix etc.
    fn add_m_sinv_z_impl(
        &self,
        alpha: Number,
        s: &dyn Vector,
        z: &dyn Vector,
        x: &mut dyn Vector,
    ) {
        let mut tmp = s.make_new_copy();
        // tmp ← (1)*Z/S + (0)*tmp  ≡  tmp = Z/S
        tmp.set(0.0);
        tmp.add_vector_quotient(1.0, z, s, 0.0);
        self.mult_vector(alpha, tmp.as_dyn_vector(), 1.0, x);
    }

    /// `X = S⁻¹ · (R + α · Z · Mᵀ · D)`. Default per upstream
    /// `Matrix::SinvBlrmZMTdBrImpl`.
    fn sinv_blrm_zmt_dbr_impl(
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

    // ---- public API (cache-aware wrappers) ----

    fn mult_vector(&self, alpha: Number, x: &dyn Vector, beta: Number, y: &mut dyn Vector) {
        self.mult_vector_impl(alpha, x, beta, y);
    }

    fn trans_mult_vector(&self, alpha: Number, x: &dyn Vector, beta: Number, y: &mut dyn Vector) {
        self.trans_mult_vector_impl(alpha, x, beta, y);
    }

    fn compute_row_amax(&self, rows_norms: &mut dyn Vector, init: bool) {
        if init {
            rows_norms.set(0.0);
        }
        self.compute_row_amax_impl(rows_norms, init);
    }

    fn compute_col_amax(&self, cols_norms: &mut dyn Vector, init: bool) {
        if init {
            cols_norms.set(0.0);
        }
        self.compute_col_amax_impl(cols_norms, init);
    }

    fn add_m_sinv_z(&self, alpha: Number, s: &dyn Vector, z: &dyn Vector, x: &mut dyn Vector) {
        self.add_m_sinv_z_impl(alpha, s, z, x);
    }

    fn sinv_blrm_zmt_dbr(
        &self,
        alpha: Number,
        s: &dyn Vector,
        r: &dyn Vector,
        z: &dyn Vector,
        d: &dyn Vector,
        x: &mut dyn Vector,
    ) {
        self.sinv_blrm_zmt_dbr_impl(alpha, s, r, z, d, x);
    }

    fn has_valid_numbers(&self) -> bool {
        let cur = self.cache().tag();
        if let Some((t, v)) = self.cache().valid.get() {
            if t == cur {
                return v;
            }
        }
        let v = self.has_valid_numbers_impl();
        self.cache().valid.set(Some((cur, v)));
        v
    }
}

/// Symmetric refinement of [`Matrix`]. A concrete symmetric matrix
/// implements both [`Matrix`] and [`SymMatrix`]; convenience helpers
/// for the symmetric-derived overrides are provided as free functions
/// `sym_default_*` which concrete impls can call from their own
/// `Matrix::*_impl` bodies.
pub trait SymMatrix: Matrix {
    /// `dim()` is identical to `n_rows()` and `n_cols()` by symmetry,
    /// but mirroring upstream's `SymMatrix::Dim()` accessor is helpful
    /// for clarity at call sites.
    fn dim(&self) -> Index {
        debug_assert_eq!(self.n_rows(), self.n_cols());
        self.n_rows()
    }
}

/// Helper that concrete symmetric matrices may forward to from
/// [`Matrix::trans_mult_vector_impl`] — exactly what upstream
/// `SymMatrix::TransMultVectorImpl` does.
#[inline]
pub fn sym_default_trans_mult_vector_impl<M: Matrix + ?Sized>(
    m: &M,
    alpha: Number,
    x: &dyn Vector,
    beta: Number,
    y: &mut dyn Vector,
) {
    m.mult_vector_impl(alpha, x, beta, y);
}

/// Helper that concrete symmetric matrices may forward to from
/// [`Matrix::compute_col_amax_impl`] — exactly what upstream
/// `SymMatrix::ComputeColAMaxImpl` does (row==col by symmetry).
#[inline]
pub fn sym_default_compute_col_amax_impl<M: Matrix + ?Sized>(
    m: &M,
    cols_norms: &mut dyn Vector,
    init: bool,
) {
    m.compute_row_amax_impl(cols_norms, init);
}
