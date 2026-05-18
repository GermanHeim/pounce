//! Vector trait + cache machinery.
//!
//! Mirrors `LinAlg/IpVector.{hpp,cpp}`. The trait splits public BLAS-1
//! / element-wise routines (which manage the change tag and cached
//! scalars) from `*_impl` methods that subclasses override. Concrete
//! implementations are responsible for **never** bumping the tag from
//! inside their own `*_impl` body — only the public wrappers do that,
//! exactly mirroring upstream's split between `Vector::Foo` and
//! `Vector::FooImpl`.

use pounce_common::cached::Cache;
use pounce_common::tagged::{Tag, TaggedCell, TaggedObject};
use pounce_common::types::{Index, Number};
use std::any::Any;
use std::cell::{Cell, RefCell};
use std::fmt::Debug;

/// Cached scalar reductions + dot cache + change tag, embedded by
/// every concrete vector type. Mirrors the mutable members of
/// upstream `Vector` (the `cached_*` fields and `dot_cache_`).
#[derive(Debug)]
pub struct VectorCache {
    tag: TaggedCell,
    nrm2: Cell<Option<(Tag, Number)>>,
    asum: Cell<Option<(Tag, Number)>>,
    amax: Cell<Option<(Tag, Number)>>,
    max: Cell<Option<(Tag, Number)>>,
    min: Cell<Option<(Tag, Number)>>,
    sum: Cell<Option<(Tag, Number)>>,
    sum_logs: Cell<Option<(Tag, Number)>>,
    valid: Cell<Option<(Tag, bool)>>,
    dot: RefCell<Cache<Number>>,
}

impl Default for VectorCache {
    fn default() -> Self {
        Self::new()
    }
}

impl VectorCache {
    /// Upstream `Vector` constructs `dot_cache_(10)`.
    pub fn new() -> Self {
        Self {
            tag: TaggedCell::new(),
            nrm2: Cell::new(None),
            asum: Cell::new(None),
            amax: Cell::new(None),
            max: Cell::new(None),
            min: Cell::new(None),
            sum: Cell::new(None),
            sum_logs: Cell::new(None),
            valid: Cell::new(None),
            dot: RefCell::new(Cache::new(10)),
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

/// Vector trait — full Ipopt `Vector` API. Object-safe.
pub trait Vector: TaggedObject + Debug + 'static {
    fn dim(&self) -> Index;
    fn cache(&self) -> &VectorCache;

    /// Create a new uninitialized vector belonging to the same
    /// `VectorSpace`. Equivalent to `Vector::MakeNew`.
    fn make_new(&self) -> Box<dyn Vector>;

    fn as_any(&self) -> &dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any;
    fn as_tagged(&self) -> &dyn TaggedObject;
    fn as_dyn_vector(&self) -> &dyn Vector;

    // ---- pure-virtual implementations ----

    fn copy_impl(&mut self, x: &dyn Vector);
    fn scal_impl(&mut self, alpha: Number);
    fn axpy_impl(&mut self, alpha: Number, x: &dyn Vector);
    fn dot_impl(&self, x: &dyn Vector) -> Number;
    fn nrm2_impl(&self) -> Number;
    fn asum_impl(&self) -> Number;
    fn amax_impl(&self) -> Number;
    fn set_impl(&mut self, alpha: Number);
    fn element_wise_divide_impl(&mut self, x: &dyn Vector);
    fn element_wise_multiply_impl(&mut self, x: &dyn Vector);
    fn element_wise_select_impl(&mut self, x: &dyn Vector);
    fn element_wise_max_impl(&mut self, x: &dyn Vector);
    fn element_wise_min_impl(&mut self, x: &dyn Vector);
    fn element_wise_reciprocal_impl(&mut self);
    fn element_wise_abs_impl(&mut self);
    fn element_wise_sqrt_impl(&mut self);
    fn element_wise_sgn_impl(&mut self);
    fn add_scalar_impl(&mut self, scalar: Number);
    fn max_impl(&self) -> Number;
    fn min_impl(&self) -> Number;
    fn sum_impl(&self) -> Number;
    fn sum_logs_impl(&self) -> Number;
    fn frac_to_bound_impl(&self, delta: &dyn Vector, tau: Number) -> Number;
    fn add_vector_quotient_impl(&mut self, a: Number, z: &dyn Vector, s: &dyn Vector, c: Number);

    // ---- defaultable implementations ----

    /// Default fallback. Concrete types override for efficiency, but
    /// the result must remain bit-identical to upstream's
    /// `DenseVector::AddTwoVectorsImpl` decision tree.
    fn add_two_vectors_impl(
        &mut self,
        a: Number,
        v1: &dyn Vector,
        b: Number,
        v2: &dyn Vector,
        c: Number,
    ) {
        if c == 0.0 {
            self.set_impl(0.0);
        } else if c != 1.0 {
            self.scal_impl(c);
        }
        if a != 0.0 {
            self.axpy_impl(a, v1);
        }
        if b != 0.0 {
            self.axpy_impl(b, v2);
        }
    }

    /// Default uses `Asum` finiteness — matches upstream
    /// `Vector::HasValidNumbersImpl`.
    fn has_valid_numbers_impl(&self) -> bool {
        self.asum_impl().is_finite()
    }

    // ---- public API (cache-aware wrappers) ----

    fn copy(&mut self, x: &dyn Vector) {
        self.copy_impl(x);
        self.cache().bump();
    }

    fn make_new_copy(&self) -> Box<dyn Vector> {
        let mut c = self.make_new();
        c.copy(self.as_dyn_vector());
        c
    }

    fn scal(&mut self, alpha: Number) {
        self.scal_impl(alpha);
        self.cache().bump();
    }

    fn axpy(&mut self, alpha: Number, x: &dyn Vector) {
        self.axpy_impl(alpha, x);
        self.cache().bump();
    }

    fn dot(&self, x: &dyn Vector) -> Number {
        // Same-vector shortcut — upstream uses `this == &x` with the
        // explanation that the cache cannot key on a self-dependency.
        // We compare cache addresses (each Vector owns a unique
        // VectorCache at a stable address).
        if std::ptr::eq(self.cache() as *const _, x.cache() as *const _) {
            let n = self.nrm2();
            return n * n;
        }
        let mut dot_cache = self.cache().dot.borrow_mut();
        if let Some(v) = dot_cache.get(&[self.as_tagged(), x.as_tagged()], &[]) {
            return v;
        }
        let v = self.dot_impl(x);
        dot_cache.add(v, &[self.as_tagged(), x.as_tagged()], &[]);
        v
    }

    fn nrm2(&self) -> Number {
        let cur = self.cache().tag();
        if let Some((t, v)) = self.cache().nrm2.get() {
            if t == cur {
                return v;
            }
        }
        let v = self.nrm2_impl();
        self.cache().nrm2.set(Some((cur, v)));
        v
    }

    fn asum(&self) -> Number {
        let cur = self.cache().tag();
        if let Some((t, v)) = self.cache().asum.get() {
            if t == cur {
                return v;
            }
        }
        let v = self.asum_impl();
        self.cache().asum.set(Some((cur, v)));
        v
    }

    fn amax(&self) -> Number {
        let cur = self.cache().tag();
        if let Some((t, v)) = self.cache().amax.get() {
            if t == cur {
                return v;
            }
        }
        let v = self.amax_impl();
        self.cache().amax.set(Some((cur, v)));
        v
    }

    fn set(&mut self, alpha: Number) {
        self.set_impl(alpha);
        self.cache().bump();
    }

    fn element_wise_divide(&mut self, x: &dyn Vector) {
        self.element_wise_divide_impl(x);
        self.cache().bump();
    }
    fn element_wise_multiply(&mut self, x: &dyn Vector) {
        self.element_wise_multiply_impl(x);
        self.cache().bump();
    }
    fn element_wise_select(&mut self, x: &dyn Vector) {
        self.element_wise_select_impl(x);
        self.cache().bump();
    }
    fn element_wise_max(&mut self, x: &dyn Vector) {
        self.element_wise_max_impl(x);
        self.cache().bump();
    }
    fn element_wise_min(&mut self, x: &dyn Vector) {
        self.element_wise_min_impl(x);
        self.cache().bump();
    }
    fn element_wise_reciprocal(&mut self) {
        self.element_wise_reciprocal_impl();
        self.cache().bump();
    }
    fn element_wise_abs(&mut self) {
        self.element_wise_abs_impl();
        self.cache().bump();
    }
    fn element_wise_sqrt(&mut self) {
        self.element_wise_sqrt_impl();
        self.cache().bump();
    }
    fn element_wise_sgn(&mut self) {
        self.element_wise_sgn_impl();
        self.cache().bump();
    }
    fn add_scalar(&mut self, scalar: Number) {
        self.add_scalar_impl(scalar);
        self.cache().bump();
    }

    fn max(&self) -> Number {
        let cur = self.cache().tag();
        if let Some((t, v)) = self.cache().max.get() {
            if t == cur {
                return v;
            }
        }
        let v = self.max_impl();
        self.cache().max.set(Some((cur, v)));
        v
    }

    fn min(&self) -> Number {
        let cur = self.cache().tag();
        if let Some((t, v)) = self.cache().min.get() {
            if t == cur {
                return v;
            }
        }
        let v = self.min_impl();
        self.cache().min.set(Some((cur, v)));
        v
    }

    fn sum(&self) -> Number {
        let cur = self.cache().tag();
        if let Some((t, v)) = self.cache().sum.get() {
            if t == cur {
                return v;
            }
        }
        let v = self.sum_impl();
        self.cache().sum.set(Some((cur, v)));
        v
    }

    fn sum_logs(&self) -> Number {
        let cur = self.cache().tag();
        if let Some((t, v)) = self.cache().sum_logs.get() {
            if t == cur {
                return v;
            }
        }
        let v = self.sum_logs_impl();
        self.cache().sum_logs.set(Some((cur, v)));
        v
    }

    fn add_one_vector(&mut self, a: Number, v1: &dyn Vector, c: Number) {
        // Upstream: AddTwoVectors(a, v1, 0., v1, c).
        self.add_two_vectors(a, v1, 0.0, v1, c);
    }

    fn add_two_vectors(
        &mut self,
        a: Number,
        v1: &dyn Vector,
        b: Number,
        v2: &dyn Vector,
        c: Number,
    ) {
        self.add_two_vectors_impl(a, v1, b, v2, c);
        self.cache().bump();
    }

    /// No cache (matches upstream comment in `IpVector.hpp:820` —
    /// caching here interferes with the quality function search).
    fn frac_to_bound(&self, delta: &dyn Vector, tau: Number) -> Number {
        self.frac_to_bound_impl(delta, tau)
    }

    fn add_vector_quotient(&mut self, a: Number, z: &dyn Vector, s: &dyn Vector, c: Number) {
        self.add_vector_quotient_impl(a, z, s, c);
        self.cache().bump();
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
