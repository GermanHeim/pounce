//! KKT-system assembly from a [`QpProblem`].
//!
//! For an equality-constrained QP `min ½xᵀHx + gᵀx s.t. Ax = b`, the
//! KKT system is the symmetric saddle-point matrix
//!
//! ```text
//!     ┌ H   Aᵀ ┐ ┌ x ┐   ┌ -g ┐
//!     │        │ │   │ = │    │
//!     └ A    0 ┘ └ λ ┘   └  b ┘
//! ```
//!
//! with Lagrangian sign convention `L = ½xᵀHx + gᵀx + λᵀ(Ax − b)` (so
//! `∇_x L = Hx + g + Aᵀλ`).
//!
//! The assembly emits triplets in the format
//! [`EMatrixFormat::TripletFormat`](pounce_linsol::EMatrixFormat::TripletFormat)
//! that FERAL / MA57 / MUMPS consume: **lower triangle, 1-based
//! indices**. Pounce-linalg's `SymTMatrix` stores one of each
//! symmetric pair (upper *or* lower; the convention is not pinned),
//! so we normalize each H entry to `irow ≥ jcol` defensively. A
//! entries land at rows `(n + i)`, which is automatically below the
//! diagonal of the augmented matrix and therefore lower-triangular
//! without further work.
//!
//! Phase 5a commit 2 covers the equality-only path. The inequality-
//! handling (bounds + general one-sided constraints + working-set
//! adds/drops) lands with the §4.2 homotopy machinery.

use crate::problem::QpProblem;
use pounce_common::{Index, Number};

/// A KKT system in FERAL-compatible triplet form.
///
/// `dim` is the dimension of the full symmetric matrix
/// (`n + n_active`, where `n_active` is the number of active rows of
/// `A` — currently equal to `m` since only equality QPs are
/// supported). `irn`, `jcn`, `vals` are parallel arrays describing
/// the lower-triangle nonzeros in 1-based indexing.
#[derive(Debug, Clone)]
pub struct KktTriplet {
    pub dim: usize,
    pub irn: Vec<Index>,
    pub jcn: Vec<Index>,
    pub vals: Vec<Number>,
}

impl KktTriplet {
    /// Assemble the equality-only KKT matrix `[H Aᵀ; A 0]` for the
    /// QP. Caller must have validated the QP (see
    /// [`QpProblem::validate`]) and verified
    /// `is_pure_equality_no_bounds(qp)`.
    pub fn assemble_equality_only(qp: &QpProblem) -> Self {
        let n = qp.n;
        let m = qp.m;
        let dim = n + m;

        let nh = qp.h.nonzeros() as usize;
        let na = qp.a.nonzeros() as usize;
        let cap = nh + na;

        let mut irn = Vec::with_capacity(cap);
        let mut jcn = Vec::with_capacity(cap);
        let mut vals = Vec::with_capacity(cap);

        // ---- H block (top-left), lower-triangle normalized ----
        let h_irows = qp.h.irows();
        let h_jcols = qp.h.jcols();
        let h_vals = qp.h.values();
        for k in 0..nh {
            let i = h_irows[k];
            let j = h_jcols[k];
            let (lo, hi) = if i >= j { (j, i) } else { (i, j) };
            irn.push(hi);
            jcn.push(lo);
            vals.push(h_vals[k]);
        }

        // ---- A block (bottom-left), rows shifted by n ----
        let a_irows = qp.a.irows();
        let a_jcols = qp.a.jcols();
        let a_vals = qp.a.values();
        let n_i = n as Index;
        for k in 0..na {
            irn.push(n_i + a_irows[k]);
            jcn.push(a_jcols[k]);
            vals.push(a_vals[k]);
        }

        // ---- (2,2) zero block is implicit ----

        Self {
            dim,
            irn,
            jcn,
            vals,
        }
    }
}

/// Right-hand side `[-g; b]` for the equality-only KKT.
///
/// For each general-constraint row the target value `b` is taken
/// from `bl` (caller has already verified `bl[i] == bu[i]`, which is
/// the equality-only contract).
pub fn rhs_equality_only(qp: &QpProblem) -> Vec<Number> {
    let mut rhs = Vec::with_capacity(qp.n + qp.m);
    rhs.extend(qp.g.iter().map(|&gi| -gi));
    rhs.extend_from_slice(qp.bl);
    rhs
}

/// Is this QP in the equality-only / no-variable-bounds subset that
/// commit 2 can solve directly? Caller routes to this fast path
/// when the predicate holds.
///
/// Concretely:
/// * every general-constraint row is an equality (`bl[i] == bu[i]`);
/// * every variable is free (`xl[i] ≤ -1e19` and `xu[i] ≥ +1e19`,
///   matching the `NLP_*_BOUND_INF` convention pounce uses).
pub fn is_pure_equality_no_bounds(qp: &QpProblem) -> bool {
    use pounce_common::types::{NLP_LOWER_BOUND_INF, NLP_UPPER_BOUND_INF};
    for (&l, &u) in qp.bl.iter().zip(qp.bu.iter()) {
        if l != u {
            return false;
        }
    }
    for (&l, &u) in qp.xl.iter().zip(qp.xu.iter()) {
        if l > NLP_LOWER_BOUND_INF || u < NLP_UPPER_BOUND_INF {
            return false;
        }
    }
    true
}
