//! Build a `pounce_qp::QpProblem` from the NLP linearization at
//! the current SQP iterate `(x, λ_g)`.
//!
//! The standard SQP QP subproblem (Nocedal-Wright §18.1):
//!
//! ```text
//!     min  ½ pᵀ ∇²L(x, λ) p + ∇f(x)ᵀ p
//!     s.t.   c_eq(x) + ∇c_eq(x) p = 0
//!            c_in(x) + ∇c_in(x) p ≥ 0           (one-sided)
//!            xl − x ≤ p ≤ xu − x                (bounds on p)
//! ```
//!
//! Phase 5b commit 1 ships only the assembly skeleton; the
//! caller / SqpAlgorithm wires it into the full loop in the
//! next commit.

// Implementation lands in the next commit alongside SqpAlgorithm.
