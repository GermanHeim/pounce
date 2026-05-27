//! Tarjan SCC + topological order → block-triangular form on each
//! connected component of the square-matched part.
//!
//! Scaffolding only — PR 1 of the auxiliary-presolve port (issue #53).
//! Implementation lands in PR 4. ripopt anchor:
//! `src/auxiliary_preprocessing.rs:2473-2552`.

/// Block-triangular ordering produced by PR 4.
#[derive(Debug, Default)]
pub struct BlockTriangularForm {
    _private: (),
}
