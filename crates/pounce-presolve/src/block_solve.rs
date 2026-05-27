//! Lightweight damped-Newton solver for square ≤ 8-dim auxiliary
//! blocks, plus the `BlockSolver` trait for the larger-block fallback.
//!
//! Scaffolding only — PR 1 of the auxiliary-presolve port (issue #53).
//! Implementation lands in PR 6. The IPM-backed fallback for blocks
//! that exceed `presolve_auxiliary_max_block_dim` lands in PR 11.
//! ripopt anchor: `src/auxiliary_preprocessing.rs:1078-1182`.

/// Outcome of solving an auxiliary block. Populated by PR 6.
#[derive(Debug, Default)]
pub struct BlockSolution {
    _private: (),
}
