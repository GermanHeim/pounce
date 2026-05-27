//! Weakly-connected component extraction on the square-matched part
//! of a Dulmage-Mendelsohn partition.
//!
//! Scaffolding only — PR 1 of the auxiliary-presolve port (issue #53).
//! Implementation lands in PR 3. ripopt anchor:
//! `src/auxiliary_preprocessing.rs:2416-2469`.

/// Component decomposition produced by PR 3.
#[derive(Debug, Default)]
pub struct ConnectedComponents {
    _private: (),
}
