//! Hopcroft-Karp bipartite matching on an [`EqualityIncidence`].
//!
//! Scaffolding only — PR 1 of the auxiliary-presolve port (issue #53).
//! Implementation lands in PR 2. ripopt anchor:
//! `src/auxiliary_preprocessing.rs:2280-2318`.

/// Maximum-cardinality bipartite matching, computed by PR 2 with
/// Hopcroft-Karp. PR 1 ships an empty placeholder so the module tree
/// compiles.
#[derive(Debug, Default)]
pub struct BipartiteMatching {
    _private: (),
}
