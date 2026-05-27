//! Bipartite incidence graph between equality rows and variables.
//!
//! Scaffolding only — PR 1 of the auxiliary-presolve port (issue #53).
//! The real implementation lands in PR 2 alongside Hopcroft-Karp
//! matching in [`crate::matching`]. ripopt anchor:
//! `src/auxiliary_preprocessing.rs:2282-2318`.

/// Placeholder for the equality-row × variable bipartite graph that
/// PR 2 will build from `eval_jac_g(Structure)` +
/// `get_constraints_linearity` + the equality filter on `(g_l, g_u)`.
#[derive(Debug, Default)]
pub struct EqualityIncidence {
    // PR 2 — fields filled in: adjacency CSR (row → vars), reverse
    // adjacency (var → rows), per-row equality flag, etc.
    _private: (),
}
