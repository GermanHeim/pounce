//! Dulmage-Mendelsohn partition into under- / square- / overdetermined
//! parts.
//!
//! Scaffolding only — PR 1 of the auxiliary-presolve port (issue #53).
//! Implementation lands in PR 3. ripopt anchor:
//! `src/auxiliary_preprocessing.rs:2320-2413`.

/// Three-way partition over (rows, vars) produced by PR 3.
#[derive(Debug, Default)]
pub struct DulmageMendelsohnPartition {
    _private: (),
}
