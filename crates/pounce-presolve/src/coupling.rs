//! Coupling classification for candidate auxiliary blocks.
//!
//! PR 1 lands the enum so [`crate::options::AuxiliaryCouplingPolicy`]
//! can refer to it; the classifier itself (objective-gradient probe
//! plus inequality-row incidence) lands in PR 5. ripopt anchor:
//! `src/auxiliary_preprocessing.rs:39-59, 1642-1687`.

/// How a candidate block is coupled to the rest of the problem.
///
/// Drives the elimination policy: `PureEquality` is always eligible
/// under `AuxiliaryCouplingPolicy::Safe`, `ObjectiveCoupled` adds in
/// under `Aggressive`, and the two inequality-coupled variants are
/// never eliminated in v1 (matches ripopt's conservative default).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AuxiliaryCouplingClass {
    /// Block touches only equality rows and not the objective.
    PureEquality,
    /// Block variables appear in the objective gradient.
    ObjectiveCoupled,
    /// Block variables appear in at least one inequality row.
    InequalityCoupled,
    /// Both of the above.
    ObjectiveAndInequalityCoupled,
}
