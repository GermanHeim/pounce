//! Cone abstraction for the convex IPM.
//!
//! Phase 2 of the LP/QP plan builds the interior-point iteration over a
//! `Cone` abstraction with only the nonnegative orthant implemented, so
//! that Phases 4–6 (SOCP / exponential / power / PSD) are cone
//! *extensions* rather than a rewrite (see `dev-notes/lp-qp-routing.md`).
//!
//! A cone owns everything the IPM needs that is cone-specific:
//! - the central-path measure `μ = ⟨s, z⟩ / degree`,
//! - the scaling block that enters the KKT system,
//! - the complementarity residual `s ∘ z - σμ e`,
//! - the fraction-to-boundary step length keeping `(s, z)` in the cone.
//!
//! The IPM driver (`crate::ipm`) is otherwise cone-agnostic. For the
//! nonnegative orthant (LP/QP) the "∘" product is elementwise and the
//! scaling block is the diagonal `s ⊘ z`; richer cones override these
//! with their Nesterov–Todd scaling.

pub mod composite;
pub mod nonneg;

pub use composite::{CompositeCone, ConeKind};
pub use nonneg::NonnegCone;

/// A symmetric cone over which the IPM maintains a primal slack `s` and
/// dual `z`. Phase 2 ships only [`NonnegCone`]; the trait exists so the
/// driver code does not bake in the orthant.
pub trait Cone {
    /// Barrier degree (the orthant's is its dimension). Used to form the
    /// central-path parameter `μ = ⟨s, z⟩ / degree`.
    fn degree(&self) -> usize;

    /// Dimension of the slack/dual vectors this cone owns.
    fn dim(&self) -> usize;

    /// Duality measure `⟨s, z⟩ / degree`.
    fn mu(&self, s: &[f64], z: &[f64]) -> f64;

    /// Diagonal of the cone's scaling block as it enters the (z, z)
    /// position of the symmetric KKT system. For the nonnegative orthant
    /// this is `s ⊘ z`; the IPM places `-scaling` on that diagonal.
    fn scaling_diag(&self, s: &[f64], z: &[f64], out: &mut [f64]);

    /// Complementarity residual `r = s ∘ z - σμ e`. With `sigma_mu = 0`
    /// this is the affine (predictor) target; with `σμ > 0` it is the
    /// centered path-following target.
    fn comp_residual(&self, s: &[f64], z: &[f64], sigma_mu: f64, out: &mut [f64]);

    /// Mehrotra corrector complementarity residual
    /// `r = s ∘ z + ds_aff ∘ dz_aff - σμ e`, where `ds_aff`/`dz_aff` are
    /// the affine-predictor steps. The `ds_aff ∘ dz_aff` second-order
    /// term is what gives Mehrotra its faster convergence; it is
    /// cone-specific (elementwise for the orthant), so it lives behind
    /// this trait rather than in the driver.
    fn comp_residual_corrector(
        &self,
        s: &[f64],
        z: &[f64],
        ds_aff: &[f64],
        dz_aff: &[f64],
        sigma_mu: f64,
        out: &mut [f64],
    );

    /// Recover the slack step `ds` from the dual step `dz` and the
    /// complementarity residual, given the current `(s, z)`:
    /// `ds = -(r_comp ⊘ z) - (s ⊘ z) ∘ dz`.
    fn recover_ds(&self, s: &[f64], z: &[f64], r_comp: &[f64], dz: &[f64], ds: &mut [f64]);

    /// Largest `α ∈ (0, 1]` such that `v + α dv` stays inside the cone,
    /// scaled by the fraction-to-boundary parameter `tau`. For the
    /// orthant: `min over dv_i<0 of -tau * v_i / dv_i`, capped at 1.
    fn max_step(&self, v: &[f64], dv: &[f64], tau: f64) -> f64;
}
