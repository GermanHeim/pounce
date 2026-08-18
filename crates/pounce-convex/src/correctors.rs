//! Gondzio multiple centrality correctors — the pieces both symmetric drivers
//! share.
//!
//! The scheme (Gondzio 1996, "Multiple centrality corrections in a primal–dual
//! method for linear programming") sits *after* the Mehrotra corrector and
//! reuses the factorization already in hand: each pass is one extra back-solve,
//! never a refactorization. It forms a trial step enlarged by [`DELTA`],
//! projects the complementarity products that step would produce into the
//! centered box `[β_lo·μ, β_hi·μ]`, solves for the correction that moves them
//! back toward the box, and keeps it only if the fraction-to-boundary step
//! grows by at least `GAMMA·DELTA`.
//!
//! Two drivers want that machinery and disagree on everything around it. The
//! HSDE loop ([`crate::hsde`]) carries the homogenizing pair `(τ, κ)` as a
//! third complementarity product and takes one *symmetric* step length; the
//! direct loop ([`crate::ipm::run_ipm`]) has no `(τ, κ)` and takes *split*
//! primal/dual lengths. What is genuinely common is the band, the projection,
//! the enlarged trial length, and the acceptance test — those live here, and
//! each driver keeps its own Schur solve and step-length rule.
//!
//! Restricted by both callers to the **pure nonnegative orthant**: there the
//! complementarity product `s ∘ z` is elementwise, so it can be box-projected
//! componentwise. A second-order or PSD block's Jordan product would need the
//! spectral machinery this module deliberately does not carry — see
//! [`crate::cones::composite::CompositeCone::is_orthant`].

/// Maximum extra corrections per iteration, and the tuning constants of the
/// acceptance rule.
///
/// Each corrector is accepted only if it lengthens the fraction-to-boundary
/// step by at least `GAMMA·DELTA`; otherwise correction stops. Lengthening the
/// step is exactly the documented purpose of the scheme — and it is what breaks
/// the degenerate-face step collapse (σ→1 centering freezing μ) on the NETLIB
/// GEN family, where the Mehrotra corrector alone stalls. `β_lo = 0.1`,
/// `β_hi = 10` is Gondzio's recommended symmetric box.
pub(crate) const MAX_CORR: usize = 3;
pub(crate) const DELTA: f64 = 0.1;
pub(crate) const GAMMA: f64 = 0.1;
pub(crate) const BETA_LO: f64 = 0.1;
pub(crate) const BETA_HI: f64 = 10.0;

/// The centered box `[β_lo·μ, β_hi·μ]` a corrector projects into.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Band {
    pub(crate) lo: f64,
    pub(crate) hi: f64,
}

impl Band {
    /// Gondzio's symmetric box around the current duality measure.
    pub(crate) fn around(mu: f64) -> Self {
        Band {
            lo: BETA_LO * mu,
            hi: BETA_HI * mu,
        }
    }

    /// The centered target `t` for one complementarity product: `ṽ` clamped
    /// into `[lo, hi]`.
    pub(crate) fn project(&self, v: f64) -> f64 {
        v.clamp(self.lo, self.hi)
    }

    /// Deviation of one complementarity product from the box: `ṽ − t`. Zero
    /// when the product is already centered, which is what makes an
    /// all-centered iteration cost nothing.
    ///
    /// Written as the subtraction rather than derived from [`Self::project`]
    /// by the caller so that both drivers agree bit-for-bit on the deviation
    /// *and* on the target; `v - (v - project(v))` is not `project(v)` in
    /// floating point once `v` is many orders above `hi`.
    pub(crate) fn deviation(&self, v: f64) -> f64 {
        v - self.project(v)
    }
}

/// The enlarged trial step length `min(α + DELTA, 1)` a corrector aims at.
pub(crate) fn trial_step(alpha: f64) -> f64 {
    (alpha + DELTA).min(1.0)
}

/// Whether a corrector earned its back-solve: it must lengthen the step by at
/// least `GAMMA·DELTA`, otherwise the caller stops correcting.
pub(crate) fn accepts(alpha_new: f64, alpha: f64) -> bool {
    alpha_new >= alpha + GAMMA * DELTA
}

/// Project the complementarity products of the enlarged trial step
/// `(s + α_p·ds) ∘ (z + α_d·dz)` into `band`, writing each deviation `ṽ − t`
/// into `out` so that the caller's `recover_ds` yields a correction with
/// `z ∘ cds + s ∘ cdz = t − ṽ`.
///
/// Returns whether *any* component left the band — an iteration whose products
/// are all centered has nothing to correct and the caller breaks out before
/// spending a back-solve.
///
/// `alpha` is the `(primal, dual)` trial pair — equal in the symmetric (HSDE)
/// case, split in the direct driver's.
pub(crate) fn project_products(
    band: Band,
    (s, ds): (&[f64], &[f64]),
    (z, dz): (&[f64], &[f64]),
    alpha: (f64, f64),
    out: &mut [f64],
) -> bool {
    let mut active = false;
    for i in 0..out.len() {
        let v = (s[i] + alpha.0 * ds[i]) * (z[i] + alpha.1 * dz[i]);
        out[i] = band.deviation(v);
        if out[i] != 0.0 {
            active = true;
        }
    }
    active
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_centered_product_has_no_deviation() {
        let b = Band::around(1.0);
        assert_eq!(b.lo, 0.1);
        assert_eq!(b.hi, 10.0);
        assert_eq!(b.deviation(1.0), 0.0);
        assert_eq!(b.deviation(0.1), 0.0);
        assert_eq!(b.deviation(10.0), 0.0);
    }

    #[test]
    fn deviation_is_signed_toward_the_nearer_face() {
        let b = Band::around(1.0);
        // Too small: the product must grow, so the deviation is negative.
        assert!((b.deviation(0.01) - (0.01 - 0.1)).abs() < 1e-18);
        // Too large: positive.
        assert!((b.deviation(100.0) - 90.0).abs() < 1e-18);
    }

    #[test]
    fn the_trial_step_is_capped_at_one() {
        assert!((trial_step(0.5) - 0.6).abs() < 1e-15);
        assert_eq!(trial_step(0.99), 1.0);
        assert_eq!(trial_step(1.0), 1.0);
    }

    #[test]
    fn acceptance_needs_the_full_gamma_delta_gain() {
        assert!(accepts(0.51, 0.5));
        assert!(accepts(0.5 + GAMMA * DELTA, 0.5));
        assert!(!accepts(0.5 + 0.5 * GAMMA * DELTA, 0.5));
        assert!(!accepts(0.4, 0.5));
    }

    /// An all-centered iterate reports "nothing to correct", which is the
    /// signal both drivers use to break before spending a back-solve.
    #[test]
    fn projection_reports_whether_anything_left_the_band() {
        let band = Band::around(1.0);
        let (s, z) = (vec![1.0, 1.0], vec![1.0, 1.0]);
        let (ds, dz) = (vec![0.0, 0.0], vec![0.0, 0.0]);
        let mut out = vec![0.0; 2];
        let a = (0.5, 0.5);
        assert!(!project_products(band, (&s, &ds), (&z, &dz), a, &mut out));
        assert_eq!(out, vec![0.0, 0.0]);

        // Drive component 0 far out of the band along the trial step.
        let ds = vec![100.0, 0.0];
        assert!(project_products(band, (&s, &ds), (&z, &dz), a, &mut out));
        assert!(out[0] > 0.0);
        assert_eq!(out[1], 0.0);
    }

    /// The split pair is what the direct driver needs; passing it twice must
    /// reproduce the symmetric case the HSDE loop uses.
    #[test]
    fn split_steps_reduce_to_the_symmetric_case() {
        let band = Band::around(1e-3);
        let s = vec![2.0, 0.5, 1e-6];
        let z = vec![1e-3, 4.0, 7.0];
        let ds = vec![-1.0, 0.25, 3e-7];
        let dz = vec![2e-4, -1.5, -2.0];
        let (mut a, mut b) = (vec![0.0; 3], vec![0.0; 3]);
        let sym = project_products(band, (&s, &ds), (&z, &dz), (0.3, 0.3), &mut a);
        let split = project_products(band, (&s, &ds), (&z, &dz), (0.3, 0.3), &mut b);
        assert_eq!(sym, split);
        assert_eq!(a, b);
    }
}
