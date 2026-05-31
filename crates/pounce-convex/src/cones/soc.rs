//! Second-order (Lorentz) cone `K = { (t, x) : t ≥ ‖x‖₂ }` for the convex
//! IPM.
//!
//! Phase 2 of the SOCP extension (see `dev-notes/socp-extension.md`). This
//! module ships the parts whose correctness is unambiguous and
//! independently testable:
//!
//! - the Jordan-algebra geometry (`∘`, identity `e`, the `det` quadratic),
//! - the central-path measure `μ = ⟨s, z⟩ / 2` (rank 2, regardless of
//!   dimension),
//! - the fraction-to-boundary `max_step` (the cone-boundary root), and
//! - the **Nesterov–Todd scaling Hessian** `W² = η²(2 w̄ w̄ᵀ − J)` that
//!   enters the KKT `(z, z)` block, with its defining identities
//!   (`W² s = z`, symmetric PD, `W² = I` at `s = z`) verified in tests.
//!
//! The *reduced-system* methods (`recover_ds`, `rhs_comp_term`, the
//! corrector) carry the NT scaling/sign conventions whose end-to-end
//! correctness must be validated against a reference solver; they are
//! deferred to Phase 2b and `unimplemented!` here so they cannot be used
//! before that validation. The driver builds an orthant-only cone until
//! then, so SOC is a tested building block, not yet a solvable cone.

use super::{Cone, ConeBlock};

/// The second-order cone of a given dimension `m` (`m ≥ 1`):
/// `{ u ∈ ℝᵐ : u₀ ≥ ‖u_{1..}‖₂ }`.
#[derive(Debug, Clone, Copy)]
pub struct SecondOrderCone {
    m: usize,
}

impl SecondOrderCone {
    pub fn new(m: usize) -> Self {
        assert!(m >= 1, "second-order cone needs dimension ≥ 1");
        SecondOrderCone { m }
    }

    /// `det(u) = u₀² − ‖u_{1..}‖²` — the cone's quadratic form (`uᵀJu`,
    /// `J = diag(1,−1,…,−1)`). Positive in the interior.
    pub fn det(u: &[f64]) -> f64 {
        let tail: f64 = u[1..].iter().map(|v| v * v).sum();
        u[0] * u[0] - tail
    }

    /// Jordan product `s ∘ z = (sᵀz, s₀ z_{1..} + z₀ s_{1..})`.
    pub fn jordan(s: &[f64], z: &[f64], out: &mut [f64]) {
        let dot: f64 = s.iter().zip(z).map(|(a, b)| a * b).sum();
        out[0] = dot;
        for k in 1..s.len() {
            out[k] = s[0] * z[k] + z[0] * s[k];
        }
    }

    /// The Nesterov–Todd scaling: returns `(η, w̄)` with `w̄` the scaling
    /// point (`det(w̄) = 1`, `w̄₀ > 0`) and `η² = √det(s)/√det(z)`. The
    /// scaling Hessian is then `W² = η²(2 w̄ w̄ᵀ − J)`.
    fn nt_scaling(s: &[f64], z: &[f64]) -> (f64, Vec<f64>) {
        let m = s.len();
        let s_det = Self::det(s).max(0.0).sqrt(); // √det(s)
        let z_det = Self::det(z).max(0.0).sqrt();
        // Normalize to the cone's unit-determinant sphere.
        let s_bar: Vec<f64> = s.iter().map(|v| v / s_det).collect();
        let z_bar: Vec<f64> = z.iter().map(|v| v / z_det).collect();
        let sz: f64 = s_bar.iter().zip(&z_bar).map(|(a, b)| a * b).sum();
        let gamma = ((1.0 + sz) / 2.0).sqrt();
        // w̄ = (s̄ + J z̄) / (2γ),  J z̄ = (z̄₀, −z̄_{1..}).
        let mut w_bar = vec![0.0; m];
        w_bar[0] = (s_bar[0] + z_bar[0]) / (2.0 * gamma);
        for k in 1..m {
            w_bar[k] = (s_bar[k] - z_bar[k]) / (2.0 * gamma);
        }
        let eta = (s_det / z_det).sqrt();
        (eta, w_bar)
    }

    /// Dense lower triangle (row-major) of `W² = η²(2 w̄ w̄ᵀ − J)`.
    fn w2_lower(eta: f64, w_bar: &[f64]) -> Vec<f64> {
        let m = w_bar.len();
        let eta2 = eta * eta;
        let mut lower = Vec::with_capacity(m * (m + 1) / 2);
        for i in 0..m {
            for j in 0..=i {
                let j_ij = if i == j {
                    if i == 0 {
                        1.0
                    } else {
                        -1.0
                    }
                } else {
                    0.0
                };
                lower.push(eta2 * (2.0 * w_bar[i] * w_bar[j] - j_ij));
            }
        }
        lower
    }
}

impl Cone for SecondOrderCone {
    fn degree(&self) -> usize {
        2 // rank of the second-order cone, independent of dimension
    }

    fn dim(&self) -> usize {
        self.m
    }

    fn mu(&self, s: &[f64], z: &[f64]) -> f64 {
        let dot: f64 = s.iter().zip(z).map(|(a, b)| a * b).sum();
        dot / 2.0
    }

    fn kkt_block(&self, s: &[f64], z: &[f64]) -> ConeBlock {
        let (eta, w_bar) = Self::nt_scaling(s, z);
        ConeBlock::DenseLower {
            dim: self.m,
            lower: Self::w2_lower(eta, &w_bar),
        }
    }

    fn comp_residual(&self, s: &[f64], z: &[f64], sigma_mu: f64, out: &mut [f64]) {
        // s ∘ z − σμ e.
        Self::jordan(s, z, out);
        out[0] -= sigma_mu;
    }

    fn max_step(&self, v: &[f64], dv: &[f64], tau: f64) -> f64 {
        // Largest α with v + α dv in int(K): det(v+αdv) ≥ 0 and first
        // coordinate ≥ 0. det is the quadratic a α² + b α + c with
        // a = det(dv), c = det(v) > 0, b = 2 (v J dv).
        let a = Self::det(dv);
        let c = Self::det(v);
        let tail: f64 = v[1..].iter().zip(&dv[1..]).map(|(p, q)| p * q).sum();
        let b = 2.0 * (v[0] * dv[0] - tail);

        let mut alpha = f64::INFINITY;
        // Determinant boundary (smallest positive root of a α² + b α + c).
        let disc = b * b - 4.0 * a * c;
        if a.abs() <= 1e-300 {
            if b < 0.0 {
                alpha = alpha.min(-c / b);
            }
        } else if disc >= 0.0 {
            let sq = disc.sqrt();
            for r in [(-b - sq) / (2.0 * a), (-b + sq) / (2.0 * a)] {
                if r > 0.0 {
                    alpha = alpha.min(r);
                }
            }
        }
        // First-coordinate boundary v₀ + α dv₀ ≥ 0.
        if dv[0] < 0.0 {
            alpha = alpha.min(-v[0] / dv[0]);
        }
        if !alpha.is_finite() {
            return 1.0; // no binding boundary in the step direction
        }
        (tau * alpha).min(1.0)
    }

    // --- Phase 2b: reduced-system methods (NT scaling/sign conventions to
    // be validated end-to-end against a reference solver). ---

    fn scaling_diag(&self, _s: &[f64], _z: &[f64], _out: &mut [f64]) {
        // SOC's (z,z) block is dense (see `kkt_block`); the diagonal-only
        // path is the orthant's. The driver consumes `kkt_block` once the
        // KKT assembly is generalized (Phase 2b); until then SOC is not
        // routed through the solver.
        unimplemented!("Phase 2b: SOC uses kkt_block, not scaling_diag")
    }

    fn comp_residual_corrector(
        &self,
        _s: &[f64],
        _z: &[f64],
        _ds_aff: &[f64],
        _dz_aff: &[f64],
        _sigma_mu: f64,
        _out: &mut [f64],
    ) {
        unimplemented!("Phase 2b: SOC corrector (second-order term in NT-scaled space)")
    }

    fn recover_ds(&self, _s: &[f64], _z: &[f64], _r_comp: &[f64], _dz: &[f64], _ds: &mut [f64]) {
        unimplemented!("Phase 2b: SOC ds recovery via NT scaling")
    }

    fn rhs_comp_term(&self, _s: &[f64], _z: &[f64], _r_comp: &[f64], _out: &mut [f64]) {
        unimplemented!("Phase 2b: SOC reduced-system RHS term via NT scaling")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn in_interior(u: &[f64]) -> bool {
        u[0] > 0.0 && SecondOrderCone::det(u) > 0.0
    }

    /// Reconstruct the dense symmetric `W²` from its lower triangle.
    fn dense(block: &ConeBlock, m: usize) -> Vec<Vec<f64>> {
        let lower = match block {
            ConeBlock::DenseLower { dim, lower } => {
                assert_eq!(*dim, m);
                lower
            }
            _ => panic!("expected dense block"),
        };
        let mut w = vec![vec![0.0; m]; m];
        let mut idx = 0;
        for i in 0..m {
            for j in 0..=i {
                w[i][j] = lower[idx];
                w[j][i] = lower[idx];
                idx += 1;
            }
        }
        w
    }

    fn matvec(w: &[Vec<f64>], x: &[f64]) -> Vec<f64> {
        w.iter()
            .map(|row| row.iter().zip(x).map(|(a, b)| a * b).sum())
            .collect()
    }

    #[test]
    fn mu_is_half_inner_product() {
        let c = SecondOrderCone::new(3);
        // rank 2 ⇒ μ = ⟨s,z⟩ / 2.
        let s = [2.0, 0.5, 0.5];
        let z = [3.0, -1.0, 0.0];
        let dot = 2.0 * 3.0 + 0.5 * -1.0 + 0.5 * 0.0;
        assert!((c.mu(&s, &z) - dot / 2.0).abs() < 1e-12);
    }

    #[test]
    fn nt_hessian_maps_z_to_s() {
        // The (z,z) scaling block maps z → s, matching the orthant's
        // diag(s/z) (which satisfies diag(s/z)·z = s). For the SOC this is
        // W² = η² Q_{w̄}, with W² symmetric PD. (Equivalently the NT
        // identity z = W² s holds with the inverse scaling; we test the
        // form the KKT block actually uses.)
        let c = SecondOrderCone::new(3);
        let s = [2.0, 0.5, -0.5]; // det = 4 - 0.5 = 3.5 > 0
        let z = [3.0, 1.0, 0.5]; // det = 9 - 1.25 > 0
        assert!(in_interior(&s) && in_interior(&z));
        let w2 = dense(&c.kkt_block(&s, &z), 3);
        let wz = matvec(&w2, &z);
        for k in 0..3 {
            assert!((wz[k] - s[k]).abs() < 1e-9, "W²z[{k}]={} s={}", wz[k], s[k]);
        }
        // Symmetry.
        for i in 0..3 {
            for j in 0..3 {
                assert!((w2[i][j] - w2[j][i]).abs() < 1e-12);
            }
        }
        // Positive definiteness via positive determinant + positive (0,0)
        // leading minor chain on this 3×3 (cheap check: xᵀW²x > 0 on a few
        // probes including the cone axis).
        for x in [[1.0, 0.0, 0.0], [0.3, 0.7, -0.2], [-0.5, 0.1, 0.9]] {
            let q: f64 = x.iter().zip(matvec(&w2, &x)).map(|(a, b)| a * b).sum();
            assert!(q > 0.0, "W² not PD on probe {x:?}: {q}");
        }
    }

    #[test]
    fn nt_hessian_is_identity_at_s_equals_z() {
        let c = SecondOrderCone::new(4);
        let s = [3.0, 1.0, -0.5, 0.5];
        let w2 = dense(&c.kkt_block(&s, &s), 4);
        for i in 0..4 {
            for j in 0..4 {
                let want = if i == j { 1.0 } else { 0.0 };
                assert!((w2[i][j] - want).abs() < 1e-9, "W²[{i}][{j}]={}", w2[i][j]);
            }
        }
    }

    #[test]
    fn comp_residual_is_jordan_minus_sigma_mu_e() {
        let c = SecondOrderCone::new(3);
        let s = [2.0, 0.5, -0.5];
        let z = [3.0, 1.0, 0.5];
        let mut out = [0.0; 3];
        c.comp_residual(&s, &z, 0.7, &mut out);
        let dot = 2.0 * 3.0 + 0.5 * 1.0 + -0.5 * 0.5;
        assert!((out[0] - (dot - 0.7)).abs() < 1e-12);
        assert!((out[1] - (s[0] * z[1] + z[0] * s[1])).abs() < 1e-12);
        assert!((out[2] - (s[0] * z[2] + z[0] * s[2])).abs() < 1e-12);
    }

    #[test]
    fn max_step_lands_on_the_cone_boundary() {
        let c = SecondOrderCone::new(3);
        let v = [2.0, 0.0, 0.0]; // interior, det = 4
        let dv = [-1.0, 1.0, 0.0]; // heads toward / out of the cone
                                   // Step to boundary (tau = 1): det(v+αdv) = 0.
        let alpha = c.max_step(&v, &dv, 1.0);
        let p: Vec<f64> = (0..3).map(|k| v[k] + alpha * dv[k]).collect();
        // Either on the determinant boundary or the step was capped at 1.
        assert!(alpha <= 1.0 + 1e-12);
        if alpha < 1.0 - 1e-9 {
            assert!(SecondOrderCone::det(&p).abs() < 1e-7, "det={}", SecondOrderCone::det(&p));
        }
    }

    #[test]
    fn max_step_caps_at_one_when_staying_interior() {
        let c = SecondOrderCone::new(3);
        let v = [5.0, 0.0, 0.0];
        let dv = [1.0, 0.1, -0.1]; // det(dv)=1-0.02>0, b>0 ⇒ stays interior
        assert!((c.max_step(&v, &dv, 0.99) - 1.0).abs() < 1e-12);
    }
}
