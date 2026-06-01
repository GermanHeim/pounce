//! Positive-semidefinite (PSD) cone primitives — Phase H7 foundation.
//!
//! The PSD cone `Sⁿ₊ = { X = Xᵀ ∈ ℝⁿˣⁿ : X ⪰ 0 }` is a **self-scaled**
//! (symmetric) cone, like the orthant and the second-order cone, so it
//! carries a Nesterov–Todd scaling. This module supplies the building
//! blocks the conic IPM needs, all in the symmetric-vectorization (`svec`)
//! coordinates the solver's slack/dual vectors live in:
//!
//! - [`svec`] / [`smat`] — the isometry between a symmetric `n×n` matrix and
//!   `ℝᵐ`, `m = n(n+1)/2`, with off-diagonals scaled by `√2` so that
//!   `⟨X, Y⟩_F = svec(X)·svec(Y)`.
//! - The log-det barrier `F(X) = −log det X`, its gradient `−X⁻¹`, and the
//!   Hessian action `D ↦ X⁻¹ D X⁻¹`.
//! - Membership / fraction-to-boundary via the eigenvalues of `X`.
//! - The **Nesterov–Todd scaling** `W` (symmetric PD, `W Z W = S`), the
//!   matrix the dense `(z,z)` KKT block `W ⊗ₛ W` is built from (driver
//!   integration is Phase H7's next step).
//!
//! Eigendecompositions reuse [`pounce_linalg::symmetric_eigen`] (the
//! cyclic-Jacobi solver shared with the NLP sensitivity path).

use pounce_linalg::symmetric_eigen;

/// The PSD cone over symmetric `n×n` matrices. Its slack/dual vectors have
/// dimension `n(n+1)/2` in [`svec`] coordinates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PsdCone {
    pub n: usize,
}

impl PsdCone {
    pub fn new(n: usize) -> Self {
        PsdCone { n }
    }

    /// Length of the `svec` vectors this cone owns, `n(n+1)/2`.
    pub fn dim(&self) -> usize {
        self.n * (self.n + 1) / 2
    }

    /// Barrier degree `ν` of `−log det` over `Sⁿ₊` — equal to `n`.
    pub fn degree(&self) -> usize {
        self.n
    }
}

/// `svec` ordering: lower triangle, column by column — `(0,0),(1,0),…,
/// (n−1,0),(1,1),(2,1),…`. Off-diagonals carry a `√2` so the map is an
/// isometry (`‖X‖_F = ‖svec(X)‖₂`). `mat` is row-major `n×n` (symmetric).
pub fn svec(mat: &[f64], n: usize, out: &mut [f64]) {
    let r2 = std::f64::consts::SQRT_2;
    let mut k = 0;
    for j in 0..n {
        for i in j..n {
            out[k] = if i == j {
                mat[i * n + i]
            } else {
                r2 * mat[i * n + j]
            };
            k += 1;
        }
    }
}

/// Inverse of [`svec`]: rebuild the symmetric `n×n` matrix (row-major) from
/// its `svec`, dividing off-diagonals by `√2`.
pub fn smat(v: &[f64], n: usize, out: &mut [f64]) {
    let inv_r2 = std::f64::consts::FRAC_1_SQRT_2;
    let mut k = 0;
    for j in 0..n {
        for i in j..n {
            let val = if i == j { v[k] } else { inv_r2 * v[k] };
            out[i * n + j] = val;
            out[j * n + i] = val;
            k += 1;
        }
    }
}

// ---- small dense symmetric-matrix helpers (row-major, modest n) ----

/// `c = a · b` for row-major `n×n` matrices.
fn matmul(a: &[f64], b: &[f64], n: usize, c: &mut [f64]) {
    for i in 0..n {
        for k in 0..n {
            let mut acc = 0.0;
            for j in 0..n {
                acc += a[i * n + j] * b[j * n + k];
            }
            c[i * n + k] = acc;
        }
    }
}

/// Symmetric matrix function `f(A) = Q diag(f(λ)) Qᵀ` for a symmetric `A`
/// (row-major). Returns `None` if the eigensolver fails to converge.
fn sym_apply(a: &[f64], n: usize, f: impl Fn(f64) -> f64) -> Option<Vec<f64>> {
    let mut vals = vec![0.0; n];
    let mut vecs = vec![0.0; n * n];
    if !symmetric_eigen(a, n, &mut vals, &mut vecs) {
        return None;
    }
    // vecs is column-major: eigenvector j has component i at vecs[j*n + i].
    let mut out = vec![0.0; n * n];
    for i in 0..n {
        for k in 0..n {
            let mut acc = 0.0;
            for j in 0..n {
                acc += f(vals[j]) * vecs[j * n + i] * vecs[j * n + k];
            }
            out[i * n + k] = acc;
        }
    }
    Some(out)
}

impl PsdCone {
    /// The cone identity `e = svec(Iₙ)` — the well-centered cold-start point.
    pub fn identity(&self, out: &mut [f64]) {
        let n = self.n;
        let mut k = 0;
        for j in 0..n {
            for i in j..n {
                out[k] = if i == j { 1.0 } else { 0.0 };
                k += 1;
            }
        }
    }

    /// Smallest eigenvalue of `smat(point)` — `> 0` iff strictly interior.
    pub fn min_eig(&self, point: &[f64]) -> f64 {
        let n = self.n;
        let mut m = vec![0.0; n * n];
        smat(point, n, &mut m);
        let mut vals = vec![0.0; n];
        let mut vecs = vec![0.0; n * n];
        if !symmetric_eigen(&m, n, &mut vals, &mut vecs) {
            return f64::NEG_INFINITY;
        }
        vals[0] // ascending
    }

    /// Whether `smat(point) ⪰ tol·I`.
    pub fn in_cone(&self, point: &[f64], tol: f64) -> bool {
        self.min_eig(point) > tol
    }

    /// The log-det barrier `F = −log det smat(point)` (`+∞` outside the cone).
    pub fn barrier(&self, point: &[f64]) -> f64 {
        let n = self.n;
        let mut m = vec![0.0; n * n];
        smat(point, n, &mut m);
        let mut vals = vec![0.0; n];
        let mut vecs = vec![0.0; n * n];
        if !symmetric_eigen(&m, n, &mut vals, &mut vecs) {
            return f64::INFINITY;
        }
        let mut acc = 0.0;
        for &l in &vals {
            if l <= 0.0 {
                return f64::INFINITY;
            }
            acc += l.ln();
        }
        -acc
    }

    /// Gradient of the barrier, `∇F = −svec(X⁻¹)` (`X = smat(point)`).
    // The eig of a correctly-sized symmetric matrix at a strictly-interior
    // (PD) point always converges, so `sym_apply` cannot return `None` here.
    #[allow(clippy::expect_used)]
    pub fn barrier_grad(&self, point: &[f64], out: &mut [f64]) {
        let n = self.n;
        let mut m = vec![0.0; n * n];
        smat(point, n, &mut m);
        let inv = sym_apply(&m, n, |l| 1.0 / l).expect("barrier_grad: eig failed");
        // out = −svec(X⁻¹).
        svec(&inv, n, out);
        for v in out.iter_mut() {
            *v = -*v;
        }
    }

    /// Hessian action `H[d] = svec(X⁻¹ · smat(d) · X⁻¹)` — the operator
    /// `∇²F(point)` applied to a direction `d` (both in `svec` coordinates).
    // See `barrier_grad`: the interior-point eig always converges.
    #[allow(clippy::expect_used)]
    pub fn barrier_hess_apply(&self, point: &[f64], dir: &[f64], out: &mut [f64]) {
        let n = self.n;
        let mut x = vec![0.0; n * n];
        smat(point, n, &mut x);
        let xinv = sym_apply(&x, n, |l| 1.0 / l).expect("hess: eig failed");
        let mut d = vec![0.0; n * n];
        smat(dir, n, &mut d);
        let mut tmp = vec![0.0; n * n];
        let mut res = vec![0.0; n * n];
        matmul(&xinv, &d, n, &mut tmp); // X⁻¹ D
        matmul(&tmp, &xinv, n, &mut res); // X⁻¹ D X⁻¹
        svec(&res, n, out);
    }

    /// Largest `α ∈ (0, tau]` with `smat(v) + α·smat(dv) ⪰ 0`, scaled by the
    /// fraction-to-boundary parameter `tau`. Computes the most-negative
    /// eigenvalue of `L⁻¹ smat(dv) L⁻ᵀ` where `smat(v) = L Lᵀ` (here via the
    /// symmetric form `V^{-1/2} smat(dv) V^{-1/2}`, `V = smat(v) ≻ 0`).
    pub fn max_step(&self, v: &[f64], dv: &[f64], tau: f64) -> f64 {
        let n = self.n;
        let mut vmat = vec![0.0; n * n];
        smat(v, n, &mut vmat);
        let vinv_half = match sym_apply(&vmat, n, |l| 1.0 / l.max(1e-300).sqrt()) {
            Some(m) => m,
            None => return tau, // can't scale; let the caller's safeguard handle it
        };
        let mut dmat = vec![0.0; n * n];
        smat(dv, n, &mut dmat);
        // M = V^{-1/2} dV V^{-1/2}  (symmetric).
        let mut tmp = vec![0.0; n * n];
        let mut mmat = vec![0.0; n * n];
        matmul(&vinv_half, &dmat, n, &mut tmp);
        matmul(&tmp, &vinv_half, n, &mut mmat);
        let mut vals = vec![0.0; n];
        let mut vecs = vec![0.0; n * n];
        if !symmetric_eigen(&mmat, n, &mut vals, &mut vecs) {
            return tau;
        }
        let min_eig = vals[0]; // ascending
        if min_eig >= 0.0 {
            1.0 // direction keeps PSD for all α ⇒ full step
        } else {
            (tau * (-1.0 / min_eig)).min(1.0)
        }
    }

    /// The Nesterov–Todd scaling matrix `W` (symmetric PD) for the
    /// primal/dual interior pair `(s, z)` (both `svec` of PD matrices):
    /// `W = S^{1/2} (S^{1/2} Z S^{1/2})^{-1/2} S^{1/2}`, which satisfies the
    /// defining identity `W Z W = S`. Returned as a row-major `n×n` matrix.
    /// The dense `(z,z)` KKT scaling block is the symmetric Kronecker
    /// product `W ⊗ₛ W` built from this (Phase H7 driver integration).
    pub fn nt_scaling(&self, s: &[f64], z: &[f64]) -> Option<Vec<f64>> {
        let n = self.n;
        let mut smat_s = vec![0.0; n * n];
        let mut smat_z = vec![0.0; n * n];
        smat(s, n, &mut smat_s);
        smat(z, n, &mut smat_z);
        let s_half = sym_apply(&smat_s, n, |l| l.max(0.0).sqrt())?;
        // M = S^{1/2} Z S^{1/2}.
        let mut tmp = vec![0.0; n * n];
        let mut m = vec![0.0; n * n];
        matmul(&s_half, &smat_z, n, &mut tmp);
        matmul(&tmp, &s_half, n, &mut m);
        let m_inv_half = sym_apply(&m, n, |l| 1.0 / l.max(1e-300).sqrt())?;
        // W = S^{1/2} M^{-1/2} S^{1/2}.
        let mut tmp2 = vec![0.0; n * n];
        let mut w = vec![0.0; n * n];
        matmul(&s_half, &m_inv_half, n, &mut tmp2);
        matmul(&tmp2, &s_half, n, &mut w);
        Some(w)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn matmul_v(a: &[f64], b: &[f64], n: usize) -> Vec<f64> {
        let mut c = vec![0.0; n * n];
        matmul(a, b, n, &mut c);
        c
    }

    #[test]
    fn svec_smat_roundtrip_and_isometry() {
        let n = 3;
        // A symmetric matrix (row-major).
        let x = vec![
            2.0, 0.5, -1.0, //
            0.5, 3.0, 0.25, //
            -1.0, 0.25, 1.5,
        ];
        let m = n * (n + 1) / 2;
        let mut v = vec![0.0; m];
        svec(&x, n, &mut v);
        let mut back = vec![0.0; n * n];
        smat(&v, n, &mut back);
        for i in 0..n * n {
            assert!((x[i] - back[i]).abs() < 1e-12, "roundtrip at {i}");
        }
        // Isometry: ⟨X,X⟩_F = ‖svec‖².
        let fro: f64 = x.iter().map(|a| a * a).sum();
        let sv: f64 = v.iter().map(|a| a * a).sum();
        assert!((fro - sv).abs() < 1e-12, "isometry {fro} vs {sv}");
    }

    #[test]
    fn inner_product_preserved() {
        let n = 2;
        let x = vec![1.0, 2.0, 2.0, 3.0];
        let y = vec![0.5, -1.0, -1.0, 4.0];
        let fro: f64 = (0..n * n).map(|i| x[i] * y[i]).sum();
        let m = n * (n + 1) / 2;
        let (mut xv, mut yv) = (vec![0.0; m], vec![0.0; m]);
        svec(&x, n, &mut xv);
        svec(&y, n, &mut yv);
        let dot: f64 = (0..m).map(|i| xv[i] * yv[i]).sum();
        assert!((fro - dot).abs() < 1e-12, "{fro} vs {dot}");
    }

    #[test]
    fn identity_is_in_cone_and_barrier_zero() {
        let c = PsdCone::new(3);
        let mut e = vec![0.0; c.dim()];
        c.identity(&mut e);
        assert!(c.in_cone(&e, 1e-9));
        assert!((c.barrier(&e) - 0.0).abs() < 1e-12); // −log det I = 0
        assert!((c.min_eig(&e) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn barrier_grad_matches_finite_difference() {
        let c = PsdCone::new(2);
        // X = [[2, 0.3],[0.3, 1.5]] ≻ 0.
        let point = {
            let x = vec![2.0, 0.3, 0.3, 1.5];
            let mut v = vec![0.0; c.dim()];
            svec(&x, 2, &mut v);
            v
        };
        let mut g = vec![0.0; c.dim()];
        c.barrier_grad(&point, &mut g);
        let h = 1e-6;
        for k in 0..c.dim() {
            let mut pp = point.clone();
            let mut pm = point.clone();
            pp[k] += h;
            pm[k] -= h;
            let fd = (c.barrier(&pp) - c.barrier(&pm)) / (2.0 * h);
            assert!((g[k] - fd).abs() < 1e-5, "grad[{k}] {} vs fd {fd}", g[k]);
        }
    }

    #[test]
    fn nt_scaling_satisfies_w_z_w_equals_s() {
        let c = PsdCone::new(3);
        // Two distinct PD matrices in svec coords.
        let to_v = |x: &[f64]| {
            let mut v = vec![0.0; c.dim()];
            svec(x, 3, &mut v);
            v
        };
        let smat_s = vec![
            4.0, 1.0, 0.0, //
            1.0, 3.0, 0.5, //
            0.0, 0.5, 2.0,
        ];
        let smat_z = vec![
            2.0, -0.3, 0.2, //
            -0.3, 1.0, 0.1, //
            0.2, 0.1, 1.5,
        ];
        let s = to_v(&smat_s);
        let z = to_v(&smat_z);
        let w = c.nt_scaling(&s, &z).expect("nt scaling");
        // Check W Z W = S.
        let wz = matmul_v(&w, &smat_z, 3);
        let wzw = matmul_v(&wz, &w, 3);
        for i in 0..9 {
            assert!(
                (wzw[i] - smat_s[i]).abs() < 1e-8,
                "W Z W ≠ S at {i}: {} vs {}",
                wzw[i],
                smat_s[i]
            );
        }
        // W is symmetric.
        for i in 0..3 {
            for j in 0..3 {
                assert!((w[i * 3 + j] - w[j * 3 + i]).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn max_step_lands_on_the_boundary() {
        let c = PsdCone::new(2);
        // v = I; dv = −I ⇒ I − α I ⪰ 0 needs α ≤ 1; with τ=1, step = 1.
        let mut v = vec![0.0; c.dim()];
        c.identity(&mut v);
        let mut dv = vec![0.0; c.dim()];
        c.identity(&mut dv);
        for x in dv.iter_mut() {
            *x = -*x;
        }
        let a = c.max_step(&v, &dv, 1.0);
        assert!((a - 1.0).abs() < 1e-9, "step {a}");
        // At α just below 1 the point is still PD; with τ = 0.99, step ≈ 0.99.
        let a2 = c.max_step(&v, &dv, 0.99);
        assert!((a2 - 0.99).abs() < 1e-9, "step {a2}");
    }

    #[test]
    fn max_step_full_when_direction_keeps_psd() {
        let c = PsdCone::new(2);
        let mut v = vec![0.0; c.dim()];
        c.identity(&mut v);
        // dv = +I ⇒ stays PD for all α ⇒ capped at 1.
        let mut dv = vec![0.0; c.dim()];
        c.identity(&mut dv);
        assert!((c.max_step(&v, &dv, 0.99) - 1.0).abs() < 1e-9);
    }
}
