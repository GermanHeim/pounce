//! Sum-of-squares (SOS) **global lower bounds** for polynomial minimization
//! — the first step of polynomial global optimization on the SDP solver.
//!
//! For a polynomial `p(x)`, the SOS relaxation of `min_x p(x)` is
//!
//! ```text
//!   max γ   s.t.   p(x) − γ  is a sum of squares,
//! ```
//!
//! and `p(x) − γ` is SOS iff there is a PSD Gram matrix `Q ⪰ 0` with
//! `p(x) − γ = z(x)ᵀ Q z(x)`, where `z(x)` is the vector of monomials up to
//! degree `d = ⌈deg p / 2⌉`. Matching the coefficient of each monomial `xᵅ`
//! turns this into a semidefinite program:
//!
//! ```text
//!   max γ   s.t.   Σ_{βᵢ+βⱼ = α} Q_{ij} = p_α − γ·[α = 0],   Q ⪰ 0.
//! ```
//!
//! The optimal `γ*` is a **certified global lower bound**: `γ* ≤ min_x p(x)`
//! always, with equality whenever `p − p*` is itself SOS (e.g. univariate
//! polynomials, quadratics, and many low-degree cases — by Hilbert's
//! theorem not *every* nonnegative polynomial is SOS, so in general `γ*` can
//! be a strict lower bound). This is built as a conic program (one
//! [`crate::ConeSpec::Psd`] block plus coefficient-matching equalities) and
//! solved through [`crate::solve_socp_ipm`].

use crate::cones::psd::svec_index;
use crate::ipm::{solve_socp_ipm, QpOptions};
use crate::qp::{QpProblem, QpStatus, Triplet};
use crate::ConeSpec;
use pounce_linsol::SparseSymLinearSolverInterface;
use std::collections::HashMap;

/// A sparse multivariate polynomial over `n_vars` variables: a list of
/// `(exponent vector, coefficient)` terms. The exponent vector has length
/// `n_vars`; e.g. over `(x, y)` the term `3·x²y` is `(vec![2, 1], 3.0)`.
#[derive(Debug, Clone)]
pub struct Polynomial {
    pub n_vars: usize,
    pub terms: Vec<(Vec<usize>, f64)>,
}

impl Polynomial {
    pub fn new(n_vars: usize, terms: Vec<(Vec<usize>, f64)>) -> Self {
        Polynomial { n_vars, terms }
    }

    /// Total degree (the largest term-exponent sum); `0` for a constant.
    pub fn degree(&self) -> usize {
        self.terms
            .iter()
            .map(|(e, _)| e.iter().sum::<usize>())
            .max()
            .unwrap_or(0)
    }

    /// Coefficients keyed by exponent vector (summing any duplicate terms).
    fn coeff_map(&self) -> HashMap<Vec<usize>, f64> {
        let mut m: HashMap<Vec<usize>, f64> = HashMap::new();
        for (e, c) in &self.terms {
            *m.entry(e.clone()).or_insert(0.0) += c;
        }
        m
    }
}

/// Result of the SOS relaxation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SosBound {
    /// The certified global lower bound `γ* ≤ min_x p(x)`.
    pub lower_bound: f64,
    /// Solve status of the underlying SDP.
    pub status: QpStatus,
}

/// All monomial exponent vectors over `n` variables with total degree
/// `≤ max_deg`, in a fixed (recursive) order.
fn monomials(n: usize, max_deg: usize) -> Vec<Vec<usize>> {
    let mut out = Vec::new();
    let mut cur = vec![0usize; n];
    fn rec(pos: usize, remaining: usize, cur: &mut [usize], out: &mut Vec<Vec<usize>>) {
        if pos == cur.len() {
            out.push(cur.to_vec());
            return;
        }
        for e in 0..=remaining {
            cur[pos] = e;
            rec(pos + 1, remaining - e, cur, out);
        }
        cur[pos] = 0;
    }
    rec(0, max_deg, &mut cur, &mut out);
    out
}

/// Build and solve the unconstrained SOS lower-bound SDP for `p`, returning
/// the certified global lower bound. See the module docs for the model.
pub fn sos_lower_bound<F>(p: &Polynomial, mut make_backend: F) -> SosBound
where
    F: FnMut() -> Box<dyn SparseSymLinearSolverInterface>,
{
    sos_lower_bound_opts(p, &QpOptions::default(), &mut make_backend)
}

/// [`sos_lower_bound`] with explicit solver options.
pub fn sos_lower_bound_opts<F>(p: &Polynomial, opts: &QpOptions, make_backend: F) -> SosBound
where
    F: FnMut() -> Box<dyn SparseSymLinearSolverInterface>,
{
    let d = p.degree().div_ceil(2); // half-degree of the SOS basis
    let basis = monomials(p.n_vars, d);
    let big_n = basis.len(); // Gram-matrix size
    let svec_dim = big_n * (big_n + 1) / 2;

    // Decision variables x = (γ, svec(Q)); γ is column 0.
    let n_x = 1 + svec_dim;

    // Group the Gram products by the monomial they contribute to: for each
    // basis pair (i ≥ j), x^{basisᵢ + basisⱼ} gets coefficient (1 if i==j
    // else √2) on the svec entry (i,j) — the √2 matching the svec scaling so
    // that 2·Q_{ij} (the two symmetric off-diagonal terms) is reproduced.
    let r2 = std::f64::consts::SQRT_2;
    let mut by_monomial: HashMap<Vec<usize>, Vec<(usize, f64)>> = HashMap::new();
    for i in 0..big_n {
        for j in 0..=i {
            let alpha: Vec<usize> = basis[i].iter().zip(&basis[j]).map(|(a, b)| a + b).collect();
            let col = 1 + svec_index(big_n, i, j);
            let coef = if i == j { 1.0 } else { r2 };
            by_monomial.entry(alpha).or_default().push((col, coef));
        }
    }

    // One coefficient-matching equality per distinct product monomial.
    let pc = p.coeff_map();
    let zero_exp = vec![0usize; p.n_vars];
    let mut a: Vec<Triplet> = Vec::new();
    let mut b: Vec<f64> = Vec::new();
    for (alpha, terms) in &by_monomial {
        let row = b.len();
        for &(col, coef) in terms {
            a.push(Triplet::new(row, col, coef));
        }
        // p(x) − γ: the constant monomial's coefficient carries the −γ.
        if *alpha == zero_exp {
            a.push(Triplet::new(row, 0, 1.0)); // + γ on the left
        }
        b.push(pc.get(alpha).copied().unwrap_or(0.0));
    }

    // Q ⪰ 0: slack s = svec(Q) via G = −I on the svec columns, h = 0.
    let mut g: Vec<Triplet> = Vec::with_capacity(svec_dim);
    for k in 0..svec_dim {
        g.push(Triplet::new(k, 1 + k, -1.0));
    }
    let h = vec![0.0; svec_dim];

    // Objective: maximize γ  ⇔  minimize −γ.
    let mut c = vec![0.0; n_x];
    c[0] = -1.0;

    let prob = QpProblem {
        n: n_x,
        p_lower: Vec::new(),
        c,
        a,
        b,
        g,
        h,
        lb: Vec::new(),
        ub: Vec::new(),
    };

    let sol = solve_socp_ipm(&prob, &[ConeSpec::Psd(big_n)], opts, make_backend);
    // γ = x₀; the reported objective is −γ.
    SosBound {
        lower_bound: sol.x.first().copied().unwrap_or(f64::NEG_INFINITY),
        status: sol.status,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pounce_feral::FeralSolverInterface;

    fn backend() -> Box<dyn SparseSymLinearSolverInterface> {
        Box::new(FeralSolverInterface::new())
    }

    #[test]
    fn monomial_count_is_binomial() {
        // #monomials over n vars of degree ≤ d is C(n+d, d).
        assert_eq!(monomials(1, 2).len(), 3); // 1, x, x²
        assert_eq!(monomials(2, 1).len(), 3); // 1, x, y
        assert_eq!(monomials(2, 2).len(), 6); // 1,x,y,x²,xy,y²
        assert_eq!(monomials(3, 2).len(), 10);
    }

    #[test]
    fn univariate_quartic_known_minimum() {
        // p(x) = x⁴ − 2x² + 3.  p' = 4x³ − 4x = 0 ⇒ x = 0, ±1; min at ±1 is
        // 1 − 2 + 3 = 2.  p − 2 = (x² − 1)² is SOS, so the bound is exact.
        let p = Polynomial::new(1, vec![(vec![4], 1.0), (vec![2], -2.0), (vec![0], 3.0)]);
        let r = sos_lower_bound(&p, backend);
        assert_eq!(r.status, QpStatus::Optimal, "{:?}", r.status);
        assert!(
            (r.lower_bound - 2.0).abs() < 1e-5,
            "bound = {}",
            r.lower_bound
        );
    }

    #[test]
    fn shifted_paraboloid_two_vars() {
        // p(x,y) = (x−1)² + y² = x² − 2x + 1 + y².  Min 0 at (1, 0); SOS-exact.
        let p = Polynomial::new(
            2,
            vec![
                (vec![2, 0], 1.0),
                (vec![1, 0], -2.0),
                (vec![0, 0], 1.0),
                (vec![0, 2], 1.0),
            ],
        );
        let r = sos_lower_bound(&p, backend);
        assert_eq!(r.status, QpStatus::Optimal, "{:?}", r.status);
        assert!(r.lower_bound.abs() < 1e-5, "bound = {}", r.lower_bound);
    }

    #[test]
    fn constant_polynomial() {
        // p ≡ 7: the global minimum (and SOS bound) is 7.
        let p = Polynomial::new(1, vec![(vec![0], 7.0)]);
        let r = sos_lower_bound(&p, backend);
        assert_eq!(r.status, QpStatus::Optimal);
        assert!(
            (r.lower_bound - 7.0).abs() < 1e-6,
            "bound = {}",
            r.lower_bound
        );
    }

    #[test]
    fn quadratic_lower_bound() {
        // p(x) = x² − 4x + 5 = (x−2)² + 1.  Min 1; basis degree d = 1.
        let p = Polynomial::new(1, vec![(vec![2], 1.0), (vec![1], -4.0), (vec![0], 5.0)]);
        let r = sos_lower_bound(&p, backend);
        assert_eq!(r.status, QpStatus::Optimal);
        assert!(
            (r.lower_bound - 1.0).abs() < 1e-5,
            "bound = {}",
            r.lower_bound
        );
    }
}
