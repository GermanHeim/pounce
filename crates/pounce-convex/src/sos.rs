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

/// A constrained polynomial program `min p(x) s.t. gᵢ(x) ≥ 0, hⱼ(x) = 0`.
#[derive(Debug, Clone)]
pub struct PolyProblem {
    pub n_vars: usize,
    pub objective: Polynomial,
    /// Inequality constraints `gᵢ(x) ≥ 0`.
    pub inequalities: Vec<Polynomial>,
    /// Equality constraints `hⱼ(x) = 0`.
    pub equalities: Vec<Polynomial>,
}

impl PolyProblem {
    pub fn new(objective: Polynomial) -> Self {
        let n_vars = objective.n_vars;
        PolyProblem {
            n_vars,
            objective,
            inequalities: Vec::new(),
            equalities: Vec::new(),
        }
    }

    /// Add an inequality `g(x) ≥ 0`.
    pub fn ge(mut self, g: Polynomial) -> Self {
        self.inequalities.push(g);
        self
    }

    /// Add an equality `h(x) = 0`.
    pub fn eq(mut self, h: Polynomial) -> Self {
        self.equalities.push(h);
        self
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
    sos_constrained_lower_bound_opts(&PolyProblem::new(p.clone()), None, opts, make_backend)
}

/// SOS / Lasserre lower bound for a **constrained** polynomial program
/// `min p s.t. gᵢ ≥ 0, hⱼ = 0` at relaxation order `order` (defaults to the
/// minimum admissible). Uses Putinar's representation
///
/// ```text
///   p(x) − γ = σ₀(x) + Σᵢ σᵢ(x) gᵢ(x) + Σⱼ λⱼ(x) hⱼ(x),
/// ```
///
/// with `σ₀, σᵢ` SOS (PSD Gram blocks; the *localizing* multipliers `σᵢ`
/// use the smaller basis of degree `d − ⌈deg gᵢ/2⌉`) and `λⱼ` free
/// polynomials. The returned `γ*` is a certified lower bound on `min p` over
/// the feasible set; raising `order` tightens it (the Lasserre hierarchy).
pub fn sos_constrained_lower_bound<F>(
    prob: &PolyProblem,
    order: Option<usize>,
    make_backend: F,
) -> SosBound
where
    F: FnMut() -> Box<dyn SparseSymLinearSolverInterface>,
{
    sos_constrained_lower_bound_opts(prob, order, &QpOptions::default(), make_backend)
}

/// [`sos_constrained_lower_bound`] with explicit solver options.
pub fn sos_constrained_lower_bound_opts<F>(
    prob: &PolyProblem,
    order: Option<usize>,
    opts: &QpOptions,
    make_backend: F,
) -> SosBound
where
    F: FnMut() -> Box<dyn SparseSymLinearSolverInterface>,
{
    let n = prob.n_vars;
    let r2 = std::f64::consts::SQRT_2;

    // Minimum relaxation order, then honor a user-requested (larger) order.
    let mut d_min = prob.objective.degree().div_ceil(2);
    for g in &prob.inequalities {
        d_min = d_min.max(g.degree().div_ceil(2));
    }
    for h in &prob.equalities {
        d_min = d_min.max(h.degree().div_ceil(2));
    }
    let d = order.map_or(d_min, |o| o.max(d_min));

    // Column layout: x = (γ, svec(Q₀), svec(Q₁)…, free λ coefficients…).
    let mut col = 1usize;
    let mut cones: Vec<ConeSpec> = Vec::new();
    let mut g_rows: Vec<Triplet> = Vec::new();
    let mut g_h: Vec<f64> = Vec::new();
    let mut by_mono: HashMap<Vec<usize>, Vec<(usize, f64)>> = HashMap::new();
    let unit = [(vec![0usize; n], 1.0)]; // weight ≡ 1 for σ₀

    // PSD (SOS) blocks: σ₀ (weight 1, basis degree d), then one localizing
    // multiplier per inequality (weight gᵢ, basis degree d − ⌈deg gᵢ/2⌉).
    let psd_specs = std::iter::once((d, &unit[..])).chain(
        prob.inequalities
            .iter()
            .map(|g| (d - g.degree().div_ceil(2), &g.terms[..])),
    );
    for (deg, weight) in psd_specs {
        let basis = monomials(n, deg);
        let bn = basis.len();
        let col_base = col;
        // Cone rows s = svec(Qₖ) ⪰ 0, and the Gram contributions to each
        // product monomial (× the weight polynomial's terms).
        for i in 0..bn {
            for j in 0..=i {
                let coef0 = if i == j { 1.0 } else { r2 };
                let qcol = col_base + svec_index(bn, i, j);
                let base: Vec<usize> = basis[i].iter().zip(&basis[j]).map(|(a, b)| a + b).collect();
                for (delta, wc) in weight {
                    let alpha: Vec<usize> = base.iter().zip(delta).map(|(a, dd)| a + dd).collect();
                    by_mono.entry(alpha).or_default().push((qcol, coef0 * wc));
                }
            }
        }
        let sd = bn * (bn + 1) / 2;
        for k in 0..sd {
            let r = g_h.len();
            g_rows.push(Triplet::new(r, col_base + k, -1.0));
            g_h.push(0.0);
        }
        cones.push(ConeSpec::Psd(bn));
        col += sd;
    }

    // Free multipliers λⱼ for equalities: a free coefficient per monomial of
    // degree ≤ 2d − deg(hⱼ), contributing (× hⱼ's terms) with no cone.
    for h in &prob.equalities {
        let basis = monomials(n, 2 * d - h.degree());
        for nu in &basis {
            let lcol = col;
            col += 1;
            for (delta, hc) in &h.terms {
                let alpha: Vec<usize> = nu.iter().zip(delta).map(|(a, dd)| a + dd).collect();
                by_mono.entry(alpha).or_default().push((lcol, *hc));
            }
        }
    }
    let n_x = col;

    // One coefficient-matching equality per distinct monomial: the SOS/Putinar
    // certificate's coefficient must equal p's, with the constant carrying −γ.
    let pc = prob.objective.coeff_map();
    let zero_exp = vec![0usize; n];
    let mut a: Vec<Triplet> = Vec::new();
    let mut b: Vec<f64> = Vec::new();
    for (alpha, terms) in &by_mono {
        let row = b.len();
        for &(c, coef) in terms {
            a.push(Triplet::new(row, c, coef));
        }
        if *alpha == zero_exp {
            a.push(Triplet::new(row, 0, 1.0)); // + γ
        }
        b.push(pc.get(alpha).copied().unwrap_or(0.0));
    }

    // Objective: maximize γ  ⇔  minimize −γ.
    let mut c = vec![0.0; n_x];
    c[0] = -1.0;

    let qp = QpProblem {
        n: n_x,
        p_lower: Vec::new(),
        c,
        a,
        b,
        g: g_rows,
        h: g_h,
        lb: Vec::new(),
        ub: Vec::new(),
    };
    let sol = solve_socp_ipm(&qp, &cones, opts, make_backend);
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

    #[test]
    fn constrained_linear_lower_bound() {
        // min x s.t. x − 1 ≥ 0  ⇒  min = 1 (the constraint binds).
        let prob = PolyProblem::new(Polynomial::new(1, vec![(vec![1], 1.0)]))
            .ge(Polynomial::new(1, vec![(vec![1], 1.0), (vec![0], -1.0)]));
        let r = sos_constrained_lower_bound(&prob, None, backend);
        assert_eq!(r.status, QpStatus::Optimal, "{:?}", r.status);
        assert!(
            (r.lower_bound - 1.0).abs() < 1e-5,
            "bound = {}",
            r.lower_bound
        );
    }

    #[test]
    fn constrained_nonconvex_box() {
        // min −x s.t. 1 − x² ≥ 0  (x ∈ [−1,1])  ⇒  min = −1 at x = 1.
        // The localizing multiplier σ₁ (a nonneg scalar) makes the bound
        // exact — a nonconvex feasible-set bound from the SDP.
        let prob = PolyProblem::new(Polynomial::new(1, vec![(vec![1], -1.0)]))
            .ge(Polynomial::new(1, vec![(vec![0], 1.0), (vec![2], -1.0)]));
        let r = sos_constrained_lower_bound(&prob, None, backend);
        assert_eq!(r.status, QpStatus::Optimal, "{:?}", r.status);
        assert!(
            (r.lower_bound + 1.0).abs() < 1e-5,
            "bound = {}",
            r.lower_bound
        );
    }

    #[test]
    fn constrained_equality_lower_bound() {
        // min x² + y² s.t. x + y − 2 = 0  ⇒  min = 2 at (1,1), via a free
        // multiplier λ(x,y) for the equality.
        let obj = Polynomial::new(2, vec![(vec![2, 0], 1.0), (vec![0, 2], 1.0)]);
        let prob = PolyProblem::new(obj).eq(Polynomial::new(
            2,
            vec![(vec![1, 0], 1.0), (vec![0, 1], 1.0), (vec![0, 0], -2.0)],
        ));
        let r = sos_constrained_lower_bound(&prob, None, backend);
        assert_eq!(r.status, QpStatus::Optimal, "{:?}", r.status);
        assert!(
            (r.lower_bound - 2.0).abs() < 1e-5,
            "bound = {}",
            r.lower_bound
        );
    }
}
