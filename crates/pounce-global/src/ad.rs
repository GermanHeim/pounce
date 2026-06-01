//! Reverse-mode automatic differentiation of an [`FbbtTape`].
//!
//! A forward sweep records every slot's value; a reverse sweep propagates
//! adjoints back to the variable leaves. Used to feed exact first derivatives
//! (objective gradient, constraint Jacobian) to the local NLP solve that
//! produces upper bounds. [`hessian`] supplies the matching exact second
//! derivatives via a point second-order forward sweep, so the local solve runs
//! on a true Newton system rather than a finite-differenced one.

// The second-order forward sweep ([`hessian`]) is matrix math over `0..n`
// indices on parallel arrays; explicit indexing reads clearer than iterators.
#![allow(clippy::needless_range_loop)]

use pounce_nlp::{FbbtOp, FbbtTape};

/// Forward sweep: value at every tape slot (`out[k]` = slot `k`'s value).
pub(crate) fn forward_vals(tape: &FbbtTape, x: &[f64]) -> Vec<f64> {
    let mut v: Vec<f64> = Vec::with_capacity(tape.ops.len());
    for op in &tape.ops {
        let r = match *op {
            FbbtOp::Const(c) => c,
            FbbtOp::Var(i) => x[i],
            FbbtOp::Add(a, b) => v[a] + v[b],
            FbbtOp::Sub(a, b) => v[a] - v[b],
            FbbtOp::Mul(a, b) => v[a] * v[b],
            FbbtOp::Div(a, b) => v[a] / v[b],
            FbbtOp::PowInt(a, n) => v[a].powi(n as i32),
            FbbtOp::Neg(a) => -v[a],
            FbbtOp::Sqrt(a) => v[a].sqrt(),
            FbbtOp::Exp(a) => v[a].exp(),
            FbbtOp::Ln(a) => v[a].ln(),
            FbbtOp::Abs(a) => v[a].abs(),
            FbbtOp::Sin(a) => v[a].sin(),
            FbbtOp::Cos(a) => v[a].cos(),
            FbbtOp::Opaque => f64::NAN,
        };
        v.push(r);
    }
    v
}

/// Accumulate `seed · ∂(tape)/∂x_i` into `grad[i]` for every variable `i` the
/// tape references. `grad` is **not** zeroed (gradients can be chained).
pub(crate) fn accumulate_gradient(tape: &FbbtTape, x: &[f64], seed: f64, grad: &mut [f64]) {
    if tape.ops.is_empty() || seed == 0.0 {
        return;
    }
    let v = forward_vals(tape, x);
    let mut adj = vec![0.0; tape.ops.len()];
    if let Some(last) = adj.last_mut() {
        *last = seed;
    }

    for k in (0..tape.ops.len()).rev() {
        let a = adj[k];
        if a == 0.0 {
            continue;
        }
        match tape.ops[k] {
            FbbtOp::Const(_) => {}
            FbbtOp::Var(i) => grad[i] += a,
            FbbtOp::Add(p, q) => {
                adj[p] += a;
                adj[q] += a;
            }
            FbbtOp::Sub(p, q) => {
                adj[p] += a;
                adj[q] -= a;
            }
            FbbtOp::Mul(p, q) => {
                adj[p] += a * v[q];
                adj[q] += a * v[p];
            }
            FbbtOp::Div(p, q) => {
                adj[p] += a / v[q];
                adj[q] -= a * v[p] / (v[q] * v[q]);
            }
            FbbtOp::PowInt(p, n) => {
                if n >= 1 {
                    adj[p] += a * n as f64 * v[p].powi(n as i32 - 1);
                }
            }
            FbbtOp::Neg(p) => adj[p] -= a,
            FbbtOp::Sqrt(p) => adj[p] += a * 0.5 / v[p].max(1e-300).sqrt(),
            FbbtOp::Exp(p) => adj[p] += a * v[k], // v[k] = exp(v[p])
            FbbtOp::Ln(p) => adj[p] += a / v[p],
            FbbtOp::Abs(p) => adj[p] += a * v[p].signum(),
            FbbtOp::Sin(p) => adj[p] += a * v[p].cos(),
            FbbtOp::Cos(p) => adj[p] -= a * v[p].sin(),
            FbbtOp::Opaque => {}
        }
    }
}

/// Gradient of `tape` at `x` into a fresh length-`n` vector.
pub(crate) fn gradient(tape: &FbbtTape, x: &[f64], n: usize) -> Vec<f64> {
    let mut g = vec![0.0; n];
    accumulate_gradient(tape, x, 1.0, &mut g);
    g
}

/// Per-slot point second-order AD payload: value, gradient, and dense Hessian
/// (`h[i * n + j]`) over the `n` problem variables.
struct PointJet {
    v: f64,
    g: Vec<f64>,
    h: Vec<f64>,
}

impl PointJet {
    /// Chain rule for a unary atom `φ(child)` with `φ' = dphi`, `φ'' = d2phi`
    /// evaluated at the child's value: `g = φ'·c.g`, `H = φ''·c.g⊗c.g + φ'·c.H`.
    fn unary(child: &PointJet, v: f64, dphi: f64, d2phi: f64, n: usize) -> PointJet {
        let mut g = vec![0.0; n];
        let mut h = vec![0.0; n * n];
        for i in 0..n {
            g[i] = dphi * child.g[i];
        }
        for i in 0..n {
            for j in 0..n {
                h[i * n + j] = d2phi * child.g[i] * child.g[j] + dphi * child.h[i * n + j];
            }
        }
        PointJet { v, g, h }
    }
}

/// Product rule for two jets: `(ab)' = a'b + ab'`,
/// `(ab)'' = a''b + a'⊗b' + b'⊗a' + ab''`.
fn mul_point(a: &PointJet, b: &PointJet, n: usize) -> PointJet {
    let mut g = vec![0.0; n];
    let mut h = vec![0.0; n * n];
    for i in 0..n {
        g[i] = a.g[i] * b.v + a.v * b.g[i];
    }
    for i in 0..n {
        for j in 0..n {
            h[i * n + j] =
                a.h[i * n + j] * b.v + a.g[i] * b.g[j] + b.g[i] * a.g[j] + a.v * b.h[i * n + j];
        }
    }
    PointJet { v: a.v * b.v, g, h }
}

/// Exact dense Hessian of `tape` at `x` (`out[i * n + j] = ∂²/∂xᵢ∂xⱼ`), by a
/// point second-order forward sweep over the tape. Returns `None` at a genuine
/// singularity (division by zero, `ln`/`√` of a non-positive value) or for a
/// non-twice-differentiable atom (`|·|`, opaque) — the caller then falls back.
pub(crate) fn hessian(tape: &FbbtTape, x: &[f64], n: usize) -> Option<Vec<f64>> {
    let mut jets: Vec<PointJet> = Vec::with_capacity(tape.ops.len());
    for op in &tape.ops {
        let jet = match *op {
            FbbtOp::Const(c) => PointJet {
                v: c,
                g: vec![0.0; n],
                h: vec![0.0; n * n],
            },
            FbbtOp::Var(k) => {
                let mut g = vec![0.0; n];
                g[k] = 1.0;
                PointJet {
                    v: x[k],
                    g,
                    h: vec![0.0; n * n],
                }
            }
            FbbtOp::Add(a, b) | FbbtOp::Sub(a, b) => {
                let sub = matches!(*op, FbbtOp::Sub(_, _));
                let (ja, jb) = (&jets[a], &jets[b]);
                let s = if sub { -1.0 } else { 1.0 };
                let mut g = vec![0.0; n];
                let mut h = vec![0.0; n * n];
                for i in 0..n {
                    g[i] = ja.g[i] + s * jb.g[i];
                }
                for idx in 0..n * n {
                    h[idx] = ja.h[idx] + s * jb.h[idx];
                }
                PointJet {
                    v: ja.v + s * jb.v,
                    g,
                    h,
                }
            }
            FbbtOp::Neg(a) => {
                let ja = &jets[a];
                PointJet {
                    v: -ja.v,
                    g: ja.g.iter().map(|x| -x).collect(),
                    h: ja.h.iter().map(|x| -x).collect(),
                }
            }
            FbbtOp::Mul(a, b) => mul_point(&jets[a], &jets[b], n),
            FbbtOp::Div(a, b) => {
                let vb = jets[b].v;
                if vb == 0.0 {
                    return None;
                }
                // 1/b with its first two derivatives, then a·(1/b).
                let inv = PointJet::unary(
                    &jets[b],
                    1.0 / vb,
                    -1.0 / (vb * vb),
                    2.0 / (vb * vb * vb),
                    n,
                );
                mul_point(&jets[a], &inv, n)
            }
            FbbtOp::PowInt(a, p) => {
                let c = &jets[a];
                let pf = p as f64;
                let val = c.v.powi(p as i32);
                // Coefficient guards keep the exponent ≥ 0, so no 0·∞.
                let d1 = if p == 0 {
                    0.0
                } else {
                    pf * c.v.powi(p as i32 - 1)
                };
                let d2 = if p <= 1 {
                    0.0
                } else {
                    pf * (pf - 1.0) * c.v.powi(p as i32 - 2)
                };
                PointJet::unary(c, val, d1, d2, n)
            }
            FbbtOp::Exp(a) => {
                let c = &jets[a];
                let e = c.v.exp();
                PointJet::unary(c, e, e, e, n)
            }
            FbbtOp::Ln(a) => {
                let c = &jets[a];
                if c.v <= 0.0 {
                    return None;
                }
                PointJet::unary(c, c.v.ln(), 1.0 / c.v, -1.0 / (c.v * c.v), n)
            }
            FbbtOp::Sqrt(a) => {
                let c = &jets[a];
                if c.v <= 0.0 {
                    return None;
                }
                let s = c.v.sqrt();
                PointJet::unary(c, s, 0.5 / s, -0.25 / (s * c.v), n)
            }
            FbbtOp::Sin(a) => {
                let c = &jets[a];
                PointJet::unary(c, c.v.sin(), c.v.cos(), -c.v.sin(), n)
            }
            FbbtOp::Cos(a) => {
                let c = &jets[a];
                PointJet::unary(c, c.v.cos(), -c.v.sin(), -c.v.cos(), n)
            }
            FbbtOp::Abs(_) | FbbtOp::Opaque => return None,
        };
        jets.push(jet);
    }
    let root = jets.last()?;
    if root.h.iter().all(|e| e.is_finite()) {
        Some(root.h.clone())
    } else {
        None
    }
}

/// Variables (0-based) that `tape` references, ascending and deduplicated.
pub(crate) fn referenced_vars(tape: &FbbtTape) -> Vec<usize> {
    let mut vs: Vec<usize> = tape
        .ops
        .iter()
        .filter_map(|op| match op {
            FbbtOp::Var(i) => Some(*i),
            _ => None,
        })
        .collect();
    vs.sort_unstable();
    vs.dedup();
    vs
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expr::{con, var};

    #[test]
    fn gradient_of_product_and_power() {
        // f = x0² · x1 + 3·x1.  ∇f = (2 x0 x1, x0² + 3).
        let f = (var(0).powi(2) * var(1)) + con(3.0) * var(1);
        let tape = f.to_tape();
        let g = gradient(&tape, &[2.0, 5.0], 2);
        assert!((g[0] - 2.0 * 2.0 * 5.0).abs() < 1e-9, "{g:?}");
        assert!((g[1] - (4.0 + 3.0)).abs() < 1e-9, "{g:?}");
    }

    #[test]
    fn gradient_of_transcendental() {
        // f = exp(x0) + ln(x1).  ∇f = (exp(x0), 1/x1).
        let f = var(0).exp() + var(1).ln();
        let tape = f.to_tape();
        let g = gradient(&tape, &[0.5, 4.0], 2);
        assert!((g[0] - 0.5_f64.exp()).abs() < 1e-9, "{g:?}");
        assert!((g[1] - 0.25).abs() < 1e-9, "{g:?}");
    }

    #[test]
    fn referenced_vars_dedup() {
        let f = var(2) * var(0) + var(2);
        assert_eq!(referenced_vars(&f.to_tape()), vec![0, 2]);
    }

    #[test]
    fn hessian_matches_analytic() {
        // f = x0² · x1 + 3·x1.
        // ∇²f = [[2 x1, 2 x0], [2 x0, 0]].
        let f = (var(0).powi(2) * var(1)) + con(3.0) * var(1);
        let h = hessian(&f.to_tape(), &[2.0, 5.0], 2).unwrap();
        assert!((h[0] - 2.0 * 5.0).abs() < 1e-9, "{h:?}"); // ∂²/∂x0² = 2 x1
        assert!((h[1] - 2.0 * 2.0).abs() < 1e-9, "{h:?}"); // ∂²/∂x0∂x1 = 2 x0
        assert!((h[2] - 2.0 * 2.0).abs() < 1e-9, "{h:?}"); // symmetric
        assert!(h[3].abs() < 1e-9, "{h:?}"); // ∂²/∂x1² = 0
    }

    #[test]
    fn hessian_of_transcendental_matches_finite_diff() {
        // f = exp(x0·x1) + ln(x1) + sin(x0).
        let f = (var(0) * var(1)).exp() + var(1).ln() + var(0).sin();
        let tape = f.to_tape();
        let x = [0.3, 1.7];
        let h = hessian(&tape, &x, 2).unwrap();
        // Central differences of the exact gradient as the reference.
        let fd = |i: usize, j: usize| {
            let hk = 1e-6;
            let mut xp = x;
            xp[j] += hk;
            let gp = gradient(&tape, &xp, 2)[i];
            xp[j] -= 2.0 * hk;
            let gm = gradient(&tape, &xp, 2)[i];
            (gp - gm) / (2.0 * hk)
        };
        for i in 0..2 {
            for j in 0..2 {
                assert!(
                    (h[i * 2 + j] - fd(i, j)).abs() < 1e-5,
                    "H[{i}][{j}]={} vs fd={}",
                    h[i * 2 + j],
                    fd(i, j)
                );
            }
        }
    }

    #[test]
    fn hessian_declines_on_nonsmooth() {
        // |x0| is not twice differentiable — the sweep declines.
        let f = var(0).abs() + var(1).powi(2);
        assert!(hessian(&f.to_tape(), &[0.5, 1.0], 2).is_none());
    }
}
