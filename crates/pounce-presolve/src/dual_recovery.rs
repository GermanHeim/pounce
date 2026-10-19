//! Postsolve dual recovery for bounds presolve *derived* (issue M24).
//!
//! Phase 1 (activity-based bound tightening) and FBBT derive variable bounds
//! from rows: `x0 = 2·x1, x0 ≥ 4` implies `x1 ≥ 2`. The reduced problem
//! carries that bound, and when it ends up active the solver is free to put
//! the multiplier on it — the reduced problem is degenerate there, since the
//! derived bound is implied by the row and the other variables' bounds. The
//! original problem has no such bound, so a multiplier on it is not a
//! multiplier of the original problem at all: written to a `.sol` it fails
//! `pounce verify --require-optimal` (measured on the example above:
//! `λ = 8`, `z_L(x1) = 20` against the true `λ = -2`, `z_L(x0) = 10`).
//!
//! # The rule
//!
//! At the postsolved primal `x*`, every multiplier the solver put on a
//! *derived* side that the original model does not have active there is
//! **spurious**. Zero it, and repair the stationarity residual that leaves by
//! re-solving for the original problem's multipliers on the active set,
//! restricted to the connected component the residual reaches:
//!
//! ```text
//!   min ‖∇f + Jᵀλ − z_L + z_U‖²   over the active rows and declared bounds
//!   s.t. λ_i ≤ 0 (row active at g_l only),  λ_i ≥ 0 (at g_u only),
//!        λ_i free (equality / both sides),   z ≥ 0
//! ```
//!
//! (Ipopt's convention.) Where the original multipliers are unique — LICQ at
//! `x*` on the active set, the usual case — this *is* the `presolve=no`
//! answer, whatever the reduced solve's split was. Where they are not
//! (degenerate), the solve is a proximal-point iteration started from the
//! reduced solve's own multipliers (spurious entries zeroed), so it returns
//! a valid KKT multiplier set of the original problem that keeps as much of
//! the reduced solve's attribution as stays valid. That is deterministic,
//! and it is the easiest rule to defend: every multiplier in the answer
//! belongs to a constraint the original model declares and that is active,
//! with the right sign, and stationarity holds.
//!
//! What this does **not** move: a multiplier on a *declared* bound that is
//! active at `x*`, even when presolve derived a tighter-or-equal bound on the
//! same side (a degenerate pair: the declared bound is the incumbent and
//! keeps it), and anything the residual does not reach.
//!
//! The component is solved dense. It is capped ([`MAX_UNKNOWNS`],
//! [`MAX_COMPONENT_VARS`]); a component past the cap, or a solve that does
//! not reduce the residual, leaves the duals exactly as the solver reported
//! them and says so through [`Outcome`].
//!
//! Activity is geometric, at `x*`, and so are the multipliers: the solve
//! reads `∇f(x*)`. That is right for a point the solver converged to, where
//! complementarity is `~μ` and an active side's slack is far below the
//! [`on_bound`] tolerance. It is not right for a warm start handed back
//! without a step: the interior-point start pushed `x` about `1e-3` off its
//! bounds, so the derived side reads inactive *and* `∇f` there is not the
//! one the supplied duals balance — widening the activity test alone was
//! measured to leave every re-derived multiplier `O(push)` off (CHAIN:
//! `[6.003, -3.003]` against `[6, -3]`). That case never reaches this
//! module: `PresolveTnlp::finalize_solution` recognises the served warm
//! start coming back and reports the supplied duals (`crate::warm_echo`).

use pounce_common::types::{Number, lower_bound_present, upper_bound_present};

/// Most multipliers one dense recovery solve will take on.
pub(crate) const MAX_UNKNOWNS: usize = 600;
/// Most variables one connected component may span.
pub(crate) const MAX_COMPONENT_VARS: usize = 5000;

/// Read-only view of the original problem at the postsolved point.
pub(crate) struct KktView<'a> {
    pub x: &'a [Number],
    pub g: &'a [Number],
    pub grad_f: &'a [Number],
    /// Jacobian triplets `(row, col, value)`, 0-based.
    pub jac: &'a [(usize, usize, Number)],
    /// The model's *declared* box and row bounds.
    pub x_l: &'a [Number],
    pub x_u: &'a [Number],
    pub g_l: &'a [Number],
    pub g_u: &'a [Number],
    /// Sides on which presolve's bound is strictly tighter than the
    /// declared one (or the declared one is absent).
    pub derived_l: &'a [bool],
    pub derived_u: &'a [bool],
}

/// What [`reattribute`] did.
#[derive(Debug, Clone, Default, PartialEq)]
pub(crate) struct Outcome {
    /// Spurious multipliers zeroed (non-zero entries on derived sides the
    /// original model does not have active at `x*`).
    pub n_spurious: usize,
    /// Multipliers the dense solve took on (0 when none was needed).
    pub n_unknowns: usize,
    /// `‖stationarity‖∞` over the component, before (with the spurious
    /// entries in place) and after.
    pub residual_before: Number,
    pub residual_after: Number,
    /// `true` when the duals were left untouched because the component was
    /// past the caps or the solve did not reduce the residual.
    pub rolled_back: bool,
}

/// Is `v` on the bound `b`, to the tolerance an interior-point solution
/// reaches a bound it is active at (well above `bound_relax_factor`)?
fn on_bound(v: Number, b: Number) -> bool {
    (v - b).abs() <= 1e-6 * b.abs().max(1.0)
}

/// Sign freedom of one unknown multiplier.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Sign {
    Free,
    NonNeg,
    NonPos,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Unknown {
    Row(usize),
    ZLower(usize),
    ZUpper(usize),
}

/// Stationarity residual `∇f + Jᵀλ − z_L + z_U`, all of it.
fn residual(v: &KktView<'_>, lambda: &[Number], z_l: &[Number], z_u: &[Number]) -> Vec<Number> {
    let mut r = v.grad_f.to_vec();
    for &(i, j, a) in v.jac {
        if let (Some(rj), Some(&l)) = (r.get_mut(j), lambda.get(i)) {
            *rj += a * l;
        }
    }
    for (j, rj) in r.iter_mut().enumerate() {
        *rj += -z_l[j] + z_u[j];
    }
    r
}

/// Zero the multipliers presolve-derived bounds carry against the original
/// model, and recover the original problem's multipliers in their place.
/// See the module docs for the rule. `lambda`, `z_l`, `z_u` are in the
/// original problem's row/variable space, Ipopt's sign convention.
pub(crate) fn reattribute(
    v: &KktView<'_>,
    lambda: &mut [Number],
    z_l: &mut [Number],
    z_u: &mut [Number],
) -> Outcome {
    let n = v.x.len();
    let m = v.g.len();
    let mut out = Outcome::default();
    if z_l.len() != n || z_u.len() != n || lambda.len() != m || v.grad_f.len() != n {
        return out;
    }
    // A spurious entry: a derived side the original does not have active.
    let declared_active_l = |j: usize| lower_bound_present(v.x_l[j]) && on_bound(v.x[j], v.x_l[j]);
    let declared_active_u = |j: usize| upper_bound_present(v.x_u[j]) && on_bound(v.x[j], v.x_u[j]);
    let mut spurious: Vec<usize> = Vec::new();
    for j in 0..n {
        let sl = v.derived_l[j] && z_l[j].is_finite() && z_l[j] != 0.0 && !declared_active_l(j);
        let su = v.derived_u[j] && z_u[j].is_finite() && z_u[j] != 0.0 && !declared_active_u(j);
        if sl || su {
            spurious.push(j);
        }
    }
    if spurious.is_empty() {
        return out;
    }
    let (lam0, zl0, zu0) = (lambda.to_vec(), z_l.to_vec(), z_u.to_vec());
    let before = residual(v, lambda, z_l, z_u);
    for &j in &spurious {
        if v.derived_l[j] && !declared_active_l(j) && z_l[j].is_finite() && z_l[j] != 0.0 {
            z_l[j] = 0.0;
            out.n_spurious += 1;
        }
        if v.derived_u[j] && !declared_active_u(j) && z_u[j].is_finite() && z_u[j] != 0.0 {
            z_u[j] = 0.0;
            out.n_spurious += 1;
        }
    }
    let r = residual(v, lambda, z_l, z_u);
    let scale = v.grad_f.iter().fold(
        1.0_f64,
        |a, &b| if b.is_finite() { a.max(b.abs()) } else { a },
    );
    let seed_tol = 1e-10 * scale;
    let seeds: Vec<usize> = spurious
        .iter()
        .copied()
        .filter(|&j| r[j].abs() > seed_tol)
        .collect();
    if seeds.is_empty() {
        // Only negligible mass was on the derived sides: zeroing it is the
        // whole repair.
        return out;
    }

    // Adjacency, both ways.
    let mut by_col: Vec<Vec<(usize, Number)>> = vec![Vec::new(); n];
    let mut by_row: Vec<Vec<(usize, Number)>> = vec![Vec::new(); m];
    for &(i, j, a) in v.jac {
        if i < m && j < n && a != 0.0 && a.is_finite() {
            by_col[j].push((i, a));
            by_row[i].push((j, a));
        }
    }
    let row_sign = |i: usize| -> Option<Sign> {
        let (gl, gu, gi) = (v.g_l[i], v.g_u[i], v.g[i]);
        if !gi.is_finite() {
            return None;
        }
        let lo = lower_bound_present(gl) && on_bound(gi, gl);
        let hi = upper_bound_present(gu) && on_bound(gi, gu);
        match (lo, hi) {
            (true, true) => Some(Sign::Free),
            (true, false) => Some(Sign::NonPos),
            (false, true) => Some(Sign::NonNeg),
            (false, false) => None,
        }
    };

    // Breadth-first over (variable, active row) incidence from the seeds.
    let mut var_slot = vec![usize::MAX; n];
    let mut vars: Vec<usize> = Vec::new();
    let mut row_seen = vec![false; m];
    let mut unknowns: Vec<(Unknown, Sign)> = Vec::new();
    let mut queue = std::collections::VecDeque::new();
    for &j in &seeds {
        if var_slot[j] == usize::MAX {
            var_slot[j] = vars.len();
            vars.push(j);
            queue.push_back(j);
        }
    }
    let mut over_cap = false;
    while let Some(j) = queue.pop_front() {
        if declared_active_l(j) {
            unknowns.push((Unknown::ZLower(j), Sign::NonNeg));
        }
        if declared_active_u(j) {
            unknowns.push((Unknown::ZUpper(j), Sign::NonNeg));
        }
        for &(i, _) in &by_col[j] {
            if row_seen[i] {
                continue;
            }
            row_seen[i] = true;
            let Some(sign) = row_sign(i) else { continue };
            unknowns.push((Unknown::Row(i), sign));
            for &(k, _) in &by_row[i] {
                if var_slot[k] == usize::MAX {
                    var_slot[k] = vars.len();
                    vars.push(k);
                    queue.push_back(k);
                }
            }
        }
        if unknowns.len() > MAX_UNKNOWNS || vars.len() > MAX_COMPONENT_VARS {
            over_cap = true;
            break;
        }
    }
    out.residual_before = vars.iter().fold(0.0_f64, |a, &j| a.max(before[j].abs()));
    let restore = |lambda: &mut [Number], z_l: &mut [Number], z_u: &mut [Number]| {
        lambda.copy_from_slice(&lam0);
        z_l.copy_from_slice(&zl0);
        z_u.copy_from_slice(&zu0);
    };
    if over_cap || unknowns.is_empty() {
        restore(lambda, z_l, z_u);
        out.rolled_back = true;
        out.residual_after = out.residual_before;
        out.n_spurious = 0;
        return out;
    }
    out.n_unknowns = unknowns.len();

    // Dense A (|vars| × k) in the transformed variable `w = s·y` so every
    // sign constraint reads `w ≥ 0`, and b = r − A·w0 so the component's
    // residual is `A·w + b`.
    let nv = vars.len();
    let k = unknowns.len();
    let mut a = vec![0.0; nv * k];
    let mut w0 = vec![0.0; k];
    let mut constrained = vec![false; k];
    for (c, &(u, sign)) in unknowns.iter().enumerate() {
        let s = if sign == Sign::NonPos { -1.0 } else { 1.0 };
        constrained[c] = sign != Sign::Free;
        match u {
            Unknown::Row(i) => {
                for &(jj, aij) in &by_row[i] {
                    let slot = var_slot[jj];
                    if slot != usize::MAX {
                        a[slot * k + c] += s * aij;
                    }
                }
                w0[c] = s * lambda[i];
            }
            Unknown::ZLower(j) => {
                a[var_slot[j] * k + c] = -1.0;
                w0[c] = z_l[j];
            }
            Unknown::ZUpper(j) => {
                a[var_slot[j] * k + c] = 1.0;
                w0[c] = z_u[j];
            }
        }
        if constrained[c] && !(w0[c] >= 0.0) {
            w0[c] = 0.0;
        }
    }
    let mut b: Vec<Number> = vars.iter().map(|&j| r[j]).collect();
    // r was computed with the current (pre-projection) values; express it
    // against the projected start.
    for (c, &(u, sign)) in unknowns.iter().enumerate() {
        let s = if sign == Sign::NonPos { -1.0 } else { 1.0 };
        let current = match u {
            Unknown::Row(i) => s * lambda[i],
            Unknown::ZLower(j) => z_l[j],
            Unknown::ZUpper(j) => z_u[j],
        };
        let current = if current.is_finite() { current } else { 0.0 };
        for (row, bv) in b.iter_mut().enumerate() {
            *bv -= a[row * k + c] * current;
        }
    }
    let w = solve_sign_constrained_ls(&a, &b, nv, k, &constrained, &w0);
    let res_after = (0..nv)
        .map(|row| {
            let mut acc = b[row];
            for c in 0..k {
                acc += a[row * k + c] * w[c];
            }
            acc.abs()
        })
        .fold(0.0_f64, Number::max);
    // Accept the recovery only when it actually closes stationarity: to the
    // solver's own residual, or to a level no tolerance would notice.
    let target = (10.0 * out.residual_before).max(1e-8 * scale);
    if !(res_after <= target) {
        restore(lambda, z_l, z_u);
        out.rolled_back = true;
        out.residual_after = out.residual_before;
        out.n_spurious = 0;
        out.n_unknowns = 0;
        return out;
    }
    for (c, &(u, sign)) in unknowns.iter().enumerate() {
        let s = if sign == Sign::NonPos { -1.0 } else { 1.0 };
        match u {
            Unknown::Row(i) => lambda[i] = s * w[c],
            Unknown::ZLower(j) => z_l[j] = w[c],
            Unknown::ZUpper(j) => z_u[j] = w[c],
        }
    }
    out.residual_after = res_after;
    out
}

/// `min ½‖A·w + b‖²` subject to `w_c ≥ 0` where `constrained[c]`, by a
/// proximal-point iteration `w⁺ = argmin ½‖A·w + b‖² + ½μ‖w − w_k‖²` started
/// at `w0`. Each step is a strictly convex bound-constrained QP, solved by a
/// primal active-set method. On a consistent system with a unique solution
/// this converges to it; on a degenerate one, to a solution near `w0`.
fn solve_sign_constrained_ls(
    a: &[Number],
    b: &[Number],
    nrows: usize,
    k: usize,
    constrained: &[bool],
    w0: &[Number],
) -> Vec<Number> {
    // Normal matrix and AᵀB.
    let mut ata = vec![0.0; k * k];
    let mut atb = vec![0.0; k];
    for row in 0..nrows {
        let ar = &a[row * k..(row + 1) * k];
        for p in 0..k {
            if ar[p] == 0.0 {
                continue;
            }
            atb[p] += ar[p] * b[row];
            for q in 0..k {
                ata[p * k + q] += ar[p] * ar[q];
            }
        }
    }
    let max_diag = (0..k).fold(0.0_f64, |m, p| m.max(ata[p * k + p]));
    let mu = 1e-6 * max_diag.max(1e-300);
    let mut h = ata.clone();
    for p in 0..k {
        h[p * k + p] += mu;
    }
    let mut w = w0.to_vec();
    for _ in 0..200 {
        // c = Aᵀb − μ w_k
        let c: Vec<Number> = (0..k).map(|p| atb[p] - mu * w[p]).collect();
        let next = box_qp(&h, &c, k, constrained, &w);
        let step = next
            .iter()
            .zip(&w)
            .fold(0.0_f64, |m, (p, q)| m.max((p - q).abs()));
        let size = next.iter().fold(1.0_f64, |m, p| m.max(p.abs()));
        w = next;
        if step <= 1e-15 * size {
            break;
        }
    }
    w
}

/// `min ½ wᵀHw + cᵀw` s.t. `w_c ≥ 0` where `constrained[c]`, `H` SPD,
/// started from the feasible `start`. Primal active-set method.
fn box_qp(
    h: &[Number],
    c: &[Number],
    k: usize,
    constrained: &[bool],
    start: &[Number],
) -> Vec<Number> {
    let mut w: Vec<Number> = start.to_vec();
    let mut fixed: Vec<bool> = (0..k).map(|p| constrained[p] && w[p] <= 0.0).collect();
    for p in 0..k {
        if fixed[p] {
            w[p] = 0.0;
        }
    }
    for _ in 0..(10 * k + 50) {
        let free: Vec<usize> = (0..k).filter(|&p| !fixed[p]).collect();
        // Solve H_FF p_F = −c_F (fixed entries are 0).
        let nf = free.len();
        let mut cand = vec![0.0; k];
        if nf > 0 {
            let mut hf = vec![0.0; nf * nf];
            let mut rhs = vec![0.0; nf];
            for (r, &p) in free.iter().enumerate() {
                rhs[r] = -c[p];
                for (s, &q) in free.iter().enumerate() {
                    hf[r * nf + s] = h[p * k + q];
                }
            }
            if !cholesky_solve(&mut hf, &mut rhs, nf) {
                return w;
            }
            for (r, &p) in free.iter().enumerate() {
                cand[p] = rhs[r];
            }
        }
        let infeasible = free.iter().any(|&p| constrained[p] && cand[p] < 0.0);
        if !infeasible {
            w = cand;
            // Multipliers of the fixed set: the gradient there.
            let mut worst: Option<(usize, Number)> = None;
            for p in 0..k {
                if !fixed[p] {
                    continue;
                }
                let mut gp = c[p];
                for q in 0..k {
                    gp += h[p * k + q] * w[q];
                }
                let tol = 1e-14 * (h[p * k + p].abs().max(1e-300)).sqrt() * (1.0 + c[p].abs());
                if gp < -tol && worst.is_none_or(|(_, g)| gp < g) {
                    worst = Some((p, gp));
                }
            }
            match worst {
                Some((p, _)) => fixed[p] = false,
                None => return w,
            }
        } else {
            // Step toward the candidate as far as feasibility allows.
            let mut alpha = 1.0_f64;
            let mut block = None;
            for &p in &free {
                if constrained[p] && cand[p] < 0.0 {
                    let t = w[p] / (w[p] - cand[p]);
                    if t < alpha {
                        alpha = t;
                        block = Some(p);
                    }
                }
            }
            for &p in &free {
                w[p] += alpha * (cand[p] - w[p]);
            }
            if let Some(p) = block {
                fixed[p] = true;
                w[p] = 0.0;
            }
            for &p in &free {
                if constrained[p] && w[p] <= 0.0 {
                    fixed[p] = true;
                    w[p] = 0.0;
                }
            }
        }
    }
    w
}

/// In-place Cholesky solve of the SPD `n×n` system `m·x = rhs`; `rhs` is
/// overwritten with `x`. `false` if `m` is not numerically SPD.
fn cholesky_solve(m: &mut [Number], rhs: &mut [Number], n: usize) -> bool {
    for j in 0..n {
        let mut d = m[j * n + j];
        for p in 0..j {
            d -= m[j * n + p] * m[j * n + p];
        }
        if !(d > 0.0) {
            return false;
        }
        let d = d.sqrt();
        m[j * n + j] = d;
        for i in (j + 1)..n {
            let mut s = m[i * n + j];
            for p in 0..j {
                s -= m[i * n + p] * m[j * n + p];
            }
            m[i * n + j] = s / d;
        }
    }
    for i in 0..n {
        let mut s = rhs[i];
        for p in 0..i {
            s -= m[i * n + p] * rhs[p];
        }
        rhs[i] = s / m[i * n + i];
    }
    for i in (0..n).rev() {
        let mut s = rhs[i];
        for p in (i + 1)..n {
            s -= m[p * n + i] * rhs[p];
        }
        rhs[i] = s / m[i * n + i];
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `min x0² + x1²  s.t.  x0 − 2·x1 = 0,  x0 ≥ 4`, at `x* = (4, 2)`, with
    /// presolve's derived `x1 ≥ 2` carrying the reduced solve's whole
    /// multiplier (`λ = −8`, `z_L(x1) = 20`): the original problem's unique
    /// multipliers are `λ = 2`, `z_L(x0) = 10`.
    #[test]
    fn a_derived_bound_multiplier_goes_back_to_the_row_and_the_declared_bound() {
        let jac = [(0, 0, 1.0), (0, 1, -2.0)];
        let v = KktView {
            x: &[4.0, 2.0],
            g: &[0.0],
            grad_f: &[8.0, 4.0],
            jac: &jac,
            x_l: &[4.0, -1e19],
            x_u: &[1e19, 1e19],
            g_l: &[0.0],
            g_u: &[0.0],
            derived_l: &[false, true],
            derived_u: &[false, false],
        };
        let (mut lam, mut zl, mut zu) = (vec![-8.0], vec![0.0, 20.0], vec![0.0, 0.0]);
        let o = reattribute(&v, &mut lam, &mut zl, &mut zu);
        assert!(!o.rolled_back, "{o:?}");
        assert_eq!(o.n_spurious, 1);
        assert!((lam[0] - 2.0).abs() < 1e-10, "λ = {lam:?}");
        assert!((zl[0] - 10.0).abs() < 1e-10, "z_L = {zl:?}");
        assert_eq!(zl[1], 0.0);
        let r = residual(&v, &lam, &zl, &zu);
        assert!(r.iter().all(|x| x.abs() < 1e-10), "{r:?}");
    }

    /// A multiplier on a declared bound the derived one ties with stays put
    /// (the incumbent keeps a degenerate pair), and nothing else moves.
    #[test]
    fn a_declared_active_bound_keeps_its_multiplier() {
        let jac = [(0, 0, 1.0), (0, 1, -2.0)];
        let v = KktView {
            x: &[4.0, 2.0],
            g: &[0.0],
            grad_f: &[8.0, 4.0],
            jac: &jac,
            x_l: &[4.0, 2.0],
            x_u: &[1e19, 1e19],
            g_l: &[0.0],
            g_u: &[0.0],
            derived_l: &[false, true],
            derived_u: &[false, false],
        };
        let (mut lam, mut zl, mut zu) = (vec![-8.0], vec![0.0, 20.0], vec![0.0, 0.0]);
        let o = reattribute(&v, &mut lam, &mut zl, &mut zu);
        assert_eq!(o, Outcome::default());
        assert_eq!((lam, zl), (vec![-8.0], vec![0.0, 20.0]));
    }

    /// Sign constraints bind: an inequality active at its lower side only
    /// takes `λ ≤ 0`, and a degenerate pair of rows resolves to a valid set.
    #[test]
    fn inequality_signs_are_respected() {
        // min (x1 − 3)²  s.t. C0: x0 = 0.5, C1: x0 + x1 ≤ 2. x* = (0.5, 1.5).
        // ∇f = (0, −3). Original: λ1 = 3 (≥ 0, upper-active), λ0 = −3.
        let jac = [(0, 0, 1.0), (1, 0, 1.0), (1, 1, 1.0)];
        let v = KktView {
            x: &[0.5, 1.5],
            g: &[0.5, 2.0],
            grad_f: &[0.0, -3.0],
            jac: &jac,
            x_l: &[-1e19, -1e19],
            x_u: &[1e19, 1e19],
            g_l: &[0.5, -1e19],
            g_u: &[0.5, 2.0],
            derived_l: &[true, false],
            derived_u: &[true, true],
        };
        let (mut lam, mut zl, mut zu) = (vec![0.0, 0.0], vec![0.0, 0.0], vec![0.0, 3.0]);
        let o = reattribute(&v, &mut lam, &mut zl, &mut zu);
        assert!(!o.rolled_back, "{o:?}");
        assert!(
            (lam[0] + 3.0).abs() < 1e-10 && (lam[1] - 3.0).abs() < 1e-10,
            "{lam:?}"
        );
        assert_eq!((zl, zu), (vec![0.0, 0.0], vec![0.0, 0.0]));
    }
}
