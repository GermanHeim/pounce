//! Bridge from the relaxation [`QpProblem`] to the revised simplex.
//!
//! OBBT's relaxation is a *pure LP* (`P` empty): `min cᵀx` over `A x = b`,
//! `G x ≤ h`, `lb ≤ x ≤ ub`. `pounce-simplex` consumes the equality-plus-bounds
//! standard form `min cᵀx s.t. A x = b, l ≤ x ≤ u`, so each inequality row
//! `gₖ·x ≤ hₖ` is turned into an equality `gₖ·x + sₖ = hₖ` with a fresh
//! nonnegative slack `sₖ`. The `n` original variables stay columns `0..n`; the
//! `m_ineq` slacks are appended as columns `n..n+m_ineq`.
//!
//! The payoff is the OBBT inner loop: all `2n` min/max LPs share this one
//! polytope and differ only in the objective, so a single [`Simplex`] solves the
//! first cold and warm-starts every subsequent objective flip from the previous
//! optimal basis — a few pivots each, instead of an IPM re-walking the central
//! path `2n` times.

use crate::obbt::{past_deadline, VarRange};
use pounce_convex::QpProblem;
use pounce_simplex::{LpProblem, LpStatus, Simplex, Triplet};
use std::time::Instant;

/// Convert a pure-LP [`QpProblem`] (no Hessian) to the simplex standard form by
/// adding one nonnegative slack per inequality row. `debug_assert`s that `P` is
/// empty — callers must only pass relaxation LPs here.
pub(crate) fn qp_to_lp(qp: &QpProblem) -> LpProblem {
    debug_assert!(
        qp.p_lower.is_empty(),
        "qp_to_lp expects a pure LP (empty P)"
    );
    let n = qp.n;
    let m_eq = qp.m_eq();
    let m_ineq = qp.m_ineq();
    let n_lp = n + m_ineq;
    let m_lp = m_eq + m_ineq;

    let mut a: Vec<Triplet> = Vec::with_capacity(qp.a.len() + qp.g.len() + m_ineq);
    // Equality rows carry over unchanged.
    for t in &qp.a {
        a.push(Triplet::new(t.row, t.col, t.val));
    }
    // Each inequality row `gₖ·x ≤ hₖ` → `gₖ·x + sₖ = hₖ`, slack in column n+k.
    for t in &qp.g {
        a.push(Triplet::new(m_eq + t.row, t.col, t.val));
    }
    for k in 0..m_ineq {
        a.push(Triplet::new(m_eq + k, n + k, 1.0));
    }

    let mut b = Vec::with_capacity(m_lp);
    b.extend_from_slice(&qp.b);
    b.extend_from_slice(&qp.h);

    // Objective: structural costs carry over; slacks cost nothing. (OBBT
    // overrides this per solve via `Simplex::solve_objective`.)
    let mut c = vec![0.0; n_lp];
    c[..n].copy_from_slice(&qp.c[..n.min(qp.c.len())]);

    // Bounds: structural from the QP (±∞ sentinels pass straight through — the
    // simplex treats |bound| ≥ 1e20 as infinite); slacks are `[0, +∞)`.
    let mut lb = vec![0.0; n_lp];
    let mut ub = vec![f64::INFINITY; n_lp];
    for j in 0..n {
        lb[j] = qp.lb_of(j);
        ub[j] = qp.ub_of(j);
    }
    // Slacks already 0 / +∞ from the initializers above.

    LpProblem {
        n: n_lp,
        m: m_lp,
        c,
        a,
        b,
        lb,
        ub,
    }
}

/// Run an OBBT pass over `qp` (the pass-start relaxation, cutoff row already
/// appended) with the warm-started revised simplex: minimize then maximize each
/// of the first `n` variables, reusing one basis across all `2n` objective
/// flips. Returns `(results, timed_out)` where `results[i] = (min xᵢ, max xᵢ)`
/// (each `None` if that solve was not optimal), matching the IPM sweep's shape.
///
/// `deadline`, when set, is polled before each variable's min/max pair; on
/// expiry the sweep stops and `timed_out` is `true` (partial results are still
/// valid bounds).
pub(crate) fn sweep(
    qp: &QpProblem,
    n: usize,
    targets: Option<&[bool]>,
    deadline: Option<Instant>,
) -> (Vec<VarRange>, bool) {
    let lp = qp_to_lp(qp);
    let n_lp = lp.n;

    let mut simplex = Simplex::new(&lp);

    // Prime: a zero-objective solve finds a feasible vertex (Phase I) and leaves
    // a reusable basis. If the polytope is infeasible/unbounded/degenerate here,
    // tighten nothing this pass (step-2's lower-bound solve handles pruning) —
    // exactly the IPM path's behaviour when its solves come back non-optimal.
    if simplex.solve().status != LpStatus::Optimal {
        return (vec![(None, None); n], false);
    }

    let mut c = vec![0.0; n_lp];
    let mut out = Vec::with_capacity(n);
    let mut timed_out = false;
    for i in 0..n {
        if past_deadline(deadline) {
            timed_out = true;
            break;
        }
        // Budgeted sweep: skip variables not selected this pass (their result is
        // `(None, None)`), matching the IPM path's `targets` mask.
        if targets.is_some_and(|t| !t[i]) {
            out.push((None, None));
            continue;
        }
        c[i] = 1.0;
        let smin = simplex.solve_objective(&c);
        let min = (smin.status == LpStatus::Optimal).then_some(smin.obj);
        c[i] = -1.0;
        let smax = simplex.solve_objective(&c);
        // max xᵢ = −min(−xᵢ).
        let max = (smax.status == LpStatus::Optimal).then_some(-smax.obj);
        c[i] = 0.0;
        out.push((min, max));
    }
    (out, timed_out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use pounce_convex::Triplet as QpTriplet;

    /// A pure-LP QpProblem: `x + y ≤ 10`, `0 ≤ x,y ≤ 8`. Each variable ranges
    /// over `[0, 8]` (its lower bound is reachable since the other can absorb
    /// the inequality; its upper bound 8 leaves the partner at 2 ≤ 10).
    fn box_lp() -> QpProblem {
        QpProblem {
            n: 2,
            p_lower: Vec::new(),
            c: vec![0.0, 0.0],
            a: Vec::new(),
            b: Vec::new(),
            g: vec![QpTriplet::new(0, 0, 1.0), QpTriplet::new(0, 1, 1.0)],
            h: vec![10.0],
            lb: vec![0.0, 0.0],
            ub: vec![8.0, 8.0],
        }
    }

    #[test]
    fn converter_adds_one_slack_per_inequality() {
        let lp = qp_to_lp(&box_lp());
        assert_eq!(lp.n, 3); // 2 structural + 1 slack
        assert_eq!(lp.m, 1); // 1 inequality → 1 equality row
        assert!(lp.lb[2] == 0.0 && lp.ub[2].is_infinite()); // slack ≥ 0
    }

    #[test]
    fn sweep_tightens_box() {
        let (res, timed) = sweep(&box_lp(), 2, None, None);
        assert!(!timed);
        let (xmin, xmax) = res[0];
        assert!((xmin.unwrap() - 0.0).abs() < 1e-6, "{xmin:?}");
        assert!((xmax.unwrap() - 8.0).abs() < 1e-6, "{xmax:?}");
        let (ymin, ymax) = res[1];
        assert!((ymin.unwrap() - 0.0).abs() < 1e-6, "{ymin:?}");
        assert!((ymax.unwrap() - 8.0).abs() < 1e-6, "{ymax:?}");
    }

    #[test]
    fn sweep_with_equality_and_negative_bounds() {
        // x + y = 4 (equality), −5 ≤ x ≤ 5, 0 ≤ y ≤ 5.
        // y ∈ [0,5] ⇒ x = 4 − y ∈ [−1, 4]; x ∈ [−5,5] ⇒ y = 4 − x ∈ [−1,9],
        // intersected with the bound [0,5] gives y ∈ [0, 5].
        let qp = QpProblem {
            n: 2,
            p_lower: Vec::new(),
            c: vec![0.0, 0.0],
            a: vec![QpTriplet::new(0, 0, 1.0), QpTriplet::new(0, 1, 1.0)],
            b: vec![4.0],
            g: Vec::new(),
            h: Vec::new(),
            lb: vec![-5.0, 0.0],
            ub: vec![5.0, 5.0],
        };
        let (res, _) = sweep(&qp, 2, None, None);
        let (xmin, xmax) = res[0];
        assert!((xmin.unwrap() - (-1.0)).abs() < 1e-6, "xmin {xmin:?}");
        assert!((xmax.unwrap() - 4.0).abs() < 1e-6, "xmax {xmax:?}");
        let (ymin, ymax) = res[1];
        assert!((ymin.unwrap() - 0.0).abs() < 1e-6, "ymin {ymin:?}");
        assert!((ymax.unwrap() - 5.0).abs() < 1e-6, "ymax {ymax:?}");
    }
}
