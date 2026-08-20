//! Newton iterations on the barrier system, against the held factor.
//!
//! A parametric step is the first-order prediction of where the
//! solution moves. It leaves a residual in the barrier KKT system at
//! the perturbed parameter values, and that residual can be driven
//! down by Newton iterations that reuse the converged factorization,
//! so each one costs a back-solve rather than a factorization.
//!
//! The operator is the one the solve left behind, evaluated at the
//! base point. That is what makes an iteration cheap and also what
//! bounds what the corrector can do: where the perturbation moves a
//! bound out of the active set, the held barrier diagonal carries that
//! bound's stiffness from the base point, `z / s` at a bound the solve
//! held tightly is `z² / μ`, and no number of iterations against that
//! operator will let the bound go. The iteration then makes no
//! progress at all, which the residual reports on the first step.
//!
//! So this is not a method that converges to a re-solve. It improves
//! the step where the active set the base point settled still fits,
//! and it says how far it got. The residual is the honest measure: the
//! distance to the true solution is not knowable without solving, and
//! the achievable accuracy varies by problem, from the held-μ offset
//! of about `1e-7` on some models to `1e-9` on others.
//!
//! The residual comes from the algorithm's own calculated quantities
//! by way of the trial iterate, so scaling, fixed variables, and the
//! bound expansions are handled exactly as the solve handles them.
//! Setting a trial point leaves `curr` alone, and `curr` is what the
//! held factorization was built from, so nothing here disturbs the
//! factor or any other consumer of the session.

use std::rc::Rc;

use pounce_common::types::{Index, Number};

use crate::backsolver::SensBacksolver;
use crate::solver::SolverError;

/// How far a step may travel toward a bound, as a fraction of the
/// distance remaining. The barrier's own rule, at the value the
/// algorithm uses once `μ` is small.
const TAU: Number = 0.9995;

/// What one call to the corrector did.
///
/// `residual` is the primal-dual barrier residual at the returned
/// point, and `initial_residual` the same quantity at the step the
/// caller handed in, so the ratio says what the iterations bought.
/// When they are equal the corrector made no progress, which happens
/// when the perturbation needs a bound the base point held to leave.
#[derive(Debug, Clone, PartialEq)]
pub struct CorrectorReport {
    /// Back-solves spent. One per iteration.
    pub iterations: usize,
    /// Residual at the returned point.
    pub residual: Number,
    /// Residual at the step handed in.
    pub initial_residual: Number,
    /// True when the loop stopped because an iteration failed to
    /// improve on the best residual seen, rather than because it ran
    /// out of budget.
    pub converged: bool,
    /// Bounds the step took out of the active set, which the corrector
    /// removed from the operator once before iterating.
    pub released: usize,
    /// The returned point's residual split the way the compound system
    /// is: how far the Lagrangian's gradient is from zero, how far the
    /// model's own equations are from being satisfied, and how far the
    /// bound multipliers are from complementarity at the held `mu`.
    ///
    /// The three carry different units and the reported `residual` is
    /// the largest of them, so a caller deciding whether an estimate is
    /// usable wants them apart. A correction can leave the equations
    /// nearly satisfied while the multipliers are far off, and only the
    /// first of those affects whether the values can be acted on.
    pub stationarity: Number,
    /// Largest violation of the model's equality and inequality rows.
    pub feasibility: Number,
    /// Largest departure from `z · s = mu` over the bound multipliers.
    pub complementarity: Number,
}

impl CorrectorReport {
    /// Whether the iterations improved on the step handed in.
    ///
    /// False means the returned point is the caller's own step, which
    /// is what happens when the operator cannot represent the
    /// active-set change the perturbation needs.
    pub fn improved(&self) -> bool {
        self.residual < self.initial_residual
    }
}

/// The barrier residual's blocks, in the compound layout's order.
///
/// Assembled from the trial iterate's calculated quantities, with the
/// parametric shift applied to the equality rows the caller pinned:
/// the corrector is aiming at the perturbed problem, whose equality
/// constraints sit at the shifted right-hand side.
pub(crate) fn residual_at(
    bs: &crate::algorithm_backsolver::PdSensBacksolver,
    flat: &[Number],
    pins: &[Index],
    deltas: &[Number],
    mu: Number,
    out: &mut [Number],
) -> Result<(), SolverError> {
    let iv = bs
        .pack_public(flat)
        .map_err(|_| SolverError::SensComputationFailed("corrector: pack failed".into()))?;
    let (data, cq, _) = bs.activity_handles();
    data.borrow_mut().set_trial(iv.freeze());

    let off = bs.offsets_public();
    let cqb = cq.borrow();
    let blocks: [(usize, std::rc::Rc<dyn pounce_linalg::Vector>, Number); 8] = [
        (0, cqb.trial_grad_lag_x(), 0.0),
        (1, cqb.trial_grad_lag_s(), 0.0),
        (2, cqb.trial_c(), 0.0),
        (3, cqb.trial_d_minus_s(), 0.0),
        (4, cqb.trial_compl_x_l(), mu),
        (5, cqb.trial_compl_x_u(), mu),
        (6, cqb.trial_compl_s_l(), mu),
        (7, cqb.trial_compl_s_u(), mu),
    ];
    for (i, v, shift) in blocks {
        let vals = crate::vec_util::dense_to_vec(&*v);
        let (a, b) = (off[i], off[i + 1]);
        if vals.len() != b - a {
            return Err(SolverError::SensComputationFailed(format!(
                "corrector: block {i} is {} long, expected {}",
                vals.len(),
                b - a
            )));
        }
        for (o, val) in out[a..b].iter_mut().zip(vals) {
            *o = val - shift;
        }
    }
    drop(cqb);

    // The perturbation moves the pinned equalities' right-hand sides,
    // so the residual there is measured against the moved value. The
    // rows are indices into `g`, which is the `y_c` block.
    let (yc_a, yc_b) = (off[2], off[3]);
    for (&row, &d) in pins.iter().zip(deltas) {
        let r = yc_a + row as usize;
        if r >= yc_b {
            return Err(SolverError::SensComputationFailed(format!(
                "corrector: pin row {row} is outside the equality block"
            )));
        }
        out[r] -= d;
    }
    Ok(())
}

/// The largest step from `val` along `dir` that keeps every entry of a
/// positive quantity at or above `1 - TAU` of where it started.
fn fraction_to_boundary(val: &[Number], dir: &[Number]) -> Number {
    let mut a = 1.0;
    for (&v, &d) in val.iter().zip(dir) {
        if d < 0.0 {
            let lim = -TAU * v / d;
            if lim < a {
                a = lim;
            }
        }
    }
    a.max(0.0)
}

/// The slacks a primal point sits at, per bound row, and the direction
/// those slacks move under a primal direction.
///
/// The bound rows carry the variable each one constrains and its side,
/// which is what turns a primal vector into the per-bound quantity the
/// fraction rule needs.
fn slacks_and_directions(
    rows: &[crate::backsolver::BoundRow],
    x: &[Number],
    dx: &[Number],
    lo: &[Number],
    hi: &[Number],
    lower: bool,
) -> (Vec<Number>, Vec<Number>) {
    let mut s = Vec::new();
    let mut ds = Vec::new();
    for b in rows.iter().filter(|b| b.lower == lower) {
        let i = b.var_row;
        if lower {
            s.push(x[i] - lo[i]);
            ds.push(dx[i]);
        } else {
            s.push(hi[i] - x[i]);
            ds.push(-dx[i]);
        }
    }
    (s, ds)
}

/// The bounds the step takes out of the active set.
///
/// A bound the solve held tightly contributes `z / s` to the barrier
/// diagonal, which at a small `mu` is a very large number, and the
/// held factorization carries it. If the perturbation moves that
/// variable off its bound, iterating against that operator cannot
/// follow: the stiffness the base point had is still there. The way
/// out is to take the bound out of the operator once, before
/// iterating, which is what `solve_released` does.
///
/// Which bounds those are is the predictor's answer, read off the step
/// it produced rather than decided here. A bound is released when the
/// solve held it, meaning its barrier diagonal is above one, and the
/// step carries the variable off it by more than roundoff. Reading the
/// primal endpoint works for every mode: `fix_relax` and `path` have
/// already applied their releases by the time the step is handed over,
/// and the plain step shows the same crossing in the coordinate it
/// carries past the bound.
fn released_rows(
    rows: &[crate::backsolver::BoundRow],
    base: &[Number],
    end: &[Number],
    lo: &[Number],
    hi: &[Number],
) -> Vec<usize> {
    let mut out = Vec::new();
    for b in rows {
        let i = b.var_row;
        let (s_base, s_end) = if b.lower {
            (base[i] - lo[i], end[i] - lo[i])
        } else {
            (hi[i] - base[i], hi[i] - end[i])
        };
        let z_base = base[b.row];
        if s_base <= 0.0 || z_base / s_base <= 1.0 {
            continue; // the solve did not hold this bound
        }
        // Off the bound by more than the slack it sat at, and by more
        // than roundoff against the variable's own size.
        if s_end > 10.0 * s_base && s_end > 1e-9 * (1.0 + base[i].abs()) {
            out.push(b.row);
        }
    }
    out
}

/// The bounds the step brings into the active set, with the barrier
/// stiffness each one needs.
///
/// A variable the solve left interior contributes almost nothing to
/// the barrier diagonal, `mu / s²` at a slack of order one, so the
/// held factorization treats it as free. If the step carries it onto a
/// bound, iterating against that operator pushes it straight back out
/// and the fraction-to-boundary rule refuses the step, which is the
/// same failure a released bound causes, in the other direction.
///
/// The stiffness a bound at slack `s` carries is `mu / s²`, which is
/// what the barrier itself would assign there, so that is what goes on
/// the diagonal. The variable sits at the margin the start put it at.
fn pinned_rows(
    rows: &[crate::backsolver::BoundRow],
    base: &[Number],
    end: &[Number],
    lo: &[Number],
    hi: &[Number],
    mu: Number,
) -> Vec<(usize, Number)> {
    let mut out = Vec::new();
    for b in rows {
        let i = b.var_row;
        let (s_base, s_end) = if b.lower {
            (base[i] - lo[i], end[i] - lo[i])
        } else {
            (hi[i] - base[i], hi[i] - end[i])
        };
        let z_base = base[b.row];
        if s_base <= 0.0 || z_base / s_base > 1.0 {
            continue; // the solve already held this bound
        }
        // On the bound now, and it was not before.
        if s_end > 0.0 && s_end < 0.1 * s_base && s_end < 1e-6 * (1.0 + base[i].abs()) {
            out.push((i, mu / (s_end * s_end)));
        }
    }
    out
}

/// Run the corrector.
///
/// `start` is the caller's compound step, `base` the converged
/// iterate, both in the compound layout. Returns the corrected step
/// and what the iterations did.
///
/// The loop keeps the iterate with the smallest residual seen and
/// stops when an iteration fails to improve on it. That one comparison
/// covers all three ways the iteration ends: reaching the accuracy the
/// held operator supports, making no progress at all because a bound
/// must leave, and settling into a cycle where the fraction rule
/// alternates between two points.
#[allow(clippy::too_many_arguments)]
pub(crate) fn run(
    bs: &crate::algorithm_backsolver::PdSensBacksolver,
    base: &[Number],
    start: &[Number],
    pins: &[Index],
    deltas: &[Number],
    lo: &[Number],
    hi: &[Number],
    mu: Number,
    max_iter: usize,
) -> Result<(Vec<Number>, CorrectorReport), SolverError> {
    let dim = bs.dim();
    let off = bs.offsets_public();
    let n_x = bs.block_dims()[0];
    let rows = bs
        .bound_rows()
        .ok_or_else(|| SolverError::SensComputationFailed("corrector: no bound rows".into()))?
        .to_vec();

    // The barrier needs every bounded coordinate strictly inside its
    // bound and every multiplier strictly positive. A step that
    // carries one past is put back just inside, since the residual is
    // undefined outside and the fraction rule cannot recover from a
    // point that is already out.
    let mut iterate: Vec<Number> = base.iter().zip(start).map(|(&b, &s)| b + s).collect();
    for b in &rows {
        let i = b.var_row;
        let margin = 1e-10 * (1.0 + base[i].abs());
        if b.lower {
            iterate[i] = iterate[i].max(lo[i] + margin);
        } else {
            iterate[i] = iterate[i].min(hi[i] - margin);
        }
    }
    for z in iterate[off[4]..off[8]].iter_mut() {
        *z = z.max(1e-12);
    }

    // The bounds the step takes out of the active set, decided once
    // here and held for every iteration. Their barrier terms come out
    // of the operator, their multipliers are held at zero, and their
    // complementarity rows leave the system, which is what an inactive
    // bound means. Everything else keeps the base point's barrier term.
    let released = released_rows(&rows, base, &iterate, lo, hi);
    let pinned = pinned_rows(&rows, base, &iterate, lo, hi, mu);
    let sigma = if released.is_empty() && pinned.is_empty() {
        None
    } else {
        for &r in &released {
            iterate[r] = 0.0;
        }
        Some(bs.active_set_sigma_x(&released, &pinned).ok_or_else(|| {
            SolverError::SensComputationFailed("corrector: active-set sigma unavailable".into())
        })?)
    };
    let solve = |rhs: &[Number], lhs: &mut [Number]| -> bool {
        match sigma.as_ref() {
            // The same Rc every call: the factorization cache keys on
            // its tag, so the released operator is factored once and
            // every later solve is a back-solve.
            Some(s) => bs.solve_released_prebuilt(&released, Rc::clone(s), rhs, lhs, false),
            None => bs.solve(rhs, lhs),
        }
    };
    // A released bound has no equation left, so its row does not count
    // toward the residual the stopping rule reads.
    let clear = |v: &mut [Number]| {
        for &r in &released {
            v[r] = 0.0;
        }
    };

    let mut resid = vec![0.0; dim];
    residual_at(bs, &iterate, pins, deltas, mu, &mut resid)?;
    clear(&mut resid);
    let norm = |v: &[Number]| v.iter().fold(0.0_f64, |a, &b| a.max(b.abs()));
    let initial_residual = norm(&resid);

    let mut best = iterate.clone();
    let mut best_residual = initial_residual;
    let mut iterations = 0usize;
    let mut converged = false;

    let mut rhs = vec![0.0; dim];
    let mut dir = vec![0.0; dim];
    while iterations < max_iter {
        for (r, s) in rhs.iter_mut().zip(&resid) {
            *r = -s;
        }
        if !solve(&rhs, &mut dir) {
            return Err(SolverError::BacksolveFailed);
        }
        iterations += 1;

        let (sl, dsl) = slacks_and_directions(&rows, &iterate[..n_x], &dir[..n_x], lo, hi, true);
        let (su, dsu) = slacks_and_directions(&rows, &iterate[..n_x], &dir[..n_x], lo, hi, false);
        let alpha_p = fraction_to_boundary(&sl, &dsl).min(fraction_to_boundary(&su, &dsu));
        let alpha_d = fraction_to_boundary(&iterate[off[4]..off[8]], &dir[off[4]..off[8]]);

        for i in 0..off[4] {
            iterate[i] += alpha_p * dir[i];
        }
        for i in off[4]..off[8] {
            iterate[i] = (iterate[i] + alpha_d * dir[i]).max(1e-14);
        }
        // A released multiplier stays at zero: the bound is out of the
        // active set for the whole correction, not something the
        // iterations decide again each step.
        for &r in &released {
            iterate[r] = 0.0;
        }

        residual_at(bs, &iterate, pins, deltas, mu, &mut resid)?;
        clear(&mut resid);
        let now = norm(&resid);
        if now < best_residual {
            best_residual = now;
            best.copy_from_slice(&iterate);
        } else {
            // An iteration that did not improve on the best residual
            // seen ends the loop, and the best point is what the
            // caller gets.
            converged = true;
            break;
        }
    }

    // the returned point's residual, split by what each block means
    residual_at(bs, &best, pins, deltas, mu, &mut resid)?;
    clear(&mut resid);
    let part = |a: usize, b: usize| norm(&resid[off[a]..off[b]]);
    let (stationarity, feasibility, complementarity) = (part(0, 2), part(2, 4), part(4, 8));

    let step: Vec<Number> = best.iter().zip(base).map(|(&v, &b)| v - b).collect();
    Ok((
        step,
        CorrectorReport {
            iterations,
            residual: best_residual,
            initial_residual,
            converged,
            released: released.len() + pinned.len(),
            stationarity,
            feasibility,
            complementarity,
        },
    ))
}
