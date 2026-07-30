//! §4.2 parametric homotopy — the qpOASES-lineage path this crate is named for.
//!
//! # Why this exists
//!
//! `ParametricActiveSetSolver` advertises "true parametric warm starting" and
//! cites the qpOASES lineage, but `solve_parametric` was a stub that discarded
//! its arguments and cold-solved. The homotopy was never implemented, and its
//! absence is *structural* rather than a missing convenience:
//!
//! The design note (`docs/src/active-set-sqp-warm-start.md` §4.3) specifies that
//! l1-elastic phase-1 works by driving the elastic slacks to zero **as the
//! homotopy proceeds**. With no homotopy, phase-1 degenerated into solving a
//! standalone, maximally-degenerate QP from cold — the hardest case for an
//! active-set method — which is exactly where it stalls and fails to terminate
//! on the Maros-Mészáros set.
//!
//! # The path
//!
//! §4.2 specifies tracing `(1-t)·QP₀ + t·QP₁` for `t ∈ [0,1]`, jumping the
//! working set wherever a multiplier reaches zero or a constraint reaches its
//! bound. Along a segment with a fixed working set `W` the KKT system
//!
//! ```text
//!   [ H   A_Wᵀ ] [ x ]   [ -g    ]
//!   [ A_W   0  ] [ λ ] = [  b_W(t) ]
//! ```
//!
//! is *affine in `t`*, so `(x(t), λ(t))` moves linearly and the next event is
//! found by two ratio tests in `t` rather than by a line search in step space.
//!
//! # Choosing `QP₀` (the part the textbook version gets wrong here)
//!
//! The canonical cold start takes `W₀ = ∅`, which makes `x(t) = −H⁻¹g(t)` and
//! therefore **requires `H` nonsingular**. Most of the Maros-Mészáros set is
//! LP-like with singular or zero `H`, where an empty working set leaves a
//! null-space direction and the KKT is singular. Repairing that needs `n` active
//! constraints — i.e. a vertex — which is the phase-1 problem again. Circular.
//!
//! This module sidesteps it: `QP₀` is the **box-only relaxation** — the target
//! QP with every general row dropped. That is solvable on an existing, tested
//! fast path ([`super::ParametricActiveSetSolver::solve_box_constrained`]), and
//! its solution comes with a working set of active bounds that makes the reduced
//! Hessian nonsingular whenever the box does. Only the **row bounds** are then
//! homotopied in, from a relaxation that `x₀` strictly satisfies to the target;
//! `H` and `g` are held fixed, so the `-g` block above never moves and the
//! direction solve has a zero primal right-hand side.
//!
//! The consequence worth stating plainly: there is **no phase-1**. `x(t)` is
//! feasible for the `t`-problem at every point on the path by construction, so
//! feasibility is never searched for and cannot stall.
//!
//! # References
//!
//! - Ferreau, Kirches, Potschka, Bock, Diehl, "qpOASES: a parametric active-set
//!   algorithm for quadratic programming", *Math. Prog. Comp.* **6** (2014) —
//!   the dense reference algorithm and the homotopy's ratio tests.
//! - Kirches, *Fast Numerical Methods for Mixed-Integer Nonlinear
//!   Model-Predictive Control* (2011), Ch. 5–7 — the sparse Schur extension.

use crate::error::{QpError, QpStatus};
use crate::kkt::{a_times_x, assemble_active_set_kkt};
use crate::options::QpOptions;
use crate::problem::{QpProblem, QpSolution, QpStats};
use crate::solver::ParametricActiveSetSolver;
use crate::solver::QpSolver as _;
use crate::working_set::{BoundStatus, ConsStatus, WorkingSet};
use pounce_common::Number;
use pounce_common::types::{NLP_LOWER_BOUND_INF, NLP_UPPER_BOUND_INF};
use pounce_linalg::triplet::{GenTMatrix, GenTMatrixSpace, SymTMatrix, SymTMatrixSpace};
use std::time::Instant;

/// How far the relaxed `t = 0` row bounds sit outside `A x₀`, relative to the
/// row's own scale. Strictly positive so every row starts *inactive* with real
/// slack — a row that started exactly at its bound would be degenerate at `t=0`
/// and the first ratio test would see a zero-length step.
const RELAX_MARGIN: Number = 1.0;

/// `t` is clamped into `[0, 1]`; an event closer than this to the current `t` is
/// treated as coincident (a degenerate tie) rather than as forward progress, so
/// the loop cannot spin on a zero-length advance.
const T_EPS: Number = 1e-12;

/// Outcome of the two ratio tests: what happens first as `t` increases.
#[derive(Debug, Clone, Copy)]
enum Event {
    /// Inactive row `i` reaches its lower bound.
    AddRowLower(usize),
    /// Inactive row `i` reaches its upper bound.
    AddRowUpper(usize),
    /// Active row `i`'s multiplier reaches zero and must leave.
    DropRow(usize),
    /// Active bound on variable `j` has its multiplier reach zero.
    DropBound(usize),
}

/// Primal regularization `δ` for the path, derived from the problem's own scale.
///
/// Needed because the path starts from the box relaxation, and that relaxation is
/// **unbounded** whenever `H` has no curvature in a box-unbounded direction —
/// which is most LP-like instances (`QAFIRO` returns `obj = -inf`). Running the
/// path on `H + δI` bounds it.
///
/// This is sound *specifically because the path is only a predictor*: the working
/// set it discovers is handed to a corrector that solves the true QP, so `δ`
/// never enters the reported answer, only the prediction of which constraints
/// end up active.
///
/// `δ` is derived, not guessed. With `H = 0` the box relaxation's solution is
/// `x₀ = clamp(−g/δ, box)`, so `δ = ‖g‖∞ / X` places `‖x₀‖` at roughly `X`, a
/// representative variable magnitude. `X` is the median finite box width, or 1
/// when no variable has two finite bounds. Putting `x₀` on the box's own scale
/// matters because the `t = 0` row bounds are relaxed outward from `A x₀`: an
/// enormous `x₀` would make them enormous too, and the path correspondingly long.
///
/// Returns `None` when `g` is zero — there is nothing to scale against, and with
/// `H = 0` and `g = 0` every feasible point is optimal anyway.
fn path_regularization_delta(qp: &QpProblem<'_>) -> Option<Number> {
    let g_inf = qp.g.iter().fold(0.0_f64, |a, v| a.max(v.abs()));
    if !(g_inf > 0.0) || !g_inf.is_finite() {
        return None;
    }
    let mut widths: Vec<Number> = (0..qp.n)
        .filter_map(|i| {
            let (l, u) = (qp.xl[i], qp.xu[i]);
            (l > NLP_LOWER_BOUND_INF && u < NLP_UPPER_BOUND_INF).then(|| (u - l).abs())
        })
        .filter(|w| w.is_finite() && *w > 0.0)
        .collect();
    let x_scale = if widths.is_empty() {
        1.0
    } else {
        widths.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        widths[widths.len() / 2]
    };
    let delta = g_inf / x_scale.max(1e-12);
    delta.is_finite().then_some(delta.clamp(1e-12, 1e12))
}

/// `H + δI`, as a fresh symmetric triplet matrix.
///
/// `H` is stored lower-triangle 1-based with each pair listed once, so `δ` is
/// added to an existing diagonal entry where one is present and appended
/// otherwise — appending unconditionally would double-count a diagonal that `H`
/// already carries.
fn regularized_hessian(qp: &QpProblem<'_>, delta: Number) -> SymTMatrix {
    let (irows, jcols, vals) = (qp.h.irows(), qp.h.jcols(), qp.h.values());
    let mut ir: Vec<i32> = irows.to_vec();
    let mut jc: Vec<i32> = jcols.to_vec();
    let mut vl: Vec<Number> = vals.to_vec();

    let mut has_diag = vec![false; qp.n];
    for k in 0..ir.len() {
        if ir[k] == jc[k] {
            let idx = (ir[k] - 1) as usize;
            if idx < qp.n {
                vl[k] += delta;
                has_diag[idx] = true;
            }
        }
    }
    for (i, seen) in has_diag.iter().enumerate() {
        if !seen {
            ir.push((i + 1) as i32);
            jc.push((i + 1) as i32);
            vl.push(delta);
        }
    }

    let space = SymTMatrixSpace::new(qp.n as i32, ir, jc);
    let mut h = SymTMatrix::new(space);
    h.set_values(&vl);
    h
}

/// Rate of change of a row bound per unit `t` (0 when the bound is infinite).
fn bound_rate(relaxed: Number, target: Number, is_lower: bool) -> Number {
    let infinite = if is_lower {
        target <= NLP_LOWER_BOUND_INF
    } else {
        target >= NLP_UPPER_BOUND_INF
    };
    if infinite { 0.0 } else { target - relaxed }
}

impl ParametricActiveSetSolver {
    /// Solve `qp` from cold by tracing the row-bound homotopy described in this
    /// module's docs.
    ///
    /// Returns `Ok(None)` when the path cannot be started (the box relaxation is
    /// unbounded or fails), which is a signal for the caller to fall back to the
    /// conventional cold path — not a verdict about `qp`.
    pub(crate) fn solve_cold_homotopy(
        &mut self,
        qp: &QpProblem<'_>,
        opts: &QpOptions,
    ) -> Result<Option<QpSolution>, QpError> {
        let started = Instant::now();
        let n = qp.n;
        let m = qp.m;
        if m == 0 {
            // No rows to bring in: the box relaxation *is* the problem, and the
            // existing fast path already handles it.
            return Ok(None);
        }

        // ---- QP₀: the box-only relaxation ----
        //
        // `m = 0` demands a genuinely 0-row Jacobian: `QpProblem::validate`
        // cross-checks `A`'s row count against `m`, so handing it the target's
        // `A` with `m = 0` is rejected outright.
        let empty_a = GenTMatrix::new(GenTMatrixSpace::new(0, n as i32, Vec::new(), Vec::new()));
        let trace = std::env::var("POUNCE_HOMOTOPY_DEBUG").is_ok();

        // Built inline rather than by a closure: a closure returning
        // `QpProblem<'_>` cannot tie the borrow of `h` to the returned value's
        // lifetime, so it fails to compile.
        macro_rules! box_qp {
            ($h:expr) => {
                QpProblem {
                    n,
                    m: 0,
                    h: $h,
                    g: qp.g,
                    a: &empty_a,
                    bl: &[],
                    bu: &[],
                    xl: qp.xl,
                    xu: qp.xu,
                    hessian_inertia: qp.hessian_inertia,
                }
            };
        }

        // Try the true `H` first, and regularize only if that fails. An
        // unbounded box relaxation means `H` has no curvature in a
        // box-unbounded direction; the *target* may still be bounded (a row
        // constraint can cut the ray off), so it is not an unboundedness verdict
        // — just a statement that the path cannot start from here.
        //
        // Regularizing only on failure keeps the problems that already work
        // bit-identical: `HS21` and `QPTEST` have positive-definite `H` and never
        // reach the retry.
        let mut h_reg_holder: Option<SymTMatrix> = None;
        let box_sol = {
            let first = self.solve(&box_qp!(qp.h), None, opts);
            if trace {
                match &first {
                    Ok(s) => eprintln!("[hom] box relaxation: {:?} obj={:.6e}", s.status, s.obj),
                    Err(e) => eprintln!("[hom] box relaxation ERROR: {e}"),
                }
            }
            match first {
                Ok(s) if s.status == QpStatus::Optimal => s,
                _ => {
                    let Some(delta) = path_regularization_delta(qp) else {
                        return Ok(None);
                    };
                    let h_reg = regularized_hessian(qp, delta);
                    let retry = self.solve(&box_qp!(&h_reg), None, opts);
                    if trace {
                        match &retry {
                            Ok(s) => eprintln!(
                                "[hom] box relaxation (delta={delta:.3e}): {:?} obj={:.6e}",
                                s.status, s.obj
                            ),
                            Err(e) => eprintln!("[hom] box relaxation (regularized) ERROR: {e}"),
                        }
                    }
                    match retry {
                        Ok(s) if s.status == QpStatus::Optimal => {
                            h_reg_holder = Some(h_reg);
                            s
                        }
                        _ => return Ok(None),
                    }
                }
            }
        };

        // Everything on the path is traced against this Hessian; the corrector at
        // the end uses the caller's `qp` and therefore the true `H`.
        let path_h: &SymTMatrix = h_reg_holder.as_ref().unwrap_or(qp.h);
        let path_qp = QpProblem {
            n,
            m,
            h: path_h,
            g: qp.g,
            a: qp.a,
            bl: qp.bl,
            bu: qp.bu,
            xl: qp.xl,
            xu: qp.xu,
            hessian_inertia: qp.hessian_inertia,
        };

        let mut x = box_sol.x.clone();
        let mut working = WorkingSet::cold(n, m);
        for (i, st) in working.bounds.iter_mut().enumerate() {
            *st = box_sol.working.bounds[i];
        }
        // Every row starts inactive, with genuine slack.
        let ax0 = a_times_x(qp.a, &x, m);
        let mut bl0 = vec![0.0; m];
        let mut bu0 = vec![0.0; m];
        for i in 0..m {
            let scale = RELAX_MARGIN * (1.0 + ax0[i].abs());
            // Relax outward from `A x₀` far enough that `x₀` is strictly
            // interior; where the target is already looser, keep the target.
            bl0[i] = if qp.bl[i] <= NLP_LOWER_BOUND_INF {
                qp.bl[i]
            } else {
                (ax0[i] - scale).min(qp.bl[i])
            };
            bu0[i] = if qp.bu[i] >= NLP_UPPER_BOUND_INF {
                qp.bu[i]
            } else {
                (ax0[i] + scale).max(qp.bu[i])
            };
        }

        let mut lambda_g = vec![0.0; m];
        let mut lambda_x = box_sol.lambda_x.clone();
        lambda_x.resize(n, 0.0);

        let mut t: Number = 0.0;
        let mut n_changes: u32 = 0;
        let mut n_refactor: u32 = 0;

        // Each iteration either advances `t` or changes the working set, and the
        // budget bounds the total.
        for _step in 0..opts.max_iter {
            if t >= 1.0 - T_EPS {
                break;
            }
            if trace && _step % 50 == 0 {
                eprintln!("[hom] step={_step} t={t:.6e}");
            }

            let active_cons: Vec<usize> = (0..m)
                .filter(|&i| working.constraints[i].is_active())
                .collect();
            let active_bounds: Vec<usize> =
                (0..n).filter(|&i| working.bounds[i].is_active()).collect();
            let (k_c, k_b) = (active_cons.len(), active_bounds.len());

            // ---- Direction: d/dt of (x, λ) along this segment ----
            //
            // `H` and `g` are fixed, so the stationarity block's right-hand side
            // does not move and the primal RHS is zero; only the active rows'
            // bounds advance, at `bound_rate`.
            let kkt = assemble_active_set_kkt(&path_qp, &active_cons, &active_bounds);
            let mut rhs = vec![0.0; n + k_c + k_b];
            for (slot, &i) in active_cons.iter().enumerate() {
                let is_lower = matches!(working.constraints[i], ConsStatus::AtLower);
                rhs[n + slot] = match working.constraints[i] {
                    ConsStatus::Equality => bound_rate(bu0[i], qp.bu[i], false),
                    _ => bound_rate(
                        if is_lower { bl0[i] } else { bu0[i] },
                        if is_lower { qp.bl[i] } else { qp.bu[i] },
                        is_lower,
                    ),
                };
            }
            // Variable bounds do not move along this path, so their rows are 0.
            match self.factorize_with_inertia_control(kkt, &mut rhs, (k_c + k_b) as i32, n, opts) {
                Ok(_) => {}
                // A rank-deficient active set on the path is the same situation
                // `solve_general` handles by pruning; rather than duplicate that
                // logic here, hand the problem back to the conventional path.
                Err(_) => return Ok(None),
            }
            n_refactor += 1;
            let dx: Vec<Number> = rhs[..n].to_vec();
            let dlam_c: Vec<Number> = (0..k_c).map(|s| rhs[n + s]).collect();
            let dlam_b: Vec<Number> = (0..k_b).map(|s| rhs[n + k_c + s]).collect();

            // ---- Ratio test 1 (primal): when does an inactive row bind? ----
            let a_dx = a_times_x(qp.a, &dx, m);
            let ax = a_times_x(qp.a, &x, m);
            let mut t_next: Number = 1.0;
            let mut event: Option<Event> = None;

            for i in 0..m {
                if working.constraints[i].is_active() {
                    continue;
                }
                // Upper: a_i·x(t) − bu_i(t) = 0.
                if qp.bu[i] < NLP_UPPER_BOUND_INF {
                    let gap = bu0[i] + t * (qp.bu[i] - bu0[i]) - ax[i];
                    let rate = a_dx[i] - bound_rate(bu0[i], qp.bu[i], false);
                    if rate > 0.0 {
                        let dt = gap / rate;
                        if dt >= -T_EPS && t + dt < t_next - T_EPS {
                            t_next = (t + dt).clamp(t, 1.0);
                            event = Some(Event::AddRowUpper(i));
                        }
                    }
                }
                // Lower: bl_i(t) − a_i·x(t) = 0.
                if qp.bl[i] > NLP_LOWER_BOUND_INF {
                    let gap = ax[i] - (bl0[i] + t * (qp.bl[i] - bl0[i]));
                    let rate = bound_rate(bl0[i], qp.bl[i], true) - a_dx[i];
                    if rate > 0.0 {
                        let dt = gap / rate;
                        if dt >= -T_EPS && t + dt < t_next - T_EPS {
                            t_next = (t + dt).clamp(t, 1.0);
                            event = Some(Event::AddRowLower(i));
                        }
                    }
                }
            }

            // ---- Ratio test 2 (dual): when does an active multiplier vanish? ----
            //
            // An inequality's multiplier must keep its sign; reaching zero means
            // the row stops binding and has to leave the working set. Equality
            // rows are exempt — their multipliers are unrestricted.
            for (slot, &i) in active_cons.iter().enumerate() {
                if matches!(working.constraints[i], ConsStatus::Equality) {
                    continue;
                }
                let lam = lambda_g[i];
                let rate = dlam_c[slot];
                // Sign convention: `AtUpper` multipliers are ≥ 0, `AtLower` ≤ 0
                // in this engine's packing (see `solve_general`'s drop test).
                let heading_to_zero = (lam > 0.0 && rate < 0.0) || (lam < 0.0 && rate > 0.0);
                if heading_to_zero {
                    let dt = -lam / rate;
                    if dt >= -T_EPS && t + dt < t_next - T_EPS {
                        t_next = (t + dt).clamp(t, 1.0);
                        event = Some(Event::DropRow(i));
                    }
                }
            }
            for (slot, &j) in active_bounds.iter().enumerate() {
                if matches!(working.bounds[j], BoundStatus::Fixed) {
                    continue;
                }
                let lam = lambda_x[j];
                let rate = dlam_b[slot];
                let heading_to_zero = (lam > 0.0 && rate < 0.0) || (lam < 0.0 && rate > 0.0);
                if heading_to_zero {
                    let dt = -lam / rate;
                    if dt >= -T_EPS && t + dt < t_next - T_EPS {
                        t_next = (t + dt).clamp(t, 1.0);
                        event = Some(Event::DropBound(j));
                    }
                }
            }

            // ---- Advance to the event (or to t = 1) ----
            let dt = t_next - t;
            for (xi, &d) in x.iter_mut().zip(dx.iter()) {
                *xi += dt * d;
            }
            for (slot, &i) in active_cons.iter().enumerate() {
                lambda_g[i] += dt * dlam_c[slot];
            }
            for (slot, &j) in active_bounds.iter().enumerate() {
                lambda_x[j] += dt * dlam_b[slot];
            }
            t = t_next;

            match event {
                None => {
                    // Nothing binds before t = 1: the path is complete.
                    t = 1.0;
                    break;
                }
                Some(Event::AddRowUpper(i)) => {
                    working.constraints[i] = ConsStatus::AtUpper;
                    n_changes += 1;
                }
                Some(Event::AddRowLower(i)) => {
                    working.constraints[i] = ConsStatus::AtLower;
                    n_changes += 1;
                }
                Some(Event::DropRow(i)) => {
                    working.constraints[i] = ConsStatus::Inactive;
                    lambda_g[i] = 0.0;
                    n_changes += 1;
                }
                Some(Event::DropBound(j)) => {
                    working.bounds[j] = BoundStatus::Inactive;
                    lambda_x[j] = 0.0;
                    n_changes += 1;
                }
            }
        }

        // The path is only a *predictor* for the final active set: `t` may have
        // stopped short of 1, and the linear algebra along the way accumulates
        // error. Hand the discovered working set to the conventional solver,
        // which corrects the iterate and applies the usual feasibility audit and
        // status logic. That keeps every existing guarantee — nothing here is
        // reported as optimal on the homotopy's own authority.
        if t < 1.0 - T_EPS {
            if trace {
                eprintln!("[hom] path did NOT reach t=1 (stopped at {t:.6e}); falling back");
            }
            return Ok(None);
        }
        if trace {
            eprintln!("[hom] reached t=1 after {n_changes} working-set changes");
        }
        let mut sol =
            <Self as crate::solver::QpSolver>::solve_with_working_set(self, qp, &working, opts)?;
        sol.stats = QpStats {
            n_working_set_changes: sol.stats.n_working_set_changes + n_changes,
            n_refactor: sol.stats.n_refactor + n_refactor,
            n_schur_updates: sol.stats.n_schur_updates,
            used_phase1: false,
            time: started.elapsed(),
        };
        Ok(Some(sol))
    }
}
