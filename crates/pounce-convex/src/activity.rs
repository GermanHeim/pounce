//! Activity classification for the convex arm.
//!
//! What each bound is *doing* at the optimum — holding its coordinate, doing
//! nothing, or vanishing together with its multiplier at a kink where the
//! derivative is two-valued. The decision rule is
//! [`pounce_sens_core::activity_kernel`], the same code the NLP arm runs; this
//! module's job is deriving the three numbers that rule reads.
//!
//! # Deriving `Σ`, `q` and `μ` without a barrier iterate
//!
//! The NLP arm reads them off the filter-IPM's eight-block iterate. The convex
//! arm has no such object — `QpSolution` carries `x`, `y`, `z`, `z_lb`, `z_ub`
//! and nothing else — so they are reconstructed from `(problem, solution)`:
//!
//! | quantity | for a variable `j` | for an inequality row `i` |
//! |---|---|---|
//! | slack | `x_j − lb_j`, `ub_j − x_j` | `s_i = h_i − (Gx)_i` |
//! | `Σ` | `z_lb/s_lb + z_ub/s_ub` | `z_i / s_i` |
//! | `q` | `P_jj` | `∇gᵢᵀ P ∇gᵢ / ‖∇gᵢ‖²` |
//!
//! `μ` is [`QpSensitivity::duality_measure`], the achieved complementarity.
//! That is not quite the barrier parameter the last interior-point iteration ran
//! at — they agree to within the centering ratio at convergence — but the
//! classification bands are decades wide, so it is the right input for this and
//! the wrong input for demanding bit-agreement with the NLP arm's `barrier_mu`.
//!
//! # The row curvature is a directional one, and still not reduced
//!
//! A row's `q` is `|∇gᵀH∇g| / ‖∇g‖²`, with `Σ` weighted by `‖∇g‖²` so the ratio
//! is invariant to a rescaling of the row — a genuine curvature along its own
//! gradient rather than a bare diagonal, which is better than the variable case
//! and still not the same as right. The other free coordinates re-optimize, so a
//! row's ratio is `reduced/directional` and a **coupled row kink reads
//! `AMBIGUOUS` at any tolerance**, exactly as a coupled variable kink does.
//!
//! This is the gh#763 / gh#804 rule, and it holds here for the same reason it
//! holds on the NLP arm: never read the activity class as a proxy for
//! kink-ness. `an_ambiguous_verdict_does_not_mean_no_kink` says so as a test
//! rather than only as prose.

use crate::qp::{BOUND_INF, QpProblem, QpSolution};
use crate::sensitivity::group_rows_by_index;
use pounce_common::types::Number;
use pounce_sens_core::activity_kernel::{
    self as kernel, EQUALITY, Entry, NOT_CLASSIFIED, UNBOUNDED,
};

/// Per-variable and per-row classification of a solved convex QP.
///
/// Field-for-field the shape the NLP arm's report has, so a caller that reads
/// one can read the other. The status codes are
/// [`pounce_sens_core::activity_kernel`]'s.
#[derive(Debug, Clone)]
pub struct ConvexActivityReport {
    /// The achieved complementarity the classification banded against.
    pub mu: Number,
    pub var_status: Vec<i8>,
    pub var_ratio: Vec<Number>,
    pub var_q_sign: Vec<i8>,
    pub var_off_central_path: Vec<bool>,
    pub var_contaminated: Vec<bool>,
    pub var_sigma: Vec<Number>,
    pub row_status: Vec<i8>,
    pub row_ratio: Vec<Number>,
    pub row_q_sign: Vec<i8>,
    pub row_off_central_path: Vec<bool>,
    pub row_contaminated: Vec<bool>,
    pub row_sigma: Vec<Number>,
}

/// Reciprocal-slack guard: a slack at or below this is treated as exactly on
/// the bound, so `z/s` does not become an infinity that classification then has
/// to reason about.
const SLACK_FLOOR: Number = 1e-300;

/// Divide, flooring the denominator so a converged-to-zero slack gives a large
/// finite `Σ` rather than an infinity.
fn ratio(z: Number, s: Number) -> Number {
    z / s.max(SLACK_FLOOR)
}

/// Classify every bounded variable and every inequality row.
///
/// `mu` is the achieved complementarity; `floor` is the noise level below which
/// a curvature is not a curvature — the caller's to derive, which is the seam
/// the kernel is built around.
pub(crate) fn classify_all(
    prob: &QpProblem,
    sol: &QpSolution,
    mu: Number,
    floor: Number,
) -> ConvexActivityReport {
    let n = prob.n;
    let m = prob.m_ineq();

    // --- variables ---
    let mut p_diag = vec![0.0; n];
    for t in &prob.p_lower {
        if t.row == t.col {
            p_diag[t.row] += t.val;
        }
    }
    let mut var = vec![NOT_CLASSIFIED; n];
    for (j, e) in var.iter_mut().enumerate() {
        let (lb, ub) = (prob.lb_of(j), prob.ub_of(j));
        let has_lo = lb > -BOUND_INF;
        let has_hi = ub < BOUND_INF;
        if !has_lo && !has_hi {
            continue; // UNBOUNDED, as initialized
        }
        if has_lo && has_hi && (ub - lb).abs() <= SLACK_FLOOR {
            e.status = kernel::FIXED;
            continue;
        }
        let s_lo = if has_lo {
            sol.x[j] - lb
        } else {
            Number::INFINITY
        };
        let s_hi = if has_hi {
            ub - sol.x[j]
        } else {
            Number::INFINITY
        };
        let mut sigma = 0.0;
        if has_lo {
            sigma += ratio(sol.z_lb[j], s_lo);
        }
        if has_hi {
            sigma += ratio(sol.z_ub[j], s_hi);
        }
        let mut entry = kernel::classify_entry(sigma, p_diag[j], floor, mu);
        entry.off_path = (has_lo && kernel::off_path(s_lo, sol.z_lb[j], mu))
            || (has_hi && kernel::off_path(s_hi, sol.z_ub[j], mu));
        *e = entry;
    }

    // --- inequality rows ---
    let mut gx = vec![0.0; m];
    prob.g_mul(&sol.x, &mut gx);
    let g_rows = group_rows_by_index(&prob.g, m);
    let mut row = vec![NOT_CLASSIFIED; m];
    for (i, e) in row.iter_mut().enumerate() {
        let s = prob.h[i] - gx[i];
        let sigma = ratio(sol.z[i], s);
        // The curvature along this row's own gradient (see below for the
        // weighting that makes the ratio scale-invariant).
        let grad = &g_rows[i];
        let norm2: Number = grad.iter().map(|&(_, v)| v * v).sum();
        if norm2 <= floor {
            *e = kernel::zero_gradient_row(sigma, floor);
            continue;
        }
        let mut quad = 0.0;
        for t in &prob.p_lower {
            let (a, b) = (
                grad.iter().find(|&&(c, _)| c == t.row).map(|&(_, v)| v),
                grad.iter().find(|&&(c, _)| c == t.col).map(|&(_, v)| v),
            );
            if let (Some(a), Some(b)) = (a, b) {
                // The lower triangle stores each off-diagonal once.
                quad += if t.row == t.col {
                    t.val * a * b
                } else {
                    2.0 * t.val * a * b
                };
            }
        }
        // Both halves carry a `‖∇g‖²`, and both are needed. Scaling a row by
        // `c` sends `s → c·s` and `z → z/c`, so `Σ → Σ/c²`, while
        // `‖∇g‖² → c²‖∇g‖²` and `∇gᵀP∇g → c²∇gᵀP∇g`. Weighting `Σ` by `‖∇g‖²`
        // and dividing the quadratic form by it makes each factor invariant, so
        // the ratio does not move with an arbitrary modelling choice. Getting
        // only one of the two right leaves the verdict scaling as `c⁻⁸`, which
        // is what `a_row_classification_is_unmoved_by_rescaling_the_row` caught.
        let q = quad / norm2;
        let mut entry = kernel::classify_entry(sigma * norm2, q, floor, mu);
        // Report the raw `Σ`, not the geometrically weighted one classification
        // used — the same convention the NLP arm follows.
        entry.sigma = sigma;
        entry.off_path = kernel::off_path(s, sol.z[i], mu);
        *e = entry;
    }

    report(mu, &var, &row)
}

fn report(mu: Number, var: &[Entry], row: &[Entry]) -> ConvexActivityReport {
    ConvexActivityReport {
        mu,
        var_status: var.iter().map(|e| e.status).collect(),
        var_ratio: var.iter().map(|e| e.ratio).collect(),
        var_q_sign: var.iter().map(|e| e.q_sign).collect(),
        var_off_central_path: var.iter().map(|e| e.off_path).collect(),
        var_contaminated: var.iter().map(|e| e.contaminated).collect(),
        var_sigma: var.iter().map(|e| e.sigma).collect(),
        row_status: row.iter().map(|e| e.status).collect(),
        row_ratio: row.iter().map(|e| e.ratio).collect(),
        row_q_sign: row.iter().map(|e| e.q_sign).collect(),
        row_off_central_path: row.iter().map(|e| e.off_path).collect(),
        row_contaminated: row.iter().map(|e| e.contaminated).collect(),
        row_sigma: row.iter().map(|e| e.sigma).collect(),
    }
}

/// The noise floor below which a curvature is not a curvature.
///
/// `√ε` relative to the largest diagonal the operator carries, matching the NLP
/// arm's derivation. Equalities are outside this classification and are reported
/// as [`EQUALITY`] by the caller, not floored here.
pub(crate) fn curvature_floor(prob: &QpProblem) -> Number {
    let mut max_abs = 1.0_f64;
    for t in &prob.p_lower {
        if t.row == t.col {
            max_abs = max_abs.max(t.val.abs());
        }
    }
    Number::EPSILON.sqrt() * max_abs
}

/// The status an equality row carries, for a caller assembling a combined view.
pub const fn equality_status() -> i8 {
    EQUALITY
}

/// The status an unbounded coordinate carries.
pub const fn unbounded_status() -> i8 {
    UNBOUNDED
}
