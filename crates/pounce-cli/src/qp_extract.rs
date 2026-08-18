//! Extract a `pounce_convex::QpProblem` (standard form) from a parsed
//! `.nl` problem, for the LP/QP dispatch path (Phase 2).
//!
//! The classifier (`crate::dispatch`) has already decided the problem is
//! an LP or convex QP; this module marshals the parsed `NlProblem` into
//! the standard form the convex IPM consumes:
//!
//! ```text
//! minimize    ½ xᵀP x + cᵀx
//! subject to  A x = b          (equalities)
//!             G x ≤ h          (inequalities)
//!             lb ≤ x ≤ ub      (the variable box)
//! ```
//!
//! Mapping from the `.nl` representation:
//! - **Objective.** `P` is the Hessian of the (degree-≤2) objective —
//!   recovered with the same `analyze_quadratic` the classifier uses, so
//!   `P` here is exactly the matrix whose definiteness was tested. `c`
//!   is the objective's linear part. A `maximize` objective is negated
//!   into a minimization.
//! - **Constraints.** Each row has a linear part and bounds `g_l ≤ row ≤
//!   g_u`. An equality (`g_l == g_u`) becomes a row of `A`; a one- or
//!   two-sided inequality becomes one or two rows of `G` (`row ≤ g_u`
//!   and/or `−row ≤ −g_l`).
//! - **Variable bounds.** Present `x_l`/`x_u` become the solver's explicit
//!   box (see [`extract_box`] for why they are no longer emitted as `G`
//!   rows). The `.nl` "infinity" sentinel is read directionally: `x_l ≤
//!   -1e19` is no lower bound, `x_u ≥ 1e19` is no upper bound. A bound past
//!   the *opposite* sentinel (an upper bound of `-5e20`) is an ordinary
//!   bound and is kept.

use crate::dispatch::analyze_quadratic_full;
use crate::nl_reader::NlProblem;
// Bound presence is read **directionally** — a lower bound is absent only at
// or below `-1e19`, an upper bound only at or above `+1e19`. This file used a
// symmetric `|v| < 1e19` test (gh #401): a real upper bound of `-5e20` failed
// it and was dropped from `G` entirely, so the QP was solved over a strictly
// larger box and reported `Optimal` at a point the model excludes.
use pounce_common::types::{lower_bound_present, upper_bound_present};
use pounce_convex::{ConeSpec, QpProblem, QpSolution, Triplet};

/// Convert a classified LP/convex-QP `NlProblem` into `QpProblem`
/// standard form. Returns `None` if the objective is not actually a
/// degree-≤2 polynomial (should not happen for a problem the classifier
/// routed here, but the conversion is total and falls back gracefully).
pub fn extract_qp(prob: &NlProblem) -> Option<QpProblem> {
    Some(extract_qp_with_map(prob)?.0) // drops con_map + reporting constant
}

/// Where each `.nl` constraint's rows landed in the standard-form QP, so
/// the QP's multipliers can be mapped back to a per-`.nl`-constraint
/// dual for the `.sol`. One entry per original constraint, in order.
#[derive(Debug, Clone)]
pub enum ConRowMap {
    /// Equality constraint → row `a_row` of `A` (multiplier `y[a_row]`).
    Eq { a_row: usize },
    /// Inequality / range constraint → up to two rows of `G`: the
    /// `row ≤ g_u` upper bound and/or the `−row ≤ −g_l` lower bound
    /// (multipliers `z[..]`, each ≥ 0).
    Ineq {
        upper: Option<usize>,
        lower: Option<usize>,
    },
}

/// Extract the QP, the constraint→row provenance map, and the objective
/// constant folded into the nonlinear tree (see below), together.
///
/// The third return value is the **degree-0 term of the nonlinear
/// objective** (e.g. the `+9` of `(x₀−3)²` that AMPL/Pyomo emit inside the
/// nonlinear tree rather than in `NlProblem::obj_constant`). The QP itself
/// ignores it — it does not move the minimizer — but the caller must add
/// it to the *reported* objective so the convex solve agrees with the NLP
/// path. It is returned in the problem's natural (user) sense, *not*
/// multiplied by the maximize/minimize `sign`.
pub fn extract_qp_with_map(prob: &NlProblem) -> Option<(QpProblem, Vec<ConRowMap>, f64)> {
    let n = prob.n;
    let sign = if prob.minimize { 1.0 } else { -1.0 };

    // --- objective Hessian P (lower triangle) + nonlinear-tree linear part
    //     + nonlinear-tree constant (degree-0 term, for reporting only) ---
    let (hess, obj_nl_linear, obj_nl_constant) = analyze_quadratic_full(&prob.obj_nonlinear, n)?;
    let mut p_lower: Vec<Triplet> = Vec::with_capacity(hess.len());
    for ((i, j), v) in &hess {
        // analyze_quadratic returns (i ≤ j) upper-ish keys; store as
        // lower triangle (row ≥ col) for the solver.
        let (row, col) = if i >= j { (*i, *j) } else { (*j, *i) };
        p_lower.push(Triplet::new(row, col, sign * v));
    }

    // --- objective linear term c ---
    // Two disjoint sources, exactly as the NLP path's eval_f sums them:
    // the `.nl` linear section (`obj_linear`) and the degree-1 terms AMPL
    // kept inside the nonlinear objective tree (e.g. the `−6·x₀` of
    // `(x₀−3)²`). Dropping the latter silently solves the wrong objective.
    let mut c = vec![0.0; n];
    for (var, coef) in &prob.obj_linear {
        c[*var] += sign * coef;
    }
    for (var, coef) in &obj_nl_linear {
        c[*var] += sign * coef;
    }

    // --- constraints: equalities → A x = b, inequalities → G x ≤ h ---
    let mut a: Vec<Triplet> = Vec::new();
    let mut b: Vec<f64> = Vec::new();
    let mut g: Vec<Triplet> = Vec::new();
    let mut h: Vec<f64> = Vec::new();
    let mut con_map: Vec<ConRowMap> = Vec::with_capacity(prob.con_linear.len());

    for (row, lin) in prob.con_linear.iter().enumerate() {
        let lo = prob.g_l[row];
        let hi = prob.g_u[row];

        // Combine the `.nl` linear section with any degree-≤1 terms AMPL
        // folded into the (here empty-Hessian) nonlinear tree — the
        // classifier admits constraint rows whose nonlinear expression
        // reduces to degree ≤ 1 (`dispatch.rs`), e.g. cancelled
        // quadratics or defined variables, and those linear/constant
        // terms live in `con_nonlinear`, not `con_linear`. Dropping them
        // silently solves the wrong constraint. The folded constant
        // shifts the bounds: `g_l ≤ row + k ≤ g_u  ⇔  g_l−k ≤ row ≤ g_u−k`.
        // This mirrors the SOCP extractor's linear-constraint handling.
        let (nl_lin, const_shift) = analyze_quadratic_full(&prob.con_nonlinear[row], n)
            .map(|(_, l, k)| (l, k))
            .unwrap_or_default();
        let mut coef = vec![0.0; n];
        for (var, v) in lin {
            coef[*var] += *v;
        }
        for (var, v) in &nl_lin {
            coef[*var] += *v;
        }
        let nonzeros = || coef.iter().enumerate().filter(|(_, v)| **v != 0.0);

        if lo == hi && lower_bound_present(lo) && upper_bound_present(hi) {
            // Equality row.
            let eq_row = next_row(&b);
            for (var, v) in nonzeros() {
                a.push(Triplet::new(eq_row, var, *v));
            }
            b.push(lo - const_shift);
            con_map.push(ConRowMap::Eq { a_row: eq_row });
        } else {
            // Upper bound: row ≤ hi.
            let upper = if upper_bound_present(hi) {
                let gr = next_row(&h);
                for (var, v) in nonzeros() {
                    g.push(Triplet::new(gr, var, *v));
                }
                h.push(hi - const_shift);
                Some(gr)
            } else {
                None
            };
            // Lower bound: row ≥ lo  ⇔  −row ≤ −lo.
            let lower = if lower_bound_present(lo) {
                let gr = next_row(&h);
                for (var, v) in nonzeros() {
                    g.push(Triplet::new(gr, var, -*v));
                }
                h.push(-(lo - const_shift));
                Some(gr)
            } else {
                None
            };
            con_map.push(ConRowMap::Ineq { upper, lower });
        }
    }

    // --- variable bounds as the explicit box (not as `G` rows) ---
    let (lb, ub) = extract_box(prob);

    Some((
        QpProblem {
            n,
            p_lower,
            c,
            a,
            b,
            g,
            h,
            lb,
            ub,
        },
        con_map,
        obj_nl_constant,
    ))
}

/// The `.nl` variable bounds as `pounce-convex`'s explicit box, with an
/// absent bound spelled `∓∞`.
///
/// Both extractors used to emit each finite bound as a `G` row (`x_i ≤ x_u`,
/// `−x_i ≤ −x_l`) and leave `lb`/`ub` empty. That was never wrong — the IPM
/// re-expands finite bounds into exactly those rows internally — but it threw
/// away the one thing the solvers can only get from the box: **that these
/// rows are a box**. Three consequences, all real:
///
/// * The empty-box screen ([`pounce_convex`]'s `screen_variable_box`, gh #491)
///   reads `lb`/`ub`, so a model with a reversed bound arrived as a pair of
///   contradictory rows instead — an infeasibility that has to be *certified*
///   numerically rather than seen. The interior-point method managed that at
///   most widths but returned `NumericalFailure` at a `NaN` iterate for
///   crossings around `1e-8`.
/// * The active-set engine handles a box with bound *statuses*, not with
///   constraint rows; feeding it `2n` extra rows made every `.nl` QP that much
///   larger in the one dimension an active-set method pays combinatorially
///   for.
/// * Presolve reasons about `tlb`/`tub` directly, so bounds hidden in rows had
///   to be rediscovered by activity-based tightening before any box reduction
///   could fire.
///
/// The bound *multipliers* now come back in the solution's `z_lb`/`z_ub`
/// rather than being decoded out of `z` by row position.
fn extract_box(prob: &NlProblem) -> (Vec<f64>, Vec<f64>) {
    let lb = (0..prob.n)
        .map(|i| {
            let v = prob.x_l[i];
            if lower_bound_present(v) {
                v
            } else {
                f64::NEG_INFINITY
            }
        })
        .collect();
    let ub = (0..prob.n)
        .map(|i| {
            let v = prob.x_u[i];
            if upper_bound_present(v) {
                v
            } else {
                f64::INFINITY
            }
        })
        .collect();
    (lb, ub)
}

/// Map the QP solver's multipliers `(y, z)` back to a per-`.nl`-
/// constraint dual vector (length `prob.m`), in the AMPL `.sol`
/// convention used by POUNCE's NLP path.
///
/// The QP solver enforces stationarity `∇f + Aᵀy + Gᵀz = 0` with
/// `z ≥ 0`, where each inequality `.nl` row contributes a `row ≤ g_u`
/// (`+row`) and/or `−row ≤ −g_l` (`−row`) `G` row. The per-constraint
/// `.nl`/Ipopt multiplier `λ` is recovered as:
/// - equality: `λ = sign · y[a_row]`;
/// - inequality: `λ = sign · (z_upper − z_lower)` — at most one of the
///   two bound rows is active at a solution.
///
/// The inequality sign (`z_upper − z_lower`, *not* `z_lower − z_upper`)
/// is fixed to match POUNCE's NLP path, which is the reference for what
/// a POUNCE `.sol` carries; this is verified empirically against the NLP
/// solve in the crate tests. `sign` undoes the maximize→minimize
/// negation so the reported dual is in the user's original sense.
pub fn recover_duals(prob: &NlProblem, con_map: &[ConRowMap], y: &[f64], z: &[f64]) -> Vec<f64> {
    let sign = if prob.minimize { 1.0 } else { -1.0 };
    con_map
        .iter()
        .map(|m| match m {
            ConRowMap::Eq { a_row } => sign * y[*a_row],
            ConRowMap::Ineq { upper, lower } => {
                let zu = upper.map(|r| z[r]).unwrap_or(0.0);
                let zl = lower.map(|r| z[r]).unwrap_or(0.0);
                sign * (zu - zl)
            }
        })
        .collect()
}

/// The next 0-based row index for a constraint block keyed by its RHS
/// vector's current length.
fn next_row(rhs: &[f64]) -> usize {
    rhs.len()
}

/// Recover the per-variable **bound multipliers** from a solved QP or SOCP.
///
/// Both extractors put the `.nl` variable bounds in the explicit box
/// ([`extract_box`]), so the solver returns their multipliers directly in
/// `z_lb`/`z_ub` and this is a length-normalizing read rather than the
/// row-position decode it used to be. Variables are 1:1 with the `.nl`
/// variables in both extractors, so no index remap is needed; a slot without
/// a finite bound stays `0.0` because no bound was active there.
///
/// The returned `z_lb` / `z_ub` are the raw non-negative multipliers of the
/// *internal minimize* problem (a maximize objective was negated during
/// extraction); the caller applies the maximize `sign` and the Ipopt
/// `ipopt_zL_out = +z_l`, `ipopt_zU_out = −z_u` output convention.
pub fn recover_bound_mults(prob: &NlProblem, sol: &QpSolution) -> (Vec<f64>, Vec<f64>) {
    let read = |v: &[f64]| -> Vec<f64> {
        (0..prob.n)
            .map(|i| v.get(i).copied().unwrap_or(0.0))
            .collect()
    };
    (read(&sol.z_lb), read(&sol.z_ub))
}

// ===========================================================================
// QCQP → SOCP extraction
// ===========================================================================

/// Where each `.nl` constraint landed in the standard-form **conic** program,
/// so the cone multipliers can be mapped back to a per-`.nl`-constraint dual.
/// One entry per original constraint, in order. (Analogue of [`ConRowMap`] for
/// the SOCP path produced by [`extract_socp_with_map`].)
#[derive(Debug, Clone)]
pub enum ConSocpMap {
    /// Linear equality → row `a_row` of `A` (multiplier `y[a_row]`).
    Eq { a_row: usize },
    /// Linear inequality / range → up to two rows of the nonnegative `G`
    /// block (`row ≤ g_u` and/or `−row ≤ −g_l`), multipliers `z[..] ≥ 0`.
    Ineq {
        upper: Option<usize>,
        lower: Option<usize>,
    },
    /// Convex quadratic inequality `g(x) ≤ g_u`, reformulated to one
    /// second-order cone. The first two cone rows both carry the linear
    /// coefficient vector `a = ∇(linear part)`, so the original constraint
    /// multiplier is recovered as `z[r0] + z[r1]` (see
    /// [`recover_socp_duals`]).
    Quad { z_row0: usize, z_row1: usize },
}

/// A deferred second-order-cone block, built after the nonnegative `G` rows
/// are known so the cones partition `G` in row order (nonneg block first,
/// then the SOCs).
/// Everything here is sized by the row's own **support** `k` — the variables
/// that actually appear in it — never by the problem width `n`. A QCQP row
/// typically touches a handful of variables out of `n` in the hundreds of
/// thousands (`nql180`: `k = 2`, `n = 129 601`), so an `n`-sized structure per
/// row is the difference between kilobytes and tens of gigabytes.
struct SocBlock {
    /// Index in `con_map` of the originating constraint, to patch with the
    /// final cone-row indices once they are assigned.
    con_idx: usize,
    /// Linear coefficients of the constraint as `(variable, coefficient)`,
    /// ascending by variable and with zeros dropped.
    a: Vec<(usize, f64)>,
    /// `b_eff = (nonlinear constant) − g_u`, the constraint's degree-0 term
    /// after moving the upper bound to the right: `½xᵀQx + aᵀx + b_eff ≤ 0`.
    b_eff: f64,
    /// Rows of the factor `F` with `FᵀF = Q`, each a sparse
    /// `(variable, coefficient)` list in the problem's own indexing.
    ///
    /// Sparse rather than length-`n` (or even length-`k`) dense: a diagonal
    /// `Q` — the `qssp180`/`nql180` regime — has rank `k` and one nonzero per
    /// factor row, so a dense factor would be `k²` to hold a `k`-nonzero
    /// object.
    f_rows: Vec<Vec<(usize, f64)>>,
}

/// Convert a classified **convex QCQP** `NlProblem` into the conic standard
/// form the SOCP IPM consumes:
///
/// ```text
/// minimize    ½ xᵀP x + cᵀx
/// subject to  A x = b
///             h − G x  ∈  K        (K = nonneg orthant × second-order cones)
/// ```
///
/// Returns `(QpProblem, con_map, obj_nl_constant, cones)`:
/// - the objective `P`/`c` exactly as the LP/QP path builds them;
/// - linear equalities in `A`/`b`; linear inequalities and finite variable
///   bounds as a leading **nonnegative** `G` block; and each convex quadratic
///   inequality `g(x) ≤ g_u` as one **second-order cone** block appended
///   after it (so `cones` covers the `G` rows in order);
/// - `con_map` mapping each original constraint to its rows for dual recovery;
/// - `obj_nl_constant`, the objective's folded degree-0 term (added back to the
///   reported value, exactly as in [`extract_qp_with_map`]).
///
/// `None` if the objective is not degree-≤2 (should not happen for a problem
/// the classifier routed here). The reformulation of a convex quadratic
/// `½xᵀQx + aᵀx + b_eff ≤ 0` (with `Q = FᵀF ⪰ 0`) is the standard rotated→
/// standard SOC: writing `s = −(aᵀx + b_eff)`, the cone slack
/// `(s+1, s−1, √2·Fx)` lies in the second-order cone iff `‖Fx‖² ≤ 2s`, i.e.
/// iff the original constraint holds.
pub fn extract_socp_with_map(
    prob: &NlProblem,
) -> Option<(QpProblem, Vec<ConSocpMap>, f64, Vec<ConeSpec>)> {
    let n = prob.n;
    let sign = if prob.minimize { 1.0 } else { -1.0 };

    // --- objective P (lower triangle) + folded linear / constant terms ---
    let (hess, obj_nl_linear, obj_nl_constant) = analyze_quadratic_full(&prob.obj_nonlinear, n)?;
    let mut p_lower: Vec<Triplet> = Vec::with_capacity(hess.len());
    for ((i, j), v) in &hess {
        let (row, col) = if i >= j { (*i, *j) } else { (*j, *i) };
        p_lower.push(Triplet::new(row, col, sign * v));
    }
    let mut c = vec![0.0; n];
    for (var, coef) in &prob.obj_linear {
        c[*var] += sign * coef;
    }
    for (var, coef) in &obj_nl_linear {
        c[*var] += sign * coef;
    }

    // --- constraints: equalities → A; linear ineqs → nonneg G block;
    //     convex quadratics → deferred SOC blocks (added after the nonneg
    //     rows so the cones partition G in row order) ---
    let mut a: Vec<Triplet> = Vec::new();
    let mut b: Vec<f64> = Vec::new();
    let mut g: Vec<Triplet> = Vec::new();
    let mut h: Vec<f64> = Vec::new();
    let mut con_map: Vec<ConSocpMap> = Vec::with_capacity(prob.m);
    let mut soc_blocks: Vec<SocBlock> = Vec::new();

    for (row, lin) in prob.con_linear.iter().enumerate() {
        let lo = prob.g_l[row];
        let hi = prob.g_u[row];
        let nl = &prob.con_nonlinear[row];
        let quad = analyze_quadratic_full(nl, n);
        let is_quadratic = matches!(&quad, Some((hmap, _, _)) if !hmap.is_empty());

        if is_quadratic {
            // Convex quadratic inequality `g(x) ≤ g_u` (the classifier
            // guarantees an upper-only bound with PSD Hessian). Build the
            // factor F (FᵀF = Q) and defer the SOC rows.
            let (hmap, nl_lin, nl_const) = quad.expect("checked above");
            // Linear coefficients a = linear-section + folded nonlinear-tree
            // linear part, accumulated sparsely: a QCQP row's linear part is
            // as narrow as its quadratic part, and `n` here can be six digits.
            let mut a_map: std::collections::BTreeMap<usize, f64> =
                std::collections::BTreeMap::new();
            for (var, coef) in lin {
                *a_map.entry(*var).or_insert(0.0) += *coef;
            }
            for (var, coef) in &nl_lin {
                *a_map.entry(*var).or_insert(0.0) += *coef;
            }
            let a_vec: Vec<(usize, f64)> = a_map.into_iter().filter(|&(_, c)| c != 0.0).collect();

            let f_rows = socp_factor_rows(&hmap);
            let con_idx = con_map.len();
            con_map.push(ConSocpMap::Quad {
                z_row0: 0,
                z_row1: 0,
            }); // patched in the SOC pass below
            soc_blocks.push(SocBlock {
                con_idx,
                a: a_vec,
                b_eff: nl_const - hi,
                f_rows,
            });
            continue;
        }

        // Linear constraint. Combine the `.nl` linear section with any
        // degree-≤1 terms AMPL folded into the (here empty-Hessian)
        // nonlinear tree, and shift the bounds by the folded constant.
        let (nl_lin, const_shift) = quad.map(|(_, l, k)| (l, k)).unwrap_or_default();
        let mut coef = vec![0.0; n];
        for (var, v) in lin {
            coef[*var] += *v;
        }
        for (var, v) in &nl_lin {
            coef[*var] += *v;
        }
        let nonzeros = || coef.iter().enumerate().filter(|(_, v)| **v != 0.0);
        if lo == hi && lower_bound_present(lo) && upper_bound_present(hi) {
            let eq_row = next_row(&b);
            for (var, v) in nonzeros() {
                a.push(Triplet::new(eq_row, var, *v));
            }
            b.push(lo - const_shift);
            con_map.push(ConSocpMap::Eq { a_row: eq_row });
        } else {
            let upper = if upper_bound_present(hi) {
                let gr = next_row(&h);
                for (var, v) in nonzeros() {
                    g.push(Triplet::new(gr, var, *v));
                }
                h.push(hi - const_shift);
                Some(gr)
            } else {
                None
            };
            let lower = if lower_bound_present(lo) {
                let gr = next_row(&h);
                for (var, v) in nonzeros() {
                    g.push(Triplet::new(gr, var, -*v));
                }
                h.push(-(lo - const_shift));
                Some(gr)
            } else {
                None
            };
            con_map.push(ConSocpMap::Ineq { upper, lower });
        }
    }

    // Variable bounds go in the explicit box, not into this orthant block —
    // see [`extract_box`]. `solve_socp_ipm` appends them as a trailing
    // nonnegative block of its own, *after* the cones, so they stay outside
    // the partition `cones` has to cover.
    let (lb, ub) = extract_box(prob);

    // The nonnegative block is every G row built so far. The cones list must
    // cover G in row order: this orthant block, then one SOC per quadratic.
    let num_nonneg = h.len();
    let mut cones: Vec<ConeSpec> = Vec::with_capacity(1 + soc_blocks.len());
    if num_nonneg > 0 {
        cones.push(ConeSpec::Nonneg(num_nonneg));
    }

    // --- emit the deferred second-order cones ---
    for blk in soc_blocks {
        let r = blk.f_rows.len();
        let dim = r + 2;
        let row0 = next_row(&h);
        // s0 = (1 − b_eff) − aᵀx  →  G row = a, h = 1 − b_eff.
        for &(var, coef) in &blk.a {
            g.push(Triplet::new(row0, var, coef));
        }
        h.push(1.0 - blk.b_eff);
        let row1 = next_row(&h);
        // s1 = −(1 + b_eff) − aᵀx  →  G row = a, h = −(1 + b_eff).
        for &(var, coef) in &blk.a {
            g.push(Triplet::new(row1, var, coef));
        }
        h.push(-(1.0 + blk.b_eff));
        // s_{2+k} = √2·(Fx)_k  →  G row = −√2·F_k, h = 0. `f` is indexed by
        // position within the row's support, so scatter back through it.
        let sqrt2 = std::f64::consts::SQRT_2;
        for f in &blk.f_rows {
            let gr = next_row(&h);
            for &(var, fv) in f {
                g.push(Triplet::new(gr, var, -sqrt2 * fv));
            }
            h.push(0.0);
        }
        cones.push(ConeSpec::SecondOrder(dim));
        con_map[blk.con_idx] = ConSocpMap::Quad {
            z_row0: row0,
            z_row1: row1,
        };
    }

    Some((
        QpProblem {
            n,
            p_lower,
            c,
            a,
            b,
            g,
            h,
            lb,
            ub,
        },
        con_map,
        obj_nl_constant,
        cones,
    ))
}

/// Map the SOCP solver's multipliers `(y, z)` back to a per-`.nl`-constraint
/// dual vector (length `prob.m`), in POUNCE's NLP-path `.sol` convention.
///
/// Linear rows reuse the QP-path recovery (`y[a_row]` for an equality;
/// `z_upper − z_lower` for an inequality). For a convex quadratic
/// `g(x) ≤ g_u` reformulated to a second-order cone, the constraint
/// multiplier is recovered as the sum of the two cone duals on the rows
/// carrying the linear coefficient vector `a`: `λ = z[r0] + z[r1]`. (At a
/// KKT point stationarity reads `λ(∇g) = (z[r0]+z[r1])·a + …`, so this sum is
/// the original multiplier; the cone's remaining rows reconstruct the `Qx`
/// part.) `sign` undoes the maximize→minimize negation.
pub fn recover_socp_duals(
    prob: &NlProblem,
    con_map: &[ConSocpMap],
    y: &[f64],
    z: &[f64],
) -> Vec<f64> {
    let sign = if prob.minimize { 1.0 } else { -1.0 };
    con_map
        .iter()
        .map(|m| match m {
            ConSocpMap::Eq { a_row } => sign * y[*a_row],
            ConSocpMap::Ineq { upper, lower } => {
                let zu = upper.map(|r| z[r]).unwrap_or(0.0);
                let zl = lower.map(|r| z[r]).unwrap_or(0.0);
                sign * (zu - zl)
            }
            ConSocpMap::Quad { z_row0, z_row1 } => sign * (z[*z_row0] + z[*z_row1]),
        })
        .collect()
}

/// Factor one quadratic row's Hessian `Q` into sparse rows `f_k` (in the
/// problem's variable indexing) with `Σ_k f_k f_kᵀ = Q`.
///
/// Two paths, and the cheap one is the common one:
///
/// * **Diagonal `Q`** — `f` is one row per positive diagonal entry, each with a
///   single nonzero `√d_i`. `O(k)` time and `O(k)` space in the row's support.
///   This is the `qssp180`/`nql180` regime, where the general path's `O(k³)`
///   factorization and `k²` factor would both be ruinous for no benefit.
/// * **Otherwise** — pivoted Cholesky on a dense `k×k` over the row's support,
///   then scatter the rows back to problem indices, dropping zeros. Sized by
///   `k`, never by `n`.
fn socp_factor_rows(
    hmap: &std::collections::BTreeMap<(usize, usize), f64>,
) -> Vec<Vec<(usize, f64)>> {
    let support = quad_support(hmap);
    let k = support.len();

    if !hmap.keys().any(|&(i, j)| i != j) {
        // Diagonal: one factor row per variable, holding that entry's square
        // root. Nonpositive diagonal entries are the zero eigenvalues of a PSD
        // diagonal matrix (convexity is already established before we get
        // here), and contribute no row.
        //
        // Two details make this a *shortcut* rather than an approximation, so
        // the fast path and the general path emit bit-identical factors and a
        // problem's trajectory cannot depend on which one ran:
        //
        //  * the tolerance is the same expression `psd_outer_factor` uses, so
        //    both drop exactly the same entries and agree on the cone's
        //    dimension;
        //  * the value is `d / √d`, not `√d`. They differ by an ulp — `√2` is
        //    `0x1.6a09e667f3bcdp+0` and `2/√2` is `0x1.6a09e667f3bccp+0` — and
        //    `d / √d` is what the general path's `a[i][p] / d_pivot` computes.
        //    Reproducing it is free; not reproducing it moved `qcqp_ball` from
        //    17 conic iterations to 12 on a 2-ulp perturbation of one `G`
        //    entry, which is precisely the kind of invisible trajectory change
        //    `scripts/sweep-fixtures.sh` exists to catch.
        //  * the rows come out in **pivot order**, largest diagonal first. On a
        //    diagonal matrix the rank-1 downdate only zeros the pivot, so the
        //    general path's complete pivoting visits entries in descending
        //    order with ties going to the lower index — which is what a
        //    *stable* sort of the (ascending-by-index) map entries gives.
        //    `‖Fx‖` does not care about row order, but `G` does: the rows land
        //    in the KKT matrix and the ordering feeds the fill-reducing
        //    permutation.
        let max_diag = hmap.values().fold(0.0_f64, |m, &v| m.max(v));
        let tol = 1e-12 * max_diag.max(1.0);
        let mut diag: Vec<(usize, f64)> = hmap
            .iter()
            .filter(|&(_, &v)| v > tol)
            .map(|(&(i, _), &v)| (i, v))
            .collect();
        diag.sort_by(|a, b| b.1.partial_cmp(&a.1).expect("PSD diagonal is finite"));
        return diag
            .into_iter()
            .map(|(i, v)| vec![(i, v / v.sqrt())])
            .collect();
    }

    let dense = dense_symmetric_on_support(hmap, &support);
    psd_outer_factor(dense, k)
        .into_iter()
        .map(|f| {
            f.into_iter()
                .enumerate()
                .filter(|&(_, fv)| fv != 0.0)
                .map(|(loc, fv)| (support[loc], fv))
                .collect()
        })
        .collect()
}

/// The variables a quadratic row touches, ascending and deduplicated.
///
/// This is the row's *support*, and it is what every downstream structure is
/// sized by. `hmap` stores only `i ≤ j`, so both coordinates must be collected.
fn quad_support(hmap: &std::collections::BTreeMap<(usize, usize), f64>) -> Vec<usize> {
    let mut s: Vec<usize> = hmap.keys().flat_map(|&(i, j)| [i, j]).collect();
    s.sort_unstable();
    s.dedup();
    s
}

/// Build a dense symmetric `k×k` matrix over a row's `support` from a
/// [`QuadHessian`]-style map of `(i ≤ j) → Hessian entry` (diagonal entries are
/// the full `∂²/∂xᵢ²`, so `½xᵀHx` reproduces the quadratic form). Off-diagonals
/// are mirrored.
///
/// Sized by `k`, never by `n`: the previous `n×n` version asked for 134 GB on a
/// two-variable row of `nql180`.
fn dense_symmetric_on_support(
    hmap: &std::collections::BTreeMap<(usize, usize), f64>,
    support: &[usize],
) -> Vec<f64> {
    let k = support.len();
    // `support` is sorted, so a binary search is the local index.
    let loc = |v: usize| support.binary_search(&v).expect("key came from support");
    let mut dense = vec![0.0; k * k];
    for (&(i, j), &v) in hmap {
        let (li, lj) = (loc(i), loc(j));
        dense[li * k + lj] = v;
        dense[lj * k + li] = v;
    }
    dense
}

/// Symmetric **pivoted (rank-revealing) Cholesky** of a PSD matrix `H`
/// (row-major `n×n`, consumed as scratch), returning the factor rows `f_k`
/// (each length `n`) such that `Σ_k f_k f_kᵀ = H` — equivalently `FᵀF = H`
/// with `F` the matrix whose rows are the `f_k`.
///
/// Callers pass a **row's support size** `k` here, not the problem width: this
/// is `O(n³)` in whatever it is handed, so the distinction is what makes a wide
/// QCQP extractable at all.
///
/// The number of rows is the
/// numerical rank, so a rank-deficient `Q` (e.g. `Q = vvᵀ`) yields the
/// minimal cone. Complete diagonal pivoting keeps the factorization stable
/// on the indefinite-looking-but-PSD matrices finite precision can produce.
fn psd_outer_factor(mut a: Vec<f64>, n: usize) -> Vec<Vec<f64>> {
    let mut rows: Vec<Vec<f64>> = Vec::new();
    // Tolerance relative to the largest initial diagonal: pivots at or below
    // this are treated as the zero eigenvalues of the PSD matrix.
    let max_diag = (0..n).map(|i| a[i * n + i]).fold(0.0_f64, f64::max);
    let tol = 1e-12 * max_diag.max(1.0);
    for _ in 0..n {
        // Largest remaining diagonal pivot.
        let mut p = 0usize;
        let mut best = f64::NEG_INFINITY;
        for i in 0..n {
            let d = a[i * n + i];
            if d > best {
                best = d;
                p = i;
            }
        }
        if best <= tol {
            break;
        }
        let d = best.sqrt();
        // f = column p of the residual, scaled by 1/d.
        let mut f = vec![0.0; n];
        for i in 0..n {
            f[i] = a[i * n + p] / d;
        }
        // Rank-1 downdate: A ← A − f fᵀ.
        for i in 0..n {
            let fi = f[i];
            if fi == 0.0 {
                continue;
            }
            for j in 0..n {
                a[i * n + j] -= fi * f[j];
            }
        }
        rows.push(f);
    }
    rows
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nl_reader::{BinOp, Expr};
    use pounce_convex::{QpOptions, QpStatus, solve_qp_ipm, solve_socp_ipm};
    use pounce_feral::FeralSolverInterface;
    use pounce_linsol::SparseSymLinearSolverInterface;

    fn backend() -> Box<dyn SparseSymLinearSolverInterface> {
        Box::new(FeralSolverInterface::new())
    }

    fn pow2(var: usize) -> Expr {
        Expr::Binary(
            BinOp::Pow,
            Box::new(Expr::Var(var)),
            Box::new(Expr::Const(2.0)),
        )
    }

    /// min −x0 − x1  s.t.  x0² + x1² ≤ 1  → x* = (1/√2, 1/√2), f* = −√2.
    /// Exercises the QCQP→SOCP reformulation end-to-end: a rank-2 ball
    /// constraint becomes one second-order cone, no nonnegative block.
    #[test]
    fn extract_and_solve_socp_ball() {
        let prob = NlProblem {
            n: 2,
            m: 1,
            num_obj: 1,
            minimize: true,
            obj_nonlinear: Expr::Const(0.0),
            obj_linear: vec![(0, -1.0), (1, -1.0)],
            obj_constant: 0.0,
            con_nonlinear: vec![Expr::Binary(
                BinOp::Add,
                Box::new(pow2(0)),
                Box::new(pow2(1)),
            )],
            con_linear: vec![vec![]],
            x_l: vec![-2e19, -2e19],
            x_u: vec![2e19, 2e19],
            g_l: vec![-2e19],
            g_u: vec![1.0],
            x0: vec![0.0, 0.0],
            lambda0: vec![0.0],
            suffixes: Default::default(),
            imported_funcs: Vec::new(),
            ampl_options: Vec::new(),
            nl_counts: None,
            var_names: Vec::new(),
            con_names: Vec::new(),
        };
        let (qp, con_map, obj_const, cones) = extract_socp_with_map(&prob).expect("extract");
        assert_eq!(obj_const, 0.0);
        // No linear inequalities / bounds → no nonneg block; one SOC of
        // dimension rank(Q)+2 = 2+2 = 4.
        assert_eq!(cones, vec![ConeSpec::SecondOrder(4)]);
        assert_eq!(qp.m_ineq(), 4);

        let sol = solve_socp_ipm(&qp, &cones, &QpOptions::default(), backend);
        assert_eq!(sol.status, QpStatus::Optimal);
        let inv_sqrt2 = 1.0 / 2.0_f64.sqrt();
        assert!((sol.x[0] - inv_sqrt2).abs() < 1e-5, "x0={}", sol.x[0]);
        assert!((sol.x[1] - inv_sqrt2).abs() < 1e-5, "x1={}", sol.x[1]);
        assert!(
            (sol.obj - (-2.0_f64.sqrt())).abs() < 1e-5,
            "obj={}",
            sol.obj
        );

        // Analytic multiplier: c + λ·2x = 0 ⇒ λ = 1/(2x0) = √2/2 ≈ 0.7071,
        // positive (active upper bound), matching the `.sol` sign convention.
        let lambda = recover_socp_duals(&prob, &con_map, &sol.y, &sol.z);
        assert_eq!(lambda.len(), 1);
        assert!(
            (lambda[0] - 0.5 * 2.0_f64.sqrt()).abs() < 1e-3,
            "ball constraint dual={}",
            lambda[0]
        );
    }

    /// min x0  s.t.  (x0−3)² ≤ 1  → feasible x0 ∈ [2, 4], optimum x0 = 2.
    /// The constraint's linear (`−6x0`) and constant (`+9`) terms are folded
    /// into the nonlinear tree; the reformulation must recover `b_eff = 9 − 1`
    /// so the cone encodes `x0² − 6x0 + 8 ≤ 0`, not `x0² ≤ 1`.
    #[test]
    fn extract_and_solve_socp_folds_constraint_constant() {
        let con = Expr::Binary(
            BinOp::Pow,
            Box::new(Expr::Binary(
                BinOp::Sub,
                Box::new(Expr::Var(0)),
                Box::new(Expr::Const(3.0)),
            )),
            Box::new(Expr::Const(2.0)),
        );
        let prob = NlProblem {
            n: 1,
            m: 1,
            num_obj: 1,
            minimize: true,
            obj_nonlinear: Expr::Const(0.0),
            obj_linear: vec![(0, 1.0)],
            obj_constant: 0.0,
            con_nonlinear: vec![con],
            con_linear: vec![vec![]],
            x_l: vec![-2e19],
            x_u: vec![2e19],
            g_l: vec![-2e19],
            g_u: vec![1.0],
            x0: vec![0.0],
            lambda0: vec![0.0],
            suffixes: Default::default(),
            imported_funcs: Vec::new(),
            ampl_options: Vec::new(),
            nl_counts: None,
            var_names: Vec::new(),
            con_names: Vec::new(),
        };
        let (qp, _con_map, obj_const, cones) = extract_socp_with_map(&prob).expect("extract");
        assert_eq!(obj_const, 0.0);
        assert_eq!(cones, vec![ConeSpec::SecondOrder(3)]); // rank 1 + 2.

        let sol = solve_socp_ipm(&qp, &cones, &QpOptions::default(), backend);
        assert_eq!(sol.status, QpStatus::Optimal);
        assert!((sol.x[0] - 2.0).abs() < 1e-5, "x0={}", sol.x[0]);
    }

    /// Build `min −x_i − x_j  s.t.  x_i² + x_j² ≤ 1` over `n` variables, where
    /// `i` and `j` are neither low-numbered nor adjacent.
    fn wide_ball(n: usize, i: usize, j: usize) -> NlProblem {
        let sq = |v: usize| {
            Expr::Binary(
                BinOp::Pow,
                Box::new(Expr::Var(v)),
                Box::new(Expr::Const(2.0)),
            )
        };
        NlProblem {
            n,
            m: 1,
            num_obj: 1,
            minimize: true,
            obj_nonlinear: Expr::Const(0.0),
            obj_linear: vec![(i, -1.0), (j, -1.0)],
            obj_constant: 0.0,
            con_nonlinear: vec![Expr::Binary(BinOp::Add, Box::new(sq(i)), Box::new(sq(j)))],
            con_linear: vec![vec![]],
            x_l: vec![-10.0; n],
            x_u: vec![10.0; n],
            g_l: vec![-2e19],
            g_u: vec![1.0],
            x0: vec![0.0; n],
            lambda0: vec![0.0],
            suffixes: Default::default(),
            imported_funcs: Vec::new(),
            ampl_options: Vec::new(),
            nl_counts: None,
            var_names: Vec::new(),
            con_names: Vec::new(),
        }
    }

    /// The cone factor is built on the row's **support**, so its columns are
    /// support-local and must be scattered back through `support` on emission.
    /// If that scatter is dropped, a row touching `x11`/`x37` silently
    /// constrains `x0`/`x1` instead — a wrong answer, not a crash.
    #[test]
    fn socp_factor_columns_scatter_back_to_original_variables() {
        let prob = wide_ball(40, 11, 37);
        let (qp, _con_map, _obj_const, cones) = extract_socp_with_map(&prob).expect("extract");
        assert_eq!(cones, vec![ConeSpec::SecondOrder(4)]); // rank 2 + 2.

        // Every G entry in the two factor rows must sit in column 11 or 37.
        // Rows 0 and 1 are the `a`-rows (here empty: no linear part).
        let factor_cols: std::collections::BTreeSet<usize> =
            qp.g.iter().filter(|t| t.row >= 2).map(|t| t.col).collect();
        assert_eq!(
            factor_cols,
            [11usize, 37].into_iter().collect(),
            "factor rows must reference the row's own variables, got {factor_cols:?}"
        );
    }

    /// The extractor must be sized by a row's support, not by the problem
    /// width. A two-variable quadratic row in a 50 000-variable problem needs
    /// kilobytes; sizing it `n×n` would ask for 20 GB and abort the process.
    #[test]
    fn socp_extraction_is_sized_by_support_not_problem_width() {
        let n = 50_000;
        let prob = wide_ball(n, 7, n - 3);
        let (qp, _con_map, _obj_const, cones) = extract_socp_with_map(&prob).expect("extract");
        assert_eq!(cones, vec![ConeSpec::SecondOrder(4)]);
        // The cone contributes exactly two nonzeros per factor row.
        assert_eq!(qp.g.iter().filter(|t| t.row >= 2).count(), 2);
    }

    /// The same scatter, but through the **dense** path: a cross term makes `Q`
    /// non-diagonal, so the row goes through the pivoted Cholesky and its
    /// support-local columns must be mapped back. Rank 1, so one factor row.
    #[test]
    fn socp_dense_factor_columns_scatter_back_to_original_variables() {
        // (x11 + x37)² ≤ 1 — off-diagonal Q, rank 1.
        let con = Expr::Binary(
            BinOp::Pow,
            Box::new(Expr::Binary(
                BinOp::Add,
                Box::new(Expr::Var(11)),
                Box::new(Expr::Var(37)),
            )),
            Box::new(Expr::Const(2.0)),
        );
        let mut prob = wide_ball(40, 11, 37);
        prob.con_nonlinear = vec![con];

        let (qp, _con_map, _obj_const, cones) = extract_socp_with_map(&prob).expect("extract");
        assert_eq!(cones, vec![ConeSpec::SecondOrder(3)], "rank 1 + 2");
        let factor_cols: std::collections::BTreeSet<usize> =
            qp.g.iter().filter(|t| t.row >= 2).map(|t| t.col).collect();
        assert_eq!(factor_cols, [11usize, 37].into_iter().collect());
    }

    /// A diagonal `Q` must not be densified. `socp_factor_rows` returns one
    /// single-nonzero row per positive diagonal entry, so the factor is `O(k)`
    /// — this is what makes the very large diagonal QCQPs (`qssp180`,
    /// `nql180`) representable at all.
    #[test]
    fn diagonal_hessian_factors_in_linear_space() {
        let mut h = std::collections::BTreeMap::new();
        for i in 0..1000usize {
            h.insert((i, i), 4.0);
        }
        let rows = socp_factor_rows(&h);
        assert_eq!(rows.len(), 1000);
        assert!(
            rows.iter().all(|r| r.len() == 1),
            "a diagonal Q must give one nonzero per factor row"
        );
        for (k, r) in rows.iter().enumerate() {
            assert_eq!(r[0].0, k);
            assert!((r[0].1 - 2.0).abs() < 1e-12, "√4 = 2, got {}", r[0].1);
        }
    }

    /// Both factor paths must satisfy the same contract, `Σ_k f_k f_kᵀ = Q`,
    /// and must agree on the rank. There are now two of them — an `O(k)`
    /// diagonal shortcut and the general pivoted Cholesky — so the contract is
    /// asserted directly rather than inferred from the shortcut's derivation.
    ///
    /// The near-zero diagonal entry is the rank agreement: it sits below the
    /// `1e-12 · max_diag` cut, and both paths must drop it, or the two would
    /// build cones of different dimension for the same constraint.
    #[test]
    fn both_factor_paths_reconstruct_q_and_agree_on_rank() {
        let recon = |rows: &[Vec<(usize, f64)>]| {
            let mut q: std::collections::BTreeMap<(usize, usize), f64> = Default::default();
            for r in rows {
                for &(i, fi) in r {
                    for &(j, fj) in r {
                        *q.entry((i, j)).or_insert(0.0) += fi * fj;
                    }
                }
            }
            q.retain(|_, v| v.abs() > 1e-12);
            q
        };

        // Diagonal path, with one entry far below the rank tolerance.
        // Deliberately *not* descending by index: the largest entry sits on
        // the highest variable, so a shortcut that emitted in index order
        // would produce the right cone with the rows in the wrong order.
        let diag: std::collections::BTreeMap<(usize, usize), f64> =
            [((3, 3), 2.0), ((8, 8), 9.0), ((9, 9), 1e-20)]
                .into_iter()
                .collect();
        let drows = socp_factor_rows(&diag);
        assert_eq!(drows.len(), 2, "the 1e-20 pivot is a zero eigenvalue");

        // The shortcut must be *bit*-identical to the general path, not merely
        // close: a 2-ulp difference in one `G` entry visibly moved `qcqp_ball`'s
        // conic trajectory (17 → 12 iterations). Run the same matrix through
        // `psd_outer_factor` and compare the raw bits.
        let support = quad_support(&diag);
        let general = psd_outer_factor(dense_symmetric_on_support(&diag, &support), support.len());
        let general_vals: Vec<f64> = general
            .iter()
            .map(|f| f.iter().copied().find(|v| *v != 0.0).expect("one nonzero"))
            .collect();
        let short_vals: Vec<f64> = drows.iter().map(|r| r[0].1).collect();
        assert_eq!(
            short_vals.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            general_vals.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "diagonal shortcut must reproduce psd_outer_factor bit for bit \
             (got {short_vals:?} vs {general_vals:?})"
        );
        assert_eq!(drows[0][0].0, 8, "largest diagonal pivots first");
        assert_eq!(drows[1][0].0, 3);
        let dq = recon(&drows);
        assert_eq!(dq.len(), 2);
        assert!((dq[&(3, 3)] - 2.0).abs() < 1e-12);
        assert!((dq[&(8, 8)] - 9.0).abs() < 1e-12);

        // General path: same matrix plus a cross term, so the diagonal
        // shortcut no longer applies. Q = [[2,1],[1,9]] on {3,8} is positive
        // definite, so full rank 2.
        let mut coupled = diag.clone();
        coupled.insert((3, 8), 1.0);
        let grows = socp_factor_rows(&coupled);
        assert_eq!(grows.len(), 2);
        let gq = recon(&grows);
        assert!((gq[&(3, 3)] - 2.0).abs() < 1e-10);
        assert!((gq[&(8, 8)] - 9.0).abs() < 1e-10);
        assert!((gq[&(3, 8)] - 1.0).abs() < 1e-10);
        assert!((gq[&(8, 3)] - 1.0).abs() < 1e-10);
        assert!(
            !gq.contains_key(&(9, 9)),
            "zero eigenvalue must not reappear"
        );
    }

    /// `psd_outer_factor` recovers a rank-1 `Q = vvᵀ` with a single factor row
    /// (minimal cone), and reconstructs `Q` exactly.
    #[test]
    fn psd_outer_factor_is_rank_revealing() {
        // Q = [[1,2],[2,4]] = v vᵀ with v = (1,2): rank 1.
        let q = vec![1.0, 2.0, 2.0, 4.0];
        let rows = psd_outer_factor(q.clone(), 2);
        assert_eq!(rows.len(), 1, "rank-1 Q must give one factor row");
        // Reconstruct Σ f fᵀ and compare to Q.
        let mut recon = vec![0.0; 4];
        for f in &rows {
            for i in 0..2 {
                for j in 0..2 {
                    recon[i * 2 + j] += f[i] * f[j];
                }
            }
        }
        for k in 0..4 {
            assert!((recon[k] - q[k]).abs() < 1e-9, "recon[{k}]={}", recon[k]);
        }
    }

    /// min (x0)^2 + (x1)^2 s.t. x0 + x1 = 2, no var bounds → (1,1), f*=2.
    #[test]
    fn extract_and_solve_equality_qp() {
        let prob = NlProblem {
            n: 2,
            m: 1,
            num_obj: 1,
            minimize: true,
            obj_nonlinear: Expr::Binary(BinOp::Add, Box::new(pow2(0)), Box::new(pow2(1))),
            obj_linear: vec![],
            obj_constant: 0.0,
            con_nonlinear: vec![Expr::Const(0.0)],
            con_linear: vec![vec![(0, 1.0), (1, 1.0)]],
            x_l: vec![-2e19, -2e19],
            x_u: vec![2e19, 2e19],
            g_l: vec![2.0],
            g_u: vec![2.0],
            x0: vec![0.0, 0.0],
            lambda0: vec![0.0],
            suffixes: Default::default(),
            imported_funcs: Vec::new(),
            ampl_options: Vec::new(),
            nl_counts: None,
            var_names: Vec::new(),
            con_names: Vec::new(),
        };
        let (qp, con_map, obj_const) = extract_qp_with_map(&prob).expect("extract");
        // No constant anywhere in this objective.
        assert_eq!(obj_const, 0.0);
        // P = 2I → two diagonal entries.
        assert_eq!(qp.p_lower.len(), 2);
        assert_eq!(qp.m_eq(), 1);
        assert_eq!(qp.m_ineq(), 0);

        let sol = solve_qp_ipm(&qp, &QpOptions::default(), backend);
        assert_eq!(sol.status, QpStatus::Optimal);
        assert!((sol.x[0] - 1.0).abs() < 1e-6, "x0={}", sol.x[0]);
        assert!((sol.x[1] - 1.0).abs() < 1e-6, "x1={}", sol.x[1]);
        assert!((sol.obj - 2.0).abs() < 1e-6, "obj={}", sol.obj);

        // KKT for the equality: ∇f + y·∇g = 0 → 2x_i + y = 0 at x=1 → y=−2.
        let lambda = recover_duals(&prob, &con_map, &sol.y, &sol.z);
        assert_eq!(lambda.len(), 1);
        assert!(
            (lambda[0] - (-2.0)).abs() < 1e-5,
            "equality dual={}",
            lambda[0]
        );
    }

    /// Regression for the dropped-linear-term bug: the objective `(x0-3)²`
    /// lives entirely in the nonlinear tree, so its linear part (`−6·x0`)
    /// must be folded into `c`. Without it the solve minimizes `x0²`
    /// (optimum 0) instead of `(x0-3)²` (optimum 3).
    #[test]
    fn extract_keeps_linear_term_from_nonlinear_tree() {
        // (x0 - 3)^2 = x0^2 - 6 x0 + 9, all in obj_nonlinear.
        let obj = Expr::Binary(
            BinOp::Pow,
            Box::new(Expr::Binary(
                BinOp::Sub,
                Box::new(Expr::Var(0)),
                Box::new(Expr::Const(3.0)),
            )),
            Box::new(Expr::Const(2.0)),
        );
        let prob = NlProblem {
            n: 1,
            m: 0,
            num_obj: 1,
            minimize: true,
            obj_nonlinear: obj,
            obj_linear: vec![],
            obj_constant: 0.0,
            con_nonlinear: vec![],
            con_linear: vec![],
            x_l: vec![-2e19],
            x_u: vec![2e19],
            g_l: vec![],
            g_u: vec![],
            x0: vec![0.0],
            lambda0: vec![],
            suffixes: Default::default(),
            imported_funcs: Vec::new(),
            ampl_options: Vec::new(),
            nl_counts: None,
            var_names: Vec::new(),
            con_names: Vec::new(),
        };
        let qp = extract_qp(&prob).expect("extract");
        assert_eq!(qp.c.len(), 1);
        assert!(
            (qp.c[0] - (-6.0)).abs() < 1e-12,
            "c[0]={} — linear term from the nonlinear tree was dropped",
            qp.c[0]
        );
        // P = 2 (one diagonal entry).
        assert_eq!(qp.p_lower.len(), 1);

        let sol = solve_qp_ipm(&qp, &QpOptions::default(), backend);
        assert_eq!(sol.status, QpStatus::Optimal);
        assert!(
            (sol.x[0] - 3.0).abs() < 1e-6,
            "x0={} (expected 3)",
            sol.x[0]
        );
    }

    /// Inequality dual sign/magnitude. min x0² s.t. x0 ≥ 1 (a one-sided
    /// inequality g_l=1, g_u=+inf). Optimum x0=1, active. The expected
    /// dual −2.0 is the value POUNCE's *NLP* path writes for this exact
    /// problem (verified by running `solver_selection=nlp` on the same
    /// `.nl`); recover_duals must match that reference convention.
    #[test]
    fn inequality_dual_recovered() {
        let prob = NlProblem {
            n: 1,
            m: 1,
            num_obj: 1,
            minimize: true,
            obj_nonlinear: pow2(0),
            obj_linear: vec![],
            obj_constant: 0.0,
            con_nonlinear: vec![Expr::Const(0.0)],
            con_linear: vec![vec![(0, 1.0)]], // g(x) = x0
            x_l: vec![-2e19],
            x_u: vec![2e19],
            g_l: vec![1.0], // x0 ≥ 1
            g_u: vec![2e19],
            x0: vec![0.0],
            lambda0: vec![0.0],
            suffixes: Default::default(),
            imported_funcs: Vec::new(),
            ampl_options: Vec::new(),
            nl_counts: None,
            var_names: Vec::new(),
            con_names: Vec::new(),
        };
        let (qp, con_map, obj_const) = extract_qp_with_map(&prob).expect("extract");
        // This model puts its constant in the `obj_constant` field, not the
        // nonlinear tree, so the tree constant is 0 here.
        assert_eq!(obj_const, 0.0);
        // One inequality row (the lower bound row −x0 ≤ −1); no upper.
        assert_eq!(qp.m_ineq(), 1);
        let sol = solve_qp_ipm(&qp, &QpOptions::default(), backend);
        assert_eq!(sol.status, QpStatus::Optimal);
        assert!((sol.x[0] - 1.0).abs() < 1e-6, "x0={}", sol.x[0]);
        let lambda = recover_duals(&prob, &con_map, &sol.y, &sol.z);
        assert!((lambda[0] - (-2.0)).abs() < 1e-5, "ineq dual={}", lambda[0]);
    }

    /// Regression (M11): a *constraint* whose linear and constant
    /// terms are folded into the nonlinear tree (not the `con_linear`
    /// section) must still reach `A`/`G`. AMPL/Pyomo emit this shape for
    /// rows the classifier admits as degree-≤1 (cancelled quadratics,
    /// defined variables): the whole `x0 − 3` lives in `con_nonlinear`
    /// and `con_linear[0]` is empty.
    ///
    ///     min x0   s.t.   x0 − 3 ≥ 0     (body in the nonlinear tree)
    ///
    /// True optimum: x0 = 3. The QP extractor used to build `A`/`G` from
    /// `con_linear` only — dropping the folded `+x0` *and* the `−3`
    /// shift, leaving a vacuous `0 ≤ 0` row, so `min x0` came out
    /// unbounded (or otherwise wrong) on the convex path.
    #[test]
    fn constraint_linear_terms_folded_in_tree_are_recovered() {
        // con body = x0 − 3, entirely in the nonlinear tree.
        let con = Expr::Binary(
            BinOp::Sub,
            Box::new(Expr::Var(0)),
            Box::new(Expr::Const(3.0)),
        );
        let prob = NlProblem {
            n: 1,
            m: 1,
            num_obj: 1,
            minimize: true,
            obj_nonlinear: Expr::Const(0.0),
            obj_linear: vec![(0, 1.0)],
            obj_constant: 0.0,
            con_nonlinear: vec![con],
            con_linear: vec![vec![]], // the `+x0` lives in the TREE
            x_l: vec![-2e19],
            x_u: vec![2e19],
            g_l: vec![0.0], // x0 − 3 ≥ 0
            g_u: vec![2e19],
            x0: vec![0.0],
            lambda0: vec![0.0],
            suffixes: Default::default(),
            imported_funcs: Vec::new(),
            ampl_options: Vec::new(),
            nl_counts: None,
            var_names: Vec::new(),
            con_names: Vec::new(),
        };
        let (qp, con_map, _obj_const) = extract_qp_with_map(&prob).expect("extract");
        // One inequality row: −x0 ≤ −3 (the lower bound, constant-shifted).
        assert_eq!(qp.m_ineq(), 1);
        let sol = solve_qp_ipm(&qp, &QpOptions::default(), backend);
        assert_eq!(sol.status, QpStatus::Optimal);
        assert!((sol.x[0] - 3.0).abs() < 1e-5, "x0={}", sol.x[0]);
        // Dual is recoverable and finite (the row carries a real coef now).
        let lambda = recover_duals(&prob, &con_map, &sol.y, &sol.z);
        assert_eq!(lambda.len(), 1);
        assert!(lambda[0].is_finite(), "dual={}", lambda[0]);
    }

    /// Regression: a constant folded into the *nonlinear objective tree*
    /// (not the `obj_constant` field) must still reach the reported
    /// objective. This is the real `.nl` shape AMPL/Pyomo emit for
    /// `min (x0-3)^2` — the whole `x0^2 - 6 x0 + 9` lives in the nonlinear
    /// tree and `obj_constant == 0`. The convex path used to drop the `+9`
    /// and report an objective 9 too small (cf. HS35 in the benchmark
    /// comparison). The minimizer is x0 = 1 (upper bound binds), where the
    /// true objective is (1-3)^2 = 4.
    #[test]
    fn tree_embedded_objective_constant_is_recovered() {
        // (x0 - 3)^2 as a single nonlinear tree: Pow(Sub(x0, 3), 2).
        let obj = Expr::Binary(
            BinOp::Pow,
            Box::new(Expr::Binary(
                BinOp::Sub,
                Box::new(Expr::Var(0)),
                Box::new(Expr::Const(3.0)),
            )),
            Box::new(Expr::Const(2.0)),
        );
        let prob = NlProblem {
            n: 1,
            m: 0,
            num_obj: 1,
            minimize: true,
            obj_nonlinear: obj,
            obj_linear: vec![],
            obj_constant: 0.0, // the +9 is in the TREE, not here
            con_nonlinear: vec![],
            con_linear: vec![],
            x_l: vec![0.0],
            x_u: vec![1.0],
            g_l: vec![],
            g_u: vec![],
            x0: vec![0.0],
            lambda0: vec![],
            suffixes: Default::default(),
            imported_funcs: Vec::new(),
            ampl_options: Vec::new(),
            nl_counts: None,
            var_names: Vec::new(),
            con_names: Vec::new(),
        };
        let (qp, _con_map, obj_const) = extract_qp_with_map(&prob).expect("extract");
        // The degree-0 term of (x0-3)^2 is +9, recovered from the tree.
        assert!((obj_const - 9.0).abs() < 1e-12, "tree constant={obj_const}");
        let sol = solve_qp_ipm(&qp, &QpOptions::default(), backend);
        assert_eq!(sol.status, QpStatus::Optimal);
        assert!((sol.x[0] - 1.0).abs() < 1e-6, "x0={}", sol.x[0]);
        // Reported objective = (½xᵀPx + cᵀx) + obj_const must equal the true
        // (1-3)^2 = 4, not the constant-dropped −5.
        let reported = sol.obj + obj_const;
        assert!((reported - 4.0).abs() < 1e-5, "reported obj={reported}");
    }

    /// Bound-constrained: min (x0-3)^2 = x0^2 - 6 x0 + 9, 0 ≤ x0 ≤ 1.
    /// Optimum x0 = 1 (upper bound binds). Here the constant 9 is carried
    /// in the `obj_constant` field (not the tree), so the extracted tree
    /// constant is 0 (asserted inside).
    #[test]
    fn extract_and_solve_bounded_qp() {
        let prob = NlProblem {
            n: 1,
            m: 0,
            num_obj: 1,
            minimize: true,
            obj_nonlinear: pow2(0),
            obj_linear: vec![(0, -6.0)],
            obj_constant: 9.0,
            con_nonlinear: vec![],
            con_linear: vec![],
            x_l: vec![0.0],
            x_u: vec![1.0],
            g_l: vec![],
            g_u: vec![],
            x0: vec![0.0],
            lambda0: vec![],
            suffixes: Default::default(),
            imported_funcs: Vec::new(),
            ampl_options: Vec::new(),
            nl_counts: None,
            var_names: Vec::new(),
            con_names: Vec::new(),
        };
        let qp = extract_qp(&prob).expect("extract");
        // The bounds are the box, not `G` rows.
        assert_eq!(qp.m_ineq(), 0);
        assert_eq!((qp.lb[0], qp.ub[0]), (0.0, 1.0));
        let sol = solve_qp_ipm(&qp, &QpOptions::default(), backend);
        assert_eq!(sol.status, QpStatus::Optimal);
        assert!((sol.x[0] - 1.0).abs() < 1e-6, "x0={}", sol.x[0]);
    }

    /// LP: min −x0 − x1, 0 ≤ x ≤ 1 → (1,1).
    #[test]
    fn extract_and_solve_lp() {
        let prob = NlProblem {
            n: 2,
            m: 0,
            num_obj: 1,
            minimize: true,
            obj_nonlinear: Expr::Const(0.0),
            obj_linear: vec![(0, -1.0), (1, -1.0)],
            obj_constant: 0.0,
            con_nonlinear: vec![],
            con_linear: vec![],
            x_l: vec![0.0, 0.0],
            x_u: vec![1.0, 1.0],
            g_l: vec![],
            g_u: vec![],
            x0: vec![0.0, 0.0],
            lambda0: vec![],
            suffixes: Default::default(),
            imported_funcs: Vec::new(),
            ampl_options: Vec::new(),
            nl_counts: None,
            var_names: Vec::new(),
            con_names: Vec::new(),
        };
        let qp = extract_qp(&prob).expect("extract");
        assert!(qp.p_lower.is_empty(), "LP has no Hessian");
        assert_eq!(qp.m_ineq(), 0, "bounds are the box, not `G` rows");
        assert_eq!(qp.lb, vec![0.0, 0.0]);
        assert_eq!(qp.ub, vec![1.0, 1.0]);
        let sol = solve_qp_ipm(&qp, &QpOptions::default(), backend);
        assert_eq!(sol.status, QpStatus::Optimal);
        assert!((sol.x[0] - 1.0).abs() < 1e-6);
        assert!((sol.x[1] - 1.0).abs() < 1e-6);
    }

    /// maximize x0 s.t. 0 ≤ x0 ≤ 5 → x0 = 5. Tests sign flip on a
    /// maximize objective.
    #[test]
    fn extract_maximize_negates() {
        let prob = NlProblem {
            n: 1,
            m: 0,
            num_obj: 1,
            minimize: false,
            obj_nonlinear: Expr::Const(0.0),
            obj_linear: vec![(0, 1.0)],
            obj_constant: 0.0,
            con_nonlinear: vec![],
            con_linear: vec![],
            x_l: vec![0.0],
            x_u: vec![5.0],
            g_l: vec![],
            g_u: vec![],
            x0: vec![0.0],
            lambda0: vec![],
            suffixes: Default::default(),
            imported_funcs: Vec::new(),
            ampl_options: Vec::new(),
            nl_counts: None,
            var_names: Vec::new(),
            con_names: Vec::new(),
        };
        let qp = extract_qp(&prob).expect("extract");
        // minimize −x0.
        assert_eq!(qp.c[0], -1.0);
        let sol = solve_qp_ipm(&qp, &QpOptions::default(), backend);
        assert_eq!(sol.status, QpStatus::Optimal);
        assert!((sol.x[0] - 5.0).abs() < 1e-6, "x0={}", sol.x[0]);
    }

    /// **gh #401.** A real bound past the *opposite* absent-bound sentinel is
    /// an ordinary bound, and must survive into `G`.
    ///
    /// `is_finite_bound` was `|v| < 1e19`, a symmetric magnitude test. An upper
    /// bound of `-5e20` failed it and the row `x_0 <= -5e20` never entered `G`,
    /// so the QP was solved over a strictly larger box — `min x_0` subject to
    /// nothing, which the IPM answers `Optimal` at a point the model excludes.
    #[test]
    fn variable_bound_past_the_opposite_sentinel_is_kept() {
        let prob = NlProblem {
            n: 1,
            m: 0,
            num_obj: 1,
            minimize: false, // maximize x0, so the -5e20 upper bound binds
            obj_nonlinear: Expr::Const(0.0),
            obj_linear: vec![(0, 1.0)],
            obj_constant: 0.0,
            con_nonlinear: vec![],
            con_linear: vec![],
            // No lower bound (`-1e21` is past the lower sentinel, so absent);
            // a real upper bound of `-5e20`, which is *not*.
            x_l: vec![-1e21],
            x_u: vec![-5e20],
            g_l: vec![],
            g_u: vec![],
            x0: vec![-7e20],
            lambda0: vec![],
            suffixes: Default::default(),
            imported_funcs: Vec::new(),
            ampl_options: Vec::new(),
            nl_counts: None,
            var_names: Vec::new(),
            con_names: Vec::new(),
        };
        let qp = extract_qp(&prob).expect("extract");
        assert_eq!(
            qp.ub[0], -5e20,
            "`x0 <= -5e20` is a real bound and must reach the box; the \
             symmetric |v| < 1e19 test dropped it, leaving an \
             unbounded-above box the model does not declare"
        );
    }

    /// **gh #401.** A row with equal bounds past the sentinel used to vanish
    /// from the problem *entirely* — contributing nothing to `A` and nothing
    /// to `G`.
    ///
    /// Directionally, `g_l = g_u = -5e20` is not an equality at all: the lower
    /// bound is absent (it is past `-1e19`) and the upper bound is real, so the
    /// row is the one-sided `x0 + x1 <= -5e20`. The old code got there by a
    /// different route and lost it: `lo == hi && is_finite_bound(lo)` failed, so
    /// the row fell into the inequality branch — where `is_finite_bound(hi)` and
    /// `is_finite_bound(lo)` were false too, leaving `upper` and `lower` both
    /// `None`. Silently deleted.
    ///
    /// (Note there is no such thing as an equality row outside `±1e19` under
    /// this convention: an equality needs both bounds present, and the two
    /// presence tests only overlap inside the band.)
    #[test]
    fn a_row_with_equal_bounds_past_the_sentinel_does_not_vanish() {
        let prob = NlProblem {
            n: 2,
            m: 1,
            num_obj: 1,
            minimize: true,
            obj_nonlinear: Expr::Const(0.0),
            obj_linear: vec![(0, 1.0)],
            obj_constant: 0.0,
            con_nonlinear: vec![Expr::Const(0.0)],
            con_linear: vec![vec![(0, 1.0), (1, 1.0)]],
            x_l: vec![-2e19, -2e19],
            x_u: vec![2e19, 2e19],
            g_l: vec![-5e20],
            g_u: vec![-5e20],
            x0: vec![0.0, 0.0],
            lambda0: vec![0.0],
            suffixes: Default::default(),
            imported_funcs: Vec::new(),
            ampl_options: Vec::new(),
            nl_counts: None,
            var_names: Vec::new(),
            con_names: Vec::new(),
        };
        let qp = extract_qp(&prob).expect("extract");
        assert_eq!(
            qp.m_eq(),
            0,
            "the lower bound is absent, so this is no equality"
        );
        assert_eq!(
            qp.m_ineq(),
            1,
            "`x0 + x1 <= -5e20` is a real constraint and must reach G; it used \
             to disappear from the problem entirely"
        );
        assert_eq!(qp.h[0], -5e20);
    }

    /// **gh #401.** The box is built with the *directional* presence test, so
    /// a bound past the opposite sentinel survives while a genuinely absent
    /// one becomes `∓∞`. Pins the case only the directional reading admits.
    ///
    /// This used to assert instead that `recover_bound_mults` walked the same
    /// `G`-row layout the builder emitted — a real hazard when the two agreed
    /// only by construction, and one that no longer exists: bounds are the
    /// box, and their multipliers come back in `z_lb`/`z_ub` with no layout
    /// to keep in step.
    #[test]
    fn the_box_is_built_with_the_directional_bound_test() {
        let prob = NlProblem {
            n: 2,
            m: 0,
            num_obj: 1,
            minimize: true,
            obj_nonlinear: Expr::Const(0.0),
            obj_linear: vec![(0, 1.0), (1, 1.0)],
            obj_constant: 0.0,
            con_nonlinear: vec![],
            con_linear: vec![],
            // x0: upper bound past the *lower* sentinel, no lower bound.
            // x1: an ordinary two-sided box.
            x_l: vec![-2e19, 0.0],
            x_u: vec![-5e20, 1.0],
            g_l: vec![],
            g_u: vec![],
            x0: vec![-6e20, 0.5],
            lambda0: vec![],
            suffixes: Default::default(),
            imported_funcs: Vec::new(),
            ampl_options: Vec::new(),
            nl_counts: None,
            var_names: Vec::new(),
            con_names: Vec::new(),
        };
        let qp = extract_qp(&prob).expect("extract");
        assert_eq!(qp.m_ineq(), 0, "bounds are the box, not `G` rows");
        // x0: no lower bound; a real upper bound past the *lower* sentinel.
        assert_eq!(qp.lb[0], f64::NEG_INFINITY);
        assert_eq!(qp.ub[0], -5e20);
        // x1: an ordinary two-sided box, carried through unchanged.
        assert_eq!(qp.lb[1], 0.0);
        assert_eq!(qp.ub[1], 1.0);
    }

    /// The bound multipliers a solve produces are handed back per variable,
    /// not decoded from a row layout. A short `z_lb`/`z_ub` (a driver that
    /// returned early without them) reads as "no bound active" rather than
    /// panicking on the index.
    #[test]
    fn bound_multipliers_come_back_per_variable() {
        let prob = NlProblem {
            n: 2,
            m: 0,
            num_obj: 1,
            minimize: true,
            obj_nonlinear: Expr::Const(0.0),
            obj_linear: vec![(0, 1.0), (1, 1.0)],
            obj_constant: 0.0,
            con_nonlinear: vec![],
            con_linear: vec![],
            x_l: vec![0.0, 0.0],
            x_u: vec![1.0, 1.0],
            g_l: vec![],
            g_u: vec![],
            x0: vec![0.5, 0.5],
            lambda0: vec![],
            suffixes: Default::default(),
            imported_funcs: Vec::new(),
            ampl_options: Vec::new(),
            nl_counts: None,
            var_names: Vec::new(),
            con_names: Vec::new(),
        };
        let sol = QpSolution {
            status: pounce_convex::QpStatus::Optimal,
            x: vec![0.0, 1.0],
            y: vec![],
            z: vec![],
            z_lb: vec![7.0, 0.0],
            z_ub: vec![0.0, 9.0],
            obj: 0.0,
            iters: 0,
            iterates: Vec::new(),
        };
        let (z_lb, z_ub) = recover_bound_mults(&prob, &sol);
        assert_eq!(z_lb, vec![7.0, 0.0]);
        assert_eq!(z_ub, vec![0.0, 9.0]);

        let empty = QpSolution {
            z_lb: Vec::new(),
            z_ub: Vec::new(),
            ..sol
        };
        let (z_lb, z_ub) = recover_bound_mults(&prob, &empty);
        assert_eq!(z_lb, vec![0.0, 0.0]);
        assert_eq!(z_ub, vec![0.0, 0.0]);
    }
}
