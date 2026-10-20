//! Square-system recovery: when the interior-point solve of a square,
//! constant-objective model fails, solve `c(x) = 0` directly.
//!
//! # Why
//!
//! A model with as many free variables as equality rows, no inequality rows
//! and a constant objective is not an optimization problem. It is a system of
//! nonlinear equations wearing an NLP's clothes, and every feasible point is a
//! global optimum. That is the class an equation-oriented flowsheet
//! initialization hands a solver (IDAES-style: fix the degrees of freedom,
//! solve the square block), and it is the class where an interior-point
//! failure has the least to do with the model: the barrier, the filter and
//! the Hessian approximation all exist to trade objective against
//! feasibility, and here there is no objective to trade.
//!
//! Measured on `square_flowsheet_resto` (32×32, `f ≡ 0`, feasible by
//! construction). Its Jacobian is nearly rank-deficient even at the solution
//! (`σ_min ≈ 3.6e-9` against `σ_max ≈ 3e6`; the near-null direction is a
//! uniform shift of the eight temperatures, pinned only by `exp(-6000/T)`),
//! so the solution set is nearly non-isolated. Under
//! `hessian_approximation=limited-memory` POUNCE on FERAL PR#222 runs to the
//! 3000-iteration cap and every rung of the second-opinion ladder fails too;
//! upstream Ipopt 3.14.19 declares local infeasibility at iteration 50. A
//! Levenberg–Marquardt least-squares solve that uses only `J` reaches
//! `max|c| ≈ 4e-10` in 18 iterations from the same start.
//!
//! # The gate
//!
//! [`probe`] opens this stage only when all of the following hold of the TNLP
//! the ladder was handed:
//!
//! * every row is an equality (`g_l == g_u`, finite) and there is at least
//!   one;
//! * the number of **free** variables — `x_l < x_u`, i.e. after removing the
//!   variables `fixed_variable_treatment` would remove — equals the row count;
//! * the objective is constant: the model does not *prove* `∇f` varies
//!   ([`TNLP::derivative_proofs`]), and `∇f` evaluates to exactly zero, every
//!   component, at the starting point.
//!
//! The last test is evaluated, not proved, and that is a deliberate choice.
//! A structural proof would be the stronger gate, but it is not available on
//! the path that matters: presolve wrappers decline `derivative_proofs` (they
//! change what the rows are), and the C, Python and CasADi surfaces hand
//! POUNCE callbacks, not algebra. What makes an evaluated gate safe is that
//! **nothing reported rests on it**. The gate only decides whether to spend
//! the attempt. The verdict rests on the certificate checked at the returned
//! point `x̂` itself (below), which includes `∇f(x̂) = 0` evaluated *there*:
//! at a feasible `x̂` with `∇f(x̂) = 0`, `(λ, z) = (0, 0)` satisfies
//! stationarity and complementarity exactly, so `x̂` is a first-order KKT
//! point by the same definition the IPM's convergence test uses — whether or
//! not `f` is constant anywhere else.
//!
//! # The algorithm
//!
//! A bounded, scaled trust-region Levenberg–Marquardt method on
//! `min ½‖c(x)‖²` over the free variables:
//!
//! * **Step.** Moré's trust-region form (Moré 1978, MINPACK `lmder`/`lmpar`):
//!   with column scaling `D`, the step `p̃ = D p` solves
//!   `(ÃᵀÃ + λI) p̃ = −Ãᵀc`, `Ã = J D⁻¹`, with `λ ≥ 0` chosen so
//!   `‖p̃‖ ≈ Δ` (within 10%) by safeguarded Newton iteration on the secular
//!   equation `1/‖p̃(λ)‖ = 1/Δ`. The Gauss–Newton step is taken when it
//!   already fits.
//! * **Linear algebra.** Each `λ` trial factors the quasi-definite augmented
//!   system `[[λI, Ãᵀ], [Ã, −I]]` with the workspace's sparse symmetric
//!   backend (FERAL) rather than forming `JᵀJ`, which squares the condition
//!   number and densifies. One extra back-solve against the same factor gives
//!   `(ÃᵀÃ + λI)⁻¹ p̃` for the Newton update. At most
//!   [`MAX_LAMBDA_TRIALS`] factorizations per iteration.
//! * **Scaling.** `D_j = max over iterates of ‖J_{:,j}‖₂` (Moré's
//!   non-decreasing column norms), which makes the method invariant to
//!   rescaling a variable — the fixture's variables span `1e-6` to `9e6`.
//!   The rows are **not** reweighted: equilibrating them by `‖J_i(x₀)‖∞` was
//!   measured on the fixture and lost it from every start tried, while the
//!   unweighted merit (MINPACK's, and scipy's `method='lm'`) solves it.
//! * **Bounds.** The trial point is the projection of `x + p` onto the box,
//!   and the gain ratio is computed for the *projected* step against the
//!   linear model, so a projection that destroys the predicted decrease
//!   shrinks `Δ` — which drives `λ` up and the step toward the scaled
//!   (projected) gradient. Every iterate is inside the box by construction.
//!   This is the projected Levenberg–Marquardt idea of Kanzow, Yamashita &
//!   Fukushima (2004), in a trust-region rather than a line-search
//!   globalization.
//! * **Update.** Gain ratio `ρ = ared/pred`; accept when `ρ ≥ 1e-4`; halve
//!   `Δ` (to at most the step taken) when `ρ < 0.25`; set `Δ = 2‖D s‖` when
//!   `ρ > 0.75` or the step was Gauss–Newton. A non-finite residual at the
//!   trial point is a rejection with `Δ` quartered.
//!
//! # Termination and honesty
//!
//! The LM stops as soon as the iterate passes the acceptance test, or fails
//! on an iteration cap (`min(max_iter, 200)`), on a vanishing projected
//! gradient of `‖c‖²` at a non-root ([`STATIONARITY_TOL`]), on `Δ` collapsing to
//! `1e-15·max(‖Dx‖, 1)` (a projected stationary point of `‖c‖²` that is not a
//! root — a *local* least-squares minimum, so the original verdict stands),
//! or on an evaluation / factorization failure.
//!
//! The acceptance test is the IPM's own, at zero multipliers: Ipopt's
//! convergence test asks for the *scaled* overall error below `tol` and the
//! *unscaled* constraint violation below `constr_viol_tol`, and at
//! `(λ, z) = (0, 0)` with `∇f = 0` the dual and complementarity terms vanish,
//! leaving `max_i s_i|c_i| ≤ tol` and `max_i |c_i| ≤ constr_viol_tol`. `s_i`
//! is the row scaling the NLP arm applies, reproduced here: gradient-based
//! (`min(1, nlp_scaling_max_gradient/‖∇c_i‖∞)`, floored at
//! `nlp_scaling_min_value`, or `nlp_scaling_constr_target_gradient/‖∇c_i‖∞`
//! when that is set) measured at the projected start, the TNLP's own
//! `g_scaling` under `user-scaling`, and `1` otherwise — which for `none` is
//! exact and for the remaining methods is the unscaled test. On top of that
//! the box is satisfied **exactly** (projection; no `bound_relax_factor`
//! widening is borrowed), and `∇f(x̂)` must be exactly zero at the returned
//! point. If any check fails the stage reports failure and the caller keeps
//! the original verdict untouched.
//!
//! # What is reported
//!
//! `Solve_Succeeded` with `x̂`, `λ = 0` and `z_l = z_u = 0`. Those are true
//! KKT multipliers, not placeholders: with `∇f(x̂) = 0` they satisfy
//! `∇f + Jᵀλ − z_l + z_u = 0` exactly. They are unique when `J(x̂)` is
//! nonsingular and no bound is active; when `J` is singular (as on the
//! fixture, numerically) the multiplier set is a cone containing zero and
//! zero is its minimum-norm member. They are **not** the multipliers an IPM
//! would have reported for the same point, because the IPM never reached it.
//!
//! The root may differ from the one an IPM would have reached: a square
//! system can have several, and the fixture's nearly non-isolated solution
//! set puts the LM root about 11% (relative, in the temperature block) from
//! the generator's `x*`. With a constant objective each is a global optimum.

use pounce_common::options_list::OptionsList;
use pounce_common::types::{Index, Number};
use pounce_linsol::{ESymSolverStatus, SparseSymLinearSolverInterface};
use pounce_nlp::SolveStatistics;
use pounce_nlp::alg_types::SolverReturn;
use pounce_nlp::constant_derivatives::DerivativeProof;
use pounce_nlp::tnlp::{
    BoundsInfo, IndexStyle, IpoptCq, IpoptData, ScalingRequest, Solution, SparsityRequest,
    StartingPoint, TNLP,
};
use std::cell::RefCell;
use std::rc::Rc;

/// The label this stage carries in the second-opinion record (`tried`,
/// `promoted_by`, the sweep's `2nd=` column).
pub const SQUARE_SYSTEM_RUNG_LABEL: &str = "square_system_lm";

/// Factorizations allowed per LM iteration while solving the secular
/// equation for `λ` (MINPACK `lmpar` allows 10 Newton steps; the extra one
/// is the Gauss–Newton trial).
pub const MAX_LAMBDA_TRIALS: usize = 11;

/// The iteration cap is `min(max_iter, this)`. Measured on the fixture from
/// eight starts (`x*` scaled by 0.1 … 10 per block): 5 to 143 iterations.
pub const LM_ITERATION_CAP: usize = 200;

/// Smallest `λ` the augmented system is factored at. `λ = 0` makes the
/// Schur complement `ÃᵀÃ` exactly singular on a rank-deficient Jacobian,
/// and `Ã` is column-normalized, so `1e-16` is an `ε`-sized regularization.
const LAMBDA_FLOOR: Number = 1e-16;

/// MINPACK's `gtol` test: stop when every column of `Ã` is orthogonal to
/// `c` to within this cosine (the columns are normalized by `D`, so each
/// `|Ã_jᵀc|/‖c‖` lies in `[0, 1]`), counting only directions the box lets
/// the iterate move in. Set at the noise level on purpose: it exists for the
/// exact case — `degenerate_start_infeasible` starts where `J = 0`, and
/// without it the method spends 57 iterations halving `Δ` at a point no
/// step can leave — and must never fire on the near-null directions of a
/// nearly singular but consistent system like `square_flowsheet_resto`.
pub const STATIONARITY_TOL: Number = 1e-12;

/// A residual system `c: R^n → R^m` with a fixed sparse Jacobian pattern.
pub trait ResidualSystem {
    /// `c(x)` into `c`. `false` when the model cannot be evaluated at `x`.
    fn eval_c(&mut self, x: &[Number], c: &mut [Number]) -> bool;
    /// Jacobian values at `x`, in the order of [`LmProblem::jac_rows`].
    fn eval_jac(&mut self, x: &[Number], vals: &mut [Number]) -> bool;
}

/// Everything the LM core needs that is not a callback.
#[derive(Debug, Clone)]
pub struct LmProblem {
    pub n: usize,
    pub m: usize,
    /// Jacobian triplet pattern, 0-based. Duplicates are summed.
    pub jac_rows: Vec<usize>,
    pub jac_cols: Vec<usize>,
    /// Box; `±∞` for an absent bound.
    pub lb: Vec<Number>,
    pub ub: Vec<Number>,
}

/// The acceptance test, in the NLP arm's terms — see the module docs.
#[derive(Debug, Clone)]
pub struct FeasibilityTest {
    /// Per-row scaling `s_i` the scaled test multiplies by.
    pub row_scale: Vec<Number>,
    /// `tol`, against `max_i s_i |c_i|`.
    pub tol: Number,
    /// `constr_viol_tol`, against `max_i |c_i|`.
    pub constr_viol_tol: Number,
}

impl FeasibilityTest {
    /// `(scaled, unscaled)` max-norm violations.
    pub fn violations(&self, c: &[Number]) -> (Number, Number) {
        let mut scaled: Number = 0.0;
        let mut unscaled: Number = 0.0;
        for (i, &ci) in c.iter().enumerate() {
            let a = ci.abs();
            // A NaN must not read as feasible: `max` would drop it.
            if !a.is_finite() {
                return (Number::INFINITY, Number::INFINITY);
            }
            unscaled = unscaled.max(a);
            scaled = scaled.max(self.row_scale.get(i).copied().unwrap_or(1.0) * a);
        }
        (scaled, unscaled)
    }

    pub fn passes(&self, c: &[Number]) -> bool {
        let (s, u) = self.violations(c);
        s <= self.tol && u <= self.constr_viol_tol
    }
}

/// How an LM run ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LmStatus {
    /// The iterate passed the [`FeasibilityTest`].
    Converged,
    /// Iteration cap reached first.
    IterationLimit,
    /// The projected gradient of `½‖c‖²` vanished at a point that is not a
    /// root: a stationary point of the least-squares problem on the box
    /// (MINPACK's `gtol` test, at [`STATIONARITY_TOL`]). No step this method
    /// can take leaves it, so it stops rather than collapse `Δ` to get there.
    Stationary,
    /// `Δ` collapsed: a projected stationary point of `‖c‖²` that is not a
    /// root, as far as this method can tell.
    TrustRegionCollapsed,
    /// The residual or Jacobian could not be evaluated at an accepted point,
    /// or was not finite there.
    EvaluationFailed,
    /// The augmented system could not be factored at any `λ` tried.
    LinearSolverFailed,
}

/// What an LM run returns.
#[derive(Debug, Clone)]
pub struct LmResult {
    pub status: LmStatus,
    /// Last accepted iterate (inside the box).
    pub x: Vec<Number>,
    /// `c(x)` at that iterate.
    pub c: Vec<Number>,
    pub iterations: usize,
    pub residual_evals: usize,
    pub jacobian_evals: usize,
    pub factorizations: usize,
}

/// Merged Jacobian pattern plus the augmented-system triplets built on it.
struct Augmented {
    /// `slot[k]` = merged entry of TNLP triplet `k`.
    slot: Vec<usize>,
    /// Merged `(row, col)` pairs.
    uniq: Vec<(usize, usize)>,
    /// 1-based lower-triangle triplets of `[[λI, Ãᵀ], [Ã, −I]]`, laid out
    /// `n` diagonal, then the merged `Ã` entries, then `m` diagonal.
    irn: Vec<Index>,
    jcn: Vec<Index>,
}

impl Augmented {
    fn new(p: &LmProblem) -> Self {
        let mut index = std::collections::HashMap::with_capacity(p.jac_rows.len());
        let mut uniq = Vec::new();
        let mut slot = Vec::with_capacity(p.jac_rows.len());
        for (&i, &j) in p.jac_rows.iter().zip(&p.jac_cols) {
            let next = uniq.len();
            let s = *index.entry((i, j)).or_insert(next);
            if s == next {
                uniq.push((i, j));
            }
            slot.push(s);
        }
        let nnz = p.n + uniq.len() + p.m;
        let mut irn = Vec::with_capacity(nnz);
        let mut jcn = Vec::with_capacity(nnz);
        for j in 0..p.n {
            irn.push((j + 1) as Index);
            jcn.push((j + 1) as Index);
        }
        for &(i, j) in &uniq {
            irn.push((p.n + i + 1) as Index);
            jcn.push((j + 1) as Index);
        }
        for i in 0..p.m {
            irn.push((p.n + i + 1) as Index);
            jcn.push((p.n + i + 1) as Index);
        }
        Self {
            slot,
            uniq,
            irn,
            jcn,
        }
    }

    fn merge(&self, raw: &[Number], out: &mut [Number]) {
        out.iter_mut().for_each(|v| *v = 0.0);
        for (k, &v) in raw.iter().enumerate() {
            out[self.slot[k]] += v;
        }
    }
}

fn norm2(v: &[Number]) -> Number {
    v.iter().map(|a| a * a).sum::<Number>().sqrt()
}

fn project(x: &mut [Number], lb: &[Number], ub: &[Number]) {
    for j in 0..x.len() {
        if x[j] < lb[j] {
            x[j] = lb[j];
        }
        if x[j] > ub[j] {
            x[j] = ub[j];
        }
    }
}

/// The factored-system half of the LM: owns the backend and the scaled
/// Jacobian values for the current iterate.
struct StepSolver {
    backend: Box<dyn SparseSymLinearSolverInterface>,
    aug: Augmented,
    n: usize,
    m: usize,
    /// Merged `Ã = J D⁻¹` values, aligned with `aug.uniq`.
    a: Vec<Number>,
    factorizations: usize,
}

impl StepSolver {
    /// Factor `[[λI, Ãᵀ], [Ã, −I]]` and return `(p̃, (ÃᵀÃ+λI)⁻¹p̃)` for the
    /// right-hand side `[0; −c]`. `None` on a failed or non-finite solve.
    fn solve(&mut self, lambda: Number, c: &[Number]) -> Option<(Vec<Number>, Vec<Number>)> {
        let (n, m) = (self.n, self.m);
        {
            let vals = self.backend.values_array_mut();
            for v in vals.iter_mut().take(n) {
                *v = lambda;
            }
            vals[n..n + self.a.len()].copy_from_slice(&self.a);
            for v in vals.iter_mut().skip(n + self.a.len()) {
                *v = -1.0;
            }
        }
        let mut rhs = vec![0.0; n + m];
        for i in 0..m {
            rhs[n + i] = -c[i];
        }
        self.factorizations += 1;
        let st =
            self.backend
                .multi_solve(true, &self.aug.irn, &self.aug.jcn, 1, &mut rhs, false, 0);
        if st != ESymSolverStatus::Success || rhs.iter().any(|v| !v.is_finite()) {
            return None;
        }
        let p: Vec<Number> = rhs[..n].to_vec();
        let mut rhs2 = vec![0.0; n + m];
        rhs2[..n].copy_from_slice(&p);
        let st =
            self.backend
                .multi_solve(false, &self.aug.irn, &self.aug.jcn, 1, &mut rhs2, false, 0);
        if st != ESymSolverStatus::Success || rhs2.iter().any(|v| !v.is_finite()) {
            return None;
        }
        Some((p, rhs2[..n].to_vec()))
    }

    /// `‖Ãᵀc‖₂`, the scaled gradient norm — MINPACK's upper bound on the
    /// `λ` that fits the trust region is `‖Ãᵀc‖/Δ`.
    fn scaled_gradient_norm(&self, c: &[Number]) -> Number {
        let mut g = vec![0.0; self.n];
        for (k, &(i, j)) in self.aug.uniq.iter().enumerate() {
            g[j] += self.a[k] * c[i];
        }
        norm2(&g)
    }

    /// Is `x` a stationary point of `½‖c‖²` on the box? See
    /// [`STATIONARITY_TOL`].
    fn is_stationary(&self, c: &[Number], x: &[Number], lb: &[Number], ub: &[Number]) -> bool {
        let cn = norm2(c);
        // A root (or a NaN norm) is not a stationary non-root.
        if cn.is_nan() || cn <= 0.0 {
            return false;
        }
        let mut g = vec![0.0; self.n];
        for (k, &(i, j)) in self.aug.uniq.iter().enumerate() {
            g[j] += self.a[k] * c[i];
        }
        g.iter().enumerate().all(|(j, &gj)| {
            // Descent moves along −g: blocked at a lower bound when g > 0,
            // at an upper bound when g < 0.
            let blocked = (gj > 0.0 && x[j] <= lb[j]) || (gj < 0.0 && x[j] >= ub[j]);
            blocked || gj.abs() <= STATIONARITY_TOL * cn
        })
    }

    /// Moré's trust-region step: `λ` such that `‖p̃(λ)‖ ≈ Δ`.
    /// Returns `(p̃, λ)`, with `λ = 0` reporting a Gauss–Newton step.
    fn trust_region_step(
        &mut self,
        c: &[Number],
        delta: Number,
        lambda_prev: Number,
    ) -> Option<(Vec<Number>, Number)> {
        // Gauss–Newton first: if it fits, it is the step.
        let mut lo: Number = 0.0;
        if let Some((p, _)) = self.solve(LAMBDA_FLOOR, c)
            && norm2(&p) <= 1.1 * delta
        {
            return Some((p, 0.0));
        }
        let gnorm = self.scaled_gradient_norm(c);
        if !(gnorm.is_finite() && gnorm > 0.0) {
            return None;
        }
        let mut hi = gnorm / delta;
        let mut lambda = if lambda_prev > LAMBDA_FLOOR {
            lambda_prev.clamp(LAMBDA_FLOOR, hi)
        } else {
            (1e-3 * hi).max(LAMBDA_FLOOR)
        };
        let mut best: Option<(Vec<Number>, Number)> = None;
        for _ in 0..MAX_LAMBDA_TRIALS - 1 {
            let Some((p, u)) = self.solve(lambda, c) else {
                // A failed factorization at this λ: more regularization is
                // the only direction that can help.
                lo = lambda;
                lambda = (10.0 * lambda).max(1e-12);
                continue;
            };
            let pn = norm2(&p);
            let fits = (pn - delta).abs() <= 0.1 * delta;
            // Keep the last inside-the-region candidate as a fallback; any
            // `p̃` with `‖p̃‖ ≤ 1.1Δ` is an acceptable (if conservative) step.
            if pn <= 1.1 * delta {
                best = Some((p.clone(), lambda));
            }
            if fits {
                return Some((p, lambda));
            }
            if pn > delta {
                lo = lo.max(lambda);
            } else {
                hi = hi.min(lambda);
            }
            let pu: Number = p.iter().zip(&u).map(|(a, b)| a * b).sum();
            let newton = if pu > 0.0 {
                lambda + (pn * pn / pu) * ((pn - delta) / delta)
            } else {
                Number::NAN
            };
            lambda = if newton.is_finite() && newton > lo && newton < hi {
                newton
            } else {
                // Outside the bracket: MINPACK's safeguard.
                (lo * hi).sqrt().max(1e-3 * hi).max(LAMBDA_FLOOR)
            };
        }
        if best.is_none() {
            // Out of trials without a fitting step. At `hi` the step is
            // inside the region by construction; take it.
            if let Some((p, _)) = self.solve(hi, c) {
                best = Some((p, hi));
            }
        }
        best
    }
}

/// Solve `c(x) = 0` over the box by bounded trust-region LM. See the module
/// docs for the method, the scaling and the termination rules.
pub fn bounded_levenberg_marquardt(
    model: &mut dyn ResidualSystem,
    prob: &LmProblem,
    x0: &[Number],
    test: &FeasibilityTest,
    max_iter: usize,
    mut backend: Box<dyn SparseSymLinearSolverInterface>,
) -> LmResult {
    let (n, m) = (prob.n, prob.m);
    let mut x = x0.to_vec();
    project(&mut x, &prob.lb, &prob.ub);
    let mut c = vec![0.0; m];
    let mut out = LmResult {
        status: LmStatus::EvaluationFailed,
        x: x.clone(),
        c: c.clone(),
        iterations: 0,
        residual_evals: 0,
        jacobian_evals: 0,
        factorizations: 0,
    };
    out.residual_evals += 1;
    if !model.eval_c(&x, &mut c) || c.iter().any(|v| !v.is_finite()) {
        return out;
    }
    out.c = c.clone();
    if test.passes(&c) {
        out.status = LmStatus::Converged;
        return out;
    }

    let aug = Augmented::new(prob);
    let nnz = aug.irn.len();
    if backend.initialize_structure((n + m) as Index, nnz as Index, &aug.irn, &aug.jcn)
        != ESymSolverStatus::Success
    {
        out.status = LmStatus::LinearSolverFailed;
        return out;
    }
    let n_uniq = aug.uniq.len();
    let mut solver = StepSolver {
        backend,
        aug,
        n,
        m,
        a: vec![0.0; n_uniq],
        factorizations: 0,
    };

    let mut raw = vec![0.0; prob.jac_rows.len()];
    let mut jac = vec![0.0; n_uniq];
    let mut d = vec![0.0; n];
    // Evaluate J at `x`, merge duplicates, raise the column norms, and
    // refresh `Ã`. `false` on a failed or non-finite Jacobian.
    let refresh = |model: &mut dyn ResidualSystem,
                   x: &[Number],
                   raw: &mut [Number],
                   jac: &mut [Number],
                   d: &mut [Number],
                   solver: &mut StepSolver|
     -> bool {
        if !model.eval_jac(x, raw) || raw.iter().any(|v| !v.is_finite()) {
            return false;
        }
        solver.aug.merge(raw, jac);
        let mut col = vec![0.0; d.len()];
        for (k, &(_, j)) in solver.aug.uniq.iter().enumerate() {
            col[j] += jac[k] * jac[k];
        }
        for j in 0..d.len() {
            d[j] = d[j].max(col[j].sqrt());
        }
        for (k, &(_, j)) in solver.aug.uniq.iter().enumerate() {
            let dj = if d[j] > 0.0 { d[j] } else { 1.0 };
            solver.a[k] = jac[k] / dj;
        }
        true
    };
    out.jacobian_evals += 1;
    if !refresh(model, &x, &mut raw, &mut jac, &mut d, &mut solver) {
        out.factorizations = solver.factorizations;
        return out;
    }
    let dscale = |d: &[Number], j: usize| if d[j] > 0.0 { d[j] } else { 1.0 };
    let scaled_norm = |d: &[Number], v: &[Number]| -> Number {
        v.iter()
            .enumerate()
            .map(|(j, &vj)| (dscale(d, j) * vj).powi(2))
            .sum::<Number>()
            .sqrt()
    };
    // MINPACK's `factor = 100` initial radius.
    let xnorm = scaled_norm(&d, &x);
    let mut delta = if xnorm > 0.0 { 100.0 * xnorm } else { 100.0 };
    let mut lambda: Number = 0.0;
    let mut ct = vec![0.0; m];
    let mut status = LmStatus::IterationLimit;
    for iter in 0..max_iter {
        if solver.is_stationary(&c, &x, &prob.lb, &prob.ub) {
            status = LmStatus::Stationary;
            break;
        }
        out.iterations = iter + 1;
        let Some((pt, lam)) = solver.trust_region_step(&c, delta, lambda) else {
            status = LmStatus::LinearSolverFailed;
            break;
        };
        lambda = lam;
        let mut xt: Vec<Number> = (0..n).map(|j| x[j] + pt[j] / dscale(&d, j)).collect();
        project(&mut xt, &prob.lb, &prob.ub);
        let s: Vec<Number> = (0..n).map(|j| xt[j] - x[j]).collect();
        let snorm = scaled_norm(&d, &s);

        out.residual_evals += 1;
        let ok = model.eval_c(&xt, &mut ct) && ct.iter().all(|v| v.is_finite());
        let rho = if ok {
            let phi: Number = c.iter().map(|v| v * v).sum();
            let phit: Number = ct.iter().map(|v| v * v).sum();
            // Linear model at the *projected* step.
            let mut lin = c.clone();
            for (k, &(i, j)) in solver.aug.uniq.iter().enumerate() {
                lin[i] += jac[k] * s[j];
            }
            let pred = phi - lin.iter().map(|v| v * v).sum::<Number>();
            if pred > 0.0 {
                (phi - phit) / pred
            } else {
                Number::NEG_INFINITY
            }
        } else {
            Number::NEG_INFINITY
        };

        if !ok {
            delta *= 0.25;
        } else if rho < 0.25 {
            delta = 0.5 * if snorm > 0.0 { delta.min(snorm) } else { delta };
        } else if rho > 0.75 || lambda == 0.0 {
            delta = delta.max(2.0 * snorm);
        }

        if rho >= 1e-4 {
            x = xt;
            c.copy_from_slice(&ct);
            if test.passes(&c) {
                status = LmStatus::Converged;
                break;
            }
            out.jacobian_evals += 1;
            if !refresh(model, &x, &mut raw, &mut jac, &mut d, &mut solver) {
                status = LmStatus::EvaluationFailed;
                break;
            }
        }
        let floor = 1e-15 * scaled_norm(&d, &x).max(1.0);
        // Written to catch a NaN radius too.
        if delta.is_nan() || delta <= floor {
            status = LmStatus::TrustRegionCollapsed;
            break;
        }
    }
    out.status = status;
    out.x = x;
    out.c = c;
    out.factorizations = solver.factorizations;
    out
}

/// Why [`probe`] did not open the stage.
#[derive(Debug, Clone, PartialEq)]
pub enum SquareSystemDecline {
    /// `square_system_recovery=no`.
    Disabled,
    /// The TNLP could not be queried (info, bounds, start, gradient).
    Unavailable,
    /// No rows, or a row that is not an equality.
    NotEqualityConstrained,
    /// Free variables and equality rows differ in number.
    NotSquare { free: usize, rows: usize },
    /// `∇f` is proved varying or is nonzero at the start.
    ObjectiveNotConstant,
}

/// A square, equality-only, constant-objective model, read off the TNLP and
/// ready to hand to [`run`].
#[derive(Debug, Clone)]
pub struct SquareSystemPlan {
    n_full: usize,
    m: usize,
    /// Full-space index of each free variable.
    free: Vec<usize>,
    /// Full-space starting point, projected into the box; fixed variables
    /// hold their fixed value.
    x_full: Vec<Number>,
    g_rhs: Vec<Number>,
    /// `(TNLP triplet index, row, free column)` for every Jacobian entry in
    /// a free column.
    kept: Vec<(usize, usize, usize)>,
    nnz_full: usize,
    lb: Vec<Number>,
    ub: Vec<Number>,
}

fn opt_num(options: &OptionsList, tag: &str, default: Number) -> Number {
    options
        .get_numeric_value(tag, "")
        .map(|(v, _)| v)
        .unwrap_or(default)
}

/// Is the stage switched on? Read as a literal tag so the option-wiring
/// scans see it.
pub fn is_enabled(options: &OptionsList) -> bool {
    options
        .get_bool_value("square_system_recovery", "")
        .map(|(v, _found)| v)
        .unwrap_or(true)
}

/// Decide whether the model the ladder was handed is in this stage's class.
/// Costs one `get_nlp_info`, one `get_bounds_info` and — only on a model that
/// is square and equality-only — one starting point, one Jacobian structure
/// query and one `∇f` evaluation.
pub fn probe(
    tnlp: &Rc<RefCell<dyn TNLP>>,
    options: &OptionsList,
) -> Result<SquareSystemPlan, SquareSystemDecline> {
    if !is_enabled(options) {
        return Err(SquareSystemDecline::Disabled);
    }
    let lo_inf = opt_num(options, "nlp_lower_bound_inf", -1e19);
    let up_inf = opt_num(options, "nlp_upper_bound_inf", 1e19);
    let mut t = tnlp.borrow_mut();
    let info = t.get_nlp_info().ok_or(SquareSystemDecline::Unavailable)?;
    let n = info.n.max(0) as usize;
    let m = info.m.max(0) as usize;
    if m == 0 {
        return Err(SquareSystemDecline::NotEqualityConstrained);
    }
    let mut x_l = vec![0.0; n];
    let mut x_u = vec![0.0; n];
    let mut g_l = vec![0.0; m];
    let mut g_u = vec![0.0; m];
    if !t.get_bounds_info(BoundsInfo {
        x_l: &mut x_l,
        x_u: &mut x_u,
        g_l: &mut g_l,
        g_u: &mut g_u,
    }) {
        return Err(SquareSystemDecline::Unavailable);
    }
    if (0..m).any(|i| !(g_l[i].is_finite() && g_l[i] == g_u[i])) {
        return Err(SquareSystemDecline::NotEqualityConstrained);
    }
    let lb: Vec<Number> = x_l
        .iter()
        .map(|&v| if v <= lo_inf { Number::NEG_INFINITY } else { v })
        .collect();
    let ub: Vec<Number> = x_u
        .iter()
        .map(|&v| if v >= up_inf { Number::INFINITY } else { v })
        .collect();
    if (0..n).any(|j| lb[j] > ub[j]) {
        return Err(SquareSystemDecline::Unavailable);
    }
    let free: Vec<usize> = (0..n).filter(|&j| lb[j] < ub[j]).collect();
    if free.len() != m {
        return Err(SquareSystemDecline::NotSquare {
            free: free.len(),
            rows: m,
        });
    }
    // Cheapest objective test first: a model that proves `∇f` varies is out
    // without an evaluation.
    if t.derivative_proofs().grad_f == DerivativeProof::Varying {
        return Err(SquareSystemDecline::ObjectiveNotConstant);
    }
    let mut x = vec![0.0; n];
    let mut z_l = vec![0.0; n];
    let mut z_u = vec![0.0; n];
    let mut lam = vec![0.0; m];
    if !t.get_starting_point(StartingPoint {
        init_x: true,
        x: &mut x,
        init_z: false,
        z_l: &mut z_l,
        z_u: &mut z_u,
        init_lambda: false,
        lambda: &mut lam,
    }) || x.iter().any(|v| !v.is_finite())
    {
        return Err(SquareSystemDecline::Unavailable);
    }
    for j in 0..n {
        if lb[j] == ub[j] {
            x[j] = lb[j];
        }
    }
    project(&mut x, &lb, &ub);
    let mut grad = vec![0.0; n];
    if !t.eval_grad_f(&x, true, &mut grad) {
        return Err(SquareSystemDecline::Unavailable);
    }
    // Exactly zero, every component: "constant" is the claim, and a small
    // gradient is a different class (the stationarity term would then not
    // vanish at λ = 0).
    if grad.iter().any(|&v| v != 0.0) {
        return Err(SquareSystemDecline::ObjectiveNotConstant);
    }
    let nnz = info.nnz_jac_g.max(0) as usize;
    let mut irow = vec![0 as Index; nnz];
    let mut jcol = vec![0 as Index; nnz];
    if !t.eval_jac_g(
        None,
        false,
        SparsityRequest::Structure {
            irow: &mut irow,
            jcol: &mut jcol,
        },
    ) {
        return Err(SquareSystemDecline::Unavailable);
    }
    let index_base: Index = match info.index_style {
        IndexStyle::Fortran => 1,
        _ => 0,
    };
    let mut free_of = vec![usize::MAX; n];
    for (k, &j) in free.iter().enumerate() {
        free_of[j] = k;
    }
    let mut kept = Vec::with_capacity(nnz);
    for k in 0..nnz {
        let i = (irow[k] - index_base) as usize;
        let j = (jcol[k] - index_base) as usize;
        if i >= m || j >= n {
            return Err(SquareSystemDecline::Unavailable);
        }
        if free_of[j] != usize::MAX {
            kept.push((k, i, free_of[j]));
        }
    }
    Ok(SquareSystemPlan {
        n_full: n,
        m,
        lb: free.iter().map(|&j| lb[j]).collect(),
        ub: free.iter().map(|&j| ub[j]).collect(),
        free,
        x_full: x,
        g_rhs: g_l,
        kept,
        nnz_full: nnz,
    })
}

/// The TNLP seen as a [`ResidualSystem`] over the free variables.
struct TnlpResidual<'a> {
    tnlp: &'a Rc<RefCell<dyn TNLP>>,
    plan: &'a SquareSystemPlan,
    x_full: Vec<Number>,
    g: Vec<Number>,
    jac_full: Vec<Number>,
}

impl TnlpResidual<'_> {
    fn scatter(&mut self, x: &[Number]) {
        for (k, &j) in self.plan.free.iter().enumerate() {
            self.x_full[j] = x[k];
        }
    }
}

impl ResidualSystem for TnlpResidual<'_> {
    fn eval_c(&mut self, x: &[Number], c: &mut [Number]) -> bool {
        self.scatter(x);
        if !self
            .tnlp
            .borrow_mut()
            .eval_g(&self.x_full, true, &mut self.g)
        {
            return false;
        }
        for ((ci, gi), ri) in c.iter_mut().zip(&self.g).zip(&self.plan.g_rhs) {
            *ci = gi - ri;
        }
        true
    }

    fn eval_jac(&mut self, x: &[Number], vals: &mut [Number]) -> bool {
        self.scatter(x);
        if !self.tnlp.borrow_mut().eval_jac_g(
            Some(&self.x_full),
            true,
            SparsityRequest::Values {
                values: &mut self.jac_full,
            },
        ) {
            return false;
        }
        for (q, &(k, _, _)) in self.plan.kept.iter().enumerate() {
            vals[q] = self.jac_full[k];
        }
        true
    }
}

/// The row scaling `s_i` the NLP arm's scaled convergence test applies —
/// see the module docs for which methods are reproduced and how.
fn acceptance_row_scale(
    tnlp: &Rc<RefCell<dyn TNLP>>,
    options: &OptionsList,
    plan: &SquareSystemPlan,
    jac_at_start: &[Number],
) -> Vec<Number> {
    let m = plan.m;
    let method = options
        .get_string_value("nlp_scaling_method", "")
        .map(|(v, _)| v)
        .unwrap_or_else(|_| "gradient-based".to_string());
    match method.as_str() {
        "gradient-based" => {
            let max_grad = opt_num(options, "nlp_scaling_max_gradient", 100.0);
            let target = opt_num(options, "nlp_scaling_constr_target_gradient", 0.0);
            let min_value = opt_num(options, "nlp_scaling_min_value", 1e-8);
            let mut row_max = vec![0.0 as Number; m];
            for (q, &(_, i, _)) in plan.kept.iter().enumerate() {
                row_max[i] = row_max[i].max(jac_at_start[q].abs());
            }
            row_max
                .iter()
                .map(|&g| {
                    let s = if target > 0.0 {
                        if g > 0.0 { target / g } else { 1.0 }
                    } else {
                        max_grad / g.max(max_grad)
                    };
                    s.max(min_value)
                })
                .collect()
        }
        "user-scaling" => {
            let mut obj = 1.0;
            let mut use_x = false;
            let mut xs = vec![1.0; plan.n_full];
            let mut use_g = false;
            let mut gs = vec![1.0; m];
            let ok = tnlp.borrow_mut().get_scaling_parameters(ScalingRequest {
                obj_scaling: &mut obj,
                use_x_scaling: &mut use_x,
                x_scaling: &mut xs,
                use_g_scaling: &mut use_g,
                g_scaling: &mut gs,
            });
            if ok && use_g {
                gs.iter().map(|v| v.abs()).collect()
            } else {
                vec![1.0; m]
            }
        }
        _ => vec![1.0; m],
    }
}

/// What [`run`] reports.
#[derive(Debug, Clone)]
pub struct SquareSystemOutcome {
    pub lm: LmResult,
    /// The certificate held and the solution was handed to the TNLP's
    /// `finalize_solution`.
    pub accepted: bool,
    /// One line for the console when not accepted.
    pub reason: String,
    /// Statistics for the promoted verdict, populated on acceptance.
    pub statistics: SolveStatistics,
    /// `(scaled, unscaled)` violation at the returned point.
    pub violation: (Number, Number),
}

/// Run the LM on a probed model and, if the result passes every check in the
/// module docs, finalize it through `tnlp` as `Solve_Succeeded`.
///
/// On any failure `tnlp` receives no `finalize_solution` call, so the caller's
/// captured solution stays the one its original verdict describes.
pub fn run(
    plan: &SquareSystemPlan,
    tnlp: &Rc<RefCell<dyn TNLP>>,
    options: &OptionsList,
    backend: Box<dyn SparseSymLinearSolverInterface>,
) -> SquareSystemOutcome {
    let wall0 = pounce_common::utils::wallclock_time();
    let cpu0 = pounce_common::utils::cpu_time();
    let max_iter = options
        .get_integer_value("max_iter", "")
        .map(|(v, _)| v.max(0) as usize)
        .unwrap_or(3000)
        .min(LM_ITERATION_CAP);
    let prob = LmProblem {
        n: plan.free.len(),
        m: plan.m,
        jac_rows: plan.kept.iter().map(|&(_, i, _)| i).collect(),
        jac_cols: plan.kept.iter().map(|&(_, _, j)| j).collect(),
        lb: plan.lb.clone(),
        ub: plan.ub.clone(),
    };
    let x0: Vec<Number> = plan.free.iter().map(|&j| plan.x_full[j]).collect();
    let mut model = TnlpResidual {
        tnlp,
        plan,
        x_full: plan.x_full.clone(),
        g: vec![0.0; plan.m],
        jac_full: vec![0.0; plan.nnz_full],
    };
    let fail = |lm: LmResult, reason: String| SquareSystemOutcome {
        violation: (Number::NAN, Number::NAN),
        lm,
        accepted: false,
        reason,
        statistics: SolveStatistics::default(),
    };
    let mut jac0 = vec![0.0; prob.jac_rows.len()];
    if !model.eval_jac(&x0, &mut jac0) {
        let lm = LmResult {
            status: LmStatus::EvaluationFailed,
            x: x0.clone(),
            c: vec![0.0; plan.m],
            iterations: 0,
            residual_evals: 0,
            jacobian_evals: 1,
            factorizations: 0,
        };
        return fail(
            lm,
            "the Jacobian could not be evaluated at the start".into(),
        );
    }
    let test = FeasibilityTest {
        row_scale: acceptance_row_scale(tnlp, options, plan, &jac0),
        tol: opt_num(options, "tol", 1e-8),
        constr_viol_tol: opt_num(options, "constr_viol_tol", 1e-4),
    };
    let lm = bounded_levenberg_marquardt(&mut model, &prob, &x0, &test, max_iter, backend);
    if lm.status != LmStatus::Converged {
        let reason = format!("{:?} after {} iteration(s)", lm.status, lm.iterations);
        return fail(lm, reason);
    }

    // The certificate, re-checked at the returned point from fresh
    // evaluations rather than read off the LM's bookkeeping.
    model.scatter(&lm.x);
    let x_full = model.x_full.clone();
    let mut g = vec![0.0; plan.m];
    let mut c = vec![0.0; plan.m];
    if !tnlp.borrow_mut().eval_g(&x_full, true, &mut g) {
        return fail(lm, "eval_g failed at the returned point".into());
    }
    for i in 0..plan.m {
        c[i] = g[i] - plan.g_rhs[i];
    }
    if !test.passes(&c) {
        return fail(
            lm,
            "the returned point does not pass the feasibility test".into(),
        );
    }
    if (0..prob.n).any(|k| !(lm.x[k] >= prob.lb[k] && lm.x[k] <= prob.ub[k])) {
        return fail(lm, "the returned point is outside the variable box".into());
    }
    let mut grad = vec![0.0; plan.n_full];
    if !tnlp.borrow_mut().eval_grad_f(&x_full, false, &mut grad) || grad.iter().any(|&v| v != 0.0) {
        return fail(
            lm,
            "the objective gradient is nonzero at the returned point".into(),
        );
    }
    let Some(obj) = tnlp.borrow_mut().eval_f(&x_full, false) else {
        return fail(lm, "eval_f failed at the returned point".into());
    };
    if !obj.is_finite() {
        return fail(
            lm,
            "the objective is not finite at the returned point".into(),
        );
    }

    let (scaled, unscaled) = test.violations(&c);
    let zeros_n = vec![0.0; plan.n_full];
    let zeros_m = vec![0.0; plan.m];
    tnlp.borrow_mut().finalize_solution(
        Solution {
            status: SolverReturn::Success,
            x: &x_full,
            z_l: &zeros_n,
            z_u: &zeros_n,
            g: &g,
            lambda: &zeros_m,
            obj_value: obj,
        },
        &IpoptData::default(),
        &IpoptCq::default(),
    );

    let statistics = SolveStatistics {
        iteration_count: lm.iterations as Index,
        total_cpu_time_secs: pounce_common::utils::cpu_time() - cpu0,
        total_wallclock_time_secs: pounce_common::utils::wallclock_time() - wall0,
        num_obj_evals: 1,
        num_constr_evals: (lm.residual_evals + 1) as Index,
        num_obj_grad_evals: 1,
        num_constr_jac_evals: (lm.jacobian_evals + 1) as Index,
        final_objective: obj,
        final_scaled_objective: obj,
        final_dual_inf: 0.0,
        final_constr_viol: scaled,
        final_compl: 0.0,
        final_kkt_error: scaled,
        final_unscaled_dual_inf: 0.0,
        final_unscaled_constr_viol: unscaled,
        final_unscaled_compl: 0.0,
        final_unscaled_kkt_error: unscaled,
        final_kkt_error_above_noise: scaled,
        final_declared_constr_viol: unscaled,
        final_declared_box_viol: 0.0,
        final_mu: 0.0,
        ..SolveStatistics::default()
    };
    SquareSystemOutcome {
        lm,
        accepted: true,
        reason: String::new(),
        statistics,
        violation: (scaled, unscaled),
    }
}

/// The backend the stage factors with: FERAL, configured from the same
/// `feral_*` options as the IPM, and with **no** summary sink, so the
/// linear-solver summary a caller reads keeps describing the interior-point
/// solves rather than absorbing this stage's factorizations.
pub fn default_backend(options: &OptionsList) -> Box<dyn SparseSymLinearSolverInterface> {
    Box::new(pounce_feral::FeralSolverInterface::with_config(
        crate::application::feral_config_from_options(options),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Dense-pattern test system from a closure pair.
    struct Sys<F, J>
    where
        F: FnMut(&[Number], &mut [Number]),
        J: FnMut(&[Number], &mut [Number]),
    {
        f: F,
        j: J,
    }

    impl<F, J> ResidualSystem for Sys<F, J>
    where
        F: FnMut(&[Number], &mut [Number]),
        J: FnMut(&[Number], &mut [Number]),
    {
        fn eval_c(&mut self, x: &[Number], c: &mut [Number]) -> bool {
            (self.f)(x, c);
            true
        }
        fn eval_jac(&mut self, x: &[Number], v: &mut [Number]) -> bool {
            (self.j)(x, v);
            true
        }
    }

    fn dense2(lb: [Number; 2], ub: [Number; 2]) -> LmProblem {
        LmProblem {
            n: 2,
            m: 2,
            jac_rows: vec![0, 0, 1, 1],
            jac_cols: vec![0, 1, 0, 1],
            lb: lb.to_vec(),
            ub: ub.to_vec(),
        }
    }

    fn strict() -> FeasibilityTest {
        FeasibilityTest {
            row_scale: vec![1.0, 1.0],
            tol: 1e-10,
            constr_viol_tol: 1e-10,
        }
    }

    fn backend() -> Box<dyn SparseSymLinearSolverInterface> {
        Box::new(pounce_feral::FeralSolverInterface::with_config(
            pounce_feral::FeralConfig::default(),
        ))
    }

    /// The root sits **on** a bound, and the unconstrained Newton steps from
    /// this start overshoot it: `x0² + x1² = 2`, `x0 = x1`, `x0 ≤ 1`, from
    /// `(0.2, 3)`. Projection must clip the overshoot onto the bound and the
    /// method must still converge — to the root on the bound, not to the
    /// mirror root `(−1, −1)` and not to a point outside the box.
    #[test]
    fn converges_to_a_root_on_an_active_bound() {
        let prob = dense2([-10.0, -10.0], [1.0, 10.0]);
        let mut sys = Sys {
            f: |x: &[Number], c: &mut [Number]| {
                c[0] = x[0] * x[0] + x[1] * x[1] - 2.0;
                c[1] = x[0] - x[1];
            },
            j: |x: &[Number], v: &mut [Number]| {
                v.copy_from_slice(&[2.0 * x[0], 2.0 * x[1], 1.0, -1.0]);
            },
        };
        let r =
            bounded_levenberg_marquardt(&mut sys, &prob, &[0.2, 3.0], &strict(), 200, backend());
        assert_eq!(r.status, LmStatus::Converged, "{r:?}");
        assert!(r.x[0] <= 1.0, "x0 = {} left the box", r.x[0]);
        assert!(
            (r.x[0] - 1.0).abs() < 1e-9 && (r.x[1] - 1.0).abs() < 1e-9,
            "{:?}",
            r.x
        );
    }

    /// The honesty half of the bound case: the only root is outside the box,
    /// so the method must say so rather than report the clipped point.
    /// `x0 = 2`, `x1 = 0` over `x0 ≤ 1` has no root in the box; the
    /// least-squares minimum on the box is `(1, 0)` with `|c| = 1`.
    #[test]
    fn a_root_outside_the_box_is_not_reported_as_converged() {
        let prob = dense2([-10.0, -10.0], [1.0, 10.0]);
        let mut sys = Sys {
            f: |x: &[Number], c: &mut [Number]| {
                c[0] = x[0] - 2.0;
                c[1] = x[1];
            },
            j: |_x: &[Number], v: &mut [Number]| v.copy_from_slice(&[1.0, 0.0, 0.0, 1.0]),
        };
        let r =
            bounded_levenberg_marquardt(&mut sys, &prob, &[0.0, 0.5], &strict(), 200, backend());
        assert_eq!(r.status, LmStatus::Stationary, "{r:?}");
        assert!(r.x[0] <= 1.0);
        assert!(
            (r.c[0] + 1.0).abs() < 1e-8,
            "should stall at the box's nearest point: {r:?}"
        );
    }

    /// `degenerate_start_infeasible`'s shape: `x0² + x1² = −25`, `x0·x1 = 9`
    /// from the origin, where `J = 0`. No step leaves a point whose gradient
    /// is exactly zero, so the method must stop at once on stationarity —
    /// not report the origin, and not spend its budget halving `Δ`.
    #[test]
    fn a_zero_jacobian_start_stops_on_stationarity_at_once() {
        let prob = dense2([Number::NEG_INFINITY; 2], [Number::INFINITY; 2]);
        let mut sys = Sys {
            f: |x: &[Number], c: &mut [Number]| {
                c[0] = x[0] * x[0] + x[1] * x[1] + 25.0;
                c[1] = x[0] * x[1] - 9.0;
            },
            j: |x: &[Number], v: &mut [Number]| {
                v.copy_from_slice(&[2.0 * x[0], 2.0 * x[1], x[1], x[0]]);
            },
        };
        let r =
            bounded_levenberg_marquardt(&mut sys, &prob, &[0.0, 0.0], &strict(), 200, backend());
        assert_eq!(r.status, LmStatus::Stationary, "{r:?}");
        assert_eq!(r.iterations, 0);
    }

    /// Rank-deficient everywhere on the solution set: `x0 = x1` and
    /// `x0² = x1²` share the root set `{x0 = x1}`, and the Jacobian
    /// `[[1, −1], [2x0, −2x1]]` is singular at every root. Gauss–Newton's
    /// system is singular there; the LM's `λ` keeps the augmented factor
    /// well-posed and the method still drives `‖c‖` to zero (the
    /// error-bound regime of Yamashita & Fukushima). The root it lands on
    /// is not unique, so only feasibility is asserted.
    #[test]
    fn converges_on_a_rank_deficient_non_isolated_root_set() {
        let prob = dense2([Number::NEG_INFINITY; 2], [Number::INFINITY; 2]);
        let mut sys = Sys {
            f: |x: &[Number], c: &mut [Number]| {
                c[0] = x[0] - x[1];
                c[1] = x[0] * x[0] - x[1] * x[1];
            },
            j: |x: &[Number], v: &mut [Number]| {
                v.copy_from_slice(&[1.0, -1.0, 2.0 * x[0], -2.0 * x[1]]);
            },
        };
        let r =
            bounded_levenberg_marquardt(&mut sys, &prob, &[3.0, -1.0], &strict(), 200, backend());
        assert_eq!(r.status, LmStatus::Converged, "{r:?}");
        assert!((r.x[0] - r.x[1]).abs() < 1e-10, "{:?}", r.x);
    }

    /// The same deficiency, structurally: a duplicated row (`J` has rank 1
    /// at every point) with duplicate triplets in the pattern, which must be
    /// summed rather than double-placed in the augmented system.
    #[test]
    fn duplicate_triplets_are_summed_and_a_repeated_row_is_tolerated() {
        let prob = LmProblem {
            n: 2,
            m: 2,
            // Row 0 lists (0, 0) twice, half the coefficient each time.
            jac_rows: vec![0, 0, 0, 1, 1],
            jac_cols: vec![0, 0, 1, 0, 1],
            lb: vec![Number::NEG_INFINITY; 2],
            ub: vec![Number::INFINITY; 2],
        };
        let mut sys = Sys {
            f: |x: &[Number], c: &mut [Number]| {
                c[0] = x[0] + x[1] - 2.0;
                c[1] = 2.0 * (x[0] + x[1] - 2.0);
            },
            j: |_x: &[Number], v: &mut [Number]| v.copy_from_slice(&[0.5, 0.5, 1.0, 2.0, 2.0]),
        };
        let r =
            bounded_levenberg_marquardt(&mut sys, &prob, &[5.0, 7.0], &strict(), 200, backend());
        assert_eq!(r.status, LmStatus::Converged, "{r:?}");
        assert!((r.x[0] + r.x[1] - 2.0).abs() < 1e-10);
    }

    /// The scaled test is the NLP arm's: a row's violation is judged after
    /// its `s_i`, and the unscaled violation separately against
    /// `constr_viol_tol`. A NaN never passes.
    #[test]
    fn feasibility_test_applies_both_halves_and_rejects_nan() {
        let t = FeasibilityTest {
            row_scale: vec![1e-3, 1.0],
            tol: 1e-8,
            constr_viol_tol: 1e-4,
        };
        assert!(t.passes(&[5e-6, 1e-9]), "5e-6 · 1e-3 = 5e-9 ≤ tol");
        assert!(
            !t.passes(&[5e-4, 0.0]),
            "scaled passes but unscaled 5e-4 > constr_viol_tol"
        );
        assert!(!t.passes(&[0.0, 2e-8]), "unscaled row over tol");
        assert!(!t.passes(&[Number::NAN, 0.0]));
    }
}
