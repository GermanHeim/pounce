//! gh#818 — the limited-memory arm on an unconstrained, ill-conditioned
//! convex quadratic.
//!
//! Reported against `pounce.minimize(f, x0, jac=g)`, which selects
//! `hessian_approximation=limited-memory` (no Hessian supplied) — the mode
//! the Python frontend and the CasADi plugin pick on their own. On a
//! 4-variable separable quadratic with `cond(H) = 1e8` the solve exhausted
//! `max_iter` where scipy's L-BFGS-B converged in 34 iterations *at the
//! same memory size*, so "scipy uses more memory" was not the difference.
//!
//! **What was actually wrong.** Not the model and not the linear algebra.
//! The published `B = σI + VVᵀ − UUᵀ` is the textbook L-BFGS matrix, and
//! the Sherman-Morrison-Woodbury solve reproduces `−B⁻¹∇f` to a relative
//! residual of ~1e-12. The cost was the *trial sequence*: the
//! quasi-Newton model understates the curvature along `d` by up to
//! `cond(H)`, so the acceptable step is `α ≈ 4e-6`, and upstream's fixed
//! `alpha *= alpha_red_factor` walks there in halves — 19–20 trial
//! points, each a full objective evaluation, every iteration. The
//! measurement that isolates it: under `alpha_red_factor 0.2` (nine
//! trials instead of twenty, nothing else changed) the 8-variable case
//! goes from `MaximumIterationsExceeded` at 2000 iterations to converged
//! at 1099.
//!
//! `BacktrackingLineSearch::next_alpha` replaces the fixed factor with a
//! safeguarded quadratic interpolation, defaulted on for the
//! limited-memory path only (`alpha_red_factor_min`) and gated behind
//! `ALPHA_INTERP_MIN_TRIALS` trial points; see those two symbols and
//! `AlgorithmBuilder`'s field doc for why the exact path keeps
//! upstream's sequence and why a short line search is left alone.
//!
//! **What this file is not evidence about.** The corpus here is
//! unconstrained and unbounded, so the filter never sees `θ > 0` and
//! restoration is never entered. It says nothing about the constrained
//! arm — `scripts/sweep-fixtures.sh` owns that, and the change is
//! deliberately confined to the leg it moves (exact leg byte-identical;
//! 3 lbfgs-leg lines move, none of them a status or a routing change).
//!
//! It is also not evidence that the change is free on this family. Over
//! a 32-cell sweep of it (`n ∈ {4, 8, 12, 20}` × cond ∈ {1e2, 1e4, 1e8,
//! 1e12} × `m ∈ {6, 10}`) one cell loses a status — `n = 8`, cond 1e12,
//! `m = 6` goes from a 2000-iteration loose-tolerance success to
//! `Diverging_Iterates` at 352, at every gate and every
//! `alpha_red_factor_min` measured. CHANGELOG.md carries the sweep and
//! the mechanism; `limited_memory_max_history 10` is the remedy, and
//! `eight_variable_high_memory_converges` below pins that it works.

use pounce_rs::builder::{Nlp, Problem};

/// `f(x) = Σ (sᵢxᵢ − 1)²`, separable and strictly convex, with
/// `x*ᵢ = 1/sᵢ` and `f* = 0` in closed form. `s = 10^linspace(0, 4, n)`
/// gives `cond(H) = (sₙ/s₁)² = 1e8`.
struct IllConditionedQuadratic {
    s: Vec<f64>,
}

impl IllConditionedQuadratic {
    fn new(n: usize) -> Self {
        Self {
            s: (0..n)
                .map(|i| 10f64.powf(4.0 * i as f64 / (n - 1) as f64))
                .collect(),
        }
    }
    fn solution(&self) -> Vec<f64> {
        self.s.iter().map(|&si| 1.0 / si).collect()
    }
}

impl Problem for IllConditionedQuadratic {
    fn objective(&self, x: &[f64]) -> f64 {
        self.s
            .iter()
            .zip(x)
            .map(|(&si, &xi)| (si * xi - 1.0).powi(2))
            .sum()
    }
    fn gradient(&self, x: &[f64], grad: &mut [f64]) -> bool {
        for (i, gi) in grad.iter_mut().enumerate() {
            *gi = 2.0 * self.s[i] * (self.s[i] * x[i] - 1.0);
        }
        true
    }
}

/// Solve at `n` variables, returning `(iterations, max relative error in
/// x, success)`. `extra`/`extra_num` set string / numeric options.
fn solve(n: usize, extra: &[(&str, &str)]) -> (i32, f64, bool) {
    solve_with(n, extra, &[])
}

fn solve_with(n: usize, extra: &[(&str, &str)], extra_num: &[(&str, f64)]) -> (i32, f64, bool) {
    let p = IllConditionedQuadratic::new(n);
    let x_star = p.solution();
    let mut nlp = Nlp::new(IllConditionedQuadratic::new(n))
        .x0(&vec![0.0; n])
        .option_int("max_iter", 2000)
        .option_int("print_level", 0)
        .option_str("hessian_approximation", "limited-memory");
    for (k, v) in extra {
        nlp = nlp.option_str(k, v);
    }
    for (k, v) in extra_num {
        nlp = nlp.option_num(k, *v);
    }
    let sol = nlp.solve();
    let rel = if sol.x.len() == n {
        (0..n)
            .map(|i| ((sol.x[i] - x_star[i]) / x_star[i]).abs())
            .fold(0.0f64, f64::max)
    } else {
        f64::INFINITY
    };
    (sol.stats.iteration_count, rel, sol.success)
}

/// The reported case. Before gh#818 this returned
/// `Maximum_Iterations_Exceeded` with `x` wrong by 3.7e-3 relative on
/// the reporter's machine, and 76 iterations here; scipy's L-BFGS-B
/// takes 34–37 at the same memory. The bound below is deliberately
/// loose — this is a "does not stall" assertion, not an iteration-count
/// pin — but it is well under the 76 the fixed-factor sequence needed,
/// so reverting `next_alpha` to `alpha *= alpha_red_factor` fails it.
#[test]
fn issue_818_four_variable_quadratic_converges_without_stalling() {
    let (iters, rel, ok) = solve(4, &[]);
    assert!(ok, "gh#818 4-variable quadratic did not converge");
    assert!(
        rel < 1e-6,
        "converged to the wrong point: max relative error {rel:.3e}"
    );
    assert!(
        iters <= 40,
        "took {iters} iterations; the fixed-factor backtracking took 76 \
         and scipy's L-BFGS-B takes 34-37 at the same memory size"
    );
}

/// The interpolation is a *default*, not a hard-coded policy, and the
/// escape hatch has to actually work: setting `alpha_red_factor_min`
/// equal to `alpha_red_factor` collapses `next_alpha`'s clamp and
/// restores upstream's fixed geometric sequence. If this assertion ever
/// reads "same trajectory", the option is registered and not reaching
/// the α-loop — which is exactly the gh#677 failure mode for
/// `limited_memory_initialization`.
#[test]
fn issue_818_alpha_red_factor_min_can_restore_the_fixed_sequence() {
    let (interp, _, _) = solve(4, &[]);
    let (fixed, _, _) = solve_with(4, &[], &[("alpha_red_factor_min", 0.5)]);
    assert_ne!(
        interp, fixed,
        "alpha_red_factor_min=0.5 (== alpha_red_factor) took the same \
         {interp} iterations as the interpolating default — the option is \
         parsed but not reaching the alpha loop"
    );
    assert!(
        fixed > interp,
        "the fixed-factor sequence took {fixed} iterations against the \
         interpolating {interp}; gh#818 is the claim that it is slower, so \
         a reversal here means the fix is not doing what its comment says"
    );
}

/// Interpolation must not be a special case of the reported model. The
/// same objective at a *benign* condition number was already fast, and
/// staying fast there is what rules out "the new trial sequence just
/// trades one regime for another".
#[test]
fn issue_818_well_conditioned_case_does_not_regress() {
    let p = IllConditionedQuadratic { s: vec![1.0; 6] };
    let sol = Nlp::new(IllConditionedQuadratic { s: vec![1.0; 6] })
        .x0(&vec![0.0; 6])
        .option_int("max_iter", 2000)
        .option_int("print_level", 0)
        .option_str("hessian_approximation", "limited-memory")
        .solve();
    assert!(sol.success, "unit-curvature quadratic did not converge");
    assert!(
        sol.stats.iteration_count <= 15,
        "{} iterations on a unit-curvature quadratic",
        sol.stats.iteration_count
    );
    let _ = p.objective(&sol.x);
}

/// `limited_memory_initialization=history-max` is the second half of
/// gh#818 and is deliberately **not** the default: it wins where the
/// window cannot span the spectrum and loses where it can, so it is an
/// option with a documented population rather than a new default. This
/// pins that it reaches the updater and changes the trajectory —
/// without it, `history-max` would be a registered no-op.
#[test]
fn issue_818_history_max_reaches_the_updater() {
    let (default_iters, _, _) = solve(4, &[]);
    let (hmax_iters, rel, ok) = solve(4, &[("limited_memory_initialization", "history-max")]);
    assert!(ok, "history-max did not converge on the gh#818 quadratic");
    assert!(
        rel < 1e-6,
        "history-max converged to the wrong point ({rel:.3e})"
    );
    assert_ne!(
        default_iters, hmax_iters,
        "history-max took the same {default_iters} iterations as scalar1 — \
         the value is registered but sigma is still read off the newest pair"
    );
}

/// The 8-variable case at the **default** memory, which the first draft
/// of this fix left open and which `ALPHA_INTERP_MIN_TRIALS` is what
/// closes. It is the constant's real test, and it pins it from both
/// sides: interpolating from the first trial leaves this at
/// `Maximum_Iterations_Exceeded`/2000 (the model never breaks out of
/// the tiny-`s` cycle), and so does any gate of 8 or more (the
/// interpolation stops firing where it was needed). Only a gate in the
/// neighbourhood of 5 converges it, in 1073 iterations. Raise, lower or
/// delete the constant and this test goes red; the fixture sweep will
/// not tell you, because no fixture in the corpus is an unconstrained
/// ill-conditioned quadratic.
#[test]
fn issue_818_eight_variable_default_memory_converges() {
    let (iters, rel, ok) = solve_with(8, &[], &[]);
    assert!(
        ok,
        "gh#818 8-variable quadratic at default memory did not converge \
         ({iters} iterations); the trial gate is what closes this case"
    );
    assert!(
        rel < 1e-4,
        "converged to the wrong point: max relative error {rel:.3e}"
    );
    assert!(
        iters < 1500,
        "took {iters} iterations against the 1073 measured for the \
         shipped gate; the fixed-factor sequence never converges here"
    );
}

/// The documented remedy for the cells this change does not fix — and
/// for the one it costs, `n = 8` at cond 1e12 with `m = 6`. Raising the
/// window is what turns a model too poor to produce a usable step into
/// one that is merely ill-conditioned. If this ever fails, the advice in
/// CHANGELOG.md and `docs/src/options.md` is wrong and the regression it
/// mitigates has no remedy left.
#[test]
fn issue_818_eight_variable_high_memory_converges() {
    let n = 8;
    let x_star = IllConditionedQuadratic::new(n).solution();
    let sol = Nlp::new(IllConditionedQuadratic::new(n))
        .x0(&vec![0.0; n])
        .option_int("max_iter", 2000)
        .option_int("print_level", 0)
        .option_int("limited_memory_max_history", 10)
        .option_str("hessian_approximation", "limited-memory")
        .solve();
    let iters = sol.stats.iteration_count;
    let rel = (0..n)
        .map(|i| ((sol.x[i] - x_star[i]) / x_star[i]).abs())
        .fold(0.0f64, f64::max);
    assert!(
        sol.success,
        "8-variable quadratic did not converge at m = 10"
    );
    assert!(rel < 1e-6, "wrong point at m = 10: {rel:.3e}");
    assert!(iters <= 120, "{iters} iterations at m = 10, expected ~41");
}
