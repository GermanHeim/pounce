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
//! arm — `scripts/sweep-fixtures.sh` owns that. Measured against
//! `a5e0a837` over 156 fixture-legs, the exact leg is byte-identical
//! (`alpha_red_factor_min` resolves to `alpha_red_factor` there) and
//! four lbfgs-leg lines move: `cresc4` gains a status
//! (`RestorationFailed`/1323 -> `SolveSucceeded`/281), `deb7` shortens
//! 1242 -> 1010 at the same status, `eigena2` 252 -> 201 at the same
//! status, and `hs13_bigstart` moves an objective digit at the same
//! iteration count. No line loses a status and none changes engine.
//!
//! It is also not evidence that the change is free on this family. Over
//! a 32-cell sweep of it (`n ∈ {4, 8, 12, 20}` × cond ∈ {1e2, 1e4, 1e8,
//! 1e12} × `m ∈ {6, 10}`) no cell loses a status and two gain one, but
//! 8 of the 30 same-status cells cost iterations — `n = 8` cond 1e4
//! `m = 6` is the worst at 646 → 822. CHANGELOG.md carries the grid.
//!
//! The sweep is also how the divergence-guard defect in
//! `pounce-rs/tests/watchdog_trial_is_not_a_divergence_verdict.rs` was
//! found: `n = 8`, cond 1e12, `m = 6` read as a status regression here
//! (`Diverging_Iterates` at 352) at every gate and every
//! `alpha_red_factor_min` measured, because the thing that had moved was
//! not in this crate at all.

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

/// The 8-variable case at the **default** memory `m = 6`, which the
/// issue thread left open and which this change does **not** close.
///
/// An earlier revision of this test asserted that it converges, and
/// that `ALPHA_INTERP_MIN_TRIALS` in "the neighbourhood of 5" is what
/// closes it. It does converge at 5 — and that is a coincidence, not a
/// property. Swept over the two knobs that reach this line search
/// (release build, `max_iter = 2000`; `MaxIter` is
/// `Maximum_Iterations_Exceeded` at 2000, i.e. did not converge):
///
/// | gate \ `alpha_red_factor_min` | 0.05 | 0.1 | 0.2 |
/// |---|---|---|---|
/// | 5 | 894 | 388 | `MaxIter` |
/// | **6** (shipped) | `MaxIter` | 1459 | `MaxIter` |
/// | 7 | 922 | `MaxIter` | `MaxIter` |
/// | 8 | `MaxIter` | `MaxIter` | 751 |
/// | 9 | `MaxIter` | `MaxIter` | `MaxIter` |
///
/// Five of fifteen cells converge, in no pattern in either knob, and
/// the counts that do span 388 to 1459. Over the same fifteen cells
/// `n = 4` converges in 21–36 iterations and `n = 8` at `m = 10` in
/// 47–134 — **every** cell. Those two are properties of the fix. This
/// one is a chaotic trajectory that sometimes escapes, and pinning it
/// tunes the constant to this cell rather than to the corpus: it is
/// what held the gate at 5, whose measured cost is 32% more solver
/// evaluations on `_rastrigin_eq` and two red assertions in
/// `python/tests/test_starts_racing.py`.
///
/// What is stable is that the case does not **stall**, which is what
/// gh#818 is actually about. From `f(x0) = 8` it reaches `f ~ 9e-13`
/// with `x` correct to 6e-7 relative — the solution to about six
/// significant figures. It exhausts `max_iter` because it cannot drive
/// dual infeasibility on the stiff coordinates far enough to *certify*
/// what it has already found, not because it is stuck; the fixed
/// sequence gets to the same place (`f ~ 2e-12`, 1.2e-6 relative) and
/// also cannot certify it. The documented remedy is memory, and
/// `issue_818_eight_variable_high_memory_converges` is where that is
/// pinned.
///
/// `success` is deliberately asserted in neither direction: converging
/// here would be a welcome improvement and must not fail this test,
/// and requiring it re-creates the coincidence above.
#[test]
fn issue_818_eight_variable_default_memory_does_not_stall() {
    let (iters, rel, _ok) = solve_with(8, &[], &[]);
    assert!(
        rel < 1e-5,
        "gh#818 8-variable quadratic at default memory got only to {rel:.3e} \
         relative after {iters} iterations; it is expected to reach the \
         solution to ~6 significant figures even when it cannot certify it, \
         so this is a stall, which is the defect gh#818 reported"
    );
}

/// The documented remedy for the cells this change does not fix, and for
/// the eight it slows: raising the window is what turns a model too poor
/// to produce a usable step into one that is merely ill-conditioned. At
/// `m = 10` this case takes 41 iterations against 1073 at `m = 6`. If
/// this ever fails, the advice in CHANGELOG.md and
/// `docs/src/options.md` is wrong.
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
