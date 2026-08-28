//! A watchdog trial iterate must not be reported as `Diverging_Iterates`.
//!
//! Found reviewing gh#818's line-search change, which did not cause the
//! defect but reached it: on the review's sweep of the issue's
//! ill-conditioned quadratic, cells exited `Diverging_Iterates` on a
//! problem bounded below by `f* = 0`, at points where the objective was
//! climbing through `+1e42`.
//!
//! **The point they exited on was not an iterate the algorithm had
//! kept.** `BacktrackingLineSearch::handle_watchdog_failure`'s
//! `accept-anyway` branch (info char `'w'`, mirroring upstream
//! `IpBacktrackingLineSearch.cpp:498-503`) promotes a trial the acceptor
//! *rejected*, deliberately does not augment the filter, and holds a
//! snapshot of the pre-watchdog iterate and direction. Within
//! `watchdog_trial_iter_max` (default 3) iterations it either finds the
//! gamble paid off or runs `StopWatchDog` and reverts. The divergence
//! guard in `IpoptAlgorithm::iterate` ran on those provisional points,
//! and `DIVERGENCE_ABS_RUNAWAY` (`1e18`) fires without the growth /
//! descent streak that `DIVERGENCE_PERSIST_ITERS` otherwise requires —
//! so a single watchdog gamble that overshot was enough.
//!
//! **Measured on the two fixtures below.** With the guard removed, each
//! stops on the *third* `'w'` row of one watchdog sequence, and the
//! printed iteration block is byte-identical to the guarded run's up to
//! that row — the guard only defers, it does not steer:
//!
//! | | exit | `|x|_∞` | objective at exit | next iteration, guarded |
//! |---|---|---|---|---|
//! | `twelve_variable` | 162 | 3.95e20 | **+1.72e42** | 163: `9.89e10` |
//! | `ten_variable` | 326 | 2.45e21 | **+7.11e44** | 327: `3.39e19` |
//!
//! A climbing objective is the *opposite* of what `Diverging_Iterates`
//! asserts — Ipopt's unboundedness verdict, the AMPL 300 "unbounded"
//! range, means `f → −∞` — and in both cases the very next iteration is
//! `StopWatchDog` throwing the excursion away, by 32 and 25 orders of
//! magnitude respectively. The guard was reporting which gamble
//! overshot furthest, not which problem was unbounded.
//!
//! **What it costs to be wrong here is the whole solve, twice over.**
//! `Diverging_Iterates` is terminal, so it also denies the gh#815
//! restoration ladder its retry: `ten_variable`'s first attempt fails
//! either way, and with the guard in place the ladder re-solves and
//! converges to a maximum relative error of 5.9e-9. Removing the guard
//! turns that into an unboundedness verdict on a strictly convex
//! quadratic.
//!
//! **What this test is not evidence about.** It pins the false *positive*
//! only. That the guard still fires on a genuine ray is owned by
//! `pounce-algorithm/tests/repro_issue248.rs` (transient excursion, must
//! not fire), `repro_issue252.rs` (decelerating descent, must not fire),
//! `repro_issue285.rs` (checked recession ray, must fire) and
//! `pounce-cli/tests/issue_314_unbounded_cubic_not_solved.rs` (must
//! fire) — the skip defers those verdicts, it does not remove them, and
//! `IpoptAlgorithm::WATCHDOG_DEFER_MAX` caps the deferral at four
//! consecutive iterations so a stale `in_watchdog` cannot hold it open.
//!
//! It is also not a claim that any one *cell* of this family reaches the
//! branch. Which cells do is chaotic in `ALPHA_INTERP_MIN_TRIALS` and in
//! `alpha_red_factor_min` — the cell this file was written against
//! (`n = 8`, cond 1e12, `m = 6`) stopped reaching it when the gate moved
//! from 5 to 6, which is exactly how a mutation-checked test quietly
//! stops testing anything. Both fixtures below were re-derived by
//! scanning `n` × cond × `m` × `alpha_red_factor_min` **under the
//! mutation** for cells that still reach it, and one of them holds the
//! shipped defaults. Re-run that scan if either goes green under the
//! mutation table.
//!
//! Mutation check: replace the `in_watchdog` binding in the divergence
//! block of `ipopt_alg.rs` with `false`, and both tests below fail with
//! `DivergingIterates` — at iteration 162 and 326 respectively.
//!
//! That binding gates the skip in **two** places — the `(amax,
//! structural_free, is_ray)` match arm and the `fire_*` binding — and
//! they are redundant, not complementary: mutating *either one alone*
//! leaves both tests green, because the other still suppresses the
//! verdict. So this file is evidence that the skip happens, and is not
//! evidence that both halves are load-bearing. They are kept because
//! each answers a different question — the match arm avoids evaluating
//! `curr_is_recession_ray` on a point that will be thrown away, and the
//! `fire_*` branch stops the streaks being *fed* a trial iterate — but
//! only the pair of them is pinned here.

use pounce_rs::ApplicationReturnStatus;
use pounce_rs::builder::{Nlp, Problem};

/// `f(x) = Σ (sᵢxᵢ − 1)²` with `s = 10^linspace(0, cond_exp/2, n)`, so
/// `cond(H) = 10^cond_exp`. Unconstrained and unbounded above, bounded
/// below by `f* = 0` at `x*ᵢ = 1/sᵢ` — an unbounded *feasible region*,
/// which is what makes the guard's structural-freedom check pass and
/// leaves the magnitude test as the only thing between these runs and a
/// false `Diverging_Iterates`.
struct IllConditionedQuadratic {
    s: Vec<f64>,
}

impl IllConditionedQuadratic {
    fn new(n: usize, cond_exp: u32) -> Self {
        let e = f64::from(cond_exp) / 2.0;
        Self {
            s: (0..n)
                .map(|i| 10f64.powf(e * i as f64 / (n - 1) as f64))
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
    fn gradient(&self, x: &[f64], g: &mut [f64]) -> bool {
        for (i, gi) in g.iter_mut().enumerate() {
            *gi = 2.0 * self.s[i] * (self.s[i] * x[i] - 1.0);
        }
        true
    }
}

/// Solve, returning `(status, iterations, objective, max relative error
/// in x)`. `alpha_red_factor_min` is left at its default unless given.
fn solve(
    n: usize,
    cond_exp: u32,
    memory: i32,
    alpha_red_factor_min: Option<f64>,
) -> (ApplicationReturnStatus, i32, f64, f64) {
    let x_star = IllConditionedQuadratic::new(n, cond_exp).solution();
    let mut nlp = Nlp::new(IllConditionedQuadratic::new(n, cond_exp))
        .x0(&vec![0.0; n])
        .option_str("hessian_approximation", "limited-memory")
        .option_int("limited_memory_max_history", memory)
        .option_int("max_iter", 2000)
        .option_int("print_level", 0);
    if let Some(a) = alpha_red_factor_min {
        nlp = nlp.option_num("alpha_red_factor_min", a);
    }
    let sol = nlp.solve();
    let rel = if sol.x.len() == n {
        (0..n)
            .map(|i| ((sol.x[i] - x_star[i]) / x_star[i]).abs())
            .fold(0.0f64, f64::max)
    } else {
        f64::INFINITY
    };
    (sol.status, sol.stats.iteration_count, sol.objective, rel)
}

/// The shipped-defaults arm: nothing set beyond the limited-memory mode
/// itself, so `alpha_red_factor_min` is the `0.05` an embedder gets
/// without asking. Guard removed, this stops at iteration 162 with
/// `obj = 1.72e42`; guarded, iteration 163 is the watchdog's own revert
/// to `9.89e10` and the solve ends `Solved_To_Acceptable_Level` at 197.
#[test]
fn a_watchdog_trial_excursion_is_not_reported_as_unbounded_twelve_variable() {
    let (status, iters, obj, rel) = solve(12, 14, 10, None);

    assert_ne!(
        status,
        ApplicationReturnStatus::DivergingIterates,
        "a problem bounded below by f* = 0 was reported unbounded, on a \
         watchdog trial point the line search had already rejected \
         (it = {iters}, obj = {obj:e})"
    );
    // Measured: `SolvedToAcceptableLevel` at 197, obj 3.74e-6, max
    // relative error 1.7e-3. The bars are loose because the point of the
    // test is the *status*, but they have to exclude "terminated
    // somewhere else, equally wrong" — the pre-fix exit carried
    // obj = 1.7e42.
    assert!(
        obj < 1e-4 && rel < 1e-1,
        "expected the neighbourhood of f* = 0; got obj = {obj:e}, max \
         relative error {rel:.3e}, status {status:?} at iteration {iters}"
    );
}

/// The other branch, and the more expensive failure: here the first
/// attempt fails either way, and it is the gh#815 restoration ladder's
/// re-solve that converges. `Diverging_Iterates` is terminal, so the
/// false verdict does not merely mislabel a failure — it takes the
/// retry that would have succeeded. Guard removed, this stops at
/// iteration 326 with `obj = 7.11e44`.
#[test]
fn a_terminal_watchdog_verdict_does_not_deny_the_ladder_its_retry() {
    let (status, iters, obj, rel) = solve(10, 12, 10, Some(0.1));

    assert_ne!(
        status,
        ApplicationReturnStatus::DivergingIterates,
        "a problem bounded below by f* = 0 was reported unbounded \
         (it = {iters}, obj = {obj:e})"
    );
    // Measured: `SolveSucceeded`, obj 7.09e-17, max relative error
    // 5.85e-9 — the right answer, reached only because the solve was
    // allowed to continue.
    assert!(
        rel < 1e-6,
        "expected convergence to x* after the ladder's retry; got max \
         relative error {rel:.3e} (obj = {obj:e}, status {status:?}, \
         iteration {iters})"
    );
}
