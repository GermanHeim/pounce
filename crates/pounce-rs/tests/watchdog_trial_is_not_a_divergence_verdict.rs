//! A watchdog trial iterate must not be reported as `Diverging_Iterates`.
//!
//! Found reviewing gh#818's line-search change, which did not cause the
//! defect but reached it: on the review's 32-cell sweep of the issue's
//! quadratic, one cell — `n = 8`, `cond(H) = 1e12`,
//! `limited_memory_max_history 6` — exited `Diverging_Iterates` at
//! iteration 352 where the pre-#818 trial sequence ran to `max_iter` and
//! returned a converged answer.
//!
//! **The point it exited on was not an iterate the algorithm had kept.**
//! `BacktrackingLineSearch::handle_watchdog_failure`'s `accept-anyway`
//! branch (info char `'w'`, mirroring upstream
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
//! Measured on the fixture below, before the fix: iterations 350, 351,
//! 352 are all `'w'` rows of one watchdog sequence; at 352 `|x|_∞ ≈ 5e22`
//! and the objective has climbed to `+2.0e45`. That is the *opposite* of
//! what `Diverging_Iterates` asserts — Ipopt's unboundedness verdict, the
//! AMPL 300 "unbounded" range, means `f → −∞` — and one iteration later
//! `StopWatchDog` would have restored an iterate at `f = 2.26e4`. The
//! same run without the interpolation reaches `f = 1.7e27` on a watchdog
//! trial of its own and survives only because `|x|` happened to stay
//! under `1e20`: the guard was reporting which excursion got luckier, not
//! which problem was unbounded.
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
//! Mutation check: drop the `in_watchdog` guard from either half of the
//! divergence block in `ipopt_alg.rs` (the `(amax, structural_free,
//! is_ray)` match arm or the `fire_*` binding) and this test goes red
//! with `DivergingIterates` at 352.

use pounce_rs::ApplicationReturnStatus;
use pounce_rs::builder::{Nlp, Problem};

/// `f(x) = Σ (sᵢxᵢ − 1)²` with `s = 10^linspace(0, 6, n)`, so
/// `cond(H) = 1e12`. Unconstrained and unbounded above, bounded below by
/// `f* = 0` at `x*ᵢ = 1/sᵢ` — an unbounded *feasible region*, which is
/// what makes the guard's structural-freedom check pass and leaves the
/// magnitude test as the only thing between this run and a false
/// `Diverging_Iterates`.
struct IllConditionedQuadratic {
    s: Vec<f64>,
}

impl IllConditionedQuadratic {
    fn new(n: usize) -> Self {
        Self {
            s: (0..n)
                .map(|i| 10f64.powf(6.0 * i as f64 / (n - 1) as f64))
                .collect(),
        }
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

#[test]
fn a_watchdog_trial_excursion_is_not_reported_as_unbounded() {
    let n = 8;
    let sol = Nlp::new(IllConditionedQuadratic::new(n))
        .x0(&vec![0.0; n])
        .option_str("hessian_approximation", "limited-memory")
        .option_int("limited_memory_max_history", 6)
        .option_int("max_iter", 2000)
        .option_int("print_level", 0)
        .solve();

    assert_ne!(
        sol.status,
        ApplicationReturnStatus::DivergingIterates,
        "a problem bounded below by f* = 0 was reported unbounded, on a \
         watchdog trial point the line search had already rejected \
         (it = {}, obj = {:e})",
        sol.stats.iteration_count,
        sol.objective,
    );
    // Measured: `SolveSucceeded` with `obj = 3.16e-15` at `max_iter`. The bar
    // is loose because the point of the test is the *status*, but it has
    // to exclude "terminated somewhere else, equally wrong" — the
    // pre-fix exit carried `obj = 2.0e45`.
    assert!(
        sol.objective < 1e-6,
        "expected the solve to reach the neighbourhood of f* = 0; got \
         obj = {:e} with status {:?} at iteration {}",
        sol.objective,
        sol.status,
        sol.stats.iteration_count,
    );
}
