# Rosenbrock-family iteration scaling: pounce vs reference Ipopt

## TL;DR — corrected

Pounce is **competitive** with reference Ipopt on chained Rosenbrock:

| problem (n=500) | iters | wall   |
| --------------- | ----- | ------ |
| pounce GENROSE  | 381   | 172 ms |
| ipopt GENROSE   | 376   | 153 ms |

The original alarm — `benchmarks/cutest/full_stderr.txt` reporting pounce at
14050 ms vs Ipopt at 197 ms on the same problem — is **stale data** from an
older pounce build and no longer reflects reality. The current cutest harness
needs a fresh sweep to retire those numbers.

The "slow Rosenbrock" we hit in the large-scale benchmark was caused by **two
unrelated, fixable mistakes in `benchmarks/large_scale/src/problems/chained_rosenbrock.rs`**, not by the algorithm:

1. **Wrong starting point.** Used `x_i = -1.2` (textbook 2-D Rosenbrock) or
   `x_i = -1.0`. Nash 1984 / CUTE `GENROSE` uses the linear ramp
   `x_i = i/(n+1)`. From -1.2 the iteration takes ~2× more steps.
2. **Asymmetric anchor.** Original formula put the `(1-x)²` term on the
   *source* of each pair (`(1 - x_i)²`); GENROSE puts it on the *target*
   (`(1 - x_{i+1})²`). The target-anchored version is the one Ipopt has
   been tuned against.

After the fix (Nash start + target-anchored formula), iteration counts match
CUTE GENROSE across n:

| n     | pounce iters | iters/n |
| ----- | ------------ | ------- |
| 250   | 199          | 0.80    |
| 500   | 387          | 0.77    |
| 1000  | 765          | 0.77    |
| 5000  | 3744         | 0.75    |

This **~0.75 iters/var** scaling is fundamental to chained Rosenbrock with
*any* local-Newton IPM — reference Ipopt scales the same way. Newton steps
propagate information one chain-link per iteration through the tridiagonal
Hessian.

## How the diagnosis ran off track

Initial trace of our broken formulation showed `lg(rg) = --` constantly,
suggesting we never apply Hessian regularization. That led to a suspicion
that the inertia check in `kkt/pd_full_space_solver.rs` was misfiring for
unconstrained problems.

The truth: when pounce is run on the *real* CUTE GENROSE, `lg(rg)` shows
values 0.7–2.4 throughout (regularization actively applied). Our IPM is
working correctly; the inertia check fires when needed. From the bad start
`x = -1.2`, the Hessian along the trajectory happens to stay PD, so no
regularization is needed — but Newton converges slowly because the iterate
is far from the optimum along a curved valley.

## Verified algorithm behavior (unchanged)

- `PdPerturbationHandler` (`crates/pounce-algorithm/src/kkt/perturbation_handler.rs`)
  — escalates `δ_x` when factorization returns `WrongInertia`.
- `PdFullSpaceSolver::solve_once`
  (`crates/pounce-algorithm/src/kkt/pd_full_space_solver.rs:483`) sets
  `check_inertia = neg_curv_test_tol <= 0.0` (default tol = 0.0 → always
  checks).
- `FeralSolverInterface::factor`
  (`crates/pounce-feral/src/lib.rs:200`) returns `WrongInertia` when the
  reported negative-eigenvalue count differs from the expected
  `num_neg_evals = y_c.dim() + y_d.dim()` (which is 0 for unconstrained).

## Follow-ups

1. ✅ Fix `chained_rosenbrock.rs` formulation + start (done).
2. **Refresh `benchmarks/cutest/full_stderr.txt`** — current numbers
   misrepresent pounce's actual performance on the CUTE suite. Worth a
   nightly rerun before the next status comparison.
3. **2× iteration gap from x=-1.2 start** is genuine but specific to
   pathological starts. Not actionable unless we plan to target hard
   initialization regimes.
