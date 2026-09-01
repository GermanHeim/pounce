# gh#884: why the biactive dual divergence is not a targeted fix

Status: **gh#884 open.** This note records what was measured while
attempting a fix, three approaches that are ruled out by measurement, and
why the issue's acceptance criteria as written are not jointly
satisfiable. It exists so the next attempt starts from the measurements
rather than from the same three ideas.

Guards: `crates/pounce-algorithm/tests/issue_884_biactive_dual_divergence.rs`
(four characterization tests, all measured on `qpec_small` and `ralph1`).

## The mechanism, sharpened

gh#884 describes the symptom — the primal reaches the exact solution and
the dual diverges. The cause is one step further back, and it is a
feedback loop.

`qpec_small` under `prod_eq` from `origin`, default options, traced per
iteration. `dual` is the `s_d`-normalised dual term that
`curr_barrier_error` actually reads; `kappa_eps_mu = barrier_tol_factor · mu`
is what it must fall below for `mu` to decrease:

| it | `mu` | `prim` | `compl` | `dual_raw` | `s_d` | `dual` | `kappa_eps_mu` |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1e-1 | 2.0e-2 | 1.9e0 | 1.5e0 | 1.0e0 | 1.5e0 | 1.0 |
| 3 | 1e-1 | 8.1e-5 | 1.1e-2 | 1.8e3 | 1.8e1 | 9.8e1 | 1.0 |
| 5 | 1e-1 | 4.1e-6 | 8.0e-7 | 3.2e6 | 1.8e5 | 1.8e1 | 1.0 |
| 7 | 1e-1 | 5.0e-8 | 8.0e-11 | 6.3e10 | 1.7e9 | 3.6e1 | 1.0 |
| 9 | 1e-1 | 1.0e-11 | 6.8e-11 | 7.8e11 | 3.5e9 | **2.2e2** | 1.0 |
| 11 | 1e-1 | 1.0e-15 | 8.0e-11 | 9.1e7 | 2.7e9 | 3.4e-2 | 1.0 |

Read the columns together:

1. **`prim` and `compl` are converged from iteration 5 on** — `1e-11`
   and `1e-11`. Neither ever blocks anything.
2. **`mu` is pinned at `1e-1` for eleven iterations**, because the
   barrier-subproblem test is `sub_err <= kappa_eps_mu` and `sub_err` is
   the `dual` column, which oscillates `1.5 → 98 → 18 → 36 → 222` and
   does not fall below `1.0` until iteration 11.
3. **The multipliers are `z = mu / s`.** The equality product row drags
   the slack of the *parallel* inequality row (`G2 >= 0`) to `~1e-15`
   while `mu` stays at `1e-1`, so `z ~ 1e14`.
4. `s_d` is the mean multiplier magnitude, so it grows in lockstep with
   the very multipliers that are exploding. `dual_raw / s_d` therefore
   stays `O(1..100)` — normalised, but never *converging*.

That is the loop: the dual error cannot fall because the multipliers are
exploding; the multipliers explode because `mu` is pinned; `mu` is pinned
because the dual error cannot fall.

The linear dependence that starts it is visible in the returned
multipliers. Rows 2 and 5 both restrict only `y2`; rows 1 and 4 are
likewise parallel:

| row | `|grad|_inf` | `lambda` | product |
|---|---:|---:|---:|
| 1 (`H1 >= 0`) | `2.0` | `-2.089e10` | `4.18e10` |
| 4 (`G1·H1 = 0`) | `2.0` | `+2.089e10` | `4.18e10` |
| 2 (`G2 >= 0`) | `1.0` | `-7.283e11` | `7.28e11` |
| 5 (`G2·H2 = 0`) | `2.3e-4` | `+1.737e15` | `4.03e11` |

`-7.283e11 + 4.03e11 = -3.25e11`, the reported residual. Note **no
regularization ever engages** — `reg = 0` at every iteration — because
the KKT matrix is singular only in the limit, never at any finite
iterate. The inertia-driven trigger cannot see this coming.

## Ruled out 1: the Hessian sparsity hypothesis. **Refuted.**

gh#884 records that the same model exits differently through the MPCC
harness's Python path (`Error_In_Step_Computation`, 118 iterations) than
through `.nl`/CLI (`Solved_To_Acceptable_Level`, 41), and names the
Hessian sparsity difference as the leading hypothesis — the harness hands
back a dense lower triangle with 6 nonzeros where the structural pattern
has 5, since `(2,1)` is identically zero. It marks this the cheap
prerequisite, to be done before the fix.

It is done, and it is not the cause. Driven from Rust with the two
patterns as the only difference, the results are **bit-identical**: same
status, same 14 iterations, same returned iterate to every digit. Pinned
by `the_hessian_sparsity_pattern_does_not_change_the_outcome`.

Whatever separates the two paths is somewhere else, and the ownership
bucket's description still needs re-checking against a harness
re-measurement — that part of gh#884 stands.

## Ruled out 2: detect the runaway and engage `delta_c`. **Too late by construction.**

Dual regularization *is* the effective remedy. With `perturb_always_cd=yes`
the same model converges in 19 iterations to an **unscaled** KKT error of
`9.96e-8` at `x = (0.999994, 0.999997, 3.7e-6)` — an honest solve, not a
residual normalised away. So an answer is reachable and gh#884 is a real
POUNCE gap. Pinned by `dual_regularization_reaches_the_answer_honestly`.

The obvious refinement is to engage it *only* when the runaway is
detected, leaving every other model alone. That does not work, and the
reason is structural rather than a matter of tuning. Engaging `delta_c`
from iteration `N`:

| engage from | outcome |
|---:|---|
| 3 | `SolveSucceeded`, 18 iters |
| 6 | `SolveSucceeded`, 42 iters |
| 8 | **`RestorationFailed`** |
| 9 | `SolveSucceeded`, 35 iters |
| 10 | **`RestorationFailed`** |
| 11 | **`RestorationFailed`** |

Non-monotone, and failing from exactly the region where the pattern first
becomes *visible*. By the time the multipliers have reached `~1e11` —
which is what a detector keyed on "the dual is running away" must wait
for — regularization can no longer recover the iterate. A detector needs
the blow-up to have happened; the fix needs to precede it.

## Ruled out 3: make `perturb_always_cd` the default. **Trades an honest failure for a silent wrong answer.**

If it must be engaged from the start, the question becomes whether it can
be on by default. It cannot.

`ralph1` under the `direct` lowering (`G·H <= 0`), `f* = 0` at the
origin, which is M-stationary but **not** S-stationary — NLP KKT *is*
S-stationarity, so no sign-feasible multiplier exists there and failing
is correct:

| | status | objective | unscaled KKT |
|---|---|---:|---:|
| default | `RestorationFailed` (39 iters) | `1.22e-3` | `1.89e10` |
| `perturb_always_cd=yes`, `max_iter=300` | `SolvedToAcceptableLevel` | `-3.81e-5` | `2.44e-6` |
| `perturb_always_cd=yes`, `max_iter=3000` | **`SolveSucceeded`** (356 iters) | **`-2.71e-5`** | `5.25e-7` |

The last row is the disqualifying one: plain `Solve_Succeeded` at an
objective **below `f* = 0`**, reachable only at a point the MPCC does not
contain. Raising the cap makes it worse, not better — the acceptable-level
exit at 300 iterations was the cap hiding it.

This is gh#884's own acceptance criterion firing ("`ralph1` must still
fail — a fix that greens all eight has over-fired"), and it is the P1a
lesson again: gh#884's current failure is *honest*, and this would
replace it with a success at the wrong answer. Pinned by
`always_on_dual_regularization_reports_success_below_the_true_optimum`.

`min -exp(x) s.t. x >= 0` — gh#274's own reproducer — is unaffected
either way (`ErrorInStepComputation`, identical iterate). That criterion
was never the binding one.

## Why no solver-side signal separates the two

`qpec_small` and `ralph1` are **structurally the same problem**:

- both have a constraint Jacobian that is rank-deficient at the limit
  point *and* at the start (`ralph1`'s product row has gradient exactly
  `(0,0)` at the origin);
- both drive `prim` to `~1e-16` while `dual_raw` explodes;
- both hold `mu` while that happens.

The trajectories differ in one observable — `‖d‖` settles to `5e-8` on
`qpec_small` and stays around `1e-2` on `ralph1` — but that is a
*consequence* of the property that actually separates them, which is
whether the limit point is S-stationary. `qpec_small`'s is (`grad f = 0`,
so `lambda = 0` certifies it with residual `0`); `ralph1`'s is not (best
sign-feasible residual `0.707`). **A solver cannot know that in advance**,
and by the time the trajectories are distinguishable, ruled-out 2 applies.

So the two criteria gh#884 states — converge the seven `qpec_small`
cells, keep `ralph1/unit/origin/direct` failing — are not jointly
satisfiable by any trigger keyed on solver-observable state. That is not
a proof that no fix exists; it is a measured statement about the class of
fixes the issue's own text proposes.

## What is left

The decision is not a patch, and belongs to the owner:

1. **Accept `perturb_always_cd` as a documented per-model lever** for the
   degenerate-complementarity class, with the measured caveat above:
   on a model whose limit point is only M-stationary it can return a
   success below `f*`. It already ships as an option; nothing to build.
2. **Re-scope gh#884** to the exact-product lowering being unsupported
   for biactive pairs, which is what `benchmarks/mpcc/`'s recommended
   route (`scholtes_then_ncp`) already assumes — it clears all eight
   cells (`1.9e-12`, `4.9e-11`, `5.7e-28`).
3. **Treat it as a genuine IPM research change** — a pair-structured KKT
   regularizer, or a `mu` rule that does not stall on a dual term
   inflated by the multipliers it is normalised by. That needs the
   fixture sweep across both legs, `benchmarks/qp` magnitude checks, and
   its own owner, per CLAUDE.md.

What should **not** happen is a fix keyed on the runaway pattern, or a
default flip. Both are measured above.
