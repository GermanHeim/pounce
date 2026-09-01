# gh#884: why the biactive dual divergence is not a targeted fix

Status: **gh#884 open.** This note records what was measured while
attempting a fix, three approaches ruled out by measurement, and one that
is **not** ruled out and looks like the most promising route. It exists so
the next attempt starts from the measurements rather than from the same
ideas.

**All measurements below were taken on `87402274`** (whose solver source is
identical to `main` at `7c42947f` — this branch adds only a test file and
this note). Re-measure rather than trusting these numbers across any
commit that touches the IPM.

Guards: `crates/pounce-algorithm/tests/issue_884_biactive_dual_divergence.rs`.

Those four tests are **invariants, not a description of the bug**. None of
them asserts that the current failure persists: a test that pinned the bug
would go red on a genuine fix, which is backwards, and would teach the next
reader to expect red in that file. The split follows an asymmetry —
`qpec_small` failing is the *bug*, so it is never asserted; `ralph1`
failing is *correct*, so it is. Everything below is measurement, and being
measurement is why it lives here rather than in an assertion.

## The mechanism, sharpened

gh#884 describes the symptom — the primal reaches the solution and the
dual diverges. To be precise about "reaches": the run stops at
`(1.0002321, 1.0001161, 2.67e-15)`, feasible to `2.2e-16` with
`f = 6.73e-8` against `f* = 0`. That is *near* the optimum, not at it, and
the `lambda = 0` certificate that holds exactly at `(1, 1, 0)` leaves an
objective-gradient residual of `4.64e-4` at the returned iterate. The
point is that the run gives up that close, not that it arrives.

The cause is one step further back, and it is a feedback loop.

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

1. **Neither `prim` nor `compl` ever blocks `mu`.** Both are already
   below `kappa_eps_mu = 1` by iteration 3 (`8.1e-5` and `1.1e-2`), and
   reach `~1e-11` by iteration 9. Only the `dual` column is ever the
   binding term.
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
by `a_structurally_zero_hessian_entry_does_not_change_the_solve`, which is
a contract in its own right: declaring an identically-zero Hessian entry
must be a no-op.

Whatever separates the two paths is somewhere else, and the ownership
bucket's description still needs re-checking against a harness
re-measurement — that part of gh#884 stands.

## Ruled out 2: detect the runaway and engage `delta_c` *in flight*. **Too late by construction.**

Dual regularization *is* the effective remedy. With `perturb_always_cd=yes`
the same model converges in 19 iterations to an **unscaled** KKT error of
`9.96e-8` at `x = (0.999994, 0.999997, 3.7e-6)` — an honest solve, not a
residual normalised away. So an answer is reachable and gh#884 is a real
POUNCE gap. Pinned by `dual_regularization_reaches_the_optimum_honestly`.

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
for — regularization can no longer recover **that iterate**.

Note precisely what this rules out: switching regularization on *within
the same run*, keeping the state the run has already reached. It says
nothing about discarding that state. See "Not ruled out" below.

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
replace it with a success at the wrong answer.

Guarded by `ralph1_must_not_claim_success_where_no_multiplier_certifies_it`
— and note *how*. An earlier draft of that test asserted the bad outcome
directly: it ran `perturb_always_cd=yes` and required a success below `f*`.
That was wrong three ways. It would go red when someone *fixed* the
underlying problem. It pinned a status the measurements above show is a
function of `max_iter` (`SolvedToAcceptableLevel` at 300,
`Solve_Succeeded` at 3000), so it would fail for reasons its name does not
describe. And it made "expected red" the norm for the file.

The test now runs at **default options** and asserts the contract: this
model must not report success, and never below `f*`. That is
direction-correct — red exactly when the solver gets worse. It still
catches the trap, because the only way to *ship* this remedy is to change
the default, which the test would then be running under.
Mutation-checked: flip that fixture to `perturb_always_cd=yes` and this
test alone fails, on the objective bound, at `-2.71e-5`.

The general rule, worth stating because it is easy to get backwards on a
known-open bug: **pin the invariant, not the state.** A conditional of the
form "if the solver claims success, the claim must be true" is vacuous
while a model is broken, load-bearing once it is fixed, and red exactly
when someone fakes the fix — which is the failure mode gh#884 names in its
own text ("a fix that only changed the exit verdict would be treating the
symptom").

`min -exp(x) s.t. x >= 0` — gh#274's own reproducer — is unaffected
either way (`ErrorInStepComputation`, identical iterate). That criterion
was never the binding one.

## Not ruled out, and the most promising route: detect late, then retry cold

Raised in review of PR #885, and it is a real gap in the three experiments
above. They cover the sparsity hypothesis, engaging `delta_c` **in flight**
at sampled iterations, and flipping the default **globally**. None of them
covers *detecting late and then restarting from iteration 0 with
regularization enabled*. "Too late to recover this iterate" is not "too
late to act" if the action is to throw the iterate away.

And the separation such a detector would need is present and large.
Minimum `||d||` over each run, taken only while the primal is converged
(`inf_pr <= 1e-8`), measured on `87402274`:

| fixture | min `||d||` | at iteration | max `inf_du` | status |
|---|---:|---:|---:|---|
| `qpec_small` / `prod_eq` / origin | **`4.309e-8`** | 12 | `7.76e11` | `RestorationFailed` |
| `ralph1` / `direct` / origin | **`7.153e-3`** | 17 | `3.63e13` | `RestorationFailed` |

Five orders apart, so any threshold in `(4.3e-8, 7.2e-3)` separates them.
That is the difference between an iterate that has *settled* while only
its multipliers run away, and one still wandering — which is exactly the
"converged primal, unbounded multiplier" versus "diverging iterate"
discriminator gh#884 asks for.

Composed with the cold retry, on these two fixtures the policy gives:

- `qpec_small` — detector fires (`||d||` reaches `4.3e-8`); retry from
  iteration 0 with `delta_c` on reaches `SolveSucceeded` at an unscaled
  KKT error of `9.96e-8`, the honest solve recorded above;
- `ralph1` — detector never fires (`||d||` bottoms out at `7.2e-3`, five
  orders above any separating threshold), so no retry happens and the
  honest failure stands.

**Both of gh#884's acceptance criteria, satisfied.** An earlier version of
this note claimed no solver-observable trigger could do that, on the
grounds that the property separating the two models is whether the limit
point is S-stationary and a solver cannot know that in advance. That
claim was too strong and is **retracted**: the solver does not need to
know S-stationarity, only to observe that the iterate stopped moving,
which it can.

What is *not* established, and would have to be before this ships:

- **Generality.** Two fixtures. Whether a `||d||` threshold separates the
  classes across `benchmarks/mpcc/`'s 79 fixture-legs, let alone the CLI
  corpus, is unmeasured — and per CLAUDE.md a threshold with no measured
  population behind it is exactly the kind that gets retuned later by
  someone who cannot see why it was chosen.
- **Cost.** A cold retry doubles the work on every model that trips the
  detector, so the false-positive rate is the price, and it is unmeasured.
- **Whether the retry converges in general.** It does here; it is a
  different trajectory on every other model.
- **`ralph1` is one of eight cells.** The other seven are `qpec_small`
  starts; the detector's behaviour on the remaining six is unmeasured.

So: promising, cheap to prototype, and the route this note recommends
trying first — not a finished answer.

## What is left

With the retry class open, the ordering changes: prototype that first,
and treat the options below as what to do if it does not survive the
corpus.

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

What should **not** happen is an *in-flight* switch keyed on the runaway
pattern, or a default flip. Both are measured above. A late detector with
a cold retry is a different mechanism and is not covered by either
measurement.
