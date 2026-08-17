# Warm-start benchmark — design notes and findings

The suite lives at `benchmarks/warmstart/`; its README is the
user-facing document. This note records *why* it is shaped the way it
is, what the prior-art survey turned up, and the two solver findings
that came out of building it.

## Prior art (searched 2026-07-31): there isn't one

No solver-agnostic, NLP-level, sequence-based warm-start benchmark
exists to reuse. What does exist and why none of it covers the case:

- **qpbenchmark family** (`qpsolvers/qpbenchmark`, `mpc_qpbenchmark`,
  `ik_qpbenchmark`, Maros-Mészáros, free-for-all). QP level, and
  explicitly cold-start: the MPC test set's README states it "does not
  reflect the warm-starting that is frequently used on robots that do
  model predictive control". Problems ship as independent instances
  with the sequence structure discarded.
- **WARP**, arXiv:2605.05728 (2026) — "A Benchmark for Primal-Dual
  Warm-Starting of Interior-Point Solvers". The closest existing
  thing. Interior-point only (it predicts the full primal-dual-barrier
  state; reports IPOPT 23 → 3 iterations for an oracle start, 76% for
  its learned model), AC-OPF only, and built around learned
  predictions over i.i.d. instances rather than consecutive
  perturbations. No active set, no working set, no path.
- **Not All Warm Starts Help**, arXiv:2606.08984 (2026) — benchmarks
  primal-dual initializations for ACOPF. An evaluation study, not a
  reusable set. Its finding that warm starts frequently *hurt* is why
  regressions are a first-class column here rather than an aggregate
  the mean absorbs.
- **OPFData** (arXiv:2406.07234) / PGLearn — 300k solved AC-OPF
  instances per grid (loads perturbed 80–120%, plus N-1 topologies).
  A dataset, no protocol; independent samples, not a path.
- **acados / CasADi NMPC papers** — closed-loop timing comparisons
  (chain-of-masses and friends) embedded in solver repos and papers.
  Per-paper harnesses; nothing portable.
- **CUTEst / Hock-Schittkowski / Vanderbei / Mittelmann /
  Maros-Mészáros** — every problem is a single cold solve.

## Why the unit is a family-plus-path

Warm starting has no meaning for one isolated NLP, and the property
that predicts payoff — how the active set moves between consecutive
instances — only exists along a path. So the benchmark's unit of work
is the whole sequence, families are tagged by the active-set regime
they exercise, and every family is run at three step sizes (×0.1, ×1,
×4 of its natural increment) because payoff is a function of how far
the problem moved. A single step size measures one point on a curve
and calls it the answer.

Families are dense-callback and solver-free by construction; the only
module that imports a solver is `adapters/pounce_adapter.py`. That is
what keeps the option open to lift the suite out of the pounce tree,
or to add an Ipopt arm, without touching a family.

## Why the harness is Python-driven and not `.nl`-driven

Every other suite runs `.nl` files through the CLI. That cannot work
here: carrying a working set between solves needs an in-process
handle, and the CLI has no cross-process working-set input (the
debugger's `resolve` seeds a primal point only). The Python
`Problem.solve(working_set=…)` / `WarmStart` surface is the one that
supports the contract, so the harness drives it directly.

## Correctness is judged on the step's own terms, not against the reference

The first cut scored a step correct iff its solution matched the
reference arm's (`cold-ipm`) to a tolerance. That is wrong on
nonconvex families, and the benchmark demonstrated it immediately: on
`rosenbrock_ring`, `cold-ipm` converges to Rosenbrock's well-known
*local* minimum (f ≈ 3.9866 at x ≈ (−1, 1, …, 1)) while the SQP arms
find the global one (f = 0). Scoring against the reference marked the
better answer wrong.

The criterion now is: converged on its own terms (success status, plus
a KKT residual and feasibility the *harness* verifies rather than
taking the solver's word for), and not a *worse* optimum than the
reference by more than `--obj-tol`. A better optimum is reported in
its own column. `‖x − x_ref‖` is a recorded diagnostic, not a gate,
because two solves can both be optimal and still differ in `x` near a
degenerate face — which the `moving_bound_qp` family produced at step
9, where the SQP and IPM arms agreed on the objective to 1e-9 while
differing in `x` by 2e-5.

## Finding 1: the inner active-set work was not observable (fixed here)

An SQP warm start's whole purpose is to skip active-set searching in
the QP subproblems, and none of that was visible from Python. The
outer iteration count is not a proxy: on a QP-shaped NLP the outer
loop terminates in one iteration whether warm started or not, so a
cold/warm comparison on `iter_count` reads exactly 1.00× while the
inner work differs by an order of magnitude.

`pounce-qp` already counted per-QP active-set changes in
`QpStats::n_working_set_changes`; nothing accumulated them. Added:

- `SqpResult::n_qp_working_set_changes` — summed over every step QP,
  including the cold-start and quasi-Newton-reset fallbacks. Excludes
  second-order-correction QPs, whose stats the line search does not
  surface.
- `SolveStatistics::sqp_qp_solves` / `sqp_qp_working_set_changes` —
  0 on the IPM path, which solves no QP subproblems.
- Python `info["n_qp_solves"]` / `info["n_qp_ws_changes"]`.

Covered by `sqp_reports_qp_working_set_changes_and_warm_start_removes_them`
in `crates/pounce-algorithm/src/sqp/tests.rs`.

With it, the first full run shows what was previously invisible:
`simplex_proj` goes from 313 active-set changes cold to 4 warm at the
default step size, and `nmpc_vanderpol` from 931 to 66 at the smallest
one — while the outer-iteration column for both reads 1.00×.

## Finding 2: exact-Hessian SQP fails on Rosenbrock from the classic start

Filed as **#416**, **fixed** in #419 — which then regressed the
unconstrained case, see Finding 3. On

    min Rosenbrock(x)  s.t.  ‖x‖² ≤ r²,   n = 10,  −5 ≤ x ≤ 5

started from the traditional `(−1.2, 1, −1.2, …)`, the default
`sqp_hessian = exact` path terminates with
`Search_Direction_Becomes_Too_Small` at **every** step of the
parameter path, at a point with stationarity residual ≈ 2.6 and
objective 9.62 (against 0.0 from other starts). `damped-bfgs` and
`lbfgs` converge from the same start, and `exact` converges from the
origin or from `0.9·1`. So it is a step-computation failure on an
indefinite exact Hessian, not a modelling problem:

| `sqp_hessian` | from (−1.2, 1, …) | from 0.9·1 | from 0 |
|---|---|---|---|
| `exact` | fails, 4 iters, KKT 2.6 | ok, 6 iters, f=0 | ok, 13 iters, f=0 |
| `damped-bfgs` | ok, 99 iters, f=0 | ok, 28 iters | ok, 123 iters |
| `lbfgs` | ok, 123 iters, f=0 | ok, 258 iters | ok, 98 iters |

Narrowed down after filing groundwork (all in #416): the inner QP hits
its `sqp_qp_max_iter` cap, which the outer loop maps to `QpStepFailed`
→ `Search_Direction_Becomes_Too_Small`. Raising the cap from its
default 200 to **250** converges. The cap is not really the bug: the
failing QP makes **zero** working-set changes in those 200 iterations,
and the 250 threshold is identical at n = 10, 20, 30 and 40, so a
fixed-count internal loop is spinning rather than active-set work
outgrowing its budget. Constraints are not involved — the same failure
occurs with the ball made unreachable and with `m = 0`. The failure
appears at the first iterate where ∇²L goes indefinite (min eigenvalue
−1.4; it is +35.4 at the start).

The family therefore starts from the origin, so the row measures warm
starting rather than a wall of identical failures.

**Corroboration from `double_well_chain`** (added later, see below).
That family is the same configuration — unconstrained, empty working
set, indefinite exact Hessian — but benign enough that the solve
*survives*: on a 12-variable instance the exact-Hessian SQP needs 24
outer iterations and 36 QP solves (12 more than outer iterations, i.e.
the rescue re-solves firing), and takes 44 ms per QP against 0.4 ms for
`damped-bfgs`. Capping `sqp_qp_max_iter` at **20** produces a
**bit-identical** result — same objective and same `x` to the last bit
at every step of the 20-step path — **8.9× faster** (33.9s → 3.8s).

That is the cleanest statement of the defect available: on this
configuration the inner QP's iterations past ~20 change nothing. They
are not work that ran out of budget, they are work that never mattered.
Rosenbrock is the same waste with a worse ending, because there the
`MaxIter` exit is not rescued.

## Finding 3: #419's fix for #416 regressed the unconstrained case

Filed as **#423**, **fixed** in #424. #419 fixed #416 by capping the inner ratio test
at the shifted step's true minimizer `α*` instead of at 1, so a
negative-curvature direction runs to its blocking bound and changes the
working set. Rosenbrock from the classic start now converges in 20
iterations.

But when there is **no** blocking bound — `m = 0`, no finite variable
bounds, which is exactly `double_well_chain` — the direction runs
forever, becomes a recession certificate, fails re-verification against
the true NLP (the objective really is bounded below), and the solve
ends. A/B on an identical script:

| build | `active-set-sqp` + `exact` | f | iters |
|---|---|--:|--:|
| 301aa84 (pre-#419) | `Solve_Succeeded` | 0.027424 | 24 |
| c15c015 (post-#419) | `Search_Direction_Becomes_Too_Small` | 26.025699 | **1** |

The IPM and `damped-bfgs` are bit-identical across the two builds, so it
is specific to the path #419 changed. The two issues are mirror images:
#416 was "the shifted step is capped at α = 1, so the solver spins
without pivoting"; #423 is "the shifted step is gone, so a solver with
nothing to pivot *to* has no step at all".

#424 fixes it by giving the driver a third branch: "the model recedes
but the NLP does not" is no longer a dead end. Verified on the same
script — `f = 0.027424` in 24 iterations, exactly the pre-#419 numbers —
and #419's own fix still holds, with Rosenbrock from the classic start
converging in 20 iterations. Both hold simultaneously; the family's rows
are back to `bad = 0` and its pre-#419 speedups (8.33× / 6.46× / 5.23×).

Worth recording as a process point: `double_well_chain` was added the day
before #419 landed, purely to close a coverage hole (no family ran
`m = 0`), and it caught the regression on its first run against the new
base. The other 21 rows were bit-identical across the two commits.

## Active-set coverage, and the two families that closed the holes

Audited by reading the working sets the solver actually returns, per
family and scale, rather than by reading the intent. What was already
covered: permanently-active equality rows, bounds activating and
releasing, inequality rows activating and releasing, and paths where
the active set is deliberately held fixed (`simplex_proj` and
`hanging_chain` at `tiny`, 0 changes end to end).

Two holes, both closed:

- **Re-activation was never tested.** `rosenbrock_ring` sweeps its
  radius monotonically, so its switch is always active → inactive.
  `rosenbrock_ring_cycle` makes the path a triangle — out past the
  switch, back in — so it crosses in both directions, at exactly the
  quarter and three-quarter steps at every scale. It measures
  measurably *less* warm-start payoff than the monotone version at
  `large` (1.76× vs 2.16× on QP active-set work, 8 residual changes vs
  1), which is the expected asymmetry: carrying an empty working set
  into a step that needs a non-empty one gives the solver nothing.
- **No path was unconstrained.** Every family carried at least one
  constraint row, so the suite never executed `m = 0`, and there was no
  zero mark to read the speedups against. `double_well_chain` is
  `m = 0` with no finite bounds — the working set is empty at every
  iterate of every step, confirmed in the data (`|A| = 0`, 20/20 steps,
  all scales).

`double_well_chain` also inverts the report's two headline columns,
which is worth understanding before reading any row: with no working
set to carry, the QP-active-set column is exactly 1.00× (0→0) and the
*entire* warm-start effect lands in outer iterations (8.3× at `tiny`).
The QP-shaped families are the mirror image — outer flat at 1.00×, all
effect inner. Neither column alone is a summary of this suite.

## The three-way: dedicated convex QP solver vs the two general paths

Added after the first results raised the question the suite could not
then answer. `cold-qp-ipm` / `warm-qp-ipm` route the three QP-shaped
families through `pounce.solve_qp` (pounce-convex). `qpform.py` does the
extraction; the self-test verifies it by re-deriving the family from the
extracted data, because a silent extraction bug would produce plausible
and wrong numbers rather than an error.

Wall time over the 9 QP-family rows, geometric mean (ms):

| cold-ipm | cold-sqp | cold-qp-ipm | warm-ipm | warm-sqp | warm-qp-ipm |
|--:|--:|--:|--:|--:|--:|
| 97.1 | 72.8 | 75.9 | 48.5 | 37.3 | **35.8** |

(Absolute wall times drift ~10–30% between runs on this machine, so read
the ordering and the ratios, not the milliseconds. Iteration counts are
noise-free and are quoted below.)

Two results worth keeping:

- **The ranking flipped once #417 was fixed.** As first measured,
  warm-started active-set SQP was fastest on 8 of 9 rows. With #422
  landed, `warm-qp-ipm` is fastest on **6 of 9** and marginally ahead
  overall (35.8 ms vs 37.3 ms geometric mean). On a problem that really
  is a QP, a properly warm-started dedicated convex solver is now
  competitive with — and usually better than — the active-set SQP.
- **The convex IPM used to warm-start weakly** — 1.17–1.50× on
  iterations, against 4–16× for the SQP's inner active-set work — which
  is what prompted #417. With #422 landed it is 1.73–3.00×, and the raw
  counts moved exactly as the prototype predicted:

  | family @ scale | warm iters before #422 | predicted | after #422 |
  |---|--:|--:|--:|
  | `simplex_proj` @ tiny | 103 | 46 | **46** |
  | `simplex_proj` @ small | 124 | 75 | **75** |
  | `simplex_proj` @ large | 141 | 96 | **96** |

  The prediction came from an env-gated prototype measured in this
  suite before the issue was filed; the shipped fix reproduces it to the
  integer, which is about as direct a confirmation of a diagnosis as a
  benchmark can give.

  Chasing that down (**#417**) ruled out the explanation that first
  looked obvious. The convex path is *not* taking a plain primal-dual
  seed: `init_iterate` recenters adaptively, sizing the interior floor
  from the warm point's KKT residual on the new problem. Scaling that
  floor across five orders of magnitude changes the iteration count by
  about one, and a *perfect* warm start (re-solving an identical
  problem from its own solution) converges in 0–2 iterations, so the
  machinery is sound. The limit is the fraction-to-boundary parameter
  `QpOptions::tau`, pinned at 0.95: the trace shows α held at exactly
  0.950 for every late iteration, so μ and the residuals fall a fixed
  ~20× per step and the count is `log₂₀(μ₀/tol)` however good the
  start is. Letting τ → 1 as μ → 0 for orthant blocks cuts warm
  iterations 35–60% with the full `pounce-convex` suite passing; doing
  it for *every* cone kind loses 60% of the SOC instances the direct
  driver currently solves. A filter line-search method can take a full
  Newton step near the solution and a barrier method with static τ
  cannot — that, not the seeding, is the whole asymmetry.

Caveat kept in the report itself: the QP arms receive matrix data once
per step where the callback arms re-evaluate per iteration, so the wall
time favors them by an amount this suite does not separate out.

## Finding 4: `sqp_qp_use_homotopy` was registered but never read

Found while adding the `-hom` arms. The option was registered by the
homotopy work (#412) and documented in detail, but
`apply_qp_subproblem_options` — the function that maps the `sqp_qp_*`
family onto `pounce_qp::QpOptions` — never read it. Setting it on the
SQP path therefore did nothing at all, silently, while the option's own
documentation described the algorithm it would select. The first
version of the `-hom` arms measured bit-identical results to their
twins, which is how it surfaced: an arm that cannot differ from its
control is either a perfect null result or a broken experiment, and it
was the second.

This is the exact inverse of #360 (read-but-unregistered), and the
guard test that issue left behind could not catch it — that test walks
the keys the *reader* consults and asserts each is registered, which
says nothing about a registered key no reader consults. Fixed here by
reading the option, and by adding
`application_every_registered_sqp_qp_option_is_read_by_the_subproblem_reader`,
which enumerates the registry and fails if the two sets diverge in
either direction. Both guards were checked for falsifiability: removing
the reader fails the round-trip test, removing the name from the
covered list fails the new one.

## Finding 5: the homotopy is a sharply mixed trade, and the split is legible

With the option working, the `-hom` arms measure it. On cold inner
solves, where it engages:

| family | conventional → homotopy | ratio |
|---|--:|--:|
| `redundant_rows` | 247 → 30 | 4.2–12.3× |
| `degenerate_corner` | 69 → 30 | 2.0–2.9× |
| `moving_bound_qp` | 793 → 685 | 0.87–2.75× |
| `degenerate_vertex` | 154 → 132 | 1.1–1.3× |
| `simplex_proj` | 978 → 1400 | 0.63–0.74× |
| `nmpc_vanderpol` | 2745 → 5115 | 0.52–0.55× |
| four others | unchanged | 1.00× |
| **all 30 rows** | **5583 → 7989** | **0.70×** |

It wins where it was designed to — degenerate, rank-deficient,
netlib-like geometry, and it *improves* with perturbation size on
`redundant_rows` because the conventional cold solve degrades there
while the homotopy does not — and loses roughly 2× on well-conditioned
MPC-shaped QPs. That is a mechanism-level account of #412's
Maros-Mészáros result (20 gained, 7 lost, the losses large instances),
and it argues for keeping the default off on the SQP path while making
the knob reachable, which is now the case.

It also bears on #413: the homotopy's cost concentrates on
`nmpc_vanderpol`, the family closest in shape to the corrector-bound
instances that issue is about.

## Finding 6: the SQP's warm-start advantage is eroded by absolute churn, and size inflates it

> **Retracted by Finding 7.** The crossover below was #428, a defect
> found by the large tier and fixed in #429, not a property of
> active-set warm starts. The churn measurements are unchanged and
> still correct; the mechanism and the conclusion drawn from them are
> not. Kept as written, with the post-fix numbers in Finding 7.

The `mpc_horizon_10/20/40/80` families are one linear-quadratic MPC at
four horizons, so only `N` differs. Warm/cold wall-time ratio, below 1
meaning warm won:

| N | mean &#124;A&#124; | `tiny` | `small` | `large` |
|--:|--:|--:|--:|--:|
| 10 | 31.5 | 0.27 | 0.38 | 0.84 |
| 20 | 61.0 | 0.22 | 0.91 | 1.29 |
| 40 | 119.6 | **0.08** | 0.66 | 1.95 |
| 80 | 206.0 | 0.31 | 1.06 | **2.57** |

Not "the SQP loses at scale": at `tiny` steps it stays excellent at
every horizon, and its best row in the entire suite is N = 40 at 0.08.
What degrades is the combination of size *and* movement. The IPM arms
stay between 0.25 and 1.00 across the whole grid; the convex QP IPM is
flatter still.

The mechanism is in the working sets. The *fraction* of the active set
that changes per step is horizon-independent (~3% at `large` for every
N — the same angular perturbation moves proportionally the same
constraints), but the *absolute* count grows with the problem: 1.05 →
5.58 changes per step from N = 10 to N = 80. An active-set method pays
for the absolute count, and each change also costs more as `|A|` grows.
Two multiplying factors.

So the suite's earlier rule — payoff tracks churn — was right but
imprecise. It tracks **absolute** churn, and problem size inflates
absolute churn even at a proportionally identical perturbation. It also
puts a number on the qualitative caveat in `docs/src/active-set-sqp.md`
("prefer the IPM for large-scale problems with thousands of active
inequalities"): the measured crossover on this problem is tens to low
hundreds of active constraints, not thousands.

Repeatability: the N = 80 numbers are deterministic — two independent
re-runs gave 1605 → 1746 inner active-set changes to the digit.

## Finding 7 — at large scale the SQP warm start was discarded, not repaired (#428, fixed in #429)

The horizon sweep stopped at n = 242 and reported a gradual erosion.
Carrying the same MPC to n = 2402 (`--tier large`, N = 200/400/800)
showed the erosion was not gradual and not intrinsic — it was a defect,
now filed as #428.

At default settings the warm-started SQP does not return an answer on
the large tier: `Maximum_Iterations_Exceeded` with `iter_count = 0` on
7 of 8 steps at every horizon, `x` left at the warm-start point.
Everything else — `cold-sqp`, both IPM arms, both convex-QP arms —
solves all 8 cleanly.

Raising the inner-QP budget until nothing truncates, inner working-set
changes for one step:

| N | n | m | cold | warm | warm, hint admitted |
|--:|--:|--:|--:|--:|--:|
| 10 | 32 | 22 | 11 | 0 | 0 |
| 20 | 62 | 42 | 25 | 43 | 0 |
| 40 | 122 | 82 | 48 | 1 | 1 |
| 80 | 242 | 162 | 66 | 164 | 2 |
| 200 | 602 | 402 | 66 | 403 | 2 |
| 400 | 1202 | 802 | 66 | 795 | 2 |
| 800 | 2402 | 1602 | 66 | 1589 | 2 |

Cold is flat at 66 across a 75× range of m. Warm is Θ(m). The correct
answer is flat at 2.

It is a step function, not a slope. At N = 200, a parameter step that
changes **zero** entries of the true active set gives 0 warm pivots; a
step that changes **one** gives 400; four gives 403. The first changed
entry costs m, every later one costs 1.

Mechanism, confirmed by bisection: `solve_with_working_set` builds
`x_init` by pinning the hinted rows to their new boundary values. When
the active set has moved, that pinned point holds a row it should have
released and therefore violates some *other* row — by about the
distance the parameter moved (bracketed here to (0.01, 0.1] against a
step of ≈0.075). `solve`'s warm-start admission pre-check
(`pounce-qp/src/solver.rs:2855`) sees an infeasible primal and returns
`solve_elastic`, whose recovery re-solve seeds `WorkingSet::cold(n, m)`
— so the hint is dropped whole and every row is re-added from scratch.

The causal test: raise only `sqp_qp_feas_tol` so the same hint is
admitted, and the solve takes 2 pivots and lands on the same optimum to
1e-11 with violation 7e-15. The work was never necessary.

That tolerance is *not* a workaround, and the reason matters. `feas_tol`
gates two unrelated decisions — whether a hint is admitted, and whether
a converged point is accepted. At 0.1 the hint is admitted and the
answer is right; at 0.5 the same solve returns objective 26.6175 with
violation 6.4e-2 and KKT 2.5. There is no setting safe for both jobs.

Two corrections to earlier findings this forces:

- Finding 6's "warm starting turns harmful above N = 20 at large
  steps" is this defect, not a property of active-set warm starts. The
  wall-time crossover numbers stand as measurements; the *explanation*
  attached to them — absolute churn growing with size — is right for
  the IPM arms and wrong for the SQP, whose cost is set by whether the
  first entry moved, not by how many did.
- The suite's rule "payoff tracks absolute churn" survives for the IPM
  and needs the caveat above for the SQP.

Why nothing caught it earlier: at m = 22 the penalty is smaller than a
cold solve outright, so the default tier's numbers look merely
unimpressive rather than wrong. It takes m in the hundreds before the
Θ(m) term separates from the constant.

### After the fix

#429 repairs the hint instead of discarding it: the rows the pinned
point violates are known, so they are pinned too and the KKT
re-factored, keeping the |A| − 1 entries the hint got right. It declines
— leaving the old elastic recovery in place — when an already-active row
is violated, when the violated rows exceed a quarter of the hint's
active set, when the repaired pin set would exceed n rows, or after
three re-pin rounds.

Re-measured on the same build:

| N | n | m | cold | warm before | warm after |
|--:|--:|--:|--:|--:|--:|
| 10 | 32 | 22 | 11 | 0 | 0 |
| 20 | 62 | 42 | 25 | 43 | 1 |
| 40 | 122 | 82 | 48 | 1 | 1 |
| 80 | 242 | 162 | 66 | 164 | 3 |
| 200 | 602 | 402 | 66 | 403 | 3 |
| 400 | 1202 | 802 | 66 | 795 | 3 |
| 800 | 2402 | 1602 | 66 | 1589 | 3 |

Flat at 3 across a 75× range of m, same optimum to 1e-11. The Δφ scan
now tracks the movement instead of jumping: 0/0/0/1/3/4 warm pivots for
true active-set diffs of 0/0/1/2/4/4, against 0/0/400/401/403 before.

Large tier at default settings: every arm correct on every step (was
7 of 8 bad on both warm SQP arms), and `warm-sqp` wall time is 0.03×,
0.03×, 0.02× its cold twin at N = 200/400/800 — the fastest arm on the
board, where it previously did not return an answer. Inner work 514 → 11
per path at all three horizons. At n = 2402: 1.34 s warm against 12.12 s
cold.

**Findings 6's conclusion is retracted.** The horizon sweep's crossover
— warm/cold wall 0.84 → 1.29 → 1.95 → 2.57 turning harmful above
N = 20 — was this defect, not a property of active-set warm starts. On
the fixed solver the same measurement runs 0.24 → 0.12 → 0.11 → 0.10 at
`large` and 0.17 → 0.08 → 0.04 → 0.02 at `tiny`: the ratio *improves*
with horizon, because cold cost grows with the problem while warm cost
is set by how far the active set moved. The churn numbers in Finding 6
were correct and are unchanged (1.05 / 2.26 / 4.21 / 5.58); it was the
interpretation built on them that was wrong. The suite's original rule
— payoff tracks how much the active set moves, and problem size has
little to do with it — stands as first written, and the "absolute churn"
sharpening should be dropped.

A caution worth keeping: Finding 6 reasoned from a plausible mechanism
(absolute churn grows with size, each change costs more as |A| grows)
that fit the data and was still wrong, because the data had a defect in
it. Two multiplying factors were invented to explain what was really a
single discarded hint. The measurement that would have caught it was
cheap — compare warm pivots against the *true* active-set difference,
which is 4 where the warm arm spent 164 — and it was not run until the
large tier made the discrepancy impossible to miss.
## What pounce#611 closed, and what it did not

pounce#611 asked for the coverage this section used to list as
missing, plus an external-solver baseline and a composite report. The
work is in `benchmarks/warmstart/` alongside the original suite; the
generated report is
[`warm-start-611-composite.md`](warm-start-611-composite.md) and its
machine-readable twin is `warm-start-611-composite.json`. Both are
regenerated by

    cd benchmarks
    python -m warmstart.run      --scales all --arms all --out warmstart/results.json
    python -m warmstart.run      --solver ipopt --scales all --arms all \
                                 --out warmstart/results-ipopt.json
    python -m warmstart.transfer --experiment all --out warmstart/transfer.json
    python -m warmstart.composite --results  warmstart/results.json \
        --ipopt warmstart/results-ipopt.json --transfer warmstart/transfer.json \
        -o ../dev-notes/warm-start-611-composite.md \
        --json-out ../dev-notes/warm-start-611-composite.json

Every table in that report is computed from those three JSON files.
None is transcribed by hand — which matters because #638 and #639 will
both move arms measured here, and a re-measure has to be a re-run
rather than an editing pass.

### Closed

- **A shift-based warm-start arm.** `warmstart/transfer.py` shifts the
  previous horizon by one stage through `WarmStart.reindex`, on
  `mpc_horizon_*` and on the genuinely closed-loop `nmpc_vanderpol`.
  **The result is negative**: on the closed-loop family the shift costs
  73 iterations against unshifted carry's 71 (cold: 136), and on
  `mpc_horizon_10` it is much worse — 87 against 23. The unseeded final
  stage costs more than the shift saves, and on `mpc_horizon_*` the path
  is a *rotation* of the initial state rather than a receding horizon,
  so the shift is not even the operation relating consecutive problems.
  The concern that the suite "understates what an MPC implementation
  would do" is measured, and it does not.
- **Large scale beyond one problem class.** Two new families with
  sparsity patterns that are not block-banded MPC:
  `elliptic_control_*` (a 1-D Poisson control problem, tridiagonal,
  symmetric coupling, conditioning growing like `h⁻²`) and
  `resistive_network_*` (flows on a ring-plus-chord graph, two
  entries per column, and a quartic loss so the Hessian is not
  constant). Both reach the `large` tier.
- **An external solver arm.** `warmstart/adapters/ipopt_adapter.py`
  drives Ipopt through cyipopt using the *same* callback object the
  pounce adapter uses, so evaluation counts mean the same thing on
  both sides. Arms Ipopt has no counterpart for are skipped with a
  recorded reason rather than dropped.
- **Composite-report integration.** `warmstart/composite.py`, wired
  in above. It stays out of `BENCHMARK_REPORT.md` — that document is
  per-problem cold-solve rows against an Ipopt reference and still has
  no shape for a per-sequence result — and is a sibling document
  instead.
- **A correctness gate that is the same for both solvers.**
  `warmstart/kkt.py` recomputes dual feasibility, primal feasibility
  and complementarity from the returned point, so neither solver's own
  status line decides whether its step counted. It agrees with pounce's
  reported unscaled KKT error to within about an order of magnitude
  across the suite (both far under the 1e-4 gate); the residual
  difference is norm and scaling convention.
- **Families built to make warm starting lose.** See below.

### The falsification families

The suite's own weakness was that every family in it had been chosen by
warm-start work, so it could measure how *much* warm starting won but
never *whether*. `rastrigin_drift` and `rastrigin_scatter` exist to
produce the opposite result, and they do: on unrelated instances the
warm arm is *faster* than cold and lands on a worse optimum on most
steps, and both racing arms do worse still. A wrong-basin step converges
cleanly — it shows up in the `bad` column, never in a status code.

Read the composite report's falsification section before quoting any
speedup from this suite. It is the section that says where the
speedups stop applying.

### Still not covered

- **A second external solver.** nlopt 2.11.0 and casadi 3.7.2 (which
  bundles its own Ipopt) both install cleanly here, but neither was
  wired to an adapter. nlopt has no dual warm-start API at all, so it
  would only populate the "previous primal only" arm; casadi's Ipopt
  would duplicate the cyipopt arm through a different binding.
- **DAE discretizations.** `elliptic_control_*` is an elliptic PDE.
  Nothing in the suite is an index-1 DAE or a collocation
  transcription, which behave differently under a warm start because
  the algebraic variables have no dynamics to carry them.
- **Memory peak.** The issue lists it as a metric. It is not measured:
  a Python-level `tracemalloc` figure would report the harness's
  allocations rather than the solver's, and RSS is too coarse at these
  problem sizes to separate the arms.
- **HSL.** The Ipopt arm runs on MUMPS. Wall-time comparisons against
  it are comparisons against a MUMPS build, and are labelled that way
  in the report.

### A solver finding this coverage produced

`elliptic_control_*` does not solve at all under the conventional
phase-1/phase-2 inner QP: `cold-sqp` returns
`Maximum_Iterations_Exceeded` with **zero completed outer iterations**
(`n_qp_solves = 1`, `n_qp_ws_changes = 88`), and a 10x larger
`sqp_max_iter` changes nothing, because the budget being exhausted is
the inner QP's. The homotopy variant solves the identical problem in
one outer iteration to a KKT residual of 3.1e-13. Reproduce with

    cd benchmarks && python -m warmstart.run --families elliptic_control_40 \
        --scales small --arms cold-sqp,cold-sqp-hom

This is behaviour of `main`, not of anything pounce#611 changed — no
Rust was touched — and it is recorded here rather than filed, per the
repo owner's standing instruction.
