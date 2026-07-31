# The Warm-Start Benchmark

Every other suite in `benchmarks/` answers "how fast does POUNCE solve
this problem?" This one answers a different question: **when you solve a
sequence of related problems, how much does starting from the previous
answer actually save — and which of POUNCE's three solvers should you
use?**

That question has no meaning for a single isolated solve, which is why
it needs its own suite. The unit of work here is a *parametric family
plus a path*: one problem shape, one scripted sweep through its
parameter space, solved end to end. MPC horizons, continuation and
homotopy, sensitivity sweeps, and design exploration all have this
shape.

There is no standard public benchmark for this. The nearest things —
the [qpbenchmark](https://github.com/qpsolvers/qpbenchmark) test sets,
[WARP](https://arxiv.org/abs/2605.05728), the AC-OPF learning datasets —
are either QP-only, interior-point-only, or ship their instances
stripped of the sequence structure that makes warm starting meaningful.
`benchmarks/warmstart/README.md` has the full survey.

## The three solvers under test

POUNCE has three solve paths that can take a sequence, and they warm
start in genuinely different ways:

| solver | `algorithm` / entry point | what it carries between solves |
|---|---|---|
| **general NLP filter-IPM** | `interior-point` (the default) | the previous primal-dual point and the converged barrier parameter μ |
| **active-set SQP** | `algorithm = active-set-sqp` | the previous **working set** — which bounds and constraints were active — plus the point |
| **convex QP interior point** | `pounce.solve_qp` (`solver_selection=qp-ipm`) | the previous primal-dual point |

Each runs cold and warm, giving six arms:

| arm | solver | seeded with | runs on |
|---|---|---|---|
| `cold-ipm` | NLP filter-IPM | nothing | every family |
| `warm-ipm` | NLP filter-IPM | previous point + μ | every family |
| `cold-sqp` | active-set SQP | nothing | every family |
| `warm-sqp` | active-set SQP | previous working set + point | every family |
| `cold-qp-ipm` | convex QP IPM | nothing | QP families only |
| `warm-qp-ipm` | convex QP IPM | previous primal-dual point | QP families only |

Each warm arm is scored against **its own** cold counterpart. That
pairing is the whole point: `warm-sqp` beating `cold-ipm` would confound
"warm started" with "switched algorithms", and only the paired
comparison isolates the warm start.

## The problems

Eight families, each run at three step sizes (`tiny` ×0.1, `small` ×1,
`large` ×4 of its natural per-step parameter increment), for 24 rows and
489 solves per arm. Warm-start payoff is a function of how far the
problem moved, so a single step size would measure one point on a curve
and call it the answer.

| family | n | m | active-set regime | perturbation enters | curvature |
|---|--:|--:|---|---|---|
| `simplex_proj` | 20 | 1 | flipping | objective | convex |
| `moving_bound_qp` | 40 | 3 | flipping | variable bounds | convex |
| `degenerate_corner` | 6 | 3 | degenerate (a multiplier passes through zero) | objective | convex |
| `hanging_chain` | 30 | 15 | flipping contacts | mixed | convex |
| `rosenbrock_ring` | 10 | 1 | one clean activation switch | constraint RHS | nonconvex |
| `rosenbrock_ring_cycle` | 10 | 1 | switch crossed in both directions | constraint RHS | nonconvex |
| `double_well_chain` | 12 | 0 | none — empty active set throughout | objective | nonconvex |
| `nmpc_vanderpol` | 47 | 32 | closed-loop MPC | constraint RHS | nonconvex |

The families are deliberately small and analytic. This is a
*measurement* of warm-start behavior, not a scaling study — the
conclusions about which solver to use transfer; the absolute timings do
not.

## How a result is produced

Three rules make the arms comparable:

1. **Every arm sees the identical parameter sequence.** For
   `nmpc_vanderpol`, whose path depends on its own solutions, the
   sequence is recorded once from the reference arm and replayed for
   the others.
2. **Step 0 of a warm arm is a cold solve** — there is nothing to warm
   from — and is excluded from the speedup ratios while still counting
   in the totals.
3. **Every step is checked.** A step must return success, actually
   achieve a small KKT residual and be feasible (verified by the
   harness, not taken from the solver's status), and not land on a
   *worse* optimum than the reference. A warm start that converges
   quickly to the wrong answer is a failure, not a win.

In the run reported below, **every step of every arm passed** — 489
steps × 4 arms, plus 180 QP-arm steps, with zero correctness failures.

## Results

Run on POUNCE 0.9.0, `tol = 1e-8`, one machine, all 24 rows.

### Does warm starting pay?

Totals across all 489 steps of all 24 rows:

| arm | Σ outer iterations | Σ solve time | incorrect steps |
|---|--:|--:|--:|
| `cold-ipm` | 6682 | 3.75 s | 0 |
| `warm-ipm` | **2156** | 1.73 s | 0 |
| `cold-sqp` | 3872 | 12.82 s | 0 |
| `warm-sqp` | **1138** | 3.05 s | 0 |

Both solvers cut outer iterations by roughly 3×. But for the active-set
SQP that number badly understates the effect, for a reason worth
understanding before reading any further.

### The metric trap: outer iterations hide the SQP's warm start

On a problem whose subproblem is already a QP, the SQP outer loop
terminates in **one** iteration whether or not it was warm started. The
work a working-set warm start actually saves is *inside* the QP
subproblems, and it is reported separately as
`info["n_qp_ws_changes"]` — active-set changes (adds + drops) summed
over the step QPs.

The two extremes make the point:

| family | SQP outer iterations, cold→warm | QP active-set changes, cold→warm |
|---|--:|--:|
| `simplex_proj` @ tiny (a QP) | 1.00× — flat | **16.0×** (285 → 0) |
| `double_well_chain` @ tiny (unconstrained) | **8.33×** | 1.00× (0 → 0) |

They are mirror images. On a QP, everything happens inside; with no
constraints there is no working set to carry, so the entire effect is in
the outer loop and comes from the primal point alone. **Neither column
alone summarizes this benchmark.** `double_well_chain` exists precisely
to be that zero mark.

### Warm-start effect per family

`SQP` is the ratio of inner QP active-set changes (raw totals in
parentheses); `IPM` is the ratio of outer iterations. Higher is better;
`worse` counts steps where warm cost *more* than cold.

| family | scale | SQP cold→warm | worse | IPM cold→warm | worse |
|---|---|--:|--:|--:|--:|
| `simplex_proj` | tiny | 16.00× (285→0) | 0 | 5.05× | 0 |
| `simplex_proj` | small | 15.55× (313→4) | 0 | 4.42× | 0 |
| `simplex_proj` | large | 15.09× (335→7) | 0 | 4.17× | 0 |
| `moving_bound_qp` | tiny | 6.00× (104→3) | 0 | 5.01× | 0 |
| `moving_bound_qp` | small | 6.10× (207→29) | 0 | 2.00× | 0 |
| `moving_bound_qp` | large | 4.99× (467→104) | 0 | 1.93× | 0 |
| `degenerate_corner` | tiny | 1.87× (19→1) | 0 | 4.67× | 0 |
| `degenerate_corner` | small | 1.87× (19→1) | 0 | 3.56× | 0 |
| `degenerate_corner` | large | 1.92× (26→4) | 0 | 3.08× | 0 |
| `hanging_chain` | tiny | 4.00× (57→0) | 0 | 1.25× | 0 |
| `hanging_chain` | small | 4.84× (99→9) | 0 | 1.54× | 0 |
| `hanging_chain` | large | 4.19× (237→72) | 1 | **0.85×** | **17** |
| `rosenbrock_ring` | tiny | 2.37× (30→1) | 0 | 11.37× | 0 |
| `rosenbrock_ring` | small | 2.48× (33→1) | 0 | 8.75× | 0 |
| `rosenbrock_ring` | large | 2.16× (30→1) | 0 | 6.73× | 0 |
| `rosenbrock_ring_cycle` | tiny | 2.24× (29→2) | 0 | 9.16× | 0 |
| `rosenbrock_ring_cycle` | small | 2.47× (35→2) | 0 | 8.62× | 0 |
| `rosenbrock_ring_cycle` | large | 1.76× (30→8) | 0 | 6.14× | 0 |
| `double_well_chain` | tiny | 1.00× (0→0) | 0 | 3.00× | 0 |
| `double_well_chain` | small | 1.00× (0→0) | 0 | 2.29× | 0 |
| `double_well_chain` | large | 1.00× (0→0) | 0 | 2.14× | 0 |
| `nmpc_vanderpol` | tiny | 30.29× (931→66) | 0 | 3.63× | 0 |
| `nmpc_vanderpol` | small | 7.76× (781→294) | 0 | 1.96× | 0 |
| `nmpc_vanderpol` | large | 2.21× (868→693) | 8 | 1.05× | 8 |

### Payoff tracks active-set churn, not problem size

Read down any family and the pattern is the same: **the further the
problem moves per step, the less a warm start buys.** `churn` is the
mean number of working-set entries that change between consecutive
steps.

| family | churn/step at tiny → large | SQP payoff at tiny → large |
|---|---|---|
| `nmpc_vanderpol` | 0.21 → 2.95 | 30.3× → 2.2× |
| `moving_bound_qp` | 0.05 → 1.63 | 6.0× → 5.0× |
| `hanging_chain` | 0.00 → 0.47 | 4.0× → 4.2× |
| `simplex_proj` | 0.00 → 0.21 | 16.0× → 15.1× |

`nmpc_vanderpol` is the clearest case: a 14× increase in churn costs a
14× reduction in payoff. This is the practical rule — **warm starting
pays in proportion to how stable your active set is**, and problem size
has little to do with it.

### Warm starting can make things worse

Two rows show it, both at the largest step size:

- `hanging_chain @ large`, `warm-ipm`: **0.85×** — the warm-started IPM
  needed *more* iterations than a cold solve on **17 of 19 steps**. The
  previous solution sits exactly on the constraint boundary, which is
  the worst possible starting point for a barrier method when the
  active set has since moved.
- `nmpc_vanderpol @ large`: 8 of 19 steps worse for both warm arms,
  where a 4× control interval makes the plant state jump far enough
  that the carried working set is a poor guess.

This is why the benchmark reports regressions per step rather than only
a mean. A single averaged speedup would hide both.

### Three-way: which solver for a sequence of QPs?

Three families are literally convex QPs, so all three solvers can take
them. Interior-point iterations and active-set pivots are not the same
unit of work, so the like-for-like column is each solver against itself:

| family | scale | convex QP IPM cold→warm | NLP IPM cold→warm | SQP cold→warm (inner) | fastest warm arm |
|---|---|--:|--:|--:|---|
| `simplex_proj` | tiny | 160→46 (3.00×) | 182→28 | 285→0 (16.00×) | `warm-qp-ipm` |
| `simplex_proj` | small | 162→75 (2.03×) | 190→38 | 313→4 (15.55×) | `warm-qp-ipm` |
| `simplex_proj` | large | 173→96 (1.73×) | 200→45 | 335→7 (15.09×) | `warm-qp-ipm` |
| `moving_bound_qp` | tiny | 202→94 (2.06×) | 228→43 | 104→3 (6.00×) | `warm-sqp` |
| `moving_bound_qp` | small | 195→121 (1.58×) | 224→116 | 207→29 (6.10×) | `warm-sqp` |
| `moving_bound_qp` | large | 229→125 (1.77×) | 240→126 | 467→104 (4.99×) | `warm-qp-ipm` |
| `degenerate_corner` | tiny | 196→74 (2.45×) | 223→41 | 19→1 (1.87×) | `warm-qp-ipm` |
| `degenerate_corner` | small | 174→77 (2.12×) | 170→40 | 19→1 (1.87×) | `warm-qp-ipm` |
| `degenerate_corner` | large | 177→98 (1.73×) | 177→53 | 26→4 (1.92×) | `warm-sqp` |

Geometric-mean wall time over those nine rows:

| cold-ipm | cold-sqp | cold-qp-ipm | warm-ipm | warm-sqp | warm-qp-ipm |
|--:|--:|--:|--:|--:|--:|
| 97.1 ms | 72.8 ms | 75.9 ms | 48.5 ms | 37.3 ms | **35.8 ms** |

The dedicated convex solver is fastest on 6 of the 9 rows, with the
active-set SQP taking the other 3 — the rows where the active set churns
hardest. Note that this ranking is recent: before
[#417](https://github.com/jkitchin/pounce/issues/417) was fixed the
convex solver's warm start was capped at 1.2–1.5× and `warm-sqp` led 8
of 9 rows.

## What to take from this

- **For a sequence of convex QPs** — `solve_qp` warm-started with the
  previous result. It leads on most rows and needs no callbacks.
- **For a general NLP whose active set is stable between solves** —
  `algorithm = active-set-sqp` carrying the working set. This is where
  the largest effects live (up to 30× less inner active-set work), and
  the whole reason the active-set path exists.
- **For a problem with no active set to speak of** — unconstrained, or
  with constraints that never bind — the warm start still helps, but
  only through the primal point. Either solver is fine; the working set
  buys nothing (`double_well_chain`: 0 → 0).
- **When each step moves the problem a long way** — check whether warm
  starting is helping at all. It can cost more than a cold solve, and
  the IPM path is more exposed to this than the SQP path.
- **Always verify.** A fast wrong answer is the failure mode that
  matters, which is why the harness re-checks KKT residuals and
  objectives itself rather than trusting a status code.

See [Active-Set SQP & Warm Starts](active-set-sqp.md) for how to drive
the warm-start APIs, and
[Initialization and Warm Starts](initialization.md) for the
interior-point side.

## Defects this benchmark found

All three are fixed. They are listed because they show what the suite is
for — two of them lived in the same configuration (nonconvex, indefinite
Hessian, nothing active) that no other suite exercised:

| issue | what it was |
|---|---|
| [#416](https://github.com/jkitchin/pounce/issues/416) | Exact-Hessian SQP spent its entire inner-QP iteration budget making **zero** working-set changes; a budget of 20 gave bit-identical answers ~9× faster. Fixed in #419. |
| [#423](https://github.com/jkitchin/pounce/issues/423) | The #416 fix regressed unconstrained problems: with nothing able to block a negative-curvature direction, the solve died at iteration 1. Caught by `double_well_chain` on its first run against the new build. Fixed in #424. |
| [#417](https://github.com/jkitchin/pounce/issues/417) | The convex QP warm start left ~40% of its iterations unclaimed — not from the seeding but from a fraction-to-boundary parameter pinned at 0.95. Fixed in #422. |

## Running it

The harness drives POUNCE in-process through the Python API, so it needs
the extension built:

```sh
cd python && maturin develop --release
```

Then:

```sh
make -C benchmarks warmstart-selftest   # finite-difference checks, no solver needed
make -C benchmarks warmstart-run        # full sweep -> results.json + results.md
make -C benchmarks warmstart-quick      # 3 families, one scale
```

or, for a narrower run:

```sh
python -m warmstart.run --families simplex_proj,nmpc_vanderpol --scales large -v
python -m warmstart.run --arms cold-sqp,warm-sqp --tol 1e-10
```

Results land in `benchmarks/warmstart/results.json` (every step of every
arm) and `results.md`. Both are regenerated per run and gitignored.

Adding a problem family or a new solver is documented in
[`benchmarks/warmstart/README.md`](https://github.com/jkitchin/pounce/blob/main/benchmarks/warmstart/README.md);
nothing outside `adapters/` imports a solver, so the families and the
protocol are reusable against any solver with a warm-start API.

## Limits of these numbers

- **Small problems by design** (n ≤ 47). The rankings reflect algorithm
  behavior, not scaling; per-iteration costs shift with size, and the
  active-set path is documented to lose ground when the active set grows
  into the thousands.
- **Wall time carries Python callback overhead** for the four
  callback-driven arms. Iteration and active-set-change counts are the
  primary measurements; times are a cross-check, and vary 10–30% between
  runs on the same machine.
- **The QP arms are handed matrix data once per step**, where the other
  arms re-evaluate the model every iteration. That is a real advantage
  of the QP path on a QP, not an artifact, but it does mean the wall
  times are not measuring identical work.
- **One machine, one run.** Iteration counts are deterministic and
  reproducible; timings are not.
