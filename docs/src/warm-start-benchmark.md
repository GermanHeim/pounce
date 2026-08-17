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
| `values-ipm` | NLP filter-IPM | previous point alone, no duals | every family |
| `cold-sqp` | active-set SQP | nothing | every family |
| `warm-sqp` | active-set SQP | previous working set + point | every family |
| `cold-sqp-hom` | active-set SQP, homotopy inner QP | nothing | every family |
| `warm-sqp-hom` | active-set SQP, homotopy inner QP | previous working set + point | every family |
| `cold-qp-ipm` | convex QP IPM | nothing | QP families only |
| `warm-qp-ipm` | convex QP IPM | previous primal-dual point | QP families only |

The `-hom` pair differs from `cold-sqp` / `warm-sqp` in exactly one
option, `sqp_qp_use_homotopy`: the inner QP's **cold** solve traces the
§4.2 parametric homotopy — start from the box-only relaxation, tighten
the row bounds along `t ∈ [0,1]`, jump the working set at each event —
instead of the conventional phase-1/phase-2 scheme. It is the algorithm
`pounce-qp` is named for.

### Why `values-ipm` exists

Every other warm arm hands the solver multipliers. That is the
comfortable case, and measuring only it left a defect invisible for
two releases: on a seed carrying *no* duals, the bound-multiplier
blocks reached the warm-start initializer as literal zeros and were
floored at `warm_start_mult_bound_push` — 1e-9 under the tightened
pushes `pounce.WarmStart` ships — so the start declared every bound
inactive and got worse the tighter the pushes were set (pounce#622).
The corpus was bit-identical across the fix on `cold-ipm`,
`warm-ipm`, `pred-ipm` and `predcorr-ipm`, because not one of them
enters that path.

It is not a synthetic regime. A caller who kept only `x` is the
default on every frontend that carries variable levels but no duals:
GAMS `x.L`, a Pyomo model whose `dual` Suffix was never loaded, a
`.nl` written without dual guesses.

Across the pounce#622 fix, on this corpus: `values-ipm` **5490 →
4385** iterations (39 of 42 rows moved), while `warm-ipm` stayed at
3404 and `cold-ipm` at 10288 to the digit. `moving_bound_qp` alone
went 1040 → 428.

One family moved the other way, and the arm is now what watches it:
`degenerate_vertex` **220 → 396**. It holds 12 rows tight in 4
variables, so the true multipliers are a mass of ties near zero, and
the pre-fix fill — a bound-multiplier push small enough to read as
"every bound inactive" — happened to be right about them. Every
honest fill loses there: `mu / slack` costs 396, and capping that at
`bound_mult_init_val` costs 341 while introducing a fresh regression
on `redundant_rows` (162 → 292), so the cap was measured and dropped.
The regression is inherent to filling the blocks rather than to the
choice of fill, and it buys the 2.4× on `moving_bound_qp`.

Each warm arm is scored against **its own** cold counterpart. That
pairing is the whole point: `warm-sqp` beating `cold-ipm` would confound
"warm started" with "switched algorithms", and only the paired
comparison isolates the warm start.

## The problems

Fourteen families in the default sweep, each run at three step sizes
(`tiny` ×0.1, `small` ×1, `large` ×4 of its natural per-step parameter
increment), for 42 rows and 855 solves per arm, plus three more in an
opt-in large tier. Warm-start payoff is a function of how far the
problem moved, so a single step size would measure one point on a curve
and call it the answer.

| family | n | m | active-set regime | perturbation enters | curvature |
|---|--:|--:|---|---|---|
| `simplex_proj` | 20 | 1 | flipping | objective | convex |
| `moving_bound_qp` | 40 | 3 | flipping | variable bounds | convex |
| `degenerate_corner` | 6 | 3 | dual degenerate (a multiplier passes through zero) | objective | convex |
| `redundant_rows` | 6 | 5 | rank-deficient (LICQ fails; duplicated rows) | objective | convex |
| `degenerate_vertex` | 4 | 12 | primal degenerate (12 rows tight in 4 variables) | objective | convex |
| `hanging_chain` | 30 | 15 | flipping contacts | mixed | convex |
| `rosenbrock_ring` | 10 | 1 | one clean activation switch | constraint RHS | nonconvex |
| `rosenbrock_ring_cycle` | 10 | 1 | switch crossed in both directions | constraint RHS | nonconvex |
| `double_well_chain` | 12 | 0 | none — empty active set throughout | objective | nonconvex |
| `nmpc_vanderpol` | 47 | 32 | closed-loop MPC | constraint RHS | nonconvex |
| `mpc_horizon_10` | 32 | 22 | control saturation | constraint RHS | convex |
| `mpc_horizon_20` | 62 | 42 | control saturation | constraint RHS | convex |
| `mpc_horizon_40` | 122 | 82 | control saturation | constraint RHS | convex |
| `mpc_horizon_80` | 242 | 162 | control saturation | constraint RHS | convex |

plus an **opt-in large tier** (`--tier large`), the same MPC carried out
to a scale where the sparse factorization is what the cost is made of:

| family | n | m | nnz(J) |
|---|--:|--:|--:|
| `mpc_horizon_200` | 602 | 402 | 1402 |
| `mpc_horizon_400` | 1202 | 802 | 2802 |
| `mpc_horizon_800` | 2402 | 1602 | 5602 |

The seven `mpc_horizon_*` families are **the same linear-quadratic MPC
problem at seven horizons** — only `N` differs, so reading down them
isolates problem size from every other property. The parameter walks the
initial state around a circle, which keeps every step about as hard as
the last while rotating the set of saturated controls. Nothing dense is
ever built for them: they declare their block-banded Jacobian and
diagonal Hessian structurally, and the convex-QP arm receives sparse
matrices, because at N = 800 a dense Hessian alone would be 46 MB
rebuilt every iteration and passing dense data to the QP solver is
60–80× slower by its own diagnostic — which would have made the QP arm
look bad for a reason that has nothing to do with the QP arm.

The three degeneracy families cover the three distinct ways an
active-set QP meets degeneracy, which are not interchangeable:
`degenerate_corner` fails strict complementarity (a zero multiplier),
`redundant_rows` fails LICQ (duplicated equality rows throughout, and a
duplicated inequality pair that activates together partway along the
path), and `degenerate_vertex` is primally degenerate (12 constraints
tight at a 4-variable vertex, so the ratio test is a mass of ties —
the case Harris's two-pass test and GMSW EXPAND exist for). The
benchmark reports that pounce prunes that vertex's active set to its
maximal independent subset: `|A|` never exceeds 4 of the 12 tight rows.

Apart from the horizon sweep, the families are deliberately small and
analytic: this is a measurement of warm-start *behavior*, and small
problems measure it cleanly.

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

In the run reported below, **every step of every arm passed** — 42 rows,
6228 solves (855 steps for each of the six callback arms, 549 for the
two QP-only ones), with zero correctness failures.

## Results

Run on POUNCE 0.9.0, `tol = 1e-8`, one machine, all 42 rows, on a build
that includes the fix for
[#428](https://github.com/jkitchin/pounce/issues/428) — which this suite
found and which moved most of the SQP numbers below.

### Does warm starting pay?

Totals across all 855 steps of all 42 rows:

| arm | Σ outer iterations | Σ solve time | incorrect steps |
|---|--:|--:|--:|
| `cold-ipm` | 10288 | 7.71 s | 0 |
| `warm-ipm` | **3628** | 3.55 s | 0 |
| `cold-sqp` | 4238 | 30.31 s | 0 |
| `warm-sqp` | **1501** | 3.46 s | 0 |

Both solvers cut outer iterations by roughly 3×. But for the active-set
SQP that number badly understates the effect — its wall time falls by
8.8× on the same iteration count — for a reason worth understanding
before reading any further.

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
| `simplex_proj` | small | 17.46× (313→0) | 0 | 4.42× | 0 |
| `simplex_proj` | large | 18.62× (335→0) | 0 | 4.17× | 0 |
| `moving_bound_qp` | tiny | 6.45× (104→0) | 0 | 5.01× | 0 |
| `moving_bound_qp` | small | 11.53× (207→0) | 0 | 2.00× | 0 |
| `moving_bound_qp` | large | 13.74× (467→19) | 0 | 1.93× | 0 |
| `degenerate_corner` | tiny | 1.87× (19→1) | 0 | 4.67× | 0 |
| `degenerate_corner` | small | 1.87× (19→1) | 0 | 3.56× | 0 |
| `degenerate_corner` | large | 1.98× (26→3) | 0 | 3.08× | 0 |
| `redundant_rows` | tiny | 2.27× (42→0) | 0 | 5.25× | 0 |
| `redundant_rows` | small | 3.16× (73→2) | 1 | 4.08× | 0 |
| `redundant_rows` | large | 5.50× (114→2) | 1 | 3.54× | 0 |
| `degenerate_vertex` | tiny | 2.16× (46→4) | 1 | 4.05× | 0 |
| `degenerate_vertex` | small | 2.23× (50→4) | 1 | 3.17× | 0 |
| `degenerate_vertex` | large | 2.16× (46→4) | 1 | 2.97× | 0 |
| `hanging_chain` | tiny | 4.00× (57→0) | 0 | 1.25× | 0 |
| `hanging_chain` | small | 4.44× (67→0) | 0 | 1.54× | 0 |
| `hanging_chain` | large | 6.84× (124→1) | 0 | 0.85× | 17 |
| `rosenbrock_ring` | tiny | 2.37× (30→1) | 0 | 11.37× | 0 |
| `rosenbrock_ring` | small | 2.26× (28→1) | 0 | 8.75× | 0 |
| `rosenbrock_ring` | large | 1.72× (18→1) | 0 | 6.73× | 0 |
| `rosenbrock_ring_cycle` | tiny | 2.32× (29→1) | 0 | 9.16× | 0 |
| `rosenbrock_ring_cycle` | small | 2.25× (28→1) | 0 | 8.62× | 0 |
| `rosenbrock_ring_cycle` | large | 1.52× (17→4) | 0 | 6.14× | 0 |
| `double_well_chain` | tiny | 1.00× (0→0) | 0 | 3.00× | 0 |
| `double_well_chain` | small | 1.00× (0→0) | 0 | 2.29× | 0 |
| `double_well_chain` | large | 1.00× (0→0) | 0 | 2.14× | 0 |
| `nmpc_vanderpol` | tiny | 18.80× (366→2) | 0 | 3.63× | 0 |
| `nmpc_vanderpol` | small | 12.55× (348→14) | 0 | 1.96× | 0 |
| `nmpc_vanderpol` | large | 7.33× (425→72) | 0 | 1.05× | 8 |
| `mpc_horizon_80` | tiny | 54.75× (1105→2) | 0 | 5.17× | 0 |
| `mpc_horizon_80` | small | 42.98× (1552→21) | 0 | 2.23× | 0 |
| `mpc_horizon_80` | large | 8.21× (1176→123) | 0 | 1.12× | 3 |

### Payoff tracks active-set churn, not problem size

Read down any family and the pattern is the same: **the further the
problem moves per step, the less a warm start buys.** `churn` is the
mean number of working-set entries that change between consecutive
steps.

| family | churn/step at tiny → large | SQP payoff at tiny → large |
|---|---|---|
| `nmpc_vanderpol` | 0.21 → 2.95 | 18.8× → 7.3× |
| `mpc_horizon_80` | 0.21 → 5.58 | 54.8× → 8.2× |
| `moving_bound_qp` | 0.05 → 1.63 | 6.5× → 13.7× |
| `hanging_chain` | 0.00 → 0.47 | 4.0× → 6.8× |
| `simplex_proj` | 0.00 → 0.21 | 16.0× → 18.6× |

The two MPC families are the clearest cases: a 14× and 27× increase in
churn costs a 2.6× and 6.7× reduction in payoff. This is the practical
rule — **warm starting pays in proportion to how stable your active set
is**, and problem size has little to do with it. (The families at the
bottom, whose churn stays below one entry per step even at `large`,
show the opposite sign: there the warm start stays essentially exact
while the cold solve gets harder, so the ratio rises.)

### Warm starting can make things worse

Two rows show it, both at the largest step size:

- `hanging_chain @ large`, `warm-ipm`: **0.85×** — the warm-started IPM
  needed *more* iterations than a cold solve on **17 of 19 steps**. The
  previous solution sits exactly on the constraint boundary, which is
  the worst possible starting point for a barrier method when the
  active set has since moved.
- `nmpc_vanderpol @ large`, `warm-ipm`: 8 of 19 steps worse, where a 4×
  control interval makes the plant state jump far enough that the
  previous point is a poor guess. The SQP arm no longer regresses on
  this row (it did before #428 was fixed), but its payoff still falls
  from 18.8× to 7.3× across the same span.

This is why the benchmark reports regressions per step rather than only
a mean. A single averaged speedup would hide both.

### How it scales: the MPC horizon sweep

The same linear MPC at four horizons, warm/cold **wall-time ratio** —
below 1.00 means warm starting won:

| N | n | mean &#124;A&#124; | `tiny` SQP / IPM | `small` SQP / IPM | `large` SQP / IPM |
|--:|--:|--:|--:|--:|--:|
| 10 | 32 | 31.0 | **0.17** / 0.37 | 0.17 / 0.38 | 0.24 / 0.69 |
| 20 | 62 | 61.2 | **0.08** / 0.37 | 0.09 / 0.70 | 0.12 / 0.74 |
| 40 | 122 | 118.3 | **0.04** / 0.32 | 0.04 / 0.59 | 0.11 / 0.85 |
| 80 | 242 | 204.1 | **0.02** / 0.26 | 0.03 / 0.49 | 0.10 / 0.86 |

Read down the SQP columns: the warm start does not merely survive the
horizon, it **improves with it** — 0.17 → 0.02 at `tiny`, and even at
the largest perturbation 0.24 → 0.10. At N = 80 a warm-started solve is
50× faster than a cold one at small steps and still 10× faster at large
ones. The reason is that cold cost grows with the problem while warm
cost is set by how far the problem moved, which is a property of the
path, not of `n`.

Reading across, the familiar pattern holds: bigger steps cost more
(0.02 → 0.10 at N = 80), because more of the active set has to change.

The mechanism is in the working sets. The *fraction* of the active set
that changes per step is essentially horizon-independent — about 3% at
`large` for every N, by construction, since the same angular
perturbation moves proportionally the same constraints:

| N | mean &#124;A&#124; | churn/step at `large` | as a fraction | SQP inner work, cold → warm |
|--:|--:|--:|--:|--:|
| 10 | 31.5 | 1.05 | 3.3% | 242 → 14 |
| 20 | 61.2 | 2.26 | 3.7% | 486 → 41 |
| 40 | 118.8 | 4.21 | 3.5% | 893 → 76 |
| 80 | 203.1 | 5.58 | 2.7% | 1176 → 123 |

Absolute churn does grow with the problem (1.05 → 5.58 changes per
step), and the warm arm's inner work grows with it — but the cold arm's
grows faster, which is why the *ratio* improves. The rule stands as
first stated: **payoff tracks how much the active set moves, and problem
size has little to do with it.**

An earlier revision of this page reported the opposite — a crossover
where warm-started SQP turned harmful above N = 20, reaching 2.57× at
N = 80. That was [#428](https://github.com/jkitchin/pounce/issues/428),
found by the large tier below and now fixed; the numbers above are the
same measurement on the fixed solver.

### At large scale: where the benchmark found a defect

Carrying the same MPC out to n = 2402 is what exposed #428, and the
before/after is the clearest single result in the suite.

At default settings the warm-started SQP **did not produce an answer**
on the large tier: `warm-sqp` and `warm-sqp-hom` returned
`Maximum_Iterations_Exceeded` with zero outer iterations on 7 of 8 steps
at every one of N = 200/400/800, leaving `x` at the warm-start point,
while every other arm solved all 8 cleanly.

Inner working-set changes for one step, before and after the fix:

| N | n | m | cold | warm, before | warm, after |
|--:|--:|--:|--:|--:|--:|
| 10 | 32 | 22 | 11 | 0 | 0 |
| 20 | 62 | 42 | 25 | 43 | 1 |
| 40 | 122 | 82 | 48 | 1 | 1 |
| 80 | 242 | 162 | 66 | 164 | 3 |
| 200 | 602 | 402 | 66 | **403** | 3 |
| 400 | 1202 | 802 | 66 | **795** | 3 |
| 800 | 2402 | 1602 | 66 | **1589** | 3 |

The warm arm was **Θ(m)** — 1589 pivots at N = 800, 24× the cost of not
warm starting at all. It is now flat at 3 across a 75× range of m, at
the same optimum to 1e-11.

The cause was not gradual erosion but a step function in how far the
problem moved. Before, at N = 200, *zero* changed entries of the true
active set cost 0 pivots and *one* cost 400. `solve_with_working_set`
pins the hinted rows to their new boundaries; once the active set has
moved, that pinned point violates some other row by roughly the distance
the parameter moved, and a feasibility pre-check in `solve` routed the
whole thing to elastic phase-1 — whose recovery re-solve starts from a
*cold* working set. The hint was discarded rather than repaired. The fix
repairs it: the violated rows are known, so they are pinned too and the
KKT re-factored, keeping the |A| − 1 entries the hint got right. Now the
cost tracks the movement, as it should:

| Δφ | entries of the true active set that changed | warm pivots, before | after |
|--:|--:|--:|--:|
| 0.002 | 0 | 0 | 0 |
| 0.005 | 0 | 0 | 0 |
| 0.01 | 1 | **400** | 0 |
| 0.02 | 2 | **401** | 1 |
| 0.05 | 4 | **403** | 3 |

On the large tier at default settings, the whole picture inverts. Every
arm is now correct on every step, and the SQP goes from unusable to the
fastest thing on the board:

| N | n | `warm-sqp` wall vs its cold twin | `warm-ipm` | `warm-qp-ipm` |
|--:|--:|--:|--:|--:|
| 200 | 602 | **0.03** | 0.58 | 0.48 |
| 400 | 1202 | **0.03** | 0.57 | 0.54 |
| 800 | 2402 | **0.02** | 0.41 | 0.50 |

Inner active-set work drops 514 → 11 per path (46.7×) identically at all
three horizons. At n = 2402 a warm-started SQP sweep takes 1.34 s
against 12.12 s cold.

This also revises the caveat in
[Active-Set SQP & Warm Starts](active-set-sqp.md) about preferring the
IPM for "large-scale problems with thousands of active inequalities".
With #428 fixed, this problem shows no such crossover up to 1645 active
constraints — the active-set path wins by 30–50× there.

### The parametric homotopy: a sharply mixed trade

The `-hom` arms differ from their twins in one option, so the delta is
the homotopy alone. Comparing inner QP active-set work on the **cold**
arms, where the homotopy actually engages (warm inner QPs mostly skip
the cold path):

| family | conventional → homotopy, cold inner work | ratio across the three scales |
|---|--:|--:|
| `simplex_proj` | 978 → 1400 | 0.63–0.74× |
| `moving_bound_qp` | 793 → 587 | 1.02–3.33× |
| `degenerate_corner` | 69 → 30 | 1.91–2.73× |
| `redundant_rows` | 247 → 30 | 3.91–11.27× |
| `degenerate_vertex` | 154 → 132 | 1.09–1.25× |
| `hanging_chain` | 257 → 257 | 1.00× |
| `rosenbrock_ring` | 79 → 79 | 1.00× |
| `rosenbrock_ring_cycle` | 77 → 77 | 1.00× |
| `double_well_chain` | 0 → 0 | — (no inner QP work at all) |
| `nmpc_vanderpol` | 1205 → 3575 | 0.33–0.36× |
| `mpc_horizon_10/20/40/80` | 9179 → 29839 | 0.25–0.37× |
| **all 42 rows** | **13038 → 36006** | **0.36×** |

Above 1.00× the homotopy did less work. The split is not random — it
tracks exactly what the homotopy was built for:

- **It wins on degenerate geometry.** `redundant_rows`, whose active
  set is linearly dependent, is its best case by a wide margin, and it
  improves with perturbation size (4.2× → 12.3× from `tiny` to
  `large`) because the conventional cold solve degrades there while the
  homotopy does not. `degenerate_corner` and `degenerate_vertex` follow
  the same pattern. This is the netlib-like geometry #412 reported it
  gaining 20 problems on.
- **It loses badly on well-conditioned MPC-shaped QPs.** Every
  `mpc_horizon_*` family and `nmpc_vanderpol` cost about **3×** the
  inner work with the homotopy on, consistently across scales, and
  `simplex_proj` costs ~1.4×.
- **It is inert on four families** — exactly 1.00×, because their inner
  QPs never take the cold path far enough for it to matter.

Net over all 42 rows it does 2.8× *more* inner work (0.36×), because
the losers are also the largest problems. That is an argument for
keeping it off by default on the SQP path and reaching for it on
degenerate models, which is what the option now allows.

### Three-way: which solver for a sequence of QPs?

Five families are literally convex QPs, so all three solvers can take
them. Interior-point iterations and active-set pivots are not the same
unit of work, so the like-for-like column is each solver against itself:

| family | scale | convex QP IPM cold→warm | NLP IPM cold→warm | SQP cold→warm (inner) | fastest warm arm |
|---|---|--:|--:|--:|---|
| `simplex_proj` | tiny | 160→46 | 182→28 | 300→15 | `warm-qp-ipm` |
| `simplex_proj` | small | 162→75 | 190→38 | 328→15 | `warm-qp-ipm` |
| `simplex_proj` | large | 173→96 | 200→45 | 350→15 | `warm-sqp` |
| `moving_bound_qp` | tiny | 202→94 | 228→43 | 109→5 | `warm-sqp` |
| `moving_bound_qp` | small | 195→121 | 224→116 | 212→5 | `warm-sqp` |
| `moving_bound_qp` | large | 229→125 | 240→126 | 472→24 | `warm-sqp` |
| `degenerate_corner` | tiny | 196→74 | 223→41 | 20→2 | `warm-qp-ipm` |
| `degenerate_corner` | small | 174→77 | 170→40 | 20→2 | `warm-qp-ipm` |
| `degenerate_corner` | large | 177→98 | 177→53 | 29→6 | `warm-sqp` |
| `redundant_rows` | tiny | 189→75 | 249→41 | 42→0 | `warm-qp-ipm` |
| `redundant_rows` | small | 173→80 | 207→44 | 82→11 | `warm-qp-ipm` |
| `redundant_rows` | large | 171→83 | 176→43 | 123→11 | `warm-qp-ipm` |
| `degenerate_vertex` | tiny | 215→73 | 192→39 | 50→8 | `warm-qp-ipm` |
| `degenerate_vertex` | small | 199→87 | 149→38 | 54→8 | `warm-sqp` |
| `degenerate_vertex` | large | 195→92 | 141→38 | 50→8 | `warm-qp-ipm` |

Geometric-mean wall time over those fifteen rows:

| cold-ipm | cold-sqp | cold-qp-ipm | warm-ipm | warm-sqp | warm-qp-ipm |
|--:|--:|--:|--:|--:|--:|
| 99.1 ms | 62.5 ms | 61.6 ms | 50.1 ms | 30.9 ms | **29.5 ms** |

The dedicated convex solver is fastest on 9 of the 15 rows and the
active-set SQP on the other 6, with the SQP taking the rows where the
active set churns hardest. The two are within 5% of each other on the
aggregate — on a problem that really is a QP, either warm-started path
is a reasonable default. Note that this ranking is recent: before
[#417](https://github.com/jkitchin/pounce/issues/417) was fixed the
convex solver's warm start was capped at 1.2–1.5× and `warm-sqp` led 8
of 9 rows.

## What to take from this

- **For a sequence of convex QPs** — `solve_qp` warm-started with the
  previous result. It leads on most rows and needs no callbacks.
- **For a general NLP whose active set is stable between solves** —
  `algorithm = active-set-sqp` carrying the working set. This is where
  the largest effects live (up to 55× less inner active-set work, and a
  50× wall-time win on the largest default horizon), and the whole
  reason the active-set path exists.
- **Scale is not the thing to worry about; movement is.** On the horizon
  sweep the SQP's warm/cold ratio *improves* with N (0.17 → 0.02 at
  small steps), because cold cost grows with the problem while warm cost
  is set by how far the active set moved. At n = 2402 a warm-started
  sweep runs 30–50× faster than cold. What costs you is a large step,
  not a large problem.
- **For a problem with no active set to speak of** — unconstrained, or
  with constraints that never bind — the warm start still helps, but
  only through the primal point. Either solver is fine; the working set
  buys nothing (`double_well_chain`: 0 → 0).
- **When each step moves the problem a long way** — check whether warm
  starting is helping at all. It can cost more than a cold solve, and
  the IPM path is more exposed to this than the SQP path.
- **On degenerate models — dependent rows, vertices where many
  constraints meet — try `sqp_qp_use_homotopy`.** It cuts inner
  active-set work by 2–12× on the degeneracy families and is the
  algorithm the active-set engine was designed around. Leave it off for
  MPC-shaped problems, where it roughly doubles the work.
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
| [#428](https://github.com/jkitchin/pounce/issues/428) | The SQP's working-set hint was discarded — not repaired — the moment the active set moved by one entry, costing one inner pivot per constraint row (1589 at n = 2402, against 3 now). Invisible below N ≈ 80; at n ≥ 602 it stopped the warm-started solve returning an answer at all. Found by the large tier on its first run, fixed in #429. |
| `sqp_qp_use_homotopy` was a no-op | Found while adding the `-hom` arms: the option was *registered* but `apply_qp_subproblem_options` never read it, so setting it on the SQP path did nothing while its documentation described what it would do. The inverse of #360 (read-but-unregistered), and invisible to that issue's guard, which only checked one direction. Fixed here, with a bidirectional guard. |

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
python -m warmstart.run --tier large --scales small   # n = 602 → 2402
```

`--tier large` is opt-in because a single active-set solve there takes
seconds; `--tier all` runs both.

Results land in `benchmarks/warmstart/results.json` (every step of every
arm) and `results.md`. Both are regenerated per run and gitignored.

Adding a problem family or a new solver is documented in
[`benchmarks/warmstart/README.md`](https://github.com/jkitchin/pounce/blob/main/benchmarks/warmstart/README.md);
nothing outside `adapters/` imports a solver, so the families and the
protocol are reusable against any solver with a warm-start API.

## Limits of these numbers

- **Mostly small problems** (n ≤ 47 outside the horizon sweep, which
  reaches n = 242 by default and n = 2402 with `--tier large`). The
  sweep gives one scaling curve on one problem shape; it is not a
  substitute for a large-scale study across problem classes, and the
  scaling it reports is specific to this MPC.
- **The large tier is one problem class.** Linear-quadratic MPC has a
  particular structure — banded, mostly equalities, a large active set
  that barely moves — and #428 was found there. Whether a large problem
  with a different sparsity pattern behaves the same way is untested,
  and is the obvious next family to add.
- **A published conclusion here has already been wrong once.** The
  horizon sweep's crossover held for one revision of this page before
  the large tier showed it was a solver defect. The measurements were
  right and the mechanism inferred from them was not; treat the
  explanations here as the current best reading of the numbers rather
  than as established behavior.
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
