# Warm-start benchmark

Every other suite in `benchmarks/` measures **cold** solves: one
problem, one solve, from scratch. This one measures the thing that
only exists in sequences — how much a solver saves by starting from
the previous problem's solution, and whether it stays correct while
doing so.

The unit of work is a **parametric family plus a path**: one NLP
shape, one scripted sweep through its parameter space, solved end to
end. That is deliberate. Warm starting has no meaning for a single
isolated NLP, and the property that decides whether it pays — how the
active set moves from one instance to the next — only exists along a
path.

## Why this suite exists at all

There is no standard warm-start benchmark for NLP solvers to reuse. The
nearest public things and why each does not cover it:

| Existing | What it is | Gap |
|---|---|---|
| [qpbenchmark](https://github.com/qpsolvers/qpbenchmark) test sets ([MPC](https://github.com/qpsolvers/mpc_qpbenchmark), [IK](https://github.com/qpsolvers/ik_qpbenchmark), Maros-Mészáros, free-for-all) | Curated QP sets + runner | QP level, not NLP. Instances ship stripped of their sequence structure; the MPC set's README states outright that it "does not reflect the warm-starting that is frequently used on robots that do model predictive control". |
| [WARP](https://arxiv.org/abs/2605.05728) (2026) | Benchmark for primal-dual warm starting of interior-point solvers | Closest in spirit, but interior-point only (it predicts the barrier state), AC-OPF only, and organized around *learned* predictions over i.i.d. instances rather than consecutive perturbations. No active set. |
| [Not All Warm Starts Help](https://arxiv.org/abs/2606.08984) (2026) | Evaluation of primal-dual initializations for ACOPF | An evaluation, not a reusable set. Its central finding — warm starts frequently make things worse — is why this suite reports regressions as a first-class column. |
| [OPFData](https://arxiv.org/abs/2406.07234) / PGLearn | 300k solved AC-OPF instances per grid, loads perturbed 80–120%, plus N-1 topologies | A dataset with no protocol or metrics, single problem class, independent samples rather than a path. |
| CUTEst, Hock-Schittkowski, Vanderbei, Mittelmann, Maros-Mészáros | The standard NLP/QP collections (several already in `benchmarks/`) | Every problem is one cold solve. No parameter, no sequence. |

## The arms

| arm | algorithm | seeded with | runs on |
|---|---|---|---|
| `cold-ipm` | general NLP filter-IPM | nothing — the family's cold start | every family |
| `cold-sqp` | active-set SQP | nothing | every family |
| `warm-ipm` | general NLP filter-IPM | previous step's primal-dual point and μ | every family |
| `values-ipm` | general NLP filter-IPM | previous step's primal point **alone** — no multipliers, no μ | every family |
| `warm-sqp` | active-set SQP | previous step's working set and point | every family |
| `cold-sqp-hom` | active-set SQP, homotopy inner QP | nothing | every family |
| `warm-sqp-hom` | active-set SQP, homotopy inner QP | previous step's working set and point | every family |
| `cold-qp-ipm` | dedicated convex QP IPM (`pounce.solve_qp`) | nothing | QP families only |
| `warm-qp-ipm` | dedicated convex QP IPM | previous step's primal-dual point | QP families only |
| `warm-ipm-norecenter` | general NLP filter-IPM | as `warm-ipm`, with `warm_start_recentering=none` | every family |
| `cold-ipm-lsq` | general NLP filter-IPM, `least_square_init_primal=yes` | nothing | every family |
| `race-fixed` | general NLP filter-IPM | winner of a fixed-budget start race | every family |
| `race-halving` | general NLP filter-IPM | winner of a successive-halving race | every family |

The last four are pounce#611's; with `values-ipm`, which pounce#622
added for its own reasons, they complete the nine initializations that
issue lists. Three are paired controls rather than candidates for
"best":

- **`values-ipm` vs `warm-ipm`** is the value of carrying the
  *dual* state. Both start at the same primal point; only one is handed
  the multipliers and μ.
- **`warm-ipm-norecenter` vs `warm-ipm`** is the pre-#606 attribution
  control. It pins `warm_start_recentering=none` per-arm, so both
  settings appear in a single run rather than needing two sweeps to
  compare — which also means a recentering change shows up as a moving
  *gap* rather than as two numbers taken at different times.
- **`cold-ipm-lsq` vs `cold-ipm`** isolates the safeguarded
  least-squares normal step. It is a *cold* arm on purpose: the option
  decides where a cold solve starts, and means nothing next to a warm
  seed that supplies the primal point directly.

The two racing arms are cold too — they choose a start rather than
reuse one, which is the alternative strategy to warm starting rather
than a variant of it. The family's own cold start is always candidate 0
in the field, so an arm that still finishes behind `cold-ipm` was beaten
by its own ranking rule and not by unlucky sampling. Their tournament
cost is charged to `init_time`, reported separately from `solve_time`,
so an arm that wins the solve and loses on the total is visible as such.

Both cold arms are there so the warm-start effect can be separated
from the algorithm change. `warm-sqp` beating `cold-ipm` proves
nothing on its own — it mixes "warm started" with "switched
algorithms". Each warm arm is therefore scored against its *own* cold
counterpart.

The two `qp-ipm` arms are the dedicated convex solver, and they are
different in kind from the other four: they take the problem as
matrices rather than through callbacks. `qpform.py` extracts
`(P, c, A, b, G, h, lb, ub)` from a family whose instances are QPs —
verified by the self-test, which re-derives the family's objective,
gradient, and every constraint row from the extracted data rather than
trusting the family's `quadratic = True` claim. Families that are not
QPs skip these arms with a stated reason. Two consequences worth
holding on to when reading the numbers: the QP arms evaluate the model
*once per step* where the others re-evaluate every iteration, and an
interior-point iteration and an active-set pivot are not the same unit
of work. That is why they live in their own report section, compared
mainly against themselves cold-vs-warm.

`cold-ipm` is also the **reference arm**: the arm every other is
compared against, and the arm that generates the parameter path for
closed-loop families. It is a baseline, not ground truth — on a
nonconvex family it can and does converge to a worse local minimum
than another arm finds, which is why correctness is judged the way it
is below.

## What is measured

**Primary — inner active-set work.** `Σ` active-set changes (adds +
drops) inside the QP subproblems, from `info["n_qp_ws_changes"]`.
This is what a working-set warm start actually reduces. Outer SQP
iterations are *not* a sufficient metric: on a family whose subproblem
is already a QP, the outer loop terminates in one iteration whether it
was warm started or not, so that column is flat by construction while
the inner work varies by more than an order of magnitude.

**Secondary** — outer iterations, evaluation counts (counted by the
harness, not the solver, so they mean the same thing across solvers),
wall time. At these problem sizes wall time is dominated by the Python
callback round trip; read it as a cross-check, not a result.

**Correctness**, and this is the point of the suite as much as the
speed is. A warm start that converges fast to the wrong answer is not
a win. Each step must:

1. return a success status,
2. actually achieve a small KKT residual and be feasible — checked by
   the harness against `--kkt-gate` / `--viol-gate`, independent of
   what the solver claims,
3. not land on a *worse* optimum than the reference arm (by more than
   `--obj-tol`).

Finding a **better** optimum than the reference is not an error; it is
reported in its own column. `‖x − x_ref‖` is recorded as a diagnostic
but is not a gate, because two solves can both be optimal and still
differ in `x` — a degenerate face, a flat direction.

**Regressions.** Steps where the warm arm cost *more* than the cold
one are counted and listed individually. A benchmark that reports only
the mean speedup hides the failure mode that matters most.

## The families

| family | n | m | regime | channel | curvature |
|---|--:|--:|---|---|---|
| `simplex_proj` | 20 | 1 | flipping | objective | convex |
| `moving_bound_qp` | 40 | 3 | flipping | bounds | convex |
| `degenerate_corner` | 6 | 3 | dual degenerate | objective | convex |
| `redundant_rows` | 6 | 5 | rank-deficient (LICQ fails) | objective | convex |
| `degenerate_vertex` | 4 | 12 | primal degenerate | objective | convex |
| `hanging_chain` | 30 | 15 | flipping | mixed | convex |
| `rosenbrock_ring` | 10 | 1 | switch | rhs | nonconvex |
| `rosenbrock_ring_cycle` | 10 | 1 | re-activation | rhs | nonconvex |
| `double_well_chain` | 12 | 0 | none (empty active set) | objective | nonconvex |
| `nmpc_vanderpol` | 47 | 32 | closed-loop | rhs | nonconvex |
| `mpc_horizon_10/20/40/80` | 32–242 | 22–162 | saturation | rhs | convex |
| `mpc_horizon_200/400/800` | 602–2402 | 402–1602 | saturation | rhs | convex |
| `badly_scaled_qp` | 12 | 4 | scaling | objective | convex |
| `rastrigin_drift` | 10 | 1 | multi-basin | objective | nonconvex |
| `rastrigin_scatter` | 10 | 1 | **unrelated** | objective | nonconvex |
| `elliptic_control_40/80/160` | 82–322 | 42–162 | moving band | rhs | convex |
| `elliptic_control_600` | 1202 | 602 | moving band | rhs | convex |
| `resistive_network_60/120` | 90–180 | 59–119 | congestion | rhs | convex |
| `resistive_network_800` | 1200 | 799 | congestion | rhs | convex |

The last block is pounce#611's. Three of them exist because the suite
had a structural blind spot rather than a missing data point:

- **`rastrigin_drift` / `rastrigin_scatter` are the falsification
  arm.** Every family above them was chosen by warm-start work, so the
  suite could measure how much warm starting won but never whether it
  did. These are shifted Rastrigin problems — a global minimum in a
  lattice of local minima spaced 1 apart — where the seed's basin is
  something the benchmark controls directly. `drift` walks a path with
  a per-coordinate step of `0.3 x scale`, so `tiny` and `small` stay in
  one basin and `large` does not; `scatter` is not a path at all, just
  independent draws, which is the issue's "unrelated global/nonconvex
  cases where continuation should not be expected to help". **A
  wrong-basin step converges cleanly and lands on a worse optimum**, so
  it appears in the `bad` column and never in a status code — and the
  arm can look *faster* on exactly the steps it got wrong.
- **`elliptic_control_*` and `resistive_network_*` are the sparsity
  and conditioning axes.** Before them the only family that reached
  size was linear-quadratic MPC: block-banded, constant Hessian, large
  active set that barely moves. The elliptic family is tridiagonal and
  symmetric with conditioning growing like `h⁻²`, so refining its mesh
  makes it harder for a reason unrelated to its size; the network
  family's incidence pattern has two entries per column with endpoints
  a third of the graph apart, so no permutation makes it banded, and
  its quartic loss means the Hessian actually changes between
  iterations.
- **`badly_scaled_qp`** spans 10⁸ in Hessian conditioning and 10³ in
  row scaling, which is the axis every initialization heuristic is
  implicitly assuming something about.

The last row is the opt-in **`large` tier** (`--tier large`; `--tier
all` runs both). It is out of the default sweep because one active-set
solve at N = 800 takes seconds, and it walks 8 steps rather than 20
since the per-step numbers are what matter, not the path length.

*regime* is how the active set behaves along the path; *channel* is
where the parameter enters (objective, constraint right-hand side,
variable bounds, or several); *curvature* is the obvious thing.

Each family runs at three **scales**, which multiply its natural
per-step parameter increment: `tiny` (0.1), `small` (1.0), `large`
(4.0). Warm-start payoff is a function of how far the problem moved,
so a single step size would measure one point on a curve and call it
the answer. `tiny` is the continuation regime where the active set
barely moves; `large` is where it churns and warm starting can hurt.

### What the active-set regimes actually cover

Read off the working sets the solver returns, not off the intent:

| situation | where it is exercised |
|---|---|
| equality rows, permanently in the working set | `nmpc_vanderpol` (32), `moving_bound_qp` (3), `simplex_proj` (1) |
| variable bounds activating and releasing | `simplex_proj` (15→19 clamped), `moving_bound_qp` (2→22), `nmpc_vanderpol` (control saturation) |
| inequality rows activating and releasing | `hanging_chain` (3→12 ground contacts), `degenerate_corner`, both Rosenbrock families |
| active set held *fixed* for a whole path | `simplex_proj` and `hanging_chain` at `tiny` — 0 changes across the path |
| a single clean active→inactive switch | `rosenbrock_ring`, on the midpoint step at every scale |
| inactive→active **re-activation** | `rosenbrock_ring_cycle`, which crosses the switch out and back |
| **empty** active set, whole path, no constraint rows at all | `double_well_chain` (`m = 0`, no finite bounds) |

The last two exist because their absence was a real hole. A
monotone sweep only ever tests active→inactive; carrying an *empty*
working set into a step that needs a non-empty one is the harder
direction and now has a family. And with every family carrying at
least one constraint row, the suite never executed the unconstrained
configuration at all — which is both the zero mark the speedups
should be read against and, per pounce#416, a configuration with real
defects hiding in it.

Two families are worth knowing about in detail:

- **`degenerate_corner`** is a correctness probe, not a speed probe.
  Its path passes exactly through a point where one constraint's and
  one bound's multipliers are zero — strict complementarity fails, and
  the multiplier-sign classifier that builds a working set from a
  converged iterate has no signal to work with. The midpoint step
  lands on the degeneracy at *every* scale, by construction.
- **`double_well_chain`** is the control, and it inverts the usual
  reading of the report. With no constraints there is no working set
  to carry, so the QP-active-set column is 0 for both arms and the
  entire warm-start effect lands in *outer* iterations instead —
  cold 480 → warm 62 at `tiny`. The QP-shaped families are the mirror
  image (flat outer, all effect inner). Any interpretation of these
  numbers that only looks at one of the two columns is wrong on half
  the suite.
- **`nmpc_vanderpol`** is the only family whose path is not scripted:
  the next parameter is the state the plant reaches after applying the
  control the previous solve produced. Because that makes the path
  depend on the solutions, the runner records the sequence the
  reference arm produces and **replays** it for the other arms — the
  arms would otherwise be solving different problems.

## What QP-solver properties this exercises

The suite reaches `pounce-qp` only through the SQP outer loop, so it is
not a QP-solver benchmark — but the questions people ask about an
active-set QP code map onto it as follows:

| property | `pounce-qp` | exercised here? |
|---|---|---|
| sparse or dense | sparse triplet KKT + sparse LDLᵀ; the Schur block alone is dense | **yes** — the `mpc_horizon_*` sweep runs the same block-banded MPC from n = 32 to n = 2402 (`--tier large`), nothing dense materialized anywhere along it |
| convex only, or indefinite | indefinite, via §4.5 inertia control and negative-curvature ratio-test handling | **yes** — 5 of 10 families are nonconvex with indefinite ∇²L along the path; two solver defects were found there |
| primal or dual | primal; l1-elastic phase-1 for an infeasible cold start, and the homotopy is primal-feasible by construction | n/a — there is no dual variant to compare |
| parametric with hot starts | both: working-set hot start, and the §4.2 qpOASES-lineage homotopy | **yes** — hot starts are the `warm-*` arms; the homotopy is the `-hom` arms |
| degeneracy | Harris two-pass, GMSW EXPAND, Bland latch, rank-deficient active sets pruned to a maximal independent subset | **yes**, all three kinds: dual (`degenerate_corner`), rank-deficient / LICQ (`redundant_rows`), primal (`degenerate_vertex`) |

The horizon sweep (`mpc_horizon_*`, the same linear MPC at seven sizes)
is what covers the scale axis, and carrying it to n = 2402
(`--tier large`) is what found
[#428](https://github.com/jkitchin/pounce/issues/428), the suite's
fourth solver defect: the working-set hint was *discarded* rather than
repaired the moment the active set moved by one entry, costing one
inner pivot per constraint row. Cold inner work is flat at 66 pivots
from N = 10 to N = 800; warm ran 0 → 43 → 164 → 403 → 795 → 1589,
tracking m exactly. At the default inner-QP budget of 200 the
warm-started SQP stopped returning an answer at all above m = 200 —
7 of 8 steps `Maximum_Iterations_Exceeded` at every large horizon,
while every other arm was clean.

Fixed in #429 (repair the hint instead of discarding it). Warm inner
work is now flat at 3 across the whole range, and the sweep says the
opposite of what it said before: the SQP's warm/cold wall ratio
*improves* with horizon — 0.17 → 0.08 → 0.04 → 0.02 at `tiny` for
N = 10…80, and 0.03 / 0.03 / 0.02 at N = 200/400/800 — because cold
cost grows with the problem while warm cost is set by how far the
active set moved.

## Running it

The harness drives the solver in-process through the Python API (the
CLI has no cross-process working-set carry for `.nl` files, so unlike
every other suite this one is not `.nl`-driven). It needs the Python
extension built:

    cd python && maturin develop --release

Then, from `benchmarks/`:

    make -C benchmarks warmstart-selftest   # derivative checks, no solver needed
    make -C benchmarks warmstart-run        # full sweep -> results.json + results.md
    make -C benchmarks warmstart-quick      # 3 families, one scale

or directly, for a narrower run:

    python -m warmstart.run --families simplex_proj,nmpc_vanderpol --scales large -v
    python -m warmstart.run --arms cold-sqp,warm-sqp --tol 1e-10
    python -m warmstart.report results.json      # re-render without re-running

`results.json` (every step of every arm) and `results.md` are
regenerated per run and gitignored, like every other suite's outputs.

### The external-solver arm

    python -m warmstart.run --solver ipopt --out warmstart/results-ipopt.json

needs cyipopt, which has no PyPI wheel and builds against a system
Ipopt. On Debian/Ubuntu:

    apt-get install -y coinor-libipopt-dev liblapack-dev libblas-dev
    pip install cython && pip install --no-build-isolation cyipopt

`--solver ipopt` without it exits with those instructions rather than
"unknown solver". Ipopt has no active-set, QP-matrix or sensitivity
path, so those arms are skipped with a recorded reason; the three it
does run (`cold-ipm`, `warm-ipm`, `values-ipm`) go through the
*same* callback object pounce is given, so evaluation counts compare.

### Changed structure

The sweep above holds each family's shape fixed, which is what an
ordinary warm start assumes. Horizon shifts and mesh refinement do not:

    python -m warmstart.transfer --experiment all --out warmstart/transfer.json

`shift` reindexes the previous horizon by one stage
(`WarmStart.reindex`) on `mpc_horizon_*`; `shift-cl` does the same on
the genuinely closed-loop `nmpc_vanderpol`; `mesh` prolongs an elliptic
control solution onto a mesh of twice the resolution
(`WarmStart.transfer` with an interpolation mapper), with and without
the mesh-dependent multiplier scaling.

### The composite report

    python -m warmstart.composite --results warmstart/results.json \
        --ipopt warmstart/results-ipopt.json \
        --transfer warmstart/transfer.json \
        -o ../dev-notes/warm-start-611-composite.md \
        --json-out ../dev-notes/warm-start-611-composite.json

Every table in that document — including the performance and data
profiles — is computed from those three JSON files. None is
transcribed, so re-measuring after a solver change is a re-run rather
than an editing pass. Sections whose input is missing say so instead of
vanishing: an absent section and an empty one mean different things to
a reader deciding whether a claim is supported.

## Adding a family

Subclass `ParametricFamily` in `families/`, supplying the usual
cyipopt-shaped callbacks in **dense** form — the harness derives the
sparsity patterns and packed value vectors from them, so the structure
and the values cannot fall out of sync. List the class in
`families/__init__.py`, then:

A family too large for a dense matrix may instead declare
`sparse_structure()` plus `jacobian_values()` / `hessian_values()`, and
set `tier = "large"`. Keep the dense methods as well where you can: the
self-test cross-checks the two against each other at any size it can
afford, and finite-differences the declared structure column by column
above that, so a structure that disagrees with its values still cannot
pass silently. That check is the only thing standing between a
mis-declared pattern and a plausible, wrong benchmark number.

    python -m warmstart.selftest

which finite-difference checks the gradient, Jacobian and Hessian of
the Lagrangian at several points along the path. A wrong derivative in
a benchmark family does not announce itself — it shows up as "the warm
arm needed more iterations", which is the exact signal the suite
exists to measure. Run it before believing any result.

## Adding a solver

Implement `SolverAdapter` in `adapters/` and register it in
`adapters/__init__.py`. Nothing outside `adapters/` imports a solver:
families, the protocol, and the report are solver-agnostic, and the
warm-start payload that crosses between steps (`WarmState`) is a plain
data record — primal point, multipliers, barrier parameter, working
set — that an adapter consumes as much of as its solver understands.
Arms an adapter does not support are reported as skipped rather than
silently dropped.

## Four solver defects this suite has caught

All four are fixed. They are recorded because the shapes recur, and
because two of them lived in the same configuration — nonconvex,
indefinite Hessian, nothing active — which is why `double_well_chain`
exists.

1. **pounce#416, fixed in #419.** From Rosenbrock's traditional
   `(-1.2, 1, -1.2, ...)` start, the exact-Hessian SQP path gave up with
   `Search_Direction_Becomes_Too_Small` at every step. The inner QP was
   spending its entire 200-iteration budget making **zero** working-set
   changes; a cap of 20 produced bit-identical answers ~9x faster. Fixed
   by capping the ratio test at the shifted step's true minimizer
   instead of at alpha = 1. `rosenbrock_ring` still starts from the
   origin, now only for continuity of its recorded numbers.

2. **pounce#423, fixed in #424.** The fix for #416 regressed the
   *unconstrained* case: with `m = 0` and no finite bounds a
   negative-curvature direction has nothing to block it, so the new
   recession-certificate path was taken at every indefinite iterate and
   the solve died at iteration 1 (f = 26.03 against the IPM's 0.027).
   `double_well_chain` caught it on its first run against the new base,
   the day after it was added. The fix gives the driver a third branch
   for "model recedes, NLP does not"; the family is back to its
   pre-#419 numbers exactly.

3. **pounce#417, fixed in #422.** `solve_qp`'s warm start was leaving
   ~40% of its iterations unclaimed — not because of the seeding, which
   is sound, but because the corrector's fraction-to-boundary parameter
   was pinned at 0.95, capping progress at ~20x per iteration however
   good the start was. Fixed by letting tau approach 1 as mu -> 0 on
   orthant blocks (the restriction matters: doing it for every cone kind
   loses 60% of the SOC instances the direct driver solves). The shipped
   fix reproduced this suite's predicted iteration counts exactly — 46,
   75 and 96 warm iterations on `simplex_proj`, against 46, 75 and 96
   measured from the prototype.

4. **pounce#428, fixed in #429.** The SQP's working-set hint was
   *discarded* rather than repaired the moment the true active set moved
   by a single entry, costing one inner pivot per constraint row —
   1589 at n = 2402, against 3 now, and 24x the cost of not warm
   starting at all. Below N = 80 the penalty is smaller than a cold
   solve, so the default tier's rows looked merely unimpressive; the
   `large` tier found it on its first run, where it stopped the warm
   arm returning an answer at all. It had also produced a *published
   wrong conclusion*: the horizon sweep's "warm starting turns harmful
   at scale" crossover was this defect, and on the fixed solver the
   ratio improves with horizon instead.

Full write-ups in `dev-notes/warm-start-benchmark.md`. The fourth is
worth reading for how it was missed: the sweep's numbers were correct
and a plausible mechanism was inferred from them that happened to be
wrong, because nobody compared the warm arm's pivot count against the
*true* active-set difference until the discrepancy grew to 164 against
4.
