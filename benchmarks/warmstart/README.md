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
| `warm-sqp` | active-set SQP | previous step's working set and point | every family |
| `cold-sqp-hom` | active-set SQP, homotopy inner QP | nothing | every family |
| `warm-sqp-hom` | active-set SQP, homotopy inner QP | previous step's working set and point | every family |
| `cold-qp-ipm` | dedicated convex QP IPM (`pounce.solve_qp`) | nothing | QP families only |
| `warm-qp-ipm` | dedicated convex QP IPM | previous step's primal-dual point | QP families only |

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
| sparse or dense | sparse triplet KKT + sparse LDLᵀ; the Schur block alone is dense | **no** — families are n ≤ 47 and near-dense. Sparse coverage is the cold `.nl` Maros-Mészáros suite |
| convex only, or indefinite | indefinite, via §4.5 inertia control and negative-curvature ratio-test handling | **yes** — 5 of 10 families are nonconvex with indefinite ∇²L along the path; two solver defects were found there |
| primal or dual | primal; l1-elastic phase-1 for an infeasible cold start, and the homotopy is primal-feasible by construction | n/a — there is no dual variant to compare |
| parametric with hot starts | both: working-set hot start, and the §4.2 qpOASES-lineage homotopy | **yes** — hot starts are the `warm-*` arms; the homotopy is the `-hom` arms |
| degeneracy | Harris two-pass, GMSW EXPAND, Bland latch, rank-deficient active sets pruned to a maximal independent subset | **yes**, all three kinds: dual (`degenerate_corner`), rank-deficient / LICQ (`redundant_rows`), primal (`degenerate_vertex`) |

Still uncovered, and worth knowing: **scale**. Every family is small by
design, so nothing here says how the sparse path or the Schur updates
behave at n in the thousands. An MPC horizon sweep would be the natural
addition.

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

## Adding a family

Subclass `ParametricFamily` in `families/`, supplying the usual
cyipopt-shaped callbacks in **dense** form — the harness derives the
sparsity patterns and packed value vectors from them, so the structure
and the values cannot fall out of sync. List the class in
`families/__init__.py`, then:

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

## Three solver defects this suite has caught

All three are fixed. They are recorded because the shapes recur, and
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

Full write-ups in `dev-notes/warm-start-benchmark.md`.
