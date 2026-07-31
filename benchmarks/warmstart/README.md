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

## The four arms

| arm | algorithm | seeded with |
|---|---|---|
| `cold-ipm` | interior point | nothing — the family's cold start |
| `cold-sqp` | active-set SQP | nothing |
| `warm-ipm` | interior point | previous step's primal-dual point and μ |
| `warm-sqp` | active-set SQP | previous step's working set and point |

Both cold arms are there so the warm-start effect can be separated
from the algorithm change. `warm-sqp` beating `cold-ipm` proves
nothing on its own — it mixes "warm started" with "switched
algorithms". Each warm arm is therefore scored against its *own* cold
counterpart.

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
| `degenerate_corner` | 6 | 3 | degenerate | objective | convex |
| `hanging_chain` | 30 | 15 | flipping | mixed | convex |
| `rosenbrock_ring` | 10 | 1 | switch | rhs | nonconvex |
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

Two families are worth knowing about in detail:

- **`degenerate_corner`** is a correctness probe, not a speed probe.
  Its path passes exactly through a point where one constraint's and
  one bound's multipliers are zero — strict complementarity fails, and
  the multiplier-sign classifier that builds a working set from a
  converged iterate has no signal to work with. The midpoint step
  lands on the degeneracy at *every* scale, by construction.
- **`nmpc_vanderpol`** is the only family whose path is not scripted:
  the next parameter is the state the plant reaches after applying the
  control the previous solve produced. Because that makes the path
  depend on the solutions, the runner records the sequence the
  reference arm produces and **replays** it for the other arms — the
  arms would otherwise be solving different problems.

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

## Known result worth not forgetting

`rosenbrock_ring` starts from the origin, not Rosenbrock's traditional
`(−1.2, 1, −1.2, …)`. From the traditional start, pounce's SQP path
with the default exact Hessian gives up with
`Search_Direction_Becomes_Too_Small` at every step of the path, at a
point with a stationarity residual of ~2.6, while `damped-bfgs` and
`lbfgs` converge from it fine. That is a cold-robustness finding, not
a warm-start one, and letting it sit inside this family would replace
the measurement with a wall of failures. It is written up in
`dev-notes/warm-start-benchmark.md`.
