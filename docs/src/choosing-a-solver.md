# Choosing a Solver

POUNCE is not a single solver but a small family of them sharing one
numerical backbone. This page is the map: what each solver is, when to
reach for it, and how they fit together.

![POUNCE solver landscape](images/solver-landscape.svg)

The one-sentence version: **convex and conic problems are solved to the
global optimum; general nonlinear problems are solved to a local (KKT)
point.** Every solver, whatever its flavor, ultimately factorizes a
symmetric KKT system through the shared `pounce-linsol` layer, which in
turn drives a pluggable backend (FERAL by default, HSL MA57 optionally).

## The solvers at a glance

| Solver | Problem class | Optimum | Crate | Entry points |
|---|---|---|---|---|
| **NLP filter-IPM** | general smooth NLP (nonconvex OK) | local (KKT) | `pounce-algorithm` + `pounce-nlp` | CLI default; Python `Problem`/`minimize`; `--solver nlp` |
| **NLP active-set SQP** | general smooth NLP | local | `pounce-algorithm` (subproblems via `pounce-qp`) | `algorithm=active-set-sqp` |
| **Convex IPM (LP/QP)** | LP, convex QP | **global** | `pounce-convex` | `solve_qp_ipm`; `pounce.qp.solve_qp`; `--solver lp-ipm`/`qp-ipm` |
| **Convex IPM (conic)** | SOCP, exponential, power cones | **global** | `pounce-convex` | `solve_socp_ipm`; `pounce.qp.solve_socp`; `pounce <file>.cbf` |
| **Active-set QP** | QP, convex *or* indefinite | local | `pounce-qp` | `ParametricActiveSetSolver`; `--solver qp-active-set` |

## When to choose each

### General nonlinear program (the common case) → **NLP filter-IPM**

If your model has nonlinear objective or constraints and you don't know
(or can't assume) convexity, this is the default and the most mature path.
It is POUNCE's port of Ipopt's filter line-search interior-point method:
robust on nonconvex problems, with a feasibility **restoration phase** for
hard starts and exact or limited-memory Hessians. It returns a local
KKT point — for a nonconvex problem there is no global guarantee.

- CLI: `pounce model.nl` (or a built-in problem).
- Python: the cyipopt-style `Problem` class, or the scipy-style
  `minimize` facade.
- Reach for **limited-memory** Hessians (`hessian_approximation=limited-memory`)
  when second derivatives are unavailable or expensive.

### A *sequence* of related NLPs, or a stable active set → **NLP active-set SQP**

Selected with `algorithm=active-set-sqp`. It solves the NLP as a sequence
of quadratic subproblems (handed to `pounce-qp`), which warm-starts
extremely well when the active set is stable across solves — e.g. a
parametric sweep or a control loop. For a single cold solve of a general
NLP, prefer the filter-IPM.

### Linear or convex quadratic program → **Convex IPM (LP/QP)**

If `P ⪰ 0` (or `P = 0` for an LP), use the convex interior-point solver:
it returns the **global** optimum, detects primal/dual infeasibility, and
offers warm-starting, batched and multiple-RHS solving, a build-once /
solve-many `QpFactorization` handle, and post-optimal **sensitivity**
(`QpSensitivity` — the sIPOPT analog). The CLI's `auto` routing classifies
an `.nl` and sends LP/convex-QP problems here automatically.

- Python: `pounce.qp.solve_qp` (and `solve_qp_batch`, `solve_qp_multi_rhs`).

### Second-order, exponential, or power cones → **Convex IPM (conic)**

The same convex solver, through its non-symmetric HSDE driver, handles
conic programs: second-order cones, and the **exponential** and **power**
cones that express geometric programming, entropy / log-sum-exp, logistic
models, and `p`-norm constraints. Also **global**. This is the path to use
when you can cast a nominally-nonconvex problem into a convex cone — you
trade modeling effort for a global guarantee.

- Python: `pounce.qp.solve_socp(..., cones=[("exp", 3), ("pow", 0.5), ...])`.
- CLI: a Conic Benchmark Format file, `pounce model.cbf` (see the CBLIB
  benchmark tier).

### Indefinite QP, or a QP inner-solver → **Active-set QP**

`pounce-qp` is a sparse parametric active-set solver that accepts an
**indefinite** Hessian (via inertia control), with two-sided bounds and
factorization-reuse across a homotopy. It is the engine behind the
active-set SQP path, and is the right choice for MPC-style problems or any
setting where you re-solve a slowly-changing QP many times. Use the convex
IPM instead when `P ⪰ 0` and you want a single robust solve with
infeasibility certificates.

## How to override the automatic routing

The CLI classifies each `.nl` problem and picks a solver, but you can force
the choice:

```sh
pounce model.nl --solver auto          # default: classify, then route
pounce model.nl --solver nlp           # filter-IPM (or active-set-sqp via algorithm=)
pounce model.nl --solver lp-ipm        # convex LP interior-point
pounce model.nl --solver qp-ipm        # convex QP interior-point
pounce model.nl --solver qp-active-set # active-set QP
```

See [LP / QP Solver Routing](lp-qp-routing.md) for how classification works
and when it falls back to the more general solver.

## The shared backbone

Every interior-point and active-set solver above assembles a symmetric KKT
system and factorizes it through **`pounce-linsol`**. That trait layer is
backend-agnostic:

- **FERAL** (`pounce-feral`) — a pure-Rust sparse symmetric LDLᵀ
  factorization. The default; no external dependencies.
- **HSL MA57** (`pounce-hsl`) — the well-known Harwell solver via
  `libcoinhsl`, enabled with the `ma57` build feature for large or
  ill-conditioned systems.

Because the backend is pluggable, the same solver code runs on either
without change.

## Cross-cutting layers

These are not solvers you select, but stages and tools the solvers share:

- **Presolve** (`pounce-presolve`) — an optional front-end that tightens
  bounds (feasibility-based bound tightening), removes redundant rows, and
  repairs LICQ degeneracies before the solve.
- **Restoration** (`pounce-restoration`) — the feasibility-recovery phase
  the filter-IPM enters when a step cannot reduce both infeasibility and
  the objective; `pounce-l1penalty` offers an ℓ₁-exact penalty
  reformulation for degenerate / LICQ-violating problems.
- **Sensitivity** — `pounce-sensitivity` gives sIPOPT-style parametric
  steps and reduced Hessians for the NLP; `QpSensitivity` does the same for
  the convex QP. See [Sensitivity Analysis](sensitivity.md).
- **Cone library** (`pounce-convex`) — nonnegative, second-order,
  exponential, and power cones today; the positive-semidefinite (PSD) cone
  is planned, which would add SDP as a convex class.
- **Solve report** — every path can emit the machine-readable
  `pounce.solve-report/v1` JSON (status, iterations, residuals, timing).
  See [JSON Solve Report](json-output.md).

## Global vs. local — the honest summary

POUNCE does not (yet) do *deterministic global optimization of nonconvex
problems* — there is no spatial branch-and-bound. What it offers is:

- **Global optima for convex problems** — LP, convex QP, SOCP, and the
  exponential / power cone classes. For these, local *is* global, so a
  convex or conic reformulation buys you a global guarantee.
- **Local optima for general nonlinear problems** — the filter-IPM and SQP
  paths converge to a KKT point.

So the practical lever for "global" answers today is **modeling**: the more
of a problem you can express in the convex cone library, the more of it the
convex solver settles globally.
