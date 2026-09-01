# Convex Solver: LP, QP, and SOCP

POUNCE ships a specialized **convex conic interior-point solver**
(`pounce-convex`) alongside the general NLP filter-IPM. It solves the
standard-form convex program

```text
minimize    ½ xᵀP x + cᵀx
subject to  A x = b
            G x ⪯_K h
            lb ≤ x ≤ ub
```

where `P ⪰ 0` and the inequality block lies in a product cone `K` of
nonnegative orthants and second-order cones. `P = 0` is an LP; an
all-orthant `K` is an LP/QP; second-order blocks make it an **SOCP**.

The method is a **Mehrotra predictor–corrector** primal–dual interior-point
algorithm with Nesterov–Todd scaling for the cones, sharing the pure-Rust
[`feral`](algorithm.md) sparse LDLᵀ backend with the NLP path. It reaches
optimality in materially fewer iterations than routing the same problem
through the general NLP solver (≈30–50% fewer on bound/inequality QPs).

> **Inspiration.** The conic interior-point design follows
> [Clarabel](https://github.com/oxfordcontrol/Clarabel.rs) (Goulart &
> Chen) — handling a quadratic objective directly and a product of
> symmetric cones — and the presolve follows
> [PaPILO](https://github.com/scipopt/papilo) (the presolving library of
> [SCIP](https://www.scipopt.org/)). POUNCE does not wrap either (the
> pure-Rust guarantee) but ports their ideas; see
> [Acknowledgments](acknowledgments.md).

This chapter covers the **Python API** (`pounce.qp` and the differentiable
`pounce.jax` layers). For automatic CLI/Pyomo routing of `.nl` LPs/QPs, see
[LP / QP Solver Routing](lp-qp-routing.md). Runnable, progressive notebooks
live in [`python/notebooks/`](https://github.com/jkitchin/pounce/tree/main/python/notebooks):
`15_convex_qp.ipynb`, `16_socp.ipynb`, `17_differentiable_convex.ipynb`.

## Quadratic programs

```python
import numpy as np
from pounce.qp import solve_qp

# min ½·2‖x‖² − 3x₀ − 4x₁  s.t.  x₀ + x₁ ≤ 1,  0 ≤ x ≤ 1
r = solve_qp(
    P=np.diag([2.0, 2.0]),
    c=[-3.0, -4.0],
    G=[[1.0, 1.0]], h=[1.0],
    lb=[0, 0], ub=[1, 1],
)
r.status   # 'optimal'
r.x        # primal solution
r.y, r.z   # equality / inequality multipliers
r.z_lb, r.z_ub  # bound multipliers (≥ 0)
r.obj, r.iters
```

`P` (lower triangle used, assumed symmetric), `A`, and `G` accept dense
arrays or scipy-sparse matrices; any of them may be omitted. The result is
a `QpResult` dataclass with a `.success` property. The solver reports
**verified** infeasibility / unboundedness (`'primal_infeasible'` /
`'dual_infeasible'`) backed by a Farkas / recession certificate rather than
an iteration-limit guess.

## Second-order cone programs

A second-order (Lorentz) cone is `{ (t, x) : t ≥ ‖x‖₂ }`. Partition the
inequality rows of `Gx ⪯_K h` with `cones` — a list of `(kind, dim)` specs
(`"nonneg"` or `"soc"`; a bare int means a second-order cone). Each slack
block `s = h − Gx` must lie in its cone.

```python
from pounce.qp import solve_socp

# minimize ‖x − x*‖  ⇔  min t s.t. (t, x − x*) ∈ SOC
r = solve_socp(
    c=[1.0, 0.0, 0.0],                 # minimize t
    G=-np.eye(3), h=[0.0, -2.0, 1.0],  # s = (t, x₀−2, x₁+1) ∈ SOC(3)
    cones=[("soc", 3)],
)
r.x   # ≈ [0, 2, -1]:  t* = 0, x = x*
```

Mixed cones compose — e.g. `cones=[("nonneg", 1), ("soc", 2)]` puts the
first slack in `ℝ₊` and the next two in a 2-D second-order cone. Large
cones use a **sparse diagonal-plus-rank-1** KKT representation (one
auxiliary variable per cone, the ECOS/Clarabel "sparse SOC" trick) so the
factorization stays sparse.

## Warm starting

Feed a previous (or nearby) solution back to seed the interior-point
iteration — useful for parametric sweeps, receding-horizon MPC, and
branch-and-bound subproblems:

```python
base = solve_qp(P=P, c=c, G=G, h=h, lb=lb, ub=ub)
nxt  = solve_qp(P=P, c=c2, G=G, h=h, lb=lb, ub=ub, warm_start=base)
```

The warm start only affects the iteration count, never the solution (a
mismatch is ignored). The recentering is **adaptive** for the orthant
(sized to the warm point's KKT residual, so it exploits a nearby problem's
duals yet self-corrects when the active set moves) and re-centers the cone
duals for second-order blocks (a converged conic point sits on the cone
boundary, where the scaling is singular).

### The step length is what makes it pay off

A warm start lowers the starting duality measure μ₀; whether that turns into
*fewer iterations* depends on how much of each Newton step the solver is
allowed to take. With a **static** fraction-to-boundary parameter τ, every
step covers at most a τ fraction of the distance to the cone boundary, so μ
falls by a fixed factor per iteration and the count is `log₁/₍₁₋τ₎(μ₀/tol)`
however good the start was — a logarithm of the perturbation, not the one or
two Newton steps a nearby problem deserves.

So on orthant blocks the step follows the Mehrotra tail
`τ = clamp(1 − μ, tau, tau_max)`: as the solve converges τ approaches 1 and a
near-optimal iterate takes a near-full Newton step. On the QP families in the
warm-start benchmark this is worth 35–60% of the warm iterations. Both ends
are tunable, and both are `method="ipm"` only:

```python
r = solve_qp(P=P, c=c2, G=G, h=h, warm_start=base,
             tau=0.95,        # floor: the flat τ far from the solution
             tau_max=0.999)   # ceiling on the tail (default: just under 1)
```

Passing `tau_max=tau` pins τ flat — the most conservative setting, and the
one to reach for if a badly-conditioned sequence starts producing
`numerical_failure`. Two scopes are deliberate and not tunable: second-order
and PSD blocks always keep the static `tau` (their boundary is curved, and an
iterate that close to it breaks the Nesterov–Todd scaling), and **cold**
solves are unaffected because they run the homogeneous self-dual embedding,
a different loop.

## Wall-clock budgets

`solve_qp`, `solve_socp`, `solve_qp_batch`, and `solve_qp_multi_rhs` take a
`time_limit` in **seconds** (`None`, the default, means unbounded). Reach for it
when an answer is needed on a schedule — a receding-horizon controller with a
fixed control period, a sweep where one pathological instance must not stall the
rest, or any solve sitting behind a request:

```python
r = solve_qp(P=P, c=c, G=G, h=h, warm_start=previous, time_limit=0.005)
if r.status == "time_limit":
    ...   # `r.x` is the best iterate reached, not a KKT point
```

`max_iter` cannot express this. One interior-point iteration may be a single KKT
solve, or a factorization plus several inertia-controlled refactorizations with
escalating shifts, and the LP route can add a simplex crossover phase — so
per-iteration cost varies by more than an order of magnitude *within* one solve,
before problem size enters into it. No iteration count means "5 ms" across two
problems.

Three properties are worth knowing:

- **A verdict outranks the clock.** `optimal`, `optimal_inaccurate`,
  `primal_infeasible`, and `dual_infeasible` survive a deadline that passed
  while the solve was finishing; only a give-up result is relabelled
  `time_limit`. So the status is always truthful about what was proved, and a
  budget can never turn into a wrong `optimal`.
- **The budget is per solve, not per call.** On the batched entry points each
  instance opens its own deadline scope, so `time_limit=10` over 100 problems
  permits 1000 s of wall clock. A shared clock would make *which* instances get
  cancelled depend on rayon's scheduling, and so on the machine. Bound the whole
  call around the call.
- **Results become machine- and load-dependent**, inherently — which is why this
  is opt-in and absent from the default path. An in-flight factorization is not
  interrupted, so expiry can overshoot by one such operation.

The differentiable layers (`pounce.jax`, `pounce.torch`) deliberately do not
take one: they raise on `time_limit` because a non-KKT iterate makes the
implicit-function gradient meaningless, and silently wrong gradients under load
are worse than a slow layer. On the CLI the same mechanism is spelled
`max_wall_time`.

## Batching and factorization reuse

```python
from pounce.qp import solve_qp_batch, QpFactorization

# Solve many independent QPs in parallel (rayon, across instances).
results = solve_qp_batch([dict(P=P, c=c_k, G=G, h=h) for c_k in cs])

# Build the KKT symbolic factor once, solve many same-structure problems.
fac = QpFactorization(P=P, c=c0, G=G, h=h, lb=lb, ub=ub)
for c_k in cs:
    rk = fac.solve(P=P, c=c_k, G=G, h=h, lb=lb, ub=ub)  # reuses the factor
```

`solve_qp_batch` parallelizes across instances (outer-parallel /
inner-serial) and `QpFactorization` reuses the AMD ordering and symbolic
factorization across solves that share a structure — the two compose with
warm starting.

## Post-optimal sensitivity (`QpSensitivity`)

`QpSensitivity` is the convex arm's sIPOPT analog: it holds the factored
active-set KKT system at the optimum, so each `parametric_step` is a single
back-substitution.

```python
from pounce.qp import QpSensitivity

# min ½‖x‖²  s.t.  x₀ + x₁ = 2   →   x* = (1, 1),  dx/db = (½, ½)
s = QpSensitivity(P=np.eye(2), c=[0.0, 0.0], A=[[1.0, 1.0]], b=[2.0])
dx = s.parametric_step([0], [1.0])       # perturb b₀ by +1
```

It perturbs the **equality right-hand side** `b`, and reports the active set,
the weakly-active set, a reduced Hessian, and two conditioning diagnostics
(`ill_conditioned`, `last_step_residual`) that let a caller detect a step it
should not trust.

### Holding the step inside the bounds

The plain step is a linear predictor, so a large enough perturbation can point
outside the variable box. `parametric_step_bounded` repairs that the way the NLP
arm does — by pinning the crossing coordinate at its bound and re-solving, so the
other coordinates move to suit and the constraints still hold. Clipping instead
would satisfy the bounds and quietly break the equalities.

```rust
let (dx, pinned, stop) =
    sens.parametric_step_bounded(&[0], &[-6.0], /* bound_eps */ 1e-3, /* max_iter */ 16)?;
```

This is not a second implementation: it runs
`pounce_sens_core::boundcheck::refine_step_onto_bounds`, the same code the NLP
arm runs, reached through `QpSensitivity::backsolver()`. That machinery is
generic over the `SensBacksolver` trait, whose whole required surface is `dim()`
and `solve(rhs, lhs)`, so an engine that can back-solve against its converged
factor gets fix-relax, path following and the directional derivative without
porting any of them.

Both halves of fix-relax are available: a coordinate the step carries past a
bound is pinned there, and a bound whose multiplier the step drives *negative*
is released so the variable can leave. Releasing is exact on this arm — the
convex active-set KKT has no barrier term to destroy, so it costs one numeric
refactorization against an unchanged sparsity pattern.

### Following the path

`parametric_step_path` applies the perturbation a little at a time, stopping
wherever the active set changes:

```rust
let (dx, segments) = sens.parametric_step_path(&[0], &[3.0], /* max_iter */ 32)?;
for s in &segments {
    println!("at {:.3}: x{} {} its {} bound",
             s.at, s.var_row, if s.pinned { "reached" } else { "left" },
             if s.lower { "lower" } else { "upper" });
}
```

A QP's solution path is piecewise affine, so within a segment the walk is exact
and the reported breakpoints are the real ones. Use this when a perturbation is
large enough to change the active set more than once, or when you want the
events rather than only the endpoint.

### Degenerate LPs need crossover

`lp_without_crossover()` is `true` when the problem is a pure LP (`P = 0`) whose
solve did not run crossover. At a degenerate optimal vertex more constraints are
active than there are variables, the active-set KKT is rank-deficient, and
`dx/db` is not single-valued — on a two-variable example the step comes back
summing to half the perturbation it should. `ill_conditioned()` already catches
that; this flag names the cause. The fix is to solve with `qp_crossover=yes`, so
the interior point is pivoted to an exact vertex basis first.

Because the flag reads `opts.crossover`, the options you hand to `build` must be
the options the solve actually ran with.

### It is an orthant capability, and says so

`QpSensitivity` covers LP and convex QP — problems whose inequality block is
a nonnegative orthant. It does **not** cover the conic arm (SOCP, QCQP,
exponential, power, PSD).

That distinction matters more than it looks, because `solve_socp_ipm` and
`solve_qp_ipm` return the *same* `QpSolution` type and the cone partition
travels beside it as a separate `cones` argument. So on the Rust API, handing
a solved conic program to `QpSensitivity::build` used to be accepted and
answered — every cone row read as an orthant row, producing a number that was
not a derivative, with no warning. It is now refused with
`SensError::NotOrthantComplementary`: an orthant row complements *row by row*
(`sᵢ ≥ 0`, `zᵢ ≥ 0`, `sᵢzᵢ ≈ μ`), while a cone satisfies only the block inner
product `⟨s, z⟩ = 0`.

Use `QpSensitivity::build_conic(prob, cones, sol, …)` for a problem that
carries cones. An all-`Nonneg` partition is the orthant problem and behaves
identically to `build`; every other family returns
`SensError::UnsupportedCone`. A cone's active object is the tangent/normal
decomposition of the face its slack sits on rather than a set of rows, and
that is not implemented yet — so the builder refuses rather than returning a
plausible wrong answer.

Python callers were never exposed to this: `pounce.qp.QpSensitivity` solves
internally with the QP interior-point solver and accepts no `cones=`.

For the NLP arm's much larger sensitivity surface — fix-relax and path modes,
the directional decision at a kink, the corrector, activity classification,
and the covariance/identifiability statistics — see
[Sensitivity Analysis](sensitivity.md). The two arms are not at parity today.

## Presolve (PaPILO-inspired)

Before the interior-point solve, POUNCE can apply a **transaction-stack
presolve** with full primal **and dual** postsolve, modeled on
[PaPILO](https://github.com/scipopt/papilo). The catalog:

- empty / **duplicate / parallel** (scalar-multiple) rows,
- fixed-variable elimination (singleton equalities),
- free columns and free-column singletons,
- activity-based redundancy and infeasibility detection,
- **forcing constraints** (a row at its activity extreme pins its variables),
- **dominated columns** (sign-definite columns optimal at a bound),
- **bound tightening** (domain propagation), with the active-bound
  multiplier re-attributed to its source row in postsolve,

iterated to a **fixpoint** so reductions cascade. Each reduction carries
the data to reverse itself, and the postsolve reconstructs a valid KKT
point of the *original* problem — the dual recovery is the contract, and is
verified by KKT-residual tests. A cone-aware variant (`presolve_conic`)
gates the `≤`-row reductions off second-order-cone blocks (which are
coupled) and recovers the reduced cone partition.

The iteration also carries a **layer cap**, and on a model with a long
bound-propagation chain — commonly, on roughly half the LP corpus — the cap
is what stops it rather than the fixpoint. That distinction is visible:
presolve reports which of the two happened and the CLI says so on its
summary line (see
[LP / QP Solver Routing](lp-qp-routing.md#when-the-reduction-is-truncated)).
A truncated reduction is still correct — every reduction it did apply is a
sound transform with its own dual recovery — and measured across the LP and
QP suites the truncation costs only box tightness, never a structural
reduction.

Presolve is applied automatically on the CLI LP/QP route; it lives in
`pounce-convex::presolve` for Rust callers. See
[LP / QP Solver Routing](lp-qp-routing.md).

## Differentiable convex layers (JAX)

`pounce.jax` exposes the solve as a differentiable JAX op via the
implicit-function theorem on the KKT system at the optimum (Amos & Kolter,
*OptNet*, 2017). The forward calls the solver; the backward is a single
linear solve through the same KKT matrix.

```python
import jax, jax.numpy as jnp
from pounce.jax import solve_qp, solve_socp, QpLayer

# x*(c) for a parametric QP, differentiable w.r.t. all of P, c, G, h, A, b.
def loss(c):
    x = solve_qp(P=P, c=c, G=G, h=h)
    return jnp.sum((x - target) ** 2)

grad_c = jax.grad(loss)(c0)        # exact gradient via implicit diff
J = jax.jacrev(lambda c: solve_qp(P=P, c=c, G=G, h=h))(c0)
```

- Gradients are provided w.r.t. **every** parameter that enters through the
  optimum: `c`, `b`, `h`, and the matrices `P`, `G`, `A` (the full OptNet
  matrix derivatives; `∇P` is the symmetric gradient).
- `solve_socp` differentiates SOCPs too — the complementarity row uses the
  cones' **arrow operators** in place of the orthant's diagonal.
- `QpLayer` captures a fixed `P`/`G`/`A` structure for use inside a larger
  JAX model, with `jax.grad` / `jacrev` / `vmap` and a parallel `.batch`.
- A warm start may be passed through (non-differentiated — it cannot change
  the solution or its gradients, only the iteration count).

All gradients are validated against finite differences in the test suite.
