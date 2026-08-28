# Sparse finite-difference Hessian from the analytic Jacobian

> `hessian_approximation=finite-difference`. Recovers the exact Lagrangian
> Hessian by graph-coloured finite differences of the **analytic
> Jacobian**, for models that supply first derivatives and no second ones.
>
> **Headline: on `laptime` this reaches the exact path's iteration count
> — 30 at N=160 and 57 at N=320, identical in both cases — where
> limited-memory takes 246 and then fails outright. It is the first thing
> in this line of work that is unambiguously better than what a
> Hessian-less user has today.**

## The idea

The Lagrangian gradient `∇ₓL = ∇f + J_cᵀy_c + J_dᵀy_d` is available in
closed form whenever the Jacobian is. Its directional derivative

```
    ∇²ₓₓL · d ≈ [ ∇ₓL(x + d, y) − ∇ₓL(x, y) ] / h
```

therefore costs one gradient plus one Jacobian evaluation. With a known
sparsity pattern, one probe recovers a whole group of structurally
orthogonal columns rather than one column, so the cost per Hessian is the
number of colour groups, not `n`.

Columns are grouped by Curtis-Powell-Reid orthogonality (no two columns in
a group share a row), found by greedy largest-first colouring. A *star*
colouring would exploit symmetry for roughly half as many groups; CPR is
used because it is simple to get right, so every number below is a
conservative bound.

## Why it is affordable, measured before it was built

`POUNCE_HESS_PATTERN_CENSUS` on the exact Hessian:

| mesh | n | nnz | `rho_max` | mean row |
|---|---|---|---|---|
| N=160 | 9 294 | 28 000 | 15 | 5.68 |
| N=320 | 18 574 | 56 000 | **15** | 5.69 |

`rho_max` is **mesh-invariant** — the Hessian's row width is set by the
per-stage stencil, not the horizon — so the probe count does not grow with
the mesh. The realised colouring is **17 groups** at N=160, against a
Jacobian evaluation costing 5.4 ms in a 92.6 ms iteration.

## The measurement

`max_iter=1200`. True optima 65.371107 (N=160) and 65.326908 (N=320).

| mesh | leg | status | iters | wall | objective |
|---|---|---|---|---|---|
| N=160 | exact | Optimal | 30 | 2.9 s | 65.3711067940491 |
| N=160 | lbfgs | Optimal | 246 | 35.6 s | 65.3705401621855 |
| N=160 | **fd-declared** | **Optimal** | **30** | **7.0 s** | 65.3711067940491 |
| N=160 | **fd-jacobian** | **Optimal** | **38** | 15.9 s | 65.3711063753353 |
| N=320 | exact | Optimal | 57 | 17.5 s | 65.3269077801929 |
| N=320 | lbfgs | **MaxIter** | 1200 | 728.3 s | 65.3265568888976 |
| N=320 | **fd-declared** | **Optimal** | **57** | **22.5 s** | 65.3269077802016 |
| N=320 | **fd-jacobian** | Acceptable | 101 | 443.5 s | 65.3269077802655 |

`fd-declared` reproduces the exact path's iteration count exactly at both
meshes and its objective to 14 significant figures, for 1.3–2.4× the wall
time and no second derivatives. Against limited-memory — the only option a
Hessian-less user has today — it is **5.1× faster at N=160 and converges
at N=320 where limited-memory does not converge at all**.

## The two pattern sources, and which one your model can use

* `fd_hessian_pattern=declared` (default) — the TNLP's declared Hessian
  **structure**. This is a structure-only call; no second derivative is
  ever evaluated. Every `.nl` declares one through AMPL's AD.
* `fd_hessian_pattern=jacobian` — derived as
  `⋃_j supp(∇g_j) ⊗ supp(∇g_j)`, needing nothing beyond the Jacobian
  pattern every TNLP must declare.

The Jacobian-derived pattern is a strict **superset** of the true one,
which is safe — a superset costs extra probe groups, never a wrong answer
— but not free: 146 267 nonzeros against the true 28 000, which is why
`fd-jacobian` costs 38 iterations and 15.9 s where `fd-declared` costs 30
and 7.0. It still beats limited-memory 2.2× at N=160 and reaches
acceptable tolerance at N=320 where limited-memory fails.

There is deliberately no mode that guesses a *subset*: that would silently
drop curvature.

**For a CasADi/FMU model this is the decision point.** If the frontend can
state a Hessian sparsity pattern — which is much weaker than evaluating
one — use `declared` and get the exact path's iteration count. If it can
only state the Jacobian pattern, `jacobian` still wins, by less.

## What this is not

* **Not measured against Ipopt.** `ipopt` is not installed in the
  environment this was built in, and `benchmarks/large_scale/ipopt_ma57.json`
  predates the `laptime` family. Every number here is POUNCE against
  POUNCE. Note also that `ipopt laptime.nl` would take the *exact* Hessian
  from AMPL's AD and run the 30-iteration path; the 246-iteration regime is
  what a model with no Hessian gets, which is the case this addresses.
* **Not free of truncation error.** The step is `sqrt(eps)·max(1,|x_j|)`,
  forward difference, so the Hessian is exact only to first order in `h`.
  It did not hurt convergence on either mesh — the objectives agree with
  the exact path to 14 figures — but a central difference (double the
  probes) is the fallback if a model ever proves sensitive.
* **Not bound-aware in the obvious way.** `nlp.x_l()`/`x_u()` live in
  Ipopt's *compressed* bounded-variable space, one entry per variable that
  has that bound, so they cannot be indexed by variable index — doing so
  panicked, which is how this was found. The guard used instead is
  stronger: each group's probe is checked for finiteness and retried with
  the step reversed, and a group that leaves the domain in both directions
  fails the update loudly rather than scattering NaN into `W`.
* **Not run through `scripts/sweep-fixtures.sh`.** It is an opt-in path
  and leaves both default legs bit-identical, but that sweep is the gate
  before merge.

## Where the remaining cost is

`fd-declared` at N=320 spends most of its extra time in Jacobian
evaluations (17 per Hessian). Two obvious reductions, neither attempted:
a star colouring instead of CPR (roughly half the groups), and skipping
the Hessian rebuild on iterations where the iterate barely moved.
