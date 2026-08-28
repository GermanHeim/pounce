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

---

## Star colouring and Hessian reuse: measured, and one of them shipped off

Two follow-ups to the above.

### Hessian reuse — small, real, kept

`fd_hessian_reuse_tol` skips the rebuild when neither the primal iterate
nor the multipliers have moved relatively more than the tolerance. **Both**
are tested: `∇²L = ∇²f + Σ yⱼ ∇²cⱼ` depends on the multipliers, so testing
`x` alone would hand back a stale Hessian through the whole endgame of an
interior-point solve, which is short steps with moving duals.

`laptime` N=160, Jacobian pattern:

| `fd_hessian_reuse_tol` | status | iters | wall |
|---|---|---|---|
| 0 (off) | Optimal | 38 | 13.5 s |
| 1e-8 | Optimal | 38 | 13.0 s |
| 1e-6 | Optimal | 38 | 12.5 s |

~7%, no iteration cost, objective unchanged to 15 figures. Off by default;
worth turning on.

### Star colouring — correct, cheaper, and the wrong choice anyway

A star colouring lets an entry be read from **either** endpoint's probe, so
it needs fewer groups: **76 → 42** on the Jacobian-derived pattern, 17 → 16
on the declared one. Its recovery is algebraically exact, verified in
`overlapping_cliques_are_validated_not_assumed` by recovering a known matrix
through it.

It still loses:

| pattern / colouring | groups | cols per group | result |
|---|---|---|---|
| declared / cpr | 17 | 546 | Optimal, 30 it, 65.3711067940 |
| declared / star | 16 | 580 | Optimal, 30 it, 65.3711067940 |
| jacobian / cpr | 76 | 122 | Optimal, 38 it, 65.3711063753 |
| jacobian / star | 42 | 221 | **Acceptable, 404 it, 65.3683344570** |

**Group size is not the cause.** `declared/star` packs the largest groups of
the four — 580 columns per probe against `jacobian/cpr`'s 122 — and
converges in 30 iterations to the exact objective. That was the first
hypothesis and the measurement refutes it.

The cause is the **finite-difference remainder**. Direct-recovery theory
assumes exact Hessian-vector products. A forward difference also carries

```
    ½ Σ_{m,p ∈ g} T_imp h_m h_p
```

into row `i`, where `T` is the third derivative. `T_imp ≠ 0` requires `i`,
`m` and `p` in a common constraint's support, hence `H_im ≠ 0` **and**
`H_ip ≠ 0`. CPR's distance-2 property forbids two such columns from sharing
a group, so those cross terms vanish *structurally*. A star colouring only
guarantees the single-neighbour property for the pair being recovered, so
they survive — and they matter precisely where the pattern is dense
(`rho_max` 59 on the Jacobian pattern against 15 on the declared one).

So the ordering of these two is the opposite of what it looks like from the
group counts, and CPR is the default.

### The check that should have caught it earlier

The recovery's soundness condition was a `debug_assert`, which is compiled
out in release. The invalid case therefore produced a wrong Hessian with no
diagnostic — it surfaced only as 404 iterations and a wrong objective, which
is the "silently wrong while reporting success" failure this whole module is
most exposed to. It is now an unconditional validation with an automatic
fallback to CPR, and `FdStats::coloring_fell_back` reports when that fires.

On `laptime` the star colouring **passes** that validation and is still
numerically wrong, which is the point worth carrying forward: the predicate
is about algebraic recoverability and cannot see the differencing error.


---

## Measured on a model that genuinely has no Hessian

Every number above this section was produced by setting an option on a
model that *does* carry a Hessian — `laptime.nl` gets one from AMPL's AD —
so it simulated the constraint rather than reproducing it. That caveat is
now removed.

`POUNCE_DROP_HESSIAN=1` installs `pounce-cli`'s `NoHessianTnlp`, a wrapper
that reports `nnz_h_lag = 0` and declines `eval_h`. That is exactly what
`pounce-py` presents for a Python problem object with no `hessian` method,
which is the real shape of an FMU- or CasADi-backed model. Verified:
`Number of nonzeros in Lagrangian Hessian` reads 28 000 without it and
**0** with it.

| mesh | leg | status | iters | wall | objective |
|---|---|---|---|---|---|
| N=160 | *ref:* `exact`, Hessian available | Optimal | 30 | 2.8 s | 65.3711067940 |
| N=160 | no-Hessian, `limited-memory` | Acceptable | 207 | 53.7 s | 65.3704926894 |
| N=160 | no-Hessian, `finite-difference` | **Optimal** | **38** | **12.6 s** | 65.3711063753 |
| N=160 | no-Hessian, `finite-difference` + reuse | Optimal | 38 | 11.7 s | 65.3711063753 |
| N=320 | *ref:* `exact`, Hessian available | Optimal | 57 | 14.5 s | 65.3269077802 |
| N=320 | no-Hessian, `limited-memory` | **MaxIter** | 1210 | 706 s | **65.3951930783** |
| N=320 | no-Hessian, `finite-difference` | Acceptable | **106** | 430 s | 65.3269077802 |

**4.6× wall and 5.4× iterations at N=160.** At N=320 limited-memory spends
1210 iterations to reach an objective of 65.395 against a true optimum of
65.326908 — not merely unconverged but visibly wrong — while the
finite-difference path reaches twelve correct digits.

### Why building the wrapper was worth it rather than reasoning about it

The simulated and genuine conditions do not agree, so the earlier numbers
were not a stand-in for these:

* limited-memory moved from 246 iterations / `Optimal` (simulated) to 207 /
  `Solved To Acceptable Level` (genuine) at N=160, and at N=320 from
  "unconverged" to "unconverged at a visibly wrong objective".
* The claim that the default `fd_hessian_pattern=declared` falls back to
  the Jacobian derivation when a model declares no Hessian was, until now,
  a reading of the match arm. It is now observed: the run reports the
  Jacobian-derived census (146 267 nonzeros, 76 groups) with no option set.

A model with no Hessian differs in more than which updater runs —
`nnz_h_lag` is 0, `h_space` is `None`, and `uninitialized_h()` returns an
empty pattern, so the augmented-system solver sees `W`'s nonzero count
change from 0 to the assembled pattern rather than staying put. None of
that was exercised by forcing the option.

The wrapper is an environment variable and not a registered option
deliberately: no real solve should ever want to discard information the
model was willing to provide.
