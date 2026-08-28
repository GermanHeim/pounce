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
* **Run through `scripts/sweep-fixtures.sh`: empty diff.** The gate is
  met, not argued around. Baseline `origin/main` (`a5e0a83`) against this
  branch (`ec78423`), 156 fixture-legs each, `exact` and `lbfgs`: not one
  line moves — status, objective, iteration count and engine all
  unchanged. The corpus is live rather than vacuous (5 `cvx-qcqp`, 36
  `cvx-qp`, 37 `nlp` per leg; zero `NO_JSON`, zero `unknown`), so the
  convex arms are covered as `CLAUDE.md` requires.

  The control that makes the empty diff mean something: the two binaries
  differ (`c14ed24a` vs `70773820`), and the baseline *rejects*
  `hessian_approximation=finite-difference` while the branch accepts it.
  So the feature is present and reachable in the swept binary and still
  moves nothing on the default legs — which is the claim, rather than
  "we changed nothing".

  What the sweep does **not** cover: `finite-difference` itself. Both legs
  run `exact` and `limited-memory` only, so the new path is never swept.
  Its evidence is the `laptime` measurements above, and those do not
  exercise restoration (`restoration_calls: 0` at N=80).

## Where the remaining cost is

`fd-declared` at N=320 spends most of its extra time in Jacobian
evaluations (17 per Hessian). Two obvious reductions, neither attempted:
a star colouring instead of CPR (roughly half the groups), and skipping
the Hessian rebuild on iterations where the iterate barely moved.

---

## The objective clique sets a hard floor on `rho_max`, and why it stays

The Jacobian-derived pattern has to stand in for `∇²f` somehow, and with
no second-derivative information from the model the only safe stand-in is
a **dense clique** over every variable the objective is nonlinear in.
Adding it was a correctness fix — without it the pattern was a *subset*
of the truth, silently dropping curvature (found in review by
@srikanth-gm) — and its cost is not small. Measured on `laptime` at
N=160 (`benchmarks/large_scale`, 9294 variables), with
`POUNCE_DROP_HESSIAN=1`:

| pattern available | nnz | groups | `rho_max` |
|---|---|---|---|
| declared (`eval_h` structure) | 34 094 | **17** | 15 |
| none — Jacobian + objective clique | 143 854 | **341** | 338 |

**The clique is the floor, not the constraints.** `laptime`'s objective is
a control-rate regulariser,

```
    laptime + W · Σ_k Σ_c (U[k+1,c] − U[k,c])²
```

over `n_int × n_controls = 160 × 2 = 320` controls. Its true `∇²f` is
**banded with a row width of 3** — each `U[k,c]` couples only to
`U[k±1,c]` — and the clique replaces that with width 320. Curtis-Powell-Reid
needs at least `rho_max` groups, so a 320-wide clique alone puts a floor
of 320 on the probe count: measured `rho_max` is 338 and measured groups
341. The clique is 51 360 of the 143 854 pattern entries (36%) but it is
100% of the reason the colouring cannot get narrow.

**So the cost is not tunable — it is the price of not knowing `∇²f`.**
Nothing in the corpus, the colouring or the mask reduces it, because any
reduction is a *subset* of the safe pattern and the mode deliberately has
no mode that guesses one.

**What removes it is the model stating any second-derivative structure at
all**, at which point `fd_hessian_pattern=declared` uses the true `∇²L`
and the clique never runs — the 17-group row above. That is available to
any model that can answer the `eval_h` *structure* call without
evaluating a value, which is every `.nl` file (AMPL's AD declares one)
and every CasADi model whose Lagrangian Hessian is symbolically
constructible.

Two things were tried and rejected while chasing this:

- **Refining the clique from the objective's own Hessian sparsity**
  (`∇²f ∪ ⋃ⱼ supp(∇gⱼ)⊗supp(∇gⱼ)`, passed from the CasADi plugin). Sound
  — it is a valid superset, and on paper it replaces the 320-wide clique
  with the true band. It was built and then removed because **its benefit
  is unreachable on that surface**: the only way to lose the symbolic
  `∇²L` in CasADi is an opaque `Callback`, and CasADi treats an opaque
  callback's Jacobian as *dense* (measured: `jac(g,x)` came back
  3480/3480 nonzeros on a model whose declared Jacobian was banded-3). A
  dense `J` makes `JᵀJ` dense, so the constraint half of the pattern is
  dense anyway and the objective refinement changes nothing. There is no
  model on that surface where it pays, so it is not worth its complexity.
- **Masking the clique by the nonlinear-variable set.** A no-op *when the
  model states its objective linearity*: the clique is then built from
  `get_objective_variables_linearity`, the objective's own nonlinear set,
  which is already a subset of the global one. When the model states
  nothing the mask is the next level down and is used — that is the
  `N`-then-all-`n` ladder, and `FdStats::objective_clique_widened` says
  which rung a run took.

The actionable consequence is not a code change but a reachability one:
the mitigation has to be available and visible from every frontend. Both
were broken and are now fixed — the CasADi plugin could not reach
`declared` at all (it failed to build `nlp_hess_l` and gave up), and
which pattern a run ended up with was only observable through
`POUNCE_FD_HESSIAN_DEBUG`. See `GetPounceFdHessianStats` and
`stats()["fd_hessian"]`, which report the source actually **used**, so a
silent fallback to the 341-group pattern is visible rather than inferred.
`objective_clique_widened` in the same report answers the follow-up — a
high `groups` caused by a widened clique is a different problem from one
caused by a genuinely dense objective, and the two are indistinguishable
from the probe count alone.

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

---

## Mesh continuation on `laptime`: primal-only warm starting makes it worse

A trajectory refinement path is the textbook case for warm starting — each
solve should be close to the last. It is not POUNCE's to exploit: the solver
sees a flat vector and cannot know which entry is which stage's state, so
mapping a coarse solution onto a fine mesh is transcription knowledge and
belongs to the frontend. `benchmarks/large_scale/mesh_continuation.py` is
that frontend for this family, mapping by arc length through the `.col`
names (`X[k,st]`, `Xc[k,j,st]`, `U[k,c]` on a uniform mesh).

`hessian_approximation=finite-difference`, every entry transferred:

| step | cold | warm (interpolated) | |
|---|---|---|---|
| N=80 → 160 | 30 it / 3.6 s | **93 it / 15.3 s** | **3.1× worse** |
| N=160 → 320 | 57 it / 18.5 s | **74 it / 24.7 s** | **1.3× worse** |

### The interpolation is not the problem

Evaluated at the N=160 mesh:

| start point | objective | max constraint violation | min relative slack | vars at bound |
|---|---|---|---|---|
| `.nl` cold | 88.889 | 6.397 | 0.0 | 641 |
| interpolated from N=80 | 65.458 | 0.814 | 7.5e-56 | 673 |

The transferred point is far better by every optimization measure — its
objective is already within 1e-1 of the converged 65.371107 and its
constraint violation is 8× smaller — and it still costs 3.1× the iterations.

The mechanism is **centrality**, not quality. A converged coarse solution
sits on its active set, with slacks down at `7.5e-56`. A barrier method
wants a well-centred interior point, and "already at the boundary" is the
worst case: the early iterations are spent restoring centrality, which costs
more than the better point saves. This is the documented
warm-start-hurts regime — `benchmarks/warmstart/README.md` cites *Not All
Warm Starts Help* (arXiv:2606.08984) for exactly this, which is why that
suite reports regressions as a first-class column rather than averaging them
away.

### What this does and does not rule out

Only the **primal** point was transferred. That is the warm-start suite's
`values-ipm` arm — no multipliers, no `mu` — and is the weakest arm it
measures, chosen here because it needs nothing but `x`. The `warm-ipm` arm,
which carries the duals and the barrier parameter so the solver resumes from
a *consistent* primal-dual-barrier state, scored 1.96× on `nmpc_vanderpol`
(the closest-shaped family) and 4.42×/9.12× on the other two. It is
untested on `laptime` and is the thing that could still work; it needs the
constraint multipliers mapped through the `.row` names as well, plus
`warm_start_init_point=yes` and `warm_start_target_mu`.

So: mesh continuation is not free, the obvious cheap version of it is a
regression on this family, and the reason is a property of interior-point
methods rather than of the transfer.
