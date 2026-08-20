# NLP and Linear-System Scaling

Optimization problems whose objective, constraints, or KKT system span
many orders of magnitude often converge poorly — or not at all — without
some form of rescaling. pounce inherits two independent scaling layers
from Ipopt and adds a third option at the linear-system level
(see [issue #61](https://github.com/jkitchin/pounce/issues/61)).

The two layers are conceptually separate:

| Layer | Option | What it touches |
|---|---|---|
| **NLP scaling** | `nlp_scaling_method` | The objective `f` and each constraint row `c_i`, before the IPM sees them. Changes algorithmic behavior (filter, `tol`, μ). |
| **Linear-system scaling** | `linear_system_scaling` | Symmetric scaling of the KKT augmented system `D K D` for the factorization. Purely numerical — the IPM sees the same iterates. |

You can configure them independently. Defaults match upstream Ipopt:
`nlp_scaling_method = gradient-based`, `linear_system_scaling = none`.

## NLP-level scaling

| Option | Default | Effect |
|---|---|---|
| `nlp_scaling_method` | `gradient-based` | `none` / `gradient-based` / `user-scaling` / `curvature-based`. |
| `nlp_scaling_max_gradient` | `100.0` | Cutoff above which gradient-based scaling applies. Per-row scale = `min(1, max_gradient / ‖∇c_i‖_∞)`. |
| `nlp_scaling_min_value` | `1e-8` | Floor on computed scale factors — prevents inverting near-zero gradients. |
| `nlp_scaling_obj_target_gradient` | `0.0` | When `> 0`, *pins* the scaled objective gradient ∞-norm to this value. Overrides the `max_gradient` cutoff. |
| `nlp_scaling_constr_target_gradient` | `0.0` | Same as above, per constraint row. |
| `obj_scaling_factor` | `1.0` | Constant multiplier on the objective, applied after the automatic factor. |

### `gradient-based` (default)

Evaluates `∇f` and `∇c_i` *once* at the starting point `x_0` and
chooses per-row scales that pull each gradient ∞-norm into a
reasonable band. Single-shot is mandatory — recomputing per iteration
would invalidate the filter's history (Wächter, 2013).

The clamp at 1.0 means scaling never *amplifies* a small row; it only
damps large ones.

Two consequences of the single shot are worth knowing before you rely
on it. The cutoff is a **per-block gate**: unless some row of a block
(the equalities, or the inequalities) exceeds
`nlp_scaling_max_gradient`, no scale vector is produced for that block
at all. And the sample is only as informative as the point it is taken
at — which is the next section.

### Quadratic rows the sampler cannot see

A row written `½·x'Qx ≤ b` about the origin has `∇g(0) = 0`. Started
from `x0 = 0` — the default for a model with free variables and no
initial guess — the sample reads nothing, and the row is assigned
factor **1.0** however far `Q` and `b` disagree in magnitude. This is
not a cutoff set too high: `100/0` and `1e-6/0` both clamp to 1.0, so
no value of `nlp_scaling_max_gradient` reaches the row.

It matters because the row's slack `s = −g(x)` then inherits the
right-hand side's scale, and so does the `−s/λ` diagonal of the KKT
system. On a QCQP whose right-hand sides run four orders of magnitude
above its curvature, supplying the row scales by hand is worth several
times the iteration count.

`pounce check-x0` reports both halves — the factors the sampler will
pick, and the coefficient magnitudes it cannot see:

```sh
pounce check-x0 model.nl
```

```text
  automatic scaling at x0 (nlp_scaling_method=gradient-based, nlp_scaling_max_gradient=100):
    objective: ||grad f|| 9.983e1 -> factor 1.000e0  (below the cutoff: unscaled)
    inequalities: 5007 row(s), no row above the cutoff -> the whole block is unscaled
                  7 row(s) have an all-zero Jacobian at x0 (the sample cannot scale them)
    quadratic rows: 7 recognized; 7 left at factor 1.0, 7 with a zero Jacobian at x0
                    worst |b|/||Q||_inf mismatch 5.588e1
```

`||Q||_inf` is the largest absolute row sum of the row's Hessian —
Gershgorin's bound on its largest eigenvalue, so the reported mismatch
is a *lower* estimate of the real one. `--scaling-max-gradient` previews
a different cutoff; `--json` puts the same numbers under a `scaling`
key.

`curvature-based` below computes exactly that correction for you;
`user-scaling` lets you supply it by hand. See
`dev-notes/quadratic-structure-exploitation.md` §8 for the derivation
and the measurements.

### `curvature-based`

Derives the scaling from the model's **quadratic coefficients** instead of
from a derivative sample, so a row's factor does not depend on where the
modeller happened to start. Two stages, both from
`dev-notes/quadratic-structure-exploitation.md` §8:

1. one **joint** variable scaling `D`, Ruiz-equilibrated across the whole
   pencil `Q_0 + Σ λ_i Q_i` — via the λ-independent magnitude envelope of
   that family, so it balances every constraint at once rather than each
   `Q_i` against its own column scaling;
2. a per-row `e_i = 1 / max(‖D Q_i D‖_∞, ‖D a_i‖_∞, |b_i|)`.

The objective is deliberately left unscaled: the Ruiz pass already anchors
the Hessian block against the constraint blocks, and shrinking it below the
constraint scale costs strong convexity.

```sh
pounce model.nl model.sol nlp_scaling_method=curvature-based
```

**It requires every row and the objective to be degree ≤ 2** — the envelope
above exists only because each `Q_i` is a constant matrix — and it refuses
with a message rather than silently solving unscaled. A model with a
genuine nonlinearity is not one this method is defined for.

What it buys is best stated as invariance rather than speed. Given a QCQP
and the same QCQP with an exact change of variables `x_j → x_j / c_j`
spanning nine orders of magnitude:

| column span | `gradient-based` | `curvature-based` |
|---|---|---|
| 1 | 75 it, `2.4779690299303e4` | 16 it, `2.4779690299303e4` |
| 1e3 | 92 it, `2.4779690302194e4` | 16 it, `2.4779690299303e4` |
| 1e6 | 154 it, `2.4779690388034e4` | 16 it, `2.4779690299303e4` |
| 1e9 | `Maximum_Iterations_Exceeded` | 16 it, `2.4779690299303e4` |

Two caveats, both measured:

* **It is off by default**, and on a model whose rows the default *can*
  see it is usually a wash — 37 of the 41 models in POUNCE's own fixture
  corpus that it accepts are unchanged in status, iterations and objective.
* **On a nonconvex model, changing the scaling changes which local minimum
  you reach.** `pooling_rt2stp` goes from `-3273.955` in 181 iterations to
  `-4391.826` in 1083 — a better point (it is the published global optimum
  of that instance) for six times the work. Neither direction is
  guaranteed; treat a nonconvex re-scale as a different search.

### `user-scaling`

The TNLP is asked for `obj_scaling`, a per-variable `x_scaling`, and a
per-constraint `g_scaling` via the `get_scaling_parameters` callback.
Use this when you know the natural units of your problem (e.g. mass in
kg vs. distance in mm) and can supply better scales than the
gradient-based heuristic.

If the TNLP's `get_scaling_parameters` returns false (the default),
pounce falls back to no automatic scaling.

> **Per-variable factors are a change of variables.** `OrigIpoptNlp`
> models `obj_scaling` and per-constraint `g_scaling` only (the design
> in [issue #61](https://github.com/jkitchin/pounce/issues/61)), so
> `x_scaling` is applied one level below the algorithm instead: a
> wrapper substitutes `x̃ = d ⊙ x`, the IPM works in the scaled
> coordinates, and everything reported back — the solution, the duals,
> the bound multipliers, and every
> [sensitivity](sensitivity.md) accessor — is in your own units
> ([issue #486](https://github.com/jkitchin/pounce/issues/486)). No
> clone of the model is made and no `propagate_solution` step is
> needed, which is what distinguishes this from Pyomo's
> `core.scale_model`.
>
> Factors must be **finite and strictly positive**. Zero and negative
> are refused rather than applied: a negative factor reverses a
> variable's direction and swaps its bounds. A factor that would push
> a finite bound past `nlp_lower_bound_inf` / `nlp_upper_bound_inf` —
> turning a bounded variable into a free one — is refused too, naming
> the threshold it crossed. Absent bounds stay absent: the `±1e19`
> sentinel is an ordinary finite number, so it is passed through
> unscaled rather than multiplied into range.
>
> One user-visible consequence: `tol` keeps comparing **scaled**
> quantities, matching upstream Ipopt, so the same `tol` stops at a
> different point than it would on the unscaled model.

#### Setting user scaling

* **From an `.nl` file (AMPL, Pyomo, any NL-writing frontend)** — attach
  a `scaling_factor` suffix to the objective, to constraints, or to
  both, and pass `nlp_scaling_method=user-scaling`. This is the same
  channel Ipopt reads through ASL. In Pyomo:

  ```python
  m.scaling_factor = Suffix(direction=Suffix.EXPORT)
  m.scaling_factor[m.obj] = 1e-3
  m.scaling_factor[m.mass_balance] = 1e2
  SolverFactory('pounce').solve(m, options={'nlp_scaling_method': 'user-scaling'})
  ```

  Components the suffix does not list are unscaled, as are components
  listed with a factor of `0` (AMPL's suffix default). With no
  `scaling_factor` suffix at all the option falls back to no scaling.
  See [Pyomo](pyomo.md) for the pyomo-pounce specifics.
* **From C** — call `SetIpoptProblemScaling(problem, obj, x_scaling,
  g_scaling)` then `AddIpoptStrOption("nlp_scaling_method",
  "user-scaling")`. See `crates/pounce-cinterface/include/pounce.h`.
* **From Rust** — implement
  [`TNLP::get_scaling_parameters`](https://github.com/jkitchin/pounce/blob/main/crates/pounce-nlp/src/tnlp.rs)
  on your problem type.
* **From Python** — `pounce.Problem.set_problem_scaling(obj_scaling,
  x_scaling=..., g_scaling=...)`, followed by
  `add_option("nlp_scaling_method", "user-scaling")`. Walked
  through end-to-end in
  [`python/notebooks/07_scaling.ipynb`](https://github.com/jkitchin/pounce/blob/main/python/notebooks/07_scaling.ipynb).

> **Specialized solvers.** A model that classifies as an LP, convex QP,
> or SOCP normally routes to `pounce-convex`, which equilibrates
> internally and never reads the TNLP scaling callback. When
> `nlp_scaling_method=user-scaling` is set and the `.nl` carries
> `scaling_factor` suffixes, `solver_selection=auto` declines that fast
> path and uses the general NLP interior-point solver so the scaling is
> honored; an explicit `solver_selection` is respected and warns.

### Target-gradient overrides

`nlp_scaling_obj_target_gradient` and
`nlp_scaling_constr_target_gradient` are subtle. When set to a
positive value, they *override* the `max_gradient` cutoff and the 1.0
clamp: the scaling is computed unconditionally as
`target / max_gradient_norm`, so the scaled gradient ∞-norm becomes
exactly the target. Useful when you have a specific numeric range you
want the IPM to see.

The default `0.0` means "use the cutoff path" — i.e. only scale rows
that are above `nlp_scaling_max_gradient`.

## Linear-system-level scaling

| Option | Default | Effect |
|---|---|---|
| `linear_system_scaling` | `none` | `none` / `ruiz` / `slack-based`. `mc19` is accepted by the option registry but not yet implemented and falls back to `none`. |
| `linear_scaling_on_demand` | `yes` | Defer scaling computation until a linear solve is poor; reduces overhead for well-conditioned KKT systems. |

The KKT augmented system is symmetric; all linear-system scalers in
pounce use the symmetric form `D K D` (single diagonal) to preserve
that structure for the downstream factorization (MA57, MUMPS,
FERAL/SSIDS).

* **`none`** — first-class choice. The inner linear solver (MA57,
  MUMPS, FERAL) often does its own scaling under some configurations;
  stacking pounce-level scaling on top can *hurt*. Default. Use
  `ma57_automatic_scaling=yes` to get MA57's internal scaling instead.
* **`ruiz`** — iterative symmetric ∞-norm equilibration (Ruiz,
  CERFACS TR/PA/01/14). Pure Rust, no Fortran dependency. Converges
  geometrically; capped at 10 iterations. A good starting point when
  MA57's internal scaling is off.
* **`slack-based`** — port of Ipopt's `IpSlackBasedTSymScalingMethod`.
  Scales the `s` block by `min(Pd_L·slack_s_L + Pd_U·slack_s_U, 1)` and
  leaves the `x`, `y_c` and `y_d` blocks at 1, so the rows whose barrier
  terms are blowing up as a slack approaches its bound are damped and
  nothing else is touched. This is the one scaler whose factors depend
  on the **iterate** rather than on the matrix, so they are recomputed
  every iteration.

  Ipopt's recommended configuration for large collocation NLPs uses it.
  It was accepted but inert before #677 — on any earlier release,
  setting it silently did nothing.
* **`mc19`** *(not yet implemented)* — intended HSL MC19 row/column
  scaling (Curtis-Reid 1972; minimizes Σ log²|a_ij|). Accepted by the
  registry but currently logs a warning and falls back to `none`.

Scaling choices only differ when scaling actually runs. With the default
`linear_scaling_on_demand=yes`, factors are computed only once a solve
looks troubled, so on a clean problem every choice behaves identically.
Set `linear_scaling_on_demand=no` to compare them. On `cresc4`, forced
on: `none` 81 iterations, `slack-based` 74, `ruiz` 61.

### Worked example — `nql180`

`nql180` is one of the Mittelmann NLP benchmarks where both default
pounce and default Ipopt fail to clear the strict `tol` gate (see
[issue #25](https://github.com/jkitchin/pounce/issues/25)). Forcing
Ruiz symmetric equilibration on the augmented KKT system is enough to
push pounce all the way to "Optimal Solution Found":

```
pounce nql180.nl presolve=yes linear_system_scaling=ruiz \
       linear_scaling_on_demand=no
```

|                          | default | + Ruiz (forced)        |
|---                       |---      |---                     |
| Exit status              | Solved To Acceptable Level | **Optimal Solution Found** |
| Iterations               | 41      | 50                     |
| Primal infeasibility     | 4.0e-11 | **1.2e-15**            |
| Dual infeasibility       | 1.0e-5  | 3.1e-4                 |
| Complementarity          | 1.2e-9  | 9.9e-10                |
| Overall NLP error        | 2.4e-7  | **9.9e-10**            |

The four-orders-of-magnitude primal-feasibility improvement and ~3
orders on the overall NLP error are the textbook Ruiz benefit:
symmetric ∞-norm equilibration lowers the condition number of the KKT
matrix enough that the back-solve residuals drop the extra fractional
digits needed to clear `tol`. The extra nine iterations are well spent
— the 50-iter Ruiz solution is mathematically of strictly higher
quality than the 41-iter unscaled "acceptable" solution.

`linear_scaling_on_demand=no` forces always-on Ruiz; the default
(`yes`) defers scaling computation until the linear solver flags an
iterate as poorly scaled, which is the right behavior for problems
that don't need it (most of the Mittelmann set, where the iter count
is unchanged with or without Ruiz).

## Reporting

All scaling effects are undone before the solve report (final
objective, multipliers, dual residuals, KKT termination metric) is
handed back to the user. You always see quantities in the natural
units of your TNLP.

Internally, the IPM operates in scaled space: stopping criteria
(`tol`, `acceptable_tol`) compare scaled values, the barrier parameter
μ is in scaled units, and the filter's history is built from scaled
function values.

## When to override the defaults

Reach for non-default scaling when:

* The constraint Jacobian has entries spanning many orders of magnitude
  (chemistry, power-flow, mixed-unit mechanics). Try `mc19` or `ruiz`
  at the linear-system level, after disabling MA57's internal scaling.
* The IPM stalls with small step sizes but no clear infeasibility.
  Worth turning `nlp_scaling_method=none` to see whether the default
  gradient scaling is doing the wrong thing; then re-enable with
  problem-specific target gradients.
* You know the natural units of your problem better than the solver
  can infer from gradients at `x_0`. Wire `user-scaling`.
* The model has quadratic constraints written about the origin and
  started from zero. `gradient-based` cannot scale those rows at all —
  see [Quadratic rows the sampler cannot
  see](#quadratic-rows-the-sampler-cannot-see) — and `pounce check-x0`
  will say so. Try `curvature-based`.
* The model is a QCQP whose variables are in wildly different units.
  `curvature-based` equilibrates the columns jointly across every
  quadratic form; `gradient-based` has no column stage at all.

Otherwise the upstream-Ipopt-style defaults (`gradient-based` at the
NLP level, `none` at the linear-system level with MA57's internal
scaling on) are a reasonable starting point.

## References

* Wächter, A. *On the effects of scaling on the performance of Ipopt.*
  arXiv:1301.7283 (2013). <https://arxiv.org/abs/1301.7283>
* Ruiz, D. *A scaling algorithm to equilibrate both rows and columns
  norms in matrices.* CERFACS TR/PA/01/14.
  <https://cerfacs.fr/wp-content/uploads/2017/06/14_DanielRuiz.pdf>
* Curtis, A. R. and Reid, J. K. *On the Automatic Scaling of Matrices
  for Gaussian Elimination.* (1972). HSL MC19 reference.
* pounce issue [#61](https://github.com/jkitchin/pounce/issues/61).
