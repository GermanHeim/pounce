# gh #689 — the direct convex driver's cold start, and what the
# `scaled_feasible` pair actually measures

Two defects, both fixed: the direct driver's cold start (§1–§2) and HSDE's
scale-relative stopping test (§3).

## 1. The divergence: a cold start with no relation to the data

`crates/pounce-cli/tests/fixtures/scaled_feasible_a.nl` at `qp_hsde=no`:

| route | status | iters | objective | `final_kkt_error` |
|---|---|---|---|---|
| HSDE (default) | Optimal | 16 | 236.85 | 2.5e2 |
| direct, before | IterationLimit | 199 | 1.14e11 | 8.4e45 |
| direct, after | Optimal | 27 | ~0 | 2.6e-6 |

The pre-fix run was diverging, not crawling. The per-iteration trace of the
Ruiz-equilibrated solve makes the mechanism plain:

```
it=0 mu=1.0e0 pinf=5.0e9 dinf=1.0e2  |x|=0  |s|=1  |z|=1
  predictor: |dx_aff|=2.85e9   ap_aff=8.4e-9  ad_aff=3.3e-10
  corrector: |dx|=7.7e17 |dz|=7.4e18  step_p=1.2e-18 step_d=2.8e-18
it=3 mu=2.7e21 ... |z|=7.3e21
```

The **predictor direction is correct**: `‖dx_aff‖ = 2.85e9` against an optimum
at `‖x̂*‖ ≈ 5e9`. What kills the solve is that `init_iterate`'s cold start was
`s = z = e` — a unit-scale point asserted over a problem whose slacks are
`O(5e9)`. Fraction-to-boundary then caps the step at `≈ 0.95·s/‖ds‖ ≈ 3e-10`,
the iterate cannot move, and the corrector — which divides `σμ` by slacks still
pinned at `1` — returns directions eighteen orders larger. `z` runs away to
`7e21` and `μ` with it.

Why the equilibration does not save it: Ruiz balances the **matrix** `K`, not
the right-hand side. On this model `‖P‖ = 2`, `‖G‖ = 1e8`, so the symmetric
sweep settles at `Dc ≈ R_G ≈ 1e-4` — which lifts `Ĝ` to `O(1)`, drops `P̂` to
`2e-8`, and leaves `ĥ` at `5e9`. The equilibrated problem is well-*conditioned*
and badly *placed*: its optimum is at `‖x̂‖ ≈ 5e9` where the original was at
`‖x‖ ≈ 5e5`. (The claim in `equilibrate.rs` that "the Ruiz pass already
normalizes the `P` block to O(1)" holds only when the `P` entries dominate
their own rows of `K`; here the `G` entries do, by eight orders.)

**Fix.** The cold seed now runs through the same Mehrotra recentering
(Mehrotra 1992 §7) that the warm path already used, with the origin as its
seed: `s̃ = h − G·0 = h`, then the positivity shift and the centering shift.
The starting slacks are the problem's own scale by construction, and nothing
about a well-scaled problem changes (`h = O(1)` ⇒ the seed is `O(1)`).

## 2. Reaching the optimum is not the same as recognizing it

With the start fixed, the driver reaches the exact optimum at iteration ~27 —
and then runs another 148 iterations past it. `pinf` floors at `5.0000e-6` and
never moves:

```
it=27 mu=8.4e-9  pinf=5.0000e-6 dinf=2.6e-10  |x|=5.0000e9
...
it=175 mu=3.3e-293 pinf=5.0000e-6 dinf=1.0e-286  → NumericalFailure
```

`5e-6` is four ulp of `5e9`: forming `Gx + s − h` cancels two `5e9` quantities,
so no iterate can drive that residual under `tol = 1e-8`. The absolute
convergence test is unreachable, the loop keeps stepping, `s` and `z` underflow
into the denormals, and the factorization breaks down — on top of the optimum
(its true KKT error at that point reads `4e-25`).

HSDE already has the answer to this: `hsde::relative_stop_permitted` opens a
scale-relative arm exactly when `max_scale·ε > tol`. The direct driver now uses
the same gate and the same normalizers — **except that complementarity stays
absolute**. That distinction matters and is not cosmetic:

- `Gx + s − h` and `Px + c + Aᵀy + Gᵀz` are *differences of like-magnitude
  terms*. Their achievable accuracy is `≈ scale·ε`; below that floor only a
  relative statement means anything.
- `μ = ⟨s,z⟩/deg` is a *sum of products of nonnegatives*. Nothing cancels; it
  reaches zero at any problem scale. There is no floor to excuse relaxing it,
  and relaxing it costs real accuracy — see §3.

Measured both ways on `scaled_feasible_a`: with the gap normalized by `|obj|`
(HSDE's shape) the solve stops at iteration 18 with objective `0.486`; with `μ`
held absolute it stops at 27 with objective `-6e-5`, i.e. the optimum.

## 3. What the `scaled_feasible` pair actually measures

#689 asks whether these two fixtures "have genuinely flat or degenerate optima
— in which case the fixtures are poor trajectory sentinels — or the success
criterion is passing on iterates that are not optimal." It is the second.

Both models are `min Σ(xᵢ − aᵢ)²` where the objective's own centre `a` is the
`.nl` file's initial guess **and is feasible**, with three constraints exactly
active there. So `x* = a` and the optimum is `0`, exactly — verifiable by hand
from the fixture. The fixed direct driver returns

```
scaled_feasible_a  x = (-3.80752631972466071, 5.0e5, 0.5, 4.99999e5)   obj 0
scaled_feasible_b  x = ( 5.0e5, -9.216393895220474, 0.5, 5.0e5)        obj 0
```

against HSDE's `236.85` and `456.33`. Those are not answers; they are iterates
HSDE's stopping test admits. The mechanism is the same relative-gap
normalization discussed in §2, in HSDE's `gap_rel = gap/(1+scale_g)` with
`scale_g = |primal obj| ∨ |dual obj|`. `QpProblem` has no constant term, so the
objective the solver sees is `½xᵀPx + cᵀx = (user objective) − Σaᵢ²`, and
`Σaᵢ² ≈ 5e11` here. A gap of `5e3` therefore passes a `1e-8` relative test —
and `5e3` of gap is `5e3` of user-visible objective, because the user's
objective is that same near-total cancellation. The `#414` guard
(`verify_or_repair_optimum`) does not catch it for the same reason: its
`cscale` is the objective magnitude, so it reads `4.9e-10` on a point whose
absolute KKT error is `2.5e2`.

**Consequence for the sweep.** `scripts/sweep-fixtures.sh` treats an objective
move as signal, and on the default (HSDE) leg these two fixtures report a
number that is a property of where the trajectory happened to stop, not of the
model. That is why `qp_gondzio_corr=0` moves `a` 236.85 → 315.67 and the
adaptive-τ tail (#690) moves it to 272.94: all three are equally "converged" by
the relative test. Until HSDE's stopping rule is revisited, an objective move
on `scaled_feasible_a`/`_b` on the default leg should not be read as a
trajectory regression — the iteration count still should.

### The fix, and the two that do not work

The gap normalizer is right in form and wrong in its input. Told the objective
constant, `scale_g` measures the objective the caller reads and everything
follows: `QpOptions::obj_constant` (default `0.0`) carries it, the CLI sets it
from the `.nl` degree-0 term it already tracks for reporting plus presolve's
`obj_offset`, in the solver's minimize sense, and it travels through both cost
scalings (divided by `hsde_cost_scale`'s `σ`, multiplied by Ruiz's). It is a
convergence-test normalizer only — no residual, no direction, no dual, and not
`QpSolution::obj` — and `0.0` is the *tightest* value, so nothing that does not
set it changes.

Two tighter-looking rules were measured first and rejected, both on the same
evidence:

*Require absolute complementarity in the relative arm* — the rule the direct
driver uses (§2). Across the whole fixture corpus exactly five models reach the
relative arm, and it separates them perfectly:

| fixture | `⟨ŝ,ẑ⟩` at the stop | verdict |
|---|---|---|
| `feasible_x0_extreme_row` | 2.3e-11 | genuine |
| `feasible_x0_sentinel_bound` | 1.1e-28 | genuine |
| `feasible_x0_wide_scale` | 7.8e2 | false (KKT error 3.6e4) |
| `scaled_feasible_a` | 1.2e3 | false |
| `scaled_feasible_b` | 1.5e3 | false |

Thirteen orders of separation — and it still fails, because the *genuine*
gh #286 huge-magnitude optima sit at `⟨ŝ,ẑ⟩` of `1.5e9`
(`huge_magnitude_qp_recovers_exact_optimum_at_default_budget`) and `1.4e13`
(`issue_286_illconditioned_huge_scale_is_optimal_and_feasible`). Those are
points the tests certify against an exact oracle; absolute complementarity is
genuinely unreachable there (`x` is `O(1)` with duals at `1e18`, so the products
floor at `~1e9`), which is what the relative arm exists for.

*Normalize the gap by the gradient scale `scale_d` instead of the objective* —
offset-invariant, and it rejects all three false stops. But `|obj| ≈ ‖∇f‖·‖x̂‖`,
so this is tighter than the current rule by a factor `‖x̂‖`: it rejects any
problem whose magnitude lives in `‖x*‖` rather than in its coefficients, which
is POWELL20 (`‖x*‖ ~ 1e7`, `‖Px̂‖ ~ 7.5e3`, gap `4e2` — `5e-2` under this rule).

The constant is the only thing that actually separates the two families, so the
constant is what the fix supplies.

## 4. A false success the fix exposed, and the guard for it

The direct driver's convergence test — absolute or relative — is applied to the
*equilibrated* problem. That is not a statement about the point the caller gets:
Ruiz's dual map divides by `Dc`, so a `Dc` spanning many decades inflates the
recovered dual residual by up to `1/min Dc`.

`feasible_x0_sentinel_bound` (coefficients from `1e-320` to `1e30`,
`min Dc ≈ 6e-16`) is the case: at the stopping iterate `‖r_d‖ = 2.3e-9` in the
scaled metric and `2.3` in the user's, at objective `1.30` against a true `0`.
`feasible_x0_extreme_row` was already doing this **before** this change — the
pre-fix direct route returned `Solve_Succeeded` at objective `5.0e11`, with an
unscaled dual infeasibility of `2.6e6`, where the true optimum is `0`.

So a direct-driver `Optimal` reached through the equilibrated path is now
re-checked in the caller's coordinates (`demote_false_equilibrated_optimum`)
and demoted to `NumericalFailure` when it does not hold up there. Note the
symmetry with gh #414: there the *unscaled* relative test was the blind one and
the equilibrated metric was decisive; here it is the reverse. Neither metric is
trusted alone — a point has to look optimal in both.

## Sweep

`scripts/sweep-fixtures.sh`, both legs, 57 fixtures.

* **Default (HSDE) leg: empty diff.** Nothing in this change is reachable from
  the default route.
* **`qp_hsde=no` leg**, every line that moved:

| fixture | before | after | note |
|---|---|---|---|
| `scaled_feasible_a` | IterLimit 199, obj 1.14e11 | Optimal 27, obj ~0 | the issue |
| `scaled_feasible_b` | NumericalFailure 99, obj 5.0e11 | Optimal 28, obj 0 | the issue |
| `lp_afiro` | Optimal 135, obj -464.7531419 | Optimal 10, obj -464.7531428 | 13× fewer, and closer to the NETLIB optimum -464.75314286 |
| `rankdef_eq_qp` | Optimal 12 | Optimal 6 | KKT 2.7e-13 → 1.6e-12 |
| `qcqp_ball` | Optimal 15 | Optimal 12 | KKT 7.3e-9 → 1.2e-8 |
| `wyndor_min` | Optimal 7, obj -35.99999997 | Optimal 6, obj -36 | KKT 1.7e-8 → 9.6e-10 |
| `dual_order` | Optimal 4 | Optimal 5 | KKT 1.6e-9 → 1.5e-11 |
| `dual_scaled` | Optimal 4 | Optimal 5 | KKT 1.5e-11 |
| `feasible_x0_wide_scale` | Optimal 4 | Optimal 13 | KKT 6.7e-16 → 5.0e-17 |
| `feasible_x0_extreme_row` | **Optimal** 4, obj 5.0e11 | NumericalFailure 4 | pre-existing false success (true optimum 0), now demoted — see §4 |
| `feasible_x0_sentinel_bound` | NumericalFailure 101 | NumericalFailure 19 | same verdict, 5× cheaper |
| `lp_row_constant`, `_expr` | Optimal 5, KKT 7.2e-10 | Optimal 5, KKT 1.8e-8 | the largest accuracy regression in the diff |

Three lines end slightly less accurate: `lp_row_constant` / `_expr`
(`7.2e-10 → 1.8e-8`), `qcqp_ball` (`7.3e-9 → 1.2e-8`) and `rankdef_eq_qp`
(`2.7e-13 → 1.6e-12`). All stay inside the solved band, and the largest of them
moves `lp_row_constant`'s objective `-5.999999998 → -5.999999961` against an
exact `-6`. This is the cost of landing on a different point of the central
path from a different start, not a change in what the driver certifies — the
same sweep shows the reverse sign on `wyndor_min` (`1.7e-8 → 9.6e-10`),
`lp_afiro` (`2.1e-7 → 2.3e-9`), `dual_order` (`1.6e-9 → 1.5e-11`) and
`feasible_x0_wide_scale` (`6.7e-16 → 5.0e-17`).

## 5. Sweep — the HSDE half

`scripts/sweep-fixtures.sh`, both legs, after §3's fix.

* **`qp_hsde=no` leg: empty diff.** `obj_constant` enters only HSDE's
  `scale_g`; the direct driver's tests do not read it.
* **Default (HSDE) leg**, every line that moved — all four are models with a
  large objective constant, which is the whole affected class:

| fixture | before | after | KKT error |
|---|---|---|---|
| `scaled_feasible_a` | Optimal 16, obj 236.85 | Optimal 123, obj 0 | 2.5e2 → 4.6e-3 |
| `scaled_feasible_b` | Optimal 21, obj 456.33 | Optimal 47, obj 0 | 9.3e2 → 1.2e-10 |
| `feasible_x0_wide_scale` | Optimal 16 | Optimal 80 | 3.6e4 → 6.6e-15 |
| `feasible_x0_extreme_row` | Optimal 32 | Optimal 33 | 7.6e-4 → 3.8e-5 |

Three of the four were false optima — `feasible_x0_wide_scale` was returning a
point with a KKT error of `3.6e4` under a `Solve_Succeeded`. The iteration
counts are the cost of the work those solves were skipping: with the constant
supplied, `scale_g` on these models is the *caller's* objective, which tends to
`0` at the optimum, so the relative test degenerates to the absolute one and
HSDE has to drive the gap to `tol` outright — which is exactly what the direct
driver does on the same models in 27 and 28 iterations.

### The gate, and why `feasible_x0_wide_scale` first went to 198

Correcting `scale_g` had one knock-on that had to be corrected with it, and it
is the more interesting half. `large_scale` — the gate that admits the relative
test at all — is keyed on `max(scale_d, scale_p, scale_g)`. Making `scale_g`
*accurate* makes it *small* on exactly the models this change targets, and on
`feasible_x0_wide_scale` that closed the gate outright, leaving only the
absolute test. The primal and dual residuals there floor at `5e-9` and `5e-6`
on `5e13`-scale data — facts about the constraint system, with nothing to do
with the objective constant — so the solve could never finish:

```
it=18 inf_pr=5.089e-09 inf_du=4.929e-05 mu=9.062e-18 a_p=9.813e-01
it=20 inf_pr=5.007e-09 inf_du=8.332e-06 mu=1.088e-19 a_p=1.000e+00
it=25 inf_pr=5.102e-09 inf_du=4.496e-06 mu=1.448e-24 a_p=1.000e+00
...
it=120 inf_pr=5.051e-09 inf_du=4.927e-08 mu=1.621e-119    <- denormals
it=130 inf_pr=5.040e-09 inf_du=6.687e-11 mu=3.465e-118 a_p=2.784e-07
it=140 inf_pr=4.343e-01 inf_du=4.984e-05 mu=1.823e+05    <- iterate discarded
it=180 inf_pr=4.846e-09 inf_du=2.023e-07 mu=1.641e+09    <- and again
```

Converged at 18, then 180 iterations of collapse-and-restart, terminating at
198 against a 200 cap on an answer it had found long before — and
`crates/pounce-cli/tests/false_local_infeasibility.rs::the_convex_route_still_solves_both_shapes`
asserts `solve_result_num == 0` on this model, so that was two iterations from
a red test, not just an ugly sweep line.

The gate is asked whether `tol`-level *absolute* accuracy is reachable on this
data at all. That is a property of the magnitudes actually being computed, not
of where the caller's zero happens to sit, so it reads the objective's own
magnitude (`scale_g_raw`) while `scale_g` supplies the gap's *normalizer*. With
that split the model converges in 80.

A gap floor (`gap ≤ N·ε·max|term|`, the same "relax to the arithmetic floor"
rule §2 applies to the direct driver) was tried on top and **dropped**. It is
faster — 23/28/39/32 iterations at `N = 8` — but it costs real accuracy on the
two fixtures the issue is about (`scaled_feasible_a` reports `6.1e-5` and `_b`
`2.4e-4` instead of `0`), and every `N` large enough to matter is a constant
fitted to one fixture's `τ`-collapse noise. The gate correction alone is
exact everywhere and needs no such constant.
