# issue #602 — `solve_parametric` eligibility, measured

[#602](https://github.com/jkitchin/pounce/issues/602) observes that
`solve_parametric` admits a warm parametric solve on a guard that checks only
shape and `H`, while the homotopy it then runs interpolates only `g` and the
row bounds — and proposes extending the guard to also require an unchanged
constraint matrix, unchanged variable bounds, and an unchanged Hessian-inertia
declaration.

The observation is correct on all three counts. This note records what each of
the three actually costs today, because the fix that follows from the data is
not the one the issue proposes.

Short version:

* This is a **cost** question, not a correctness one. Every route measured
  lands on the same answer to `1e-14`; the path is a predictor and the
  corrector re-solves against the true problem.
* Bullets 1 and 2 (`A`, variable bounds) are real and measurable. Bullet 3
  (inertia) has **no effect on this path at all** — `hessian_inertia` is never
  read by the homotopy or by the KKT factorization.
* **Rejecting an ineligible pair usually costs more than admitting it**,
  because the fallback in the current code is a *cold* solve — and on this
  family the ineligible cases beat cold in 10 of 13 rows. The useful change is
  not a stricter guard, it is a **better fallback**: `solve_with_working_set`,
  which is already what the SQP driver uses and which is the fastest of the
  three routes in 12 of 17 rows measured.

---

## What the guard checks, and what the path models

`solve_parametric_scoped` (`crates/pounce-qp/src/solver.rs`) admits the warm
path when `(n, m)` match, `H` is bit-identical (nonzeros, values, `irows`,
`jcols`), `sol_prev.status == Optimal`, and `sol_prev.x.len() == qp_new.n`.

`solve_homotopy`'s warm arm (`crates/pounce-qp/src/homotopy.rs`) then starts at
`x = sol_prev.x` with `working = sol_prev.working`, and traces to `t = 1`
carrying exactly two moving quantities:

* `dg = qp.g − prev.g`, entering the direction solve as the primal right-hand
  side `−dg`;
* row bounds, interpolating `prev.bl → qp.bl` and `prev.bu → qp.bu` through
  `bound_rate`.

Everything else the tracer touches is taken from the **new** problem and
treated as constant along the path: `qp.a` in both `a_times_x` calls and in
`assemble_active_set_kkt`, and `qp.xl` / `qp.xu` in the pinned box. So three
quantities can differ between the two problems with the path modelling the
difference not at all:

| quantity | checked by the guard? | modelled by the path? |
|---|---|---|
| `n`, `m`, `H` | yes | `H` held fixed (correctly — it is not interpolated) |
| `g`, `bl`, `bu` | no (correctly) | **yes**, interpolated |
| `A` (structure and values) | **no** | **no** |
| `xl`, `xu` | **no** | **no** |
| `hessian_inertia` | **no** | not read at all — see below |

The mechanism behind the issue's concern is worth stating precisely, because it
bounds how bad the damage can get. The tracer is a **pure predictor with no
corrector inside the loop**: each step solves for `d/dt (x, λ)` and takes a
linear step. It never evaluates a residual and never projects back onto the
manifold. So an initial off-manifold error does not grow *through a feedback
mechanism* — it is simply never removed, and the linear extrapolation is taken
about the wrong point for the whole path. That is why the failure shows up as a
degraded active-set *prediction* rather than as a wrong answer.

## Bullet 3 first: the inertia declaration is a no-op here

`hessian_inertia` is read in exactly one place in the crate — `elastic.rs`,
where `ElasticProblem::as_qp` maps it onto the augmented problem. The homotopy
never reads it, and neither does `factorize_with_inertia_control`, which takes
its `expected_neg` as an explicit argument and is documented as deciding from
the factor rather than the hint:

> `expected_neg` is required (no bypass) so the inertia signal is always
> checked. The `HessianInertia::Indefinite` hint merely tells the caller
> "shifts may be needed"; the algorithm decides what to do based on the
> factor's report.
> — `solver.rs`

Measured, flipping the declaration with `H` bit-identical changes nothing:

| change | homotopy |
|---|---|
| `Psd` → `Psd` (baseline) | `Optimal chg=4` |
| `Psd` → `Indefinite` | `Optimal chg=4` |
| `Psd` → `Unknown` | `Optimal chg=4` |

Since `H` must already be bit-identical to reach this path, a differing
declaration means the *caller* contradicted itself about one matrix, which is
worth rejecting as hygiene — but it is not a source of wasted path work, and it
should not be sold as one.

## Bullet 2 is real, and the cause is worse than "not interpolated"

The issue asks for unchanged variable bounds "unless box-bound interpolation is
implemented". Interpolating the box would not be sufficient, because the
homotopy has **no bound-adding event at all**. The `Event` enum is

```rust
enum Event { AddRowLower(usize), AddRowUpper(usize), DropRow(usize), DropBound(usize) }
```

— rows can be added and dropped, and bounds can be *dropped*, but no inactive
variable bound can ever *become* active on the path. The primal ratio test
loops `for i in 0..m` over general rows only, and `worst_path_violation`
documents the matching blind spot ("variable bounds do not move along this
path, so they are not checked here").

The consequence is not confined to the parametric entry point: on the cold arm
too, nothing stops `x(t)` walking straight out of its box. For #602 it means
that if the new problem tightens `xu` below the previous solution, the path
starts outside the new box, never notices, and hands the corrector a working
set built along an infeasible trajectory. Measured (`xu` capped, `g`/`b` moved
slightly):

| change | homotopy | cold | ws-only |
|---|---|---|---|
| `xu = 2` | `chg=4` 8.1 ms | `chg=97` 304.7 ms | `chg=3` **5.2 ms** |
| `xu = 0.5` | `chg=77` 241.4 ms | `chg=94` 299.3 ms | `chg=7` **11.9 ms** |
| `xu = 0.1` | `chg=77` 217.2 ms | `chg=103` 338.5 ms | `chg=75` **210.0 ms** |

The `xu = 0.5` row is the sharpest single result in this note: the homotopy
spends **20×** what simply reusing the previous working set costs, and a
stricter guard as proposed would have replaced it with the cold solve, which is
**25×**. The waste is real; the proposed remedy is the wrong end of it.

## Bullet 1 is real, and its damage is not monotone

Perturbing every entry of `A` by a relative `da` (structure fixed), with `g`
and `b` moved slightly:

| `da` | homotopy | cold | ws-only |
|---|---|---|---|
| 0.02 | `chg=3` 8.6 ms | `chg=18` 56.2 ms | `chg=3` 5.6 ms |
| 0.05 | `chg=65` 189.7 ms | `chg=82` 257.2 ms | `chg=64` 182.0 ms |
| 0.10 | `chg=5` 17.5 ms | `chg=19` 61.5 ms | `chg=4` 6.9 ms |
| 0.20 | `chg=6` 19.1 ms | `chg=19` 71.6 ms | `chg=5` 9.3 ms |
| 0.30 | `chg=8` 23.7 ms | `chg=19` 64.8 ms | `chg=7` 12.7 ms |
| 0.40 | `chg=53` 180.8 ms | `chg=16` **59.7 ms** | `chg=52` 157.0 ms |
| 0.50 | `chg=54` 167.8 ms | `chg=16` 52.7 ms | `chg=9` **14.0 ms** |
| 0.60 | `chg=53` 152.2 ms | `chg=17` **63.1 ms** | `chg=52` 156.8 ms |
| 0.80 | `chg=55` 157.4 ms | `chg=71` 215.7 ms | `chg=54` 156.8 ms |
| 1.00 | `chg=45` 138.7 ms | `chg=61` 183.4 ms | `chg=45` 140.9 ms |

Two things to read off it. The homotopy loses to cold at `da ∈ {0.40, 0.60}` —
so the issue's concern reproduces. And the damage is **not monotone in `‖ΔA‖`**:
`da = 0.05` is expensive on all three routes (a harder target, not a warm-start
failure) while `da = 0.30` is cheap and `da = 1.00` is fine again. A guard
thresholded on how much `A` moved would be fitting noise — which is precisely
the trap [#434](https://github.com/jkitchin/pounce/issues/434) documented when
its own candidate guard turned out to be separated from destroying a genuine
gain by a 3% margin on one instance (`dev-notes/issue-434-homotopy-cost.md`,
"Result 2 — the guard, declined").

What *is* monotone is a quantity the path already computes. The
`POUNCE_HOMOTOPY_DEBUG` handoff line reports the target violation of the point
the path hands the corrector, and it tracks `‖ΔA‖` cleanly:

| `da` | handoff violation |
|---|---|
| 0.00 | 5.3e-2 |
| 0.02 | 1.5e-1 |
| 0.10 | 2.1e-1 |
| 0.30 | 3.5e-1 |
| 0.60 | 5.7e-1 |

That makes it a candidate *runtime* signal (measure the path, do not predict
it — #434's preferred shape), but it is measured here on one synthetic family
and separates the good rows from the bad ones only loosely: `da = 0.30` at
3.5e-1 is cheap and `da = 0.60` at 5.7e-1 is not, with nothing in between to
place a threshold on. It needs the real suite before it is worth anything.

## Correctness: not at risk

Across all 17 rows measured, `|x_warm − x_cold| ≤ 8.4e-15` and
`|obj_warm − obj_cold| ≤ 1.4e-14`, every route `Optimal`. That is by
construction rather than luck: `trace_path` never reports its own point as a
solution, it hands the discovered working set to `solve_with_working_set`,
which pins a primal, routes through `solve`'s infeasible-warm-start pre-check
into l1-elastic phase-1 when the pin is unusable, and finishes under
`audit_and_repair`. The M5 audit re-checks every row and bound against
`feas_tol` before an `Optimal` is allowed out.

So #602 should be scoped as a performance issue. Nothing here justifies a
correctness-flavoured urgency, and framing it that way would invite exactly the
"it cannot produce a wrong answer, therefore ship it" reasoning that `CLAUDE.md`
warns about — in the opposite direction.

## Two incidental findings

**`solve_parametric` ignores `use_homotopy`.** `solve_scoped` gates the cold
homotopy on `ws.is_none() && opts.use_homotopy`; `solve_parametric_scoped`
calls `solve_homotopy` unconditionally. Verified: the parametric route produces
identical statistics (`chg=4 refac=8`) with the flag `false` and `true`. Since
`use_homotopy` defaults to `false` in `pounce-qp` — deliberately, per the #434
work — the public parametric entry point is the one place the homotopy runs
without an opt-in and without a kill switch. Whether that is intended is a
separate question from #602, but any option-level mitigation for #602 would be
built on a flag that this path does not currently honour.

**`solve_parametric` has no production caller.** The only callers in-tree are
`crates/pounce-qp/tests/homotopy.rs` and `crates/pounce-rs/tests/qp_surface.rs`.
The SQP driver (`pounce-algorithm/src/sqp/sqp_alg.rs`) warm-starts through
`solve_with_working_set`, precisely because each SQP linearization moves `A` and
translates the row bounds by `−c(x_k)`, so the previous *primal* does not carry
over. It is worth being explicit about what that implies for #602's framing:
the natural consumer of a parametric API is the SQP outer loop, and in that
loop `A` changes at **every** iteration. A guard that requires `A` unchanged
would make `solve_parametric` permanently ineligible for the workload it exists
to serve. The eligible population is the fixed-`A` parametric family — MPC with
fixed dynamics, an RHS/objective sweep, a continuation step — which is real, but
is a narrower claim than the crate currently makes.

(Relatedly, `pounce-qp`'s package description advertises "the parametric
corrector inside `pounce-sensitivity`". `pounce-sensitivity` does not depend on
`pounce-qp`. Cosmetic, ships to crates.io, not part of this issue.)

## What to do

Ranked by measured value, not by how closely it matches the issue text.

1. **Change the fallback, not the guard.** — **done, see below.**
   `solve_parametric_scoped`'s ineligible branch was
   `self.solve(qp_new, None, opts)` — a cold solve that throws away a working
   set the caller just handed us. Routing it to
   `solve_with_working_set(qp_new, &sol_prev.working, opts)` costs nothing when
   the guard already passes and turns every rejection from "start over" into
   "keep the active-set guess". `ws-only` is the fastest of the three routes in
   **12 of the 17** rows above, and never loses to the homotopy by more than
   noise. The three rows it loses (`da ∈ {0.40, 0.60, 1.00}`) are ones where
   *both* warm routes lose to cold, i.e. where the previous active set is
   genuinely a bad guess — a case for a runtime bail-out, not for preferring a
   cold restart by default. This is worth doing on its own merits, before any
   eligibility change, and it is what makes a stricter guard affordable.

2. **Then tighten the guard, in this order.** With (1) in place, rejecting an
   ineligible pair is cheap, so the guard can be honest about what the path
   models: require identical `A` (structure *and* values — compare
   `nonzeros/values/irows/jcols`, the same test already applied to `H`), and
   identical `xl`/`xu`. Reject a mismatched `hessian_inertia` too if desired,
   but document it as a caller-consistency check, not a path-cost one.

3. **Or fix the box blind spot instead**, which is the larger prize and is not
   specific to #602: add `AddBoundLower` / `AddBoundUpper` events and the
   matching primal ratio test over `j in 0..n`. That removes the reason
   variable-bound changes hurt, benefits the **cold** arm as well (where `x(t)`
   can currently leave the box on any problem), and is the prerequisite for the
   box-bound interpolation the issue asks about. Interpolating `xl`/`xu`
   without it would move a bound the ratio test still cannot see.

4. **Interpolating `A` is a different algorithm — do not scope it here.** With
   `A(t) = (1−t)A₀ + tA₁` the KKT matrix itself becomes `t`-dependent, so
   `(x(t), λ(t))` is no longer affine along a segment and the two ratio tests
   in `t` stop being exact. It needs a genuine predictor-corrector continuation
   (a Newton correction back onto the manifold at each step), not an extension
   of the current tracer. Given (1), the payoff over `solve_with_working_set`
   would have to be demonstrated before the complexity is worth it.

## What must be measured before any of it merges

All four options above reroute which correction the solver reaches for, which
makes them **trajectory changes** under `CLAUDE.md` — `scripts/sweep-fixtures.sh`
against a baseline binary, diffed before merge, with every moving line
explained. In addition:

* The synthetic family in this note is `n = 30`, `m = 20`, diagonal PD `H`. It
  is an instrument for *mechanism*, not evidence about the shipped workload.
  The real comparison is the 138-problem Maros-Mészáros sweep through
  `crates/pounce-convex/examples/homotopy_sweep.rs` and the warm-start suite in
  `benchmarks/warmstart` (whose `-hom` arms differ by exactly one option).
* Option 1 needs an A/B on the warm-start suite specifically, since that is
  where a changed fallback shows up.
* Any threshold-shaped rule needs the #434 treatment: replay candidate rules
  against recorded per-step trajectories, not against endpoint summaries, and
  decline the rule if nothing separates the losses from the gains with margin.

## Reproducing

```text
cargo run -p pounce-qp --example parametric_eligibility_sweep
POUNCE_HOMOTOPY_DEBUG=1 cargo run -p pounce-qp --example parametric_eligibility_sweep
```

The tables above are from the debug profile, so read the ratios between columns
rather than the absolute milliseconds. With the trace enabled each row is
preceded by three `[hom] summary` lines — previous solve, warm path, cold solve
— and the middle one carries the handoff violation quoted above.

---

## Step 1, implemented

`solve_parametric_scoped`'s declined branch now routes to
`solve_with_working_set` when the previous solve reached `Optimal` and its
working set is dimensionally valid for the new problem, falling back to a cold
solve otherwise.

The measurement above could not show this, because every row in it is a pair
the guard *admits* — the fallback is never reached. The sweep therefore gained
a section that perturbs `H`, which is the one quantity the guard rejects on, so
`solve_parametric` never reaches the path and the `homotopy` column is the
fallback and nothing else:

| change | before | after | `ws-only` |
|---|---|---|---|
| `H` 1% | `chg=18` 14.8 ms | `chg=3` 1.3 ms | `chg=3` 1.2 ms |
| `H` 1%, `A` 10% | `chg=19` 16.7 ms | `chg=4` 1.5 ms | `chg=4` 1.8 ms |
| `H` 5% | `chg=18` 15.5 ms | `chg=3` 1.1 ms | `chg=3` 1.1 ms |
| `H` 10% | `chg=18` 14.6 ms | `chg=3` 1.1 ms | `chg=3` 1.3 ms |
| `H` 10%, `A` 10% | `chg=84` 58.1 ms | `chg=67` 42.5 ms | `chg=67` 35.8 ms |
| `H` 25% | `chg=18` 15.3 ms | `chg=3` 1.4 ms | `chg=3` 1.7 ms |
| `H` 50% | `chg=17` 15.1 ms | `chg=2` 0.8 ms | `chg=2` 0.8 ms |
| `H` 50%, `A` 10% | `chg=82` 51.6 ms | `chg=65` 37.1 ms | `chg=65` 34.8 ms |

Two things worth reading off it. The "before" column is *identical* to the
`cold` column, which is the direct confirmation that the declined branch was a
cold solve; the "after" column is identical to `ws-only`, which is the
confirmation that it now isn't. And the hint survives a **50%** Hessian
perturbation (2 changes against a cold solve's 17) — because what a working set
encodes is which constraints bind, and that is far more stable under a change
of `H` than the iterate is. This is why the change is worth making even though
the guard rejects these pairs for a good reason.

Answers are unchanged: agreement with a cold solve of the same target within
`1.8e-15` on every row.

### Conditions on the hint

* `sol_prev.status == Optimal`. A `TimeLimit` result carries
  `WorkingSet::cold` — all-inactive, i.e. no information — and a `MaxIter` one
  carries a set that was still moving. Neither is covered by the measurement,
  so neither gets the hint.
* `sol_prev.working.validate_dims(qp_new.n, qp_new.m).is_ok()`. Without it, a
  shape change reaches `solve_with_working_set` and comes back
  `WarmStartDimensionMismatch` — converting "no warm start available" into
  "solve failed" for a call that has a perfectly good cold answer available.

### Test

`ineligible_parametric_reuses_the_working_set` in
`crates/pounce-qp/tests/homotopy.rs` asserts the answer matches a cold solve
**and** that the change count is strictly below it. The second half is the
point: the old behaviour returned the right answer and would pass any
answer-only assertion, which is precisely the gh #544 shape. Verified to fail
against the previous code (7 changes against cold's 7) and pass after.

The two existing closed-form cases are too small to carry that assertion — a
cold solve reaches their optima in 0–2 working-set changes — so the test builds
an `n = 12`, `m = 8` case with several rows binding.

### Fixture sweep

`scripts/sweep-fixtures.sh` over all 57 CLI fixtures, baseline against new, on
both the default arm and `algorithm=active-set-sqp`: **empty diff on both**.

That is the expected result and it is *not* evidence the change is good.
`solve_parametric` has no production caller, so no CLI fixture can reach the
branch that moved. The sweep's value here is confirming the change did not leak
into anything else in `solver.rs` — it bounds collateral damage, it does not
measure the intended effect. The intended effect is the table above, on a
synthetic family, and it stays synthetic until something in-tree calls
`solve_parametric`.

### Review finding: reusing a working set needs more than matching dimensions

[@GermanHeim caught a bug](https://github.com/jkitchin/pounce/issues/602#issuecomment-5303884538)
in the change above, and it was a **wrong answer**, not a slow one. Reproduced
before fixing:

> prev: `min ½x² s.t. x == 1` → `x = 1`
> new:  `min x²  s.t. x ≤ 2` (`H` changed, so the guard declines)
>
> | route | status | `x` | objective |
> |---|---|---|---|
> | parametric | `Optimal` | **−1e19** | **1e38** |
> | cold | `Optimal` | 0 | 0 |

The dimensional check the change shipped with is necessary but not sufficient.
`ConsStatus::Equality` and `BoundStatus::Fixed` are not "active" markers — they
are assertions about the *problem's* bound topology (`bl == bu`, `xl == xu`),
and they carry the semantics that follow: always in the working set, multiplier
unrestricted in sign, and a drop score hard-wired to `0.0` — never dropped.
Carried onto a problem where the row is a range, that is a false statement
nothing downstream can retract. `pin_working_set` pins an `Equality` row to
`qp.bl[i]`, which for the new problem is the `-1e20` infinity sentinel; the
resulting iterate is feasible for `x ≤ 2`, so the M5 audit passes it; no drop
test can reconsider it; and the solve reports `Optimal`.

The same shape reproduces through `Fixed` when a previously-pinned variable is
freed (`x = [−0, −1e19]`), which the report predicted and which the fix covers.

**Why the audit does not catch it.** `audit_and_repair` checks feasibility, and
`x = −1e19` *is* feasible. The point is optimal for the problem the working set
describes; it is the working set that describes the wrong problem. No
feasibility-based check can see that.

**The distinction that matters.** `AtLower` pinned to a lower bound the new
problem does not have lands on the same sentinel — but it is *droppable*, so
the dual ratio test sees a multiplier of the wrong sign and removes it. The
damage is iterations. Only `Equality` and `Fixed`, which no drop test can
reconsider, convert a bad pin into a wrong answer. That is the whole reason the
first two rules of the fix are load-bearing and the rest are hygiene.

**The fix.** `WorkingSet::reconciled_with(qp, opts)` re-derives both
topology-dependent statuses from the new problem and drops any remaining status
that names a bound the new problem does not have, matching the predicates the
solver uses when it builds a working set itself (exact `bl == bu` for rows,
both-finite-within-`feas_tol` for bounds). What survives is the part of a
working set that is only ever a guess about which constraints bind — the part
that costs at most iterations when it is wrong.

Four regression tests in `crates/pounce-qp/tests/homotopy.rs`. Two of them
(`Equality` → range, `Fixed` → free) fail against the unfixed change; the other
two pin rules that currently cost iterations rather than correctness, and their
doc comments say so rather than implying more coverage than they have. The
step-1 measurements are unchanged — reconciliation is a no-op when the topology
matches, which is every row of the sweep.

### The same hazard exists in `solve_with_working_set` itself

Worth recording separately, because the fix above does **not** address it.
`solve_with_working_set` is public API and applies no such reconciliation: a
caller who passes a working set from a problem whose bound topology has changed
gets the `Optimal`-at-`-1e19` behaviour directly. What the change above did was
make `solve_parametric` reach that hazard *on its own*, from two well-formed
problems and no caller mistake — so fixing it there was the regression fix, not
the whole problem.

The in-tree caller that matters is the active-set SQP driver, which is safe by
construction: an SQP linearization moves `A` and translates the row bounds, but
a constraint that is an equality stays an equality across iterations, and a
fixed variable stays fixed. That is why this has never been observed there, and
also why it is not a reason to relax about it — it is an unstated precondition
holding by accident of the caller, not by contract.

Options, none of them taken here: document the precondition on
`solve_with_working_set`; make it reconcile internally (a trajectory change for
the SQP driver, so it needs its own A/B on `benchmarks/warmstart`); or promote
`reconciled_with` to public API so callers can opt in. The middle one is
probably right, and it is the kind of change that should be measured rather than
assumed.
