# gh#712 — the complementarity normalizer, and the cut that separates the families

`optimum_is_genuine` certifies a point either on its absolute KKT error or, when
that fails, on `equilibrated_kkt_rel` against `FALSE_OPTIMUM_REL_TOL = 1e-3`.
The relative arm's complementarity term was normalized by
`½x̂ᵀP̂x̂ + ĉᵀx̂` — the `QpProblem` objective, which on a model carrying a
degree-0 term is the caller's objective *displaced* by it. On
`scaled_feasible_a` (`Σaᵢ² ≈ 5e11`) that read `4.57e-9` on a point whose
absolute KKT error is `2.28e3`, and the gh#293 Ruiz retry's `Optimal` was
returned to the caller as `SolveSucceeded`.

This is the second of the two sites #696 named. It corrected HSDE's `scale_g`
with `QpOptions::obj_constant` and left this one, which the #689 dev-note had
already flagged as reading `4.9e-10` on a point whose absolute error is `2.5e2`.
Post-#696 it is no longer merely failing to catch a bad point — it is what
*certifies* the retry's answer.

## Why the obvious fix does not work

Adding the constant, `cscale = |½x̂ᵀP̂x̂ + ĉᵀx̂ + const·σ|`, does not re-centre
the objective — on these models it **cancels** it. The two parts are equal and
opposite (`-5.000e11` and `+5.000e11` on `scaled_feasible_a`), so the sum
carries no scale at all, the `max(1.0)` floor takes over, and the relative test
silently becomes an absolute one.

That is right for a model whose data is `O(1)` and wrong for one whose
magnitudes are enormous. It rejects `feasible_x0_wide_scale`, whose certified
point matches the NLP oracle to 13 digits (`499998.30314177` against `…78`) and
whose absolute KKT error of `1.32e-1` is `5.3e-13` of its own objective terms.
Measured: that variant takes the model from 80 iterations to 199 against a 200
cap — handing back exactly the margin `cb43358` fixed when it took the same
model from 198 to 80.

It is the same split `cb43358` had to make inside HSDE, which corrected the
*gap's normalizer* with the constant while keeping the *gate* — "is `tol`-level
absolute accuracy reachable on this data at all" — on the objective's own
magnitude, because that is a property of the magnitudes being computed and not
of where the caller's zero sits.

## The calibration

Every point that reaches the normalizer, with its ground truth. `rel` columns
are `comp / cscale` against the `1e-3` cut. Measured by dumping the
normalizer's ingredients (`POUNCE_DUMP_CAL`) and driving the gh#414 and gh#286
family tests plus the CLI fixtures through it.

| point | truth | abs KKT | current | +const | +const, floored |
|---|---|---|---|---|---|
| `414` n3dec3 false | reject | 8.28e+03 | 1.23e+02 ✓ | 1.23e+02 ✓ | 1.23e+02 ✓ |
| `414` n4dec4 false | reject | 1.56e-01 | 2.01e-02 ✓ | 2.01e-02 ✓ | 2.01e-02 ✓ |
| `414` n6dec6 false | reject | 3.05e+03 | 2.80e+01 ✓ | 2.80e+01 ✓ | 2.80e+01 ✓ |
| `414` repaired ×4 | accept | ≤2.1e-04 | 2.9e-10‥1.0e-9 ✓ | same ✓ | same ✓ |
| `286` huge 1e18 | accept | 1.50e+09 | 3.88e-10 ✓ | same ✓ | same ✓ |
| `286` illcond 1e12×1e10 | accept | 1.39e+13 | 1.47e-08 ✓ | same ✓ | same ✓ |
| `286` mid 1e12 | accept | 1.43e+03 | 3.70e-10 ✓ | same ✓ | same ✓ |
| `712` `scaled_feasible_a` | **reject** | 2.28e+03 | 4.57e-09 ✗ | 2.28e+03 ✓ | 1.53e-01 ✓ |
| `712` `feasible_x0_wide_scale` | **accept** | 1.32e-01 | 5.33e-13 ✓ | 1.33e-01 ✗ | 1.79e-05 ✓ |
| `712` `feasible_x0_extreme_row` | accept | 3.81e-05 | 3.81e-20 ✓ | 3.41e-13 ✓ | 2.29e-17 ✓ |

(The `+const, floored` column is the denominator-floor variant described
below under *the rule that did not ship*. It is kept because it is the
measurement that established the shape of the problem, not because it is
what the tree does.)

Two things this settles that argument could not:

* **No cut on the current normalizer can work.** The genuine `286` illcond
  point reads `1.47e-8` and the false `scaled_feasible_a` point reads
  `4.57e-9` — the good one is *worse* than the bad one. The constant is
  missing information, not a mis-tuned threshold.
* **A data-scale normalizer is not the answer either**, which was the first
  guess. The `286` objectives are huge because the magnitude lives in `‖x*‖`,
  not in the data: `datascale` is `2e9` where the objective is `3.87e18`, so
  normalizing by data rejects the whole family outright.

## The rule

Restoring the constant is only half a rule, because — per the section above —
it hands back the margin on `feasible_x0_wide_scale`. Something has to keep
that model certified. Two candidates were built and measured.

### What ships: filter the numerator

A slack that is smaller than the rounding quantum of the operands that
produced it is not a constraint violation, it is the residue of computing
`h - Gx` in floating point. Such a term is dropped from the complementarity
max rather than allowed to define it:

```rust
const SLACK_NOISE_KAPPA: f64 = 64.0;

fn slack_is_resolvable(slack: f64, magnitude: f64) -> bool {
    slack.abs() > SLACK_NOISE_KAPPA * f64::EPSILON * magnitude
}
```

with `magnitude` the larger of the two operands on that row (`|hᵢ|` vs
`|gᵢx|`, `|xᵢ|` vs `|lb|`/`|ub|` on the bound pairs). The denominator keeps
the plain `|quad + lin + const·σ|` with its `max(1.0)` floor.

`64ε` is not a new calibration. It is the quantum this codebase already uses
for "numerically zero" in `pounce_algorithm` — `ROW_NOISE_KAPPA` and
`primal_noise_floor_kappa`, from gh#446 and gh#528.

The question it asks is *local*: for this one row, is the slack
distinguishable from zero at the precision of the numbers that produced it?
The relaxation it grants is bounded by that row's own operands, and does not
depend on how badly the objective happens to cancel.

### The rule that did not ship: floor the denominator

```rust
cscale = max(|quad + lin + const·σ|,
             OBJ_CANCELLATION_FLOOR * max(|quad|, |lin|, |const·σ|),
             1.0)      // φ = √ε = 1.49e-8
```

The floor is *relative to the terms*: the objective is never trusted as a
scale below a fixed fraction of the magnitudes that produced it. A model whose
sum retains any of its terms' magnitude is untouched, which is every gh#414
and gh#286 instance, so the family keeps the numbers `FALSE_OPTIMUM_REL_TOL`
was calibrated against and the cut does not move. The cancelling models admit
any `φ ∈ (2.7e-10, 2.3e-6)`; `√ε` sits 56× above that floor and 153× below its
ceiling. Plain `ε` is inside the *lower* bound and rejects
`feasible_x0_wide_scale`.

It works. It was not taken for one reason, recorded here because it is the
kind of thing that gets rediscovered: **the floor is not conditioned on a
constant being supplied.** Cancellation is a property of the arithmetic, so an
objective that cancels on its own (`½xᵀPx ≈ −cᵀx`, both huge, no degree-0
term) is affected too — and there the effect runs the other way. It *raises*
`cscale` off a sum that carries no scale, relaxing the gh#414 guard rather
than tightening it, by as much as `1/√ε ≈ 6.7e7`. No fixture in the corpus is
in that regime and neither family exercises it, so unlike every other row in
the table above that surface rests on argument and not measurement.

Both candidates are *unconditional* — neither is gated on a non-zero
`obj_constant`, and both therefore move constant-free callers. The difference
is the bound on the relaxation each grants: the numerator filter's is tied to
the operands of the row it drops, the denominator floor's is tied to how badly
an unrelated objective cancels.

### Measured equivalence

On the fixture corpus the two are indistinguishable. Swept against a common
baseline at `9b713ddd`, 120 fixture-legs each, both legs: both variants move
exactly one line — `scaled_feasible_a`, below — and their corpora are
byte-identical to each other. The corpus cannot arbitrate between them, which
is why the choice above is argued from the bound and not from a diff.

## The second half of the fix

The normalizer alone changes nothing on the target model. `solve_qp_ipm_core`'s
gh#293 Ruiz retry accepted `Optimal` on *status alone* and returned it, and
that early return bypasses `verify_or_repair_optimum` entirely — so the
corrected test was never consulted. It was the one `Optimal` in that function
reaching a caller without a genuineness check. It now earns it the same way,
and a retry that cannot certify leaves the original status standing.

## What ships

Fixture sweep, both legs — **one model moves**:

```
scaled_feasible_a   SolveSucceeded 123it  ->  MaximumIterationsExceeded 199it
```

Everything else in the corpus is bit-identical on both legs, including
`feasible_x0_wide_scale` at 80, `scaled_feasible_b` at 47 and
`feasible_x0_extreme_row` at 33.

`scaled_feasible_a` at the default budget is now an honest budget exhaustion
carrying the right objective (`0`, `final_kkt_error` 2.97e-07 against the
`4.57e-03` it used to certify). Reaching a point this solver will genuinely
certify takes it **~3596 iterations** and the default cap is 200; it certifies
at `max_iter=4000`. That cost was always being paid — #712 is what made it
visible instead of skipped.

`issue_689_direct_driver_scaled_feasible::the_default_route_reaches_the_same_optimum`
is updated to match: the objective assertion, which is the sharp half, still
holds on both models unchanged; the verdict is asserted per model, with the
budget `_a` actually needs recorded so a future "it certifies at 200 again"
has to say whether it is a speed-up or this regressing.

**Open, and deliberately not folded in:** why the model needs 3596 iterations
at all. That is a convergence-rate question on a least-squares QP whose optimum
is a near-total cancellation, and it is what #690's adaptive-τ study kept
tripping over. It wants its own measurement.
