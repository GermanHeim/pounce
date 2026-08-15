# issue #592 — `Solve_Succeeded` at a point POUNCE immediately improves on restart

[#592](https://github.com/jkitchin/pounce/issues/592): on a fixed-policy NLP
from LyoPRONTO's Problem 2 GDP, POUNCE returns `Solve_Succeeded`, and
restarting POUNCE from the returned primal point improves the objective twice —
by **25.096 s (0.079 %)** in total — landing on the point IPOPT 3.14.16 reaches
in one solve. Reported against 0.10.0.

**Two defects were found and fixed.** Neither is in the convergence test; both
are in the inertia-correction path, and together they were sending this model
down a trajectory IPOPT never takes. With both fixed, the first solve reaches
`31785.744274` in 27 iterations — the reporter's IPOPT answer — and there is
nothing left for a restart to improve.

The first half of this note (through "Why the certificate is nonetheless
legal") records the investigation as it ran, because that analysis is still
correct and explains why the obvious fix — tightening the exit test — is the
wrong one. The second half records the two defects.

---

## Reproduction, reduced

The issue's recipe needs LyoPRONTO, GDPopt and glpk. It reduces to two `.nl`
captures: hook `call_after_subproblem_solve`, write the model out, and the
whole thing replays in 0.09 s.

    pounce 0_gdpopt.nl -AMPL tol=1e-5 acceptable_tol=1e-3 print_level=5

* `0_gdpopt.nl` — GDPopt's **first** subproblem, from its own starting point.
  This is the solve that produced the certificate the issue complains about.
* `8_restart1.nl` — the same model started from the point that solve returned.
  Solving it *is* the issue's "restart 1".

**113 variables, 155 constraints.** Before the fix, both reproduce every digit
the issue reports:

| solve | objective | status |
|---|---|---|
| `0_gdpopt.nl` (GDPopt's POUNCE result) | 31810.840250047 | `SolveSucceeded` |
| restart 1 = `8_restart1.nl` | 31803.787989278 | `SolveSucceeded` |
| restart 2 | 31785.744273619 | `SolveSucceeded` |
| restart 3 | 31785.744273619 | `SolveSucceeded` (fixed point) |
| IPOPT 3.14.16 from the capture (reporter) | 31785.744272683 | `Optimal Solution Found` |

The fixture is **not vendored**: LyoPRONTO is GPL-3.0 and POUNCE is EPL-2.0, so
a `.nl` encoding their model equations is not ours to add. The reporter has
since published it themselves, with provenance and a SHA-256, at
<https://gist.github.com/bernalde/19ae04607cb953fe92c3077902205dd3> — under
GPL-3.0-or-later, and explicitly *not* as permission to vendor it here. We
reference it by link only. See "Regression coverage" below for what stands in
for it.

## The returned point, measured

`pounce verify` — which re-derives everything from the model and does not trust
the `.sol` — rejects the point POUNCE certified:

| | premature exit (`SolveSucceeded`) | true optimum (`tol=1e-8`) |
|---|---|---|
| objective | 31803.788 | 31785.744 |
| max constraint violation | 2.844e-6 | 6.925e-7 |
| KKT stationarity residual | 9.994e-3 | 9.104e-6 |
| row complementarity ‖λ‖·slack | 3.273 | 1.869e-5 |
| bound complementarity | 9.091e-7 | 2.506e-9 |

## Why the certificate is nonetheless legal

At the exit, `‖λ‖∞ = 4.09e6` and mean `|λ| = 2.13e5` over 155 rows. That drives
the optimality-error scaling `s_d ≈ 1011`, so the aggregate divides the dual
infeasibility by a thousand:

```text
                       (scaled)     (unscaled)
Dual infeasibility:    9.994e-3     9.994e-3
Constraint violation:  2.358e-12    2.844e-6
Complementarity:       9.091e-7     9.091e-7
Overall NLP error:     9.887e-6     9.994e-3      <- 9.994e-3 / 1011
```

`9.887e-6 <= tol = 1e-5`, and all three of IPOPT's *unscaled component* gates
pass at their defaults too — dual `9.994e-3 <= dual_inf_tol = 1`, violation
`2.844e-6 <= constr_viol_tol = 1e-4`, complementarity
`9.091e-7 <= compl_inf_tol = 1e-4`. **IPOPT's convergence test would certify
this same point.** The exit test is not the defect.

### The complementarity subtlety

`verify` reports row complementarity 3.273 where the solver reports 9.091e-7,
and both are right. The solver's quantity is `z·s` on the *internal* slack; the
verifier recomputes the row value from `x`. They differ by the row residual, and
at `|λ| ~ 4e6` a residual of 2.8e-6 is worth ~11 s of Lagrangian — the same
order as the 18 s the restart recovers.

That is the cleanest statement of why the *symptom* is not itself a bug:

> The feasibility and stationarity tolerances are **absolute**, but their
> consequence for the objective scales with the **multipliers**. At
> `‖λ‖∞ ≈ 4e6`, `constr_viol_tol = 1e-4` licenses an objective error of order
> `4e2` seconds. The certified point is off by 18 s, entirely consistent with
> its 2.8e-6 residual. Neither solver's default tolerance set constrains this
> model's answer to better than tens of seconds.

So restart idempotence is not something these tolerances promise at this
conditioning, and changing the convergence test so this point fails would
diverge parity on every badly-scaled model in the corpus. What needed
explaining — and what turned out to be broken — is why POUNCE's *trajectory*
lands on the bad point and IPOPT's does not.

---

## Defect 1 — the inertia-trust floor was dimension-blind

`feral_inertia_pivot_floor` was introduced in
[#540](https://github.com/jkitchin/pounce/issues/540) (see
`issue-540-eigena2-inertia-noise.md`). When the factorisation reports an
inertia count that disagrees with the expected one *and* the smallest pivot of
the equilibrated matrix is below the floor, the count is treated as noise and
the system is reported `Singular` rather than `WrongInertia` — which routes to
the `δ_c` branch of the perturbation handler instead of climbing the `δ_x`
ladder.

The floor's own rationale is a backward-error argument: an equilibrated pivot
loses its sign at roughly `n · eps`. But the value shipped was the **constant**
`1e-12`, which is `n · eps` at `n ≈ 4500`. The KKT systems in this issue are
order 165–311, where `n · eps` is 3.7e-14 … 6.9e-14 — the constant was two
decades too generous, convicting pivots that were still measurable at the
solver's own scale.

**Fix:** `feral_inertia_pivot_floor` now defaults to `None`, which selects the
dimension-aware floor `n * f64::EPSILON` (`pounce_feral::inertia_trust_floor`).
Setting the option explicitly still pins an absolute floor for every dimension,
and `0` still disables the trigger entirely, so #540's opt-out is unchanged.
This is a breaking API change on `FeralConfig::inertia_pivot_floor`
(`f64` → `Option<f64>`).

Sweep over the 57-fixture CLI corpus: exactly two models moved, both the #540
models, both improved (eigena2 27 → 26 iterations, eigenb2 68 → 67), zero
status changes, the other 55 byte-identical. `8_restart1.nl` went 41
iterations / 31803.786 / `Acceptable` → 27 iterations / 31785.744271776 /
`SolveSucceeded`.

**This alone did not close the issue.** With only the floor fixed, the
end-to-end symptom was unchanged — the improvement available on restart was
still 25.096 s in total; the floor fix had merely merged two restarts into one.

## Defect 2 — `δ_c` was spent on a full-rank Jacobian and never withdrawn

On `0_gdpopt.nl` iteration 11 the #540 trigger fires *legitimately*
(`min_piv = 3.44e-14 < 6.9e-14`), so the handler raises `δ_c`. And `δ_c`
works, in the narrow sense: the next factorisation's smallest pivot is a
healthy 4.9e-10. But the inertia count is *still* wrong — because the Jacobian
was not rank-deficient in the first place; the small pivot came from the
Hessian block. `δ_c` was the wrong medicine.

The handler then does the one thing it must not: it **keeps** `δ_c` and starts
climbing the `δ_x` ladder on top of it. Four rungs later `δ_w = 1e2` (IPOPT
accepts this system at `δ_w = 1e-4`). The resulting step is so over-damped that
the objective is frozen for eight iterations, after which the loose-tolerance
exit fires at 31810.840 — the certificate the issue reports.

Things that were tried and do not discriminate:

* **A different floor value.** Sweeping 6.9e-14 / 3e-14 / 1e-14 / 5e-15 /
  1e-15 / 0 across both instances is wildly non-monotone; only `floor = 0`
  reaches the optimum on both, and that is just #540 regressed. No floor value
  separates the two populations.
* **The inertia count after `δ_c`.** eigena2's post-`δ_c` counts are also
  wrong (60/61 against an expected 55), so "count still wrong ⇒ `δ_c` was
  wrong" convicts the case #540 exists to serve.
* **Which block owns the smallest pivot.** This is the discriminator one
  actually wants, and it needs a `min_pivot_index` from feral. feral is an
  external crates.io dependency here, not a path dependency, so its API could
  not be extended in this change.

What does separate them cleanly is **how many `δ_x` rungs get climbed while
`δ_c` is up**:

| model | rungs-under-`δ_c` histogram |
|---|---|
| eigena2 (#540) | `{1: 4}` |
| eigenb2 (#540) | `{0: 5, 1: 1}` |
| `0_gdpopt.nl` (#592) | `{0: 6, 1: 4, 4: 1}` |
| `pooling_rt2stp` | `{…, 3: 3, 5: 1, 6: 1, 8: 1, 14: 1}` |

When `δ_c` is the right medicine it is *immediately* right — never more than
one rung. When it is the wrong medicine the ladder climbs without limit.

**Fix:** a `δ_c` walk-back. If the `δ_x` ladder has climbed
`perturb_delta_c_max_rungs` rungs (default `3`) while `δ_c` is up, `δ_c` was
not what this system needed: all four deltas are withdrawn to zero, the
degeneracy probe is reset to `NoTest`, and `δ_c` is latched off for the rest of
this iterate so the `Singular` branch cannot raise it again. `w` is appended to
the iteration line's info string when this fires. The latch clears on the next
`consider_new_system`. `perturb_delta_c_max_rungs = 0` disables the walk-back
and restores the previous behaviour exactly.

## Result

| model | before | after |
|---|---|---|
| `0_gdpopt.nl` (the reported first solve) | 19 it, 31810.840250, `SolveSucceeded` | **27 it, 31785.744274, `SolveSucceeded`** |
| `8_restart1.nl` (restart 1) | 41 it, 31803.788, `Acceptable` | **14 it, 31785.744276, `SolveSucceeded`** |
| IPOPT 3.14.19 reference | 23 it, 31785.744272683 | — |
| eigena2 (#540) | 27 it, `Optimal` | 26 it, `Optimal` |
| eigenb2 (#540) | 68 it, `Optimal` | 67 it, `Optimal` |
| `pooling_rt2stp` | 812 it | **298 it** (pre-#544 was 206) |
| `unbounded_exp` | 32 it, `ErrorInStepComputation` | 27 it, same status |
| other 53 CLI fixtures | — | byte-identical |

The first solve now lands on the reporter's IPOPT point, so the restart has
nothing to improve and the reported non-idempotence is gone. The `pooling`
number is the incidental one worth flagging: #544 had cost that model 812
iterations against a pre-#544 206, and the walk-back returns most of it.

Opt-outs verified: `perturb_delta_c_max_rungs=0` reproduces `0_gdpopt.nl` at
exactly 19 iterations / 31810.840250, and `feral_inertia_pivot_floor=0` still
fully disables the #540 trigger.

### The cold GDP path

The table above is measured on the two captured `.nl` files. The reporter
[pointed out](https://github.com/jkitchin/pounce/issues/592#issuecomment-5298835858)
that the stricter criterion is the **original cold GDP pipeline** — GDPopt
driving POUNCE from LyoPRONTO's own starting point — and that every
*option-level* workaround that repairs the captured restart fails there: with
`nlp_scaling_method=none` or `nlp_scaling_max_gradient=1e8` the cold solve is
slightly *worse* than the default, not better.

That is a useful negative result in its own right: it rules out gradient
scaling as the root cause, which the captured-restart evidence had made look
plausible. The fix here is not an option setting — it is a default-behaviour
change in the inertia-correction path — so it acts on the cold solve directly.
Running the reporter's pipeline in one environment, released 0.10.0 against
this branch, and reporting the two phase-switch times the downstream model is
judged on:

| | objective (s) | switch 1 (h) | switch 2 (h) |
|---|---|---|---|
| **0.10.0 cold** | 31810.840250047 | 1.925104404813 | 3.924408024351 |
| 0.10.0 restart 1 | 31803.786223117 | 1.753942060567 | 3.922548330275 |
| 0.10.0 restart 2 | 31785.744273619 | 1.575762164995 | 3.917595808629 |
| **this branch, cold** | **31785.744273619** | **1.575762164995** | **3.917595808629** |
| this branch, restart 1 | 31785.744273619 | 1.575762164995 | 3.917595808629 |
| this branch, restart 2 | 31785.744273619 | 1.575762164995 | 3.917595808629 |
| IPOPT reference | 31785.744272683 | 1.575762157937 | 3.917595808370 |

The 0.10.0 row reproduces the reporter's own numbers to every digit they
quoted, which is what makes the comparison trustworthy. The cold solve now
reaches the IPOPT switch times on the first attempt, and restarts 1 and 2 are
identical to it to twelve digits. The residual disagreement with IPOPT is
7e-9 h in switch 1 — the same order that separates the reporter's `tol=1e-6`
and `barrier_tol_factor=1000` rows from each other, i.e. inside the
tolerance-limited band described above. The 0.35 h error is gone.

## Regression coverage

Because the reproducer cannot be vendored (GPL-3.0 model into an EPL-2.0
repository — the reporter's published gist is explicit that sharing it is not
permission to vendor), the behaviour is pinned three ways instead:

* `crates/pounce-common/src/pd_perturbation.rs` — four unit tests on the
  walk-back state machine: it fires after the configured rungs, a withdrawn
  `δ_c` is not raised again until the next iterate, one rung leaves `δ_c`
  alone, and `0` disables it.
* `crates/pounce-feral/src/lib.rs` — three unit tests on the floor: the
  dimension-aware default, an explicitly pinned floor, and an N=400 system
  whose trailing 2×2 block produces a pivot that *is* measurable at `n · eps`
  and must not be convicted. (Note when writing these: `min_pivot_magnitude`
  is measured on the **equilibrated** matrix, so a bare small diagonal entry is
  lifted back to 1 by scaling — only a near-singular *block* survives.)
* `crates/pounce-cli/tests/issue_592_delta_c_walkback.rs` — three end-to-end
  tests on the already-vendored `pooling_rt2stp` fixture, which exhibits the
  same wrong-medicine pattern: the walk-back removes the #544 detour, disabling
  it restores the long run, and both routes reach the same objective.

## What is still open

* **The multiplier-magnitude diagnostic.** The tolerance analysis above stands
  on its own regardless of this fix: at `‖λ‖∞ ≈ 4e6` the default tolerances
  determine this model's answer only to ±10 s, and nothing tells the user so.
  POUNCE already has "masked certificate" machinery for the *objective-scale*
  channel (`obj_scale_certificate_threshold`); the multiplier-magnitude channel
  is the same phenomenon and is unguarded. When `‖λ‖∞ · max_constr_viol` is a
  non-trivial fraction of `|f|`, the run could say the certificate is
  tolerance-limited — in the console summary and the JSON report — without
  changing the status or breaking parity.
* **`dual_inf_tol` is silently inert on models like this one.** Tightening it
  to `1e-3` or `1e-2` changed nothing, because `dual_inf_scale_kappa` raises
  the effective floor above it. That is documented behaviour
  (`docs/src/options.md`, "the floor is a floor"), but the documentation's
  inertness argument — the floor does not rise above `dual_inf_tol` until
  `dual_scale` exceeds `dual_inf_tol/tol = 1e8` — is computed at the *default*
  `tol = 1e-8`. At `tol = 1e-5` the crossover is `dual_scale > 1e5`, a thousand
  times easier to reach. The docs should say so.
* **`min_pivot_index` in feral.** Knowing which block owns the smallest pivot
  would let the handler pick `δ_c` versus `δ_x` directly, instead of raising
  `δ_c` speculatively and walking it back. The walk-back is a three-iteration
  recovery from a wrong guess; the index would avoid the guess.
* **A synthetic reproducer for the premature certificate** (independent of the
  trajectory fix). One was attempted and did not work: 8-variable NLPs with an
  objective of order 1e4 and equality rows carrying a near-cancelling
  `1e6..1e8` pair alongside O(1) nonlinear terms — heterogeneous *within* the
  row, which row scaling cannot repair the way it repairs a uniform row
  multiplier. 60 randomised instances, each solved at `tol=1e-5` and
  `tol=1e-8`, produced **no** instance where both report `Solve_Succeeded` and
  the loose solve is measurably worse. Ill-conditioning alone is not enough;
  the real model also brought the degenerate active set and the frozen-iterate
  corner (`lg(rg) > 10`, `‖d‖ ~ 1e-11`) that the synthetic instances never
  entered.
