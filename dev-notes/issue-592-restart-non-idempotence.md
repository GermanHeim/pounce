# issue #592 — `Solve_Succeeded` at a point POUNCE immediately improves on restart

[#592](https://github.com/jkitchin/pounce/issues/592): on a fixed-policy NLP
from LyoPRONTO's Problem 2 GDP, POUNCE returns `Solve_Succeeded`, and
restarting POUNCE from the returned primal point improves the objective twice —
by **25.096 s (0.079 %)** in total — landing on the point IPOPT 3.14.16 reaches
in one solve. Reported against 0.10.0; reproduces on `main` at the time of
writing.

This note records the investigation. **No fix landed.** The short version is
that the exit is legal under every tolerance the run configured, including
IPOPT's own component gates, so "tighten the convergence test" is a parity
break rather than a fix; what actually differs from IPOPT is the *trajectory*,
and localising that needs an IPOPT oracle the investigation did not have.

---

## Reproduction, reduced

The issue's recipe needs LyoPRONTO, GDPopt and glpk. It reduces to a single
`.nl`: capture GDPopt's first subproblem (`call_after_subproblem_solve`), write
it out, and the whole thing replays in 0.09 s.

    pounce sub592.nl -AMPL tol=1e-5 acceptable_tol=1e-3 print_level=5

**113 variables, 155 constraints.** The captured model starts at the point
GDPopt's POUNCE solve left it (objective 31810.840), so solving it *is* the
issue's "restart 1", and it reproduces every digit the issue reports:

| solve | objective | status |
|---|---:|---|
| capture (GDPopt's POUNCE result) | 31810.840250047 | `SolveSucceeded` |
| restart 1 = `sub592.nl` | 31803.787989278 | `SolveSucceeded` |
| restart 2 | 31785.744273619 | `SolveSucceeded` |
| restart 3 | 31785.744273619 | `SolveSucceeded` (fixed point) |
| IPOPT 3.14.16 from the capture (reporter) | 31785.744272683 | `Optimal Solution Found` |

The fixture is **not vendored**: LyoPRONTO is GPL-3.0 and POUNCE is EPL-2.0, so
a `.nl` encoding their model equations is not ours to add. See "Fixture" below.

## The returned point, measured

`pounce verify` — which re-derives everything from the model and does not trust
the `.sol` — rejects the point POUNCE certified:

| | premature exit (`SolveSucceeded`) | true optimum (`tol=1e-8`) |
|---|---:|---:|
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

That is the cleanest statement of the whole issue:

> The feasibility and stationarity tolerances are **absolute**, but their
> consequence for the objective scales with the **multipliers**. At
> `‖λ‖∞ ≈ 4e6`, `constr_viol_tol = 1e-4` licenses an objective error of order
> `4e2` seconds. The certified point is off by 18 s, entirely consistent with
> its 2.8e-6 residual. Neither solver's default tolerance set constrains this
> model's answer to better than tens of seconds.

So restart idempotence is not something these tolerances promise at this
conditioning. What still needs explaining is why IPOPT's trajectory lands on the
good point and POUNCE's does not.

## Where the conditioning comes from

Default gradient-based scaling (`nlp_scaling_max_gradient = 100`) assigns row
scale factors ~1e-6, which is what inflates the multipliers and `s_d`. Every
configuration that removes or loosens that reaches the true optimum, and does so
in *fewer* iterations than the failing run:

| configuration | objective | iters |
|---|---:|---:|
| default (`tol=1e-5`) | 31803.788 | 30 |
| `tol=1e-4` | 31802.803 | 68 |
| `tol=1e-6` | 31785.744 | 43 |
| `tol=1e-8` | 31785.744 | 29 |
| `mu_strategy=adaptive` | 31785.744 | 31 |
| `barrier_tol_factor=1000` | 31785.744 | 14 |
| `nlp_scaling_method=none` | 31785.744 | **17** |
| `nlp_scaling_max_gradient=1e8` | 31785.744 | **17** |

The failing trajectory is visibly pathological from iteration 9: `inf_du` spikes
to 1.9e9, regularisation climbs to `lg(rg) = 10.6`, and from iteration 25 the
iterate is frozen (`‖d‖ ≈ 1e-11`, objective constant to 8 digits) while the dual
infeasibility decays 2.34e6 → 9.99e-3 purely through dual updates. It exits the
moment that decay crosses `s_d · tol = 1.01e-2`.

## Hypotheses tested and eliminated

* **μ collapses 3.5 decades in one iteration** (iteration 17→18, `lg(mu)`
  −2.5 → −6.0, immediately preceding the blow-up). Legal: upstream's
  `MonotoneMuUpdate::UpdateBarrierParameter` loops the reduction while
  `sub_problem_error <= kappa_eps_mu`, and `mu_allow_fast_monotone_decrease`
  defaults to `yes`. POUNCE implements the cap and the same loop. The floor μ
  lands on (9.09e-7 = `min(tol, compl_inf_tol)/(barrier_tol_factor+1)`) is
  upstream's `CalcNewMuAndTau` floor.
* **`dual_inf_scale_kappa` lets the dual gate through.** It does raise the
  strict dual bar here (to `kappa·tol·dual_scale`, well above
  `dual_inf_tol = 1`), but the unscaled dual infeasibility is 9.994e-3 and would
  clear even the bare `1`. Not what admits the exit. *Worth noting separately:*
  a user tightening `dual_inf_tol` on this model is silently overridden —
  `dual_inf_tol=1e-3` and `=1e-2` both changed nothing. That is documented
  behaviour (`docs/src/options.md`, "the floor is a floor"), but the
  documentation's inertness argument — "the floor does not rise above
  `dual_inf_tol` until `dual_scale` exceeds `dual_inf_tol/tol = 1e8`" — is
  computed at the *default* `tol = 1e-8`. At `tol = 1e-5` the crossover is
  `dual_scale > 1e5`, a thousand times easier to reach. The docs should say so.

## What is left

The open question is the trajectory: why POUNCE goes through the iteration-9–23
blow-up on this model when IPOPT does not. Answering it needs a side-by-side
IPOPT run on the identical evaluator (the `cyipopt`-on-the-same-`.nl` pattern
used for #257 / #266), which was not available in the investigating
environment — no IPOPT binary, Ubuntu ships library-only 3.11.9, and `cyipopt`
will not build without a pkg-config'd Ipopt.

Two candidate directions for whoever picks this up, in preference order:

1. **Trajectory parity.** Diff the two iteration logs from the same start.
   First suspects, in order of how early they act: the gradient-based scaling
   factors themselves (compare against `IpGradientScaling.cpp`'s
   `min(1, max_gradient/‖∇g_i‖∞)` row by row — if these already differ, nothing
   downstream is comparable), then the inertia-correction / regularisation path
   that reaches `lg(rg) = 10.6`, then the second-order corrections (the `H`
   flags at iterations 3, 6–10).
2. **A tolerance-limited diagnostic, not a status change.** POUNCE already has
   "masked certificate" machinery for the *objective-scale* channel
   (`obj_scale_certificate_threshold`). The multiplier-magnitude channel is the
   same phenomenon and is unguarded: when `‖λ‖∞ · max_constr_viol` is a
   non-trivial fraction of `|f|`, the certificate is tolerance-limited and the
   run could say so — in the console summary and the JSON report — without
   changing the status or breaking parity. This would have told the reporter
   immediately that the answer was determined only to ±10 s.

Changing the convergence test so this point fails is **not** recommended: it
passes IPOPT's own gates, so it would diverge parity on every badly-scaled model
in the corpus, not just this one.

## Fixture

The reporter offered the captured `.nl`. Before it can be vendored, the licence
has to be settled: LyoPRONTO is GPL-3.0, POUNCE is EPL-2.0, and a `.nl` encodes
the model's equations. Either the reporter contributes it under EPL-2.0-
compatible terms, or the regression needs a synthetic model.

**A synthetic reproducer was attempted and did not work.** The construction
targeted the mechanism directly: 8-variable NLPs with an objective on the order
of 1e4 and equality rows carrying a near-cancelling `1e6..1e8` pair alongside
O(1) nonlinear terms — heterogeneous *within* the row, which row scaling cannot
repair the way it repairs a uniform row multiplier. 60 randomised instances,
each solved at `tol=1e-5` and `tol=1e-8`, produced **no** instance where both
report `Solve_Succeeded` and the loose solve is measurably worse. Ill-
conditioning alone is evidently not enough; the real model also brings a
degenerate active set and the frozen-iterate corner (`lg(rg) > 10`, `‖d‖ ~
1e-11`) that the synthetic instances never entered. A generator aimed at *that*
corner, rather than at the conditioning that precedes it, is the more promising
second attempt.
