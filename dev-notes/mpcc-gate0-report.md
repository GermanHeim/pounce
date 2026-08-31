# MPCC Gate 0 — supported route and failure boundary

> The gate report gh#794 asks for. It states which POUNCE route is
> supported for small MPCCs, where that route's boundary is, and which
> observed gaps belong to POUNCE, to the modelling layer, or to nobody
> yet. It is the exit criterion for Gate 0 of gh#776 and the input to
> the decision about whether Gate 1 (a single flash with phase
> appearance) should begin.
>
> Harness: [`benchmarks/mpcc/`](../benchmarks/mpcc/README.md). Raw
> numbers: `benchmarks/mpcc/results-full.{json,md}`, regenerable with
> `make -C benchmarks mpcc-run`.

## Provenance of the run this report is written from

| field | value |
|---|---|
| pounce commit | `2b5e4cb` (clean) |
| model-data revision | `689e154931c00735` |
| pounce version | 0.10.0 (Python extension, release build) |
| discopt | absent — a cross-repository *design* dependency of gh#776, not a runtime dependency of this harness. No DiscOpt comparison was run. |
| CCOpt | absent. Optional per gh#794; the pin a comparison would have used is `ccopt==0.4.1`. |
| matrix | 11 cases × 2 scaling legs × 3 starts × 9 routes × 5 controls = 2880 records, 244 s |
| pinned options | `tol=1e-8`, `max_iter=300`, `bound_relax_factor=0`, `honor_original_bounds=yes` |
| linear algebra | identical across all nine routes — one POUNCE build, one process, the default linear solver, no per-route backend selection. No comparison below crosses a linear-algebra boundary. |
| relaxation schedule | `tau = 1e0 … 1e-8`, ×0.1, one bisection allowed after a rejected stage |

`bound_relax_factor=0` is not a detail. Its default of 1e-8 relaxes every
constraint bound before the solve, which on this corpus means accepting
`G >= -1e-8` — a source-level sign violation of the same order as the
complementarity products the whole report is trying to measure.

## The decision

**Supported route: `scholtes_then_ncp`** — Scholtes continuation with
full primal/dual/barrier warm starts to locate the branch, then **one
exact-product NCP-equality solve seeded from it**. It is the only route
in the comparison that solves every cell, and the only one that does so
while never returning a point below the MPCC's optimum.

Neither half is sufficient alone, and the reasons are different:

* the continuation always converges, but its answer is only ever
  feasible for `G*H <= tau` — on `ralph1` and `ralph2` and `scholtes4`
  that means an objective *below* the true optimum, at complementarity
  pinned at the schedule floor;
* the exact-product solve returns an MPCC-feasible point, but from a
  cold start it fails on the cases where a pair is biactive at the
  solution (finding P2).

Run in sequence they cover each other. That is a measurement, not a
hope: the composition clears all eight cells that P2 breaks, and returns
complementarity at `1.9e-12` on `qpec_small/origin` and `5.7e-28` on
`ralph1` where the continuation alone leaves `1e-08`.

### Route comparison

Cells are `(case, scaling, start)`, 60 per route with `infeasible_pair`
excluded and reported separately. `at f*` counts the cells that reached
the global optimum the branch-enumeration oracle knows independently;
`other local` counts cells that reached a different genuine local
solution, which is expected of a local solver; `below f*` counts cells
that returned an objective *lower* than the optimum, possible only at a
point the source model does not admit.

| route | solved | at f* | other local | below f* | not solved | median iters | worst source compl |
|---|---:|---:|---:|---:|---:|---:|---|
| `direct` | 58 | 47 | 9 | 2 | 2 | 43 | 1.8e-11 |
| `ncp_eq` | 58 | 47 | 11 | 0 | 2 | 20 | 9.7e-10 |
| `ncp_eq_l1` | 60 | 44 | 4 | 12 | 0 | 18 | 9.8e-09 |
| `ncp_eq_l1_fallback` | 58 | 41 | 11 | 6 | 2 | 18 | 9.7e-10 |
| `ncp_eq_auto_l1` | 58 | 47 | 11 | 0 | 2 | 20 | 9.7e-10 |
| `scholtes_cold` | 60 | 40 | 8 | 12 | 0 | 164 | 1.0e-08 |
| `scholtes_warm_primal` | 60 | 48 | 0 | 12 | 0 | 60 | 1.0e-08 |
| `scholtes_warm_full` | 60 | 48 | 0 | 12 | 0 | 49 | 1.0e-08 |
| **`scholtes_then_ncp`** | **60** | **57** | **3** | **0** | **0** | **79** | **4.9e-11** |

The composition wins on every axis at once: it solves 60 of 60 where the
exact-product routes solve 58, reaches the global optimum in 57 where
nothing else exceeds 48, never returns a point below the optimum where
every continuation returns 12, and leaves complementarity three orders
tighter than the schedule floor. It costs about 4× the iterations of a
single `ncp_eq` solve. That is the trade, and on this corpus it is not
close.

### Per case, `at f* / other local / below f* / not solved`

| case | direct | ncp_eq | l1 | l1_fb | auto_l1 | s_cold | s_prim | s_full | **s+ncp** |
|---|---|---|---|---|---|---|---|---|---|
| `regular_strict` | 3/3/0/0 | 3/3/0/0 | 4/2/0/0 | 3/3/0/0 | 3/3/0/0 | 3/3/0/0 | 6/0/0/0 | 6/0/0/0 | **6/0/0/0** |
| `biactive_positive` | 6/0/0/0 | 5/1/0/0 | 6/0/0/0 | 5/1/0/0 | 5/1/0/0 | 6/0/0/0 | 6/0/0/0 | 6/0/0/0 | **6/0/0/0** |
| `ralph1` | 4/0/1/1 | 6/0/0/0 | 0/0/6/0 | 6/0/0/0 | 6/0/0/0 | 0/0/6/0 | 0/0/6/0 | 0/0/6/0 | **3/3/0/0** |
| `ctrap` | 4/2/0/0 | 6/0/0/0 | 6/0/0/0 | 6/0/0/0 | 6/0/0/0 | 5/1/0/0 | 6/0/0/0 | 6/0/0/0 | **6/0/0/0** |
| `infeasible_pair` | 0/0/0/4 | 0/0/0/4 | 0/0/0/4 | 0/0/0/4 | 0/0/0/4 | 0/0/0/4 | 0/0/0/4 | 0/0/0/4 | **0/0/0/4** |
| `selector_theta_025` | 4/2/0/0 | 3/3/0/0 | 5/1/0/0 | 3/3/0/0 | 3/3/0/0 | 4/2/0/0 | 6/0/0/0 | 6/0/0/0 | **6/0/0/0** |
| `selector_theta_050` | 6/0/0/0 | 6/0/0/0 | 6/0/0/0 | 6/0/0/0 | 6/0/0/0 | 6/0/0/0 | 6/0/0/0 | 6/0/0/0 | **6/0/0/0** |
| `selector_theta_075` | 4/2/0/0 | 4/2/0/0 | 5/1/0/0 | 4/2/0/0 | 4/2/0/0 | 4/2/0/0 | 6/0/0/0 | 6/0/0/0 | **6/0/0/0** |
| `ralph2` | 6/0/0/0 | 4/2/0/0 | 6/0/0/0 | 4/2/0/0 | 4/2/0/0 | 6/0/0/0 | 6/0/0/0 | 6/0/0/0 | **6/0/0/0** |
| `scholtes4` | 5/0/1/0 | 6/0/0/0 | 0/0/6/0 | 0/0/6/0 | 6/0/0/0 | 0/0/6/0 | 0/0/6/0 | 0/0/6/0 | **6/0/0/0** |
| `qpec_small` | 5/0/0/1 | 4/0/0/2 | 6/0/0/0 | 4/0/0/2 | 4/0/0/2 | 6/0/0/0 | 6/0/0/0 | 6/0/0/0 | **6/0/0/0** |

Two rows are worth reading closely. On `scholtes4` — whose solution is
M-stationary and not S-stationary — every continuation and every ℓ₁
route lands below `f*`, and only the routes that finish on an exact
product reach it. On `regular_strict` the composition finds the *global*
optimum from all six starts, including the two that begin on the other
branch, where every single-solve route stops at the local `f = 4`: the
relaxation's interior lets it cross branches, which no exact-product
solve from a cold start on the wrong axis can do.

## The complementarity tolerance floor

The single most important number Gate 1 inherits, and it is not a solver
property.

`G*H` is quadratically flat at the corner. A solve that drives the
product to `eps` therefore pins each side of the pair only to
`sqrt(eps)`, and the objective follows that excursion. At the default
`tol = 1e-8` an MPCC objective is good to about `1e-4`, no matter which
route produced it:

| case | route | source complementarity | objective below f* | `sqrt(residual)` |
|---|---|---|---|---|
| `ralph1` | `ncp_eq_l1` | 2.6e-09 | 5.07e-05 | 5.1e-05 |
| `scholtes4` | `ncp_eq_l1` | 1.9e-10 | 2.2e-05 | 1.4e-05 |
| `ralph2` | `ncp_eq_l1` | 2.1e-10 | 4.3e-10 | — (linear here: this case's relaxed optimum is `-2 tau`) |

The same effect governs the *stationarity class*. At a residual of
`eps` the recovered MPCC multipliers on a biactive pair are themselves
`O(sqrt(eps))`, and S- and C-stationarity differ only in their signs —
so at `nu = w = -2.9e-05` against a corner band of `1e-04`, the class is
not resolved by the data. 61 of the 512 control-free observations carry
this label; none of them is a defect and none of them is hidden.

Two consequences for Gate 1. A phase state read off a complementarity
pair is determined only to `sqrt(tol)`, so a regime decision taken on a
margin smaller than that is arithmetic, not physics. And an objective
comparison against a reference — a published trajectory, a DiscOpt GDP
solution — cannot be quoted tighter than `sqrt(tol)` without tightening
`tol` first.

## The documented failure boundary

1. **A continuation alone never returns an MPCC-feasible point.** Its
   answer carries complementarity at the schedule floor by construction:
   `1.0e-08` = `tau_min` in every continuation cell that ran the
   schedule out, and on `ralph2` exactly `-2 tau_min` in the objective.
   This is the method behaving as designed, and it is why the supported
   route finishes on an exact product. 126 of the 576 triaged
   observations carry this label.

2. **Below `tau = 1e-8` is untested**, and so is any schedule shape
   other than the geometric one here.

3. **Where no S-stationary point exists, no route can certify one.** On
   `ralph1` MPCC-LICQ fails and the multiplier system admits no
   nonnegative pair; on `scholtes4` the multiplier set is a line that
   never enters the nonnegative orthant. The best available certificate
   is M. **The stationarity class, not the NLP status, is the thing to
   read.**

4. **The biactive cases lean on acceptable-level termination.** The
   `no_acceptable` control (`acceptable_iter=0`) flips **9** verdicts,
   every one a `ncp_eq`-family solve on `biactive_positive` or
   `scholtes4` that succeeded only as `Solved_To_Acceptable_Level` and,
   with the criterion off, ends in `Error_In_Step_Computation`. A Gate 1
   model with biactive phase pairs should expect this and should not be
   tuned with `acceptable_iter=0`.

5. **Scaling is load-bearing, and the scaled KKT error is not the one to
   read.** The `skew` leg spans six orders across the variables — mild
   next to a real flash — and moves **33 of 288** `(case, start, route)`
   cells. The `no_scaling` control moves 24 objectives and 10 verdicts.
   Fourteen observations triage as `scaling`: POUNCE converged the
   problem it internally scaled while its own `final_unscaled_kkt_error`
   is three or more orders larger, and the MPCC stationarity residual,
   measured in the user's units, agrees with the unscaled figure. POUNCE
   reports both (gh#173); a Gate 1 harness that reads only the scaled
   one will believe a point it should not.

6. **A biactive pair breaks a cold exact-product solve** — finding P2,
   below. Covered by the supported route, not by any single solve.

7. **A POUNCE-only convergence mechanism is load-bearing in 3 cells.**
   `upstream_heuristics` (`acceptable_progress_kappa`,
   `dual_inf_scale_kappa`, `obj_scale_certificate_threshold` all zeroed,
   which each option's own documentation calls bit-for-bit upstream
   behaviour) turns `qpec_small/skew/upper_left` from `Solve_Succeeded`
   into `Error_In_Step_Computation` for three routes. Worth knowing
   before any of those three mechanisms is changed for an unrelated
   reason.

8. **Infeasibility is detected, on every route.** No route ran to
   completion on `infeasible_pair` in any of its 36 cells and none
   returned a success status. `direct` and `ncp_eq_l1` return
   `Infeasible_Problem_Detected`; the rest of the exact-product family
   returns `Not_Enough_Degrees_Of_Freedom`, which is structurally
   correct. The continuation accepts `tau = 1` and the bisected
   `tau = 0.316` and rejects `tau = 0.1` — the relaxation is feasible
   exactly for `tau >= 1/4`, so the observed crossover sits on the
   analytic one. That agreement is the strongest single check that the
   harness measures what it claims to.

9. **The corpus itself.** Every pair is affine, every objective and row
   quadratic, nothing exceeds three variables and two pairs, and there
   is no dynamics, thermodynamics or phase equilibrium anywhere. Nothing
   here bounds conditioning, fill, or trajectory cost at process-model
   size, and nothing here says anything about a *nonlinear*
   complementarity function — which a VLE pair will be.

## Warm starting across the continuation

Total inner iterations over the 60 non-infeasible cells, same stages,
same schedule, same stopping requirements:

| arm | inner iterations | vs cold |
|---|---:|---:|
| `scholtes_cold` (independent solves) | 11 174 | 1.0× |
| `scholtes_warm_primal` | 3 342 | 3.3× fewer |
| `scholtes_warm_full` (x, λ, z, μ) | 3 054 | 3.7× fewer |
| `scholtes_then_ncp` (adds the finishing solve) | 4 855 | 2.3× fewer |

The restart ladder was never needed on the continuation arms: 0 restarts
and 0 rejected stages across all three on every case except
`infeasible_pair`, where the rejection is the correct answer. It *is*
needed by the composition — 8 restarts, 8 rejected stages, all on the
finishing solve, which is precisely the exact-product fragility of
finding P2 being absorbed by the ladder rather than surfacing as a
failure.

Most of the primal-only arm's gain is already the whole gain; carrying
the duals and `mu` adds 9%. Worth stating plainly, because the
warm-start machinery's cost to a caller is dominated by capturing and
replaying the dual blocks correctly, and on this corpus that work buys
very little.

**The finishing solve takes the point and nothing else.** The relaxed
problem's duals are duals of a different constraint — the product row is
an active inequality at `G*H <= tau` and an equality at `G*H = 0` — and
seeding them measurably sent a three-variable finish into a thrash that
had not returned in 25 s, where the primal-only transfer converges in
six iterations.

## Ownership of the observed gaps

576 observations triaged mechanically (rules in
`benchmarks/mpcc/report.py`). **No observation is unassigned.**

| owner | observations | what it means |
|---|---:|---|
| converged, nothing to assign | 331 | reached an S- or M-stationary point |
| relaxation limit | 126 | a continuation's `tau`-feasible answer (boundary item 1) |
| complementarity tolerance floor | 61 | the `sqrt(tol)` section above |
| source formulation | 36 | `infeasible_pair`: failing is the correct answer |
| scaling | 14 | boundary item 5 |
| POUNCE candidate | 8 | finding P2 — since confirmed against Ipopt 3.14.19, which solves the reproducer POUNCE fails |

### Nothing is assigned to DiscOpt

DiscOpt is not a runtime dependency of this harness and none of the
lowerings compared here is DiscOpt's. What Gate 0 hands back to
[jkitchin/discopt#1123](https://github.com/jkitchin/discopt/issues/1123)
and [#1124](https://github.com/jkitchin/discopt/issues/1124) is a
requirement rather than a defect: **a lowering that reports only the
NLP's residuals is not reportable as an MPCC result.** The three
quantities that have to survive the lowering are the source
complementarity product, the sign residuals of `G, H >= 0` in the
model's own units, and enough of the source model to classify MPCC
stationarity. `benchmarks/mpcc/schema.json` is the concrete shape that
satisfies that, and it is deliberately solver-agnostic.

### P1 — `l1_exact_penalty_barrier` reported success on a violated constraint. **Fixed.**

The ℓ₁ wrapper solves the augmented NLP `c(x) − p + n = target`, whose
equality rows the slacks satisfy to machine precision by construction.
That residual was reported as the solve's constraint violation, and the
wrapper's exit verdict was argued from `Σ(p + n)` against its own
`l1_slack_tol`. On `ralph1`:

```
status  Solve_Succeeded
f       -5.0e-04          (the MPCC optimum is 0)
report  final_constr_viol = 9.6e-15
actual  |c(x) − target|   = 2.5e-07
```

`final_unscaled_constr_viol` said `9.6e-15` too, so unscaling did not
disclose it either — there was no field in the result a caller could
have read to notice.

Fixed in `run_l1_penalty_outer_loop` (this PR). The statistics now carry
the violation of the **user's own** rows and bounds at the returned
point, measured by `original_space_feasibility`, and both the ρ-escalation
loop's termination test and the exit verdict are judged by the
tolerances the caller set — `is_negligible` at `tol` and at
`acceptable_tol`, scale-relative — instead of by `Σ(p + n)` against
`l1_slack_tol`. `Σ(p + n)` is not the violation (that is `|p_i − n_i|`
per row, and at the barrier's interior both slacks stay positive where
their difference is zero) and `l1_slack_tol`'s `1e-6` is four orders
looser than a `tol = 1e-8` solve asked for.

The change is downgrade-only: a solve that reaches the strict standard
on the user's rows keeps the status the inner gave it, and nothing can
turn a failure into a success. Where the measurement is unavailable the
historical slack-sum test still applies, so "not measured" never reads
as "feasible".

After the fix, on the same reproducer: the wrapper escalates ρ instead
of stopping at the penalty point, `final_constr_viol` reads `2.6e-09`
and **equals the actual violation exactly**, and the objective improves
an order of magnitude to `-5.1e-05` — the remainder being the
complementarity tolerance floor above, not a hidden residual. Across the
corpus the reported violation now equals the source complementarity
product in every ℓ₁ cell, and `ralph1` under `ncp_eq_l1_fallback`
reaches the true optimum (`1.8e-09`, product exactly 0) where it used to
stop at `-4.3e-10`.

841 tests across `pounce-algorithm` and `pounce-l1penalty` pass,
including all 11 ℓ₁ integration tests. The change is confined to solves
that opt into the wrapper, so the CLI fixture corpus — which never sets
the option — is untouched by construction.

### P2 — a biactive pair breaks a cold exact-product solve. **Open: the exit is not what this section said, twice over.**

> **Status of this section.** It has now been rewritten three times, and
> the first two versions were wrong in opposite directions — first
> "inherent to the reformulation, nothing to fix", then "a POUNCE defect,
> upstream solves it". Both were drawn from too little measurement. What
> follows is confined to what has actually been run, and P2 is left
> **open pending re-measurement through the harness**, not resolved. The
> ownership bucket keeps its 8 cells and its `POUNCE candidate` label,
> because that label was always the mechanical triage's and never rested
> on either story.

Eight of the 576 observations, all with the same exit **through the
harness's Python path**. On `qpec_small` from `origin` and `upper_left`
at unit scaling, the `ncp_eq` family ends in
`Error_In_Step_Computation`; so does `direct` on
`qpec_small/skew/upper_right` and on `ralph1/unit/origin`. Both
exact-product lowerings are affected at similar rates, so it is not a
property of one route.

**"All eight are one exit" is a statement about that path, and the same
model exits differently through another.** Run as an `.nl` file through
the CLI, at the same options, `qpec_small/origin` exits
`Solved_To_Acceptable_Level` in 41 iterations, not
`Error_In_Step_Computation` in 118. Both runs use exact Hessians, so it
is not an L-BFGS difference; the sparsity does differ (the harness hands
back a dense lower triangle, 6 nonzeros, where the `.nl` path detects 5)
and that is the leading hypothesis, but it has not been established.
Until it is, the exit named in this bucket should be read as the Python
path's, and **the ownership bucket's description needs re-checking
against a harness re-measurement** — the count of 8 is unaffected, since
it comes from the triage rules and not from the exit string.

**It is not a property of the reformulation class either.** That was the
first version's claim, never tested against another solver and inferred
from the vanishing row gradient — which is real, but does not stop an
interior-point method converging.

Ipopt 3.14.19 and POUNCE 0.11.0 were then run on the **same `.nl` file**,
same options (`tol=1e-8 max_iter=300 bound_relax_factor=0
honor_original_bounds=yes`), exact Hessians on both sides, at default
`recalc_y=no`:

| `qpec_small`, `ncp_eq`, via `.nl` | start | iters | exit | **unscaled** overall |
|---|---|---:|---|---:|
| Ipopt 3.14.19 | `origin` | 58 | `Optimal Solution Found` | **`7.523e-09`** |
| POUNCE 0.11.0 | `origin` | 41 | `Solved To Acceptable Level` | **`7.897e+04`** |
| Ipopt 3.14.19 | `upper_left` | — | `Optimal Solution Found` | **`5.934e-02`** |
| POUNCE 0.11.0 | `upper_left` | 76 | `Optimal Solution Found` | **`1.564e-08`** |

**Neither implementation dominates, and both report success at a point
that is not stationary in the model's own units.** On `origin` POUNCE
reports acceptable-level at `7.9e+04` where Ipopt reaches `7.5e-09`; on
`upper_left` the roles reverse — POUNCE converges to `1.6e-08` and Ipopt
reports optimal at `5.9e-02`. So the second version of this section
("a POUNCE defect, upstream solves it") was drawn from the `origin` row
alone and does not survive the second start.

What both rows share is the mechanism, and it is the same one the
`recalc_y` note below describes: the aggregate the gates read is
`s_d`-normalised, `s_d` grows with the multipliers, and on a biactive
pair the multipliers are unbounded — so the scaled error passes while the
unscaled residual it stands for is orders larger. **`recalc_y` is not
required to reach that state**; both rows above are at default options.
The subject is the normalised gate, not the option.

The mechanism is identified exactly. The solve reaches
`x = (1.0, 1.0, 4.8e-09)` — the exact solution — with objective
`2.2e-16` and constraint violation `3.5e-19`, and then the *dual*
diverges, doubling every iteration from `0.31` to `164` while `‖d‖`
halves. The exit is the gh#274 near-feasible restoration re-entry gate:

```
[POUNCE] near-feasible restoration re-entry at theta 3.803e-19 but the point
fails the acceptable-level tolerances (nlp_err 4.049e-5); reporting
ErrorInStepComputation rather than Solved_To_Acceptable_Level (gh#274).
```

At a biactive pair the equality row `G_i H_i = 0` has gradient
`H_i ∇G_i + G_i ∇H_i`, and both terms vanish together there. Measured on
the cell that does converge, the biactive product row carries a
multiplier of `1.5e+11` against a row gradient of `3.2e-05`.

**The eight cells are not one phenomenon, and an earlier draft of this
section said they were.** That draft read the vanishing row gradient as
"no bounded multiplier exists in the limit". That inference is wrong: a
row whose gradient vanishes imposes no first-order restriction, so its
multiplier is *arbitrary* rather than nonexistent — `λ · 0 = 0` for every
`λ`, including `0`. Checking the KKT system directly at each solution
splits the eight:

| cells | best dual residual over sign-feasible multipliers | what it means |
|---|---|---|
| 7 — all `qpec_small` | **`0`**, at `λ = 0` exactly | the lowered NLP *has* a KKT point here; POUNCE does not reach it |
| 1 — `ralph1/unit/origin/direct` | **`0.707`** | no sign-feasible multiplier exists; failing is correct |

`qpec_small`'s solution `(1, 1, 0)` has `∇f = 0` exactly — the case
factory's own docstring says so, and says the zero multiplier vector is
admissible — and the biactive product row's gradient is exactly
`(0, 0, 0)`. So `λ = 0` satisfies stationarity with residual `0`. The
solver reaches that point (`cv 3.5e-19`, objective `2.2e-16` against
`f* = 0`) and then fails on a dual residual of `164` carried by
multipliers that blew up along the barrier path. What is wrong is the
multiplier estimate, not the point, and not the existence of an answer.

`ralph1` is the opposite and its docstring already said so: the origin is
M-stationary but **not** S-stationary, and NLP KKT is S-stationarity, so
no sign-feasible multiplier exists. Solving for one, the best achievable
residual is `0.707` — three orders above any tolerance in play. (An
*unsigned* least-squares fit reaches `4e-16` there, which is why sign
feasibility has to be part of the test and not an afterthought.)

The consequence for the ledger below: seven of the eight are a POUNCE
gap with a reachable answer, and the eighth is correct behaviour that
should never be "fixed". Lumping them together is the mistake this
report's own rule about branch coverage exists to prevent.

**Why it is not fixed *in this PR*** — which is a different claim from
the one this section used to make, and a much weaker one. It is a real
defect and it should be filed and fixed; it is not this PR's to fix.

1. **The supported route covers the corpus meanwhile, measured.**
   `scholtes_then_ncp` clears all eight cells and returns
   MPCC-feasible points on them (`1.9e-12`, `4.9e-11`, `5.7e-28`).
   The restart ladder absorbs the fragility — its 8 restarts are exactly
   these cells — rather than surfacing it. So Gate 0's recommendation
   does not depend on the fix landing first. That is a reason to file
   rather than to rush, not a reason to close.

2. **The obvious fix is the wrong one, and that is measured too** (see
   below). gh#274 requires the full acceptable-level triplet before a
   near-feasible restoration re-entry may claim acceptability, *because*
   an earlier, looser version reported a diverging iterate as solved:
   `min -exp(x) s.t. x >= 0` re-enters restoration with
   `inf_pr = 1.7e-10` and `inf_du = 8.8e+47`, and Pyomo maps
   `Solved_To_Acceptable_Level` into the solved family. Relaxing that
   gate to admit these cells is not the fix — upstream does not reach
   the gate at all, it converges. **The defect is upstream of the gate,
   in whatever makes POUNCE's dual residual grow to `164` over 118
   iterations on a model Ipopt finishes in 58.** A fix that only changed
   the exit verdict would be treating the symptom.

   **Re-estimating the multipliers is not the discriminator, and this
   was measured rather than reasoned.** The obvious candidate — re-solve
   for the multipliers at the returned point and test the residual that
   gives — is a mechanism POUNCE already ships, as `recalc_y` (the
   least-squares estimate of `IpLeastSquareMults`, applied once the
   iterate is feasible enough). Turning it on does flip every one of
   these cells to `Solve_Succeeded`. It should not be taken:

   | cell (with `recalc_y=yes`) | scaled `final_kkt_error` | **unscaled** `final_kkt_error` |
   |---|---:|---:|
   | `ralph1/unit/origin/direct` | `4.607e-10` | **`2.659e-02`** |
   | `qpec_small/unit/origin` | `1.091e-09` | `5.066e-08` |
   | `qpec_small/unit/upper_left` | `3.384e-11` | **`4.228e-03`** |
   | `qpec_small/skew/upper_right` | `1.664e-09` | **`6.179e-05`** |

   Three of the four are not stationary in the model's own units, by
   three to six orders, and they pass anyway: the re-estimated
   multipliers are enormous, `s_d` grows with them, and the *scaled*
   aggregate the strict gate reads is that residual divided by `s_d`.
   The verdict is normalised by the very quantity that is wrong. That is
   the P1 failure class — a status that passes because it is measured
   against the wrong denominator — and it is what this report would be
   recommending if it took the obvious route.

   The reason the estimate can do this is the vanishing gradient itself.
   At the exact solution the product row's gradient is exactly zero and
   `λ = 0` is the only bounded answer; a hair off it, the gradient is
   `~1e-9`, and a multiplier of `~1e9` on that row reproduces any vector
   you like. So an *unbounded* least-squares multiplier fit does not test
   stationarity near a biactive pair — it manufactures it. (Drafted here
   as the recommended fix on exactly that mistake, and caught by checking
   the unscaled column.) A test that would work has to bound the
   multipliers, or read the unscaled residual, and neither is what
   `recalc_y` does.

   **The gate does not always refuse them, and an earlier draft of this
   line said it did.** On the `.nl`/CLI path, at default options, this
   same model exits `Solved_To_Acceptable_Level` after 9 restoration
   iterations with an unscaled dual infeasibility of `7.897e+04` — i.e.
   *through* that gate, not refused by it. Reproduced locally; it is not
   a claim about one machine.

   **Why it passes is worth getting right, because the obvious answer is
   wrong.** It is not that the acceptable-level triplet reads
   `s_d`-normalised quantities: the three component tests read the
   *unscaled* residuals (`current_is_acceptable_with_state` calls
   `curr_unscaled_dual_infeasibility_max` and its siblings), exactly as
   `acceptable_dual_inf_tol`'s own help text specifies. Both halves of
   the test pass on their own terms:

   | test | quantity | threshold | |
   |---|---:|---:|---|
   | aggregate | **scaled** `8.234e-11` | `acceptable_tol` `1e-6` | passes — this one *is* `s_d`-normalised, and `s_d ≈ 1e15` |
   | dual component | **unscaled** `7.897e+04` | `acceptable_dual_inf_tol` `1e10` | passes — on the threshold's size, not on any normalisation |

   So there is **no implementation defect here to fix**. POUNCE reports
   `Solved_To_Acceptable_Level` because a point with an unscaled dual
   infeasibility of `7.9e+04` *is* acceptable under the registered
   defaults, and those defaults are upstream Ipopt's. `acceptable_tol` is
   documented as applying to the scaled overall error and
   `acceptable_dual_inf_tol` to the unscaled dual infeasibility at `1e10`;
   both are implemented as documented.

   That does not make the outcome harmless — it is still
   `Solved_To_Acceptable_Level`, which Pyomo maps into the solved family,
   at a dual residual of `7.9e+04`. But it relocates the question from
   "what is broken" to "is `acceptable_dual_inf_tol = 1e10` the right
   default", which is a policy choice with a blast radius of every
   acceptable-level exit in the solver, and a deliberate deviation from
   upstream if changed. Refusing this point needs *either* threshold
   moved: the aggregate would refuse it if read unscaled (`7.9e+04` is not
   `≤ 1e-6`), and the component would refuse it at any threshold below
   `7.9e+04`. Both are decisions, not repairs.

   One thing that *is* a gap rather than a policy: `kkt_fidelity_tol`
   (gh#173) exists for precisely this shape — "the scaled convergence test
   passes but the user-space duals have drifted" — and it only ever
   downgrades `Solve_Succeeded`. An exit that is already
   `Solved_To_Acceptable_Level` is outside its reach, so the one guard
   built for this case cannot be aimed at it even by a caller who opts in.

3. **It is a core-IPM change and needs the fixture sweep and an owner.**
   The verdict is reached on an `s_d`-normalised aggregate, so the blast
   radius of touching that is every model, not merely those reaching the
   gh#274 gate; `scripts/sweep-fixtures.sh` over both legs is the minimum
   evidence, and a measured regression there needs an issue and an owner
   of its own. That does not belong in a benchmark PR.

   One acceptance criterion is available and worth writing down even
   though it is only half the picture: **on `origin`, `qpec_small` under
   `ncp_eq` must not report success at an unscaled residual of `7.9e+04`,
   and Ipopt's 58 iterations to `7.5e-09` show a point exists to converge
   to.** It is half the picture because on `upper_left` the comparison
   runs the other way, so "match Ipopt" is not the criterion — "do not
   report success at a residual four orders above `tol` in the model's
   own units" is.

What a future fix would have to establish, stated so it does not have to
be re-derived — and narrowed by the `recalc_y` measurement above, which
rules out the first thing anyone will try:

- a discriminator between "converged primal, unbounded multiplier" and
  "diverging iterate" that is checkable at the gate **and that a
  multiplier of `1e9` on a `1e-9` gradient cannot satisfy**. An
  unbounded least-squares multiplier fit is not one; nor is anything
  read off the `s_d`-normalised aggregate, since `s_d` is itself a
  function of the multipliers being estimated. The unscaled residual is
  the quantity that stays honest here, and on these cells it refuses
  three of the four outright;
- evidence it does not relabel any `-exp(x)`-shaped case;
- and a fixture sweep across both legs showing what else moves.

The cheaper half of that is already available: `recalc_y=yes` is a
one-option experiment that reproduces the wrong answer on demand, so any
proposed fix can be checked against it before it is written.

gh#794's issue-splitting rule is satisfied for filing this — minimal
model, commit-stamped comparators, kill-switch evidence
(`upstream_heuristics` moves 3 of the cells, `no_scaling` moves others,
neither explains it) — but no issue is opened here.

**A separate candidate this turned up**, recorded rather than filed:
`recalc_y=yes` can take a point whose unscaled KKT error is `2.7e-02`
and report `Solve_Succeeded`, because the re-estimated multipliers
inflate `s_d` and the strict gate reads the quotient. What makes it
worth an issue rather than a footnote is the direction: an option whose
purpose is *better* multipliers converts an honest failure into a false
success, and it does so through a feedback loop — the estimate changes
the denominator the estimate is judged against.

Not a regression, off by default, and **upstream Ipopt 3.14.19 does the
same thing — measured, not inferred.** On `qpec_small/unit/origin` under
`ncp_eq`, with `recalc_y=yes` and everything else as above:

| `recalc_y` | iters | exit | objective | scaled overall | **unscaled overall** | `Σ‖λ‖₁` |
|---|---:|---|---:|---:|---:|---:|
| `no` | 58 | `Optimal Solution Found` | `4.545e-10` | `4.262e-10` | `7.523e-09` | `8.96e+07` |
| `yes` | 8 | `Optimal Solution Found` | `4.765e-06` | `1.428e-10` | **`2.636e-03`** | `1.48e+10` |

The multiplier mass is the mechanism, visible directly: `recalc_y`
multiplies `Σ‖λ‖₁` by 165, `s_d` rises with it, the scaled aggregate the
strict gate reads falls to `1.4e-10`, and the unscaled residual it is
covering for *worsens* by six orders. Upstream stops at iteration 8 on a
point four orders worse in objective than the one it reaches without the
option, and calls both `Optimal Solution Found`.

Upstream reproduces it on the second start too: `upper_left` with
`recalc_y=yes` exits `Optimal Solution Found` in 23 iterations at scaled
`7.327e-13` over unscaled `3.476e-02`, carrying `λ[p2] = -5.692e+13`
against a row gradient of `1.308e-08` — the "a huge multiplier on a
vanishing gradient costs nothing in the fit" argument at `1e13` scale.

**But `recalc_y` is not necessary, and framing the finding around it is
too narrow.** At *default* options the same false success is reachable in
both implementations: POUNCE reports `Solved_To_Acceptable_Level` at an
unscaled `7.897e+04` on `origin`, and Ipopt reports `Optimal Solution
Found` at an unscaled `5.934e-02` on `upper_left`. The option makes the
trap easier to reach; it does not create it.

So the subject is **the `s_d`-normalised gate itself**, in both
implementations, and an issue filed on the narrow "`recalc_y` converts an
honest failure into a false success" framing would attract a
correspondingly narrow fix. The question worth asking is whether a
verdict may be reached on an aggregate normalised by the multipliers,
when the multipliers are exactly what is unbounded — and gh#532 already
answers a neighbouring version of it for the *strict* dual component,
which is where a fix should start reading.

### P3 — `presolve_licq_action=auto_l1` did nothing

`ncp_eq_auto_l1` produced results **identical to `ncp_eq` in all 64
cells** — same status, same iteration count, same objective — and the
`no_presolve` control moves nothing anywhere in the matrix. On a corpus
where every case's lowering is rank-deficient at every feasible point by
construction, the presolve LICQ check never fired.

Not filed as a defect: the check is documented as detecting
rank-deficient *equality blocks* before the IPM starts, and the
degeneracy here is at the solution rather than in the initial block
structure. Recorded because it is a live trap — `auto_l1` looks like the
natural setting for an MPCC and, on this corpus, is a no-op.

## Gate decision

**Proceed to Gate 1, with two conditions.** The exit criterion — a
supported route and default settings plus a documented failure boundary
— is met: `scholtes_then_ncp` solves 60 of 60 cells, reaches the global
optimum in 57 and a genuine local one in 3, never returns a point the
MPCC does not contain, and correctly certifies M on the two cases where
S is unavailable. Representative cases are repeatably reliable, so the
stop decision gh#794 provides for is not the right one.

The conditions:

1. **Gate 1 must report source-level complementarity and MPCC
   stationarity separately from the NLP residuals**, using the contract
   in `benchmarks/mpcc/schema.json`, and must not quote a regime
   decision or an objective comparison tighter than the
   complementarity tolerance floor above. Every route that looks good on
   NLP residuals alone fails on one of the source columns.

2. **The flash pairs are nonlinear and this corpus is not.** Gate 1's
   first task is to re-establish boundary items 1, 5 and 6 with a
   nonlinear `G`/`H`: the product row's Hessian stops being constant,
   the `sqrt(tol)` floor acquires the pair's own curvature, and the
   whole trajectory argument changes.

Nothing in this report authorises tray or column work; gh#794's last
acceptance criterion stands.
