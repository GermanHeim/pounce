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
| POUNCE candidate | 8 | finding P2 |

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

### P2 — a biactive pair breaks a cold exact-product solve. **Not fixed; covered.**

Eight of the 576 observations, and all of them the same exit. On
`qpec_small` from `origin` and `upper_left` at unit scaling, the
`ncp_eq` family ends in `Error_In_Step_Computation`; so does `direct` on
`qpec_small/skew/upper_right` and on `ralph1/unit/origin`. Both
exact-product lowerings are affected at similar rates, so this is a
property of the reformulation class, not of one route.

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

The diverging dual is not a numerical accident. At a biactive pair the
equality row `G_i H_i = 0` has gradient `H_i ∇G_i + G_i ∇H_i`, and both
terms vanish together there: no bounded multiplier exists in the limit.
Measured on the cell that does converge, the biactive product row
carries a multiplier of `1.5e+11` against a row gradient of `3.2e-05`.

**Why it is not fixed here.** Three reasons, in order of weight:

1. **The supported route already covers it, measured.**
   `scholtes_then_ncp` clears all eight cells and returns
   MPCC-feasible points on them (`1.9e-12`, `4.9e-11`, `5.7e-28`).
   The restart ladder absorbs the fragility — its 8 restarts are exactly
   these cells — rather than surfacing it. The route table above is the
   evidence; no cell of the corpus is left unsolved.

2. **A solver-side fix means loosening a gate that exists to prevent a
   wrong answer.** gh#274 requires the full acceptable-level triplet
   before a near-feasible restoration re-entry may claim acceptability,
   *because* an earlier, looser version reported a diverging iterate as
   solved: `min -exp(x) s.t. x >= 0` re-enters restoration with
   `inf_pr = 1.7e-10` and `inf_du = 8.8e+47`, and Pyomo maps
   `Solved_To_Acceptable_Level` into the solved family. Distinguishing
   that from our case needs a discriminator — ours has a *flat, finite*
   objective where the unbounded case's objective diverges alongside the
   dual — plus a definition of flatness, a window, and an interaction
   with `acceptable_progress_kappa`'s existing machinery.

3. **It is a core-IPM termination change and would need the fixture
   sweep and an owner.** Every model that reaches this gate is in its
   blast radius; `scripts/sweep-fixtures.sh` over both legs is the
   minimum evidence, and a measured regression there needs an issue and
   an owner of its own. That does not belong in a benchmark PR, and
   nothing in this corpus establishes the acceptance criterion such a
   change would have to meet.

What a future fix would have to establish, stated so it does not have to
be re-derived: a discriminator between "converged primal, unbounded
multiplier" and "diverging iterate" that is checkable at the gate;
evidence it does not relabel any `-exp(x)`-shaped case; and a fixture
sweep across both legs showing what else moves. gh#794's
issue-splitting rule is satisfied for filing this — minimal model,
commit-stamped comparators, kill-switch evidence (`upstream_heuristics`
moves 3 of the cells, `no_scaling` moves others, neither explains it) —
but no issue is opened here.

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
