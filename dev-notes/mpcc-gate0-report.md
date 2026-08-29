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
| pounce commit | `235956e` (clean) |
| model-data revision | `93f01eeffb84c866` |
| pounce version | 0.10.0 (Python extension, release build) |
| discopt | absent — a cross-repository *design* dependency of gh#776, not a runtime dependency of this harness. No DiscOpt comparison was run. |
| CCOpt | absent. Optional per gh#794; the pin a comparison would have used is `ccopt==0.4.1`. |
| matrix | 11 cases × 2 scaling legs × 3 starts × 8 routes × 5 controls = 2560 records, 188 s |
| pinned options | `tol=1e-8`, `max_iter=300`, `bound_relax_factor=0`, `honor_original_bounds=yes` |
| linear algebra | identical across all eight routes — one POUNCE build, one process, the default linear solver, no per-route backend selection. No comparison below crosses a linear-algebra boundary. |
| relaxation schedule | `tau = 1e0 … 1e-8`, ×0.1, one bisection allowed after a rejected stage |

`bound_relax_factor=0` is not a detail. Its default of 1e-8 relaxes every
constraint bound before the solve, which on this corpus means accepting
`G >= -1e-8` — a source-level sign violation of the same order as the
complementarity products the whole report is trying to measure.

## The decision

**Supported route: `ncp_eq` — the exact product / NCP equality lowering
(`G >= 0`, `H >= 0`, `G*H = 0`) solved by ordinary POUNCE at default
settings.** It is the only route in the comparison that never returned
an objective below the MPCC's own optimum, and the only one that reaches
the right answer on both cases where no S-stationary point exists.

**Recommended default for a model whose branch is not known in advance:
Scholtes continuation with full warm starts to locate the branch,
followed by one `ncp_eq` solve seeded from it.** The continuation never
failed to converge on this corpus and reaches the global optimum more
often than any single solve; its answer is only ever `tau`-feasible, so
it must be finished by an exact-product solve rather than reported.

**Not supported on this evidence: the `l1_exact_penalty_barrier` routes**
(`ncp_eq_l1`, `ncp_eq_l1_fallback`). See finding P1.

### Route comparison

Cells are `(case, scaling, start)`, 60 per route with `infeasible_pair`
excluded and reported separately. `at f*` counts the cells that reached
the global optimum the branch-enumeration oracle knows independently;
`other local` counts cells that reached a different genuine local
solution, which is expected of a local solver and is not a failure;
`below f*` counts cells that returned an objective *lower* than the
optimum, which is possible only at a point the source model does not
admit.

| route | solved | at f* | other local | below f* | not solved | median iters |
|---|---:|---:|---:|---:|---:|---:|
| `direct` | 58 | 47 | 9 | 2 | 2 | 43 |
| **`ncp_eq`** | **58** | **47** | **11** | **0** | **2** | **20** |
| `ncp_eq_l1` | 60 | 45 | 3 | 12 | 0 | 18 |
| `ncp_eq_l1_fallback` | 58 | 41 | 11 | 6 | 2 | 18 |
| `ncp_eq_auto_l1` | 58 | 47 | 11 | 0 | 2 | 20 |
| `scholtes_cold` | 60 | 40 | 8 | 12 | 0 | 164 |
| `scholtes_warm_primal` | 60 | 48 | 0 | 12 | 0 | 60 |
| `scholtes_warm_full` | 60 | 48 | 0 | 12 | 0 | 49 |

The `below f*` column is the one that decides the recommendation. A
route with a higher solved count and a nonzero `below f*` is not better;
it is returning points the MPCC does not contain and calling them
solutions.

### Per case, `at f* / other local / below f* / not solved`

| case | direct | ncp_eq | l1 | l1_fb | auto_l1 | s_cold | s_prim | s_full |
|---|---|---|---|---|---|---|---|---|
| `regular_strict` | 3/3/0/0 | 3/3/0/0 | 5/1/0/0 | 3/3/0/0 | 3/3/0/0 | 3/3/0/0 | 6/0/0/0 | 6/0/0/0 |
| `biactive_positive` | 6/0/0/0 | 5/1/0/0 | 6/0/0/0 | 5/1/0/0 | 5/1/0/0 | 6/0/0/0 | 6/0/0/0 | 6/0/0/0 |
| `ralph1` | 4/0/1/1 | **6/0/0/0** | 0/0/6/0 | 6/0/0/0 | 6/0/0/0 | 0/0/6/0 | 0/0/6/0 | 0/0/6/0 |
| `ctrap` | 4/2/0/0 | 6/0/0/0 | 6/0/0/0 | 6/0/0/0 | 6/0/0/0 | 5/1/0/0 | 6/0/0/0 | 6/0/0/0 |
| `infeasible_pair` | 0/0/0/4 | 0/0/0/4 | 0/0/0/4 | 0/0/0/4 | 0/0/0/4 | 0/0/0/4 | 0/0/0/4 | 0/0/0/4 |
| `selector_theta_025` | 4/2/0/0 | 3/3/0/0 | 5/1/0/0 | 3/3/0/0 | 3/3/0/0 | 4/2/0/0 | 6/0/0/0 | 6/0/0/0 |
| `selector_theta_050` | 6/0/0/0 | 6/0/0/0 | 6/0/0/0 | 6/0/0/0 | 6/0/0/0 | 6/0/0/0 | 6/0/0/0 | 6/0/0/0 |
| `selector_theta_075` | 4/2/0/0 | 4/2/0/0 | 5/1/0/0 | 4/2/0/0 | 4/2/0/0 | 4/2/0/0 | 6/0/0/0 | 6/0/0/0 |
| `ralph2` | 6/0/0/0 | 4/2/0/0 | 6/0/0/0 | 4/2/0/0 | 4/2/0/0 | 6/0/0/0 | 6/0/0/0 | 6/0/0/0 |
| `scholtes4` | 5/0/1/0 | **6/0/0/0** | 0/0/6/0 | 0/0/6/0 | 6/0/0/0 | 0/0/6/0 | 0/0/6/0 | 0/0/6/0 |
| `qpec_small` | 5/0/0/1 | 4/0/0/2 | 6/0/0/0 | 4/0/0/2 | 4/0/0/2 | 6/0/0/0 | 6/0/0/0 | 6/0/0/0 |

`ralph1` and `scholtes4` are the two cases whose solution is
M-stationary and **not** S-stationary. Only `ncp_eq` (and its `auto_l1`
alias, which is the same solve — see P3) reaches them, 6 of 6 cells
each, with the classifier certifying M and refusing S. Every other route
either returns a point below the optimum or, on `direct`, fails.

## The documented failure boundary

1. **Scholtes continuation never returns an MPCC-feasible point.** Its
   answer carries complementarity at the schedule floor by construction:
   on this corpus the returned `max |G_i H_i|` is `1.0e-8` = `tau_min` in
   every continuation cell that ran the schedule out. On `ralph1` that is
   an objective of `-1.0e-4 ≈ -sqrt(tau_min)`, and on `ralph2` exactly
   `-2*tau_min`, both *below* the true optimum of 0. This is the method
   behaving as designed and it is why the harness computes source
   feasibility separately; it is also why the recommended default
   finishes with an exact-product solve. 162 of the 512 triaged
   observations carry this label.

2. **Below `tau = 1e-8` is untested.** Nothing here says whether the
   schedule can be driven further, or what happens when it is.

3. **Where no S-stationary point exists, no route can certify one.** On
   `ralph1` MPCC-LICQ fails and the multiplier system admits no
   nonnegative pair; on `scholtes4` the multiplier set is a line that
   never enters the nonnegative orthant. The best available certificate
   is M, `ncp_eq` reaches it, and a consumer that treats "converged NLP"
   as "S-stationary MPCC solution" is wrong on both. **The stationarity
   class, not the NLP status, is the thing to read.**

4. **The biactive cases lean on acceptable-level termination.** The
   `no_acceptable` control (`acceptable_iter=0`) flips **9** verdicts,
   every one of them a `ncp_eq`-family solve on `biactive_positive` or
   `scholtes4` that succeeded only as `Solved_To_Acceptable_Level` and,
   with the criterion off, ends in `Error_In_Step_Computation`. The
   supported route's success on biactive points is therefore partly a
   success of the fallback verdict, not of the strict one. A Gate 1
   model with biactive phase pairs should expect this and should not be
   tuned with `acceptable_iter=0`.

5. **Scaling is load-bearing, and the scaled KKT error is not the one to
   read.** The `skew` leg spans six orders across the variables — mild
   next to a real flash — and moves **31 of 256** `(case, start, route)`
   cells. The `no_scaling` control moves 22 objectives and 10 verdicts.
   Eleven observations triage as `scaling`: POUNCE converged the problem
   it internally scaled (`final_kkt_error` ~1e-9 to 1e-13) while its own
   `final_unscaled_kkt_error` is three or more orders larger, and the
   MPCC stationarity residual, measured in the user's units, agrees with
   the unscaled figure. POUNCE reports both (gh#173); a Gate 1 harness
   that reads only the scaled one will believe a point it should not.

6. **A second pair is where the exact-product route starts to fail.**
   `qpec_small` is the only case with two complementarity pairs, one
   strict and one biactive. The `ncp_eq` family fails 2 of its 6 cells
   (unit scaling, `origin` and `upper_left`) with
   `Error_In_Step_Computation`, where `direct` and all three continuation
   routes solve it. One pair is not evidence about two.

7. **A POUNCE-only convergence mechanism is load-bearing in 3 cells.**
   `upstream_heuristics` (`acceptable_progress_kappa`,
   `dual_inf_scale_kappa`, `obj_scale_certificate_threshold` all zeroed,
   which each option's own documentation calls bit-for-bit upstream
   behaviour) turns `qpec_small/skew/upper_left` from `Solve_Succeeded`
   into `Error_In_Step_Computation` for `ncp_eq`, `ncp_eq_l1_fallback`
   and `ncp_eq_auto_l1`. Worth knowing before any of those three
   mechanisms is changed for an unrelated reason.

8. **Infeasibility is detected, on every route.** No route ran to
   completion on `infeasible_pair` in any of its 32 cells, and none
   returned a success status. `direct` and
   `ncp_eq_l1` return `Infeasible_Problem_Detected`; the rest of the
   exact-product family returns `Not_Enough_Degrees_Of_Freedom`, which is
   structurally correct (two variables, two source equalities and an
   equality product row). The continuation accepts `tau = 1` and the
   bisected `tau = 0.316` and rejects `tau = 0.1` — the relaxation is
   feasible exactly for `tau >= 1/4`, so the observed crossover sits on
   the analytic one. That agreement is the strongest single check that
   the harness measures what it claims to.

9. **The corpus itself.** Every pair is affine, every objective and row
   quadratic, nothing exceeds three variables and two pairs, and there is
   no dynamics, thermodynamics or phase equilibrium anywhere. Nothing
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

The restart ladder was never needed: 0 restarts and 0 rejected stages
across all three arms on every case except `infeasible_pair`, where the
rejection is the correct answer. Active-pair regime changes along the
schedule: 32 for the cold arm against 24 for both warm arms — the cold
arm wanders between branches that the warm arms hold.

Most of the primal-only arm's gain is already the whole gain; carrying
the duals and `mu` adds 9%. That is worth stating plainly, because the
warm-start machinery's cost to a caller is dominated by capturing and
replaying the dual blocks correctly, and on this corpus that work buys
very little. It may buy more on a model whose barrier is far from its
default.

## Ownership of the observed gaps

512 observations were triaged mechanically (rules in
`benchmarks/mpcc/report.py`). **No observation is unassigned**, so
gh#794's issue-splitting rule is not triggered by anything except the
two POUNCE findings below, both of which already have a minimal
reproducer.

| owner | observations | what it means |
|---|---:|---|
| converged, nothing to assign | 265 | reached an S- or M-stationary point |
| relaxation limit | 162 | the Scholtes route's `tau`-feasible answer (boundary item 1) |
| POUNCE candidate | 42 | findings P1 and P2 below |
| source formulation | 32 | `infeasible_pair`: failing is the correct answer |
| scaling | 11 | boundary item 5 |

### Nothing is assigned to DiscOpt

DiscOpt is not a runtime dependency of this harness and none of the
three lowerings compared here is DiscOpt's. What Gate 0 hands back to
[jkitchin/discopt#1123](https://github.com/jkitchin/discopt/issues/1123)
and [#1124](https://github.com/jkitchin/discopt/issues/1124) is a
requirement rather than a defect: **a lowering that reports only the
NLP's residuals is not reportable as an MPCC result.** The three
quantities that have to survive the lowering are the source
complementarity product, the sign residuals of `G, H >= 0` measured in
the model's own units, and enough of the source model to classify MPCC
stationarity. `benchmarks/mpcc/schema.json` is the concrete shape that
satisfies that, and it is deliberately solver-agnostic.

### P1 — `l1_exact_penalty_barrier` returns `Solve_Succeeded` at a point that violates the constraints it was given

The most consequential finding, and the reason the ℓ₁ routes are not
recommended. Reproducer, 25 lines, no harness:

```python
import numpy as np, pounce
# ralph1 as an NCP-equality NLP:
#   min 2x - y  s.t.  x >= 0,  G = y >= 0,  H = y - x >= 0,  G*H = 0
class P:
    def objective(self, z):  return 2*z[0] - z[1]
    def gradient(self, z):   return np.array([2., -1.])
    def constraints(self, z):
        x, y = z; return np.array([y, y - x, y*(y - x)])
    def jacobian(self, z):
        x, y = z; return np.array([0., 1.,  -1., 1.,  -y, 2*y - x])
    def jacobianstructure(self):
        return (np.repeat(np.arange(3), 2), np.tile(np.arange(2), 3))
    def hessianstructure(self):
        r, c = np.tril_indices(2); return (r, c)
    def hessian(self, z, lam, of):
        H = lam[2]*np.array([[0., -1.], [-1., 2.]])
        r, c = np.tril_indices(2); return H[r, c]

B = 1e20
for l1 in ("no", "yes"):
    prob = pounce.Problem(n=2, m=3, problem_obj=P(),
                          lb=[0., -B], ub=[B, B], cl=[0, 0, 0], cu=[B, B, 0])
    for k, v in [("print_level", 0), ("tol", 1e-8), ("bound_relax_factor", 0.0),
                 ("l1_exact_penalty_barrier", l1)]:
        prob.add_option(k, v)
    x, info = prob.solve(x0=np.zeros(2))
    print(l1, info["status_msg"], info["obj_val"], x,
          "reported viol", info["final_constr_viol"],
          "actual |c3|", abs(P().constraints(x)[2]))
```

Output on `235956e`:

```
no  Solve_Succeeded  1.818182e-09  [1.8e-09 1.8e-09]  reported viol 5.5e-26  actual |c3| 1.7e-29
yes Solve_Succeeded -5.000757e-04  [1.7e-08 5.0e-04]  reported viol 9.6e-15  actual |c3| 2.5e-07
```

With the wrapper on, the returned point violates the equality row it was
given by `2.5e-07` while the solve reports `final_constr_viol = 9.6e-15`
— and `final_unscaled_constr_viol` reports the same `9.6e-15`, so there
is no field in the result a caller could have read to notice. The
objective is `5.0e-4` **below** the true optimum, which is only possible
off the feasible set.

The shape of the number suggests the residual is being measured on the
augmented problem `c(x) - p + n = target`, which the slacks satisfy
exactly by construction, rather than on `c(x)` in the original space.
`crates/pounce-l1penalty`'s README describes original-space reporting, so
either the reporting or the README is wrong. The wrapper's *answer* may
also be defensible as a penalty solution — but a penalty solution
reported as `Solve_Succeeded` with a 1e-14 constraint violation is not.

Reach across the corpus: **34 of the 42** POUNCE-candidate observations
are on an ℓ₁ route — 24 on `ncp_eq_l1` and 10 on `ncp_eq_l1_fallback`,
across `ralph1`, `scholtes4`, `ralph2` and `qpec_small` and both scaling
legs. The fallback route inherits the behaviour wherever the fallback
engages, which on `scholtes4` is all six of its cells. This meets gh#794's
issue-splitting bar in full: minimal source model and initial point,
commit-stamped baseline and comparator (`ncp_eq` at the same commit),
kill-switch evidence (no control changes it), and a measurable
acceptance criterion — **`final_constr_viol` must be the violation of
the constraints the caller declared, or the status must not be
`Solve_Succeeded`.** It is not filed here; filing is a call for the
maintainer.

### P2 — `Error_In_Step_Computation` on the two-pair case where other routes succeed

`qpec_small` at unit scaling from the `origin` and `upper_left` starts:
`ncp_eq`, `ncp_eq_l1_fallback` and `ncp_eq_auto_l1` all end in
`Error_In_Step_Computation`, while `direct` (the same model with `G*H <=
0` instead of `= 0`) and all three continuation routes reach `f* = 0`.
Six observations. The distinguishing feature of the case is a second
complementarity pair, one of which is biactive at the solution, so the
equality lowering presents two rows whose gradients vanish together
there.

Smaller and separately interesting: the same three routes recover on the
`skew` leg where the unit leg fails, and `upstream_heuristics` reverses
that recovery (boundary item 7). That is enough of a thread to pull, and
not yet enough to name a mechanism.

### P3 — `presolve_licq_action=auto_l1` did nothing

`ncp_eq_auto_l1` produced results **identical to `ncp_eq` in all 64
cells** — same status, same iteration count, same objective — and the
`no_presolve` control moves nothing anywhere in the matrix. On a corpus
where every case's lowering is rank-deficient at every feasible point by
construction, the presolve LICQ check never fired.

This is not filed as a defect because the check is documented as
detecting rank-deficient *equality blocks* before the IPM starts, and the
degeneracy here is at the solution rather than in the initial block
structure. It is recorded because it is a live trap: `auto_l1` looks
like the natural setting for an MPCC and, on this corpus, it is a no-op
with an ℓ₁ route (P1) waiting behind it if it ever does fire.

## Gate decision

**Proceed to Gate 1, with three conditions.** The exit criterion — a
supported route and default settings plus a documented failure boundary
— is met: `ncp_eq` solves 58 of 60 cells, reaches the global optimum in
47 and a genuine local one in 11, never returns a point the MPCC does
not contain, and correctly certifies M on the two cases where S is
unavailable. Representative cases are repeatably reliable, so the stop
decision gh#794 provides for is not the right one.

The conditions:

1. **Gate 1 must report source-level complementarity and MPCC
   stationarity separately from the NLP residuals**, using the contract
   in `benchmarks/mpcc/schema.json`. Every route in this comparison that
   looks good on NLP residuals alone fails on one of the source columns.

2. **P1 is resolved or the ℓ₁ routes are documented as unsupported for
   complementarity constraints** before any Gate 1 model uses them. The
   present behaviour — a success status on a point that violates the
   caller's constraints, with no field disclosing it — is worse for a
   phase-change model than an outright failure would be, because a
   physically wrong phase state is exactly what it will produce.

3. **The flash pairs are nonlinear and this corpus is not.** Gate 1's
   first task is to re-establish items 1, 5 and 6 of the boundary with a
   nonlinear `G`/`H`, because the product row's Hessian stops being
   constant and the whole trajectory argument changes.

Nothing in this report authorises tray or column work; gh#794's last
acceptance criterion stands.
