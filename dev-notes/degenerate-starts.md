# Degenerate starting points, and what actually recovers them

The measurement behind `start_point_perturbation`,
`start_point_conditioner`, `infeasibility_perturbed_start_retry`, and
the start-point audit in `crates/pounce-nlp/src/degeneracy.rs`.

User-facing versions of this material live in
`docs/src/initialization.md` ("Conditioning the starting point"),
`docs/src/troubleshooting.md` ("What POUNCE says when it stops from a
degenerate point") and `docs/src/acknowledgments.md`. This note is the
evidence, kept for the next person who wants to change a default.

## Where the corpus came from

KRONOS (Ahmed, M. G. T. & Hasan, M. M. F., *Comput. Chem. Eng.* **215**
(2026) 109839, doi:10.1016/j.compchemeng.2026.109839) ships a
244-problem benchmark set. We ran it head-to-head at K=1 from identical
bundled starting points:

| Solver | solved | global |
|---|---|---|
| KRONOS | 225 | 175 |
| POUNCE, models as given (squared-slack form) | 191 | 155 |
| POUNCE, natural form | 223 | 189 |

On the 208 both solved: KRONOS 35.88 s, POUNCE 3.11 s.

The two solved sets are not nested. The interesting direction is the
fifteen models POUNCE fails from their bundled start, ten of which
KRONOS proves feasible to 2.4e-7 or better — i.e. ten wrong verdicts,
not ten hard problems.

## What recovers the fifteen

| Remedy | recovered |
|---|---|
| default | 0 |
| `start_with_resto=yes` | 0 |
| `expect_infeasible_problem=yes` | 0 |
| `mu_strategy=adaptive` | 4 |
| Adam warm-up (KRONOS stage 0) | 3 |
| **one displaced start, relative 1e-2** | **13** |
| restoration + displaced start | 14 |

The two zeros at the top are the finding. Restoration is the textbook
answer to an infeasibility verdict and it recovers *nothing*, because
restoration inherits the same degenerate point — it is a different
objective solved from the same place. The iterate does not need to be
*better*, it needs to be *non-degenerate*.

The mechanism: at a start where the constraint Jacobian is
rank-deficient, LICQ fails, and a filter line search has no descent
direction to find whatever else you hand it. Two ways in dominate the
corpus:

- **A squared slack at zero.** KRONOS's own reformulation
  (`g ≤ 0 → g + s² = 0`) has `∂(s²)/∂s = 0` at `s = 0`, so the Jacobian
  loses rank *exactly on the active set*. This is why POUNCE does far
  better on the natural form (223) than on the models as KRONOS states
  them (191) — the reformulation is a good idea for KRONOS's own
  least-norm Newton step and a bad one for an IPM.
- **An origin start on a homogeneous quadratic.** `x'Qx = b` has a zero
  gradient at `x = 0`, and an all-zeros default start is extremely
  common. This is also why the displacement is
  `scale·(1 + |x_i|)·u_i` and not `scale·|x_i|·u_i`: a purely relative
  perturbation is identically zero at the origin, which is the case it
  most needs to fix.

## The whole corpus, re-run against the built branch

The table above is a per-remedy probe. This is the end-to-end number:
all 244 problems, one start each (`K=1`), identical starting points,
re-run after the branch was built.

| Leg | solved | global optimum | solve time |
|---|---|---|---|
| KRONOS | 225/244 | 175/244 | 42.9 s |
| POUNCE, models as KRONOS states them (squared slacks) | 191/244 | 155/244 | 14.6 s |
| POUNCE, natural inequality form | 223/244 | 189/244 | 7.4 s |
| **POUNCE, natural form + the second-opinion ladder** | **239/244** | **199/244** | **9.4 s** |

The first three rows reproduce the pre-branch run to the problem. That
is the point of including them: the diagnosis changes nothing an
embedder can observe, so the library legs *must* not move, and they do
not. The fourth row is the new one.

Sixteen problems move, and every promoted answer is genuinely feasible
(worst residual 1.0e-8, on `hs020`):

| rung | promotions |
|---|---|
| `feral_scaling=mc64` | 1 |
| `mu_strategy=adaptive` | 3 |
| `start_point_perturbation=1e-2` | 12 |

Twelve of the sixteen are rung 3's. Four of the sixteen were
`Invalid_Number_Detected`, not `Infeasible_Problem_Detected` — those
are the ones no earlier rung would have been offered, since rungs 1 and
2 do not fire on an invalid number.

Cost of the ladder over the whole corpus: **34 extra solves, +2.00 s**,
spread over 244 problems, spent only on runs that had already failed.

Set difference against KRONOS: 17 problems the ladder solves and KRONOS
does not; **3** the other way (`a10_perm`, `a29_rump`, `hong`, down from
fifteen); 2 neither solves (`bt13`, `lewispol`).

On the 222 both solve, KRONOS spends 37.6 s to POUNCE's 5.6 s — 6.7x.
That ratio is *worse* than the 12x on the 208 both solved before the
ladder, and it should be: the fourteen problems added to the
intersection are exactly the hard ones, and POUNCE now pays two or
three failed solves on each before it gets them.

One caveat on how this was measured. **The second-opinion ladder is
CLI-only.** `grep -rn "second_opinion\|scaling_retry" crates/pounce-py/src/
python/` returns nothing: the conditioner options themselves are
reachable from Python, but the retry logic that drives them is in
`crates/pounce-cli/src/main.rs`. The harness therefore reproduces the
ladder option-for-option rather than measuring it through the library —
including its non-cumulativeness, which is load-bearing (gh#524). A
Python embedder does not get rung 3 today, and that is a real gap
between the two frontends rather than an artifact of the benchmark.

`lch` needed its 300 s harness timeout lifted, for a reason that is not
POUNCE's: it is n=600, and JAX takes 3m30s to compile its Lagrangian
Hessian. The third leg pushed the total past the cap. Given the room it
solves, in 1.14 s and 27 iterations.

## Why Adam is an option and not a default

KRONOS's stage 0 is Adam on `f(x) + ρ‖h(x)‖²` (200 iters, lr 5e-2,
ρ 10). POUNCE implements it generalised to two-sided bounds, so it
applies to an arbitrary NLP: the violation of a row is its distance
outside `[g_l, g_u]`, which reduces to `g - b` on an equality row.

Measured over 40 problems POUNCE already solves (i.e. deliberately
*not* the failing fifteen — the question is what it costs the healthy
majority):

- 40/40 still solved. It breaks nothing.
- 22 improved. `rk23` 82 → 11, `bt5` 45 → 9, `chnrosnb` 40 → 10,
  `hs056` 42 → 12.
- median 0.83×, geomean 0.79×.
- **total 1.62× worse**: 3030 → 4900 iterations.

The total is the whole argument. It is carried by `palmer1c`
71 → 1023 and `biggs6` 1906 → 2938; excluding those two the ratio is
0.89×. `palmer1c` is badly scaled, and a *fixed, unscaled* ρ against a
badly-scaled model walks the iterate somewhere the barrier method then
has to walk back from. The defect is in `adam_warmup_penalty`'s
default, not in the idea — a scaled ρ is the obvious follow-up and is
not implemented.

A median win with a 14× tail is an option, not a default. It also
recovered only 3 of the fifteen, against the displaced start's 13, so
it is not the remedy for the problem that motivated the work either.

The implementation is guarded: if the warm-up does not reduce the merit
it restores the original point and reports zero iterations, so enabling
it can never cost more than the evaluations it spent. It also cannot
rescue a NaN start — like KRONOS's own, it breaks on the first
non-finite gradient and keeps the last good iterate, and the last good
iterate of a run that started at NaN is NaN. That is what the
sanitisation step in `ConditionedStartTnlp` is for, and it runs
*before* the displacement for the same reason.

## Why the retry is rung 3 and not rung 1

The [second-opinion ladder](../docs/src/troubleshooting.md) probes three
independent things, in increasing order of how much they disturb the
run:

1. `feral_scaling=mc64` — numerical diversity (does the verdict survive
   different, equally backward-stable linear algebra?).
2. `mu_strategy=adaptive` — algorithmic diversity (does it survive a
   different barrier trajectory?).
3. `start_point_perturbation=1e-2` — the point the trajectory starts
   from.

Rung 3 is last because it is the only one that changes the *question*
rather than the method: rungs 1 and 2 answer "is this the same problem
solved differently", rung 3 answers "is this a different problem". A
user who supplied a considered initial guess is entitled to have it
used, so displacing it is the last thing tried and never the first.

Rungs are **not** cumulative — each restores every earlier knob to
baseline first. That is load-bearing and gh#524 (`cresc4`) is why:
stacking `mu_strategy=adaptive` on top of MC64 loses the fix.

Rung 3 also fires on `Invalid_Number_Detected`, which rungs 1 and 2 do
not: a NaN at the starting point is not a statement about scaling or
about the barrier schedule, and re-running the same evaluation with
different linear algebra reproduces it exactly. This is the only case
where the ladder opens with a single rung.

## Why the audit reports zero rows, not rank

`degeneracy.rs` reports identically zero Jacobian rows and columns
rather than estimating rank. An SVD is not affordable to run
speculatively on every failed solve, and it would be the wrong tool
anyway: the degeneracy that shows up in this corpus is structural and
exact, not a small singular value. Consequences worth stating:

- **Absence of the finding is not a clean bill of health.** A
  full-rank-looking Jacobian can still be numerically rank-deficient.
- **Structural absence is not a finding.** Only a column the model
  *declared* and then evaluated to zero is reported. Otherwise every
  sparse model reports every variable it does not use.
- **A non-finite entry counts as nonzero**, so the two diagnostics
  never contradict each other — a NaN Jacobian entry is reported by the
  audit, not silently as a zero row.
- Out-of-range and negative indices are skipped, not panicked on. This
  runs on the failure path, where the model is already known to be
  misbehaving, and a panic there would replace a diagnosis with a crash.

The audit runs on the user's own TNLP, before presolve, elimination,
scaling and the counting wrapper. Naming `x[3]` of a presolved model
points at a *neighbouring* variable's answer — the gh#450 failure mode,
plausible and wrong. Going around the counting wrapper also keeps the
reported eval counts the solver's rather than the diagnosis's.

## Trajectory check

Per `CLAUDE.md` this is a trajectory change, so
`scripts/sweep-fixtures.sh` was run against a baseline built at
`f6231f40`, both legs, whole corpus: **142 fixture-legs each, empty
diff**.

The empty diff is evidence *because* the corpus exercises the new code:
22 of those 142 legs end `Infeasible_Problem_Detected`, so rung 3 ran
on all 22 and promoted none of them. The fixtures that assert
infeasibility are genuinely infeasible and the new rung does not
manufacture false promotions from them.

What the corpus cannot tell you is anything about magnitude on large
degenerate models — every fixture is small. The measurements above are
the substitute, and they are on a different corpus, which is a real
gap.

## What rung 3 costs the models that already ship in `benchmarks/`

The corpus rung 3 was designed against is degenerate by construction.
The existing benchmark corpus is not, so the honest question is what the
new rung costs there, and the honest answer is: it fires on the models
that were already failing, and nothing else.

Seven entries in `benchmarks/BENCHMARK_REPORT.json` end in a status that
opens the ladder. Six were regenerated and re-measured against a
baseline binary built at `f6231f40` (`iprob`/`model8` via
`benchmarks/lp/generate_nl.py --meszaros-only`; `cresc100`, `discs`,
`himmelbj`, `launch` from Vanderbei's `.mod` sources per
`benchmarks/vanderbei/README.md`). Iteration counts, which are
deterministic where wall-clock is not:

| model | engine | base solve(s) | + rung 3 | wall-clock |
|---|---|---|---|---|
| `cresc100` | nlp | 1473, 3000, 780, 2711 | +198 | 15.25 → 15.42 s |
| `discs`    | nlp | 73, 73, 106 | +126 | 0.233 → 0.299 s |
| `launch`   | nlp | 90, 90, 99  | +59  | 0.151 → 0.194 s |
| `himmelbj` | nlp | 63          | +62  | 0.043 → 0.055 s |
| `iprob`    | cvx-lp | 15 | — | 0.423 → 0.412 s |
| `model8`   | — (presolve-certified) | 0 | — | 0.043 → 0.040 s |

**+0.28 s on 16.14 s across the six, or +1.8 %, and no promotions** — every
verdict is unchanged. That is the expected shape: none of these models
has a degenerate *start*, which is the only thing rung 3 addresses. A
run with `infeasibility_perturbed_start_retry=no` reproduces the
baseline timings everywhere, so the whole delta is rung 3 and none of it
is a change to the solve itself.

Two of the six never reach the ladder at all, for reasons worth stating
because they bound how much of the corpus the new rung can ever touch:

- `model8` is **presolve-certified** infeasible, and the ladder is gated
  on `presolve_certified.is_none()`. A certificate is not a numerical
  verdict and there is nothing for a second opinion to overturn.
- `iprob` routes to the **convex LP interior point**, and a model that
  routes to `pounce-convex` never reaches `optimize_tnlp`, where the
  ladder lives (`main.rs`, the comment at the `solver_selection`
  dispatch). Its infeasibility is numerically determined, not certified,
  and it *still* runs zero rungs. The ladder is an NLP-arm feature.

The seventh, `gaslib40_dynamic`, could not be regenerated: its generator
imports `gas_net` as an editable install pointing at a checkout that no
longer exists on this machine. It reaches the ladder like the other four
NLP models, so it is expected to carry the same shape of cost, but that
is inference and not a measurement.

## A non-promoted rung must not replace the solution it rejected

Measuring the six above turned up a defect, and it is older than this
branch.

Every rung is a full `optimize_tnlp` through the same TNLP, so every
rung overwrites the captured `(x, lambda)` — both the IPM's
`on_converged` capture and `CountingTnlp::finalize_solution`'s.
`resolve_scaling_retry_outcome` puts `status` and the statistics back
when nothing promotes, but it never had a handle on the solution
vectors. The `.sol` therefore shipped the original verdict over the
*last non-promoted rung's* iterate: a point the solver had just decided
not to believe, written out as the answer, under a status line that was
identical either way. Nothing downstream that checks status could see
it.

It reproduces on the pre-branch binary for rungs 1–2 (`cresc100`,
`discs`, `launch`: the `.sol` differs from the same solve with
`feral_infeasibility_scaling_retry=no infeasibility_mu_strategy_retry=no`),
and this branch extended it to the `Invalid_Number_Detected` class,
where `himmelbj` had previously run zero rungs and so had nothing to
leak.

The fix snapshots both captures before the ladder and restores them when
no rung promotes; on promotion the promoted rung's capture is kept,
which is what should ship. Pinned by
`a_non_promoted_second_opinion_does_not_replace_the_solution_it_rejected`
in `crates/pounce-cli/tests/issue_508_infeasibility_gap_status.rs`,
which asserts the whole `.sol` is byte-identical to the ladder-disabled
run. Mutation-checked: on the pre-fix binary that fixture's two runs
differ.

With the fix in place, all four ladder-running models produce a `.sol`
byte-identical to their own ladder-disabled run, and the ladder-disabled
runs are byte-identical across the two binaries — i.e. the branch
changes no trajectory on this corpus, only what gets reported when a
second opinion is declined.
