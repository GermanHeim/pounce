# A promoted second opinion used to read as a speed-up (gh#850)

`scripts/sweep-fixtures.sh` could not see a second-opinion ladder promotion.
When the base solve fails and a rung recovers it, the JSON report's `status`
and `statistics.iteration_count` both become the **promoted rung's**, and
nothing else said the base solver had failed — so a fixture that *lost* its
baseline solve and is now only rescued by a retry read in the diff as a large
improvement.

This is the same shape of invisibility the engine column was added to close
(gh#760). CLAUDE.md already states the rule for engines: "Status, objective and
iteration count can all be unchanged while a model silently changes arms … so a
routing regression used to leave no trace in the diff." A ladder promotion is
that shape.

It is worse than a gap in the evidence. `scripts/sweep-fixtures.sh` is the
repo's primary trajectory guard and CLAUDE.md makes it the *required* evidence
for a trajectory change, so a guard that converts a lost solve into a recorded
win produces positive evidence for the wrong conclusion.

## What was fixed

- `SecondOpinionOutcome` now carries `base_status`, `base_iteration_count` and
  `rung_iteration_counts`, so the base solve's verdict and cost survive a
  promotion instead of being overwritten by the rung's.
- The JSON report gained a `second_opinion` block (additive; absent entirely
  when the verdict opened no ladder, so its *presence* is itself the signal).
- `scripts/sweep-fixtures.sh` gained a `2nd=` column, built from that block:
  `-`, `kept(n),tot=N`, or `<rung>@<base status>/<base iters>,tot=N`.

Pinned by `crates/pounce-cli/tests/issue850_second_opinion_is_recorded.rs`.

## What the new column immediately revealed — and who owns it

Two fixtures in the corpus are solved **only** by a ladder rung. Measured at
the commit that added the column (`infeasibility_perturbed_start_retry=no`
turns the rung off):

| fixture | defaults | rung off |
|---|---|---|
| `square_flowsheet_resto` | `SolveSucceeded`, 54 iters | **`RestorationFailed`, 131 iters** |
| `degenerate_start_hs008` | `SolveSucceeded`, 5 iters | **`InfeasibleProblemDetected`, 7 iters** |

`square_flowsheet_resto` is the one gh#850 reports, and it is a **regression**,
not merely a fixture that has always needed the rung:

| | status | iters | final constr viol |
|---|---|---|---|
| `v0.10.0`, defaults | `SolveSucceeded` | 116 | 4.2e-10 |
| HEAD, defaults | `SolveSucceeded` | 54 | 3.9e-10 |
| HEAD, rung off | `RestorationFailed` | 131 | 6.7e-4 |

`v0.10.0` does not have `infeasibility_perturbed_start_retry` at all — it
rejects the option with `OPTION_INVALID` — so that 116 is the *base solver*
converging, and HEAD's base solver no longer does. The rung that saves it was
added in the same release window. gh#850 bisects the loss to `2c4f25f1`
("perf(feral): wire increase_quality, and turn the backend refinement off for
the IPM (gh#698 obs 5)").

## The regression itself, and what it turned out to be

Made visible first, then fixed. The visible form is what the sweep now prints
on that line — `start_point_perturbation=1e-2@Restoration_Failed/131,tot=185`
— and it is what let the second, worse instance be found at all.

**The lbfgs leg was worse than the reported exact one and nothing was rescuing
it.** With the column in place, `lbfgs square_flowsheet_resto` reads
`MaximumIterationsExceeded`, `it=3000`, `2nd=-`: it ran to the cap, and no
ladder opened because that verdict does not trigger one. The exact leg at
least came back `SolveSucceeded`.

**Which half of `2c4f25f1` did it.** That commit does two things — wires
`increase_quality`, and turns the backend refinement off — and only the first
is responsible. Measured, one binary, the rung switchable:

| `increase_quality` | `feral_refine` | exact leg |
|---|---|---|
| on (0.11 default) | off (default) | `RestorationFailed`, 131 |
| on | **on** | `RestorationFailed`, 131 |
| **off** | off | **`Optimal`, 99** |
| off | on | `Optimal`, 99 |

Refinement makes no difference to this model in either direction, so the
68.9 s → 18.8 s win that `2c4f25f1` bought — which lives entirely in
`refine = false` — is untouched by the fix.

**Why the rung costs a solve.** Ipopt calls `IncreaseQuality` when
`PdFullSpaceSolver`'s refinement stalls, and MA57 answers by raising `pivtol`
toward `pivtolmax`: strictly more conservative each time, so keeping it raised
for the rest of the solve can only make the factorization safer. FERAL's ladder
changes *which pivots are taken*, which is a lateral move in trajectory terms,
and it persists the same way — across every later factorization, including a
restoration sub-solve's. On this fixture it fires exactly twice **in the base solve**: once in the
main loop, and once inside restoration at `76r`. Process-wide the exact leg
fires six times and the lbfgs leg twenty-five per solve — see the gh#857
section below, which also explains why the second firing is invisible in the
printed `q` column.

**There is no firing-count policy that separates the cases.** `deb7` and
`square_flowsheet_resto` each fire the rung exactly twice on their exact legs;
one gains 16% of its iterations and the other loses its verdict. So the fix is
a lever, not a cap: `feral_increase_quality` stays **on**, because it also buys
things nothing else supplies, and `=no` recovers a problem the rung costs.

**The trade, from the sweep — 18 fixture-legs move.** Within the CLI corpus the
rung costs a *verdict* twice and buys *iterations* five times:

| | with the rung | without |
|---|---|---|
| `exact square_flowsheet_resto` | `RestorationFailed`/131, rescued at 185 total | **`Optimal`/99, no ladder** |
| `lbfgs square_flowsheet_resto` | `MaximumIterationsExceeded`/3000 — **recovered automatically since gh#857**: `Optimal`/178, 3178 total | **`Optimal`/178** |
| `exact deb7` | 147 | 171 |
| `exact pooling_rt2stp` | 109 | 128 |
| `lbfgs eigena2` | 186 | 202 (`ErrorInStepComputation` either way) |
| `lbfgs pooling_rt2stp` | 295 | **273** |
| three ladder `tot=` counts on infeasible fixtures | | +3 ‥ +24 |

On that evidence alone the rung looks like a bad trade, and the first draft of
this work flipped the default off. **The workspace suite refuted it**, which is
worth recording because the fixture corpus could not:
`pounce-rs/tests/watchdog_trial_is_not_a_divergence_verdict.rs`'s 12-variable
model ends `SolvedToAcceptableLevel` at `obj = 3.7e-6` **with** the rung and at
`obj = 3.42` against `f* = 0` **without** it. That is a wrong-ish answer under a
success-shaped status — worse than an honest cap failure — and the 158-leg sweep
is blind to it because the model is not a CLI fixture. The same lesson as the
corpus notes in CLAUDE.md, one layer out.

**And nothing separates the two sides.** Measured with a process-global firing
cap, the rung fires exactly twice on `square_flowsheet_resto` — once in the main
solve at iteration 25 and once inside restoration at `76r` — and allowing *only
the first* still loses the leg, so declining it for the restoration sub-solve
would not help. Nor does a count: `deb7` and `square_flowsheet_resto` each fire
it exactly twice on their exact legs, one gaining 16% of its iterations and the
other losing its verdict.

**So the default stands and the rung gets a lever, not a flip.**
`feral_increase_quality=no` recovers both legs of `square_flowsheet_resto`
cleanly (99 and 178 iterations), and is the documented recovery for a model this
rung costs. Pinned by
`crates/pounce-cli/tests/issue850_increase_quality_regression.rs`, which asserts
the trade in both directions.

**What a real fix looked like it needed — and did not.** The reading at the
time was a *revertible* escalation, one that does not govern every later
factorization including a restoration sub-solve's, which FERAL's `quality_level`
could not express because it only ratcheted up. That was filed as
**jkitchin/feral#192**, it landed as `reset_quality`, and **it does not fix
this** — see "The feral route, and why it is closed rather than pending" below,
which is the measurement. The remedy that works is a re-solve
(`feral_increase_quality_retry`, gh#857), not a re-baselining.

**One thing checked and worth knowing:** gh#590's badly-scaled LP grid
(`issue_590_primal_noise_floor_component`, data scale `1e10` and `1e11`, six
seeds), which `2c4f25f1` cites as needing the escalation once refinement came
off, passes with the rung off — so that particular justification no longer
binds. The perf claim that commit measured lives in `feral_refine`, which none
of this touches.

The cost is understated on the same lines, and the `tot=` field is what says
so. `square_flowsheet_resto` really costs `131 + 54 = 185`, 3.4× its reported
`it=54`; `degenerate_start_hs008` costs 30 against a reported 5; and among the
fixtures where the ladder runs and promotes *nothing*,
`issue_508_infeasible_gap_1em4` costs 982 against a reported 441. Fifteen
fixture-legs carry a `2nd=` entry, and every one of them was previously
reporting a fraction of its true cost.

## Note for the next sweep baseline

Adding the column moves **every** line in the sweep output, so a diff taken
across this commit is not comparable field-by-field with an older baseline.
Re-baseline against a binary built at or after it.

## gh#857 — the losing direction recovers itself, and the count that made it possible

Two things shipped under gh#857. They are separable and were measured
separately, which is the only reason the second one is trustworthy.

### Part A: `quality_escalations`, a statistic

`increase_quality` left **no trace in any report**. Two runs could agree on
status, objective, iteration count and engine and still have factorized the
KKT systems along different pivot sequences — the same reporting gap gh#850
closed for ladder verdicts and gh#760 for engine routing, one layer down.
`SolveStatistics::quality_escalations` counts the escalations the backend
*accepted*, is carried in the JSON report and the console summary (printed
only when nonzero, so a non-escalating summary is byte-identical to what it
was), and is the sweep's new `q=` column.

The counter is **shared with the restoration sub-solve**, and that is the
part that needed plumbing rather than a field. `PdFullSpaceSolver` already
appended a `q` to the info-string column, so counting the printed `q`s looks
like it would do. It would report half: `square_flowsheet_resto`'s base solve
prints **one** `q`, on the row labelled 26, while **two** escalations
happened — the second at `76r`, inside restoration, whose rows carry no info
column at all. `the_restoration_escalation_is_counted_though_it_never_prints`
in `crates/pounce-cli/tests/issue857_quality_escalations_are_reported.rs` is
what fails if someone narrows the counter back to the main loop; removing the
counter wiring turns 4 of that file's 6 tests red.

Cross-checked three ways before being trusted: it reproduces the base-solve
`2` this note derived above with a process-global firing cap; it reconciles
with a separately instrumented process-wide count (exact 2 + 4 = 6, lbfgs
25 + 25 = 50 pounce-side); and it exceeds the printed `q`s by exactly the
restoration firing.

**Scope correction to this note and to the option text.** "Fires exactly
twice" is a *base-solve* figure and is correct as such. Process-wide the
exact leg escalates **six** times — the second-opinion rung is a whole second
solve — and the lbfgs leg reaches **25 per solve**. The old text did not say
which, which made it read as a property of the run.

**Corpus coverage, measured.** 13 of 158 fixture-legs escalate at all:

```
exact  deb7                           q=2     exact  square_flowsheet_resto   q=4
exact  infeasible_square_scaled_1em4  q=4     lbfgs  eigena2                  q=9
exact  issue_508_infeasible_gap_1em2  q=1     lbfgs  eigmaxa                  q=2
exact  issue_508_infeasible_gap_1em4  q=3     lbfgs  infeasible_sq…_1em4      q=3
exact  pooling_rt2stp                 q=2     lbfgs  issue_508…_gap_1em2      q=5
                                              lbfgs  issue_508…_gap_1em4      q=1
                                              lbfgs  pooling_rt2stp           q=9
                                              lbfgs  square_flowsheet_resto   q=25
```

Swept against `73e064c1`, Part A moves **nothing**: all 158 lines identical
field-for-field once the new column is stripped. It is report-only by
construction and by measurement.

### Part B: rung 4, gated on that count

`feral_increase_quality_retry` (default on) re-solves once with
`feral_increase_quality=no` when a solve ends `Restoration_Failed`,
`Maximum_Iterations_Exceeded` or `Infeasible_Problem_Detected` **and**
escalated at least once.

The `Maximum_Iterations_Exceeded` half required opening a trigger that
`SecondOpinionTrigger::for_status` deliberately refused, on the sound
reasoning that the answer to a budget exit is a bigger budget. That reasoning
has exactly one exception and this is it: an escalation persists across every
later factorization, so the wall may be the *escalated trajectory's* rather
than the model's, and a bigger budget re-runs the same wall. 178 iterations
were available on the un-escalated path.

**The count is what makes it affordable, and it is a gate, not a threshold.**
`for_status` now names a trigger for every budget exit; `second_opinion_rungs`
drops the rung when the count is zero, and the driver returns on an empty rung
list *before* it narrates, so a non-escalating capped run is byte-identical to
its pre-gh#857 self. It cannot be a threshold because `deb7` and
`square_flowsheet_resto`'s base solve each escalate exactly twice, one gaining
and one losing — the finding this note already records. Only the verdict
separates them, and `deb7`'s is `Optimal`.

**Appended, never prepended.** On a `Restoration_Failed` the gh#815
displacement rung runs and promotes first. `square_flowsheet_resto`'s exact
leg is that case, and reaches the same answer at the same 185 iterations it
did before.

**The sweep: exactly one line moves in 158.**

```
- lbfgs square_flowsheet_resto nlp MaximumIterationsExceeded it=3000 q=25 2nd=-
+ lbfgs square_flowsheet_resto nlp SolveSucceeded            it=178  q=0
    2nd=feral_increase_quality=no@Maximum_Iterations_Exceeded/3000,tot=3178
```

That is the whole collateral of the budget-exit half: of the 13 escalating
legs, this is the only one whose verdict is an unrecovered
`Restoration_Failed` or budget exit. The row in the table above that read
`lbfgs square_flowsheet_resto | MaximumIterationsExceeded/3000 | Optimal/178`
is now `Optimal/178 automatically, 3178 total`.

### The third trigger, and the platform that made it necessary

`Infeasible_Problem_Detected` joined the trigger set after CI disagreed with
the development machine about this fixture. On linux/x86_64 the same
limited-memory leg spends the same 3000 iterations and takes the same 25
escalations and then exits `Infeasible_Problem_Detected` rather than at the
cap — a **false infeasibility verdict on a feasible model**, which is a
strictly worse failure mode than a budget exit because it is a wrong answer
reported as a verdict. Rungs 1–3 all run on it (that status has opened a
ladder since long before gh#857) and all three fail: `mc64` returns
`Restoration_Failed`, `mu_strategy=adaptive` and the perturbed start return
`Infeasible_Problem_Detected`. Rung 4 recovers it, and did not open, because
the trigger set named only the two statuses macOS produces.

The lesson is the branch rule in a new dimension: a corpus can be uniform in
the *platform* it was measured on, and a gate written from that corpus names
the shapes one platform's arithmetic produces. Nothing about the defect is
platform-specific — it is the same escalation on the same trajectory — only
which terminal status the walk into the wall ends at.

**That trigger is not free, and the six lines it moves are the price.** All
six are infeasibility fixtures that escalated, and all six are models that
really are infeasible, so the rung confirms the verdict at the cost of one
more solve:

```
- exact infeasible_square_scaled_1em4 InfeasibleProblemDetected it=17  q=4 2nd=kept(3),tot=61
+ exact infeasible_square_scaled_1em4 InfeasibleProblemDetected it=17  q=4 2nd=kept(4),tot=78
- exact issue_508_infeasible_gap_1em2 InfeasibleProblemDetected it=114 q=1 2nd=kept(3),tot=290
+ exact issue_508_infeasible_gap_1em2 InfeasibleProblemDetected it=114 q=1 2nd=kept(4),tot=404
- exact issue_508_infeasible_gap_1em4 InfeasibleProblemDetected it=441 q=3 2nd=kept(3),tot=982
+ exact issue_508_infeasible_gap_1em4 InfeasibleProblemDetected it=441 q=3 2nd=kept(4),tot=1423
```

plus the same three on the `lbfgs` leg. No status, objective, `it=`, `q=` or
engine moves on any of them — only the rung count and the ladder total.

**The gate is what bounds it, and both of its branches are in the corpus.** Of
the eight NLP-arm infeasibility fixture-legs, four escalated and take the rung;
`infeasible_equalities`, `issue_372_infeasible_bounds` and
`degenerate_start_infeasible` never escalated and are untouched at three rungs.
Both branches are pinned in
`crates/pounce-cli/tests/issue857_escalation_gated_quality_rung.rs`.

**The cost, stated plainly.** One extra solve on a run that was already going
to report failure — with one case worth naming, because it is the only
trigger a user induces deliberately: a small `max_iter` on an escalating model
now spends a second budget before reporting.
`feral_increase_quality_retry=no` holds a capped run to exactly the budget it
was given.

**Four existing pins needed that flag, and what they have in common is the
finding worth carrying forward.** `issue850_increase_quality_regression.rs`'s
lbfgs test asserts the base solve runs to the cap, and this rung recovers it.
The other three all wanted *no ladder at all* and each said so by naming one
rung's flag:

| pin | what it wanted | how it said so |
|---|---|---|
| `issue850_second_opinion_is_recorded.rs::the_base_solver_alone_does_not_solve_this_fixture` | a bare `Restoration_Failed` on the exact leg | `infeasibility_perturbed_start_retry=no` |
| `issue_815_restoration_ladder.rs::the_ladder_can_be_switched_off` | the escape hatch, asserting nothing is announced | `infeasibility_perturbed_start_retry=no` |
| `issue_819_restoration_iteration_count.rs::run_to_restoration_failure` | a solve that actually terminates in restoration, to count its `r` rows | `infeasibility_perturbed_start_retry=no` |
| `python/tests/test_second_opinion.py::test_turning_the_whole_ladder_off_restores_upstream_behaviour` | `second_opinion is None`, i.e. no ladder ran | a `LADDER_OFF` dict naming the other three |

Rung 4 opens on `Restoration_Failed` too, so "the base solver alone" quietly
became "the base solver plus one rung", the escape hatch stopped being an
escape hatch, and the run gh#819 measures stopped terminating in restoration.
All four now name every flag.

The Python one is worth a second look, because it did not fail where it was
written. Widening rung 4's trigger to `Infeasible_Problem_Detected` is what
reached it, and it went red on linux/x86_64 while staying green on
macOS/aarch64 — not because the models differ but because *the arithmetic
does*, and rung 4's gate is a measurement. Two sibling tests in that file
wrote the rung list down as three literal strings and had to be re-anchored
on a list derived from a ladder-free base solve's `quality_escalations`. The
general rule: **a pin on the ladder's shape is a pin on a count the platform
gets to choose**, so derive it. `deb7` escalates twice on macOS/aarch64 and
zero times on linux/x86_64 under identical options; the same trap took
`issue857_quality_escalations_are_reported.rs`'s `deb7` arm, which now
asserts the portable claim (an `Optimal` verdict opens no rung, whatever the
count) instead of the count.

**There is no ladder-wide switch, and that is the defect these four expose.**
The ladder is disabled a rung at a time — `feral_infeasibility_scaling_retry`,
`infeasibility_mu_strategy_retry`, `infeasibility_perturbed_start_retry`, and
now `feral_increase_quality_retry` — and every one of their option texts ends
"set to no to keep behaviour bit-for-bit faithful to upstream IPOPT", which
is true of that rung and false of the solver. So a caller who wants upstream's
one-solve-one-verdict behaviour must know four names today and will silently
acquire a fifth rung the next time one is added, exactly as these three pins
did. The pins were fixed by enumeration because that is what exists; a single
`second_opinion=no` master switch would make the next rung free, and is
**not** part of gh#857 — it is a separate change with a permanent option name
attached, recorded here so the next person adding a rung finds the cost
already counted rather than paying it a fifth time.

What the second one tells you is not in the sweep: **the exact leg's
`Restoration_Failed` is recoverable by rung 4 as well as by rung 3.** The sweep
cannot show it, because rung 3 runs first and promotes, which is exactly the
"appended last costs nothing new" property working — but it means the recovery
is redundant on that leg rather than absent, and a future change to rung 3
would not leave the leg uncovered.

A fifth pin, one of gh#857's own, needed it for a different reason worth
separating: `issue857_quality_escalations_are_reported.rs`'s lbfgs test reads
the statistic out of the JSON, and **the JSON carries the promoted solve's
statistics, not the base solve's** — the rule `iteration_count` has always
followed. The recovery rung promotes a solve that by construction escalated
zero times, so a run whose base solve escalated twenty-five times now reports
`quality_escalations = 0`. That is correct and is documented on the field, but
it is a genuine sharp edge for anyone debugging an escalation: the number you
want is the base solve's, and `feral_increase_quality_retry=no` is how you get
it. The rung's own gate is unaffected — the driver reads the base statistics
before any promotion.

Worth noticing rather than editing past: a recovery rung is exactly the shape
of change that makes an existing regression pin go quiet, and all five of
these went *red* only because they asserted a specific number or status. A pin that
had asserted "did not reach `Optimal`" in some looser form would have gone
green and stayed green, which is the gh#544 lesson one level up. Any rung added
later that catches `Restoration_Failed` or `Maximum_Iterations_Exceeded` has
the same five files to check, and the table above is the list.

### Corrections this work owes the text above

- **The scaling rung of FERAL's ladder is unreachable as pounce ships.**
  feral 0.17.0 takes it only when `numeric_params.scaling` is
  `ScalingStrategy::Identity`; pounce's default is `ScalingStrategy::Auto`
  (`crates/pounce-feral/src/lib.rs`). Every escalation a pounce user sees is a
  `pivot_threshold` bump, the first 1e-8 → 1e-6. So "scaling, then pivot
  threshold" describes FERAL, not this repo.
- **A milder ladder cannot be the fix.** On the lbfgs leg every static
  `feral_pivtol` in {1e-6, 3.16e-5, 4.2e-4, 1e-2, 0.5} loses the leg from
  iteration 0; only 1e-8 solves it. The harm is the destination, not the size
  of the step to it — which is why the recovery is a re-solve and not a
  re-baselining. See the section below: that is not an inference, it was
  measured against the real `reset_quality()`.
- **`lbfgs eigena2` is not a win.** 202 → 186 exits
  `ErrorInStepComputation` either way: 16 fewer iterations to the same
  non-answer.
- **`lbfgs pooling_rt2stp` goes the other way.** 273 → 295 — the rung costs
  iterations there. The option text listed it among the legs it buys.

### The feral route, and why it is closed rather than pending

gh#857 was filed "blocked on jkitchin/feral#192", under the diagnosis that the
distinguishing factor is *for how long* to escalate rather than *whether* to.
The remedy landed and refutes the diagnosis, which is worth recording because
the reasoning is reusable:

- `Solver::reset_quality` (feral#192/#193) was plumbed into `pounce-feral` and
  instrumented — **376 escalations against 376 matching resets on one solve**,
  so the mechanism works — and it recovers **neither leg at either
  re-baselining boundary**.
- That agrees with the static-`feral_pivtol` sweep from the other direction:
  every value in {1e-6, 3.16e-5, 4.2e-4, 1e-2, 0.5} loses the lbfgs leg from
  iteration 0 and only 1e-8 solves it. **The harm is the destination, not the
  duration.** A trajectory that has visited the raised threshold once is
  already on the other path, so reverting afterwards reverts nothing. Duration
  was the wrong axis, and a remedy aimed at it could not have worked.
- Taking the bump is separately unattractive: feral at `91ace05` costs the
  *clean* exact leg on its own, `Optimal`/99 → `RestorationFailed`/128 with the
  escalation rung already off (jkitchin/feral#196). pounce stays on 0.17.0.

The general shape: **a fix aimed at the wrong axis of a problem fails
silently — it works exactly as specified and changes nothing.** The cheap
discriminator here existed before the upstream work did (the static-pivtol
sweep) and would have predicted the outcome; it was run afterwards.

### `laptime`, and what the benchmark says about rung 4

gh#857's remaining owed item was re-running `2c4f25f1`'s 126k-KKT measurement,
unrepeated since. On the rung-4 branch, `max_iter=100`, limited-memory,
monotone, single-threaded BLAS:

| `feral_refine` | rep 1 | rep 2 | rep 3 | median |
|---|---|---|---|---|
| `no` (the default `2c4f25f1` set) | 22.95 s | 22.55 s | 22.75 s | **22.75 s** |
| `yes` (pre-0.11 behaviour) | 104.61 s | 103.97 s | 113.10 s | **104.61 s** |

**4.6x**, against the **3.7x** (68.9 s -> 18.8 s) recorded at `2c4f25f1`. The
absolute numbers are higher on both sides because this is a different machine
and a release-with-debuginfo build, which is why the ratio is the thing to
read. All six runs: 100 iterations, `Maximum_Iterations_Exceeded`,
`quality_escalations = 0`, no ladder announced.

The ratio `2c4f25f1` bought is intact. The run also supplies something the
fixture corpus structurally cannot, per this repo's own rule that the corpus
gives no magnitude at benchmark scale: `laptime` exits
`Maximum_Iterations_Exceeded` — rung 4's **new** trigger — with
`quality_escalations = 0`, so the gate declines, no ladder is announced, and
the summary block is byte-identical to its pre-gh#857 self. The largest model
in the repo pays nothing for the new rung, measured rather than argued from
813-variable fixtures.
