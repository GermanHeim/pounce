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

**This regression is not fixed here, and per CLAUDE.md it needs an owner.**
gh#850 is closed by making it *visible* — the sweep now prints
`start_point_perturbation=1e-2@Restoration_Failed/131,tot=185` on that line, so
the next reader meets the fact rather than a 2× win. The underlying question —
why `2c4f25f1` cost the base solver this model, and whether the refinement
should be restored for restoration sub-problems — is separate work on a
different commit and belongs in its own issue.

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
