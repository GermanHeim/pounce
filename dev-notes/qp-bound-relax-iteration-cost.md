# The QP suite's +515 iterations, and who owns them

`4c02817d` — "Apply bound_relax_factor on the convex arm too" (Refs #744,
#745) — raised POUNCE's iteration counts across the 138-problem
Maros–Mészáros QP suite. This note is the record CLAUDE.md asks for: *a
measured regression that gets recorded as "an accepted cost of the fix" needs
an issue and an owner. Without one it is indistinguishable from noise to the
next reader.* The issue is gh#760; this note is the number.

**It is not a revert request.** The evidence below says the commit is right
and the cost is the honest price of correctness.

## What moved

`benchmarks/qp` (138 Maros–Mészáros QPs), before vs after, same machine:

* 138/138 solved both ways, **no status flips**
* total iterations **2633 → 3148 (+515)**; 31 models worse, 11 better
* wall **354.7 s → 382.0 s (+7.7 %)**

Worst movers, as originally measured:

| model | iterations | Δt |
|---|---|---|
| QSCFXM3 | 38 → 168 | +1.47 s |
| QSCFXM2 | 35 → 145 | +0.82 s |
| QSCFXM1 | 30 → 131 | +0.38 s |
| Q25FV47 | 41 → 142 | +9.20 s |
| STADAT1 | 22 → 34 | +4.08 s |

Bisected to `4c02817d` over the 588-commit window `fa20c200..4bd7e59c` (8
steps, both endpoints verified).

## Attribution, re-verified

The attribution does not need the old binary. `bound_relax_factor=0` on a
current build restores the pre-commit counts, because zero is exactly what the
convex extractors used to read. Re-run on `fdea82b5` (2026-08-29), same host:

| model | default (relaxed) | `bound_relax_factor=0` | as first measured |
|---|---|---|---|
| QSCFXM1 | 131 | 30 | 30 → 131 |
| QSCFXM2 | 145 | 35 | 35 → 145 |
| QSCFXM3 | 168 | 38 | 38 → 168 |
| Q25FV47 | 120 | 35 | 41 → 142 |
| STADAT1 | 92 | 53 | 22 → 34 |

On the QSCFXM family the zeroed column reproduces the pre-commit counts *to
the iteration*. So a reader who finds the QP suite slower than an old report
and reaches for `git bisect` can stop here: run it once with
`bound_relax_factor=0` and compare.

`Q25FV47` and `STADAT1` are the two that no longer land on their 2026-08-23
numbers, and the zeroed column is how you can tell that is a different
question. Both moved in **both** columns — `Q25FV47` 41 → 35 with the
relaxation off, `STADAT1` 22 → 53 — so what moved them is not the relaxation.
Both still carry the relaxation cost this note is about (3.4× and 1.7×).

What did move them is `d18c289e`, "escalate the primal (x,x) regularization on
wrong inertia", and it is a **speed-up that shows up as an iteration count
going the wrong way** — the one shape a trajectory sweep reads backwards.
Built at that commit and its parent, one problem each way:

| | `d18c289e^` | `d18c289e` |
|---|---|---|
| STADAT1 | 34 it / 4.89 s | 92 it / **0.72 s** |
| Q25FV47 | 142 it / 10.96 s | 120 it / **3.20 s** |
| STADAT2 (control) | 18 it / 0.16 s | 18 it / 0.16 s |

The parent reproduces the committed baseline to the iteration, so nothing
else in the 245-commit window contributes.

STADAT1 is the one to read. It is the same size as `STADAT2`/`STADAT3`, which
did not move at all, and it used to cost **0.127 s per iteration against their
0.0094 s** — thirteen times its own siblings. It now costs 0.0074 s, in line
with them. That is `d18c289e`'s subject exactly: the escalation loop was
answering a wrong-inertia deficit in the `(x,x)` block by escalating `delta_c`
and `(z,z)`, which cannot repair it, so it refactored up to its whole try
budget every iteration — 4.19 tries per iteration on that commit's own
instrumentation. Thirty-four iterations of mis-targeted regularization is not
a better trajectory than 92 honest ones; the old count was low because each
step was doing several factorizations and biasing the equality residual.

Suite-wide the same run shows it: of the 97 problems over 0.05 s, the median
time ratio is 1.00×, **eight are faster by more than 20 % and none is slower
by more than 20 %** (worst 1.12×, noise).

## Why the cost is right

The convex arm used to read the `.nl` bounds verbatim while the NLP arm
relaxed them, so one binary on one file solved two different models depending
on `solver_selection`. `4c02817d` closed that. The cheap old iteration counts
were the cost of solving an easier, wrong model.

Comparing the two arms on current `main` settles it. The NLP arm, which has
always relaxed, is **slower than the convex arm now is**:

| model | convex (`auto`) | NLP (`solver_selection=nlp`) |
|---|---|---|
| QSCFXM3 | 168 | 282 |
| QSCFXM1 | 131 | 198 |
| Q25FV47 | 142 | 422 |

(`make -C benchmarks qp-convex-run`, which runs the same `.nl` twice through
one binary at `solver_selection=qp-ipm` and `=nlp`.)

Objectives now agree between the arms to 12 significant digits (QSCFXM3
`3.0816354268674165e+07` vs `3.0816354268676057e+07`). Before the commit they
disagreed in the first two digits on degenerate models — LISWET1 returned
36.1224 against the Ipopt-MA57 reference's 27.1221.

So the convex arm is still the fastest route to the right answer. It simply
stopped being a fast route to a different one.

## How it got past review, and what actually failed

`4c02817d`'s own message flags the risk and names the gap:

> This is a trajectory change on the convex path, not the NLP path, so
> `scripts/sweep-fixtures.sh` does not cover it; the corpus comparison above
> is the equivalent evidence.

The substituted evidence was **objective parity across the convex corpus**,
which cannot see an iteration-count regression by construction. That is the
gh#544 blind spot — "the right answer, slowly" — and it is the entire reason
the fixture sweep asserts trajectory.

The premise was also wrong: the sweep was never blind to the convex arm. Both
legs run at the default `solver_selection=auto`, which routes to the most
specialized engine available, so most of the corpus never touches the NLP arm
at all (42 of 79 fixtures as of `fdea82b5`). Run across `4c02817d` with the
unmodified script, **52 fixture-legs move**, including a status flip:

```
lbfgs scaled_feasible_a  MaximumIterationsExceeded it=199 -> SolveSucceeded it=69
exact feasible_x0_wide_scale                it=80  -> it=32
exact lp_degen2                             it=15  -> it=20
exact presolve_float_trap                   it=0   -> it=7
```

The tool was fine. The reasoning about the tool is what shipped, and
`scripts/sweep-fixtures.sh`'s header and CLAUDE.md now both say so in as many
words (gh#761), so the same sentence cannot be written again.

## The part the sweep genuinely could not tell you — and now can

A moved convex line is a signal to go measure `benchmarks/qp`. It is not a
bound on what you will find there, and in 2026-08 the corpus could not have
predicted the magnitude, because **no convex fixture in it was one on which
the relaxation is expensive.**

Measured on `fdea82b5`, sweeping the whole corpus twice — default versus
`bound_relax_factor=0` — every convex line that moves at all:

| fixture | relaxed | `bound_relax_factor=0` |
|---|---|---|
| `lp_degen2` (534 cols, the largest well-posed convex fixture) | 18 | 15 |
| `feasible_x0_sentinel_bound` | 27 | 25 |
| `feasible_x0_extreme_row` | 32 | 33 |
| `scaled_feasible_b` | 44 | 47 |
| `qcqp_ball` | 12 | 17 |
| `lp_afiro`, `convex_qp_share1b`, `lp_israel`, `lp_share1b` | unmoved | unmoved |

Tens of percent, in both directions, plus two ill-scaled stress fixtures
(`scaled_feasible_a` 69 → 199, `feasible_x0_wide_scale` 32 → 80) whose numbers
are about their scaling rather than about the relaxation. A reviewer reading
that diff learns "small and mixed" — true of the corpus, false of the suite.

`crates/pounce-cli/tests/fixtures/convex_qp_qscfxm1.nl` is the missing row.
Same binary, same measurement:

| fixture | relaxed | `bound_relax_factor=0` |
|---|---|---|
| `convex_qp_qscfxm1` | 131 | 30 |

4.4×, on both sweep legs — the suite's own signature rather than a scaled-down
analogue of it. It is `QSCFXM1`, the cheapest member of the family the
benchmark measured (457 columns, 0.40 s per leg) against `QSCFXM3`'s 1371
columns and 1.7 s; the sweep's wall time goes 10.9 s → 11.9 s for it (three
runs each way, spread under 0.3 s).

`crates/pounce-cli/tests/issue_760_convex_bound_relax_magnitude.rs` pins the
routing, the dimensions, the published DOC 97/6 optimum, and the ratio — never
an absolute count, which is the sweep's job to measure and the most
platform-sensitive number in this repository. Mutation-checked: setting the
convex arm's relax factor back to zero (i.e. reverting `4c02817d`) takes the
ratio to 1.00× and two of the three tests go red.

This closes the *class*, not the whole gap. `QSCFXM1` is 457 columns; nothing
in the corpus is within two orders of magnitude of `BOYD2` (93 263 columns), so
magnitude on models that size is still only knowable by running
`benchmarks/qp`. See `dev-notes/convex-fixture-corpus.md`.

## The committed baseline, refreshed

`benchmarks/qp/pounce.json` is gitignored — the durable record is
`benchmarks/BENCHMARK_REPORT.md`. It was already post-relaxation before this
note existed: `e84f4259` regenerated it at `32fea00e`, downstream of
`4c02817d`, replacing the pre-fix report generated at `a798ae1b` on
2026-08-22. gh#760's second follow-up — "refresh the committed QP baseline so
the new counts are the reference" — was satisfied by that, on paper.

It is now satisfied by measurement instead. The suite was re-run on this
branch (`make -C benchmarks qp-rerun`, same host, same committed
`ipopt_ma57.json` reference; the report stamps the exact commit in its own
provenance table), because the spot checks above turned up two problems whose
committed counts the current tree no longer produces. Whole-suite delta
against the 2026-08-23 run:

* 138/138 solved both ways, **no status flips, no objective changes**
* total iterations **3148 → 3164 (+16)**; six models moved, none of the others
  by a single iteration
* total solve time 358.3 s → 354.9 s

| model | 2026-08-23 | current |
|---|---|---|
| STADAT1 | 34 | 92 |
| Q25FV47 | 142 | 120 |
| QSHIP12S | 38 | 30 |
| QSHIP08S | 23 | 18 |
| QSTANDAT | 36 | 31 |
| QSIERRA | 41 | 39 |

In the report's QP table that is mean iterations 22.3 → 22.4 and total time
102.79 s → 93.45 s over the 133 commonly-solved problems — **the suite got
faster while its iteration total rose**, which is `d18c289e` and is explained
above, not `4c02817d` and not the bound relaxation.

That is the reason to read this table by time and not only by iterations. Six
models moved and no verdict did, in a six-day window of 245 commits; the two
that moved most are the two that got several times faster. A reader who
diffs only the iteration column here sees a regression that is not there.

## Related

* gh#760 — this issue; the measurement and the bisect.
* gh#744 / gh#745 — the defect `4c02817d` fixed.
* gh#761 — the engine column on each sweep line, and the corrected coverage
  prose in `scripts/sweep-fixtures.sh` and CLAUDE.md.
* gh#544 / gh#592 — "the right answer, slowly", and why the sweep asserts
  trajectory: `dev-notes/trajectory-regressions-and-the-fixture-sweep.md`.
* gh#690 — the same population failure one class over:
  `dev-notes/convex-fixture-corpus.md`.
