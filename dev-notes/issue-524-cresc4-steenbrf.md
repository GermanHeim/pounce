# issue #524 — `cresc4` false infeasibility and `steenbrf` stall

[#524](https://github.com/jkitchin/pounce/issues/524): the two Vanderbei
problems where POUNCE fails and the committed Ipopt-MA57 reference succeeds.
The issue's hypothesis was that they share a cause, because both are fixed by
`mu_strategy=adaptive`.

Outcome: **`cresc4` reproduced, diagnosed and fixed. `steenbrf` did not
reproduce**, and the reason it did not is itself a finding about the benchmark
corpus — see the second half.

All measurements below were taken on `a664dc0` plus the change described here,
FERAL backend, release build.

---

## Working without the corpus

`$POUNCE_BENCH_DATA` is not available in this environment and
`vanderbei.princeton.edu` is not reachable through the agent proxy, so the
reporter's `.nl` files could not be obtained. Both problems were instead
transcribed from their CUTEst `mastsif` SIF sources (reachable at
`https://bitbucket.org/optrove/sif/raw/master/<NAME>.SIF`) into Pyomo and
written to `.nl` with Pyomo's own writer — no AMPL needed.

That is a weaker artifact than the reporter's, and the discipline in
`AGENTS.md` applies: a transcription is only evidence once it reproduces the
reported *signature*, not merely the reported *symptom*. What was checked
before believing it:

- the objective POUNCE reaches on a healthy encoding matches the published
  reference (`cresc4`: `0.87189753860735963` vs `0.8718976`);
- the failure signature matches the issue's table cell for cell — status,
  near-zero objective at exit, and recovery under both `mu_strategy=adaptive`
  and `nlp_scaling_method=none`;
- the transcription machinery was validated end to end on an unrelated control
  (`STEENBRB`, below) that has a published reference number.

Iteration counts still differ from the corpus files, and every claim below is
about the transcribed models. Both transcriptions are committed so they can be
audited against the SIF rather than taken on trust:

- `crates/pounce-cli/tests/fixtures/cresc4.py` regenerates the committed
  fixture `cresc4.nl` byte for byte;
- `dev-notes/issue-524/steenbrf.py` emits `STEENBRF`, and with `--as-b` the
  `STEENBRB` control below.

---

## `cresc4`

### The encoding matters, and that is not noise

The straightforward transcription **solves** — 101 iterations, `0.8718975386`.
The failure only appears under some equally-valid AMPL surface encodings of the
same model. Twelve encodings were generated, varying three things a `.mod`
author chooses freely:

| varied | effect on the verdict |
|---|---|
| variable declaration order | **none** (the `.nl` writer reorders to the ASL convention regardless) |
| constraint declaration order (`is2` first vs `os1` first) | flips it |
| radius term on the RHS (`lhs <= (d+r)^2`) vs folded into the body (`lhs - (d+r)^2 <= 0`) | flips it |

Six of the twelve report `Infeasible_Problem_Detected`. This is worth stating
plainly: on this problem, POUNCE's verdict about whether the feasible set is
empty depends on which order the user typed their constraints in. That is the
real severity of the bug, and it is not visible from a single `.nl`.

Two encodings are used below and both are committed-quality repros;
`crates/pounce-cli/tests/fixtures/cresc4.nl` is the `is2`-first / RHS-form one.

### Signature match

```
                          issue (corpus .nl)        fixture      other encoding
default                   Infeasible_Problem_Det.   same         same
  iterations              74                        61           103
  obj at exit             2.8658120760032343e-09   -9.483023e-10  2.0924048e-09
  constraint violation    5.1235431712319179e-01    4.172438e-01  4.584213e-01
mu_strategy=adaptive      Optimal, 0.87189752899…   0.87189752729 0.87189754551
nlp_scaling_method=none   Optimal                   Optimal       Optimal
```

Iteration counts differ from the corpus file, as expected from a different
`.nl` encoding; everything that identifies the *failure mode* matches — a
verdict of local infeasibility at a near-zero objective with a constraint
violation of roughly a half, recovered by either knob.

### What the trajectory does

Traced on the `os1`-first encoding (`103` iterations), which shows the shape
most clearly; the committed fixture's shorter trace does the same thing in
fewer restoration cycles. Numbers in this section are that encoding's and
should not be quoted against the fixture's.

The main loop repeatedly drives the objective to ~0 — and past it, to
`-6.3e-09` — while the constraint violation stalls. The objective *is* the
crescent's area, so an objective of zero is a degenerate crescent: the solver
is chasing a shape that cannot contain the four points, and the barrier problem
at `mu = 0.1` rewards it for doing so. Restoration is entered six times
(iterations 17r, 64r, 75r, 80r, 84r, 98r), claws the violation down a little
each time (`1.83 → 1.60 → 1.35 → 0.91 → 0.76`), hands back to the main loop,
which immediately spends the gain re-collapsing the crescent.

The last restoration converges properly: over iterations 106r–123r its own dual
infeasibility falls to `2.91e-11` while `inf_pr` sits pinned at exactly
`7.50e-01`. That is a legitimate stationary point *of the feasibility problem*
— restoration did its job. The verdict is wrong because of where the main loop
delivered it, not because restoration misbehaved.

### The actual defect: the retry corroborated itself

The guard that exists for exactly this case fired, and made it worse:

```
EXIT: Converged to a point of local infeasibility. Problem may be infeasible.
pounce: local infeasibility under the current FERAL scaling — re-solving once
        with MC64 before believing it …
pounce: MC64 re-solve did not recover (InfeasibleProblemDetected); keeping the
        original local-infeasibility verdict (now corroborated by a second
        scaling).
```

Diffing the two traces on the committed fixture: the MC64 re-solve is
**character-identical to the original for iterations 0–15**, and diverges at
iteration 16 in the eighth significant digit —

```
orig  16  2.2741476e-08 2.98e+02 6.31e+05  -1.0 5.47e+02  1.6 1.59e-04 8.75e-05h  1
mc64  16  2.2739801e-08 2.98e+02 6.31e+05  -1.0 5.47e+02  1.6 1.59e-04 8.75e-05h  1
```

— which is the exact hypersensitivity signature the guard was written for. It
still lands in the same basin and returns the same verdict.

That is the whole bug in the guard, and it is sharper than "MC64 didn't
perturb anything": MC64 *did* perturb the trajectory, at ULP scale, in the way
`discs.nl` taught us to expect — and it made no difference. `feral_scaling`
varies only the linear algebra, so whether the perturbation escapes the basin
is luck. On `discs.nl` it was lucky. Here it was not, and the CLI reported the
unlucky draw as corroboration, which converts "I re-ran it and nothing changed"
into confidence. That is worse than not having retried: it is the step that
turns a numerical failure into a reported fact.

`nlp_scaling_method=none` also solves `cresc4`, which settles the question the
issue raised — a third scaling *does* disagree, so the two-way agreement was
never about scaling at all.

### Fix

`crates/pounce-cli/src/main.rs`: the retry becomes a two-rung ladder.

1. `feral_scaling=mc64` — numerical diversity (unchanged, still
   `feral_infeasibility_scaling_retry`).
2. `mu_strategy=adaptive` — algorithmic diversity (new,
   `infeasibility_mu_strategy_retry`, default on).

Three properties are deliberate:

- **Rungs are not cumulative.** Rung 2 re-asserts the baseline `feral_scaling`
  before applying `mu_strategy`. Measured on the fixture:
  `mu_strategy=adaptive` solves it, and `mu_strategy=adaptive` *with*
  `feral_scaling=mc64` still reports local infeasibility. A stacked ladder
  would have thrown the fix away — and would have looked correct in a test that
  only checked the final status on the other encoding, where the stacked
  combination happens to work.
- **Promotion is verified, not trusted.** A rung overturns the verdict only on
  `Solve_Succeeded` / `Solved_To_Acceptable_Level`, i.e. the retry's own KKT
  check. Adaptive µ is never assumed to be right; it is only ever allowed to
  produce a point that passes the same convergence test everything else passes.
  So the change cannot manufacture a false *optimal* — the failure direction
  that would be worse than the one being fixed.
- **A no-op rung is skipped**, so nothing is spent re-deriving an answer under
  settings the baseline already had, and the ladder costs zero on runs that
  succeed.

Why automate it at all rather than document it: retrying with a different
barrier strategy is already what IPOPT's documentation tells a user to do when
it reports infeasibility on a problem they believe is feasible. Doing it in the
solver spares the round trip, and — the point of this issue — spares the user
from being told the verdict was corroborated when it was not.

### Known gap, pre-existing and not widened by choice

On a ladder that does *not* promote, `status` and the statistics revert to the
original solve (`resolve_scaling_retry_outcome`, code review L23) but the point
written to the `.sol` is whatever the last rung's solve left in
`nominal_capture`. That was already true when the ladder had one rung; adding a
second only changes which failing point it is. It is a real inconsistency —
original verdict, original iteration count, someone else's `x` — and worth
fixing at the same place gh #508 fixed the console banner, but it is a separate
change and is not attempted here.

### Not fixed here

The ladder is a second opinion, not a cure. The underlying question stands and
is worth its own issue: **why does monotone µ let the main loop trade constraint
violation for a degenerate zero-area crescent, repeatedly, undoing restoration's
progress each time?** The filter line search is accepting steps that collapse
the objective while `inf_pr` stalls; six restoration entries that each get
partially reverted is the shape of a filter that is not holding the ground
restoration bought. `cresc4` is six variables, so that trajectory is fully
inspectable — the issue's own observation, and still the best next step.

---

## `steenbrf` — did not reproduce, and the corpus file is the reason

The transcription of `mastsif/STEENBRF.SIF` **solves cleanly**: 53 iterations,
`Optimal Solution Found`, objective `8991.85`. No stall, nothing like the
reported 3000-iteration grind.

That is not a transcription bug. The control:

- `STEENBRB.SIF` is byte-identical to `STEENBRF.SIF` except for four lines
  (`AM LA(4) LA(4) 0.5`, "half investment cost for arc 4") and the problem name.
- Transcribing it with the same code and un-halving that one coefficient gives
  **`9075.8553865777394`**, against the published reference `9075.855` in
  `benchmarks/vanderbei/cute_table_status.json` — every digit the table
  carries. `STEENBRB.SIF` also records `SOLTN 9098.9319884` in its own header.

So the machinery is right, and mastsif `STEENBRF` has an optimum near `8991`.
But every number in the issue and in the reference table puts the corpus
`steenbrf` two orders of magnitude lower: POUNCE stalls at `397.818`, POUNCE
under adaptive µ reaches `282.678`, Ipopt reaches `1321.652`, and the table's
`ref_obj` is `282.7578`.

A floor argument confirms the two cannot be the same model. In mastsif
`STEENBRF` the objective's linear term alone is `Σ_arcs 0.01 · COST · FLOW`, and
commodity 1 must move 2000 units from node 2 to node 3, whose cheapest path
(arcs 5 and 8) costs 40 — contributing `2000 · 40 · 0.01 = 800` before any of
the other eleven commodities, the cubic congestion term, or the capacity term.
An objective of `282.76` is not reachable.

**Conclusion: the corpus's `vanderbei/nl/steenbrf.nl` is not a transliteration
of CUTEst `STEENBRF`.** Two things follow, both for the maintainer to decide:

1. The `steenbrf` half of #524 cannot be worked without the corpus artifact.
   It needs `$POUNCE_BENCH_DATA` or a reachable copy of Vanderbei's
   `cresc4.mod`-style `steenbrf.mod`.
2. It is worth checking whether that `.mod` is faithful at all before spending
   more on it. The reference table already flags it: `steenbrf` is one of the
   few entries with `solvers_agree: false` (`nitro` 282.7578, `snopt` 319.0946,
   `loqo` no answer), and the siblings that *do* agree — `steenbra` 16957.67,
   `steenbrb` 9075.855, `steenbrd` 9030.082, `steenbre` 27459.16 — are all in
   the band the SIF sources predict. `steenbrf` is the outlier in its own
   family by a factor of ~32. A solver being blamed for a 3000-iteration stall
   on a model whose three reference solvers cannot agree on the answer is worth
   confirming before it is worth optimising for.

### The second `steenbrf` question, answered from existing work

The issue also asks whether a stall that flat — objective unchanged to eight
digits over hundreds of iterations, primal feasible throughout — should be
detected and terminated early rather than grinding to the iteration cap.

That has already been prototyped and discarded once, and the write-up applies
directly here: `dev-notes/issue-131-monotone-lbfgs-stall.md`, "Follow-up A".
An opt-in `monotone_stall_iter` was wired end to end and reverted, because
building it falsified its own premise — what reads as a hard stall in a
`print_level=5` trace was a uniformly decelerating crawl with no clean
bimodality separating "slow but will clear" from "doomed", leaving only
problem-specific thresholds. The conclusion there was that if a
"don't silently grind to `max_iter`" signal is ever wanted, it should be a
*post-hoc* diagnostic keyed on the existing `max_iter` + frozen-µ exit state,
never a mid-solve heuristic that perturbs the trajectory. Nothing in #524
changes that, and `steenbrf`'s trace (`inf_du` oscillating *upward* while the
objective is pinned) is if anything a cleaner candidate for the post-hoc
diagnostic than for a mid-solve gate.
