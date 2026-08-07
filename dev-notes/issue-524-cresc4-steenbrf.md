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

**The `steenbrf` half now uses the reporter's own file.** It was attached to
[a comment on #524](https://github.com/jkitchin/pounce/issues/524#issuecomment-5219526863)
as a base64 tarball all along, and was later uploaded into the container
directly; `sha256 bba26942…78d0ab` matches the issue's checksum, so the
`steenbrf` section below is measured on the exact benchmark input. `cresc4` has
no attachment on the issue and its transcription remains the only route.

So `cresc4` below is measured on a transcription whose iteration counts differ
from the corpus file, and `steenbrf` below is measured on the corpus file
itself. Both transcriptions are committed so they can be audited against the
SIF rather than taken on trust:

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

## `steenbrf` — the reporter's file, and why it crawls

**This section was rewritten once the reporter's actual `.nl` reached the
container** (uploaded directly; `sha256
bba26942506ca72bd77bdb98150a9cf1409f0fc1e2c4d14377a7fe059278d0ab`, matching the
issue comment's checksum exactly, 19004 B). Everything below is measured on
that file. The earlier version of this note reasoned from a `mastsif`
transcription and reached the right *conclusion* — the corpus file is not
CUTEst `STEENBRF` — for incomplete reasons. The real answer is sharper.

### It does not stall here

On this machine, at the reporter's own commit, on the byte-identical file:

| | reporter (`a664dc05`) | here, `a664dc0` | here, +this PR |
|---|---|---|---|
| defaults | Maximum_Iterations_Exceeded, 3000 | **Optimal, 2570 iters, 282.678** | identical, 2570 |
| `mu_strategy=adaptive` | Solved To Acceptable Level, 567 | Solved To Acceptable Level, 567 | identical, 567 |

The adaptive column matches the reporter to the iteration, which is what says
the file and the setup are right. The default column does not: it converges,
2570 iterations, 430 short of the cap.

That is not run-to-run noise — three runs give 2570 exactly, and
`RAYON_NUM_THREADS=1` and `=4` give 2570 too. It is not this PR either: the
parent binary produces bit-identical output. So the trajectory is **platform
dependent**, and the reporter's machine and this container fall on opposite
sides of a 3000-iteration cap that this problem approaches either way.

Do not read that as "cannot reproduce, closing". Read it as: the margin to
`max_iter` on this model is about 15 %, and which side of it you land on is
decided by the floating-point environment. The interesting question was never
"does it hit 3000" but "why does the default path need 4.5× the iterations
adaptive needs", and that reproduces perfectly.

### Why: 360 of 468 variables carry no objective at all

Vanderbei's `steenbrf` is **not** a transliteration of CUTEst `STEENBRF`, and
the difference is not a scale factor — it is a dropped index set.

Read straight off the `.nl` header and the `G0` segment:

```
steenbrb.nl:   0 468 0      # nonlinear vars in constraints, objectives, both
               864 468      # nonzeros in Jacobian, objective gradient
steenbrf.nl:   0 108 0
               864 108
```

Both files have 468 variables and 108 rows. In `steenbrb` every flow variable
appears in the objective. In `steenbrf` only **108** do, and the `.col` map
names them: `cd1..cd18`, `cr1..cr18` (the 36 capacities), plus `d11_k`,
`r11_k`, `d12_k`, `r12_k` for each arc — the flows of **commodities 11 and 12
only**. The congestion term is `LC_k · (d11_k + d12_k)³ / cd_k²`, summing two
of the twelve commodities.

Commodities 1 through 10 — **360 variables** — appear in no objective term,
neither linear nor nonlinear. They are pinned only by flow conservation and
`x ≥ 0`.

CUTEst `STEENBRF` sums all twelve (`Σ_{i=1..12} d[i,k]`). So Vanderbei's `.mod`
looks like a transliteration defect — an index set written over two commodities
where the source has twelve.

That single fact explains every anomaly in this problem:

- **The objective value.** 282.678 here versus ≥ 8251.6 for the CUTEst model.
  That floor is not an estimate: dropping the (non-negative) congestion term
  from `mastsif` STEENBRF leaves a pure min-cost multicommodity-flow LP, and
  POUNCE solves it at **8250.0**; the capacity term adds ≥ 1.64. So 282.678 is
  not a different local minimum of the CUTEst model — it is **unreachable** by
  it, which is what proves the two are different problems rather than two
  answers to one.
- **The reference table disagreeing with itself.** `steenbrf` is one of the few
  entries in `benchmarks/vanderbei/cute_table_status.json` with
  `solvers_agree: false` (nitro 282.7578, snopt 319.0946, loqo no answer). With
  360 cost-free variables the optimal face is a large flat polytope; solvers
  stop at different points of it and report different objectives.
- **The crawl.** This is the part that matters for #524. Those 360 variables
  enter the barrier problem *only* through their `log(x)` terms — zero
  objective gradient, zero Hessian contribution. Nothing pulls them anywhere;
  they are driven purely by µ, and the monotone schedule has to walk all 360 of
  them down decade by decade with the line search fighting the bound. That is
  precisely the trace the issue quotes: objective pinned to eight digits,
  `inf_pr` at 1e-9 (feasible throughout), tiny steps, 20+ backtracks per
  iteration. Adaptive µ, which sets µ from the current centrality rather than
  from a fixed schedule, is not hostage to that and finishes in 567.

The controlled comparison, same network, same data, same start point, the only
difference being whether the other ten commodities carry cost:

| model | congestion term | iterations (defaults) |
|---|---|---|
| CUTEst `STEENBRF` (transcribed) | all 12 commodities | **53** |
| Vanderbei `steenbrf` (reporter's file) | commodities 11–12 only | **2570** |

Fifty-three against two thousand five hundred and seventy, on the same network.
The slowness is a property of the degenerate model, not of the solver.

### What to do about it

Nothing in the solver, on this evidence. POUNCE reaches 282.678, which agrees
with the best reference (nitro's 282.7578) to four digits, and does it in
2570 iterations under defaults and 567 under adaptive µ. There is no wrong
answer here — only a slow path on a model whose objective ignores 77 % of its
variables.

Three things are worth doing, none of them a code change:

1. **Check the corpus entry.** If `steenbrf.mod` really does sum two
   commodities where the source sums twelve, the benchmark is scoring solvers
   on a typo. The same check is worth running across the family — this note
   only establishes it for `steenbrf`, and `steenbrb` is *not* affected (its
   objective touches all 468 variables and its optimum, 9075.8553865777394,
   matches the published 9075.855 exactly, verified against Vanderbei's own
   file, not just the table).
2. **If the entry stays, expect the iteration count.** A model with a large
   cost-free subspace is a legitimate stress case for a monotone barrier, but
   it should be filed as "monotone µ is slow on degenerate models", not as a
   stall bug, and it should not be measured against a 3000-iteration cap it
   sits 15 % inside.
3. **The `max_iter` margin is the only real robustness question**, and it is
   the one from `dev-notes/issue-131-monotone-lbfgs-stall.md`: a post-hoc
   diagnostic on the `max_iter` + frozen-µ exit that says "this looks like a
   degenerate crawl, try `mu_strategy=adaptive`" would have turned the
   reporter's 3000-iteration wall into a one-line answer. Follow-up A in that
   note establishes that the *mid-solve* version of this is an unprincipled
   patience knob; the post-hoc version is still unbuilt and is still the right
   shape.

### On the earlier version of this note

It claimed the corpus file "could not be obtained" and then, after the
attachment was found, that it could not be decoded. Both were true at the time
and both are now moot. What survives unchanged is the STEENBRB control, which
is what made the "different model" conclusion safe to state before the file
arrived — and which the reporter's own `steenbrb.nl` has since confirmed
outright: transcription and original solve to the same objective in the same 49
iterations.
