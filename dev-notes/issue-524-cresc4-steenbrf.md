# issue #524 — `cresc4` false infeasibility and `steenbrf` stall

[#524](https://github.com/jkitchin/pounce/issues/524): the two Vanderbei
problems where POUNCE fails and the committed Ipopt-MA57 reference succeeds.
The issue's hypothesis was that they share a cause, because both are fixed by
`mu_strategy=adaptive`.

Outcome: **`cresc4` reproduced, diagnosed and fixed — confirmed on the corpus
file itself. `steenbrf` is a real solver defect and is not fixed here**; it
shares the `cresc4` failure shape (restoration progress repeatedly discarded)
and needs its own issue.

Most measurements below were taken on `a664dc0` plus the change described here,
FERAL backend, release build, in a container without the corpus. The two
sections that say so were re-measured later on the reporter's own machine
against `$POUNCE_BENCH_DATA` — `cresc4` end-to-end on the corpus `.nl`, and the
whole of the `steenbrf` half.

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
itself.

**The corpus `cresc4.nl` has since been run directly**, on the reporter's
machine, and the fix holds on it — rung 1 (`feral_scaling=mc64`) still returns
local infeasibility, rung 2 (`mu_strategy=adaptive`) returns
`Optimal Solution Found` at `0.87189752899987727`, matching the committed
Ipopt-MA57 reference `0.8718975393` to nine significant figures. Merged `main`
still reports `Infeasible_Problem_Detected` on that same file. The
transcription and the corpus file differ in iteration counts and agree on
everything this change turns on.

Both transcriptions are committed so they can be audited against the
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

### What the extra rung costs

Nothing on a successful solve — the ladder only runs on a local-infeasibility
verdict, and a 733-problem Vanderbei sweep (this branch's binary *and* driver
against `54219714`, same host) shows no objective drift on any of the 700
problems both solve, nothing broken, and exactly one status change from the
solver: `cresc4`, `Infeasible_Problem_Detected` → `Solve_Succeeded`.

Four further rows change, and none of them are the solver — they are the
driver fix below reading logs it previously could not classify:
`indef` and `static3` `Solver_Error` → `Diverging_Iterates`, `grouping` and
`lewispol` `Solver_Error` → `Not_Enough_Degrees_Of_Freedom`. The last two
abort before any `EXIT:` banner is printed, so the new `Status:` line is the
only thing that resolves them.

The cost lands entirely on problems that report infeasible, and there it is not
negligible. Measured on that sweep:

| problem | `54219714` | with the ladder |
|---|---|---|
| `cresc132` | 3.65s | **73.36s** (20×) |
| `cresc100` | 0.97s | 5.95s |
| `cresc50` | 2.33s | 4.56s |
| `launch` | 0.10s | 0.22s |
| `cresc4` | 0.06s | 0.07s (and now solves) |
| vanderbei suite, total | 1430s | 1523s (+7 %) |

Two extra solves on a hard infeasible problem is inherent to the design, and
+7 % across a suite is a fair price for the class of wrong answer it removes.
The 20× on `cresc132` is worth stating plainly, because it is the number a
branch-and-bound driver pruning many infeasible nodes would feel: the barrier
rung there runs to `max_iter` before giving up.

A rung only ever counts when it *converges*, so a rung that runs to the
iteration cap has burned a full solve to contribute nothing. Capping the rungs'
`max_iter` below the baseline's would bound this worst case, at the cost of
declining any recovery that genuinely needs the iterations. Not attempted here —
it changes what the ladder can find, so it needs its own justification and its
own sweep rather than being folded into this change.

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

### It does not stall in the container; it does stall on the reporter's machine

The first version of this section was written in a container, where the default
path converges at 2570 iterations — 430 short of the cap — and concluded from
that margin that there was no stall to fix. Re-measured on the reporter's own
machine (Darwin 25.5.0 arm64, release + FERAL, same `.nl`), that conclusion does
not survive:

| | container, `a664dc0` | reporter's machine, `54219714` |
|---|---|---|
| defaults | Optimal, 2570 iters, 282.678 | **Maximum_Iterations_Exceeded, 3000** |
| defaults, `max_iter=6000` | — | **Restoration_Failed at 3039** |
| `mu_strategy=adaptive` | Solved To Acceptable Level, 567 | Solved To Acceptable Level, 567 |

Two things follow, and the second is the one that matters.

The trajectory really is **platform dependent** — same commit, same bytes, two
different answers, and on neither machine is it run-to-run noise (three runs
agree exactly, `RAYON_NUM_THREADS=1` and `=4` agree too). The adaptive row is
identical on both, which is what confirms the file and the setup are right.

But the 15 % margin the container seemed to show is **not a margin**. Raising
the cap does not let the default path finish: with `max_iter=6000` it fails at
iteration 3039 with `Restoration Failed`. The container's 2570-iteration
"success" is the lucky side of a coin, not evidence that the model merely needs
patience. There is no cap to sit safely inside.

### The barrier parameter stops moving, and restoration is why

Both solvers walk µ down identically through the first three barrier levels and
then part at the fourth. Counting non-restoration iterations at each `lg(mu)`,
from the reporter's machine and the committed reference log
(`benchmarks/vanderbei/logs/vanderbei/steenbrf.ipopt-ma57.log`, Ipopt 3.14.20 +
MA57):

| lg(µ) | −1.0 | −1.7 | −2.5 | **−3.8** | −5.7 | −8.6 |
|---|---|---|---|---|---|---|
| Ipopt-MA57 | 43 | 133 | 422 | **29** | 18 | 1192 |
| POUNCE, defaults | 37 | 78 | 242 | **2360** | — | — |

POUNCE reaches `lg(mu) = -3.8` at iteration 366 and never leaves it. Ipopt
clears the same level in 29 iterations. They *enter* it in near-identical
states, so nothing before this point is the cause:

```
ipopt   608  5.1501578e+02 1.48e-07 3.23e+03  -3.8 2.69e+01  -3.0 8.76e-01 1.48e-01f  1
pounce  366  3.6948567e+02 1.52e-07 3.81e+03  -3.8 4.27e-01  -1.4 1.00e+00 1.49e-01f  1
```

The mechanism is visible in the next thirty iterations, and it is not that
POUNCE fails to make progress. It makes the progress and then throws it away:

```
 389  obj=3.667223e+02 inf_pr=2.19e-08 inf_du=6.20e-03   ← essentially at the µ-update test
 390  obj=3.656657e+02 inf_pr=1.30e-08 inf_du=6.42e-03
 391r obj=1.764715e+02 inf_pr=1.30e-08 inf_du=1.00e+03   ← restoration
 …
 394  obj=3.680784e+02 inf_pr=6.81e-09 inf_du=3.44e+03   ← back out, five orders worse
```

`inf_du` gets to 6.2e-03 with `inf_pr` at 2e-08 — feasible and one short step
from the dual tolerance that would drop µ — and restoration then resets it to
3.4e+03. That cycle repeats for the remaining 2600 iterations. The objective is
not pinned as the issue's quoted tail suggests; over iterations 2500–3000 it
swings between 395.4 and 436.1 while `inf_du` swings between 1.06 and 1.46e+04.
It is a limit cycle, not a crawl.

The restoration counts are the whole story in one line:

| run | iterations | restoration iterations | **restoration episodes** | outcome |
|---|---|---|---|---|
| POUNCE, defaults | 3000 | 346 | **62** | Maximum_Iterations_Exceeded |
| POUNCE, `mu_strategy=adaptive` | 567 | 10 | **1** | Solved To Acceptable Level |
| Ipopt-MA57, defaults | 1846 | 10 | **1** | Solved To Acceptable Level |

Adaptive µ and Ipopt have *identical* restoration profiles on this problem — ten
restoration iterations in a single episode. The monotone default enters
restoration sixty-two times.

One earlier reading has to be withdrawn: it looked like POUNCE was taking wild
unregularized steps (‖d‖ = 5.4e+03, 6.9e+03 at iterations 368–369) that Ipopt
did not. It is not a discriminator — Ipopt takes a 7.9e+04 step at its
iteration 614 and a 1.1e+05 step later in the same phase, both larger than
anything POUNCE takes here. Big unregularized steps are normal on this model.
The repeated restoration entry is what is not.

It *looks* like the `cresc4` shape — restoration progress the main loop does not
hold on to — and the previous version of this note said so. That reading is now
withdrawn: the repeated restoration entry is a symptom, not the cause. The cause
is one line of missing state, and it is in the line search rather than the
restoration hand-off. See the next section.

### The actual defect: the watchdog counter survives restoration

Ablation isolates it in one run. Against the three globalization heuristics
active in the stall, only one matters:

| run | outcome |
|---|---|
| defaults | `Maximum_Iterations_Exceeded`, 3000 |
| `soft_resto_pderror_reduction_factor=0` (soft resto off) | `Maximum_Iterations_Exceeded`, 3000 |
| **`watchdog_shortened_iter_trigger=0` (watchdog off)** | **`Solve_Succeeded`, 481** |
| both off | `Error_In_Step_Computation`, 2389 |

The marker tally says the same thing structurally — POUNCE spends the stall in
mechanisms Ipopt never enters at all:

| run | `f` | `s` (soft resto) | `h` | `w` (watchdog) | `R` |
|---|---|---|---|---|---|
| POUNCE, defaults | 1668 | 472 | 398 | **334** | 124 |
| Ipopt-MA57 | 1783 | **0** | 42 | **0** | 1 |

The watchdog arms after `watchdog_shortened_iter_trigger` (default 10)
**consecutive** shortened steps. Upstream zeroes that counter when the
restoration phase succeeds — `IpBacktrackingLineSearch.cpp:624-631`, alongside
`in_soft_resto_phase_` and `soft_resto_counter_` — because an iterate returned
by restoration is a different point, so a run of shortened steps does not
continue through it.

POUNCE had no equivalent, and the reason is structural rather than an oversight
in reading the source. Upstream calls `PerformRestoration()` from *inside*
`FindAcceptableTrialPoint`, so the four resets sit immediately after it with the
line-search state in scope. POUNCE returns `Outcome::Failed` and lets
`IpoptAlgorithm::invoke_restoration` run restoration, so there was no site where
that code could land — and nothing on the recovery path told the line search a
restoration had happened.

The arithmetic on this problem, straight off the log:

| iterations | what | counter |
|---|---|---|
| 381–385 | five shortened steps (`ls` = 11, 12, 13, 15, 16) | 1 → 5 |
| 386–390 | soft restoration (`s`) — neither side touches the counter | 5 |
| 391–394 | full restoration, succeeds | upstream → **0**; POUNCE → **5** |
| 399–403 | five more shortened steps | upstream 5; POUNCE → **10 = trigger** |
| 404 | — | POUNCE arms the watchdog |

Then the collapse, which is what actually costs the solve:

```
 403  3.6779355e+02 1.22e-08 4.61e+02  -3.8 1.27e-03  4.1 1.00e+00 1.53e-05f 17
 404  3.6744474e+02 3.68e-07 9.11e+01  -3.8 2.29e-03  3.7 1.00e+00 1.00e+00w  1
 405  3.6709336e+02 1.23e-07 1.87e+02  -3.8 2.86e-03  3.2 1.00e+00 9.88e-01w  1
 406  3.6675112e+02 1.35e-07 1.97e+02  -3.8 4.23e-03  2.7 1.00e+00 9.30e-01w  1
 407  3.6779355e+02 1.22e-08 4.72e+02  -3.8 7.70e-03  2.2 1.00e+00 4.77e-07f 21
 408  3.6779355e+02 1.22e-08 4.89e+04  -3.8 8.68e-02  1.8 1.00e+00 4.04e-08f 20
 409  3.6779355e+02 1.22e-08 1.08e+04  -3.8 1.27e-01  1.3 7.83e-01 2.91e-08f 23
```

404–406 are exactly `watchdog_trial_iter_max = 3` trial iterations; iteration
407's objective and `inf_pr` are bit-identical to 403's, which is `StopWatchDog`
reverting to the snapshot. The three `w` steps bought nothing and the line
search comes out of the revert backtracking 20+ times to `alpha` ~1e-08. That
cycle ran 105 times. Ipopt's longest run of consecutive shortened steps on this
problem is **6** — it never reaches the trigger, which is exactly what the
missing reset predicts.

The fix is `BacktrackingLineSearch::reset_after_restoration()`, called from the
`RestorationOutcome::Recovered` arm of `invoke_restoration`. It ports the three
resets that apply (`count_successive_shortened_steps_` is not ported at all —
upstream reads it only under `expect_infeasible_problem_`, `cpp:798-804`).

With it, `steenbrf` converges in 481 iterations to `Solve_Succeeded`, at
constraint violation 4.4e-11 and objective 282.678 — a *lower* minimum than the
Ipopt-MA57 reference reaches (1321.65, and only to acceptable level, in 1846
iterations). The marker tally drops to 0 `w` and 12 `s`.

On the 733-problem Vanderbei sweep this fixes a second problem nobody was
looking at — `brainpc2`, `Maximum_Iterations_Exceeded` at 3000 →
`Solved_To_Acceptable_Level` at 1003 — breaks nothing, drifts no objective, and
takes 6.5 % off total solve time and 7.3 % off total iterations. That last part
is the tell that this was a defect and not a tuning preference: a heuristic that
fires when it should not costs iterations everywhere, quietly.

### Why the model is hard anyway: 360 of 468 variables carry no objective at all

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

That single fact explains why the problem is hard, and it explains two of the
three anomalies outright. It does **not** explain the stall — see the third
bullet.

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
- **The difficulty — but not the stall.** Those 360 variables enter the barrier
  problem *only* through their `log(x)` terms: zero objective gradient, zero
  Hessian contribution. Nothing pulls them anywhere, so they are driven purely
  by µ, the optimal face is a large flat polytope, and the whole model is a
  legitimately nasty case for a monotone barrier. That is a real property of
  the model and it is why every solver takes hundreds of iterations on it.
  It is **not** an explanation of POUNCE's failure, because Ipopt solves *that
  same file*, with those same 360 unpriced variables, and its µ keeps
  advancing. A model property that both solvers face cannot account for a
  behaviour only one of them shows. What separates them is the sixty-two
  restoration episodes measured above.

The controlled comparison, same network, same data, same start point, the only
difference being whether the other ten commodities carry cost:

| model | congestion term | iterations (defaults) |
|---|---|---|
| CUTEst `STEENBRF` (transcribed) | all 12 commodities | **53** |
| Vanderbei `steenbrf` (reporter's file) | commodities 11–12 only | 2570 (container) / **3000, cap** (reporter) |

Fifty-three against thousands, on the same network: the corpus model is
enormously harder than the CUTEst one it was transcribed from. That is worth
knowing on its own, and it is a separate fact from POUNCE failing where Ipopt
does not.

### What to do about it

Two separate things, and the earlier versions of this note got the split wrong
twice — first collapsing them into "nothing in the solver", then into "the same
unfixed shared cause as `cresc4`".

**The solver defect is fixed**: the missing post-restoration reset of the
watchdog counter, above. `steenbrf` converges. It was *not* the `cresc4` shape —
`cresc4`'s "Not fixed here" item stands on its own and is unchanged by this.

Then, separately, three things about the corpus entry, none of them code:

1. **Check the corpus entry.** If `steenbrf.mod` really does sum two
   commodities where the source sums twelve, the benchmark is scoring solvers
   on a typo. The same check is worth running across the family — this note
   only establishes it for `steenbrf`, and `steenbrb` is *not* affected (its
   objective touches all 468 variables and its optimum, 9075.8553865777394,
   matches the published 9075.855 exactly, verified against Vanderbei's own
   file, not just the table).
2. **If the entry stays, expect the iteration count.** A model with a large
   cost-free subspace is a legitimate stress case for a monotone barrier, and
   hundreds of iterations on it are not by themselves a bug — 481 post-fix is
   still an order of magnitude more than the family's other members. What was a
   bug was the limit cycle, and that is now fixed; keep the two claims apart
   when reading the suite.
3. **The post-hoc diagnostic is still worth building**, and it is the one from
   `dev-notes/issue-131-monotone-lbfgs-stall.md`: a message on the `max_iter` +
   frozen-µ exit that says "µ has not moved in N iterations and restoration was
   entered M times — try `mu_strategy=adaptive`" would have turned the
   reporter's 3000-iteration wall into a one-line answer. Follow-up A in that
   note establishes that the *mid-solve* version of this is an unprincipled
   patience knob; the post-hoc version is still unbuilt and is still the right
   shape. The restoration-episode count is the signal to key it off — 62 versus
   1 separated the failing run from both healthy ones cleanly. `steenbrf` is no
   longer the motivating example, but the diagnostic is not specific to it.

### On the earlier versions of this note

Three claims have been withdrawn, all from reasoning that ran ahead of the
measurement:

- *"The corpus file could not be obtained"*, then *"could not be decoded"*.
  True at the time, moot now — the whole `steenbrf` half is measured on the
  reporter's file.
- *"It does not stall here, so there is nothing in the solver."* The container's
  2570-iteration convergence was real but is the lucky side of a
  platform-dependent trajectory, and the 15 % margin to `max_iter` it seemed to
  establish does not exist (`max_iter=6000` → `Restoration Failed` at 3039). A
  narrower claim from that same session — that POUNCE takes large unregularized
  steps Ipopt does not — is also withdrawn: Ipopt takes larger ones on this
  problem.
- *"It is the `cresc4` shape — restoration progress the main loop does not hold
  on to — and it wants a single issue covering both."* Written when the
  repeated restoration entry was the most visible thing in the trace. It was
  the symptom: restoration was being re-entered because the watchdog kept
  wrecking the line search, not the other way round. Both problems do respond
  to `mu_strategy=adaptive`, which is what made the shared-cause reading
  attractive, but a heuristic that only fires under the monotone schedule will
  produce that signature without sharing anything else.

What survives unchanged is the STEENBRB control, which is what made the
"different model" conclusion safe to state before the file arrived — and which
the reporter's own `steenbrb.nl` has since confirmed outright: transcription and
original solve to the same objective in the same 49 iterations. The corpus
finding and the solver finding are independent, and both hold.
