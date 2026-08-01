# issue #434 — the homotopy's "pure cost" losses, measured

[#434](https://github.com/jkitchin/pounce/issues/434) records that the §4.2
parametric homotopy (`pounce-qp`'s `homotopy.rs`, on by default in the convex
QP driver) is a net win on Maros-Mészáros but not a uniform one: a handful of
large instances that used to solve now hit the time cap, and on those the path
is **pure cost** — it burns the whole budget without reaching `t = 1`.

The issue proposes a runtime guard (abandon a path that is not getting
anywhere, fall back to the conventional cold start) and is explicit about the
order of work:

> A guard has to fire on the 7 losses without touching the 20 gains, and no
> threshold should be chosen without data at the shipped code state. […]
> **If none does, this issue should be closed rather than shipping a guess** —
> the failure mode of a bad threshold is giving back more than it recovers,
> silently.

This note records that measurement. Two things came out of it, and they are
about different phenomena that the issue treats as one:

1. The instances where the homotopy **wedges** — burns the cap without
   finishing — were mostly a numerical defect, #434's own "also open, and
   cheaper" item. Fixing it recovers them.
2. The **Ω(|A|) pivot cost** the issue's title describes is separate, real,
   and untouched by that fix. The warm-start corroboration reproduces
   *identically* after it.

And the guard the issue asks for is **declined**, with the data for why.

---

## The instrument

`crates/pounce-convex/examples/homotopy_sweep.rs` solves one QP through
`solve_qp_active_set` with `use_homotopy` forced on or off — the two arms
differing by exactly that option, everything else the shipped configuration —
and prints status, objective, and time as JSON. Per-path telemetry (steps,
final `t`, longest run of steps that did not advance `t`) comes from the
existing `POUNCE_HOMOTOPY_DEBUG` trace, which now emits a machine-readable
`[hom] summary …` line at every exit from `trace_path`.

Problem data is the same 138-problem Maros-Mészáros convex QP set the `qp`
suite uses, read from the `qpsolvers/maros_meszaros_qpbenchmark` `.mat`
mirror. Objectives are checked against `benchmarks/qp/ipopt_ma57.json` (with
the `.mat` constant offset `r` restored), so "solved" means *solved correctly*
and a solved-but-wrong answer is not counted as a win.

### Environment caveat — read the deltas, not the totals

The sweep ran on a 4-core container with 3 solves concurrent at a 120 s cap.
That is materially slower than the machine behind #434's tables, and the
absolute counts reflect it: the conventional arm scores 52/138 here against
the 58/138 in the issue, and the baseline homotopy arm 46/138 against 71/138.
Wall-clock-sensitive verdicts (`Timeout` in particular) therefore differ from
the issue's per-problem lists.

Everything below is an **A/B on one machine in one run**, which is the
comparison the question needs. Path step counts and `t` values are
deterministic and unaffected by the contention.

---

## What the losses actually are

Baseline (pre-fix) homotopy against the conventional arm, same run: **+2 / −8**.
The eight losses split cleanly into two mechanisms, and the split is the whole
point:

| loss | path steps | final `t` | path ended | homotopy arm | conventional arm |
|---|--:|--:|---|---|---|
| AUG2D | 1000+ | 0.500 | killed at the cap | Timeout | Optimal, 8.8 s |
| AUG2DC | 2100+ | 0.500 | killed at the cap | Timeout | Optimal, 8.2 s |
| DTOC3 | 2600+ | 0.992 | killed at the cap | Timeout | Optimal, 0.7 s |
| QSHIP04L | 1050+ | 0.552 | killed at the cap | Timeout | Optimal, 43.0 s |
| STADAT2 | 6500+ | 0.9999 | killed at the cap | Timeout | Optimal, 56.4 s |
| UBH1 | 1200+ | 0.612 | killed at the cap | Timeout | Optimal, 97.8 s |
| **DUALC1** | **4** | **1.0** | **complete** | IterationLimit, 26.8 s | Optimal, 0.02 s |
| **STCQP2** | **1163** | **1.0** | **complete** | Timeout | Optimal, 102.6 s |

The first six are the mode #434 describes — the path never finishes. The last
two are **a different failure**: the path completes, quickly, and what follows
is bad. `DUALC1`'s path takes four steps to reach `t = 1` and hands over a
working set the corrector then spends 2457 iterations failing to use, on a
problem the conventional route solves in 20 ms. That is a *bad prediction*, not
an expensive one, and no guard on path cost can reach it — the path is already
over when the damage starts.

Keeping the two apart matters for the guard question: a guard is scored against
the six it could reach, not the eight.

---

## Result 1 — the ratio test was stepping over crossings

`AUG2DC`'s trace is what redirected the investigation. Its path reaches
`t = 0.5` inside 50 steps and then **stops**:

```
[hom] step=50  t=5.000000e-1
[hom] step=100 t=5.000000e-1
…
[hom] step=1750 t=5.000000e-1
```

Not a slow path — a stopped one, spending thousands of KKT factorizations
without moving the parameter at all.

The cause is #434's other item. Both ratio tests selected the next event with

```rust
if dt >= -T_EPS && t + dt < t_next - T_EPS { … }
```

The `- T_EPS` margin reads as a don't-bother-for-a-hair guard. It is not one:
it makes a crossing that happens **earlier** than the incumbent, by less than
`T_EPS = 1e-12`, lose to it — so the step knowingly overshoots the earlier
crossing. #413 measured exactly that on `QSHARE2B`: row 132 crossed at
`dt = 2.9e-16` and lost to a step of `1.1e-14`.

Overshooting is not a rounding-level mistake, because violation is absorbing.
The primal ratio test only ever *prevents* a violation and can never repair
one, so a row stepped over stays inactive and violated for the rest of the
path while the direction solve pushes it further out. The same comparison also
discarded crossings *tied* with the incumbent, leaving a row sitting exactly on
a bound it was not in the working set for — which the next direction pushes it
across.

The repair (`RatioTest` in `homotopy.rs`): compare crossings exactly, and fire
the whole coincident set rather than one member of it. `tests/homotopy_unit.rs`
pins the rule directly, including the measured 2.9e-16-vs-1.1e-14 case.

Re-running all 138 with only that change (the conventional arm is untouched by
it, so it is reused from the same run):

| arm | correct | solved-but-wrong | timeouts |
|---|--:|--:|--:|
| conventional | 52/138 | 4 | 58 |
| homotopy, baseline | 46/138 | 1 | 72 |
| homotopy, **fixed** | **48/138** | 3 | 69 |

**Fixed against baseline homotopy: +2 / −0.** `AUG2D` and `AUG2DC` recover;
nothing regresses. Both are correct to `1e-14` against the Ipopt reference:

| instance | baseline homotopy | fixed homotopy |
|---|---|---|
| AUG2D | Timeout, path stuck at `t = 0.50` | **Optimal, 38.2 s** |
| AUG2DC | Timeout, path stuck at `t = 0.50` | **Optimal, 17.0 s**, path completes in **104 steps** |
| QSHARE2B | the #413 loss | **OptimalInaccurate**, matches the published `11703.69` |

The effect on paths generally is larger than the two status flips suggest —
cold paths that reach `t = 1` go 92 → 98 of 138, those killed mid-flight
37 → 31, and the median completed path halves:

| | complete | killed mid-flight | median steps | max |
|---|--:|--:|--:|--:|
| baseline | 92 | 37 | 216 | 2725 |
| fixed | 98 | 31 | **102** | 2725 |

`QSHIP04L` is worth noting separately: its path went from *killed at 1050
steps* to *complete in 615*, and it is **still a loss**. The fix moved it from
the never-finishes mode into the bad-prediction mode. Path cost was not what
was wrong with it.

### The fix does **not** touch the warm-start corroboration — and that matters

#434 corroborates its cost claim with the `benchmarks/warmstart` suite, whose
`-hom` arms differ from their twins by exactly `use_homotopy`. Re-running that
suite (`python -m warmstart.run --quick`, the three families in the issue's
table) against the **fixed** build reproduces the issue's numbers *exactly*:

| family | conventional | homotopy | ratio | mean \|A\|/n |
|---|--:|--:|--:|--:|
| `simplex_proj` | 328 | 523 | 0.63× | 16.4/20 = 82% |
| `rosenbrock_ring` | 29 | 29 | 1.00× | 0.5/10 = 5% |
| `nmpc_vanderpol` | 370 | 1140 | **0.32×** | 46.4/47 = 99% |
| **total** | **727** | **1692** | **0.43×** | |

Not close to the issue's figures — identical. These are deterministic counts,
and the ratio-test defect never fires on these families: they are small,
non-degenerate QPs where no two crossings land within `1e-12` of each other.

So the two phenomena are **separate**, and this note's first result must not be
read as covering both:

* The **Ω(\|A\|) pivot cost** the issue describes is real, reproduced, tracks
  the active-set fraction exactly as claimed (82% → 0.63×, 5% → 1.00×,
  99% → 0.32×), and is **completely untouched** by the ratio-test fix.
* The **Maros-Mészáros wedging** — `AUG2D`/`AUG2DC` stuck at `t = 0.5` and
  burning the cap — was the defect, and is fixed.

Worth stating plainly because it is easy to overclaim from the first result:
the fix explains why some large cold solves *stopped*, not why the homotopy
costs more pivots when the active set is a large fraction of `n`. #434's
central mechanism claim survives the fix intact.

Two things bound how much that costs in practice. All arms return correct
answers with **identical outer iteration counts** (20/20, 301/301, 86/86), so
the extra inner work buys nothing and loses nothing — it is overhead, not
damage. And `use_homotopy` defaults to `false` in `pounce-qp`, so the SQP
inner-QP path does not take it unless asked; these `-hom` arms are opt-in
measurements, not shipped behaviour. The default that *is* on is the convex QP
driver's, which is what the Maros-Mészáros sweep above measures.

---

## Result 2 — the guard, declined

After the fix, six losses remain, and only three of them are even reachable by
a guard on path cost — the other three (`DUALC1`, `QSHIP04L`, `STCQP2`)
complete their paths and fail afterwards.

A guard fires **during** a path, so each candidate rule was replayed against
every path's recorded `(step, t)` trajectory rather than against the
`(steps, t)` it ended at. That distinction is not pedantic: scored on
endpoints, "abandon at 50 steps with `t < 0.9999`" looks like it catches every
reachable loss and harms no gain — but `KSIP` reaches `t = 1` only after 1065
steps and spends the early part of that path indistinguishable from one that
never finishes, so the endpoint score silently credits a rule that would have
thrown `KSIP` away.

Replayed properly, exactly one rule shape catches all three reachable losses
without firing on either gain:

> abandon at `steps >= 1100` while `t < 0.99`

**It should not be shipped.** Two independent reasons, both from the data:

1. **It is fitted to one point, with a 3% margin.** `KSIP` — a genuine gain,
   0.4 s with the homotopy against a 120 s timeout without it — completes its
   path at **1065** steps. The threshold is **1100**. Nothing but that single
   instance separates the rule from destroying a gain, and #434's own tables
   list twenty gains on the machine it was filed from, none of whose path
   lengths are known here.

2. **It is not actually harmless.** The gain/loss framing hides the cost,
   because a problem *both* arms solve cannot appear as a loss. Replaying the
   rule against all 48 problems the fixed homotopy solves correctly, it fires
   on `LASER` — path complete in 2725 steps, **Optimal in 16.3 s**, against
   41.4 s on the conventional route. The guard would abandon a working fast
   path there and hand the problem to a route 2.5× slower; at a tighter cap
   that is a timeout it manufactured.

So the answer to the issue's question 3 is **no**: no threshold on
`(steps, t)` separates the losses from the gains. Per the issue's own
instruction —

> If none does, this issue should be closed rather than shipping a guess.

— no guard is added. The measurement is the deliverable.

This also matches the precedent in
`dev-notes/issue-131-monotone-lbfgs-stall.md`, where an analogous stall
detector was prototyped and discarded for the same reason: what looks like a
hard stall in a trace is a decelerating crawl, and no fixed-window gate
separates "slow but will finish" from "doomed".

---

## What is left

* **The `DUALC1` / `QSHIP04L` / `STCQP2` mode is untouched and is the more
  interesting one.** The path completes and the prediction is still bad —
  `DUALC1` reaches `t = 1` in four steps and the corrector then spends 2457
  iterations failing on a problem the conventional route solves in 20 ms. That
  is a question about what the path predicts, not about what it costs, and it
  wants its own issue rather than a line in this one.
* **The rank-repair tabu**, the other and larger source of uncapped crossings
  in #413's measurement (10 of 14, against the 4 fixed here). `tabu_cons`
  hides a row from the *primal ratio test* rather than only from the add
  decision, so the step is computed as if the row were absent and crosses it.
  Repairing it properly needs an exchange pivot at the degenerate vertex.
* **`DTOC3`, `STADAT2`, `UBH1`** remain genuinely long paths — `t` advancing
  the whole way, no stall — which is the mechanism #434 describes and which
  this note does not solve. What it establishes is that they cannot be
  separated from the gains by a runtime threshold.

## Reproducing

The harness is `crates/pounce-convex/examples/homotopy_sweep.rs`; the sweep
driver, the `.mat` → flat-text converter, and the analysis scripts are not
committed (they depend on a downloaded copy of the Maros-Mészáros `.mat`
mirror). To re-run:

```
cargo run -p pounce-convex --release --example homotopy_sweep -- <file.qp> on|off
POUNCE_HOMOTOPY_DEBUG=1   # adds the [hom] summary / step trace on stderr
```
