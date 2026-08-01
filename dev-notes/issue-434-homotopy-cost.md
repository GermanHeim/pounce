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

This note records that measurement and what it turned up, which was not what
the issue expected: **most of the cost was a numerical defect, not an
algorithmic one.** #434's own "also open, and cheaper" item — crossings
escaping the primal ratio test — was the cause of the expensive one.

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

Spot-checked on the instances #434 names, each verified against the Ipopt
reference objective:

| instance | baseline homotopy | fixed homotopy |
|---|---|---|
| AUG2D | Timeout, path stuck at `t = 0.50` | **Optimal, 38.2 s** (`rel = 1.7e-14`) |
| AUG2DC | Timeout, path stuck at `t = 0.50` | **Optimal, 17.0 s**, path completes in **104 steps** |
| QSHARE2B | the #413 loss | **OptimalInaccurate**, matches the published `11703.69` |

*(Full 138-problem re-run in progress; this section gets the complete table.)*

---

## Result 2 — the guard

*(Pending the full re-run. The question is whether any rule of the form
"abandon after `K` steps with `t` still below `T`" fires on the losses that
remain after the fix without firing on any gain.)*

One methodological point is already settled, and it is the reason this took a
second sweep. A guard fires **during** a path, so a candidate rule must be
replayed against the path's `(step, t)` **trajectory**, not against the
`(steps, t)` it ended at. Scored on endpoints, "abandon at 50 steps with
`t < 0.9999`" looks like it catches every reachable loss and harms no gain —
but `KSIP`, a gain, reaches `t = 1` only after 1065 steps, and spends the
early part of that path indistinguishable from one that never finishes. The
endpoint score silently credits a rule that would have thrown `KSIP` away.
The harness therefore records every `(step, t)` tick and replays rules against
the whole path.

---

## What is left

*(Pending.)*
