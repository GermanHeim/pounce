# QP head-to-head — dedicated convex solver vs general NLP

The **same** Maros-Mészáros convex QP problems as the `benchmarks/qp`
suite, solved twice by the **same** pounce binary:

1. routed to the dedicated convex interior-point solver
   (`pounce-convex`) via `solver_selection=qp-ipm`, and
2. routed through the general NLP filter-IPM via `solver_selection=nlp`.

This is a **pounce-vs-pounce** comparison (no Ipopt reference). The point
is to quantify the speedup the dedicated convex solver buys over the
general NLP path on its own home turf — convex QPs, where the convex IPM
is the textbook-correct method.

> Note: pounce's *default* routing (`solver_selection=auto`) already sends
> convex QPs to `pounce-convex`, so the standard `benchmarks/qp` suite is
> already running the dedicated solver against Ipopt. This suite is the
> explicit dedicated-vs-NLP head-to-head, which `qp` does not provide.
>
> The active-set QP engine (`pounce-qp`, `solver_selection=qp-active-set`)
> is a *different* solver, aimed at warm-started SQP/MPC subproblem
> sequences rather than cold one-shot solves, and is not benchmarked here.

## Data (reused — nothing new generated)

This suite reuses the QP `.nl` already produced for the `qp` suite at
`$POUNCE_BENCH_DATA/qp/nl/*.nl` (the 138-problem Maros-Mészáros convex QP
set; see `benchmarks/qp/README.md` and `benchmarks/qp/generate_nl.py`). If
those `.nl` are missing, generate them once with:

    make -C benchmarks qp-generate

## Two arms / result files

Both produced by `benchmarks/scripts/run_nl_bench.sh` (mode `pounce`,
with the extra `solver_selection=` arg and a custom solver label):

| file | solver | `solver_selection` |
|------|--------|--------------------|
| `convex.json` | `convex` | `qp-ipm` (pounce-convex) |
| `nlp.json`    | `nlp`    | `nlp` (general filter-IPM) |

Both are regenerated each release and are `.gitignore`d, exactly like
every suite's `pounce.json`. Only this `README.md` is committed.

## How to run

    make -C benchmarks qp-convex-run     # incremental
    make -C benchmarks qp-convex-rerun   # wipe both JSON, re-run

## Where results appear

The **Dedicated Convex Solver vs. General NLP (head-to-head)** section of
`benchmarks/BENCHMARK_REPORT.md`, with per-arm Optimal counts, median /
total time, mean / median iterations, and the geometric-mean speedup of
the convex arm over the NLP arm. These suites are deliberately kept out of
the Ipopt-reference machinery (performance profiles, regressions, the
saved baseline) — they are a different comparison axis.
