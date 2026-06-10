# LP head-to-head — dedicated convex solver vs general NLP

The **same** NETLIB + small Mészáros LP problems as the `benchmarks/lp`
suite, solved twice by the **same** pounce binary:

1. routed to the dedicated convex interior-point solver
   (`pounce-convex`) via `solver_selection=lp-ipm`, and
2. routed through the general NLP filter-IPM via `solver_selection=nlp`.

This is a **pounce-vs-pounce** comparison (no Ipopt reference). The point
is to quantify the speedup the dedicated convex solver buys over the
general NLP path on its own home turf — LPs, where the convex IPM is the
textbook-correct method.

> Note: pounce's *default* routing (`solver_selection=auto`) already sends
> LPs to `pounce-convex`, so the standard `benchmarks/lp` suite is already
> running the dedicated solver against Ipopt. This suite is the explicit
> dedicated-vs-NLP head-to-head, which `lp` does not provide.

## Data (reused — nothing new generated)

This suite reuses the LP `.nl` already produced for the `lp` suite at
`$POUNCE_BENCH_DATA/lp/nl/*.nl` (NETLIB + size-filtered Mészáros; see
`benchmarks/lp/README.md` and `benchmarks/lp/generate_nl.py`). If those
`.nl` are missing, generate them once with:

    make -C benchmarks lp-generate

## Two arms / result files

Both produced by `benchmarks/scripts/run_nl_bench.sh` (mode `pounce`,
with the extra `solver_selection=` arg and a custom solver label):

| file | solver | `solver_selection` |
|------|--------|--------------------|
| `convex.json` | `convex` | `lp-ipm` (pounce-convex) |
| `nlp.json`    | `nlp`    | `nlp` (general filter-IPM) |

Both are regenerated each release and are `.gitignore`d, exactly like
every suite's `pounce.json`. Only this `README.md` is committed.

## How to run

    make -C benchmarks lp-convex-run     # incremental
    make -C benchmarks lp-convex-rerun   # wipe both JSON, re-run

## Where results appear

The **Dedicated Convex Solver vs. General NLP (head-to-head)** section of
`benchmarks/BENCHMARK_REPORT.md`, with per-arm Optimal counts, median /
total time, mean / median iterations, and the geometric-mean speedup of
the convex arm over the NLP arm. These suites are deliberately kept out of
the Ipopt-reference machinery (performance profiles, regressions, the
saved baseline) — they are a different comparison axis.
