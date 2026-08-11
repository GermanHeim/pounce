# WM_CFy: the last "Ipopt only" solve is a time-limit artifact

**Date:** 2026-08-11
**Binary:** `target/release/pounce` built from `ce41b5bc` (merged main, 0.10.0)
**Status:** measured, not folded into `BENCHMARK_REPORT.md`

## Claim

`Mittelmann/WM_CFy` is the single instance the 0.10.0 report counts as solved
by Ipopt and not by POUNCE. It is not a capability gap. The two columns are
run at different time limits on that suite, and POUNCE is the faster of the
two solvers once given the same clock.

## The asymmetry

`benchmarks/ipopt_ma57.provenance.json` carries a per-suite override:

    "mittelmann": { "timelimit": 1800, "generated": "2026-08-07",
                    "threads": "pinned to 1 (OMP/OPENBLAS/VECLIB/RAYON)" }

The POUNCE arm has no time-limit flag at all — `run_nl_bench.sh` wraps it in
`timeout $BENCH_TIMELIMIT` (default 300) and relabels rc=124 as
`Maximum_CpuTime_Exceeded`. A plain `make benchmark-rerun` therefore runs the
mittelmann POUNCE arm at 300s against a reference that got 1800s.

## Ipopt never solved it at 300s either

Before the reference was regenerated (`e157b036`, 2026-08-08), Ipopt's own
record for this instance was a timeout:

| | `e157b036~1` | `e157b036` |
|---|---|---|
| status | `Maximum_CpuTime_Exceeded` | `Solve_Succeeded` |
| time | 340.5s | 1146.5s |
| iterations | 157 | 556 |
| objective | 1.232202588975379 (incumbent) | 1.2218235931043862 |

So `WM_CFy` used to be a *mutual* failure. It became an Ipopt-only solve on
2026-08-08 by raising Ipopt's ceiling, not by any change in either solver.

That regeneration flipped four instances, and it was justified for three of
them — `henon120` (108.5s → 77.5s), `lane_emden120` (89.3s → 69.7s) and
`qcqp1000-2c` (104.2s → 129.7s) were killed by the documented threading bug
(`max_cpu_time` counts CPU summed across threads, so an unpinned run burns a
300s budget in ~90s wall) and all three now finish *inside* 300s. Pinning
fixed those; the raised ceiling was incidental. `WM_CFy` is the only one that
actually consumed the extra ceiling.

## POUNCE at Ipopt's clock

Rerun with the driver's exact invocation (`timeout 1800 pounce <nl> --no-sol`,
all four thread vars = 1, same host):

| | POUNCE @1800s | Ipopt ma57 @1800s |
|---|---|---|
| status | `Optimal Solution Found` | `Solve_Succeeded` |
| wall | **672.8s** (664.5s in solver) | 1146.5s |
| iterations | **239** | 556 |
| objective | 1.2234907111354598 | 1.2218235931043862 |
| NLP error | 6.3403053477504722e-09 | — |
| constraint violation | 9.3537964873657842e-10 | — |

POUNCE converges in 59% of Ipopt's wall time and 43% of its iterations.

## Caveat: different local solutions

The objectives disagree by **0.136%**, well past the report's 0.01% match
threshold. The problem minimizes (`O0 0` in the .nl), so Ipopt's 1.22182 is
the better point and POUNCE converged to a worse local optimum. Both are
legitimately converged — POUNCE's KKT error is 6.3e-09 with 9.4e-10 violation
— so this is two local optima on a nonconvex problem, not an error. Folding
this run into the report would move `WM_CFy` from "Ipopt only" to "both
Optimal, objectives disagree", not to a clean match.

## What was done about it

Nothing to the numbers. `benchmarks/mittelmann/pounce.json` still records the
300s timeout, which is the honest record of the sweep as configured. The
report now discloses the override, the inferred POUNCE cutoff, and names
`WM_CFy` as the instance the gap decides (`time_limit_note` in
`benchmark_report.py`).

The like-for-like fix — rerunning the mittelmann POUNCE arm at
`BENCH_TIMELIMIT=1800` — is deferred. It is a few hours of machine time and
only this one instance can change status, since the other five POUNCE
timeouts in the corpus (`vanderbei/cresc132`, `drcav3lq`, `drcavty3`,
`lpopt/ex10`, `irish-electricity`) are instances Ipopt fails as well.
