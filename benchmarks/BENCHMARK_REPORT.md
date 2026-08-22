# POUNCE Benchmark Report

Generated: 2026-08-22 04:06:16

## Provenance

| Component | Version / Detail |
|-----------|------------------|
| POUNCE | v0.10.0 (perf/feral-refine-target @ 96846891-dirty) |
| POUNCE linear solver | feral (default) |
| Ipopt | Ipopt 3.14.20 (Darwin arm64), ASL(20241202) |
| Ipopt linear solver | ma57 (via ref/Ipopt/install-ma57) |
| Platform | Darwin 25.5.0 arm64 |

POUNCE results were produced this run by `make -C benchmarks
<suite>-run` (pounce only). The Ipopt column is a saved reference
(`make -C benchmarks ipopt-reference`), rerun only when explicitly
regenerated — generated 2026-06-11 21:49:49 EDT on Johns-Mac-mini.local (Darwin 25.5.0 arm64), git 659d98a, timelimit 300s. Ipopt solve *times* are
from that reference machine and only comparable to POUNCE when this
report is generated on the same host.

The GAMS solver-link path is exercised separately as a liveness
smoke check (`make -C benchmarks gams-bench`) and is not aggregated here.

> **Threading & timing.** These POUNCE runs carry no per-suite thread stamp — they predate it, so the
> settings they ran under were not recorded.
> At report time they are not pinned (`OMP_NUM_THREADS` unset, `OPENBLAS_NUM_THREADS` unset, `VECLIB_MAXIMUM_THREADS` unset, `RAYON_NUM_THREADS` unset), which says nothing about the runs themselves.
> Treat POUNCE-vs-Ipopt time comparisons below as unverified on this axis.

> **Time limits.** The saved Ipopt reference ran at `max_cpu_time` = 300s unless overridden below. The POUNCE arm carries no time-limit flag — it is wrapped in `timeout $BENCH_TIMELIMIT` (default 300s) and a kill is recorded as `Maximum_CpuTime_Exceeded` — so its limit is not stamped in the results and is inferred here from the longest run that was killed.
> Override — **mittelmann**: Ipopt reference at 1800s (regenerated 2026-08-07, threads pinned to 1 (OMP/OPENBLAS/VECLIB/RAYON)).
> Reason given: the 300s max_cpu_time in the base stamp was reached at ~90s wall on unpinned multithreaded BLAS, leaving 6 instances truncated; see dev-notes/research/mittelmann-post-546-sweep.md
> POUNCE runs in this suite were cut off at ~300s, so the two columns are **not** held to the same clock here.
> Decided by that gap: **WM_CFy** — POUNCE cut off at 300s (105 iters), Ipopt Optimal at 1147s (556 iters), i.e. past POUNCE's cutoff. It is counted here as an Ipopt-only solve, on a limit POUNCE was never given. Measured out of band on 2026-08-11 at the reference's own 1800s limit (same host, threads pinned, binary from ce41b5bc): POUNCE returns `Optimal Solution Found` in 673s / 239 iterations, vs Ipopt's 1147s / 556. Objectives differ by 0.136% (two local optima; Ipopt's is the better point). Deliberately not merged into the results, which record the sweep as configured — see dev-notes/research/wm-cfy-timelimit.md.

## Executive Summary

| Metric | POUNCE | Ipopt |
|--------|--------|-------|
| Optimal (strict) | **1281/1327** (96.5%) | **1241/1327** (93.5%) |
| Acceptable (informational, *not* counted as solved) | 6 | 24 |
| Solved exclusively (strict Optimal) | 46 | 6 |
| Both Optimal | 1235 | |
| Matching objectives (< 0.01%) | 1176/1235 | |

> **Note:** All headline counts use strict Optimal status only. `Acceptable`
> means the iterate met relaxed tolerances but not the requested tolerance —
> per CLAUDE.md's "Honesty in Benchmarks" rule it is reported separately and
> never folded into the pass rate. See the "Acceptable (not Optimal)" and
> "Different Local Minima" sections below.

## Performance Profiles

[Dolan & Moré (2002)](https://doi.org/10.1007/s101070100263) performance profiles pooled over every suite with an Ipopt reference. ρ_s(τ) is the fraction of problems a solver solves within a factor τ of the fastest solver on each problem: the **height at τ=1** is how often it was the quickest, and the **right-hand plateau** is its overall robustness (fraction solved at all). A problem counts as solved only at strict/acceptable success; failures and timeouts are charged infinite cost. Regenerate or slice these with `python3 scripts/perf_profile.py <suite…> [--metric iters] [--mode data]`.

![**Performance profile by wall-clock time.** Valid because POUNCE and Ipopt-MA57 were run interleaved on this host (see Provenance).](figures/profile_performance_time.png)

**Performance profile by wall-clock time.** Valid because POUNCE and Ipopt-MA57 were run interleaved on this host (see Provenance).
  
_1292 problems; solvers: pounce, ipopt._

![**Performance profile by iteration count** — machine-independent, so it stays comparable across hosts and reruns.](figures/profile_performance_iters.png)

**Performance profile by iteration count** — machine-independent, so it stays comparable across hosts and reruns.
  
_1292 problems; solvers: pounce, ipopt._

![**Data profile (absolute-time ECDF).** Fraction of problems solved within a given wall-clock budget, without best-solver normalization — reads directly as “how many by 1 s? by 10 s?”.](figures/profile_data_time.png)

**Data profile (absolute-time ECDF).** Fraction of problems solved within a given wall-clock budget, without best-solver normalization — reads directly as “how many by 1 s? by 10 s?”.
  
_1327 problems; solvers: pounce, ipopt._

## Per-Suite Summary

| Suite | Problems | POUNCE Optimal | Ipopt Optimal | POUNCE only | Ipopt only | Both Optimal | Match |
|-------|----------|---------------|--------------|-------------|------------|--------------|-------|
| Vanderbei | 733 | 694 (94.7%) | 683 (93.2%) | 15 | 4 | 679 | 656/679 |
| Electrolyte | 13 | 13 (100.0%) | 13 (100.0%) | 0 | 0 | 13 | 13/13 |
| Grid | 4 | 4 (100.0%) | 4 (100.0%) | 0 | 0 | 4 | 4/4 |
| CHO | 1 | 0 (0.0%) | 1 (100.0%) | 0 | 1 | 0 | 0/1 |
| Water | 6 | 6 (100.0%) | 6 (100.0%) | 0 | 0 | 6 | 4/6 |
| Gas | 4 | 3 (75.0%) | 3 (75.0%) | 0 | 0 | 3 | 3/3 |
| LargeScale | 6 | 6 (100.0%) | 5 (83.3%) | 1 | 0 | 5 | 5/5 |
| Mittelmann | 47 | 46 (97.9%) | 41 (87.2%) | 6 | 1 | 40 | 39/40 |
| QP | 138 | 138 (100.0%) | 133 (96.4%) | 5 | 0 | 133 | 125/133 |
| LP | 371 | 369 (99.5%) | 352 (94.9%) | 17 | 0 | 352 | 327/352 |
| LPopt | 4 | 2 (50.0%) | 0 (0.0%) | 2 | 0 | 0 | 0/1 |

## Vanderbei Reference Cross-Check

Per-problem status from R. Vanderbei's `cute_table.pdf` (`vanderbei/cute_table_status.json`). The meaningful denominator is the **expected-solvable** set — problems with a documented finite optimum — not all 733: the CUTE collection deliberately includes unbounded, infeasible, and no-solver-finishes problems.

| cute_table status | problems | POUNCE solved | meaning |
|---|---|---|---|
| optimum | 684 | 659 | finite reference optimum exists (expected-solvable) |
| hard | 14 | 8 | in table, but SNOPT+NITRO+LOQO all hit time/iter limits |
| infeasible | 3 | 0 | a reference solver declared infeasibility |
| unbounded | 1 | 0 | unbounded below |
| untabulated | 31 | 27 | not in cute_table — no reference datum |

**POUNCE solved 659 / 684 expected-solvable (96.3%).** The hard / infeasible / unbounded / untabulated rows above are excluded from this denominator — a POUNCE failure there is shared with the commercial reference solvers and is not counted as a miss.

**Genuine misses — expected-solvable but POUNCE did not reach Optimal (25):**

> brainpc0 brainpc2 britgas coshfun cresc100 cresc50 csfi2 dallasl dallasm deconvb discs eigenc2 flosp2hh grouping himmelbj nonmsqrt orthrds2 palmer5e polak3 sineali ssebnln steenbrc steenbrd steenbrf steenbrg

**Objective disagreements vs. cute_table reference (21)** — POUNCE converged but to a different value than the agreed reference optimum (possible wrong basin or misread problem):

| Problem | POUNCE obj | reference obj | rel. diff |
|---|---|---|---|
| broydn7d | 3.450050e+02 | 3.823419e+00 | 8.9e+01 |
| liswet9 | 1.963305e+03 | 2.499976e+01 | 7.8e+01 |
| liswet8 | 7.144874e+02 | 2.499977e+01 | 2.8e+01 |
| liswet7 | 4.987922e+02 | 2.499979e+01 | 1.9e+01 |
| palmer1c | 7.932114e+00 | 9.759799e-02 | 7.8e+00 |
| eigenbco | 4.732885e-19 | 9.000000e+00 | 1.0e+00 |
| liswet10 | 4.948391e+01 | 2.499967e+01 | 9.8e-01 |
| orthregd | 1.523900e+03 | 4.245801e+04 | 9.6e-01 |
| orthrgds | 1.523900e+03 | 2.603509e+04 | 9.4e-01 |
| bt4 | -3.704768e+00 | -4.551055e+01 | 9.2e-01 |
| camel6 | -2.154638e-01 | -1.031628e+00 | 7.9e-01 |
| liswet1 | 3.612062e+01 | 2.500304e+01 | 4.4e-01 |
| fletcher | 1.165685e+01 | 1.952537e+01 | 4.0e-01 |
| liswet12 | -3.314381e+03 | -5.026353e+03 | 3.4e-01 |
| cresc132 | 8.577054e-01 | 6.848460e-01 | 1.7e-01 |
| hs044 | -1.300000e+01 | -1.500000e+01 | 1.3e-01 |
| avgasb | -4.483219e+00 | -4.132819e+00 | 8.5e-02 |
| steenbre | 2.851495e+04 | 2.745916e+04 | 3.8e-02 |
| haldmads | 3.303041e-02 | 1.223712e-04 | 3.3e-02 |
| errinros | 4.040449e+01 | 3.990415e+01 | 1.3e-02 |
| trainh | 1.231200e+01 | 1.236996e+01 | 4.7e-03 |

## Vanderbei Suite — Performance

On 679 commonly-solved problems:

| Metric | POUNCE | Ipopt |
|--------|--------|-------|
| Median time | 34.4ms | 44.3ms |
| Total time | 281.01s | 234.62s |
| Mean iterations | 46.9 | 46.9 |
| Median iterations | 15 | 16 |

- **Geometric mean speedup**: 1.0x
- **Median speedup**: 1.1x
- POUNCE faster: 473/679 (70%)
- POUNCE 10x+ faster: 1/679
- Ipopt faster: 206/679

## Electrolyte Suite — Performance

On 13 commonly-solved problems:

| Metric | POUNCE | Ipopt |
|--------|--------|-------|
| Median time | 29.5ms | 37.6ms |
| Total time | 395.6ms | 503.3ms |
| Mean iterations | 12.3 | 12.2 |
| Median iterations | 10 | 10 |

- **Geometric mean speedup**: 1.3x
- **Median speedup**: 1.2x
- POUNCE faster: 13/13 (100%)
- POUNCE 10x+ faster: 0/13
- Ipopt faster: 0/13

## Grid Suite — Performance

On 4 commonly-solved problems:

| Metric | POUNCE | Ipopt |
|--------|--------|-------|
| Median time | 33.8ms | 41.9ms |
| Total time | 135.0ms | 157.2ms |
| Mean iterations | 15.5 | 15.5 |
| Median iterations | 17 | 17 |

- **Geometric mean speedup**: 1.2x
- **Median speedup**: 1.2x
- POUNCE faster: 4/4 (100%)
- POUNCE 10x+ faster: 0/4
- Ipopt faster: 0/4

## Water Suite — Performance

On 6 commonly-solved problems:

| Metric | POUNCE | Ipopt |
|--------|--------|-------|
| Median time | 111.2ms | 122.5ms |
| Total time | 692.7ms | 696.0ms |
| Mean iterations | 198.2 | 205.2 |
| Median iterations | 183 | 209 |

- **Geometric mean speedup**: 1.0x
- **Median speedup**: 0.9x
- POUNCE faster: 2/6 (33%)
- POUNCE 10x+ faster: 0/6
- Ipopt faster: 4/6

## Gas Suite — Performance

On 3 commonly-solved problems:

| Metric | POUNCE | Ipopt |
|--------|--------|-------|
| Median time | 84.2ms | 113.3ms |
| Total time | 282.7ms | 374.0ms |
| Mean iterations | 39.7 | 39.7 |
| Median iterations | 20 | 20 |

- **Geometric mean speedup**: 1.4x
- **Median speedup**: 1.3x
- POUNCE faster: 3/3 (100%)
- POUNCE 10x+ faster: 0/3
- Ipopt faster: 0/3

## LargeScale Suite — Performance

On 5 commonly-solved problems:

| Metric | POUNCE | Ipopt |
|--------|--------|-------|
| Median time | 2.48s | 573.2ms |
| Total time | 11.57s | 9.43s |
| Mean iterations | 309.6 | 305.6 |
| Median iterations | 5 | 2 |

- **Geometric mean speedup**: 0.6x
- **Median speedup**: 0.5x
- POUNCE faster: 2/5 (40%)
- POUNCE 10x+ faster: 0/5
- Ipopt faster: 3/5

## Mittelmann Suite — Performance

On 40 commonly-solved problems:

| Metric | POUNCE | Ipopt |
|--------|--------|-------|
| Median time | 5.97s | 6.88s |
| Total time | 849.31s | 1488.30s |
| Mean iterations | 153.6 | 110.7 |
| Median iterations | 55 | 55 |

- **Geometric mean speedup**: 1.0x
- **Median speedup**: 0.7x
- POUNCE faster: 14/40 (35%)
- POUNCE 10x+ faster: 4/40
- Ipopt faster: 26/40

## QP Suite — Performance

On 133 commonly-solved problems:

| Metric | POUNCE | Ipopt |
|--------|--------|-------|
| Median time | 81.5ms | 92.9ms |
| Total time | 85.27s | 172.97s |
| Mean iterations | 18.4 | 75.6 |
| Median iterations | 18 | 24 |

- **Geometric mean speedup**: 1.2x
- **Median speedup**: 1.1x
- POUNCE faster: 82/133 (62%)
- POUNCE 10x+ faster: 3/133
- Ipopt faster: 51/133

## LP Suite — Performance

On 352 commonly-solved problems:

| Metric | POUNCE | Ipopt |
|--------|--------|-------|
| Median time | 146.7ms | 157.8ms |
| Total time | 496.95s | 422.20s |
| Mean iterations | 25.0 | 107.3 |
| Median iterations | 23 | 56 |

- **Geometric mean speedup**: 1.1x
- **Median speedup**: 1.0x
- POUNCE faster: 181/352 (51%)
- POUNCE 10x+ faster: 10/352
- Ipopt faster: 171/352

## Failure Analysis

### Vanderbei Suite

| Failure Mode | POUNCE | Ipopt |
|-------------|--------|-------|
| Acceptable | 5 | 6 |
| Diverging_Iterates | 3 | 0 |
| Error_In_Step_Computation | 4 | 0 |
| Infeasible_Problem_Detected | 3 | 4 |
| Invalid_Number_Detected | 1 | 3 |
| Maximum_CpuTime_Exceeded | 2 | 8 |
| Maximum_Iterations_Exceeded | 12 | 16 |
| Not_Enough_Degrees_Of_Freedom | 2 | 0 |
| Restoration_Failed | 3 | 3 |
| Search_Direction_Becomes_Too_Small | 2 | 1 |
| Solver_Error | 2 | 2 |
| Unknown_Error | 0 | 7 |

### CHO Suite

| Failure Mode | POUNCE | Ipopt |
|-------------|--------|-------|
| Acceptable | 1 | 0 |

### Gas Suite

| Failure Mode | POUNCE | Ipopt |
|-------------|--------|-------|
| Infeasible_Problem_Detected | 1 | 1 |

### LargeScale Suite

| Failure Mode | POUNCE | Ipopt |
|-------------|--------|-------|
| N/A | 0 | 1 |

### Mittelmann Suite

| Failure Mode | POUNCE | Ipopt |
|-------------|--------|-------|
| Maximum_CpuTime_Exceeded | 1 | 2 |
| Maximum_Iterations_Exceeded | 0 | 3 |
| Solver_Error | 0 | 1 |

### QP Suite

| Failure Mode | POUNCE | Ipopt |
|-------------|--------|-------|
| Acceptable | 0 | 4 |
| Maximum_CpuTime_Exceeded | 0 | 1 |

### LP Suite

| Failure Mode | POUNCE | Ipopt |
|-------------|--------|-------|
| Acceptable | 0 | 14 |
| Infeasible_Problem_Detected | 2 | 1 |
| Maximum_CpuTime_Exceeded | 0 | 1 |
| Maximum_Iterations_Exceeded | 0 | 1 |
| Restoration_Failed | 0 | 1 |
| Unknown_Error | 0 | 1 |

### LPopt Suite

| Failure Mode | POUNCE | Ipopt |
|-------------|--------|-------|
| Maximum_CpuTime_Exceeded | 2 | 4 |

## Regressions (Ipopt Optimal, POUNCE not Optimal)

| Problem | Suite | n | m | POUNCE status | Ipopt obj |
|---------|-------|---|---|--------------|-----------|
| WM_CFy | Mittelmann | 8709 | 12850 | Maximum_CpuTime_Exceeded | 1.221824e+00 |
| cho_parmest | CHO | 21672 | 21660 | Acceptable | 6.767287e+04 |
| discs | Vanderbei | 36 | 69 | Infeasible_Problem_Detected | 1.444952e+01 |
| orthrds2 | Vanderbei | 203 | 100 | Acceptable | 1.544297e+03 |
| ssebnln | Vanderbei | 194 | 96 | Error_In_Step_Computation | 1.617060e+07 |
| steenbrd | Vanderbei | 468 | 108 | Error_In_Step_Computation | 9.030082e+03 |

## Wins (POUNCE Optimal, Ipopt not Optimal) — 46 problems

| Problem | Suite | n | m | Ipopt status | POUNCE obj |
|---------|-------|---|---|-------------|------------|
| BOYD1 | QP | 93261 | 18 | Acceptable | -6.173522e+07 |
| BOYD2 | QP | 93263 | 186531 | Maximum_CpuTime_Exceeded | 2.125677e+01 |
| QPILOTNO | QP | 2172 | 975 | Acceptable | 4.728587e+06 |
| QRECIPE | QP | 180 | 91 | Acceptable | -2.666160e+02 |
| QSCORPIO | QP | 358 | 388 | Acceptable | 1.880510e+03 |
| aa4 | LP | 7195 | 426 | Acceptable | 2.587761e+04 |
| air05 | LP | 7195 | 426 | Acceptable | 2.587761e+04 |
| bore3d | LP | 315 | 233 | Acceptable | 1.373080e+03 |
| brainpc1 | Vanderbei | 6905 | 6900 | Restoration_Failed | 4.382655e-04 |
| brainpc5 | Vanderbei | 6905 | 6900 | Maximum_CpuTime_Exceeded | 3.750404e-04 |
| brainpc7 | Vanderbei | 6905 | 6900 | Maximum_CpuTime_Exceeded | 3.927066e-04 |
| bt8 | Vanderbei | 5 | 2 | Acceptable | 1.000000e+00 |
| co5 | LP | 7993 | 5715 | Acceptable | 7.144696e+05 |
| complex | LP | 1408 | 1023 | Acceptable | -9.966667e+01 |
| coolhans | Vanderbei | 9 | 0 | Unknown_Error | 0.000000e+00 |
| cq5 | LP | 7530 | 5025 | Acceptable | 4.001338e+05 |
| cresc132 | Vanderbei | 6 | 2654 | Infeasible_Problem_Detected | 8.577054e-01 |
| cvxqp3 | Vanderbei | 10000 | 7500 | Maximum_CpuTime_Exceeded | 1.157111e+08 |
| dallass | Vanderbei | 46 | 31 | Invalid_Number_Detected | -3.239323e+04 |
| de063155 | LP | 1488 | 852 | Restoration_Failed | 9.883094e+09 |
| drcav2lq | Vanderbei | 10816 | 816 | Maximum_CpuTime_Exceeded | 1.119702e-03 |
| drcavty2 | Vanderbei | 10816 | 816 | Maximum_CpuTime_Exceeded | 1.119702e-03 |
| finnis | LP | 614 | 497 | Acceptable | 1.727911e+05 |
| flosp2th | Vanderbei | 691 | 0 | Maximum_Iterations_Exceeded | 1.000000e+01 |
| greenbea | LP | 5405 | 2389 | Maximum_Iterations_Exceeded | -7.248917e+07 |
| greenbeb | LP | 5405 | 2389 | Acceptable | -4.302260e+06 |
| laptime | LargeScale | 58014 | 62014 | N/A | 6.529464e+01 |
| manne | Vanderbei | 1094 | 730 | Acceptable | -9.741512e-01 |
| maros | LP | 1443 | 845 | Acceptable | -5.806374e+04 |
| nql180 | Mittelmann | 129601 | 130080 | Solver_Error | -9.277211e-01 |
| palmer7e | Vanderbei | 8 | 0 | Maximum_Iterations_Exceeded | 1.015390e+01 |
| pilot.ja | LP | 1988 | 940 | Acceptable | -6.113136e+03 |
| pilotnov | LP | 2172 | 975 | Acceptable | -4.497276e+03 |
| polak6 | Vanderbei | 5 | 4 | Unknown_Error | -4.400000e+01 |
| qap15 | LPopt | 22275 | 6330 | Maximum_CpuTime_Exceeded | 1.040994e+03 |
| qcqp1500-1c | Mittelmann | 1500 | 10508 | Maximum_CpuTime_Exceeded | 3.882979e+06 |
| qcqp1500-1nc | Mittelmann | 1500 | 10508 | Maximum_CpuTime_Exceeded | 4.778480e+06 |
| recipe | LP | 180 | 91 | Acceptable | -2.666160e+02 |
| robot_a | Mittelmann | 1001 | 52013 | Maximum_Iterations_Exceeded | 1.043195e+00 |
| robot_b | Mittelmann | 1001 | 52013 | Maximum_Iterations_Exceeded | 2.333099e+00 |
| robot_c | Mittelmann | 1001 | 52013 | Maximum_Iterations_Exceeded | 1.405976e+00 |
| scfxm1-2r-27 | LP | 6189 | 4088 | Acceptable | 2.886965e+03 |
| scorpion | LP | 358 | 388 | Acceptable | 1.878125e+03 |
| scrs8-2r-256 | LP | 9765 | 7196 | Maximum_CpuTime_Exceeded | 1.144161e+03 |
| steenbre | Vanderbei | 540 | 126 | Acceptable | 2.851495e+04 |
| supportcase10 | LPopt | 14630 | 165684 | Maximum_CpuTime_Exceeded | 3.383924e+00 |

## Acceptable (not Optimal) — 6 problems

These problems converged within relaxed tolerances but not strict tolerances.

| Problem | Suite | n | m | Ipopt status | POUNCE obj | Ipopt obj |
|---------|-------|---|---|-------------|------------|-----------|
| brainpc0 | Vanderbei | 6905 | 6900 | Restoration_Failed | 3.899805e-01 | 1.700213e+06 |
| brainpc2 | Vanderbei | 13805 | 13800 | Maximum_CpuTime_Exceeded | 4.389516e-04 | 5.293887e-04 |
| cho_parmest | CHO | 21672 | 21660 | Optimal | 6.767287e+04 | 6.767287e+04 |
| csfi2 | Vanderbei | 5 | 4 | Acceptable | 5.501760e+01 | 5.501760e+01 |
| orthrds2 | Vanderbei | 203 | 100 | Optimal | 1.461041e+03 | 1.544297e+03 |
| steenbrf | Vanderbei | 468 | 108 | Acceptable | 7.205566e+02 | 1.321652e+03 |

## POUNCE-Only Suite Details

These suites currently run POUNCE only — no Ipopt-side comparison is captured in their result files. Per-problem timing and iteration counts are shown so users can inspect the whole picture.

### LPopt

| Problem | n | m | Status | Objective | Iters | Time |
|---------|---|---|--------|-----------|-------|------|
| ex10 | 17,680 | 69,608 | Maximum_CpuTime_Exceeded | N/A | 0 | 300.14s |
| irish-electricity | 61,728 | 104,259 | Maximum_CpuTime_Exceeded | N/A | 363 | 300.08s |
| qap15 | 22,275 | 6,330 | Optimal | 1.0410e+03 | 26 | 30.67s |
| supportcase10 | 14,630 | 165,684 | Optimal | 3.3839e+00 | 32 | 124.22s |

POUNCE: **2/4 Optimal** in 755.10s total

## Dedicated Convex Solver vs. General NLP (head-to-head)

The same LP / convex-QP `.nl` problems solved twice by the **same**
pounce binary: once routed to the dedicated convex interior-point
solver (`pounce-convex`, via `solver_selection=lp-ipm` / `qp-ipm`) and
once through the general NLP filter-IPM (`solver_selection=nlp`). This
quantifies the speedup the dedicated solver buys on its home turf. It
is a pounce-vs-pounce comparison and is independent of the Ipopt
reference used by the suites above.

### LP — convex vs NLP

| Metric | pounce-convex | pounce-nlp |
|--------|---------------|------------|
| Optimal | 366/371 (98.7%) | 356/371 (96.0%) |
| Solved exclusively | 13 | 3 |
| Both Optimal | 353 | |
| Matching objectives (< 0.01%) | 328/353 | |

On 353 problems solved by both arms:

| Metric | pounce-convex | pounce-nlp |
|--------|---------------|------------|
| Median time | 146.3ms | 196.8ms |
| Total time | 153.29s | 626.43s |
| Mean iterations | 25.0 | 118.0 |
| Median iterations | 23 | 57 |

- **Geometric-mean speedup (convex over nlp)**: 1.4x
- **Median speedup**: 1.1x
- pounce-convex faster: 245/353 (69%)
- pounce-convex 10x+ faster: 11/353
- pounce-nlp faster: 108/353

### QP — convex vs NLP

| Metric | pounce-convex | pounce-nlp |
|--------|---------------|------------|
| Optimal | 137/138 (99.3%) | 135/138 (97.8%) |
| Solved exclusively | 3 | 1 |
| Both Optimal | 134 | |
| Matching objectives (< 0.01%) | 126/134 | |

On 134 problems solved by both arms:

| Metric | pounce-convex | pounce-nlp |
|--------|---------------|------------|
| Median time | 83.8ms | 97.5ms |
| Total time | 344.39s | 435.74s |
| Mean iterations | 19.0 | 82.8 |
| Median iterations | 18 | 25 |

- **Geometric-mean speedup (convex over nlp)**: 1.0x
- **Median speedup**: 1.0x
- pounce-convex faster: 60/134 (45%)
- pounce-convex 10x+ faster: 1/134
- pounce-nlp faster: 74/134

### Mittelmann — exact vs L-BFGS

| Metric | pounce-convex | pounce-nlp |
|--------|---------------|------------|
| Optimal | 46/47 (97.9%) | 29/47 (61.7%) |
| Solved exclusively | 17 | 0 |
| Both Optimal | 29 | |
| Matching objectives (< 0.01%) | 26/29 | |

On 29 problems solved by both arms:

| Metric | pounce-convex | pounce-nlp |
|--------|---------------|------------|
| Median time | 8.09s | 9.55s |
| Total time | 904.53s | 1119.34s |
| Mean iterations | 72.2 | 119.6 |
| Median iterations | 48 | 62 |

- **Geometric-mean speedup (convex over nlp)**: 1.5x
- **Median speedup**: 1.7x
- pounce-convex faster: 24/29 (83%)
- pounce-convex 10x+ faster: 0/29
- pounce-nlp faster: 5/29

---
*Generated by benchmark_report.py*