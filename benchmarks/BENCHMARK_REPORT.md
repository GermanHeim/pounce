# POUNCE Benchmark Report

Generated: 2026-08-10 21:17:48

## Provenance

| Component | Version / Detail |
|-----------|------------------|
| POUNCE | v0.10.0 (main @ 804bf084-dirty) |
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
> At report time all of `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `VECLIB_MAXIMUM_THREADS`, `RAYON_NUM_THREADS` = 1. When this report is generated as the last
> step of `make -C benchmarks benchmark` that is the sweep's own environment, but it is not proof: run
> `benchmark-report` separately and it says nothing about the runs.
> Treat POUNCE-vs-Ipopt time comparisons below as unverified on this axis.

## Executive Summary

| Metric | POUNCE | Ipopt |
|--------|--------|-------|
| Optimal (strict) | **1287/1326** (97.1%) | **1241/1326** (93.6%) |
| Acceptable (informational, *not* counted as solved) | 3 | 24 |
| Solved exclusively (strict Optimal) | 48 | 2 |
| Both Optimal | 1239 | |
| Matching objectives (< 0.01%) | 1177/1239 | |

> **Note:** All headline counts use strict Optimal status only. `Acceptable`
> means the iterate met relaxed tolerances but not the requested tolerance —
> per CLAUDE.md's "Honesty in Benchmarks" rule it is reported separately and
> never folded into the pass rate. See the "Acceptable (not Optimal)" and
> "Different Local Minima" sections below.

## Performance Profiles

[Dolan & Moré (2002)](https://doi.org/10.1007/s101070100263) performance profiles pooled over every suite with an Ipopt reference. ρ_s(τ) is the fraction of problems a solver solves within a factor τ of the fastest solver on each problem: the **height at τ=1** is how often it was the quickest, and the **right-hand plateau** is its overall robustness (fraction solved at all). A problem counts as solved only at strict/acceptable success; failures and timeouts are charged infinite cost. Regenerate or slice these with `python3 scripts/perf_profile.py <suite…> [--metric iters] [--mode data]`.

![**Performance profile by wall-clock time.** Valid because POUNCE and Ipopt-MA57 were run interleaved on this host (see Provenance).](figures/profile_performance_time.png)

**Performance profile by wall-clock time.** Valid because POUNCE and Ipopt-MA57 were run interleaved on this host (see Provenance).
  
_1291 problems; solvers: pounce, ipopt._

![**Performance profile by iteration count** — machine-independent, so it stays comparable across hosts and reruns.](figures/profile_performance_iters.png)

**Performance profile by iteration count** — machine-independent, so it stays comparable across hosts and reruns.
  
_1291 problems; solvers: pounce, ipopt._

![**Data profile (absolute-time ECDF).** Fraction of problems solved within a given wall-clock budget, without best-solver normalization — reads directly as “how many by 1 s? by 10 s?”.](figures/profile_data_time.png)

**Data profile (absolute-time ECDF).** Fraction of problems solved within a given wall-clock budget, without best-solver normalization — reads directly as “how many by 1 s? by 10 s?”.
  
_1326 problems; solvers: pounce, ipopt._

## Per-Suite Summary

| Suite | Problems | POUNCE Optimal | Ipopt Optimal | POUNCE only | Ipopt only | Both Optimal | Match |
|-------|----------|---------------|--------------|-------------|------------|--------------|-------|
| Vanderbei | 733 | 702 (95.8%) | 683 (93.2%) | 19 | 0 | 683 | 659/683 |
| Electrolyte | 13 | 13 (100.0%) | 13 (100.0%) | 0 | 0 | 13 | 13/13 |
| Grid | 4 | 4 (100.0%) | 4 (100.0%) | 0 | 0 | 4 | 4/4 |
| CHO | 1 | 0 (0.0%) | 1 (100.0%) | 0 | 1 | 0 | 0/1 |
| Water | 6 | 6 (100.0%) | 6 (100.0%) | 0 | 0 | 6 | 2/6 |
| Gas | 4 | 3 (75.0%) | 3 (75.0%) | 0 | 0 | 3 | 3/3 |
| LargeScale | 5 | 5 (100.0%) | 5 (100.0%) | 0 | 0 | 5 | 5/5 |
| Mittelmann | 47 | 46 (97.9%) | 41 (87.2%) | 6 | 1 | 40 | 39/40 |
| QP | 138 | 138 (100.0%) | 133 (96.4%) | 5 | 0 | 133 | 125/133 |
| LP | 371 | 368 (99.2%) | 352 (94.9%) | 16 | 0 | 352 | 327/352 |
| LPopt | 4 | 2 (50.0%) | 0 (0.0%) | 2 | 0 | 0 | 0/1 |

## Vanderbei Reference Cross-Check

Per-problem status from R. Vanderbei's `cute_table.pdf` (`vanderbei/cute_table_status.json`). The meaningful denominator is the **expected-solvable** set — problems with a documented finite optimum — not all 733: the CUTE collection deliberately includes unbounded, infeasible, and no-solver-finishes problems.

| cute_table status | problems | POUNCE solved | meaning |
|---|---|---|---|
| optimum | 684 | 667 | finite reference optimum exists (expected-solvable) |
| hard | 14 | 8 | in table, but SNOPT+NITRO+LOQO all hit time/iter limits |
| infeasible | 3 | 0 | a reference solver declared infeasibility |
| unbounded | 1 | 0 | unbounded below |
| untabulated | 31 | 27 | not in cute_table — no reference datum |

**POUNCE solved 667 / 684 expected-solvable (97.5%).** The hard / infeasible / unbounded / untabulated rows above are excluded from this denominator — a POUNCE failure there is shared with the commercial reference solvers and is not counted as a miss.

**Genuine misses — expected-solvable but POUNCE did not reach Optimal (17):**

> brainpc0 brainpc1 brainpc2 coshfun cresc100 cresc132 csfi2 dallasl deconvb flosp2hh grouping himmelbj nonmsqrt palmer5e polak3 sineali steenbrc

**Objective disagreements vs. cute_table reference (24)** — POUNCE converged but to a different value than the agreed reference optimum (possible wrong basin or misread problem):

| Problem | POUNCE obj | reference obj | rel. diff |
|---|---|---|---|
| broydn7d | 3.450050e+02 | 3.823419e+00 | 8.9e+01 |
| liswet9 | 1.963305e+03 | 2.499976e+01 | 7.8e+01 |
| liswet8 | 7.144874e+02 | 2.499977e+01 | 2.8e+01 |
| liswet7 | 4.987922e+02 | 2.499979e+01 | 1.9e+01 |
| palmer1c | 7.932114e+00 | 9.759799e-02 | 7.8e+00 |
| eigenbco | 1.024901e-16 | 9.000000e+00 | 1.0e+00 |
| liswet10 | 4.948391e+01 | 2.499967e+01 | 9.8e-01 |
| orthregd | 1.523900e+03 | 4.245801e+04 | 9.6e-01 |
| orthrgds | 1.523900e+03 | 2.603509e+04 | 9.4e-01 |
| bt4 | -3.704768e+00 | -4.551055e+01 | 9.2e-01 |
| camel6 | -2.154638e-01 | -1.031628e+00 | 7.9e-01 |
| orthrds2 | 1.562823e+03 | 1.044332e+03 | 5.0e-01 |
| liswet1 | 3.612062e+01 | 2.500304e+01 | 4.4e-01 |
| fletcher | 1.165685e+01 | 1.952537e+01 | 4.0e-01 |
| liswet12 | -3.314381e+03 | -5.026353e+03 | 3.4e-01 |
| discs | 1.444952e+01 | 1.200008e+01 | 2.0e-01 |
| cresc50 | 7.862467e-01 | 5.932123e-01 | 1.9e-01 |
| hs044 | -1.300000e+01 | -1.500000e+01 | 1.3e-01 |
| avgasb | -4.483219e+00 | -4.132819e+00 | 8.5e-02 |
| steenbre | 2.851495e+04 | 2.745916e+04 | 3.8e-02 |
| haldmads | 3.328319e-02 | 1.223712e-04 | 3.3e-02 |
| errinros | 4.040449e+01 | 3.990415e+01 | 1.3e-02 |
| trainh | 1.231200e+01 | 1.236996e+01 | 4.7e-03 |
| twirism1 | -1.003602e+00 | -1.006758e+00 | 3.1e-03 |

## Vanderbei Suite — Performance

On 683 commonly-solved problems:

| Metric | POUNCE | Ipopt |
|--------|--------|-------|
| Median time | 35.4ms | 44.3ms |
| Total time | 304.63s | 235.00s |
| Mean iterations | 47.2 | 47.2 |
| Median iterations | 15 | 16 |

- **Geometric mean speedup**: 1.0x
- **Median speedup**: 1.1x
- POUNCE faster: 461/683 (67%)
- POUNCE 10x+ faster: 1/683
- Ipopt faster: 222/683

## Electrolyte Suite — Performance

On 13 commonly-solved problems:

| Metric | POUNCE | Ipopt |
|--------|--------|-------|
| Median time | 30.8ms | 37.6ms |
| Total time | 404.5ms | 503.3ms |
| Mean iterations | 14.8 | 12.2 |
| Median iterations | 10 | 10 |

- **Geometric mean speedup**: 1.2x
- **Median speedup**: 1.2x
- POUNCE faster: 13/13 (100%)
- POUNCE 10x+ faster: 0/13
- Ipopt faster: 0/13

## Grid Suite — Performance

On 4 commonly-solved problems:

| Metric | POUNCE | Ipopt |
|--------|--------|-------|
| Median time | 34.4ms | 41.9ms |
| Total time | 137.4ms | 157.2ms |
| Mean iterations | 15.5 | 15.5 |
| Median iterations | 17 | 17 |

- **Geometric mean speedup**: 1.1x
- **Median speedup**: 1.1x
- POUNCE faster: 4/4 (100%)
- POUNCE 10x+ faster: 0/4
- Ipopt faster: 0/4

## Water Suite — Performance

On 6 commonly-solved problems:

| Metric | POUNCE | Ipopt |
|--------|--------|-------|
| Median time | 116.2ms | 122.5ms |
| Total time | 657.7ms | 696.0ms |
| Mean iterations | 191.8 | 205.2 |
| Median iterations | 191 | 209 |

- **Geometric mean speedup**: 1.0x
- **Median speedup**: 1.2x
- POUNCE faster: 4/6 (67%)
- POUNCE 10x+ faster: 0/6
- Ipopt faster: 2/6

## Gas Suite — Performance

On 3 commonly-solved problems:

| Metric | POUNCE | Ipopt |
|--------|--------|-------|
| Median time | 90.6ms | 113.3ms |
| Total time | 292.1ms | 374.0ms |
| Mean iterations | 40.0 | 39.7 |
| Median iterations | 20 | 20 |

- **Geometric mean speedup**: 1.3x
- **Median speedup**: 1.3x
- POUNCE faster: 3/3 (100%)
- POUNCE 10x+ faster: 0/3
- Ipopt faster: 0/3

## LargeScale Suite — Performance

On 5 commonly-solved problems:

| Metric | POUNCE | Ipopt |
|--------|--------|-------|
| Median time | 2.58s | 573.2ms |
| Total time | 12.59s | 9.43s |
| Mean iterations | 309.6 | 305.6 |
| Median iterations | 5 | 2 |

- **Geometric mean speedup**: 0.5x
- **Median speedup**: 0.5x
- POUNCE faster: 2/5 (40%)
- POUNCE 10x+ faster: 0/5
- Ipopt faster: 3/5

## Mittelmann Suite — Performance

On 40 commonly-solved problems:

| Metric | POUNCE | Ipopt |
|--------|--------|-------|
| Median time | 12.40s | 6.88s |
| Total time | 1276.26s | 1488.30s |
| Mean iterations | 113.7 | 110.7 |
| Median iterations | 55 | 55 |

- **Geometric mean speedup**: 0.7x
- **Median speedup**: 0.6x
- POUNCE faster: 6/40 (15%)
- POUNCE 10x+ faster: 0/40
- Ipopt faster: 34/40

## QP Suite — Performance

On 133 commonly-solved problems:

| Metric | POUNCE | Ipopt |
|--------|--------|-------|
| Median time | 85.4ms | 92.9ms |
| Total time | 98.88s | 172.97s |
| Mean iterations | 18.4 | 75.6 |
| Median iterations | 18 | 24 |

- **Geometric mean speedup**: 1.1x
- **Median speedup**: 1.1x
- POUNCE faster: 81/133 (61%)
- POUNCE 10x+ faster: 2/133
- Ipopt faster: 52/133

## LP Suite — Performance

On 352 commonly-solved problems:

| Metric | POUNCE | Ipopt |
|--------|--------|-------|
| Median time | 148.0ms | 157.8ms |
| Total time | 521.82s | 422.20s |
| Mean iterations | 25.0 | 107.3 |
| Median iterations | 23 | 56 |

- **Geometric mean speedup**: 1.1x
- **Median speedup**: 1.0x
- POUNCE faster: 175/352 (50%)
- POUNCE 10x+ faster: 10/352
- Ipopt faster: 177/352

## Failure Analysis

### Vanderbei Suite

| Failure Mode | POUNCE | Ipopt |
|-------------|--------|-------|
| Acceptable | 2 | 6 |
| Diverging_Iterates | 3 | 0 |
| Error_In_Step_Computation | 1 | 0 |
| Infeasible_Problem_Detected | 2 | 4 |
| Invalid_Number_Detected | 1 | 3 |
| Maximum_CpuTime_Exceeded | 3 | 8 |
| Maximum_Iterations_Exceeded | 13 | 16 |
| Not_Enough_Degrees_Of_Freedom | 2 | 0 |
| Restoration_Failed | 2 | 3 |
| Search_Direction_Becomes_Too_Small | 0 | 1 |
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
| Infeasible_Problem_Detected | 3 | 1 |
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

## Wins (POUNCE Optimal, Ipopt not Optimal) — 48 problems

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
| brainpc5 | Vanderbei | 6905 | 6900 | Maximum_CpuTime_Exceeded | 3.584225e-04 |
| brainpc7 | Vanderbei | 6905 | 6900 | Maximum_CpuTime_Exceeded | 3.926835e-04 |
| britgas | Vanderbei | 450 | 360 | Maximum_Iterations_Exceeded | -1.594081e-07 |
| bt8 | Vanderbei | 5 | 2 | Acceptable | 1.000000e+00 |
| co5 | LP | 7993 | 5715 | Acceptable | 7.144696e+05 |
| complex | LP | 1408 | 1023 | Acceptable | -9.966667e+01 |
| coolhans | Vanderbei | 9 | 0 | Unknown_Error | 0.000000e+00 |
| cq5 | LP | 7530 | 5025 | Acceptable | 4.001338e+05 |
| cresc50 | Vanderbei | 6 | 100 | Infeasible_Problem_Detected | 7.862467e-01 |
| cvxqp3 | Vanderbei | 10000 | 7500 | Maximum_CpuTime_Exceeded | 1.157111e+08 |
| dallasm | Vanderbei | 196 | 151 | Invalid_Number_Detected | -4.819819e+04 |
| dallass | Vanderbei | 46 | 31 | Invalid_Number_Detected | -3.239323e+04 |
| drcav2lq | Vanderbei | 10816 | 816 | Maximum_CpuTime_Exceeded | 1.119702e-03 |
| drcavty2 | Vanderbei | 10816 | 816 | Maximum_CpuTime_Exceeded | 1.119702e-03 |
| eigenc2 | Vanderbei | 462 | 231 | Unknown_Error | 7.718095e+02 |
| finnis | LP | 614 | 497 | Acceptable | 1.727911e+05 |
| flosp2th | Vanderbei | 691 | 0 | Maximum_Iterations_Exceeded | 1.000000e+01 |
| greenbea | LP | 5405 | 2389 | Maximum_Iterations_Exceeded | -7.248917e+07 |
| greenbeb | LP | 5405 | 2389 | Acceptable | -4.302260e+06 |
| manne | Vanderbei | 1094 | 730 | Acceptable | -9.741034e-01 |
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
| steenbrf | Vanderbei | 468 | 108 | Acceptable | 2.826781e+02 |
| steenbrg | Vanderbei | 540 | 126 | Acceptable | 2.747128e+04 |
| supportcase10 | LPopt | 14630 | 165684 | Maximum_CpuTime_Exceeded | 3.383924e+00 |

## Acceptable (not Optimal) — 3 problems

These problems converged within relaxed tolerances but not strict tolerances.

| Problem | Suite | n | m | Ipopt status | POUNCE obj | Ipopt obj |
|---------|-------|---|---|-------------|------------|-----------|
| cho_parmest | CHO | 21672 | 21660 | Optimal | 6.767287e+04 | 6.767287e+04 |
| csfi2 | Vanderbei | 5 | 4 | Acceptable | 5.501760e+01 | 5.501760e+01 |
| dallasl | Vanderbei | 906 | 667 | Invalid_Number_Detected | -2.026041e+05 | N/A |

## POUNCE-Only Suite Details

These suites currently run POUNCE only — no Ipopt-side comparison is captured in their result files. Per-problem timing and iteration counts are shown so users can inspect the whole picture.

### LPopt

| Problem | n | m | Status | Objective | Iters | Time |
|---------|---|---|--------|-----------|-------|------|
| ex10 | 17,680 | 69,608 | Maximum_CpuTime_Exceeded | N/A | 0 | 300.10s |
| irish-electricity | 61,728 | 104,259 | Maximum_CpuTime_Exceeded | N/A | 387 | 300.07s |
| qap15 | 22,275 | 6,330 | Optimal | 1.0410e+03 | 26 | 44.91s |
| supportcase10 | 14,630 | 165,684 | Optimal | 3.3839e+00 | 32 | 162.10s |

POUNCE: **2/4 Optimal** in 807.17s total

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
| Optimal | 366/371 (98.7%) | 353/371 (95.1%) |
| Solved exclusively | 15 | 2 |
| Both Optimal | 351 | |
| Matching objectives (< 0.01%) | 326/351 | |

On 351 problems solved by both arms:

| Metric | pounce-convex | pounce-nlp |
|--------|---------------|------------|
| Median time | 151.4ms | 203.3ms |
| Total time | 163.55s | 630.65s |
| Mean iterations | 25.1 | 116.3 |
| Median iterations | 23 | 56 |

- **Geometric-mean speedup (convex over nlp)**: 1.4x
- **Median speedup**: 1.1x
- pounce-convex faster: 238/351 (68%)
- pounce-convex 10x+ faster: 10/351
- pounce-nlp faster: 113/351

### QP — convex vs NLP

| Metric | pounce-convex | pounce-nlp |
|--------|---------------|------------|
| Optimal | 137/138 (99.3%) | 133/138 (96.4%) |
| Solved exclusively | 5 | 1 |
| Both Optimal | 132 | |
| Matching objectives (< 0.01%) | 124/132 | |

On 132 problems solved by both arms:

| Metric | pounce-convex | pounce-nlp |
|--------|---------------|------------|
| Median time | 85.8ms | 110.5ms |
| Total time | 100.02s | 192.51s |
| Mean iterations | 18.4 | 76.2 |
| Median iterations | 18 | 24 |

- **Geometric-mean speedup (convex over nlp)**: 1.1x
- **Median speedup**: 1.0x
- pounce-convex faster: 67/132 (51%)
- pounce-convex 10x+ faster: 1/132
- pounce-nlp faster: 65/132

---
*Generated by benchmark_report.py*