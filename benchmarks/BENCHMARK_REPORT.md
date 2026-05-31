# POUNCE Benchmark Report

Generated: 2026-05-31 10:31:45

## Provenance

| Component | Version / Detail |
|-----------|------------------|
| POUNCE | v0.3.0 (benchmark-suites-lp-qp @ 608868b-dirty) |
| POUNCE linear solver | feral (default) |
| Ipopt | Ipopt 3.14.20 (Darwin arm64), ASL(20241202) |
| Ipopt linear solver | ma57 (via ref/Ipopt/install-ma57) |
| Platform | Darwin 25.5.0 arm64 |

POUNCE results were produced this run by `make -C benchmarks
<suite>-run` (pounce only). The Ipopt column is a saved reference
(`make -C benchmarks ipopt-reference`), rerun only when explicitly
regenerated — generated 2026-05-30 18:15:14 EDT on Johns-Mac-mini.local (Darwin 25.5.0 arm64), git 911c870, timelimit 300s. Ipopt solve *times* are
from that reference machine and only comparable to POUNCE when this
report is generated on the same host.

The GAMS solver-link path is exercised separately as a liveness
smoke check (`make -C benchmarks gams-bench`) and is not aggregated here.

## Executive Summary

| Metric | POUNCE | Ipopt |
|--------|--------|-------|
| Optimal (strict) | **1224/1326** (92.3%) | **868/1326** (65.5%) |
| Acceptable (informational, *not* counted as solved) | 36 | 10 |
| Solved exclusively (strict Optimal) | 381 | 25 |
| Both Optimal | 843 | |
| Matching objectives (< 0.01%) | 829/843 | |

> **Note:** All headline counts use strict Optimal status only. `Acceptable`
> means the iterate met relaxed tolerances but not the requested tolerance —
> per CLAUDE.md's "Honesty in Benchmarks" rule it is reported separately and
> never folded into the pass rate. See the "Acceptable (not Optimal)" and
> "Different Local Minima" sections below.

## Per-Suite Summary

| Suite | Problems | POUNCE Optimal | Ipopt Optimal | POUNCE only | Ipopt only | Both Optimal | Match |
|-------|----------|---------------|--------------|-------------|------------|--------------|-------|
| Vanderbei | 733 | 677 (92.4%) | 682 (93.0%) | 17 | 22 | 660 | 649/660 |
| Electrolyte | 13 | 13 (100.0%) | 13 (100.0%) | 0 | 0 | 13 | 13/13 |
| Grid | 4 | 4 (100.0%) | 4 (100.0%) | 0 | 0 | 4 | 4/4 |
| CHO | 1 | 1 (100.0%) | 1 (100.0%) | 0 | 0 | 1 | 1/1 |
| Water | 6 | 6 (100.0%) | 6 (100.0%) | 0 | 0 | 6 | 4/6 |
| Gas | 4 | 3 (75.0%) | 3 (75.0%) | 0 | 0 | 3 | 3/3 |
| LargeScale | 5 | 5 (100.0%) | 5 (100.0%) | 0 | 0 | 5 | 5/5 |
| Mittelmann | 47 | 38 (80.9%) | 31 (66.0%) | 8 | 1 | 30 | 30/30 |
| QP | 138 | 133 (96.4%) | 123 (89.1%) | 12 | 2 | 121 | 120/121 |
| LP | 371 | 344 (92.7%) | 0 (0.0%) | 344 | 0 | 0 | 0/1 |
| LPopt | 4 | 0 (0.0%) | 0 (0.0%) | 0 | 0 | 0 | 0/1 |

## Vanderbei Reference Cross-Check

Per-problem status from R. Vanderbei's `cute_table.pdf` (`vanderbei/cute_table_status.json`). The meaningful denominator is the **expected-solvable** set — problems with a documented finite optimum — not all 733: the CUTE collection deliberately includes unbounded, infeasible, and no-solver-finishes problems.

| cute_table status | problems | POUNCE solved | meaning |
|---|---|---|---|
| optimum | 684 | 643 | finite reference optimum exists (expected-solvable) |
| hard | 14 | 7 | in table, but SNOPT+NITRO+LOQO all hit time/iter limits |
| infeasible | 3 | 0 | a reference solver declared infeasibility |
| unbounded | 1 | 0 | unbounded below |
| untabulated | 31 | 27 | not in cute_table — no reference datum |

**POUNCE solved 643 / 684 expected-solvable (94.0%).** The hard / infeasible / unbounded / untabulated rows above are excluded from this denominator — a POUNCE failure there is shared with the commercial reference solvers and is not counted as a miss.

**Genuine misses — expected-solvable but POUNCE did not reach Optimal (41):**

> artif brainpc0 britgas chebyqad coshfun cragglvy cresc100 cresc132 cresc4 cresc50 cvxqp3 dallasl dallasm dallass discs djtl flosp2hh flosp2hm grouping helix himmelbj hubfit nonmsqrt orthrds2 palmer5e palmer7e polak3 s368 sawpath scon1dls semicon1 sensors sineali spanhyd ssebnln steenbrc steenbrf trainh twirism1 yfit yfitu

**Objective disagreements vs. cute_table reference (117)** — POUNCE converged but to a different value than the agreed reference optimum (possible wrong basin or misread problem):

| Problem | POUNCE obj | reference obj | rel. diff |
|---|---|---|---|
| broydn7d | 3.450050e+02 | 3.823419e+00 | 8.9e+01 |
| liswet9 | 1.899426e+03 | 2.499976e+01 | 7.5e+01 |
| liswet8 | 6.509716e+02 | 2.499977e+01 | 2.5e+01 |
| liswet7 | 3.911518e+02 | 2.499979e+01 | 1.5e+01 |
| eigenbco | 5.617567e-26 | 9.000000e+00 | 1.0e+00 |
| scurly10 | -1.003163e-02 | -1.003163e+06 | 1.0e+00 |
| scurly20 | -1.003163e-02 | -1.003163e+06 | 1.0e+00 |
| scurly30 | -1.003163e-02 | -1.003163e+06 | 1.0e+00 |
| meyer3 | 8.794586e-07 | 8.794586e+01 | 1.0e+00 |
| hs099 | -3.456347e+02 | -8.310799e+08 | 1.0e+00 |
| himmelbd | 6.699286e-05 | 5.922563e+00 | 1.0e+00 |
| arglinb | 1.469851e-04 | 4.634146e+00 | 1.0e+00 |
| hs084 | -2.872406e+02 | -5.280335e+06 | 1.0e+00 |
| brownden | 4.823391e+00 | 8.582220e+04 | 1.0e+00 |
| arglinc | 3.679793e-04 | 6.135135e+00 | 1.0e+00 |
| lakes | 6.298140e+01 | 3.505248e+05 | 1.0e+00 |
| optctrl3 | 5.120041e-01 | 2.048017e+03 | 1.0e+00 |
| optctrl6 | 5.120041e-01 | 2.048017e+03 | 1.0e+00 |
| bdqrtic | 1.333272e+00 | 3.983818e+03 | 1.0e+00 |
| penalty2 | 3.848429e+01 | 9.709608e+04 | 1.0e+00 |
| bratu1d | -4.242482e-03 | -8.518927e+00 | 1.0e+00 |
| hs064 | 4.375194e+00 | 6.299842e+03 | 1.0e+00 |
| scosine | -8.553258e+00 | -9.999000e+03 | 1.0e+00 |
| jensmp | 1.422873e-01 | 1.243622e+02 | 1.0e+00 |
| errinros | 4.833189e-02 | 3.990415e+01 | 1.0e+00 |
| cvxbqp1 | 4.286142e+03 | 2.250225e+06 | 1.0e+00 |
| ncvxbqp1 | -3.781988e+07 | -1.985544e+10 | 1.0e+00 |
| cvxqp2 | 1.558904e+05 | 8.184246e+07 | 1.0e+00 |
| lch | -1.649122e-02 | -4.318289e+00 | 1.0e+00 |
| avion2 | 4.115108e+05 | 9.468017e+07 | 1.0e+00 |
| chainwoo | 2.788601e-01 | 6.362471e+01 | 1.0e+00 |
| hs119 | 1.313276e+00 | 2.448997e+02 | 9.9e-01 |
| palmer1b | 2.409219e-02 | 3.447355e+00 | 9.9e-01 |
| 3pk | 1.677022e-02 | 1.720119e+00 | 9.9e-01 |
| hs062 | -2.624873e+02 | -2.627251e+04 | 9.9e-01 |
| himmelbf | 3.640806e+00 | 3.185717e+02 | 9.9e-01 |
| model | 7.764926e+01 | 5.742163e+03 | 9.9e-01 |
| steenbre | 5.005874e+02 | 2.745916e+04 | 9.8e-01 |
| ncvxqp4 | -1.789977e+06 | -9.397879e+07 | 9.8e-01 |
| cvxqp1 | 2.071451e+04 | 1.087512e+06 | 9.8e-01 |
| hs107 | 1.027441e+02 | 5.055012e+03 | 9.8e-01 |
| ncvxqp2 | -1.541020e+06 | -5.781269e+07 | 9.7e-01 |
| steenbrd | 2.614067e+02 | 9.030082e+03 | 9.7e-01 |
| steenbrb | 2.627318e+02 | 9.075855e+03 | 9.7e-01 |
| palmer4b | 2.097297e-01 | 6.835139e+00 | 9.7e-01 |
| palmer3b | 1.373521e-01 | 4.227647e+00 | 9.7e-01 |
| orthregd | 1.523900e+03 | 4.245801e+04 | 9.6e-01 |
| palmer1 | 4.262081e+02 | 1.175460e+04 | 9.6e-01 |
| lsnnodoc | 4.808004e+00 | 1.231124e+02 | 9.6e-01 |
| hs020 | 1.670770e+00 | 4.019873e+01 | 9.6e-01 |
| hs017 | 4.156277e-02 | 1.000000e+00 | 9.6e-01 |
| eigenc2 | 4.011484e+01 | 7.718095e+02 | 9.5e-01 |
| orthrgds | 1.523900e+03 | 2.603509e+04 | 9.4e-01 |
| palmer5d | 5.193696e+00 | 8.733940e+01 | 9.4e-01 |
| sseblin | 1.078040e+06 | 1.617060e+07 | 9.3e-01 |
| palmer2 | 2.646634e+02 | 3.651090e+03 | 9.3e-01 |
| freuroth | 4.458645e+04 | 6.081592e+05 | 9.3e-01 |
| bt4 | -3.704768e+00 | -4.551055e+01 | 9.2e-01 |
| explin | -6.031302e+04 | -7.237563e+05 | 9.2e-01 |
| expquad | -3.020500e+05 | -3.624600e+06 | 9.2e-01 |
| explin2 | -6.037160e+04 | -7.244591e+05 | 9.2e-01 |
| qrtquad | -3.065620e+05 | -3.648088e+06 | 9.2e-01 |
| powell20 | 5.211962e+06 | 5.214578e+07 | 9.0e-01 |
| palmer5c | 2.632140e-01 | 2.128087e+00 | 8.8e-01 |
| hs103 | 7.378044e+01 | 5.436680e+02 | 8.6e-01 |
| hs102 | 1.239180e+02 | 9.118806e+02 | 8.6e-01 |
| hs101 | 2.461046e+02 | 1.809765e+03 | 8.6e-01 |
| hs019 | -1.157377e+03 | -6.961814e+03 | 8.3e-01 |
| camel6 | -2.154638e-01 | -1.031628e+00 | 7.9e-01 |
| catena | -5.175437e+03 | -2.307775e+04 | 7.8e-01 |
| hong | 3.964576e-01 | 1.347307e+00 | 7.1e-01 |
| fccu | 3.716370e+00 | 1.114911e+01 | 6.7e-01 |
| hs083 | -1.059902e+04 | -3.066554e+04 | 6.5e-01 |
| palmer1d | 1.550473e-06 | 6.526826e-01 | 6.5e-01 |
| palmer2b | 1.952286e-02 | 6.233947e-01 | 6.0e-01 |
| palmer7c | 1.385267e-05 | 6.019857e-01 | 6.0e-01 |
| batch | 1.036721e+05 | 2.591804e+05 | 6.0e-01 |
| gpp | 5.783505e+03 | 1.440093e+04 | 6.0e-01 |
| liswet10 | 3.930234e+01 | 2.499967e+01 | 5.7e-01 |
| harkerp2 | -3.697727e-05 | -5.000000e-01 | 5.0e-01 |
| hs114 | -9.211388e+02 | -1.768807e+03 | 4.8e-01 |
| hs105 | 5.918833e+02 | 1.136361e+03 | 4.8e-01 |
| eigena2 | 4.583333e+01 | 8.250000e+01 | 4.4e-01 |
| fletcher | 1.165685e+01 | 1.952537e+01 | 4.0e-01 |
| eigenb2 | 1.000000e+00 | 1.600000e+00 | 3.8e-01 |
| dixmaand | 6.503642e-01 | 1.000000e+00 | 3.5e-01 |
| dixmaanh | 6.560532e-01 | 1.000000e+00 | 3.4e-01 |
| dixmaanl | 6.599015e-01 | 1.000000e+00 | 3.4e-01 |
| liswet12 | -3.379107e+03 | -5.026353e+03 | 3.3e-01 |
| smmpsf | 7.321578e+05 | 1.046985e+06 | 3.0e-01 |
| cliff | 2.135747e-09 | 1.997866e-01 | 2.0e-01 |
| engval1 | 4.474733e+03 | 5.548668e+03 | 1.9e-01 |
| hs093 | 1.102206e+02 | 1.350760e+02 | 1.8e-01 |
| hairy | 1.639999e+01 | 2.000000e+01 | 1.8e-01 |
| qudlin | -6.000000e+03 | -7.200000e+03 | 1.7e-01 |
| palmer8c | 1.415888e-05 | 1.597681e-01 | 1.6e-01 |
| hs044 | -1.300000e+01 | -1.500000e+01 | 1.3e-01 |
| hs113 | 2.170197e+01 | 2.430621e+01 | 1.1e-01 |
| palmer1c | 1.984316e-08 | 9.759799e-02 | 9.8e-02 |
| palmer1a | 4.213860e-04 | 8.988363e-02 | 8.9e-02 |
| avgasb | -4.483219e+00 | -4.132819e+00 | 8.5e-02 |
| liswet1 | 2.712027e+01 | 2.500304e+01 | 8.5e-02 |
| mwright | 2.312853e+01 | 2.497881e+01 | 7.4e-02 |
| palmer8a | 1.927299e-03 | 7.400970e-02 | 7.2e-02 |
| palmer6a | 2.800103e-03 | 5.594884e-02 | 5.3e-02 |
| palmer4c | 4.753927e-07 | 5.031070e-02 | 5.0e-02 |
| palmer4a | 7.908980e-04 | 4.060614e-02 | 4.0e-02 |
| expfitc | 1.159793e-04 | 2.330257e-02 | 2.3e-02 |
| palmer3a | 4.343676e-04 | 2.043142e-02 | 2.0e-02 |
| palmer3c | 1.843055e-07 | 1.953764e-02 | 2.0e-02 |
| palmer2a | 3.838581e-04 | 1.716074e-02 | 1.7e-02 |
| palmer6c | 1.644281e-06 | 1.638742e-02 | 1.6e-02 |
| palmer2c | 3.935677e-08 | 1.442139e-02 | 1.4e-02 |
| palmer5b | 1.928087e-05 | 9.752493e-03 | 9.7e-03 |
| penalty1 | 6.439498e-08 | 9.686175e-03 | 9.7e-03 |
| palmer8e | 5.501854e-05 | 6.339307e-03 | 6.3e-03 |
| expfitb | 1.249092e-04 | 5.019366e-03 | 4.9e-03 |

## Vanderbei Suite — Performance

On 660 commonly-solved problems:

| Metric | POUNCE | Ipopt |
|--------|--------|-------|
| Median time | 32.1ms | 36.7ms |
| Total time | 237.67s | 135.68s |
| Mean iterations | 45.9 | 45.6 |
| Median iterations | 15 | 16 |

- **Geometric mean speedup**: 0.9x
- **Median speedup**: 1.1x
- POUNCE faster: 384/660 (58%)
- POUNCE 10x+ faster: 0/660
- Ipopt faster: 276/660

## Electrolyte Suite — Performance

On 13 commonly-solved problems:

| Metric | POUNCE | Ipopt |
|--------|--------|-------|
| Median time | 29.8ms | 31.5ms |
| Total time | 393.3ms | 412.3ms |
| Mean iterations | 12.0 | 12.2 |
| Median iterations | 10 | 10 |

- **Geometric mean speedup**: 1.0x
- **Median speedup**: 1.1x
- POUNCE faster: 12/13 (92%)
- POUNCE 10x+ faster: 0/13
- Ipopt faster: 1/13

## Grid Suite — Performance

On 4 commonly-solved problems:

| Metric | POUNCE | Ipopt |
|--------|--------|-------|
| Median time | 34.1ms | 33.9ms |
| Total time | 137.7ms | 133.6ms |
| Mean iterations | 15.5 | 15.5 |
| Median iterations | 17 | 17 |

- **Geometric mean speedup**: 1.0x
- **Median speedup**: 1.0x
- POUNCE faster: 1/4 (25%)
- POUNCE 10x+ faster: 0/4
- Ipopt faster: 3/4

## CHO Suite — Performance

On 1 commonly-solved problems:

| Metric | POUNCE | Ipopt |
|--------|--------|-------|
| Median time | 4.51s | 1.61s |
| Total time | 4.51s | 1.61s |
| Mean iterations | 35.0 | 33.0 |
| Median iterations | 35 | 33 |

- **Geometric mean speedup**: 0.4x
- **Median speedup**: 0.4x
- POUNCE faster: 0/1 (0%)
- POUNCE 10x+ faster: 0/1
- Ipopt faster: 1/1

## Water Suite — Performance

On 6 commonly-solved problems:

| Metric | POUNCE | Ipopt |
|--------|--------|-------|
| Median time | 98.4ms | 70.1ms |
| Total time | 655.4ms | 412.9ms |
| Mean iterations | 210.8 | 205.2 |
| Median iterations | 183 | 209 |

- **Geometric mean speedup**: 0.7x
- **Median speedup**: 0.7x
- POUNCE faster: 0/6 (0%)
- POUNCE 10x+ faster: 0/6
- Ipopt faster: 6/6

## Gas Suite — Performance

On 3 commonly-solved problems:

| Metric | POUNCE | Ipopt |
|--------|--------|-------|
| Median time | 81.6ms | 70.9ms |
| Total time | 276.7ms | 202.2ms |
| Mean iterations | 39.0 | 39.7 |
| Median iterations | 20 | 20 |

- **Geometric mean speedup**: 0.8x
- **Median speedup**: 0.9x
- POUNCE faster: 1/3 (33%)
- POUNCE 10x+ faster: 0/3
- Ipopt faster: 2/3

## LargeScale Suite — Performance

On 5 commonly-solved problems:

| Metric | POUNCE | Ipopt |
|--------|--------|-------|
| Median time | 2.61s | 467.8ms |
| Total time | 27.11s | 3.31s |
| Mean iterations | 306.4 | 305.6 |
| Median iterations | 1 | 2 |

- **Geometric mean speedup**: 0.3x
- **Median speedup**: 0.4x
- POUNCE faster: 2/5 (40%)
- POUNCE 10x+ faster: 0/5
- Ipopt faster: 3/5

## Mittelmann Suite — Performance

On 30 commonly-solved problems:

| Metric | POUNCE | Ipopt |
|--------|--------|-------|
| Median time | 8.20s | 3.09s |
| Total time | 656.43s | 786.19s |
| Mean iterations | 96.2 | 95.2 |
| Median iterations | 47 | 41 |

- **Geometric mean speedup**: 0.6x
- **Median speedup**: 0.6x
- POUNCE faster: 10/30 (33%)
- POUNCE 10x+ faster: 0/30
- Ipopt faster: 20/30

## QP Suite — Performance

On 121 commonly-solved problems:

| Metric | POUNCE | Ipopt |
|--------|--------|-------|
| Median time | 97.4ms | 114.2ms |
| Total time | 53.12s | 72.01s |
| Mean iterations | 78.3 | 76.4 |
| Median iterations | 24 | 24 |

- **Geometric mean speedup**: 1.1x
- **Median speedup**: 1.1x
- POUNCE faster: 72/121 (60%)
- POUNCE 10x+ faster: 3/121
- Ipopt faster: 49/121

## Failure Analysis

### Vanderbei Suite

| Failure Mode | POUNCE | Ipopt |
|-------------|--------|-------|
| Acceptable | 8 | 6 |
| Infeasible_Problem_Detected | 3 | 4 |
| Invalid_Number_Detected | 0 | 3 |
| Maximum_CpuTime_Exceeded | 3 | 9 |
| Maximum_Iterations_Exceeded | 13 | 16 |
| Restoration_Failed | 1 | 3 |
| Search_Direction_Becomes_Too_Small | 2 | 1 |
| Solver_Error | 26 | 2 |
| Unknown_Error | 0 | 7 |

### Gas Suite

| Failure Mode | POUNCE | Ipopt |
|-------------|--------|-------|
| Infeasible_Problem_Detected | 1 | 1 |

### Mittelmann Suite

| Failure Mode | POUNCE | Ipopt |
|-------------|--------|-------|
| Acceptable | 2 | 0 |
| Maximum_CpuTime_Exceeded | 7 | 12 |
| Maximum_Iterations_Exceeded | 0 | 3 |
| Solver_Error | 0 | 1 |

### QP Suite

| Failure Mode | POUNCE | Ipopt |
|-------------|--------|-------|
| Acceptable | 3 | 4 |
| Maximum_CpuTime_Exceeded | 1 | 11 |
| Search_Direction_Becomes_Too_Small | 1 | 0 |

### LP Suite

| Failure Mode | POUNCE | Ipopt |
|-------------|--------|-------|
| Acceptable | 23 | 0 |
| Infeasible_Problem_Detected | 1 | 0 |
| Maximum_CpuTime_Exceeded | 1 | 0 |
| Maximum_Iterations_Exceeded | 1 | 0 |
| N/A | 0 | 371 |
| Solver_Error | 1 | 0 |

### LPopt Suite

| Failure Mode | POUNCE | Ipopt |
|-------------|--------|-------|
| Maximum_CpuTime_Exceeded | 3 | 0 |
| Maximum_Iterations_Exceeded | 1 | 0 |
| N/A | 0 | 4 |

## Regressions (Ipopt Optimal, POUNCE not Optimal)

| Problem | Suite | n | m | POUNCE status | Ipopt obj |
|---------|-------|---|---|--------------|-----------|
| QSHELL | QP | 1775 | 536 | Acceptable | 2.674980e+08 |
| QSIERRA | QP | 2036 | 1227 | Search_Direction_Becomes_Too_Small | 3.762568e+04 |
| artif | Vanderbei | 5002 | 2 | Solver_Error | 1.897534e-17 |
| chebyqad | Vanderbei | 50 | 0 | Solver_Error | 5.386315e-03 |
| cragglvy | Vanderbei | 5000 | 0 | Solver_Error | 2.988096e+01 |
| cresc4 | Vanderbei | 6 | 8 | Solver_Error | 8.718975e-01 |
| discs | Vanderbei | 36 | 69 | Infeasible_Problem_Detected | 1.444952e+01 |
| djtl | Vanderbei | 2 | 0 | Solver_Error | -8.951545e-05 |
| flosp2hm | Vanderbei | 691 | 0 | Maximum_Iterations_Exceeded | 1.598325e-02 |
| helix | Vanderbei | 3 | 0 | Solver_Error | 3.794741e-26 |
| hubfit | Vanderbei | 2 | 1 | Solver_Error | 1.689350e-02 |
| orthrds2 | Vanderbei | 203 | 100 | Acceptable | 1.544297e+03 |
| qcqp1000-1nc | Mittelmann | 1000 | 154 | Acceptable | -1.841514e+06 |
| s332 | Vanderbei | 2 | 102 | Solver_Error | 1.874879e+01 |
| s368 | Vanderbei | 100 | 0 | Acceptable | 2.005351e-18 |
| sawpath | Vanderbei | 593 | 786 | Infeasible_Problem_Detected | 1.815730e+02 |
| scon1dls | Vanderbei | 1002 | 2 | Solver_Error | 3.511377e-10 |
| semicon1 | Vanderbei | 1002 | 2 | Solver_Error | 3.511377e-10 |
| sensors | Vanderbei | 1000 | 0 | Maximum_CpuTime_Exceeded | -8.600696e+04 |
| spanhyd | Vanderbei | 97 | 33 | Acceptable | 5.193068e-04 |
| ssebnln | Vanderbei | 194 | 96 | Acceptable | 1.078040e+06 |
| trainh | Vanderbei | 20008 | 10002 | Solver_Error | 1.231200e+01 |
| twirism1 | Vanderbei | 343 | 313 | Search_Direction_Becomes_Too_Small | -1.001141e+00 |
| yfit | Vanderbei | 3 | 0 | Solver_Error | 1.261201e-14 |
| yfitu | Vanderbei | 3 | 0 | Solver_Error | 1.252218e-14 |

## Wins (POUNCE Optimal, Ipopt not Optimal) — 381 problems

| Problem | Suite | n | m | Ipopt status | POUNCE obj |
|---------|-------|---|---|-------------|------------|
| 25fv47 | LP | 1571 | 820 | N/A | 5.501846e+03 |
| 80bau3b | LP | 9799 | 2237 | N/A | 2.948346e+04 |
| BOYD1 | QP | 93261 | 18 | Acceptable | -6.415515e+05 |
| CONT-100 | QP | 10197 | 9801 | Maximum_CpuTime_Exceeded | -4.644398e+00 |
| CONT-101 | QP | 10197 | 10098 | Maximum_CpuTime_Exceeded | 1.955273e-01 |
| CONT-200 | QP | 40397 | 39601 | Maximum_CpuTime_Exceeded | -4.684876e+00 |
| CONT-201 | QP | 40397 | 40198 | Maximum_CpuTime_Exceeded | 1.924834e-01 |
| CONT-300 | QP | 90597 | 90298 | Maximum_CpuTime_Exceeded | 1.915123e-01 |
| CVXQP1_L | QP | 10000 | 5000 | Maximum_CpuTime_Exceeded | 1.035284e+06 |
| CVXQP2_L | QP | 10000 | 2500 | Maximum_CpuTime_Exceeded | 7.794520e+05 |
| EXDATA | QP | 3000 | 3001 | Maximum_CpuTime_Exceeded | -1.418434e+02 |
| QPILOTNO | QP | 2172 | 975 | Acceptable | 6.433450e+05 |
| QSCORPIO | QP | 358 | 388 | Acceptable | 9.357630e+02 |
| STCQP1 | QP | 4097 | 2052 | Maximum_CpuTime_Exceeded | 1.551436e+05 |
| aa01 | LP | 8904 | 823 | N/A | 2.459495e+03 |
| aa03 | LP | 8627 | 825 | N/A | 2.668980e+03 |
| aa3 | LP | 8627 | 825 | N/A | 2.668980e+03 |
| aa5 | LP | 8308 | 801 | N/A | 2.441431e+03 |
| aa6 | LP | 7292 | 646 | N/A | 2.882177e+03 |
| adlittle | LP | 97 | 56 | N/A | 6.812537e+03 |
| afiro | LP | 32 | 27 | N/A | -4.647531e+02 |
| air02 | LP | 6774 | 50 | N/A | 7.951536e+01 |
| air04 | LP | 8904 | 823 | N/A | 2.459495e+03 |
| air05 | LP | 7195 | 426 | N/A | 9.659420e+02 |
| air06 | LP | 8627 | 825 | N/A | 2.668980e+03 |
| aircraft | LP | 7517 | 3754 | N/A | 1.567042e+03 |
| bandm | LP | 472 | 305 | N/A | -1.586280e+02 |
| beaconfd | LP | 262 | 173 | N/A | 3.081879e+04 |
| blend | LP | 83 | 74 | N/A | -3.081215e+01 |
| bnl1 | LP | 1175 | 632 | N/A | 1.977629e+03 |
| bnl2 | LP | 3489 | 2280 | N/A | 1.811237e+03 |
| boeing1 | LP | 384 | 348 | N/A | -3.352136e+02 |
| boeing2 | LP | 143 | 140 | N/A | -3.150187e+02 |
| brainpc1 | Vanderbei | 6905 | 6900 | Restoration_Failed | 4.370310e-04 |
| brainpc2 | Vanderbei | 13805 | 13800 | Maximum_CpuTime_Exceeded | 4.418439e-04 |
| brainpc5 | Vanderbei | 6905 | 6900 | Maximum_CpuTime_Exceeded | 3.750387e-04 |
| brainpc7 | Vanderbei | 6905 | 6900 | Maximum_CpuTime_Exceeded | 3.927054e-04 |
| brandy | LP | 249 | 182 | N/A | 1.518510e+03 |
| bt8 | Vanderbei | 5 | 2 | Acceptable | 1.000000e+00 |
| capri | LP | 353 | 271 | N/A | 2.690013e+03 |
| cari | LP | 1200 | 400 | N/A | 5.818930e+02 |
| cep1 | LP | 3248 | 1521 | N/A | 3.551601e+05 |
| complex | LP | 1408 | 1023 | N/A | -9.966667e+01 |
| coolhans | Vanderbei | 9 | 0 | Unknown_Error | 0.000000e+00 |
| cq5 | LP | 7530 | 5025 | N/A | 2.842296e+03 |
| cr42 | LP | 1513 | 905 | N/A | 2.801850e+01 |
| crew1 | LP | 6469 | 135 | N/A | 2.055556e+01 |
| csfi2 | Vanderbei | 5 | 4 | Acceptable | 5.501760e+01 |
| cycle | LP | 2857 | 1886 | N/A | -5.226389e+00 |
| czprob | LP | 3523 | 927 | N/A | 7.111419e+05 |
| d2q06c | LP | 5167 | 2171 | N/A | 1.227842e+05 |
| d6cube | LP | 6184 | 404 | N/A | 3.154916e+02 |
| de080285 | LP | 1488 | 936 | N/A | 1.392390e+00 |
| deconvb | Vanderbei | 51 | 0 | Maximum_Iterations_Exceeded | 1.974309e-03 |
| degen2 | LP | 534 | 444 | N/A | -1.435178e+03 |
| delf000 | LP | 5464 | 3128 | N/A | 3.073872e-01 |
| delf001 | LP | 5462 | 3098 | N/A | 2.358603e+02 |
| delf002 | LP | 5460 | 3135 | N/A | 2.830333e-01 |
| delf003 | LP | 5460 | 3065 | N/A | 9.115538e+02 |
| delf004 | LP | 5464 | 3142 | N/A | 1.586912e+01 |
| delf005 | LP | 5464 | 3103 | N/A | 2.287301e+02 |
| delf006 | LP | 5469 | 3147 | N/A | 2.284287e+01 |
| delf007 | LP | 5471 | 3137 | N/A | 3.799532e+01 |
| delf008 | LP | 5472 | 3148 | N/A | 2.412330e+01 |
| delf009 | LP | 5472 | 3135 | N/A | 4.841221e+01 |
| delf010 | LP | 5472 | 3147 | N/A | 2.291042e+01 |
| delf011 | LP | 5471 | 3134 | N/A | 4.734915e+01 |
| delf012 | LP | 5471 | 3151 | N/A | 1.786776e+01 |
| delf013 | LP | 5472 | 3116 | N/A | 2.616674e+02 |
| delf014 | LP | 5472 | 3170 | N/A | 1.534961e+01 |
| delf015 | LP | 5471 | 3161 | N/A | 7.370340e+01 |
| delf017 | LP | 5471 | 3176 | N/A | 4.615156e+01 |
| delf018 | LP | 5471 | 3196 | N/A | 1.421314e+01 |
| delf019 | LP | 5471 | 3185 | N/A | 2.342124e+02 |
| delf020 | LP | 5472 | 3213 | N/A | 3.516788e+01 |
| delf021 | LP | 5471 | 3208 | N/A | 3.947018e+01 |
| delf022 | LP | 5472 | 3214 | N/A | 3.649904e+01 |
| delf023 | LP | 5472 | 3214 | N/A | 3.541146e+01 |
| delf024 | LP | 5466 | 3207 | N/A | 3.514143e+01 |
| delf025 | LP | 5464 | 3197 | N/A | 3.502944e+01 |
| delf026 | LP | 5462 | 3190 | N/A | 3.516109e+01 |
| delf027 | LP | 5457 | 3187 | N/A | 3.194531e+01 |
| delf028 | LP | 5452 | 3177 | N/A | 2.972432e+01 |
| delf029 | LP | 5454 | 3179 | N/A | 2.752232e+01 |
| delf030 | LP | 5469 | 3199 | N/A | 2.538378e+01 |
| delf031 | LP | 5455 | 3176 | N/A | 2.342961e+01 |
| delf032 | LP | 5467 | 3196 | N/A | 2.160368e+01 |
| delf033 | LP | 5456 | 3173 | N/A | 2.002692e+01 |
| delf034 | LP | 5455 | 3175 | N/A | 1.894350e+01 |
| delf035 | LP | 5468 | 3193 | N/A | 1.769464e+01 |
| delf036 | LP | 5459 | 3170 | N/A | 1.644569e+01 |
| deter0 | LP | 5468 | 1923 | N/A | -2.045908e+00 |
| deter4 | LP | 9133 | 3235 | N/A | -1.428015e+00 |
| df2177 | LP | 9728 | 630 | N/A | 9.099996e+01 |
| dirichlet120 | Mittelmann | 53881 | 241 | Maximum_CpuTime_Exceeded | 3.503609e-02 |
| disp3 | LP | 1856 | 2182 | N/A | 8.083178e+04 |
| drcav2lq | Vanderbei | 10816 | 816 | Maximum_CpuTime_Exceeded | 1.104453e-03 |
| drcavty1 | Vanderbei | 10816 | 816 | Maximum_CpuTime_Exceeded | 2.651744e-14 |
| drcavty2 | Vanderbei | 10816 | 816 | Maximum_CpuTime_Exceeded | 1.104453e-03 |
| dsbmip | LP | 1877 | 1182 | N/A | -3.051982e+02 |
| e226 | LP | 282 | 223 | N/A | -1.163893e+01 |
| eigenc2 | Vanderbei | 462 | 231 | Unknown_Error | 4.011484e+01 |
| etamacro | LP | 688 | 400 | N/A | -9.693758e+01 |
| ex1_320 | Mittelmann | 203522 | 101761 | Maximum_CpuTime_Exceeded | 6.541997e-02 |
| ex4_2_320 | Mittelmann | 204798 | 103037 | Maximum_CpuTime_Exceeded | 3.639168e+00 |
| farm | LP | 12 | 7 | N/A | 4.385965e+03 |
| fffff800 | LP | 854 | 524 | N/A | 5.556796e+05 |
| finnis | LP | 614 | 497 | N/A | 4.288360e+03 |
| fit1d | LP | 1026 | 24 | N/A | -6.351652e+02 |
| fit1p | LP | 1677 | 627 | N/A | 9.146378e+03 |
| flosp2th | Vanderbei | 691 | 0 | Maximum_Iterations_Exceeded | 1.000000e+01 |
| forplan | LP | 421 | 135 | N/A | -6.642190e+02 |
| fxm2-16 | LP | 5602 | 3900 | N/A | 1.841676e+04 |
| fxm2-6 | LP | 2172 | 1520 | N/A | 1.841707e+04 |
| fxm3_6 | LP | 9492 | 6200 | N/A | 1.861604e+04 |
| gams10a | LP | 61 | 114 | N/A | 1.000000e+00 |
| gams30a | LP | 181 | 354 | N/A | 1.000000e+00 |
| ganges | LP | 1681 | 1309 | N/A | -1.095857e+05 |
| gen | LP | 2560 | 769 | N/A | -1.330523e-05 |
| gen1 | LP | 2560 | 769 | N/A | -1.330523e-05 |
| gen2 | LP | 3264 | 1121 | N/A | 3.292791e+00 |
| gen4 | LP | 4297 | 1537 | N/A | -2.213326e-05 |
| gfrd-pnc | LP | 1092 | 616 | N/A | 8.412347e+03 |
| greenbeb | LP | 5405 | 2389 | N/A | -4.302260e+06 |
| grow15 | LP | 645 | 300 | N/A | -1.068709e+08 |
| grow22 | LP | 946 | 440 | N/A | -1.608343e+08 |
| grow7 | LP | 301 | 140 | N/A | -4.778781e+07 |
| henon120 | Mittelmann | 32401 | 241 | Maximum_CpuTime_Exceeded | 1.332947e+02 |
| iiasa | LP | 2970 | 669 | N/A | 6.278482e+05 |
| israel | LP | 142 | 174 | N/A | -2.981858e+04 |
| jendrec1 | LP | 4228 | 2109 | N/A | 1.566579e+03 |
| kb2 | LP | 41 | 43 | N/A | -1.749900e+03 |
| kleemin3 | LP | 3 | 3 | N/A | -1.000000e+04 |
| kleemin4 | LP | 4 | 4 | N/A | -1.000000e+05 |
| kleemin5 | LP | 5 | 5 | N/A | -1.000000e+06 |
| kleemin6 | LP | 6 | 6 | N/A | -1.000000e+07 |
| l9 | LP | 1401 | 244 | N/A | 9.982146e-01 |
| lane_emden120 | Mittelmann | 57721 | 241 | Maximum_CpuTime_Exceeded | 9.340251e+00 |
| large000 | LP | 6833 | 4239 | N/A | 7.260546e+00 |
| large001 | LP | 6834 | 4162 | N/A | 3.318139e+03 |
| large002 | LP | 6835 | 4249 | N/A | 2.878458e-01 |
| large003 | LP | 6835 | 4200 | N/A | 2.459547e+03 |
| large004 | LP | 6836 | 4250 | N/A | 1.747032e+01 |
| large005 | LP | 6837 | 4237 | N/A | 4.118496e+01 |
| large006 | LP | 6837 | 4249 | N/A | 2.474461e+01 |
| large007 | LP | 6836 | 4236 | N/A | 5.026082e+01 |
| large008 | LP | 6837 | 4248 | N/A | 2.681109e+01 |
| large009 | LP | 6837 | 4237 | N/A | 5.028209e+01 |
| large010 | LP | 6837 | 4247 | N/A | 2.562472e+01 |
| large011 | LP | 6837 | 4236 | N/A | 4.935814e+01 |
| large012 | LP | 6838 | 4253 | N/A | 1.996981e+01 |
| large013 | LP | 6838 | 4248 | N/A | 5.710028e+01 |
| large014 | LP | 6838 | 4271 | N/A | 1.715729e+01 |
| large015 | LP | 6838 | 4265 | N/A | 6.725271e+01 |
| large016 | LP | 6838 | 4287 | N/A | 1.634838e+01 |
| large017 | LP | 6837 | 4277 | N/A | 6.256793e+01 |
| large018 | LP | 6837 | 4297 | N/A | 2.146131e+01 |
| large019 | LP | 6836 | 4300 | N/A | 5.255244e+01 |
| large020 | LP | 6837 | 4315 | N/A | 3.775156e+01 |
| large021 | LP | 6838 | 4311 | N/A | 4.221277e+01 |
| large022 | LP | 6834 | 4312 | N/A | 3.774823e+01 |
| large023 | LP | 6835 | 4302 | N/A | 3.624436e+01 |
| large024 | LP | 6831 | 4292 | N/A | 3.560283e+01 |
| large025 | LP | 6832 | 4297 | N/A | 3.554429e+01 |
| large026 | LP | 6824 | 4284 | N/A | 3.563724e+01 |
| large027 | LP | 6821 | 4275 | N/A | 3.337026e+01 |
| large028 | LP | 6833 | 4302 | N/A | 3.113620e+01 |
| large029 | LP | 6832 | 4301 | N/A | 2.874745e+01 |
| large030 | LP | 6823 | 4285 | N/A | 2.651229e+01 |
| large031 | LP | 6826 | 4294 | N/A | 2.450246e+01 |
| large032 | LP | 6827 | 4292 | N/A | 2.261425e+01 |
| large033 | LP | 6817 | 4273 | N/A | 2.098350e+01 |
| large034 | LP | 6831 | 4294 | N/A | 1.987645e+01 |
| large035 | LP | 6829 | 4293 | N/A | 1.856036e+01 |
| large036 | LP | 6822 | 4282 | N/A | 1.710623e+01 |
| lotfi | LP | 308 | 153 | N/A | -2.526471e+01 |
| manne | Vanderbei | 1094 | 730 | Acceptable | -9.741654e-01 |
| maros | LP | 1443 | 845 | N/A | -4.720630e+04 |
| maros-r7 | LP | 9408 | 3136 | N/A | 1.497185e+06 |
| model1 | LP | 798 | 362 | N/A | 0.000000e+00 |
| model2 | LP | 1212 | 379 | N/A | -3.524043e+03 |
| model4 | LP | 4549 | 1337 | N/A | 1.589575e+05 |
| model6 | LP | 5001 | 2088 | N/A | 1.175077e+05 |
| modszk1 | LP | 1620 | 686 | N/A | 8.720824e+00 |
| multi | LP | 102 | 60 | N/A | 4.440472e+02 |
| nemsafm | LP | 2252 | 334 | N/A | -6.792374e+03 |
| nemscem | LP | 1570 | 651 | N/A | 8.977233e+04 |
| nemspmm1 | LP | 8622 | 2342 | N/A | -3.274158e+05 |
| nemspmm2 | LP | 8413 | 2281 | N/A | -2.917948e+05 |
| nesm | LP | 2923 | 662 | N/A | 1.407604e+06 |
| nl | LP | 9718 | 7031 | N/A | 1.961718e+04 |
| nsic1 | LP | 463 | 451 | N/A | -9.168554e+06 |
| nsic2 | LP | 463 | 459 | N/A | -8.203512e+06 |
| nsir1 | LP | 5717 | 4407 | N/A | -2.890903e+07 |
| nsir2 | LP | 5717 | 4451 | N/A | -2.717559e+07 |
| nug05 | LP | 225 | 210 | N/A | 5.000000e+01 |
| nug06 | LP | 486 | 372 | N/A | 8.599999e+01 |
| nug07 | LP | 931 | 602 | N/A | 1.480000e+02 |
| nug08 | LP | 1632 | 912 | N/A | 2.034999e+02 |
| orna1 | LP | 882 | 882 | N/A | -1.831426e+01 |
| orna2 | LP | 882 | 882 | N/A | -2.122480e+01 |
| orna3 | LP | 882 | 882 | N/A | -2.119990e+01 |
| orna4 | LP | 882 | 882 | N/A | 5.655257e+02 |
| orna7 | LP | 882 | 882 | N/A | -2.123633e+01 |
| orswq2 | LP | 80 | 80 | N/A | 4.847428e-01 |
| p0033 | LP | 33 | 15 | N/A | 4.875380e+02 |
| p0040 | LP | 40 | 23 | N/A | 7.572178e+02 |
| p0201 | LP | 201 | 133 | N/A | 7.161458e+01 |
| p0282 | LP | 282 | 241 | N/A | 1.100977e+02 |
| p0291 | LP | 291 | 252 | N/A | 4.262814e+00 |
| p05 | LP | 9500 | 5081 | N/A | 4.633336e+05 |
| p0548 | LP | 548 | 176 | N/A | 2.865946e+00 |
| p19 | LP | 586 | 284 | N/A | 2.539646e+03 |
| p2756 | LP | 2756 | 755 | N/A | 2.444316e+01 |
| p6000 | LP | 5872 | 2093 | N/A | -2.581354e+03 |
| pcb1000 | LP | 2428 | 1565 | N/A | 1.979424e+04 |
| pcb3000 | LP | 6810 | 3960 | N/A | 3.295358e+04 |
| perold | LP | 1376 | 625 | N/A | -9.380756e+03 |
| pf2177 | LP | 900 | 9728 | N/A | 9.000000e+01 |
| pgp2 | LP | 9220 | 4034 | N/A | 4.473243e+02 |
| pilot | LP | 3652 | 1441 | N/A | -5.574897e+02 |
| pilot.we | LP | 2789 | 722 | N/A | -1.274055e+03 |
| pilot4 | LP | 1000 | 410 | N/A | -2.581140e+03 |
| pilot87 | LP | 4883 | 2030 | N/A | 3.017104e+02 |
| pilotnov | LP | 2172 | 975 | N/A | -4.497276e+03 |
| pldd000b | LP | 3267 | 3069 | N/A | 2.740677e-01 |
| pldd001b | LP | 3267 | 3069 | N/A | 3.808764e-01 |
| pldd002b | LP | 3267 | 3069 | N/A | 4.062523e-01 |
| pldd003b | LP | 3267 | 3069 | N/A | 4.032281e-01 |
| pldd004b | LP | 3267 | 3069 | N/A | 4.179595e-01 |
| pldd005b | LP | 3267 | 3069 | N/A | 4.151384e-01 |
| pldd006b | LP | 3267 | 3069 | N/A | 4.179809e-01 |
| pldd007b | LP | 3267 | 3069 | N/A | 4.041209e-01 |
| pldd008b | LP | 3267 | 3069 | N/A | 4.169518e-01 |
| pldd009b | LP | 3267 | 3069 | N/A | 4.540568e-01 |
| pldd010b | LP | 3267 | 3069 | N/A | 4.824974e-01 |
| pldd011b | LP | 3267 | 3069 | N/A | 4.965518e-01 |
| pldd012b | LP | 3267 | 3069 | N/A | 4.610722e-01 |
| pltexpa2-16 | LP | 4540 | 1726 | N/A | -4.831658e+00 |
| pltexpa2-6 | LP | 1820 | 686 | N/A | -4.739684e+00 |
| polak6 | Vanderbei | 5 | 4 | Unknown_Error | -4.400000e+01 |
| problem | LP | 46 | 12 | N/A | -1.591240e-05 |
| progas | LP | 1425 | 1650 | N/A | 7.607732e+05 |
| qcqp1000-2c | Mittelmann | 1000 | 5107 | Maximum_CpuTime_Exceeded | 1.112949e+04 |
| qcqp1500-1c | Mittelmann | 1500 | 10508 | Maximum_CpuTime_Exceeded | 1.751829e+04 |
| qiulp | LP | 840 | 1192 | N/A | -8.169834e+02 |
| qssp180 | Mittelmann | 196024 | 130141 | Maximum_CpuTime_Exceeded | -6.639447e+00 |
| r05 | LP | 9500 | 5171 | N/A | 4.648599e+05 |
| rat1 | LP | 9408 | 3136 | N/A | 1.999532e+06 |
| rat5 | LP | 9408 | 3136 | N/A | 3.083707e+06 |
| refine | LP | 33 | 29 | N/A | -3.926918e+05 |
| rosen1 | LP | 1024 | 520 | N/A | -2.761274e+04 |
| rosen10 | LP | 4096 | 2056 | N/A | -1.742155e+05 |
| rosen2 | LP | 2048 | 1032 | N/A | -5.441751e+04 |
| rosen7 | LP | 512 | 264 | N/A | -2.032970e+04 |
| rosen8 | LP | 1024 | 520 | N/A | -4.212268e+04 |
| sc105 | LP | 103 | 104 | N/A | -5.220206e+01 |
| sc205 | LP | 203 | 204 | N/A | -5.220206e+01 |
| sc205-2r-100 | LP | 2214 | 2212 | N/A | -1.007049e+01 |
| sc205-2r-16 | LP | 366 | 364 | N/A | -5.538771e+01 |
| sc205-2r-200 | LP | 4414 | 4412 | N/A | -1.007049e+01 |
| sc205-2r-27 | LP | 608 | 606 | N/A | -1.510574e+01 |
| sc205-2r-32 | LP | 718 | 716 | N/A | -5.538771e+01 |
| sc205-2r-4 | LP | 102 | 100 | N/A | -6.042296e+01 |
| sc205-2r-400 | LP | 8814 | 8812 | N/A | -1.007049e+01 |
| sc205-2r-50 | LP | 1114 | 1112 | N/A | -3.076411e+01 |
| sc205-2r-64 | LP | 1422 | 1420 | N/A | -5.538771e+01 |
| sc205-2r-8 | LP | 190 | 188 | N/A | -6.042296e+01 |
| sc50a | LP | 48 | 49 | N/A | -6.457508e+01 |
| sc50b | LP | 48 | 48 | N/A | -7.000000e+01 |
| scagr25 | LP | 500 | 471 | N/A | -2.228615e+06 |
| scagr7 | LP | 140 | 129 | N/A | -3.521737e+05 |
| scagr7-2b-16 | LP | 660 | 623 | N/A | -1.258160e+05 |
| scagr7-2b-4 | LP | 180 | 167 | N/A | -1.258220e+05 |
| scagr7-2c-16 | LP | 660 | 623 | N/A | -1.257190e+05 |
| scagr7-2c-4 | LP | 180 | 167 | N/A | -1.257190e+05 |
| scagr7-2c-64 | LP | 2580 | 2447 | N/A | -1.236737e+05 |
| scagr7-2r-108 | LP | 4340 | 4119 | N/A | -1.259877e+05 |
| scagr7-2r-16 | LP | 660 | 623 | N/A | -1.258160e+05 |
| scagr7-2r-216 | LP | 8660 | 8223 | N/A | -1.259877e+05 |
| scagr7-2r-27 | LP | 1100 | 1041 | N/A | -1.259130e+05 |
| scagr7-2r-32 | LP | 1300 | 1231 | N/A | -1.258160e+05 |
| scagr7-2r-4 | LP | 180 | 167 | N/A | -1.258160e+05 |
| scagr7-2r-54 | LP | 2180 | 2067 | N/A | -1.259568e+05 |
| scagr7-2r-64 | LP | 2580 | 2447 | N/A | -1.258160e+05 |
| scagr7-2r-8 | LP | 340 | 319 | N/A | -1.258160e+05 |
| scfxm1 | LP | 457 | 330 | N/A | 1.841676e+04 |
| scfxm1-2b-16 | LP | 3714 | 2460 | N/A | 2.877564e+03 |
| scfxm1-2b-4 | LP | 1014 | 684 | N/A | 2.875983e+03 |
| scfxm1-2c-4 | LP | 1014 | 684 | N/A | 2.875983e+03 |
| scfxm1-2r-16 | LP | 3714 | 2460 | N/A | 2.877564e+03 |
| scfxm1-2r-27 | LP | 6189 | 4088 | N/A | 2.886965e+03 |
| scfxm1-2r-32 | LP | 7314 | 4828 | N/A | 2.877564e+03 |
| scfxm1-2r-4 | LP | 1014 | 684 | N/A | 2.877564e+03 |
| scfxm1-2r-8 | LP | 1914 | 1276 | N/A | 2.877564e+03 |
| scfxm2 | LP | 914 | 660 | N/A | 3.666026e+04 |
| scfxm3 | LP | 1371 | 990 | N/A | 5.490125e+04 |
| scorpion | LP | 358 | 388 | N/A | 9.345764e+02 |
| scrs8 | LP | 1169 | 490 | N/A | 1.704490e+01 |
| scrs8-2b-16 | LP | 645 | 476 | N/A | 2.113027e+00 |
| scrs8-2b-4 | LP | 189 | 140 | N/A | 2.113027e+00 |
| scrs8-2b-64 | LP | 2469 | 1820 | N/A | 2.113800e+01 |
| scrs8-2c-4 | LP | 189 | 140 | N/A | 2.113027e+00 |
| scrs8-2c-64 | LP | 2469 | 1820 | N/A | 2.108824e+00 |
| scrs8-2r-16 | LP | 645 | 476 | N/A | 2.321727e+00 |
| scrs8-2r-27 | LP | 1063 | 784 | N/A | 1.106737e+01 |
| scrs8-2r-4 | LP | 189 | 140 | N/A | 2.321726e+00 |
| scrs8-2r-64b | LP | 2469 | 1820 | N/A | 2.537228e+01 |
| scrs8-2r-8 | LP | 341 | 252 | N/A | 2.116663e+01 |
| scsd1 | LP | 760 | 77 | N/A | 8.666651e+00 |
| scsd6 | LP | 1350 | 147 | N/A | 5.049997e+01 |
| scsd8 | LP | 2750 | 397 | N/A | 9.049999e+02 |
| scsd8-2b-16 | LP | 2310 | 330 | N/A | 2.750000e+01 |
| scsd8-2b-4 | LP | 630 | 90 | N/A | 1.525000e+01 |
| scsd8-2c-16 | LP | 2310 | 330 | N/A | 1.500000e+01 |
| scsd8-2c-4 | LP | 630 | 90 | N/A | 1.500000e+01 |
| scsd8-2r-16 | LP | 2310 | 330 | N/A | 1.599950e+01 |
| scsd8-2r-27 | LP | 3850 | 550 | N/A | 2.400000e+01 |
| scsd8-2r-32 | LP | 4550 | 650 | N/A | 1.599901e+01 |
| scsd8-2r-4 | LP | 630 | 90 | N/A | 1.550000e+01 |
| scsd8-2r-54 | LP | 7630 | 1090 | N/A | 2.385001e+01 |
| scsd8-2r-64 | LP | 9030 | 1290 | N/A | 1.584277e+01 |
| scsd8-2r-8 | LP | 1190 | 170 | N/A | 1.600000e+01 |
| scsd8-2r-8b | LP | 1190 | 170 | N/A | 1.600000e+01 |
| sctap1 | LP | 480 | 300 | N/A | 1.412250e+03 |
| sctap1-2b-16 | LP | 1584 | 990 | N/A | 2.808000e+02 |
| sctap1-2b-4 | LP | 432 | 270 | N/A | 2.392500e+02 |
| sctap1-2c-16 | LP | 1584 | 990 | N/A | 3.264000e+02 |
| sctap1-2c-4 | LP | 432 | 270 | N/A | 2.362500e+02 |
| sctap1-2c-64 | LP | 5424 | 3390 | N/A | 2.003906e+02 |
| sctap1-2r-16 | LP | 1584 | 990 | N/A | 3.590000e+02 |
| sctap1-2r-27 | LP | 2640 | 1650 | N/A | 2.475000e+02 |
| sctap1-2r-32 | LP | 3120 | 1950 | N/A | 3.540000e+02 |
| sctap1-2r-4 | LP | 432 | 270 | N/A | 2.805000e+02 |
| sctap1-2r-54 | LP | 5232 | 3270 | N/A | 2.492500e+02 |
| sctap1-2r-64 | LP | 6192 | 3870 | N/A | 3.440000e+02 |
| sctap1-2r-8 | LP | 816 | 510 | N/A | 3.605000e+02 |
| sctap1-2r-8b | LP | 816 | 510 | N/A | 2.500000e+02 |
| sctap2 | LP | 1880 | 1090 | N/A | 1.724807e+03 |
| sctap3 | LP | 2480 | 1480 | N/A | 1.424000e+03 |
| seba | LP | 1028 | 515 | N/A | 3.265087e+03 |
| seymourl | LP | 1372 | 4944 | N/A | 4.038465e+02 |
| share1b | LP | 225 | 117 | N/A | -7.658932e+04 |
| share2b | LP | 79 | 96 | N/A | -4.157322e+02 |
| shell | LP | 1775 | 536 | N/A | 2.981074e+07 |
| ship04l | LP | 2118 | 360 | N/A | 2.434598e+04 |
| ship04s | LP | 1458 | 360 | N/A | 2.441915e+04 |
| ship08l | LP | 4283 | 712 | N/A | 2.353662e+04 |
| ship08s | LP | 2387 | 712 | N/A | 2.367277e+04 |
| ship12l | LP | 5427 | 1042 | N/A | 2.611810e+04 |
| ship12s | LP | 2763 | 1042 | N/A | 2.645650e+04 |
| sierra | LP | 2036 | 1227 | N/A | 2.438788e+04 |
| slptsk | LP | 3347 | 2861 | N/A | 2.989537e+01 |
| small000 | LP | 1140 | 709 | N/A | 2.128221e+00 |
| small001 | LP | 1140 | 687 | N/A | 2.007631e+02 |
| small002 | LP | 1140 | 713 | N/A | 3.765713e+00 |
| small003 | LP | 1140 | 711 | N/A | 1.854361e+01 |
| small004 | LP | 1140 | 717 | N/A | 5.468393e+00 |
| small005 | LP | 1140 | 717 | N/A | 3.417117e+00 |
| small006 | LP | 1138 | 710 | N/A | 2.443084e+00 |
| small007 | LP | 1137 | 711 | N/A | 1.388197e+00 |
| small008 | LP | 1134 | 712 | N/A | 8.053325e-01 |
| small009 | LP | 1135 | 710 | N/A | 4.907638e-01 |
| small010 | LP | 1138 | 711 | N/A | 2.262410e-01 |
| small011 | LP | 1133 | 705 | N/A | 8.861410e-02 |
| small012 | LP | 1134 | 706 | N/A | 6.225608e-02 |
| small013 | LP | 1131 | 701 | N/A | 9.625452e-02 |
| small014 | LP | 1130 | 687 | N/A | 1.229335e-01 |
| small015 | LP | 1130 | 683 | N/A | 1.679980e-01 |
| small016 | LP | 1130 | 677 | N/A | 2.202576e-01 |
| stair | LP | 467 | 356 | N/A | -2.512670e+02 |
| standata | LP | 1075 | 359 | N/A | 1.257699e+03 |
| standmps | LP | 1075 | 467 | N/A | 1.406017e+03 |
| steenbre | Vanderbei | 540 | 126 | Acceptable | 5.005874e+02 |
| steenbrg | Vanderbei | 540 | 126 | Acceptable | 5.006421e+02 |
| stocfor1 | LP | 111 | 117 | N/A | -1.387503e+04 |
| stocfor2 | LP | 2031 | 2157 | N/A | -3.902441e+04 |
| tuff | LP | 587 | 294 | N/A | 2.921487e-01 |
| vtp.base | LP | 203 | 198 | N/A | 1.298315e+05 |
| woodw | LP | 8405 | 1098 | N/A | 1.304438e+00 |
| zed | LP | 43 | 116 | N/A | -3.576841e+02 |

## Acceptable (not Optimal) — 36 problems

These problems converged within relaxed tolerances but not strict tolerances.

| Problem | Suite | n | m | Ipopt status | POUNCE obj | Ipopt obj |
|---------|-------|---|---|-------------|------------|-----------|
| CVXQP3_L | QP | 10000 | 7500 | Maximum_CpuTime_Exceeded | 1.102011e+06 | 2.593116e+04 |
| QRECIPE | QP | 180 | 91 | Acceptable | -2.666160e+02 | -2.666160e+02 |
| QSHELL | QP | 1775 | 536 | Optimal | 2.674981e+08 | 2.674980e+08 |
| aa4 | LP | 7195 | 426 | N/A | 9.659420e+02 | N/A |
| agg | LP | 163 | 488 | N/A | -3.588038e+07 | N/A |
| agg2 | LP | 302 | 516 | N/A | -2.007526e+07 | N/A |
| agg3 | LP | 302 | 516 | N/A | 1.030387e+07 | N/A |
| bore3d | LP | 315 | 233 | N/A | 4.094410e+02 | N/A |
| ch | LP | 5062 | 3682 | N/A | 5.981401e+03 | N/A |
| co5 | LP | 7993 | 5715 | N/A | 5.075138e+03 | N/A |
| cvxqp3 | Vanderbei | 10000 | 7500 | Maximum_CpuTime_Exceeded | 2.204021e+05 | 2.046225e+05 |
| de063155 | LP | 1488 | 852 | N/A | 9.883094e+08 | N/A |
| degen3 | LP | 1818 | 1503 | N/A | -9.853701e+02 | N/A |
| himmelbj | Vanderbei | 45 | 16 | Restoration_Failed | N/A | -1.903503e+03 |
| kleemin7 | LP | 7 | 7 | N/A | -1.000000e+08 | N/A |
| kleemin8 | LP | 8 | 8 | N/A | -1.000000e+09 | N/A |
| model3 | LP | 3840 | 1609 | N/A | 1.750657e+03 | N/A |
| model7 | LP | 8007 | 3358 | N/A | 4.942951e+03 | N/A |
| nql180 | Mittelmann | 129601 | 130080 | Solver_Error | -9.277190e-01 | N/A |
| orthrds2 | Vanderbei | 203 | 100 | Optimal | 1.544296e+03 | 1.544297e+03 |
| pilot.ja | LP | 1988 | 940 | N/A | -6.113137e+03 | N/A |
| qcqp1000-1nc | Mittelmann | 1000 | 154 | Optimal | -1.841514e+06 | -1.841514e+06 |
| recipe | LP | 180 | 91 | N/A | -2.666160e+02 | N/A |
| s368 | Vanderbei | 100 | 0 | Optimal | -1.175134e-19 | 2.005351e-18 |
| scrs8-2c-16 | LP | 645 | 476 | N/A | 2.119200e+00 | N/A |
| scrs8-2c-32 | LP | 1253 | 924 | N/A | 2.114008e+00 | N/A |
| scrs8-2c-8 | LP | 341 | 252 | N/A | 2.118902e+00 | N/A |
| scrs8-2r-128 | LP | 4901 | 3612 | N/A | 2.219152e+01 | N/A |
| scrs8-2r-256 | LP | 9765 | 7196 | N/A | 2.156604e+01 | N/A |
| scrs8-2r-32 | LP | 1253 | 924 | N/A | 2.321728e+00 | N/A |
| scrs8-2r-64 | LP | 2469 | 1820 | N/A | 2.321729e+00 | N/A |
| spanhyd | Vanderbei | 97 | 33 | Optimal | 5.889003e-04 | 5.193068e-04 |
| ssebnln | Vanderbei | 194 | 96 | Optimal | 1.079313e+06 | 1.078040e+06 |
| steenbrc | Vanderbei | 540 | 126 | Unknown_Error | 9.448802e+03 | 1.258424e+04 |
| steenbrf | Vanderbei | 468 | 108 | Acceptable | 4.415272e+03 | 6.378341e+02 |
| wood1p | LP | 2594 | 244 | N/A | 1.443041e+00 | N/A |

## POUNCE-Only Suite Details

These suites currently run POUNCE only — no Ipopt-side comparison is captured in their result files. Per-problem timing and iteration counts are shown so users can inspect the whole picture.

### LP

| Problem | n | m | Status | Objective | Iters | Time |
|---------|---|---|--------|-----------|-------|------|
| 25fv47 | 1,571 | 820 | Optimal | 5.5018e+03 | 203 | 784.5ms |
| 80bau3b | 9,799 | 2,237 | Optimal | 2.9483e+04 | 216 | 3.11s |
| aa01 | 8,904 | 823 | Optimal | 2.4595e+03 | 386 | 25.14s |
| aa03 | 8,627 | 825 | Optimal | 2.6690e+03 | 664 | 42.57s |
| aa3 | 8,627 | 825 | Optimal | 2.6690e+03 | 449 | 26.92s |
| aa4 | 7,195 | 426 | Acceptable | 9.6594e+02 | 198 | 6.43s |
| aa5 | 8,308 | 801 | Optimal | 2.4414e+03 | 530 | 30.57s |
| aa6 | 7,292 | 646 | Optimal | 2.8822e+03 | 519 | 21.54s |
| adlittle | 97 | 56 | Optimal | 6.8125e+03 | 66 | 45.2ms |
| afiro | 32 | 27 | Optimal | -4.6475e+02 | 25 | 35.2ms |
| agg | 163 | 488 | Acceptable | -3.5880e+07 | 256 | 302.9ms |
| agg2 | 302 | 516 | Acceptable | -2.0075e+07 | 197 | 363.5ms |
| agg3 | 302 | 516 | Acceptable | 1.0304e+07 | 223 | 545.2ms |
| air02 | 6,774 | 50 | Optimal | 7.9515e+01 | 79 | 1.09s |
| air04 | 8,904 | 823 | Optimal | 2.4595e+03 | 386 | 24.80s |
| air05 | 7,195 | 426 | Optimal | 9.6594e+02 | 220 | 7.05s |
| air06 | 8,627 | 825 | Optimal | 2.6690e+03 | 664 | 42.38s |
| aircraft | 7,517 | 3,754 | Optimal | 1.5670e+03 | 26 | 183.6ms |
| bandm | 472 | 305 | Optimal | -1.5863e+02 | 78 | 81.2ms |
| beaconfd | 262 | 173 | Optimal | 3.0819e+04 | 35 | 49.7ms |
| blend | 83 | 74 | Optimal | -3.0812e+01 | 20 | 33.3ms |
| bnl1 | 1,175 | 632 | Optimal | 1.9776e+03 | 196 | 325.9ms |
| bnl2 | 3,489 | 2,280 | Optimal | 1.8112e+03 | 278 | 2.36s |
| boeing1 | 384 | 348 | Optimal | -3.3521e+02 | 88 | 108.3ms |
| boeing2 | 143 | 140 | Optimal | -3.1502e+02 | 55 | 55.3ms |
| bore3d | 315 | 233 | Acceptable | 4.0944e+02 | 296 | 158.3ms |
| brandy | 249 | 182 | Optimal | 1.5185e+03 | 106 | 95.1ms |
| capri | 353 | 271 | Optimal | 2.6900e+03 | 192 | 156.8ms |
| cari | 1,200 | 400 | Optimal | 5.8189e+02 | 29 | 1.53s |
| cep1 | 3,248 | 1,521 | Optimal | 3.5516e+05 | 341 | 940.1ms |
| ch | 5,062 | 3,682 | Acceptable | 5.9814e+03 | 381 | 17.43s |
| co5 | 7,993 | 5,715 | Acceptable | 5.0751e+03 | 268 | 50.00s |
| complex | 1,408 | 1,023 | Optimal | -9.9667e+01 | 64 | 1.97s |
| cq5 | 7,530 | 5,025 | Optimal | 2.8423e+03 | 201 | 18.25s |
| cr42 | 1,513 | 905 | Optimal | 2.8018e+01 | 46 | 103.4ms |
| crew1 | 6,469 | 135 | Optimal | 2.0556e+01 | 30 | 255.3ms |
| cycle | 2,857 | 1,886 | Optimal | -5.2264e+00 | 173 | 2.23s |
| czprob | 3,523 | 927 | Optimal | 7.1114e+05 | 418 | 898.9ms |
| d2q06c | 5,167 | 2,171 | Optimal | 1.2278e+05 | 396 | 4.95s |
| d6cube | 6,184 | 404 | Optimal | 3.1549e+02 | 55 | 544.2ms |
| de063155 | 1,488 | 852 | Acceptable | 9.8831e+08 | 857 | 99.52s |
| de080285 | 1,488 | 936 | Optimal | 1.3924e+00 | 44 | 105.3ms |
| degen2 | 534 | 444 | Optimal | -1.4352e+03 | 98 | 482.2ms |
| degen3 | 1,818 | 1,503 | Acceptable | -9.8537e+02 | 100 | 3.03s |
| delf000 | 5,464 | 3,128 | Optimal | 3.0739e-01 | 44 | 267.5ms |
| delf001 | 5,462 | 3,098 | Optimal | 2.3586e+02 | 46 | 280.4ms |
| delf002 | 5,460 | 3,135 | Optimal | 2.8303e-01 | 40 | 261.3ms |
| delf003 | 5,460 | 3,065 | Optimal | 9.1155e+02 | 69 | 386.0ms |
| delf004 | 5,464 | 3,142 | Optimal | 1.5869e+01 | 53 | 316.4ms |
| delf005 | 5,464 | 3,103 | Optimal | 2.2873e+02 | 73 | 424.3ms |
| delf006 | 5,469 | 3,147 | Optimal | 2.2843e+01 | 58 | 351.7ms |
| delf007 | 5,471 | 3,137 | Optimal | 3.7995e+01 | 72 | 440.1ms |
| delf008 | 5,472 | 3,148 | Optimal | 2.4123e+01 | 66 | 381.3ms |
| delf009 | 5,472 | 3,135 | Optimal | 4.8412e+01 | 74 | 423.9ms |
| delf010 | 5,472 | 3,147 | Optimal | 2.2910e+01 | 66 | 412.1ms |
| delf011 | 5,471 | 3,134 | Optimal | 4.7349e+01 | 69 | 389.7ms |
| delf012 | 5,471 | 3,151 | Optimal | 1.7868e+01 | 65 | 389.3ms |
| delf013 | 5,472 | 3,116 | Optimal | 2.6167e+02 | 87 | 486.8ms |
| delf014 | 5,472 | 3,170 | Optimal | 1.5350e+01 | 61 | 385.5ms |
| delf015 | 5,471 | 3,161 | Optimal | 7.3703e+01 | 76 | 490.8ms |
| delf017 | 5,471 | 3,176 | Optimal | 4.6152e+01 | 69 | 437.1ms |
| delf018 | 5,471 | 3,196 | Optimal | 1.4213e+01 | 57 | 370.8ms |
| delf019 | 5,471 | 3,185 | Optimal | 2.3421e+02 | 70 | 444.3ms |
| delf020 | 5,472 | 3,213 | Optimal | 3.5168e+01 | 61 | 426.5ms |
| delf021 | 5,471 | 3,208 | Optimal | 3.9470e+01 | 62 | 406.2ms |
| delf022 | 5,472 | 3,214 | Optimal | 3.6499e+01 | 49 | 334.6ms |
| delf023 | 5,472 | 3,214 | Optimal | 3.5411e+01 | 51 | 342.8ms |
| delf024 | 5,466 | 3,207 | Optimal | 3.5141e+01 | 48 | 323.7ms |
| delf025 | 5,464 | 3,197 | Optimal | 3.5029e+01 | 50 | 331.3ms |
| delf026 | 5,462 | 3,190 | Optimal | 3.5161e+01 | 50 | 337.7ms |
| delf027 | 5,457 | 3,187 | Optimal | 3.1945e+01 | 49 | 322.4ms |
| delf028 | 5,452 | 3,177 | Optimal | 2.9724e+01 | 46 | 314.2ms |
| delf029 | 5,454 | 3,179 | Optimal | 2.7522e+01 | 46 | 303.0ms |
| delf030 | 5,469 | 3,199 | Optimal | 2.5384e+01 | 41 | 284.8ms |
| delf031 | 5,455 | 3,176 | Optimal | 2.3430e+01 | 42 | 294.9ms |
| delf032 | 5,467 | 3,196 | Optimal | 2.1604e+01 | 40 | 289.6ms |
| delf033 | 5,456 | 3,173 | Optimal | 2.0027e+01 | 53 | 346.5ms |
| delf034 | 5,455 | 3,175 | Optimal | 1.8944e+01 | 54 | 357.2ms |
| delf035 | 5,468 | 3,193 | Optimal | 1.7695e+01 | 50 | 328.8ms |
| delf036 | 5,459 | 3,170 | Optimal | 1.6446e+01 | 50 | 335.4ms |
| deter0 | 5,468 | 1,923 | Optimal | -2.0459e+00 | 26 | 148.3ms |
| deter4 | 9,133 | 3,235 | Optimal | -1.4280e+00 | 29 | 234.9ms |
| df2177 | 9,728 | 630 | Optimal | 9.1000e+01 | 4 | 431.1ms |
| disp3 | 1,856 | 2,182 | Optimal | 8.0832e+04 | 62 | 214.8ms |
| dsbmip | 1,877 | 1,182 | Optimal | -3.0520e+02 | 85 | 405.9ms |
| e226 | 282 | 223 | Optimal | -1.1639e+01 | 66 | 75.6ms |
| etamacro | 688 | 400 | Optimal | -9.6938e+01 | 115 | 146.7ms |
| farm | 12 | 7 | Optimal | 4.3860e+03 | 15 | 33.6ms |
| fffff800 | 854 | 524 | Optimal | 5.5568e+05 | 544 | 962.5ms |
| finnis | 614 | 497 | Optimal | 4.2884e+03 | 162 | 199.3ms |
| fit1d | 1,026 | 24 | Optimal | -6.3517e+02 | 30 | 85.3ms |
| fit1p | 1,677 | 627 | Optimal | 9.1464e+03 | 121 | 237.6ms |
| forplan | 421 | 135 | Optimal | -6.6422e+02 | 91 | 100.5ms |
| fxm2-16 | 5,602 | 3,900 | Optimal | 1.8417e+04 | 271 | 2.39s |
| fxm2-6 | 2,172 | 1,520 | Optimal | 1.8417e+04 | 170 | 634.4ms |
| fxm3_6 | 9,492 | 6,200 | Optimal | 1.8616e+04 | 411 | 5.34s |
| gams10a | 61 | 114 | Optimal | 1.0000e+00 | 8 | 30.1ms |
| gams30a | 181 | 354 | Optimal | 1.0000e+00 | 9 | 33.9ms |
| ganges | 1,681 | 1,309 | Optimal | -1.0959e+05 | 415 | 723.6ms |
| gen | 2,560 | 769 | Optimal | -1.3305e-05 | 21 | 864.5ms |
| gen1 | 2,560 | 769 | Optimal | -1.3305e-05 | 21 | 881.8ms |
| gen2 | 3,264 | 1,121 | Optimal | 3.2928e+00 | 13 | 842.7ms |
| gen4 | 4,297 | 1,537 | Optimal | -2.2133e-05 | 23 | 4.94s |
| gfrd-pnc | 1,092 | 616 | Optimal | 8.4123e+03 | 144 | 137.6ms |
| greenbea | 5,405 | 2,389 | Maximum_Iterations_Exceeded | -5.6030e+07 | 2999 | 26.89s |
| greenbeb | 5,405 | 2,389 | Optimal | -4.3023e+06 | 2071 | 19.41s |
| grow15 | 645 | 300 | Optimal | -1.0687e+08 | 153 | 213.9ms |
| grow22 | 946 | 440 | Optimal | -1.6083e+08 | 159 | 308.9ms |
| grow7 | 301 | 140 | Optimal | -4.7788e+07 | 111 | 92.3ms |
| iiasa | 2,970 | 669 | Optimal | 6.2785e+05 | 519 | 979.0ms |
| iprob | 3,001 | 3,001 | Infeasible_Problem_Detected | 2.7144e+03 | 1019 | 42.57s |
| israel | 142 | 174 | Optimal | -2.9819e+04 | 135 | 90.9ms |
| jendrec1 | 4,228 | 2,109 | Optimal | 1.5666e+03 | 98 | 1.24s |
| kb2 | 41 | 43 | Optimal | -1.7499e+03 | 46 | 38.9ms |
| kleemin3 | 3 | 3 | Optimal | -1.0000e+04 | 20 | 36.0ms |
| kleemin4 | 4 | 4 | Optimal | -1.0000e+05 | 42 | 33.3ms |
| kleemin5 | 5 | 5 | Optimal | -1.0000e+06 | 85 | 37.1ms |
| kleemin6 | 6 | 6 | Optimal | -1.0000e+07 | 167 | 44.5ms |
| kleemin7 | 7 | 7 | Acceptable | -1.0000e+08 | 328 | 55.5ms |
| kleemin8 | 8 | 8 | Acceptable | -1.0000e+09 | 573 | 73.9ms |
| l9 | 1,401 | 244 | Optimal | 9.9821e-01 | 17 | 64.1ms |
| large000 | 6,833 | 4,239 | Optimal | 7.2605e+00 | 44 | 445.2ms |
| large001 | 6,834 | 4,162 | Optimal | 3.3181e+03 | 151 | 1.41s |
| large002 | 6,835 | 4,249 | Optimal | 2.8785e-01 | 49 | 561.1ms |
| large003 | 6,835 | 4,200 | Optimal | 2.4595e+03 | 102 | 1.01s |
| large004 | 6,836 | 4,250 | Optimal | 1.7470e+01 | 56 | 608.8ms |
| large005 | 6,837 | 4,237 | Optimal | 4.1185e+01 | 77 | 777.4ms |
| large006 | 6,837 | 4,249 | Optimal | 2.4745e+01 | 77 | 863.6ms |
| large007 | 6,836 | 4,236 | Optimal | 5.0261e+01 | 87 | 911.3ms |
| large008 | 6,837 | 4,248 | Optimal | 2.6811e+01 | 78 | 821.4ms |
| large009 | 6,837 | 4,237 | Optimal | 5.0282e+01 | 84 | 775.9ms |
| large010 | 6,837 | 4,247 | Optimal | 2.5625e+01 | 77 | 839.0ms |
| large011 | 6,837 | 4,236 | Optimal | 4.9358e+01 | 82 | 861.8ms |
| large012 | 6,838 | 4,253 | Optimal | 1.9970e+01 | 74 | 825.4ms |
| large013 | 6,838 | 4,248 | Optimal | 5.7100e+01 | 82 | 863.2ms |
| large014 | 6,838 | 4,271 | Optimal | 1.7157e+01 | 66 | 746.4ms |
| large015 | 6,838 | 4,265 | Optimal | 6.7253e+01 | 76 | 820.4ms |
| large016 | 6,838 | 4,287 | Optimal | 1.6348e+01 | 65 | 730.2ms |
| large017 | 6,837 | 4,277 | Optimal | 6.2568e+01 | 75 | 806.7ms |
| large018 | 6,837 | 4,297 | Optimal | 2.1461e+01 | 64 | 694.9ms |
| large019 | 6,836 | 4,300 | Optimal | 5.2552e+01 | 73 | 850.9ms |
| large020 | 6,837 | 4,315 | Optimal | 3.7752e+01 | 74 | 837.6ms |
| large021 | 6,838 | 4,311 | Optimal | 4.2213e+01 | 77 | 867.0ms |
| large022 | 6,834 | 4,312 | Optimal | 3.7748e+01 | 63 | 688.8ms |
| large023 | 6,835 | 4,302 | Optimal | 3.6244e+01 | 64 | 714.3ms |
| large024 | 6,831 | 4,292 | Optimal | 3.5603e+01 | 56 | 667.6ms |
| large025 | 6,832 | 4,297 | Optimal | 3.5544e+01 | 59 | 690.2ms |
| large026 | 6,824 | 4,284 | Optimal | 3.5637e+01 | 60 | 686.7ms |
| large027 | 6,821 | 4,275 | Optimal | 3.3370e+01 | 55 | 620.0ms |
| large028 | 6,833 | 4,302 | Optimal | 3.1136e+01 | 55 | 618.6ms |
| large029 | 6,832 | 4,301 | Optimal | 2.8747e+01 | 55 | 620.2ms |
| large030 | 6,823 | 4,285 | Optimal | 2.6512e+01 | 50 | 571.4ms |
| large031 | 6,826 | 4,294 | Optimal | 2.4502e+01 | 49 | 563.1ms |
| large032 | 6,827 | 4,292 | Optimal | 2.2614e+01 | 48 | 561.7ms |
| large033 | 6,817 | 4,273 | Optimal | 2.0983e+01 | 56 | 609.4ms |
| large034 | 6,831 | 4,294 | Optimal | 1.9876e+01 | 56 | 614.3ms |
| large035 | 6,829 | 4,293 | Optimal | 1.8560e+01 | 56 | 600.5ms |
| large036 | 6,822 | 4,282 | Optimal | 1.7106e+01 | 54 | 689.9ms |
| lotfi | 308 | 153 | Optimal | -2.5265e+01 | 55 | 59.8ms |
| maros | 1,443 | 845 | Optimal | -4.7206e+04 | 577 | 1.94s |
| maros-r7 | 9,408 | 3,136 | Optimal | 1.4972e+06 | 663 | 43.58s |
| model1 | 798 | 362 | Optimal | 0.0000e+00 | 35 | 70.1ms |
| model2 | 1,212 | 379 | Optimal | -3.5240e+03 | 301 | 1.11s |
| model3 | 3,840 | 1,609 | Acceptable | 1.7507e+03 | 212 | 1.35s |
| model4 | 4,549 | 1,337 | Optimal | 1.5896e+05 | 397 | 4.47s |
| model6 | 5,001 | 2,088 | Optimal | 1.1751e+05 | 227 | 2.81s |
| model7 | 8,007 | 3,358 | Acceptable | 4.9430e+03 | 268 | 4.44s |
| model8 | 6,464 | 2,896 | Solver_Error | 0.0000e+00 | 0 | 35.3ms |
| modszk1 | 1,620 | 686 | Optimal | 8.7208e+00 | 82 | 148.9ms |
| multi | 102 | 60 | Optimal | 4.4405e+02 | 23 | 39.2ms |
| nemsafm | 2,252 | 334 | Optimal | -6.7924e+03 | 141 | 143.7ms |
| nemscem | 1,570 | 651 | Optimal | 8.9772e+04 | 265 | 347.6ms |
| nemspmm1 | 8,622 | 2,342 | Optimal | -3.2742e+05 | 443 | 6.06s |
| nemspmm2 | 8,413 | 2,281 | Optimal | -2.9179e+05 | 760 | 14.57s |
| nesm | 2,923 | 662 | Optimal | 1.4076e+06 | 564 | 1.79s |
| nl | 9,718 | 7,031 | Optimal | 1.9617e+04 | 266 | 40.28s |
| nsic1 | 463 | 451 | Optimal | -9.1686e+06 | 20 | 49.4ms |
| nsic2 | 463 | 459 | Optimal | -8.2035e+06 | 153 | 160.8ms |
| nsir1 | 5,717 | 4,407 | Optimal | -2.8909e+07 | 49 | 3.23s |
| nsir2 | 5,717 | 4,451 | Optimal | -2.7176e+07 | 259 | 37.74s |
| nug05 | 225 | 210 | Optimal | 5.0000e+01 | 13 | 47.6ms |
| nug06 | 486 | 372 | Optimal | 8.6000e+01 | 13 | 120.5ms |
| nug07 | 931 | 602 | Optimal | 1.4800e+02 | 18 | 437.4ms |
| nug08 | 1,632 | 912 | Optimal | 2.0350e+02 | 13 | 1.00s |
| nug12 | 8,856 | 3,192 | Maximum_CpuTime_Exceeded | N/A | 28 | 120.08s |
| orna1 | 882 | 882 | Optimal | -1.8314e+01 | 33 | 93.4ms |
| orna2 | 882 | 882 | Optimal | -2.1225e+01 | 27 | 85.6ms |
| orna3 | 882 | 882 | Optimal | -2.1200e+01 | 27 | 84.4ms |
| orna4 | 882 | 882 | Optimal | 5.6553e+02 | 37 | 105.5ms |
| orna7 | 882 | 882 | Optimal | -2.1236e+01 | 28 | 89.7ms |
| orswq2 | 80 | 80 | Optimal | 4.8474e-01 | 22 | 65.6ms |
| p0033 | 33 | 15 | Optimal | 4.8754e+02 | 29 | 45.0ms |
| p0040 | 40 | 23 | Optimal | 7.5722e+02 | 31 | 43.5ms |
| p0201 | 201 | 133 | Optimal | 7.1615e+01 | 52 | 76.4ms |
| p0282 | 282 | 241 | Optimal | 1.1010e+02 | 41 | 69.7ms |
| p0291 | 291 | 252 | Optimal | 4.2628e+00 | 29 | 61.8ms |
| p05 | 9,500 | 5,081 | Optimal | 4.6333e+05 | 45 | 2.49s |
| p0548 | 548 | 176 | Optimal | 2.8659e+00 | 53 | 78.2ms |
| p19 | 586 | 284 | Optimal | 2.5396e+03 | 33 | 76.8ms |
| p2756 | 2,756 | 755 | Optimal | 2.4443e+01 | 24 | 120.5ms |
| p6000 | 5,872 | 2,093 | Optimal | -2.5814e+03 | 53 | 304.4ms |
| pcb1000 | 2,428 | 1,565 | Optimal | 1.9794e+04 | 78 | 467.0ms |
| pcb3000 | 6,810 | 3,960 | Optimal | 3.2954e+04 | 80 | 1.68s |
| perold | 1,376 | 625 | Optimal | -9.3808e+03 | 592 | 1.80s |
| pf2177 | 900 | 9,728 | Optimal | 9.0000e+01 | 5 | 1.19s |
| pgp2 | 9,220 | 4,034 | Optimal | 4.4732e+02 | 58 | 822.1ms |
| pilot | 3,652 | 1,441 | Optimal | -5.5749e+02 | 286 | 5.48s |
| pilot.ja | 1,988 | 940 | Acceptable | -6.1131e+03 | 526 | 2.95s |
| pilot.we | 2,789 | 722 | Optimal | -1.2741e+03 | 214 | 901.3ms |
| pilot4 | 1,000 | 410 | Optimal | -2.5811e+03 | 203 | 580.5ms |
| pilot87 | 4,883 | 2,030 | Optimal | 3.0171e+02 | 324 | 26.54s |
| pilotnov | 2,172 | 975 | Optimal | -4.4973e+03 | 331 | 4.84s |
| pldd000b | 3,267 | 3,069 | Optimal | 2.7407e-01 | 49 | 237.6ms |
| pldd001b | 3,267 | 3,069 | Optimal | 3.8088e-01 | 50 | 238.6ms |
| pldd002b | 3,267 | 3,069 | Optimal | 4.0625e-01 | 49 | 237.8ms |
| pldd003b | 3,267 | 3,069 | Optimal | 4.0323e-01 | 50 | 250.2ms |
| pldd004b | 3,267 | 3,069 | Optimal | 4.1796e-01 | 50 | 241.0ms |
| pldd005b | 3,267 | 3,069 | Optimal | 4.1514e-01 | 49 | 238.7ms |
| pldd006b | 3,267 | 3,069 | Optimal | 4.1798e-01 | 46 | 228.1ms |
| pldd007b | 3,267 | 3,069 | Optimal | 4.0412e-01 | 46 | 224.8ms |
| pldd008b | 3,267 | 3,069 | Optimal | 4.1695e-01 | 52 | 254.7ms |
| pldd009b | 3,267 | 3,069 | Optimal | 4.5406e-01 | 48 | 237.9ms |
| pldd010b | 3,267 | 3,069 | Optimal | 4.8250e-01 | 54 | 269.2ms |
| pldd011b | 3,267 | 3,069 | Optimal | 4.9655e-01 | 49 | 240.3ms |
| pldd012b | 3,267 | 3,069 | Optimal | 4.6107e-01 | 50 | 244.2ms |
| pltexpa2-16 | 4,540 | 1,726 | Optimal | -4.8317e+00 | 59 | 298.1ms |
| pltexpa2-6 | 1,820 | 686 | Optimal | -4.7397e+00 | 53 | 140.6ms |
| problem | 46 | 12 | Optimal | -1.5912e-05 | 24 | 40.1ms |
| progas | 1,425 | 1,650 | Optimal | 7.6077e+05 | 259 | 758.5ms |
| qiulp | 840 | 1,192 | Optimal | -8.1698e+02 | 15 | 79.4ms |
| r05 | 9,500 | 5,171 | Optimal | 4.6486e+05 | 45 | 22.21s |
| rat1 | 9,408 | 3,136 | Optimal | 1.9995e+06 | 1337 | 83.86s |
| rat5 | 9,408 | 3,136 | Optimal | 3.0837e+06 | 1450 | 106.49s |
| recipe | 180 | 91 | Acceptable | -2.6662e+02 | 45 | 64.3ms |
| refine | 33 | 29 | Optimal | -3.9269e+05 | 70 | 51.4ms |
| rosen1 | 1,024 | 520 | Optimal | -2.7613e+04 | 90 | 276.1ms |
| rosen10 | 4,096 | 2,056 | Optimal | -1.7422e+05 | 137 | 1.13s |
| rosen2 | 2,048 | 1,032 | Optimal | -5.4418e+04 | 109 | 592.6ms |
| rosen7 | 512 | 264 | Optimal | -2.0330e+04 | 86 | 137.4ms |
| rosen8 | 1,024 | 520 | Optimal | -4.2123e+04 | 99 | 239.3ms |
| sc105 | 103 | 104 | Optimal | -5.2202e+01 | 19 | 47.7ms |
| sc205 | 203 | 204 | Optimal | -5.2202e+01 | 26 | 55.9ms |
| sc205-2r-100 | 2,214 | 2,212 | Optimal | -1.0070e+01 | 30 | 140.2ms |
| sc205-2r-16 | 366 | 364 | Optimal | -5.5388e+01 | 20 | 58.2ms |
| sc205-2r-200 | 4,414 | 4,412 | Optimal | -1.0070e+01 | 38 | 279.7ms |
| sc205-2r-27 | 608 | 606 | Optimal | -1.5106e+01 | 28 | 83.1ms |
| sc205-2r-32 | 718 | 716 | Optimal | -5.5388e+01 | 21 | 75.7ms |
| sc205-2r-4 | 102 | 100 | Optimal | -6.0423e+01 | 18 | 52.0ms |
| sc205-2r-400 | 8,814 | 8,812 | Optimal | -1.0070e+01 | 35 | 504.6ms |
| sc205-2r-50 | 1,114 | 1,112 | Optimal | -3.0764e+01 | 39 | 111.0ms |
| sc205-2r-64 | 1,422 | 1,420 | Optimal | -5.5388e+01 | 23 | 117.6ms |
| sc205-2r-8 | 190 | 188 | Optimal | -6.0423e+01 | 18 | 50.4ms |
| sc50a | 48 | 49 | Optimal | -6.4575e+01 | 17 | 44.7ms |
| sc50b | 48 | 48 | Optimal | -7.0000e+01 | 15 | 38.1ms |
| scagr25 | 500 | 471 | Optimal | -2.2286e+06 | 400 | 313.3ms |
| scagr7 | 140 | 129 | Optimal | -3.5217e+05 | 138 | 77.6ms |
| scagr7-2b-16 | 660 | 623 | Optimal | -1.2582e+05 | 81 | 115.8ms |
| scagr7-2b-4 | 180 | 167 | Optimal | -1.2582e+05 | 71 | 62.6ms |
| scagr7-2c-16 | 660 | 623 | Optimal | -1.2572e+05 | 76 | 104.9ms |
| scagr7-2c-4 | 180 | 167 | Optimal | -1.2572e+05 | 67 | 65.6ms |
| scagr7-2c-64 | 2,580 | 2,447 | Optimal | -1.2367e+05 | 79 | 296.4ms |
| scagr7-2r-108 | 4,340 | 4,119 | Optimal | -1.2599e+05 | 136 | 745.4ms |
| scagr7-2r-16 | 660 | 623 | Optimal | -1.2582e+05 | 79 | 122.5ms |
| scagr7-2r-216 | 8,660 | 8,223 | Optimal | -1.2599e+05 | 119 | 1.50s |
| scagr7-2r-27 | 1,100 | 1,041 | Optimal | -1.2591e+05 | 125 | 243.0ms |
| scagr7-2r-32 | 1,300 | 1,231 | Optimal | -1.2582e+05 | 83 | 188.2ms |
| scagr7-2r-4 | 180 | 167 | Optimal | -1.2582e+05 | 83 | 67.5ms |
| scagr7-2r-54 | 2,180 | 2,067 | Optimal | -1.2596e+05 | 142 | 428.4ms |
| scagr7-2r-64 | 2,580 | 2,447 | Optimal | -1.2582e+05 | 81 | 296.9ms |
| scagr7-2r-8 | 340 | 319 | Optimal | -1.2582e+05 | 80 | 84.3ms |
| scfxm1 | 457 | 330 | Optimal | 1.8417e+04 | 135 | 166.7ms |
| scfxm1-2b-16 | 3,714 | 2,460 | Optimal | 2.8776e+03 | 94 | 549.8ms |
| scfxm1-2b-4 | 1,014 | 684 | Optimal | 2.8760e+03 | 104 | 204.3ms |
| scfxm1-2c-4 | 1,014 | 684 | Optimal | 2.8760e+03 | 105 | 248.3ms |
| scfxm1-2r-16 | 3,714 | 2,460 | Optimal | 2.8776e+03 | 101 | 659.2ms |
| scfxm1-2r-27 | 6,189 | 4,088 | Optimal | 2.8870e+03 | 127 | 2.16s |
| scfxm1-2r-32 | 7,314 | 4,828 | Optimal | 2.8776e+03 | 108 | 1.83s |
| scfxm1-2r-4 | 1,014 | 684 | Optimal | 2.8776e+03 | 113 | 236.9ms |
| scfxm1-2r-8 | 1,914 | 1,276 | Optimal | 2.8776e+03 | 109 | 393.2ms |
| scfxm2 | 914 | 660 | Optimal | 3.6660e+04 | 209 | 385.4ms |
| scfxm3 | 1,371 | 990 | Optimal | 5.4901e+04 | 281 | 740.4ms |
| scorpion | 358 | 388 | Optimal | 9.3458e+02 | 59 | 84.5ms |
| scrs8 | 1,169 | 490 | Optimal | 1.7045e+01 | 88 | 169.0ms |
| scrs8-2b-16 | 645 | 476 | Optimal | 2.1130e+00 | 37 | 89.1ms |
| scrs8-2b-4 | 189 | 140 | Optimal | 2.1130e+00 | 39 | 73.2ms |
| scrs8-2b-64 | 2,469 | 1,820 | Optimal | 2.1138e+01 | 61 | 359.2ms |
| scrs8-2c-16 | 645 | 476 | Acceptable | 2.1192e+00 | 52 | 122.1ms |
| scrs8-2c-32 | 1,253 | 924 | Acceptable | 2.1140e+00 | 59 | 210.1ms |
| scrs8-2c-4 | 189 | 140 | Optimal | 2.1130e+00 | 39 | 63.1ms |
| scrs8-2c-64 | 2,469 | 1,820 | Optimal | 2.1088e+00 | 69 | 457.6ms |
| scrs8-2c-8 | 341 | 252 | Acceptable | 2.1189e+00 | 57 | 93.5ms |
| scrs8-2r-128 | 4,901 | 3,612 | Acceptable | 2.2192e+01 | 69 | 2.46s |
| scrs8-2r-16 | 645 | 476 | Optimal | 2.3217e+00 | 40 | 88.9ms |
| scrs8-2r-256 | 9,765 | 7,196 | Acceptable | 2.1566e+01 | 69 | 19.61s |
| scrs8-2r-27 | 1,063 | 784 | Optimal | 1.1067e+01 | 47 | 126.8ms |
| scrs8-2r-32 | 1,253 | 924 | Acceptable | 2.3217e+00 | 56 | 197.0ms |
| scrs8-2r-4 | 189 | 140 | Optimal | 2.3217e+00 | 40 | 60.6ms |
| scrs8-2r-64 | 2,469 | 1,820 | Acceptable | 2.3217e+00 | 58 | 379.7ms |
| scrs8-2r-64b | 2,469 | 1,820 | Optimal | 2.5372e+01 | 51 | 251.7ms |
| scrs8-2r-8 | 341 | 252 | Optimal | 2.1167e+01 | 51 | 73.7ms |
| scsd1 | 760 | 77 | Optimal | 8.6667e+00 | 13 | 55.4ms |
| scsd6 | 1,350 | 147 | Optimal | 5.0500e+01 | 17 | 68.0ms |
| scsd8 | 2,750 | 397 | Optimal | 9.0500e+02 | 17 | 90.0ms |
| scsd8-2b-16 | 2,310 | 330 | Optimal | 2.7500e+01 | 13 | 77.6ms |
| scsd8-2b-4 | 630 | 90 | Optimal | 1.5250e+01 | 9 | 51.8ms |
| scsd8-2c-16 | 2,310 | 330 | Optimal | 1.5000e+01 | 14 | 77.5ms |
| scsd8-2c-4 | 630 | 90 | Optimal | 1.5000e+01 | 9 | 52.2ms |
| scsd8-2r-16 | 2,310 | 330 | Optimal | 1.6000e+01 | 14 | 74.9ms |
| scsd8-2r-27 | 3,850 | 550 | Optimal | 2.4000e+01 | 24 | 133.5ms |
| scsd8-2r-32 | 4,550 | 650 | Optimal | 1.5999e+01 | 15 | 104.2ms |
| scsd8-2r-4 | 630 | 90 | Optimal | 1.5500e+01 | 9 | 45.8ms |
| scsd8-2r-54 | 7,630 | 1,090 | Optimal | 2.3850e+01 | 25 | 215.1ms |
| scsd8-2r-64 | 9,030 | 1,290 | Optimal | 1.5843e+01 | 21 | 248.9ms |
| scsd8-2r-8 | 1,190 | 170 | Optimal | 1.6000e+01 | 12 | 52.9ms |
| scsd8-2r-8b | 1,190 | 170 | Optimal | 1.6000e+01 | 12 | 59.4ms |
| sctap1 | 480 | 300 | Optimal | 1.4122e+03 | 42 | 76.9ms |
| sctap1-2b-16 | 1,584 | 990 | Optimal | 2.8080e+02 | 42 | 138.6ms |
| sctap1-2b-4 | 432 | 270 | Optimal | 2.3925e+02 | 32 | 62.7ms |
| sctap1-2c-16 | 1,584 | 990 | Optimal | 3.2640e+02 | 44 | 139.2ms |
| sctap1-2c-4 | 432 | 270 | Optimal | 2.3625e+02 | 32 | 62.9ms |
| sctap1-2c-64 | 5,424 | 3,390 | Optimal | 2.0039e+02 | 52 | 577.0ms |
| sctap1-2r-16 | 1,584 | 990 | Optimal | 3.5900e+02 | 42 | 125.5ms |
| sctap1-2r-27 | 2,640 | 1,650 | Optimal | 2.4750e+02 | 44 | 206.4ms |
| sctap1-2r-32 | 3,120 | 1,950 | Optimal | 3.5400e+02 | 46 | 244.9ms |
| sctap1-2r-4 | 432 | 270 | Optimal | 2.8050e+02 | 29 | 62.7ms |
| sctap1-2r-54 | 5,232 | 3,270 | Optimal | 2.4925e+02 | 52 | 667.0ms |
| sctap1-2r-64 | 6,192 | 3,870 | Optimal | 3.4400e+02 | 58 | 834.1ms |
| sctap1-2r-8 | 816 | 510 | Optimal | 3.6050e+02 | 39 | 88.4ms |
| sctap1-2r-8b | 816 | 510 | Optimal | 2.5000e+02 | 33 | 87.9ms |
| sctap2 | 1,880 | 1,090 | Optimal | 1.7248e+03 | 31 | 145.7ms |
| sctap3 | 2,480 | 1,480 | Optimal | 1.4240e+03 | 30 | 170.2ms |
| seba | 1,028 | 515 | Optimal | 3.2651e+03 | 216 | 357.2ms |
| seymourl | 1,372 | 4,944 | Optimal | 4.0385e+02 | 73 | 1.18s |
| share1b | 225 | 117 | Optimal | -7.6589e+04 | 336 | 173.7ms |
| share2b | 79 | 96 | Optimal | -4.1573e+02 | 29 | 50.4ms |
| shell | 1,775 | 536 | Optimal | 2.9811e+07 | 544 | 984.5ms |
| ship04l | 2,118 | 360 | Optimal | 2.4346e+04 | 54 | 152.9ms |
| ship04s | 1,458 | 360 | Optimal | 2.4419e+04 | 47 | 107.2ms |
| ship08l | 4,283 | 712 | Optimal | 2.3537e+04 | 69 | 288.7ms |
| ship08s | 2,387 | 712 | Optimal | 2.3673e+04 | 69 | 198.7ms |
| ship12l | 5,427 | 1,042 | Optimal | 2.6118e+04 | 69 | 377.6ms |
| ship12s | 2,763 | 1,042 | Optimal | 2.6456e+04 | 69 | 242.4ms |
| sierra | 2,036 | 1,227 | Optimal | 2.4388e+04 | 64 | 259.1ms |
| slptsk | 3,347 | 2,861 | Optimal | 2.9895e+01 | 173 | 27.72s |
| small000 | 1,140 | 709 | Optimal | 2.1282e+00 | 38 | 114.4ms |
| small001 | 1,140 | 687 | Optimal | 2.0076e+02 | 59 | 131.2ms |
| small002 | 1,140 | 713 | Optimal | 3.7657e+00 | 49 | 114.4ms |
| small003 | 1,140 | 711 | Optimal | 1.8544e+01 | 57 | 127.5ms |
| small004 | 1,140 | 717 | Optimal | 5.4684e+00 | 45 | 104.7ms |
| small005 | 1,140 | 717 | Optimal | 3.4171e+00 | 38 | 97.7ms |
| small006 | 1,138 | 710 | Optimal | 2.4431e+00 | 48 | 105.8ms |
| small007 | 1,137 | 711 | Optimal | 1.3882e+00 | 47 | 105.5ms |
| small008 | 1,134 | 712 | Optimal | 8.0533e-01 | 36 | 90.9ms |
| small009 | 1,135 | 710 | Optimal | 4.9076e-01 | 36 | 99.1ms |
| small010 | 1,138 | 711 | Optimal | 2.2624e-01 | 44 | 99.5ms |
| small011 | 1,133 | 705 | Optimal | 8.8614e-02 | 44 | 101.5ms |
| small012 | 1,134 | 706 | Optimal | 6.2256e-02 | 44 | 100.6ms |
| small013 | 1,131 | 701 | Optimal | 9.6255e-02 | 43 | 96.4ms |
| small014 | 1,130 | 687 | Optimal | 1.2293e-01 | 43 | 101.9ms |
| small015 | 1,130 | 683 | Optimal | 1.6800e-01 | 43 | 110.0ms |
| small016 | 1,130 | 677 | Optimal | 2.2026e-01 | 45 | 100.9ms |
| stair | 467 | 356 | Optimal | -2.5127e+02 | 72 | 113.8ms |
| standata | 1,075 | 359 | Optimal | 1.2577e+03 | 52 | 99.9ms |
| standmps | 1,075 | 467 | Optimal | 1.4060e+03 | 69 | 120.7ms |
| stocfor1 | 111 | 117 | Optimal | -1.3875e+04 | 43 | 49.9ms |
| stocfor2 | 2,031 | 2,157 | Optimal | -3.9024e+04 | 173 | 572.8ms |
| tuff | 587 | 294 | Optimal | 2.9215e-01 | 147 | 216.6ms |
| vtp.base | 203 | 198 | Optimal | 1.2983e+05 | 382 | 208.7ms |
| wood1p | 2,594 | 244 | Acceptable | 1.4430e+00 | 102 | 1.92s |
| woodw | 8,405 | 1,098 | Optimal | 1.3044e+00 | 70 | 862.2ms |
| zed | 43 | 116 | Optimal | -3.5768e+02 | 128 | 79.1ms |

POUNCE: **344/371 Optimal** in 1231.18s total

### LPopt

| Problem | n | m | Status | Objective | Iters | Time |
|---------|---|---|--------|-----------|-------|------|
| ex10 | 17,680 | 69,608 | Maximum_CpuTime_Exceeded | N/A | 43 | 1800.06s |
| irish-electricity | 61,728 | 104,259 | Maximum_Iterations_Exceeded | 2.4570e+06 | 2999 | 1136.12s |
| qap15 | 22,275 | 6,330 | Maximum_CpuTime_Exceeded | N/A | 23 | 1800.06s |
| supportcase10 | 14,630 | 165,684 | Maximum_CpuTime_Exceeded | N/A | 53 | 1800.09s |

POUNCE: **0/4 Optimal** in 6536.33s total

---
*Generated by benchmark_report.py*