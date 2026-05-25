# POUNCE Benchmark Report

Generated: 2026-05-25 08:37:23

## Executive Summary

| Metric | POUNCE | Ipopt |
|--------|--------|-------|
| Optimal (strict) | **574/744** (77.2%) | **557/744** (74.9%) |
| Acceptable (informational, *not* counted as solved) | 4 | 5 |
| Solved exclusively (strict Optimal) | 24 | 7 |
| Both Optimal | 550 | |
| Matching objectives (< 0.01%) | 541/550 | |

> **Note:** All headline counts use strict Optimal status only. `Acceptable`
> means the iterate met relaxed tolerances but not the requested tolerance —
> per CLAUDE.md's "Honesty in Benchmarks" rule it is reported separately and
> never folded into the pass rate. See the "Acceptable (not Optimal)" and
> "Different Local Minima" sections below.

## Per-Suite Summary

| Suite | Problems | POUNCE Optimal | Ipopt Optimal | POUNCE only | Ipopt only | Both Optimal | Match |
|-------|----------|---------------|--------------|-------------|------------|--------------|-------|
| CUTEst | 727 | 558 (76.8%) | 557 (76.6%) | 8 | 7 | 550 | 541/550 |
| Electrolyte | 13 | 12 (92.3%) | 0 (0.0%) | 12 | 0 | 0 | 0/1 |
| Grid | 4 | 4 (100.0%) | 0 (0.0%) | 4 | 0 | 0 | 0/1 |

## CUTEst Suite — Performance

On 550 commonly-solved problems:

| Metric | POUNCE | Ipopt |
|--------|--------|-------|
| Median time | 2.1ms | 2.9ms |
| Total time | 19.25s | 20.23s |
| Mean iterations | 35.9 | 39.4 |
| Median iterations | 13 | 12 |

- **Geometric mean speedup**: 1.3x
- **Median speedup**: 1.3x
- POUNCE faster: 487/550 (89%)
- POUNCE 10x+ faster: 1/550
- Ipopt faster: 63/550

## Failure Analysis

### CUTEst Suite

| Failure Mode | POUNCE | Ipopt |
|-------------|--------|-------|
| Acceptable | 4 | 5 |
| Diverging_Iterates | 2 | 2 |
| Error_In_Step_Computation | 0 | 2 |
| Infeasible_Problem_Detected | 16 | 11 |
| Invalid_Number_Detected | 0 | 1 |
| Maximum_Iterations_Exceeded | 12 | 12 |
| Not_Enough_Degrees_Of_Freedom | 123 | 123 |
| Restoration_Failed | 4 | 4 |
| Search_Direction_Becomes_Too_Small | 0 | 1 |
| Timeout | 8 | 9 |

### Electrolyte Suite

| Failure Mode | POUNCE | Ipopt |
|-------------|--------|-------|
| Infeasible_Problem_Detected | 1 | 0 |
| N/A | 0 | 13 |

### Grid Suite

| Failure Mode | POUNCE | Ipopt |
|-------------|--------|-------|
| N/A | 0 | 4 |

## Regressions (Ipopt Optimal, POUNCE not Optimal)

| Problem | Suite | n | m | POUNCE status | Ipopt obj |
|---------|-------|---|---|--------------|-----------|
| BT8 | CUTEst | 5 | 2 | Acceptable | 1.000000e+00 |
| CRESC50 | CUTEst | 6 | 100 | Infeasible_Problem_Detected | 7.862467e-01 |
| DECONVU | CUTEst | 63 | 0 | Acceptable | 4.146188e-13 |
| DMN15102LS | CUTEst | 66 | 0 | Timeout | 6.637446e+02 |
| HATFLDFL | CUTEst | 3 | 0 | Maximum_Iterations_Exceeded | 6.016804e-05 |
| MSS1 | CUTEst | 90 | 73 | Maximum_Iterations_Exceeded | -1.400000e+01 |
| PFIT4 | CUTEst | 3 | 3 | Infeasible_Problem_Detected | 0.000000e+00 |

## Wins (POUNCE Optimal, Ipopt not Optimal) — 24 problems

| Problem | Suite | n | m | Ipopt status | POUNCE obj |
|---------|-------|---|---|-------------|------------|
| BuOH-water LLE | Electrolyte | 2 | 2 | N/A | 7.825841e-10 |
| CO2-water speciation | Electrolyte | 5 | 2 | N/A | -6.907755e-03 |
| CaCl2+NaCl mixed | Electrolyte | 6 | 4 | N/A | -7.723715e-01 |
| DECONVNE | CUTEst | 63 | 40 | Acceptable | 0.000000e+00 |
| DENSCHNDNE | CUTEst | 3 | 3 | Acceptable | 0.000000e+00 |
| DIAMON2DLS | CUTEst | 66 | 0 | Timeout | 6.325671e+02 |
| HCl mean activity | Electrolyte | 1 | 0 | N/A | 1.003840e-15 |
| HIMMELBJ | CUTEst | 45 | 14 | Error_In_Step_Computation | N/A |
| Multi-salt DH fit | Electrolyte | 8 | 0 | N/A | 4.292394e-12 |
| NaCl solubility | Electrolyte | 1 | 0 | N/A | 1.935429e-17 |
| NaCl speciation | Electrolyte | 4 | 3 | N/A | -4.832682e-01 |
| POLAK6 | CUTEst | 5 | 4 | Maximum_Iterations_Exceeded | -4.400000e+01 |
| PRICE4NE | CUTEst | 2 | 2 | Acceptable | 0.000000e+00 |
| Phosphoric acid | Electrolyte | 6 | 2 | N/A | -5.531314e-02 |
| Pitzer NaCl fit | Electrolyte | 3 | 0 | N/A | 4.345896e-15 |
| ROBOT | CUTEst | 14 | 2 | Search_Direction_Becomes_Too_Small | 6.593299e+00 |
| Saturated brine | Electrolyte | 3 | 3 | N/A | 0.000000e+00 |
| TAXR13322 | CUTEst | 72 | 1261 | Acceptable | -6.449419e+04 |
| Water autoionization | Electrolyte | 1 | 0 | N/A | 2.629926e-07 |
| case14_ieee | Grid | 38 | 68 | N/A | 2.178080e+03 |
| case30_ieee | Grid | 72 | 142 | N/A | 8.208515e+03 |
| case3_lmbd | Grid | 12 | 12 | N/A | 5.812643e+03 |
| case5_pjm | Grid | 20 | 22 | N/A | 1.755189e+04 |
| eNRTL T-dep fit | Electrolyte | 4 | 0 | N/A | 1.685581e-12 |

## Acceptable (not Optimal) — 4 problems

These problems converged within relaxed tolerances but not strict tolerances.

| Problem | Suite | n | m | Ipopt status | POUNCE obj | Ipopt obj |
|---------|-------|---|---|-------------|------------|-----------|
| BT8 | CUTEst | 5 | 2 | Optimal | 1.000000e+00 | 1.000000e+00 |
| DECONVU | CUTEst | 63 | 0 | Optimal | 8.107592e-14 | 4.146188e-13 |
| DJTL | CUTEst | 2 | 0 | Acceptable | -8.951545e+03 | -8.951545e+03 |
| EQC | CUTEst | 9 | 3 | Error_In_Step_Computation | -8.630052e+02 | -8.651227e+02 |

---
*Generated by benchmark_report.py*