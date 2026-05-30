# POUNCE Benchmark Report

Generated: 2026-05-29 15:13:51

## Provenance

| Component | Version / Detail |
|-----------|------------------|
| POUNCE | v0.3.0 (main @ ce6337c-dirty) |
| POUNCE linear solver | feral (default) |
| Ipopt | Ipopt 3.14.20 (Darwin arm64), ASL(20241202) |
| Ipopt linear solver | ma57 (via ref/Ipopt/install-ma57) |
| Platform | Darwin 25.3.0 arm64 |

Suites in this report were each produced by their respective
`make -C benchmarks <suite>-run` target. GAMS results are sourced from
`gams/nlpbench/runsolver/*.csv` and use GAMS's bundled linear solver,
not the Ipopt install above.

## Executive Summary

| Metric | POUNCE | Ipopt |
|--------|--------|-------|
| Optimal (strict) | **1362/1582** (86.1%) | **1268/1582** (80.2%) |
| Acceptable (informational, *not* counted as solved) | 5 | 5 |
| Solved exclusively (strict Optimal) | 115 | 21 |
| Both Optimal | 1247 | |
| Matching objectives (< 0.01%) | 1215/1247 | |

> **Note:** All headline counts use strict Optimal status only. `Acceptable`
> means the iterate met relaxed tolerances but not the requested tolerance —
> per CLAUDE.md's "Honesty in Benchmarks" rule it is reported separately and
> never folded into the pass rate. See the "Acceptable (not Optimal)" and
> "Different Local Minima" sections below.

## Per-Suite Summary

| Suite | Problems | POUNCE Optimal | Ipopt Optimal | POUNCE only | Ipopt only | Both Optimal | Match |
|-------|----------|---------------|--------------|-------------|------------|--------------|-------|
| CUTEst | 727 | 557 (76.6%) | 555 (76.3%) | 8 | 6 | 549 | 544/549 |
| Electrolyte | 13 | 13 (100.0%) | 13 (100.0%) | 0 | 0 | 13 | 13/13 |
| Grid | 4 | 4 (100.0%) | 4 (100.0%) | 0 | 0 | 4 | 4/4 |
| CHO | 1 | 1 (100.0%) | 0 (0.0%) | 1 | 0 | 0 | 0/1 |
| Water | 6 | 6 (100.0%) | 6 (100.0%) | 0 | 0 | 6 | 4/6 |
| Gas | 4 | 3 (75.0%) | 3 (75.0%) | 0 | 0 | 3 | 3/3 |
| Mittelmann | 47 | 44 (93.6%) | 0 (0.0%) | 44 | 0 | 0 | 0/1 |
| LargeScale | 15 | 15 (100.0%) | 0 (0.0%) | 15 | 0 | 0 | 0/1 |
| GAMS | 765 | 719 (94.0%) | 687 (89.8%) | 47 | 15 | 672 | 647/672 |

## CUTEst Suite — Performance

On 549 commonly-solved problems:

| Metric | POUNCE | Ipopt |
|--------|--------|-------|
| Median time | 7.9ms | 728us |
| Total time | 20.65s | 14.53s |
| Mean iterations | 36.4 | 35.9 |
| Median iterations | 13 | 12 |

- **Geometric mean speedup**: 0.1x
- **Median speedup**: 0.1x
- POUNCE faster: 4/549 (1%)
- POUNCE 10x+ faster: 0/549
- Ipopt faster: 545/549

## Electrolyte Suite — Performance

On 13 commonly-solved problems:

| Metric | POUNCE | Ipopt |
|--------|--------|-------|
| Median time | 28.0ms | 32.0ms |
| Total time | 379.1ms | 427.0ms |
| Mean iterations | 12.0 | 12.2 |
| Median iterations | 10 | 10 |

- **Geometric mean speedup**: 1.1x
- **Median speedup**: 1.1x
- POUNCE faster: 12/13 (92%)
- POUNCE 10x+ faster: 0/13
- Ipopt faster: 1/13

## Grid Suite — Performance

On 4 commonly-solved problems:

| Metric | POUNCE | Ipopt |
|--------|--------|-------|
| Median time | 34.6ms | 38.2ms |
| Total time | 139.7ms | 144.0ms |
| Mean iterations | 15.5 | 15.5 |
| Median iterations | 17 | 17 |

- **Geometric mean speedup**: 1.0x
- **Median speedup**: 1.1x
- POUNCE faster: 3/4 (75%)
- POUNCE 10x+ faster: 0/4
- Ipopt faster: 1/4

## Water Suite — Performance

On 6 commonly-solved problems:

| Metric | POUNCE | Ipopt |
|--------|--------|-------|
| Median time | 124.4ms | 101.6ms |
| Total time | 841.9ms | 510.2ms |
| Mean iterations | 210.8 | 188.5 |
| Median iterations | 183 | 163 |

- **Geometric mean speedup**: 0.7x
- **Median speedup**: 0.8x
- POUNCE faster: 0/6 (0%)
- POUNCE 10x+ faster: 0/6
- Ipopt faster: 6/6

## Gas Suite — Performance

On 3 commonly-solved problems:

| Metric | POUNCE | Ipopt |
|--------|--------|-------|
| Median time | 94.0ms | 103.4ms |
| Total time | 462.2ms | 297.7ms |
| Mean iterations | 39.0 | 39.7 |
| Median iterations | 20 | 20 |

- **Geometric mean speedup**: 0.8x
- **Median speedup**: 1.1x
- POUNCE faster: 2/3 (67%)
- POUNCE 10x+ faster: 0/3
- Ipopt faster: 1/3

## GAMS Suite — Performance

On 670 commonly-solved problems:

| Metric | POUNCE | Ipopt |
|--------|--------|-------|
| Median time | 9.0ms | 8.0ms |
| Total time | 4810.20s | 3430.01s |
| Mean iterations | 43.7 | 52.3 |
| Median iterations | 15 | 15 |

- **Geometric mean speedup**: 1.0x
- **Median speedup**: 1.0x
- POUNCE faster: 226/670 (34%)
- POUNCE 10x+ faster: 9/670
- Ipopt faster: 247/670

## Failure Analysis

### CUTEst Suite

| Failure Mode | POUNCE | Ipopt |
|-------------|--------|-------|
| Acceptable | 5 | 4 |
| Diverging_Iterates | 3 | 3 |
| Error_In_Step_Computation | 0 | 2 |
| Infeasible_Problem_Detected | 12 | 12 |
| Invalid_Number_Detected | 0 | 1 |
| Maximum_Iterations_Exceeded | 14 | 12 |
| Not_Enough_Degrees_Of_Freedom | 123 | 123 |
| Restoration_Failed | 4 | 5 |
| Timeout | 9 | 10 |

### CHO Suite

| Failure Mode | POUNCE | Ipopt |
|-------------|--------|-------|
| Acceptable | 0 | 1 |

### Gas Suite

| Failure Mode | POUNCE | Ipopt |
|-------------|--------|-------|
| Infeasible_Problem_Detected | 1 | 1 |

### Mittelmann Suite

| Failure Mode | POUNCE | Ipopt |
|-------------|--------|-------|
| ERROR | 3 | 0 |
| N/A | 0 | 47 |

### LargeScale Suite

| Failure Mode | POUNCE | Ipopt |
|-------------|--------|-------|
| N/A | 0 | 15 |

### GAMS Suite

| Failure Mode | POUNCE | Ipopt |
|-------------|--------|-------|
| GAMS_ms13_ss10 | 0 | 3 |
| GAMS_ms13_ss9 | 3 | 0 |
| GAMS_ms16_ss1 | 3 | 3 |
| GAMS_ms5_ss1 | 0 | 6 |
| GAMS_ms7_ss1 | 14 | 0 |
| GAMS_ms7_ss10 | 4 | 0 |
| GAMS_ms7_ss2 | 5 | 0 |
| N/A | 1 | 40 |
| TerminatedBySolver | 7 | 15 |
| Timeout | 9 | 11 |

## Regressions (Ipopt Optimal, POUNCE not Optimal)

| Problem | Suite | n | m | POUNCE status | Ipopt obj |
|---------|-------|---|---|--------------|-----------|
| DECONVU | CUTEst | 63 | 0 | Acceptable | 5.098630e-11 |
| ELATTAR | CUTEst | 7 | 102 | Acceptable | 1.054115e+00 |
| HATFLDFL | CUTEst | 3 | 0 | Maximum_Iterations_Exceeded | 1.332275e-19 |
| MSS1 | CUTEst | 90 | 73 | Maximum_Iterations_Exceeded | -1.500000e+01 |
| PFIT1 | CUTEst | 3 | 3 | Infeasible_Problem_Detected | 0.000000e+00 |
| TAXR13322 | CUTEst | 72 | 1261 | Maximum_Iterations_Exceeded | -6.449419e+04 |
| globallib.gms/arki0007.gms | GAMS | 4786 | 5046 | TerminatedBySolver | -7.447090e+02 |
| globallib.gms/arki0012.gms | GAMS | 19315 | 17738 | Timeout | -2.242524e+05 |
| globallib.gms/arki0014.gms | GAMS | 19306 | 17693 | Timeout | -2.239770e+05 |
| globallib.gms/arki0017.gms | GAMS | 4333 | 2573 | GAMS_ms7_ss1 | -1.218331e+02 |
| globallib.gms/ex8_3_4.gms | GAMS | 111 | 77 | GAMS_ms7_ss10 | -3.060711e+00 |
| globallib.gms/ex9_1_1.gms | GAMS | 14 | 13 | GAMS_ms7_ss1 | -1.300000e+01 |
| globallib.gms/ex9_1_10.gms | GAMS | 15 | 13 | GAMS_ms7_ss1 | -3.250000e+00 |
| globallib.gms/ex9_1_8.gms | GAMS | 15 | 13 | GAMS_ms7_ss1 | -3.250000e+00 |
| globallib.gms/turkey.gms | GAMS | 519 | 288 | GAMS_ms7_ss1 | -2.933016e+04 |
| large.gms/arki0017.gms | GAMS | 4333 | 2573 | GAMS_ms7_ss1 | -1.218331e+02 |
| medium.gms/ex8_3_4.gms | GAMS | 111 | 77 | GAMS_ms7_ss10 | -3.060711e+00 |
| mittelmann.gms/clnlbeam.gms.gz | GAMS | 60000 | 40001 | GAMS_ms7_ss2 | 3.448761e+02 |
| mittelmann.gms/nql180.gms.gz | GAMS | 129602 | 130081 | N/A | -9.277280e-01 |
| mittelmann.gms/qcqp1500-1nc.gms.gz | GAMS | 1501 | 10509 | Timeout | 4.778295e+06 |
| mittelmann.gms/robot_c.gms.gz | GAMS | 1002 | 52014 | Timeout | 1.405969e+00 |

## Wins (POUNCE Optimal, Ipopt not Optimal) — 115 problems

| Problem | Suite | n | m | Ipopt status | POUNCE obj |
|---------|-------|---|---|-------------|------------|
| BratuProblem@s0.100 | LargeScale | 1000 | 998 | N/A | 0.000000e+00 |
| BratuProblem@s0.500 | LargeScale | 5000 | 4998 | N/A | 0.000000e+00 |
| BratuProblem@s1.000 | LargeScale | 10000 | 9998 | N/A | 0.000000e+00 |
| CRESC100 | CUTEst | 6 | 200 | Infeasible_Problem_Detected | 7.443588e-01 |
| CRESC132 | CUTEst | 6 | 2654 | Timeout | 8.577054e-01 |
| ChainedRosenbrock@s0.100 | LargeScale | 200 | 0 | N/A | 1.000000e+00 |
| ChainedRosenbrock@s0.500 | LargeScale | 1000 | 0 | N/A | 1.000000e+00 |
| ChainedRosenbrock@s1.000 | LargeScale | 2000 | 0 | N/A | 1.000000e+00 |
| DENSCHNDNE | CUTEst | 3 | 3 | Acceptable | 0.000000e+00 |
| HIMMELBJ | CUTEst | 45 | 14 | Error_In_Step_Computation | N/A |
| NARX_CFy | Mittelmann | 0 | 0 | N/A | 8.601459e-03 |
| OET7 | CUTEst | 7 | 1002 | Maximum_Iterations_Exceeded | 4.445208e-05 |
| OptimalControl@s0.100 | LargeScale | 10001 | 5001 | N/A | 1.174241e-01 |
| OptimalControl@s0.500 | LargeScale | 50001 | 25001 | N/A | 1.173518e-01 |
| OptimalControl@s1.000 | LargeScale | 100001 | 50001 | N/A | 1.173428e-01 |
| PFIT3 | CUTEst | 3 | 3 | Restoration_Failed | 0.000000e+00 |
| POLAK6 | CUTEst | 5 | 4 | Restoration_Failed | -4.400000e+01 |
| PRICE4NE | CUTEst | 2 | 2 | Acceptable | 0.000000e+00 |
| PoissonControl@s0.100 | LargeScale | 800 | 400 | N/A | 9.939498e-02 |
| PoissonControl@s0.500 | LargeScale | 20000 | 10000 | N/A | 9.946757e-02 |
| PoissonControl@s1.000 | LargeScale | 80000 | 40000 | N/A | 9.947002e-02 |
| SparseQP@s0.100 | LargeScale | 5000 | 5000 | N/A | -1.249817e+03 |
| SparseQP@s0.500 | LargeScale | 25000 | 25000 | N/A | -6.249817e+03 |
| SparseQP@s1.000 | LargeScale | 50000 | 50000 | N/A | -1.249982e+04 |
| WM_CFy | Mittelmann | 0 | 0 | N/A | 1.221803e+00 |
| arki0003 | Mittelmann | 0 | 0 | N/A | 3.795201e+03 |
| arki0009 | Mittelmann | 0 | 0 | N/A | -1.775525e+04 |
| bearing_400 | Mittelmann | 0 | 0 | N/A | -1.546765e-01 |
| camshape_6400 | Mittelmann | 0 | 0 | N/A | -4.448378e+00 |
| cho_parmest | CHO | 21672 | 21660 | Acceptable | 4.746068e+04 |
| clnlbeam | Mittelmann | 0 | 0 | N/A | 3.448761e+02 |
| cont5_1_l | Mittelmann | 0 | 0 | N/A | 2.720975e+00 |
| cont5_2_1_l | Mittelmann | 0 | 0 | N/A | 6.627472e-04 |
| cont5_2_2_l | Mittelmann | 0 | 0 | N/A | 5.199975e-04 |
| cont5_2_3_l | Mittelmann | 0 | 0 | N/A | 5.207821e-04 |
| cont5_2_4_l | Mittelmann | 0 | 0 | N/A | 6.637143e-02 |
| corkscrw | Mittelmann | 0 | 0 | N/A | 9.809596e+01 |
| dirichlet120 | Mittelmann | 0 | 0 | N/A | 3.503609e-02 |
| dtoc1nd | Mittelmann | 0 | 0 | N/A | 1.364558e+02 |
| dtoc2 | Mittelmann | 0 | 0 | N/A | 2.863299e+00 |
| elec_400 | Mittelmann | 0 | 0 | N/A | 1.001292e+02 |
| ex1_160 | Mittelmann | 0 | 0 | N/A | 6.386126e-02 |
| ex1_320 | Mittelmann | 0 | 0 | N/A | 6.541997e-02 |
| ex4_2_160 | Mittelmann | 0 | 0 | N/A | 3.646117e+00 |
| ex4_2_320 | Mittelmann | 0 | 0 | N/A | 3.639168e+00 |
| ex8_2_2 | Mittelmann | 0 | 0 | N/A | -5.526663e+02 |
| ex8_2_3 | Mittelmann | 0 | 0 | N/A | -3.731079e+03 |
| gasoil_3200 | Mittelmann | 0 | 0 | N/A | 5.236596e-03 |
| globallib.gms/camshape12800.gms | GAMS | 25600 | 25601 | N/A | 4.751979e+00 |
| globallib.gms/camshape3200.gms | GAMS | 6400 | 6401 | Timeout | 4.322945e+00 |
| globallib.gms/camshape6400.gms | GAMS | 12800 | 12801 | N/A | 4.448385e+00 |
| globallib.gms/ex7_3_4.gms | GAMS | 13 | 18 | TerminatedBySolver | 8.460472e+00 |
| globallib.gms/ex7_3_5.gms | GAMS | 14 | 16 | TerminatedBySolver | 1.206893e+00 |
| globallib.gms/ex9_2_6.gms | GAMS | 17 | 13 | TerminatedBySolver | 3.000000e+00 |
| globallib.gms/glider2.gms | GAMS | 666 | 610 | GAMS_ms5_ss1 | 1.282400e+03 |
| globallib.gms/glider_org.gms | GAMS | 666 | 610 | GAMS_ms5_ss1 | 1.282400e+03 |
| globallib.gms/glidera.gms | GAMS | 666 | 610 | GAMS_ms5_ss1 | 1.282400e+03 |
| globallib.gms/glidera3200.gms | GAMS | 41616 | 38410 | Timeout | 1.247987e+03 |
| globallib.gms/hs62.gms | GAMS | 4 | 2 | TerminatedBySolver | -2.627251e+04 |
| globallib.gms/methanol12800.gms.gz | GAMS | 384006 | 383998 | Timeout | 9.022293e-03 |
| globallib.gms/polygon75.gms | GAMS | 151 | 2850 | TerminatedBySolver | -7.844636e-01 |
| globallib.gms/rocket6400.gms.gz | GAMS | 38408 | 32003 | Timeout | 1.012837e+00 |
| globallib.gms/rocket800.gms.gz | GAMS | 4808 | 4003 | Timeout | 1.012837e+00 |
| henon120 | Mittelmann | 0 | 0 | N/A | 1.332947e+02 |
| issue51.gms/arki0013.gms | GAMS | 19315 | 17738 | N/A | -2.242531e+05 |
| issue51.gms/cvxqp3.gms | GAMS | 10001 | 7501 | N/A | 1.157111e+05 |
| issue51.gms/dallasl.gms | GAMS | 10001 | 7501 | N/A | 1.157111e+05 |
| issue51.gms/hues-mod.gms | GAMS | 10002 | 3 | N/A | 6.545183e+00 |
| issue51.gms/huestis.gms | GAMS | 10002 | 3 | N/A | 6.545191e+00 |
| issue51.gms/marine_1600.gms.gz | GAMS | 38416 | 38393 | N/A | 1.974653e+07 |
| issue51.gms/narx_cfy.gms.gz | GAMS | 45486 | 46745 | N/A | 8.609220e-03 |
| issue51.gms/nuffield2_trap.gms | GAMS | 10087 | 13203 | N/A | -5.232006e+00 |
| issue51.gms/powerflow10.gms.gz | GAMS | 1873729 | 940249 | N/A | 0.000000e+00 |
| issue51.gms/powerflow14.gms.gz | GAMS | 1888801 | 955321 | N/A | 0.000000e+00 |
| issue51.gms/powerflow15.gms.gz | GAMS | 1888801 | 1017433 | N/A | 3.410011e+01 |
| issue51.gms/powerflow18.gms.gz | GAMS | 1888801 | 955321 | N/A | 0.000000e+00 |
| issue51.gms/powerflow19.gms.gz | GAMS | 1888803 | 1017506 | N/A | 3.917828e-01 |
| issue51.gms/powerflow20.gms.gz | GAMS | 1873729 | 940249 | N/A | 0.000000e+00 |
| issue51.gms/powerflow21.gms.gz | GAMS | 1888801 | 1017433 | N/A | 1.315241e+01 |
| issue51.gms/powerflow22.gms.gz | GAMS | 1873729 | 940249 | N/A | 0.000000e+00 |
| issue51.gms/powerflow23.gms.gz | GAMS | 1888801 | 1017433 | N/A | 1.184445e+00 |
| issue51.gms/powerflow24.gms.gz | GAMS | 1873729 | 940249 | N/A | 0.000000e+00 |
| issue51.gms/powerflow25.gms.gz | GAMS | 1888801 | 955321 | N/A | 0.000000e+00 |
| issue51.gms/powerflow26.gms.gz | GAMS | 1873729 | 940153 | N/A | 0.000000e+00 |
| issue51.gms/powerflow27.gms.gz | GAMS | 1888801 | 955321 | N/A | 0.000000e+00 |
| issue51.gms/powerflow28.gms.gz | GAMS | 1873729 | 940153 | N/A | 0.000000e+00 |
| issue51.gms/qcqp1000-2c.gms.gz | GAMS | 1001 | 5108 | N/A | 7.381274e+05 |
| issue51.gms/qcqp1500-1c.gms.gz | GAMS | 1501 | 10509 | N/A | 3.882979e+06 |
| issue51.gms/qcqp750-2nc.gms.gz | GAMS | 751 | 139 | N/A | -2.161899e+07 |
| issue51.gms/robota3200.gms | GAMS | 35213 | 25603 | N/A | 9.140917e+00 |
| just01.gms/powerflow01.gms.gz | GAMS | 239545 | 126745 | N/A | 0.000000e+00 |
| just22.gms/powerflow22.gms.gz | GAMS | 1873729 | 940249 | N/A | 0.000000e+00 |
| lane_emden120 | Mittelmann | 0 | 0 | N/A | 9.340251e+00 |
| large.gms/polygon75.gms | GAMS | 151 | 2850 | TerminatedBySolver | -7.844636e-01 |
| marine_1600 | Mittelmann | 0 | 0 | N/A | 1.974653e+07 |
| mittelmann.gms/camshape_6400.gms.gz | GAMS | 12798 | 12801 | N/A | 4.448394e+00 |
| mittelmann.gms/dtoc2.gms.gz | GAMS | 64951 | 38971 | N/A | 2.863299e+00 |
| nql180 | Mittelmann | 0 | 0 | N/A | -9.277190e-01 |
| optmass | Mittelmann | 0 | 0 | N/A | -1.204045e-01 |
| pinene_3200 | Mittelmann | 0 | 0 | N/A | 1.987217e+01 |
| powerflow.gms/powerflow03.gms.gz | GAMS | 251233 | 163153 | Timeout | 4.414222e-01 |
| qcqp1000-1nc | Mittelmann | 0 | 0 | N/A | -1.841514e+06 |
| qcqp1000-2c | Mittelmann | 0 | 0 | N/A | 1.112949e+04 |
| qcqp1000-2nc | Mittelmann | 0 | 0 | N/A | 1.098821e+03 |
| qcqp1500-1c | Mittelmann | 0 | 0 | N/A | 1.751829e+04 |
| qcqp1500-1nc | Mittelmann | 0 | 0 | N/A | 1.537714e+04 |
| qcqp500-3c | Mittelmann | 0 | 0 | N/A | -8.517295e+03 |
| qcqp500-3nc | Mittelmann | 0 | 0 | N/A | -1.369706e+04 |
| qcqp750-2c | Mittelmann | 0 | 0 | N/A | -1.202807e+04 |
| qcqp750-2nc | Mittelmann | 0 | 0 | N/A | -1.445230e+04 |
| qssp180 | Mittelmann | 0 | 0 | N/A | -6.639447e+00 |
| robot_1600 | Mittelmann | 0 | 0 | N/A | 9.140924e+00 |
| rocket_12800 | Mittelmann | 0 | 0 | N/A | -1.012792e+00 |
| steering_12800 | Mittelmann | 0 | 0 | N/A | 5.545709e-01 |
| svanberg | Mittelmann | 0 | 0 | N/A | 8.362382e+04 |

## Acceptable (not Optimal) — 5 problems

These problems converged within relaxed tolerances but not strict tolerances.

| Problem | Suite | n | m | Ipopt status | POUNCE obj | Ipopt obj |
|---------|-------|---|---|-------------|------------|-----------|
| BT8 | CUTEst | 5 | 2 | Acceptable | 1.000000e+00 | 1.000000e+00 |
| DECONVU | CUTEst | 63 | 0 | Optimal | 8.107592e-14 | 5.098630e-11 |
| DJTL | CUTEst | 2 | 0 | Acceptable | -8.951545e+03 | -8.951545e+03 |
| ELATTAR | CUTEst | 7 | 102 | Optimal | 1.869735e+02 | 1.054115e+00 |
| EQC | CUTEst | 9 | 3 | Error_In_Step_Computation | -8.617445e+02 | -8.576466e+02 |

## POUNCE-Only Suite Details

These suites currently run POUNCE only — no Ipopt-side comparison is captured in their result files. Per-problem timing and iteration counts are shown so users can inspect the whole picture.

### CHO

| Problem | n | m | Status | Objective | Iters | Time |
|---------|---|---|--------|-----------|-------|------|
| cho_parmest | 21,672 | 21,660 | Optimal | 4.7461e+04 | 35 | 4.76s |

POUNCE: **1/1 Optimal** in 4.76s total

### Mittelmann

| Problem | n | m | Status | Objective | Iters | Time |
|---------|---|---|--------|-----------|-------|------|
| NARX_CFy | — | — | Optimal | 8.6015e-03 | 604 | 506.81s |
| WM_CFy | — | — | Optimal | 1.2218e+00 | 707 | 2709.53s |
| arki0003 | — | — | Optimal | 3.7952e+03 | 317 | 1.42s |
| arki0009 | — | — | Optimal | -1.7755e+04 | 355 | 10.75s |
| bearing_400 | — | — | Optimal | -1.5468e-01 | 16 | 6.29s |
| camshape_6400 | — | — | Optimal | -4.4484e+00 | 87 | 1.93s |
| clnlbeam | — | — | Optimal | 3.4488e+02 | 552 | 42.81s |
| cont5_1_l | — | — | Optimal | 2.7210e+00 | 16 | 5.88s |
| cont5_2_1_l | — | — | Optimal | 6.6275e-04 | 41 | 12.28s |
| cont5_2_2_l | — | — | Optimal | 5.2000e-04 | 48 | 14.57s |
| cont5_2_3_l | — | — | Optimal | 5.2078e-04 | 55 | 15.18s |
| cont5_2_4_l | — | — | Optimal | 6.6371e-02 | 12 | 6.49s |
| corkscrw | — | — | Optimal | 9.8096e+01 | 313 | 23.02s |
| dirichlet120 | — | — | Optimal | 3.5036e-02 | 56 | 68.21s |
| dtoc1nd | — | — | Optimal | 1.3646e+02 | 8 | 3.09s |
| dtoc2 | — | — | Optimal | 2.8633e+00 | 5 | 4.75s |
| elec_400 | — | — | Optimal | 1.0013e+02 | 126 | 42.48s |
| ex1_160 | — | — | Optimal | 6.3861e-02 | 15 | 1.78s |
| ex1_320 | — | — | Optimal | 6.5420e-02 | 8 | 5.09s |
| ex4_2_160 | — | — | Optimal | 3.6461e+00 | 21 | 2.45s |
| ex4_2_320 | — | — | Optimal | 3.6392e+00 | 17 | 10.67s |
| ex8_2_2 | — | — | Optimal | -5.5267e+02 | 67 | 503.0ms |
| ex8_2_3 | — | — | Optimal | -3.7311e+03 | 69 | 1.19s |
| gasoil_3200 | — | — | Optimal | 5.2366e-03 | 14 | 3.62s |
| henon120 | — | — | Optimal | 1.3329e+02 | 266 | 111.42s |
| lane_emden120 | — | — | Optimal | 9.3403e+00 | 84 | 95.47s |
| marine_1600 | — | — | Optimal | 1.9747e+07 | 14 | 13.69s |
| nql180 | — | — | Optimal | -9.2772e-01 | 41 | 46.63s |
| optmass | — | — | Optimal | -1.2040e-01 | 47 | 5.68s |
| pinene_3200 | — | — | Optimal | 1.9872e+01 | 10 | 6.01s |
| qcqp1000-1nc | — | — | Optimal | -1.8415e+06 | 192 | 21.50s |
| qcqp1000-2c | — | — | Optimal | 1.1129e+04 | 102 | 51.91s |
| qcqp1000-2nc | — | — | Optimal | 1.0988e+03 | 295 | 91.53s |
| qcqp1500-1c | — | — | Optimal | 1.7518e+04 | 102 | 219.49s |
| qcqp1500-1nc | — | — | Optimal | 1.5377e+04 | 256 | 623.08s |
| qcqp500-3c | — | — | Optimal | -8.5173e+03 | 397 | 422.92s |
| qcqp500-3nc | — | — | Optimal | -1.3697e+04 | 121 | 54.87s |
| qcqp750-2c | — | — | Optimal | -1.2028e+04 | 67 | 116.07s |
| qcqp750-2nc | — | — | Optimal | -1.4452e+04 | 78 | 154.16s |
| qssp180 | — | — | Optimal | -6.6394e+00 | 33 | 40.03s |
| robot_1600 | — | — | Optimal | 9.1409e+00 | 38 | 1.86s |
| robot_a | — | — | ERROR | 8.1276e+00 | 2999 | 931.48s |
| robot_b | — | — | ERROR | 2.4874e+01 | 2999 | 923.81s |
| robot_c | — | — | ERROR | 3.3747e+01 | 2999 | 1159.82s |
| rocket_12800 | — | — | Optimal | -1.0128e+00 | 24 | 142.40s |
| steering_12800 | — | — | Optimal | 5.5457e-01 | 20 | 85.14s |
| svanberg | — | — | Optimal | 8.3624e+04 | 34 | 8.51s |

POUNCE: **44/47 Optimal** in 8828.25s total

### LargeScale

| Problem | n | m | Status | Objective | Iters | Time |
|---------|---|---|--------|-----------|-------|------|
| BratuProblem@s0.100 | 1,000 | 998 | Optimal | 0.0000e+00 | 2 | 15.4ms |
| BratuProblem@s0.500 | 5,000 | 4,998 | Optimal | 0.0000e+00 | 2 | 11.0ms |
| BratuProblem@s1.000 | 10,000 | 9,998 | Optimal | 0.0000e+00 | 1 | 24.3ms |
| ChainedRosenbrock@s0.100 | 200 | — | Optimal | 1.0000e+00 | 146 | 249.7ms |
| ChainedRosenbrock@s0.500 | 1,000 | — | Optimal | 1.0000e+00 | 765 | 475.1ms |
| ChainedRosenbrock@s1.000 | 2,000 | — | Optimal | 1.0000e+00 | 1484 | 1.64s |
| OptimalControl@s0.100 | 10,001 | 5,001 | Optimal | 1.1742e-01 | 1 | 31.8ms |
| OptimalControl@s0.500 | 50,001 | 25,001 | Optimal | 1.1735e-01 | 1 | 224.3ms |
| OptimalControl@s1.000 | 100,001 | 50,001 | Optimal | 1.1734e-01 | 1 | 271.8ms |
| PoissonControl@s0.100 | 800 | 400 | Optimal | 9.9395e-02 | 1 | 3.0ms |
| PoissonControl@s0.500 | 20,000 | 10,000 | Optimal | 9.9468e-02 | 1 | 126.1ms |
| PoissonControl@s1.000 | 80,000 | 40,000 | Optimal | 9.9470e-02 | 1 | 397.3ms |
| SparseQP@s0.100 | 5,000 | 5,000 | Optimal | -1.2498e+03 | 7 | 68.7ms |
| SparseQP@s0.500 | 25,000 | 25,000 | Optimal | -6.2498e+03 | 7 | 512.2ms |
| SparseQP@s1.000 | 50,000 | 50,000 | Optimal | -1.2500e+04 | 7 | 838.1ms |

POUNCE: **15/15 Optimal** in 4.89s total

---
*Generated by benchmark_report.py*