# POUNCE Benchmark Report

Generated: 2026-05-25 10:38:49

## Provenance

| Component | Version / Detail |
|-----------|------------------|
| POUNCE | v0.1.0 (claude/release-prep-bench-docs @ 9eb217b-dirty) |
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
| Optimal (strict) | **2297/2611** (88.0%) | **2272/2611** (87.0%) |
| Acceptable (informational, *not* counted as solved) | 4 | 5 |
| Solved exclusively (strict Optimal) | 100 | 75 |
| Both Optimal | 2197 | |
| Matching objectives (< 0.01%) | 2077/2197 | |

> **Note:** All headline counts use strict Optimal status only. `Acceptable`
> means the iterate met relaxed tolerances but not the requested tolerance —
> per CLAUDE.md's "Honesty in Benchmarks" rule it is reported separately and
> never folded into the pass rate. See the "Acceptable (not Optimal)" and
> "Different Local Minima" sections below.

## Per-Suite Summary

| Suite | Problems | POUNCE Optimal | Ipopt Optimal | POUNCE only | Ipopt only | Both Optimal | Match |
|-------|----------|---------------|--------------|-------------|------------|--------------|-------|
| CUTEst | 727 | 558 (76.8%) | 557 (76.6%) | 8 | 7 | 550 | 541/550 |
| Water | 6 | 6 (100.0%) | 6 (100.0%) | 0 | 0 | 6 | 4/6 |
| Mittelmann | 47 | 44 (93.6%) | 0 (0.0%) | 44 | 0 | 0 | 0/1 |
| LargeScale | 15 | 15 (100.0%) | 0 (0.0%) | 15 | 0 | 0 | 0/1 |
| GAMS | 1816 | 1674 (92.2%) | 1709 (94.1%) | 33 | 68 | 1641 | 1532/1641 |

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

## Water Suite — Performance

On 6 commonly-solved problems:

| Metric | POUNCE | Ipopt |
|--------|--------|-------|
| Median time | 113.1ms | 74.0ms |
| Total time | 783.0ms | 422.6ms |
| Mean iterations | 219.8 | 188.5 |
| Median iterations | 183 | 163 |

- **Geometric mean speedup**: 0.6x
- **Median speedup**: 0.7x
- POUNCE faster: 0/6 (0%)
- POUNCE 10x+ faster: 0/6
- Ipopt faster: 6/6

## GAMS Suite — Performance

On 1617 commonly-solved problems:

| Metric | POUNCE | Ipopt |
|--------|--------|-------|
| Median time | 5.0ms | 5.0ms |
| Total time | 10217.12s | 4318.71s |
| Mean iterations | 45.5 | 54.6 |
| Median iterations | 14 | 14 |

- **Geometric mean speedup**: 1.0x
- **Median speedup**: 1.0x
- POUNCE faster: 528/1617 (33%)
- POUNCE 10x+ faster: 11/1617
- Ipopt faster: 633/1617

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
| GAMS_ms13_ss10 | 0 | 6 |
| GAMS_ms13_ss5 | 0 | 10 |
| GAMS_ms13_ss9 | 6 | 0 |
| GAMS_ms16_ss1 | 3 | 3 |
| GAMS_ms3_ss1 | 0 | 7 |
| GAMS_ms5_ss1 | 0 | 14 |
| GAMS_ms6_ss2 | 0 | 1 |
| GAMS_ms6_ss5 | 1 | 0 |
| GAMS_ms7_ss1 | 44 | 0 |
| GAMS_ms7_ss10 | 3 | 0 |
| GAMS_ms7_ss2 | 20 | 0 |
| GAMS_ms7_ss5 | 0 | 1 |
| N/A | 1 | 6 |
| TerminatedBySolver | 51 | 30 |
| Timeout | 13 | 29 |

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
| globallib.gms/arki0003.gms | GAMS | 2283 | 2583 | GAMS_ms7_ss2 | 3.795206e+03 |
| globallib.gms/arki0007.gms | GAMS | 4786 | 5046 | TerminatedBySolver | -7.447090e+02 |
| globallib.gms/arki0012.gms | GAMS | 19315 | 17738 | Timeout | -2.242524e+05 |
| globallib.gms/arki0017.gms | GAMS | 4333 | 2573 | GAMS_ms7_ss1 | -1.218331e+02 |
| globallib.gms/ex3_1_1.gms | GAMS | 9 | 7 | TerminatedBySolver | 7.049248e+03 |
| globallib.gms/ex5_4_2.gms | GAMS | 9 | 7 | TerminatedBySolver | 7.512230e+03 |
| globallib.gms/ex8_3_10a.gms | GAMS | 142 | 109 | GAMS_ms7_ss1 | -1.319326e+00 |
| globallib.gms/ex8_3_2.gms | GAMS | 111 | 77 | GAMS_ms7_ss1 | -3.313267e-01 |
| globallib.gms/ex8_3_6.gms | GAMS | 111 | 77 | GAMS_ms7_ss1 | -5.000000e-01 |
| globallib.gms/ex9_1_1.gms | GAMS | 14 | 13 | GAMS_ms7_ss1 | -1.300000e+01 |
| globallib.gms/ex9_1_10.gms | GAMS | 15 | 13 | GAMS_ms7_ss1 | -3.250000e+00 |
| globallib.gms/ex9_1_8.gms | GAMS | 15 | 13 | GAMS_ms7_ss1 | -3.250000e+00 |
| globallib.gms/glidera800.gms | GAMS | 10416 | 9610 | TerminatedBySolver | 1.247984e+03 |
| globallib.gms/mathopt2.gms | GAMS | 3 | 5 | TerminatedBySolver | 1.906685e-23 |
| globallib.gms/st_e34.gms | GAMS | 7 | 5 | TerminatedBySolver | 1.561951e-02 |
| globallib.gms/turkey.gms | GAMS | 519 | 288 | GAMS_ms7_ss1 | -2.933016e+04 |
| large.gms/arki0003.gms | GAMS | 2283 | 2583 | GAMS_ms7_ss2 | 3.795206e+03 |
| large.gms/arki0017.gms | GAMS | 4333 | 2573 | GAMS_ms7_ss1 | -1.218331e+02 |
| medium.gms/ex8_3_10a.gms | GAMS | 142 | 109 | GAMS_ms7_ss1 | -1.319326e+00 |
| medium.gms/ex8_3_2.gms | GAMS | 111 | 77 | GAMS_ms7_ss1 | -3.313267e-01 |
| medium.gms/ex8_3_6.gms | GAMS | 111 | 77 | GAMS_ms7_ss1 | -5.000000e-01 |
| mittelmann.gms/clnlbeam.gms.gz | GAMS | 60000 | 40001 | Timeout | 3.448761e+02 |
| mittelmann.gms/nql180.gms.gz | GAMS | 129602 | 130081 | N/A | -9.277280e-01 |
| mittelmann.gms/qcqp1000-1nc.gms.gz | GAMS | 1001 | 155 | GAMS_ms7_ss1 | -2.662887e+07 |
| mittelmann.gms/qcqp1500-1nc.gms.gz | GAMS | 1501 | 10509 | Timeout | 4.778295e+06 |
| mittelmann.gms/qcqp750-2nc.gms.gz | GAMS | 751 | 139 | Timeout | -2.161899e+07 |
| mittelmann.gms/robot_c.gms.gz | GAMS | 1002 | 52014 | Timeout | 1.405969e+00 |
| powerflow.gms/powerflow15.gms.gz | GAMS | 1888801 | 1017433 | Timeout | 3.411712e+01 |
| princetonlib.gms/bt8.gms | GAMS | 6 | 3 | GAMS_ms7_ss1 | 1.000000e+00 |
| princetonlib.gms/cvxqp3.gms | GAMS | 10001 | 7501 | Timeout | 1.157111e+05 |
| princetonlib.gms/dallasl.gms | GAMS | 10001 | 7501 | Timeout | 1.157111e+05 |
| princetonlib.gms/dallasm.gms | GAMS | 197 | 152 | GAMS_ms7_ss1 | -4.819819e+04 |
| princetonlib.gms/emfl_socp_vareps.gms | GAMS | 5677 | 5626 | GAMS_ms7_ss1 | 4.686754e+01 |
| princetonlib.gms/emfl_vareps.gms | GAMS | 52 | 1 | GAMS_ms7_ss2 | 4.686754e+01 |
| princetonlib.gms/fermat2_vareps.gms | GAMS | 4 | 1 | GAMS_ms7_ss2 | 4.472136e+00 |
| princetonlib.gms/flosp2tm.gms | GAMS | 692 | 1 | GAMS_ms7_ss1 | 1.000000e+01 |
| princetonlib.gms/gridnetf.gms | GAMS | 7566 | 3845 | GAMS_ms7_ss1 | 2.421090e+02 |
| princetonlib.gms/hs055.gms | GAMS | 7 | 7 | GAMS_ms7_ss1 | 6.779703e+00 |
| princetonlib.gms/hs069.gms | GAMS | 8 | 6 | GAMS_ms7_ss1 | -9.955213e+02 |
| princetonlib.gms/hs091.gms | GAMS | 36 | 32 | TerminatedBySolver | 1.362657e+00 |
| princetonlib.gms/hs095.gms | GAMS | 7 | 5 | TerminatedBySolver | 1.561951e-02 |
| princetonlib.gms/hs096.gms | GAMS | 7 | 5 | TerminatedBySolver | 1.561954e-02 |
| princetonlib.gms/hs097.gms | GAMS | 7 | 5 | TerminatedBySolver | 3.135809e+00 |
| princetonlib.gms/hs098.gms | GAMS | 7 | 5 | TerminatedBySolver | 3.135809e+00 |
| princetonlib.gms/hs106.gms | GAMS | 9 | 7 | TerminatedBySolver | 7.049248e+03 |
| princetonlib.gms/manne.gms | GAMS | 1096 | 731 | GAMS_ms7_ss1 | -9.745717e-01 |
| princetonlib.gms/maxcut.gms | GAMS | 51 | 31 | GAMS_ms7_ss1 | -7.499947e-01 |
| princetonlib.gms/maxmineig1.gms | GAMS | 201 | 111 | GAMS_ms7_ss1 | -4.430301e+00 |
| princetonlib.gms/modell.gms | GAMS | 1832 | 340 | TerminatedBySolver | 5.742163e+03 |
| princetonlib.gms/nuffield2_trap.gms | GAMS | 10087 | 13203 | Timeout | -5.232065e+00 |
| princetonlib.gms/oet2.gms | GAMS | 4 | 1003 | TerminatedBySolver | 8.651078e-02 |
| princetonlib.gms/palmer5a.gms | GAMS | 9 | 1 | GAMS_ms7_ss2 | 2.252129e-02 |
| princetonlib.gms/palmer7a.gms | GAMS | 7 | 1 | GAMS_ms7_ss2 | 1.033486e+01 |
| princetonlib.gms/palmer7e.gms | GAMS | 9 | 1 | GAMS_ms7_ss2 | 6.352034e+00 |
| princetonlib.gms/price.gms | GAMS | 3 | 1 | GAMS_ms7_ss1 | 2.365112e-16 |
| princetonlib.gms/princeton_launch.gms | GAMS | 27 | 30 | GAMS_ms6_ss5 | 3.970923e-06 |
| princetonlib.gms/qpnboei2.gms | GAMS | 144 | 160 | GAMS_ms7_ss1 | 1.271826e+06 |
| princetonlib.gms/qpnstair.gms | GAMS | 468 | 357 | GAMS_ms7_ss1 | 5.146033e+06 |
| princetonlib.gms/s372.gms | GAMS | 10 | 13 | TerminatedBySolver | 1.339009e+04 |
| princetonlib.gms/saw_sawpath.gms | GAMS | 594 | 785 | TerminatedBySolver | 1.815730e+02 |
| princetonlib.gms/sawpath.gms | GAMS | 594 | 787 | TerminatedBySolver | 1.815730e+02 |
| princetonlib.gms/sineali.gms | GAMS | 21 | 1 | GAMS_ms7_ss2 | -1.901000e+03 |
| princetonlib.gms/ssebnln.gms | GAMS | 195 | 97 | GAMS_ms7_ss1 | 1.617060e+07 |
| princetonlib.gms/steenbre.gms | GAMS | 541 | 127 | GAMS_ms7_ss1 | 2.745917e-01 |
| princetonlib.gms/steenbrf.gms | GAMS | 469 | 109 | GAMS_ms7_ss1 | 2.826795e+02 |
| princetonlib.gms/steenbrg.gms | GAMS | 541 | 127 | GAMS_ms7_ss1 | 2.742093e-01 |
| princetonlib.gms/structure_socp_vareps.gms | GAMS | 5135 | 1904 | GAMS_ms7_ss10 | 1.003269e-02 |
| small.gms/st_e34.gms | GAMS | 7 | 5 | TerminatedBySolver | 1.561951e-02 |

## Wins (POUNCE Optimal, Ipopt not Optimal) — 100 problems

| Problem | Suite | n | m | Ipopt status | POUNCE obj |
|---------|-------|---|---|-------------|------------|
| BratuProblem@s0.100 | LargeScale | 1000 | 998 | N/A | 0.000000e+00 |
| BratuProblem@s0.500 | LargeScale | 5000 | 4998 | N/A | 0.000000e+00 |
| BratuProblem@s1.000 | LargeScale | 10000 | 9998 | N/A | 0.000000e+00 |
| ChainedRosenbrock@s0.100 | LargeScale | 200 | 0 | N/A | 1.000000e+00 |
| ChainedRosenbrock@s0.500 | LargeScale | 1000 | 0 | N/A | 1.000000e+00 |
| ChainedRosenbrock@s1.000 | LargeScale | 2000 | 0 | N/A | 1.000000e+00 |
| DECONVNE | CUTEst | 63 | 40 | Acceptable | 0.000000e+00 |
| DENSCHNDNE | CUTEst | 3 | 3 | Acceptable | 0.000000e+00 |
| DIAMON2DLS | CUTEst | 66 | 0 | Timeout | 6.325671e+02 |
| HIMMELBJ | CUTEst | 45 | 14 | Error_In_Step_Computation | N/A |
| NARX_CFy | Mittelmann | 0 | 0 | N/A | 8.589393e-03 |
| OptimalControl@s0.100 | LargeScale | 10001 | 5001 | N/A | 1.174241e-01 |
| OptimalControl@s0.500 | LargeScale | 50001 | 25001 | N/A | 1.173518e-01 |
| OptimalControl@s1.000 | LargeScale | 100001 | 50001 | N/A | 1.173428e-01 |
| POLAK6 | CUTEst | 5 | 4 | Maximum_Iterations_Exceeded | -4.400000e+01 |
| PRICE4NE | CUTEst | 2 | 2 | Acceptable | 0.000000e+00 |
| PoissonControl@s0.100 | LargeScale | 800 | 400 | N/A | 9.939498e-02 |
| PoissonControl@s0.500 | LargeScale | 20000 | 10000 | N/A | 9.946757e-02 |
| PoissonControl@s1.000 | LargeScale | 80000 | 40000 | N/A | 9.947002e-02 |
| ROBOT | CUTEst | 14 | 2 | Search_Direction_Becomes_Too_Small | 6.593299e+00 |
| SparseQP@s0.100 | LargeScale | 5000 | 5000 | N/A | -1.249817e+03 |
| SparseQP@s0.500 | LargeScale | 25000 | 25000 | N/A | -6.249817e+03 |
| SparseQP@s1.000 | LargeScale | 50000 | 50000 | N/A | -1.249982e+04 |
| TAXR13322 | CUTEst | 72 | 1261 | Acceptable | -6.449419e+04 |
| WM_CFy | Mittelmann | 0 | 0 | N/A | 1.221831e+00 |
| arki0003 | Mittelmann | 0 | 0 | N/A | 3.795201e+03 |
| arki0009 | Mittelmann | 0 | 0 | N/A | -1.775525e+04 |
| bearing_400 | Mittelmann | 0 | 0 | N/A | -1.546765e-01 |
| camshape_6400 | Mittelmann | 0 | 0 | N/A | -4.448378e+00 |
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
| globallib.gms/camshape3200.gms | GAMS | 6400 | 6401 | Timeout | 4.322955e+00 |
| globallib.gms/camshape6400.gms | GAMS | 12800 | 12801 | N/A | 4.448394e+00 |
| globallib.gms/ex7_3_4.gms | GAMS | 13 | 18 | TerminatedBySolver | 1.000000e+01 |
| globallib.gms/ex7_3_5.gms | GAMS | 14 | 16 | TerminatedBySolver | 1.206893e+00 |
| globallib.gms/ex9_2_6.gms | GAMS | 17 | 13 | TerminatedBySolver | -1.000000e+00 |
| globallib.gms/glider2.gms | GAMS | 666 | 610 | GAMS_ms5_ss1 | 1.282400e+03 |
| globallib.gms/glidera.gms | GAMS | 666 | 610 | GAMS_ms5_ss1 | 1.282400e+03 |
| globallib.gms/glidera3200.gms | GAMS | 41616 | 38410 | Timeout | 1.247987e+03 |
| globallib.gms/hs62.gms | GAMS | 4 | 2 | TerminatedBySolver | -2.627251e+04 |
| globallib.gms/methanol12800.gms.gz | GAMS | 384006 | 383998 | Timeout | 9.022293e-03 |
| globallib.gms/polygon75.gms | GAMS | 151 | 2850 | TerminatedBySolver | -7.844636e-01 |
| globallib.gms/rocket6400.gms.gz | GAMS | 38408 | 32003 | Timeout | 1.012837e+00 |
| globallib.gms/rocket800.gms.gz | GAMS | 4808 | 4003 | Timeout | 1.012837e+00 |
| globallib.gms/st_e04.gms | GAMS | 5 | 3 | TerminatedBySolver | 6.079386e+03 |
| henon120 | Mittelmann | 0 | 0 | N/A | 1.332947e+02 |
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
| princetonlib.gms/drcav2lq.gms | GAMS | 10806 | 10001 | Timeout | 1.198386e-14 |
| princetonlib.gms/himmelbj.gms | GAMS | 127 | 17 | GAMS_ms7_ss5 | -1.970127e+03 |
| princetonlib.gms/hs089.gms | GAMS | 34 | 32 | Timeout | 1.362646e+00 |
| princetonlib.gms/hs090.gms | GAMS | 35 | 32 | TerminatedBySolver | 1.362646e+00 |
| princetonlib.gms/hs092.gms | GAMS | 37 | 32 | Timeout | 1.362646e+00 |
| princetonlib.gms/pca.gms | GAMS | 10 | 5 | GAMS_ms5_ss1 | N/A |
| princetonlib.gms/putt_trap.gms | GAMS | 971 | 910 | GAMS_ms3_ss1 | N/A |
| princetonlib.gms/s214.gms | GAMS | 3 | 1 | GAMS_ms13_ss5 | 0.000000e+00 |
| princetonlib.gms/s281.gms | GAMS | 11 | 1 | TerminatedBySolver | 0.000000e+00 |
| princetonlib.gms/s348.gms | GAMS | 21 | 18 | GAMS_ms5_ss1 | 3.697422e+01 |
| princetonlib.gms/s365mod.gms | GAMS | 10 | 8 | GAMS_ms5_ss1 | 5.213990e+01 |
| princetonlib.gms/shear_midpt.gms | GAMS | 1409 | 1201 | GAMS_ms5_ss1 | 1.732559e+02 |
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

## Acceptable (not Optimal) — 4 problems

These problems converged within relaxed tolerances but not strict tolerances.

| Problem | Suite | n | m | Ipopt status | POUNCE obj | Ipopt obj |
|---------|-------|---|---|-------------|------------|-----------|
| BT8 | CUTEst | 5 | 2 | Optimal | 1.000000e+00 | 1.000000e+00 |
| DECONVU | CUTEst | 63 | 0 | Optimal | 8.107592e-14 | 4.146188e-13 |
| DJTL | CUTEst | 2 | 0 | Acceptable | -8.951545e+03 | -8.951545e+03 |
| EQC | CUTEst | 9 | 3 | Error_In_Step_Computation | -8.630052e+02 | -8.651227e+02 |

## POUNCE-Only Suite Details

These suites currently run POUNCE only — no Ipopt-side comparison is captured in their result files. Per-problem timing and iteration counts are shown so users can inspect the whole picture.

### Mittelmann

| Problem | n | m | Status | Objective | Iters | Time |
|---------|---|---|--------|-----------|-------|------|
| NARX_CFy | — | — | Optimal | 8.5894e-03 | 418 | 340.45s |
| WM_CFy | — | — | Optimal | 1.2218e+00 | 605 | 2246.08s |
| arki0003 | — | — | Optimal | 3.7952e+03 | 308 | 1.49s |
| arki0009 | — | — | Optimal | -1.7755e+04 | 362 | 10.09s |
| bearing_400 | — | — | Optimal | -1.5468e-01 | 16 | 4.66s |
| camshape_6400 | — | — | Optimal | -4.4484e+00 | 87 | 1.99s |
| clnlbeam | — | — | Optimal | 3.4488e+02 | 552 | 42.90s |
| cont5_1_l | — | — | Optimal | 2.7210e+00 | 16 | 5.75s |
| cont5_2_1_l | — | — | Optimal | 6.6275e-04 | 41 | 11.60s |
| cont5_2_2_l | — | — | Optimal | 5.2000e-04 | 48 | 14.54s |
| cont5_2_3_l | — | — | Optimal | 5.2078e-04 | 55 | 15.38s |
| cont5_2_4_l | — | — | Optimal | 6.6371e-02 | 12 | 6.27s |
| corkscrw | — | — | Optimal | 9.8096e+01 | 313 | 20.61s |
| dirichlet120 | — | — | Optimal | 3.5036e-02 | 56 | 63.59s |
| dtoc1nd | — | — | Optimal | 1.3646e+02 | 8 | 3.07s |
| dtoc2 | — | — | Optimal | 2.8633e+00 | 5 | 4.92s |
| elec_400 | — | — | Optimal | 1.0013e+02 | 126 | 40.62s |
| ex1_160 | — | — | Optimal | 6.3861e-02 | 15 | 1.74s |
| ex1_320 | — | — | Optimal | 6.5420e-02 | 8 | 5.18s |
| ex4_2_160 | — | — | Optimal | 3.6461e+00 | 21 | 2.50s |
| ex4_2_320 | — | — | Optimal | 3.6392e+00 | 17 | 10.80s |
| ex8_2_2 | — | — | Optimal | -5.5267e+02 | 67 | 582.0ms |
| ex8_2_3 | — | — | Optimal | -3.7311e+03 | 69 | 1.32s |
| gasoil_3200 | — | — | Optimal | 5.2366e-03 | 14 | 3.59s |
| henon120 | — | — | Optimal | 1.3329e+02 | 266 | 106.89s |
| lane_emden120 | — | — | Optimal | 9.3403e+00 | 84 | 98.11s |
| marine_1600 | — | — | Optimal | 1.9747e+07 | 13 | 70.44s |
| nql180 | — | — | Optimal | -9.2772e-01 | 41 | 45.55s |
| optmass | — | — | Optimal | -1.2040e-01 | 47 | 5.68s |
| pinene_3200 | — | — | Optimal | 1.9872e+01 | 10 | 5.34s |
| qcqp1000-1nc | — | — | Optimal | -1.8415e+06 | 192 | 19.63s |
| qcqp1000-2c | — | — | Optimal | 1.1129e+04 | 102 | 265.14s |
| qcqp1000-2nc | — | — | Optimal | 1.0988e+03 | 295 | 89.36s |
| qcqp1500-1c | — | — | Optimal | 1.7518e+04 | 102 | 2528.78s |
| qcqp1500-1nc | — | — | Optimal | 1.5377e+04 | 256 | 592.23s |
| qcqp500-3c | — | — | Optimal | -8.5173e+03 | 397 | 385.33s |
| qcqp500-3nc | — | — | Optimal | -1.3697e+04 | 121 | 50.17s |
| qcqp750-2c | — | — | Optimal | -1.2028e+04 | 67 | 105.61s |
| qcqp750-2nc | — | — | Optimal | -1.4452e+04 | 78 | 142.00s |
| qssp180 | — | — | Optimal | -6.6394e+00 | 33 | 39.54s |
| robot_1600 | — | — | Optimal | 9.1409e+00 | 38 | 1.82s |
| robot_a | — | — | ERROR | 2.2161e+01 | 2999 | 882.36s |
| robot_b | — | — | ERROR | 1.5177e+01 | 2999 | 820.35s |
| robot_c | — | — | ERROR | 3.3746e+01 | 1660 | 464.19s |
| rocket_12800 | — | — | Optimal | -1.0128e+00 | 24 | 16.56s |
| steering_12800 | — | — | Optimal | 5.5457e-01 | 20 | 12.43s |
| svanberg | — | — | Optimal | 8.3624e+04 | 34 | 8.77s |

POUNCE: **44/47 Optimal** in 9616.00s total

### LargeScale

| Problem | n | m | Status | Objective | Iters | Time |
|---------|---|---|--------|-----------|-------|------|
| BratuProblem@s0.100 | 1,000 | 998 | Optimal | 0.0000e+00 | 2 | 2.9ms |
| BratuProblem@s0.500 | 5,000 | 4,998 | Optimal | 0.0000e+00 | 2 | 12.5ms |
| BratuProblem@s1.000 | 10,000 | 9,998 | Optimal | 0.0000e+00 | 1 | 24.1ms |
| ChainedRosenbrock@s0.100 | 200 | — | Optimal | 1.0000e+00 | 146 | 34.0ms |
| ChainedRosenbrock@s0.500 | 1,000 | — | Optimal | 1.0000e+00 | 765 | 555.1ms |
| ChainedRosenbrock@s1.000 | 2,000 | — | Optimal | 1.0000e+00 | 1484 | 1.89s |
| OptimalControl@s0.100 | 10,001 | 5,001 | Optimal | 1.1742e-01 | 1 | 33.9ms |
| OptimalControl@s0.500 | 50,001 | 25,001 | Optimal | 1.1735e-01 | 1 | 238.9ms |
| OptimalControl@s1.000 | 100,001 | 50,001 | Optimal | 1.1734e-01 | 1 | 275.8ms |
| PoissonControl@s0.100 | 800 | 400 | Optimal | 9.9395e-02 | 1 | 2.9ms |
| PoissonControl@s0.500 | 20,000 | 10,000 | Optimal | 9.9468e-02 | 1 | 129.9ms |
| PoissonControl@s1.000 | 80,000 | 40,000 | Optimal | 9.9470e-02 | 1 | 394.6ms |
| SparseQP@s0.100 | 5,000 | 5,000 | Optimal | -1.2498e+03 | 7 | 84.7ms |
| SparseQP@s0.500 | 25,000 | 25,000 | Optimal | -6.2498e+03 | 7 | 569.0ms |
| SparseQP@s1.000 | 50,000 | 50,000 | Optimal | -1.2500e+04 | 7 | 884.6ms |

POUNCE: **15/15 Optimal** in 5.13s total

---
*Generated by benchmark_report.py*