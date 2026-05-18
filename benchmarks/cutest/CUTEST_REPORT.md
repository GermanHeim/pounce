# CUTEst Benchmark Report

Comparison of ripopt vs Ipopt (C++) on the CUTEst test set.

## Executive Summary

- **Total problems**: 727
- **ripopt solved**: 560/727 (77.0%)
- **Ipopt solved**: 562/727 (77.3%)
- **Both solved**: 551/727
- **Matching solutions** (rel obj diff < 1e-4): 520/551

## Accuracy Statistics (where both solve)

Relative difference = |r_obj - i_obj| / max(|r_obj|, |i_obj|, 1.0).  
The 1.0 floor prevents near-zero objectives from inflating the metric.

**Matching solutions** (520 problems, rel diff < 1e-4):

| Metric | Rel Diff |
|--------|----------|
| Mean   | 1.99e-07 |
| Median | 3.95e-16 |
| Max    | 2.92e-05 |

**All both-solved** (551 problems, including 31 mismatches):

| Metric | Rel Diff |
|--------|----------|
| Mean   | 2.96e-02 |
| Median | 5.72e-16 |
| Max    | 1.00e+00 |

## Category Breakdown

| Category | Total | ripopt | Ipopt | Both | Match |
|----------|-------|--------|-------|------|-------|
| constrained | 493 | 338 | 343 | 333 | 325 |
| unconstrained | 234 | 222 | 219 | 218 | 195 |

## Detailed Results

| Problem | n | m | ripopt | Ipopt | Obj Diff | r_iter | i_iter | r_time | i_time | Speedup | Status |
|---------|---|---|--------|-------|----------|--------|--------|--------|--------|---------|--------|
| 3PK | 30 | 0 | Solve_Succee | Solve_Succee | 3.59e-14 | 4 | 9 | 111us | 8.6ms | 77.3x | PASS |
| ACOPP14 | 38 | 68 | Solve_Succee | Solve_Succee | 7.99e-15 | 9 | 9 | 5.2ms | 3.9ms | 0.8x | PASS |
| ACOPP30 | 72 | 142 | Solve_Succee | Solve_Succee | 9.38e-14 | 13 | 13 | 12.2ms | 6.4ms | 0.5x | PASS |
| ACOPR14 | 38 | 82 | Solve_Succee | Solve_Succee | 2.81e-15 | 13 | 13 | 6.1ms | 5.1ms | 0.8x | PASS |
| ACOPR30 | 72 | 172 | Solved_To_Ac | Solve_Succee | 3.21e-02 | 72 | 221 | 56.6ms | 114.7ms | 2.0x | MISMATCH |
| AIRCRFTA | 8 | 5 | Solve_Succee | Solve_Succee | 0.00e+00 | 3 | 3 | 626us | 964us | 1.5x | PASS |
| AIRCRFTB | 8 | 0 | Solve_Succee | Solve_Succee | 4.07e-18 | 12 | 15 | 42us | 2.8ms | 66.8x | PASS |
| AIRPORT | 84 | 42 | Solve_Succee | Solve_Succee | 0.00e+00 | 13 | 13 | 7.6ms | 5.7ms | 0.8x | PASS |
| AKIVA | 2 | 0 | Solve_Succee | Solve_Succee | 5.76e-16 | 6 | 6 | 54us | 1.3ms | 23.2x | PASS |
| ALLINIT | 4 | 0 | Solve_Succee | Solve_Succee | 0.00e+00 | 7 | 20 | 28us | 3.7ms | 130.1x | PASS |
| ALLINITA | 4 | 4 | Solve_Succee | Solve_Succee | 1.58e-13 | 23 | 12 | 2.9ms | 2.6ms | 0.9x | PASS |
| ALLINITC | 4 | 1 | Solve_Succee | Solve_Succee | 3.17e-12 | 19 | 17 | 2.8ms | 3.3ms | 1.2x | PASS |
| ALLINITU | 4 | 0 | Solve_Succee | Solve_Succee | 3.09e-16 | 16 | 14 | 40us | 2.4ms | 61.1x | PASS |
| ALSOTAME | 2 | 1 | Solve_Succee | Solve_Succee | 0.00e+00 | 8 | 8 | 1.6ms | 1.9ms | 1.2x | PASS |
| ANTWERP | 27 | 10 | Solve_Succee | Solve_Succee | 5.35e-07 | 113 | 108 | 24.5ms | 23.5ms | 1.0x | PASS |
| ARGAUSS | 3 | 15 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 17us | 625us | 35.9x | BOTH_FAIL |
| AVGASA | 8 | 10 | Solve_Succee | Solve_Succee | 0.00e+00 | 9 | 9 | 2.2ms | 2.2ms | 1.0x | PASS |
| AVGASB | 8 | 10 | Solve_Succee | Solve_Succee | 0.00e+00 | 11 | 11 | 2.3ms | 2.5ms | 1.1x | PASS |
| AVION2 | 49 | 15 | Maximum_Iter | Maximum_Iter | N/A | 2999 | 3000 | 727.9ms | 675.6ms | 0.9x | BOTH_FAIL |
| BA-L1 | 57 | 12 | Solve_Succee | Solve_Succee | 0.00e+00 | 5 | 6 | 1.3ms | 1.9ms | 1.5x | PASS |
| BA-L1LS | 57 | 0 | Solve_Succee | Solve_Succee | 4.38e-15 | 6 | 10 | 379us | 2.6ms | 6.9x | PASS |
| BA-L1SP | 57 | 12 | Error_In_Ste | Solve_Succee | N/A | 42 | 5 | 29.8ms | 2.9ms | 0.1x | ripopt_FAIL |
| BA-L1SPLS | 57 | 0 | Solve_Succee | Solve_Succee | 6.48e-17 | 6 | 9 | 1.7ms | 4.8ms | 2.8x | PASS |
| BARD | 3 | 0 | Solve_Succee | Solve_Succee | 6.59e-17 | 8 | 8 | 36us | 2.0ms | 55.5x | PASS |
| BARDNE | 3 | 15 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 17us | 296us | 17.2x | BOTH_FAIL |
| BATCH | 48 | 73 | Solve_Succee | Solve_Succee | 4.17e-09 | 41 | 29 | 15.2ms | 9.2ms | 0.6x | PASS |
| BEALE | 2 | 0 | Solve_Succee | Solve_Succee | 5.11e-16 | 7 | 8 | 27us | 2.7ms | 99.4x | PASS |
| BEALENE | 2 | 3 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 25us | 303us | 12.2x | BOTH_FAIL |
| BENNETT5 | 3 | 154 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 43us | 338us | 7.9x | BOTH_FAIL |
| BENNETT5LS | 3 | 0 | Solve_Succee | Solve_Succee | 1.94e-05 | 9 | 21 | 398us | 5.9ms | 14.9x | PASS |
| BIGGS3 | 6 | 0 | Solve_Succee | Solve_Succee | 1.44e-18 | 8 | 9 | 81us | 3.1ms | 38.3x | PASS |
| BIGGS5 | 6 | 0 | Solve_Succee | Solve_Succee | 6.69e-20 | 20 | 20 | 142us | 5.6ms | 39.5x | PASS |
| BIGGS6 | 6 | 0 | Solve_Succee | Solve_Succee | 1.60e-20 | 85 | 79 | 394us | 16.1ms | 41.0x | PASS |
| BIGGS6NE | 6 | 13 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 29us | 303us | 10.6x | BOTH_FAIL |
| BIGGSC4 | 4 | 7 | Solve_Succee | Solve_Succee | 0.00e+00 | 17 | 17 | 5.1ms | 4.6ms | 0.9x | PASS |
| BLEACHNG | 17 | 0 | Solved_To_Ac | Timeout | N/A | 12 | 0 | 14.08s | 60.00s | 4.3x | ipopt_FAIL |
| BOOTH | 2 | 2 | Solve_Succee | Solve_Succee | 0.00e+00 | 1 | 1 | 871us | 687us | 0.8x | PASS |
| BOX2 | 3 | 0 | Solve_Succee | Solve_Succee | 5.93e-22 | 7 | 8 | 23us | 2.4ms | 104.2x | PASS |
| BOX3 | 3 | 0 | Solve_Succee | Solve_Succee | 3.72e-20 | 8 | 9 | 25us | 2.6ms | 101.0x | PASS |
| BOX3NE | 3 | 10 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 27us | 299us | 11.2x | BOTH_FAIL |
| BOXBOD | 2 | 6 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 27us | 209us | 7.7x | BOTH_FAIL |
| BOXBODLS | 2 | 0 | Solve_Succee | Solve_Succee | 8.80e-01 | 26 | 13 | 62us | 3.8ms | 61.7x | MISMATCH |
| BQP1VAR | 1 | 0 | Solve_Succee | Solve_Succee | 1.10e-07 | 20 | 5 | 37us | 2.1ms | 58.0x | PASS |
| BQPGABIM | 50 | 0 | Solve_Succee | Solve_Succee | 9.96e-07 | 32 | 12 | 1.1ms | 4.2ms | 3.7x | PASS |
| BQPGASIM | 50 | 0 | Solve_Succee | Solve_Succee | 9.61e-07 | 32 | 12 | 1.2ms | 3.6ms | 3.2x | PASS |
| BRANIN | 2 | 0 | Solve_Succee | Solve_Succee | 1.28e-15 | 7 | 7 | 28us | 2.6ms | 91.2x | PASS |
| BRKMCC | 2 | 0 | Solve_Succee | Solve_Succee | 0.00e+00 | 3 | 3 | 22us | 1.2ms | 51.8x | PASS |
| BROWNBS | 2 | 0 | Solve_Succee | Solve_Succee | 0.00e+00 | 9 | 7 | 28us | 1.4ms | 49.1x | PASS |
| BROWNBSNE | 2 | 3 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 24us | 298us | 12.4x | BOTH_FAIL |
| BROWNDEN | 4 | 0 | Solve_Succee | Solve_Succee | 1.70e-16 | 8 | 8 | 59us | 2.1ms | 35.9x | PASS |
| BROWNDENE | 4 | 20 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 32us | 304us | 9.5x | BOTH_FAIL |
| BT1 | 2 | 1 | Solve_Succee | Solve_Succee | 1.21e-09 | 7 | 7 | 1.9ms | 2.3ms | 1.2x | PASS |
| BT10 | 2 | 2 | Solve_Succee | Solve_Succee | 2.79e-09 | 7 | 6 | 1.7ms | 2.0ms | 1.1x | PASS |
| BT11 | 5 | 3 | Solve_Succee | Solve_Succee | 2.22e-16 | 8 | 8 | 1.9ms | 2.2ms | 1.2x | PASS |
| BT12 | 5 | 3 | Solve_Succee | Solve_Succee | 0.00e+00 | 4 | 4 | 1.5ms | 1.1ms | 0.7x | PASS |
| BT13 | 5 | 1 | Solve_Succee | Solve_Succee | 0.00e+00 | 537 | 24 | 71.8ms | 4.3ms | 0.1x | PASS |
| BT2 | 3 | 1 | Solve_Succee | Solve_Succee | 0.00e+00 | 12 | 12 | 1.9ms | 2.7ms | 1.4x | PASS |
| BT3 | 5 | 3 | Solve_Succee | Solve_Succee | 0.00e+00 | 1 | 1 | 508us | 653us | 1.3x | PASS |
| BT4 | 3 | 2 | Solve_Succee | Solve_Succee | 1.20e-16 | 9 | 9 | 2.3ms | 2.6ms | 1.1x | PASS |
| BT5 | 3 | 2 | Solve_Succee | Solve_Succee | 0.00e+00 | 7 | 7 | 1.7ms | 2.1ms | 1.2x | PASS |
| BT6 | 5 | 2 | Solve_Succee | Solve_Succee | 1.09e-14 | 13 | 13 | 2.5ms | 2.9ms | 1.1x | PASS |
| BT7 | 5 | 3 | Solve_Succee | Solve_Succee | 0.00e+00 | 16 | 16 | 3.2ms | 4.3ms | 1.4x | PASS |
| BT8 | 5 | 2 | Solve_Succee | Solve_Succee | 0.00e+00 | 14 | 14 | 2.6ms | 3.2ms | 1.3x | PASS |
| BT9 | 4 | 2 | Solve_Succee | Solve_Succee | 0.00e+00 | 13 | 13 | 2.4ms | 3.1ms | 1.3x | PASS |
| BURKEHAN | 1 | 1 | Infeasible_P | Infeasible_P | N/A | 3 | 11 | 4.1ms | 4.4ms | 1.1x | BOTH_FAIL |
| BYRDSPHR | 3 | 2 | Infeasible_P | Solve_Succee | N/A | 21 | 12 | 8.2ms | 3.0ms | 0.4x | ripopt_FAIL |
| CAMEL6 | 2 | 0 | Solve_Succee | Solve_Succee | 7.91e-01 | 19 | 8 | 47us | 2.5ms | 52.7x | MISMATCH |
| CANTILVR | 5 | 1 | Solve_Succee | Solve_Succee | 2.77e-09 | 11 | 11 | 3.5ms | 3.7ms | 1.0x | PASS |
| CB2 | 3 | 3 | Solve_Succee | Solve_Succee | 0.00e+00 | 8 | 8 | 2.7ms | 2.6ms | 1.0x | PASS |
| CB3 | 3 | 3 | Solve_Succee | Solve_Succee | 2.22e-16 | 8 | 8 | 2.2ms | 2.3ms | 1.0x | PASS |
| CERI651A | 7 | 61 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 21us | 240us | 11.7x | BOTH_FAIL |
| CERI651ALS | 7 | 0 | Solve_Succee | Solve_Succee | 6.82e-08 | 191 | 95 | 2.7ms | 16.1ms | 6.0x | PASS |
| CERI651B | 7 | 66 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 23us | 370us | 15.8x | BOTH_FAIL |
| CERI651BLS | 7 | 0 | Solve_Succee | Solve_Succee | 2.65e-09 | 49 | 56 | 812us | 8.9ms | 10.9x | PASS |
| CERI651C | 7 | 56 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 21us | 231us | 11.1x | BOTH_FAIL |
| CERI651CLS | 7 | 0 | Solve_Succee | Solve_Succee | 1.86e-09 | 50 | 53 | 672us | 7.9ms | 11.7x | PASS |
| CERI651D | 7 | 67 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 22us | 238us | 10.6x | BOTH_FAIL |
| CERI651DLS | 7 | 0 | Solve_Succee | Solve_Succee | 2.54e-08 | 25 | 60 | 434us | 10.2ms | 23.5x | PASS |
| CERI651E | 7 | 64 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 21us | 233us | 10.9x | BOTH_FAIL |
| CERI651ELS | 7 | 0 | Solve_Succee | Solve_Succee | 3.50e-10 | 45 | 45 | 742us | 6.7ms | 9.0x | PASS |
| CHACONN1 | 3 | 3 | Solve_Succee | Solve_Succee | 0.00e+00 | 6 | 6 | 1.3ms | 1.6ms | 1.2x | PASS |
| CHACONN2 | 3 | 3 | Solve_Succee | Solve_Succee | 0.00e+00 | 6 | 6 | 1.2ms | 1.6ms | 1.3x | PASS |
| CHWIRUT1 | 3 | 214 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 24us | 263us | 11.0x | BOTH_FAIL |
| CHWIRUT1LS | 3 | 0 | Solve_Succee | Solve_Succee | 5.72e-16 | 6 | 6 | 146us | 1.6ms | 11.1x | PASS |
| CHWIRUT2 | 3 | 54 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 18us | 221us | 12.0x | BOTH_FAIL |
| CHWIRUT2LS | 3 | 0 | Solve_Succee | Solve_Succee | 2.22e-16 | 6 | 6 | 48us | 1.5ms | 31.6x | PASS |
| CLIFF | 2 | 0 | Solve_Succee | Solve_Succee | 7.45e-03 | 26 | 23 | 24us | 3.2ms | 130.5x | MISMATCH |
| CLUSTER | 2 | 2 | Solve_Succee | Solve_Succee | 0.00e+00 | 9 | 9 | 1.2ms | 1.9ms | 1.6x | PASS |
| CLUSTERLS | 2 | 0 | Solve_Succee | Solve_Succee | 1.61e-17 | 16 | 17 | 24us | 2.7ms | 111.5x | PASS |
| CONCON | 15 | 11 | Solve_Succee | Solve_Succee | 0.00e+00 | 7 | 7 | 1.8ms | 1.9ms | 1.1x | PASS |
| CONGIGMZ | 3 | 5 | Solve_Succee | Solve_Succee | 0.00e+00 | 25 | 20 | 3.8ms | 3.8ms | 1.0x | PASS |
| COOLHANS | 9 | 9 | Solve_Succee | Solve_Succee | 0.00e+00 | 9 | 9 | 1.3ms | 1.7ms | 1.3x | PASS |
| COOLHANSLS | 9 | 0 | Solve_Succee | Solve_Succee | 2.92e-05 | 43 | 25 | 214us | 4.1ms | 19.3x | PASS |
| CORE1 | 65 | 59 | Solve_Succee | Solve_Succee | 4.68e-16 | 31 | 33 | 8.7ms | 8.1ms | 0.9x | PASS |
| CRESC100 | 6 | 200 | Infeasible_P | Infeasible_P | N/A | 1535 | 155 | 5.55s | 116.9ms | 0.0x | BOTH_FAIL |
| CRESC132 | 6 | 2654 | Infeasible_P | Timeout | N/A | 54 | 0 | 51.55s | 60.00s | 1.2x | BOTH_FAIL |
| CRESC4 | 6 | 8 | Solve_Succee | Solve_Succee | 5.28e-09 | 428 | 64 | 83.0ms | 13.1ms | 0.2x | PASS |
| CRESC50 | 6 | 100 | Infeasible_P | Solve_Succee | N/A | 1527 | 194 | 2.51s | 83.5ms | 0.0x | ripopt_FAIL |
| CSFI1 | 5 | 4 | Solve_Succee | Solve_Succee | 1.45e-16 | 11 | 11 | 2.3ms | 2.5ms | 1.1x | PASS |
| CSFI2 | 5 | 4 | Solve_Succee | Solve_Succee | 2.59e-11 | 14 | 14 | 2.6ms | 3.1ms | 1.2x | PASS |
| CUBE | 2 | 0 | Solve_Succee | Solve_Succee | 4.04e-14 | 26 | 27 | 25us | 4.3ms | 169.3x | PASS |
| CUBENE | 2 | 2 | Solve_Succee | Solve_Succee | 0.00e+00 | 1 | 1 | 623us | 653us | 1.0x | PASS |
| DALLASS | 46 | 31 | Solve_Succee | Solve_Succee | 1.99e-05 | 28 | 22 | 7.0ms | 5.1ms | 0.7x | PASS |
| DANIWOOD | 2 | 6 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 15us | 220us | 14.3x | BOTH_FAIL |
| DANIWOODLS | 2 | 0 | Solve_Succee | Solve_Succee | 3.47e-18 | 12 | 10 | 25us | 2.0ms | 78.2x | PASS |
| DANWOOD | 2 | 6 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 17us | 218us | 13.0x | BOTH_FAIL |
| DANWOODLS | 2 | 0 | Solve_Succee | Solve_Succee | 1.86e-16 | 8 | 11 | 23us | 2.1ms | 93.4x | PASS |
| DECONVB | 63 | 0 | Solve_Succee | Maximum_Iter | N/A | 1672 | 3000 | 145.2ms | 743.8ms | 5.1x | ipopt_FAIL |
| DECONVBNE | 63 | 40 | Solve_Succee | Solve_Succee | 0.00e+00 | 161 | 505 | 83.8ms | 166.2ms | 2.0x | PASS |
| DECONVC | 63 | 1 | Solve_Succee | Solve_Succee | 3.27e-10 | 77 | 31 | 18.7ms | 8.9ms | 0.5x | PASS |
| DECONVNE | 63 | 40 | Solve_Succee | Solved_To_Ac | 0.00e+00 | 1 | 26 | 1.2ms | 24.7ms | 20.3x | PASS |
| DECONVU | 63 | 0 | Solve_Succee | Solve_Succee | 1.47e-09 | 50 | 333 | 4.8ms | 88.1ms | 18.2x | PASS |
| DEGENLPA | 20 | 15 | Solve_Succee | Solve_Succee | 1.40e-11 | 18 | 18 | 3.6ms | 4.3ms | 1.2x | PASS |
| DEGENLPB | 20 | 15 | Solve_Succee | Solve_Succee | 1.65e-11 | 19 | 19 | 3.6ms | 4.0ms | 1.1x | PASS |
| DEMBO7 | 16 | 20 | Solve_Succee | Solve_Succee | 3.62e-09 | 59 | 45 | 10.4ms | 9.0ms | 0.9x | PASS |
| DEMYMALO | 3 | 3 | Solve_Succee | Solve_Succee | 1.48e-16 | 9 | 9 | 2.3ms | 2.1ms | 0.9x | PASS |
| DENSCHNA | 2 | 0 | Solve_Succee | Solve_Succee | 5.88e-39 | 6 | 6 | 15us | 1.2ms | 77.7x | PASS |
| DENSCHNB | 2 | 0 | Solve_Succee | Solve_Succee | 9.99e-16 | 6 | 7 | 16us | 1.5ms | 95.9x | PASS |
| DENSCHNBNE | 2 | 3 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 15us | 224us | 14.9x | BOTH_FAIL |
| DENSCHNC | 2 | 0 | Solve_Succee | Solve_Succee | 0.00e+00 | 10 | 10 | 20us | 1.6ms | 84.2x | PASS |
| DENSCHNCNE | 2 | 2 | Solve_Succee | Solve_Succee | 0.00e+00 | 7 | 7 | 1.3ms | 1.4ms | 1.1x | PASS |
| DENSCHND | 3 | 0 | Solve_Succee | Solve_Succee | 1.94e-04 | 25 | 26 | 34us | 3.9ms | 113.8x | MISMATCH |
| DENSCHNDNE | 3 | 3 | Solve_Succee | Solved_To_Ac | 0.00e+00 | 23 | 22 | 2.5ms | 3.6ms | 1.5x | PASS |
| DENSCHNE | 3 | 0 | Solve_Succee | Solve_Succee | 1.86e-17 | 10 | 14 | 19us | 2.8ms | 145.1x | PASS |
| DENSCHNENE | 3 | 3 | Infeasible_P | Infeasible_P | N/A | 14 | 10 | 5.6ms | 2.5ms | 0.4x | BOTH_FAIL |
| DENSCHNF | 2 | 0 | Solve_Succee | Solve_Succee | 0.00e+00 | 6 | 6 | 16us | 1.2ms | 77.0x | PASS |
| DENSCHNFNE | 2 | 2 | Solve_Succee | Solve_Succee | 0.00e+00 | 5 | 5 | 901us | 1.2ms | 1.3x | PASS |
| DEVGLA1 | 4 | 0 | Solve_Succee | Solve_Succee | 3.32e-15 | 16 | 23 | 103us | 4.0ms | 39.3x | PASS |
| DEVGLA1B | 4 | 0 | Solve_Succee | Solve_Succee | 1.19e-18 | 19 | 20 | 127us | 7.7ms | 60.7x | PASS |
| DEVGLA1NE | 4 | 24 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 32us | 220us | 6.9x | BOTH_FAIL |
| DEVGLA2 | 5 | 0 | Solve_Succee | Solve_Succee | 3.64e-15 | 14 | 13 | 192us | 3.4ms | 18.0x | PASS |
| DEVGLA2B | 5 | 0 | Solve_Succee | Solve_Succee | 3.49e-08 | 50 | 24 | 295us | 7.2ms | 24.3x | PASS |
| DEVGLA2NE | 5 | 16 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 31us | 300us | 9.8x | BOTH_FAIL |
| DGOSPEC | 3 | 0 | Solve_Succee | Solve_Succee | 2.02e-10 | 32 | 27 | 60us | 7.6ms | 126.9x | PASS |
| DIAMON2D | 66 | 4643 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 912us | 7.4ms | 8.2x | BOTH_FAIL |
| DIAMON2DLS | 66 | 0 | Timeout | Timeout | N/A | 0 | 0 | 60.00s | 60.00s | 1.0x | BOTH_FAIL |
| DIAMON3D | 99 | 4643 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 1.3ms | 15.0ms | 11.9x | BOTH_FAIL |
| DIAMON3DLS | 99 | 0 | Timeout | Timeout | N/A | 0 | 0 | 60.00s | 60.00s | 1.0x | BOTH_FAIL |
| DIPIGRI | 7 | 4 | Solve_Succee | Solve_Succee | 0.00e+00 | 9 | 9 | 3.0ms | 3.3ms | 1.1x | PASS |
| DISC2 | 29 | 23 | Solve_Succee | Solve_Succee | 0.00e+00 | 24 | 24 | 8.7ms | 7.5ms | 0.9x | PASS |
| DISCS | 36 | 66 | Solve_Succee | Solve_Succee | 4.33e-11 | 128 | 184 | 61.5ms | 69.7ms | 1.1x | PASS |
| DIXCHLNG | 10 | 5 | Solve_Succee | Solve_Succee | 0.00e+00 | 10 | 10 | 2.0ms | 2.5ms | 1.2x | PASS |
| DJTL | 2 | 0 | Solve_Succee | Solved_To_Ac | 1.63e-15 | 1827 | 1538 | 2.8ms | 169.0ms | 61.3x | PASS |
| DMN15102 | 66 | 4643 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 977us | 8.1ms | 8.3x | BOTH_FAIL |
| DMN15102LS | 66 | 0 | Timeout | Solve_Succee | N/A | 0 | 1189 | 60.00s | 37.12s | 0.6x | ripopt_FAIL |
| DMN15103 | 99 | 4643 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 1.3ms | 15.0ms | 11.8x | BOTH_FAIL |
| DMN15103LS | 99 | 0 | Search_Direc | Timeout | N/A | 620 | 0 | 44.29s | 60.00s | 1.4x | BOTH_FAIL |
| DMN15332 | 66 | 4643 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 1.0ms | 8.4ms | 8.1x | BOTH_FAIL |
| DMN15332LS | 66 | 0 | Timeout | Timeout | N/A | 0 | 0 | 60.00s | 60.00s | 1.0x | BOTH_FAIL |
| DMN15333 | 99 | 4643 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 1.5ms | 13.8ms | 9.4x | BOTH_FAIL |
| DMN15333LS | 99 | 0 | Timeout | Timeout | N/A | 0 | 0 | 60.00s | 60.00s | 1.0x | BOTH_FAIL |
| DMN37142 | 66 | 4643 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 1.1ms | 7.2ms | 6.7x | BOTH_FAIL |
| DMN37142LS | 66 | 0 | Error_In_Ste | Timeout | N/A | 563 | 0 | 17.39s | 60.00s | 3.5x | BOTH_FAIL |
| DMN37143 | 99 | 4643 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 1.2ms | 15.2ms | 12.5x | BOTH_FAIL |
| DMN37143LS | 99 | 0 | Timeout | Timeout | N/A | 0 | 0 | 60.00s | 60.00s | 1.0x | BOTH_FAIL |
| DNIEPER | 61 | 24 | Solve_Succee | Solve_Succee | 1.91e-12 | 30 | 23 | 9.3ms | 6.5ms | 0.7x | PASS |
| DUAL1 | 85 | 1 | Solve_Succee | Solve_Succee | 4.37e-16 | 15 | 15 | 13.5ms | 7.6ms | 0.6x | PASS |
| DUAL2 | 96 | 1 | Solve_Succee | Solve_Succee | 1.11e-16 | 12 | 12 | 11.2ms | 6.9ms | 0.6x | PASS |
| DUAL4 | 75 | 1 | Solve_Succee | Solve_Succee | 5.55e-16 | 12 | 12 | 7.8ms | 5.5ms | 0.7x | PASS |
| DUALC1 | 9 | 215 | Solve_Succee | Solve_Succee | 2.96e-16 | 18 | 18 | 23.6ms | 12.3ms | 0.5x | PASS |
| DUALC2 | 7 | 229 | Solve_Succee | Solve_Succee | 1.28e-16 | 12 | 12 | 13.9ms | 8.8ms | 0.6x | PASS |
| DUALC5 | 8 | 278 | Solve_Succee | Solve_Succee | 5.32e-16 | 11 | 11 | 18.2ms | 9.7ms | 0.5x | PASS |
| DUALC8 | 8 | 503 | Solve_Succee | Solve_Succee | 1.99e-16 | 13 | 13 | 30.6ms | 15.2ms | 0.5x | PASS |
| ECKERLE4 | 3 | 35 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 27us | 285us | 10.4x | BOTH_FAIL |
| ECKERLE4LS | 3 | 0 | Solve_Succee | Solve_Succee | 6.98e-01 | 17 | 36 | 146us | 8.7ms | 59.5x | MISMATCH |
| EG1 | 3 | 0 | Solve_Succee | Solve_Succee | 2.07e-01 | 20 | 8 | 54us | 2.9ms | 54.3x | MISMATCH |
| EGGCRATE | 2 | 0 | Solve_Succee | Solve_Succee | 5.62e-16 | 6 | 5 | 17us | 1.9ms | 109.5x | PASS |
| EGGCRATEB | 2 | 0 | Solve_Succee | Solve_Succee | 5.62e-16 | 8 | 6 | 31us | 2.1ms | 65.7x | PASS |
| EGGCRATENE | 2 | 4 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 27us | 300us | 11.2x | BOTH_FAIL |
| ELATTAR | 7 | 102 | Solve_Succee | Solve_Succee | 9.98e-01 | 216 | 81 | 125.6ms | 34.0ms | 0.3x | MISMATCH |
| ELATVIDU | 2 | 0 | Solve_Succee | Solve_Succee | 0.00e+00 | 11 | 11 | 30us | 2.6ms | 85.4x | PASS |
| ELATVIDUB | 2 | 0 | Solve_Succee | Solve_Succee | 1.30e-16 | 11 | 11 | 33us | 2.9ms | 86.7x | PASS |
| ELATVIDUNE | 2 | 3 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 27us | 307us | 11.4x | BOTH_FAIL |
| ENGVAL2 | 3 | 0 | Solve_Succee | Solve_Succee | 5.70e-16 | 25 | 21 | 57us | 5.0ms | 87.2x | PASS |
| ENGVAL2NE | 3 | 5 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 17us | 296us | 17.1x | BOTH_FAIL |
| ENSO | 9 | 168 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 44us | 377us | 8.6x | BOTH_FAIL |
| ENSOLS | 9 | 0 | Solve_Succee | Solve_Succee | 7.21e-16 | 8 | 7 | 1.1ms | 3.1ms | 2.7x | PASS |
| EQC | 9 | 3 | Search_Direc | Error_In_Ste | N/A | 27 | 15 | 7.0ms | 6.2ms | 0.9x | BOTH_FAIL |
| ERRINBAR | 18 | 9 | Solve_Succee | Solve_Succee | 2.46e-12 | 43 | 37 | 11.2ms | 9.5ms | 0.8x | PASS |
| EXP2 | 2 | 0 | Solve_Succee | Solve_Succee | 0.00e+00 | 7 | 7 | 35us | 1.8ms | 50.3x | PASS |
| EXP2B | 2 | 0 | Solve_Succee | Solve_Succee | 3.05e-15 | 7 | 7 | 23us | 2.3ms | 98.0x | PASS |
| EXP2NE | 2 | 10 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 25us | 326us | 12.9x | BOTH_FAIL |
| EXPFIT | 2 | 0 | Solve_Succee | Solve_Succee | 8.60e-16 | 10 | 8 | 45us | 2.3ms | 52.4x | PASS |
| EXPFITA | 5 | 22 | Solve_Succee | Solve_Succee | 3.90e-18 | 13 | 13 | 4.7ms | 4.3ms | 0.9x | PASS |
| EXPFITB | 5 | 102 | Solve_Succee | Solve_Succee | 7.81e-17 | 16 | 16 | 10.3ms | 7.3ms | 0.7x | PASS |
| EXPFITC | 5 | 502 | Solve_Succee | Solve_Succee | 1.17e-07 | 19 | 18 | 33.0ms | 16.8ms | 0.5x | PASS |
| EXPFITNE | 2 | 10 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 25us | 276us | 10.9x | BOTH_FAIL |
| EXTRASIM | 2 | 1 | Solve_Succee | Solve_Succee | 0.00e+00 | 3 | 3 | 1.5ms | 1.5ms | 1.0x | PASS |
| FBRAIN | 2 | 2211 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 61us | 428us | 7.0x | BOTH_FAIL |
| FBRAIN2 | 4 | 2211 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 67us | 572us | 8.5x | BOTH_FAIL |
| FBRAIN2LS | 4 | 0 | Solve_Succee | Solve_Succee | 1.01e-07 | 24 | 10 | 14.8ms | 8.8ms | 0.6x | PASS |
| FBRAIN2NE | 4 | 2211 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 67us | 590us | 8.8x | BOTH_FAIL |
| FBRAIN3 | 6 | 2211 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 88us | 840us | 9.6x | BOTH_FAIL |
| FBRAIN3LS | 6 | 0 | Maximum_Iter | Maximum_Iter | N/A | 3000 | 3000 | 3.00s | 3.59s | 1.2x | BOTH_FAIL |
| FBRAINLS | 2 | 0 | Solve_Succee | Solve_Succee | 2.11e-15 | 8 | 7 | 3.2ms | 3.9ms | 1.2x | PASS |
| FBRAINNE | 2 | 2211 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 63us | 420us | 6.6x | BOTH_FAIL |
| FCCU | 19 | 8 | Solve_Succee | Solve_Succee | 0.00e+00 | 9 | 9 | 2.7ms | 2.8ms | 1.0x | PASS |
| FEEDLOC | 90 | 259 | Solve_Succee | Solve_Succee | 7.50e-13 | 23 | 23 | 50.3ms | 14.6ms | 0.3x | PASS |
| FLETCHER | 4 | 4 | Solve_Succee | Solve_Succee | 5.40e-11 | 28 | 28 | 6.0ms | 6.5ms | 1.1x | PASS |
| FLT | 2 | 2 | Solve_Succee | Solve_Succee | 0.00e+00 | 25 | 5 | 4.8ms | 2.1ms | 0.4x | PASS |
| GAUSS1 | 8 | 250 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 49us | 294us | 6.1x | BOTH_FAIL |
| GAUSS1LS | 8 | 0 | Solve_Succee | Solve_Succee | 0.00e+00 | 5 | 5 | 444us | 1.5ms | 3.4x | PASS |
| GAUSS2 | 8 | 250 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 28us | 259us | 9.3x | BOTH_FAIL |
| GAUSS2LS | 8 | 0 | Solve_Succee | Solve_Succee | 0.00e+00 | 5 | 5 | 351us | 1.8ms | 5.2x | PASS |
| GAUSS3 | 8 | 250 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 49us | 370us | 7.6x | BOTH_FAIL |
| GAUSS3LS | 8 | 0 | Solve_Succee | Solve_Succee | 1.83e-15 | 14 | 11 | 1.4ms | 4.1ms | 3.0x | PASS |
| GAUSSIAN | 3 | 0 | Solve_Succee | Solve_Succee | 0.00e+00 | 2 | 2 | 29us | 983us | 34.0x | PASS |
| GBRAIN | 2 | 2200 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 59us | 441us | 7.5x | BOTH_FAIL |
| GBRAINLS | 2 | 0 | Solve_Succee | Solve_Succee | 0.00e+00 | 6 | 6 | 2.2ms | 3.2ms | 1.5x | PASS |
| GENHS28 | 10 | 8 | Solve_Succee | Solve_Succee | 4.44e-16 | 1 | 1 | 748us | 916us | 1.2x | PASS |
| GIGOMEZ1 | 3 | 3 | Solve_Succee | Solve_Succee | 3.08e-09 | 13 | 13 | 3.1ms | 3.8ms | 1.2x | PASS |
| GIGOMEZ2 | 3 | 3 | Solve_Succee | Solve_Succee | 0.00e+00 | 7 | 7 | 1.4ms | 1.8ms | 1.3x | PASS |
| GIGOMEZ3 | 3 | 3 | Solve_Succee | Solve_Succee | 1.11e-16 | 8 | 8 | 1.6ms | 2.0ms | 1.2x | PASS |
| GOFFIN | 51 | 50 | Solve_Succee | Solve_Succee | 1.33e-14 | 6 | 7 | 5.7ms | 4.0ms | 0.7x | PASS |
| GOTTFR | 2 | 2 | Solve_Succee | Solve_Succee | 0.00e+00 | 5 | 5 | 1.1ms | 1.2ms | 1.2x | PASS |
| GOULDQP1 | 32 | 17 | Solve_Succee | Solve_Succee | 0.00e+00 | 15 | 15 | 3.3ms | 3.7ms | 1.1x | PASS |
| GROUPING | 100 | 125 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 28us | 245us | 8.7x | BOTH_FAIL |
| GROWTH | 3 | 12 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 18us | 372us | 21.2x | BOTH_FAIL |
| GROWTHLS | 3 | 0 | Solve_Succee | Solve_Succee | 6.84e-13 | 69 | 71 | 145us | 11.7ms | 80.4x | PASS |
| GULF | 3 | 0 | Solve_Succee | Solve_Succee | 3.40e-22 | 25 | 28 | 624us | 5.4ms | 8.7x | PASS |
| GULFNE | 3 | 99 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 21us | 234us | 11.2x | BOTH_FAIL |
| HAHN1 | 7 | 236 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 30us | 262us | 8.8x | BOTH_FAIL |
| HAHN1LS | 7 | 0 | Solve_Succee | Solve_Succee | 9.54e-01 | 42 | 78 | 2.2ms | 17.4ms | 7.9x | MISMATCH |
| HAIFAM | 99 | 150 | Solve_Succee | Solve_Succee | 5.72e-09 | 107 | 40 | 132.4ms | 15.7ms | 0.1x | PASS |
| HAIFAS | 13 | 9 | Solve_Succee | Solve_Succee | 1.50e-12 | 26 | 16 | 4.6ms | 3.7ms | 0.8x | PASS |
| HAIRY | 2 | 0 | Solve_Succee | Solve_Succee | 0.00e+00 | 60 | 62 | 52us | 10.3ms | 199.5x | PASS |
| HALDMADS | 6 | 42 | Solve_Succee | Solve_Succee | 1.00e+00 | 32 | 8 | 7.8ms | 3.1ms | 0.4x | MISMATCH |
| HART6 | 6 | 0 | Solve_Succee | Solve_Succee | 3.47e-15 | 6 | 7 | 26us | 2.0ms | 77.2x | PASS |
| HATFLDA | 4 | 0 | Solve_Succee | Solve_Succee | 1.11e-12 | 8 | 13 | 21us | 2.5ms | 121.4x | PASS |
| HATFLDANE | 4 | 4 | Solve_Succee | Solve_Succee | 0.00e+00 | 7 | 6 | 1.7ms | 2.6ms | 1.5x | PASS |
| HATFLDB | 4 | 0 | Solve_Succee | Solve_Succee | 1.01e-07 | 18 | 8 | 29us | 1.8ms | 62.6x | PASS |
| HATFLDBNE | 4 | 4 | Infeasible_P | Infeasible_P | N/A | 78 | 13 | 13.9ms | 3.4ms | 0.2x | BOTH_FAIL |
| HATFLDC | 25 | 0 | Solve_Succee | Solve_Succee | 4.61e-14 | 5 | 5 | 50us | 1.4ms | 28.0x | PASS |
| HATFLDCNE | 25 | 25 | Solve_Succee | Solve_Succee | 0.00e+00 | 4 | 4 | 1.3ms | 1.4ms | 1.1x | PASS |
| HATFLDD | 3 | 0 | Solve_Succee | Solve_Succee | 4.58e-20 | 23 | 21 | 50us | 3.3ms | 65.2x | PASS |
| HATFLDDNE | 3 | 10 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 17us | 215us | 13.0x | BOTH_FAIL |
| HATFLDE | 3 | 0 | Solve_Succee | Solve_Succee | 1.04e-19 | 20 | 20 | 75us | 3.1ms | 41.1x | PASS |
| HATFLDENE | 3 | 21 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 17us | 227us | 13.1x | BOTH_FAIL |
| HATFLDF | 3 | 3 | Solve_Succee | Solve_Succee | 0.00e+00 | 140 | 135 | 14.8ms | 24.0ms | 1.6x | PASS |
| HATFLDFL | 3 | 0 | Solve_Succee | Solve_Succee | 3.75e-10 | 1201 | 1281 | 936us | 206.7ms | 220.8x | PASS |
| HATFLDFLNE | 3 | 3 | Solve_Succee | Solve_Succee | 0.00e+00 | 15 | 15 | 2.4ms | 3.1ms | 1.3x | PASS |
| HATFLDFLS | 3 | 0 | Solve_Succee | Solve_Succee | 3.78e-18 | 34 | 36 | 33us | 6.5ms | 196.6x | PASS |
| HATFLDG | 25 | 25 | Solve_Succee | Solve_Succee | 0.00e+00 | 7 | 7 | 1.5ms | 1.8ms | 1.2x | PASS |
| HATFLDGLS | 25 | 0 | Solve_Succee | Solve_Succee | 7.52e-16 | 12 | 14 | 117us | 2.8ms | 23.7x | PASS |
| HATFLDH | 4 | 7 | Solve_Succee | Solve_Succee | 1.45e-16 | 17 | 17 | 3.0ms | 3.5ms | 1.2x | PASS |
| HEART6 | 6 | 6 | Solve_Succee | Solve_Succee | 0.00e+00 | 54 | 22 | 7.6ms | 5.5ms | 0.7x | PASS |
| HEART6LS | 6 | 0 | Solve_Succee | Solve_Succee | 9.46e-23 | 884 | 875 | 2.1ms | 142.6ms | 68.2x | PASS |
| HEART8 | 8 | 8 | Solve_Succee | Solve_Succee | 0.00e+00 | 14 | 12 | 1.7ms | 2.6ms | 1.6x | PASS |
| HEART8LS | 8 | 0 | Solve_Succee | Solve_Succee | 8.88e-19 | 77 | 106 | 246us | 17.7ms | 71.8x | PASS |
| HELIX | 3 | 0 | Solve_Succee | Solve_Succee | 4.38e-12 | 9 | 13 | 20us | 2.4ms | 120.6x | PASS |
| HELIXNE | 3 | 3 | Solve_Succee | Solve_Succee | 0.00e+00 | 7 | 7 | 1.3ms | 1.5ms | 1.2x | PASS |
| HET-Z | 2 | 1002 | Solve_Succee | Solve_Succee | 4.44e-15 | 13 | 11 | 24.4ms | 18.7ms | 0.8x | PASS |
| HIELOW | 3 | 0 | Solve_Succee | Solve_Succee | 2.86e-15 | 4 | 8 | 4.6ms | 11.4ms | 2.5x | PASS |
| HIMMELBA | 2 | 2 | Solve_Succee | Solve_Succee | 0.00e+00 | 1 | 1 | 539us | 567us | 1.1x | PASS |
| HIMMELBB | 2 | 0 | Solve_Succee | Solve_Succee | 1.40e-17 | 9 | 18 | 16us | 3.1ms | 192.9x | PASS |
| HIMMELBC | 2 | 2 | Solve_Succee | Solve_Succee | 0.00e+00 | 6 | 6 | 1.1ms | 1.4ms | 1.3x | PASS |
| HIMMELBCLS | 2 | 0 | Solve_Succee | Solve_Succee | 1.91e-18 | 9 | 6 | 17us | 1.3ms | 75.7x | PASS |
| HIMMELBD | 2 | 2 | Infeasible_P | Infeasible_P | N/A | 27 | 22 | 4.7ms | 5.1ms | 1.1x | BOTH_FAIL |
| HIMMELBE | 3 | 3 | Solve_Succee | Solve_Succee | 0.00e+00 | 2 | 2 | 678us | 724us | 1.1x | PASS |
| HIMMELBF | 4 | 0 | Solve_Succee | Solve_Succee | 1.47e-12 | 121 | 75 | 176us | 11.0ms | 62.6x | PASS |
| HIMMELBFNE | 4 | 7 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 17us | 215us | 12.3x | BOTH_FAIL |
| HIMMELBG | 2 | 0 | Solve_Succee | Solve_Succee | 9.15e-18 | 6 | 6 | 15us | 1.5ms | 97.8x | PASS |
| HIMMELBH | 2 | 0 | Solve_Succee | Solve_Succee | 0.00e+00 | 4 | 4 | 15us | 1.2ms | 77.0x | PASS |
| HIMMELBI | 100 | 12 | Solve_Succee | Solve_Succee | 0.00e+00 | 13 | 13 | 4.1ms | 3.8ms | 0.9x | PASS |
| HIMMELBJ | 45 | 14 | Solve_Succee | Error_In_Ste | N/A | 571 | 580 | 99.5ms | 139.8ms | 1.4x | ipopt_FAIL |
| HIMMELBK | 24 | 14 | Solve_Succee | Solve_Succee | 2.88e-09 | 17 | 18 | 3.2ms | 4.1ms | 1.3x | PASS |
| HIMMELP1 | 2 | 0 | Solve_Succee | Solve_Succee | 3.66e-15 | 10 | 10 | 20us | 2.2ms | 112.4x | PASS |
| HIMMELP2 | 2 | 1 | Solve_Succee | Solve_Succee | 2.97e-10 | 18 | 17 | 3.1ms | 3.7ms | 1.2x | PASS |
| HIMMELP3 | 2 | 2 | Solve_Succee | Solve_Succee | 0.00e+00 | 11 | 11 | 2.2ms | 2.5ms | 1.2x | PASS |
| HIMMELP4 | 2 | 3 | Solve_Succee | Solve_Succee | 1.45e-12 | 22 | 23 | 3.7ms | 4.7ms | 1.3x | PASS |
| HIMMELP5 | 2 | 3 | Solve_Succee | Solve_Succee | 3.86e-12 | 18 | 46 | 2.8ms | 8.3ms | 3.0x | PASS |
| HIMMELP6 | 2 | 5 | Solve_Succee | Solve_Succee | 5.22e-13 | 33 | 31 | 5.2ms | 6.4ms | 1.2x | PASS |
| HONG | 4 | 1 | Solve_Succee | Solve_Succee | 1.57e-16 | 7 | 7 | 1.4ms | 1.9ms | 1.4x | PASS |
| HS1 | 2 | 0 | Solve_Succee | Solve_Succee | 1.63e-15 | 24 | 28 | 25us | 5.4ms | 217.4x | PASS |
| HS10 | 2 | 1 | Solve_Succee | Solve_Succee | 0.00e+00 | 12 | 12 | 1.9ms | 2.5ms | 1.3x | PASS |
| HS100 | 7 | 4 | Solve_Succee | Solve_Succee | 0.00e+00 | 9 | 9 | 1.8ms | 2.4ms | 1.3x | PASS |
| HS100LNP | 7 | 2 | Solve_Succee | Solve_Succee | 0.00e+00 | 20 | 20 | 2.1ms | 3.0ms | 1.4x | PASS |
| HS100MOD | 7 | 4 | Solve_Succee | Solve_Succee | 3.90e-14 | 8 | 14 | 1.8ms | 3.2ms | 1.8x | PASS |
| HS101 | 7 | 5 | Solve_Succee | Solve_Succee | 6.57e-13 | 40 | 39 | 6.1ms | 9.8ms | 1.6x | PASS |
| HS102 | 7 | 5 | Solve_Succee | Solve_Succee | 1.97e-10 | 30 | 52 | 5.0ms | 10.1ms | 2.0x | PASS |
| HS103 | 7 | 5 | Solve_Succee | Solve_Succee | 6.02e-10 | 31 | 21 | 5.6ms | 4.6ms | 0.8x | PASS |
| HS104 | 8 | 5 | Solve_Succee | Solve_Succee | 0.00e+00 | 8 | 8 | 2.2ms | 2.0ms | 0.9x | PASS |
| HS105 | 8 | 1 | Solve_Succee | Solve_Succee | 0.00e+00 | 23 | 23 | 6.2ms | 7.0ms | 1.1x | PASS |
| HS106 | 8 | 6 | Solve_Succee | Solve_Succee | 1.95e-12 | 12 | 18 | 2.3ms | 3.6ms | 1.6x | PASS |
| HS107 | 9 | 6 | Solve_Succee | Solve_Succee | 0.00e+00 | 7 | 7 | 1.5ms | 1.9ms | 1.3x | PASS |
| HS108 | 9 | 13 | Solve_Succee | Solve_Succee | 1.53e-09 | 14 | 11 | 3.1ms | 2.9ms | 0.9x | PASS |
| HS109 | 9 | 10 | Solve_Succee | Solve_Succee | 1.21e-12 | 16 | 14 | 2.9ms | 3.0ms | 1.0x | PASS |
| HS11 | 2 | 1 | Solve_Succee | Solve_Succee | 0.00e+00 | 6 | 6 | 1.3ms | 1.7ms | 1.3x | PASS |
| HS111 | 10 | 3 | Solve_Succee | Solve_Succee | 1.49e-16 | 15 | 15 | 2.6ms | 3.4ms | 1.3x | PASS |
| HS111LNP | 10 | 3 | Solve_Succee | Solve_Succee | 1.49e-16 | 15 | 15 | 1.9ms | 2.6ms | 1.3x | PASS |
| HS112 | 10 | 3 | Solve_Succee | Solve_Succee | 1.49e-16 | 10 | 10 | 2.1ms | 2.3ms | 1.1x | PASS |
| HS113 | 10 | 8 | Solve_Succee | Solve_Succee | 8.04e-15 | 9 | 9 | 2.0ms | 2.3ms | 1.2x | PASS |
| HS114 | 10 | 11 | Solve_Succee | Solve_Succee | 8.48e-13 | 13 | 13 | 2.4ms | 2.8ms | 1.1x | PASS |
| HS116 | 13 | 14 | Solve_Succee | Solve_Succee | 1.59e-09 | 19 | 19 | 3.6ms | 4.1ms | 1.1x | PASS |
| HS117 | 15 | 5 | Solve_Succee | Solve_Succee | 1.93e-12 | 18 | 19 | 3.6ms | 4.1ms | 1.2x | PASS |
| HS118 | 15 | 17 | Solve_Succee | Solve_Succee | 0.00e+00 | 10 | 10 | 2.4ms | 2.5ms | 1.1x | PASS |
| HS119 | 16 | 8 | Solve_Succee | Solve_Succee | 0.00e+00 | 17 | 17 | 3.1ms | 3.7ms | 1.2x | PASS |
| HS12 | 2 | 1 | Solve_Succee | Solve_Succee | 2.99e-13 | 6 | 6 | 1.4ms | 1.7ms | 1.2x | PASS |
| HS13 | 2 | 1 | Solve_Succee | Solve_Succee | 4.89e-11 | 50 | 47 | 6.3ms | 8.2ms | 1.3x | PASS |
| HS14 | 2 | 2 | Solve_Succee | Solve_Succee | 0.00e+00 | 5 | 5 | 1.2ms | 1.4ms | 1.2x | PASS |
| HS15 | 2 | 2 | Solve_Succee | Solve_Succee | 3.77e-11 | 13 | 13 | 2.3ms | 2.7ms | 1.2x | PASS |
| HS16 | 2 | 2 | Solve_Succee | Solve_Succee | 0.00e+00 | 10 | 10 | 2.1ms | 2.4ms | 1.2x | PASS |
| HS17 | 2 | 2 | Solve_Succee | Solve_Succee | 1.90e-08 | 22 | 22 | 3.4ms | 4.2ms | 1.2x | PASS |
| HS18 | 2 | 2 | Solve_Succee | Solve_Succee | 4.23e-10 | 12 | 10 | 1.8ms | 2.1ms | 1.2x | PASS |
| HS19 | 2 | 2 | Solve_Succee | Solve_Succee | 7.84e-16 | 12 | 12 | 2.3ms | 2.8ms | 1.2x | PASS |
| HS1NE | 2 | 2 | Solve_Succee | Solve_Succee | 0.00e+00 | 21 | 30 | 3.1ms | 6.7ms | 2.2x | PASS |
| HS2 | 2 | 0 | Solve_Succee | Solve_Succee | 2.57e-08 | 24 | 10 | 25us | 2.2ms | 90.9x | PASS |
| HS20 | 2 | 3 | Solve_Succee | Solve_Succee | 0.00e+00 | 5 | 5 | 1.2ms | 1.4ms | 1.2x | PASS |
| HS21 | 2 | 1 | Solve_Succee | Solve_Succee | 0.00e+00 | 6 | 6 | 1.6ms | 1.5ms | 1.0x | PASS |
| HS21MOD | 7 | 1 | Solve_Succee | Solve_Succee | 0.00e+00 | 13 | 13 | 2.2ms | 2.7ms | 1.2x | PASS |
| HS22 | 2 | 2 | Solve_Succee | Solve_Succee | 0.00e+00 | 5 | 5 | 1.2ms | 1.5ms | 1.2x | PASS |
| HS23 | 2 | 5 | Solve_Succee | Solve_Succee | 2.34e-11 | 11 | 9 | 2.2ms | 2.1ms | 1.0x | PASS |
| HS24 | 2 | 3 | Solve_Succee | Solve_Succee | 0.00e+00 | 14 | 14 | 2.6ms | 3.3ms | 1.3x | PASS |
| HS25 | 3 | 0 | Solve_Succee | Solve_Succee | 1.00e+00 | 0 | 27 | 29us | 5.9ms | 202.0x | MISMATCH |
| HS25NE | 3 | 99 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 23us | 231us | 10.1x | BOTH_FAIL |
| HS26 | 3 | 1 | Solve_Succee | Solve_Succee | 3.07e-26 | 25 | 25 | 2.2ms | 3.5ms | 1.6x | PASS |
| HS268 | 5 | 5 | Solve_Succee | Solve_Succee | 1.09e-11 | 14 | 14 | 2.3ms | 3.0ms | 1.3x | PASS |
| HS27 | 3 | 1 | Solve_Succee | Solve_Succee | 2.21e-12 | 58 | 57 | 5.0ms | 8.9ms | 1.8x | PASS |
| HS28 | 3 | 1 | Solve_Succee | Solve_Succee | 9.24e-31 | 1 | 1 | 663us | 656us | 1.0x | PASS |
| HS29 | 3 | 1 | Solve_Succee | Solve_Succee | 0.00e+00 | 7 | 7 | 1.5ms | 1.9ms | 1.3x | PASS |
| HS2NE | 2 | 2 | Infeasible_P | Infeasible_P | N/A | 16 | 12 | 3.4ms | 3.3ms | 1.0x | BOTH_FAIL |
| HS3 | 2 | 0 | Solve_Succee | Solve_Succee | 1.10e-07 | 12 | 4 | 18us | 1.2ms | 66.9x | PASS |
| HS30 | 3 | 1 | Solve_Succee | Solve_Succee | 1.89e-08 | 8 | 7 | 1.5ms | 1.8ms | 1.2x | PASS |
| HS31 | 3 | 1 | Solve_Succee | Solve_Succee | 0.00e+00 | 6 | 6 | 1.5ms | 1.6ms | 1.0x | PASS |
| HS32 | 3 | 2 | Solve_Succee | Solve_Succee | 0.00e+00 | 15 | 15 | 2.5ms | 3.0ms | 1.2x | PASS |
| HS33 | 3 | 2 | Solve_Succee | Solve_Succee | 0.00e+00 | 9 | 9 | 1.8ms | 2.1ms | 1.2x | PASS |
| HS34 | 3 | 2 | Solve_Succee | Solve_Succee | 5.92e-09 | 8 | 7 | 1.7ms | 1.9ms | 1.1x | PASS |
| HS35 | 3 | 1 | Solve_Succee | Solve_Succee | 2.50e-16 | 7 | 7 | 1.6ms | 1.7ms | 1.1x | PASS |
| HS35I | 3 | 1 | Solve_Succee | Solve_Succee | 6.80e-16 | 7 | 7 | 1.6ms | 1.7ms | 1.1x | PASS |
| HS35MOD | 3 | 1 | Solve_Succee | Solve_Succee | 0.00e+00 | 14 | 14 | 2.3ms | 2.8ms | 1.2x | PASS |
| HS36 | 3 | 1 | Solve_Succee | Solve_Succee | 0.00e+00 | 11 | 11 | 2.5ms | 2.6ms | 1.0x | PASS |
| HS37 | 3 | 2 | Solve_Succee | Solve_Succee | 2.63e-16 | 11 | 11 | 2.4ms | 2.6ms | 1.1x | PASS |
| HS38 | 4 | 0 | Solve_Succee | Solve_Succee | 2.38e-19 | 41 | 39 | 49us | 7.0ms | 144.2x | PASS |
| HS39 | 4 | 2 | Solve_Succee | Solve_Succee | 0.00e+00 | 13 | 13 | 1.5ms | 2.2ms | 1.5x | PASS |
| HS3MOD | 2 | 0 | Solve_Succee | Solve_Succee | 1.10e-07 | 23 | 4 | 27us | 1.1ms | 42.1x | PASS |
| HS4 | 2 | 0 | Solve_Succee | Solve_Succee | 9.37e-08 | 32 | 4 | 32us | 1.2ms | 35.9x | PASS |
| HS40 | 4 | 3 | Solve_Succee | Solve_Succee | 0.00e+00 | 3 | 3 | 773us | 990us | 1.3x | PASS |
| HS41 | 4 | 1 | Solve_Succee | Solve_Succee | 0.00e+00 | 7 | 7 | 1.4ms | 1.8ms | 1.3x | PASS |
| HS42 | 4 | 2 | Solve_Succee | Solve_Succee | 0.00e+00 | 4 | 4 | 861us | 1.1ms | 1.2x | PASS |
| HS43 | 4 | 3 | Solve_Succee | Solve_Succee | 0.00e+00 | 8 | 8 | 1.5ms | 2.0ms | 1.3x | PASS |
| HS44 | 4 | 6 | Solve_Succee | Solve_Succee | 4.80e-08 | 86 | 24 | 7.7ms | 4.9ms | 0.6x | PASS |
| HS44NEW | 4 | 6 | Solve_Succee | Solve_Succee | 0.00e+00 | 18 | 18 | 3.4ms | 4.0ms | 1.2x | PASS |
| HS45 | 5 | 0 | Solve_Succee | Solve_Succee | 5.50e-07 | 32 | 11 | 38us | 2.5ms | 66.3x | PASS |
| HS46 | 5 | 2 | Solve_Succee | Solve_Succee | 2.74e-24 | 19 | 19 | 2.1ms | 2.8ms | 1.3x | PASS |
| HS47 | 5 | 3 | Solve_Succee | Solve_Succee | 1.02e-14 | 19 | 19 | 1.9ms | 2.9ms | 1.5x | PASS |
| HS48 | 5 | 2 | Solve_Succee | Solve_Succee | 5.42e-31 | 1 | 1 | 624us | 672us | 1.1x | PASS |
| HS49 | 5 | 2 | Solve_Succee | Solve_Succee | 0.00e+00 | 19 | 19 | 2.1ms | 2.8ms | 1.3x | PASS |
| HS5 | 2 | 0 | Solve_Succee | Solve_Succee | 1.04e-15 | 9 | 7 | 22us | 1.6ms | 75.2x | PASS |
| HS50 | 5 | 3 | Solve_Succee | Solve_Succee | 0.00e+00 | 9 | 9 | 1.1ms | 1.7ms | 1.6x | PASS |
| HS51 | 5 | 3 | Solve_Succee | Solve_Succee | 0.00e+00 | 1 | 1 | 524us | 676us | 1.3x | PASS |
| HS52 | 5 | 3 | Solve_Succee | Solve_Succee | 5.00e-16 | 1 | 1 | 642us | 677us | 1.1x | PASS |
| HS53 | 5 | 3 | Solve_Succee | Solve_Succee | 2.17e-16 | 6 | 6 | 1.4ms | 1.5ms | 1.1x | PASS |
| HS54 | 6 | 1 | Solve_Succee | Solve_Succee | 0.00e+00 | 15 | 15 | 2.8ms | 3.3ms | 1.2x | PASS |
| HS55 | 6 | 6 | Solve_Succee | Solve_Succee | 1.92e-06 | 7 | 18 | 1.4ms | 4.9ms | 3.4x | PASS |
| HS56 | 7 | 4 | Solve_Succee | Solve_Succee | 0.00e+00 | 10 | 10 | 1.5ms | 2.7ms | 1.8x | PASS |
| HS57 | 2 | 1 | Solve_Succee | Solve_Succee | 2.43e-17 | 20 | 10 | 3.8ms | 2.9ms | 0.8x | PASS |
| HS59 | 2 | 3 | Solve_Succee | Solve_Succee | 2.33e-14 | 18 | 17 | 5.0ms | 5.2ms | 1.0x | PASS |
| HS6 | 2 | 1 | Solve_Succee | Solve_Succee | 4.93e-32 | 5 | 5 | 1.7ms | 2.0ms | 1.2x | PASS |
| HS60 | 3 | 1 | Solve_Succee | Solve_Succee | 0.00e+00 | 6 | 6 | 2.1ms | 2.3ms | 1.0x | PASS |
| HS61 | 3 | 2 | Solve_Succee | Solve_Succee | 3.18e-13 | 7 | 10 | 1.7ms | 2.4ms | 1.4x | PASS |
| HS62 | 3 | 1 | Solve_Succee | Solve_Succee | 2.77e-16 | 6 | 6 | 2.4ms | 2.4ms | 1.0x | PASS |
| HS63 | 3 | 2 | Solve_Succee | Solve_Succee | 0.00e+00 | 5 | 5 | 1.9ms | 2.0ms | 1.1x | PASS |
| HS64 | 3 | 1 | Solve_Succee | Solve_Succee | 7.24e-10 | 16 | 16 | 4.2ms | 4.6ms | 1.1x | PASS |
| HS65 | 3 | 1 | Solve_Succee | Solve_Succee | 0.00e+00 | 16 | 16 | 4.7ms | 5.0ms | 1.1x | PASS |
| HS66 | 3 | 2 | Solve_Succee | Solve_Succee | 8.06e-10 | 12 | 10 | 2.9ms | 3.0ms | 1.0x | PASS |
| HS67 | 3 | 14 | Solve_Succee | Solve_Succee | 1.37e-15 | 9 | 9 | 3.6ms | 3.0ms | 0.8x | PASS |
| HS68 | 4 | 2 | Solve_Succee | Solve_Succee | 1.04e-13 | 16 | 16 | 4.3ms | 4.6ms | 1.1x | PASS |
| HS69 | 4 | 2 | Solve_Succee | Solve_Succee | 2.73e-15 | 10 | 10 | 3.4ms | 3.2ms | 1.0x | PASS |
| HS7 | 2 | 1 | Solve_Succee | Solve_Succee | 0.00e+00 | 27 | 27 | 4.8ms | 6.2ms | 1.3x | PASS |
| HS70 | 4 | 1 | Solve_Succee | Solve_Succee | 1.79e-01 | 30 | 46 | 7.3ms | 11.2ms | 1.5x | MISMATCH |
| HS71 | 4 | 2 | Solve_Succee | Solve_Succee | 1.46e-15 | 8 | 8 | 2.7ms | 2.8ms | 1.0x | PASS |
| HS72 | 4 | 2 | Solve_Succee | Solve_Succee | 1.48e-09 | 21 | 16 | 4.8ms | 4.2ms | 0.9x | PASS |
| HS73 | 4 | 3 | Solve_Succee | Solve_Succee | 0.00e+00 | 8 | 8 | 2.8ms | 2.6ms | 1.0x | PASS |
| HS74 | 4 | 5 | Solve_Succee | Solve_Succee | 1.77e-16 | 8 | 8 | 3.0ms | 2.9ms | 1.0x | PASS |
| HS75 | 4 | 5 | Solve_Succee | Solve_Succee | 1.76e-16 | 8 | 8 | 2.9ms | 2.8ms | 1.0x | PASS |
| HS76 | 4 | 3 | Solve_Succee | Solve_Succee | 0.00e+00 | 7 | 7 | 2.7ms | 2.6ms | 1.0x | PASS |
| HS76I | 4 | 3 | Solve_Succee | Solve_Succee | 0.00e+00 | 6 | 6 | 2.4ms | 2.4ms | 1.0x | PASS |
| HS77 | 5 | 2 | Solve_Succee | Solve_Succee | 1.51e-13 | 11 | 11 | 2.2ms | 2.8ms | 1.3x | PASS |
| HS78 | 5 | 3 | Solve_Succee | Solve_Succee | 0.00e+00 | 4 | 4 | 1.2ms | 1.5ms | 1.2x | PASS |
| HS79 | 5 | 3 | Solve_Succee | Solve_Succee | 0.00e+00 | 4 | 4 | 1.4ms | 1.6ms | 1.2x | PASS |
| HS8 | 2 | 2 | Solve_Succee | Solve_Succee | 0.00e+00 | 5 | 5 | 1.5ms | 1.4ms | 1.0x | PASS |
| HS80 | 5 | 3 | Solve_Succee | Solve_Succee | 3.47e-17 | 5 | 5 | 1.9ms | 2.1ms | 1.1x | PASS |
| HS81 | 5 | 3 | Solve_Succee | Solve_Succee | 3.47e-12 | 7 | 68 | 2.6ms | 16.6ms | 6.4x | PASS |
| HS83 | 5 | 3 | Solve_Succee | Solve_Succee | 0.00e+00 | 9 | 9 | 3.2ms | 2.9ms | 0.9x | PASS |
| HS84 | 5 | 3 | Solve_Succee | Solve_Succee | 9.96e-01 | 15 | 9 | 4.4ms | 3.0ms | 0.7x | MISMATCH |
| HS85 | 5 | 21 | Solve_Succee | Solve_Succee | 2.31e-09 | 20 | 13 | 7.3ms | 5.4ms | 0.7x | PASS |
| HS86 | 5 | 10 | Solve_Succee | Solve_Succee | 2.20e-16 | 10 | 10 | 3.4ms | 3.4ms | 1.0x | PASS |
| HS87 | 6 | 4 | Maximum_Iter | Maximum_Iter | N/A | 2999 | 3000 | 312.5ms | 461.1ms | 1.5x | BOTH_FAIL |
| HS88 | 2 | 1 | Solve_Succee | Solve_Succee | 1.01e-12 | 25 | 18 | 7.7ms | 5.3ms | 0.7x | PASS |
| HS89 | 3 | 1 | Solve_Succee | Solve_Succee | 7.33e-15 | 15 | 15 | 5.4ms | 5.2ms | 1.0x | PASS |
| HS9 | 2 | 1 | Solve_Succee | Solve_Succee | 1.11e-16 | 7 | 3 | 1.8ms | 971us | 0.5x | PASS |
| HS90 | 4 | 1 | Solve_Succee | Solve_Succee | 3.10e-15 | 17 | 16 | 6.2ms | 6.0ms | 1.0x | PASS |
| HS91 | 5 | 1 | Solve_Succee | Solve_Succee | 0.00e+00 | 16 | 16 | 6.9ms | 6.1ms | 0.9x | PASS |
| HS92 | 6 | 1 | Solve_Succee | Solve_Succee | 1.15e-13 | 19 | 35 | 8.8ms | 12.9ms | 1.5x | PASS |
| HS93 | 6 | 2 | Solve_Succee | Solve_Succee | 2.10e-16 | 7 | 7 | 2.7ms | 2.4ms | 0.9x | PASS |
| HS95 | 6 | 4 | Solve_Succee | Solve_Succee | 4.39e-10 | 9 | 9 | 3.3ms | 3.2ms | 1.0x | PASS |
| HS96 | 6 | 4 | Solve_Succee | Solve_Succee | 4.39e-10 | 8 | 8 | 3.1ms | 2.9ms | 1.0x | PASS |
| HS97 | 6 | 4 | Solve_Succee | Solve_Succee | 1.12e-08 | 24 | 24 | 6.8ms | 6.7ms | 1.0x | PASS |
| HS98 | 6 | 4 | Solve_Succee | Solve_Succee | 1.12e-08 | 13 | 13 | 4.0ms | 3.9ms | 1.0x | PASS |
| HS99 | 7 | 2 | Solve_Succee | Solve_Succee | 0.00e+00 | 4 | 5 | 1.8ms | 1.9ms | 1.0x | PASS |
| HS99EXP | 31 | 21 | Solve_Succee | Solve_Succee | 0.00e+00 | 26 | 17 | 6.8ms | 4.9ms | 0.7x | PASS |
| HUBFIT | 2 | 1 | Solve_Succee | Solve_Succee | 3.47e-18 | 7 | 7 | 2.5ms | 2.4ms | 1.0x | PASS |
| HUMPS | 2 | 0 | Solve_Succee | Solve_Succee | 2.73e-17 | 672 | 1533 | 760us | 217.1ms | 285.7x | PASS |
| HYDC20LS | 99 | 0 | Solve_Succee | Solve_Succee | 7.12e-08 | 375 | 639 | 66.4ms | 160.5ms | 2.4x | PASS |
| HYDCAR20 | 99 | 99 | Solve_Succee | Solve_Succee | 0.00e+00 | 9 | 9 | 5.2ms | 3.7ms | 0.7x | PASS |
| HYDCAR6 | 29 | 29 | Solve_Succee | Solve_Succee | 0.00e+00 | 5 | 5 | 2.2ms | 2.2ms | 1.0x | PASS |
| HYDCAR6LS | 29 | 0 | Solve_Succee | Solve_Succee | 1.77e-15 | 143 | 149 | 3.0ms | 36.8ms | 12.3x | PASS |
| HYPCIR | 2 | 2 | Solve_Succee | Solve_Succee | 0.00e+00 | 5 | 5 | 1.3ms | 1.7ms | 1.3x | PASS |
| JENSMP | 2 | 0 | Solve_Succee | Solve_Succee | 1.14e-16 | 9 | 9 | 40us | 2.2ms | 56.9x | PASS |
| JENSMPNE | 2 | 10 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 30us | 299us | 9.9x | BOTH_FAIL |
| JUDGE | 2 | 0 | Solve_Succee | Solve_Succee | 1.55e-15 | 8 | 9 | 40us | 2.1ms | 52.0x | PASS |
| JUDGEB | 2 | 0 | Solve_Succee | Solve_Succee | 3.53e-15 | 8 | 9 | 46us | 2.9ms | 62.4x | PASS |
| JUDGENE | 2 | 20 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 29us | 303us | 10.6x | BOTH_FAIL |
| KIRBY2 | 5 | 151 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 41us | 336us | 8.2x | BOTH_FAIL |
| KIRBY2LS | 5 | 0 | Solve_Succee | Solve_Succee | 8.87e-15 | 15 | 11 | 600us | 3.4ms | 5.7x | PASS |
| KIWCRESC | 3 | 2 | Solve_Succee | Solve_Succee | 3.76e-11 | 9 | 8 | 2.9ms | 2.5ms | 0.9x | PASS |
| KOEBHELB | 3 | 0 | Solve_Succee | Solve_Succee | 3.09e-01 | 72 | 71 | 1.9ms | 20.9ms | 11.0x | MISMATCH |
| KOEBHELBNE | 3 | 156 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 37us | 329us | 8.9x | BOTH_FAIL |
| KOWOSB | 4 | 0 | Solve_Succee | Solve_Succee | 2.01e-18 | 9 | 8 | 52us | 2.4ms | 45.6x | PASS |
| KOWOSBNE | 4 | 11 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 19us | 328us | 17.6x | BOTH_FAIL |
| KSIP | 20 | 1001 | Solve_Succee | Solve_Succee | 1.09e-08 | 28 | 22 | 143.7ms | 65.1ms | 0.5x | PASS |
| LAKES | 90 | 78 | Solve_Succee | Solve_Succee | 3.32e-15 | 11 | 11 | 6.5ms | 4.7ms | 0.7x | PASS |
| LANCZOS1 | 6 | 24 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 34us | 310us | 9.0x | BOTH_FAIL |
| LANCZOS1LS | 6 | 0 | Solve_Succee | Solve_Succee | 8.25e-08 | 58 | 115 | 553us | 24.9ms | 45.0x | PASS |
| LANCZOS2 | 6 | 24 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 29us | 304us | 10.7x | BOTH_FAIL |
| LANCZOS2LS | 6 | 0 | Solve_Succee | Solve_Succee | 1.05e-07 | 56 | 101 | 532us | 22.7ms | 42.7x | PASS |
| LANCZOS3 | 6 | 24 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 28us | 310us | 10.9x | BOTH_FAIL |
| LANCZOS3LS | 6 | 0 | Solve_Succee | Solve_Succee | 1.03e-07 | 54 | 174 | 513us | 37.0ms | 72.0x | PASS |
| LAUNCH | 25 | 28 | Solve_Succee | Solve_Succee | 2.64e-09 | 23 | 12 | 7.2ms | 3.8ms | 0.5x | PASS |
| LEVYMONE10 | 10 | 20 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 33us | 304us | 9.3x | BOTH_FAIL |
| LEVYMONE5 | 2 | 4 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 26us | 298us | 11.7x | BOTH_FAIL |
| LEVYMONE6 | 3 | 6 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 28us | 296us | 10.7x | BOTH_FAIL |
| LEVYMONE7 | 4 | 8 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 18us | 296us | 16.6x | BOTH_FAIL |
| LEVYMONE8 | 5 | 10 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 29us | 300us | 10.4x | BOTH_FAIL |
| LEVYMONE9 | 8 | 16 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 33us | 218us | 6.7x | BOTH_FAIL |
| LEVYMONT10 | 10 | 0 | Solve_Succee | Solve_Succee | 3.47e-16 | 4 | 4 | 54us | 1.8ms | 33.5x | PASS |
| LEVYMONT5 | 2 | 0 | Solve_Succee | Solve_Succee | 1.00e+00 | 7 | 10 | 36us | 2.8ms | 77.9x | MISMATCH |
| LEVYMONT6 | 3 | 0 | Solve_Succee | Solve_Succee | 0.00e+00 | 7 | 8 | 36us | 2.9ms | 81.4x | PASS |
| LEVYMONT7 | 4 | 0 | Solve_Succee | Solve_Succee | 0.00e+00 | 7 | 7 | 24us | 2.8ms | 117.4x | PASS |
| LEVYMONT8 | 5 | 0 | Solve_Succee | Solve_Succee | 0.00e+00 | 4 | 4 | 21us | 1.4ms | 68.8x | PASS |
| LEVYMONT9 | 8 | 0 | Solve_Succee | Solve_Succee | 3.42e-16 | 4 | 4 | 44us | 1.7ms | 38.8x | PASS |
| LEWISPOL | 6 | 9 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 17us | 301us | 17.9x | BOTH_FAIL |
| LHAIFAM | 99 | 150 | Restoration_ | Invalid_Numb | N/A | 1 | 0 | 465.1ms | 265us | 0.0x | BOTH_FAIL |
| LIN | 4 | 2 | Solve_Succee | Solve_Succee | 3.87e-09 | 6 | 7 | 2.0ms | 2.4ms | 1.2x | PASS |
| LINSPANH | 97 | 33 | Solve_Succee | Solve_Succee | 7.31e-11 | 23 | 24 | 8.8ms | 7.5ms | 0.8x | PASS |
| LOADBAL | 31 | 31 | Solve_Succee | Solve_Succee | 0.00e+00 | 13 | 13 | 5.3ms | 4.4ms | 0.8x | PASS |
| LOGHAIRY | 2 | 0 | Solve_Succee | Solve_Succee | 0.00e+00 | 1496 | 2747 | 1.9ms | 393.7ms | 210.0x | PASS |
| LOGROS | 2 | 0 | Solve_Succee | Solve_Succee | 2.24e-14 | 51 | 49 | 67us | 13.7ms | 205.9x | PASS |
| LOOTSMA | 3 | 2 | Solve_Succee | Solve_Succee | 2.36e-11 | 10 | 13 | 3.1ms | 4.1ms | 1.3x | PASS |
| LOTSCHD | 12 | 7 | Solve_Succee | Solve_Succee | 0.00e+00 | 9 | 9 | 3.1ms | 2.9ms | 0.9x | PASS |
| LRCOVTYPE | 54 | 0 | Solve_Succee | Solve_Succee | 3.01e-03 | 16 | 33 | 3.08s | 6.20s | 2.0x | MISMATCH |
| LRIJCNN1 | 22 | 0 | Solve_Succee | Solve_Succee | 1.05e-14 | 15 | 11 | 241.8ms | 194.6ms | 0.8x | PASS |
| LSC1 | 3 | 6 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 19us | 220us | 11.6x | BOTH_FAIL |
| LSC1LS | 3 | 0 | Solve_Succee | Solve_Succee | 9.21e-16 | 17 | 16 | 29us | 3.0ms | 104.0x | PASS |
| LSC2 | 3 | 6 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 16us | 222us | 13.7x | BOTH_FAIL |
| LSC2LS | 3 | 0 | Solve_Succee | Solve_Succee | 1.75e-05 | 35 | 38 | 46us | 5.2ms | 113.0x | PASS |
| LSNNODOC | 5 | 4 | Solve_Succee | Solve_Succee | 7.87e-02 | 26 | 10 | 5.6ms | 2.5ms | 0.5x | MISMATCH |
| LSQFIT | 2 | 1 | Solve_Succee | Solve_Succee | 0.00e+00 | 7 | 7 | 1.4ms | 1.8ms | 1.3x | PASS |
| MADSEN | 3 | 6 | Solve_Succee | Solve_Succee | 0.00e+00 | 18 | 18 | 2.9ms | 5.3ms | 1.9x | PASS |
| MAKELA1 | 3 | 2 | Solve_Succee | Solve_Succee | 1.57e-16 | 12 | 12 | 2.2ms | 2.7ms | 1.3x | PASS |
| MAKELA2 | 3 | 3 | Solve_Succee | Solve_Succee | 1.23e-16 | 6 | 6 | 1.1ms | 1.7ms | 1.5x | PASS |
| MAKELA3 | 21 | 20 | Solve_Succee | Solve_Succee | 1.06e-11 | 17 | 11 | 3.1ms | 2.7ms | 0.9x | PASS |
| MAKELA4 | 21 | 40 | Solve_Succee | Solve_Succee | 2.89e-18 | 5 | 5 | 1.6ms | 1.7ms | 1.0x | PASS |
| MARATOS | 2 | 1 | Solve_Succee | Solve_Succee | 0.00e+00 | 4 | 4 | 738us | 1.0ms | 1.4x | PASS |
| MARATOSB | 2 | 0 | Solve_Succee | Solve_Succee | 1.52e-13 | 669 | 672 | 352us | 102.9ms | 292.7x | PASS |
| MATRIX2 | 6 | 2 | Solve_Succee | Solve_Succee | 5.28e-11 | 51 | 42 | 5.6ms | 7.3ms | 1.3x | PASS |
| MAXLIKA | 8 | 0 | Solve_Succee | Solve_Succee | 9.74e-11 | 64 | 23 | 7.1ms | 7.1ms | 1.0x | PASS |
| MCONCON | 15 | 11 | Solve_Succee | Solve_Succee | 0.00e+00 | 7 | 7 | 1.6ms | 1.8ms | 1.1x | PASS |
| MDHOLE | 2 | 0 | Solve_Succee | Solve_Succee | 1.10e-07 | 45 | 42 | 38us | 8.8ms | 231.7x | PASS |
| MESH | 41 | 48 | Diverging_It | Diverging_It | N/A | 79 | 79 | 25.1ms | 21.0ms | 0.8x | BOTH_FAIL |
| METHANB8 | 31 | 31 | Solve_Succee | Solve_Succee | 0.00e+00 | 3 | 3 | 951us | 1.2ms | 1.2x | PASS |
| METHANB8LS | 31 | 0 | Solve_Succee | Solve_Succee | 4.45e-15 | 9 | 8 | 179us | 1.8ms | 10.0x | PASS |
| METHANL8 | 31 | 31 | Solve_Succee | Solve_Succee | 0.00e+00 | 4 | 4 | 1.4ms | 1.3ms | 0.9x | PASS |
| METHANL8LS | 31 | 0 | Solve_Succee | Solve_Succee | 3.47e-17 | 39 | 40 | 784us | 8.5ms | 10.8x | PASS |
| MEXHAT | 2 | 0 | Solve_Succee | Solve_Succee | 6.77e-10 | 27 | 26 | 26us | 4.0ms | 153.1x | PASS |
| MEYER3 | 3 | 0 | Solve_Succee | Solve_Succee | 1.25e-12 | 180 | 194 | 363us | 30.7ms | 84.7x | PASS |
| MEYER3NE | 3 | 16 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 17us | 213us | 12.7x | BOTH_FAIL |
| MGH09 | 4 | 11 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 19us | 220us | 11.4x | BOTH_FAIL |
| MGH09LS | 4 | 0 | Solve_Succee | Solve_Succee | 1.21e-03 | 28 | 72 | 62us | 12.2ms | 194.8x | MISMATCH |
| MGH10 | 3 | 16 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 18us | 220us | 11.9x | BOTH_FAIL |
| MGH10LS | 3 | 0 | Solve_Succee | Solve_Succee | 6.62e-12 | 764 | 1828 | 1.4ms | 296.5ms | 207.6x | PASS |
| MGH10S | 3 | 16 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 18us | 215us | 12.2x | BOTH_FAIL |
| MGH10SLS | 3 | 0 | Solve_Succee | Solve_Succee | 8.40e-13 | 333 | 354 | 645us | 56.4ms | 87.3x | PASS |
| MGH17 | 5 | 33 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 20us | 224us | 11.4x | BOTH_FAIL |
| MGH17LS | 5 | 0 | Solve_Succee | Solve_Succee | 1.00e+00 | 5 | 47 | 47us | 9.0ms | 190.8x | MISMATCH |
| MGH17S | 5 | 33 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 20us | 227us | 11.1x | BOTH_FAIL |
| MGH17SLS | 5 | 0 | Solve_Succee | Solve_Succee | 9.76e-01 | 10 | 41 | 69us | 8.0ms | 115.9x | MISMATCH |
| MIFFLIN1 | 3 | 2 | Solve_Succee | Solve_Succee | 4.44e-16 | 5 | 5 | 1.2ms | 1.4ms | 1.1x | PASS |
| MIFFLIN2 | 3 | 2 | Solve_Succee | Solve_Succee | 2.59e-11 | 9 | 11 | 1.7ms | 2.7ms | 1.6x | PASS |
| MINMAXBD | 5 | 20 | Solve_Succee | Solve_Succee | 1.19e-11 | 28 | 25 | 5.2ms | 5.9ms | 1.1x | PASS |
| MINMAXRB | 3 | 4 | Solve_Succee | Solve_Succee | 7.30e-16 | 8 | 8 | 1.5ms | 2.0ms | 1.3x | PASS |
| MINSURF | 64 | 0 | Solve_Succee | Solve_Succee | 0.00e+00 | 4 | 4 | 178us | 1.5ms | 8.5x | PASS |
| MISRA1A | 2 | 14 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 18us | 223us | 12.3x | BOTH_FAIL |
| MISRA1ALS | 2 | 0 | Solve_Succee | Solve_Succee | 6.74e-15 | 31 | 40 | 56us | 6.4ms | 114.4x | PASS |
| MISRA1B | 2 | 14 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 17us | 217us | 12.9x | BOTH_FAIL |
| MISRA1BLS | 2 | 0 | Solve_Succee | Solve_Succee | 6.75e-14 | 25 | 34 | 49us | 5.4ms | 110.5x | PASS |
| MISRA1C | 2 | 14 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 18us | 225us | 12.8x | BOTH_FAIL |
| MISRA1CLS | 2 | 0 | Solve_Succee | Solve_Succee | 2.32e-14 | 13 | 14 | 30us | 2.8ms | 93.4x | PASS |
| MISRA1D | 2 | 14 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 16us | 221us | 13.7x | BOTH_FAIL |
| MISRA1DLS | 2 | 0 | Solve_Succee | Solve_Succee | 1.01e-15 | 20 | 30 | 40us | 4.9ms | 122.9x | PASS |
| MISTAKE | 9 | 13 | Solve_Succee | Solve_Succee | 7.14e-11 | 19 | 16 | 3.7ms | 3.8ms | 1.0x | PASS |
| MRIBASIS | 36 | 55 | Solve_Succee | Solve_Succee | 0.00e+00 | 15 | 15 | 5.2ms | 4.5ms | 0.9x | PASS |
| MSS1 | 90 | 73 | Solved_To_Ac | Solve_Succee | 9.81e-01 | 734 | 95 | 1.44s | 52.7ms | 0.0x | MISMATCH |
| MUONSINE | 1 | 512 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 22us | 254us | 11.8x | BOTH_FAIL |
| MUONSINELS | 1 | 0 | Solve_Succee | Solve_Succee | 4.19e-01 | 7 | 8 | 288us | 2.0ms | 7.0x | MISMATCH |
| MWRIGHT | 5 | 3 | Solve_Succee | Solve_Succee | 0.00e+00 | 10 | 10 | 1.3ms | 2.0ms | 1.6x | PASS |
| NASH | 72 | 24 | Not_Enough_D | Infeasible_P | N/A | 0 | 45 | 23us | 12.3ms | 531.9x | BOTH_FAIL |
| NELSON | 3 | 128 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 20us | 232us | 11.5x | BOTH_FAIL |
| NET1 | 48 | 57 | Solve_Succee | Solve_Succee | 2.14e-12 | 23 | 26 | 6.0ms | 6.2ms | 1.0x | PASS |
| NYSTROM5 | 18 | 20 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 22us | 224us | 10.4x | BOTH_FAIL |
| NYSTROM5C | 18 | 20 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 19us | 228us | 12.0x | BOTH_FAIL |
| ODFITS | 10 | 6 | Solve_Succee | Solve_Succee | 1.91e-16 | 8 | 8 | 1.5ms | 2.0ms | 1.3x | PASS |
| OET1 | 3 | 1002 | Solve_Succee | Solve_Succee | 4.14e-10 | 34 | 33 | 60.3ms | 48.0ms | 0.8x | PASS |
| OET2 | 3 | 1002 | Solve_Succee | Solve_Succee | 7.49e-16 | 121 | 181 | 766.2ms | 278.8ms | 0.4x | PASS |
| OET3 | 4 | 1002 | Solve_Succee | Solve_Succee | 4.65e-09 | 12 | 13 | 25.6ms | 21.3ms | 0.8x | PASS |
| OET4 | 4 | 1002 | Solve_Succee | Solve_Succee | 8.53e-01 | 33 | 165 | 59.8ms | 251.2ms | 4.2x | MISMATCH |
| OET5 | 5 | 1002 | Solve_Succee | Solve_Succee | 7.31e-13 | 74 | 64 | 190.5ms | 118.6ms | 0.6x | PASS |
| OET6 | 5 | 1002 | Solve_Succee | Solve_Succee | 1.34e-16 | 185 | 126 | 555.6ms | 359.1ms | 0.6x | PASS |
| OET7 | 7 | 1002 | Solve_Succee | Solve_Succee | 2.02e-07 | 139 | 193 | 886.4ms | 537.9ms | 0.6x | PASS |
| OPTCNTRL | 32 | 20 | Solve_Succee | Solve_Succee | 1.45e-15 | 9 | 9 | 2.1ms | 2.4ms | 1.1x | PASS |
| OPTPRLOC | 30 | 30 | Solve_Succee | Solve_Succee | 6.74e-11 | 13 | 13 | 3.4ms | 3.4ms | 1.0x | PASS |
| ORTHREGB | 27 | 6 | Solve_Succee | Solve_Succee | 4.52e-20 | 1 | 2 | 556us | 1.1ms | 2.0x | PASS |
| OSBORNE1 | 5 | 33 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 20us | 238us | 11.9x | BOTH_FAIL |
| OSBORNE2 | 11 | 65 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 23us | 252us | 10.8x | BOTH_FAIL |
| OSBORNEA | 5 | 0 | Solve_Succee | Solve_Succee | 3.46e-18 | 24 | 64 | 145us | 10.9ms | 74.7x | PASS |
| OSBORNEB | 11 | 0 | Solve_Succee | Solve_Succee | 1.39e-17 | 20 | 19 | 536us | 3.6ms | 6.7x | PASS |
| OSLBQP | 8 | 0 | Solve_Succee | Solve_Succee | 8.71e-08 | 31 | 15 | 41us | 2.9ms | 69.8x | PASS |
| PALMER1 | 4 | 0 | Solve_Succee | Solve_Succee | 1.55e-16 | 842 | 13 | 3.1ms | 2.8ms | 0.9x | PASS |
| PALMER1A | 6 | 0 | Solve_Succee | Solve_Succee | 9.98e-01 | 30 | 48 | 166us | 11.6ms | 69.8x | MISMATCH |
| PALMER1ANE | 6 | 35 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 21us | 223us | 10.4x | BOTH_FAIL |
| PALMER1B | 4 | 0 | Solve_Succee | Solve_Succee | 4.61e-12 | 21 | 17 | 94us | 3.5ms | 37.0x | PASS |
| PALMER1BNE | 4 | 35 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 21us | 224us | 10.8x | BOTH_FAIL |
| PALMER1C | 8 | 0 | Solve_Succee | Solve_Succee | 1.06e-13 | 1 | 1 | 24us | 581us | 24.6x | PASS |
| PALMER1D | 7 | 0 | Solve_Succee | Solve_Succee | 5.86e-14 | 1 | 1 | 21us | 595us | 28.2x | PASS |
| PALMER1E | 8 | 0 | Solve_Succee | Solve_Succee | 5.00e-11 | 123 | 55 | 1.7ms | 15.7ms | 9.3x | PASS |
| PALMER1ENE | 8 | 35 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 33us | 299us | 9.0x | BOTH_FAIL |
| PALMER1NE | 4 | 31 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 29us | 308us | 10.7x | BOTH_FAIL |
| PALMER2 | 4 | 0 | Solve_Succee | Solve_Succee | 4.98e-16 | 2305 | 28 | 10.5ms | 8.7ms | 0.8x | PASS |
| PALMER2A | 6 | 0 | Solve_Succee | Solve_Succee | 7.97e-14 | 100 | 91 | 694us | 25.7ms | 37.0x | PASS |
| PALMER2ANE | 6 | 23 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 38us | 321us | 8.5x | BOTH_FAIL |
| PALMER2B | 4 | 0 | Solve_Succee | Solve_Succee | 1.79e-14 | 16 | 15 | 57us | 4.8ms | 84.5x | PASS |
| PALMER2BNE | 4 | 23 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 32us | 297us | 9.3x | BOTH_FAIL |
| PALMER2C | 8 | 0 | Solve_Succee | Solve_Succee | 1.92e-14 | 1 | 1 | 34us | 818us | 23.9x | PASS |
| PALMER2E | 8 | 0 | Solve_Succee | Solve_Succee | 1.28e-07 | 168 | 114 | 1.6ms | 31.2ms | 19.5x | PASS |
| PALMER2ENE | 8 | 23 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 19us | 312us | 16.1x | BOTH_FAIL |
| PALMER2NE | 4 | 23 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 33us | 305us | 9.1x | BOTH_FAIL |
| PALMER3 | 4 | 0 | Solve_Succee | Solve_Succee | 6.25e-02 | 261 | 44 | 1.4ms | 11.6ms | 8.3x | MISMATCH |
| PALMER3A | 6 | 0 | Solve_Succee | Solve_Succee | 7.98e-14 | 90 | 73 | 637us | 20.4ms | 32.0x | PASS |
| PALMER3ANE | 6 | 23 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 19us | 302us | 16.3x | BOTH_FAIL |
| PALMER3B | 4 | 0 | Solve_Succee | Solve_Succee | 3.78e-15 | 11 | 15 | 85us | 5.1ms | 59.9x | PASS |
| PALMER3BNE | 4 | 23 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 34us | 301us | 8.9x | BOTH_FAIL |
| PALMER3C | 8 | 0 | Solve_Succee | Solve_Succee | 2.26e-15 | 1 | 1 | 35us | 829us | 23.9x | PASS |
| PALMER3E | 8 | 0 | Solve_Succee | Solve_Succee | 4.06e-08 | 21 | 32 | 255us | 9.6ms | 37.6x | PASS |
| PALMER3ENE | 8 | 23 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 36us | 224us | 6.2x | BOTH_FAIL |
| PALMER3NE | 4 | 23 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 31us | 309us | 10.1x | BOTH_FAIL |
| PALMER4 | 4 | 0 | Solve_Succee | Solve_Succee | 3.98e-16 | 136 | 16 | 721us | 5.8ms | 8.0x | PASS |
| PALMER4A | 6 | 0 | Solve_Succee | Solve_Succee | 8.07e-14 | 104 | 53 | 728us | 14.8ms | 20.3x | PASS |
| PALMER4ANE | 6 | 23 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 34us | 222us | 6.5x | BOTH_FAIL |
| PALMER4B | 4 | 0 | Solve_Succee | Solve_Succee | 9.36e-15 | 10 | 16 | 79us | 5.2ms | 66.4x | PASS |
| PALMER4BNE | 4 | 23 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 32us | 302us | 9.6x | BOTH_FAIL |
| PALMER4C | 8 | 0 | Solve_Succee | Solve_Succee | 3.43e-15 | 1 | 1 | 35us | 634us | 17.9x | PASS |
| PALMER4E | 8 | 0 | Solve_Succee | Solve_Succee | 9.09e-06 | 18 | 25 | 206us | 7.1ms | 34.4x | PASS |
| PALMER4ENE | 8 | 23 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 33us | 265us | 8.0x | BOTH_FAIL |
| PALMER4NE | 4 | 23 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 33us | 302us | 9.1x | BOTH_FAIL |
| PALMER5A | 8 | 0 | Solve_Succee | Maximum_Iter | N/A | 2367 | 3000 | 13.4ms | 639.9ms | 47.6x | ipopt_FAIL |
| PALMER5ANE | 8 | 12 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 31us | 312us | 10.0x | BOTH_FAIL |
| PALMER5B | 9 | 0 | Solve_Succee | Solve_Succee | 1.88e-02 | 35 | 113 | 255us | 28.0ms | 109.9x | MISMATCH |
| PALMER5BNE | 9 | 12 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 31us | 315us | 10.0x | BOTH_FAIL |
| PALMER5C | 6 | 0 | Solve_Succee | Solve_Succee | 3.96e-15 | 1 | 1 | 26us | 794us | 30.5x | PASS |
| PALMER5D | 4 | 0 | Solve_Succee | Solve_Succee | 3.25e-16 | 1 | 1 | 23us | 806us | 35.2x | PASS |
| PALMER5E | 8 | 0 | Maximum_Iter | Maximum_Iter | N/A | 3000 | 3000 | 15.8ms | 479.2ms | 30.4x | BOTH_FAIL |
| PALMER5ENE | 8 | 12 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 35us | 213us | 6.1x | BOTH_FAIL |
| PALMER6A | 6 | 0 | Solve_Succee | Solve_Succee | 4.00e-12 | 171 | 105 | 791us | 26.2ms | 33.1x | PASS |
| PALMER6ANE | 6 | 13 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 29us | 306us | 10.4x | BOTH_FAIL |
| PALMER6C | 8 | 0 | Solve_Succee | Solve_Succee | 6.56e-15 | 1 | 1 | 30us | 793us | 26.2x | PASS |
| PALMER6E | 8 | 0 | Solve_Succee | Solve_Succee | 2.53e-11 | 23 | 30 | 185us | 8.9ms | 48.3x | PASS |
| PALMER6ENE | 8 | 13 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 19us | 302us | 15.5x | BOTH_FAIL |
| PALMER7A | 6 | 0 | Maximum_Iter | Maximum_Iter | N/A | 3000 | 3000 | 12.5ms | 486.9ms | 39.0x | BOTH_FAIL |
| PALMER7ANE | 6 | 13 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 32us | 218us | 6.9x | BOTH_FAIL |
| PALMER7C | 8 | 0 | Solve_Succee | Solve_Succee | 5.96e-14 | 1 | 1 | 28us | 813us | 28.7x | PASS |
| PALMER7E | 8 | 0 | Solve_Succee | Maximum_Iter | N/A | 981 | 3000 | 6.7ms | 628.5ms | 94.0x | ipopt_FAIL |
| PALMER7ENE | 8 | 13 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 31us | 241us | 7.8x | BOTH_FAIL |
| PALMER8A | 6 | 0 | Solve_Succee | Solve_Succee | 7.07e-14 | 31 | 36 | 161us | 11.4ms | 70.8x | PASS |
| PALMER8ANE | 6 | 12 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 31us | 328us | 10.5x | BOTH_FAIL |
| PALMER8C | 8 | 0 | Solve_Succee | Solve_Succee | 2.84e-14 | 1 | 1 | 29us | 817us | 27.9x | PASS |
| PALMER8E | 8 | 0 | Solve_Succee | Solve_Succee | 2.03e-11 | 25 | 23 | 187us | 7.3ms | 39.4x | PASS |
| PALMER8ENE | 8 | 12 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 32us | 304us | 9.5x | BOTH_FAIL |
| PARKCH | 15 | 0 | Solve_Succee | Solve_Succee | 5.92e-02 | 10 | 17 | 2.18s | 3.67s | 1.7x | MISMATCH |
| PENTAGON | 6 | 15 | Solve_Succee | Solve_Succee | 1.36e-19 | 19 | 19 | 5.7ms | 5.7ms | 1.0x | PASS |
| PFIT1 | 3 | 3 | Infeasible_P | Infeasible_P | N/A | 223 | 266 | 43.7ms | 45.4ms | 1.0x | BOTH_FAIL |
| PFIT1LS | 3 | 0 | Solve_Succee | Solve_Succee | 8.51e-13 | 306 | 263 | 501us | 51.5ms | 102.8x | PASS |
| PFIT2 | 3 | 3 | Solve_Succee | Restoration_ | N/A | 113 | 247 | 19.6ms | 48.7ms | 2.5x | ipopt_FAIL |
| PFIT2LS | 3 | 0 | Solve_Succee | Solve_Succee | 1.08e-13 | 85 | 82 | 96us | 18.1ms | 189.3x | PASS |
| PFIT3 | 3 | 3 | Infeasible_P | Solve_Succee | N/A | 4 | 133 | 13.7ms | 30.3ms | 2.2x | ripopt_FAIL |
| PFIT3LS | 3 | 0 | Solve_Succee | Solve_Succee | 4.02e-14 | 125 | 132 | 227us | 28.7ms | 126.5x | PASS |
| PFIT4 | 3 | 3 | Infeasible_P | Solve_Succee | N/A | 214 | 190 | 36.6ms | 36.1ms | 1.0x | ripopt_FAIL |
| PFIT4LS | 3 | 0 | Solve_Succee | Solve_Succee | 2.45e-14 | 245 | 215 | 434us | 43.4ms | 100.0x | PASS |
| POLAK1 | 3 | 2 | Solve_Succee | Solve_Succee | 0.00e+00 | 5 | 5 | 1.7ms | 1.8ms | 1.1x | PASS |
| POLAK2 | 11 | 2 | Solve_Succee | Solve_Succee | 4.61e-10 | 11 | 10 | 3.0ms | 3.2ms | 1.1x | PASS |
| POLAK3 | 12 | 10 | Restoration_ | Maximum_Iter | N/A | 109 | 3000 | 28.2ms | 649.4ms | 23.0x | BOTH_FAIL |
| POLAK4 | 3 | 3 | Solve_Succee | Solve_Succee | 1.52e-12 | 5 | 4 | 1.9ms | 1.6ms | 0.8x | PASS |
| POLAK5 | 3 | 2 | Solve_Succee | Solve_Succee | 1.06e-11 | 31 | 31 | 6.0ms | 7.5ms | 1.2x | PASS |
| POLAK6 | 5 | 4 | Solved_To_Ac | Maximum_Iter | N/A | 158 | 3000 | 33.0ms | 819.2ms | 24.8x | ipopt_FAIL |
| PORTFL1 | 12 | 1 | Solve_Succee | Solve_Succee | 0.00e+00 | 9 | 9 | 3.1ms | 3.1ms | 1.0x | PASS |
| PORTFL2 | 12 | 1 | Solve_Succee | Solve_Succee | 0.00e+00 | 8 | 8 | 3.1ms | 2.8ms | 0.9x | PASS |
| PORTFL3 | 12 | 1 | Solve_Succee | Solve_Succee | 0.00e+00 | 9 | 9 | 3.3ms | 3.3ms | 1.0x | PASS |
| PORTFL4 | 12 | 1 | Solve_Succee | Solve_Succee | 3.47e-18 | 8 | 8 | 3.4ms | 3.0ms | 0.9x | PASS |
| PORTFL6 | 12 | 1 | Solve_Succee | Solve_Succee | 3.47e-18 | 8 | 8 | 3.1ms | 2.9ms | 0.9x | PASS |
| POWELLBS | 2 | 2 | Solve_Succee | Solve_Succee | 0.00e+00 | 11 | 11 | 2.1ms | 2.5ms | 1.2x | PASS |
| POWELLBSLS | 2 | 0 | Solve_Succee | Solve_Succee | 3.73e-17 | 85 | 91 | 113us | 18.0ms | 159.2x | PASS |
| POWELLSQ | 2 | 2 | Infeasible_P | Infeasible_P | N/A | 37 | 29 | 7.2ms | 7.3ms | 1.0x | BOTH_FAIL |
| POWELLSQLS | 2 | 0 | Solve_Succee | Solve_Succee | 9.77e-20 | 10 | 10 | 29us | 3.1ms | 107.7x | PASS |
| PRICE3NE | 2 | 2 | Solve_Succee | Solve_Succee | 0.00e+00 | 7 | 7 | 1.7ms | 1.9ms | 1.1x | PASS |
| PRICE4 | 2 | 0 | Solve_Succee | Solve_Succee | 2.73e-19 | 9 | 8 | 30us | 2.3ms | 77.1x | PASS |
| PRICE4B | 2 | 0 | Solve_Succee | Solve_Succee | 1.41e-15 | 9 | 8 | 24us | 2.8ms | 118.2x | PASS |
| PRICE4NE | 2 | 2 | Solve_Succee | Solved_To_Ac | 0.00e+00 | 25 | 23 | 3.3ms | 5.5ms | 1.7x | PASS |
| PRODPL0 | 60 | 29 | Solve_Succee | Solve_Succee | 3.63e-16 | 15 | 15 | 7.3ms | 5.1ms | 0.7x | PASS |
| PRODPL1 | 60 | 29 | Solve_Succee | Solve_Succee | 1.99e-16 | 28 | 28 | 11.1ms | 8.6ms | 0.8x | PASS |
| PSPDOC | 4 | 0 | Solve_Succee | Solve_Succee | 4.43e-08 | 23 | 5 | 45us | 2.2ms | 49.4x | PASS |
| PT | 2 | 501 | Solve_Succee | Solve_Succee | 3.15e-10 | 82 | 106 | 79.0ms | 75.7ms | 1.0x | PASS |
| QC | 9 | 4 | Solve_Succee | Solve_Succee | 4.49e-11 | 44 | 44 | 10.2ms | 11.2ms | 1.1x | PASS |
| QCNEW | 9 | 3 | Solve_Succee | Solve_Succee | 1.52e-07 | 75 | 6 | 9.2ms | 1.9ms | 0.2x | PASS |
| QPCBLEND | 83 | 74 | Solve_Succee | Solve_Succee | 2.63e-10 | 19 | 19 | 10.3ms | 6.8ms | 0.7x | PASS |
| QPNBLEND | 83 | 74 | Solve_Succee | Solve_Succee | 8.67e-18 | 18 | 18 | 10.6ms | 6.7ms | 0.6x | PASS |
| RAT42 | 3 | 9 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 29us | 300us | 10.2x | BOTH_FAIL |
| RAT42LS | 3 | 0 | Solve_Succee | Solve_Succee | 1.32e-15 | 15 | 28 | 39us | 7.2ms | 185.8x | PASS |
| RAT43 | 4 | 15 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 19us | 218us | 11.7x | BOTH_FAIL |
| RAT43LS | 4 | 0 | Solve_Succee | Solve_Succee | 6.00e-15 | 18 | 34 | 143us | 8.0ms | 56.3x | PASS |
| RECIPE | 3 | 3 | Solve_Succee | Solve_Succee | 0.00e+00 | 16 | 16 | 2.4ms | 3.9ms | 1.6x | PASS |
| RECIPELS | 3 | 0 | Solve_Succee | Solve_Succee | 3.01e-09 | 17 | 29 | 43us | 6.7ms | 156.5x | PASS |
| RES | 20 | 14 | Solve_Succee | Solve_Succee | 0.00e+00 | 10 | 10 | 3.3ms | 3.1ms | 0.9x | PASS |
| RK23 | 17 | 11 | Infeasible_P | Solve_Succee | N/A | 286 | 10 | 63.3ms | 2.7ms | 0.0x | ripopt_FAIL |
| ROBOT | 14 | 2 | Solve_Succee | Search_Direc | N/A | 20 | 18 | 5.4ms | 5.5ms | 1.0x | ipopt_FAIL |
| ROSENBR | 2 | 0 | Solve_Succee | Solve_Succee | 0.00e+00 | 21 | 21 | 38us | 5.1ms | 133.8x | PASS |
| ROSENBRTU | 2 | 0 | Solve_Succee | Solve_Succee | 4.75e-20 | 93 | 87 | 103us | 18.9ms | 183.4x | PASS |
| ROSENMMX | 5 | 4 | Solve_Succee | Solve_Succee | 5.28e-14 | 13 | 13 | 3.9ms | 4.3ms | 1.1x | PASS |
| ROSZMAN1 | 4 | 25 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 30us | 312us | 10.3x | BOTH_FAIL |
| ROSZMAN1LS | 4 | 0 | Solve_Succee | Solve_Succee | 1.53e-01 | 15 | 28 | 106us | 7.6ms | 71.3x | MISMATCH |
| RSNBRNE | 2 | 2 | Solve_Succee | Solve_Succee | 0.00e+00 | 1 | 1 | 934us | 891us | 1.0x | PASS |
| S268 | 5 | 5 | Solve_Succee | Solve_Succee | 1.09e-11 | 14 | 14 | 4.2ms | 4.1ms | 1.0x | PASS |
| S308 | 2 | 0 | Solve_Succee | Solve_Succee | 1.11e-16 | 9 | 9 | 30us | 2.4ms | 83.0x | PASS |
| S308NE | 2 | 3 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 25us | 298us | 12.1x | BOTH_FAIL |
| S316-322 | 2 | 1 | Solve_Succee | Solve_Succee | 6.19e-12 | 7 | 7 | 1.9ms | 2.2ms | 1.2x | PASS |
| S365 | 7 | 5 | Restoration_ | Restoration_ | N/A | 0 | 1 | 614us | 1.4ms | 2.3x | BOTH_FAIL |
| S365MOD | 7 | 5 | Restoration_ | Restoration_ | N/A | 0 | 1 | 624us | 1.4ms | 2.2x | BOTH_FAIL |
| SANTA | 21 | 23 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 37us | 312us | 8.5x | BOTH_FAIL |
| SANTALS | 21 | 0 | Solve_Succee | Solve_Succee | 4.33e-17 | 33 | 31 | 306us | 10.6ms | 34.5x | PASS |
| SIM2BQP | 2 | 0 | Solve_Succee | Solve_Succee | 1.08e-07 | 30 | 5 | 45us | 1.9ms | 43.2x | PASS |
| SIMBQP | 2 | 0 | Solve_Succee | Solve_Succee | 1.08e-07 | 30 | 5 | 45us | 2.0ms | 45.3x | PASS |
| SIMPLLPA | 2 | 2 | Solve_Succee | Solve_Succee | 1.11e-16 | 8 | 8 | 2.9ms | 2.9ms | 1.0x | PASS |
| SIMPLLPB | 2 | 3 | Solve_Succee | Solve_Succee | 0.00e+00 | 7 | 7 | 2.5ms | 2.6ms | 1.0x | PASS |
| SINEVAL | 2 | 0 | Solve_Succee | Solve_Succee | 2.00e-40 | 42 | 42 | 63us | 10.4ms | 165.2x | PASS |
| SINVALNE | 2 | 2 | Solve_Succee | Solve_Succee | 0.00e+00 | 1 | 1 | 623us | 881us | 1.4x | PASS |
| SIPOW1 | 2 | 2000 | Solve_Succee | Solve_Succee | 9.34e-09 | 73 | 81 | 210.2ms | 201.1ms | 1.0x | PASS |
| SIPOW1M | 2 | 2000 | Solve_Succee | Solve_Succee | 2.91e-11 | 82 | 88 | 248.7ms | 235.6ms | 0.9x | PASS |
| SIPOW2 | 2 | 2000 | Solve_Succee | Solve_Succee | 1.87e-07 | 93 | 69 | 213.7ms | 176.1ms | 0.8x | PASS |
| SIPOW2M | 2 | 2000 | Solve_Succee | Solve_Succee | 6.46e-14 | 70 | 73 | 191.5ms | 183.7ms | 1.0x | PASS |
| SIPOW3 | 4 | 2000 | Solve_Succee | Solve_Succee | 9.18e-07 | 73 | 12 | 142.7ms | 36.4ms | 0.3x | PASS |
| SIPOW4 | 4 | 2000 | Solve_Succee | Solve_Succee | 3.59e-09 | 13 | 11 | 63.7ms | 34.5ms | 0.5x | PASS |
| SISSER | 2 | 0 | Solve_Succee | Solve_Succee | 1.56e-11 | 16 | 18 | 22us | 2.6ms | 115.4x | PASS |
| SISSER2 | 2 | 0 | Solve_Succee | Solve_Succee | 4.12e-11 | 16 | 20 | 21us | 2.9ms | 142.4x | PASS |
| SNAIL | 2 | 0 | Solve_Succee | Solve_Succee | 1.10e-28 | 64 | 63 | 52us | 9.9ms | 190.9x | PASS |
| SNAKE | 2 | 2 | Solve_Succee | Solve_Succee | 3.57e-15 | 8 | 8 | 2.0ms | 2.1ms | 1.1x | PASS |
| SPANHYD | 97 | 33 | Search_Direc | Solve_Succee | N/A | 21 | 20 | 9.0ms | 5.8ms | 0.6x | ripopt_FAIL |
| SPIRAL | 3 | 2 | Solved_To_Ac | Infeasible_P | N/A | 2440 | 370 | 194.5ms | 63.7ms | 0.3x | ipopt_FAIL |
| SSI | 3 | 0 | Maximum_Iter | Maximum_Iter | N/A | 3000 | 3000 | 1.8ms | 457.6ms | 250.7x | BOTH_FAIL |
| SSINE | 3 | 2 | Maximum_Iter | Solve_Succee | N/A | 2999 | 224 | 290.5ms | 35.9ms | 0.1x | ripopt_FAIL |
| STANCMIN | 3 | 2 | Solve_Succee | Solve_Succee | 0.00e+00 | 9 | 9 | 1.7ms | 2.0ms | 1.2x | PASS |
| STRATEC | 10 | 0 | Solve_Succee | Solve_Succee | 1.67e-14 | 21 | 34 | 1.96s | 3.19s | 1.6x | PASS |
| STREG | 4 | 0 | Solve_Succee | Solve_Succee | 8.90e-02 | 20 | 13 | 26us | 2.5ms | 95.6x | MISMATCH |
| STREGNE | 4 | 2 | Solve_Succee | Solve_Succee | 0.00e+00 | 2 | 2 | 704us | 861us | 1.2x | PASS |
| SUPERSIM | 2 | 2 | Solve_Succee | Solve_Succee | 2.22e-16 | 5 | 1 | 1.1ms | 726us | 0.7x | PASS |
| SWOPF | 83 | 92 | Solve_Succee | Solve_Succee | 6.84e-14 | 13 | 13 | 5.6ms | 4.1ms | 0.7x | PASS |
| SYNTHES1 | 6 | 6 | Solve_Succee | Solve_Succee | 3.55e-15 | 8 | 8 | 1.8ms | 2.0ms | 1.2x | PASS |
| SYNTHES2 | 11 | 14 | Solve_Succee | Solve_Succee | 8.88e-16 | 14 | 14 | 3.1ms | 3.1ms | 1.0x | PASS |
| SYNTHES3 | 17 | 23 | Solve_Succee | Solve_Succee | 0.00e+00 | 13 | 13 | 3.1ms | 3.2ms | 1.0x | PASS |
| TAME | 2 | 1 | Solve_Succee | Solve_Succee | 0.00e+00 | 5 | 5 | 879us | 1.4ms | 1.6x | PASS |
| TAX13322 | 72 | 1261 | Maximum_Iter | Maximum_Iter | N/A | 2999 | 3000 | 20.11s | 13.29s | 0.7x | BOTH_FAIL |
| TAXR13322 | 72 | 1261 | Maximum_Iter | Solved_To_Ac | N/A | 2999 | 56 | 20.27s | 2.71s | 0.1x | ripopt_FAIL |
| TENBARS1 | 18 | 9 | Solve_Succee | Solve_Succee | 0.00e+00 | 36 | 39 | 6.2ms | 7.6ms | 1.2x | PASS |
| TENBARS2 | 18 | 8 | Solve_Succee | Solve_Succee | 3.95e-16 | 31 | 33 | 5.3ms | 6.8ms | 1.3x | PASS |
| TENBARS3 | 18 | 8 | Solve_Succee | Solve_Succee | 6.07e-16 | 47 | 34 | 6.5ms | 6.9ms | 1.1x | PASS |
| TENBARS4 | 18 | 9 | Solve_Succee | Solve_Succee | 1.91e-16 | 14 | 14 | 2.8ms | 3.6ms | 1.3x | PASS |
| TFI1 | 3 | 101 | Solve_Succee | Solve_Succee | 1.10e-11 | 18 | 19 | 7.1ms | 7.0ms | 1.0x | PASS |
| TFI2 | 3 | 101 | Solve_Succee | Solve_Succee | 4.67e-09 | 8 | 8 | 2.8ms | 2.9ms | 1.0x | PASS |
| TFI3 | 3 | 101 | Solve_Succee | Solve_Succee | 2.26e-10 | 13 | 13 | 4.0ms | 4.3ms | 1.1x | PASS |
| THURBER | 7 | 37 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 22us | 228us | 10.6x | BOTH_FAIL |
| THURBERLS | 7 | 0 | Solve_Succee | Solve_Succee | 6.27e-01 | 28 | 19 | 305us | 3.7ms | 12.2x | MISMATCH |
| TOINTGOR | 50 | 0 | Solve_Succee | Solve_Succee | 0.00e+00 | 7 | 7 | 158us | 1.5ms | 9.5x | PASS |
| TOINTPSP | 50 | 0 | Solve_Succee | Solve_Succee | 0.00e+00 | 17 | 20 | 320us | 5.0ms | 15.6x | PASS |
| TOINTQOR | 50 | 0 | Solve_Succee | Solve_Succee | 1.93e-16 | 1 | 1 | 43us | 636us | 14.8x | PASS |
| TRIGGER | 7 | 6 | Solve_Succee | Solve_Succee | 0.00e+00 | 15 | 15 | 1.7ms | 2.8ms | 1.7x | PASS |
| TRO3X3 | 30 | 13 | Diverging_It | Solve_Succee | N/A | 247 | 47 | 53.6ms | 10.6ms | 0.2x | ripopt_FAIL |
| TRO4X4 | 63 | 25 | Diverging_It | Diverging_It | N/A | 464 | 157 | 178.5ms | 45.0ms | 0.3x | BOTH_FAIL |
| TRO6X2 | 45 | 21 | Infeasible_P | Restoration_ | N/A | 214 | 353 | 70.2ms | 94.2ms | 1.3x | BOTH_FAIL |
| TRUSPYR1 | 11 | 4 | Solve_Succee | Solve_Succee | 1.51e-12 | 10 | 10 | 2.0ms | 2.3ms | 1.2x | PASS |
| TRUSPYR2 | 11 | 11 | Solve_Succee | Solve_Succee | 1.20e-11 | 13 | 13 | 2.3ms | 3.1ms | 1.3x | PASS |
| TRY-B | 2 | 1 | Solve_Succee | Solve_Succee | 0.00e+00 | 23 | 23 | 3.4ms | 4.6ms | 1.4x | PASS |
| TWOBARS | 2 | 2 | Solve_Succee | Solve_Succee | 0.00e+00 | 8 | 8 | 1.6ms | 1.9ms | 1.2x | PASS |
| VESUVIA | 8 | 1025 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 64us | 404us | 6.3x | BOTH_FAIL |
| VESUVIALS | 8 | 0 | Solve_Succee | Solve_Succee | 0.00e+00 | 341 | 48 | 73.9ms | 18.8ms | 0.3x | PASS |
| VESUVIO | 8 | 1025 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 65us | 391us | 6.0x | BOTH_FAIL |
| VESUVIOLS | 8 | 0 | Solve_Succee | Solve_Succee | 2.41e-15 | 19 | 10 | 6.1ms | 4.7ms | 0.8x | PASS |
| VESUVIOU | 8 | 1025 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 63us | 397us | 6.3x | BOTH_FAIL |
| VESUVIOULS | 8 | 0 | Solve_Succee | Solve_Succee | 4.44e-16 | 14 | 8 | 3.3ms | 3.6ms | 1.1x | PASS |
| VIBRBEAM | 8 | 0 | Solve_Succee | Solve_Succee | 9.14e-01 | 21 | 58 | 421us | 9.8ms | 23.3x | MISMATCH |
| VIBRBEAMNE | 8 | 30 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 24us | 224us | 9.5x | BOTH_FAIL |
| WACHBIEG | 3 | 2 | Infeasible_P | Infeasible_P | N/A | 10 | 15 | 3.3ms | 3.7ms | 1.1x | BOTH_FAIL |
| WATER | 31 | 10 | Solve_Succee | Solve_Succee | 3.31e-12 | 19 | 17 | 4.0ms | 3.6ms | 0.9x | PASS |
| WAYSEA1 | 2 | 0 | Solve_Succee | Solve_Succee | 3.94e-31 | 14 | 14 | 19us | 2.0ms | 108.5x | PASS |
| WAYSEA1B | 2 | 0 | Solve_Succee | Solve_Succee | 1.89e-12 | 14 | 14 | 20us | 2.7ms | 135.0x | PASS |
| WAYSEA1NE | 2 | 2 | Solve_Succee | Solve_Succee | 0.00e+00 | 7 | 7 | 1.1ms | 1.3ms | 1.2x | PASS |
| WAYSEA2 | 2 | 0 | Solve_Succee | Solve_Succee | 1.61e-11 | 21 | 22 | 23us | 3.0ms | 130.6x | PASS |
| WAYSEA2B | 2 | 0 | Solve_Succee | Solve_Succee | 1.61e-11 | 21 | 22 | 23us | 4.0ms | 173.1x | PASS |
| WAYSEA2NE | 2 | 2 | Solve_Succee | Solve_Succee | 0.00e+00 | 11 | 11 | 2.2ms | 2.7ms | 1.2x | PASS |
| WEEDS | 3 | 0 | Solve_Succee | Solve_Succee | 1.24e-14 | 34 | 28 | 130us | 7.9ms | 61.0x | PASS |
| WEEDSNE | 3 | 12 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 30us | 301us | 9.9x | BOTH_FAIL |
| WOMFLET | 3 | 3 | Solve_Succee | Solve_Succee | 1.44e-10 | 10 | 8 | 3.3ms | 2.6ms | 0.8x | PASS |
| YFIT | 3 | 0 | Solve_Succee | Solve_Succee | 6.17e-15 | 41 | 36 | 107us | 9.8ms | 91.4x | PASS |
| YFITNE | 3 | 17 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 30us | 223us | 7.5x | BOTH_FAIL |
| YFITU | 3 | 0 | Solve_Succee | Solve_Succee | 2.70e-17 | 35 | 36 | 163us | 8.0ms | 49.3x | PASS |
| ZANGWIL2 | 2 | 0 | Solve_Succee | Solve_Succee | 0.00e+00 | 1 | 1 | 18us | 800us | 44.2x | PASS |
| ZANGWIL3 | 3 | 3 | Solve_Succee | Solve_Succee | 0.00e+00 | 1 | 1 | 822us | 814us | 1.0x | PASS |
| ZECEVIC2 | 2 | 2 | Solve_Succee | Solve_Succee | 0.00e+00 | 9 | 8 | 2.7ms | 2.7ms | 1.0x | PASS |
| ZECEVIC3 | 2 | 2 | Solve_Succee | Solve_Succee | 0.00e+00 | 17 | 17 | 4.3ms | 4.6ms | 1.1x | PASS |
| ZECEVIC4 | 2 | 2 | Solve_Succee | Solve_Succee | 0.00e+00 | 10 | 10 | 3.3ms | 3.4ms | 1.0x | PASS |
| ZY2 | 3 | 2 | Solve_Succee | Solve_Succee | 0.00e+00 | 14 | 14 | 4.4ms | 4.4ms | 1.0x | PASS |

## Performance Comparison (where both solve)

### Iteration Comparison

| Metric | ripopt | Ipopt |
|--------|--------|-------|
| Mean   | 47.3 | 44.5 |
| Median | 13 | 13 |
| Total  | 26047 | 24529 |

- ripopt fewer iterations: 126/551
- Ipopt fewer iterations: 152/551
- Tied: 273/551

### Timing Comparison

| Metric | ripopt | Ipopt |
|--------|--------|-------|
| Mean   | 27.0ms | 37.6ms |
| Median | 1.7ms | 3.1ms |
| Total  | 14.86s | 20.74s |

- Geometric mean speedup (Ipopt_time/ripopt_time): **4.32x**
  - \>1 means ripopt is faster, <1 means Ipopt is faster
- ripopt faster: 415/551 problems
- Ipopt faster: 136/551 problems
- Overall speedup (total time): 1.40x

## Failure Analysis

### Problems where only ripopt fails (11)

| Problem | n | m | ripopt status | Ipopt obj |
|---------|---|---|---------------|-----------|
| BA-L1SP | 57 | 12 | Error_In_Step_Computation | 0.000000e+00 |
| BYRDSPHR | 3 | 2 | Infeasible_Problem_Detected | -4.683300e+00 |
| CRESC50 | 6 | 100 | Infeasible_Problem_Detected | 7.862467e-01 |
| DMN15102LS | 66 | 0 | Timeout | 6.637446e+02 |
| PFIT3 | 3 | 3 | Infeasible_Problem_Detected | 0.000000e+00 |
| PFIT4 | 3 | 3 | Infeasible_Problem_Detected | 0.000000e+00 |
| RK23 | 17 | 11 | Infeasible_Problem_Detected | 8.333327e-02 |
| SPANHYD | 97 | 33 | Search_Direction_Becomes_Too_Small | 2.397380e+02 |
| SSINE | 3 | 2 | Maximum_Iterations_Exceeded | 0.000000e+00 |
| TAXR13322 | 72 | 1261 | Maximum_Iterations_Exceeded | -3.429089e+02 |
| TRO3X3 | 30 | 13 | Diverging_Iterates | 8.967478e+00 |

### Problems where only Ipopt fails (9)

| Problem | n | m | Ipopt status | ripopt obj |
|---------|---|---|--------------|------------|
| BLEACHNG | 17 | 0 | Timeout | 1.823872e+04 |
| DECONVB | 63 | 0 | Maximum_Iterations_Exceeded | 4.991591e-03 |
| HIMMELBJ | 45 | 14 | Error_In_Step_Computation | N/A |
| PALMER5A | 8 | 0 | Maximum_Iterations_Exceeded | 4.589404e-02 |
| PALMER7E | 8 | 0 | Maximum_Iterations_Exceeded | 6.777943e+00 |
| PFIT2 | 3 | 3 | Restoration_Failed | 0.000000e+00 |
| POLAK6 | 5 | 4 | Maximum_Iterations_Exceeded | 3.870766e+02 |
| ROBOT | 14 | 2 | Search_Direction_Becomes_Too_Small | 6.593299e+00 |
| SPIRAL | 3 | 2 | Infeasible_Problem_Detected | 1.475539e+03 |

### Problems where both fail (156)

| Problem | n | m | ripopt status | Ipopt status |
|---------|---|---|---------------|--------------|
| ARGAUSS | 3 | 15 | Not_Enough_Degrees_Of_Freedom | Not_Enough_Degrees_Of_Freedom |
| AVION2 | 49 | 15 | Maximum_Iterations_Exceeded | Maximum_Iterations_Exceeded |
| BARDNE | 3 | 15 | Not_Enough_Degrees_Of_Freedom | Not_Enough_Degrees_Of_Freedom |
| BEALENE | 2 | 3 | Not_Enough_Degrees_Of_Freedom | Not_Enough_Degrees_Of_Freedom |
| BENNETT5 | 3 | 154 | Not_Enough_Degrees_Of_Freedom | Not_Enough_Degrees_Of_Freedom |
| BIGGS6NE | 6 | 13 | Not_Enough_Degrees_Of_Freedom | Not_Enough_Degrees_Of_Freedom |
| BOX3NE | 3 | 10 | Not_Enough_Degrees_Of_Freedom | Not_Enough_Degrees_Of_Freedom |
| BOXBOD | 2 | 6 | Not_Enough_Degrees_Of_Freedom | Not_Enough_Degrees_Of_Freedom |
| BROWNBSNE | 2 | 3 | Not_Enough_Degrees_Of_Freedom | Not_Enough_Degrees_Of_Freedom |
| BROWNDENE | 4 | 20 | Not_Enough_Degrees_Of_Freedom | Not_Enough_Degrees_Of_Freedom |
| BURKEHAN | 1 | 1 | Infeasible_Problem_Detected | Infeasible_Problem_Detected |
| CERI651A | 7 | 61 | Not_Enough_Degrees_Of_Freedom | Not_Enough_Degrees_Of_Freedom |
| CERI651B | 7 | 66 | Not_Enough_Degrees_Of_Freedom | Not_Enough_Degrees_Of_Freedom |
| CERI651C | 7 | 56 | Not_Enough_Degrees_Of_Freedom | Not_Enough_Degrees_Of_Freedom |
| CERI651D | 7 | 67 | Not_Enough_Degrees_Of_Freedom | Not_Enough_Degrees_Of_Freedom |
| CERI651E | 7 | 64 | Not_Enough_Degrees_Of_Freedom | Not_Enough_Degrees_Of_Freedom |
| CHWIRUT1 | 3 | 214 | Not_Enough_Degrees_Of_Freedom | Not_Enough_Degrees_Of_Freedom |
| CHWIRUT2 | 3 | 54 | Not_Enough_Degrees_Of_Freedom | Not_Enough_Degrees_Of_Freedom |
| CRESC100 | 6 | 200 | Infeasible_Problem_Detected | Infeasible_Problem_Detected |
| CRESC132 | 6 | 2654 | Infeasible_Problem_Detected | Timeout |
| DANIWOOD | 2 | 6 | Not_Enough_Degrees_Of_Freedom | Not_Enough_Degrees_Of_Freedom |
| DANWOOD | 2 | 6 | Not_Enough_Degrees_Of_Freedom | Not_Enough_Degrees_Of_Freedom |
| DENSCHNBNE | 2 | 3 | Not_Enough_Degrees_Of_Freedom | Not_Enough_Degrees_Of_Freedom |
| DENSCHNENE | 3 | 3 | Infeasible_Problem_Detected | Infeasible_Problem_Detected |
| DEVGLA1NE | 4 | 24 | Not_Enough_Degrees_Of_Freedom | Not_Enough_Degrees_Of_Freedom |
| DEVGLA2NE | 5 | 16 | Not_Enough_Degrees_Of_Freedom | Not_Enough_Degrees_Of_Freedom |
| DIAMON2D | 66 | 4643 | Not_Enough_Degrees_Of_Freedom | Not_Enough_Degrees_Of_Freedom |
| DIAMON2DLS | 66 | 0 | Timeout | Timeout |
| DIAMON3D | 99 | 4643 | Not_Enough_Degrees_Of_Freedom | Not_Enough_Degrees_Of_Freedom |
| DIAMON3DLS | 99 | 0 | Timeout | Timeout |
| DMN15102 | 66 | 4643 | Not_Enough_Degrees_Of_Freedom | Not_Enough_Degrees_Of_Freedom |
| DMN15103 | 99 | 4643 | Not_Enough_Degrees_Of_Freedom | Not_Enough_Degrees_Of_Freedom |
| DMN15103LS | 99 | 0 | Search_Direction_Becomes_Too_Small | Timeout |
| DMN15332 | 66 | 4643 | Not_Enough_Degrees_Of_Freedom | Not_Enough_Degrees_Of_Freedom |
| DMN15332LS | 66 | 0 | Timeout | Timeout |
| DMN15333 | 99 | 4643 | Not_Enough_Degrees_Of_Freedom | Not_Enough_Degrees_Of_Freedom |
| DMN15333LS | 99 | 0 | Timeout | Timeout |
| DMN37142 | 66 | 4643 | Not_Enough_Degrees_Of_Freedom | Not_Enough_Degrees_Of_Freedom |
| DMN37142LS | 66 | 0 | Error_In_Step_Computation | Timeout |
| DMN37143 | 99 | 4643 | Not_Enough_Degrees_Of_Freedom | Not_Enough_Degrees_Of_Freedom |
| DMN37143LS | 99 | 0 | Timeout | Timeout |
| ECKERLE4 | 3 | 35 | Not_Enough_Degrees_Of_Freedom | Not_Enough_Degrees_Of_Freedom |
| EGGCRATENE | 2 | 4 | Not_Enough_Degrees_Of_Freedom | Not_Enough_Degrees_Of_Freedom |
| ELATVIDUNE | 2 | 3 | Not_Enough_Degrees_Of_Freedom | Not_Enough_Degrees_Of_Freedom |
| ENGVAL2NE | 3 | 5 | Not_Enough_Degrees_Of_Freedom | Not_Enough_Degrees_Of_Freedom |
| ENSO | 9 | 168 | Not_Enough_Degrees_Of_Freedom | Not_Enough_Degrees_Of_Freedom |
| EQC | 9 | 3 | Search_Direction_Becomes_Too_Small | Error_In_Step_Computation |
| EXP2NE | 2 | 10 | Not_Enough_Degrees_Of_Freedom | Not_Enough_Degrees_Of_Freedom |
| EXPFITNE | 2 | 10 | Not_Enough_Degrees_Of_Freedom | Not_Enough_Degrees_Of_Freedom |
| FBRAIN | 2 | 2211 | Not_Enough_Degrees_Of_Freedom | Not_Enough_Degrees_Of_Freedom |
| FBRAIN2 | 4 | 2211 | Not_Enough_Degrees_Of_Freedom | Not_Enough_Degrees_Of_Freedom |
| FBRAIN2NE | 4 | 2211 | Not_Enough_Degrees_Of_Freedom | Not_Enough_Degrees_Of_Freedom |
| FBRAIN3 | 6 | 2211 | Not_Enough_Degrees_Of_Freedom | Not_Enough_Degrees_Of_Freedom |
| FBRAIN3LS | 6 | 0 | Maximum_Iterations_Exceeded | Maximum_Iterations_Exceeded |
| FBRAINNE | 2 | 2211 | Not_Enough_Degrees_Of_Freedom | Not_Enough_Degrees_Of_Freedom |
| GAUSS1 | 8 | 250 | Not_Enough_Degrees_Of_Freedom | Not_Enough_Degrees_Of_Freedom |
| GAUSS2 | 8 | 250 | Not_Enough_Degrees_Of_Freedom | Not_Enough_Degrees_Of_Freedom |
| GAUSS3 | 8 | 250 | Not_Enough_Degrees_Of_Freedom | Not_Enough_Degrees_Of_Freedom |
| GBRAIN | 2 | 2200 | Not_Enough_Degrees_Of_Freedom | Not_Enough_Degrees_Of_Freedom |
| GROUPING | 100 | 125 | Not_Enough_Degrees_Of_Freedom | Not_Enough_Degrees_Of_Freedom |
| GROWTH | 3 | 12 | Not_Enough_Degrees_Of_Freedom | Not_Enough_Degrees_Of_Freedom |
| GULFNE | 3 | 99 | Not_Enough_Degrees_Of_Freedom | Not_Enough_Degrees_Of_Freedom |
| HAHN1 | 7 | 236 | Not_Enough_Degrees_Of_Freedom | Not_Enough_Degrees_Of_Freedom |
| HATFLDBNE | 4 | 4 | Infeasible_Problem_Detected | Infeasible_Problem_Detected |
| HATFLDDNE | 3 | 10 | Not_Enough_Degrees_Of_Freedom | Not_Enough_Degrees_Of_Freedom |
| HATFLDENE | 3 | 21 | Not_Enough_Degrees_Of_Freedom | Not_Enough_Degrees_Of_Freedom |
| HIMMELBD | 2 | 2 | Infeasible_Problem_Detected | Infeasible_Problem_Detected |
| HIMMELBFNE | 4 | 7 | Not_Enough_Degrees_Of_Freedom | Not_Enough_Degrees_Of_Freedom |
| HS25NE | 3 | 99 | Not_Enough_Degrees_Of_Freedom | Not_Enough_Degrees_Of_Freedom |
| HS2NE | 2 | 2 | Infeasible_Problem_Detected | Infeasible_Problem_Detected |
| HS87 | 6 | 4 | Maximum_Iterations_Exceeded | Maximum_Iterations_Exceeded |
| JENSMPNE | 2 | 10 | Not_Enough_Degrees_Of_Freedom | Not_Enough_Degrees_Of_Freedom |
| JUDGENE | 2 | 20 | Not_Enough_Degrees_Of_Freedom | Not_Enough_Degrees_Of_Freedom |
| KIRBY2 | 5 | 151 | Not_Enough_Degrees_Of_Freedom | Not_Enough_Degrees_Of_Freedom |
| KOEBHELBNE | 3 | 156 | Not_Enough_Degrees_Of_Freedom | Not_Enough_Degrees_Of_Freedom |
| KOWOSBNE | 4 | 11 | Not_Enough_Degrees_Of_Freedom | Not_Enough_Degrees_Of_Freedom |
| LANCZOS1 | 6 | 24 | Not_Enough_Degrees_Of_Freedom | Not_Enough_Degrees_Of_Freedom |
| LANCZOS2 | 6 | 24 | Not_Enough_Degrees_Of_Freedom | Not_Enough_Degrees_Of_Freedom |
| LANCZOS3 | 6 | 24 | Not_Enough_Degrees_Of_Freedom | Not_Enough_Degrees_Of_Freedom |
| LEVYMONE10 | 10 | 20 | Not_Enough_Degrees_Of_Freedom | Not_Enough_Degrees_Of_Freedom |
| LEVYMONE5 | 2 | 4 | Not_Enough_Degrees_Of_Freedom | Not_Enough_Degrees_Of_Freedom |
| LEVYMONE6 | 3 | 6 | Not_Enough_Degrees_Of_Freedom | Not_Enough_Degrees_Of_Freedom |
| LEVYMONE7 | 4 | 8 | Not_Enough_Degrees_Of_Freedom | Not_Enough_Degrees_Of_Freedom |
| LEVYMONE8 | 5 | 10 | Not_Enough_Degrees_Of_Freedom | Not_Enough_Degrees_Of_Freedom |
| LEVYMONE9 | 8 | 16 | Not_Enough_Degrees_Of_Freedom | Not_Enough_Degrees_Of_Freedom |
| LEWISPOL | 6 | 9 | Not_Enough_Degrees_Of_Freedom | Not_Enough_Degrees_Of_Freedom |
| LHAIFAM | 99 | 150 | Restoration_Failed | Invalid_Number_Detected |
| LSC1 | 3 | 6 | Not_Enough_Degrees_Of_Freedom | Not_Enough_Degrees_Of_Freedom |
| LSC2 | 3 | 6 | Not_Enough_Degrees_Of_Freedom | Not_Enough_Degrees_Of_Freedom |
| MESH | 41 | 48 | Diverging_Iterates | Diverging_Iterates |
| MEYER3NE | 3 | 16 | Not_Enough_Degrees_Of_Freedom | Not_Enough_Degrees_Of_Freedom |
| MGH09 | 4 | 11 | Not_Enough_Degrees_Of_Freedom | Not_Enough_Degrees_Of_Freedom |
| MGH10 | 3 | 16 | Not_Enough_Degrees_Of_Freedom | Not_Enough_Degrees_Of_Freedom |
| MGH10S | 3 | 16 | Not_Enough_Degrees_Of_Freedom | Not_Enough_Degrees_Of_Freedom |
| MGH17 | 5 | 33 | Not_Enough_Degrees_Of_Freedom | Not_Enough_Degrees_Of_Freedom |
| MGH17S | 5 | 33 | Not_Enough_Degrees_Of_Freedom | Not_Enough_Degrees_Of_Freedom |
| MISRA1A | 2 | 14 | Not_Enough_Degrees_Of_Freedom | Not_Enough_Degrees_Of_Freedom |
| MISRA1B | 2 | 14 | Not_Enough_Degrees_Of_Freedom | Not_Enough_Degrees_Of_Freedom |
| MISRA1C | 2 | 14 | Not_Enough_Degrees_Of_Freedom | Not_Enough_Degrees_Of_Freedom |
| MISRA1D | 2 | 14 | Not_Enough_Degrees_Of_Freedom | Not_Enough_Degrees_Of_Freedom |
| MUONSINE | 1 | 512 | Not_Enough_Degrees_Of_Freedom | Not_Enough_Degrees_Of_Freedom |
| NASH | 72 | 24 | Not_Enough_Degrees_Of_Freedom | Infeasible_Problem_Detected |
| NELSON | 3 | 128 | Not_Enough_Degrees_Of_Freedom | Not_Enough_Degrees_Of_Freedom |
| NYSTROM5 | 18 | 20 | Not_Enough_Degrees_Of_Freedom | Not_Enough_Degrees_Of_Freedom |
| NYSTROM5C | 18 | 20 | Not_Enough_Degrees_Of_Freedom | Not_Enough_Degrees_Of_Freedom |
| OSBORNE1 | 5 | 33 | Not_Enough_Degrees_Of_Freedom | Not_Enough_Degrees_Of_Freedom |
| OSBORNE2 | 11 | 65 | Not_Enough_Degrees_Of_Freedom | Not_Enough_Degrees_Of_Freedom |
| PALMER1ANE | 6 | 35 | Not_Enough_Degrees_Of_Freedom | Not_Enough_Degrees_Of_Freedom |
| PALMER1BNE | 4 | 35 | Not_Enough_Degrees_Of_Freedom | Not_Enough_Degrees_Of_Freedom |
| PALMER1ENE | 8 | 35 | Not_Enough_Degrees_Of_Freedom | Not_Enough_Degrees_Of_Freedom |
| PALMER1NE | 4 | 31 | Not_Enough_Degrees_Of_Freedom | Not_Enough_Degrees_Of_Freedom |
| PALMER2ANE | 6 | 23 | Not_Enough_Degrees_Of_Freedom | Not_Enough_Degrees_Of_Freedom |
| PALMER2BNE | 4 | 23 | Not_Enough_Degrees_Of_Freedom | Not_Enough_Degrees_Of_Freedom |
| PALMER2ENE | 8 | 23 | Not_Enough_Degrees_Of_Freedom | Not_Enough_Degrees_Of_Freedom |
| PALMER2NE | 4 | 23 | Not_Enough_Degrees_Of_Freedom | Not_Enough_Degrees_Of_Freedom |
| PALMER3ANE | 6 | 23 | Not_Enough_Degrees_Of_Freedom | Not_Enough_Degrees_Of_Freedom |
| PALMER3BNE | 4 | 23 | Not_Enough_Degrees_Of_Freedom | Not_Enough_Degrees_Of_Freedom |
| PALMER3ENE | 8 | 23 | Not_Enough_Degrees_Of_Freedom | Not_Enough_Degrees_Of_Freedom |
| PALMER3NE | 4 | 23 | Not_Enough_Degrees_Of_Freedom | Not_Enough_Degrees_Of_Freedom |
| PALMER4ANE | 6 | 23 | Not_Enough_Degrees_Of_Freedom | Not_Enough_Degrees_Of_Freedom |
| PALMER4BNE | 4 | 23 | Not_Enough_Degrees_Of_Freedom | Not_Enough_Degrees_Of_Freedom |
| PALMER4ENE | 8 | 23 | Not_Enough_Degrees_Of_Freedom | Not_Enough_Degrees_Of_Freedom |
| PALMER4NE | 4 | 23 | Not_Enough_Degrees_Of_Freedom | Not_Enough_Degrees_Of_Freedom |
| PALMER5ANE | 8 | 12 | Not_Enough_Degrees_Of_Freedom | Not_Enough_Degrees_Of_Freedom |
| PALMER5BNE | 9 | 12 | Not_Enough_Degrees_Of_Freedom | Not_Enough_Degrees_Of_Freedom |
| PALMER5E | 8 | 0 | Maximum_Iterations_Exceeded | Maximum_Iterations_Exceeded |
| PALMER5ENE | 8 | 12 | Not_Enough_Degrees_Of_Freedom | Not_Enough_Degrees_Of_Freedom |
| PALMER6ANE | 6 | 13 | Not_Enough_Degrees_Of_Freedom | Not_Enough_Degrees_Of_Freedom |
| PALMER6ENE | 8 | 13 | Not_Enough_Degrees_Of_Freedom | Not_Enough_Degrees_Of_Freedom |
| PALMER7A | 6 | 0 | Maximum_Iterations_Exceeded | Maximum_Iterations_Exceeded |
| PALMER7ANE | 6 | 13 | Not_Enough_Degrees_Of_Freedom | Not_Enough_Degrees_Of_Freedom |
| PALMER7ENE | 8 | 13 | Not_Enough_Degrees_Of_Freedom | Not_Enough_Degrees_Of_Freedom |
| PALMER8ANE | 6 | 12 | Not_Enough_Degrees_Of_Freedom | Not_Enough_Degrees_Of_Freedom |
| PALMER8ENE | 8 | 12 | Not_Enough_Degrees_Of_Freedom | Not_Enough_Degrees_Of_Freedom |
| PFIT1 | 3 | 3 | Infeasible_Problem_Detected | Infeasible_Problem_Detected |
| POLAK3 | 12 | 10 | Restoration_Failed | Maximum_Iterations_Exceeded |
| POWELLSQ | 2 | 2 | Infeasible_Problem_Detected | Infeasible_Problem_Detected |
| RAT42 | 3 | 9 | Not_Enough_Degrees_Of_Freedom | Not_Enough_Degrees_Of_Freedom |
| RAT43 | 4 | 15 | Not_Enough_Degrees_Of_Freedom | Not_Enough_Degrees_Of_Freedom |
| ROSZMAN1 | 4 | 25 | Not_Enough_Degrees_Of_Freedom | Not_Enough_Degrees_Of_Freedom |
| S308NE | 2 | 3 | Not_Enough_Degrees_Of_Freedom | Not_Enough_Degrees_Of_Freedom |
| S365 | 7 | 5 | Restoration_Failed | Restoration_Failed |
| S365MOD | 7 | 5 | Restoration_Failed | Restoration_Failed |
| SANTA | 21 | 23 | Not_Enough_Degrees_Of_Freedom | Not_Enough_Degrees_Of_Freedom |
| SSI | 3 | 0 | Maximum_Iterations_Exceeded | Maximum_Iterations_Exceeded |
| TAX13322 | 72 | 1261 | Maximum_Iterations_Exceeded | Maximum_Iterations_Exceeded |
| THURBER | 7 | 37 | Not_Enough_Degrees_Of_Freedom | Not_Enough_Degrees_Of_Freedom |
| TRO4X4 | 63 | 25 | Diverging_Iterates | Diverging_Iterates |
| TRO6X2 | 45 | 21 | Infeasible_Problem_Detected | Restoration_Failed |
| VESUVIA | 8 | 1025 | Not_Enough_Degrees_Of_Freedom | Not_Enough_Degrees_Of_Freedom |
| VESUVIO | 8 | 1025 | Not_Enough_Degrees_Of_Freedom | Not_Enough_Degrees_Of_Freedom |
| VESUVIOU | 8 | 1025 | Not_Enough_Degrees_Of_Freedom | Not_Enough_Degrees_Of_Freedom |
| VIBRBEAMNE | 8 | 30 | Not_Enough_Degrees_Of_Freedom | Not_Enough_Degrees_Of_Freedom |
| WACHBIEG | 3 | 2 | Infeasible_Problem_Detected | Infeasible_Problem_Detected |
| WEEDSNE | 3 | 12 | Not_Enough_Degrees_Of_Freedom | Not_Enough_Degrees_Of_Freedom |
| YFITNE | 3 | 17 | Not_Enough_Degrees_Of_Freedom | Not_Enough_Degrees_Of_Freedom |

### Objective mismatches (31)

Both solvers converged but found different objective values (rel diff > 1e-4).

- **Different local minimum** (both Optimal): 0
- **Convergence gap** (one Acceptable): 31
- **Better objective found by**: ripopt 10, Ipopt 21

| Problem | ripopt obj | Ipopt obj | Rel Diff | r_status | i_status | Better |
|---------|-----------|-----------|----------|----------|----------|--------|
| HS25 | 3.283500e+01 | 3.082490e-20 | 1.00e+00 | Solve_Succ | Solve_Succ | ipopt |
| LEVYMONT5 | 1.248706e+01 | 1.239502e-25 | 1.00e+00 | Solve_Succ | Solve_Succ | ipopt |
| HALDMADS | 1.223614e-04 | 2.218282e+00 | 1.00e+00 | Solve_Succ | Solve_Succ | ripopt |
| MGH17LS | 1.022432e+00 | 7.898394e-05 | 1.00e+00 | Solve_Succ | Solve_Succ | ipopt |
| PALMER1A | 5.975432e+01 | 8.988363e-02 | 9.98e-01 | Solve_Succ | Solve_Succ | ipopt |
| ELATTAR | 1.427073e-01 | 7.420618e+01 | 9.98e-01 | Solve_Succ | Solve_Succ | ripopt |
| HS84 | -1.175337e+09 | -5.280335e+06 | 9.96e-01 | Solve_Succ | Solve_Succ | ripopt |
| MSS1 | -2.641499e-01 | -1.400000e+01 | 9.81e-01 | Solved_To_ | Solve_Succ | ipopt |
| MGH17SLS | 1.022041e+00 | 2.451788e-02 | 9.76e-01 | Solve_Succ | Solve_Succ | ipopt |
| HAHN1LS | 1.532438e+00 | 3.338424e+01 | 9.54e-01 | Solve_Succ | Solve_Succ | ripopt |
| VIBRBEAM | 3.866826e+00 | 3.322376e-01 | 9.14e-01 | Solve_Succ | Solve_Succ | ipopt |
| BOXBODLS | 1.168009e+03 | 9.771500e+03 | 8.80e-01 | Solve_Succ | Solve_Succ | ripopt |
| OET4 | 8.577575e-01 | 4.295421e-03 | 8.53e-01 | Solve_Succ | Solve_Succ | ipopt |
| CAMEL6 | -2.154638e-01 | -1.031628e+00 | 7.91e-01 | Solve_Succ | Solve_Succ | ipopt |
| ECKERLE4LS | 6.996962e-01 | 1.463589e-03 | 6.98e-01 | Solve_Succ | Solve_Succ | ipopt |
| THURBERLS | 1.512117e+04 | 5.642708e+03 | 6.27e-01 | Solve_Succ | Solve_Succ | ipopt |
| MUONSINELS | 2.549180e+04 | 4.387412e+04 | 4.19e-01 | Solve_Succ | Solve_Succ | ripopt |
| KOEBHELB | 1.122226e+02 | 7.751635e+01 | 3.09e-01 | Solve_Succ | Solve_Succ | ipopt |
| EG1 | -1.132801e+00 | -1.429307e+00 | 2.07e-01 | Solve_Succ | Solve_Succ | ipopt |
| HS70 | 1.863491e-01 | 7.498464e-03 | 1.79e-01 | Solve_Succ | Solve_Succ | ipopt |
| ROSZMAN1LS | 1.531345e-01 | 4.948485e-04 | 1.53e-01 | Solve_Succ | Solve_Succ | ipopt |
| STREG | 8.517075e-12 | 8.901950e-02 | 8.90e-02 | Solve_Succ | Solve_Succ | ripopt |
| LSNNODOC | 1.336336e+02 | 1.231124e+02 | 7.87e-02 | Solve_Succ | Solve_Succ | ipopt |
| PALMER3 | 2.265958e+03 | 2.416980e+03 | 6.25e-02 | Solve_Succ | Solve_Succ | ripopt |
| PARKCH | 1.725842e+03 | 1.623743e+03 | 5.92e-02 | Solve_Succ | Solve_Succ | ipopt |
| ACOPR30 | 5.960093e+02 | 5.768924e+02 | 3.21e-02 | Solved_To_ | Solve_Succ | ipopt |
| PALMER5B | 2.856549e-02 | 9.752496e-03 | 1.88e-02 | Solve_Succ | Solve_Succ | ipopt |
| CLIFF | 1.997866e-01 | 2.072380e-01 | 7.45e-03 | Solve_Succ | Solve_Succ | ripopt |
| LRCOVTYPE | 5.753182e-01 | 5.723072e-01 | 3.01e-03 | Solve_Succ | Solve_Succ | ipopt |
| MGH09LS | 1.521328e-03 | 3.075056e-04 | 1.21e-03 | Solve_Succ | Solve_Succ | ipopt |
| DENSCHND | 2.861336e-05 | 2.221899e-04 | 1.94e-04 | Solve_Succ | Solve_Succ | ripopt |

---
*Generated by benchmarks/cutest/compare.py*