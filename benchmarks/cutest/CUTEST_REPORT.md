# CUTEst Benchmark Report

Comparison of pounce vs Ipopt (C++) on the CUTEst test set.

## Executive Summary

- **Total problems**: 727
- **pounce solved**: 562/727 (77.3%)
- **Ipopt solved**: 562/727 (77.3%)
- **Both solved**: 557/727
- **Matching solutions** (rel obj diff < 1e-4): 547/557

## Accuracy Statistics (where both solve)

Relative difference = |r_obj - i_obj| / max(|r_obj|, |i_obj|, 1.0).  
The 1.0 floor prevents near-zero objectives from inflating the metric.

**Matching solutions** (547 problems, rel diff < 1e-4):

| Metric | Rel Diff |
|--------|----------|
| Mean   | 2.87e-08 |
| Median | 2.25e-23 |
| Max    | 1.24e-05 |

**All both-solved** (557 problems, including 10 mismatches):

| Metric | Rel Diff |
|--------|----------|
| Mean   | 7.58e-03 |
| Median | 1.74e-22 |
| Max    | 9.96e-01 |

## Category Breakdown

`low-dof` = constrained problems with degrees of freedom (n &minus; n_eq) &le; 0 &mdash; near-square systems split out so they don't skew the harder cohorts. Override with `--low-dof-max N`.

| Category | Total | pounce | Ipopt | Both | Match |
|----------|-------|--------|-------|------|-------|
| constrained | 322 | 304 | 302 | 300 | 291 |
| low-dof | 171 | 40 | 41 | 40 | 40 |
| unconstrained | 234 | 218 | 219 | 217 | 216 |

## Detailed Results

| Problem | n | m | dof | pounce | Ipopt | Obj Diff | r_iter | i_iter | r_time | i_time | Speedup | Status |
|---------|---|---|-----|--------|-------|----------|--------|--------|--------|--------|---------|--------|
| 3PK | 30 | 0 | 30 | Solve_Succee | Solve_Succee | 3.87e-16 | 9 | 9 | 2.3ms | 7.1ms | 3.1x | PASS |
| ACOPP14 | 38 | 68 | 10 | Solve_Succee | Solve_Succee | 3.38e-16 | 9 | 9 | 4.6ms | 4.2ms | 0.9x | PASS |
| ACOPP30 | 72 | 142 | 12 | Solve_Succee | Solve_Succee | 1.22e-13 | 13 | 13 | 10.4ms | 6.7ms | 0.6x | PASS |
| ACOPR14 | 38 | 82 | 10 | Solve_Succee | Solve_Succee | 1.13e-16 | 13 | 13 | 6.0ms | 5.6ms | 0.9x | PASS |
| ACOPR30 | 72 | 172 | 12 | Solve_Succee | Solve_Succee | 8.61e-14 | 200 | 221 | 136.3ms | 121.9ms | 0.9x | PASS |
| AIRCRFTA | 8 | 5 | 3 | Solve_Succee | Solve_Succee | 0.00e+00 | 3 | 3 | 860us | 1.1ms | 1.3x | PASS |
| AIRCRFTB | 8 | 0 | 8 | Solve_Succee | Solve_Succee | 2.62e-29 | 15 | 15 | 1.9ms | 2.9ms | 1.5x | PASS |
| AIRPORT | 84 | 42 | 84 | Solve_Succee | Solve_Succee | 0.00e+00 | 13 | 13 | 8.2ms | 6.2ms | 0.8x | PASS |
| AKIVA | 2 | 0 | 2 | Solve_Succee | Solve_Succee | 0.00e+00 | 6 | 6 | 857us | 1.3ms | 1.5x | PASS |
| ALLINIT | 4 | 0 | 4 | Solve_Succee | Solve_Succee | 0.00e+00 | 20 | 20 | 2.6ms | 3.8ms | 1.5x | PASS |
| ALLINITA | 4 | 4 | 2 | Solve_Succee | Solve_Succee | 5.63e-11 | 13 | 12 | 2.4ms | 2.7ms | 1.1x | PASS |
| ALLINITC | 4 | 1 | 3 | Solve_Succee | Solve_Succee | 3.19e-12 | 20 | 17 | 2.7ms | 3.5ms | 1.3x | PASS |
| ALLINITU | 4 | 0 | 4 | Solve_Succee | Solve_Succee | 0.00e+00 | 14 | 14 | 1.8ms | 2.7ms | 1.5x | PASS |
| ALSOTAME | 2 | 1 | 1 | Solve_Succee | Solve_Succee | 0.00e+00 | 8 | 8 | 1.4ms | 1.9ms | 1.4x | PASS |
| ANTWERP | 27 | 10 | 19 | Solve_Succee | Solve_Succee | 5.35e-07 | 103 | 108 | 23.8ms | 25.4ms | 1.1x | PASS |
| ARGAUSS | 3 | 15 | -12 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 24us | 240us | 9.8x | BOTH_FAIL |
| AVGASA | 8 | 10 | 8 | Solve_Succee | Solve_Succee | 1.92e-16 | 9 | 9 | 2.1ms | 2.3ms | 1.1x | PASS |
| AVGASB | 8 | 10 | 8 | Solve_Succee | Solve_Succee | 0.00e+00 | 11 | 11 | 2.5ms | 2.6ms | 1.0x | PASS |
| AVION2 | 49 | 15 | 34 | Maximum_Iter | Maximum_Iter | N/A | 2999 | 3000 | 761.6ms | 689.6ms | 0.9x | BOTH_FAIL |
| BA-L1 | 57 | 12 | 45 | Solve_Succee | Solve_Succee | 0.00e+00 | 6 | 6 | 2.0ms | 1.9ms | 1.0x | PASS |
| BA-L1LS | 57 | 0 | 57 | Solve_Succee | Solve_Succee | 1.57e-24 | 10 | 10 | 2.5ms | 2.8ms | 1.1x | PASS |
| BA-L1SP | 57 | 12 | 45 | Solve_Succee | Solve_Succee | 0.00e+00 | 5 | 5 | 3.7ms | 3.0ms | 0.8x | PASS |
| BA-L1SPLS | 57 | 0 | 57 | Solve_Succee | Solve_Succee | 3.02e-22 | 9 | 9 | 4.5ms | 4.7ms | 1.0x | PASS |
| BARD | 3 | 0 | 3 | Solve_Succee | Solve_Succee | 1.39e-17 | 8 | 8 | 1.3ms | 1.7ms | 1.3x | PASS |
| BARDNE | 3 | 15 | -12 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 26us | 258us | 10.1x | BOTH_FAIL |
| BATCH | 48 | 73 | 36 | Solve_Succee | Solve_Succee | 4.17e-09 | 41 | 29 | 9.8ms | 7.8ms | 0.8x | PASS |
| BEALE | 2 | 0 | 2 | Solve_Succee | Solve_Succee | 1.77e-24 | 8 | 8 | 1.3ms | 2.1ms | 1.7x | PASS |
| BEALENE | 2 | 3 | -1 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 28us | 242us | 8.7x | BOTH_FAIL |
| BENNETT5 | 3 | 154 | -151 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 29us | 279us | 9.8x | BOTH_FAIL |
| BENNETT5LS | 3 | 0 | 3 | Solve_Succee | Solve_Succee | 3.02e-15 | 21 | 21 | 2.8ms | 4.4ms | 1.6x | PASS |
| BIGGS3 | 6 | 0 | 6 | Solve_Succee | Solve_Succee | 3.91e-27 | 9 | 9 | 1.5ms | 2.3ms | 1.5x | PASS |
| BIGGS5 | 6 | 0 | 6 | Solve_Succee | Solve_Succee | 5.49e-26 | 20 | 20 | 2.3ms | 3.9ms | 1.7x | PASS |
| BIGGS6 | 6 | 0 | 6 | Solve_Succee | Solve_Succee | 3.78e-18 | 79 | 79 | 7.3ms | 12.3ms | 1.7x | PASS |
| BIGGS6NE | 6 | 13 | -7 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 23us | 249us | 10.8x | BOTH_FAIL |
| BIGGSC4 | 4 | 7 | 4 | Solve_Succee | Solve_Succee | 0.00e+00 | 17 | 17 | 2.7ms | 3.8ms | 1.4x | PASS |
| BLEACHNG | 17 | 0 | 17 | Timeout | Timeout | N/A | 0 | 0 | 60.00s | 60.00s | 1.0x | BOTH_FAIL |
| BOOTH | 2 | 2 | 0 | Solve_Succee | Solve_Succee | 0.00e+00 | 1 | 1 | 989us | 612us | 0.6x | PASS |
| BOX2 | 3 | 0 | 3 | Solve_Succee | Solve_Succee | 0.00e+00 | 8 | 8 | 1.3ms | 1.6ms | 1.3x | PASS |
| BOX3 | 3 | 0 | 3 | Solve_Succee | Solve_Succee | 9.25e-28 | 9 | 9 | 1.2ms | 1.8ms | 1.6x | PASS |
| BOX3NE | 3 | 10 | -7 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 23us | 233us | 10.3x | BOTH_FAIL |
| BOXBOD | 2 | 6 | -4 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 24us | 284us | 11.6x | BOTH_FAIL |
| BOXBODLS | 2 | 0 | 2 | Solve_Succee | Solve_Succee | 0.00e+00 | 13 | 13 | 1.7ms | 2.6ms | 1.5x | PASS |
| BQP1VAR | 1 | 0 | 1 | Solve_Succee | Solve_Succee | 8.45e-22 | 5 | 5 | 1.1ms | 1.4ms | 1.3x | PASS |
| BQPGABIM | 50 | 0 | 50 | Solve_Succee | Solve_Succee | 2.88e-12 | 12 | 12 | 2.2ms | 2.9ms | 1.3x | PASS |
| BQPGASIM | 50 | 0 | 50 | Solve_Succee | Solve_Succee | 7.32e-12 | 12 | 12 | 2.4ms | 2.9ms | 1.2x | PASS |
| BRANIN | 2 | 0 | 2 | Solve_Succee | Solve_Succee | 0.00e+00 | 7 | 7 | 1.4ms | 1.8ms | 1.3x | PASS |
| BRKMCC | 2 | 0 | 2 | Solve_Succee | Solve_Succee | 0.00e+00 | 3 | 3 | 638us | 950us | 1.5x | PASS |
| BROWNBS | 2 | 0 | 2 | Solve_Succee | Solve_Succee | 0.00e+00 | 7 | 7 | 983us | 1.4ms | 1.5x | PASS |
| BROWNBSNE | 2 | 3 | -1 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 21us | 234us | 10.9x | BOTH_FAIL |
| BROWNDEN | 4 | 0 | 4 | Solve_Succee | Solve_Succee | 0.00e+00 | 8 | 8 | 1.1ms | 1.6ms | 1.4x | PASS |
| BROWNDENE | 4 | 20 | -16 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 27us | 270us | 10.0x | BOTH_FAIL |
| BT1 | 2 | 1 | 1 | Solve_Succee | Solve_Succee | 0.00e+00 | 7 | 7 | 1.1ms | 1.7ms | 1.5x | PASS |
| BT10 | 2 | 2 | 0 | Solve_Succee | Solve_Succee | 2.79e-09 | 7 | 6 | 1.1ms | 1.4ms | 1.3x | PASS |
| BT11 | 5 | 3 | 2 | Solve_Succee | Solve_Succee | 2.22e-16 | 8 | 8 | 1.2ms | 1.7ms | 1.4x | PASS |
| BT12 | 5 | 3 | 2 | Solve_Succee | Solve_Succee | 0.00e+00 | 4 | 4 | 795us | 1.2ms | 1.5x | PASS |
| BT13 | 5 | 1 | 4 | Solve_Succee | Solve_Succee | 0.00e+00 | 24 | 24 | 3.0ms | 4.6ms | 1.5x | PASS |
| BT2 | 3 | 1 | 2 | Solve_Succee | Solve_Succee | 0.00e+00 | 12 | 12 | 1.4ms | 1.9ms | 1.4x | PASS |
| BT3 | 5 | 3 | 2 | Solve_Succee | Solve_Succee | 0.00e+00 | 1 | 1 | 595us | 720us | 1.2x | PASS |
| BT4 | 3 | 2 | 1 | Solve_Succee | Solve_Succee | 1.20e-16 | 9 | 9 | 1.3ms | 2.0ms | 1.5x | PASS |
| BT5 | 3 | 2 | 1 | Solve_Succee | Solve_Succee | 0.00e+00 | 7 | 7 | 976us | 1.5ms | 1.5x | PASS |
| BT6 | 5 | 2 | 3 | Solve_Succee | Solve_Succee | 0.00e+00 | 13 | 13 | 1.5ms | 2.4ms | 1.6x | PASS |
| BT7 | 5 | 3 | 2 | Solve_Succee | Solve_Succee | 0.00e+00 | 16 | 16 | 1.9ms | 3.0ms | 1.6x | PASS |
| BT8 | 5 | 2 | 3 | Solved_To_Ac | Solve_Succee | 3.73e-09 | 47 | 14 | 5.4ms | 2.3ms | 0.4x | PASS |
| BT9 | 4 | 2 | 2 | Solve_Succee | Solve_Succee | 0.00e+00 | 13 | 13 | 1.4ms | 2.2ms | 1.6x | PASS |
| BURKEHAN | 1 | 1 | 1 | Infeasible_P | Infeasible_P | N/A | 3 | 11 | 2.5ms | 3.0ms | 1.2x | BOTH_FAIL |
| BYRDSPHR | 3 | 2 | 1 | Solve_Succee | Solve_Succee | 4.52e-13 | 13 | 12 | 1.9ms | 2.6ms | 1.4x | PASS |
| CAMEL6 | 2 | 0 | 2 | Solve_Succee | Solve_Succee | 0.00e+00 | 8 | 8 | 1.5ms | 2.2ms | 1.4x | PASS |
| CANTILVR | 5 | 1 | 5 | Solve_Succee | Solve_Succee | 2.77e-09 | 11 | 11 | 1.9ms | 2.7ms | 1.4x | PASS |
| CB2 | 3 | 3 | 3 | Solve_Succee | Solve_Succee | 0.00e+00 | 8 | 8 | 1.6ms | 2.0ms | 1.3x | PASS |
| CB3 | 3 | 3 | 3 | Solve_Succee | Solve_Succee | 2.22e-16 | 8 | 8 | 1.7ms | 2.0ms | 1.2x | PASS |
| CERI651A | 7 | 61 | -54 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 26us | 260us | 10.1x | BOTH_FAIL |
| CERI651ALS | 7 | 0 | 7 | Solve_Succee | Solve_Succee | 6.95e-09 | 96 | 95 | 8.6ms | 16.8ms | 2.0x | PASS |
| CERI651B | 7 | 66 | -59 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 28us | 262us | 9.4x | BOTH_FAIL |
| CERI651BLS | 7 | 0 | 7 | Solve_Succee | Solve_Succee | 1.05e-08 | 51 | 56 | 4.8ms | 9.2ms | 1.9x | PASS |
| CERI651C | 7 | 56 | -49 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 28us | 263us | 9.4x | BOTH_FAIL |
| CERI651CLS | 7 | 0 | 7 | Solve_Succee | Solve_Succee | 1.84e-10 | 53 | 53 | 4.6ms | 8.4ms | 1.8x | PASS |
| CERI651D | 7 | 67 | -60 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 26us | 261us | 10.0x | BOTH_FAIL |
| CERI651DLS | 7 | 0 | 7 | Solve_Succee | Solve_Succee | 5.34e-11 | 59 | 60 | 6.1ms | 11.1ms | 1.8x | PASS |
| CERI651E | 7 | 64 | -57 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 27us | 248us | 9.3x | BOTH_FAIL |
| CERI651ELS | 7 | 0 | 7 | Solve_Succee | Solve_Succee | 6.80e-10 | 43 | 45 | 3.8ms | 6.9ms | 1.8x | PASS |
| CHACONN1 | 3 | 3 | 3 | Solve_Succee | Solve_Succee | 0.00e+00 | 6 | 6 | 1.3ms | 1.7ms | 1.3x | PASS |
| CHACONN2 | 3 | 3 | 3 | Solve_Succee | Solve_Succee | 0.00e+00 | 6 | 6 | 1.3ms | 1.7ms | 1.3x | PASS |
| CHWIRUT1 | 3 | 214 | -211 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 30us | 274us | 9.2x | BOTH_FAIL |
| CHWIRUT1LS | 3 | 0 | 3 | Solve_Succee | Solve_Succee | 3.81e-16 | 6 | 6 | 1.2ms | 1.8ms | 1.5x | PASS |
| CHWIRUT2 | 3 | 54 | -51 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 28us | 244us | 8.9x | BOTH_FAIL |
| CHWIRUT2LS | 3 | 0 | 3 | Solve_Succee | Solve_Succee | 2.22e-16 | 6 | 6 | 1.2ms | 1.7ms | 1.5x | PASS |
| CLIFF | 2 | 0 | 2 | Solve_Succee | Solve_Succee | 2.78e-16 | 23 | 23 | 1.8ms | 3.2ms | 1.8x | PASS |
| CLUSTER | 2 | 2 | 0 | Solve_Succee | Solve_Succee | 0.00e+00 | 9 | 9 | 1.1ms | 2.0ms | 1.9x | PASS |
| CLUSTERLS | 2 | 0 | 2 | Solve_Succee | Solve_Succee | 8.93e-27 | 17 | 17 | 1.8ms | 2.8ms | 1.5x | PASS |
| CONCON | 15 | 11 | 4 | Solve_Succee | Solve_Succee | 0.00e+00 | 7 | 7 | 1.5ms | 2.0ms | 1.3x | PASS |
| CONGIGMZ | 3 | 5 | 3 | Solve_Succee | Solve_Succee | 1.27e-16 | 20 | 20 | 2.9ms | 4.1ms | 1.4x | PASS |
| COOLHANS | 9 | 9 | 0 | Solve_Succee | Solve_Succee | 0.00e+00 | 9 | 9 | 1.3ms | 1.8ms | 1.4x | PASS |
| COOLHANSLS | 9 | 0 | 9 | Solve_Succee | Solve_Succee | 1.27e-22 | 25 | 25 | 2.7ms | 4.4ms | 1.6x | PASS |
| CORE1 | 65 | 59 | 24 | Solve_Succee | Solve_Succee | 0.00e+00 | 33 | 33 | 9.6ms | 8.3ms | 0.9x | PASS |
| CRESC100 | 6 | 200 | 6 | Infeasible_P | Infeasible_P | N/A | 1191 | 155 | 1.92s | 121.3ms | 0.1x | BOTH_FAIL |
| CRESC132 | 6 | 2654 | 6 | Infeasible_P | Timeout | N/A | 357 | 0 | 6.34s | 60.00s | 9.5x | BOTH_FAIL |
| CRESC4 | 6 | 8 | 6 | Solve_Succee | Solve_Succee | 1.98e-08 | 126 | 64 | 19.1ms | 12.4ms | 0.6x | PASS |
| CRESC50 | 6 | 100 | 6 | Infeasible_P | Solve_Succee | N/A | 251 | 194 | 139.1ms | 82.1ms | 0.6x | pounce_FAIL |
| CSFI1 | 5 | 4 | 3 | Solve_Succee | Solve_Succee | 1.45e-16 | 11 | 11 | 2.3ms | 3.1ms | 1.4x | PASS |
| CSFI2 | 5 | 4 | 3 | Solve_Succee | Solve_Succee | 0.00e+00 | 14 | 14 | 2.4ms | 3.2ms | 1.3x | PASS |
| CUBE | 2 | 0 | 2 | Solve_Succee | Solve_Succee | 0.00e+00 | 27 | 27 | 2.6ms | 4.5ms | 1.7x | PASS |
| CUBENE | 2 | 2 | 0 | Solve_Succee | Solve_Succee | 0.00e+00 | 1 | 1 | 684us | 739us | 1.1x | PASS |
| DALLASS | 46 | 31 | 15 | Solve_Succee | Solve_Succee | 4.58e-02 | 28 | 22 | 6.4ms | 5.2ms | 0.8x | MISMATCH |
| DANIWOOD | 2 | 6 | -4 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 20us | 290us | 14.2x | BOTH_FAIL |
| DANIWOODLS | 2 | 0 | 2 | Solve_Succee | Solve_Succee | 4.16e-17 | 10 | 10 | 1.3ms | 2.1ms | 1.7x | PASS |
| DANWOOD | 2 | 6 | -4 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 24us | 236us | 10.0x | BOTH_FAIL |
| DANWOODLS | 2 | 0 | 2 | Solve_Succee | Solve_Succee | 2.25e-16 | 11 | 11 | 1.4ms | 2.2ms | 1.5x | PASS |
| DECONVB | 63 | 0 | 63 | Maximum_Iter | Maximum_Iter | N/A | 2999 | 3000 | 637.9ms | 717.0ms | 1.1x | BOTH_FAIL |
| DECONVBNE | 63 | 40 | 23 | Solve_Succee | Solve_Succee | 0.00e+00 | 252 | 505 | 130.9ms | 162.4ms | 1.2x | PASS |
| DECONVC | 63 | 1 | 62 | Solve_Succee | Solve_Succee | 9.11e-18 | 31 | 31 | 9.4ms | 9.0ms | 1.0x | PASS |
| DECONVNE | 63 | 40 | 23 | Solve_Succee | Solved_To_Ac | 0.00e+00 | 2 | 26 | 1.9ms | 24.4ms | 13.0x | PASS |
| DECONVU | 63 | 0 | 63 | Solved_To_Ac | Solve_Succee | 3.34e-13 | 345 | 333 | 86.3ms | 85.7ms | 1.0x | PASS |
| DEGENLPA | 20 | 15 | 5 | Solve_Succee | Solve_Succee | 6.52e-12 | 18 | 18 | 3.4ms | 4.0ms | 1.2x | PASS |
| DEGENLPB | 20 | 15 | 5 | Solve_Succee | Solve_Succee | 7.28e-15 | 19 | 19 | 3.1ms | 4.0ms | 1.3x | PASS |
| DEMBO7 | 16 | 20 | 16 | Solve_Succee | Solve_Succee | 2.56e-10 | 54 | 45 | 9.8ms | 8.9ms | 0.9x | PASS |
| DEMYMALO | 3 | 3 | 3 | Solve_Succee | Solve_Succee | 1.48e-16 | 9 | 9 | 1.9ms | 2.2ms | 1.2x | PASS |
| DENSCHNA | 2 | 0 | 2 | Solve_Succee | Solve_Succee | 4.24e-36 | 6 | 6 | 817us | 1.2ms | 1.5x | PASS |
| DENSCHNB | 2 | 0 | 2 | Solve_Succee | Solve_Succee | 0.00e+00 | 7 | 7 | 1.1ms | 1.6ms | 1.4x | PASS |
| DENSCHNBNE | 2 | 3 | -1 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 22us | 325us | 14.5x | BOTH_FAIL |
| DENSCHNC | 2 | 0 | 2 | Solve_Succee | Solve_Succee | 0.00e+00 | 10 | 10 | 1.3ms | 1.7ms | 1.3x | PASS |
| DENSCHNCNE | 2 | 2 | 0 | Solve_Succee | Solve_Succee | 0.00e+00 | 7 | 7 | 951us | 1.5ms | 1.6x | PASS |
| DENSCHND | 3 | 0 | 3 | Solve_Succee | Solve_Succee | 5.31e-17 | 26 | 26 | 2.5ms | 4.0ms | 1.6x | PASS |
| DENSCHNDNE | 3 | 3 | 0 | Solve_Succee | Solved_To_Ac | 0.00e+00 | 23 | 22 | 2.0ms | 3.5ms | 1.7x | PASS |
| DENSCHNE | 3 | 0 | 3 | Solve_Succee | Solve_Succee | 0.00e+00 | 14 | 14 | 1.6ms | 3.0ms | 1.8x | PASS |
| DENSCHNENE | 3 | 3 | 0 | Infeasible_P | Infeasible_P | N/A | 14 | 10 | 2.9ms | 2.5ms | 0.8x | BOTH_FAIL |
| DENSCHNF | 2 | 0 | 2 | Solve_Succee | Solve_Succee | 0.00e+00 | 6 | 6 | 979us | 1.3ms | 1.3x | PASS |
| DENSCHNFNE | 2 | 2 | 0 | Solve_Succee | Solve_Succee | 0.00e+00 | 5 | 5 | 914us | 1.2ms | 1.3x | PASS |
| DEVGLA1 | 4 | 0 | 4 | Solve_Succee | Solve_Succee | 1.71e-22 | 23 | 23 | 2.7ms | 4.1ms | 1.5x | PASS |
| DEVGLA1B | 4 | 0 | 4 | Solve_Succee | Solve_Succee | 1.00e-22 | 20 | 20 | 3.6ms | 4.9ms | 1.4x | PASS |
| DEVGLA1NE | 4 | 24 | -20 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 25us | 241us | 9.5x | BOTH_FAIL |
| DEVGLA2 | 5 | 0 | 5 | Solve_Succee | Solve_Succee | 1.46e-23 | 13 | 13 | 1.5ms | 2.4ms | 1.7x | PASS |
| DEVGLA2B | 5 | 0 | 5 | Solve_Succee | Solve_Succee | 0.00e+00 | 24 | 24 | 3.3ms | 5.0ms | 1.5x | PASS |
| DEVGLA2NE | 5 | 16 | -11 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 21us | 239us | 11.3x | BOTH_FAIL |
| DGOSPEC | 3 | 0 | 3 | Solve_Succee | Solve_Succee | 0.00e+00 | 27 | 27 | 3.8ms | 5.4ms | 1.4x | PASS |
| DIAMON2D | 66 | 4643 | -4577 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 1.2ms | 11.8ms | 9.7x | BOTH_FAIL |
| DIAMON2DLS | 66 | 0 | 66 | Solve_Succee | Timeout | N/A | 1244 | 0 | 42.50s | 60.00s | 1.4x | ipopt_FAIL |
| DIAMON3D | 99 | 4643 | -4544 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 1.6ms | 19.3ms | 12.1x | BOTH_FAIL |
| DIAMON3DLS | 99 | 0 | 99 | Timeout | Timeout | N/A | 0 | 0 | 60.00s | 60.00s | 1.0x | BOTH_FAIL |
| DIPIGRI | 7 | 4 | 7 | Solve_Succee | Solve_Succee | 0.00e+00 | 9 | 9 | 1.8ms | 2.3ms | 1.3x | PASS |
| DISC2 | 29 | 23 | 12 | Solve_Succee | Solve_Succee | 0.00e+00 | 24 | 24 | 5.3ms | 6.0ms | 1.1x | PASS |
| DISCS | 36 | 66 | 18 | Solve_Succee | Solve_Succee | 2.15e-01 | 137 | 184 | 49.2ms | 71.4ms | 1.5x | MISMATCH |
| DIXCHLNG | 10 | 5 | 5 | Solve_Succee | Solve_Succee | 0.00e+00 | 10 | 10 | 1.3ms | 1.9ms | 1.4x | PASS |
| DJTL | 2 | 0 | 2 | Solved_To_Ac | Solved_To_Ac | 0.00e+00 | 1538 | 1538 | 72.1ms | 165.6ms | 2.3x | PASS |
| DMN15102 | 66 | 4643 | -4577 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 1.5ms | 11.0ms | 7.5x | BOTH_FAIL |
| DMN15102LS | 66 | 0 | 66 | Timeout | Solve_Succee | N/A | 0 | 1189 | 60.00s | 39.84s | 0.7x | pounce_FAIL |
| DMN15103 | 99 | 4643 | -4544 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 1.5ms | 19.5ms | 13.3x | BOTH_FAIL |
| DMN15103LS | 99 | 0 | 99 | Timeout | Timeout | N/A | 0 | 0 | 60.00s | 60.00s | 1.0x | BOTH_FAIL |
| DMN15332 | 66 | 4643 | -4577 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 1.2ms | 11.0ms | 9.0x | BOTH_FAIL |
| DMN15332LS | 66 | 0 | 66 | Timeout | Timeout | N/A | 0 | 0 | 60.00s | 60.00s | 1.0x | BOTH_FAIL |
| DMN15333 | 99 | 4643 | -4544 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 1.5ms | 18.8ms | 12.9x | BOTH_FAIL |
| DMN15333LS | 99 | 0 | 99 | Timeout | Timeout | N/A | 0 | 0 | 60.00s | 60.00s | 1.0x | BOTH_FAIL |
| DMN37142 | 66 | 4643 | -4577 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 1.1ms | 9.0ms | 7.9x | BOTH_FAIL |
| DMN37142LS | 66 | 0 | 66 | Timeout | Timeout | N/A | 0 | 0 | 60.00s | 60.00s | 1.0x | BOTH_FAIL |
| DMN37143 | 99 | 4643 | -4544 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 1.3ms | 16.9ms | 12.6x | BOTH_FAIL |
| DMN37143LS | 99 | 0 | 99 | Timeout | Timeout | N/A | 0 | 0 | 60.00s | 60.00s | 1.0x | BOTH_FAIL |
| DNIEPER | 61 | 24 | 37 | Solve_Succee | Solve_Succee | 1.91e-12 | 30 | 23 | 5.8ms | 5.1ms | 0.9x | PASS |
| DUAL1 | 85 | 1 | 84 | Solve_Succee | Solve_Succee | 4.37e-16 | 15 | 15 | 8.9ms | 6.4ms | 0.7x | PASS |
| DUAL2 | 96 | 1 | 95 | Solve_Succee | Solve_Succee | 1.11e-16 | 12 | 12 | 8.3ms | 6.1ms | 0.7x | PASS |
| DUAL4 | 75 | 1 | 74 | Solve_Succee | Solve_Succee | 5.55e-16 | 12 | 12 | 5.8ms | 4.7ms | 0.8x | PASS |
| DUALC1 | 9 | 215 | 8 | Solve_Succee | Solve_Succee | 2.96e-16 | 18 | 18 | 13.5ms | 10.8ms | 0.8x | PASS |
| DUALC2 | 7 | 229 | 6 | Solve_Succee | Solve_Succee | 3.84e-16 | 12 | 12 | 8.7ms | 7.4ms | 0.9x | PASS |
| DUALC5 | 8 | 278 | 7 | Solve_Succee | Solve_Succee | 1.33e-16 | 11 | 11 | 10.1ms | 8.2ms | 0.8x | PASS |
| DUALC8 | 8 | 503 | 7 | Solve_Succee | Solve_Succee | 1.99e-16 | 13 | 13 | 18.6ms | 14.3ms | 0.8x | PASS |
| ECKERLE4 | 3 | 35 | -32 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 24us | 220us | 9.1x | BOTH_FAIL |
| ECKERLE4LS | 3 | 0 | 3 | Solve_Succee | Solve_Succee | 0.00e+00 | 36 | 36 | 3.4ms | 6.1ms | 1.8x | PASS |
| EG1 | 3 | 0 | 3 | Solve_Succee | Solve_Succee | 0.00e+00 | 8 | 8 | 1.6ms | 2.0ms | 1.3x | PASS |
| EGGCRATE | 2 | 0 | 2 | Solve_Succee | Solve_Succee | 0.00e+00 | 5 | 5 | 947us | 1.2ms | 1.3x | PASS |
| EGGCRATEB | 2 | 0 | 2 | Solve_Succee | Solve_Succee | 0.00e+00 | 6 | 6 | 1.3ms | 1.6ms | 1.2x | PASS |
| EGGCRATENE | 2 | 4 | -2 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 18us | 219us | 12.1x | BOTH_FAIL |
| ELATTAR | 7 | 102 | 7 | Solve_Succee | Solve_Succee | 2.68e-07 | 225 | 81 | 91.4ms | 34.1ms | 0.4x | PASS |
| ELATVIDU | 2 | 0 | 2 | Solve_Succee | Solve_Succee | 0.00e+00 | 11 | 11 | 987us | 1.8ms | 1.8x | PASS |
| ELATVIDUB | 2 | 0 | 2 | Solve_Succee | Solve_Succee | 1.30e-16 | 11 | 11 | 1.7ms | 2.2ms | 1.3x | PASS |
| ELATVIDUNE | 2 | 3 | -1 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 19us | 212us | 11.1x | BOTH_FAIL |
| ENGVAL2 | 3 | 0 | 3 | Solve_Succee | Solve_Succee | 1.11e-31 | 21 | 21 | 1.8ms | 3.4ms | 1.9x | PASS |
| ENGVAL2NE | 3 | 5 | -2 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 19us | 216us | 11.6x | BOTH_FAIL |
| ENSO | 9 | 168 | -159 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 30us | 267us | 9.0x | BOTH_FAIL |
| ENSOLS | 9 | 0 | 9 | Solve_Succee | Solve_Succee | 1.44e-16 | 7 | 7 | 2.0ms | 2.3ms | 1.2x | PASS |
| EQC | 9 | 3 | 9 | Solved_To_Ac | Error_In_Ste | N/A | 65 | 15 | 9.5ms | 4.5ms | 0.5x | ipopt_FAIL |
| ERRINBAR | 18 | 9 | 10 | Solve_Succee | Solve_Succee | 0.00e+00 | 37 | 37 | 5.8ms | 7.5ms | 1.3x | PASS |
| EXP2 | 2 | 0 | 2 | Solve_Succee | Solve_Succee | 0.00e+00 | 7 | 7 | 977us | 1.5ms | 1.5x | PASS |
| EXP2B | 2 | 0 | 2 | Solve_Succee | Solve_Succee | 0.00e+00 | 7 | 7 | 1.3ms | 1.7ms | 1.3x | PASS |
| EXP2NE | 2 | 10 | -8 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 29us | 216us | 7.3x | BOTH_FAIL |
| EXPFIT | 2 | 0 | 2 | Solve_Succee | Solve_Succee | 2.50e-16 | 8 | 8 | 1.0ms | 1.6ms | 1.6x | PASS |
| EXPFITA | 5 | 22 | 5 | Solve_Succee | Solve_Succee | 3.90e-18 | 13 | 13 | 2.5ms | 3.0ms | 1.2x | PASS |
| EXPFITB | 5 | 102 | 5 | Solve_Succee | Solve_Succee | 2.17e-17 | 16 | 16 | 6.1ms | 5.4ms | 0.9x | PASS |
| EXPFITC | 5 | 502 | 5 | Solve_Succee | Solve_Succee | 8.67e-17 | 18 | 18 | 22.9ms | 15.8ms | 0.7x | PASS |
| EXPFITNE | 2 | 10 | -8 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 21us | 214us | 10.0x | BOTH_FAIL |
| EXTRASIM | 2 | 1 | 1 | Solve_Succee | Solve_Succee | 0.00e+00 | 3 | 3 | 1.1ms | 1.0ms | 1.0x | PASS |
| FBRAIN | 2 | 2211 | -2209 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 71us | 474us | 6.6x | BOTH_FAIL |
| FBRAIN2 | 4 | 2211 | -2207 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 84us | 650us | 7.7x | BOTH_FAIL |
| FBRAIN2LS | 4 | 0 | 4 | Solve_Succee | Solve_Succee | 1.11e-16 | 10 | 10 | 8.5ms | 8.9ms | 1.1x | PASS |
| FBRAIN2NE | 4 | 2211 | -2207 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 85us | 624us | 7.4x | BOTH_FAIL |
| FBRAIN3 | 6 | 2211 | -2205 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 101us | 773us | 7.7x | BOTH_FAIL |
| FBRAIN3LS | 6 | 0 | 6 | Maximum_Iter | Maximum_Iter | N/A | 2999 | 3000 | 3.39s | 3.63s | 1.1x | BOTH_FAIL |
| FBRAINLS | 2 | 0 | 2 | Solve_Succee | Solve_Succee | 0.00e+00 | 7 | 7 | 3.6ms | 4.0ms | 1.1x | PASS |
| FBRAINNE | 2 | 2211 | -2209 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 57us | 441us | 7.7x | BOTH_FAIL |
| FCCU | 19 | 8 | 11 | Solve_Succee | Solve_Succee | 9.56e-16 | 9 | 9 | 1.6ms | 2.1ms | 1.3x | PASS |
| FEEDLOC | 90 | 259 | 71 | Solve_Succee | Solve_Succee | 1.14e-12 | 23 | 23 | 23.9ms | 14.4ms | 0.6x | PASS |
| FLETCHER | 4 | 4 | 3 | Solve_Succee | Solve_Succee | 5.40e-11 | 28 | 28 | 4.1ms | 5.6ms | 1.4x | PASS |
| FLT | 2 | 2 | 0 | Solve_Succee | Solve_Succee | 0.00e+00 | 6 | 5 | 877us | 1.3ms | 1.5x | PASS |
| GAUSS1 | 8 | 250 | -242 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 34us | 276us | 8.2x | BOTH_FAIL |
| GAUSS1LS | 8 | 0 | 8 | Solve_Succee | Solve_Succee | 0.00e+00 | 5 | 5 | 1.1ms | 1.4ms | 1.3x | PASS |
| GAUSS2 | 8 | 250 | -242 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 31us | 265us | 8.6x | BOTH_FAIL |
| GAUSS2LS | 8 | 0 | 8 | Solve_Succee | Solve_Succee | 0.00e+00 | 5 | 5 | 1.1ms | 1.5ms | 1.3x | PASS |
| GAUSS3 | 8 | 250 | -242 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 32us | 262us | 8.2x | BOTH_FAIL |
| GAUSS3LS | 8 | 0 | 8 | Solve_Succee | Solve_Succee | 0.00e+00 | 11 | 11 | 2.0ms | 2.8ms | 1.4x | PASS |
| GAUSSIAN | 3 | 0 | 3 | Solve_Succee | Solve_Succee | 0.00e+00 | 2 | 2 | 551us | 723us | 1.3x | PASS |
| GBRAIN | 2 | 2200 | -2198 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 74us | 442us | 6.0x | BOTH_FAIL |
| GBRAINLS | 2 | 0 | 2 | Solve_Succee | Solve_Succee | 0.00e+00 | 6 | 6 | 3.0ms | 3.4ms | 1.1x | PASS |
| GENHS28 | 10 | 8 | 2 | Solve_Succee | Solve_Succee | 4.44e-16 | 1 | 1 | 555us | 686us | 1.2x | PASS |
| GIGOMEZ1 | 3 | 3 | 3 | Solve_Succee | Solve_Succee | 4.44e-16 | 13 | 13 | 1.9ms | 2.8ms | 1.4x | PASS |
| GIGOMEZ2 | 3 | 3 | 3 | Solve_Succee | Solve_Succee | 0.00e+00 | 7 | 7 | 1.3ms | 1.8ms | 1.4x | PASS |
| GIGOMEZ3 | 3 | 3 | 3 | Solve_Succee | Solve_Succee | 1.11e-16 | 8 | 8 | 1.4ms | 1.9ms | 1.3x | PASS |
| GOFFIN | 51 | 50 | 51 | Solve_Succee | Solve_Succee | 1.65e-14 | 6 | 7 | 5.0ms | 3.9ms | 0.8x | PASS |
| GOTTFR | 2 | 2 | 0 | Solve_Succee | Solve_Succee | 0.00e+00 | 5 | 5 | 751us | 1.2ms | 1.6x | PASS |
| GOULDQP1 | 32 | 17 | 15 | Solve_Succee | Solve_Succee | 0.00e+00 | 15 | 15 | 3.0ms | 3.4ms | 1.1x | PASS |
| GROUPING | 100 | 125 | -25 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 34us | 232us | 6.9x | BOTH_FAIL |
| GROWTH | 3 | 12 | -9 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 20us | 225us | 11.3x | BOTH_FAIL |
| GROWTHLS | 3 | 0 | 3 | Solve_Succee | Solve_Succee | 8.85e-16 | 71 | 71 | 5.1ms | 11.2ms | 2.2x | PASS |
| GULF | 3 | 0 | 3 | Solve_Succee | Solve_Succee | 7.15e-28 | 28 | 28 | 3.1ms | 5.4ms | 1.7x | PASS |
| GULFNE | 3 | 99 | -96 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 25us | 240us | 9.4x | BOTH_FAIL |
| HAHN1 | 7 | 236 | -229 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 36us | 276us | 7.6x | BOTH_FAIL |
| HAHN1LS | 7 | 0 | 7 | Solve_Succee | Solve_Succee | 8.51e-16 | 78 | 78 | 10.6ms | 17.5ms | 1.6x | PASS |
| HAIFAM | 99 | 150 | 99 | Solve_Succee | Solve_Succee | 5.72e-09 | 50 | 40 | 40.0ms | 16.2ms | 0.4x | PASS |
| HAIFAS | 13 | 9 | 13 | Solve_Succee | Solve_Succee | 2.78e-16 | 16 | 16 | 3.7ms | 3.9ms | 1.0x | PASS |
| HAIRY | 2 | 0 | 2 | Solve_Succee | Solve_Succee | 0.00e+00 | 64 | 62 | 5.2ms | 11.0ms | 2.1x | PASS |
| HALDMADS | 6 | 42 | 6 | Solve_Succee | Solve_Succee | 9.85e-01 | 27 | 8 | 6.7ms | 3.1ms | 0.5x | MISMATCH |
| HART6 | 6 | 0 | 6 | Solve_Succee | Solve_Succee | 0.00e+00 | 7 | 7 | 1.5ms | 2.0ms | 1.4x | PASS |
| HATFLDA | 4 | 0 | 4 | Solve_Succee | Solve_Succee | 4.74e-26 | 13 | 13 | 1.8ms | 2.6ms | 1.4x | PASS |
| HATFLDANE | 4 | 4 | 0 | Solve_Succee | Solve_Succee | 0.00e+00 | 7 | 6 | 1.4ms | 1.7ms | 1.2x | PASS |
| HATFLDB | 4 | 0 | 4 | Solve_Succee | Solve_Succee | 0.00e+00 | 8 | 8 | 1.6ms | 1.9ms | 1.2x | PASS |
| HATFLDBNE | 4 | 4 | 0 | Infeasible_P | Infeasible_P | N/A | 78 | 13 | 13.8ms | 3.3ms | 0.2x | BOTH_FAIL |
| HATFLDC | 25 | 0 | 25 | Solve_Succee | Solve_Succee | 0.00e+00 | 5 | 5 | 1.1ms | 1.4ms | 1.3x | PASS |
| HATFLDCNE | 25 | 25 | 0 | Solve_Succee | Solve_Succee | 0.00e+00 | 4 | 4 | 1.2ms | 1.4ms | 1.2x | PASS |
| HATFLDD | 3 | 0 | 3 | Solve_Succee | Solve_Succee | 2.02e-19 | 21 | 21 | 2.0ms | 3.3ms | 1.7x | PASS |
| HATFLDDNE | 3 | 10 | -7 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 23us | 231us | 10.2x | BOTH_FAIL |
| HATFLDE | 3 | 0 | 3 | Solve_Succee | Solve_Succee | 8.77e-20 | 20 | 20 | 1.8ms | 3.1ms | 1.7x | PASS |
| HATFLDENE | 3 | 21 | -18 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 22us | 225us | 10.4x | BOTH_FAIL |
| HATFLDF | 3 | 3 | 0 | Solve_Succee | Solve_Succee | 0.00e+00 | 140 | 135 | 14.1ms | 26.0ms | 1.8x | PASS |
| HATFLDFL | 3 | 0 | 3 | Maximum_Iter | Solve_Succee | N/A | 2999 | 1281 | 207.5ms | 207.6ms | 1.0x | pounce_FAIL |
| HATFLDFLNE | 3 | 3 | 0 | Solve_Succee | Solve_Succee | 0.00e+00 | 15 | 15 | 2.1ms | 3.2ms | 1.5x | PASS |
| HATFLDFLS | 3 | 0 | 3 | Solve_Succee | Solve_Succee | 3.12e-25 | 36 | 36 | 2.9ms | 6.0ms | 2.1x | PASS |
| HATFLDG | 25 | 25 | 0 | Solve_Succee | Solve_Succee | 0.00e+00 | 7 | 7 | 1.3ms | 1.8ms | 1.4x | PASS |
| HATFLDGLS | 25 | 0 | 25 | Solve_Succee | Solve_Succee | 2.06e-31 | 14 | 14 | 1.8ms | 2.7ms | 1.5x | PASS |
| HATFLDH | 4 | 7 | 4 | Solve_Succee | Solve_Succee | 1.45e-16 | 17 | 17 | 3.0ms | 3.6ms | 1.2x | PASS |
| HEART6 | 6 | 6 | 0 | Solve_Succee | Solve_Succee | 0.00e+00 | 32 | 22 | 5.1ms | 5.6ms | 1.1x | PASS |
| HEART6LS | 6 | 0 | 6 | Solve_Succee | Solve_Succee | 9.46e-23 | 875 | 875 | 78.4ms | 147.4ms | 1.9x | PASS |
| HEART8 | 8 | 8 | 0 | Solve_Succee | Solve_Succee | 0.00e+00 | 13 | 12 | 1.7ms | 2.7ms | 1.6x | PASS |
| HEART8LS | 8 | 0 | 8 | Solve_Succee | Solve_Succee | 4.36e-29 | 106 | 106 | 9.5ms | 18.5ms | 2.0x | PASS |
| HELIX | 3 | 0 | 3 | Solve_Succee | Solve_Succee | 2.03e-29 | 13 | 13 | 1.4ms | 2.4ms | 1.8x | PASS |
| HELIXNE | 3 | 3 | 0 | Solve_Succee | Solve_Succee | 0.00e+00 | 7 | 7 | 972us | 1.6ms | 1.6x | PASS |
| HET-Z | 2 | 1002 | 2 | Solve_Succee | Solve_Succee | 4.00e-15 | 11 | 11 | 20.8ms | 19.5ms | 0.9x | PASS |
| HIELOW | 3 | 0 | 3 | Solve_Succee | Solve_Succee | 0.00e+00 | 8 | 8 | 11.2ms | 11.6ms | 1.0x | PASS |
| HIMMELBA | 2 | 2 | 0 | Solve_Succee | Solve_Succee | 0.00e+00 | 1 | 1 | 526us | 631us | 1.2x | PASS |
| HIMMELBB | 2 | 0 | 2 | Solve_Succee | Solve_Succee | 1.95e-24 | 18 | 18 | 2.0ms | 3.1ms | 1.6x | PASS |
| HIMMELBC | 2 | 2 | 0 | Solve_Succee | Solve_Succee | 0.00e+00 | 6 | 6 | 862us | 1.4ms | 1.6x | PASS |
| HIMMELBCLS | 2 | 0 | 2 | Solve_Succee | Solve_Succee | 0.00e+00 | 6 | 6 | 979us | 1.4ms | 1.4x | PASS |
| HIMMELBD | 2 | 2 | 0 | Infeasible_P | Infeasible_P | N/A | 27 | 22 | 4.3ms | 5.0ms | 1.2x | BOTH_FAIL |
| HIMMELBE | 3 | 3 | 0 | Solve_Succee | Solve_Succee | 0.00e+00 | 2 | 2 | 579us | 753us | 1.3x | PASS |
| HIMMELBF | 4 | 0 | 4 | Solve_Succee | Solve_Succee | 7.14e-16 | 75 | 75 | 5.9ms | 11.5ms | 2.0x | PASS |
| HIMMELBFNE | 4 | 7 | -3 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 18us | 231us | 13.1x | BOTH_FAIL |
| HIMMELBG | 2 | 0 | 2 | Solve_Succee | Solve_Succee | 3.03e-30 | 6 | 6 | 1.1ms | 1.5ms | 1.4x | PASS |
| HIMMELBH | 2 | 0 | 2 | Solve_Succee | Solve_Succee | 0.00e+00 | 4 | 4 | 969us | 1.2ms | 1.2x | PASS |
| HIMMELBI | 100 | 12 | 100 | Solve_Succee | Solve_Succee | 0.00e+00 | 13 | 13 | 4.2ms | 3.8ms | 0.9x | PASS |
| HIMMELBJ | 45 | 14 | 31 | Solve_Succee | Error_In_Ste | N/A | 615 | 580 | 105.7ms | 143.8ms | 1.4x | ipopt_FAIL |
| HIMMELBK | 24 | 14 | 10 | Solve_Succee | Solve_Succee | 6.94e-18 | 18 | 18 | 3.5ms | 4.1ms | 1.2x | PASS |
| HIMMELP1 | 2 | 0 | 2 | Solve_Succee | Solve_Succee | 0.00e+00 | 11 | 10 | 1.9ms | 2.3ms | 1.2x | PASS |
| HIMMELP2 | 2 | 1 | 2 | Solve_Succee | Solve_Succee | 2.97e-10 | 18 | 17 | 2.9ms | 3.8ms | 1.3x | PASS |
| HIMMELP3 | 2 | 2 | 2 | Solve_Succee | Solve_Succee | 0.00e+00 | 11 | 11 | 2.1ms | 2.5ms | 1.2x | PASS |
| HIMMELP4 | 2 | 3 | 2 | Solve_Succee | Solve_Succee | 1.45e-12 | 22 | 23 | 3.9ms | 4.8ms | 1.2x | PASS |
| HIMMELP5 | 2 | 3 | 2 | Solve_Succee | Solve_Succee | 2.89e-14 | 51 | 46 | 6.7ms | 8.5ms | 1.3x | PASS |
| HIMMELP6 | 2 | 5 | 2 | Solve_Succee | Solve_Succee | 2.05e-12 | 31 | 31 | 5.0ms | 6.4ms | 1.3x | PASS |
| HONG | 4 | 1 | 3 | Solve_Succee | Solve_Succee | 1.57e-16 | 7 | 7 | 1.4ms | 1.8ms | 1.3x | PASS |
| HS1 | 2 | 0 | 2 | Solve_Succee | Solve_Succee | 0.00e+00 | 28 | 28 | 3.2ms | 5.5ms | 1.7x | PASS |
| HS10 | 2 | 1 | 2 | Solve_Succee | Solve_Succee | 0.00e+00 | 12 | 12 | 1.8ms | 2.5ms | 1.4x | PASS |
| HS100 | 7 | 4 | 7 | Solve_Succee | Solve_Succee | 0.00e+00 | 9 | 9 | 1.7ms | 2.3ms | 1.3x | PASS |
| HS100LNP | 7 | 2 | 5 | Solve_Succee | Solve_Succee | 0.00e+00 | 20 | 20 | 1.7ms | 2.9ms | 1.7x | PASS |
| HS100MOD | 7 | 4 | 7 | Solve_Succee | Solve_Succee | 0.00e+00 | 14 | 14 | 2.3ms | 3.0ms | 1.3x | PASS |
| HS101 | 7 | 5 | 7 | Solve_Succee | Solve_Succee | 8.63e-14 | 47 | 39 | 7.4ms | 9.8ms | 1.3x | PASS |
| HS102 | 7 | 5 | 7 | Solve_Succee | Solve_Succee | 1.97e-10 | 28 | 52 | 4.5ms | 10.6ms | 2.4x | PASS |
| HS103 | 7 | 5 | 7 | Solve_Succee | Solve_Succee | 6.06e-10 | 16 | 21 | 3.1ms | 4.5ms | 1.5x | PASS |
| HS104 | 8 | 5 | 8 | Solve_Succee | Solve_Succee | 0.00e+00 | 8 | 8 | 1.7ms | 2.0ms | 1.2x | PASS |
| HS105 | 8 | 1 | 8 | Solve_Succee | Solve_Succee | 0.00e+00 | 23 | 23 | 6.1ms | 7.0ms | 1.1x | PASS |
| HS106 | 8 | 6 | 8 | Solve_Succee | Solve_Succee | 1.95e-12 | 18 | 18 | 2.9ms | 3.6ms | 1.2x | PASS |
| HS107 | 9 | 6 | 3 | Solve_Succee | Solve_Succee | 0.00e+00 | 7 | 7 | 1.5ms | 1.8ms | 1.2x | PASS |
| HS108 | 9 | 13 | 9 | Solve_Succee | Solve_Succee | 1.11e-16 | 11 | 11 | 2.4ms | 2.8ms | 1.2x | PASS |
| HS109 | 9 | 10 | 3 | Solve_Succee | Solve_Succee | 1.23e-12 | 15 | 14 | 2.7ms | 3.1ms | 1.2x | PASS |
| HS11 | 2 | 1 | 2 | Solve_Succee | Solve_Succee | 0.00e+00 | 6 | 6 | 1.2ms | 1.6ms | 1.3x | PASS |
| HS111 | 10 | 3 | 7 | Solve_Succee | Solve_Succee | 1.49e-16 | 15 | 15 | 2.4ms | 3.3ms | 1.4x | PASS |
| HS111LNP | 10 | 3 | 7 | Solve_Succee | Solve_Succee | 1.49e-16 | 15 | 15 | 1.5ms | 2.7ms | 1.7x | PASS |
| HS112 | 10 | 3 | 7 | Solve_Succee | Solve_Succee | 1.49e-16 | 10 | 10 | 1.8ms | 2.3ms | 1.3x | PASS |
| HS113 | 10 | 8 | 10 | Solve_Succee | Solve_Succee | 1.46e-16 | 9 | 9 | 1.9ms | 2.3ms | 1.2x | PASS |
| HS114 | 10 | 11 | 7 | Solve_Succee | Solve_Succee | 3.09e-15 | 13 | 13 | 2.4ms | 2.8ms | 1.2x | PASS |
| HS116 | 13 | 14 | 13 | Solve_Succee | Solve_Succee | 1.59e-09 | 19 | 19 | 3.5ms | 4.2ms | 1.2x | PASS |
| HS117 | 15 | 5 | 15 | Solve_Succee | Solve_Succee | 0.00e+00 | 19 | 19 | 3.4ms | 4.2ms | 1.2x | PASS |
| HS118 | 15 | 17 | 15 | Solve_Succee | Solve_Succee | 1.71e-16 | 10 | 10 | 2.4ms | 2.6ms | 1.1x | PASS |
| HS119 | 16 | 8 | 8 | Solve_Succee | Solve_Succee | 0.00e+00 | 17 | 17 | 3.0ms | 3.7ms | 1.2x | PASS |
| HS12 | 2 | 1 | 2 | Solve_Succee | Solve_Succee | 1.18e-16 | 6 | 6 | 1.2ms | 1.6ms | 1.4x | PASS |
| HS13 | 2 | 1 | 2 | Solve_Succee | Solve_Succee | 9.41e-13 | 49 | 47 | 6.0ms | 8.1ms | 1.3x | PASS |
| HS14 | 2 | 2 | 1 | Solve_Succee | Solve_Succee | 0.00e+00 | 5 | 5 | 1.1ms | 1.4ms | 1.2x | PASS |
| HS15 | 2 | 2 | 2 | Solve_Succee | Solve_Succee | 0.00e+00 | 13 | 13 | 2.2ms | 2.8ms | 1.3x | PASS |
| HS16 | 2 | 2 | 2 | Solve_Succee | Solve_Succee | 0.00e+00 | 10 | 10 | 2.0ms | 2.5ms | 1.3x | PASS |
| HS17 | 2 | 2 | 2 | Solve_Succee | Solve_Succee | 0.00e+00 | 22 | 22 | 3.1ms | 4.3ms | 1.4x | PASS |
| HS18 | 2 | 2 | 2 | Solve_Succee | Solve_Succee | 5.33e-16 | 10 | 10 | 1.7ms | 2.2ms | 1.3x | PASS |
| HS19 | 2 | 2 | 2 | Solve_Succee | Solve_Succee | 7.84e-16 | 12 | 12 | 2.1ms | 2.8ms | 1.3x | PASS |
| HS1NE | 2 | 2 | 0 | Solve_Succee | Solve_Succee | 0.00e+00 | 21 | 30 | 2.6ms | 6.8ms | 2.6x | PASS |
| HS2 | 2 | 0 | 2 | Solve_Succee | Solve_Succee | 0.00e+00 | 10 | 10 | 1.6ms | 2.3ms | 1.5x | PASS |
| HS20 | 2 | 3 | 2 | Solve_Succee | Solve_Succee | 0.00e+00 | 5 | 5 | 1.2ms | 1.5ms | 1.2x | PASS |
| HS21 | 2 | 1 | 2 | Solve_Succee | Solve_Succee | 0.00e+00 | 6 | 6 | 1.3ms | 1.6ms | 1.3x | PASS |
| HS21MOD | 7 | 1 | 7 | Solve_Succee | Solve_Succee | 0.00e+00 | 13 | 13 | 2.1ms | 2.8ms | 1.4x | PASS |
| HS22 | 2 | 2 | 2 | Solve_Succee | Solve_Succee | 0.00e+00 | 5 | 5 | 1.1ms | 1.4ms | 1.3x | PASS |
| HS23 | 2 | 5 | 2 | Solve_Succee | Solve_Succee | 0.00e+00 | 9 | 9 | 1.7ms | 2.1ms | 1.3x | PASS |
| HS24 | 2 | 3 | 2 | Solve_Succee | Solve_Succee | 0.00e+00 | 14 | 14 | 2.3ms | 3.3ms | 1.4x | PASS |
| HS25 | 3 | 0 | 3 | Solve_Succee | Solve_Succee | 0.00e+00 | 27 | 27 | 4.2ms | 6.2ms | 1.5x | PASS |
| HS25NE | 3 | 99 | -96 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 27us | 265us | 9.7x | BOTH_FAIL |
| HS26 | 3 | 1 | 2 | Solve_Succee | Solve_Succee | 3.07e-26 | 25 | 25 | 1.9ms | 3.4ms | 1.8x | PASS |
| HS268 | 5 | 5 | 5 | Solve_Succee | Solve_Succee | 1.09e-11 | 14 | 14 | 2.3ms | 2.9ms | 1.3x | PASS |
| HS27 | 3 | 1 | 2 | Solve_Succee | Solve_Succee | 2.21e-12 | 58 | 57 | 4.8ms | 9.1ms | 1.9x | PASS |
| HS28 | 3 | 1 | 2 | Solve_Succee | Solve_Succee | 9.24e-31 | 1 | 1 | 554us | 701us | 1.3x | PASS |
| HS29 | 3 | 1 | 3 | Solve_Succee | Solve_Succee | 0.00e+00 | 7 | 7 | 1.3ms | 1.8ms | 1.3x | PASS |
| HS2NE | 2 | 2 | 0 | Infeasible_P | Infeasible_P | N/A | 16 | 12 | 3.0ms | 3.1ms | 1.0x | BOTH_FAIL |
| HS3 | 2 | 0 | 2 | Solve_Succee | Solve_Succee | 1.74e-22 | 4 | 4 | 944us | 1.3ms | 1.3x | PASS |
| HS30 | 3 | 1 | 3 | Solve_Succee | Solve_Succee | 1.89e-08 | 8 | 7 | 1.6ms | 1.8ms | 1.2x | PASS |
| HS31 | 3 | 1 | 3 | Solve_Succee | Solve_Succee | 0.00e+00 | 6 | 6 | 1.3ms | 1.7ms | 1.3x | PASS |
| HS32 | 3 | 2 | 2 | Solve_Succee | Solve_Succee | 0.00e+00 | 15 | 15 | 2.3ms | 3.1ms | 1.4x | PASS |
| HS33 | 3 | 2 | 3 | Solve_Succee | Solve_Succee | 0.00e+00 | 9 | 9 | 1.7ms | 2.2ms | 1.3x | PASS |
| HS34 | 3 | 2 | 3 | Solve_Succee | Solve_Succee | 5.92e-09 | 8 | 7 | 1.6ms | 1.8ms | 1.1x | PASS |
| HS35 | 3 | 1 | 3 | Solve_Succee | Solve_Succee | 2.50e-16 | 7 | 7 | 1.5ms | 1.8ms | 1.2x | PASS |
| HS35I | 3 | 1 | 3 | Solve_Succee | Solve_Succee | 6.80e-16 | 7 | 7 | 1.3ms | 1.8ms | 1.4x | PASS |
| HS35MOD | 3 | 1 | 3 | Solve_Succee | Solve_Succee | 0.00e+00 | 14 | 14 | 2.2ms | 2.8ms | 1.3x | PASS |
| HS36 | 3 | 1 | 3 | Solve_Succee | Solve_Succee | 0.00e+00 | 11 | 11 | 2.1ms | 2.7ms | 1.3x | PASS |
| HS37 | 3 | 2 | 3 | Solve_Succee | Solve_Succee | 2.63e-16 | 11 | 11 | 2.1ms | 2.6ms | 1.2x | PASS |
| HS38 | 4 | 0 | 4 | Solve_Succee | Solve_Succee | 1.91e-27 | 39 | 39 | 4.5ms | 7.4ms | 1.6x | PASS |
| HS39 | 4 | 2 | 2 | Solve_Succee | Solve_Succee | 0.00e+00 | 13 | 13 | 1.4ms | 2.3ms | 1.7x | PASS |
| HS3MOD | 2 | 0 | 2 | Solve_Succee | Solve_Succee | 2.20e-21 | 4 | 4 | 1.0ms | 1.1ms | 1.1x | PASS |
| HS4 | 2 | 0 | 2 | Solve_Succee | Solve_Succee | 0.00e+00 | 4 | 4 | 932us | 1.2ms | 1.3x | PASS |
| HS40 | 4 | 3 | 1 | Solve_Succee | Solve_Succee | 0.00e+00 | 3 | 3 | 891us | 962us | 1.1x | PASS |
| HS41 | 4 | 1 | 3 | Solve_Succee | Solve_Succee | 0.00e+00 | 7 | 7 | 1.3ms | 1.8ms | 1.4x | PASS |
| HS42 | 4 | 2 | 2 | Solve_Succee | Solve_Succee | 0.00e+00 | 4 | 4 | 788us | 1.0ms | 1.3x | PASS |
| HS43 | 4 | 3 | 4 | Solve_Succee | Solve_Succee | 0.00e+00 | 8 | 8 | 1.6ms | 2.0ms | 1.3x | PASS |
| HS44 | 4 | 6 | 4 | Solve_Succee | Solve_Succee | 0.00e+00 | 24 | 24 | 3.5ms | 5.1ms | 1.4x | PASS |
| HS44NEW | 4 | 6 | 4 | Solve_Succee | Solve_Succee | 0.00e+00 | 18 | 18 | 3.0ms | 4.2ms | 1.4x | PASS |
| HS45 | 5 | 0 | 5 | Solve_Succee | Solve_Succee | 0.00e+00 | 11 | 11 | 1.8ms | 2.5ms | 1.4x | PASS |
| HS46 | 5 | 2 | 3 | Solve_Succee | Solve_Succee | 2.74e-24 | 19 | 19 | 1.6ms | 2.8ms | 1.8x | PASS |
| HS47 | 5 | 3 | 2 | Solve_Succee | Solve_Succee | 0.00e+00 | 19 | 19 | 1.7ms | 2.8ms | 1.6x | PASS |
| HS48 | 5 | 2 | 3 | Solve_Succee | Solve_Succee | 5.42e-31 | 1 | 1 | 599us | 675us | 1.1x | PASS |
| HS49 | 5 | 2 | 3 | Solve_Succee | Solve_Succee | 0.00e+00 | 19 | 19 | 1.6ms | 2.9ms | 1.8x | PASS |
| HS5 | 2 | 0 | 2 | Solve_Succee | Solve_Succee | 0.00e+00 | 7 | 7 | 1.3ms | 1.7ms | 1.3x | PASS |
| HS50 | 5 | 3 | 2 | Solve_Succee | Solve_Succee | 0.00e+00 | 9 | 9 | 1.0ms | 1.7ms | 1.6x | PASS |
| HS51 | 5 | 3 | 2 | Solve_Succee | Solve_Succee | 0.00e+00 | 1 | 1 | 685us | 748us | 1.1x | PASS |
| HS52 | 5 | 3 | 2 | Solve_Succee | Solve_Succee | 5.00e-16 | 1 | 1 | 607us | 688us | 1.1x | PASS |
| HS53 | 5 | 3 | 2 | Solve_Succee | Solve_Succee | 0.00e+00 | 6 | 6 | 1.3ms | 1.6ms | 1.2x | PASS |
| HS54 | 6 | 1 | 5 | Solve_Succee | Solve_Succee | 0.00e+00 | 15 | 15 | 2.5ms | 3.3ms | 1.3x | PASS |
| HS55 | 6 | 6 | 0 | Solve_Succee | Solve_Succee | 1.92e-06 | 23 | 18 | 3.8ms | 4.9ms | 1.3x | PASS |
| HS56 | 7 | 4 | 3 | Solve_Succee | Solve_Succee | 0.00e+00 | 10 | 10 | 1.3ms | 2.0ms | 1.5x | PASS |
| HS57 | 2 | 1 | 2 | Solve_Succee | Solve_Succee | 4.16e-17 | 10 | 10 | 1.4ms | 2.0ms | 1.4x | PASS |
| HS59 | 2 | 3 | 2 | Solve_Succee | Solve_Succee | 1.93e-14 | 17 | 17 | 2.9ms | 3.8ms | 1.3x | PASS |
| HS6 | 2 | 1 | 1 | Solve_Succee | Solve_Succee | 4.93e-32 | 5 | 5 | 880us | 1.4ms | 1.6x | PASS |
| HS60 | 3 | 1 | 2 | Solve_Succee | Solve_Succee | 0.00e+00 | 6 | 6 | 1.2ms | 1.6ms | 1.3x | PASS |
| HS61 | 3 | 2 | 1 | Solve_Succee | Solve_Succee | 3.18e-13 | 7 | 10 | 1.1ms | 1.8ms | 1.6x | PASS |
| HS62 | 3 | 1 | 2 | Solve_Succee | Solve_Succee | 2.77e-16 | 6 | 6 | 1.3ms | 1.8ms | 1.4x | PASS |
| HS63 | 3 | 2 | 1 | Solve_Succee | Solve_Succee | 0.00e+00 | 5 | 5 | 1.1ms | 1.4ms | 1.3x | PASS |
| HS64 | 3 | 1 | 3 | Solve_Succee | Solve_Succee | 7.24e-10 | 16 | 16 | 2.3ms | 3.3ms | 1.4x | PASS |
| HS65 | 3 | 1 | 3 | Solve_Succee | Solve_Succee | 0.00e+00 | 16 | 16 | 2.7ms | 3.7ms | 1.4x | PASS |
| HS66 | 3 | 2 | 3 | Solve_Succee | Solve_Succee | 1.52e-13 | 10 | 10 | 1.7ms | 2.2ms | 1.3x | PASS |
| HS67 | 3 | 14 | 3 | Solve_Succee | Solve_Succee | 4.50e-15 | 9 | 9 | 2.1ms | 2.2ms | 1.0x | PASS |
| HS68 | 4 | 2 | 2 | Solve_Succee | Solve_Succee | 4.44e-16 | 16 | 16 | 2.4ms | 3.3ms | 1.4x | PASS |
| HS69 | 4 | 2 | 2 | Solve_Succee | Solve_Succee | 2.73e-15 | 10 | 10 | 1.9ms | 2.5ms | 1.4x | PASS |
| HS7 | 2 | 1 | 1 | Solve_Succee | Solve_Succee | 0.00e+00 | 27 | 27 | 2.7ms | 4.5ms | 1.7x | PASS |
| HS70 | 4 | 1 | 4 | Solve_Succee | Solve_Succee | 1.19e-16 | 46 | 46 | 5.8ms | 8.8ms | 1.5x | PASS |
| HS71 | 4 | 2 | 3 | Solve_Succee | Solve_Succee | 1.46e-15 | 8 | 8 | 1.7ms | 2.0ms | 1.2x | PASS |
| HS72 | 4 | 2 | 4 | Solve_Succee | Solve_Succee | 3.33e-14 | 16 | 16 | 2.5ms | 3.2ms | 1.3x | PASS |
| HS73 | 4 | 3 | 3 | Solve_Succee | Solve_Succee | 0.00e+00 | 8 | 8 | 1.5ms | 2.0ms | 1.3x | PASS |
| HS74 | 4 | 5 | 1 | Solve_Succee | Solve_Succee | 1.77e-16 | 8 | 8 | 1.7ms | 2.1ms | 1.2x | PASS |
| HS75 | 4 | 5 | 1 | Solve_Succee | Solve_Succee | 1.76e-16 | 8 | 8 | 1.6ms | 2.0ms | 1.3x | PASS |
| HS76 | 4 | 3 | 4 | Solve_Succee | Solve_Succee | 0.00e+00 | 7 | 7 | 1.6ms | 1.8ms | 1.1x | PASS |
| HS76I | 4 | 3 | 4 | Solve_Succee | Solve_Succee | 0.00e+00 | 6 | 6 | 1.5ms | 1.6ms | 1.1x | PASS |
| HS77 | 5 | 2 | 3 | Solve_Succee | Solve_Succee | 0.00e+00 | 11 | 11 | 1.3ms | 2.0ms | 1.6x | PASS |
| HS78 | 5 | 3 | 2 | Solve_Succee | Solve_Succee | 0.00e+00 | 4 | 4 | 851us | 1.1ms | 1.3x | PASS |
| HS79 | 5 | 3 | 2 | Solve_Succee | Solve_Succee | 0.00e+00 | 4 | 4 | 809us | 1.1ms | 1.4x | PASS |
| HS8 | 2 | 2 | 0 | Solve_Succee | Solve_Succee | 0.00e+00 | 5 | 5 | 868us | 1.2ms | 1.4x | PASS |
| HS80 | 5 | 3 | 2 | Solve_Succee | Solve_Succee | 3.47e-17 | 5 | 5 | 1.2ms | 1.6ms | 1.4x | PASS |
| HS81 | 5 | 3 | 2 | Solve_Succee | Solve_Succee | 1.13e-11 | 73 | 68 | 9.7ms | 12.5ms | 1.3x | PASS |
| HS83 | 5 | 3 | 5 | Solve_Succee | Solve_Succee | 0.00e+00 | 9 | 9 | 1.8ms | 2.1ms | 1.2x | PASS |
| HS84 | 5 | 3 | 5 | Solve_Succee | Solve_Succee | 9.96e-01 | 14 | 9 | 2.3ms | 2.3ms | 1.0x | MISMATCH |
| HS85 | 5 | 21 | 5 | Solve_Succee | Solve_Succee | 6.80e-09 | 22 | 13 | 4.9ms | 4.0ms | 0.8x | PASS |
| HS86 | 5 | 10 | 5 | Solve_Succee | Solve_Succee | 2.20e-16 | 10 | 10 | 1.9ms | 2.4ms | 1.3x | PASS |
| HS87 | 6 | 4 | 2 | Maximum_Iter | Maximum_Iter | N/A | 2999 | 3000 | 275.9ms | 492.1ms | 1.8x | BOTH_FAIL |
| HS88 | 2 | 1 | 2 | Solve_Succee | Solve_Succee | 8.15e-16 | 18 | 18 | 3.1ms | 4.0ms | 1.3x | PASS |
| HS89 | 3 | 1 | 3 | Solve_Succee | Solve_Succee | 7.33e-15 | 15 | 15 | 3.1ms | 3.8ms | 1.2x | PASS |
| HS9 | 2 | 1 | 1 | Solve_Succee | Solve_Succee | 0.00e+00 | 3 | 3 | 725us | 1.0ms | 1.4x | PASS |
| HS90 | 4 | 1 | 4 | Solve_Succee | Solve_Succee | 3.26e-16 | 16 | 16 | 3.8ms | 4.4ms | 1.2x | PASS |
| HS91 | 5 | 1 | 5 | Solve_Succee | Solve_Succee | 0.00e+00 | 16 | 16 | 4.0ms | 4.7ms | 1.2x | PASS |
| HS92 | 6 | 1 | 6 | Solve_Succee | Solve_Succee | 1.08e-13 | 37 | 35 | 9.6ms | 10.7ms | 1.1x | PASS |
| HS93 | 6 | 2 | 6 | Solve_Succee | Solve_Succee | 2.10e-16 | 7 | 7 | 1.5ms | 2.0ms | 1.3x | PASS |
| HS95 | 6 | 4 | 6 | Solve_Succee | Solve_Succee | 4.39e-10 | 9 | 9 | 1.8ms | 2.2ms | 1.2x | PASS |
| HS96 | 6 | 4 | 6 | Solve_Succee | Solve_Succee | 4.39e-10 | 8 | 8 | 1.7ms | 2.2ms | 1.3x | PASS |
| HS97 | 6 | 4 | 6 | Solve_Succee | Solve_Succee | 1.12e-08 | 24 | 24 | 4.0ms | 5.1ms | 1.3x | PASS |
| HS98 | 6 | 4 | 6 | Solve_Succee | Solve_Succee | 1.12e-08 | 13 | 13 | 2.3ms | 2.8ms | 1.2x | PASS |
| HS99 | 7 | 2 | 5 | Solve_Succee | Solve_Succee | 0.00e+00 | 4 | 5 | 1.1ms | 1.5ms | 1.4x | PASS |
| HS99EXP | 31 | 21 | 10 | Solve_Succee | Solve_Succee | 0.00e+00 | 17 | 17 | 3.1ms | 3.8ms | 1.2x | PASS |
| HUBFIT | 2 | 1 | 2 | Solve_Succee | Solve_Succee | 3.47e-18 | 7 | 7 | 1.3ms | 1.8ms | 1.4x | PASS |
| HUMPS | 2 | 0 | 2 | Solve_Succee | Solve_Succee | 2.76e-17 | 188 | 1533 | 13.7ms | 220.0ms | 16.0x | PASS |
| HYDC20LS | 99 | 0 | 99 | Solve_Succee | Solve_Succee | 4.19e-15 | 639 | 639 | 181.8ms | 175.5ms | 1.0x | PASS |
| HYDCAR20 | 99 | 99 | 0 | Solve_Succee | Solve_Succee | 0.00e+00 | 9 | 9 | 3.1ms | 3.0ms | 1.0x | PASS |
| HYDCAR6 | 29 | 29 | 0 | Solve_Succee | Solve_Succee | 0.00e+00 | 5 | 5 | 1.4ms | 1.6ms | 1.2x | PASS |
| HYDCAR6LS | 29 | 0 | 29 | Solve_Succee | Solve_Succee | 2.86e-18 | 148 | 149 | 19.1ms | 31.1ms | 1.6x | PASS |
| HYPCIR | 2 | 2 | 0 | Solve_Succee | Solve_Succee | 0.00e+00 | 5 | 5 | 1.1ms | 1.3ms | 1.2x | PASS |
| JENSMP | 2 | 0 | 2 | Solve_Succee | Solve_Succee | 1.14e-16 | 9 | 9 | 1.0ms | 1.6ms | 1.6x | PASS |
| JENSMPNE | 2 | 10 | -8 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 21us | 227us | 10.7x | BOTH_FAIL |
| JUDGE | 2 | 0 | 2 | Solve_Succee | Solve_Succee | 0.00e+00 | 9 | 9 | 1.2ms | 1.6ms | 1.4x | PASS |
| JUDGEB | 2 | 0 | 2 | Solve_Succee | Solve_Succee | 2.21e-16 | 9 | 9 | 1.6ms | 2.0ms | 1.3x | PASS |
| JUDGENE | 2 | 20 | -18 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 20us | 220us | 10.7x | BOTH_FAIL |
| KIRBY2 | 5 | 151 | -146 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 30us | 247us | 8.2x | BOTH_FAIL |
| KIRBY2LS | 5 | 0 | 5 | Solve_Succee | Solve_Succee | 2.39e-15 | 11 | 11 | 1.6ms | 2.4ms | 1.5x | PASS |
| KIWCRESC | 3 | 2 | 3 | Solve_Succee | Solve_Succee | 8.11e-17 | 8 | 8 | 1.6ms | 2.0ms | 1.2x | PASS |
| KOEBHELB | 3 | 0 | 3 | Solve_Succee | Solve_Succee | 1.10e-15 | 71 | 71 | 9.1ms | 15.8ms | 1.7x | PASS |
| KOEBHELBNE | 3 | 156 | -153 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 29us | 238us | 8.2x | BOTH_FAIL |
| KOWOSB | 4 | 0 | 4 | Solve_Succee | Solve_Succee | 7.59e-19 | 8 | 8 | 1.2ms | 2.0ms | 1.7x | PASS |
| KOWOSBNE | 4 | 11 | -7 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 23us | 220us | 9.5x | BOTH_FAIL |
| KSIP | 20 | 1001 | 20 | Solve_Succee | Solve_Succee | 1.80e-10 | 22 | 22 | 96.4ms | 69.8ms | 0.7x | PASS |
| LAKES | 90 | 78 | 12 | Solve_Succee | Solve_Succee | 3.94e-14 | 11 | 11 | 4.3ms | 3.6ms | 0.8x | PASS |
| LANCZOS1 | 6 | 24 | -18 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 24us | 230us | 9.4x | BOTH_FAIL |
| LANCZOS1LS | 6 | 0 | 6 | Solve_Succee | Solve_Succee | 4.19e-18 | 114 | 115 | 9.5ms | 19.7ms | 2.1x | PASS |
| LANCZOS2 | 6 | 24 | -18 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 23us | 242us | 10.7x | BOTH_FAIL |
| LANCZOS2LS | 6 | 0 | 6 | Solve_Succee | Solve_Succee | 4.44e-18 | 101 | 101 | 9.0ms | 17.7ms | 2.0x | PASS |
| LANCZOS3 | 6 | 24 | -18 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 25us | 232us | 9.3x | BOTH_FAIL |
| LANCZOS3LS | 6 | 0 | 6 | Solve_Succee | Solve_Succee | 6.07e-17 | 174 | 174 | 13.9ms | 31.5ms | 2.3x | PASS |
| LAUNCH | 25 | 28 | 16 | Solve_Succee | Solve_Succee | 2.61e-09 | 22 | 12 | 4.4ms | 3.2ms | 0.7x | PASS |
| LEVYMONE10 | 10 | 20 | -10 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 28us | 233us | 8.4x | BOTH_FAIL |
| LEVYMONE5 | 2 | 4 | -2 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 23us | 225us | 9.7x | BOTH_FAIL |
| LEVYMONE6 | 3 | 6 | -3 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 20us | 225us | 11.1x | BOTH_FAIL |
| LEVYMONE7 | 4 | 8 | -4 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 26us | 221us | 8.6x | BOTH_FAIL |
| LEVYMONE8 | 5 | 10 | -5 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 24us | 228us | 9.4x | BOTH_FAIL |
| LEVYMONE9 | 8 | 16 | -8 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 23us | 230us | 10.1x | BOTH_FAIL |
| LEVYMONT10 | 10 | 0 | 10 | Solve_Succee | Solve_Succee | 0.00e+00 | 4 | 4 | 963us | 1.2ms | 1.3x | PASS |
| LEVYMONT5 | 2 | 0 | 2 | Solve_Succee | Solve_Succee | 1.17e-28 | 10 | 10 | 1.7ms | 2.3ms | 1.4x | PASS |
| LEVYMONT6 | 3 | 0 | 3 | Solve_Succee | Solve_Succee | 0.00e+00 | 8 | 8 | 1.7ms | 2.0ms | 1.2x | PASS |
| LEVYMONT7 | 4 | 0 | 4 | Solve_Succee | Solve_Succee | 0.00e+00 | 7 | 7 | 1.5ms | 2.0ms | 1.3x | PASS |
| LEVYMONT8 | 5 | 0 | 5 | Solve_Succee | Solve_Succee | 0.00e+00 | 4 | 4 | 947us | 1.2ms | 1.3x | PASS |
| LEVYMONT9 | 8 | 0 | 8 | Solve_Succee | Solve_Succee | 0.00e+00 | 4 | 4 | 1.0ms | 1.3ms | 1.3x | PASS |
| LEWISPOL | 6 | 9 | -3 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 24us | 231us | 9.6x | BOTH_FAIL |
| LHAIFAM | 99 | 150 | 99 | Restoration_ | Invalid_Numb | N/A | 1 | 0 | 432.2ms | 288us | 0.0x | BOTH_FAIL |
| LIN | 4 | 2 | 2 | Solve_Succee | Solve_Succee | 3.06e-03 | 5 | 7 | 1.1ms | 1.8ms | 1.6x | MISMATCH |
| LINSPANH | 97 | 33 | 64 | Solve_Succee | Solve_Succee | 6.57e-11 | 17 | 24 | 4.8ms | 5.8ms | 1.2x | PASS |
| LOADBAL | 31 | 31 | 20 | Solve_Succee | Solve_Succee | 0.00e+00 | 13 | 13 | 3.3ms | 3.4ms | 1.0x | PASS |
| LOGHAIRY | 2 | 0 | 2 | Solve_Succee | Solve_Succee | 0.00e+00 | 2003 | 2747 | 148.9ms | 404.0ms | 2.7x | PASS |
| LOGROS | 2 | 0 | 2 | Solve_Succee | Solve_Succee | 0.00e+00 | 49 | 49 | 5.4ms | 9.7ms | 1.8x | PASS |
| LOOTSMA | 3 | 2 | 3 | Solve_Succee | Solve_Succee | 1.07e-12 | 13 | 13 | 2.4ms | 3.0ms | 1.3x | PASS |
| LOTSCHD | 12 | 7 | 5 | Solve_Succee | Solve_Succee | 0.00e+00 | 9 | 9 | 1.7ms | 2.1ms | 1.3x | PASS |
| LRCOVTYPE | 54 | 0 | 54 | Solve_Succee | Solve_Succee | 5.55e-15 | 37 | 33 | 7.22s | 6.10s | 0.8x | PASS |
| LRIJCNN1 | 22 | 0 | 22 | Solve_Succee | Solve_Succee | 0.00e+00 | 11 | 11 | 181.3ms | 184.1ms | 1.0x | PASS |
| LSC1 | 3 | 6 | -3 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 21us | 226us | 10.8x | BOTH_FAIL |
| LSC1LS | 3 | 0 | 3 | Solve_Succee | Solve_Succee | 0.00e+00 | 16 | 16 | 1.6ms | 3.0ms | 1.8x | PASS |
| LSC2 | 3 | 6 | -3 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 21us | 225us | 10.6x | BOTH_FAIL |
| LSC2LS | 3 | 0 | 3 | Solve_Succee | Solve_Succee | 1.24e-05 | 40 | 38 | 2.5ms | 5.2ms | 2.1x | PASS |
| LSNNODOC | 5 | 4 | 1 | Solve_Succee | Solve_Succee | 5.96e-13 | 10 | 10 | 1.7ms | 2.4ms | 1.4x | PASS |
| LSQFIT | 2 | 1 | 2 | Solve_Succee | Solve_Succee | 0.00e+00 | 7 | 7 | 1.2ms | 1.8ms | 1.5x | PASS |
| MADSEN | 3 | 6 | 3 | Solve_Succee | Solve_Succee | 0.00e+00 | 18 | 18 | 2.7ms | 3.7ms | 1.4x | PASS |
| MAKELA1 | 3 | 2 | 3 | Solve_Succee | Solve_Succee | 1.57e-16 | 12 | 12 | 1.9ms | 2.7ms | 1.4x | PASS |
| MAKELA2 | 3 | 3 | 3 | Solve_Succee | Solve_Succee | 1.23e-16 | 6 | 6 | 1.1ms | 1.6ms | 1.4x | PASS |
| MAKELA3 | 21 | 20 | 21 | Solve_Succee | Solve_Succee | 1.06e-11 | 17 | 11 | 2.7ms | 2.6ms | 1.0x | PASS |
| MAKELA4 | 21 | 40 | 21 | Solve_Succee | Solve_Succee | 4.07e-20 | 5 | 5 | 1.3ms | 1.7ms | 1.3x | PASS |
| MARATOS | 2 | 1 | 1 | Solve_Succee | Solve_Succee | 0.00e+00 | 4 | 4 | 714us | 1.0ms | 1.4x | PASS |
| MARATOSB | 2 | 0 | 2 | Solve_Succee | Solve_Succee | 0.00e+00 | 672 | 672 | 42.0ms | 100.1ms | 2.4x | PASS |
| MATRIX2 | 6 | 2 | 6 | Solve_Succee | Solve_Succee | 5.27e-11 | 56 | 42 | 6.2ms | 7.3ms | 1.2x | PASS |
| MAXLIKA | 8 | 0 | 8 | Solve_Succee | Solve_Succee | 0.00e+00 | 23 | 23 | 5.5ms | 6.8ms | 1.2x | PASS |
| MCONCON | 15 | 11 | 4 | Solve_Succee | Solve_Succee | 0.00e+00 | 7 | 7 | 1.4ms | 1.8ms | 1.3x | PASS |
| MDHOLE | 2 | 0 | 2 | Solve_Succee | Solve_Succee | 0.00e+00 | 42 | 42 | 4.8ms | 9.1ms | 1.9x | PASS |
| MESH | 41 | 48 | 17 | Diverging_It | Diverging_It | N/A | 81 | 79 | 21.2ms | 20.1ms | 0.9x | BOTH_FAIL |
| METHANB8 | 31 | 31 | 0 | Solve_Succee | Solve_Succee | 0.00e+00 | 3 | 3 | 1.0ms | 1.1ms | 1.1x | PASS |
| METHANB8LS | 31 | 0 | 31 | Solve_Succee | Solve_Succee | 1.01e-26 | 8 | 8 | 1.3ms | 1.7ms | 1.4x | PASS |
| METHANL8 | 31 | 31 | 0 | Solve_Succee | Solve_Succee | 0.00e+00 | 4 | 4 | 1.2ms | 1.3ms | 1.1x | PASS |
| METHANL8LS | 31 | 0 | 31 | Solve_Succee | Solve_Succee | 2.22e-21 | 40 | 40 | 5.8ms | 8.4ms | 1.4x | PASS |
| MEXHAT | 2 | 0 | 2 | Solve_Succee | Solve_Succee | 6.94e-18 | 26 | 26 | 1.9ms | 3.8ms | 2.0x | PASS |
| MEYER3 | 3 | 0 | 3 | Solve_Succee | Solve_Succee | 6.02e-12 | 196 | 194 | 13.1ms | 30.7ms | 2.3x | PASS |
| MEYER3NE | 3 | 16 | -13 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 22us | 222us | 10.1x | BOTH_FAIL |
| MGH09 | 4 | 11 | -7 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 21us | 218us | 10.6x | BOTH_FAIL |
| MGH09LS | 4 | 0 | 4 | Solve_Succee | Solve_Succee | 0.00e+00 | 72 | 72 | 5.4ms | 11.6ms | 2.1x | PASS |
| MGH10 | 3 | 16 | -13 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 23us | 224us | 9.8x | BOTH_FAIL |
| MGH10LS | 3 | 0 | 3 | Solve_Succee | Solve_Succee | 3.68e-12 | 1960 | 1828 | 132.5ms | 272.8ms | 2.1x | PASS |
| MGH10S | 3 | 16 | -13 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 21us | 218us | 10.4x | BOTH_FAIL |
| MGH10SLS | 3 | 0 | 3 | Solve_Succee | Solve_Succee | 2.21e-12 | 354 | 354 | 23.3ms | 52.8ms | 2.3x | PASS |
| MGH17 | 5 | 33 | -28 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 22us | 222us | 10.0x | BOTH_FAIL |
| MGH17LS | 5 | 0 | 5 | Solve_Succee | Solve_Succee | 1.91e-07 | 50 | 47 | 4.6ms | 9.0ms | 1.9x | PASS |
| MGH17S | 5 | 33 | -28 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 22us | 236us | 10.8x | BOTH_FAIL |
| MGH17SLS | 5 | 0 | 5 | Solve_Succee | Solve_Succee | 6.36e-08 | 38 | 41 | 4.1ms | 8.2ms | 2.0x | PASS |
| MIFFLIN1 | 3 | 2 | 3 | Solve_Succee | Solve_Succee | 4.44e-16 | 5 | 5 | 1.2ms | 1.4ms | 1.2x | PASS |
| MIFFLIN2 | 3 | 2 | 3 | Solve_Succee | Solve_Succee | 1.99e-10 | 11 | 11 | 1.8ms | 2.5ms | 1.4x | PASS |
| MINMAXBD | 5 | 20 | 5 | Solve_Succee | Solve_Succee | 1.19e-11 | 25 | 25 | 4.6ms | 6.0ms | 1.3x | PASS |
| MINMAXRB | 3 | 4 | 3 | Solve_Succee | Solve_Succee | 1.20e-16 | 8 | 8 | 1.4ms | 1.9ms | 1.4x | PASS |
| MINSURF | 64 | 0 | 64 | Solve_Succee | Solve_Succee | 0.00e+00 | 4 | 4 | 1.1ms | 1.5ms | 1.3x | PASS |
| MISRA1A | 2 | 14 | -12 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 21us | 229us | 11.1x | BOTH_FAIL |
| MISRA1ALS | 2 | 0 | 2 | Solve_Succee | Solve_Succee | 3.86e-15 | 40 | 40 | 3.5ms | 6.4ms | 1.8x | PASS |
| MISRA1B | 2 | 14 | -12 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 21us | 213us | 10.2x | BOTH_FAIL |
| MISRA1BLS | 2 | 0 | 2 | Solve_Succee | Solve_Succee | 4.30e-14 | 34 | 34 | 2.6ms | 5.3ms | 2.1x | PASS |
| MISRA1C | 2 | 14 | -12 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 21us | 220us | 10.6x | BOTH_FAIL |
| MISRA1CLS | 2 | 0 | 2 | Solve_Succee | Solve_Succee | 1.33e-14 | 14 | 14 | 1.5ms | 2.7ms | 1.9x | PASS |
| MISRA1D | 2 | 14 | -12 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 20us | 223us | 11.1x | BOTH_FAIL |
| MISRA1DLS | 2 | 0 | 2 | Solve_Succee | Solve_Succee | 6.94e-18 | 30 | 30 | 2.3ms | 5.0ms | 2.1x | PASS |
| MISTAKE | 9 | 13 | 9 | Solve_Succee | Solve_Succee | 0.00e+00 | 16 | 16 | 3.1ms | 3.8ms | 1.2x | PASS |
| MRIBASIS | 36 | 55 | 27 | Solve_Succee | Solve_Succee | 0.00e+00 | 15 | 15 | 5.0ms | 4.4ms | 0.9x | PASS |
| MSS1 | 90 | 73 | 17 | Maximum_Iter | Solve_Succee | N/A | 2999 | 95 | 12.38s | 49.2ms | 0.0x | pounce_FAIL |
| MUONSINE | 1 | 512 | -511 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 24us | 249us | 10.2x | BOTH_FAIL |
| MUONSINELS | 1 | 0 | 1 | Solve_Succee | Solve_Succee | 0.00e+00 | 8 | 8 | 1.4ms | 1.9ms | 1.4x | PASS |
| MWRIGHT | 5 | 3 | 2 | Solve_Succee | Solve_Succee | 0.00e+00 | 10 | 10 | 1.5ms | 1.9ms | 1.2x | PASS |
| NASH | 72 | 24 | 48 | Infeasible_P | Infeasible_P | N/A | 15 | 45 | 6.7ms | 12.1ms | 1.8x | BOTH_FAIL |
| NELSON | 3 | 128 | -125 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 26us | 233us | 9.1x | BOTH_FAIL |
| NET1 | 48 | 57 | 10 | Solve_Succee | Solve_Succee | 2.14e-12 | 23 | 26 | 5.6ms | 6.2ms | 1.1x | PASS |
| NYSTROM5 | 18 | 20 | -2 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 27us | 227us | 8.4x | BOTH_FAIL |
| NYSTROM5C | 18 | 20 | -2 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 26us | 228us | 8.8x | BOTH_FAIL |
| ODFITS | 10 | 6 | 4 | Solve_Succee | Solve_Succee | 1.91e-16 | 8 | 8 | 1.4ms | 1.9ms | 1.4x | PASS |
| OET1 | 3 | 1002 | 3 | Solve_Succee | Solve_Succee | 2.22e-16 | 33 | 33 | 53.7ms | 45.5ms | 0.8x | PASS |
| OET2 | 3 | 1002 | 3 | Solve_Succee | Solve_Succee | 7.20e-10 | 172 | 181 | 294.1ms | 265.8ms | 0.9x | PASS |
| OET3 | 4 | 1002 | 4 | Solve_Succee | Solve_Succee | 4.08e-17 | 13 | 13 | 22.7ms | 20.4ms | 0.9x | PASS |
| OET4 | 4 | 1002 | 4 | Solve_Succee | Solve_Succee | 8.53e-01 | 52 | 165 | 85.8ms | 240.4ms | 2.8x | MISMATCH |
| OET5 | 5 | 1002 | 5 | Solve_Succee | Solve_Succee | 4.73e-13 | 67 | 64 | 142.4ms | 112.9ms | 0.8x | PASS |
| OET6 | 5 | 1002 | 5 | Solve_Succee | Solve_Succee | 4.56e-14 | 161 | 126 | 428.7ms | 344.6ms | 0.8x | PASS |
| OET7 | 7 | 1002 | 7 | Solve_Succee | Solve_Succee | 2.07e-07 | 205 | 193 | 612.3ms | 514.6ms | 0.8x | PASS |
| OPTCNTRL | 32 | 20 | 12 | Solve_Succee | Solve_Succee | 1.45e-15 | 9 | 9 | 1.9ms | 2.2ms | 1.2x | PASS |
| OPTPRLOC | 30 | 30 | 30 | Solve_Succee | Solve_Succee | 6.74e-11 | 13 | 13 | 3.2ms | 3.6ms | 1.1x | PASS |
| ORTHREGB | 27 | 6 | 21 | Solve_Succee | Solve_Succee | 0.00e+00 | 2 | 2 | 768us | 1.0ms | 1.3x | PASS |
| OSBORNE1 | 5 | 33 | -28 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 22us | 241us | 10.8x | BOTH_FAIL |
| OSBORNE2 | 11 | 65 | -54 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 26us | 247us | 9.5x | BOTH_FAIL |
| OSBORNEA | 5 | 0 | 5 | Solve_Succee | Solve_Succee | 2.36e-18 | 64 | 64 | 4.9ms | 10.5ms | 2.1x | PASS |
| OSBORNEB | 11 | 0 | 11 | Solve_Succee | Solve_Succee | 1.39e-17 | 19 | 19 | 2.2ms | 3.6ms | 1.6x | PASS |
| OSLBQP | 8 | 0 | 8 | Solve_Succee | Solve_Succee | 0.00e+00 | 15 | 15 | 2.0ms | 2.8ms | 1.4x | PASS |
| PALMER1 | 4 | 0 | 4 | Solve_Succee | Solve_Succee | 1.55e-16 | 13 | 13 | 1.9ms | 2.8ms | 1.4x | PASS |
| PALMER1A | 6 | 0 | 6 | Solve_Succee | Solve_Succee | 1.24e-15 | 48 | 48 | 5.7ms | 9.9ms | 1.7x | PASS |
| PALMER1ANE | 6 | 35 | -29 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 24us | 232us | 9.8x | BOTH_FAIL |
| PALMER1B | 4 | 0 | 4 | Solve_Succee | Solve_Succee | 3.45e-14 | 17 | 17 | 2.4ms | 3.4ms | 1.4x | PASS |
| PALMER1BNE | 4 | 35 | -31 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 27us | 223us | 8.3x | BOTH_FAIL |
| PALMER1C | 8 | 0 | 8 | Solve_Succee | Solve_Succee | 3.44e-14 | 1 | 1 | 547us | 597us | 1.1x | PASS |
| PALMER1D | 7 | 0 | 7 | Solve_Succee | Solve_Succee | 1.71e-13 | 1 | 1 | 421us | 601us | 1.4x | PASS |
| PALMER1E | 8 | 0 | 8 | Solve_Succee | Solve_Succee | 3.88e-13 | 55 | 55 | 6.8ms | 10.9ms | 1.6x | PASS |
| PALMER1ENE | 8 | 35 | -27 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 25us | 234us | 9.4x | BOTH_FAIL |
| PALMER1NE | 4 | 31 | -27 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 24us | 229us | 9.3x | BOTH_FAIL |
| PALMER2 | 4 | 0 | 4 | Solve_Succee | Solve_Succee | 2.49e-16 | 28 | 28 | 3.9ms | 6.6ms | 1.7x | PASS |
| PALMER2A | 6 | 0 | 6 | Solve_Succee | Solve_Succee | 8.64e-16 | 92 | 91 | 10.9ms | 19.5ms | 1.8x | PASS |
| PALMER2ANE | 6 | 23 | -17 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 23us | 218us | 9.4x | BOTH_FAIL |
| PALMER2B | 4 | 0 | 4 | Solve_Succee | Solve_Succee | 1.74e-14 | 15 | 15 | 2.2ms | 3.5ms | 1.5x | PASS |
| PALMER2BNE | 4 | 23 | -19 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 24us | 220us | 9.1x | BOTH_FAIL |
| PALMER2C | 8 | 0 | 8 | Solve_Succee | Solve_Succee | 5.09e-15 | 1 | 1 | 535us | 613us | 1.1x | PASS |
| PALMER2E | 8 | 0 | 8 | Solve_Succee | Solve_Succee | 1.16e-01 | 91 | 114 | 10.3ms | 24.0ms | 2.3x | MISMATCH |
| PALMER2ENE | 8 | 23 | -15 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 23us | 223us | 9.6x | BOTH_FAIL |
| PALMER2NE | 4 | 23 | -19 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 22us | 215us | 9.8x | BOTH_FAIL |
| PALMER3 | 4 | 0 | 4 | Solve_Succee | Solve_Succee | 0.00e+00 | 44 | 44 | 4.5ms | 7.8ms | 1.7x | PASS |
| PALMER3A | 6 | 0 | 6 | Solve_Succee | Solve_Succee | 5.79e-16 | 73 | 73 | 8.3ms | 14.9ms | 1.8x | PASS |
| PALMER3ANE | 6 | 23 | -17 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 22us | 219us | 9.8x | BOTH_FAIL |
| PALMER3B | 4 | 0 | 4 | Solve_Succee | Solve_Succee | 1.47e-15 | 15 | 15 | 2.5ms | 3.5ms | 1.4x | PASS |
| PALMER3BNE | 4 | 23 | -19 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 24us | 214us | 8.8x | BOTH_FAIL |
| PALMER3C | 8 | 0 | 8 | Solve_Succee | Solve_Succee | 4.71e-15 | 1 | 1 | 534us | 594us | 1.1x | PASS |
| PALMER3E | 8 | 0 | 8 | Solve_Succee | Solve_Succee | 3.02e-17 | 32 | 32 | 3.9ms | 6.0ms | 1.5x | PASS |
| PALMER3ENE | 8 | 23 | -15 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 24us | 224us | 9.4x | BOTH_FAIL |
| PALMER3NE | 4 | 23 | -19 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 23us | 224us | 9.9x | BOTH_FAIL |
| PALMER4 | 4 | 0 | 4 | Solve_Succee | Solve_Succee | 0.00e+00 | 16 | 16 | 2.6ms | 3.9ms | 1.5x | PASS |
| PALMER4A | 6 | 0 | 6 | Solve_Succee | Solve_Succee | 2.20e-15 | 53 | 53 | 6.1ms | 10.5ms | 1.7x | PASS |
| PALMER4ANE | 6 | 23 | -17 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 26us | 229us | 8.8x | BOTH_FAIL |
| PALMER4B | 4 | 0 | 4 | Solve_Succee | Solve_Succee | 6.50e-15 | 15 | 16 | 2.2ms | 3.8ms | 1.7x | PASS |
| PALMER4BNE | 4 | 23 | -19 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 25us | 229us | 9.1x | BOTH_FAIL |
| PALMER4C | 8 | 0 | 8 | Solve_Succee | Solve_Succee | 4.23e-15 | 1 | 1 | 523us | 626us | 1.2x | PASS |
| PALMER4E | 8 | 0 | 8 | Solve_Succee | Solve_Succee | 2.59e-17 | 25 | 25 | 3.2ms | 5.1ms | 1.6x | PASS |
| PALMER4ENE | 8 | 23 | -15 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 25us | 230us | 9.2x | BOTH_FAIL |
| PALMER4NE | 4 | 23 | -19 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 24us | 226us | 9.5x | BOTH_FAIL |
| PALMER5A | 8 | 0 | 8 | Maximum_Iter | Maximum_Iter | N/A | 2999 | 3000 | 344.9ms | 634.9ms | 1.8x | BOTH_FAIL |
| PALMER5ANE | 8 | 12 | -4 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 23us | 219us | 9.6x | BOTH_FAIL |
| PALMER5B | 9 | 0 | 9 | Solve_Succee | Solve_Succee | 1.66e-13 | 105 | 113 | 12.3ms | 21.8ms | 1.8x | PASS |
| PALMER5BNE | 9 | 12 | -3 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 25us | 226us | 8.9x | BOTH_FAIL |
| PALMER5C | 6 | 0 | 6 | Solve_Succee | Solve_Succee | 1.25e-15 | 1 | 1 | 517us | 612us | 1.2x | PASS |
| PALMER5D | 4 | 0 | 4 | Solve_Succee | Solve_Succee | 6.51e-16 | 1 | 1 | 510us | 593us | 1.2x | PASS |
| PALMER5E | 8 | 0 | 8 | Maximum_Iter | Maximum_Iter | N/A | 2999 | 3000 | 249.9ms | 485.0ms | 1.9x | BOTH_FAIL |
| PALMER5ENE | 8 | 12 | -4 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 23us | 220us | 9.4x | BOTH_FAIL |
| PALMER6A | 6 | 0 | 6 | Solve_Succee | Solve_Succee | 8.33e-16 | 106 | 105 | 10.8ms | 20.0ms | 1.8x | PASS |
| PALMER6ANE | 6 | 13 | -7 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 23us | 212us | 9.1x | BOTH_FAIL |
| PALMER6C | 8 | 0 | 8 | Solve_Succee | Solve_Succee | 2.54e-14 | 1 | 1 | 587us | 597us | 1.0x | PASS |
| PALMER6E | 8 | 0 | 8 | Solve_Succee | Solve_Succee | 2.91e-15 | 30 | 30 | 3.8ms | 6.2ms | 1.7x | PASS |
| PALMER6ENE | 8 | 13 | -5 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 22us | 233us | 10.8x | BOTH_FAIL |
| PALMER7A | 6 | 0 | 6 | Maximum_Iter | Maximum_Iter | N/A | 2999 | 3000 | 259.5ms | 489.4ms | 1.9x | BOTH_FAIL |
| PALMER7ANE | 6 | 13 | -7 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 24us | 215us | 8.8x | BOTH_FAIL |
| PALMER7C | 8 | 0 | 8 | Solve_Succee | Solve_Succee | 4.20e-13 | 1 | 1 | 434us | 602us | 1.4x | PASS |
| PALMER7E | 8 | 0 | 8 | Maximum_Iter | Maximum_Iter | N/A | 2999 | 3000 | 332.7ms | 644.4ms | 1.9x | BOTH_FAIL |
| PALMER7ENE | 8 | 13 | -5 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 28us | 223us | 7.8x | BOTH_FAIL |
| PALMER8A | 6 | 0 | 6 | Solve_Succee | Solve_Succee | 5.41e-16 | 36 | 36 | 5.3ms | 8.8ms | 1.7x | PASS |
| PALMER8ANE | 6 | 12 | -6 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 25us | 237us | 9.6x | BOTH_FAIL |
| PALMER8C | 8 | 0 | 8 | Solve_Succee | Solve_Succee | 6.44e-15 | 1 | 1 | 528us | 608us | 1.2x | PASS |
| PALMER8E | 8 | 0 | 8 | Solve_Succee | Solve_Succee | 1.02e-16 | 23 | 23 | 3.3ms | 5.0ms | 1.5x | PASS |
| PALMER8ENE | 8 | 12 | -4 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 24us | 231us | 9.8x | BOTH_FAIL |
| PARKCH | 15 | 0 | 15 | Solve_Succee | Solve_Succee | 4.20e-16 | 17 | 17 | 4.07s | 4.05s | 1.0x | PASS |
| PENTAGON | 6 | 15 | 6 | Solve_Succee | Solve_Succee | 5.42e-20 | 19 | 19 | 3.9ms | 4.5ms | 1.2x | PASS |
| PFIT1 | 3 | 3 | 0 | Infeasible_P | Infeasible_P | N/A | 523 | 266 | 797.2ms | 47.8ms | 0.1x | BOTH_FAIL |
| PFIT1LS | 3 | 0 | 3 | Solve_Succee | Solve_Succee | 1.28e-21 | 263 | 263 | 25.4ms | 49.1ms | 1.9x | PASS |
| PFIT2 | 3 | 3 | 0 | Infeasible_P | Restoration_ | N/A | 6 | 247 | 41.6ms | 50.1ms | 1.2x | BOTH_FAIL |
| PFIT2LS | 3 | 0 | 3 | Solve_Succee | Solve_Succee | 6.31e-22 | 81 | 82 | 8.0ms | 16.6ms | 2.1x | PASS |
| PFIT3 | 3 | 3 | 0 | Solve_Succee | Solve_Succee | 0.00e+00 | 235 | 133 | 44.2ms | 28.4ms | 0.6x | PASS |
| PFIT3LS | 3 | 0 | 3 | Solve_Succee | Solve_Succee | 3.72e-22 | 132 | 132 | 13.0ms | 24.8ms | 1.9x | PASS |
| PFIT4 | 3 | 3 | 0 | Infeasible_P | Solve_Succee | N/A | 264 | 190 | 129.8ms | 38.8ms | 0.3x | pounce_FAIL |
| PFIT4LS | 3 | 0 | 3 | Solve_Succee | Solve_Succee | 1.33e-20 | 215 | 215 | 20.6ms | 39.4ms | 1.9x | PASS |
| POLAK1 | 3 | 2 | 3 | Solve_Succee | Solve_Succee | 0.00e+00 | 5 | 5 | 1.2ms | 1.4ms | 1.3x | PASS |
| POLAK2 | 11 | 2 | 11 | Solve_Succee | Solve_Succee | 3.14e-10 | 10 | 10 | 1.8ms | 2.4ms | 1.3x | PASS |
| POLAK3 | 12 | 10 | 12 | Restoration_ | Maximum_Iter | N/A | 1191 | 3000 | 251.8ms | 694.2ms | 2.8x | BOTH_FAIL |
| POLAK4 | 3 | 3 | 3 | Solve_Succee | Solve_Succee | 1.52e-12 | 5 | 4 | 1.1ms | 1.3ms | 1.2x | PASS |
| POLAK5 | 3 | 2 | 3 | Solve_Succee | Solve_Succee | 1.06e-11 | 31 | 31 | 3.6ms | 5.7ms | 1.6x | PASS |
| POLAK6 | 5 | 4 | 5 | Solve_Succee | Maximum_Iter | N/A | 153 | 3000 | 17.6ms | 875.4ms | 49.8x | ipopt_FAIL |
| PORTFL1 | 12 | 1 | 11 | Solve_Succee | Solve_Succee | 0.00e+00 | 9 | 9 | 1.8ms | 2.3ms | 1.3x | PASS |
| PORTFL2 | 12 | 1 | 11 | Solve_Succee | Solve_Succee | 0.00e+00 | 8 | 8 | 1.7ms | 2.2ms | 1.3x | PASS |
| PORTFL3 | 12 | 1 | 11 | Solve_Succee | Solve_Succee | 0.00e+00 | 9 | 9 | 1.8ms | 2.2ms | 1.2x | PASS |
| PORTFL4 | 12 | 1 | 11 | Solve_Succee | Solve_Succee | 3.47e-18 | 8 | 8 | 1.8ms | 2.1ms | 1.2x | PASS |
| PORTFL6 | 12 | 1 | 11 | Solve_Succee | Solve_Succee | 3.47e-18 | 8 | 8 | 2.0ms | 2.1ms | 1.0x | PASS |
| POWELLBS | 2 | 2 | 0 | Solve_Succee | Solve_Succee | 0.00e+00 | 11 | 11 | 1.1ms | 1.8ms | 1.6x | PASS |
| POWELLBSLS | 2 | 0 | 2 | Solve_Succee | Solve_Succee | 2.25e-23 | 90 | 91 | 5.8ms | 13.3ms | 2.3x | PASS |
| POWELLSQ | 2 | 2 | 0 | Infeasible_P | Infeasible_P | N/A | 37 | 29 | 4.6ms | 5.6ms | 1.2x | BOTH_FAIL |
| POWELLSQLS | 2 | 0 | 2 | Solve_Succee | Solve_Succee | 1.09e-30 | 10 | 10 | 1.4ms | 2.1ms | 1.5x | PASS |
| PRICE3NE | 2 | 2 | 0 | Solve_Succee | Solve_Succee | 0.00e+00 | 7 | 7 | 877us | 1.5ms | 1.7x | PASS |
| PRICE4 | 2 | 0 | 2 | Solve_Succee | Solve_Succee | 6.18e-26 | 8 | 8 | 965us | 1.6ms | 1.6x | PASS |
| PRICE4B | 2 | 0 | 2 | Solve_Succee | Solve_Succee | 0.00e+00 | 8 | 8 | 1.4ms | 2.0ms | 1.4x | PASS |
| PRICE4NE | 2 | 2 | 0 | Solve_Succee | Solved_To_Ac | 0.00e+00 | 25 | 23 | 1.9ms | 3.8ms | 2.0x | PASS |
| PRODPL0 | 60 | 29 | 40 | Solve_Succee | Solve_Succee | 3.63e-16 | 15 | 15 | 4.2ms | 3.9ms | 0.9x | PASS |
| PRODPL1 | 60 | 29 | 40 | Solve_Succee | Solve_Succee | 1.99e-16 | 28 | 28 | 6.6ms | 6.8ms | 1.0x | PASS |
| PSPDOC | 4 | 0 | 4 | Solve_Succee | Solve_Succee | 0.00e+00 | 5 | 5 | 1.2ms | 1.6ms | 1.3x | PASS |
| PT | 2 | 501 | 2 | Solve_Succee | Solve_Succee | 0.00e+00 | 106 | 106 | 76.0ms | 82.8ms | 1.1x | PASS |
| QC | 9 | 4 | 9 | Solve_Succee | Solve_Succee | 5.35e-11 | 49 | 44 | 7.5ms | 9.6ms | 1.3x | PASS |
| QCNEW | 9 | 3 | 9 | Solve_Succee | Solve_Succee | 0.00e+00 | 6 | 6 | 1.4ms | 1.7ms | 1.3x | PASS |
| QPCBLEND | 83 | 74 | 40 | Solve_Succee | Solve_Succee | 1.40e-12 | 19 | 19 | 6.5ms | 5.6ms | 0.9x | PASS |
| QPNBLEND | 83 | 74 | 40 | Solve_Succee | Solve_Succee | 1.04e-17 | 18 | 18 | 6.5ms | 5.4ms | 0.8x | PASS |
| RAT42 | 3 | 9 | -6 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 26us | 238us | 9.2x | BOTH_FAIL |
| RAT42LS | 3 | 0 | 3 | Solve_Succee | Solve_Succee | 0.00e+00 | 28 | 28 | 2.7ms | 5.1ms | 1.9x | PASS |
| RAT43 | 4 | 15 | -11 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 24us | 228us | 9.4x | BOTH_FAIL |
| RAT43LS | 4 | 0 | 4 | Solve_Succee | Solve_Succee | 6.21e-16 | 34 | 34 | 3.1ms | 5.9ms | 1.9x | PASS |
| RECIPE | 3 | 3 | 0 | Solve_Succee | Solve_Succee | 0.00e+00 | 16 | 16 | 1.5ms | 2.9ms | 2.0x | PASS |
| RECIPELS | 3 | 0 | 3 | Solve_Succee | Solve_Succee | 1.34e-17 | 29 | 29 | 2.8ms | 5.4ms | 1.9x | PASS |
| RES | 20 | 14 | 8 | Solve_Succee | Solve_Succee | 0.00e+00 | 10 | 10 | 1.7ms | 2.2ms | 1.3x | PASS |
| RK23 | 17 | 11 | 6 | Solve_Succee | Solve_Succee | 1.39e-17 | 10 | 10 | 2.3ms | 2.8ms | 1.2x | PASS |
| ROBOT | 14 | 2 | 12 | Solve_Succee | Search_Direc | N/A | 20 | 18 | 4.1ms | 4.5ms | 1.1x | ipopt_FAIL |
| ROSENBR | 2 | 0 | 2 | Solve_Succee | Solve_Succee | 1.41e-26 | 21 | 21 | 1.8ms | 3.7ms | 2.0x | PASS |
| ROSENBRTU | 2 | 0 | 2 | Solve_Succee | Solve_Succee | 0.00e+00 | 86 | 87 | 6.6ms | 14.9ms | 2.2x | PASS |
| ROSENMMX | 5 | 4 | 5 | Solve_Succee | Solve_Succee | 5.28e-14 | 13 | 13 | 2.3ms | 3.3ms | 1.4x | PASS |
| ROSZMAN1 | 4 | 25 | -21 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 26us | 231us | 8.8x | BOTH_FAIL |
| ROSZMAN1LS | 4 | 0 | 4 | Solve_Succee | Solve_Succee | 1.73e-18 | 27 | 28 | 2.6ms | 5.0ms | 2.0x | PASS |
| RSNBRNE | 2 | 2 | 0 | Solve_Succee | Solve_Succee | 0.00e+00 | 1 | 1 | 501us | 628us | 1.3x | PASS |
| S268 | 5 | 5 | 5 | Solve_Succee | Solve_Succee | 1.09e-11 | 14 | 14 | 3.0ms | 3.0ms | 1.0x | PASS |
| S308 | 2 | 0 | 2 | Solve_Succee | Solve_Succee | 0.00e+00 | 9 | 9 | 1.2ms | 1.8ms | 1.5x | PASS |
| S308NE | 2 | 3 | -1 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 20us | 228us | 11.2x | BOTH_FAIL |
| S316-322 | 2 | 1 | 1 | Solve_Succee | Solve_Succee | 6.19e-12 | 7 | 7 | 1.1ms | 1.5ms | 1.4x | PASS |
| S365 | 7 | 5 | 7 | Restoration_ | Restoration_ | N/A | 0 | 1 | 328us | 990us | 3.0x | BOTH_FAIL |
| S365MOD | 7 | 5 | 7 | Restoration_ | Restoration_ | N/A | 0 | 1 | 322us | 986us | 3.1x | BOTH_FAIL |
| SANTA | 21 | 23 | -2 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 26us | 228us | 8.7x | BOTH_FAIL |
| SANTALS | 21 | 0 | 21 | Solve_Succee | Solve_Succee | 3.39e-19 | 31 | 31 | 5.2ms | 7.3ms | 1.4x | PASS |
| SIM2BQP | 2 | 0 | 2 | Solve_Succee | Solve_Succee | 2.30e-19 | 5 | 5 | 996us | 1.4ms | 1.4x | PASS |
| SIMBQP | 2 | 0 | 2 | Solve_Succee | Solve_Succee | 4.07e-20 | 5 | 5 | 1.0ms | 1.4ms | 1.3x | PASS |
| SIMPLLPA | 2 | 2 | 2 | Solve_Succee | Solve_Succee | 1.11e-16 | 8 | 8 | 1.6ms | 1.9ms | 1.2x | PASS |
| SIMPLLPB | 2 | 3 | 2 | Solve_Succee | Solve_Succee | 0.00e+00 | 7 | 7 | 1.4ms | 1.9ms | 1.4x | PASS |
| SINEVAL | 2 | 0 | 2 | Solve_Succee | Solve_Succee | 1.22e-40 | 42 | 42 | 3.6ms | 7.0ms | 2.0x | PASS |
| SINVALNE | 2 | 2 | 0 | Solve_Succee | Solve_Succee | 0.00e+00 | 1 | 1 | 568us | 635us | 1.1x | PASS |
| SIPOW1 | 2 | 2000 | 2 | Solve_Succee | Solve_Succee | 9.35e-09 | 82 | 81 | 214.9ms | 219.3ms | 1.0x | PASS |
| SIPOW1M | 2 | 2000 | 2 | Solve_Succee | Solve_Succee | 2.22e-16 | 89 | 88 | 235.3ms | 234.3ms | 1.0x | PASS |
| SIPOW2 | 2 | 2000 | 2 | Solve_Succee | Solve_Succee | 6.98e-10 | 71 | 69 | 192.0ms | 175.6ms | 0.9x | PASS |
| SIPOW2M | 2 | 2000 | 2 | Solve_Succee | Solve_Succee | 1.93e-08 | 69 | 73 | 182.3ms | 185.6ms | 1.0x | PASS |
| SIPOW3 | 4 | 2000 | 4 | Solve_Succee | Solve_Succee | 0.00e+00 | 12 | 12 | 41.8ms | 37.6ms | 0.9x | PASS |
| SIPOW4 | 4 | 2000 | 4 | Solve_Succee | Solve_Succee | 0.00e+00 | 11 | 11 | 48.9ms | 34.8ms | 0.7x | PASS |
| SISSER | 2 | 0 | 2 | Solve_Succee | Solve_Succee | 2.52e-27 | 18 | 18 | 1.5ms | 2.6ms | 1.7x | PASS |
| SISSER2 | 2 | 0 | 2 | Solve_Succee | Solve_Succee | 1.01e-28 | 20 | 20 | 1.8ms | 3.0ms | 1.7x | PASS |
| SNAIL | 2 | 0 | 2 | Solve_Succee | Solve_Succee | 3.03e-29 | 63 | 63 | 5.0ms | 10.2ms | 2.0x | PASS |
| SNAKE | 2 | 2 | 2 | Solve_Succee | Solve_Succee | 1.87e-15 | 8 | 8 | 1.7ms | 2.2ms | 1.3x | PASS |
| SPANHYD | 97 | 33 | 64 | Solve_Succee | Solve_Succee | 5.74e-03 | 29 | 20 | 10.8ms | 5.8ms | 0.5x | MISMATCH |
| SPIRAL | 3 | 2 | 3 | Infeasible_P | Infeasible_P | N/A | 1607 | 370 | 314.1ms | 64.4ms | 0.2x | BOTH_FAIL |
| SSI | 3 | 0 | 3 | Maximum_Iter | Maximum_Iter | N/A | 2999 | 3000 | 203.4ms | 462.9ms | 2.3x | BOTH_FAIL |
| SSINE | 3 | 2 | 1 | Solve_Succee | Solve_Succee | 0.00e+00 | 224 | 224 | 17.3ms | 35.9ms | 2.1x | PASS |
| STANCMIN | 3 | 2 | 3 | Solve_Succee | Solve_Succee | 0.00e+00 | 9 | 9 | 1.6ms | 2.1ms | 1.3x | PASS |
| STRATEC | 10 | 0 | 10 | Solve_Succee | Solve_Succee | 4.93e-15 | 24 | 34 | 2.26s | 3.21s | 1.4x | PASS |
| STREG | 4 | 0 | 4 | Solve_Succee | Solve_Succee | 1.91e-13 | 13 | 13 | 1.5ms | 2.5ms | 1.7x | PASS |
| STREGNE | 4 | 2 | 2 | Solve_Succee | Solve_Succee | 0.00e+00 | 2 | 2 | 681us | 845us | 1.2x | PASS |
| SUPERSIM | 2 | 2 | 0 | Solve_Succee | Solve_Succee | 2.22e-16 | 5 | 1 | 1.1ms | 749us | 0.7x | PASS |
| SWOPF | 83 | 92 | 5 | Solve_Succee | Solve_Succee | 6.62e-14 | 13 | 13 | 5.2ms | 4.1ms | 0.8x | PASS |
| SYNTHES1 | 6 | 6 | 6 | Solve_Succee | Solve_Succee | 3.55e-15 | 8 | 8 | 1.8ms | 2.1ms | 1.2x | PASS |
| SYNTHES2 | 11 | 14 | 10 | Solve_Succee | Solve_Succee | 0.00e+00 | 14 | 14 | 2.5ms | 3.2ms | 1.3x | PASS |
| SYNTHES3 | 17 | 23 | 15 | Solve_Succee | Solve_Succee | 3.89e-15 | 13 | 13 | 2.6ms | 3.2ms | 1.2x | PASS |
| TAME | 2 | 1 | 1 | Solve_Succee | Solve_Succee | 0.00e+00 | 5 | 5 | 974us | 1.3ms | 1.4x | PASS |
| TAX13322 | 72 | 1261 | 72 | Maximum_Iter | Maximum_Iter | N/A | 2999 | 3000 | 11.73s | 13.54s | 1.2x | BOTH_FAIL |
| TAXR13322 | 72 | 1261 | 72 | Solve_Succee | Solved_To_Ac | 9.95e-01 | 2431 | 56 | 9.75s | 2.73s | 0.3x | MISMATCH |
| TENBARS1 | 18 | 9 | 10 | Solve_Succee | Solve_Succee | 0.00e+00 | 39 | 39 | 6.0ms | 7.6ms | 1.3x | PASS |
| TENBARS2 | 18 | 8 | 10 | Solve_Succee | Solve_Succee | 0.00e+00 | 33 | 33 | 4.8ms | 7.0ms | 1.4x | PASS |
| TENBARS3 | 18 | 8 | 10 | Solve_Succee | Solve_Succee | 2.02e-16 | 34 | 34 | 5.1ms | 6.9ms | 1.4x | PASS |
| TENBARS4 | 18 | 9 | 10 | Solve_Succee | Solve_Succee | 1.91e-16 | 14 | 14 | 2.7ms | 3.6ms | 1.3x | PASS |
| TFI1 | 3 | 101 | 3 | Solve_Succee | Solve_Succee | 0.00e+00 | 19 | 19 | 6.5ms | 7.0ms | 1.1x | PASS |
| TFI2 | 3 | 101 | 3 | Solve_Succee | Solve_Succee | 1.21e-13 | 8 | 8 | 2.8ms | 3.0ms | 1.1x | PASS |
| TFI3 | 3 | 101 | 3 | Solve_Succee | Solve_Succee | 1.33e-10 | 14 | 13 | 4.2ms | 4.5ms | 1.1x | PASS |
| THURBER | 7 | 37 | -30 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 28us | 246us | 8.7x | BOTH_FAIL |
| THURBERLS | 7 | 0 | 7 | Solve_Succee | Solve_Succee | 1.13e-15 | 19 | 19 | 2.0ms | 3.8ms | 1.9x | PASS |
| TOINTGOR | 50 | 0 | 50 | Solve_Succee | Solve_Succee | 3.31e-16 | 7 | 7 | 1.2ms | 1.6ms | 1.3x | PASS |
| TOINTPSP | 50 | 0 | 50 | Solve_Succee | Solve_Succee | 0.00e+00 | 20 | 20 | 2.7ms | 4.9ms | 1.8x | PASS |
| TOINTQOR | 50 | 0 | 50 | Solve_Succee | Solve_Succee | 0.00e+00 | 1 | 1 | 623us | 670us | 1.1x | PASS |
| TRIGGER | 7 | 6 | 1 | Solve_Succee | Solve_Succee | 0.00e+00 | 15 | 15 | 1.6ms | 2.8ms | 1.8x | PASS |
| TRO3X3 | 30 | 13 | 18 | Solve_Succee | Solve_Succee | 6.18e-03 | 90 | 47 | 18.6ms | 10.4ms | 0.6x | MISMATCH |
| TRO4X4 | 63 | 25 | 39 | Diverging_It | Diverging_It | N/A | 246 | 157 | 82.2ms | 45.4ms | 0.6x | BOTH_FAIL |
| TRO6X2 | 45 | 21 | 25 | Infeasible_P | Restoration_ | N/A | 380 | 353 | 280.7ms | 96.5ms | 0.3x | BOTH_FAIL |
| TRUSPYR1 | 11 | 4 | 8 | Solve_Succee | Solve_Succee | 0.00e+00 | 10 | 10 | 1.9ms | 2.3ms | 1.2x | PASS |
| TRUSPYR2 | 11 | 11 | 8 | Solve_Succee | Solve_Succee | 3.16e-16 | 13 | 13 | 2.6ms | 3.0ms | 1.1x | PASS |
| TRY-B | 2 | 1 | 1 | Solve_Succee | Solve_Succee | 0.00e+00 | 23 | 23 | 3.0ms | 4.8ms | 1.6x | PASS |
| TWOBARS | 2 | 2 | 2 | Solve_Succee | Solve_Succee | 0.00e+00 | 8 | 8 | 1.5ms | 2.0ms | 1.3x | PASS |
| VESUVIA | 8 | 1025 | -1017 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 72us | 426us | 5.9x | BOTH_FAIL |
| VESUVIALS | 8 | 0 | 8 | Solve_Succee | Solve_Succee | 0.00e+00 | 51 | 48 | 16.9ms | 19.0ms | 1.1x | PASS |
| VESUVIO | 8 | 1025 | -1017 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 84us | 433us | 5.1x | BOTH_FAIL |
| VESUVIOLS | 8 | 0 | 8 | Solve_Succee | Solve_Succee | 0.00e+00 | 10 | 10 | 4.1ms | 5.0ms | 1.2x | PASS |
| VESUVIOU | 8 | 1025 | -1017 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 69us | 422us | 6.1x | BOTH_FAIL |
| VESUVIOULS | 8 | 0 | 8 | Solve_Succee | Solve_Succee | 5.55e-17 | 8 | 8 | 3.4ms | 3.7ms | 1.1x | PASS |
| VIBRBEAM | 8 | 0 | 8 | Solve_Succee | Solve_Succee | 1.67e-15 | 58 | 58 | 6.2ms | 10.5ms | 1.7x | PASS |
| VIBRBEAMNE | 8 | 30 | -22 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 29us | 239us | 8.3x | BOTH_FAIL |
| WACHBIEG | 3 | 2 | 1 | Infeasible_P | Infeasible_P | N/A | 10 | 15 | 3.0ms | 3.7ms | 1.2x | BOTH_FAIL |
| WATER | 31 | 10 | 21 | Solve_Succee | Solve_Succee | 3.13e-12 | 18 | 17 | 3.7ms | 3.7ms | 1.0x | PASS |
| WAYSEA1 | 2 | 0 | 2 | Solve_Succee | Solve_Succee | 3.94e-31 | 14 | 14 | 1.3ms | 2.1ms | 1.6x | PASS |
| WAYSEA1B | 2 | 0 | 2 | Solve_Succee | Solve_Succee | 0.00e+00 | 14 | 14 | 1.9ms | 2.9ms | 1.5x | PASS |
| WAYSEA1NE | 2 | 2 | 0 | Solve_Succee | Solve_Succee | 0.00e+00 | 7 | 7 | 1.0ms | 1.4ms | 1.4x | PASS |
| WAYSEA2 | 2 | 0 | 2 | Solve_Succee | Solve_Succee | 7.08e-24 | 22 | 22 | 1.6ms | 2.9ms | 1.9x | PASS |
| WAYSEA2B | 2 | 0 | 2 | Solve_Succee | Solve_Succee | 0.00e+00 | 22 | 22 | 2.5ms | 3.9ms | 1.6x | PASS |
| WAYSEA2NE | 2 | 2 | 0 | Solve_Succee | Solve_Succee | 0.00e+00 | 11 | 11 | 1.2ms | 2.0ms | 1.6x | PASS |
| WEEDS | 3 | 0 | 3 | Solve_Succee | Solve_Succee | 3.09e-15 | 28 | 28 | 4.0ms | 6.5ms | 1.6x | PASS |
| WEEDSNE | 3 | 12 | -9 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 22us | 226us | 10.5x | BOTH_FAIL |
| WOMFLET | 3 | 3 | 3 | Solve_Succee | Solve_Succee | 0.00e+00 | 8 | 8 | 1.5ms | 2.0ms | 1.4x | PASS |
| YFIT | 3 | 0 | 3 | Solve_Succee | Solve_Succee | 5.31e-21 | 36 | 36 | 4.0ms | 7.1ms | 1.8x | PASS |
| YFITNE | 3 | 17 | -14 | Not_Enough_D | Not_Enough_D | N/A | 0 | 0 | 21us | 223us | 10.6x | BOTH_FAIL |
| YFITU | 3 | 0 | 3 | Solve_Succee | Solve_Succee | 4.59e-21 | 36 | 36 | 2.8ms | 5.6ms | 2.0x | PASS |
| ZANGWIL2 | 2 | 0 | 2 | Solve_Succee | Solve_Succee | 0.00e+00 | 1 | 1 | 471us | 570us | 1.2x | PASS |
| ZANGWIL3 | 3 | 3 | 0 | Solve_Succee | Solve_Succee | 0.00e+00 | 1 | 1 | 576us | 613us | 1.1x | PASS |
| ZECEVIC2 | 2 | 2 | 2 | Solve_Succee | Solve_Succee | 0.00e+00 | 8 | 8 | 1.6ms | 2.0ms | 1.3x | PASS |
| ZECEVIC3 | 2 | 2 | 2 | Solve_Succee | Solve_Succee | 0.00e+00 | 17 | 17 | 2.6ms | 3.5ms | 1.3x | PASS |
| ZECEVIC4 | 2 | 2 | 2 | Solve_Succee | Solve_Succee | 0.00e+00 | 10 | 10 | 1.8ms | 2.4ms | 1.4x | PASS |
| ZY2 | 3 | 2 | 3 | Solve_Succee | Solve_Succee | 0.00e+00 | 14 | 14 | 2.7ms | 3.3ms | 1.2x | PASS |

## Performance Comparison (where both solve)

### Iteration Comparison

| Metric | pounce | Ipopt |
|--------|--------|-------|
| Mean   | 43.3 | 42.5 |
| Median | 13 | 13 |
| Total  | 24144 | 23660 |

- pounce fewer iterations: 34/557
- Ipopt fewer iterations: 63/557
- Tied: 460/557

### Timing Comparison

| Metric | pounce | Ipopt |
|--------|--------|-------|
| Mean   | 52.4ms | 41.7ms |
| Median | 2.1ms | 2.9ms |
| Total  | 29.17s | 23.25s |

- Geometric mean speedup (Ipopt_time/pounce_time): **1.34x**
  - \>1 means pounce is faster, <1 means Ipopt is faster
- pounce faster: 491/557 problems
- Ipopt faster: 66/557 problems
- Overall speedup (total time): 0.80x

## Failure Analysis

### Problems where only pounce fails (5)

| Problem | n | m | pounce status | Ipopt obj |
|---------|---|---|---------------|-----------|
| CRESC50 | 6 | 100 | Infeasible_Problem_Detected | 7.862467e-01 |
| DMN15102LS | 66 | 0 | Timeout | 6.637446e+02 |
| HATFLDFL | 3 | 0 | Maximum_Iterations_Exceeded | 6.016804e-05 |
| MSS1 | 90 | 73 | Maximum_Iterations_Exceeded | -1.400000e+01 |
| PFIT4 | 3 | 3 | Infeasible_Problem_Detected | 0.000000e+00 |

### Problems where only Ipopt fails (5)

| Problem | n | m | Ipopt status | pounce obj |
|---------|---|---|--------------|------------|
| DIAMON2DLS | 66 | 0 | Timeout | 6.325671e+02 |
| EQC | 9 | 3 | Error_In_Step_Computation | -8.630052e+02 |
| HIMMELBJ | 45 | 14 | Error_In_Step_Computation | N/A |
| POLAK6 | 5 | 4 | Maximum_Iterations_Exceeded | -4.400000e+01 |
| ROBOT | 14 | 2 | Search_Direction_Becomes_Too_Small | 6.593299e+00 |

### Problems where both fail (160)

| Problem | n | m | pounce status | Ipopt status |
|---------|---|---|---------------|--------------|
| ARGAUSS | 3 | 15 | Not_Enough_Degrees_Of_Freedom | Not_Enough_Degrees_Of_Freedom |
| AVION2 | 49 | 15 | Maximum_Iterations_Exceeded | Maximum_Iterations_Exceeded |
| BARDNE | 3 | 15 | Not_Enough_Degrees_Of_Freedom | Not_Enough_Degrees_Of_Freedom |
| BEALENE | 2 | 3 | Not_Enough_Degrees_Of_Freedom | Not_Enough_Degrees_Of_Freedom |
| BENNETT5 | 3 | 154 | Not_Enough_Degrees_Of_Freedom | Not_Enough_Degrees_Of_Freedom |
| BIGGS6NE | 6 | 13 | Not_Enough_Degrees_Of_Freedom | Not_Enough_Degrees_Of_Freedom |
| BLEACHNG | 17 | 0 | Timeout | Timeout |
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
| DECONVB | 63 | 0 | Maximum_Iterations_Exceeded | Maximum_Iterations_Exceeded |
| DENSCHNBNE | 2 | 3 | Not_Enough_Degrees_Of_Freedom | Not_Enough_Degrees_Of_Freedom |
| DENSCHNENE | 3 | 3 | Infeasible_Problem_Detected | Infeasible_Problem_Detected |
| DEVGLA1NE | 4 | 24 | Not_Enough_Degrees_Of_Freedom | Not_Enough_Degrees_Of_Freedom |
| DEVGLA2NE | 5 | 16 | Not_Enough_Degrees_Of_Freedom | Not_Enough_Degrees_Of_Freedom |
| DIAMON2D | 66 | 4643 | Not_Enough_Degrees_Of_Freedom | Not_Enough_Degrees_Of_Freedom |
| DIAMON3D | 99 | 4643 | Not_Enough_Degrees_Of_Freedom | Not_Enough_Degrees_Of_Freedom |
| DIAMON3DLS | 99 | 0 | Timeout | Timeout |
| DMN15102 | 66 | 4643 | Not_Enough_Degrees_Of_Freedom | Not_Enough_Degrees_Of_Freedom |
| DMN15103 | 99 | 4643 | Not_Enough_Degrees_Of_Freedom | Not_Enough_Degrees_Of_Freedom |
| DMN15103LS | 99 | 0 | Timeout | Timeout |
| DMN15332 | 66 | 4643 | Not_Enough_Degrees_Of_Freedom | Not_Enough_Degrees_Of_Freedom |
| DMN15332LS | 66 | 0 | Timeout | Timeout |
| DMN15333 | 99 | 4643 | Not_Enough_Degrees_Of_Freedom | Not_Enough_Degrees_Of_Freedom |
| DMN15333LS | 99 | 0 | Timeout | Timeout |
| DMN37142 | 66 | 4643 | Not_Enough_Degrees_Of_Freedom | Not_Enough_Degrees_Of_Freedom |
| DMN37142LS | 66 | 0 | Timeout | Timeout |
| DMN37143 | 99 | 4643 | Not_Enough_Degrees_Of_Freedom | Not_Enough_Degrees_Of_Freedom |
| DMN37143LS | 99 | 0 | Timeout | Timeout |
| ECKERLE4 | 3 | 35 | Not_Enough_Degrees_Of_Freedom | Not_Enough_Degrees_Of_Freedom |
| EGGCRATENE | 2 | 4 | Not_Enough_Degrees_Of_Freedom | Not_Enough_Degrees_Of_Freedom |
| ELATVIDUNE | 2 | 3 | Not_Enough_Degrees_Of_Freedom | Not_Enough_Degrees_Of_Freedom |
| ENGVAL2NE | 3 | 5 | Not_Enough_Degrees_Of_Freedom | Not_Enough_Degrees_Of_Freedom |
| ENSO | 9 | 168 | Not_Enough_Degrees_Of_Freedom | Not_Enough_Degrees_Of_Freedom |
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
| NASH | 72 | 24 | Infeasible_Problem_Detected | Infeasible_Problem_Detected |
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
| PALMER5A | 8 | 0 | Maximum_Iterations_Exceeded | Maximum_Iterations_Exceeded |
| PALMER5ANE | 8 | 12 | Not_Enough_Degrees_Of_Freedom | Not_Enough_Degrees_Of_Freedom |
| PALMER5BNE | 9 | 12 | Not_Enough_Degrees_Of_Freedom | Not_Enough_Degrees_Of_Freedom |
| PALMER5E | 8 | 0 | Maximum_Iterations_Exceeded | Maximum_Iterations_Exceeded |
| PALMER5ENE | 8 | 12 | Not_Enough_Degrees_Of_Freedom | Not_Enough_Degrees_Of_Freedom |
| PALMER6ANE | 6 | 13 | Not_Enough_Degrees_Of_Freedom | Not_Enough_Degrees_Of_Freedom |
| PALMER6ENE | 8 | 13 | Not_Enough_Degrees_Of_Freedom | Not_Enough_Degrees_Of_Freedom |
| PALMER7A | 6 | 0 | Maximum_Iterations_Exceeded | Maximum_Iterations_Exceeded |
| PALMER7ANE | 6 | 13 | Not_Enough_Degrees_Of_Freedom | Not_Enough_Degrees_Of_Freedom |
| PALMER7E | 8 | 0 | Maximum_Iterations_Exceeded | Maximum_Iterations_Exceeded |
| PALMER7ENE | 8 | 13 | Not_Enough_Degrees_Of_Freedom | Not_Enough_Degrees_Of_Freedom |
| PALMER8ANE | 6 | 12 | Not_Enough_Degrees_Of_Freedom | Not_Enough_Degrees_Of_Freedom |
| PALMER8ENE | 8 | 12 | Not_Enough_Degrees_Of_Freedom | Not_Enough_Degrees_Of_Freedom |
| PFIT1 | 3 | 3 | Infeasible_Problem_Detected | Infeasible_Problem_Detected |
| PFIT2 | 3 | 3 | Infeasible_Problem_Detected | Restoration_Failed |
| POLAK3 | 12 | 10 | Restoration_Failed | Maximum_Iterations_Exceeded |
| POWELLSQ | 2 | 2 | Infeasible_Problem_Detected | Infeasible_Problem_Detected |
| RAT42 | 3 | 9 | Not_Enough_Degrees_Of_Freedom | Not_Enough_Degrees_Of_Freedom |
| RAT43 | 4 | 15 | Not_Enough_Degrees_Of_Freedom | Not_Enough_Degrees_Of_Freedom |
| ROSZMAN1 | 4 | 25 | Not_Enough_Degrees_Of_Freedom | Not_Enough_Degrees_Of_Freedom |
| S308NE | 2 | 3 | Not_Enough_Degrees_Of_Freedom | Not_Enough_Degrees_Of_Freedom |
| S365 | 7 | 5 | Restoration_Failed | Restoration_Failed |
| S365MOD | 7 | 5 | Restoration_Failed | Restoration_Failed |
| SANTA | 21 | 23 | Not_Enough_Degrees_Of_Freedom | Not_Enough_Degrees_Of_Freedom |
| SPIRAL | 3 | 2 | Infeasible_Problem_Detected | Infeasible_Problem_Detected |
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

### Objective mismatches (10)

Both solvers converged but found different objective values (rel diff > 1e-4).

- **Different local minimum** (both Optimal): 0
- **Convergence gap** (one Acceptable): 10
- **Better objective found by**: pounce 4, Ipopt 6

| Problem | pounce obj | Ipopt obj | Rel Diff | r_status | i_status | Better |
|---------|-----------|-----------|----------|----------|----------|--------|
| HS84 | -1.175337e+09 | -5.280335e+06 | 9.96e-01 | Solve_Succ | Solve_Succ | pounce |
| TAXR13322 | -6.449419e+04 | -3.429089e+02 | 9.95e-01 | Solve_Succ | Solved_To_ | pounce |
| HALDMADS | 3.294056e-02 | 2.218282e+00 | 9.85e-01 | Solve_Succ | Solve_Succ | pounce |
| OET4 | 8.576995e-01 | 4.295421e-03 | 8.53e-01 | Solve_Succ | Solve_Succ | ipopt |
| DISCS | 1.528822e+01 | 1.200007e+01 | 2.15e-01 | Solve_Succ | Solve_Succ | ipopt |
| PALMER2E | 1.163043e-01 | 2.065001e-04 | 1.16e-01 | Solve_Succ | Solve_Succ | ipopt |
| DALLASS | -3.090997e+04 | -3.239322e+04 | 4.58e-02 | Solve_Succ | Solve_Succ | ipopt |
| TRO3X3 | 8.912035e+00 | 8.967478e+00 | 6.18e-03 | Solve_Succ | Solve_Succ | pounce |
| SPANHYD | 2.411214e+02 | 2.397380e+02 | 5.74e-03 | Solve_Succ | Solve_Succ | ipopt |
| LIN | -1.451448e-02 | -1.757754e-02 | 3.06e-03 | Solve_Succ | Solve_Succ | ipopt |

---
*Generated by benchmarks/cutest/compare.py*