# POUNCE-QP vs Clarabel vs POUNCE-NLP vs POUNCE active-set — Maros-Meszaros QP benchmark

> **STALE for the pounce QP-IPM column (2026-09-01).** This report was
> generated at 0.9.0, before `bound_relax_factor` was made opt-in on the convex
> arm. Its pounce QP-IPM figures are from the *unrelaxed* arm, and are the
> evidence that reversing gh #744 was right — 137/138 correct, 0
> solved-but-wrong, while `LISWET1(re=2.5e-01)` sits in Ipopt-MA57's wrong
> column and `LISWET1(re=2.7e-01)` in Clarabel's, against a ground truth of
> `36.1224`.
>
> Re-measured on 2026-09-01 against the same DOC 97/6 optima and the same
> `|obj-opt| <= 1e-5 + 1e-4·max(|obj|,|opt|)` rule, the convex arm is
> **138/138 correct** (from **130/138** with the widening), recovering `YAO`
> and `LISWET1/7/8/9/10/11/12` — precisely this file's list of Ipopt-MA57's
> wrong objectives — with 0 newly wrong, 0 status changes, and total iterations
> 3164 → 2658.
>
> The Clarabel and Ipopt columns are unaffected: neither solver changed.
>
> **Not regenerated here** because a faithful run needs Clarabel installed and
> it was not available on this machine; a partial regeneration would drop arms
> and be worse than this note. To refresh:
> `python3 benchmarks/scripts/compare_qp_four_way.py` after
> `cargo build --release --bin pounce` and `pip install clarabel`.

138 problems; 138 with ground-truth optima (DOC 97/6, BPMPD reference). A solve is **correct** when `|obj-opt| <= 1e-05 + 0.0001·max(|obj|,|opt|)`.

Produced by **pounce 0.9.0 (commit 87051c11+dirty, built 2026-07-28T19:36:41Z)**. Numbers here are only comparable to another run of the same binary: the previous committed report was six weeks stale, and the NLP column moved 129/138 → 105/138 across that gap on nothing but binary drift. Rebuild before regenerating (`cargo build --release --bin pounce`).

### pounce QP-IPM (solver_selection=qp-ipm)
- Solved (own status): **137/138**
- Correct vs ground truth: **137/138**
- Solved-but-wrong (status OK, obj off): **0**
- Median rel-err on correct solves: 1.4e-09
### Clarabel
- Solved (own status): **133/138**
- Correct vs ground truth: **124/138**
- Solved-but-wrong (status OK, obj off): **9**
- Median rel-err on correct solves: 2.3e-09
- Wrong objectives: YAO(re=5.4e-01), UBH1(re=5.6e-04), LISWET7(re=9.2e-01), LISWET10(re=4.9e-01), LISWET1(re=2.7e-01), LISWET8(re=9.2e-01), LISWET11(re=2.6e-01), LISWET12(re=8.6e-01), LISWET9(re=7.2e-01)
### pounce NLP (solver_selection=nlp)
- Solved (own status): **113/138**
- Correct vs ground truth: **105/138**
- Solved-but-wrong (status OK, obj off): **8**
- Median rel-err on correct solves: 1.4e-08
- Wrong objectives: YAO(re=7.7e-03), LISWET7(re=2.2e-01), LISWET10(re=2.1e-01), LISWET1(re=2.5e-01), LISWET8(re=8.9e-02), LISWET11(re=6.0e-02), LISWET12(re=3.7e-02), LISWET9(re=3.3e-02)
### pounce active-set QP (solver_selection=qp-active-set)
- Solved (own status): **46/138**
- Correct vs ground truth: **46/138**
- Solved-but-wrong (status OK, obj off): **0**
- Median rel-err on correct solves: 1.0e-09
### Speed (geomean over 99 all-three-correct problems)
- pounce QP-IPM : 0.050s
- Clarabel      : 0.008s
- pounce NLP    : 0.059s
- QP-IPM vs Clarabel: 6.46×  (Clarabel faster)
- QP-IPM vs NLP     : 1.18×  (QP-IPM faster)

Basis note: membership is now `solve_time is not None` rather than a truthiness test. The old form dropped any solve finishing in under a millisecond (0.0 is falsy), which biased the geomean upward and excluded *every* correct active-set solve. Counts here may therefore differ slightly from reports generated before that fix. The active-set engine is timed separately below, on its own pairwise basis, so adding it does not move these three figures.

### Speed — active-set vs QP-IPM

**Not reported: the active-set path does not populate `total_wallclock_time_secs` in `--json-output`** (it emits `0.0` regardless of actual runtime, while `qp-ipm` on the same problem reports correctly). Any geomean built on that field would read as "instantaneous" and be meaningless. Tracked separately; this section will populate once the field is correct.

### Problems where pounce-QP is correct but another solver is not

| problem | opt | clarabel | nlp | qp-active-set |
|---|---|---|---|---|
| QAFIRO | -1.59078 | ✓ | SearchDirectionBecomesTooSmall | ✓ |
| QSCAGR7 | 2.68659e+07 | ✓ | ✓ | SearchDirectionBecomesTooSmall |
| QSC205 | -0.00581395 | ✓ | ✓ | MaximumIterationsExceeded |
| QRECIPE | -266.616 | ✓ | ✓ | SearchDirectionBecomesTooSmall |
| QSHARE2B | 11703.7 | ✓ | ✓ | SearchDirectionBecomesTooSmall |
| QADLITTL | 480319 | ✓ | ✓ | SearchDirectionBecomesTooSmall |
| CVXQP1_S | 11590.7 | ✓ | ✓ | SearchDirectionBecomesTooSmall |
| CVXQP3_S | 11943.4 | ✓ | ✓ | SearchDirectionBecomesTooSmall |
| QPCBLEND | -0.00784254 | ✓ | ✓ | SearchDirectionBecomesTooSmall |
| DUALC2 | 3551.31 | ✓ | ✓ | SearchDirectionBecomesTooSmall |
| PRIMALC2 | -3551.31 | DualInfeasible | ✓ | ✓ |
| QSCTAP1 | 1415.86 | ✓ | ✓ | SearchDirectionBecomesTooSmall |
| DUALC5 | 427.232 | ✓ | ✓ | SearchDirectionBecomesTooSmall |
| DUALC1 | 6155.25 | ✓ | ✓ | SearchDirectionBecomesTooSmall |
| QSHARE1B | 720078 | ✓ | ErrorInStepComputation | SearchDirectionBecomesTooSmall |
| QSCAGR25 | 2.01738e+08 | ✓ | InfeasibleProblemDetected | SearchDirectionBecomesTooSmall |
| QSCORPIO | 1880.51 | ✓ | InfeasibleProblemDetected | SearchDirectionBecomesTooSmall |
| QPCBOEI2 | 8.17196e+06 | ✓ | ✓ | SearchDirectionBecomesTooSmall |
| PRIMALC1 | -6155.25 | DualInfeasible | ✓ | ✓ |
| QISRAEL | 2.53478e+07 | InsufficientProgress | ✓ | InternalError |
| QSCSD1 | 8.66667 | ✓ | InfeasibleProblemDetected | SearchDirectionBecomesTooSmall |
| DPKLO1 | 0.370096 | ✓ | SearchDirectionBecomesTooSmall | ✓ |
| QBORE3D | 3100.2 | ✓ | SearchDirectionBecomesTooSmall | SearchDirectionBecomesTooSmall |
| QBRANDY | 28375.1 | ✓ | ErrorInStepComputation | SearchDirectionBecomesTooSmall |
| QSTANDAT | 6411.84 | ✓ | SearchDirectionBecomesTooSmall | SearchDirectionBecomesTooSmall |
| QGFRDXPN | 1.00791e+11 | ✓ | ✓ | SearchDirectionBecomesTooSmall |
| QBEACONF | 164712 | ✓ | SearchDirectionBecomesTooSmall | SearchDirectionBecomesTooSmall |
| DUALC8 | 18309.4 | ✓ | ✓ | SearchDirectionBecomesTooSmall |
| QCAPRI | 6.67933e+07 | ✓ | ✓ | SearchDirectionBecomesTooSmall |
| QSCSD6 | 50.8082 | ✓ | ✓ | SearchDirectionBecomesTooSmall |
| QE226 | 212.653 | ✓ | ✓ | SearchDirectionBecomesTooSmall |
| QSCFXM1 | 1.68827e+07 | ✓ | InfeasibleProblemDetected | SearchDirectionBecomesTooSmall |
| QETAMACR | 86760.4 | ✓ | InfeasibleProblemDetected | SearchDirectionBecomesTooSmall |
| QFORPLAN | 7.45663e+09 | ✓ | InfeasibleProblemDetected | SearchDirectionBecomesTooSmall |
| QGROW7 | -4.27987e+07 | ✓ | ErrorInStepComputation | InternalError |
| QBANDM | 16352.3 | ✓ | InfeasibleProblemDetected | SearchDirectionBecomesTooSmall |
| QSEBA | 8.14818e+07 | ✓ | ✓ | SearchDirectionBecomesTooSmall |
| GOULDQP2 | 0.000184275 | ✓ | ✓ | SearchDirectionBecomesTooSmall |
| LASER | 2.4096e+06 | ✓ | ✓ | SearchDirectionBecomesTooSmall |
| QSHIP04S | 2.42499e+06 | ✓ | ✓ | SearchDirectionBecomesTooSmall |
| QSCTAP2 | 1735.03 | ✓ | ✓ | SearchDirectionBecomesTooSmall |
| QPCBOEI1 | 1.15039e+07 | ✓ | ✓ | SearchDirectionBecomesTooSmall |
| QFFFFF80 | 873147 | ✓ | InfeasibleProblemDetected | InternalError |
| QSCRS8 | 904.56 | ✓ | InfeasibleProblemDetected | SearchDirectionBecomesTooSmall |
| QSIERRA | 2.37505e+07 | ✓ | ✓ | SearchDirectionBecomesTooSmall |
| QSHIP04L | 2.42002e+06 | ✓ | ✓ | SearchDirectionBecomesTooSmall |
| GOULDQP3 | 2.06278 | ✓ | ✓ | SearchDirectionBecomesTooSmall |
| QSCTAP3 | 1438.75 | ✓ | ✓ | SearchDirectionBecomesTooSmall |
| QSCSD8 | 940.764 | ✓ | InfeasibleProblemDetected | SearchDirectionBecomesTooSmall |
| QGROW15 | -1.01694e+08 | ✓ | ErrorInStepComputation | TimeOut |
| QSCFXM2 | 2.77762e+07 | ✓ | InfeasibleProblemDetected | SearchDirectionBecomesTooSmall |
| QSTAIR | 7.98545e+06 | ✓ | InfeasibleProblemDetected | SearchDirectionBecomesTooSmall |
| AUG3DQP | 675.238 | ✓ | ✓ | SearchDirectionBecomesTooSmall |
| QPCSTAIR | 6.20439e+06 | ✓ | ✓ | SearchDirectionBecomesTooSmall |
| CVXQP2_M | 820155 | ✓ | ✓ | SearchDirectionBecomesTooSmall |
| YAO | 197.704 | off re=5.4e-01 | off re=7.7e-03 | SearchDirectionBecomesTooSmall |
| AUG3DCQP | 993.362 | ✓ | ✓ | SearchDirectionBecomesTooSmall |
| CVXQP1_M | 1.08751e+06 | ✓ | ✓ | SearchDirectionBecomesTooSmall |
| QGROW22 | -1.49629e+08 | ✓ | ✓ | TimeOut |
| CVXQP3_M | 1.36283e+06 | ✓ | ✓ | SearchDirectionBecomesTooSmall |
| QSHIP08S | 2.38573e+06 | ✓ | ✓ | SearchDirectionBecomesTooSmall |
| QSCFXM3 | 3.08164e+07 | ✓ | InfeasibleProblemDetected | SearchDirectionBecomesTooSmall |
| QSHELL | 1.57264e+12 | ✓ | ✓ | SearchDirectionBecomesTooSmall |
| CONT-050 | -4.56385 | ✓ | ✓ | SearchDirectionBecomesTooSmall |
| QSHIP12S | 3.05696e+06 | ✓ | ✓ | SearchDirectionBecomesTooSmall |
| QPILOTNO | 4.72859e+06 | ✓ | InfeasibleProblemDetected | InternalError |
| QSHIP08L | 2.37604e+06 | ✓ | ✓ | SearchDirectionBecomesTooSmall |
| Q25FV47 | 1.37444e+07 | ✓ | ✓ | SearchDirectionBecomesTooSmall |
| POWELL20 | 5.20896e+10 | PrimalInfeasible | ✓ | SearchDirectionBecomesTooSmall |
| STCQP2 | 22327.3 | ✓ | ✓ | SearchDirectionBecomesTooSmall |
| STCQP1 | 155144 | ✓ | ✓ | SearchDirectionBecomesTooSmall |
| CONT-101 | 0.195527 | ✓ | ✓ | SearchDirectionBecomesTooSmall |
| QSHIP12L | 3.01888e+06 | ✓ | ✓ | SearchDirectionBecomesTooSmall |
| UBH1 | 1.116 | off re=5.6e-04 | ✓ | ✓ |
| DTOC3 | 235.262 | ✓ | InfeasibleProblemDetected | ✓ |
| LISWET5 | 25.0343 | ✓ | ✓ | TimeOut |
| LISWET6 | 24.9957 | ✓ | ✓ | TimeOut |
| LISWET7 | 498.841 | off re=9.2e-01 | off re=2.2e-01 | ✓ |
| LISWET10 | 49.4858 | off re=4.9e-01 | off re=2.1e-01 | TimeOut |
| LISWET1 | 36.1224 | off re=2.7e-01 | off re=2.5e-01 | ✓ |
| LISWET8 | 714.47 | off re=9.2e-01 | off re=8.9e-02 | TimeOut |
| LISWET11 | 49.524 | off re=2.6e-01 | off re=6.0e-02 | SearchDirectionBecomesTooSmall |
| LISWET12 | 1736.93 | off re=8.6e-01 | off re=3.7e-02 | ✓ |
| LISWET9 | 1963.25 | off re=7.2e-01 | off re=3.3e-02 | ✓ |
| LISWET3 | 25.0012 | ✓ | ✓ | TimeOut |
| LISWET4 | 25.0001 | ✓ | ✓ | TimeOut |
| KSIP | 0.575798 | ✓ | ✓ | SearchDirectionBecomesTooSmall |
| CONT-100 | -4.6444 | ✓ | ✓ | SearchDirectionBecomesTooSmall |
| AUG2DQP | 6.23701e+06 | ✓ | ✓ | SearchDirectionBecomesTooSmall |
| AUG2DCQP | 6.49813e+06 | ✓ | ✓ | SearchDirectionBecomesTooSmall |
| HUESTIS | 3.48247e+11 | ✓ | ✓ | SearchDirectionBecomesTooSmall |
| HUES-MOD | 3.48247e+07 | ✓ | ✓ | SearchDirectionBecomesTooSmall |
| CVXQP2_L | 8.18425e+07 | ✓ | ✓ | SearchDirectionBecomesTooSmall |
| CVXQP1_L | 1.08705e+08 | ✓ | ✓ | TimeOut |
| CVXQP3_L | 1.15711e+08 | ✓ | ✓ | TimeOut |
| CONT-201 | 0.192483 | ✓ | ✓ | TimeOut |
| CONT-200 | -4.68488 | ✓ | ✓ | SearchDirectionBecomesTooSmall |
| CONT-300 | 0.191512 | ✓ | ✓ | TimeOut |
| BOYD2 | 21.2568 | MaxIterations | TimeOut | TimeOut |
| BOYD1 | -6.17352e+07 | ✓ | ✓ | SearchDirectionBecomesTooSmall |
| EXDATA | -141.843 | ✓ | ✓ | TimeOut |

### Active-set QP: 92 of 138 not matching ground truth

Of these, **0** report a successful status with a wrong objective (the dangerous kind); the rest fail visibly.

| problem | n | m | opt | status | objective | rel-err |
|---|---|---|---|---|---|---|
| AUG2DCQP | 20200 | 30200 | 6.49813e+06 | SearchDirectionBecomesTooSmall | 10050.5 | 1.0e+00 |
| AUG2DQP | 20200 | 30200 | 6.23701e+06 | SearchDirectionBecomesTooSmall | 9801 | 1.0e+00 |
| AUG3DCQP | 3873 | 4873 | 993.362 | SearchDirectionBecomesTooSmall | 1693.5 | 4.1e-01 |
| AUG3DQP | 3873 | 4873 | 675.238 | SearchDirectionBecomesTooSmall | 1093.5 | 3.8e-01 |
| BOYD1 | 93261 | 93279 | -6.17352e+07 | SearchDirectionBecomesTooSmall | -2.15855e+07 | 6.5e-01 |
| BOYD2 | 93263 | 279794 | 21.2568 | TimeOut | — | — |
| CONT-050 | 2597 | 4998 | -4.56385 | SearchDirectionBecomesTooSmall | 0 | 1.0e+00 |
| CONT-100 | 10197 | 19998 | -4.6444 | SearchDirectionBecomesTooSmall | 0 | 1.0e+00 |
| CONT-101 | 10197 | 20295 | 0.195527 | SearchDirectionBecomesTooSmall | 0 | 1.0e+00 |
| CONT-200 | 40397 | 79998 | -4.68488 | SearchDirectionBecomesTooSmall | 0 | 1.0e+00 |
| CONT-201 | 40397 | 80595 | 0.192483 | TimeOut | — | — |
| CONT-300 | 90597 | 180895 | 0.191512 | TimeOut | — | — |
| CVXQP1_L | 10000 | 15000 | 1.08705e+08 | TimeOut | — | — |
| CVXQP1_M | 1000 | 1500 | 1.08751e+06 | SearchDirectionBecomesTooSmall | 22522.5 | 9.8e-01 |
| CVXQP1_S | 100 | 150 | 11590.7 | SearchDirectionBecomesTooSmall | 227.25 | 9.8e-01 |
| CVXQP2_L | 10000 | 12500 | 8.18425e+07 | SearchDirectionBecomesTooSmall | 2.25023e+06 | 9.7e-01 |
| CVXQP2_M | 1000 | 1250 | 820155 | SearchDirectionBecomesTooSmall | 22522.5 | 9.7e-01 |
| CVXQP3_L | 10000 | 17500 | 1.15711e+08 | TimeOut | — | — |
| CVXQP3_M | 1000 | 1750 | 1.36283e+06 | SearchDirectionBecomesTooSmall | 22522.5 | 9.8e-01 |
| CVXQP3_S | 100 | 175 | 11943.4 | SearchDirectionBecomesTooSmall | 227.25 | 9.8e-01 |
| DUALC1 | 9 | 224 | 6155.25 | SearchDirectionBecomesTooSmall | 0 | 1.0e+00 |
| DUALC2 | 7 | 236 | 3551.31 | SearchDirectionBecomesTooSmall | 0 | 1.0e+00 |
| DUALC5 | 8 | 286 | 427.232 | SearchDirectionBecomesTooSmall | 0 | 1.0e+00 |
| DUALC8 | 8 | 511 | 18309.4 | SearchDirectionBecomesTooSmall | 0 | 1.0e+00 |
| EXDATA | 3000 | 6001 | -141.843 | TimeOut | — | — |
| GOULDQP2 | 699 | 1048 | 0.000184275 | SearchDirectionBecomesTooSmall | 0.000163507 | 1.1e-01 |
| GOULDQP3 | 699 | 1048 | 2.06278 | SearchDirectionBecomesTooSmall | 3.38844 | 3.9e-01 |
| HUES-MOD | 10000 | 10002 | 3.48247e+07 | SearchDirectionBecomesTooSmall | 0 | 1.0e+00 |
| HUESTIS | 10000 | 10002 | 3.48247e+11 | SearchDirectionBecomesTooSmall | 0 | 1.0e+00 |
| KSIP | 20 | 1021 | 0.575798 | SearchDirectionBecomesTooSmall | 0 | 1.0e+00 |
| LASER | 1002 | 2002 | 2.4096e+06 | SearchDirectionBecomesTooSmall | 0 | 1.0e+00 |
| LISWET10 | 10002 | 20002 | 49.4858 | TimeOut | — | — |
| LISWET11 | 10002 | 20002 | 49.524 | SearchDirectionBecomesTooSmall | 2525.75 | 9.8e-01 |
| LISWET3 | 10002 | 20002 | 25.0012 | TimeOut | — | — |
| LISWET4 | 10002 | 20002 | 25.0001 | TimeOut | — | — |
| LISWET5 | 10002 | 20002 | 25.0343 | TimeOut | — | — |
| LISWET6 | 10002 | 20002 | 24.9957 | TimeOut | — | — |
| LISWET8 | 10002 | 20002 | 714.47 | TimeOut | — | — |
| POWELL20 | 10000 | 20000 | 5.20896e+10 | SearchDirectionBecomesTooSmall | 0 | 1.0e+00 |
| Q25FV47 | 1571 | 2391 | 1.37444e+07 | SearchDirectionBecomesTooSmall | 0 | 1.0e+00 |
| QADLITTL | 97 | 153 | 480319 | SearchDirectionBecomesTooSmall | 0 | 1.0e+00 |
| QBANDM | 472 | 777 | 16352.3 | SearchDirectionBecomesTooSmall | 0 | 1.0e+00 |
| QBEACONF | 262 | 435 | 164712 | SearchDirectionBecomesTooSmall | 0 | 1.0e+00 |
| QBORE3D | 315 | 548 | 3100.2 | SearchDirectionBecomesTooSmall | 2250.95 | 2.7e-01 |
| QBRANDY | 249 | 469 | 28375.1 | SearchDirectionBecomesTooSmall | 0 | 1.0e+00 |
| QCAPRI | 353 | 624 | 6.67933e+07 | SearchDirectionBecomesTooSmall | 0 | 1.0e+00 |
| QE226 | 282 | 505 | 212.653 | SearchDirectionBecomesTooSmall | 7.113 | 9.7e-01 |
| QETAMACR | 688 | 1088 | 86760.4 | SearchDirectionBecomesTooSmall | -133.586 | 1.0e+00 |
| QFFFFF80 | 854 | 1378 | 873147 | InternalError | — | — |
| QFORPLAN | 421 | 582 | 7.45663e+09 | SearchDirectionBecomesTooSmall | 2.43936e+08 | 9.7e-01 |
| QGFRDXPN | 1092 | 1708 | 1.00791e+11 | SearchDirectionBecomesTooSmall | 4.9e+10 | 5.1e-01 |
| QGROW15 | 645 | 945 | -1.01694e+08 | TimeOut | — | — |
| QGROW22 | 946 | 1386 | -1.49629e+08 | TimeOut | — | — |
| QGROW7 | 301 | 441 | -4.27987e+07 | InternalError | — | — |
| QISRAEL | 142 | 316 | 2.53478e+07 | InternalError | — | — |
| QPCBLEND | 83 | 157 | -0.00784254 | SearchDirectionBecomesTooSmall | 0 | 1.0e+00 |
| QPCBOEI1 | 384 | 735 | 1.15039e+07 | SearchDirectionBecomesTooSmall | 0 | 1.0e+00 |
| QPCBOEI2 | 143 | 309 | 8.17196e+06 | SearchDirectionBecomesTooSmall | 0 | 1.0e+00 |
| QPCSTAIR | 467 | 823 | 6.20439e+06 | SearchDirectionBecomesTooSmall | 204468 | 9.7e-01 |
| QPILOTNO | 2172 | 3147 | 4.72859e+06 | InternalError | — | — |
| QRECIPE | 180 | 271 | -266.616 | SearchDirectionBecomesTooSmall | -0.324 | 1.0e+00 |
| QSC205 | 203 | 408 | -0.00581395 | MaximumIterationsExceeded | 0 | 1.0e+00 |
| QSCAGR25 | 500 | 971 | 2.01738e+08 | SearchDirectionBecomesTooSmall | 0 | 1.0e+00 |
| QSCAGR7 | 140 | 269 | 2.68659e+07 | SearchDirectionBecomesTooSmall | 0 | 1.0e+00 |
| QSCFXM1 | 457 | 787 | 1.68827e+07 | SearchDirectionBecomesTooSmall | 0 | 1.0e+00 |
| QSCFXM2 | 914 | 1574 | 2.77762e+07 | SearchDirectionBecomesTooSmall | 0 | 1.0e+00 |
| QSCFXM3 | 1371 | 2361 | 3.08164e+07 | SearchDirectionBecomesTooSmall | 0 | 1.0e+00 |
| QSCORPIO | 358 | 746 | 1880.51 | SearchDirectionBecomesTooSmall | 0 | 1.0e+00 |
| QSCRS8 | 1169 | 1659 | 904.56 | SearchDirectionBecomesTooSmall | 0 | 1.0e+00 |
| QSCSD1 | 760 | 837 | 8.66667 | SearchDirectionBecomesTooSmall | 0 | 1.0e+00 |
| QSCSD6 | 1350 | 1497 | 50.8082 | SearchDirectionBecomesTooSmall | 0 | 1.0e+00 |
| QSCSD8 | 2750 | 3147 | 940.764 | SearchDirectionBecomesTooSmall | 0 | 1.0e+00 |
| QSCTAP1 | 480 | 780 | 1415.86 | SearchDirectionBecomesTooSmall | 0 | 1.0e+00 |
| QSCTAP2 | 1880 | 2970 | 1735.03 | SearchDirectionBecomesTooSmall | 0 | 1.0e+00 |
| QSCTAP3 | 2480 | 3960 | 1438.75 | SearchDirectionBecomesTooSmall | 0 | 1.0e+00 |
| QSEBA | 1028 | 1543 | 8.14818e+07 | SearchDirectionBecomesTooSmall | 85.35 | 1.0e+00 |
| QSHARE1B | 225 | 342 | 720078 | SearchDirectionBecomesTooSmall | 0 | 1.0e+00 |
| QSHARE2B | 79 | 175 | 11703.7 | SearchDirectionBecomesTooSmall | 0 | 1.0e+00 |
| QSHELL | 1775 | 2311 | 1.57264e+12 | SearchDirectionBecomesTooSmall | 9.88354e+10 | 9.4e-01 |
| QSHIP04L | 2118 | 2520 | 2.42002e+06 | SearchDirectionBecomesTooSmall | 0 | 1.0e+00 |
| QSHIP04S | 1458 | 1860 | 2.42499e+06 | SearchDirectionBecomesTooSmall | 0 | 1.0e+00 |
| QSHIP08L | 4283 | 5061 | 2.37604e+06 | SearchDirectionBecomesTooSmall | 0 | 1.0e+00 |
| QSHIP08S | 2387 | 3165 | 2.38573e+06 | SearchDirectionBecomesTooSmall | 0 | 1.0e+00 |
| QSHIP12L | 5427 | 6578 | 3.01888e+06 | SearchDirectionBecomesTooSmall | 0 | 1.0e+00 |
| QSHIP12S | 2763 | 3914 | 3.05696e+06 | SearchDirectionBecomesTooSmall | 0 | 1.0e+00 |
| QSIERRA | 2036 | 3263 | 2.37505e+07 | SearchDirectionBecomesTooSmall | 0 | 1.0e+00 |
| QSTAIR | 467 | 823 | 7.98545e+06 | SearchDirectionBecomesTooSmall | 234559 | 9.7e-01 |
| QSTANDAT | 1075 | 1434 | 6411.84 | SearchDirectionBecomesTooSmall | 0 | 1.0e+00 |
| STCQP1 | 4097 | 6149 | 155144 | SearchDirectionBecomesTooSmall | 0 | 1.0e+00 |
| STCQP2 | 4097 | 6149 | 22327.3 | SearchDirectionBecomesTooSmall | 0 | 1.0e+00 |
| VALUES | 202 | 203 | -1.39662 | ParseError:JSONDecodeError | — | — |
| YAO | 2002 | 4002 | 197.704 | SearchDirectionBecomesTooSmall | 273.128 | 2.8e-01 |

