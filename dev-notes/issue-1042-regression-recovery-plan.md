# gh#1042 — recovering the 0.13.0 accepted trajectory costs

Plan for the agent loop on branch `claude/pounce-regressions-recovery-tz8uh2`.
One work item per iteration. This file is also the status board: update the
**Status** column of the inventory and the per-item log at the end of every
iteration, in the same commit as the code.

Written against `0a299151` (main after 0.13.0, 2026-10-22), on **x86_64**
(4 CPUs, AVX2+FMA, FERAL 0.21.0). Every number marked *measured* below was
taken on that machine with a release `pounce` built from `0a299151`
(`p-head`) and from tag `v0.12.0` = `a31dab9c` (`p-base`), built in a
separate worktree with its own `CARGO_TARGET_DIR`. Numbers marked *issue*
are copied from gh#1042 / PR #1038 and were **not** reproduced here (arm64
numbers cannot be).

## 1. Summary and inventory

gh#1042 is the tracking record CLAUDE.md requires for the trajectory costs
0.13.0 shipped: the FERAL 0.20 → 0.21.0 bump (#1038, commits `97bc0f15`,
`5fc8c797`, `2599c2be`), the gh#1000 convex-routing change (#1002,
`f6a4e4c2`) and the MLSL `gamma` default change (#993, `91b6dcda`). An issue
comment (id 6344966966, moved from jkitchin/feral#224) adds two benchmark
diagnoses: the qcqp1000-2c slowdown is pounce reusing the multiplier-init
matrix's symbolic ordering, and brainpc0/1's slowdown is 811 restoration
iterations after a now-*correct* inertia report.

### What the measurements here add or correct

| claim in the issue | measured here (x86_64) | verdict |
|---|---|---|
| exact `pooling_rt2stp` 162 → 492 (x86) | `p-base` 162 it / q=4 / 3 resto calls; `p-head` 492 it / q=3 / **24 resto calls, 106 resto it**, **40 iterations with lg(rg) ≥ 19 and 171 with lg(rg) ≥ 10** | confirmed; the ladder-at-cap diagnosis is consistent with the log but still unmeasured at the handler level |
| exact `dallass` 41 → 123 (x86) | `p-base` 41 it / q=6 / **2 resto calls**; `p-head` 123 it / **q=19 / 12 resto calls (53 resto it)** | confirmed; the cost is 10 extra non-finite-Hessian restoration entries, not the strike exemption |
| exact `deb7` Acceptable/180 **arm64 only**, x86 improves 143 → 114 | `p-base` 143, `p-head` 114, both `SolveSucceeded` | confirmed as an x86 improvement; the arm64 regression is unverifiable here |
| lbfgs `dallass` Optimal/224 → ErrorInStep/308 **arm64 only** | `p-base` **ErrorInStep/213**, `p-head` ErrorInStep/146 (ladder tot 292) | **discrepancy**: on x86 the lbfgs leg fails on 0.12.0 too, so there is no x86 regression to recover and the "arm64 only" is about the *verdict* on arm64, not the leg being healthy elsewhere |
| lbfgs `pooling_rt2stp` worse optimum **arm64 only**; x86 failing → solved | `p-base` ErrorInStep/716, `p-head` SolveSucceeded/82 at −3273.95 | confirmed x86 gain |
| `scaled_feasible_a` nlp 22/19 → cvx-qp 200/200 | `p-base` nlp 22 / 19; `p-head` cvx-qp 200 / 200, obj 0, final kkt 5e-19 | confirmed, **mechanism found** (item R1) |
| minima `circle_parabola_t0p5` mlsl 2 → 1 | `p-base` (gamma 2) found 2 in 11 solves; `p-head` (gamma 0.5) found 1 in 9 solves, `converged` = stagnation at `patience 8` | confirmed; gamma grid measured (item R4) |
| not in the issue | lbfgs `deb7`: `p-base` ErrorInStep/1010 → `p-head` RestorationFailed/1206 (ladder tot 1738). Both fail; the sweep diff line exists but is not a lost solve | recorded, out of scope |

### Inventory

| id | fixture / benchmark (leg) | v0.12.0 → 0.13.0 | suspected cause | recoverable? | Status |
|---|---|---|---|---|---|
| **R1** | `scaled_feasible_a` (exact and lbfgs) | nlp 22 / 19 it → cvx-qp **200 / 200** it, right answer | #1002 `f6a4e4c2`: HSDE endgame certifies the gh#293 equilibrated retry, whose loop never meets `pres < tol` (plateau 5.0e-6 → 5.7e-7) while `mu` runs to 1e-121; `Optimal` is stamped only by the post-loop `true_kkt_error` upgrade | **yes** — a stall-gated in-loop certificate, or a primal-residual floor that reads the stalled `pres` as the regularization floor it is | **not recoverable this way** (2026-10-22: hypothesis refuted — `m_eq = 0`, the plateau is a rounding floor, and the iterate is not optimal until it 165; see log + §5) |
| **R2** | `pooling_rt2stp` (exact, x86) | 162 → **492** it (arm64: 109 → 242 total, *issue*) | FERAL 0.21 trajectory; `2599c2be` removed the latch (904 → 492); the remaining cost is the `δ_x` ladder climbing to `max_hessian_perturbation` with `δ_c` on (*issue*, unmeasured) | **partial** — the "stop the ladder once inertia is guaranteed" follow-up the issue names; must be measured first | **not recoverable this way** (2026-10-22: hypothesis refuted — 0 of 146 `WrongInertia` factors with `δ_c > 0` had a PD (1,1) block; the climb answers a genuinely indefinite `W`, λ_min down to −1.2e19 vs v0.12.0's −2.2e9; trigger found at iteration 4, see log + §5) |
| **R3** | `dallass` (exact, x86) | 41 → **123** it (arm64 41 → 65, *issue*) | FERAL 0.21 path lands on `x = 0` of `|x|²·|x|^0.85` terms 12 times instead of 2; each is a non-finite Hessian → restoration entry; `5fc8c797` only stops the third from ending the solve | **partial** — (A) the Hessian is an evaluator artefact (`0·∞` from the product rule; the true second derivative of `|x|^2.85` at 0 is 0), fixable in `pounce-nl`; (B) R2 may move it; re-baseline after R2 | todo |
| **R4** | minima `circle_parabola_t0p5` mlsl | found 2 of 2 → **1** | #993 `91b6dcda`: `MLSL_GAMMA` 2 → 0.5; at 0.5 the extra launches re-find the first minimum and the stagnation counter (`patience 8`) stops the search | **partial** — no single gamma wins at seed 0 (table in R4); needs a 10-seed measurement or a structural stagnation rule; non-blocking | todo |
| **B1** | benchmark `qcqp1000-2c` (mittelmann) | 99 it / 28 s → 163 it / 112 s (*comment*) | pounce reuses the symbolic ordering FERAL chose for the least-squares multiplier-init matrix (W = 0, D = I) for every later KKT; `invalidate_symbolic_cache()` after the init factor halves factor time (*comment*) | **yes** for the code change; **measurement blocked** here (no benchmark corpus on this machine); a branch `perf/feral-symbolic-after-init` is said to be in progress — check `origin` first | todo |
| B2 | benchmark `brainpc0/1` (vanderbei) | 58 it / 1.1 s → 1073 + 52 it / 75 s; 185 + 24 → 2117 + 24 (*comment*) | after a now-correct inertia at one iterate, pounce's restoration takes 811 of 1073 iterations (*comment*) | **unknown** — investigate only after B1 lands (B1 changes the trajectory) and only with the corpus | blocked |
| B3 | benchmark `qcqp1500-1c` | Optimal → `Maximum_CpuTime_Exceeded` at 300 s (*issue*) | presumably B1 (same family); **unverified** | follows B1 | blocked |
| B4 | benchmark `cresc132` Optimal → Infeasible (Ipopt agrees); `ssebnln` Optimal → Acceptable | (*issue*) | FERAL 0.21 trajectory | **no / out of scope** (§5) | n/a |
| X1 | exact `deb7` arm64 Optimal/147 → Acceptable/180 | (*issue*) | FERAL 0.21 NEON trajectory | **not recoverable here** (x86 improves 143 → 114) (§5) | n/a |
| X2 | lbfgs `dallass` arm64 Optimal/224 → ErrorInStep/308 | (*issue*); x86 fails on both versions | FERAL 0.21 trajectory | **not recoverable here**; the x86 lbfgs failure predates 0.13.0 (§5) | n/a |
| X3 | lbfgs `pooling_rt2stp` arm64 −4391.83 → −3273.95 | (*issue*); x86 failing → solved | bistable nonconvex model | **no** — both are certified optima (§5) | n/a |
| F1 | `issue_592` pin `< 500` sits 8 above x86's 492 | — | R2 | resolved by R2; **do not raise the pin** | open — R2 did not land; pin untouched, margin still 8 (follow-up in §5) |

## 2. Ordering

1. **R1** `scaled_feasible_a` — self-contained in `pounce-convex`, mechanism
   established, cheapest to validate (convex-arm sweep lines + `benchmarks/qp`
   if the corpus exists), interacts with nothing else.
2. **R2** `pooling_rt2stp` ladder — the central NLP-arm item; it moves the
   `δ_x` ladder for every exact-Hessian solve, so **R3 is re-baselined after
   it** (dallass's 12 restoration entries may or may not survive).
3. **R3** `dallass` — only after R2's numbers are in; option (A) is a
   `pounce-nl` change independent of R2, option (B) is "R2 already fixed it".
4. **B1** symbolic-ordering reuse — independent of R1–R3 (it changes *which
   factorization* FERAL performs, not the IPM's decisions, but through
   pivoting it can move trajectories, so it still needs the full sweep). Code
   change is implementable and unit-testable without the corpus; the
   benchmark claim can only be *verified* with it.
5. **R4** MLSL gamma — independent, has its own sweep (`sweep-minima.sh`),
   lowest certainty; do it last among the recoverable items.
6. **B2 / B3** — only if `POUNCE_BENCH_DATA` resolves to a corpus
   (`. scripts/bench_data_root.sh && bench_data_root`); otherwise record as
   blocked and stop.

Items that interact: R2 → R3 (re-baseline), B1 → B2/B3 (re-baseline). R1 and
R4 touch neither arm the others touch, but every item still diffs the full
fixture sweep against the **branch tip before the item** (`p-prev`), so that
each commit's own blast radius is accounted for separately.

## 3. Work items

Common to every item: build `p-prev` from the branch tip before you start
(`cargo build --release -p pounce-cli && cp target/release/pounce
/tmp/p-prev`), keep `p-base` (v0.12.0) around for the regressed fixtures'
"before" numbers, and sweep **both legs** with `scripts/sweep-fixtures.sh`.
Fixture paths are `crates/pounce-cli/tests/fixtures/<name>.nl`.

### R1 — `scaled_feasible_a` runs the convex IPM to `max_iter` before reporting Optimal

**Goal.** The convex arm certifies `scaled_feasible_a` `Optimal` in a number
of iterations comparable to a healthy HSDE solve (≤ 60 on both legs; the
pre-#1002 NLP route took 22 / 19), without changing any verdict.

**Files / symbols.**
- `crates/pounce-convex/src/hsde.rs`: the main loop (`for it in 0..opts.max_iter`,
  ~line 631), the stopping test (`abs_ok`, `rel_in_floor`, `comp_ok`,
  `converged`, ~lines 898–1000), the relative-candidate stash
  (`rel_candidate`, `restore_candidate`, ~1010–1050), the adaptive static
  regularization `base_reg` (~1094–1110), the post-loop `true_kkt_error`
  upgrade (~1620–1640).
- `crates/pounce-convex/src/ipm.rs`: `solve_qp_ipm_core` (the gh#293
  equilibrated retry, ~527–700), `equilibrated_retry_budget` (~807),
  `optimum_is_genuine` (~1104), `verify_or_repair_optimum`.
- `crates/pounce-cli/src/main.rs` ~2941 and ~4902: the gh#535 convex → NLP
  decline rule, whose comment still says this model "needs 3596 iterations".
- Tests to read first: `crates/pounce-cli/tests/issue_689_direct_driver_scaled_feasible.rs`
  (`ITER_BUDGET = 60` on the `qp_hsde=no` route; its header explains the
  `obj_constant` normalizer), `declared_optimum_sentinel.rs` (~163–220,
  routing at default and at `bound_relax_factor=1e-8`),
  `issue_1000_*`, `crates/pounce-convex/tests/issue984_stopping_units.rs`,
  `issue_689_direct_cold_start_scale.rs`, `issue414_cost_normalized_false_optimal.rs`,
  `illconditioned_huge_scale.rs`; `dev-notes/issue-689-direct-driver-cold-start.md`.

**Measured mechanism (x86, `p-head`, `--json-detail full`).** With
`qp_equilibrate=no` the convex arm does *not* certify and the model falls to
the NLP arm (22 it), so the returned 200-iteration run is the **gh#293
equilibrated retry** (full `max_iter` budget because `P ≠ 0`). Its trace:
`inf_pr` falls to 3.9e-5 at iteration 13, then sits at **5.000e-6** from
iteration 20 to ~125 while `inf_du` drifts 5e-5 → 5e-9 and `mu` collapses
from 1e-19 to 1e-121; around 125–150 the iterate is thrown away (`inf_pr`
back to 4.8e9, `alpha 0`), recovers by 155, and then `inf_pr` oscillates
5.7e-7 ↔ 2.8e-6 with `inf_du` ~1e-14 until the cap. `pres < opts.tol` (1e-8)
is never met, so neither `abs_ok` nor `rel_in_floor` (which also requires
`pres <= cap_p`) is, and `converged` never fires. After the loop the
un-homogenized point's `true_kkt_error` is 5e-19, so the verdict is upgraded
to `Optimal`. The 5e-6 plateau is the static regularization floor
`δ·‖ŷ‖` on the equality block in the equilibrated metric (the commit message
of `f6a4e4c2` records that lowering the equality-block floor `delta_c` was
tried and dropped because it cost QFORPLAN 33 → 139).

**Correction (R1 iteration, 2026-10-22).** The last sentence above is wrong:
`scaled_feasible_a` has no equality rows (`m_eq = 0`), every bit of `pres` is
the inequality block, and `5.0e-6` is the rounding floor of `Gx + s − hτ` at
`‖x̂‖ = 5e9` (≈ 4 ulp, as `ipm.rs`'s own gh#689 comment says). Through the
plateau the iterate is not optimal either (`true_kkt_error` ≥ 23 until it
150). The hypothesis below is refuted; see the R1 item log and §5.

**Hypothesis.** The loop lacks a stop for an iterate that is optimal in the
caller's coordinates but whose *homogeneous, equilibrated* primal residual is
floored by regularization. The post-loop certificate already encodes the
right test; it is just never consulted while iterating.

**Steps.**
1. Reproduce: `pounce scaled_feasible_a.nl out.sol --json-output a.json
   --json-detail full` → `statistics.iteration_count == 200`,
   `solution.engine == "cvx-qp"`. Confirm the plateau in `iterations[]`.
   Also run `hessian_approximation=limited-memory` (same numbers expected —
   the convex arm ignores it).
2. Instrument (temporarily) which of `pres`, `dres`, `gap`, `comp_max`,
   `true_comp` fails at iterations 20–120 and what `base_reg`/`delta_c`
   floor is in force. Record the numbers in the item log.
3. Implement the narrowest fix that the measurement supports. Preferred
   shape, in the loop after `converged` is computed: when the iterate has
   **stalled** — `pres` within a factor of 2 of its value `k` iterations ago
   (k = 5 is a starting point), `mu` below `REL_FLOOR_KAPPA·ε·tol` or similar
   so no polishing is happening, and `dres`/`gap`/`comp_ok` already satisfied
   — compute `true_kkt_error(prob, cone, x/τ, y/τ, z/τ, 1.0)` and stop
   `Optimal` if it is `< opts.tol`. This honours the existing comment in
   `ipm.rs` ("deliberately runs *after* the loop … breaking early on the
   certificate would hand back a less-polished iterate"): a stalled loop is
   not polishing anything. Alternative shape, if the measurement says the
   plateau is exactly the regularization floor: let `cap_p` include the
   regularization floor `δ·‖ŷ‖`, as the comment on `QpOptions::reg` already
   describes it. Do **not** change `equilibrated_retry_budget` to hide the
   cost, and do not change the gh#535 decline rule.
4. Update the stale "3596 iterations" comments in `main.rs` only if the
   number you measure is different; cite the measurement.
5. Add a regression test in `crates/pounce-convex/tests/` (or extend
   `issue_689_direct_driver_scaled_feasible.rs`) that pins the default
   route: engine `cvx-qp`, `Optimal`, `iteration_count <= 60`, objective
   `|f| <= 1e-3`. Mutation-check it: with the fix reverted it must fail on
   the iteration count.

**Acceptance.** `scaled_feasible_a`: `cvx-qp`, `SolveSucceeded`, obj
`|f| ≤ 1e-3`, `iteration_count ≤ 60` on **both** legs (measured on this
machine, 200 → target). `scaled_feasible_b` unchanged or better.
`pounce verify` on the `.sol` still reports 0 violation.

**Validation.**
- `scripts/sweep-fixtures.sh` both legs, `p-prev` vs new. Only
  `scaled_feasible_a` lines (and possibly other convex lines that *improve*)
  may move; any convex line whose iteration count **rises** or whose
  engine/status changes is a stop-and-explain. Recount the engine split in
  the item log.
- `cargo test -p pounce-convex -p pounce-cli` (the `issue_689_*`,
  `declared_optimum_sentinel`, `issue_760_*`, `issue_1000_*`,
  `issue984_stopping_units`, `issue414_*`, `illconditioned_huge_scale`
  files are the guards that must not regress).
- If `bench_data_root` resolves: `benchmarks/qp` A/B total iterations
  (0.13.0 release: 137 problems optimal, total 2998 at #1002); a rise of more
  than a few percent, or any status flip, is a stop.
- `python/tests/test_qp*.py` if the Python extension builds here
  (`maturin develop` in `python/`); otherwise record "not run".

**Must not regress.** The gh#414 false-optimum guards (`optimum_is_genuine`,
`verify_or_repair_optimum`), the gh#984 stash rule ("every stash satisfies
`comp_ok`"), gh#1000 I.D/I.8 numbers (battery QP optimal at tol 1e-6/1e-8;
blend LP 14–23 iterations across scales — both pinned in `issue984_*` /
`issue_1000_*` tests), the gh#689 `ITER_BUDGET` on the direct driver.

### R2 — `pooling_rt2stp` exact: stop the `δ_x` ladder once the inertia is guaranteed

**Goal.** Bring x86 `pooling_rt2stp` (exact) from 492 iterations back toward
0.12.0's 162 without reintroducing the latch `2599c2be` removed, and restore
the margin under the `issue_592` `< 500` pin by lowering the count, not the
pin.

**Files / symbols.**
- `crates/pounce-common/src/pd_perturbation.rs`: `perturb_for_wrong_inertia`,
  `get_deltas_for_wrong_inertia` (the ladder: `delta_xs_init`,
  `delta_xs_first_inc_fact`, `delta_xs_inc_fact`, `delta_xs_max`, the
  overflow branch at ~687 with the `2599c2be` reset), `maybe_withdraw_delta_c`
  (gh#592 walk-back), `perturb_for_singular_with` (gh#981 reinstatement),
  the unit tests at the bottom (`an_overflowed_ladder_does_not_poison_the_next_system`,
  the gh#592 walk-back tests).
- `crates/pounce-algorithm/src/kkt/pd_full_space_solver.rs` ~1640–1760: the
  retry loop that reads `ESymSolverStatus` and calls the handler
  (`Singular` → `perturb_for_singular_with`; `WrongInertia` with too few
  negatives → `escalate_aug_quality` then `perturb_for_singular`; other
  `WrongInertia`/`Singular` → `perturb_for_wrong_inertia`).
- `crates/pounce-algorithm/src/kkt/aug_system_solver.rs`: `AugSysCoeffs`
  (`w: Option<&dyn SymMatrix>`, `w_factor`, `d_x`, `delta_x`, `d_s`,
  `delta_s`, `delta_c`, `delta_d`) — everything a Gershgorin bound on the
  (1,1) block needs is in this struct.
- `crates/pounce-algorithm/src/kkt/std_aug_system_solver.rs` ~200–320:
  downcasts `W` to `SymTMatrix` (exact Hessian) or `DiagMatrix`; the L-BFGS
  arm uses `LowRankUpdateSymMatrix` through
  `low_rank_aug_system_solver.rs`, for which a row-max is not implemented
  (the issue says so) — the lbfgs leg keeps the current behaviour.
- `crates/pounce-feral/src/lib.rs` ~1350–1440: where `WrongInertia` /
  `Singular` are decided (`inertia_trust_floor`, `sub_floor_count_is_noise`,
  gh#1003).
- Guards: `crates/pounce-cli/tests/issue_592_delta_c_walkback.rs`
  (`< 500` pin, both optima accepted, walk-back inert), `issue_250_dual_guard_never_worse.rs`
  (`pooling_rt2stp_solves_at_default_settings`), `crates/pounce-wasm/src/lib.rs`
  ~942–965 (pooling shim), `issue857_*` (quality-escalation counts on
  `deb7`/`circle_parabola_t0p5`), `issue_884_biactive_dual_divergence.rs`,
  `pd_perturbation` unit tests; `dev-notes/issue-592-restart-non-idempotence.md`,
  `dev-notes/issue-981-delta-c-walkback.md`, `dev-notes/issue-540-eigena2-inertia-noise.md`.
- Debug aid: `POUNCE_DBG_PERT=1` makes the handler emit `[PERT]` lines
  through the sink's `debug` (tracing at debug level; check how the CLI
  enables it, e.g. `RUST_LOG`/`--log-level`).

**Hypothesis (the issue's, unmeasured).** With `δ_c > 0` the KKT matrix is
quasi-definite as soon as the (1,1) block `W + Σ_x + δ_x I` (and the slack
block `Σ_s + δ_s I`) is positive definite, and then its inertia is exactly
`(n + m_s, m_c + m_d, 0)` by Sylvester. When FERAL's count disagrees on such
a matrix the count is rounding, and the ladder should accept the factor
instead of climbing (×8 per rung) to `max_hessian_perturbation`. The log is
consistent: 40 of 492 iterations carry `lg(rg) ≥ 19`, 171 carry
`lg(rg) ≥ 10`, and the over-damped steps feed 24 restoration entries
(0.12.0: 3).

**Steps.**
1. Reproduce: `pounce pooling_rt2stp.nl o.sol --json-output p.json` →
   492 it, `-3273.955`, no second opinion; count `lg(rg) ≥ 19` lines.
2. **Measure before coding.** With `POUNCE_DBG_PERT=1` (and the tracing
   level that shows it) record per ladder: `δ_c` on/off, rungs climbed,
   whether it overflowed, and — with a temporary Gershgorin probe on
   `AugSysCoeffs` — at which rung the (1,1) block became diagonally dominant.
   The item is only worth implementing if a material share of the 171
   high-`δ_x` iterations had `δ_c > 0` and a provably PD block at a rung well
   below where the ladder stopped. Write the histogram into the item log.
   If the hypothesis is false, record that, mark R2 "not recoverable this
   way", and propose the alternative the measurement suggests (e.g. the
   `delta_c` reinstatement/withdrawal interplay of gh#592/gh#981) as a
   follow-up issue instead of forcing a change.
3. Implement, if supported: in `pd_full_space_solver.rs`, when the status is
   `WrongInertia` (not `Singular`), `d.delta_c > 0` **or** `m_c == 0`, and a
   helper (new `AugSystemSolver` method, default `false`; implemented in
   `StdAugSystemSolver` for `SymTMatrix`/`DiagMatrix` W only) proves
   `W·w_factor + D_x + δ_x I` strictly diagonally dominant with positive
   diagonal and `D_s + δ_s > 0`, treat the factor as `Success` and record an
   info-string letter (e.g. `g`) so the iteration table shows it. Note the
   gh#1003 reasoning in `pounce-feral`: a *surplus* negative on a clean factor
   stays `WrongInertia` — this item accepts exactly that case when the matrix
   proves the count wrong. Do not touch the `Singular` arm (that is the
   gh#981 reinstatement's territory) and do not alter the overflow reset.
4. Unit test in `pounce-algorithm` (a 2×2 or 3×3 KKT whose (1,1) block is
   diagonally dominant with `δ_c > 0` and a fake linear solver reporting a
   wrong count) and a CLI pin: `pooling_rt2stp` exact `SolveSucceeded`, a
   known optimum, `iteration_count` below a bound set from the measurement
   (leave the `< 500` pin where it is).

**Acceptance.** `pooling_rt2stp` exact (x86): `SolveSucceeded`, objective one
of −3273.955 / −4391.826, `iteration_count ≤ 300` (0.12.0: 162; anything
≤ 300 restores the `issue_592` margin; report the actual number). Fewer
restoration calls than 24. No new `max_hessian_perturbation` overflows on
systems where the block was provably PD.

**Validation.**
- `scripts/sweep-fixtures.sh` both legs, `p-prev` vs new, **and** the same
  lines against `p-base` for the regressed fixtures. Every moved line is
  explained: an exact-leg line whose `q=` or `it=` moves is expected on
  nonconvex models; any status loss, any objective change outside a known
  bistable pair, any `nlp → square-lm` move, and any lbfgs line moving at
  all (the helper must be a no-op there) are stops.
- `cargo test -p pounce-common -p pounce-algorithm -p pounce-cli -p pounce-wasm`.
  Guards named above; in particular `the_walkback_is_inert_on_this_fixture_since_gh693`
  (`< 300` with the walk-back off) and `issue857_*`.
- Python `python/tests/test_mehrotra_feasible_gh986.py` if the extension
  builds; otherwise record.
- If the corpus is available: `benchmarks/vanderbei` status A/B (0.13.0
  release: 707 of 733) — `steenbrd/e/f/g`, `eigena2/b2` and `dallas*` are
  the models the inertia path has historically moved.

**Must not regress.** `an_overflowed_ladder_does_not_poison_the_next_system`
(the `2599c2be` fix); gh#592's walk-back tests; gh#981's reinstatement test
(`issue981_cstr_dup_row`); gh#540/gh#1003 `pounce-feral` unit tests; the
`issue_250` hang guard; `deb7` must stay `SolveSucceeded` on the exact leg.

**Refuted (R2 iteration, 2026-10-22).** Measured with a temporary dense
probe on every ladder factorization (log below): no `WrongInertia` factor
with `δ_c > 0` has a positive-definite (1,1) block, so the planned
"accept when Sylvester guarantees the inertia" arm would never fire on this
fixture. The ladder stops within ~1 rung of the (1,1) PD threshold
(median `δ_x / −λ_min(W+Σ_x)` = 5.7); it climbs high because `W` itself is
indefinite by 1e13–1e19 after iteration 5, which v0.12.0 never is. The
trigger is an iteration-4 step from a `δ_c`-free, numerically singular
factor that iterative refinement accepts because the matrix's single
quality escalation was already spent by the ladder's too-few-negatives
branch. Follow-up in §5.

### R3 — `dallass` exact: the `|x|²·|x|^0.85` kink

**Goal.** x86 `dallass` exact back toward 41 iterations (0.12.0) with the
strike exemption kept.

**Files / symbols.**
- `crates/pounce-cli/tests/fixtures/dallass.nl` lines ~300–340: the objective
  terms are `o2 (o5 (o15 vK) n2) (o5 (o15 vK) n0.85)`, i.e. `|x|² · |x|^0.85`.
  The product rule gives `u''v + 2u'v' + uv''` with `v' = 0.85|x|^-0.15` and
  `v'' = -0.1275|x|^-1.15`, so at `x = 0` two terms are `0·∞ = NaN` while the
  true second derivative of `|x|^2.85`, `2.85·1.85·|x|^0.85`, is **0**.
- `crates/pounce-nl/src/nl_tape.rs`: `TapeOp::Pow` / `TapeOp::Abs` forward,
  gradient and Hessian arms (~373, 511, 694, 951, 1130, 1470, 1838); the
  `hessian_pow_*_matches_fd` tests (~5131–5150).
- `crates/pounce-algorithm/src/ipopt_alg.rs` ~3175–3195 (`resto_entry_nonfinite_w`)
  and ~4120 (the strike exemption); `crates/pounce-cli/tests/nonfinite_hessian_restoration_strike.rs`.

**Measured.** `p-base`: 41 it, 2 restoration calls (14 resto it), q=6.
`p-head`: 123 it, **12 restoration calls (53 resto it)**, q=19; the `R`
entries sit at iterations 31, 39, 45, 53, … — the trajectory keeps returning
a flow to exactly 0. Hessian evaluations = iterations + 1 on both, so the
cost is the restoration entries and the post-restoration re-convergence, not
extra evaluations.

**Hypotheses.** (A) The NaN is an evaluator artefact: merging the two powers
of the same `Abs` operand (`|x|^a · |x|^b → |x|^(a+b)`, valid because the
base is non-negative), or evaluating the product-rule Hessian of
`pow(abs(x), p)` nodes so that an exact-zero factor annihilates an infinite
co-factor **only when the merged exponent proves the limit is finite**, gives
a finite Hessian at the kink and removes the restoration entries entirely.
(B) R2 changes the path and the kink is met less often — re-measure first.

**Steps.**
1. After R2 lands, re-run `dallass` both legs; record. If exact ≤ ~60 with
   ≤ 3 restoration calls, mark R3 "recovered by R2" and stop.
2. Otherwise prototype (A) as a tape-level peephole at `.nl` load time
   (`nl_reader.rs` / `nl_tape.rs`): a product whose two factors are
   `Pow(Abs(v), c1)` and `Pow(Abs(v), c2)` with constant exponents becomes
   `Pow(Abs(v), c1 + c2)`. Values are unchanged up to rounding; first
   derivatives at `x ≠ 0` unchanged; at `x = 0` the Hessian becomes the
   mathematically correct finite value. Add a unit test that the Hessian of
   `|x|²·|x|^0.85` at 0 is finite and matches a finite difference at
   `x = ±1e-3`, and that a model without the pattern builds an identical
   tape.
3. Measure `dallass` again: expect **no** non-finite-Hessian restoration
   entries (`resto_entry_nonfinite_w` never set; add a counter to the
   statistics or read the `R` lines).

**Acceptance.** `dallass` exact (x86): `SolveSucceeded` at −32393.2257,
`iteration_count ≤ 60`, restoration calls ≤ 3. The lbfgs leg is not an
acceptance criterion (it fails on 0.12.0 too on x86) but must not get
worse (`ErrorInStepComputation`/146 today).

**Validation.** Full fixture sweep both legs (the peephole changes the
Hessian of any model with that pattern — grep the fixture corpus for
`o15` under `o5` under `o2` and list the affected fixtures beforehand);
`cargo test -p pounce-nl -p pounce-cli`; `nonfinite_hessian_restoration_strike.rs`
must still pass — note its header says the exemption is what makes it pass,
so if (A) removes the kink entirely that test no longer exercises the
exemption: keep the exemption, and give the test a second model that still
reaches a non-finite Hessian (e.g. `sqrt(x)` at `x = 0` with
`x ≥ 0`), mutation-checked with the exemption reverted.

**Must not regress.** The `5fc8c797` exemption and its test; `dallasl`,
`dallasm` on the corpus (same term structure) if available.

### B1 — pounce reuses the multiplier-init matrix's symbolic ordering

**Goal.** The first IPM KKT factorization gets its own symbolic analysis
instead of inheriting the ordering FERAL's value-dependent `LdltCompress`
preprocessing chose for the least-squares multiplier-init system
(W = 0, D_x = D_s = I). Comment 6344966966: replaying qcqp1000-2c, 98.98 s
→ 56.39 s factor time, delays 2.95M → 1.61M, identical inertia on all 165
systems.

**Before starting:** `git fetch origin && git branch -r | grep -i symbolic`
and search open PRs for `perf/feral-symbolic-after-init`. If that branch or
PR exists, do not duplicate it — record its state in the item log and skip
B1 (it is the owner's in-progress work). It was **not** on `origin` on
2026-10-22.

**Files / symbols.**
- `crates/pounce-feral/src/lib.rs`: the per-pattern symbolic cache ("pattern-
  fingerprint cache likewise reuses the symbolic factorization", header
  ~line 17), `FeralSolverInterface::factor` (~1307), the invalidation block
  (~1604); FERAL 0.21's `Solver::invalidate_symbolic_cache` (registry source
  `feral-0.21.0/src/numeric/solver.rs` ~2812).
- `crates/pounce-algorithm/src/eq_mult/least_square.rs` (~49–200): the
  multiplier-init solve goes through the same `AugSystemSolver`
  (`StdAugSystemSolver` "pins its triplet structure" there).
- Other first-factorization sites the comment lists: restoration's first
  factorization (`AugRestoSystemSolver`), the second-opinion ladder and
  gh#983 audit re-solves, warm starts.
- `crates/pounce-linalg` `SparseSymLinearSolverInterface` trait, for where a
  "next factor is a different regime, re-analyse" hint belongs.

**Steps.**
1. Trace which `FeralSolverInterface` instance the init solve and the first
   IPM KKT share, and confirm (with a counter or `RUST_LOG`) that the second
   factor does not re-run symbolic analysis. Record it.
2. Add an interface method (e.g. `invalidate_symbolic_cache`, default no-op)
   and call it once after the multiplier-init factor (and at the other
   regime boundaries only if step 1 shows they share a cache). Keep the
   per-pattern cache otherwise: re-analysing every factor is the slowdown
   the cache exists to prevent.
3. Unit test in `pounce-feral`: two factors with the same pattern but
   different values re-use the analysis by default and re-analyse after the
   invalidation; an `pounce-algorithm` test that the init → first-KKT
   transition triggers exactly one re-analysis.

**Acceptance.** Without the corpus: the tests above, and the fixture sweep
explained. With the corpus: qcqp1000-2c wall time materially down from 112 s
toward the comment's replay (56 s on FERAL 0.21; the "99 it" of 0.12.0 is a
different trajectory and not a target), `LinearSystemFactorization` time
reported in the JSON report, inertia unchanged.

**Validation.** Full fixture sweep both legs — a different ordering can
change pivots, so lines may move; a line moving only in `q=` or `it=` on a
nonconvex model is a trajectory change to explain, any status loss is a
stop. `cargo test -p pounce-feral -p pounce-algorithm -p pounce-cli`.
`benchmarks/mittelmann` and `vanderbei` A/B if the corpus is available.

**Must not regress.** `pounce-feral`'s factor-stats tests (#955), gh#857
quality-escalation accounting (escalations never step back down), the
`issue_983_scale_audit_retry_budget` test.

### R4 — MLSL default `gamma` and `circle_parabola_t0p5`

**Goal.** `circle_parabola_t0p5` under `--minima mlsl --seed 0 --n-minima 6
--max-solves 30 --patience 8` finds both minima again without losing the
gains #993 bought on camel / wavy / wavy_1e12.

**Files / symbols.** `crates/pounce-minima/src/search.rs`: `MLSL_GAMMA`
(~line 75), `run_mlsl` (~647–690: `radius = gamma·√n·(ln N/N)^(1/n)`,
`better_sample_within`, `near_any` against the archive), `consider` /
`reject` / `patience` (~530–605); `options.rs` (`gamma: Option<f64>`);
`crates/pounce-cli/src/cli.rs` ~328 (help text says 0.5);
`python/pounce/_minima.py`; `docs/src/find-minima-cli.md` ~84,
`docs/src/find-minima.md` ~174. Guards:
`crates/pounce-cli/tests/issue989_mlsl_cli.rs`,
`python/tests/test_minima.py::test_mlsl_default_gamma_launches_solves_on_a_2d_box`
and `::test_mlsl_sample_cap_status_names_the_cap`,
`python/tests/test_minima_cli_parity.py`; `scripts/sweep-minima.sh`.

**Measured (x86, `p-head`, seed 0, the sweep's budget `--n-minima 6
--max-solves 30 --patience 8`; "found" = minima, status in brackets):**

| gamma | camel | circle_parabola_t0p5 | haverly_t3_brf0 | wavy | wavy_1e12 |
|---|---|---|---|---|---|
| 0.5 (default) | 4 (converged, 12 solves) | **1** (converged, 9) | 2 (10) | 6 (target) | 6 (target) |
| 0.75 | 5 (sample cap, 5) | 2 (17) | 2 (10) | 6 | 6 |
| 1.0 | 4 (sample cap, 4) | 2 (15) | **1** (sample cap, 1) | 6 | 6 |
| 1.5 | 2 (sample cap) | 2 (12) | 1 (sample cap) | **1** (sample cap) | 1 |
| 2.0 (0.12.0) | 2 (sample cap) | 2 (11) | 1 (sample cap) | 1 (sample cap) | 1 |

(`nonconvex_qcqp`, `nonconvex_two_escapes`, `saddle_axis_first`,
`wavy_accept_anything` read `found=0` at every gamma, as the sweep expects.)
`p-base` at its default 2 reproduces found=2 / 11 solves; `p-base
--gamma 0.5` reproduces found=1 / 9. So the loss is the gamma value, not the
gh#966 extraction, and at gamma 0.5 the search ends by **stagnation** (8
consecutive rejected solves re-finding minimum 1), not by budget. No single
value dominates at one seed: 0.75 is the only row that loses nothing in this
table, and it is one seed.

**Steps.**
1. Reproduce the row above with `p-prev`.
2. Measure over **seeds 0–9** the five fixtures above at gamma ∈ {0.5, 0.6,
   0.75, 1.0} (CLI loop; ~25 runs per cell are sub-second). Score per fixture:
   mean minima found and mean solves. This is the measurement #993 did not
   record for the constrained fixtures.
3. Decide from the measurement: (a) a new default (only if one value is
   never worse on mean found than 0.5 on camel/wavy and recovers
   circle_parabola on most seeds); or (b) a structural change that does not
   move gamma — e.g. in `run_mlsl`, a solve launched from a sample that
   converges to an already-archived minimum extends the archive's
   `near_any` radius for that minimum (so the clustering filter learns the
   basin) instead of counting as a stagnation strike. (b) is the better
   shape — the stagnation rule was written for multistart, where every solve
   is an independent draw, and MLSL's launches are not independent — but it
   is a search change and needs the same 10-seed table. Pick one; do not do
   both in one iteration.
4. Update the help text, the two docs pages and the Python docstring if the
   default moves; update `issue989_mlsl_cli.rs` and the Python gamma tests
   only to the extent their *stated* contracts (all six camel minima at
   `--max-solves 60 --patience 15`; `sample_cap_reached` at gamma 100) need
   it — those contracts must still hold.

**Acceptance.** `scripts/sweep-minima.sh` `p-prev` vs new: the
`circle_parabola_t0p5 mlsl` line reads `found=2`; no other `mlsl` line reads
fewer minima than before; non-mlsl lines (the other five methods) **diff
empty**. The 10-seed table is in the item log and in the commit message.

**Validation.** `scripts/sweep-minima.sh` (54 lines) `p-prev` vs new;
`scripts/sweep-fixtures.sh` must diff **empty** (the single-solve path is
untouched); `cargo test -p pounce-minima -p pounce-cli` (`issue989_mlsl_cli`,
`minima_*`); `python/tests/test_minima.py` and `test_minima_cli_parity.py`
if the extension builds — the parity test is the one that catches a CLI /
Python default drifting apart, so if the default moves and Python cannot be
run here, say so loudly in the log and keep the Python default in
`python/pounce/_minima.py` in lockstep by inspection.

**Must not regress.** `mlsl_default_gamma_finds_all_six_camel_minima`,
`the_sample_cap_is_named_as_such`, the gh#966 parity contract, the
`sweep-minima.sh` `found=0` saddle lines and the `kkt_rej` line.

### B2 / B3 — brainpc0/1 restoration path, qcqp1500-1c (corpus required)

Only with `bench_data_root` resolving, and only after B1 has landed and the
vanderbei/mittelmann A/B has been re-taken on top of it. Goal for B2: find
why restoration spends 811 of 1073 iterations on brainpc1 after a *correct*
inertia at iteration 41 (the comment's diagnosis), with the restoration
cycle detectors in `ipopt_alg.rs` (`resto_no_outer_progress_count`,
`resto_near_feasible_count`, the gh#1000 I.20 inner budget inheriting
`max_iter`) as the first suspects. No acceptance numbers can be set without
the corpus; if it is absent, record "blocked: no benchmark corpus on this
machine" and stop. B3 is presumed to follow B1 (same family); verify, do not
assume.

## 4. Loop protocol

**Pick the next item** in the §2 order whose Status is `todo` and whose
dependencies are `done` or `skipped` (R3 waits for R2; B2/B3 wait for B1 and
the corpus). One item per iteration. Never start a second item in the same
iteration, even if the first was quick — its sweep is the evidence for its
own commit.

**Per-iteration checklist.**
1. `git status` clean, on `claude/pounce-regressions-recovery-tz8uh2`,
   `git pull --ff-only origin claude/pounce-regressions-recovery-tz8uh2`.
2. Build `p-prev` from the current tip (release, `pounce-cli`), and if
   missing, `p-base` from `v0.12.0` in a worktree with its own
   `CARGO_TARGET_DIR`. Keep both outside the tree.
3. **Reproduce** the regression with `p-prev` exactly as the item's
   "reproduce" step says, and write the numbers into the item log *before*
   changing code. If the regression does not reproduce, stop and record
   that (it may have been recovered by an earlier item — that is a result).
4. Measure what the item says to measure before implementing (R2 step 2, R4
   step 2). If the measurement refutes the hypothesis, mark the item
   `not recoverable (reason)` with the data, suggest a follow-up issue
   title in the log, and end the iteration with a docs-only commit of this
   file.
5. Implement the narrowest change; add the regression test; mutation-check
   it (revert the fix, the test fails; restore).
6. Validate per the item: `cargo fmt --all -- --check`, `cargo clippy
   --workspace --exclude pounce-hsl --all-targets -- -D clippy::correctness
   -D clippy::suspicious`, the crate tests named, then the sweep(s) named,
   `p-prev` vs new (and vs `p-base` for the item's own fixtures). Paste the
   full `diff` of the sweep into the item log, with one line of explanation
   per moved line. An unexplained moved line is a stop, not a footnote.
7. Update this file: the inventory Status cell, the item log (numbers
   before/after, sweep diff, what was not run and why).
8. Commit code + this file together. Message format:

   ```
   <area>: <what changed> (#1042 <id>)

   <2–6 lines: before → after numbers on this machine, root cause in one
   sentence, what the sweep moved and why, what was not run.>

   Refs #1042
   <the attribution trailers your session's reminder specifies>
   ```

   Use `Refs`, not `Fixes`/`Closes`: #1042 stays open until the owner
   closes it. Name the architecture every number was taken on.
9. `git push origin claude/pounce-regressions-recovery-tz8uh2`.
10. If CI on the branch goes red on something this iteration touched, the
    next iteration fixes it first (docs-only commits excepted).

**Stop / skip conditions.** Mark an item `not recoverable` (with the
measurement) rather than forcing it when any of these holds: the
measurement in step 4 refutes the hypothesis; the fix needs the fix that
caused the regression to be weakened (the `2599c2be` reset, the `5fc8c797`
exemption, the gh#1000 stopping-test corrections, the gh#414/gh#984
guards); the sweep moves a status, an engine, or an objective outside a
known bistable pair and you cannot explain it from the change; the item
needs arm64 or the benchmark corpus and neither is available; or two
iterations have been spent on it. In every such case write the suggested
follow-up issue (title + the three numbers that justify it) into §5 of this
file and move on.

**Keeping the branch green.** Every commit carries its own passing tests
and formatted code; never commit a half-implemented item (stash it or
revert). Do not re-point an existing pin to make a number fit — the
`issue_592` `< 500` bound in particular is left alone unless R2 lowers the
count well below it, in which case tighten it to the measured margin, never
raise it. Docs-only iterations (skips) still run `cargo fmt --check`.

## 5. Judged unrecoverable or out of scope

- **X1 exact `deb7` arm64 Optimal/147 → Acceptable/180.** Same objective to
  7 digits; on x86 the same change is an improvement (143 → 114, measured).
  A NEON-vs-AVX2 rounding difference on a nonconvex model is not something
  this branch can measure or should steer. Leave recorded in #1042; if the
  owner wants it chased it needs an arm64 machine and the R2 ladder
  measurement there.
- **X2 lbfgs `dallass` arm64 Optimal/224 → ErrorInStep/308.** On x86 the
  lbfgs leg fails on **both** 0.12.0 (ErrorInStep/213) and 0.13.0
  (ErrorInStep/146 + ladder, tot 292), so the issue's "arm64 only" names
  where the *verdict* changed, not a leg that is otherwise healthy. The
  lbfgs failure on this model is older than 0.13.0 and is a separate
  defect (suggested follow-up: "dallass under limited-memory ends
  Error_In_Step_Computation on x86 on 0.12.0 and 0.13.0").
- **X3 lbfgs `pooling_rt2stp` arm64 lands on −3273.95 instead of −4391.83.**
  Both are certified optima of a bistable nonconvex model; the tests already
  accept either; on x86 the leg went from failing to solved. Nothing to
  recover.
- **lbfgs `deb7` x86 ErrorInStep/1010 → RestorationFailed/1206 (tot 1738).**
  Not in the issue; both fail; recorded here so the next sweep reader does
  not mistake it for a new loss.
- **B4 `cresc132` Optimal → Infeasible** (Ipopt's verdict too) and
  **`ssebnln` Optimal → Acceptable** (same objective): corpus-only,
  FERAL-trajectory, no mechanism named by anyone, and `cresc132` was already
  in the report's "genuine misses" list. Out of scope unless B1/B2 work
  happens to move them; record if it does.
- **R1 `scaled_feasible_a` cvx-qp 200 / 200 it (both legs).** Not
  recoverable with the planned stall-gated certificate or `cap_p` floor: the
  instrumented loop (log below) shows the model has **no equality rows**
  (`m_eq = 0`), the `pres = 5.0e-6` plateau is the rounding floor of
  inequality rows at `‖x̂‖ = 5e9` (`ε·5e9 = 1.1e-6`; `cap_p = 7.1e-5`, which
  `pres` already satisfies), and through the plateau the un-homogenized point
  is **not optimal** (`true_kkt_error` 2.6e5 at it 20, 9.8e3 at it 120, 23 at
  it 125). The earliest any in-loop certificate could stop is it 165, after a
  blow-up (`inf_pr` 4.8e9 at it 140) and a 15-iteration recovery — 200 → ~166
  at best, and dependent on that blow-up. Suggested follow-up issue: **"HSDE:
  μ collapses ~8×/iteration while true complementarity crawls linearly once
  τ ≈ 1e-10 (`scaled_feasible_a`)"** — numbers: retry it 20 → 120 `mu`
  5.0e-19 → 3.2e-117 with `alpha = 1` every step while `true_comp` only goes
  2.6e5 → 9.8e3 and `dres` falls by a constant 5.0e-7/iteration; the first
  (unequilibrated) HSDE solve also ends at the 200 cap with
  `true_kkt_error 0.149` (τ creeping linearly 1.6e-9 → 7.6e-8, `true_comp`
  ∝ 1/τ), so the model costs 400 HSDE iterations in all; `max_iter=400`
  converges at 230. Candidate shapes (both trajectory changes across the
  whole convex arm, so both need `benchmarks/qp`, absent here): a centering
  safeguard that keeps the `σμ` target from falling below the
  *true*-slack complementarity when the internal `s` has decoupled from
  `h − Gx̂` (the gh#689 decoupling), or an HSDE cold start sized to the
  data the way the direct driver's cold seed is (gh#689: Mehrotra
  recentering of the seed in the `run_ipm` starting-iterate builder,
  `crates/pounce-convex/src/ipm.rs` ~4485, which took the direct solve of
  this model 175 → 27 iterations). The `ipm.rs` doc comment on the direct
  driver's scale-relative arm (~4303) already names this plateau as `≈ 4 ulp`
  of a `5e9` cancellation — the plan's "regularization floor" reading was
  wrong.
- **R2 `pooling_rt2stp` exact 492 it (v0.12.0: 162).** Not recoverable
  with the planned Gershgorin / quasi-definite acceptance: of 1310 ladder
  factorizations, the 146 `WrongInertia` ones with `δ_c > 0` **all** have an
  indefinite (1,1) block (0 PD, 0 Gershgorin-dominant), so the rule is
  inert here. The high `δ_x` is the correct answer to a `W` whose λ_min is
  −1e13…−1.2e19 (151 of 516 systems below −1e10; v0.12.0: worst −2.2e9,
  none below −1e10). The `W` blow-up starts at iteration 4: the gh#592
  walk-back withdraws `δ_c` (both versions do this) and the `δ_c`-free
  factor at `δ_x = 1e-4` is numerically singular (min |eig| 2e-17, FERAL
  reports `Success`, 72 negatives); its solve has a dual part of **6.4e16**.
  On v0.12.0 the same decision gives 3.5e16, but iterative refinement stalls
  (7.5e-9), escalates pivot quality, refactors and gets **3.0e6**. On
  0.13.0 the ladder's first factor at iteration 4 reported 71 < 72
  negatives, which takes `pd_full_space_solver.rs`'s too-few-negatives
  branch and spends the matrix's one quality escalation there
  (`augsys_improved = true`); refinement then stalls at 1.6e-8, cannot
  escalate, takes the `S` exit (ratio < `residual_ratio_singular`) and keeps
  the 6.4e16 step; λ_min(W) is −4.1e13 at iteration 5. Counterfactual
  (probe build only, env-gated, not committed): letting refinement escalate
  even when `augsys_improved` is set takes `pooling_rt2stp` exact
  **492 → 109** (−4391.826, the other certified optimum; 3 restoration
  calls), `dallass` 123 and `deb7` 114 unchanged — but as a blanket rule it
  moves 19 sweep lines (escalations run to the cap of 12/24 on 15 of them;
  lbfgs `circle_parabola_t0p5` changes second-opinion rung and objective
  8.155 → 10.428; lbfgs `scale_audit_retry_meyer3` 112 → 152). Suggested
  follow-up issue: **"PD solver: a quality escalation spent by the inertia
  ladder's too-few-negatives branch leaves iterative refinement none for
  the same matrix, so a numerically singular `δ_c`-free factor's step is
  kept (`pooling_rt2stp`, 492 vs 109 it)"** — numbers: iteration-4 dual
  step 6.4e16 kept vs 3.0e6 after escalation; λ_min(W) −2.2e9 (v0.12.0) vs
  −1.2e19 (0.13.0); 492 → 109 under the blanket counterfactual with 19
  collateral sweep lines. Candidate shapes for that issue: track the two
  escalation reasons separately (one for the inertia branch, one for
  refinement, per matrix), or refuse the `S` exit when the step's dual part
  is many orders above the right-hand side's. Both are trajectory changes
  across the NLP arm; they need the full sweep and the vanderbei A/B.
- **Anything that requires the benchmark corpus** (B1's benefit, B2, B3) is
  blocked on this machine: `POUNCE_BENCH_DATA` is unset and no
  `~/projects/pounce-bench-data` exists. The code half of B1 is still in
  scope.

## 6. Item logs

Append one subsection per iteration: `### <id> — <date> — <status>` with
the before/after table, the sweep diff and its explanations, tests run,
tests not run, and the commit SHA.

### Baseline — 2026-10-22 — measured on x86_64 at `0a299151` vs `v0.12.0`

| fixture (leg) | `p-base` v0.12.0 | `p-head` 0a299151 |
|---|---|---|
| pooling_rt2stp (exact) | nlp SolveSucceeded 162 it q=4 −3273.954992, 3 resto calls | nlp SolveSucceeded 492 it q=3 −3273.95499, 24 resto calls / 106 resto it |
| pooling_rt2stp (lbfgs) | nlp ErrorInStepComputation 716 it q=12 −4391.82 | nlp SolveSucceeded 82 it q=0 −3273.954992 |
| dallass (exact) | nlp SolveSucceeded 41 it q=6 −32393.22574, 2 resto calls | nlp SolveSucceeded 123 it q=19, 12 resto calls / 53 resto it |
| dallass (lbfgs) | nlp ErrorInStepComputation 213 it q=7 | nlp ErrorInStepComputation 146 it q=4 (ladder kept, tot 292) |
| deb7 (exact) | nlp SolveSucceeded 143 it 97.55993795 | nlp SolveSucceeded 114 it 97.55993795 |
| deb7 (lbfgs) | nlp ErrorInStepComputation 1010 it | nlp RestorationFailed 1206 it (ladder kept, tot 1738) |
| scaled_feasible_a (exact / lbfgs) | nlp SolveSucceeded 22 / 19 it, obj 1.1e-8 / 2.8e-7 | cvx-qp SolveSucceeded 200 / 200 it, obj 0, kkt 5e-19 |
| circle_parabola_t0p5 mlsl seed 0 | found 2 in 11 solves (gamma 2) | found 1 in 9 solves (gamma 0.5), status converged |

### R1 — 2026-10-22 — not recoverable this way (hypothesis refuted)

Measured on x86_64 (4 CPUs, AVX2+FMA) with `p-prev` = release `pounce` built
from `75e410d2` in a separate worktree with its own `CARGO_TARGET_DIR`.

**Reproduced (before any change).**

| run (`p-prev`) | engine | status | it | objective | final kkt |
|---|---|---|---|---|---|
| exact (default) | cvx-qp | SolveSucceeded | 200 | 0.0 | 5.0e-19 |
| lbfgs (`hessian_approximation=limited-memory`) | cvx-qp | SolveSucceeded | 200 | 0.0 | 5.0e-19 |
| `bound_relax_factor=0` (and `+ solver_selection=qp-ipm`) | cvx-qp | SolveSucceeded | 200 | 0.0 | 5.0e-19 |
| `max_iter=400` | cvx-qp | SolveSucceeded | 230 | 0.0 | 0.0 |
| `qp_equilibrate=no` / `qp_hsde=no` | nlp | SolveSucceeded | 22 | 1.1e-8 | 9.0e-9 |
| `scaled_feasible_b` default | cvx-qp | SolveSucceeded | 36 | 0.0 | — |

The `iterations[]` trace matches the plan (inf_pr 3.9e-5 at it 13, 5.000e-6
from it 20 to ~125, alpha 1.0 throughout, blow-up to 4.8e9 around 130–150,
recovery by 153, then 5.7e-7 ↔ 2.8e-6 to the cap).

**Instrumented (temporary `eprintln!` in `hsde.rs`, reverted; not
committed).** Per iteration: `pres` split into `ρ_y`/`ρ_z`, `dres`, `gap`,
`comp_max`, `true_comp`, the stop flags, `base_reg`/`reg_eff`/`δ_c`/`δ_w`,
and `true_kkt_error(prob, cone, x, y, z, τ)`. Both HSDE solves of the call
were traced (the first, unequilibrated one also runs the full 200).

| retry it | pres | dres | gap | comp_max | true_comp | `true_kkt_error` | mu | τ |
|---|---|---|---|---|---|---|---|---|
| 13 | 3.9e-5 | 2.9e-2 | 7.0e8 | 8.8e7 | 8.8e7 | 8.8e7 | 4.2e-13 | 1.0e-10 |
| 20 | 5.0e-6 | 5.4e-5 | 6.6e2 | 1.0e2 | 2.6e5 | 2.6e5 | 5.0e-19 | 9.7e-11 |
| 60 | 5.0e-6 | 3.2e-5 | 1.7e-2 | 8.3e-37 | 1.6e5 | 1.6e5 | 7.9e-57 | 9.8e-11 |
| 120 | 5.0e-6 | 2.0e-6 | 1.5e-2 | 3.5e-97 | 9.8e3 | 9.8e3 | 3.2e-117 | 9.8e-11 |
| 125 | 5.0e-6 | 4.6e-9 | 5.1e-4 | 7.2e-101 | 2.3e1 | 2.3e1 | 4.6e-121 | 9.8e-11 |
| 140 | 4.8e9 | 1.2e-7 | 5.6e-2 | 5.6e-2 | 8.3 | 8.3 | 5.0e-23 | 9.8e-11 |
| 160 | 5.7e-7 | 1.7e-8 | 5.8e-4 | 1.5e-4 | 0 | 1.5e-4 | 7.9e-25 | 9.8e-11 |
| 165 | 1.1e-6 | 1.3e-10 | 1.3e-4 | 8.6e-9 | 0 | **8.6e-9** (first < tol) | 3.5e-29 | 9.8e-11 |
| 199 | 8.5e-6 | 7.8e-15 | 3.2e-5 | 9.1e-21 | 0 | 9.1e-15 | 8.5e-42 | 9.8e-11 |

- `m_eq = 0`, `m_ineq = 10`, `n = 4`: `ρ_y` is empty, so there is no
  equality-block `δ·‖ŷ‖` floor; `δ_c` sits at 1e-10, `δ_w` at 1e-10,
  `base_reg = reg_eff = 1e-14`, zero inertia retries, on every iteration.
- `cap_p = 7.1e-5` (`64·ε·scale_p`, `scale_p ≈ 5e9`): `pres = 5e-6` already
  satisfies it. What blocks a stop through it 20–125 is `comp_ok`
  (`true_comp` 2.6e5 → 23 against `tol_true_comp = 1e-6`) — the point is
  genuinely not optimal, with internal `s·z` at 1e-37…1e-101 while the true
  row slack times `ẑ` is 1e5. `true_kkt_error` tracks `true_comp`.
- After the blow-up the loop recovers and converges in ~15 iterations
  (151 → 165); from 165 on `comp_ok` holds and `true_kkt_error < tol`, and
  the stop is blocked only by `pres < tol` (unreachable, ~4 ulp) and by
  `gap_rel` (the relative arm has no gap excuse).
- First, unequilibrated HSDE solve: τ creeps linearly 1.6e-9 → 7.6e-8 over
  it 19–199, `true_comp` falls ∝ 1/τ (7.2 → 0.149), `mu` ~1e-37; ends
  `IterationLimit` at `true_kkt_error 0.149`, so the gh#293 retry runs.

**Verdict.** Both planned shapes are refuted by the measurement: (1) a
stall-gated in-loop `true_kkt_error` certificate cannot fire before it 165,
because the plateau iterate is not optimal; it would buy 200 → ~166 at best,
only by virtue of the chaotic blow-up-and-recovery, and against `ipm.rs`'s
reasoning for judging only after the loop; (2) widening `cap_p` changes
nothing, since `pres ≤ cap_p` already holds. The acceptance target (≤ 60 on
both legs) is out of reach for either. The real cost is the 20–125 crawl —
`mu` dividing by ~8 per full step while the true complementarity falls
linearly — which is an HSDE centering / cold-start problem whose fix would
move trajectories across the whole convex arm and needs `benchmarks/qp`
(not on this machine). Follow-up issue and numbers in §5. Not touched, per
the plan: `equilibrated_retry_budget`, the gh#535 decline rule, the gh#414 /
gh#984 guards. The "3596 iterations" comments in `main.rs` were not
re-measured and are left as they are.

**Tests / sweep.** Docs-only commit: no code changed, so no regression test,
no mutation check, no fixture sweep (nothing can move) and no `cargo test`;
`cargo fmt --all -- --check` run (clean). Not run: `benchmarks/qp` (no
corpus: `POUNCE_BENCH_DATA` unset, `bench_data_root` finds none), Python
tests (no code change).


### R2 — 2026-10-22 — not recoverable this way (hypothesis refuted)

Measured on x86_64 (4 CPUs, AVX2+FMA). `p-prev` = release `pounce` built
from the clean tip `463eeeae` (code-identical to `75e410d2`) with its own
`CARGO_TARGET_DIR`; the cached binary from the R1 iteration was not reused
because a rebuild from the clean tree was not byte-identical (R1's
reverted instrumentation had touched `pounce-convex`). `p-base` = v0.12.0.

**Reproduced (before any change).**

| run | engine | status | it | objective | resto calls / it | q | lg(rg) ≥ 19 / ≥ 10 |
|---|---|---|---|---|---|---|---|
| `p-prev` exact | nlp | SolveSucceeded | 492 | −3273.95499 | 24 / 106 | 3 | 40 / 171 |
| `p-prev` lbfgs | nlp | SolveSucceeded | 82 | −3273.954992 | — | 0 | — |
| `p-base` exact | nlp | SolveSucceeded | 162 | −3273.954992 | 3 / 9 | 4 | 0 / 0 |
| `p-prev` exact, `perturb_delta_c_max_rungs=0` | nlp | SolveSucceeded | 115 (stalls, dual inf 1.8e8) + re-solve 205 | −3273.95500 | 0 | — | 2 / 50 |
| `p-base` exact, `perturb_delta_c_max_rungs=0` | nlp | SolveSucceeded | 116 | — | 2 / 7 | — | 0 / 0 |

No second opinion on the default exact run. `p-base`'s largest `δ_x` is
3.6e9 (40 iterations ≥ 1e4); `p-prev`'s is 9.7e19 (183 iteration records
≥ 1e10, restoration included).

**Instrumented (temporary probe, reverted; not committed).** In
`StdAugSystemSolver::solve`, after every factorization: the assembled KKT
densified and its eigenvalues taken (faer), the (1,1) block `W + Σ_x + δ_x I`
— its λ_min and Gershgorin margin — the slack diagonal minimum, `δ_x`,
`δ_c`, FERAL's reported negative count against the required 72, and the
max-norm of the primal and dual parts of the solution; in
`PdFullSpaceSolver` the iteration at each `consider_new_system`, the
accepted deltas, and the iterative-refinement residual ratios. The same
probe was applied to a v0.12.0 worktree (built with its own target dir) so
both versions were measured identically. Restoration ladders run in the
same 46-variable space (`AugRestoSystemSolver` condenses), so the counts
below include them. The probe does not perturb the run: 492 / 162
reproduced under it.

All 1310 ladder factorizations on `p-prev` (re-solves after a quality
escalation excluded), by status, `δ_c`, and whether the (1,1) block is PD
(λ_min > 0) / Gershgorin-dominant:

| status | `δ_c > 0` | (1,1) PD | Gershgorin | count |
|---|---|---|---|---|
| WrongInertia | yes | no | no | **146** |
| WrongInertia | yes | yes | — | **0** |
| WrongInertia | no | yes | 7 of 9 | 9 |
| WrongInertia | no | no | no | 138 |
| Singular | yes | yes | yes | 5 |
| Singular | yes | no | no | 202 |
| Singular | no | yes | 55 of 57 | 57 |
| Singular | no | no | no | 237 |
| Success | (all) | | | 516 |

- The planned arm (`WrongInertia`, `δ_c > 0` or `m_c = 0`, block provably
  PD) has **zero** candidates: `m_c = 40`, and every `WrongInertia` with
  `δ_c > 0` has an indefinite (1,1) block. The 9 PD `WrongInertia` factors
  have `δ_c = 0`, where Sylvester proves nothing without a rank argument for
  `J`. The 5 provable cases are `Singular` (excluded by the plan: gh#981
  territory).
- Of 516 ladders, 170 end at `δ_x ≥ 1e10` (1 overflows). In those, the
  first-rung λ_min(W+Σ_x) is negative in 165 (median −1.5e14, min −1.2e19).
  At acceptance `δ_x / −λ_min` has quantiles 0.00016 / 1.4 / 2.4 / **5.7** /
  2.1e5 / 3.7e8 (min/10/25/50/75/90 %): the median ladder stops about one
  rung past the (1,1) PD threshold. 88 of 169 accept at their first
  non-zero rung (the start is `δ_x_last/3`), and the over-damped tail
  (41 + 14 ladders > 100× the threshold) comes from that inherited start,
  not from climbing.
- `W` itself: λ_min(W+Σ_x) at each system, quantiles min/5/10/25/50 %:
  `p-prev` −1.2e19 / −1.2e19 / −3.4e15 / −6.0e12 / −1.1e6 (151 of 516 below
  −1e10, 347 below −1e4); `p-base` −2.2e9 / −2.2e9 / −2.2e9 / −8.2e5 /
  −6.4e5 (0 of 162 below −1e10, 132 below −1e4).

**Where `W` blows up (iteration 4 → 5).**

| | `p-base` v0.12.0 | `p-prev` 0.13.0 |
|---|---|---|
| it-4 ladder | WI(73), WI, WI with `δ_c` → walk-back `w` → Success at `δ_x` 3.3e-3, `δ_c` 0 | WI(**71** < 72) → too-few-negatives branch **escalates quality** (`q`) → Singular → WI(73) ×3 with `δ_c` → walk-back → Success at `δ_x` 1e-4, `δ_c` 0 |
| min eig of the accepted factor | 1.6e-17 | 2.0e-17 |
| dual part of the solve | 3.5e16 | 6.4e16 |
| refinement ratios | 9.0e-7, 7.5e-9, 7.5e-9 → stall, `augsys_improved` false → escalate, refactor | 9.1e-6, 2.2e-8, 1.2e-8, 1.6e-8 → stall, `augsys_improved` already true → `S` exit (< `residual_ratio_singular`), step kept |
| dual part after refinement | **3.0e6** (ratio 1e-16) | **6.4e16** |
| λ_min(W+Σ_x) at iteration 5 | −7.4e2 | **−4.1e13** |

So both versions withdraw `δ_c` and accept a numerically singular factor
at iteration 4; v0.12.0 is rescued by the refinement's quality escalation,
0.13.0 is not, because FERAL 0.21's count at that iteration's unperturbed
factor (71 instead of 73) spent the escalation in the inertia branch. The
`2599c2be` reset is not involved.

**Counterfactual (probe build only, env-gated, not committed).** Allowing
the refinement branch to escalate even when `augsys_improved` is set:
`pooling_rt2stp` exact 492 → **109**, −4391.82601 (the other certified
optimum, accepted by `issue_592`), 3 restoration calls, q = 4; `dallass`
exact 123 (unchanged), `deb7` exact 114 (unchanged). Full sweep, `p-prev`
vs counterfactual, 19 lines move:

```
exact  infeasible_square_scaled_1em4  q=3 -> q=12
exact  pooling_rt2stp                 it=492 q=3 obj=-3273.95499 -> it=109 q=4 obj=-4391.82601
exact  square_flowsheet_resto         it=95 q=4 -> it=76 q=13
lbfgs  circle_parabola_t0p5           it=77 q=4 2nd=feral_scaling=mc64@.../48,tot=125 obj=8.155059653
                                   -> it=51 q=24 2nd=start_point_perturbation=1e-2@.../48,tot=227 obj=10.42794559
lbfgs  circle_parabola_t1p5           q=10 -> q=12
lbfgs  cresc4                         q=0 2nd=feral_increase_quality=no@.../3000,tot=3200 -> q=24 2nd=-
lbfgs  degenerate_start_infeasible    tot=74 -> tot=77
lbfgs  eigena2                        it=239 q=3 tot=478 -> it=164 q=11 tot=328 (ErrorInStep both)
lbfgs  eigmaxa                        q=2 -> q=12
lbfgs  infeasible_square_scaled_1em4  q=1 -> q=12
lbfgs  issue880_sigma_uncertified     it=67 q=8 -> it=59 q=12
lbfgs  issue981_water_main            q=1 -> q=12
lbfgs  issue_508_infeasible_gap_1em2  q=5 tot=284 -> q=12 tot=290
lbfgs  issue_508_infeasible_gap_1em4  q=1 tot=346 -> q=12 tot=349
lbfgs  scale_audit_retry_meyer3       it=112 q=5 -> it=152 q=12
```

No status or engine moves, but the blanket rule runs escalations to the
cap and changes an lbfgs objective, so it is a measurement of where the
lever is, not a candidate fix. The narrower shapes are in §5.

**Verdict.** The plan's hypothesis — FERAL miscounting the inertia of a
provably quasi-definite matrix, so that the ladder climbs past a rung where
Sylvester already guarantees it — is false on this fixture: the arm it
describes has zero candidates in 1310 factorizations. The climb is the
correct response to a Hessian of the Lagrangian that is indefinite by up to
1.2e19, and that comes from multipliers blown up by one kept step at
iteration 4. Per §4 step 4 no code change was made; the follow-up issue
(title + numbers) is in §5. The `issue_592` `< 500` pin is untouched (F1
stays open with a margin of 8). The unit tests named in the plan were not
needed — nothing they guard changed.

**Effect on R3.** R2 changed no code, so R3's re-baseline is the baseline:
`dallass` exact 123 it / 12 restoration calls / q = 19, lbfgs
ErrorInStepComputation 146 (tot 292), both re-measured on `p-prev` in this
iteration's sweep. R3's dependency is resolved (R2 closed as not
recoverable); R3 proceeds to its step 2 (the `pounce-nl` peephole).

**Tests / sweep.** Docs-only commit: no code changed, so no regression
test, no mutation check, no `cargo test`; `cargo fmt --all -- --check` run
(clean). `scripts/sweep-fixtures.sh` was run on `p-prev` (234 fixture-legs;
engine split nlp 127, cvx-qp 92, cvx-qcqp 14, square-lm 1) only as the
baseline for the counterfactual above; there is no new binary to diff
against. Not run: `benchmarks/vanderbei` (no corpus: `POUNCE_BENCH_DATA`
unset), Python tests (no code change), arm64 (not available).
