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
| **R1** | `scaled_feasible_a` (exact and lbfgs) | nlp 22 / 19 it → cvx-qp **200 / 200** it, right answer | #1002 `f6a4e4c2`: HSDE endgame certifies the gh#293 equilibrated retry, whose loop never meets `pres < tol` (plateau 5.0e-6 → 5.7e-7) while `mu` runs to 1e-121; `Optimal` is stamped only by the post-loop `true_kkt_error` upgrade | **yes** — a stall-gated in-loop certificate, or a primal-residual floor that reads the stalled `pres` as the regularization floor it is | todo |
| **R2** | `pooling_rt2stp` (exact, x86) | 162 → **492** it (arm64: 109 → 242 total, *issue*) | FERAL 0.21 trajectory; `2599c2be` removed the latch (904 → 492); the remaining cost is the `δ_x` ladder climbing to `max_hessian_perturbation` with `δ_c` on (*issue*, unmeasured) | **partial** — the "stop the ladder once inertia is guaranteed" follow-up the issue names; must be measured first | todo |
| **R3** | `dallass` (exact, x86) | 41 → **123** it (arm64 41 → 65, *issue*) | FERAL 0.21 path lands on `x = 0` of `|x|²·|x|^0.85` terms 12 times instead of 2; each is a non-finite Hessian → restoration entry; `5fc8c797` only stops the third from ending the solve | **partial** — (A) the Hessian is an evaluator artefact (`0·∞` from the product rule; the true second derivative of `|x|^2.85` at 0 is 0), fixable in `pounce-nl`; (B) R2 may move it; re-baseline after R2 | todo |
| **R4** | minima `circle_parabola_t0p5` mlsl | found 2 of 2 → **1** | #993 `91b6dcda`: `MLSL_GAMMA` 2 → 0.5; at 0.5 the extra launches re-find the first minimum and the stagnation counter (`patience 8`) stops the search | **partial** — no single gamma wins at seed 0 (table in R4); needs a 10-seed measurement or a structural stagnation rule; non-blocking | todo |
| **B1** | benchmark `qcqp1000-2c` (mittelmann) | 99 it / 28 s → 163 it / 112 s (*comment*) | pounce reuses the symbolic ordering FERAL chose for the least-squares multiplier-init matrix (W = 0, D = I) for every later KKT; `invalidate_symbolic_cache()` after the init factor halves factor time (*comment*) | **yes** for the code change; **measurement blocked** here (no benchmark corpus on this machine); a branch `perf/feral-symbolic-after-init` is said to be in progress — check `origin` first | todo |
| B2 | benchmark `brainpc0/1` (vanderbei) | 58 it / 1.1 s → 1073 + 52 it / 75 s; 185 + 24 → 2117 + 24 (*comment*) | after a now-correct inertia at one iterate, pounce's restoration takes 811 of 1073 iterations (*comment*) | **unknown** — investigate only after B1 lands (B1 changes the trajectory) and only with the corpus | blocked |
| B3 | benchmark `qcqp1500-1c` | Optimal → `Maximum_CpuTime_Exceeded` at 300 s (*issue*) | presumably B1 (same family); **unverified** | follows B1 | blocked |
| B4 | benchmark `cresc132` Optimal → Infeasible (Ipopt agrees); `ssebnln` Optimal → Acceptable | (*issue*) | FERAL 0.21 trajectory | **no / out of scope** (§5) | n/a |
| X1 | exact `deb7` arm64 Optimal/147 → Acceptable/180 | (*issue*) | FERAL 0.21 NEON trajectory | **not recoverable here** (x86 improves 143 → 114) (§5) | n/a |
| X2 | lbfgs `dallass` arm64 Optimal/224 → ErrorInStep/308 | (*issue*); x86 fails on both versions | FERAL 0.21 trajectory | **not recoverable here**; the x86 lbfgs failure predates 0.13.0 (§5) | n/a |
| X3 | lbfgs `pooling_rt2stp` arm64 −4391.83 → −3273.95 | (*issue*); x86 failing → solved | bistable nonconvex model | **no** — both are certified optima (§5) | n/a |
| F1 | `issue_592` pin `< 500` sits 8 above x86's 492 | — | R2 | resolved by R2; **do not raise the pin** | with R2 |

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
