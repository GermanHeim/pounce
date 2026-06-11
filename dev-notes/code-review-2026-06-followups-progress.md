# Code-review 2026-06 — follow-ups (F1–F8) progress

Tracks the follow-ups raised in `code-review-2026-06-verification.md` (the
re-verification of the L1–L56 fix batch). Each entry: verification by running
code, a fail-first test where constructible, the fix, and the result.

| ID | Title | Sev | Status |
|----|-------|-----|--------|
| F1 | H3 duals off by `obj_scale_factor` | High | ✅ fixed |
| F2 | H1 inertia-shift δ + unbounded-QP false positive | Med-High | ✅ fixed |
| F3 | H11 fix dormant (no `get_variables_linearity`) | High | ✅ fixed |
| F4 | L7 watchdog `alpha_primal_max` — reopen | Medium | ✅ fixed |
| F5 | L56 incomplete — session FFI unguarded | Medium | ⬜ todo |
| F6 | H12 no Phase-0 rollback on FBBT infeasibility | Medium | ⬜ todo |
| F7 | L10 MA57 grow paths unguarded | Low | ⬜ todo |
| F8 | M9 zero-fill in pounce-sensitivity `dense_to_vec` | Low | ⬜ todo |

---

## F1 detail — duals reported scaled by `obj_scale_factor` (High)

**Finding.** When gradient-based objective scaling triggers
(`‖∇f(x0)‖∞ > nlp_scaling_max_gradient`, default 100), the solution duals
(`lambda`, and also the bound duals `z_l`/`z_u`) were reported scaled by
`obj_scale_factor` instead of in the user's unscaled-Lagrangian convention
`∇f + λ·∇g + z = 0`.

**Root cause.** `OrigIpoptNlp` carries two parallel dual-lifting families:
- `pack_lambda_for_user` / `pack_z_l_for_user` / `pack_z_u_for_user` — apply
  `c_scale`/`d_scale` but **not** the `1/obj_scale_factor` division. These feed
  the *scaled* `eval_h` and are correct there.
- `finalize_solution_lambda` / `finalize_solution_z_l` / `finalize_solution_z_u`
  — apply scale **and** divide by `obj_scale_factor`. These are the correct
  *final-solution* convention (mirror of upstream
  `IpOrigIpoptNLP::FinalizeSolution`). They existed but had **zero callers**
  (dead code).

The solution hooks called the `pack_*` family:
- `crates/pounce-cli/src/main.rs:643` (`on_converged` → JSON / `.sol`),
- `crates/pounce-algorithm/src/application.rs:2215` (`finalize_via_orig_nlp`,
  the `finalize_solution` TNLP callback used by the Python bindings),
- `application.rs:2371` (`finalize_via_sqp`, the SQP analog),

so every dual came back scaled whenever scaling kicked in. Only `lambda` was
flagged in the review, but `z_l`/`z_u` shared the identical root cause via
`pack_z_*_for_user`.

**Verification by running code.** New fixture `dual_scaled.nl` =
`dual_order.nl` with the y-target moved 30 → 3000:
`min (x-3)^2 + (y-3000)^2  s.t.  x ≤ 2 (active),  y == 1`. At x0=(0,0)
`‖∇f‖∞ = 2·3000 = 6000`, so `obj_scale_factor = 100/6000 = 1/60`. Running the
release binary:

```
# before fix, default scaling:        lambda = [0.0333, 99.97]   (scaled, WRONG)
# nlp_scaling_method=none:            lambda = [2.0,   5998.0]   (true)
# after fix, default scaling:         lambda = [2.0,   5998.0]   (true, FIXED)
```

The pre-fix `[0.0333, 99.97]` is exactly `[2, 5998]/60` — confirming the missing
`obj_scale_factor` division. Regression: `dual_order.nl` (obj_scale=1) is
unchanged at `[2.0, 58.0]`.

**Fix.** Switch all six call sites in `finalize_via_orig_nlp` /
`finalize_via_sqp` (and the lambda site in `main.rs`) from the `pack_*` family
to the `finalize_solution_*` family. This fixes `lambda` (F1's explicit ask)
**and** the latent `z_l`/`z_u` bug consistently, and retires the dead code the
verifier flagged. The `pack_*` methods stay (still used by the `eval_h` path).

**Test.** `crates/pounce-cli/tests/json_report.rs::lambda_is_unscaled_by_obj_scale_factor_under_gradient_scaling`
— solves `dual_scaled.nl` with default scaling ON and asserts the unscaled
`lambda = [≈2, ≈5998]` (decisive guard `|lambda[1]| > 1000`; pre-fix it was
≈99.97). Fail-first demonstrated directly on the pre-fix binary (`[0.033, 99.97]`
fails both `>1000` and `≈5998`).

**Result.** `pounce-cli` full suite green (incl. 7 json_report tests);
`pounce-algorithm` 258 lib tests, `pounce-nlp` 39 lib tests green; fmt + clippy
(correctness/suspicious gate) clean.

---

## F2 detail — inertia-shift δ and unbounded-QP detection (Med-High)

The verifier raised two coupled defects around the QP inertia-control shift δ
in `crates/pounce-qp/src/solver.rs`:

- **N1 false positive** — the equality-only fast path (`solve_equality_only`)
  used a *magnitude heuristic* (`δ·‖x‖∞ > 1e-3·‖g‖∞`) to declare unboundedness.
  It could not distinguish a large-but-finite minimizer in a **curved**
  direction from a genuine blow-up along a **flat** descent ray, so a bounded
  singular QP like `H = diag(1e-6, 0)`, `g = (-1, 0)` (true minimizer
  `x₁ = 1e6`, obj `-5e5`) was wrongly reported `Unbounded`.
- **F2(a) δ discarded** — the general active-set loop (`solve_general`) and the
  opt-in Schur loop (`solve_general_schur`) threw away the δ returned by
  `factorize_with_inertia_control`. An unbounded QP carrying a general
  inequality (which routes off the equality-only path) just took unbounded full
  steps until `MaxIter` — never certifying the recession ray.

**Root cause.** Both defects are the same missing piece: a `δ > 0` solve is
consistent with *both* a bounded QP (the regularizer picks the min-norm point
along a flat, gradient-free direction) and an unbounded one, so the shift alone
proves nothing. Magnitude of `‖x‖` is not a discriminator either — a shallow but
genuinely curved bowl has a large finite minimizer.

**Fix — certified recession ray.** A QP `min ½xᵀHx + gᵀx s.t. Ax = b` is
unbounded below iff there is a direction `d` with `Hd = 0` (zero curvature; for
PSD H ⟺ `dᵀHd = 0`), `Ad = 0` (feasible), and `gᵀd < 0` (descent). New shared
helper `ray_is_unbounded_descent(h, g, dir)` checks the two intrinsic clauses
— **zero curvature** `dᵀHd ≈ 0` relative to `‖H‖`, and **strict descent**
`gᵀd < 0` — leaving feasibility of the ray to the caller:

- `solve_equality_only`: the saddle solve maintains `Ax = b`, so the candidate
  ray `d = x/‖x‖` has `Ad = b/‖x‖ → 0`; verified by an inline `‖Ad‖∞` guard,
  then `δ > 0 && feasible && ray_is_unbounded_descent` ⇒ `Unbounded`.
- `solve_general`: the ratio test having an **empty candidate list** certifies
  `+p` is feasible for every step length (and `p` lies in the active null
  space); gated additionally on `δ > 0` (captured from the factorize call).
- `solve_general_schur`: same empty-candidate certificate, but δ is hidden
  inside `SchurState`, so it relies on the curvature clause alone to reject
  curved (PD-reduced-Hessian) steps — `ray_is_unbounded_descent` returns
  `false` on any `dᵀHd ≉ 0`, so the unconditional check is safe.

Why the curvature clause is robust: as the inertia shift `δ → 0` the flat
descent component of `-g` is amplified by `1/δ`, so `d = x/‖x‖` converges to the
recession ray and `dᵀHd/‖H‖` shrinks like `O((δ/‖H‖)²)`; a curved minimizer
keeps `dᵀHd ≈ ‖H‖` (ratio `O(1)`). The `1e-3·‖H‖` threshold sits in that gap,
and ambiguous near-floor-curvature cases fall on the conservative side
(reported bounded, never falsely unbounded).

**Verification by running code (fail-first).** Three new analytical tests in
`crates/pounce-qp/src/tests/analytical.rs`:

```
n1_bounded_singular_qp_is_not_falsely_unbounded   pre-fix: Unbounded x=[990099,0]  → Optimal (FIXED)
n1_partial_curvature_descent_ray_is_unbounded     pre/post: Unbounded              (guard)
f2_general_active_set_detects_unbounded_ray        pre-fix: MaxIter x=[0,2.0e10]   → Unbounded (FIXED)
                                                   (+ same problem on the Schur path)
```

The N1 case (`Unbounded` pre-fix) and the active-set case (`MaxIter` pre-fix,
`x₂` ran to `2.01e10`) both fail before the fix and pass after.

**Result.** `pounce-qp` full suite green — 83 lib tests + integration
(`mm_published_optima` real Maros-Meszaros QPs confirm bounded problems are not
falsely flagged). fmt + clippy (correctness/suspicious gate) clean.

---

## F3 detail — H11 presolve safeguard was dormant (High)

**Finding.** The L-batch fix H11 added a guard in the Phase-0 presolve
auxiliary-elimination pass (`pounce-presolve/src/auxiliary.rs:270-281`): it
unions every variable upstream tagged `NonLinear` into the objective support,
so a variable that is nonlinear in the objective but merely *zero-gradient at
the single probe point* (the canonical case `f = (x − x0)²` warm-started at
`x0`, where `∇f(x0) = 0`) is not mis-classified objective-free and eliminated.
The guard reads the tags from `TNLP::get_variables_linearity`. But the only
implementation of that trait method was the **default stub**
(`pounce-nlp/src/tnlp.rs:223`, `-> false`, slice untouched), so on every
production path `have_var_linearity` was `false`, `probe.var_linearity` was
`None`, and the H11 union never ran. The safeguard was dead code: its unit
tests (`auxiliary.rs:1002-1065`) pass *tags by hand* and so never exercised the
real (untagged) production path.

**Root cause.** `NlTnlp` (the `.nl`-backed TNLP, the production entry point for
every AMPL/`pounce`-CLI solve) implemented `get_constraints_linearity` but not
`get_variables_linearity`, falling through to the `false` default. The tape
already knows exactly which variables are nonlinear, so the information was
available — just never surfaced.

**Fix.** Implement `get_variables_linearity` on `NlTnlp`
(`crates/pounce-nl/src/nl_reader.rs`, beside `get_constraints_linearity`) with
the upstream **global** semantics: a variable is `NonLinear` iff it appears in
the nonlinear part of the objective or of any constraint, else `Linear`. The
parsed `.nl` already separates each row into a linear coefficient list and a
nonlinear `Expr`, so the nonlinear set is exactly the structural union of the
existing `collect_vars` walk over `obj_nonlinear` and every `con_nonlinear`
row. This honors the documented contract and engages H11 on every solve. (The
alternative the verifier floated — making the *untagged* case conservative in
the presolve — was rejected: it would broadly regress legitimate
auxiliary-elimination for every TNLP that genuinely returns no tags.)

**Verification by running code (fail-first).** New unit test
`crates/pounce-nl/src/nl_reader.rs::variables_linearity_tags_obj_nonlinear_vs_linear_vars`
builds `min (x0 − 1)² + 3·x1` (x0 nonlinear in the objective, x1 only in the
linear part) and asserts `get_variables_linearity` returns `true` with
`[NonLinear, Linear]`. Demonstrated fail-first by temporarily reverting the
body to the stub (`-> false`, slice untouched): the test panics
`get_variables_linearity must report it filled the slice`. Post-fix it passes.

**Result.** `pounce-nl` full suite green (89 lib tests, incl. the new one);
`pounce-presolve` 226 lib tests green (the H11 path now active on the real
no-hand-tags route, no regression). fmt + clippy (correctness/suspicious gate)
clean.

---

## F4 detail — watchdog StopWatchDog retry reused the failed direction's FTB cap (Medium)

**Finding.** The filter line-search watchdog snapshots an iterate + search
direction; after `watchdog_trial_iter_max` (default 3) failed outer iterations
it reverts ("StopWatchDog") to that snapshot and re-runs the alpha-loop on the
saved `delta` with `skip_first = true`. Upstream
`IpBacktrackingLineSearch::FindAcceptableTrialPoint` (stable/3.14) recomputes
`alpha_primal_max` / `alpha_dual_max` from `actual_delta_` — which
`StopWatchDog` has reverted to the snapshot — so the entire body re-runs on the
recovered direction, fraction-to-the-boundary (FTB) caps included. pounce's
`handle_watchdog_failure` (`backtracking.rs`) instead **reused the
`alpha_init` / `alpha_dual` it had been handed** — the caps computed for the
*pre-revert* iterate and the *now-abandoned* failed direction — and fed them to
`run_alpha_loop` over `snap_delta`.

**Root cause.** The retry's first trial length is `cap × alpha_red_factor`. With
a stale cap the first trial is mis-sized for `snap_delta` at the reverted
iterate: if the failed cap is looser than the snapshot's true FTB limit, the
trial overshoots the boundary (negative slack / bound multiplier → non-finite
barrier objective, wasted backtracking trials); if tighter, it needlessly
shortens an otherwise-feasible step. Either way the recovered search starts from
the wrong place.

**Fix.** In the StopWatchDog branch, after `set_curr(snap)` (so the CQ now
reflects the snapshot), recompute both caps from `snap_delta` at the reverted
iterate and stop reusing the handed-in values:

```rust
let tau = data.borrow().curr_tau;
let (alpha_primal_retry, alpha_dual_retry) = {
    let cq_ref = cq.borrow();
    (1.0_f64.min(cq_ref.aff_step_alpha_primal_max(&snap_delta, tau)),
     1.0_f64.min(cq_ref.aff_step_alpha_dual_max(&snap_delta, tau)))
};
```

clamped to the full step `1.0` (the default `alpha_max`, matching the main
path's `alpha_init.min(alpha_primal_max)` at `ipopt_alg.rs:1045`;
`BacktrackingLineSearch` carries no `alpha_max` field). The now-unused
`alpha_init` parameter was removed from `handle_watchdog_failure`'s signature
and its call site in `run_filter_line_search`. Both the primal **and** dual cap
are recomputed: the verification doc named only `alpha_primal_max`, but the dual
cap reuse is the identical staleness bug — applying the failed direction's dual
cap to `snap_delta`'s bound-multiplier components at the reverted iterate can
violate FTB on the `z`'s just as readily.

**Verification by running code (fail-first).** New focused unit test
`crates/pounce-algorithm/src/line_search/backtracking.rs::stop_watchdog_retry_recomputes_ftb_cap_from_snapshot_direction`.
The watchdog/backtracking path had no existing unit harness, so the test builds
one: an `F4MockNlp` (n=1, `x[0] ≥ 0`, `f = x[0]²`, no constraints) and a
`RecordingAcceptor` that records the *first* trial alpha offered and always
accepts. It arms the watchdog (snapshot `x = 2` ⇒ slack 2, `z_L = 0.5`,
`τ = 1`; `snap_delta` `Δx = -4` ⇒ true FTB cap `τ·s/|Δx| = 1·2/4 = 0.5`) with
`watchdog_trial_iter = watchdog_trial_iter_max`, then calls
`handle_watchdog_failure`. It asserts `Outcome::Accepted` and that the recorded
first retry alpha is `0.25` (= recomputed cap `0.5` × `alpha_red_factor` `0.5`).
Fail-first was demonstrated by temporarily hard-coding the call to pass a
*failed-direction* cap of `0.6` in place of `alpha_primal_retry`: the recorded
first alpha became `0.3` (≠ asserted `0.25`) — the test fails on the pre-fix
behavior and passes after. Numbers were chosen so both the pre-fix (`0.3`,
`x = 0.8`) and post-fix (`0.25`) first trials are *feasible*, avoiding the trap
where an infeasible pre-fix trial backtracks and coincidentally lands on `0.25`.

**Result.** `pounce-algorithm` 258 lib tests + the new one green (one unrelated
pre-existing flake, `iter_dump::tests::header_writes_magic_and_version`, races on
a shared `ENV_DUMP_PATH` env var under parallel execution and passes in
isolation — not touched by this change). fmt + clippy (correctness/suspicious
gate) clean.
