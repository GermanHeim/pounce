# Code-review 2026-06 — follow-ups (F1–F8) progress

Tracks the follow-ups raised in `code-review-2026-06-verification.md` (the
re-verification of the L1–L56 fix batch). Each entry: verification by running
code, a fail-first test where constructible, the fix, and the result.

| ID | Title | Sev | Status |
|----|-------|-----|--------|
| F1 | H3 duals off by `obj_scale_factor` | High | ✅ fixed |
| F2 | H1 inertia-shift δ + unbounded-QP false positive | Med-High | ⬜ todo |
| F3 | H11 fix dormant (no `get_variables_linearity`) | High | ⬜ todo |
| F4 | L7 watchdog `alpha_primal_max` — reopen | Medium | ⬜ todo |
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
