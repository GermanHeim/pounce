# Code-review 2026-06 — follow-ups (F1–F8) progress

Tracks the follow-ups raised in `code-review-2026-06-verification.md` (the
re-verification of the L1–L56 fix batch). Each entry: verification by running
code, a fail-first test where constructible, the fix, and the result.

| ID | Title | Sev | Status |
|----|-------|-----|--------|
| F1 | H3 duals off by `obj_scale_factor` | High | ✅ fixed |
| F2 | H1 inertia-shift δ + unbounded-QP false positive | Med-High | ✅ fixed |
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
