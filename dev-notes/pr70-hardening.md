# PR #70 Hardening — Loop-Driven Verification Tracker

This file is the **state** for the PR #70 hardening loop. Plan:
`~/.claude/plans/woolly-launching-parnas.md`.

## Loop prompt (`/loop`)

> Work the **first unchecked** item below. Do only that one item end-to-end,
> update its section (Findings + checkbox), commit, then stop. Do not start the
> next item.

## Per-iteration protocol

1. **Select** the first `- [ ]` item; re-confirm scope from the plan.
2. **Implement** the named tests, reusing the oracle patterns below.
3. **Run** the item's command. Triage: test bug → fix test; real defect → fix if
   small & obviously correct, else record under Findings with a minimal repro +
   severity. Never paper over a wrong-answer defect.
4. **Record** Findings (tests added, pass/fail, defects, follow-ups). Flip
   `[ ]`→`[x]` only when Done criteria hold.
5. **Commit** one per item: `test(pr70): <item> — <result>` (with the required
   `Co-Authored-By` trailer; never `--no-verify`). Stop.

## Reusable oracle patterns (in-repo)

- **vs-NLP cross-check**: `crates/pounce-cli/tests/{cblib_vs_nlp,exp_cone_vs_nlp,qp_vs_nlp_iterations}.rs`
- **Known optima**: `crates/pounce-qp/tests/mm_published_optima.rs`, `crates/pounce-convex/tests/qp_known_optima.rs`
- **Routing unit**: `crates/pounce-cli/tests/dispatch_routing.rs` + `#[cfg(test)]` in `dispatch.rs`; fixtures `crates/pounce-cli/tests/fixtures/*.nl`
- **External validation**: `benchmarks/scripts/compare_pounce_clarabel.py`
- **`--json-output` schema**: `solution.status`, `statistics.{final_objective,iteration_count,total_wallclock_time_secs}`

## Baseline (captured at bootstrap)

- `cargo test --workspace`: **GREEN** — true exit 0, **1649 passed, 0 failed**
  (confirmed on a clean re-run, not piped through `tail`).
- Clarabel comparison (Item B input) — **full suite**, outputs in
  `benchmarks/clarabel_compare.md` + `clarabel_compare_{lp,qp}.json`:
  - **LP**: 467 problems, 419 both-solved, **412/419 agree** (reldiff < 1e-4).
    3 pounce-only, 28 clarabel-only. POUNCE non-solves incl. InternalError
    (greenbea, ch, nemsemm1, nemsemm2), several TimeOut/MaxIter.
  - **QP**: 138 problems, 114 both-solved, **110/114 agree**. 3 pounce-only,
    19 clarabel-only. `VALUES` failed with `ParseError:JSONDecodeError` on the
    pounce side — likely a JSON-report/harness bug, flag in B or G.
  - **Objective disagreements to triage in Item B** (both solved, reldiff ≥ 1e-4):
    - Near-zero-objective artifacts (both ≈ 0, published optimum 0 — almost
      certainly fine): LP `model11`; QP `S268`/`HS268`.
    - **Genuine, investigate**: QP `YAO` (pounce 197.70 vs clarabel 91.02,
      reldiff 0.54); LP `capri` (2625.0 vs 2690.0, reldiff 0.024).
    - Borderline (≈1–4e-4, likely tolerance): LP `lpl2`, `pltexpa3_16`,
      `pltexpa4_6`, `large001`, `fxm3_16`; QP `UBH1`.
  - POUNCE correct live; stored `benchmarks/lp/pounce.json` is STALE
    (adlittle/stocfor1 wrong) — regenerate in B.

---

## [ ] A1 — Routing classification (HIGHEST RISK)
- Scope: `classify_problem` must never under-classify nonconvex as convex.
  Cover: indefinite Hessian → `NonconvexQp`; near-PSD boundary at `±PSD_TOL`
  (1e-9) resolves conservatively (inconclusive → NLP); maximize-of-convex
  (concave) → nonconvex; zero Hessian → `Lp`; pure linear; genuinely convex
  QP/QCQP still convex (no false fallback).
- Files: `crates/pounce-cli/src/dispatch.rs` (PSD test ~L564+),
  `crates/pounce-cli/tests/dispatch_routing.rs`, fixtures `crates/pounce-cli/tests/fixtures/*.nl`.
- Run: `cargo test -p pounce-cli dispatch`
- Done: new cases green; any misclassification recorded as a Finding.
- Findings:

## [ ] A2 — Forced `solver_selection` mismatch must error, not mis-solve
- Scope: `qp-ipm`/`lp-ipm`/`qp-active-set` forced on a non-matching/nonconvex
  `.nl` returns a clear error (nonzero exit / error status), never a wrong
  "optimal." `auto` on the same routes safely (NLP/global).
- Files: `crates/pounce-cli/tests/qp_dispatch_end_to_end.rs`,
  `resolve_solver` mismatch arms in `dispatch.rs` (L298–320).
- Run: `cargo test -p pounce-cli`
- Done: mismatch cases assert error; green.
- Findings:

## [ ] B — Objective validation vs known optima + Clarabel
- Scope: netlib LP + Maros–Mészáros QP objectives from pounce match Clarabel /
  published optima within tol (rel < 1e-4); disagreements triaged. **Regenerate
  the stale `benchmarks/lp/pounce.json`** from live pounce. Conic/CBLIB covered
  via `cblib_vs_nlp`.
- Files: `benchmarks/scripts/compare_pounce_clarabel.py` (add `--check` mode +
  nonzero exit on disagreement), `benchmarks/lp/pounce.json` (regenerate),
  optionally `benchmarks/qp/pounce.json`.
- Run: `python3 benchmarks/scripts/compare_pounce_clarabel.py --class both`
- Done: all problems agree within tol or each disagreement is explained;
  `pounce.json` no longer stale.
- Findings:

## [ ] C — Status / edge-case honesty
- Scope: Infeasible, Unbounded, and limit cases (iteration/time/node) report the
  correct status — **never "optimal."** Edge inputs: empty constraints, fixed
  variable, free variable, single variable, zero-Hessian QP-as-LP.
- Files: `crates/pounce-convex/tests/infeasibility.rs` (+bounded_form.rs),
  `crates/pounce-convex/src/{ipm,hsde,hsde_nonsym}.rs`;
  `crates/pounce-global/tests/global.rs` + `bnb.rs` `GlobalStatus::{Infeasible,NodeLimit,TimeLimit}`.
- Run: `cargo test -p pounce-convex infeasib && cargo test -p pounce-global`
- Done: status assertions green for every edge case.
- Findings:

## [ ] D — Nonsymmetric cones & SDP (riskiest numerics)
- Scope: exp/power cones (`hsde_nonsym` path) and `psd`/`chordal` least
  battle-tested. Adversarial: ill-conditioned, near-cone-boundary, a few larger
  instances; validate via vs-NLP and/or known optima (geometric/entropy for exp,
  small SDPs for psd).
- Files: `crates/pounce-convex/src/cones/{exp,power,psd,chordal,nonsym}.rs`,
  `crates/pounce-convex/src/hsde_nonsym.rs`; tests alongside cone tests +
  `crates/pounce-cli/tests/exp_cone_vs_nlp.rs`.
- Run: `cargo test -p pounce-convex cone && cargo test -p pounce-cli exp_cone`
- Done: new adversarial cases green or defects logged.
- Findings:

## [ ] E — Global solver soundness
- Scope: (1) certified **lower bound always a valid global bound**; relaxations
  (αBB/RLT/OBBT/McCormick) are valid outer approximations; (2) **parallel ==
  serial** optimum; (3) node/time limits return best-incumbent with correct
  status.
- Files: `crates/pounce-global/src/{bnb,alphabb,rlt,obbt,envelope,relax,branching}.rs`,
  `crates/pounce-global/tests/global.rs`.
- Run: `cargo test -p pounce-global`
- Done: bound-validity + serial==parallel + limit-status tests green.
- Findings:

## [ ] F — Presolve round-trip (primal AND dual)
- Scope: presolve + postsolve recovers true primal and **dual** solution,
  including on heavily-reduced problems.
- Files: `crates/pounce-convex/src/presolve.rs`,
  `crates/pounce-convex/tests/presolve_roundtrip.rs` (+ presolve_reductions/
  forcing/conic/bound_tightening).
- Run: `cargo test -p pounce-convex presolve`
- Done: primal+dual recovery asserted; green.
- Findings:

## [ ] G — FFI / Python surface
- Scope: `minimize()` auto-routing picks the right solver; JAX differentiable-QP
  gradients match finite differences; `--json-output` schema uniform across all
  solver paths.
- Files: `python/pounce/{_route.py,qp.py,jax/_qp.py,global_opt.py,sos.py}`,
  `python/tests/test_{minimize_autoroute,qp,qp_jax,qp_sensitivity,socp,global,sos}.py`.
- Run: `pytest python/tests -q` (build the extension first per repo norm).
- Done: pytest green; gradient finite-diff check within tol.
- Findings:

## [ ] H — Hygiene (build / clippy / full suite)
- Scope: clean `cargo build` + `cargo clippy` across the feature matrix (fix the
  known `unused import: QpStatus` in
  `crates/pounce-qp/.../illconditioned_fallback.rs`); full `cargo test` +
  `pytest` green; no new warnings.
- Run: `cargo clippy --workspace --all-targets && cargo test --workspace`
- Done: zero warnings; both suites green.
- Findings:
