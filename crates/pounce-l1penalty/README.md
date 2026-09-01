# pounce-l1penalty

Thierry-Biegler (2020) ℓ₁-exact penalty-barrier TNLP wrapper for POUNCE.

Internal crate. Wraps a user [`TNLP`](../pounce-nlp) so that the IPM
solves the augmented problem

```
min   f(x) + ρ · 1ᵀ(p + n)
s.t.  c(x) − p + n = g_target,
      x_L ≤ x ≤ x_U,   p ≥ 0,   n ≥ 0
```

instead of the original. The augmented NLP automatically satisfies LICQ
on the slack variables `(p, n)`, which is the property that makes the
standard interior-point machinery (filter LS, inertia correction,
fraction-to-boundary) work on degenerate / MPCC-like cases that the
stock filter line search thrashes on.

## Status

Feature-complete through Phase 3.5. Available in the CLI as both an
explicit mode (`l1_exact_penalty_barrier=yes`) and an auto-fallback
(`l1_fallback_on_restoration_failure=yes`) — see the
[CLI README](../pounce-cli/README.md#degenerate-mpcc-nlps-l-exact-penalty-barrier-wrapper)
for usage. The wrapper carries:

- TNLP wrapper with full solution back-projection and multiplier
  recovery into the original variable space;
- Byrd-Nocedal-Waltz dynamic-ρ outer loop whose termination and honest
  infeasibility upgrade are judged on the **original** model's
  feasibility at the returned point, against the caller's own `tol` /
  `acceptable_tol` (gh#794). The slack sum `Σ(p + n)` steers ρ, which is
  what it is the right quantity for; it is not the constraint violation
  (that is `|pᵢ − nᵢ|` per row) and it is no longer what decides the
  verdict;
- original-space reporting that includes the constraint violation:
  `final_constr_viol` is the violation of the rows the caller declared,
  not the augmented problem's residual, which the slacks satisfy to
  machine precision by construction;
- opt-in auto-fallback on `Restoration_Failed`,
  `Infeasible_Problem_Detected`, `Solved_To_Acceptable_Level`,
  `Maximum_Iterations_Exceeded`, `Not_Enough_Degrees_Of_Freedom`.

The MPCC benchmark sweep this crate was long listed as needing now
exists: [`benchmarks/mpcc/`](../../benchmarks/mpcc/README.md), with the
gate report at
[`dev-notes/mpcc-gate0-report.md`](../../dev-notes/mpcc-gate0-report.md).
It is what found the reporting defect above. On that corpus the ℓ₁
routes are **not** the recommended one — see the report's route table.
Tracking: [pounce#10](https://github.com/jkitchin/pounce/issues/10),
[pounce#794](https://github.com/jkitchin/pounce/issues/794).

## Algorithmic reference

Thierry, D. & Biegler, L.T. (2020). *"The ℓ₁ Exact Penalty-Barrier
Phase for Degenerate Nonlinear Programming Problems in Ipopt"*,
IFAC-PapersOnLine.

ripopt 0.8.0's `src/l1_penalty_barrier_nlp.rs` (commit `7847bba9`) is
the canonical port source.

## License

EPL-2.0.
