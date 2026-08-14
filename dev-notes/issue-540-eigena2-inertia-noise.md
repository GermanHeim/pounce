# gh #540 — `eigena2`: the inertia test read from a factorization that could not answer it

Pre-fix measurements below were taken on `270a0502`, the commit the issue
quotes; post-fix ones on this branch (they are reproducible on either by
setting `feral_inertia_pivot_floor` — `0` is the pre-fix routing). All
single-threaded
(`OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
RAYON_NUM_THREADS=1`), FERAL backend, on the reporter's own `eigena2.nl`
(attached to the issue; now checked in as
`crates/pounce-cli/tests/fixtures/eigena2.nl`).

## The reported symptom, reproduced

```
  26  8.2500000e+01 2.44e-15 2.93e-07   -8.6 4.52e-08   -0.3 1.00e+00 1.00e+00h  1
  27  8.2500000e+01 1.11e-15 4.63e-07   -8.6 2.23e-08   -0.8 1.00e+00 1.00e+00h  1
  28  8.2500000e+01 4.44e-16 2.24e-07   -8.6 8.20e-09    1.4 1.00e+00 1.00e+00h  1
  29  8.2500000e+01 8.88e-16 7.73e-08   -8.6 8.47e-09    1.0 1.00e+00 1.00e+00h  1
EXIT: Solved To Acceptable Level.   dual inf 3.3144912958031115e-07
```

Bit-identical to the issue. `POUNCE_DBG_PERT=1` shows the ladder at the
iteration that breaks it: one δ_x step down the decrement ladder
(`1.60e-1 → 5.35e-2`), then three ×8 escalations
(`5.35e-2 → 4.28e-1 → 3.42 → 27.4`) before a factorization is accepted.
`27.4 = 10^1.44` is the `lg(rg) = 1.4` in the table.

## The δ_w update rule is not the defect

`get_deltas_for_wrong_inertia` is an exact port; every value in that ladder is
what upstream's rule produces from the state it was in. The issue's reading —
"the escalation logic throws away the progress after a single non-monotone
iteration" — describes the *consequence*. The cause is upstream of it: the
loop was fed an inertia count that carries no information.

## What the factorizations were actually reporting

`--dump kkt:all` on the run, then an exact `numpy.linalg.eigvalsh` on each
dumped triplet:

(`iter` is the dump's own numbering, `pounce-dump-*/iter_NNN/`; the last four
rows are the four factorizations of the single iteration that breaks the run.)

| iter | δ_x     | feral `n_neg` | exact `n_neg` | `\|λ\|` below 1e-10 | min `\|λ\|` |
|-----:|---------|--------------:|--------------:|-------------------:|-----------:|
| 020  | 0       | 64            | 61            | 51                 | 5.1e-16    |
| 022  | 0       | 60            | 47            | 53                 | 4.7e-18    |
| 024  | 0       | 56            | 40            | 54                 | 9.2e-19    |
| 026  | 0       | 64            | 52            | 54                 | 6.6e-19    |
| 026  | 5.35e-2 | 58            | 45            | 45                 | 1.1e-18    |
| 026  | 4.28e-1 | 62            | 62            | 45                 | 7.1e-11    |
| 026  | 27.4    | 55            | 55            | 45                 | 7.1e-11    |

The expected count is 55 (one per equality constraint) on a `165 × 165` KKT.
Down the whole tail, **45–54 eigenvalues sit at `1e-16` or below** on a matrix
of norm 240 — the system is singular to working precision, and neither number
in the middle two columns means anything. Note where the two columns start
agreeing: the last two rows, which are the factorizations taken *after* `δ_c`
was applied.

Cause: the constraint Jacobian degenerates as the iterate converges. Its
singular values at iteration 27 are ten at `O(1)` and forty-five at `~1e-8`,
halving every iteration (`4.26e-7` at 19, `2.13e-7` at 20, `1e-8` at 27).
`eigena2` is degenerate at its solution; LICQ fails there. This is a property
of the model, not of the scaling — the same collapse happens under
`nlp_scaling_method=none`.

Consequence: with `δ_c = 0` the (2,2) block is exactly zero and those 45
near-dependent constraint directions produce no pivot at all. With
`δ_c = 1e-8·μ^0.25 = 7.07e-11` each of them contributes exactly one negative
pivot of magnitude `δ_c`, the count becomes 55, and the smallest pivot rises
from `~1e-16` to `5.8e-9`.

## Why `δ_c` was never reached

Three routes to `δ_c` exist, and all three were closed:

1. **`jac_degenerate = Degenerate`** — upstream's designed answer; once
   latched, `ConsiderNewSystem` seeds `δ_c` every iteration. But the
   degeneracy probe only runs while a flag is `NotYetDetermined`, and both
   flags latch to `NotDegenerate` on the first successful unperturbed
   factorization (iteration 0, where the Jacobian is full rank). A Jacobian
   that degenerates only near the solution can never be detected. Upstream has
   the same gap.
2. **`residual_ratio_singular` → `pretend_singular`** — iterative refinement
   converges here despite the singularity (the right-hand side happens to lie
   in the range), so the residual ratio stays under `residual_ratio_max` and
   this never arms.
3. **The linear solver reporting `Singular`** — feral's `ZeroPivotAction::
   ForceAccept` completes the factorization and the `feral_singular_pivot_
   floor` (`1e-20`, the MA57 `CNTL(2)` value) is far below the `~1e-16` pivots
   here. Worse, that check ran *after* the inertia comparison, so on a matrix
   that is both singular and reports a wrong count — which is every one of
   them — the `WrongInertia` return preempted it. The check could only fire on
   factorizations whose inertia was already correct, i.e. exactly where the
   caller does not need it.

Upstream's MA27 / MA57 / MUMPS interfaces all test singularity before
comparing the count (`IpMumpsSolverInterface.cpp`: `if (error == -10) return
SYMSOLVER_SINGULAR;` sits above `negevals_ = ...`), and MUMPS's null-pivot
threshold is relative to `‖A_pre‖` rather than absolute. Ipopt-with-MUMPS
therefore takes route 3 on this model and pounce did not.

## The fix

`feral_inertia_pivot_floor` (default `1e-12`; **superseded by the
dimension-aware `n · eps` default in gh#592** — see
`issue-592-restart-non-idempotence.md`): when the negative-eigenvalue
count *already disagrees* with what the IPM asked for, and the smallest
accepted pivot is under this floor, report `Singular` instead of
`WrongInertia`.

It is deliberately a second, higher floor rather than a raise of
`feral_singular_pivot_floor`, and the distinction matters:

- `feral_singular_pivot_floor` says "this factor is unusable", and applies to
  factors that were otherwise going to *succeed*. Raising it can turn a good
  factorization into a failure.
- `feral_inertia_pivot_floor` says "this factor cannot measure inertia", and
  is consulted only on factors that were already going to be rejected. It
  cannot cost a usable factorization — it only chooses which perturbation the
  rejection is repaired with.

That asymmetry is what makes the default safe to change without the benchmark
corpus in hand. It is pinned by
`well_conditioned_inertia_mismatch_still_reports_wrong_inertia`: an SPD `2×2`
with a mismatching demand still returns `WrongInertia`, so the ordinary
negative-curvature signal the δ_w ladder exists to answer is untouched.

### Choosing `1e-12`

feral's pivots are reported in its equilibrated space, so the relevant
question is where a pivot of an `O(1)` matrix stops carrying a reliable sign:
`n·eps`, which spans `2.2e-15` at `n = 10` to `2.2e-10` at `n = 10^6`. `1e-12`
sits in the middle. The measured sensitivity on `eigena2` is flat across that
whole band and well beyond it:

| floor  | iters | dual inf  | status     |
|--------|------:|-----------|------------|
| 0      | 29    | 3.31e-07  | Acceptable |
| 1e-16  | 29    | 3.31e-07  | Acceptable |
| 1e-15  | 27    | 1.08e-09  | Optimal    |
| 1e-14  | 26    | 5.72e-09  | Optimal    |
| 1e-13  | 26    | 3.98e-09  | Optimal    |
| 1e-12  | 27    | 5.73e-10  | Optimal    |
| 1e-10  | 32    | 8.85e-09  | Optimal    |

### The probe assertion

Routing a wrong-inertia verdict to `PerturbForSingularity` means that call can
now arrive with `δ_x` already off zero, which trips upstream's
`DBG_ASSERT(delta_x_curr_ == 0. && ...)` in the `DxEq0` arms. The reachable
sequence is: `Singular` at `δ_x = 0` (probe raises `δ_c`, moves to
`DcGt0DxEq0`, leaves `jac_degenerate` undetermined) → `WrongInertia` (`δ_x`
leaves zero; `finalize_test` resolves only the Hessian flag) → `Singular`
again, now at `δ_x > 0` under a `DxEq0` status. On `270a0502` that panics at
`pd_perturbation.rs:293`; it is pinned by
`a_second_singular_verdict_from_the_ladder_does_not_trip_the_probe`.

Those `DBG_ASSERT`s are assumptions, not invariants — MUMPS `INFO(1) = -10`
and MA27 `IFLAG = 3` can both fire from any rung of the ladder, so upstream
can reach them too. `perturb_for_singular` now checks whether the probe's
precondition
still holds; when it does not, the probe is abandoned (`test_status =
NoTest`) and the determined-state path runs, which asserts nothing and does
the right thing from wherever the perturbations are: raise `δ_c` if it is
still zero, else take a `δ_x` step. The rung already paid for is kept.

## Result

```
  24  8.2500000e+01 3.81e-13 2.90e-06   -8.6 6.68e-07    0.6 1.00e+00 1.00e+00h  1
  25  8.2500000e+01 5.31e-14 3.89e-07   -8.6 2.69e-07    0.2 1.00e+00 1.00e+00h  1
  26  8.2500000e+01 1.11e-15 1.90e-08   -8.6 3.95e-08   -0.3 1.00e+00 1.00e+00h  1
  27  8.2500000e+01 2.22e-16 3.18e-10   -9.0 1.98e-09   -0.8 1.00e+00 1.00e+00h  1
EXIT: Optimal Solution Found.   dual inf 5.7300837831461680e-10
```

The `lg(rg)` ladder now walks `2.1 → 1.6 → 1.1 → 0.6 → 0.2 → -0.3 → -0.8`
without the re-escalation, matching the shape of Ipopt's
`2.0 → … → -1.4`, and the tail is superlinear again
(`3.89e-7 → 1.90e-8 → 3.18e-10`).

## Not settled here

- **`eigenb2` (gh #541)** is the sister issue and was not investigated. It is
  described as the opposite symptom — *no* regularization where Ipopt applies
  `10^2.9` — and this note does not claim it shares a root cause. Whether the
  same rank-deficiency mechanism is behind it is an open question; the KKT
  dump + exact-eigenvalue procedure above is the way to answer it.
- **Corpus-wide effect.** The benchmark archive is gitignored and was not
  available in the environment this was done in, so the only regression
  evidence is the in-tree `cargo test` suite (green) plus the
  cannot-lose-a-factorization argument above. A benchmark sweep before release
  is still worth running.
- **Route 1 (`jac_degenerate` re-arming)** remains closed, in pounce and in
  upstream. A Jacobian that degenerates only near the solution is still never
  latched as degenerate; `δ_c` is now reached reactively, one factorization
  per iteration later than a latched flag would.
