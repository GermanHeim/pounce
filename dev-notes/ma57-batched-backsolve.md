# MA57's batched back-substitution is not bit-identical, and what that costs

Recorded because it was measured once, on a model nobody else has, and the
alternative to writing it down is rediscovering it. It is finding **(c)** of
@srikanth-gm's review of [#809](https://github.com/jkitchin/pounce/pull/809),
and it is the reason `ma57_batched_backsolve` is an opt-in option rather than
a default `true`.

## The claim

`ma57cd_` with `nrhs > 1` does not reproduce the per-column result
bit-for-bit. The difference is about one ulp, and one ulp is enough to move
the trajectory.

The evidence is a single-variable A/B: one binary, one model, one host, one
toolchain, with a three-line addition as the only difference —

```rust
fn multi_solve_matches_single_solve(&self, _nrhs: usize) -> bool { true }
```

The two arms diverge at iteration 20, in the last printed digit of the
objective:

```text
  gate closed   20  9.7709874e+01 3.29e+00 5.87e+01  -1.5 3.71e+00  - 2.75e-01 2.21e-01f 1
  gate open     20  9.7709873e+01 3.29e+00 5.87e+01  -1.5 3.71e+00  - 2.75e-01 2.21e-01f 1
```

and finish at different iteration counts and different objectives. Across the
four arms of that experiment: 201 / 206 / 173 / 160 iterations.

This is exactly the property `SparseSymLinearSolverInterface::multi_solve_matches_single_solve`
was introduced to gate (gh#729), and the trait's default `false` is doing real
work for MA57. On a nonconvex problem a perturbation this size can select a
different local optimum: gh#729 is MA57 taking `pooling_rt2stp` to an
objective 25% worse while still reporting `Optimal Solution Found`.

## The model

gh#809's review model: a 118276-row KKT system solved through the
limited-memory quasi-Newton path, so `LowRankAugSystemSolver` is live and the
SMW correction block is what offers the batch. The widths it offers are small
— instrumentation on that run recorded `n_cols = 2` on the batched
evaluations. Five binaries, two interleaved replicates per arm, arms adjacent
in time; every arm was deterministic, with both replicates producing identical
trajectories and objectives to the last digit.

## The trap

**Do not quote wall-clock across the gate.** The figures from that experiment
were Ipopt 326.0 s, gate-closed POUNCE 356.5 s, gate-open POUNCE 269.2 s. The
gate-open arm converging in 160 iterations against 206 is a one-ulp
perturbation landing favourably on a chaotic trajectory. It is not work
removed, and it could as easily have gone the other way.

Only per-iteration figures compare, and even a per-iteration comparison across
the gate is not single-variable at the trajectory level — the two arms walk
different paths. That is a limitation of the measurement, not a defect in it,
and it is why the numbers below are stated as ratios that reproduce
independently on feral rather than as a headline speed-up.

## What opening the gate actually buys

Per iteration, on that model, gate-open with and without #809's merged call:

| per iteration (s) | one call per column | merged call | Δ |
|---|---:|---:|---:|
| numeric factorization | 0.1443 | 0.1711 | +18.5% |
| back-solve | 0.1256 | 0.0854 | **−32.0%** |
| linear algebra total | 0.2796 | 0.2678 | −4.2% |
| solver internal | 0.3755 | 0.3544 | −5.6% |

Against per-arm replicate spreads of 3.4–5.3%, the −4.2% net is "probably
real, small". The same ratios on feral's `laptime` are −30.8% back-solve and
+16.4% factorization, so **MA57's fixed-traversal to per-RHS cost ratio is not
materially different from feral's** — an argument for the batch that rested on
MA57 having a much larger fixed cost would not survive this.

The case that does survive is the comparison against Ipopt/MA57, on an
equal-iteration basis:

| per iteration (s) | Ipopt/MA57 | POUNCE, gate closed | POUNCE, gate open + #809 |
|---|---:|---:|---:|
| numeric factorization | 0.1344 | 0.1467 (+9.1%) | 0.1702 (+26.6%) |
| back-solve | 0.1055 | 0.1594 (**+51.0%**) | 0.0856 (**−18.9%**) |
| linear algebra total | 0.2465 | 0.3142 (+27.5%) | 0.2662 (+8.0%) |
| non-linear-algebra | 0.0368 | 0.0943 (+156%) | 0.0871 (+137%) |
| solver internal | 0.2833 | 0.4085 (+44.2%) | 0.3533 (+24.7%) |

The back-solve row inverts, from 51% worse than Ipopt to 19% better. What
remains is not linear algebra: of the residual 0.0700 s/iter gap, 0.0503 —
72% — is the non-linear-algebra row.

## Why the option has no width ceiling

feral answers `multi_solve_matches_single_solve` with a **measurement**:
below its BLAS-3 threshold each column runs the same rank-1 cascade a
single-RHS solve would, so `nrhs <= FERAL_BITWISE_MULTI_SOLVE_MAX_NRHS` is
true and `multi_solve_bitwise_matches_single_solve_at_the_documented_ceiling`
re-derives it on every run.

MA57 cannot be given the same treatment here. CoinHSL is licensed and cannot
be linked in CI, so any width ceiling on this backend would be a constant no
test in this repository could re-derive — which is the shape of defect
`FERAL_BITWISE_MULTI_SOLVE_MAX_NRHS`'s own doc comment warns about, and which
`dev-notes/trajectory-regressions-and-the-fixture-sweep.md` is the post-mortem
of. So `ma57_batched_backsolve` is a flat permission: it applies at every
width, and the user granting it is accepting the perturbation, not being told
there is none.

If someone with an HSL licence establishes a width below which `ma57cd_`
genuinely reproduces the per-column result on representative KKT systems, that
turns the permission into a ceiling of feral's kind.
`crates/pounce-hsl/tests/ma57_batched_backsolve.rs` is where the probe lives;
its fixture is a 500-row band, which is deliberately *not* claimed as evidence
either way about the 118276-row result above.

## Status

- `ma57_batched_backsolve` exists and defaults to `no`. Nothing about an
  existing MA57 run changes.
- #809's merged call is what makes the batch worth turning on rather than
  merely possible; the two are independent changes and land separately.
- Nobody has measured what the option does across `benchmarks/` or
  `scripts/sweep-fixtures.sh`, because neither harness can select MA57 without
  CoinHSL. A user who turns it on is on the trajectory-change side of
  `CLAUDE.md`'s rule and should expect iteration counts to move.
