# `qcqp_columns_wellcond.nl` / `qcqp_columns_illcond.nl` (gh #703)

**The same convex QCQP, twice, in two coordinate systems.** Both were
written by

```sh
scripts/gen-qcqp-nl.py --n 24 --quad-rows 1 --quad-density 0.25 \
    --linear-rows 10 --eqns 3 --linear-nnz 5 --seed 11 \
    --column-scale <S> --out <file>
```

with `S = 1` for the well-conditioned one and `S = 1e8` for its twin.
`--column-scale S` applies the exact substitution `x_j -> x_j / c_j` with
`c_j` spanning `[S^-0.5, S^0.5]`: every function value, the objective and
every right-hand side are unchanged (the two files' `r` sections agree to
the last ulp), and so is the optimum. Only the *conditioning* moves — the
columns of `Q` and of the linear block span eight orders of magnitude in
the second file.

They exist as a pair because a scaling scheme is much easier to trust when
the right answer is known in advance rather than merely "fewer iterations".
On the NLP path, against the shared optimum `-3.642102538232e+02`:

| file | `nlp_scaling_method` | iters | objective |
|---|---|---:|---|
| wellcond | `gradient-based` | 10 | `-3.642102538232e+02` |
| wellcond | `curvature-based` | 14 | `-3.642102538232e+02` |
| illcond | `gradient-based` | 27 | `-3.642102163442e+02` |
| illcond | `curvature-based` | **14** | **`-3.642102538232e+02`** |

`gradient-based` loses seven digits on the ill-conditioned twin;
`curvature-based` reproduces the well-conditioned answer to **one ulp** in
the same iteration count, which is the joint variable scaling recovering
`c`. One ulp rather than bit equality because `D` recovers `c` only up to
how the Ruiz sweep's factors round; the test pins the measured distance.

`crates/pounce-cli/tests/curvature_scaling.rs` pins that, with
`solver_selection=nlp` — see "Why the gh #703 tests still pin
`solver_selection=nlp`" below. At this size both models clear the conic
guards, so their *default* route is the SOCP driver, and that route is
pinned separately.

## Why one quadratic row and not two

`quad_evaluator_differential.rs` bounds how far the constant-matrix
evaluator may drift from the AD tape it replaces. It carries two numbers,
and only one of them is a guarantee:

* `WORST_OBSERVED_HESS_REL` — the largest *relative* deviation any Hessian
  entry shows, one eps across the corpus. This is the correctness claim.
* `MAX_HESS_ULPS` — the largest distance in representable doubles, pinned
  at **1**. This is a corpus measurement, kept because gh #711 showed a
  one-ulp coefficient difference moving a fixture from 17 conic iterations
  to 12. It is not a property of the evaluator.

These are the corpus's only models with a quadratic objective over the same
variables as a quadratic row, so every shared diagonal entry of `∇²L` has
two writers: the fast path scatters the row's share and adds the
objective's on top, the tape sums both in one pass, and the two orders
differ by an ulp.

Regenerating the pair with `--quad-rows 2` moves that, and — this is the
part worth recording — **it moves only on the ill-conditioned twin**:

| variant | worst ulp | worst relative |
|---|---:|---:|
| `--quad-rows 2 --column-scale 1` | 1 | `2.19e-16` |
| `--quad-rows 2 --column-scale 1e8` | **2** | `2.84e-16` |
| `--quad-rows 3 --column-scale 1` | **2** | `3.79e-16` |

So a second row would have forced `MAX_HESS_ULPS` up from 1 — loosening an
accuracy guard for the whole fast path to accommodate a fixture added for
an unrelated reason — while the relative deviation, the number that carries
the claim, never leaves the eps range. One row demonstrates the same
invariance and leaves the pin where the rest of the corpus put it.

What drives the ulp count is **cancellation in the sum, not the number of
writers**. An earlier draft of this file said the opposite — three writers,
therefore two ulp — and the arithmetic does not work that way: reassociating
`k` terms carries a relative error of order `(k−1)·eps·Σ|terms|/|Σ terms|`,
and it is the conditioning factor that runs away while `k` barely moves.
The table above is that in miniature (same `k`, different conditioning,
different ulp), and
`the_ulp_pin_is_a_corpus_measurement_and_the_relative_bound_is_the_guarantee`
makes it explicit: a dense synthetic **two**-row model reaches 8 ulp and a
dense **four**-row model reaches 2.

The pair still moves the *frequency* assertion (they add ~96 one-ulp
entries where the rest of the corpus produces 2); that is recorded beside
it, and carries no claim.

## The defect these fixtures found on the conic route (fixed here)

At this size both files clear the conic guards, so **the default route for
them is the SOCP driver, not the NLP path** — which is how these fixtures
came to test something they were not built for.

On that route, before the fix in this branch, `qcqp_columns_illcond.nl`
returned

```
SolveSucceeded   28 iterations   objective -4.0065155951e+02
Constraint violation ... 2.66e-15   (as reported by the solver)
```

against a true optimum of `-3.6421025382e+02`. Re-evaluating the returned
point with POUNCE's own evaluator violated the quadratic row by `4.948e+01`
— 38% of that row's right-hand side:

```sh
pounce check-x0 qcqp_columns_illcond.nl --x0-file <the returned x>
#   c[0]: g = 1.793545e2, bounds [-1e19, 1.298705e2], violation 4.948e1
```

The well-conditioned twin was fine on the same route, and the NLP route was
fine on both. It reproduced identically on binaries built before gh #703,
so it is **pre-existing and unrelated to the scaling method these fixtures
were added for** — they only exposed it, which is what a pair of models
that differ *only* in conditioning is for.

### The cause was the rank test, not the missing equilibration

`crates/pounce-convex/src/equilibrate.rs` predicts a nearby failure in its
own module docs — Ruiz equilibration is wired only into `solve_qp_ipm`,
because per-row scaling of `G` breaks a second-order cone, so a
badly-column-scaled conic problem gets no scaling — and it predicts it
surfaces as a `NumericalFailure`. That is not what happened here, and the
difference is the whole point: this was a **wrong answer, reported as a
success**, and it was produced upstream of the solver, in the reduction.

`qp_extract::psd_outer_factor` turns each quadratic row into a cone by
factoring `Σ_k f_k f_kᵀ = Q`, and the number of rows it emits is the
dimension of that cone. Its rank test cut at `1e-12 · max_diag` — one
absolute threshold for the whole matrix. The injected column scaling moves
`max_diag` by nineteen orders of magnitude, so on the ill-conditioned twin
the cut landed at `4.3e-3` and discarded **7 of 24** genuine directions:

| file | pivots kept | pivot range | residual `‖Q − Σ f fᵀ‖` |
|---|---:|---|---:|
| wellcond | 24 of 24 | `8.84e0 … 5.47e1` | `1.07e-14` |
| illcond | **17** of 24 | `1.54e-7 … 4.26e9` | `2.30e-3` |

Seven missing directions make the cone larger than the constraint, so the
solver reached a *better* objective than the true optimum and certified it
against the cone it was given. Its self-reported violation was honest about
that cone (`2.66e-15`) and silent about the model. A relative residual
check would not have caught it either: `2.30e-3` against `‖Q‖ = 4.3e9` is
`5.4e-13`.

The fix is to make the rank test relative to each pivot's *own* starting
diagonal rather than to the largest one, which is invariant under the
diagonal congruence `Q → CQC` that a change of units is. Two regression
tests pin it directly —
`rank_does_not_depend_on_the_units_the_columns_are_measured_in` and
`a_spanned_direction_is_dropped_at_every_column_scaling` — plus
`the_conic_route_gives_both_twins_the_same_answer` at the CLI level.

One consequence for reading history: the sweep line for
`qcqp_columns_illcond` in any baseline taken before this fix records the
*wrong* answer as the expected one. The fix moves exactly two sweep lines,
both of them that fixture, on both legs:

```
- exact  qcqp_columns_illcond  SolveSucceeded  it=28  obj=-400.6515595
+ exact  qcqp_columns_illcond  SolveSucceeded  it=26  obj=-364.2102508
```

### Why the gh #703 tests still pin `solver_selection=nlp`

Not because of the defect above — that is fixed. `curvature-based` reaches
the engine through `TNLP::get_scaling_parameters`, which only the general
NLP path calls, so the tests that measure *it* have to name that path to be
measuring anything. Asking for the option without naming a solver now
declines the convex route and says so, rather than accepting the option and
solving without it (gh#483); `curvature_scaling.rs` pins that separately.
