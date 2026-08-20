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

`crates/pounce-cli/tests/curvature_scaling.rs` pins that. It passes
`solver_selection=nlp` deliberately: at this size the models clear both
conic guards and would otherwise route to the SOCP driver, and gh #703 is
about the NLP path.

## Why one quadratic row and not two

`quad_evaluator_differential.rs` bounds how far the constant-matrix
evaluator may drift from the AD tape it replaces, and the bound is a
*measurement* of the corpus rather than a tolerance. These are the corpus's
only models with a quadratic objective over the same variables as a
quadratic row, so every shared diagonal entry of `∇²L` has two writers and
reassociates by one ulp. At **two** quadratic rows there are three writers
and the distance becomes two ulp — which would have forced `MAX_HESS_ULPS`
up from 1, loosening an accuracy guard for the whole fast path to
accommodate a fixture added for an unrelated reason. One row demonstrates
the same invariance and leaves that bound where the rest of the corpus put
it. The pair still moves the *frequency* assertion (they add ~96 one-ulp
entries where the rest of the corpus produces 2); that is recorded beside
it.

## A separate defect these fixtures expose (not gh #703)

At this size both files clear the conic guards, so **the default route for
them is the SOCP driver, not the NLP path**. On that route
`qcqp_columns_illcond.nl` returns

```
SolveSucceeded   28 iterations   objective -4.0065155951e+02
Constraint violation ... 2.66e-15   (as reported by the solver)
```

against a true optimum of `-3.6421025382e+02`. Re-evaluating the point it
returns, with POUNCE's own evaluator, violates the quadratic row by
`4.948e+01` — 38% of that row's right-hand side:

```sh
pounce check-x0 qcqp_columns_illcond.nl --x0-file <the returned x>
#   c[0]: g = 1.793545e2, bounds [-1e19, 1.298705e2], violation 4.948e1
```

The well-conditioned twin is fine on the same route (`-3.6421025082e+02`,
no violated rows), and the NLP route is fine on both. This reproduces
identically on binaries built before gh #703, so it is **pre-existing and
unrelated to the scaling method these fixtures were added for** — but it is
why the tests in `curvature_scaling.rs` pin `solver_selection=nlp`, and why
the sweep line for `qcqp_columns_illcond` must not be read as a correct
answer.

The mechanism is the one `crates/pounce-convex/src/equilibrate.rs` states
in its own module docs: Ruiz equilibration is wired only into
`solve_qp_ipm` because per-row scaling of `G` breaks a second-order cone,
so a badly-column-scaled conic problem gets no scaling at all. That module
predicts the failure surfaces as a `NumericalFailure`. Here it surfaces as
a silent wrong answer instead.
