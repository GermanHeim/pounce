#!/usr/bin/env python3
"""Generate structurally-equivalent QCQP `.nl` instances (gh #588).

Why this exists
---------------
The Mittelmann `qcqp*` family is the target of the quadratic-structure work,
and the instances are not redistributable — a machine without
`$POUNCE_BENCH_DATA` (and without egress to `plato.asu.edu`) can measure
nothing on them. This writes a model in the *same shape* so a phase can be
probed in the regime it claims to help, and so the claim can be reproduced
by anyone.

A generated instance is **not** a Mittelmann instance and no number taken on
one may be reported as though it were. What it can honestly support is a
statement about a *regime* — "evaluation-bound at this size, and here is what
the phase does to the evaluation timers" — and only after
`--shape qcqp1000-2c` has been checked against the real `qcqp1000-2c`, which
is the one instance available to calibrate against.

The shape it reproduces
-----------------------
`qcqp1000-2c` as AMPL emits it:

    1000 vars, 5107 constraints, 1 objective, 0 ranges, 100 eqns;
    7 nonlinear constraints, 1 nonlinear objective;
    nlvc = nlvo = nlvb = 1000; 63121 Jacobian nonzeros.

* every variable free (`b` code 3), no initial guess, so `x0 = 0`;
* `k` quadratic rows `½xᵀQᵢx ≤ bᵢ`, each written as one `o54` sumlist of
  `o2 n0.5 o2 o2 n<c> v<i> v<j>` summands — one per **stored** entry of
  `Qᵢ`, both triangles, exactly as AMPL emits them (which is why the merge
  in the recognizer has to be exact: it is folding `(i,j)` and `(j,i)` with
  the identical coefficient);
* a quadratic objective in the same form plus a dense `G0` linear part;
* the remaining rows linear and sparse, a stated number of them equalities.

Each `Qᵢ` is symmetric with a random sparse off-diagonal pattern and a
diagonal set to `--dominance` times the row's off-diagonal absolute sum, so
it is positive semidefinite by diagonal dominance (`--dominance 1.0` puts it
on the boundary, which is the ill-conditioned end). A reference point
`x_ref` is drawn first and every right-hand side is set from it, so the model
is strictly feasible and Slater holds.

Usage
-----
    gen-qcqp-nl.py --shape qcqp1000-2c --out /tmp/gen1000.nl
    gen-qcqp-nl.py --n 500 --quad-rows 10 --quad-density 1.0 \
                   --linear-rows 110 --eqns 10 --out /tmp/gen500.nl
"""

from __future__ import annotations

import argparse
import random
import sys

# Shapes taken from the real headers. `quad_density` is the fraction of the
# n x n symmetric pattern that is stored, so `1.0` is the dense row
# `qcqp500-3c` has and `0.042` is what `qcqp1000-2c`'s 41 868 summands work
# out to.
SHAPES = {
    # Measured directly from mittelmann/nl/qcqp1000-2c.nl.
    "qcqp1000-2c": dict(
        n=1000, quad_rows=7, quad_density=0.0419, linear_rows=5100,
        eqns=100, linear_nnz=11, dominance=1.02,
    ),
    # From the design note's table (the instance itself is not available
    # here): 120 rows, 10 of them nonlinear, dense per-row Q.
    "qcqp500-3c": dict(
        n=500, quad_rows=10, quad_density=1.0, linear_rows=110,
        eqns=10, linear_nnz=11, dominance=1.02,
    ),
}


def build_quadratic(n: int, density: float, dominance: float, rng: random.Random):
    """A sparse symmetric PSD matrix as {row: {col: value}}, both triangles."""
    q: dict[int, dict[int, float]] = {i: {} for i in range(n)}
    if density >= 1.0:
        pairs = ((i, j) for i in range(n) for j in range(i + 1, n))
    else:
        # Sample off-diagonal pairs without materializing all n(n-1)/2.
        want = int(density * n * (n - 1) / 2)
        seen = set()
        while len(seen) < want:
            i = rng.randrange(n)
            j = rng.randrange(n)
            if i != j:
                seen.add((min(i, j), max(i, j)))
        pairs = iter(sorted(seen))
    for i, j in pairs:
        v = rng.uniform(-10.0, 10.0)
        q[i][j] = v
        q[j][i] = v
    for i in range(n):
        s = sum(abs(v) for v in q[i].values())
        # Strictly positive even for an empty row, so `Q` is nonsingular at
        # dominance > 1 and every variable appears in the row (which is what
        # makes `J` dense, as it is in the real instance).
        q[i][i] = dominance * s + 1.0
    return q


def quad_summands(q: dict[int, dict[int, float]]):
    """The `o54` operand list for `½xᵀQx`, in AMPL's emission order."""
    for i in sorted(q):
        for j in sorted(q[i]):
            yield i, j, q[i][j]


def quad_value(q, x):
    total = 0.0
    for i in q:
        for j, v in q[i].items():
            total += v * x[i] * x[j]
    return 0.5 * total


def emit(out, args) -> None:
    rng = random.Random(args.seed)
    n = args.n
    m = args.quad_rows + args.linear_rows
    x_ref = [rng.uniform(-1.0, 1.0) for _ in range(n)]

    # --- quadratic rows and objective ---
    quads = [
        build_quadratic(n, args.quad_density, args.dominance, rng)
        for _ in range(args.quad_rows)
    ]
    obj_q = build_quadratic(n, args.quad_density, args.dominance, rng)
    obj_g = [rng.uniform(-100.0, 100.0) for _ in range(n)]

    # --- linear rows: `linear_nnz` entries each, first `eqns` are equalities ---
    lin_rows = []
    for _ in range(args.linear_rows):
        cols = sorted(rng.sample(range(n), min(args.linear_nnz, n)))
        lin_rows.append([(c, rng.uniform(-10.0, 10.0)) for c in cols])

    jac_nnz = args.quad_rows * n + sum(len(r) for r in lin_rows)

    w = out.write
    w("g3 0 1 0\t# problem generated-qcqp\n")
    w(f" {n} {m} 1 0 {args.eqns}\t# vars, constraints, objectives, ranges, eqns\n")
    w(f" {args.quad_rows} 1\t# nonlinear constraints, objectives\n")
    w(" 0 0\t# network constraints: nonlinear, linear\n")
    w(f" {n} {n} {n}\t# nonlinear vars in constraints, objectives, both\n")
    w(" 0 0 0 1\t# linear network variables; functions; arith, flags\n")
    w(" 0 0 0 0 0\t# discrete variables: binary, integer, nonlinear (b,c,o)\n")
    w(f" {jac_nnz} {n}\t# nonzeros in Jacobian, gradients\n")
    w(" 0 0\t# max name lengths: constraints, variables\n")
    w(" 0 0 0 0 0\t# common exprs: b,c,o,c1,o1\n")

    # Variable bounds: all free, as in the real instance.
    w("b\n")
    for _ in range(n):
        w("3\n")

    # Row bounds. Quadratic rows and the linear inequalities are `<= b`
    # (code 1); the first `eqns` linear rows are equalities (code 4). Every
    # right-hand side is taken at `x_ref` plus a slack, so `x_ref` is
    # strictly feasible.
    w("r\n")
    for q in quads:
        w(f"1 {quad_value(q, x_ref) + args.slack!r}\n")
    for k, row in enumerate(lin_rows):
        val = sum(c * x_ref[j] for j, c in row)
        if k < args.eqns:
            w(f"4 {val!r}\n")
        else:
            w(f"1 {val + args.slack!r}\n")

    # Constraint bodies.
    for k, q in enumerate(quads):
        terms = list(quad_summands(q))
        w(f"C{k}\n")
        w("o54\n")
        w(f"{len(terms)}\n")
        for i, j, v in terms:
            w(f"o2\nn0.5\no2\no2\nn{v!r}\nv{i}\nv{j}\n")
    for k in range(len(lin_rows)):
        w(f"C{args.quad_rows + k}\nn0\n")

    # Objective (minimize).
    obj_terms = list(quad_summands(obj_q))
    w("O0 0\n")
    w("o54\n")
    w(f"{len(obj_terms)}\n")
    for i, j, v in obj_terms:
        w(f"o2\nn0.5\no2\no2\nn{v!r}\nv{i}\nv{j}\n")

    # `k` section: cumulative count of Jacobian entries in columns 0..n-2.
    col_counts = [0] * n
    for _ in range(args.quad_rows):
        for j in range(n):
            col_counts[j] += 1
    for row in lin_rows:
        for j, _ in row:
            col_counts[j] += 1
    w(f"k{n - 1}\n")
    run = 0
    for j in range(n - 1):
        run += col_counts[j]
        w(f"{run}\n")

    # `J` blocks: the quadratic rows carry a dense zero linear part (their
    # whole body is in the tree), the linear rows carry their coefficients.
    for k in range(args.quad_rows):
        w(f"J{k} {n}\n")
        for j in range(n):
            w(f"{j} 0\n")
    for k, row in enumerate(lin_rows):
        w(f"J{args.quad_rows + k} {len(row)}\n")
        for j, c in row:
            w(f"{j} {c!r}\n")

    # `G` block: the objective's linear part, dense.
    w(f"G0 {n}\n")
    for j in range(n):
        w(f"{j} {obj_g[j]!r}\n")


def main(argv: list[str]) -> int:
    # Shape-able options default to `None` so a preset can fill them in
    # *without* overriding a flag the caller passed explicitly — precedence
    # is flag, then `--shape`, then the fallback below. (Applying the preset
    # unconditionally silently ignored `--dominance`, which made two
    # calibration runs come back byte-identical and look like the knob did
    # nothing.)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--shape", choices=sorted(SHAPES), help="a preset from the real headers")
    p.add_argument("--n", type=int)
    p.add_argument("--quad-rows", type=int)
    p.add_argument("--quad-density", type=float)
    p.add_argument("--linear-rows", type=int)
    p.add_argument("--eqns", type=int)
    p.add_argument("--linear-nnz", type=int)
    p.add_argument("--dominance", type=float)
    p.add_argument("--slack", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)
    fallback = dict(
        n=1000, quad_rows=7, quad_density=0.0419, linear_rows=5100,
        eqns=100, linear_nnz=11, dominance=1.02,
    )
    preset = SHAPES.get(args.shape, {})
    for k, v in fallback.items():
        if getattr(args, k) is None:
            setattr(args, k, preset.get(k, v))
    with open(args.out, "w") as fh:
        emit(fh, args)
    print(f"wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
