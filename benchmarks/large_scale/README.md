# Large-Scale Synthetic Suite

Large, sparse, synthetic NLPs designed to stress the sparse linear algebra
path and workspace sizing of both POUNCE and Ipopt. Problems are
parameterised by a size and scaled up to around 100K variables. They are
emitted as AMPL `.nl` files by `generate_nl.py` (Pyomo) and run through the
same dual-solver `.nl` driver (`benchmarks/scripts/run_nl_bench.sh`) as
every other suite — there is no compiled Rust harness and no libipopt FFI.

The five problems cover the main structural patterns POUNCE needs to handle
efficiently:

- **rosenbrock** — generalized/chained Rosenbrock (CUTE `GENROSE`),
  unconstrained, tridiagonal Hessian, `f* = 1` at `x = 1`. Default `n = 2000`
  (kept small: it is fundamentally O(n) Newton iterations).
- **bratu** — 1-D Bratu BVP `-u'' = λ e^u` with a 3-point stencil; pure
  feasibility (objective ≡ 0), nonlinear equality constraints. Default
  `n = 10000`.
- **optcontrol** — discretised linear-quadratic optimal control; quadratic
  objective, block-tridiagonal linear dynamics. Default `T = 50000`
  (`n = 100001`, `m = 50001`).
- **poisson** — 2-D Poisson boundary control on a K×K grid; quadratic
  objective, 5-point-stencil linear constraints. Default `K = 200`
  (`n = 80000`, `m = 40000`).
- **sparseqp** — convex sparse QP, tridiagonal `Q`, cyclic three-term
  inequality rows, box bounds. Default `n = 50000`.
- **laptime** — minimum-lap-time vehicle trajectory on a closed circuit,
  transcribed by degree-3 Radau direct collocation. Nonconvex objective,
  saturating (Pacejka) tyre curves, a friction-ellipse path constraint, and a
  periodicity row closing the lap. Default `N = 1000` intervals with 8
  steering-lag states, i.e. `n_x = 14` — `n = 58014`, `m = 62014`, 2000
  degrees of freedom. Added for pounce #698.

These are intentionally synthetic rather than drawn from a public library so
the size can be scaled freely without shipping giant fixtures, and so both
solvers see the exact same problem.

### Why `laptime` is not a sixth variation on the first five

The first five are all large and sparse, and none of them is shaped like the
models that have actually broken POUNCE's limited-memory path. `optcontrol`
is the closest and it is a single-state, linear-dynamics, convex QP: two
Jacobian entries per row, no active inequalities, no restoration, and an
exact Hessian available for free.

That gap is not hypothetical. `scripts/scaling-probe.sh` measured the
limited-memory path as linear from `n = 2,000` to `n = 128,000` **on these
families** and reported no hidden quadratic — while pounce #684 was, at that
moment, allocating a dense `n(n+1)/2` Hessian triangle the instant
restoration was entered under `hessian_approximation=limited-memory`. The
probe was right about what it measured and blind to the defect, because none
of its problems enters restoration.

`laptime` is the shape that found #677, #684, #686 and #688: a 60,000-variable
collocation model with analytic Jacobians and no analytic Hessian. It is an
independent problem built from published vehicle-dynamics modelling, not a
copy of the reporter's proprietary model.

It already separates the two Hessian legs on its own. Mesh refinement, POUNCE
0.10.0, 2 lag states, iterations to convergence:

| `N` | exact | limited-memory | lap time (exact) |
|---|---|---|---|
| 40 | 27 | 71 | 65.658683 |
| 80 | 29 | 144 | 65.462561 |
| 160 | 33 | 276 | 65.370889 |
| 320 | 89 | **884, `ErrorInStepComputation`** | 65.326142 |
| 640 | 339 | **did not finish in 900 s** | 65.303131 |

The exact-Hessian column is a well-behaved transcription: lap time converges
first-order in the mesh (the differences halve — the expected rate for a
minimum-time problem whose optimal control switches), and the solve stays
cheap. The limited-memory column diverges from it by roughly 3x per
refinement and then falls over at a size the exact path handles in 89
iterations. Whether that is L-BFGS legitimately struggling on a hard
nonconvex problem or a defect is **not established here** — but it is the
first problem in this repo that poses the question at all.

## Contents

- `generate_nl.py` — Pyomo generator; writes one `.nl` (plus matching
  `.row`/`.col` name maps) per problem into `nl/`
- `nl/` — generated `.nl` files (gitignored; regenerated locally)
- `pounce.json` / `ipopt_ma57.json` — latest POUNCE and Ipopt/MA57 results

## Prerequisites

- `pyomo` (for `generate_nl.py`)
- `ipopt` (MA57 build) for the comparison side, same as the other `.nl`
  suites

## How to run

From the repo root:

```bash
make -C benchmarks large-scale            # generate .nl if missing, then run
make -C benchmarks large-scale-rerun      # force a rerun
make -C benchmarks large-scale-generate   # (re)generate the .nl files only
```

Regenerate at a different scale, or generate a single problem:

```bash
python3 generate_nl.py --scale 0.1          # 10% of every default size
python3 generate_nl.py optcontrol --optcontrol-t 1000
```

## Output

- `nl/*.nl` — generated problems
- `pounce.json` / `ipopt_ma57.json` — POUNCE and Ipopt per-problem results
  in the canonical
  `{solver,name,n,m,status,objective,iterations,solve_time}` schema

This suite feeds the composite `benchmarks/BENCHMARK_REPORT.md` via
`load_domain_results()` in `benchmark_report.py`.
