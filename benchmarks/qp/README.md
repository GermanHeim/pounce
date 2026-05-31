# Maros-Mészáros Convex QP Benchmarks

The [Maros-Mészáros test set][marmes] is the standard collection of convex
quadratic programs, the QP analogue of the NETLIB LP set. Each instance is

```
minimize    1/2 xᵀ P x + qᵀ x + r
subject to  l_c <= C x <= u_c        (general linear rows)
            lb  <=  x  <= ub          (variable bounds)
```

with `P` symmetric positive semidefinite (convex), so every problem has a
global optimum. The problems are deliberately hard — ill-conditioned,
degenerate, or large — and were curated by Maros & Mészáros (1997),
[*A Repository of Convex Quadratic Programming Problems*][report], Imperial
College tech. report DOC 97/6.

This suite lives under `benchmarks/qp/`. The problems are emitted as AMPL
`.nl` files by `generate_nl.py` and run through the same dual-solver `.nl`
driver (`benchmarks/scripts/run_nl_bench.sh`) as every other suite — no
compiled harness, no libipopt FFI. The suite feeds into the composite
`benchmarks/BENCHMARK_REPORT.md`.

## Problems

- **Instance count:** 138 convex QPs (the full standard set).
- **Size range:** from `n = 2` (e.g. `HS21`, `QPTEST`, `TAME`, `ZECEVIC2`)
  up to roughly `n = 93,000` variables / `m = 187,000` rows (`BOYD2`,
  `CONT-300`).
- **Class:** all are minimizations of a convex quadratic objective subject
  to linear equality/inequality rows and variable bounds.

## Source and how the `.nl` were produced

The original distribution is QPS (MPS plus a quadratic-objective section)
on Csaba Mészáros' SZTAKI FTP. That FTP is effectively dead (the
`~meszaros/public_ftp/qpdata/marosmeszaros/` path now redirects into a 404
loop), so this suite pulls a well-known faithful mirror maintained by the
qpsolvers project:

> <https://github.com/qpsolvers/maros_meszaros_qpbenchmark> — `data/*.mat`

Those `.mat` files were produced from the original SIF problems by
`sif2mat.m` in `proxqp_benchmark`. Each is a MATLAB v5 file (read with
`scipy.io.loadmat`) holding the QP in the documented form

| key | meaning |
|-----|---------|
| `P` (`n×n` sparse) | Hessian of the `1/2 xᵀPx` term |
| `q` (`n`)          | linear objective coefficients |
| `r` (scalar)       | constant objective offset |
| `A` (`m×n` sparse) | `== vstack([C, eye(n)])` — the **last `n` rows** are the variable-bound identity block; the rows above are the general linear rows `C` |
| `l`, `u` (`m`)     | double-sided bounds `l <= A x <= u` |
| `n`, `m`           | dimensions |

with the infinity constant set to `1e20`. This is exactly the convention
used by qpbenchmark's own loader (`maros_meszaros.py`,
`load_problem_from_mat_file` / `convert_problem_from_double_sided`).

`generate_nl.py` reproduces that decoding and, for each problem, builds a
Pyomo `ConcreteModel` with the objective `1/2 xᵀPx + qᵀx + r` (the `0.5`
factor on the full symmetric `P`, `minimize` sense, and the constant offset
`r` retained so the emitted objective matches the published
Maros-Mészáros optimal value), the general rows `l_c <= C x <= u_c`, and the
variable bounds `lb <= x <= ub`. It then writes each as `<NAME>.nl`.

### Validation

The `0.5` factor / sense / offset were verified before bulk conversion. On
sample instances POUNCE matches both the published reference optima and an
independent HiGHS QP solve:

| Problem | POUNCE objective | independent check |
|---------|------------------|-------------------|
| `HS21`     | -99.960000 | HiGHS -99.960000; published −99.96 |
| `QPTEST`   |   4.371875 | HiGHS 4.371875 |
| `DUAL1`    |   0.035013 | HiGHS 0.035013 |
| `DUAL2`    |   0.033734 | — |
| `ZECEVIC2` |  -4.125000 | published −4.125 |
| `HS35`     |   0.111111 | published 1/9 |

All reported `Optimal Solution Found`.

## Contents

- `generate_nl.py` — downloads the `.mat` mirror (into `data/`, gitignored)
  and converts each QP to one `.nl` (plus `.row`/`.col` name maps) in `nl/`
- `data/` — cached `.mat` downloads (gitignored; re-fetched on demand)
- `nl/` — generated `.nl` files (gitignored; regenerated locally)
- `ipopt_ma57.json` — committed Ipopt-MA57 reference (run via
  `make -C benchmarks ipopt-ref-qp`)
- `pounce.json` — latest POUNCE results (gitignored, regenerated each release)

## Prerequisites

- `pyomo`, `scipy`, `numpy` (for `generate_nl.py`)
- network access on first run to fetch the `.mat` mirror
- `ipopt` (MA57 build) for the comparison side, same as the other `.nl`
  suites

## How to run

From the repo root:

```bash
make -C benchmarks qp-run         # generate .nl if missing, then run POUNCE
make -C benchmarks qp-rerun       # force a POUNCE rerun
make -C benchmarks qp-generate    # (re)generate the .nl files only
```

Generate a single problem, or skip the download and use cached `.mat`:

```bash
python3 generate_nl.py QPTEST HS21      # only the named problems
python3 generate_nl.py --no-fetch       # use already-downloaded data/*.mat
```

Refresh the Ipopt reference for this suite (rare):

```bash
make -C benchmarks ipopt-ref-qp
```

[marmes]: https://www.cuter.rl.ac.uk/Problems/marmes.shtml
[report]: https://www.doc.ic.ac.uk/~rr2000/DTR97-6.pdf
