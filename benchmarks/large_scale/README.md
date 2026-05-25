# Large-Scale Synthetic Suite

Large, sparse, synthetic NLPs designed to stress the sparse linear algebra
path and the workspace sizing of both ripopt and Ipopt. Problems are
parameterised by a size `n` and scaled up to around 100K variables. The
problems cover the main structural patterns ripopt needs to handle
efficiently:

- **Bratu** — 2D Bratu nonlinear PDE discretisation (sparse symmetric
  Jacobian, indefinite Hessian)
- **OptControl** — discretised optimal control with state and control
  bounds (block-tridiagonal Jacobian)
- **PoissonControl** — Poisson boundary control (dense blocks on a sparse
  skeleton)
- **SparseQP** — very large convex sparse quadratic program

These problems are intentionally synthetic rather than drawn from CUTEst
so we can scale `n` without shipping giant SIF files, and so that both
solvers see the exact same Rust `NlpProblem` struct for a fair comparison.

## Contents

- `problems.rs` — NlpProblem definitions shared by
  `tests/large_scale.rs`, `tests/large_scale_benchmark.rs`, and the
  optional comparison example
- `large_scale_results.txt` — captured stdout/stderr from the last run,
  parsed by `benchmarks/benchmark_report.py` for the large-scale section

## Prerequisites

None beyond ripopt itself; the comparison run requires `ipopt-native`.

## How to run

From the repo root:

```bash
make large-scale
```

or directly:

```bash
cargo test --release --features ipopt-native -- --ignored \
    large_scale_vs_ipopt --nocapture \
    2>&1 | tee benchmarks/large_scale/large_scale_results.txt
```

The tests are marked `#[ignore]` so they do not run during `cargo test`;
the benchmark target explicitly opts in.

## Output

- `large_scale_results.txt` — full run log with per-problem `BENCH:` lines
  that the composite report parser recognises

This suite feeds the composite `benchmarks/BENCHMARK_REPORT.md` via the
`load_large_scale_results()` parser in `benchmark_report.py`.
