# Benchmarks

Comparison harnesses that exercise POUNCE against upstream Ipopt across
several NLP test suites. Each suite lives in its own subdirectory with
its own README explaining the problems, prerequisites, and how to run
it.

The benchmark *inputs* (large `.nl` exports, compiled SIF problem
libraries) and the per-run *outputs* (logs, JSON results, generated
reports) are regenerated locally and not tracked in the repository.
The per-suite README files and the source harnesses are tracked.

## Suites

| Suite | Problem class | Size | Notes |
|-------|---------------|------|-------|
| [`cho/`](cho/README.md)             | Parameter estimation, kinetic ODE | 1 problem, medium dense | CHO cell kinetics from Pyomo `parmest`; stress test for restoration |
| [`cutest/`](cutest/README.md)       | Canonical NLP collection | 727 / 1542 problems | Gould–Orban–Toint MASTSIF; requires `prepare.sh` to compile SIF libraries |
| [`electrolyte/`](electrolyte/README.md) | Gibbs free-energy minimization | 13 problems | Aqueous electrolyte equilibrium; ill-conditioned by construction |
| [`gas/`](gas/README.md)             | Gas-pipeline network NLP | 4 problems, up to 21k vars | Finite-volume Euler discretization, GasLib networks |
| [`grid/`](grid/README.md)           | AC optimal power flow | MATPOWER cases | Polar-form ACOPF; canonical nonconvex grid NLP |
| [`large_scale/`](large_scale/README.md) | Synthetic large sparse NLPs | up to 100k vars | Bratu, OptControl, PoissonControl, SparseQP — stresses sparse linsol |
| [`mittelmann/`](mittelmann/README.md) | Mittelmann ampl-nlp | 47 problems, up to 261k vars | Standard public NLP benchmark |
| [`water/`](water/README.md)         | Water-network design | 6 problems | MINLPLib instances, signomial nonlinearities |

## Common targets

The repo-root `Makefile` exposes shortcuts that build the `pounce` CLI
first and then drive the per-suite harnesses:

```sh
make bench-cho          # CHO parameter-estimation
make bench-gas          # GasLib pipelines
make bench-water        # Water-network design
make bench-mittelmann   # Mittelmann ampl-nlp
make bench-cutest       # CUTEst (run after `make bench-cutest-prepare` once)
```

Some suites (`electrolyte`, `grid`, `large_scale`) currently run
through `cargo run` / `cargo test` directly — see each suite's README
for the exact invocation.

## Adding a new suite

1. Create `benchmarks/<suite>/` with a `README.md` describing the
   problem class, sources, prerequisites, and how to run.
2. The `.gitignore` whitelists every `benchmarks/*/README.md` and the
   `cutest/` source tree; everything else under `benchmarks/` is
   ignored by default. Add explicit `!benchmarks/<suite>/<file>`
   whitelist lines for source files that should be tracked.
3. Wire a `make bench-<suite>` target in the repo-root `Makefile` if
   the suite is intended to be a one-shot run.
