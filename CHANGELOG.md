# Changelog

All notable changes to POUNCE are tracked here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html)
once it reaches `1.0.0`. Pre-1.0 minor bumps may include breaking
changes.

## [0.2.0] — 2026-05-25

First tagged release. The `0.1.0` work-in-progress version was never
tagged; everything below summarizes the state of `main` as of this
release.

### Solver core

- **Full Ipopt-parity C ABI**: `CreateIpoptProblem`, `IpoptSolve`,
  `AddIpoptStrOption` / `AddIpoptNumOption` / `AddIpoptIntOption`,
  `OpenIpoptOutputFile`, `SetIpoptProblemScaling`,
  `SetIntermediateCallback`, `GetIpoptCurrentIterate`,
  `GetIpoptCurrentViolations`, plus a new `IpoptSolver` session
  handle (`IpoptSolverSolve`, `IpoptSolverResolve`,
  `IpoptSolverKktSolve`, `IpoptSolverParametricStep`).
- **Restoration phase** wired through `IpoptSolve` with the soft
  restoration line search; nested IPM honors the parent's
  `print_iter_output` gate.
- **Rapid infeasibility detection** in the main loop; convergence
  statuses certified against upstream Ipopt.
- **Option-parity (tier-A waves 1-4)**: convergence options
  (`tol`, `acceptable_tol`, etc.), mu/watchdog/output toggles,
  iteration-output flags, warm-start machinery,
  `fixed_variable_treatment`, `nlp_*_bound_inf`,
  `barrier_tol_factor`, `sigma_min` / `sigma_max` for the adaptive
  quality-function oracle.
- **Sensitivity (sIPOPT)**: Phase D landed — convenience API,
  eigendecomposition, fixed-variable lifting, boundcheck. New
  `Solver` session API on top: value-typed `Factorization` handle
  in `pounce-linsol` enables factor-once / solve-many; `Solver`
  exposes `kkt_solve`, `parametric_step`, and
  `compute_reduced_hessian` without callback shapes.
- **Presolve** crate (`pounce-presolve`) as an opt-in TNLP wrapper.

### Backends and bindings

- **Python** (`pounce-solver`): PyO3 bindings with a cyipopt-style
  `Problem` class and a scipy-style `minimize()` facade. The wheel
  bundles the `pounce` CLI executable.
- **Python session API** (`pounce.Solver`): pyclass that wraps the
  Rust `Solver`, enabling warm-start sequences (MPC / parametric /
  B&B) and many-RHS sensitivity workflows without the
  callback-based shape.
- **pyomo-pounce** (`pyomo-pounce`): Pyomo SolverFactory plugin
  that drives the `pounce` CLI on the user's PATH.
- **GAMS link**: native solver link (`libGamsPounce`) for GAMS;
  Jacobian eval skips dense memsets and pure-linear rows.
- **CLI**: bundled `pounce` binary writes AMPL `.sol` solution
  output; new `--about` prints version / build / features / paths;
  `--dump` writes per-iteration KKT artefacts; the sIPOPT
  sensitivity step is folded in.

### Linear-solver layer

- **Public `Factorization`** in `pounce-linsol`: factor once,
  back-solve many RHS, refactor with new values reusing the
  symbolic factor / AMD ordering.
- **MA57** backend (`pounce-hsl`) honors the `linear_solver`
  option default (`"ma57"`).
- **Feral** backend: cascade-break and FMA default off (opt-in via
  env); near-singular factorizations are flagged via an absolute
  pivot floor; explicit-zero stripping before KKT factor; skips
  refactor on same-matrix back-solve.

### Numerical robustness

- TNLP `eval_*` user-callback failures surface as NaN instead of
  panicking.
- Round-off-tolerant `Compare_le` in the Armijo line-search test.
- Unconstrained problems routed through the IPM (no degenerate
  paths).
- `push_x_into_interior` uses `dim()` (not `values().len()`),
  fixing a subtle off-by-one on partially-filled vectors.
- `OrigIpoptNlp::eval_h` always uses the `h_entry_in_full`
  mapping; closes the panic when an entire Hessian row sits on a
  fixed variable.

### Benchmarks

- **Composite report** (`make benchmark` →
  `benchmarks/BENCHMARK_REPORT.md`) covering 9 suites: CUTEst (727
  curated; 1542 full sweep), Mittelmann LP/QP, water-network
  design, gas-network, electrolyte, grid, CHO, large-scale, and
  the GAMS link.
- **Incremental per-suite targets**: `make benchmark-<suite>`
  skips when `results.json` is fresh; `make benchmark-<suite>-rerun`
  forces a rebuild.
- **MA57 baseline** integrated into the composite report.

### Studio & tooling

- **studio/mcp** MCP server (`pounce-studio-mcp`) with
  `analyze`, `run`, `explain`, `citations` tools and an embedded
  glossary; backed by `pounce-studio-core` via PyO3.
- **Linear-solver post-mortem** aggregated end-to-end and
  surfaced through the studio.

### Infrastructure

- CI workflow with format / clippy / build / test, plus
  wheel-smoke for `pounce-solver` and `pyomo-pounce`.
- mdbook documentation built and deployed to GitHub Pages via the
  new `docs.yml` workflow.
- Zenodo metadata (`.zenodo.json`) and `CITATION.cff` for
  archival on every GitHub Release.

[0.2.0]: https://github.com/jkitchin/pounce/releases/tag/v0.2.0
