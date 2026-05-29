# pounce-py

PyO3 bindings for POUNCE. Builds the `_pounce` Python extension module that
backs the `pounce` Python package (see `python/`).

The Python-facing surface is a cyipopt-compatible `Problem` class plus a
scipy-style `minimize()` facade and a `pounce.jax` subpackage providing:

- AD-built objective gradient, constraint Jacobian, and Lagrangian Hessian
  from JAX-traced `f(x)` and `g(x)` functions (`from_jax`).
- A `custom_vjp` wrapper around the solver that differentiates `x*(p)` via
  the implicit-function theorem on the active KKT system — slack
  inequality rows are dropped from the backward block (`solve`, pounce#73).
- A dual-warm-start variant `solve_with_warm` that hands `(x, λ, z)` from
  one call into the next (pounce#74).
- A batching rule `vmap_solve` plus a `ThreadPoolExecutor`-backed
  `vmap_solve_parallel` that releases the GIL inside each solve via
  `py.allow_threads` (pounce#74).
- A `JaxProblem` build-once / solve-many handle that caches the JIT
  artefacts, sparsity probe, and underlying `pounce.Problem` across
  calls; each worker thread of `vmap_solve_parallel` keeps its own
  cached `Problem` via `threading.local` (pounce#75).

Build:

```sh
# Develop install (needs maturin):
cd python && maturin develop --release

# Wheel:
cd python && maturin build --release
```

`cargo build` in the workspace builds this crate as a regular rlib for
type-checking; the wheel build adds `--features extension-module` so PyO3
links against the right Python ABI.
