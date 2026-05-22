# Sensitivity Analysis

POUNCE includes a parametric sensitivity capability compatible with
upstream Ipopt's `contrib/sIPOPT/` (Pirnay, López-Negrete & Biegler
2012, DOI
[10.1007/s12532-012-0043-2](https://doi.org/10.1007/s12532-012-0043-2)).
It computes the first-order change in the optimal primal solution with
respect to a problem parameter, reusing the KKT factorization from the
converged solve. Three entry points cover the common workflows.

## AMPL CLI

The main `pounce` driver auto-detects the sIPOPT suffixes
(`sens_state_1`, `sens_state_value_1`, `sens_init_constr`) in an input
`.nl`, runs a post-optimal sensitivity step after the solve, and
writes the perturbed primal back as a `sens_sol_state_1` suffix — no
separate binary or flag needed:

```sh
pounce problem.nl                   # writes problem.sol
pounce problem.nl out.sol --json-output result.json --json-detail full
```

`pounce_sens` is retained as a thin backward-compatibility alias:
`pounce_sens in.nl out.sol` is identical to `pounce in.nl out.sol`, so
existing AMPL / solver scripts keep working unchanged.

Related flags:

- `--sens-boundcheck` / `--sens-bound-eps EPS` — clamp the perturbed
  primal `x* + Δx` onto the declared `[x_l, x_u]` box.
- `--compute-red-hessian` / `--rh-eigendecomp` — compute the reduced
  Hessian (and its eigendecomposition) over the variables tagged by
  the `red_hessian` integer var-suffix.

## Rust library

`SensSolve` is a builder that wraps the `on_converged` callback
plumbing into a single call:

```rust
use pounce_sensitivity::SensSolve;

let result = SensSolve::new(vec![2, 3])
    .with_deltas(vec![0.05, 0.0])
    .with_reduced_hessian()
    .run(&mut app, tnlp);
// result.dx, result.reduced_hessian, result.status
```

`with_reduced_hessian_eigen()` adds the eigendecomposition;
`with_boundcheck(eps)` enables the bound projection.

## Python

`solve_with_sens` exposes the same capability from the
cyipopt-compatible Python wrapper:

```python
result = prob.solve_with_sens(x0=x0, sens_boundcheck=True)
```

`rh_eigendecomp=True` requests the reduced-Hessian eigendecomposition;
`sens_bound_eps=…` tunes the bound projection. See
[`python/notebooks/04_sensitivity.ipynb`](https://github.com/jkitchin/pounce/blob/main/python/notebooks/04_sensitivity.ipynb)
for a walkthrough.

## Verification

All three entry points are verified against upstream sIPOPT 3.14.19's
`parametric_cpp` golden output to within roughly 6e-9 per component.
The bound projection is a single-pass clamp; upstream's iterative
Schur refinement (re-factorize on each violation) is intentionally not
ported.
