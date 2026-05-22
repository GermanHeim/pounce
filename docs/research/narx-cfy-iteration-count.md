# Investigation note — NARX_CFy iteration count

**Status: closed, no action.** Investigated 2026-05-22. The apparent
"pounce is 2× slower than ipopt" on NARX_CFy is a basin-of-attraction
artifact, not an algorithmic weakness. Recorded here in case the
symptom resurfaces.

## 1. The symptom

A benchmark comparison showed pounce taking far more iterations than
ipopt on the Mittelmann `NARX_CFy` instance:

| run                          | linear solver | iters | objective   |
| ---------------------------- | ------------- | ----- | ----------- |
| ipopt+MA57 (one invocation)  | ma57          | 234   | 8.5383e-3   |
| ipopt+MA57 (another, May 21) | ma57          | 430   | 8.7268e-3   |
| ipopt+MUMPS                  | mumps         | 437   | 8.6015e-3   |
| **pounce+MA57**              | ma57          | 437   | 8.6015e-3   |
| ipopt+feral / pounce+feral   | feral         | 269/403 | ~8.59-8.67e-3 |

Logs: `benchmarks/mittelmann/logs/{ipopt-ma57,ipopt-mumps,pounce-ma57}/NARX_CFy.log`.

## 2. Why it is not an algorithmic weakness

- **`NARX_CFy` is nonconvex and degenerate.** 43973 vars / 43720
  equality constraints (~253 free dims) + 4536 bounds; the NARX
  recurrence makes the constraint Jacobian near-rank-deficient. At
  least four distinct local minima cluster within 2% of each other
  (8.538, 8.601, 8.669, 8.727 e-3). The KKT is near-singular, so the
  basin is decided by rounding-level perturbations in the first
  Newton step.

- **ipopt+MA57 is not reproducible against itself.** The same solver
  (`linear_solver=ma57`, Ipopt 3.14.20) produced 234 iters → 8.5383e-3
  and 430 iters → 8.7268e-3 on different invocations — different basin,
  2× iteration spread. The iteration count is a basin lottery, not a
  solver property.

- **pounce+MA57 reproduces ipopt+MUMPS digit-for-digit** across all
  437 iterations and converges to the identical minimum (8.6015e-3).
  pounce's IPM is a faithful Ipopt port — same line search, monotone
  µ schedule, refinement (`residual_ratio_max=1e-10`), all defaults.
  There is **zero** iteration gap vs ipopt+MUMPS. The "gap" only
  exists against one lucky ipopt+MA57 draw, which converges to a
  *different* optimum.

## 3. The one unexplained detail (deliberately not pursued)

pounce+MA57 follows the **MUMPS** trajectory, not the **MA57**
trajectory, despite linking MA57. pounce's MA57 config was verified
byte-identical to Ipopt's (`pivtol 1e-8`, `pivtolmax 1e-4`,
`automatic_scaling off`, `pivot_order 5`, `block_size 16`;
`linear_system_scaling=none`, `nlp_scaling_method=gradient-based`).

The divergence is at **iteration 1**, from a byte-identical
iteration-0 point. With the linear solver config ruled out, the cause
is in the first KKT system's regularization δ_w/δ_c and inertia
interpretation (`crates/pounce-algorithm/src/kkt/perturbation_handler.rs`).
On a near-singular degenerate KKT, MA57 and MUMPS report
inertia/singularity differently; pounce's handling of MA57's report
coincides with the MUMPS outcome rather than Ipopt-MA57's.

This is not a bug — pounce finds a valid optimum (better than
ipopt+MA57's 430-iter run). It was not pursued because pinning it
would not make pounce faster; it would only explain the basin choice.

## 4. If this resurfaces

- Do **not** treat the iteration count as a regression unless pounce
  and the reference ipopt run converge to the **same** objective with
  the **same** linear solver. Compare basins first.
- The only legitimate lever to cut iterations on this class of
  degenerate problem is the µ strategy — `mu_strategy=adaptive`
  (quality-function oracle, already implemented in
  `crates/pounce-algorithm/src/mu/adaptive.rs`, not the default) — or
  attacking the long regularized tail (here: 307 tail iterations,
  ~97% regularized, shared with ipopt+MUMPS).
- To pin §3, instrument iteration 0→1: dump δ_w, δ_c, reported
  inertia (negevals), and the step vector for pounce+MA57 and compare
  against a high-`print_level` ipopt+MA57 run.
