# Rust API

POUNCE is written in Rust, so the Rust API is the solver itself rather than a
binding over it. Everything is reached through one crate — **`pounce-rs`**, a
facade that re-exports a curated public surface so your code does not depend
on POUNCE's internal crate layout.

```sh
cargo add pounce-rs
```

```toml
[dependencies]
pounce-rs = "0.9"
```

The default build is the NLP path. The convex/conic, active-set QP, and
sensitivity solvers are behind [feature flags](#feature-flags).

> **Why one crate.** The solver is split across ~20 workspace crates
> (`pounce-nlp`, `pounce-algorithm`, `pounce-convex`, …) whose boundaries move
> as the code evolves. Depending on them directly couples you to that layout.
> `pounce-rs` is the stability boundary; everything below is internal.

## Two APIs

### The builder — for the common case

Implement [`Problem`] — only `objective` is required — then configure and
solve. Anything you leave out is supplied: missing gradients and Jacobians are
approximated by finite differences, and the Hessian defaults to a
limited-memory (L-BFGS) approximation.

```rust
use pounce_rs::prelude::*;

// min (x0−1)² + (x1−2)²  s.t.  x0 + x1 == 3,  0 ≤ x ≤ 5
struct P;
impl Problem for P {
    fn objective(&self, x: &[f64]) -> f64 {
        (x[0] - 1.0).powi(2) + (x[1] - 2.0).powi(2)
    }
    fn n_constraints(&self) -> usize { 1 }
    fn constraints(&self, x: &[f64], g: &mut [f64]) { g[0] = x[0] + x[1]; }
}

let sol = Nlp::new(P)
    .var_bounds(&[0.0, 0.0], &[5.0, 5.0])
    .constraint_bounds(&[3.0], &[3.0])   // equality: lower == upper
    .x0(&[0.0, 0.0])
    .option_num("tol", 1e-10)
    .solve();

assert!(sol.success);
assert!((sol.x[0] - 1.0).abs() < 1e-5 && (sol.x[1] - 2.0).abs() < 1e-5);
```

`n` is inferred from `var_bounds` or `x0` (they must agree). Options use the
same names as the CLI and upstream Ipopt — `option_num`, `option_int`,
`option_str`; see [Solver Options](options.md).

The returned `Solution` carries `success` / `status`, `x`, `objective`,
`multipliers`, the constraint values `g`, the bound multipliers `z_l` / `z_u`,
and `stats` (wall time, iteration count, evaluation counts, final
infeasibilities). The vector fields are filled by `finalize_solution`, so they
stay **empty** if a solve aborts before finalizing — check `success` before
indexing.

To supply exact derivatives, implement `gradient` and `jacobian` and return
`true`; returning `false` (the default) selects finite differences for that
callback.

### `TNLP` — for full control

For an exact Hessian, custom Jacobian/Hessian *sparsity*, or NLP scaling,
implement the [`TNLP`] trait directly and drive it with `IpoptApplication`.
This is the same trait the CLI and the C ABI sit on.

```rust
use pounce_rs::prelude::*;
use std::cell::RefCell;
use std::rc::Rc;

let mut app = IpoptApplication::new();
app.initialize()?;
let prob = Rc::new(RefCell::new(MyTnlp::default()));
let status = app.optimize_tnlp(Rc::clone(&prob) as Rc<RefCell<dyn TNLP>>);
assert_eq!(status, ApplicationReturnStatus::SolveSucceeded);
```

You provide `get_nlp_info` (sizes and nonzero counts), `get_bounds_info`,
`get_starting_point`, the evaluators (`eval_f`, `eval_grad_f`, `eval_g`,
`eval_jac_g`, `eval_h`), and `finalize_solution` to receive the answer.
`eval_jac_g` and `eval_h` are called in two modes — `SparsityRequest::Structure`
for the pattern, then `SparsityRequest::Values` — so the pattern is declared
once and reused across iterations.

The [crate documentation on docs.rs](https://docs.rs/pounce-rs) has a complete
HS071 walkthrough.

## Iteration capture and logging

Opt into the per-iteration trajectory with `.capture_iterations()` on the
builder; the records land in `sol.stats.iterations`. Outside the builder,
`with_iter_capture` wraps any closure and returns the records alongside its
result:

```rust
use pounce_rs::prelude::*;

let (sol, iters) = with_iter_capture(|| {
    Nlp::new(P)
        .var_bounds(&[0.0, 0.0], &[5.0, 5.0])
        .constraint_bounds(&[3.0], &[3.0])
        .solve()
});
assert!(sol.success && !iters.is_empty());
```

On the `IpoptApplication` path, install `collector_scope()` for the duration of
the solve and read the history back from `statistics()`. `init_subscriber()`
turns on console logging without your crate taking a `tracing` dependency.

## Feature flags

Everything beyond the NLP path is off by default and lands in its own module.
The two QP families both name their types `QpProblem` / `QpSolution` /
`QpStatus`, so they cannot share one flat namespace.

| feature | module | covers |
|---|---|---|
| `convex` | `pounce_rs::convex` | LP, convex QP, SOCP / exponential / power / PSD cones, SOS; batched and warm-started solves; symbolic-factorization reuse; QP sensitivity |
| `qp` | `pounce_rs::qp`, `pounce_rs::sqp` | sparse **parametric active-set** QP, and the SQP working-set warm-start contract |
| `sensitivity` | `pounce_rs::sensitivity` | sIPOPT-style `∂x*/∂p` predictors, parametric warm starts, reduced Hessian |
| `full` | — | all three |

```toml
[dependencies]
pounce-rs = { version = "0.9", features = ["convex", "sensitivity"] }
```

Enabling a feature widens what the crate **exports**, not what it builds: the
default NLP path already compiles `pounce-qp`, `pounce-linsol`, and
`pounce-feral` transitively, so `qp` costs nothing at build time and only
`convex` and `sensitivity` add crates.

`convex` and `qp` also enable **`pounce_rs::linsol`**, which supplies the
sparse symmetric factorization those solvers take as an argument —
`backend()` for the default parallel FERAL factor, `serial_backend()` for the
inner-serial one used under an outer-parallel batch.

### Convex: LP, QP, and conic

```rust
use pounce_rs::convex::{QpOptions, QpProblem, QpStatus, Triplet, solve_qp_ipm};
use pounce_rs::linsol::backend;

// min ‖x‖² − 0.5·x0 − 1.5·x1  s.t.  x0 + x1 == 1,  0 ≤ x ≤ 5
let prob = QpProblem {
    n: 2,
    p_lower: vec![Triplet::new(0, 0, 2.0), Triplet::new(1, 1, 2.0)],
    c: vec![-0.5, -1.5],
    a: vec![Triplet::new(0, 0, 1.0), Triplet::new(0, 1, 1.0)],
    b: vec![1.0],
    g: vec![],
    h: vec![],
    lb: vec![0.0, 0.0],
    ub: vec![5.0, 5.0],
};

let sol = solve_qp_ipm(&prob, &QpOptions::default(), backend);
assert_eq!(sol.status, QpStatus::Optimal);
```

`P` is the **lower triangle** of the Hessian in triplet form; an empty `P` is
an LP. Cone blocks beyond the nonnegative orthant are declared with `ConeSpec`
and solved by `solve_socp_ipm`. For many instances, `solve_qp_batch_parallel`
runs one per rayon worker, and `QpFactorization` reuses the AMD ordering and
symbolic analysis across instances that share a sparsity pattern. See
[Convex Solver](convex-solver.md).

### Active-set QP and SQP warm starts

`pounce_rs::qp` is the parametric active-set engine — a different solver
family from the convex IPM, for sequences of nearby QPs, and it accepts an
indefinite Hessian. `pounce_rs::sqp` is the NLP-level counterpart: carrying a
working set from one SQP solve into the next. See
[Active-Set SQP & Warm Starts](active-set-sqp.md).

### Sensitivity

```rust
use pounce_rs::prelude::*;
use pounce_rs::sensitivity::SensSolve;

let result = SensSolve::new(vec![2, 3])      // pinned constraint rows
    .with_deltas(vec![-0.5, 0.0])            // Δp
    .with_reduced_hessian()
    .run(&mut app, tnlp);

let dx = result.dx.expect("populated when with_deltas was set");
```

A sensitivity-stage failure is reported through `result.error`, **not**
`result.status` — the underlying solve can converge while the post-solve step
fails. See [Sensitivity Analysis](sensitivity.md) and
[Sessions](sessions.md).

## Escape hatch

Each feature module also re-exports the crate behind it — `pounce_rs::convex`
re-exports `pounce_convex`, and so on — so anything outside the curated
surface stays reachable without adding a dependency. Reaching for it is a
signal the facade is missing something; those are worth
[filing](https://github.com/jkitchin/pounce/issues).

## See also

- [docs.rs/pounce-rs](https://docs.rs/pounce-rs) — the full API reference
- [Choosing a Solver](choosing-a-solver.md) — which solver fits which problem
- [Solver Options](options.md) — the option names shared by every frontend
- [Python API](python.md) — the same solvers from Python
