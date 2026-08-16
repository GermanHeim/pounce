# Initialization and Warm Starts

POUNCE is a local NLP solver: every solve starts from a point, and that
point often decides whether the solve takes 15 iterations or 150, or
whether it converges at all. This page collects the initialization
story in one place: where the starting point comes from on each
frontend, what the solver does with it (the part that surprises
people), how to warm-start each algorithm path, and how to diagnose a
bad start. The per-algorithm details live in their own pages; this is
the map.

## Where the starting point comes from

| Frontend | Primal starting point |
|---|---|
| Python `Problem.solve(x0=...)` | the `x0` argument |
| Python `minimize(fun, x0, ...)` | the `x0` argument |
| CLI / AMPL | the `.nl` file's initial-guess segment; zeros for variables without one |
| Pyomo | each `Var`'s `.value`, serialized into the `.nl` by Pyomo's writer |
| GAMS | variable levels (`x.L`) via GMO |
| Rust | `Nlp::new(problem).x0(&[...])`, or `TNLP::get_starting_point` |

Two silent-zero traps hide in that table:

* **Pyomo:** a `Var` whose `.value` was never set is written as `0`
  in the `.nl` file. A model initialized "nowhere" is actually
  initialized at the origin, which for many process models is outside
  every variable's meaningful range (and a domain error for `log`,
  `/`, and friends).
* **GAMS:** levels default to `0` unless assigned. Set `x.L` before
  the `solve` statement.

Dual estimates can be seeded too: `Problem.solve` accepts
`lagrange=`, `zl=`, `zu=` keyword arguments, and the `.nl` format
carries constraint-dual guesses when the modeling layer writes them.
Dual seeds are ignored unless you opt into a warm start (below). The
scipy-style `minimize` facade does not expose dual seeding; use
`pounce.Problem` directly when you need it.

## What the solver does with your point (cold start)

The default interior-point path ports Ipopt's iterate initializer
(`crates/pounce-algorithm/src/init/default.rs`). The sequence:

1. **The primal point is pushed into the interior of the bounds.**
   Per component, with bounds `lo <= x <= hi`:
   `p_l = min(bound_push * max(|lo|, 1), bound_frac * (hi - lo))`,
   likewise `p_u`, and `x` is clamped into `[lo + p_l, hi - p_u]`.
   One-sided bounds use the `bound_push` term alone; free variables
   are untouched. With the defaults (`bound_push = bound_frac =
   1e-2`), a variable sitting exactly on its lower bound `1.0` starts
   at `1.01` instead. Your point is honored *approximately*, and the
   deliberately-at-a-bound part of it is not honored at all. This is
   the single most common reason a "perfect" starting point does not
   behave like one.
2. **Slacks** are set to `s = d(x)` and pushed into the slack bounds
   the same way.
3. **Duals** get fixed defaults: constraint multipliers start at
   `y = 0` and are then replaced by a least-square estimate, unless
   that estimate exceeds `constr_mult_init_max` (in which case it is
   discarded and `y` stays at zero); bound multipliers are
   `z = v = bound_mult_init_val = 1.0`.
4. **The barrier parameter** starts at `mu_init = 0.1` (monotone
   `mu_strategy`, the default) regardless of how good your point is.

The knobs, all Ipopt-compatible, and all settable through every
frontend's option path (`Problem.solve(options={...})`, `pounce
model.nl bound_push=0.1`, an `ipopt.opt` line, `IpoptApplication::
options_mut`):

| Option | Default | Meaning |
|---|---|---|
| `bound_push` | `1e-2` | Absolute push off each bound (relative to `max(|bound|, 1)`). |
| `bound_frac` | `1e-2` | Cap on the push as a fraction of the bound interval. |
| `slack_bound_push` / `slack_bound_frac` | `1e-2` | Same, for inequality slacks. |
| `bound_mult_init_val` | `1.0` | Initial bound-multiplier value. |
| `bound_mult_init_method` | `constant` | `constant` is the only implemented mode; upstream's `mu-based` parses and is then refused rather than silently served as `constant`. |
| `constr_mult_init_max` | `1e3` | Cap on the least-square constraint-multiplier estimate; `0` keeps `y = 0`. |
| `least_square_init_primal` | `no` | Replace the starting `x` with the min-norm solution of the linearized constraints before the interior push — but only if that actually reduces the true nonlinear violation (see [Safeguarding the least-square start](#safeguarding-the-least-square-start)). |
| `mu_init` | `0.1` | Initial barrier parameter (monotone strategy). |
| `start_with_resto` | `no` | Jump straight into feasibility restoration at iteration 1 (aborts if the start is already feasible). |

An *infeasible* starting point is fine: the IPM does not require
feasibility, and `least_square_init_primal=yes` can cheaply reduce
iteration-0 infeasibility on mostly-linear models (the
`mehrotra_algorithm` LP/QP cascade turns it on for you, along with
more aggressive `bound_push` / `bound_frac` / `bound_mult_init_val`).
A point where a function *fails to evaluate* is not fine; see
[Diagnosing a bad start](#diagnosing-a-bad-start).

### Safeguarding the least-square start

The min-norm solution of the *linearized* constraints is a local model
step, not automatically a better starting point. Where the Jacobian is
small relative to the residual, the linearization asks for a very large
correction and the true nonlinear violation at the far end can be far
worse than where it started. On `x₀² + x₁² = 1` from `(0.05, 0.05)` the
Jacobian is `(0.1, 0.1)`, the linearized correction is about 7 units
long, and the violation at the far end is `48.5` against the `0.995` it
started with.

So the step is scored before it is taken. Writing `θ(x)` for the
unscaled max-norm nonlinear violation —
`max(‖c(x)‖∞, ‖max(d_l − d(x), d(x) − d_u, 0)‖∞)`, the same quantity the
CLI reports as the model's constraint violation — the initializer:

1. evaluates `θ₀` at your point, after the interior push;
2. computes the least-square direction `d = x_ls − x₀` once;
3. tries `α = 1, ½, ¼, …` (at most `least_square_init_max_trials`,
   default 4), pushing each candidate into the bound interior *before*
   measuring it, so the accepted merit is the merit of the point the
   algorithm will really start from;
4. accepts the first `α` with `θ(α) ≤ (1 − η·α)·θ₀`, where
   `η = least_square_init_accept_ratio` (default `1e-2`). The linear
   model predicts `θ → 0` at `α = 1`, so this is exactly "the actual
   feasibility reduction is at least `η` times the predicted one";
5. keeps your original `x` if no trial qualifies.

`least_square_init_max_trials` and `least_square_init_accept_ratio` are
fields on `DefaultIterateInitializer`, not registered options: unlike
every knob in the table above they are not settable from a frontend,
and setting them by name is rejected with `Unknown option`. They are
named here because the safeguard's behaviour is defined in terms of
them, not because you can tune them.

Each trial costs one constraint evaluation; none costs a Jacobian or a
KKT solve, because only the length of the step changes. A point that is
already feasible is left alone — no step can improve a violation of
zero.

The decision is readable after the solve:

```rust
if let Some(r) = app.least_square_init_report() {
    println!("{} -> {} (alpha {}, {} rejected, {})",
             r.violation_initial, r.violation_final,
             r.alpha, r.rejected_trials, r.termination);
}
```

## Warm-starting the interior-point path

From Python, the packaged form is one object:

```python
x, info = prob.solve(x0=x0)                  # cold solve
ws = pounce.WarmStart.from_info(x, info)     # captures x, duals, mu
x2, info2 = prob.solve(warm_start=ws)        # warm re-solve
ws.save("state.npz")                         # reuse across processes
```

`warm_start=` is accepted by `Problem.solve` and `pounce.minimize`,
seeds the primal and dual iterates, applies the enabling options
below, and forwards the SQP working set when the state was captured
from that path. The rest of this section is what it does under the
hood (and the only route from the CLI or an options file).

The enabling options are **scoped to the call**: they are installed for
that one solve and taken back afterwards, including when the solve
raises. A warm solve therefore never changes what the next ordinary
`solve` on the same `Problem` does. (Before pounce#607 it did, and the
cost was invisible: on HS071 an ordinary cold solve went from 17
iterations to 24 on a `Problem` that had served one warm solve, with the
same objective to ten digits.)

Passing a previous solution as `x0` is **not** a warm start by
itself. The IPM warm start is a package of three things, and skipping
any one of them silently degrades to (roughly) a cold solve:

1. **Opt in and seed the duals.** Set `warm_start_init_point=yes` and
   pass the previous multipliers.
2. **Lower `mu_init`.** The default `0.1` makes the solver walk the
   barrier schedule down from scratch even when started at the
   optimum. Seed it near the converged complementarity (e.g. `1e-7`
   after a `tol=1e-8` solve). Since #606 this is a *floor*: the solver
   measures the point you supplied and raises `mu` above it if the
   point cannot support that barrier (see below).
3. **Tighten the warm-start pushes.** The warm initializer applies
   its own interior clamp with `warm_start_bound_push` / `_frac`
   (default `1e-3`), which shoves an at-the-bound solution back off
   its bounds. Tighten them to keep the point.

```python
x, info = make_problem().solve(x0=x0_cold)      # cold solve

warm = make_problem()
warm.add_option("warm_start_init_point", "yes")
warm.add_option("mu_init", 1e-7)
for k in ("warm_start_bound_push", "warm_start_bound_frac",
          "warm_start_slack_bound_push", "warm_start_slack_bound_frac",
          "warm_start_mult_bound_push"):
    warm.add_option(k, 1e-9)

x2, info2 = warm.solve(
    x0=x,
    lagrange=np.asarray(info["mult_g"]),
    zl=np.asarray(info["mult_x_L"]),
    zu=np.asarray(info["mult_x_U"]),
)
```

On HS071 this takes the re-solve from 11 iterations to 5, while
`warm_start_init_point=yes` alone saves nothing; the full runnable
comparison is `python/examples/hs071_warm_start.py`. On the CLI the
same options apply as `KEY=VALUE` pairs, with dual seeds coming from
the `.nl` file's dual segment when present.

| Option | Default | Meaning |
|---|---|---|
| `warm_start_init_point` | `no` | Master switch: honor supplied primal *and* dual seeds. |
| `warm_start_bound_push` / `warm_start_bound_frac` | `1e-3` | Interior clamp used instead of `bound_push` / `bound_frac`. |
| `warm_start_slack_bound_push` / `warm_start_slack_bound_frac` | `1e-3` | Same, for slacks. |
| `warm_start_mult_bound_push` | `1e-3` | Floor on seeded bound multipliers (a carried-in `z = 0` must not start on the barrier's boundary). |
| `warm_start_mult_init_max` | `1e6` | Cap on seeded equality multipliers. |
| `warm_start_recentering` | `residual` | Reconstruct the multiplier blocks the caller did not supply, and raise `mu` when the supplied point cannot support it. `none` restores the pre-#606 constants. |

### What the solver does with a partial warm start

Two things about the list above are worth knowing before you tune it.

**You cannot seed every multiplier block.** `TNLP::get_starting_point`
— which is what `lagrange` / `zl` / `zu` reach — carries the equality
multipliers and the *variable*-bound multipliers. The interior-point
method also needs a multiplier for each inequality row's slack
(`v_L` / `v_U` internally), and there is no field for those on any
frontend. On every warm start ever run they arrived as zero and were
floored at `warm_start_mult_bound_push`.

**A constant is the wrong fill.** `warm_start_mult_bound_push` is a
number chosen with no reference to the slacks it is paired against, so
the "warm" point it produces is not a stationary point of anything.

Under `warm_start_recentering=residual` (the default since #606) the
initializer instead rebuilds what it was not given:

- a bound-multiplier entry that arrives as exactly `0` (or `NaN`) is
  not a legal barrier multiplier, so it was never a seed; it is
  re-derived from the stationarity identity
  `P_L z_L − P_U z_U = ∇f + J_c^T y_c + J_d^T y_d` (and its slack-block
  twin `P_L v_L − P_U v_U = −y_d`), floored at `μ / slack` so an
  inactive bound still gets the value complementarity implies;
- an equality-multiplier block that is identically zero goes through
  the same regularized least-squares augmented solve the cold path
  uses, now with real bound multipliers in its right-hand side;
- `mu` is raised to the point's measured average complementarity when
  that exceeds `mu_init` **by more than a factor of ten**, so a stale
  seed gets a looser barrier instead of being trusted while a merely
  imperfect one keeps the barrier it asked for. Moving `mu` reroutes
  the whole trajectory, so a near miss is not worth what the reroute
  costs. The measurement is clamped to `[1e-11, 0.1]`; `mu_init`
  itself is not, so an explicit setting outside that band is a floor
  and is never capped. The primal and dual residuals deliberately do
  **not** move `mu`: a warm point at a moved parameter carries both by
  construction, and reacting to them discards the warm start to pay for
  a Newton step that was about to happen anyway.

`warm_start_target_mu`, when set, still pins `mu` outright.

What happened is reported back. From Python it is `info["warm_start"]`:

```python
x2, info2 = warm.solve(x0=x, zl=..., zu=...)
info2["warm_start"]
# {'primal_residual': 1.6e-09, 'dual_residual': 3.5e-10,
#  'complementarity': 4.2e-09, 'mu_in': 2.5e-09, 'mu_out': 4.2e-09,
#  'bound_duals': 'reconstructed', 'eq_duals': 'accepted',
#  'bound_duals_reconstructed': 1, 'recentering_disabled': False}
```

From Rust it is `IpoptApplication::warm_start_diagnostics()`. At
`print_level=5` the iteration line carries `wz` (bound multipliers
rebuilt), `wy` (equality multipliers rebuilt), `wy0` (a reconstruction
was discarded) and `wmu` (the barrier was loosened).

### Two options that are refused

`warm_start_same_structure` and `warm_start_entire_iterate` are
registered — an `ipopt.opt` written for Ipopt parses unchanged — but
both name Ipopt's `TNLP::GetWarmStartIterate` surface, which pounce
does not expose. Setting either to `yes` used to parse, set a field
nothing read, and change nothing at all. Since #606 it fails with a
message instead. `warm_start_init_point=yes` is the supported route
and carries the primal point and every multiplier block the TNLP
surface has.

### Which model does this warm start belong to?

A warm start is a point in *one* model's variable space, with
multipliers in that model's constraint space. Replay it against a model
whose variables have been reordered, whose bounds have moved, or which
is simply a different model of the same shape, and the arrays are still
the right *length* — so nothing objects. What comes back is a wrong
answer, or the right answer down a much longer trajectory.

Pass `problem=` when you capture, and the object records a
**signature** of the model as well: dimensions, the bound signature, the
declared sparsity, the scaling convention, the algorithm/backend, and
the model-defining options.

```python
ws = pounce.WarmStart.from_info(x, info, problem=prob)
ws.save("state.npz")

# ... later, possibly in another process
ws = pounce.WarmStart.load("state.npz")
x2, info2 = prob.solve(warm_start=ws)     # checked before the solver runs
```

A mismatch is refused *before the solver is entered*, with a report
naming every facet that moved:

```text
warm start is not compatible with this problem (1 mismatch,
exact-structure replay, schema v2):
  - bounds: captured '51e5c8cd33c97b92', target '42ae305673e91939'
resolve it by one of:
  - re-capture against this problem: WarmStart.from_info(x, info, problem=prob)
  - transfer it explicitly: ws.transfer(prob, mapper) or, with stable IDs
    on both sides, ws.reindex(prob)
  - assert it transfers as-is: ws.migrate(prob)
  - downgrade the check: compat='warn' or compat='unsafe'
```

`compat` picks how hard that is enforced — `"strict"` (the default)
raises, `"warn"` emits the same report as a warning and proceeds,
`"unsafe"` skips the comparison. Set it on the object, on `load()`, or
per call: `prob.solve(warm_start=ws, compat="warn")`.
`ws.describe_compatibility(prob)` returns the report as a string without
raising, which is the dry run for a replay you are unsure of.

One structural change a fingerprint cannot see is a **reordering**:
permuting a model with a uniform box and a dense jacobian leaves every
digest bit-identical. Ordering is knowledge only you have, so name it:

```python
ws = pounce.WarmStart.from_info(x, info, problem=prob,
                                var_ids=names, con_ids=con_names)
...
prob2.solve(warm_start=ws, var_ids=names_in_prob2_order)   # refused
```

### Transferring a warm start: horizon shifts and reindexing

When the model *has* changed and you know how, say so. `transfer()`
takes a mapper and produces a **mapped** replay — labelled as such, and
still refused on any problem other than the one it was mapped to:

```python
def shift(ctx):                       # ctx: source, target, problem
    m = ctx.index_map("var")          # target-indexed source positions, -1 = new
    return {"x": ..., "lagrange": ..., "zl": ..., "zu": ...}

moved = ws.transfer(next_prob, shift, var_ids=next_ids, con_ids=next_con_ids)
```

With stable IDs on both sides, `reindex` writes that mapper for you —
entries the target shares with the source move to their new positions,
entries only the target has are left *unseeded* (`NaN`, which the warm
initializer reads as "you decide") rather than fabricated:

```python
moved = ws.reindex(next_prob, var_ids=next_ids, con_ids=next_con_ids)
x, info = next_prob.solve(warm_start=moved)
```

That covers both cases: a reordering, where the ID sets are equal, and a
receding horizon, where they overlap. Note what it does *not* buy you —
a transferred interior-point start is about validity, not speed. On a
slew-limited tracking model the mapped point costs 12 iterations against
7 for a cold solve; on a longer sinusoidal track the gap widens with the
horizon (12 vs 9 at horizon 5, 30 vs 10 at horizon 40). That is the same
barrier/active-set limit described just below, and the reason the SQP
path exists.

### Artifacts written before pounce#607

Archives from earlier releases carry no signature. They are
*unverifiable*, not incompatible, so they still load and still replay;
what you get is one `WarmStartLegacyWarning` and a dimension check
(the only facet their own arrays witness). Two ways to clear it:

```python
ws = pounce.WarmStart.load("old.npz")      # warns on replay
ws = ws.migrate(prob)                      # re-sign it against this problem
ws.save("old.npz")                         # ... and it is a v2 artifact now
```

`migrate` is an **assertion**, not a conversion: it re-signs the arrays
without touching them, so use it only when they really do belong to this
problem. When they need rearranging, that is `reindex` / `transfer`. An
unsigned warm start held only in memory — `from_info(x, info)` with no
`problem=` — behaves exactly as it always has, and says nothing.

Even a well-executed IPM warm start has a structural limit: the
barrier pushes iterates off the bounds, so the active-set information
in a converged solution cannot be fully exploited. When you are
solving a *sequence* of related NLPs (MPC steps, branch-and-bound
nodes, homotopy paths), that limit is the reason the active-set SQP
path exists.

## Warm-starting the active-set SQP path

With `algorithm=active-set-sqp`, the warm-start payload is different:
alongside the primal/dual seeds it carries the **working set** (which
bounds and constraints are active), and an unchanged working set means
the next solve converges in a handful of QP iterations.

```python
prob.add_option("algorithm", "active-set-sqp")

ws = None
for k in range(horizon_steps):
    x, info = prob.solve(x0=x_prev, working_set=ws)
    ws = info["working_set"]
    x_prev = x
```

The two paths' warm-start inputs are deliberately path-local: the
IPM-side options above (`warm_start_init_point`, `mu_init`,
`bound_push`, ...) are silently ignored on the SQP path, and
`working_set=` is ignored on the IPM path. Details, the
`classify_working_set` helper for reconstructing a working set from
multipliers, and the GAMS `sqp_state_file` / marginal-based routes are
in [Active-Set SQP & Warm Starts](active-set-sqp.md). Note the GAMS
warm-start features currently live in the native C link only, not the
pip link (see [GAMS](gams.md)).

## Sequences of solves: batch chaining and sessions

For MPC chains, parametric sweeps, and B&B node relaxations from
Python, `solve_nlp_batch` packages the whole IPM warm-start recipe
for you:

```python
results = pounce.solve_nlp_batch(batch_t)                   # cold
results = pounce.solve_nlp_batch(batch_t1, warms=results)   # warm
```

Each instance is seeded with the previous primal and duals, the
converged `mu` is threaded into `mu_init`, and
`warm_start_init_point=yes` is forced; see
[Batched NLP solving](python.md#batched-nlp-solving-solve_nlp_batch).
For post-solve sensitivity queries against the converged KKT factor
(a different kind of reuse, no re-solve at all), see
[Sessions](sessions.md). JAX users get warm-start hand-off along a
parameter trajectory via `JaxProblem`; see
[the Python guide](python.md).

## Diagnosing a bad start

The first stop is the preflight check, which evaluates the model once
at its starting point (no solve) and reports everything this page has
warned about: NaN/inf evaluations, bound violations, how far the
interior clamp will move the point, initial constraint violation, and
derivative scale spread.

```sh
pounce check-x0 model.nl              # text report; --json for tools
pounce check-x0 model.nl --x0-file candidate.txt
```

```python
report = pounce.preflight(problem_obj, x0, lb=lb, ub=ub, cl=cl, cu=cu)
print(report)          # report.fatal, report.warnings, report.to_dict()
```

Exit code 0 means the model evaluates cleanly at x0 (warnings allowed);
21 means a solve from this point would abort. The other diagnostics:

* **`Invalid_Number_Detected`** means an evaluator returned NaN/inf,
  and the very first evaluation at the starting point is the usual
  culprit (`log(0)` or a division at an all-zeros default start).
  The interior clamp only repairs bound violations; it cannot fix
  domain errors on free variables. Move the start into the domain,
  or add bounds that keep the clamp inside it.
* **`derivative_test=first-order`** runs the derivative checker at
  the starting point; wrong derivatives look exactly like a bad
  start (immediate restoration, tiny steps).
* **The [interactive debugger](debugger.md)** (`--debug`) breaks at
  iteration 0, so you can inspect the initial objective, `inf_pr`,
  and `inf_du` before a single step is taken, and `resolve` from an
  edited iterate.
* **Presolve** (`presolve=yes`) reports structural trouble that no
  starting point can fix, like rank-deficient equality blocks
  (LICQ check), and its bound tightening shrinks the box the
  interior clamp places you in. See
  [Troubleshooting Recipes](troubleshooting.md) and [FBBT](fbbt.md).
* **`pounce-studio analyze-nl`** gives a structural pre-flight of a
  model file without solving.

## No good starting point at all?

Three composable primitives cover the "generate or repair a point"
workflows from Python:

```python
# N diverse starts (the sampler behind find_minima): sobol / uniform /
# jitter / bounds midpoint. Feed them to solve_nlp_batch or race them.
starts = pounce.generate_starts(16, bounds=bounds, seed=0)

# Safeguarded sparse elastic repair of a candidate onto the constraints
# + bounds (the standalone form of least_square_init_primal). Never
# returns a point whose true nonlinear violation is worse than the one
# you gave it; pass return_report=True for the diagnostics.
x0 = pounce.project_to_feasible(problem_obj, x0, lb=lb, ub=ub, cl=cl, cu=cu)
x0, rep = pounce.project_to_feasible(problem_obj, x0, lb=lb, ub=ub,
                                     cl=cl, cu=cu, return_report=True)
# rep.violation_initial / .violation_final / .step_norm /
# .rejected_trials / .elastic_total / .termination

# Cheap tournament: a few iterations from each start, ranked; continue
# the winner at full effort with a WarmStart.
best = pounce.race_starts(fun, starts, bounds=bounds, iters=10)[0]
res = pounce.minimize(fun, best.x,
                      warm_start=pounce.WarmStart.from_info(best.x, best.info))
```

### Racing starts: the successive-halving ladder

`race_starts` used to spend the same budget on every candidate, from a
cold start, and rank the field once at the end. That kept most of
multistart's cost — the candidate that was hopeless after two iterations
was still charged for ten — and threw away the solver state between
rounds. Since pounce#610 the default policy is an adaptive
**successive-halving ladder**:

1. every candidate runs for a small budget;
2. the field is ranked on five signals (below);
3. the weakest fraction is eliminated;
4. the survivors are **resumed from their held solver state** with a
   budget `eta` times larger, and the ladder repeats.

The winner ends up with about the effort `iters` would have given it
under the old policy, so the two are comparable on quality; what changes
is what the losers cost.

```python
best, race = pounce.race_starts(fun, starts, jac=jac, bounds=bounds,
                                constraints=cons, iters=20,
                                return_report=True)
print(race.report())
# race: policy=halving eta=3 candidates=16 rungs=2
#   rung 0: budget=37 evals entrants=16 -> survivors=7 spent=530 evals / 112 iters (0 resumed, 16 started)
#       - #10: duplicate of candidate 6 (scaled distance 0.000606 <= 0.001)
#       - #14: below halving cut (rank 7 of 15, keep 6)
#       - #5: below halving cut (rank 8 of 15, keep 6)
#       ... seven more
#   rung 1: budget=111 evals entrants=7 -> survivors=7 spent=272 evals / 61 iters (7 resumed, 0 started)
#   total 834 evals / 173 iters, 7 resumes
```

(HS71, 16 Sobol starts, `iters=20` — the `hs71` row of the benchmark
table below. The fixed policy spends 259 iterations on the same field.)

`RaceReport` carries the per-rung resource spend and a reason for every
candidate's exit; `RaceCandidate` carries each one's evaluations,
iterations, resumes, restoration calls and final residuals.

**What "resumed" means, precisely.** POUNCE has no API for suspending an
IPM mid-iteration and re-entering the same algorithm object — every
`Solver.solve` builds its application afresh. What a pause carries is the
whole interior-point *iterate*: the primal point, the constraint
multipliers, both bound-multiplier blocks, and the barrier parameter μ,
replayed through the warm-start machinery above so that pounce#606's
recentering measures the point it is actually handed. That is materially
not a cold restart. Measured on the `rastrigin_eq` fixture in
`python/tests/test_starts_racing.py`:

| paused at | resumed (state + point) | restarted (point only) |
|---|---|---|
| 3 iterations | **32 iters** / 330 evals | 43 iters / 368 evals |
| 5 iterations | **17 iters** / 250 evals | 43 iters / 376 evals |
| 8 iterations | **0 iters** / 80 evals | 43 iters / 372 evals |

Both arms start from the identical iterate and reach the identical
objective, start for start. The last row is the clearest: by 8
iterations every candidate has converged, the resumed solve recognises
it immediately because the carried duals and μ satisfy the convergence
check on entry, and the restarted solve — handed the same point and
nothing else — needs 5 to 8 iterations each to re-derive the same
certificate.

The size of that gap is model-dependent. On HS71 the same comparison is
a wash (98/92/77 iterations resumed against 102/87/79 restarted), which
is the regime pounce#608 warns about: a warm-started IPM often converges
in one iteration per step, and where it does a resume has nothing left
to remove. What a pause does *not* carry is the filter history and the
line-search state; that would need a `Solver.resolve()`, which does not
exist yet.

**Ranking.** Eliminations are decided on a weighted sum of five
rank-normalized signals — rank-normalized so that a violation in mol/s
and a dimensionless KKT residual can be combined without an invented
scale factor:

| signal | what it reads | default weight |
|---|---|---|
| `violation` | how infeasible the iterate is now | 3.0 |
| `feasibility_progress` | how much of its *initial* violation it has removed | 1.0 |
| `kkt` | the scaled first-order residual, in log units | 1.5 |
| `objective_progress` | objective removed per evaluation spent, damped while infeasible | 1.0 |
| `health` | restoration share, non-finite objective, failed exit | 1.0 |

Feasibility carries the most weight because an infeasible candidate's
objective is not a number about the problem being solved. Pass
`weights=` to re-balance. Diversity is protected two ways: survivors
within `cluster_tol` of each other in scaled units are collapsed to the
best of the group, and `explore` candidates from *outside* the cut are
retained anyway, chosen farthest-first from those already kept.

**Evaluations, not iterations, are the resource.** Rung 0 has no
evaluation budget — it *is* the calibration — and every later rung's
budget is a multiple of what rung 0 actually cost. Each candidate then
converts its remaining budget into an iteration cap through *its own*
measured evaluations-per-iteration, so a candidate whose iterations are
expensive (a dozen line-search trials, a restoration excursion) is
granted fewer of them for the same resource. A cumulative iteration
ceiling rising to `iters` bounds the other side.

**When not to use it.** A rung boundary costs a fresh solver application
and a re-evaluation at the seed. On a model where that fixed cost is a
large fraction of the whole solve — one variable, no constraints, a
handful of evaluations per iteration — the ladder cuts iterations but
comes out level or slightly up on evaluations. Measured over
`benchmarks/scripts/race_starts_bench.py` (six multi-basin models × three
field sizes): **17.9% fewer** user-callable evaluations overall with no
quality regression anywhere, ranging from **43.8% fewer** on HS71 with 27
starts to **5.5% more** on the two-variable `himmelblau_disc` with 16.
Iterations fall in every one of the eighteen configurations. Where the
ladder does not pay, `policy="fixed"` is the pre-#610 policy, kept
verbatim and reproducing its old answers exactly:

```python
best = pounce.race_starts(fun, starts, bounds=bounds, iters=10,
                          policy="fixed")
```

`policy="halving"` runs on the NLP path only — it holds a
`pounce.Solver` session per candidate, which is what a pause suspends —
and refuses a non-`"nlp"` `solver_selection` rather than silently losing
the session it needs.

When the model has many local minima and you want *all* of them (or a
managed search rather than a tournament), the
[global search drivers](find-minima.md) (`multistart`, `mlsl`,
`deflation`, `flooding`, `tunneling`, `basinhopping`) manage
populations of starting points and warm-start bookkeeping for you,
from Python (`pounce.find_minima`) or the CLI (`--minima`).
