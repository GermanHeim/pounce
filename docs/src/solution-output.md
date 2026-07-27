# Solution Output

## The `.sol` file

Following the AMPL solver convention, solving a positional `.nl` file
writes a sibling `<stub>.sol` next to it — `pounce problem.nl`
produces `problem.sol`. The file carries the primal `x` and dual
`lambda` blocks plus an `objno` line with the AMPL `solve_result_num`,
so AMPL (or any `.sol` reader) can pull the solution back:

```sh
pounce problem.nl                       # writes problem.sol
pounce problem.nl --sol-output out.sol  # write to an explicit path
pounce problem.nl --no-sol              # skip the .sol write
```

A `.sol` is written even when the solve fails, so the
`solve_result_num` is always recoverable. Built-in problems
(`--problem …`) have no `.nl` stub, so they only produce a `.sol`
when `--sol-output` is given explicitly.

## Reading `solve_result_num`

The `objno` line carries an AMPL `solve_result_num` (Gay 2005, *Hooking Your
Solver to AMPL* §5). Consumers key on the **band**, not the exact number:

| Band | Meaning |
|---|---|
| `0`–`99` | solved |
| `100`–`199` | solved, with a warning |
| `200`–`299` | infeasible |
| `300`–`399` | unbounded |
| `400`–`499` | limit reached (iterations, time) |
| `500`–`599` | failure |

Pyomo maps each band to a `TerminationCondition`, so anything in `200`–`299`
arrives as `TerminationCondition.infeasible`.

### Infeasible: proved vs. local

Within the infeasible band POUNCE distinguishes *how* it knows:

| Code | Verdict | What it means |
|---|---|---|
| `200` | `InfeasibleProblemDetected` | The solver converged to a point of **local** infeasibility — a stationary point of the constraint violation with the violation bounded away from zero. |
| `201` | `... (detected by presolve: …)` | Presolve's bound propagation / interval arithmetic found the feasible region empty before any iteration. |

The difference is real, not cosmetic. `201` is a *structural* detection made on
the model's bounds before iterating, not a certified proof — it is subject to
the same floating-point limits as any interval computation, and is withheld
whenever the violation is smaller than the feasibility tolerance. `200` is
different in kind — on a nonconvex problem a positive local minimum of the
violation does **not** rule out a feasible point elsewhere, which is why the
console message says "Problem may be infeasible."

When the region is found empty the solve is skipped entirely and the message
names how it was found, so the claim is checkable:

```text
POUNCE 0.9.0: InfeasibleProblemDetected (detected by presolve: bound propagation)
objno 0 201
```

`201` requires [presolve](auxiliary-presolve.md) to be enabled (`presolve=yes`);
it is off by default. A presolve-derived infeasibility is only reported when the
contradiction holds on the *original* box — one produced by presolve's own
auxiliary elimination is re-checked after rollback and never certified.

## Choosing an output format

| You want… | Use |
|---|---|
| AMPL / Pyomo to read the result back | the `.sol` file (default) |
| A structured, schema-versioned report for tooling | `--json-output` (see [JSON Solve Report](json-output.md)) |
| Just the console summary | `--no-sol` |

The `.sol` and JSON outputs are not exclusive — you can request both
in the same run.
