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

## Choosing an output format

| You want… | Use |
|---|---|
| AMPL / Pyomo to read the result back | the `.sol` file (default) |
| A structured, schema-versioned report for tooling | `--json-output` (see [JSON Solve Report](json-output.md)) |
| Just the console summary | `--no-sol` |

The `.sol` and JSON outputs are not exclusive — you can request both
in the same run.
