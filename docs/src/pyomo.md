# Pyomo

Because POUNCE speaks the AMPL NL/SOL protocol, it drops into
[Pyomo](https://www.pyomo.org/) through the AMPL Solver Library
interface — exactly how Pyomo drives Ipopt.

The [`pyomo-pounce`](https://github.com/jkitchin/pounce/tree/main/pyomo-pounce)
package registers `pounce` as a Pyomo `SolverFactory` solver:

```python
import pyomo_pounce  # registers 'pounce'
from pyomo.environ import ConcreteModel, Var, Objective, SolverFactory

model = ConcreteModel()
model.x = Var(bounds=(-10, 10))
model.obj = Objective(expr=(model.x - 3) ** 2)

solver = SolverFactory('pounce')
solver.solve(model)
```

Options pass through the usual Pyomo mechanism:

```python
solver.solve(model, options={'tol': 1e-10, 'max_iter': 500})
```

Under the hood, Pyomo writes the model to an AMPL `.nl` file, invokes
`pounce problem.nl -AMPL`, and reads the result back from the `.sol`
file. See [Running Solves](cli.md) for the `-AMPL` solver mode.

## Preflight and initialization

A `Var` whose `.value` was never set is written as **0** into the
`.nl` file, so an uninitialized model actually starts at the origin
(see [Initialization and Warm Starts](initialization.md)). The package
ships three helpers for exactly this:

```python
import pyomo_pounce

report = pyomo_pounce.preflight(model)   # what will POUNCE see at x0?
print(report)                            # unset vars, bound/constraint
if report.fatal:                         # violations, NaN/inf evaluations
    ...

pyomo_pounce.initialize_missing_values(model)   # bounds-aware fill
                                                # (midpoint / one unit
                                                # inside / zero)

rep = pyomo_pounce.block_initialize(model)      # experimental: solve the
                                                # equality system's square
                                                # blocks in DM order and
                                                # write the values back
```

`preflight` evaluates every active constraint and the objective at the
current values with unset values treated as 0 (exactly what the NL
writer sends), restores the model untouched, and reports what
iteration 0 will see; `report.fatal` means the solve would abort with
`Invalid_Number_Detected`.

`block_initialize` is IDAES-flavored initialization without hand-written
routines: it takes the active equality constraints, finds the square
part of the incidence graph (Dulmage-Mendelsohn, via
`pyomo.contrib.incidence_analysis`), and solves the diagonal blocks in
topological order — 1x1 blocks by Newton, larger blocks as square
subsystem solves with POUNCE — filling `Var.value` along the way.
Variables in the under- or over-determined parts are left untouched;
follow with `initialize_missing_values` to fill the remainder. Fix a
variable to treat it as a known input.
