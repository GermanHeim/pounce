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
