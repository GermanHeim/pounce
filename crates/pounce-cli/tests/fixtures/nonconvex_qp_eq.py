"""gh #871 — a nonconvex QP whose negative curvature is hidden behind an
equality row.

    min  -x0**2
    s.t. x0 + x1 + x2 == 0
         x0 in [0, 1],  x1, x2 in [-1, 1]

The origin is first-order clean: the gradient vanishes, so `x0`'s lower bound
is *weakly* active and its multiplier is zero. The true minimum is `-1` at
`x0 = 1` (feasible with `x1 = -1, x2 = 0`), which ipopt and POUNCE's own NLP
arm both reach.

What makes it the fixture for #871 rather than another instance of #848: every
`QpProblem` in both #848 test files has `a: vec![]`, and `max_feasible_step`
opens with a hard rejection of any direction carrying an equality component.
The curvature search runs on `P`, so it returns `e0` -- and `A e0 = 1`. The
guard was defeated on a branch its corpus could not reach.

Regenerate with:  python nonconvex_qp_eq.py
"""

from pyomo.environ import ConcreteModel, Constraint, Objective, Var

m = ConcreteModel()
m.x0 = Var(bounds=(0, 1), initialize=0.0)
m.x1 = Var(bounds=(-1, 1), initialize=0.0)
m.x2 = Var(bounds=(-1, 1), initialize=0.0)
m.c = Constraint(expr=m.x0 + m.x1 + m.x2 == 0)
m.o = Objective(expr=-m.x0**2)
m.write("nonconvex_qp_eq.nl", io_options={"symbolic_solver_labels": False})
