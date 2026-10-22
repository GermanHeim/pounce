"""Regenerate `mu_fallback_declined_bound_active.nl`.

    python3 mu_fallback_declined_bound_active.py mu_fallback_declined_bound_active.nl

Needs only Pyomo. Rosenbrock in `(a, b)` beside three controls pushed onto
their upper bound:

    min  100 (b - a^2)^2 + (1 - a)^2 - x1 - x2 - x3
    s.t. a^2 + x1 + x2 + x3 >= 1.5,   0 <= x_i <= 1

Under `max_iter=11` the base attempt stops `Maximum_Iterations_Exceeded` with
the controls still inside their box, the mu-strategy fallback's retry
(adaptive) gets closer and also stops at the cap -- `9.46e-9` outside the
declared box, on the widened bound -- and is declined, so the base attempt's
point is restored. Before gh#983's declared-box fix the restore left the
retry's `final_declared_box_viol` behind: the report said `9.46e-9` beside a
point inside its box.
"""

import sys

import pyomo.environ as pe

m = pe.ConcreteModel()
m.I = pe.RangeSet(0, 2)
m.x = pe.Var(m.I, bounds=(0, 1), initialize=0.5)
m.a = pe.Var(initialize=-1.2)
m.b = pe.Var(initialize=1.0)
m.obj = pe.Objective(
    expr=100 * (m.b - m.a**2) ** 2 + (1 - m.a) ** 2 - sum(m.x[i] for i in m.I)
)
m.c = pe.Constraint(expr=m.a**2 + sum(m.x[i] for i in m.I) >= 1.5)
m.write(sys.argv[1], io_options={"symbolic_solver_labels": False})
