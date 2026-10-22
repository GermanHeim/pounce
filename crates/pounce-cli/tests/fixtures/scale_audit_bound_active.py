"""Regenerate `scale_audit_bound_active.nl` (the gh#983 scale audit's
declared-box defect) and its fixed-variable variant.

    python3 scale_audit_bound_active.py scale_audit_bound_active.nl
    python3 scale_audit_bound_active.py scale_audit_fixed_var.nl --fixed-var

`--fixed-var` adds `z` with the single-point box `[0.5, 0.5]` and an
objective term `(z - 0.2)^2` (so `f* = -2.91e-3`). Under the default
`fixed_variable_treatment=make_parameter` it is spliced in at its value;
under `relax_bounds` it stays in the iterate with its point widened into a
band, which is the branch of the re-solve's box handling that fixture exists
to reach.

Needs only Pyomo (its `.nl` writer is native -- no AMPL involved).

    min  1e-3 * ((y - 1)^2 - x1 - x2 - x3)
    s.t. y^3 + x1 + x2 + x3 >= 1.5
         0 <= x_i <= 1,  -10 <= y <= 10

The optimum is `x = (1, 1, 1)`, `y = 1`, `f* = -3e-3` exactly, and `-3e-3`
is also the infimum of `f` over the declared box -- no point inside the
bounds the model writes down can do better. Every `x_i` sits on its upper
bound with multiplier `1e-3`, equal to the objective's largest gradient, and
that is what reaches the defect: the objective is small, so the scale audit
re-solves it multiplied up by `~1e3`, which multiplies the bound multipliers
by the same factor and shrinks each bound's slack `mu / z` below the
`bound_relax_factor` widening (`1e-8`). Before the fix the promoted re-solve
reported `x_i = 1 + 4.97e-9` and `f = -3.0000000149e-3` -- below `f*`, an
objective the model cannot reach -- while the base attempt's point was inside
its box. The cubic keeps the model off the convex arms.
"""

import sys

import pyomo.environ as pe

fixed = "--fixed-var" in sys.argv[2:]
m = pe.ConcreteModel()
m.I = pe.RangeSet(0, 2)
m.x = pe.Var(m.I, bounds=(0, 1), initialize=0.5)
m.y = pe.Var(bounds=(-10, 10), initialize=0.0)
e = (m.y - 1) ** 2 - sum(m.x[i] for i in m.I)
if fixed:
    m.z = pe.Var(bounds=(0.5, 0.5), initialize=0.5)
    e = e + (m.z - 0.2) ** 2
m.obj = pe.Objective(expr=1e-3 * e)
m.c = pe.Constraint(expr=m.y**3 + sum(m.x[i] for i in m.I) >= 1.5)
m.write(sys.argv[1], io_options={"symbolic_solver_labels": False})
