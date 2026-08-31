# gh#873 D2: a genuinely indefinite model whose whole objective lives below the
# absolute acceptance bar.
#
#   min  k*x0*x1   s.t.  x0 + x1 == 2,  -9 <= x <= 9
#
# The reduced Hessian on the null space of the equality is -k: as indefinite at
# k = 1e-30 as at k = 1. The true minimum is -63k, at the corner (9, -7); the
# stationary point (1, 1) is the constrained *maximum*, worth +k.
#
# Two absolute floors used to sit between the two. `ev[0] >= -1e-8 * h_scale`
# with `h_scale = ...max(1.0)` stopped seeing the curvature below k ~ 1e-8, and
# `f_trial < f_curr - 1e-10` rejected every genuine improvement below k ~ 1e-10
# as round-off. Either alone returns the maximum.
#
# `sqp_tiny_objective_convex.py` is the paired negative control: the fix must
# not degenerate into "refute anything small".
#
# Regenerate:  python sqp_tiny_objective.py 1e-20 sqp_tiny_objective_k1em20.nl
import sys

from pyomo.environ import ConcreteModel, Constraint, Objective, Var

k = float(sys.argv[1])
name = sys.argv[2]

m = ConcreteModel()
m.x0 = Var(bounds=(-9.0, 9.0), initialize=1.0)
m.x1 = Var(bounds=(-9.0, 9.0), initialize=1.0)
m.c = Constraint(expr=m.x0 + m.x1 == 2.0)
m.o = Objective(expr=k * m.x0 * m.x1)
m.write(name, io_options={"symbolic_solver_labels": False})
