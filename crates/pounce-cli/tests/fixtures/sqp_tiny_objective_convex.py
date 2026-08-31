# gh#873: the negative control for `sqp_tiny_objective.py`, and the same lesson
# gh#872's `units_qp_convex.py` carries -- lowering a floor is only a fix if
# what sits below it is still judged on its merits.
#
#   min  k*(x0**2 + x1**2)   s.t.  x0 + x1 == 2,  -9 <= x <= 9
#
# Strictly convex at every k, minimum 2k at (1, 1). The reduced Hessian on the
# null space of the equality is +4k, which after gh#873 is compared against a
# floor that scales with k -- so it stays positive at k = 1e-20 rather than
# being swamped. A fix that merely lowered the thresholds without keeping them
# relative would start refuting this point, and the SQP arm would report a
# "better" point that does not exist.
#
# Regenerate:  python sqp_tiny_objective_convex.py 1e-20 sqp_tiny_objective_convex.nl
import sys

from pyomo.environ import ConcreteModel, Constraint, Objective, Var

k = float(sys.argv[1])
name = sys.argv[2]

m = ConcreteModel()
m.x0 = Var(bounds=(-9.0, 9.0), initialize=0.0)
m.x1 = Var(bounds=(-9.0, 9.0), initialize=0.0)
m.c = Constraint(expr=m.x0 + m.x1 == 2.0)
m.o = Objective(expr=k * (m.x0**2 + m.x1**2))
m.write(name, io_options={"symbolic_solver_labels": False})
