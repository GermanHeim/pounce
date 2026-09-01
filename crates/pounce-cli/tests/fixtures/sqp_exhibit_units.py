# gh#873 D2: the same model in two systems of units, u = S*x.
#
#   min  -(u0/S)**2 + (u1/S)**2   s.t. -2S <= u0 <= 5e-6*S,  -S <= u1 <= S
#
# The objective *values* and the optimum are the same for every S: obj = -4 at
# u0 = -2S. Only the units of the variables change. But the active-set test in
# `negative_curvature_at_kkt_point` compared the distance to a bound against
# `constr_viol_tol`, an absolute distance in x units -- so shrinking the units
# moves a genuinely inactive bound *into* the active set, freezing x0 into the
# null space and closing the only direction of negative curvature the model has.
#
# Measured at S = 1e-1 and below the verdict flipped to 0.0, the constrained
# maximum, while S = 1 and S = 1e2 returned -4.
#
# Regenerate:  python sqp_exhibit_units.py 1e-2 sqp_exhibit_units_s1em2.nl
import sys

from pyomo.environ import ConcreteModel, Objective, Var

S = float(sys.argv[1])
name = sys.argv[2]

m = ConcreteModel()
m.u0 = Var(bounds=(-2.0 * S, 5e-6 * S), initialize=0.0)
m.u1 = Var(bounds=(-1.0 * S, 1.0 * S), initialize=0.0)
m.o = Objective(expr=-((m.u0 / S) ** 2) + (m.u1 / S) ** 2)
m.write(name, io_options={"symbolic_solver_labels": False})
