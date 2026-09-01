# gh#873: the exhibition has to search *both* signs of the curvature direction,
# and has to score them rather than take the first that clears its bar.
#
#   min  -x0**2 + x1**2     s.t.  -2 <= x0 <= g,  -1 <= x1 <= 1,  x0 = x1 = 0
#
# (0, 0) is a first-order point and a maximum along x0. The curvature direction
# is +-e0, and the two signs are worth wildly different amounts:
#
#   +e0 walks to the near wall x0 = g   ->  f = -g**2   (tiny, but an improvement)
#   -e0 walks to the far wall  x0 = -2  ->  f = -4      (the global minimum)
#
# `exhibit_better_point` scans `[+1, -1]` in that order, so returning the first
# acceptable point returns `-g**2` and throws the global minimum away. Whether
# that happened used to depend on an *absolute* acceptance bar accidentally
# rejecting the small improvement -- which is a coincidence, not a design, and
# it inverts as soon as the bar is made scale-relative.
#
# g is chosen well clear of `constr_viol_tol` so the near bound is genuinely
# inactive and both signs are live; `sqp_exhibit_units.py` is the fixture that
# varies that distance instead.
#
# Regenerate:  python sqp_exhibit_signs.py 5e-2 sqp_exhibit_signs.nl
import sys

from pyomo.environ import ConcreteModel, Objective, Var

g = float(sys.argv[1])
name = sys.argv[2]

m = ConcreteModel()
m.x0 = Var(bounds=(-2.0, g), initialize=0.0)
m.x1 = Var(bounds=(-1.0, 1.0), initialize=0.0)
m.o = Objective(expr=-m.x0**2 + m.x1**2)
m.write(name, io_options={"symbolic_solver_labels": False})
