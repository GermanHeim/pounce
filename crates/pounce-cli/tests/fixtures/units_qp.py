# gh#872: one model in two systems of units, X = K*x.
#
#   min  0.5*(X0/K)**2 + 0.5*(X1/K)**2 + 5*(X0/K)*(X1/K)   s.t. -2K <= X <= 2K
#
# H = K**-2 * [[1, 5], [5, 1]], the box scales with K, and the minimum is
# obj = -16 at X = (2K, -2K) for every K. Only the units of X change -- metres
# to micrometres -- so a verdict that moves with K is a verdict that depends on
# the units the user chose. |lambda_min| / lambda_max = 2/3 at every scale, so
# the Hessian is strongly indefinite throughout and nothing here is anywhere
# near the round-off the PSD tolerances exist to absorb.
#
# Regenerate:  python units_qp.py 1 units_qp_k1.nl
#              python units_qp.py 1e5 units_qp_k1e5.nl
import sys

from pyomo.environ import ConcreteModel, Objective, Var

K = float(sys.argv[1])
name = sys.argv[2]

m = ConcreteModel()
m.X0 = Var(bounds=(-2 * K, 2 * K), initialize=0.0)
m.X1 = Var(bounds=(-2 * K, 2 * K), initialize=0.0)
m.o = Objective(
    expr=0.5 * (m.X0 / K) ** 2 + 0.5 * (m.X1 / K) ** 2 + 5 * (m.X0 / K) * (m.X1 / K)
)
m.write(name, io_options={"symbolic_solver_labels": False})
