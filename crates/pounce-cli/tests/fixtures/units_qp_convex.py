# gh#872: a genuinely convex QP at ||H||inf ~ 1e-10.
#
# The corpus this fix landed in had 48 of 49 classifier-reaching fixtures at
# ||H||inf >= 1, where `psd_band`'s .min(1.0) clamp makes the change a no-op --
# so an empty sweep diff said almost nothing (gh#690/#760's lesson). This is the
# other branch: a matrix the tightened band must keep certifying PSD, so a fix
# that simply rejected everything small would move this line off `cvx-qp`.
#
# H = K^-2 * [[1, 0.2], [0.2, 1]], eigenvalues (1 +/- 0.2)/K^2 -- positive
# definite at every K, with |lambda_min|/lambda_max = 0.667 mirrored from the
# indefinite fixture so the two differ only in the sign of the off-diagonal.
from pyomo.environ import ConcreteModel, Objective, Var

K = 1e5
m = ConcreteModel()
m.X0 = Var(bounds=(-2 * K, 2 * K), initialize=0.0)
m.X1 = Var(bounds=(-2 * K, 2 * K), initialize=0.0)
m.o = Objective(
    expr=0.5 * (m.X0 / K) ** 2
    + 0.5 * (m.X1 / K) ** 2
    + 0.2 * (m.X0 / K) * (m.X1 / K)
    - 3.0 * (m.X0 / K)
    + 1.0 * (m.X1 / K)
)
m.write("units_qp_convex.nl", io_options={"symbolic_solver_labels": False})
