"""Rigorous lower bound on the CUTEst STEENBRF/STEENBRB objective (gh #524).

obj = sum_k [ LT(k)*(FD+FR) + LC(k)*(FD^3/cd^2 + FR^3/cr^2) + LA(k)*(sqrt(cd)+sqrt(cr)) ]

Every term is >= 0, so dropping the cubic congestion term gives a valid lower
bound; the capacity term is bounded below by its own bound cd,cr >= 0.01.
What is left is a pure min-cost multicommodity flow LP, which POUNCE solves at
8250.0; the capacity term adds at least 1.64. So the CUTEst model cannot go
below ~8251.6.

That is what rules out the alternative explanation for the corpus `steenbrf`
optimum of 282.678 — that it is simply a different local minimum of the same
nonconvex problem. It is not reachable at all, so the corpus file is a
different model. Run from this directory (it imports `steenbrf.py`):

    python3 floor_bound.py            # STEENBRB coefficients
    python3 floor_bound.py --as-f     # STEENBRF (LA[4] halved)
    pounce floor_lp.nl --no-sol       # -> 8250.0000000
"""
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import steenbrf as S
from pyomo.environ import (
    ConcreteModel, Var, Objective, Constraint, RangeSet, minimize, SolverFactory,
)

half_la4 = "--as-f" in sys.argv          # STEENBRF halves LA(4); STEENBRB does not
LA = dict(S.LA)
if not half_la4:
    LA[4] *= 2.0

m = ConcreteModel()
m.A, m.T, m.N = RangeSet(1, 18), RangeSet(1, 12), RangeSet(1, 9)
m.d = Var(m.T, m.A, initialize=0.1, bounds=(0.0, None))
m.r = Var(m.T, m.A, initialize=0.1, bounds=(0.0, None))

m.obj = Objective(
    expr=sum(S.LT[k] * sum(m.d[i, k] + m.r[i, k] for i in m.T) for k in m.A),
    sense=minimize,
)
m.con = Constraint(m.N, m.T, rule=lambda m, n, i: (
    sum(s * m.d[i, k] - s * m.r[i, k] for k, s in S.NODE_ROWS[n])
    == S.RHS.get((n, i), 0.0)))

m.write("floor_lp.nl", io_options={"symbolic_solver_labels": True})
cap_floor = sum(LA[k] * 2.0 * (0.01 ** 0.5) for k in LA)
print(f"capacity-term floor (cd=cr=0.01): {cap_floor:.4f}")
