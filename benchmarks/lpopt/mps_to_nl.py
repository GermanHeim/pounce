#!/usr/bin/env python3
"""Convert an MPS (LP) or QPS (QP) file to an AMPL .nl file.

Parses the model with HiGHS (highspy), which reads fixed/free MPS and the
QPS quadratic-objective extension, then rebuilds it as a Pyomo ConcreteModel
and writes .nl via Pyomo's ASL writer -- the same .nl pipeline used by
benchmarks/large_scale/generate_nl.py.

Model convention (matches MPS/QPS and HiGHS):
    minimize   0.5 x' Q x + c' x + offset
    subject to row_lower <= A x <= row_upper
               col_lower <= x   <= col_upper

Usage:
    mps_to_nl.py INPUT.mps OUTPUT.nl
"""
import sys
import math

import highspy
from pyomo.environ import ConcreteModel, Var, Objective, Constraint, minimize, Reals
from pyomo.opt import ProblemFormat

INF = highspy.kHighsInf


def _bound(v):
    """Map a HiGHS +/-inf bound to None (unbounded) for Pyomo."""
    if v <= -INF or v == -math.inf:
        return None
    if v >= INF or v == math.inf:
        return None
    return float(v)


def convert(inp, out):
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    status = h.readModel(inp)
    # readModel returns a HighsStatus; kError is fatal.
    if str(status).endswith("kError"):
        raise RuntimeError(f"HiGHS failed to read {inp}: {status}")

    model = h.getModel()
    lp = model.lp_
    n = lp.num_col_
    m = lp.num_row_
    cost = list(lp.col_cost_)
    clo = list(lp.col_lower_)
    cup = list(lp.col_upper_)
    rlo = list(lp.row_lower_)
    rup = list(lp.row_upper_)
    offset = float(lp.offset_)
    maximize = str(lp.sense_).endswith("kMaximize")

    # A in compressed form. HiGHS MPS load is column-wise (CSC):
    #   for column j, entries a_start_[j] .. a_start_[j+1]-1
    #   index_[k] = row, value_[k] = coeff.
    A = lp.a_matrix_
    start = list(A.start_)
    index = list(A.index_)
    value = list(A.value_)

    # Build per-row coefficient lists from the CSC arrays.
    rows = [dict() for _ in range(m)]
    for j in range(n):
        for k in range(start[j], start[j + 1]):
            rows[index[k]][j] = rows[index[k]].get(j, 0.0) + value[k]

    # Optional quadratic objective (QP). Hessian is CSC of the full Q
    # (HiGHS stores the symmetric matrix); objective uses 0.5 x'Qx.
    hess = model.hessian_
    qdim = getattr(hess, "dim_", 0)
    quad = {}
    if qdim and qdim > 0:
        qstart = list(hess.start_)
        qindex = list(hess.index_)
        qvalue = list(hess.value_)
        for j in range(qdim):
            for k in range(qstart[j], qstart[j + 1]):
                quad[(qindex[k], j)] = qvalue[k]

    mdl = ConcreteModel()
    mdl.N = range(n)
    mdl.x = Var(mdl.N, domain=Reals)
    for j in range(n):
        lo, up = _bound(clo[j]), _bound(cup[j])
        mdl.x[j].setlb(lo)
        mdl.x[j].setub(up)

    sign = -1.0 if maximize else 1.0  # emit as a minimization

    def obj_rule(mm):
        e = offset + sum(cost[j] * mm.x[j] for j in range(n) if cost[j] != 0.0)
        if quad:
            e = e + 0.5 * sum(q * mm.x[i] * mm.x[j] for (i, j), q in quad.items())
        return sign * e

    mdl.obj = Objective(rule=obj_rule, sense=minimize)

    def con_rule(mm, r):
        coeffs = rows[r]
        if not coeffs:
            return Constraint.Skip
        body = sum(c * mm.x[j] for j, c in coeffs.items())
        lo, up = _bound(rlo[r]), _bound(rup[r])
        if lo is not None and up is not None:
            if lo == up:
                return body == lo
            return (lo, body, up)
        if lo is not None:
            return body >= lo
        if up is not None:
            return body <= up
        return Constraint.Skip

    mdl.con = Constraint(range(m), rule=con_rule)

    mdl.write(out, format=ProblemFormat.nl,
              io_options={"symbolic_solver_labels": False})
    return n, m, len(quad) > 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("usage: mps_to_nl.py INPUT.mps OUTPUT.nl")
    n, m, isqp = convert(sys.argv[1], sys.argv[2])
    print(f"wrote {sys.argv[2]}: n={n} m={m} {'QP' if isqp else 'LP'}")
