"""STEENBRF transcribed from CUTEst mastsif STEENBRF.SIF (Ph. Toint, June 1990).

A totally separable nonconvex multi-commodity network problem
(Steenbrink, "Optimization of Transport Networks", Wiley 1974, p.124).
468 variables, 108 linear equality constraints.

    python3 steenbrf.py steenbrf.nl           # STEENBRF
    python3 steenbrf.py steenbrb.nl --as-b    # STEENBRB, the control

This is the control behind the `steenbrf` half of gh #524 (see
`dev-notes/issue-524-cresc4-steenbrf.md`): the corpus file the issue reports on
is NOT this problem. Vanderbei's `steenbrf` prices only commodities 11 and 12,
leaving 360 of its 468 variables out of the objective entirely; this one prices
all twelve, as the SIF does. Same network, same data, same start point —
53 iterations here against 2570 there.

The control is what makes that claim checkable rather than an assertion.
`STEENBRB.SIF` is byte-identical to `STEENBRF.SIF` except for one line —
`AM LA(4) LA(4) 0.5`, halving the investment cost of arc 4 — so `--as-b`
un-halves that single coefficient and changes nothing else. It reproduces
`9075.8553865777394`, against the `9075.855` recorded for `steenbrb` in
`benchmarks/vanderbei/cute_table_status.json` and the `SOLTN 9098.9319884` in
STEENBRB.SIF's own header: every digit those sources carry. Vanderbei's own
`steenbrb.nl` has since confirmed it outright — his file and this one solve to
the same objective in the same 49 iterations.

So the transcription is right, and mastsif STEENBRF has an optimum near 8991
(POUNCE solves it cleanly, 53 iterations, no stall). The corpus `steenbrf`
optimum is 282.678 — below a floor this model provably cannot reach. Dropping
the non-negative congestion term leaves a pure min-cost multicommodity-flow LP,
which POUNCE solves at 8250.0, and the capacity term adds at least 1.64. So
282.678 is not a second local minimum of this model; it is unreachable by it,
which is what makes "different problem" a measurement rather than a guess.
"""

import sys

from pyomo.environ import (
    ConcreteModel, Var, Objective, Constraint, RangeSet, sqrt, minimize,
)

NARCS = 18
NTRIPS = 12

COST = {
    1: 35.0, 2: 40.0, 3: 30.0, 4: 100.0, 5: 15.0, 6: 55.0,
    7: 100.0, 8: 25.0, 9: 60.0, 10: 35.0, 11: 55.0, 12: 15.0,
    13: 40.0, 14: 60.0, 15: 25.0, 16: 30.0, 17: 50.0, 18: 50.0,
}
ALPH, TZERO, CCR, NONZ = 0.01, 0.01, 0.01, 0.01
MNONZ = -NONZ

CMD = {i: 0.0 + NONZ for i in COST}          # minimal capacities
CMR = {i: 0.0 + NONZ for i in COST}
LA = {i: COST[i] * ALPH for i in COST}
LT = {i: COST[i] * TZERO for i in COST}
LC = {i: COST[i] * CCR for i in COST}
LA[4] *= 0.5                                  # half investment cost for arc 4

# CMIN of the IIJ elements: SHIFT = CMD(i) + MNONZ  (= 0.0 here)
SHIFT_D = {i: CMD[i] + MNONZ for i in COST}
SHIFT_R = {i: CMR[i] + MNONZ for i in COST}

# Flow-conservation rows: node -> list of (arc, sign_on_D).  R carries -sign.
NODE_ROWS = {
    1: [(1, -1), (2, -1), (3, -1)],
    2: [(4, -1), (5, -1), (6, -1), (1, +1)],
    3: [(7, -1), (8, -1), (9, -1), (2, +1)],
    4: [(10, -1), (11, -1), (12, -1), (4, +1)],
    5: [(13, -1), (14, -1), (15, -1), (7, +1)],
    6: [(16, -1), (10, +1), (13, +1)],
    7: [(17, -1), (3, +1), (5, +1), (8, +1)],
    8: [(18, -1), (6, +1), (9, +1), (11, +1), (14, +1), (17, +1)],
    9: [(12, +1), (15, +1), (16, +1), (18, +1)],
}

# CONSTANTS section: (node, trip) -> rhs
RHS = {
    (2, 1): -2000.0, (3, 1): 2000.0,
    (2, 2): -2000.0, (4, 2): 2000.0,
    (2, 3): -1000.0, (5, 3): 1000.0,
    (3, 4): -1000.0, (4, 4): 1000.0,
    (3, 5): -2000.0, (5, 5): 2000.0,
    (4, 6): -1000.0, (5, 6): 1000.0,
    (3, 7): -200.0, (2, 7): 200.0,
    (4, 8): -200.0, (2, 8): 200.0,
    (5, 9): -100.0, (2, 9): 100.0,
    (4, 10): -100.0, (3, 10): 100.0,
    (5, 11): -200.0, (3, 11): 200.0,
    (5, 12): -100.0, (4, 12): 100.0,
}


def build():
    m = ConcreteModel()
    m.A = RangeSet(1, NARCS)
    m.T = RangeSet(1, NTRIPS)
    m.N = RangeSet(1, 9)

    # START POINT: 'DEFAULT' 0.1 for every variable.
    m.cd = Var(m.A, initialize=0.1, bounds=(CMD[1], None))
    m.cr = Var(m.A, initialize=0.1, bounds=(CMR[1], None))
    m.d = Var(m.T, m.A, initialize=0.1, bounds=(0.0, None))
    m.r = Var(m.T, m.A, initialize=0.1, bounds=(0.0, None))

    def xt(flow, cap, lt, lc):          # XT element
        return lt * flow + lc * flow**3 / cap**2

    def obj(m):
        total = 0.0
        for k in m.A:
            fd = sum(m.d[i, k] for i in m.T)
            fr = sum(m.r[i, k] for i in m.T)
            total += xt(fd, m.cd[k], LT[k], LC[k])
            total += xt(fr, m.cr[k], LT[k], LC[k])
            total += LA[k] * sqrt(m.cd[k] - SHIFT_D[k])     # IIJ element
            total += LA[k] * sqrt(m.cr[k] - SHIFT_R[k])
        return total

    m.obj = Objective(rule=obj, sense=minimize)

    def flow(m, n, i):
        return (
            sum(s * m.d[i, k] - s * m.r[i, k] for k, s in NODE_ROWS[n])
            == RHS.get((n, i), 0.0)
        )

    m.con = Constraint(m.N, m.T, rule=flow)
    return m


if __name__ == "__main__":
    if "--as-b" in sys.argv[2:]:
        LA[4] *= 2.0                              # undo the STEENBRF-only halving
    build().write(sys.argv[1], io_options={"symbolic_solver_labels": True})
