"""Regenerate `cresc4.nl` (gh #524).

CRESC4 is a CUTE problem (Ph. Toint, June 1993, after J.P. Rasson): find the
smallest-area crescent containing a set of points in the plane. Transcribed
from CUTEst `mastsif/CRESC4.SIF`, which is the authority for the model, the
bounds and the start point.

    python3 cresc4.py cresc4.nl

Needs only Pyomo (its `.nl` writer is native — no AMPL involved).

Why this file is committed alongside the `.nl`: the fixture's whole value is
that it is a faithful transcription of a real published problem rather than a
shape built to reproduce a symptom, and that claim is only checkable if the
transcription is readable. See `dev-notes/issue-524-cresc4-steenbrf.md`.

Geometry, in the SIF's parametrisation:

    circle 1: center (v1, w1),                         radius r1 = a*d + r
    circle 2: center (v1, w1) + a*d*(cos t, sin t),    radius r2 = d + r

Every data point must lie inside circle 2 (`is2`) and outside circle 1
(`os1`); the objective is the crescent's area — circle 2 minus the lens where
the two overlap — which is the SIF's `SC` element, transcribed term for term
below. The known optimum is 0.8718976 (LOQO, SNOPT and Ipopt-MA57 all agree;
the SIF header records 0.87189692).

ENCODING WARNING. POUNCE's verdict on this problem depends on the surface
encoding, which is the point of gh #524: with the constraints declared
`is2`-first and the radius term written on the right-hand side (what this file
emits) POUNCE on defaults reports local infeasibility, and declaring `os1`
first or folding the radius into the constraint body flips it back to a clean
solve. Six of the twelve encodings tried fail. Changing the declaration order
or the `<=` form below therefore changes what the fixture tests.
"""

import sys

from pyomo.environ import (
    ConcreteModel, Var, Objective, Constraint, RangeSet, acos, cos, sin, minimize,
)

#: The four data points to be enclosed (SIF `X1..X4` / `Y1..Y4`).
PTS = [(1.0, 0.0), (0.0, 1.0), (0.0, -1.0), (0.5, 0.0)]


def build():
    m = ConcreteModel()
    # Bounds and initial values are the SIF's BOUNDS / START POINT blocks.
    m.v1 = Var(initialize=-40.0)
    m.w1 = Var(initialize=5.0)
    m.d = Var(initialize=1.0, bounds=(1.0e-8, None))
    m.a = Var(initialize=2.0, bounds=(1.0, None))
    m.t = Var(initialize=1.5, bounds=(0.0, 6.2831852))
    m.r = Var(initialize=0.75, bounds=(0.39, None))

    # --- SIF element SC: the crescent's area. ---
    r2 = m.d + m.r                      # radius of the enclosing circle
    r1 = m.a * m.d + m.r                # radius of the excluded circle
    sep = m.a * m.d                     # distance between the two centers
    e = 2.0 * r2 * sep
    p = 2.0 * r1 * sep
    h = (sep * sep + r1 * r1 - r2 * r2) / p
    ell = -(sep * sep - r1 * r1 + r2 * r2) / e
    m.obj = Objective(
        expr=r2 * r2 * acos(ell) - r1 * r1 * acos(h) + 0.5 * e * sin(acos(ell)),
        sense=minimize,
    )

    m.I = RangeSet(0, len(PTS) - 1)

    def is2(m, i):  # inside circle 2
        x, y = PTS[i]
        return (m.v1 + m.a * m.d * cos(m.t) - x) ** 2 + (
            m.w1 + m.a * m.d * sin(m.t) - y
        ) ** 2 <= (m.d + m.r) ** 2

    def os1(m, i):  # outside circle 1
        x, y = PTS[i]
        return (m.v1 - x) ** 2 + (m.w1 - y) ** 2 >= (m.a * m.d + m.r) ** 2

    # Declaration order is load-bearing — see the ENCODING WARNING above.
    m.is2 = Constraint(m.I, rule=is2)
    m.os1 = Constraint(m.I, rule=os1)
    return m


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "cresc4.nl"
    build().write(out, io_options={"symbolic_solver_labels": True})
