"""Degeneracy the ratio test has to survive, not just weak activity.

`degenerate_corner` in :mod:`.quadratic` covers *dual* degeneracy — a
multiplier passing exactly through zero, so strict complementarity
fails and a sign-based working-set classifier has no signal. That is
one of the three ways an active-set QP meets degeneracy, and it was the
only one this suite exercised.

The two families here cover the other two, both of which are properties
of the *constraint geometry* rather than of the multipliers:

- :class:`RedundantRows` — the active constraints are linearly
  dependent (LICQ fails). Exact-duplicate equality rows make the
  Jacobian rank-deficient everywhere; a duplicated inequality pair
  activates together partway along the path, so the rank deficiency
  *arrives* as an event rather than sitting there from the start. The
  solver has to prune the active set to a maximal independent subset
  and keep going.
- :class:`DegenerateVertex` — many more constraints are tight at the
  solution than there are variables (12 rows, n = 4), so the ratio test
  is a mass of ties and the vertex is the classic cycling risk that
  Harris / GMSW EXPAND exist to handle. The path then walks the
  solution *off* that vertex, which is where the ties actually have to
  be broken consistently.

Both are convex QPs, so all three solvers can take them.
"""

from __future__ import annotations

from itertools import combinations
from typing import List, Optional

import numpy as np

from ..spec import Bounds, ParametricFamily

_INF = 1e20


class RedundantRows(ParametricFamily):
    """``min ½‖x − a(θ)‖²`` over a constraint set that violates LICQ.

    The rows, in order:

    ===  =====================  ====================================
    row  constraint             why it is here
    ===  =====================  ====================================
    0    ``x₀ + x₁ = 1``        an ordinary equality
    1    ``2x₀ + 2x₁ = 2``      its exact duplicate — the equality
                                block is rank-deficient at every
                                point, LICQ never holds
    2    ``x₂ + x₃ ≤ b(θ)``     activates partway along the path
    3    ``3x₂ + 3x₃ ≤ 3b(θ)``  its exact multiple, so the two
                                activate *together* and the active
                                set becomes dependent as an event
    4    ``x₄ + x₅ ≤ 2``        slack throughout, a control row
    ===  =====================  ====================================

    A solver that prunes the active set to a maximal linearly
    independent subset handles this; one that factors the raw active-set
    KKT hits a singular system. The multipliers on a duplicated pair are
    not unique, which is also why the suite's correctness gate checks
    objectives rather than multiplier values.
    """

    name = "redundant_rows"
    quadratic = True
    tags = {"regime": "rank-deficient", "channel": "objective", "curvature": "convex"}
    n_steps = 21

    _N = 6
    _DELTA = 0.03

    def __init__(self):
        self._A = np.array(
            [
                [1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [2.0, 2.0, 0.0, 0.0, 0.0, 0.0],  # duplicate of row 0
                [0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 3.0, 3.0, 0.0, 0.0],  # 3× row 2
                [0.0, 0.0, 0.0, 0.0, 1.0, 1.0],
            ]
        )
        # The inequality pair binds at 0.4; a(θ) sweeps x₂+x₃ through it.
        self._b_ineq = 0.4
        self._a = self._a_c()

    def _a_c(self) -> np.ndarray:
        return np.array([0.5, 0.5, 0.2, 0.2, 0.3, 0.3])

    def _direction(self) -> np.ndarray:
        # Only the x₂/x₃ block moves, and it moves *through* the
        # duplicated inequality pair's boundary at the midpoint step.
        return np.array([0.0, 0.0, 1.0, 1.0, 0.0, 0.0])

    @property
    def n(self) -> int:
        return self._N

    @property
    def m(self) -> int:
        return 5

    def bounds(self) -> Bounds:
        cl = np.array([1.0, 2.0, -_INF, -_INF, -_INF])
        cu = np.array([1.0, 2.0, self._b_ineq, 3.0 * self._b_ineq, 2.0])
        return Bounds(
            lb=np.zeros(self._N),
            ub=np.full(self._N, _INF),
            cl=cl,
            cu=cu,
        )

    def cold_x0(self) -> np.ndarray:
        return np.full(self._N, 0.3)

    def set_theta(self, theta: np.ndarray) -> None:
        t = float(np.asarray(theta).ravel()[0])
        self._a = self._a_c() + t * self._direction()

    def theta_path(self, scale: float) -> Optional[List[np.ndarray]]:
        half = (self.n_steps - 1) // 2
        # θ = 0 puts x₂ + x₃ exactly on the duplicated pair's boundary.
        return [
            np.array([scale * self._DELTA * (k - half)])
            for k in range(self.n_steps)
        ]

    def objective(self, x):
        d = x - self._a
        return 0.5 * float(d @ d)

    def gradient(self, x):
        return x - self._a

    def constraints(self, x):
        return self._A @ x

    def jacobian_dense(self, x):
        return self._A

    def hessian_dense(self, x, lagrange, obj_factor):
        return obj_factor * np.eye(self._N)


class DegenerateVertex(ParametricFamily):
    """A vertex where 12 rows are tight in 4 variables, then a walk off it.

    The feasible set is the nonnegative orthant in ``R⁴`` written three
    times over: the four rows ``−xⱼ ≤ 0``, the six pairwise rows
    ``−(xᵢ + xⱼ) ≤ 0``, and two triple rows. Every one of them is tight
    at the origin and all but four are redundant, so the origin is a
    vertex with **three times more active constraints than variables** —
    primal degeneracy of the kind Harris's two-pass test and GMSW EXPAND
    exist to survive, and where a naive ratio test can take a zero step
    and cycle.

    ``a(θ)`` starts deep in the polar cone (every component negative), so
    the projection is pinned at that degenerate vertex with all 12 rows
    active, and sweeps to strictly positive, where the solution is
    ``a(θ)`` itself and nothing is active. The interesting steps are in
    between: the active set has to shed rows in groups, and every drop
    is a tie among redundant candidates.
    """

    name = "degenerate_vertex"
    quadratic = True
    tags = {"regime": "primal-degenerate", "channel": "objective", "curvature": "convex"}
    n_steps = 21

    _N = 4
    _DELTA = 0.06

    def __init__(self):
        rows = [-np.eye(self._N)[j] for j in range(self._N)]
        for i, j in combinations(range(self._N), 2):
            r = np.zeros(self._N)
            r[i] = r[j] = -1.0
            rows.append(r)
        for trip in ((0, 1, 2), (1, 2, 3)):
            r = np.zeros(self._N)
            for j in trip:
                r[j] = -1.0
            rows.append(r)
        self._A = np.array(rows)  # 4 + 6 + 2 = 12 rows, rank 4
        self._a = self._a_at(0.0)

    def _a_at(self, t: float) -> np.ndarray:
        # t < 0: deep in the polar cone (solution pinned at the vertex).
        # t > 0: inside the feasible cone (solution = a, nothing active).
        base = np.array([1.0, 0.8, 1.2, 0.9])
        return t * base

    @property
    def n(self) -> int:
        return self._N

    @property
    def m(self) -> int:
        return 12

    def bounds(self) -> Bounds:
        return Bounds(
            lb=np.full(self._N, -_INF),
            ub=np.full(self._N, _INF),
            cl=np.full(12, -_INF),
            cu=np.zeros(12),
        )

    def cold_x0(self) -> np.ndarray:
        return np.full(self._N, 0.1)

    def set_theta(self, theta: np.ndarray) -> None:
        self._a = self._a_at(float(np.asarray(theta).ravel()[0]))

    def theta_path(self, scale: float) -> Optional[List[np.ndarray]]:
        half = (self.n_steps - 1) // 2
        # θ = 0 is the fully degenerate step: a = 0, every row tight,
        # every multiplier free to be zero. It lands exactly on a step
        # at every scale.
        return [
            np.array([scale * self._DELTA * (k - half)])
            for k in range(self.n_steps)
        ]

    def objective(self, x):
        d = x - self._a
        return 0.5 * float(d @ d)

    def gradient(self, x):
        return x - self._a

    def constraints(self, x):
        return self._A @ x

    def jacobian_dense(self, x):
        return self._A

    def hessian_dense(self, x, lagrange, obj_factor):
        return obj_factor * np.eye(self._N)
