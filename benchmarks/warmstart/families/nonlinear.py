"""Nonlinear families: curvature and a genuinely nonconvex path.

- :class:`HangingChain` — the canonical spring-chain problem with a
  parabolic ground the chain rests on. Nonlinear inequality
  constraints; the active set is "which nodes touch the ground", and
  it changes as the ground rises and the anchor moves. Convex, but
  with real constraint curvature, so a warm start has to survive
  Jacobian *and* Hessian drift, not just a moving right-hand side.
- :class:`RosenbrockRing` — nonconvex objective inside a trust ball
  whose radius sweeps through the unconstrained solution's norm. The
  path crosses a single clean activation switch at its midpoint at
  every scale, which separates "warm start across an unchanged active
  set" from "warm start across the one step where it changes".
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from ..spec import Bounds, ParametricFamily

_INF = 1e20


class HangingChain(ParametricFamily):
    """Spring chain resting on a parabolic ground.

    Free nodes ``p_i = (x_i, y_i)``, ``i = 1..N``, with both ends
    pinned. Minimize spring energy plus gravitational potential
    subject to ``y_i ≥ ground(x_i)``, where

        ``ground(x) = g₀(θ) + q·(x − x_c)²``.

    The parameter moves the right anchor and raises the ground, so
    nodes make and break contact along the path. Variables carry no
    bounds — every active-set event comes from the nonlinear
    constraint rows.
    """

    name = "hanging_chain"
    tags = {"regime": "flipping", "channel": "mixed", "curvature": "convex"}
    n_steps = 20

    _N = 15  # free nodes
    _K = 40.0  # spring constant
    _W = 4.0  # node weight (m·g)
    _Q = 0.4  # ground curvature
    _XC = 1.5  # ground vertex
    _ANCHOR_Y = 1.0
    _DELTA_A = 0.02  # anchor drift per step
    _DELTA_G = 0.015  # ground rise per step
    _A0 = 2.5
    _G0 = -1.6

    def __init__(self):
        self._anchor_x = self._A0
        self._ground0 = self._G0

    # -- layout: z = [x_1, y_1, x_2, y_2, ...] ---------------------

    @property
    def n(self) -> int:
        return 2 * self._N

    @property
    def m(self) -> int:
        return self._N

    def _nodes(self, z):
        """(N+2, 2) array of all node positions, anchors included."""
        p = np.empty((self._N + 2, 2))
        p[0] = (0.0, self._ANCHOR_Y)
        p[1:-1, 0] = z[0::2]
        p[1:-1, 1] = z[1::2]
        p[-1] = (self._anchor_x, self._ANCHOR_Y)
        return p

    def bounds(self) -> Bounds:
        return Bounds(
            lb=np.full(self.n, -_INF),
            ub=np.full(self.n, _INF),
            cl=np.zeros(self._N),
            cu=np.full(self._N, _INF),
        )

    def cold_x0(self) -> np.ndarray:
        # Straight line between the anchors at the *initial* anchor
        # position — deliberately θ-independent, so every cold solve
        # along the path starts from the same place.
        t = np.arange(1, self._N + 1) / (self._N + 1)
        z = np.empty(self.n)
        z[0::2] = t * self._A0
        z[1::2] = self._ANCHOR_Y
        return z

    def set_theta(self, theta: np.ndarray) -> None:
        t = np.asarray(theta, dtype=float).ravel()
        self._anchor_x, self._ground0 = float(t[0]), float(t[1])

    def theta_path(self, scale: float) -> Optional[List[np.ndarray]]:
        return [
            np.array(
                [
                    self._A0 + scale * self._DELTA_A * k,
                    self._G0 + scale * self._DELTA_G * k,
                ]
            )
            for k in range(self.n_steps)
        ]

    # -- functions -------------------------------------------------

    def objective(self, z):
        p = self._nodes(z)
        d = np.diff(p, axis=0)
        return float(0.5 * self._K * np.sum(d * d) + self._W * np.sum(z[1::2]))

    def gradient(self, z):
        p = self._nodes(z)
        # dE/dp_i = K·(2p_i − p_{i−1} − p_{i+1}) for free nodes.
        g = self._K * (2.0 * p[1:-1] - p[:-2] - p[2:])
        g[:, 1] += self._W
        out = np.empty(self.n)
        out[0::2] = g[:, 0]
        out[1::2] = g[:, 1]
        return out

    def constraints(self, z):
        x, y = z[0::2], z[1::2]
        return y - (self._ground0 + self._Q * (x - self._XC) ** 2)

    def jacobian_dense(self, z):
        x = z[0::2]
        j = np.zeros((self._N, self.n))
        rows = np.arange(self._N)
        j[rows, 2 * rows] = -2.0 * self._Q * (x - self._XC)
        j[rows, 2 * rows + 1] = 1.0
        return j

    def hessian_dense(self, z, lagrange, obj_factor):
        n = self.n
        h = np.zeros((n, n))
        # Objective: tridiagonal in node index, decoupled per coordinate.
        for i in range(self._N):
            for c in (0, 1):
                idx = 2 * i + c
                h[idx, idx] += obj_factor * 2.0 * self._K
                if i + 1 < self._N:
                    nxt = 2 * (i + 1) + c
                    h[idx, nxt] += -obj_factor * self._K
                    h[nxt, idx] += -obj_factor * self._K
        # Constraints: −2q on each node's x-diagonal.
        for i in range(self._N):
            h[2 * i, 2 * i] += lagrange[i] * (-2.0 * self._Q)
        return h


class RosenbrockRing(ParametricFamily):
    """``min Rosenbrock(x)  s.t.  ‖x‖² ≤ r(θ)²`` with a sweeping radius.

    The unconstrained minimizer is ``x* = 1`` with ``‖x*‖ = √n``, so
    the ball constraint is active for ``r < √n`` and inactive above
    it. The path is centered on ``r = √n``: the switch always happens
    at the midpoint step, at every scale, and at that step the
    constraint is exactly weakly active. Nonconvex objective, so this
    is also the family where a warm start can land a solver in a
    different basin than a cold solve — which the correctness gate
    reports rather than hides.

    The cold start is the origin, not Rosenbrock's traditional
    ``(−1.2, 1, −1.2, …)``. That start is a *robustness* test rather
    than a warm-start test: from it, pounce's exact-Hessian SQP path
    gives up with ``Search_Direction_Becomes_Too_Small`` at every step
    of the path (the quasi-Newton modes converge from it fine), so
    every arm would be measuring failure instead of warm-start effect.
    From the origin both algorithms converge and agree at every step,
    which is what makes the switch measurable. See
    ``dev-notes/warm-start-benchmark.md`` for the failing case.
    """

    name = "rosenbrock_ring"
    tags = {"regime": "switch", "channel": "rhs", "curvature": "nonconvex"}
    n_steps = 21  # odd, so a step lands exactly on the switch

    _N = 10
    _DELTA = 0.05

    def __init__(self):
        self._r = float(np.sqrt(self._N))

    @property
    def n(self) -> int:
        return self._N

    @property
    def m(self) -> int:
        return 1

    def bounds(self) -> Bounds:
        return Bounds(
            lb=np.full(self._N, -5.0),
            ub=np.full(self._N, 5.0),
            cl=np.array([-_INF]),
            cu=np.array([self._r**2]),
        )

    def cold_x0(self) -> np.ndarray:
        return np.zeros(self._N)

    def set_theta(self, theta: np.ndarray) -> None:
        self._r = float(np.asarray(theta).ravel()[0])

    def theta_path(self, scale: float) -> Optional[List[np.ndarray]]:
        half = (self.n_steps - 1) // 2
        r_switch = float(np.sqrt(self._N))
        return [
            np.array([r_switch + scale * self._DELTA * (k - half)])
            for k in range(self.n_steps)
        ]

    def objective(self, x):
        a = x[1:] - x[:-1] ** 2
        b = 1.0 - x[:-1]
        return float(100.0 * (a @ a) + b @ b)

    def gradient(self, x):
        n = self._N
        g = np.zeros(n)
        a = x[1:] - x[:-1] ** 2
        g[:-1] += -400.0 * x[:-1] * a - 2.0 * (1.0 - x[:-1])
        g[1:] += 200.0 * a
        return g

    def constraints(self, x):
        return np.array([float(x @ x)])

    def jacobian_dense(self, x):
        return (2.0 * x).reshape(1, self._N)

    def hessian_dense(self, x, lagrange, obj_factor):
        n = self._N
        h = np.zeros((n, n))
        a = x[1:] - x[:-1] ** 2
        # d²/dx_i² from the i-th (i < n−1) Rosenbrock term, plus the
        # 200 from the (i−1)-th term's linear appearance in x_i.
        diag = np.zeros(n)
        diag[:-1] += -400.0 * a + 800.0 * x[:-1] ** 2 + 2.0
        diag[1:] += 200.0
        h[np.arange(n), np.arange(n)] = diag
        off = -400.0 * x[:-1]
        h[np.arange(n - 1), np.arange(1, n)] = off
        h[np.arange(1, n), np.arange(n - 1)] = off
        h *= obj_factor
        h[np.arange(n), np.arange(n)] += 2.0 * lagrange[0]
        return h
