"""Quadratic families: clean active-set behavior, no curvature confounds.

These three isolate the *active set* as the only thing that changes
along the path. The objective is quadratic and the constraints are
linear, so second derivatives are constant and a Newton method solves
each instance in a handful of iterations from a good point — which is
exactly what makes them a sharp measurement of warm-start quality: any
iteration a warm-started solve spends here is spent finding the active
set, not chasing curvature.

- :class:`SimplexProjection` — objective-channel perturbation, active
  set flips one component at a time (projection onto the simplex).
- :class:`MovingBoundQP` — bound-channel perturbation, monotonically
  growing active set as a wall sweeps across the solution.
- :class:`DegenerateCorner` — built so a multiplier passes exactly
  through zero at the midpoint of the path, at every scale. This is
  the case a sign-based working-set classifier is documented to be
  lossy on (docs/src/active-set-sqp-warm-start.md), so it is a
  correctness probe rather than a speed probe.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from ..spec import Bounds, ParametricFamily

_INF = 1e20


class SimplexProjection(ParametricFamily):
    """``min ½‖x − p‖²  s.t.  Σx = 1, x ≥ 0`` with drifting ``p``.

    Solution is the water-filling projection ``x_i = max(p_i − λ, 0)``,
    so the active set is "which components are clamped at zero". The
    path drifts each component of ``p`` at a different rate, so
    components cross the water level one at a time: at ``tiny`` scale
    the active set never moves, at ``large`` it churns.
    """

    name = "simplex_proj"
    quadratic = True
    tags = {"regime": "flipping", "channel": "objective", "curvature": "convex"}
    n_steps = 20

    _N = 20
    _DELTA = 0.02  # natural per-step drift

    def __init__(self):
        self._p = self._p0()

    def _p0(self) -> np.ndarray:
        return np.linspace(-0.4, 0.6, self._N)

    def _direction(self) -> np.ndarray:
        # Different components drift at different rates and signs, so
        # crossings of the water level are spread along the path
        # rather than happening all at once.
        return np.cos(2.0 * np.pi * np.arange(self._N) / self._N)

    @property
    def n(self) -> int:
        return self._N

    @property
    def m(self) -> int:
        return 1

    def bounds(self) -> Bounds:
        return Bounds(
            lb=np.zeros(self._N),
            ub=np.full(self._N, _INF),
            cl=np.array([1.0]),
            cu=np.array([1.0]),
        )

    def cold_x0(self) -> np.ndarray:
        return np.full(self._N, 1.0 / self._N)

    def set_theta(self, theta: np.ndarray) -> None:
        self._p = np.asarray(theta, dtype=float).copy()

    def theta_path(self, scale: float) -> Optional[List[np.ndarray]]:
        p0, d = self._p0(), self._direction()
        return [p0 + scale * self._DELTA * k * d for k in range(self.n_steps)]

    def objective(self, x):
        d = x - self._p
        return 0.5 * float(d @ d)

    def gradient(self, x):
        return x - self._p

    def constraints(self, x):
        return np.array([float(x.sum())])

    def jacobian_dense(self, x):
        return np.ones((1, self._N))

    def hessian_dense(self, x, lagrange, obj_factor):
        return obj_factor * np.eye(self._N)


class MovingBoundQP(ParametricFamily):
    """Equality-constrained QP whose lower bounds sweep upward.

    ``min ½xᵀQx + cᵀx  s.t.  Ax = b,  x ≥ ℓ(θ)``

    ``Q`` is tridiagonal SPD and ``c`` is set so the bound-free
    solution is a fixed reference profile ``x_ref``; the lower bound
    wall then rises through that profile, activating components one
    after another. The perturbation is in the **bound** channel — the
    objective and the constraint rows never move — which is the case
    where an interior-point warm start is at its most awkward (the
    previous solution sits exactly where the new bound is) and an
    active-set method should shine.
    """

    name = "moving_bound_qp"
    quadratic = True
    tags = {"regime": "flipping", "channel": "bounds", "curvature": "convex"}
    n_steps = 20

    _N = 40
    _M = 3
    _DELTA = 0.05

    def __init__(self):
        n = self._N
        i = np.arange(n)
        # Tridiagonal SPD Hessian (discrete 1-D Laplacian + shift).
        self._Q = (
            np.diag(np.full(n, 2.2))
            + np.diag(np.full(n - 1, -1.0), 1)
            + np.diag(np.full(n - 1, -1.0), -1)
        )
        self._x_ref = np.sin(2.0 * np.pi * i / n)
        # Deterministic dense equality rows.
        self._A = np.cos(
            np.outer(np.arange(1, self._M + 1), i) * 0.7
        )
        self._b = self._A @ self._x_ref
        self._c = -(self._Q @ self._x_ref)
        self._shape = 0.3 * np.cos(3.0 * np.pi * i / n) - 1.2
        self._theta = 0.0

    @property
    def n(self) -> int:
        return self._N

    @property
    def m(self) -> int:
        return self._M

    def bounds(self) -> Bounds:
        return Bounds(
            lb=self._theta + self._shape,
            ub=np.full(self._N, _INF),
            cl=self._b.copy(),
            cu=self._b.copy(),
        )

    def cold_x0(self) -> np.ndarray:
        return np.zeros(self._N)

    def set_theta(self, theta: np.ndarray) -> None:
        self._theta = float(np.asarray(theta).ravel()[0])

    def theta_path(self, scale: float) -> Optional[List[np.ndarray]]:
        return [
            np.array([scale * self._DELTA * k]) for k in range(self.n_steps)
        ]

    def objective(self, x):
        return float(0.5 * x @ self._Q @ x + self._c @ x)

    def gradient(self, x):
        return self._Q @ x + self._c

    def constraints(self, x):
        return self._A @ x

    def jacobian_dense(self, x):
        return self._A

    def hessian_dense(self, x, lagrange, obj_factor):
        return obj_factor * self._Q


class DegenerateCorner(ParametricFamily):
    """A path that passes exactly through a degenerate active set.

    ``min ½‖x − a(θ)‖²  s.t.  Ax ≤ b,  x ≥ 0``

    ``a(θ)`` sweeps along a line chosen so that at the **midpoint step
    of the path — at every scale —** the unconstrained minimizer sits
    exactly on one constraint's boundary and exactly on one variable's
    lower bound. Both multipliers are zero there: the constraint is
    weakly active, strict complementarity fails, and the
    multiplier-sign classifier that builds a working set from a
    converged iterate has no signal to work with.

    Nothing here is fast or hard to solve. The point is whether a warm
    start stays *correct* across the degeneracy — a solver that
    misclassifies is free to converge to the wrong face, and only the
    correctness gate catches it.
    """

    name = "degenerate_corner"
    quadratic = True
    tags = {"regime": "degenerate", "channel": "objective", "curvature": "convex"}
    n_steps = 21  # odd, so a step lands exactly on the midpoint

    _N = 6
    _DELTA = 0.04

    def __init__(self):
        n = self._N
        self._A = np.array(
            [
                [1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
                [0.5, 0.0, 0.5, 0.0, 1.0, 1.0],
            ]
        )
        # a(θ) = a_c + θ·d. The crossing configuration is at θ = 0.
        self._a_c = np.array([0.6, 0.4, 0.3, 0.2, 0.0, 0.5])
        self._d = np.array([0.5, 0.5, -0.2, 0.0, 0.4, -0.3])
        # b makes row 0 exactly satisfied-with-equality at θ = 0, so
        # its multiplier passes through zero there. Component 4 of
        # a_c is exactly 0, so that bound is weakly active too.
        self._b = self._A @ self._a_c
        self._b[1] += 0.75  # rows 1,2 stay strictly inactive nearby
        self._b[2] += 0.75
        self._a = self._a_c.copy()

    @property
    def n(self) -> int:
        return self._N

    @property
    def m(self) -> int:
        return 3

    def bounds(self) -> Bounds:
        return Bounds(
            lb=np.zeros(self._N),
            ub=np.full(self._N, _INF),
            cl=np.full(3, -_INF),
            cu=self._b.copy(),
        )

    def cold_x0(self) -> np.ndarray:
        return np.full(self._N, 0.25)

    def set_theta(self, theta: np.ndarray) -> None:
        t = float(np.asarray(theta).ravel()[0])
        self._a = self._a_c + t * self._d

    def theta_path(self, scale: float) -> Optional[List[np.ndarray]]:
        half = (self.n_steps - 1) // 2
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
