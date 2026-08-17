"""Poorly scaled constraints and variables.

The issue lists "rank-deficient or poorly scaled constraints" as one
problem family. The suite already covers rank deficiency three ways
(``redundant_rows``, ``degenerate_corner``, ``degenerate_vertex``); it
had nothing for scaling.

Scaling is worth its own family because it is the axis every
initialization heuristic in the solver is implicitly making an
assumption about. A warm start that carries multipliers forward carries
them in the *scaled* space the previous solve worked in; a recentering
rule that pushes an iterate a fixed distance off its bound is measuring
that distance in units that differ by ``10⁴`` between two coordinates
of this problem; and the safeguarded least-squares normal step
(``least_square_init_primal``) is solving a least-squares problem whose
conditioning is exactly what this family controls. None of those is
exercised by a well-scaled problem, and all of them can degrade
silently — the solve still converges, it just costs more.

    min  ½ Σⱼ (sⱼ xⱼ − aⱼ(θ))²
    s.t. ρᵢ Σⱼ wᵢⱼ sⱼ xⱼ  ≤  ρᵢ bᵢ,     i = 1 … 4
         −50 ≤ xⱼ ≤ 50

with column scales ``sⱼ`` spanning ``10⁻²…10²`` and row scales ``ρᵢ``
spanning ``10⁻¹·⁵…10¹·⁵``. The Hessian is ``diag(sⱼ²)``, so its
condition number is ``10⁸``, and the constraint rows are individually
multiplied by a factor that changes nothing about the feasible set but
everything about what a residual norm on that row means.

The bounds are deliberately *uniform* while the columns are not: at
the optimum ``xⱼ ≈ aⱼ/sⱼ``, so the small-``s`` coordinates run out to
``|x| ≈ 100`` and clamp on the box while the large-``s`` ones sit near
zero. Which bounds are active is therefore a consequence of the
scaling, not of a separately chosen active set — and it moves as θ
does.

Everything is quadratic in the objective and linear in the
constraints, so this is a convex QP and the dedicated QP arms apply:
the same scaling question can be asked of the matrix-form solver and
the callback-driven ones side by side.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from ..spec import Bounds, ParametricFamily

_INF = 1e20


class BadlyScaledQP(ParametricFamily):
    """Convex QP with 10⁸ Hessian conditioning and 10³ row scaling."""

    name = "badly_scaled_qp"
    tags = {
        "regime": "scaling",
        "channel": "objective",
        "curvature": "convex",
    }
    quadratic = True
    n_steps = 20

    _N = 12
    _M = 4
    _BOX = 50.0
    _RHS = 0.5
    _RADIUS = 1.0
    _DPHI = 0.05

    def __init__(self):
        self._theta = self._theta_at(0.0)

    # -- the scaling ------------------------------------------------

    def _col_scale(self) -> np.ndarray:
        """``sⱼ`` from 10⁻² to 10²."""
        j = np.arange(self._N)
        return 10.0 ** (4.0 * j / (self._N - 1) - 2.0)

    def _row_scale(self) -> np.ndarray:
        """``ρᵢ`` from 10⁻¹·⁵ to 10¹·⁵."""
        i = np.arange(self._M)
        return 10.0 ** (3.0 * i / (self._M - 1) - 1.5)

    def _w(self) -> np.ndarray:
        i = np.arange(self._M)[:, None]
        j = np.arange(self._N)[None, :]
        return np.cos(1.7 * i + 0.9 * j)

    def _a(self) -> np.ndarray:
        """Target in the *scaled* coordinates, moved by θ."""
        j = np.arange(self._N)
        base = np.cos(0.7 * j)
        p = np.sin(0.4 * j)
        q = np.cos(0.31 * j + 1.0)
        return base + self._theta[0] * p + self._theta[1] * q

    def _theta_at(self, phi: float) -> np.ndarray:
        return self._RADIUS * np.array([np.cos(phi), np.sin(phi)])

    # -- shape ------------------------------------------------------

    @property
    def n(self) -> int:
        return self._N

    @property
    def m(self) -> int:
        return self._M

    def bounds(self) -> Bounds:
        return Bounds(
            lb=np.full(self._N, -self._BOX),
            ub=np.full(self._N, self._BOX),
            cl=np.full(self._M, -_INF),
            cu=self._row_scale() * self._RHS,
        )

    def cold_x0(self) -> np.ndarray:
        # Feasible (every row evaluates to 0 ≤ ρᵢ·0.5) and θ-independent.
        return np.zeros(self._N)

    def set_theta(self, theta: np.ndarray) -> None:
        self._theta = np.asarray(theta, dtype=float).ravel().copy()

    def theta_path(self, scale: float) -> Optional[List[np.ndarray]]:
        return [
            self._theta_at(scale * self._DPHI * k) for k in range(self.n_steps)
        ]

    # -- functions --------------------------------------------------

    def objective(self, x):
        r = self._col_scale() * x - self._a()
        return float(0.5 * (r @ r))

    def gradient(self, x):
        s = self._col_scale()
        return s * (s * x - self._a())

    def constraints(self, x):
        s = self._col_scale()
        return self._row_scale() * (self._w() @ (s * x))

    def jacobian_dense(self, x):
        return self._row_scale()[:, None] * self._w() * self._col_scale()[None, :]

    def hessian_dense(self, x, lagrange, obj_factor):
        # Constraints are linear; only the objective contributes.
        return obj_factor * np.diag(self._col_scale() ** 2)
