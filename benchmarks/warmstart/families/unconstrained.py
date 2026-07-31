"""The empty-active-set control: nothing to carry but the point itself.

Every other family hands the next solve a working set worth having.
This one hands it nothing: no constraint rows at all (``m = 0``), no
finite variable bounds, so the working set is empty at every iterate
of every step. Whatever speedup shows up here is what a warm start
buys from the *primal point alone* — the zero mark the other families'
numbers should be read against.

It is also the one configuration the rest of the suite never executes.
That matters: pounce#416 (exact-Hessian SQP burning its whole inner-QP
budget) reproduces precisely here — unconstrained, bounds inactive or
absent, indefinite Hessian — and the suite found it only because a
*constrained* family happened to pass through an interior iterate on
its way somewhere else.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from ..spec import Bounds, ParametricFamily

_INF = 1e20


class DoubleWellChain(ParametricFamily):
    """Coupled double wells with drifting depths, no constraints.

        min Σᵢ (xᵢ² − cᵢ(θ))²  +  κ Σᵢ (xᵢ₊₁ − xᵢ)²

    Each coordinate has two symmetric minima at ``±√cᵢ``; the spring
    coupling pulls neighbours together so the wells do not separate
    into ``n`` independent scalar problems. Nonconvex, and the
    Hessian is genuinely indefinite away from the solution (the well
    term contributes ``12xᵢ² − 4cᵢ``, negative whenever
    ``xᵢ² < cᵢ/3``), so this exercises unconstrained *and* indefinite
    — with no active set anywhere to rescue the step computation.

    The parameter drifts the well depths, which moves the minimizer
    without ever changing which branch it sits in, so a warm start
    should keep the solve inside the same basin. A cold solve has to
    re-find it every step.
    """

    name = "double_well_chain"
    tags = {"regime": "none", "channel": "objective", "curvature": "nonconvex"}
    n_steps = 20

    _N = 12
    _K = 0.5  # coupling
    _DELTA = 0.05

    def __init__(self):
        self._c = self._c0()

    def _c0(self) -> np.ndarray:
        # Depths in [0.6, 2.4] — the minimizers √c are then O(1) and
        # the cold start below sits inside the indefinite region.
        return 0.6 + 1.8 * np.arange(self._N) / (self._N - 1)

    def _direction(self) -> np.ndarray:
        return np.cos(2.0 * np.pi * np.arange(self._N) / self._N)

    @property
    def n(self) -> int:
        return self._N

    @property
    def m(self) -> int:
        return 0

    def bounds(self) -> Bounds:
        return Bounds(
            lb=np.full(self._N, -_INF),
            ub=np.full(self._N, _INF),
            cl=np.zeros(0),
            cu=np.zeros(0),
        )

    def cold_x0(self) -> np.ndarray:
        # Deliberately inside the negative-curvature region
        # (x² < c/3 for every coordinate), so the path to the solution
        # runs through an indefinite Hessian with no constraint in the
        # working set to stabilize the step.
        return np.full(self._N, 0.35)

    def set_theta(self, theta: np.ndarray) -> None:
        self._c = np.asarray(theta, dtype=float).copy()

    def theta_path(self, scale: float) -> Optional[List[np.ndarray]]:
        c0, d = self._c0(), self._direction()
        return [c0 + scale * self._DELTA * k * d for k in range(self.n_steps)]

    def objective(self, x):
        w = x**2 - self._c
        d = np.diff(x)
        return float(w @ w + self._K * (d @ d))

    def gradient(self, x):
        g = 4.0 * x * (x**2 - self._c)
        # κ·Σ(x_{i+1} − x_i)² → κ·2·(2x_i − x_{i−1} − x_{i+1}) interior.
        g[:-1] -= 2.0 * self._K * (x[1:] - x[:-1])
        g[1:] += 2.0 * self._K * (x[1:] - x[:-1])
        return g

    def constraints(self, x):
        return np.zeros(0)

    def jacobian_dense(self, x):
        return np.zeros((0, self._N))

    def hessian_dense(self, x, lagrange, obj_factor):
        n = self._N
        h = np.zeros((n, n))
        diag = 12.0 * x**2 - 4.0 * self._c
        # Coupling: tridiagonal, +2κ per incident spring on the
        # diagonal and −2κ off it.
        diag[:-1] += 2.0 * self._K
        diag[1:] += 2.0 * self._K
        h[np.arange(n), np.arange(n)] = diag
        off = np.full(n - 1, -2.0 * self._K)
        h[np.arange(n - 1), np.arange(1, n)] = off
        h[np.arange(1, n), np.arange(n - 1)] = off
        return obj_factor * h
