"""The closed-loop family: an actual receding-horizon sequence.

Every other family in this suite walks a *scripted* parameter path.
This one does not: the next problem's parameter is the state the plant
reaches after applying the control the previous solve produced. That
is the setting warm starting was invented for, and it behaves
differently from a scripted sweep — the perturbation size is set by
the closed loop rather than by us, and it shrinks as the controller
drives the plant toward the setpoint.

Because the path depends on the solutions, the runner records the
parameter sequence produced by the reference arm once and *replays*
it for every other arm (see :mod:`..runner`). Otherwise each arm would
be solving a slightly different set of problems and the iteration
counts would not be comparable.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from ..spec import Bounds, ParametricFamily

_INF = 1e20


class VanDerPolNMPC(ParametricFamily):
    """Direct-transcription NMPC of a Van der Pol oscillator.

    Variables ``z = [x₀ … x_N, u₀ … u_{N−1}]`` (states first, then
    controls), single-shooting-free: the dynamics are equality
    constraints (explicit Euler), the initial state is an equality
    constraint whose right-hand side is the parameter, and the
    controls are bounded. The active set is *which controls
    saturate*, which changes as the plant state moves — the textbook
    reason MPC codes warm start.

    Nonconvex: the ``x₁²x₂`` term in the dynamics makes the constraint
    Hessian state-dependent.
    """

    name = "nmpc_vanderpol"
    tags = {"regime": "closed-loop", "channel": "rhs", "curvature": "nonconvex"}
    n_steps = 20
    adaptive = True

    _NH = 15  # horizon steps
    _H = 0.1  # discretization step
    _MU = 1.0  # Van der Pol parameter
    _U_MAX = 0.75
    _Q = np.array([1.0, 0.1])
    _R = 0.05
    _QT = 10.0  # terminal weight
    _X_INIT = np.array([2.0, 0.0])

    def __init__(self):
        self._theta = self._X_INIT.copy()
        self._plant_steps_per_move = 1.0

    # -- layout ----------------------------------------------------

    @property
    def n(self) -> int:
        return 2 * (self._NH + 1) + self._NH

    @property
    def m(self) -> int:
        return 2 + 2 * self._NH

    @property
    def _u_off(self) -> int:
        return 2 * (self._NH + 1)

    def bounds(self) -> Bounds:
        lb = np.full(self.n, -5.0)
        ub = np.full(self.n, 5.0)
        lb[self._u_off :] = -self._U_MAX
        ub[self._u_off :] = self._U_MAX
        return Bounds(
            lb=lb,
            ub=ub,
            cl=np.zeros(self.m),
            cu=np.zeros(self.m),
        )

    def cold_x0(self) -> np.ndarray:
        return np.zeros(self.n)

    def set_theta(self, theta: np.ndarray) -> None:
        self._theta = np.asarray(theta, dtype=float).ravel().copy()

    def theta_path(self, scale: float) -> Optional[List[np.ndarray]]:
        return None  # adaptive

    def initial_theta(self, scale: float) -> np.ndarray:
        # `scale` sets how far the plant advances between solves, i.e.
        # how big a perturbation the next problem sees: 1.0 is one
        # control interval (the textbook receding-horizon step).
        self._plant_steps_per_move = float(scale)
        return self._X_INIT.copy()

    def next_theta(self, x_solution: np.ndarray) -> np.ndarray:
        u0 = float(x_solution[self._u_off])
        return self._simulate(self._theta, u0, self._H * self._plant_steps_per_move)

    # -- plant (RK4, finer than the controller's Euler model) -------

    def _rhs(self, x, u):
        return np.array(
            [x[1], self._MU * (1.0 - x[0] ** 2) * x[1] - x[0] + u]
        )

    def _simulate(self, x0, u, duration, substeps=20):
        x = np.asarray(x0, dtype=float).copy()
        dt = duration / substeps
        for _ in range(substeps):
            k1 = self._rhs(x, u)
            k2 = self._rhs(x + 0.5 * dt * k1, u)
            k3 = self._rhs(x + 0.5 * dt * k2, u)
            k4 = self._rhs(x + dt * k3, u)
            x = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        return x

    # -- functions -------------------------------------------------

    def _split(self, z):
        X = z[: self._u_off].reshape(self._NH + 1, 2)
        U = z[self._u_off :]
        return X, U

    def objective(self, z):
        X, U = self._split(z)
        stage = float(np.sum(self._Q * X[:-1] ** 2))
        terminal = float(self._QT * np.sum(self._Q * X[-1] ** 2))
        return stage + terminal + float(self._R * (U @ U))

    def gradient(self, z):
        X, U = self._split(z)
        g = np.zeros(self.n)
        gx = g[: self._u_off].reshape(self._NH + 1, 2)
        gx[:-1] = 2.0 * self._Q * X[:-1]
        gx[-1] = 2.0 * self._QT * self._Q * X[-1]
        g[self._u_off :] = 2.0 * self._R * U
        return g

    def constraints(self, z):
        X, U = self._split(z)
        c = np.empty(self.m)
        c[:2] = X[0] - self._theta
        h, mu = self._H, self._MU
        x1, x2 = X[:-1, 0], X[:-1, 1]
        c[2::2] = X[1:, 0] - x1 - h * x2
        c[3::2] = (
            X[1:, 1] - x2 - h * (mu * (1.0 - x1**2) * x2 - x1 + U)
        )
        return c

    def jacobian_dense(self, z):
        X, U = self._split(z)
        h, mu = self._H, self._MU
        j = np.zeros((self.m, self.n))
        j[0, 0] = 1.0
        j[1, 1] = 1.0
        for k in range(self._NH):
            x1, x2 = X[k]
            r1, r2 = 2 + 2 * k, 3 + 2 * k
            i1, i2 = 2 * k, 2 * k + 1  # x_k
            n1, n2 = 2 * (k + 1), 2 * (k + 1) + 1  # x_{k+1}
            uk = self._u_off + k
            # x1_{k+1} − x1_k − h·x2_k
            j[r1, n1] = 1.0
            j[r1, i1] = -1.0
            j[r1, i2] = -h
            # x2_{k+1} − x2_k − h(μ(1−x1²)x2 − x1 + u)
            j[r2, n2] = 1.0
            j[r2, i1] = 2.0 * h * mu * x1 * x2 + h
            j[r2, i2] = -1.0 - h * mu * (1.0 - x1**2)
            j[r2, uk] = -h
        return j

    def hessian_dense(self, z, lagrange, obj_factor):
        X, _ = self._split(z)
        h, mu = self._H, self._MU
        H = np.zeros((self.n, self.n))
        # Objective (constant, diagonal).
        diag = np.zeros(self.n)
        dx = diag[: self._u_off].reshape(self._NH + 1, 2)
        dx[:-1] = 2.0 * self._Q
        dx[-1] = 2.0 * self._QT * self._Q
        diag[self._u_off :] = 2.0 * self._R
        H[np.arange(self.n), np.arange(self.n)] = obj_factor * diag
        # Constraints: only the x1²x2 term has curvature.
        for k in range(self._NH):
            x1, x2 = X[k]
            lam = lagrange[3 + 2 * k]
            i1, i2 = 2 * k, 2 * k + 1
            H[i1, i1] += lam * 2.0 * h * mu * x2
            H[i1, i2] += lam * 2.0 * h * mu * x1
            H[i2, i1] += lam * 2.0 * h * mu * x1
        return H
