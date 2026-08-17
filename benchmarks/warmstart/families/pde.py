"""PDE-constrained optimal control, swept over mesh refinement.

The issue asks for "PDE/DAE discretizations and mesh refinement" as a
problem family, and the suite's own `dev-notes` note that its only
large family is linear-quadratic MPC — "banded, mostly equalities, a
large active set that barely moves" — so whether the warm-start results
survive a *different* sparsity pattern at size was untested.

This is the standard 1-D elliptic control problem:

    min  h/2 Σ (yᵢ − y_dᵢ)²  +  αh/2 Σ uᵢ²
    s.t. −(y_{i−1} − 2yᵢ + y_{i+1})/h² − uᵢ = 0,   i = 1…N
         y₀ = θ₀,  y_{N+1} = θ₁
         |uᵢ| ≤ u_max

discretized on a uniform mesh with ``h = 1/(N+1)``. The state is
tracked against a fixed profile; the two Dirichlet boundary values are
the parameter.

Why this is not another MPC in disguise
---------------------------------------

The transcription is superficially similar — states, controls, one
equality row per node — but the two properties that decide warm-start
behaviour are different:

* **The coupling is symmetric, not causal.** MPC's dynamics row
  ``x⁺ = Ax + Bu`` reaches backwards only; the Laplacian row reaches
  both ways. The Jacobian is tridiagonal in the state block rather than
  block-lower-banded, so the KKT factor has a different fill pattern
  and an active-set change at node ``i`` propagates in both directions.
* **The active set is an interval whose endpoints move.** The control
  saturates wherever the unconstrained optimum ``≈ π² sin πx`` exceeds
  ``u_max``, which is a contiguous band in the middle of the domain.
  Moving the boundary data slides the band's endpoints. So the number
  of active bounds stays roughly constant while *which* bounds are
  active changes at the edges — the opposite of MPC's saturation
  regime, where the active set is large and barely moves.

**Conditioning moves with the mesh, and that is the point.** The
discrete Laplacian's condition number grows like ``h⁻²``, so refining
the mesh makes the problem harder in a way that has nothing to do with
its size. Reading ``elliptic_control_40`` → ``_160`` is therefore *not*
the same experiment as reading ``mpc_horizon_10`` → ``_80``, where
conditioning is flat and only the dimension grows. A warm-start result
that holds across the MPC sweep and fails across this one is a
conditioning result, and the two sweeps together are what separates
them.

Every instance is a convex QP (quadratic objective, linear
constraints), so the dedicated QP arms apply here as well, and θ enters
through two pin-equality rows, so the sensitivity-predictor arms do
too.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Type

import numpy as np

from ..spec import Bounds, ParametricFamily


class EllipticControlBase(ParametricFamily):
    """Poisson-constrained tracking control on ``N`` interior nodes.

    ``z = [y₀ … y_{N+1}, u₁ … u_N]`` — the boundary nodes are carried as
    variables and pinned by equality rows, rather than eliminated, so
    that θ enters as a right-hand side exactly the way
    :attr:`ParametricFamily.pin_rows` requires.
    """

    quadratic = True
    n_steps = 20

    #: Rows 0 and 1 are ``y₀ − θ₀ = 0`` and ``y_{N+1} − θ₁ = 0``.
    pin_rows = (0, 1)

    _N = 40  # interior nodes; overridden per mesh
    _ALPHA = 1e-4  # control regularization
    _U_MAX = 6.0  # saturates where π²·sin πx exceeds it
    _AMP = 1.0  # target amplitude
    _RADIUS = 0.5  # boundary data walks a circle of this radius
    _DPHI = 0.05  # radians per step, before the scale multiplier

    def __init__(self):
        self._theta = self._theta_at(0.0)

    # -- layout ----------------------------------------------------

    @property
    def n(self) -> int:
        return 2 * self._N + 2

    @property
    def m(self) -> int:
        return self._N + 2

    @property
    def _u_off(self) -> int:
        return self._N + 2

    @property
    def _h(self) -> float:
        return 1.0 / (self._N + 1)

    def _target(self) -> np.ndarray:
        """Desired state at the interior nodes."""
        xs = np.arange(1, self._N + 1) * self._h
        return self._AMP * np.sin(np.pi * xs)

    def _theta_at(self, phi: float) -> np.ndarray:
        return self._RADIUS * np.array([np.cos(phi), np.sin(phi)])

    def bounds(self) -> Bounds:
        lb = np.full(self.n, -10.0)
        ub = np.full(self.n, 10.0)
        lb[self._u_off :] = -self._U_MAX
        ub[self._u_off :] = self._U_MAX
        return Bounds(lb=lb, ub=ub, cl=np.zeros(self.m), cu=np.zeros(self.m))

    def cold_x0(self) -> np.ndarray:
        return np.zeros(self.n)

    def set_theta(self, theta: np.ndarray) -> None:
        self._theta = np.asarray(theta, dtype=float).ravel().copy()

    def theta_path(self, scale: float) -> Optional[List[np.ndarray]]:
        return [
            self._theta_at(scale * self._DPHI * k) for k in range(self.n_steps)
        ]

    # -- functions -------------------------------------------------

    def _split(self, z):
        return z[: self._u_off], z[self._u_off :]

    def objective(self, z):
        y, u = self._split(z)
        d = y[1:-1] - self._target()
        return float(
            0.5 * self._h * (d @ d) + 0.5 * self._ALPHA * self._h * (u @ u)
        )

    def gradient(self, z):
        y, u = self._split(z)
        g = np.zeros(self.n)
        g[1 : self._N + 1] = self._h * (y[1:-1] - self._target())
        g[self._u_off :] = self._ALPHA * self._h * u
        return g

    def constraints(self, z):
        y, u = self._split(z)
        c = np.empty(self.m)
        c[0] = y[0] - self._theta[0]
        c[1] = y[-1] - self._theta[1]
        # -(y_{i-1} - 2 y_i + y_{i+1})/h^2 - u_i, for i = 1..N
        lap = (y[:-2] - 2.0 * y[1:-1] + y[2:]) / self._h**2
        c[2:] = -lap - u
        return c

    def jacobian_dense(self, z):
        j = np.zeros((self.m, self.n))
        j[0, 0] = 1.0
        j[1, self._N + 1] = 1.0
        inv = 1.0 / self._h**2
        for i in range(1, self._N + 1):
            r = 1 + i  # rows 2 .. N+1
            j[r, i - 1] = -inv
            j[r, i] = 2.0 * inv
            j[r, i + 1] = -inv
            j[r, self._u_off + i - 1] = -1.0
        return j

    def hessian_dense(self, z, lagrange, obj_factor):
        # Constraints are linear; the objective is a separable quadratic.
        return obj_factor * np.diag(self._hess_diag())

    def _hess_diag(self) -> np.ndarray:
        diag = np.zeros(self.n)
        # The boundary nodes carry no tracking cost — they are fixed by
        # the pin rows, so a zero here is correct rather than missing.
        diag[1 : self._N + 1] = self._h
        diag[self._u_off :] = self._ALPHA * self._h
        return diag

    # -- sparse path -----------------------------------------------

    def sparse_structure(self):
        jr: List[int] = [0, 1]
        jc: List[int] = [0, self._N + 1]
        for i in range(1, self._N + 1):
            r = 1 + i
            jr += [r, r, r, r]
            jc += [i - 1, i, i + 1, self._u_off + i - 1]
        idx = np.arange(self.n)
        return (
            np.array(jr, dtype=np.int64),
            np.array(jc, dtype=np.int64),
            idx.copy(),  # Hessian is diagonal
            idx.copy(),
        )

    def jacobian_values(self, z):
        inv = 1.0 / self._h**2
        vals = np.empty(2 + 4 * self._N)
        vals[0] = 1.0
        vals[1] = 1.0
        vals[2:] = np.tile(
            np.array([-inv, 2.0 * inv, -inv, -1.0]), self._N
        )
        return vals

    def hessian_values(self, z, lagrange, obj_factor):
        return obj_factor * self._hess_diag()


def mesh_family(nn: int, tier: str = "default") -> Type[EllipticControlBase]:
    return type(
        f"EllipticControl{nn}",
        (EllipticControlBase,),
        {
            "name": f"elliptic_control_{nn}",
            "_N": nn,
            "tier": tier,
            "n_steps": 20 if tier == "default" else 8,
            "tags": {
                "regime": "moving-band",
                "channel": "rhs",
                "curvature": "convex",
                "mesh": str(nn),
            },
        },
    )


#: Interior-node counts. ``n = 2N+2`` runs 82 → 322 in the default
#: sweep; the opt-in large mesh reaches n = 1202 with a condition
#: number roughly 225× the coarsest one.
MESHES = (40, 80, 160)
LARGE_MESHES = (600,)

PDE_FAMILIES: List[Type[EllipticControlBase]] = [
    mesh_family(nn) for nn in MESHES
] + [mesh_family(nn, tier="large") for nn in LARGE_MESHES]

PDE_BY_NAME: Dict[str, Type[EllipticControlBase]] = {
    f.name: f for f in PDE_FAMILIES
}
