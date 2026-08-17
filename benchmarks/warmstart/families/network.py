"""A large sparse energy-network model: scattered sparsity, not banded.

The suite's own notes name this gap directly — the only family that
reaches size is linear-quadratic MPC, which is "banded, mostly
equalities, a large active set that barely moves", and "whether #428
dominates on a large problem with a different sparsity pattern is
untested". The issue asks for "large sparse process/energy models".

This is a nonlinear resistive network at steady state: flows on the
edges of a graph, conserved at every node, with a convex loss that is
quadratic at small flow and quartic at large flow —

    min  Σ_e  r_e/2 · f_e²  +  c_e/4 · f_e⁴
    s.t. (A f)_v = d_v,        v = 1 … V−1
         |f_e| ≤ f_max

where ``A`` is the node-edge incidence matrix. Node 0 carries no row
and acts as the slack bus, which is what makes the demand vector free
to move without a balance condition on it.

Three properties this has and ``mpc_horizon_*`` does not
--------------------------------------------------------

* **The sparsity is scattered.** ``A`` has exactly two entries per
  column, at the edge's endpoints. The ring-plus-chord topology puts
  those endpoints ``V/3`` apart for a third of the edges, so no
  permutation makes this banded — the KKT factor fills in a way the MPC
  sweep never exercises.
* **It is not a QP.** The quartic loss term makes the Hessian
  state-dependent (``r_e + 3c_e f_e²``), so the Newton system genuinely
  changes from iteration to iteration. Every existing large family has
  a constant Hessian, which flatters any method that reuses a factor.
  The dedicated QP arms correctly skip here.
* **The active set is sparse and scattered too.** Congested edges are
  the ones on the short paths between a moved demand and the slack bus,
  so the active bounds move around the graph as θ rotates rather than
  sliding along an index range.

The topology and coefficients are deterministic functions of the node
index — no RNG anywhere — so an instance is reproducible from ``V``
alone.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Type

import numpy as np

from ..spec import Bounds, ParametricFamily


def _edges(nv: int) -> List[Tuple[int, int]]:
    """Ring plus evenly spaced chords. Deterministic in ``nv``."""
    ring = [(k, (k + 1) % nv) for k in range(nv)]
    # Chords reach a third of the way around, which is what makes the
    # pattern non-bandable; there are nv//2 of them so E = 1.5·V.
    chords = [(k, (k + nv // 3) % nv) for k in range(nv // 2)]
    return ring + chords


class ResistiveNetworkBase(ParametricFamily):
    """Steady-state flow on a ring-plus-chord network, ``_NV`` nodes."""

    quadratic = False  # quartic loss term
    n_steps = 20

    #: Rows 0 and 1 are the balance rows of nodes 1 and 2, whose demands
    #: are θ — so stepping θ is exactly stepping those rows' right-hand
    #: side, which is what the sensitivity predictor's `deltas` means.
    pin_rows = (0, 1)

    _NV = 100
    _F_MAX = 1.2
    _D_AMP = 0.4  # fixed demand amplitude at the non-parametric nodes
    _RADIUS = 1.0  # θ walks a circle of this radius
    _DPHI = 0.05

    def __init__(self):
        self._edge_list = _edges(self._NV)
        self._theta = self._theta_at(0.0)

    # -- layout ----------------------------------------------------

    @property
    def n(self) -> int:
        return len(self._edge_list)

    @property
    def m(self) -> int:
        return self._NV - 1

    def _coeffs(self) -> Tuple[np.ndarray, np.ndarray]:
        """``(r, c)`` per edge — deterministic, mildly heterogeneous."""
        e = np.arange(self.n)
        r = 0.5 + 0.15 * ((e * 37) % 11)
        c = 0.01 * (1 + ((e * 53) % 7))
        return r, c

    def _fixed_demand(self) -> np.ndarray:
        """Demand at rows 2 … V−2 (nodes 3 … V−1); θ supplies rows 0, 1."""
        v = np.arange(3, self._NV)
        return self._D_AMP * np.sin(2.0 * np.pi * v / self._NV)

    def _theta_at(self, phi: float) -> np.ndarray:
        return self._RADIUS * np.array([np.cos(phi), np.sin(phi)])

    def bounds(self) -> Bounds:
        return Bounds(
            lb=np.full(self.n, -self._F_MAX),
            ub=np.full(self.n, self._F_MAX),
            cl=np.zeros(self.m),
            cu=np.zeros(self.m),
        )

    def cold_x0(self) -> np.ndarray:
        return np.zeros(self.n)

    def set_theta(self, theta: np.ndarray) -> None:
        self._theta = np.asarray(theta, dtype=float).ravel().copy()

    def theta_path(self, scale: float) -> Optional[List[np.ndarray]]:
        return [
            self._theta_at(scale * self._DPHI * k) for k in range(self.n_steps)
        ]

    # -- functions -------------------------------------------------

    def objective(self, f):
        r, c = self._coeffs()
        return float(0.5 * np.sum(r * f**2) + 0.25 * np.sum(c * f**4))

    def gradient(self, f):
        r, c = self._coeffs()
        return r * f + c * f**3

    def _demand(self) -> np.ndarray:
        d = np.empty(self.m)
        d[0] = self._theta[0]
        d[1] = self._theta[1]
        d[2:] = self._fixed_demand()
        return d

    def constraints(self, f):
        # (A f)_v - d_v for v = 1 .. V-1; node 0 is the slack bus.
        net = np.zeros(self._NV)
        for e, (a, b) in enumerate(self._edge_list):
            net[a] += f[e]
            net[b] -= f[e]
        return net[1:] - self._demand()

    def jacobian_dense(self, f):
        j = np.zeros((self.m, self.n))
        for e, (a, b) in enumerate(self._edge_list):
            if a >= 1:
                j[a - 1, e] = 1.0
            if b >= 1:
                j[b - 1, e] = -1.0
        return j

    def hessian_dense(self, f, lagrange, obj_factor):
        return obj_factor * np.diag(self._hess_diag(f))

    def _hess_diag(self, f) -> np.ndarray:
        r, c = self._coeffs()
        # Constraints are linear, so only the objective contributes.
        return r + 3.0 * c * np.asarray(f, dtype=float) ** 2

    # -- sparse path -----------------------------------------------

    def sparse_structure(self):
        jr: List[int] = []
        jc: List[int] = []
        for e, (a, b) in enumerate(self._edge_list):
            if a >= 1:
                jr.append(a - 1)
                jc.append(e)
            if b >= 1:
                jr.append(b - 1)
                jc.append(e)
        idx = np.arange(self.n)
        return (
            np.array(jr, dtype=np.int64),
            np.array(jc, dtype=np.int64),
            idx.copy(),  # Hessian is diagonal
            idx.copy(),
        )

    def jacobian_values(self, f):
        vals: List[float] = []
        for a, b in self._edge_list:
            if a >= 1:
                vals.append(1.0)
            if b >= 1:
                vals.append(-1.0)
        return np.array(vals, dtype=float)

    def hessian_values(self, f, lagrange, obj_factor):
        return obj_factor * self._hess_diag(f)


def _network_family(nv: int, tier: str = "default") -> Type[ResistiveNetworkBase]:
    return type(
        f"ResistiveNetwork{nv}",
        (ResistiveNetworkBase,),
        {
            "name": f"resistive_network_{nv}",
            "_NV": nv,
            "tier": tier,
            "n_steps": 20 if tier == "default" else 8,
            "tags": {
                "regime": "congestion",
                "channel": "rhs",
                "curvature": "convex",
                "nodes": str(nv),
            },
        },
    )


#: ``n = 1.5·V`` edges, ``m = V−1`` balance rows.
NETWORK_SIZES = (60, 120)
LARGE_NETWORK_SIZES = (800,)

NETWORK_FAMILIES: List[Type[ResistiveNetworkBase]] = [
    _network_family(nv) for nv in NETWORK_SIZES
] + [_network_family(nv, tier="large") for nv in LARGE_NETWORK_SIZES]

NETWORK_BY_NAME: Dict[str, Type[ResistiveNetworkBase]] = {
    f.name: f for f in NETWORK_FAMILIES
}
