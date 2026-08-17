"""The falsification arm: multi-basin problems where warm starting should lose.

Every other family in this suite was chosen by warm-start work, and
every one of them is a family where carrying the previous answer
forward is a defensible thing to do — a smooth path, a moving active
set, a solution that drifts. A suite built only out of those measures
how *much* warm starting wins, never *whether* it wins, and it cannot
produce the result that would tell a reader not to use it.

These two families exist to produce that result. Both are Rastrigin
shifts: a global minimum sitting in a lattice of local minima spaced
one unit apart, so "which basin is the seed in" is a property the
benchmark can control directly rather than hope for.

    f(x) = Σᵢ (xᵢ − θᵢ)² + A(1 − cos 2π(xᵢ − θᵢ))

The global minimum is at ``x = θ`` for every θ. A cold solve from a
fixed point re-finds whichever basin that point leads to. A warm solve
starts at the *previous* θ — so the seed is exactly ``‖Δθ‖`` away from
the answer, and whether it lands in the right basin is decided by how
far θ moved, which is precisely what :data:`~..spec.SCALES` sweeps.

What each family is for
-----------------------

``rastrigin_drift``
    A smooth path, like every other family here, but through a
    multi-basin landscape. Per-coordinate step is ``0.3 × scale``:
    ``tiny`` (0.03) and ``small`` (0.3) stay inside one basin and warm
    starting should behave like it does everywhere else, while ``large``
    (1.2) steps past at least one basin boundary every step and the seed
    is in the wrong well. Reading the three scales in order shows where
    continuation stops working, rather than asserting that it does or
    does not.

``rastrigin_scatter``
    Consecutive instances that are **not** a path. θ is a fixed seeded
    draw per step, so step k+1 has no relationship to step k beyond
    sharing a shape. This is the issue's "unrelated global/nonconvex
    cases where continuation should not be expected to help", and at
    ``large`` the draws span ±4 units — several basins apart — so the
    previous solution is worth strictly less than the cold start, which
    at least sits at a point chosen for the problem.

Both are deterministic: the scatter draws come from a seeded generator
evaluated at construction, not at solve time, so a re-run reproduces the
same instances.

How to read a bad result here
-----------------------------

Landing in the wrong basin does not show up as a failed solve. It shows
up as a *converged* step with a worse objective than the reference arm
— which the runner already scores as ``correct = False`` (see
:func:`~..runner._score_correctness`) and the report already counts.
That is the failure mode these families are here to produce, and a
warm arm losing steps on them is the suite working, not the suite
broken.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from ..spec import Bounds, ParametricFamily

_INF = 1e20


class _RastriginBase(ParametricFamily):
    """Shifted Rastrigin under one linear inequality.

    The constraint row is not decoration: without it the family would
    be unconstrained and would duplicate ``double_well_chain``'s
    coverage. ``Σ xᵢ ≤ c`` binds on part of every path and releases on
    the rest, so there is a working set to carry as well as a basin to
    get wrong — and the two failure modes are separable in the report,
    because a wrong-basin step shows up in the objective column and a
    wrong-working-set step shows up in the iteration column.
    """

    tags = {
        "regime": "multi-basin",
        "channel": "objective",
        "curvature": "nonconvex",
    }
    n_steps = 20

    _N = 10
    _A = 3.0  # well depth; the lattice spacing is 1 regardless
    _BOX = 6.0
    _SUM_CAP = 2.0

    def __init__(self):
        self._theta = self.theta_path(1.0)[0]

    @property
    def n(self) -> int:
        return self._N

    @property
    def m(self) -> int:
        return 1

    def bounds(self) -> Bounds:
        return Bounds(
            lb=np.full(self._N, -self._BOX),
            ub=np.full(self._N, self._BOX),
            cl=np.array([-_INF]),
            cu=np.array([self._SUM_CAP]),
        )

    def cold_x0(self) -> np.ndarray:
        # Fixed and θ-independent, as the spec requires. The origin is
        # a deliberate choice: it is a local minimum of the θ = 0
        # instance, so the cold arm is not being handed the answer.
        return np.zeros(self._N)

    def set_theta(self, theta: np.ndarray) -> None:
        self._theta = np.asarray(theta, dtype=float).ravel().copy()

    # -- functions -------------------------------------------------

    def objective(self, x):
        d = x - self._theta
        return float(d @ d + self._A * np.sum(1.0 - np.cos(2.0 * np.pi * d)))

    def gradient(self, x):
        d = x - self._theta
        return 2.0 * d + 2.0 * np.pi * self._A * np.sin(2.0 * np.pi * d)

    def constraints(self, x):
        return np.array([float(np.sum(x))])

    def jacobian_dense(self, x):
        return np.ones((1, self._N))

    def hessian_dense(self, x, lagrange, obj_factor):
        d = x - self._theta
        diag = 2.0 + 4.0 * np.pi**2 * self._A * np.cos(2.0 * np.pi * d)
        # The constraint is linear, so it contributes nothing.
        return obj_factor * np.diag(diag)


class RastriginDrift(_RastriginBase):
    """Smooth path through a multi-basin landscape.

    Per-coordinate increment ``0.3 × scale`` against a lattice spacing
    of 1: ``tiny`` and ``small`` stay in-basin, ``large`` does not.
    """

    name = "rastrigin_drift"

    _DELTA = 0.3

    def theta_path(self, scale: float) -> Optional[List[np.ndarray]]:
        # Every coordinate moves by the same amount, so the step size in
        # basin-widths is exactly `scale * _DELTA` and can be read
        # straight off the scale column.
        start = np.full(self._N, -1.5)
        return [
            start + scale * self._DELTA * k * np.ones(self._N)
            for k in range(self.n_steps)
        ]


class RastriginScatter(_RastriginBase):
    """Unrelated instances: θ is redrawn, not stepped.

    The draws are generated once from a fixed seed, so the sequence is
    part of the benchmark rather than a property of the run.
    """

    name = "rastrigin_scatter"
    tags = dict(_RastriginBase.tags, regime="unrelated")

    _SEED = 611
    _SPREAD = 1.0

    def theta_path(self, scale: float) -> Optional[List[np.ndarray]]:
        rng = np.random.default_rng(self._SEED)
        base = np.zeros(self._N)
        return [
            base + scale * self._SPREAD * rng.uniform(-1.0, 1.0, self._N)
            for _ in range(self.n_steps)
        ]
