"""Dense family callbacks → cyipopt-shaped sparse callbacks, with counters.

Families in this suite write their Jacobian and Hessian densely
(:meth:`ParametricFamily.jacobian_dense` /
:meth:`~ParametricFamily.hessian_dense`). This module turns one into
the ``jacobianstructure`` / ``jacobian`` / ``hessianstructure`` /
``hessian`` quartet every Ipopt-style interface expects.

Two reasons to go through here rather than hand-writing structures per
family:

1. **The structure and the values cannot disagree.** The classic
   silent-wrong-answer bug in this interface is a structure array that
   no longer matches the order the value callback packs entries in.
   Here both come from the same dense matrix, so the correspondence is
   constructive.
2. **Evaluation counts mean the same thing for every solver.** The
   wrapper counts calls itself instead of trusting a solver-reported
   statistic, which is the only way ``n_hess`` compares across a
   Newton method and a quasi-Newton one.

The sparsity pattern is *sampled*: the dense callbacks are evaluated
at a handful of randomized points (and, for the Hessian, randomized
multipliers and objective factors) and the union of nonzeros is taken.
That avoids pinning a pattern to an accidental zero at one special
point. Structural entries that are zero at every sample are simply
absent from the pattern — harmless, since they are zero wherever the
solver would have read them, for every θ along the path (families are
required to keep sparsity θ-invariant).
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from .spec import ParametricFamily

#: Number of randomized probes used to infer a sparsity pattern.
_N_PROBES = 8

#: Entries with |value| below this at every probe are treated as
#: structurally zero.
_PATTERN_TOL = 0.0


def _probe_points(family: ParametricFamily, rng: np.random.Generator) -> list:
    """Points to sample the sparsity pattern at.

    The cold start plus random points inside the bounds (finite bounds
    respected so a family that would divide by zero outside its domain
    is not evaluated there).
    """
    b = family.bounds()
    lo = np.where(np.isfinite(b.lb) & (b.lb > -1e19), b.lb, -1.0)
    hi = np.where(np.isfinite(b.ub) & (b.ub < 1e19), b.ub, 1.0)
    # Keep the box non-degenerate and modest in size.
    span = np.clip(hi - lo, 1e-3, 4.0)
    base = np.clip(family.cold_x0(), lo, hi)
    pts = [base]
    for _ in range(_N_PROBES - 1):
        pts.append(np.clip(base + span * (rng.random(family.n) - 0.5), lo, hi))
    return pts


class SparseCallbacks:
    """cyipopt-style callback object over a :class:`ParametricFamily`.

    Instances are passed straight to a solver's problem constructor.
    The sparsity pattern is computed once at construction and reused
    for the whole path.
    """

    def __init__(self, family: ParametricFamily, seed: int = 0):
        self.family = family
        rng = np.random.default_rng(seed)
        pts = _probe_points(family, rng)

        n, m = family.n, family.m

        jac_pat = np.zeros((m, n), dtype=bool)
        hess_pat = np.zeros((n, n), dtype=bool)
        for x in pts:
            if m:
                jac_pat |= np.abs(family.jacobian_dense(x)) > _PATTERN_TOL
            lam = rng.standard_normal(m) if m else np.zeros(0)
            for obj_factor in (1.0, 0.0):
                h = family.hessian_dense(x, lam, obj_factor)
                hess_pat |= np.abs(h) > _PATTERN_TOL

        self._jac_rows, self._jac_cols = np.nonzero(jac_pat)
        # Ipopt-style interfaces want the *lower* triangle of the
        # (symmetric) Hessian of the Lagrangian.
        hess_pat = np.tril(hess_pat | hess_pat.T)
        self._hess_rows, self._hess_cols = np.nonzero(hess_pat)

        self.reset_counts()

    # -- instrumentation -------------------------------------------

    def reset_counts(self) -> None:
        self.n_obj = 0
        self.n_grad = 0
        self.n_cons = 0
        self.n_jac = 0
        self.n_hess = 0

    def counts(self) -> dict:
        return {
            "n_obj": self.n_obj,
            "n_grad": self.n_grad,
            "n_cons": self.n_cons,
            "n_jac": self.n_jac,
            "n_hess": self.n_hess,
        }

    @property
    def nnz_jac(self) -> int:
        return int(self._jac_rows.size)

    @property
    def nnz_hess(self) -> int:
        return int(self._hess_rows.size)

    # -- cyipopt-shaped callbacks ----------------------------------

    def objective(self, x):
        self.n_obj += 1
        return float(self.family.objective(np.asarray(x, dtype=float)))

    def gradient(self, x):
        self.n_grad += 1
        return np.asarray(
            self.family.gradient(np.asarray(x, dtype=float)), dtype=float
        )

    def constraints(self, x):
        self.n_cons += 1
        return np.asarray(
            self.family.constraints(np.asarray(x, dtype=float)), dtype=float
        )

    def jacobianstructure(self) -> Tuple[np.ndarray, np.ndarray]:
        return (
            self._jac_rows.astype(np.int64),
            self._jac_cols.astype(np.int64),
        )

    def jacobian(self, x):
        self.n_jac += 1
        j = self.family.jacobian_dense(np.asarray(x, dtype=float))
        return np.ascontiguousarray(j[self._jac_rows, self._jac_cols], dtype=float)

    def hessianstructure(self) -> Tuple[np.ndarray, np.ndarray]:
        return (
            self._hess_rows.astype(np.int64),
            self._hess_cols.astype(np.int64),
        )

    def hessian(self, x, lagrange, obj_factor):
        self.n_hess += 1
        h = self.family.hessian_dense(
            np.asarray(x, dtype=float),
            np.asarray(lagrange, dtype=float),
            float(obj_factor),
        )
        return np.ascontiguousarray(
            h[self._hess_rows, self._hess_cols], dtype=float
        )
