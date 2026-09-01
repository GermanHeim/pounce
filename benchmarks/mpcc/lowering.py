"""MPCC -> smooth NLP.

Three lowerings, and the differences between them are the whole subject
of the benchmark:

``prod_ineq``  ``G >= 0``, ``H >= 0``, ``G*H <= 0``. The "direct POUNCE
               NLP formulation". Its feasible points are exactly the
               MPCC's, because ``G,H >= 0`` makes ``G*H <= 0``
               equivalent to ``G*H = 0``.
``prod_eq``    the same with ``G*H = 0``: the exact product / NCP
               equality form.
``scholtes``   ``G*H <= tau``: Scholtes' regularisation, feasible for the
               MPCC only in the limit ``tau -> 0``.

All three violate MFCQ at every MPCC-feasible point -- that is a theorem
about the reformulation, not a defect of any of them, and it is why the
benchmark exists. ``prod_eq`` is the most degenerate of the three (an
equality row whose gradient is ``H grad G + G grad H``, and both terms
vanish together at a biactive point), and it is where the l1
exact-penalty routes are expected to matter.

Two construction choices worth knowing:

* **Dense Jacobian and Hessian structures.** Every case is at most six
  variables, so declaring the dense pattern removes a whole class of
  index bug from the harness at no measurable cost. A sparsity bug here
  would be indistinguishable from a solver bug in the results, which is
  exactly the confusion gh#794 exists to prevent.

* **``bound_relax_factor`` is pinned to 0 by the route layer**, not
  here. Its default relaxes every constraint bound by 1e-8, which on
  this corpus means accepting ``G >= -1e-8``: a source-level sign
  violation of the same size as the numbers the report is trying to
  measure. See `routes.BASE_OPTIONS`.

Row order is fixed and is part of the contract, because the runner reads
multipliers back out of ``info["mult_g"]`` positionally::

    [ source rows ] [ G_1 ] [ H_1 ] ... [ G_q ] [ H_q ] [ prod_1 ] ... [ prod_q ]
"""

from __future__ import annotations

import dataclasses
from typing import List, Optional, Tuple

import numpy as np

from .spec import MpccCase, Quad, product

#: The lowerings, by the name that appears in a record.
LOWERINGS = ("prod_ineq", "prod_eq", "scholtes")


@dataclasses.dataclass
class LoweredNlp:
    """A cyipopt-style callback object plus the bookkeeping to read it back."""

    case: MpccCase
    lowering: str
    tau: Optional[float]
    n: int
    m: int
    lb: np.ndarray
    ub: np.ndarray
    cl: np.ndarray
    cu: np.ndarray
    forms: List[Quad]
    #: Row index of the first ``G_i`` row, the first ``prod_i`` row.
    g_row0: int
    prod_row0: int
    n_source_rows: int

    # -- cyipopt-style callbacks ----------------------------------

    def objective(self, x):
        return self.case.objective.value(np.asarray(x, dtype=float))

    def gradient(self, x):
        return self.case.objective.grad(np.asarray(x, dtype=float))

    def constraints(self, x):
        x = np.asarray(x, dtype=float)
        return np.array([f.value(x) for f in self.forms])

    def jacobian(self, x):
        x = np.asarray(x, dtype=float)
        return np.concatenate([f.grad(x) for f in self.forms]) if self.m else np.zeros(0)

    def jacobianstructure(self):
        rows = np.repeat(np.arange(self.m), self.n)
        cols = np.tile(np.arange(self.n), self.m)
        return rows, cols

    def hessianstructure(self):
        rows, cols = np.tril_indices(self.n)
        return rows, cols

    def hessian(self, x, lagrange, obj_factor):
        x = np.asarray(x, dtype=float)
        h = obj_factor * self.case.objective.hess(x)
        for lam, f in zip(np.asarray(lagrange, dtype=float), self.forms):
            if lam != 0.0:
                h = h + lam * f.hess(x)
        r, c = np.tril_indices(self.n)
        return h[r, c]

    # -- reading a solve back -------------------------------------

    def pair_multipliers(self, mult_g: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(mult_G, mult_H, mult_prod)`` sliced out of ``info['mult_g']``.

        These are the *NLP* multipliers of the lowered rows. They are
        reported, but they are not the MPCC multipliers and the report
        never presents them as such: `stationarity.classify` recovers
        the MPCC multipliers from the source model's own gradients.
        """
        q = self.case.q
        g0 = self.g_row0
        mg = np.array([mult_g[g0 + 2 * i] for i in range(q)])
        mh = np.array([mult_g[g0 + 2 * i + 1] for i in range(q)])
        mp = np.array([mult_g[self.prod_row0 + i] for i in range(q)])
        return mg, mh, mp


def lower(case: MpccCase, lowering: str, tau: Optional[float] = None) -> LoweredNlp:
    """Build the smooth NLP for ``case`` under ``lowering``.

    ``tau`` is required by ``scholtes`` and rejected by the others --
    a relaxation parameter on a non-relaxed lowering would be silently
    ignored, and a record whose ``tau`` field says ``1e-8`` when nothing
    read it is worse than no field at all.
    """
    if lowering not in LOWERINGS:
        raise ValueError(f"unknown lowering {lowering!r}")
    if lowering == "scholtes":
        if tau is None:
            raise ValueError("scholtes lowering needs tau")
    elif tau is not None:
        raise ValueError(f"{lowering} lowering takes no tau")

    forms: List[Quad] = []
    cl: List[float] = []
    cu: List[float] = []

    for row in case.rows:
        forms.append(row.form)
        cl.append(row.lo)
        cu.append(row.hi)
    n_source_rows = len(forms)

    g_row0 = len(forms)
    for p in case.pairs:
        forms.append(p.G.as_quad())
        cl.append(0.0)
        cu.append(np.inf)
        forms.append(p.H.as_quad())
        cl.append(0.0)
        cu.append(np.inf)

    prod_row0 = len(forms)
    for p in case.pairs:
        forms.append(product(p.G, p.H))
        if lowering == "prod_eq":
            cl.append(0.0)
            cu.append(0.0)
        elif lowering == "prod_ineq":
            cl.append(-np.inf)
            cu.append(0.0)
        else:
            cl.append(-np.inf)
            cu.append(float(tau))

    return LoweredNlp(
        case=case,
        lowering=lowering,
        tau=tau,
        n=case.n,
        m=len(forms),
        lb=case.lb.copy(),
        ub=case.ub.copy(),
        cl=np.array(cl),
        cu=np.array(cu),
        forms=forms,
        g_row0=g_row0,
        prod_row0=prod_row0,
        n_source_rows=n_source_rows,
    )


def fd_check(nlp: LoweredNlp, x: np.ndarray, h: float = 1e-6) -> dict:
    """Central-difference check of every derivative the NLP declares.

    Run by `selftest` at several points per case. The corpus is
    quadratic, so the second differences are exact to round-off and a
    loose tolerance here would be hiding something rather than
    tolerating noise.
    """
    x = np.asarray(x, dtype=float)
    n = nlp.n

    def num_grad(f):
        out = np.zeros(n)
        for j in range(n):
            e = np.zeros(n)
            e[j] = h
            out[j] = (f(x + e) - f(x - e)) / (2 * h)
        return out

    gerr = float(np.max(np.abs(num_grad(nlp.objective) - nlp.gradient(x))))
    jac = nlp.jacobian(x).reshape(nlp.m, n) if nlp.m else np.zeros((0, n))
    jerr = 0.0
    for i in range(nlp.m):
        num = num_grad(lambda z, i=i: nlp.constraints(z)[i])
        jerr = max(jerr, float(np.max(np.abs(num - jac[i]))))

    # Hessian of the Lagrangian against a central difference of the
    # analytic gradient of the same Lagrangian.
    rng = np.random.default_rng(794)
    lam = rng.normal(size=nlp.m)
    of = 1.7

    def lag_grad(z):
        g = of * nlp.gradient(z)
        if nlp.m:
            g = g + lam @ nlp.jacobian(z).reshape(nlp.m, n)
        return g

    num_h = np.zeros((n, n))
    for j in range(n):
        e = np.zeros(n)
        e[j] = h
        num_h[:, j] = (lag_grad(x + e) - lag_grad(x - e)) / (2 * h)
    tri = nlp.hessian(x, lam, of)
    full = np.zeros((n, n))
    r, c = np.tril_indices(n)
    full[r, c] = tri
    full = full + np.tril(full, -1).T
    herr = float(np.max(np.abs(full - 0.5 * (num_h + num_h.T))))
    return {"grad": gerr, "jac": jerr, "hess": herr}
