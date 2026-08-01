"""Extract standard-form QP data from a family whose instances are QPs.

Some families in this suite are literally convex QPs dressed as NLPs —
quadratic objective, linear constraints — and a dedicated convex QP
solver can take them directly instead of going through callbacks. This
module does the conversion, in solver-agnostic form, so an adapter only
has to map the result onto its own QP entry point.

The conversion is exact and constructive rather than assumed. For a
family with ``f(x) = ½xᵀPx + cᵀx + f₀`` and ``g(x) = Jx + g₀``:

    P  = ∇²f            (constant, so evaluated once, anywhere)
    c  = ∇f(0)          (since ∇f(x) = Px + c)
    f₀ = f(0)
    J  = ∇g             (constant)
    g₀ = g(0)

``f₀`` matters: the QP solver reports ``½xᵀPx + cᵀx`` with no constant
term, so an objective compared against the other arms without adding
``f₀`` back would be wrong by a fixed offset — silently, and differently
per step, since ``f₀`` moves with the parameter.

Two-sided constraint rows are split into the one-sided ``Gx ≤ h`` form
the QP interface takes; rows with ``cl == cu`` become equalities.
:func:`verify` re-derives the family's own callbacks from the extracted
data and is exercised by the suite's self-test — a wrong extraction here
would produce plausible, wrong benchmark numbers.
"""

from __future__ import annotations

import dataclasses
from typing import Optional

import numpy as np

from .spec import ParametricFamily

#: Rows whose bounds are this close together are treated as equalities.
_EQ_TOL = 1e-12

#: Matches the ±1e19/1e20 sentinels the families use for "no bound".
_BOUND_INF = 1e19


@dataclasses.dataclass
class QpData:
    """``min ½xᵀPx + cᵀx + f₀  s.t.  Ax = b,  Gx ≤ h,  lb ≤ x ≤ ub``."""

    P: np.ndarray
    c: np.ndarray
    f0: float
    A: Optional[np.ndarray]
    b: Optional[np.ndarray]
    G: Optional[np.ndarray]
    h: Optional[np.ndarray]
    lb: np.ndarray
    ub: np.ndarray


def _sparse_extract(family, src) -> QpData:
    """Sparse counterpart of :func:`extract`, for families that declare
    a structure. Nothing dense is ever materialized — passing a dense
    ``P`` to the convex solver at n in the thousands is 60-80× slower
    by its own diagnostic, which would make the QP arm look bad for a
    reason that has nothing to do with the solver."""
    from scipy import sparse

    n, m = family.n, family.m
    jr, jc, hr, hc = (np.asarray(a) for a in family.sparse_structure())
    zero = np.zeros(n)

    hv = np.asarray(family.hessian_values(zero, np.zeros(m), 1.0), dtype=float)
    # Stored lower triangle -> full symmetric, which is what the QP
    # interface's `P` expects to be handed (it re-takes the triangle).
    P = sparse.coo_matrix((hv, (hr, hc)), shape=(n, n)).tocsc()
    strict = sparse.coo_matrix(
        (hv[hr != hc], (hc[hr != hc], hr[hr != hc])), shape=(n, n)
    ).tocsc()
    P = (P + strict).tocsc()

    c = np.asarray(src.gradient(zero), dtype=float).copy()
    f0 = float(src.objective(zero))

    b_ = family.bounds()
    lb = np.where(b_.lb > -_BOUND_INF, b_.lb, -np.inf)
    ub = np.where(b_.ub < _BOUND_INF, b_.ub, np.inf)

    A = b = G = h = None
    if m:
        jv = np.asarray(family.jacobian_values(zero), dtype=float)
        J = sparse.coo_matrix((jv, (jr, jc)), shape=(m, n)).tocsr()
        g0 = np.asarray(src.constraints(zero), dtype=float)
        cl, cu = b_.cl, b_.cu
        eq = np.abs(cu - cl) <= _EQ_TOL
        if eq.any():
            A = J[np.flatnonzero(eq)]
            b = cl[eq] - g0[eq]
        rows, hs = [], []
        for i in np.flatnonzero(~eq):
            if cu[i] < _BOUND_INF:
                rows.append(J[i])
                hs.append(cu[i] - g0[i])
            if cl[i] > -_BOUND_INF:
                rows.append(-J[i])
                hs.append(-(cl[i] - g0[i]))
        if rows:
            G = sparse.vstack(rows).tocsc()
            h = np.array(hs)
    return QpData(
        P=P, c=c, f0=f0,
        A=A.tocsc() if A is not None else None,
        b=b, G=G, h=h, lb=lb, ub=ub,
    )


def extract(family: ParametricFamily, callbacks=None) -> QpData:
    """Build :class:`QpData` for the family at its *current* parameter.

    Pass ``callbacks`` (a :class:`~.sparsity.SparseCallbacks`) to route
    the three evaluations through the harness's counters, so the
    assembly cost shows up in the step's evaluation counts the same way
    the other arms' per-iteration callbacks do.
    """
    src = callbacks if callbacks is not None else family
    if family.sparse_structure() is not None:
        return _sparse_extract(family, src)

    n, m = family.n, family.m
    zero = np.zeros(n)

    # `SparseCallbacks` returns packed sparse values, so the dense
    # matrices come from the family either way; the callbacks object is
    # used for the vector evaluations, which is what the counters track.
    lam = np.zeros(m)
    P = np.asarray(family.hessian_dense(zero, lam, 1.0), dtype=float)
    P = 0.5 * (P + P.T)  # symmetrize; the families build full symmetric
    c = np.asarray(src.gradient(zero), dtype=float).copy()
    f0 = float(src.objective(zero))

    bounds = family.bounds()
    lb = np.where(bounds.lb > -_BOUND_INF, bounds.lb, -np.inf)
    ub = np.where(bounds.ub < _BOUND_INF, bounds.ub, np.inf)

    A = b = G = h = None
    if m:
        J = np.asarray(family.jacobian_dense(zero), dtype=float)
        g0 = np.asarray(src.constraints(zero), dtype=float)
        cl, cu = bounds.cl, bounds.cu

        eq = np.abs(cu - cl) <= _EQ_TOL
        if eq.any():
            A = J[eq]
            b = cl[eq] - g0[eq]

        g_rows, h_vals = [], []
        for i in np.nonzero(~eq)[0]:
            if cu[i] < _BOUND_INF:  #  Jx + g₀ ≤ cu
                g_rows.append(J[i])
                h_vals.append(cu[i] - g0[i])
            if cl[i] > -_BOUND_INF:  # −(Jx + g₀) ≤ −cl
                g_rows.append(-J[i])
                h_vals.append(-(cl[i] - g0[i]))
        if g_rows:
            G = np.array(g_rows)
            h = np.array(h_vals)

    return QpData(P=P, c=c, f0=f0, A=A, b=b, G=G, h=h, lb=lb, ub=ub)


def verify(family: ParametricFamily, qp: QpData, rng, n_points: int = 5) -> list:
    """Re-derive the family from ``qp`` and report any disagreement.

    Returns a list of human-readable failures (empty when the extraction
    reproduces the family exactly). Checks the objective value, the
    gradient, the constraint values, and that ``P`` is symmetric positive
    semidefinite — the convexity the QP solver is entitled to assume and
    would otherwise report a silently-wrong "optimal" for.
    """
    out = []
    n = family.n
    dense = lambda M: (M.toarray() if hasattr(M, "toarray") else M)
    P, A, G = dense(qp.P), dense(qp.A), dense(qp.G)
    for _ in range(n_points):
        x = rng.standard_normal(n)
        f_qp = 0.5 * x @ P @ x + qp.c @ x + qp.f0
        f_fam = float(family.objective(x))
        if abs(f_qp - f_fam) > 1e-9 * (1.0 + abs(f_fam)):
            out.append(f"objective mismatch: qp={f_qp:.12g} family={f_fam:.12g}")
        g_qp = P @ x + qp.c
        if not np.allclose(g_qp, family.gradient(x), rtol=1e-9, atol=1e-9):
            out.append("gradient mismatch")
        if family.m:
            cons = np.asarray(family.constraints(x), dtype=float)
            bounds = family.bounds()
            eq = np.abs(bounds.cu - bounds.cl) <= _EQ_TOL
            # Equalities: Ax − b must be the family's own residual
            # g(x) − cl on those rows.
            if A is not None and not np.allclose(
                A @ x - qp.b, cons[eq] - bounds.cl[eq], rtol=1e-9, atol=1e-9
            ):
                out.append("equality row mismatch")
            # Inequalities: each split row's slack h − Gx must be the
            # family's own distance to the bound it came from.
            expected = []
            for i in np.nonzero(~eq)[0]:
                if bounds.cu[i] < _BOUND_INF:
                    expected.append(bounds.cu[i] - cons[i])
                if bounds.cl[i] > -_BOUND_INF:
                    expected.append(cons[i] - bounds.cl[i])
            if expected:
                if G is None or not np.allclose(
                    qp.h - G @ x, np.array(expected), rtol=1e-9, atol=1e-9
                ):
                    out.append("inequality row mismatch")
            elif qp.G is not None:
                out.append("unexpected inequality rows in the extraction")
    if not np.allclose(P, P.T, atol=1e-12):
        out.append("P is not symmetric")
    if P.size and P.shape[0] <= 400:
        w = np.linalg.eigvalsh(P)
        if w.min() < -1e-9 * max(1.0, abs(w.max())):
            out.append(
                f"P is indefinite (min eigenvalue {w.min():.3e}); the convex "
                "QP solver is not applicable to this family"
            )
    return out
