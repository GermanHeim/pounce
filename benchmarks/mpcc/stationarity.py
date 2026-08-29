"""MPCC stationarity classification.

Why this file is not a residual check
-------------------------------------

An MPCC violates MFCQ at every feasible point of every lowering in
`lowering.py`, so the NLP KKT conditions of the lowered problem are not
the first-order conditions of the MPCC. A converged NLP solve therefore
tells you that *the reformulation's* residuals are small. It does not
tell you which MPCC stationarity concept -- if any -- the returned point
satisfies, and the four concepts are genuinely different: on `ralph1` no
S-stationary point exists at all, and on `ctrap` the origin is
C-stationary, has every ordinary residual at zero, and is not even a
local minimiser.

The classes, with ``L = f + sum_k lambda_k c_k - sum_i nu_i G_i
- sum_i w_i H_i - zL'(x - lb) + zU'(x - ub)`` and ``grad_x L = 0``:

============  ==================================================
weak (W)      ``nu_i = 0`` where ``G_i > 0``; ``w_i = 0`` where
              ``H_i > 0``; ordinary signs on lambda and z. No
              condition on the biactive entries.
Clarke (C)    additionally ``nu_i * w_i >= 0`` on biactive pairs.
Mordukhovich  additionally, per biactive pair, either both are
(M)           strictly positive or their product is zero.
strong (S)    additionally ``nu_i >= 0`` and ``w_i >= 0`` on
              biactive pairs. Equivalent to being a KKT point of
              the tightened NLP, and to B-stationarity under
              MPCC-LICQ.
============  ==================================================

With no biactive pair the four coincide, and the classifier says so
rather than implying a discrimination it did not make.

Why it searches the multiplier set
----------------------------------

The class is a property of the multiplier *set*, not of one multiplier
vector. On `scholtes4` the set at the solution is a line: the
least-squares point of it gives ``nu = w = -1`` (C, not M), while the
point ``lambda = (1/4, 3/4)`` gives ``nu = 0`` (M). Reporting the class
of a single least-squares vector would therefore report **C** for a
point that is genuinely M-stationary. So each class is tested as a
*feasibility* question -- does a multiplier vector with these signs
reproduce the gradient? -- with `scipy.optimize.lsq_linear`, over an
explicit enumeration of the sign branches the class allows. The biactive
set is tiny here (at most one pair in this corpus, and the enumeration
is 3**b), and the classifier refuses rather than guesses above
`MAX_ENUM`.

`selftest` mutation-checks this: `ralph1` must come back M and not S,
`scholtes4` M and not S, `ctrap`'s origin C and not M, and every strict
case S.
"""

from __future__ import annotations

import itertools
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import lsq_linear

from .spec import ACTIVE_TOL, MpccCase, pair_activity

#: Largest biactive set the branch enumeration will attempt. Above it
#: the classifier reports "not enumerated" rather than a class it did
#: not establish.
MAX_ENUM = 8

_CLASSES = ("S", "M", "C", "W")


def _sets(case: MpccCase, x: np.ndarray, tol: float):
    """Index sets, using `spec.pair_activity`'s sqrt-aware threshold.

    Reading membership with a fixed tolerance is what made this
    classifier report `none` — not even weakly stationary — for points
    that had reached the optimum to nine digits; see `pair_activity`.
    """
    g, h = case.pair_values(x)
    g_act, h_act = pair_activity(g, h, tol)
    return g, h, g_act, h_act


def _columns(case: MpccCase, x: np.ndarray, tol: float):
    """Columns of the stationarity system and their base bounds."""
    n, q = case.n, case.q
    cols: List[np.ndarray] = []
    lo: List[float] = []
    hi: List[float] = []
    names: List[str] = []

    cvals = case.row_values(x)
    for k, row in enumerate(case.rows):
        cols.append(row.form.grad(x))
        names.append(f"lam[{row.name}]")
        if row.is_equality:
            lo.append(-np.inf)
            hi.append(np.inf)
        else:
            at_hi = np.isfinite(row.hi) and abs(cvals[k] - row.hi) <= tol
            at_lo = np.isfinite(row.lo) and abs(cvals[k] - row.lo) <= tol
            if at_hi and not at_lo:
                lo.append(0.0)
                hi.append(np.inf)
            elif at_lo and not at_hi:
                lo.append(-np.inf)
                hi.append(0.0)
            elif at_hi and at_lo:  # numerically pinned between two bounds
                lo.append(-np.inf)
                hi.append(np.inf)
            else:
                lo.append(0.0)
                hi.append(0.0)

    _, _, g_act, h_act = _sets(case, x, tol)
    for i, p in enumerate(case.pairs):
        cols.append(-p.G.grad(x))
        names.append(f"nu[{p.name}]")
        lo.append(-np.inf if g_act[i] else 0.0)
        hi.append(np.inf if g_act[i] else 0.0)
    for i, p in enumerate(case.pairs):
        cols.append(-p.H.grad(x))
        names.append(f"w[{p.name}]")
        lo.append(-np.inf if h_act[i] else 0.0)
        hi.append(np.inf if h_act[i] else 0.0)

    for j in range(n):
        e = np.zeros(n)
        e[j] = -1.0
        cols.append(e)
        names.append(f"zL[{j}]")
        act = np.isfinite(case.lb[j]) and abs(x[j] - case.lb[j]) <= tol
        lo.append(0.0)
        hi.append(np.inf if act else 0.0)
    for j in range(n):
        e = np.zeros(n)
        e[j] = 1.0
        cols.append(e)
        names.append(f"zU[{j}]")
        act = np.isfinite(case.ub[j]) and abs(x[j] - case.ub[j]) <= tol
        lo.append(0.0)
        hi.append(np.inf if act else 0.0)

    A = np.array(cols).T if cols else np.zeros((n, 0))
    return A, np.array(lo), np.array(hi), names, g_act, h_act


def _solve(A, b, lo, hi):
    if A.shape[1] == 0:
        return np.zeros(0), float(np.max(np.abs(b))) if b.size else 0.0
    # lsq_linear rejects a bound pair that is exactly equal, so pin the
    # forced-zero columns by dropping them instead of squeezing them.
    fixed = (lo == hi)
    free = ~fixed
    b2 = b - A[:, fixed] @ lo[fixed] if fixed.any() else b
    if not free.any():
        return np.where(fixed, lo, 0.0), float(np.max(np.abs(b2))) if b2.size else 0.0
    res = lsq_linear(A[:, free], b2, bounds=(lo[free], hi[free]), method="bvls")
    theta = np.zeros(A.shape[1])
    theta[fixed] = lo[fixed]
    theta[free] = res.x
    resid = float(np.max(np.abs(A @ theta - b)))
    return theta, resid


def _branch_bounds(lo, hi, nu_idx, w_idx, biactive, klass):
    """All sign-branch variants of ``(lo, hi)`` the class admits."""
    bi = [i for i in range(len(biactive)) if biactive[i]]
    if klass == "W":
        yield lo.copy(), hi.copy()
        return
    if klass == "S":
        l, h = lo.copy(), hi.copy()
        for i in bi:
            l[nu_idx[i]] = max(l[nu_idx[i]], 0.0)
            l[w_idx[i]] = max(l[w_idx[i]], 0.0)
        yield l, h
        return
    if klass == "C":
        options = ("pp", "nn")
    else:  # "M": both-positive, or one of the two pinned to zero
        options = ("pp", "nu0", "w0")
    for combo in itertools.product(options, repeat=len(bi)):
        l, h = lo.copy(), hi.copy()
        for i, opt in zip(bi, combo):
            a, c = nu_idx[i], w_idx[i]
            if opt == "pp":
                l[a] = max(l[a], 0.0)
                l[c] = max(l[c], 0.0)
            elif opt == "nn":
                h[a] = min(h[a], 0.0)
                h[c] = min(h[c], 0.0)
            elif opt == "nu0":
                l[a] = h[a] = 0.0
            else:
                l[c] = h[c] = 0.0
        yield l, h


def mpcc_licq(case: MpccCase, x: np.ndarray, tol: float = ACTIVE_TOL) -> Optional[bool]:
    """Are the active MPCC gradients (G and H counted separately) independent?"""
    rows: List[np.ndarray] = []
    cvals = case.row_values(x)
    for k, row in enumerate(case.rows):
        active = row.is_equality or (
            (np.isfinite(row.hi) and abs(cvals[k] - row.hi) <= tol)
            or (np.isfinite(row.lo) and abs(cvals[k] - row.lo) <= tol)
        )
        if active:
            rows.append(row.form.grad(x))
    _, _, g_act, h_act = _sets(case, x, tol)
    for i, p in enumerate(case.pairs):
        if g_act[i]:
            rows.append(p.G.grad(x))
        if h_act[i]:
            rows.append(p.H.grad(x))
    for j in range(case.n):
        if (np.isfinite(case.lb[j]) and abs(x[j] - case.lb[j]) <= tol) or (
            np.isfinite(case.ub[j]) and abs(x[j] - case.ub[j]) <= tol
        ):
            e = np.zeros(case.n)
            e[j] = 1.0
            rows.append(e)
    if not rows:
        return True
    M = np.array(rows)
    return bool(np.linalg.matrix_rank(M, tol=1e-9) == M.shape[0])


def classify(
    case: MpccCase,
    x: np.ndarray,
    act_tol: float = ACTIVE_TOL,
    resid_tol: float = 1e-6,
) -> Dict[str, object]:
    """Classify ``x`` as an MPCC stationary point.

    Returns the strongest class whose sign branches admit a multiplier
    vector reproducing ``grad f``, together with the residual of every
    class tried, the index sets, MPCC-LICQ, and the multiplier vector
    itself. Everything the verdict rests on is in the record; nothing
    has to be re-derived to audit it.
    """
    x = np.asarray(x, dtype=float)
    n, q = case.n, case.q
    A, lo, hi, names, g_act, h_act = _columns(case, x, act_tol)
    b = -case.objective.grad(x)
    # The residual tolerance is relative to the **dual scale** -- the
    # magnitude of the largest term the stationarity sum is assembled
    # from, counting every column a multiplier is allowed to be nonzero
    # in, not only the ones the fit happened to use. This is the same
    # argument `dual_inf_scale_kappa` makes about POUNCE's own strict
    # gate (see docs/src/options.md): what a scale-relative floor
    # forgives is a residual small next to the terms that produced it,
    # never a genuine non-stationarity.
    #
    # It is load-bearing on the `skew` scaling leg, and an absolute
    # threshold here would be exactly the "absolute threshold on a
    # scale-dependent quantity" that `/sens-review` entry 3 is about:
    # under `x = diag(d) xt` every gradient column carries a factor of
    # `d`, so a fixed 1e-6 declares a converged point non-stationary for
    # no reason but the units it was written in.
    free_cols = lo != hi
    col_scale = (
        float(np.max(np.abs(A[:, free_cols]))) if free_cols.any() and A.size else 0.0
    )
    # `grad f` itself is a sum, and on the `skew` leg its terms cancel by
    # orders of magnitude: at `regular_strict`'s solution under
    # `d = (1e-3, 1e3)` the second component is `2e6 * 2e-3 - 4e3`, two
    # terms of 4e3 that cancel to zero. Judging the leftover against the
    # *result* rather than against the terms would declare a point five
    # significant figures from the minimiser non-stationary. So the dual
    # scale counts the terms `grad f` is assembled from too.
    obj_scale = 0.0
    if case.objective.P.size:
        obj_scale = float(np.max(np.abs(case.objective.P * x[None, :])))
    if case.objective.c.size:
        obj_scale = max(obj_scale, float(np.max(np.abs(case.objective.c))))
    scale = max(
        1.0,
        float(np.max(np.abs(b))) if b.size else 1.0,
        col_scale,
        obj_scale,
    )
    tol = resid_tol * scale

    biactive = [bool(g_act[i] and h_act[i]) for i in range(q)]
    nb = sum(biactive)
    m_src = len(case.rows)
    nu_idx = [m_src + i for i in range(q)]
    w_idx = [m_src + q + i for i in range(q)]

    out: Dict[str, object] = {
        "act_tol": act_tol,
        "resid_tol": tol,
        "dual_scale": scale,
        "n_biactive": nb,
        "biactive": biactive,
        "regime": case.regime(x, act_tol),
        "mpcc_licq": mpcc_licq(case, x, act_tol),
        "classes_coincide": nb == 0,
        "residuals": {},
    }

    if nb > MAX_ENUM:
        out["klass"] = "not-enumerated"
        out["reason"] = f"{nb} biactive pairs exceeds MAX_ENUM={MAX_ENUM}"
        return out

    best: Optional[Tuple[str, np.ndarray, float]] = None
    for klass in _CLASSES:
        bres = np.inf
        btheta = None
        for l, h in _branch_bounds(lo, hi, nu_idx, w_idx, biactive, klass):
            theta, r = _solve(A, b, l, h)
            if r < bres:
                bres, btheta = r, theta
        out["residuals"][klass] = float(bres)
        if best is None and bres <= tol:
            best = (klass, btheta, bres)

    if best is None:
        out["klass"] = "none"
        out["reason"] = (
            "no multiplier vector reproduces grad f to the residual "
            f"tolerance {tol:.2e} (dual scale {scale:.2e}) -- the point is not "
            "even weakly stationary in the model's own units. Compare the "
            "record's nlp block: a large scaled/unscaled KKT gap there means "
            "the solve converged its internally scaled problem and this is "
            "the same statement in the user's units."
        )
        return out

    klass, theta, _ = best
    out["klass"] = klass
    out["multipliers"] = {nm: float(v) for nm, v in zip(names, theta) if abs(v) > 0.0}
    if nb == 0:
        out["note"] = (
            "no biactive pair; the four MPCC stationarity classes coincide "
            "here and S carries no extra information"
        )
    return out
