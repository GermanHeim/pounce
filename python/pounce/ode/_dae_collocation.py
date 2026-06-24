"""Fixed-mesh BDF collocation for a fully-implicit DAE, with the numpy pieces
the differentiable frontends need: a host forward solve and the transpose-
Jacobian solve for the implicit-function-theorem backward.

On a *fixed* mesh ``t_0 < ... < t_{m-1}`` the node states ``Y = (y_1, ...,
y_{m-1})`` (``y_0`` given) solve the residual ``r_k = F(t_{k+1}, y_{k+1},
y'_{k+1}) = 0``, with ``y'`` approximated by a backward-difference stencil:

* **order 1 (backward Euler):** ``y'_{k+1} = (y_{k+1} - y_k) / h_k``.
* **order 2 (BDF2):** the variable-step 3-point stencil over
  ``y_{k-1}, y_k, y_{k+1}`` (BE is used for the first step, which has no
  ``y_{k-1}``).

Both are L-stable (correct for stiff / index-1 DAEs); order is mesh-controlled.
``R_Y`` is block lower-triangular with ``order`` subdiagonals (``r_k`` couples
``y_{k+1}`` and the previous ``order`` nodes). The IFT backward needs
``R_Y^T u = v`` (FERAL sparse LU); the parameter VJP is supplied by the
frameworks' autodiff of a traced residual at ``Y*``.
"""

from __future__ import annotations

import numpy as np

from .._pounce import SparseLU

_FD = np.sqrt(np.finfo(float).eps)


def _stencil(t, k, order):
    """Backward-difference coefficients for ``y'`` at node ``k+1``.

    Returns ``(c_w, [(offset, coeff), ...])`` with ``y'_{k+1} = c_w * y_{k+1}
    + sum(coeff * y_{k+1+offset})`` over already-known nodes (offset < 0).
    """
    h = t[k + 1] - t[k]
    if order == 1 or k == 0:                      # backward Euler (also startup)
        return 1.0 / h, [(-1, -1.0 / h)]
    hm = t[k] - t[k - 1]
    rho = h / hm                                  # variable-step BDF2
    c_w = (1.0 + 2.0 * rho) / (1.0 + rho) / h
    c_k = -(1.0 + rho) / h
    c_km1 = rho * rho / (1.0 + rho) / h
    return c_w, [(-1, c_k), (-2, c_km1)]


def _node_jacs(Ffun, t1, w, wp, jac):
    """``(dF/dy, dF/dy', F0)`` at a node, analytic if ``jac`` else forward-diff."""
    F0 = np.asarray(Ffun(t1, w, wp), float)
    if jac is not None:
        Fy, Fyp = jac(t1, w, wp)
        return np.asarray(Fy, float), np.asarray(Fyp, float), F0
    n = w.size
    Fy = np.empty((n, n)); Fyp = np.empty((n, n))
    for j in range(n):
        dy = _FD * max(1.0, abs(w[j]))
        wj = w.copy(); wj[j] += dy
        Fy[:, j] = (np.asarray(Ffun(t1, wj, wp), float) - F0) / dy
        dp = _FD * max(1.0, abs(wp[j]))
        pj = wp.copy(); pj[j] += dp
        Fyp[:, j] = (np.asarray(Ffun(t1, w, pj), float) - F0) / dp
    return Fy, Fyp, F0


def _yp_known(t, Y, k, order):
    """``(c_w, yp_known)``: the ``y'`` contribution from already-known nodes
    (everything but the ``c_w * y_{k+1}`` term). ``Y`` is full ``(n, m)``."""
    c_w, terms = _stencil(t, k, order)
    yp = np.zeros(Y.shape[0])
    for off, c in terms:
        yp = yp + c * Y[:, k + 1 + off]
    return c_w, yp


def collocation_forward(Ffun, t, y0, *, order=2, jac=None, tol=1e-10,
                        maxiter=50):
    """Sequential BDF march; returns ``Y`` of shape ``(n, m)``."""
    t = np.asarray(t, float)
    y0 = np.asarray(y0, float)
    n, m = y0.size, t.size
    Y = np.empty((n, m))
    Y[:, 0] = y0
    for k in range(m - 1):
        c_w, yp_k = _yp_known(t, Y, k, order)
        w = Y[:, k].copy()                        # warm start from prev node
        for _ in range(maxiter):
            wp = c_w * w + yp_k
            Fy, Fyp, F0 = _node_jacs(Ffun, t[k + 1], w, wp, jac)
            if np.linalg.norm(F0) <= tol * (1.0 + np.linalg.norm(w)):
                break
            w = w + np.linalg.solve(Fy + c_w * Fyp, -F0)
        Y[:, k + 1] = w
    return Y


def _coo_pattern(n, M, order):
    """COO (rows, cols) for ``R_Y`` (``N = n*M``): diagonal + up to ``order``
    subdiagonal blocks (a subdiagonal is dropped when it would hit ``y_0``)."""
    rows = []; cols = []
    for k in range(M):
        r0 = k * n
        for off in range(0, order + 1):           # diag (off 0) + subdiagonals
            col_blk = k - off
            if col_blk < 0:                        # would reference y_0 (fixed)
                continue
            for a in range(n):
                for b in range(n):
                    rows.append(r0 + a); cols.append(col_blk * n + b)
    return np.asarray(rows, np.int64), np.asarray(cols, np.int64)


def _jac_values(Ffun, t, Y, n, M, order, jac):
    """Values aligned to :func:`_coo_pattern`, evaluated at ``Y`` (full)."""
    t = np.asarray(t, float)
    vals = []
    for k in range(M):
        c_w, yp_k = _yp_known(t, Y, k, order)
        w = Y[:, k + 1]
        wp = c_w * w + yp_k
        Fy, Fyp, _ = _node_jacs(Ffun, t[k + 1], w, wp, jac)
        _, terms = _stencil(t, k, order)
        sub_coeff = {-off: c for off, c in terms}   # subdiagonal -> coeff
        for sub in range(0, order + 1):
            col_blk = k - sub
            if col_blk < 0:
                continue
            if sub == 0:
                vals.append((Fy + c_w * Fyp).reshape(-1))
            else:
                vals.append((sub_coeff[sub] * Fyp).reshape(-1))
    return np.concatenate(vals)


def collocation_transpose_solve(Ffun, t, Y, v, *, order=2, jac=None):
    """Solve ``R_Y^T u = v`` at the converged nodes ``Y`` (full ``(n, m)``).

    Returns ``u`` of shape ``(n*(m-1),)`` (the unknown block ``y_1..y_{m-1}``).
    """
    n = Y.shape[0]
    M = Y.shape[1] - 1
    rows, cols = _coo_pattern(n, M, order)
    lu = SparseLU(n * M, rows, cols)
    lu.factor(_jac_values(Ffun, t, Y, n, M, order, jac))
    return np.asarray(lu.solve_transpose(np.asarray(v, float).reshape(-1)))
