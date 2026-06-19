"""SciPy-signature boundary value problem solver on top of pounce.

:func:`solve_bvp` matches the call signature and return shape of
:func:`scipy.integrate.solve_bvp`, but solves the Hermite--Simpson
collocation system (see :mod:`pounce.bvp._core`) as a **pounce feasibility
NLP** — ``min 0`` subject to the square collocation residual
``R(z) = 0`` — rather than SciPy's bespoke damped-Newton iteration.

Differences from SciPy worth knowing:

* **Fixed mesh.** The mesh ``x`` you pass is used as-is; there is no
  adaptive refinement. This is deliberate: a fixed mesh makes the
  solution map ``theta -> y`` smooth, which is what the differentiable
  ``pounce.jax`` / ``pounce.torch`` layers exploit. Refine by passing a
  denser ``x``. ``max_nodes`` is accepted for signature compatibility.
* **Derivatives.** The collocation Jacobian handed to the interior-point
  solver is formed by forward finite differences of the residual; the
  Hessian uses pounce's limited-memory quasi-Newton approximation. The
  ``fun_jac`` / ``bc_jac`` arguments are accepted for signature
  compatibility (a future revision can assemble the exact sparse
  collocation Jacobian from them).
* **Singular term ``S``** is not yet supported.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from .._pounce import Problem
from . import _core
from ._jac import CollocationJacobian


@dataclass
class BVPResult:
    """Result of :func:`solve_bvp`, mirroring SciPy's ``Bunch``.

    Attributes match :func:`scipy.integrate.solve_bvp` so existing code can
    consume the result unchanged: ``sol`` (callable cubic-Hermite
    interpolant returning shape ``(n, ...)``), ``x`` (mesh), ``y`` (states
    at the mesh, ``(n, m)``), ``yp`` (derivatives at the mesh), ``p``
    (converged unknown parameters or ``None``), ``rms_residuals``,
    ``niter``, ``status``, ``message``, ``success``.
    """

    sol: Callable
    p: Any
    x: np.ndarray
    y: np.ndarray
    yp: np.ndarray
    rms_residuals: np.ndarray
    niter: int
    status: int
    message: str
    success: bool
    info: dict = field(default_factory=dict, repr=False)


def _make_spline(x, y, yp):
    """Lazily-built cubic-Hermite interpolant ``sol(xq) -> (n, ...)`` from
    node states ``y`` ``(n, m)`` and node derivatives ``yp`` ``(n, m)``.

    Construction is deferred to the first call so callers that only read
    ``res.y`` / ``res.p`` don't pay for the spline build.
    """
    cache = {}

    def sol(xq):
        if "spline" not in cache:
            from scipy.interpolate import CubicHermiteSpline

            # CubicHermiteSpline interpolates along axis 0; feed (m, n) and
            # transpose the query result back to SciPy's (n, ...) layout.
            cache["spline"] = CubicHermiteSpline(x, y.T, yp.T)
        return cache["spline"](xq).T

    return sol


def _make_jac_adapters(fun_jac, bc_jac, uses_p, k):
    """Adapt SciPy-style ``fun_jac`` / ``bc_jac`` to the per-node block
    callables :class:`CollocationJacobian` expects, or return ``None`` to
    select the finite-difference path.

    ``fun_jac(x, y[, p])`` returns ``df_dy`` ``(n, n, mq)`` (and ``df_dp``
    ``(n, k, mq)`` when ``p`` is present); we transpose to the ``(mq, n,
    *)`` block layout. ``bc_jac(ya, yb[, p])`` returns
    ``(dbc_dya, dbc_dyb[, dbc_dp])``.
    """
    df_blocks = None
    if fun_jac is not None:
        def df_blocks(xq, Yq, p, fq):
            if uses_p:
                out = fun_jac(xq, Yq, p)
                df_dy, df_dp = out if isinstance(out, tuple) else (out, None)
            else:
                df_dy, df_dp = fun_jac(xq, Yq), None
            J = np.transpose(np.asarray(df_dy, dtype=np.float64), (2, 0, 1))
            if k > 0:
                K = np.transpose(np.asarray(df_dp, dtype=np.float64), (2, 0, 1))
            else:
                K = None
            return J, K

    dbc_block = None
    if bc_jac is not None:
        def dbc_block(ya, yb, p, b0):
            if uses_p:
                dya, dyb, dp = bc_jac(ya, yb, p)
            else:
                dya, dyb = bc_jac(ya, yb)
                dp = np.zeros((ya.shape[0], 0), dtype=np.float64)
            return (np.asarray(dya, dtype=np.float64),
                    np.asarray(dyb, dtype=np.float64),
                    np.asarray(dp, dtype=np.float64))

    return df_blocks, dbc_block


class _BvpNlp:
    """Cyipopt-shaped feasibility problem: ``min 0`` s.t. ``R(z) = 0``.

    The objective and its gradient are identically zero, so the
    interior-point method reduces to a Newton iteration on the square
    collocation residual. The constraint Jacobian is the exact **sparse**
    collocation Jacobian (:class:`CollocationJacobian`).

    The Lagrangian Hessian is supplied as **exactly zero**. This is not an
    approximation: for ``min 0`` s.t. ``R(z) = 0`` with a square,
    nonsingular ``J = dR/dz``, the KKT stationarity ``Jᵀλ = 0`` forces the
    optimal multipliers ``λ* = 0``, so the Lagrangian Hessian
    ``Σ_t λ_t ∇²R_t`` vanishes at the solution. With the zero-Hessian
    block the IPM step solves ``[[0, Jᵀ],[J, 0]] [dz; dλ] = [-Jᵀλ; -R]``,
    whose primal part is ``dz = -J⁻¹ R`` — precisely the Newton step on the
    collocation system that SciPy's ``solve_bvp`` takes (SciPy likewise
    uses only the residual Jacobian, never second derivatives of ``f``).
    Supplying it directly avoids the limited-memory quasi-Newton machinery
    and converges in one Newton step on linear problems.
    """

    def __init__(self, residual_fn, jac, n, m, k):
        self._r = residual_fn
        self._jac = jac
        self._n = n
        self._m = m
        self._N = n * m + k
        self._empty = np.zeros(0, dtype=np.float64)
        self._empty_idx = np.zeros(0, dtype=np.int64)

    def objective(self, z):
        return 0.0

    def gradient(self, z):
        return np.zeros(self._N, dtype=np.float64)

    def constraints(self, z):
        return np.asarray(self._r(z), dtype=np.float64)

    def jacobianstructure(self):
        return self._jac.structure()

    def jacobian(self, z):
        z = np.asarray(z, dtype=np.float64)
        Y = z[: self._n * self._m].reshape(self._n, self._m)
        p = z[self._n * self._m :]
        return self._jac.values(Y, p)

    def hessianstructure(self):
        return (self._empty_idx, self._empty_idx)

    def hessian(self, z, lagrange, obj_factor):
        return self._empty


def solve_bvp(
    fun,
    bc,
    x,
    y,
    p=None,
    S=None,
    fun_jac=None,
    bc_jac=None,
    tol=1e-3,
    max_nodes=1000,
    verbose=0,
    bc_tol=None,
):
    """Solve a boundary value problem on a fixed mesh with pounce.

    Drop-in for :func:`scipy.integrate.solve_bvp`. ``fun(x, y[, p])``
    returns the ``(n, m)`` RHS over the mesh; ``bc(ya, yb[, p])`` returns
    the ``n + k`` boundary residuals. See the module docstring for the
    (small) behavioural differences from SciPy.

    Returns
    -------
    BVPResult
        SciPy-compatible result bunch.
    """
    if S is not None:
        raise NotImplementedError(
            "pounce.bvp.solve_bvp does not yet support the singular term S."
        )

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if y.ndim != 2:
        raise ValueError("`y` must be 2-D with shape (n, m).")
    n, m = y.shape
    if x.shape != (m,):
        raise ValueError(f"`x` must have shape ({m},) to match `y`.")
    if np.any(np.diff(x) <= 0):
        raise ValueError("`x` must be strictly increasing.")

    uses_p = p is not None
    p0 = np.asarray(p, dtype=np.float64).ravel() if uses_p else np.zeros(0)
    k = p0.shape[0]

    nfun, nbc = _core._make_normalized(fun, bc, theta=None, uses_p=uses_p)

    # Sanity-check the boundary-residual count: collocation contributes
    # n*(m-1) equations, so bc must supply the remaining n + k.
    bc0 = np.asarray(nbc(y[:, 0], y[:, -1], p0), dtype=np.float64)
    if bc0.shape != (n + k,):
        raise ValueError(
            f"`bc` must return {n + k} residuals (n + k); got {bc0.shape}."
        )

    N = _core.num_unknowns(n, m, k)

    def residual_fn(z):
        return _core.residual_of_z(z, nfun, nbc, x, n, m, k, np.concatenate)

    df_blocks, dbc_block = _make_jac_adapters(fun_jac, bc_jac, uses_p, k)
    jac = CollocationJacobian(
        nfun, nbc, x, n, m, k, df_blocks=df_blocks, dbc_block=dbc_block,
    )
    obj = _BvpNlp(residual_fn, jac, n, m, k)
    cl = np.zeros(N, dtype=np.float64)
    cu = np.zeros(N, dtype=np.float64)
    problem = Problem(n=N, m=N, problem_obj=obj, cl=cl, cu=cu)
    problem.add_option("tol", float(tol))
    # Collocation residuals are naturally well-scaled; skip the scaling
    # pass (its setup cost buys nothing here).
    problem.add_option("nlp_scaling_method", "none")
    problem.add_option("print_level", 5 if verbose >= 2 else 0)

    z0 = _core.pack_z(y, p0, np.concatenate)
    z_star, info = problem.solve(x0=z0)
    z_star = np.asarray(z_star, dtype=np.float64)

    Y, p_star = _core.unpack_z(z_star, n, m)
    Y = np.array(Y)
    yp = np.asarray(nfun(x, Y, p_star), dtype=np.float64)

    # Per-interval RMS of the collocation residual (state-major block).
    r_star = residual_fn(z_star)
    col = r_star[: n * (m - 1)].reshape(n, m - 1)
    rms_residuals = np.sqrt(np.mean(col**2, axis=0))

    status = 0 if info.get("status", 1) in (0, 1) else 1
    success = status == 0
    message = info.get("status_msg", "")
    sol = _make_spline(x, Y, yp)

    return BVPResult(
        sol=sol,
        p=(p_star.copy() if uses_p else None),
        x=x,
        y=Y,
        yp=yp,
        rms_residuals=rms_residuals,
        niter=int(info.get("iter_count", 0)),
        status=status,
        message=message,
        success=success,
        info=info,
    )
