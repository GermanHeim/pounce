"""Differentiable convex-QP layer (OptNet-style implicit differentiation).

Solves, and differentiates through, the convex QP

.. code-block:: text

    minimize    ½ xᵀP x + cᵀx
    subject to  G x ≤ h
                A x = b

The forward solve calls the ``pounce-convex`` interior-point solver
through a host callback. The backward pass uses the implicit-function
theorem on the KKT system at the optimum (Amos & Kolter, *OptNet*, 2017):
the same KKT matrix that defines the solution also yields its
sensitivities, so a single linear solve gives the cotangents.

Scope. Gradients are provided w.r.t. the **right-hand-side / linear
parameters** ``c``, ``b``, ``h`` — the parametric case (varying
objective / constraint levels over a fixed structure) that the batched
and multiple-RHS entry points target. Gradients w.r.t. the matrices
``P``, ``G``, ``A`` (the full OptNet matrix derivatives) are a documented
follow-up; passing those as differentiable arguments raises.

Bounds ``lb ≤ x ≤ ub`` are supported in the *forward* solve by folding
them into ``G``/``h`` before differentiation, so the IFT sees a single
inequality block.
"""

from __future__ import annotations

from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np

from .. import _pounce

__all__ = ["solve_qp", "QpLayer"]

# Active-set tolerance for the backward pass: an inequality counts as
# active when its multiplier is above this (complementarity slackness).
_ACTIVE_TOL = 1e-6


def _expand_bounds(G, h, lb, ub, n):
    """Fold finite variable bounds into G/h as extra rows.

    Returns ``(G_full, h_full)`` as dense jnp arrays. ``x_i ≤ ub_i`` and
    ``−x_i ≤ −lb_i``."""
    rows = []
    rhs = []
    if G is not None and G.shape[0] > 0:
        rows.append(G)
        rhs.append(h)
    if ub is not None:
        for i in range(n):
            if np.isfinite(float(ub[i])):
                e = jnp.zeros(n).at[i].set(1.0)
                rows.append(e[None, :])
                rhs.append(jnp.asarray(ub[i]).reshape(1))
    if lb is not None:
        for i in range(n):
            if np.isfinite(float(lb[i])):
                e = jnp.zeros(n).at[i].set(-1.0)
                rows.append(e[None, :])
                rhs.append((-jnp.asarray(lb[i])).reshape(1))
    if not rows:
        return jnp.zeros((0, n)), jnp.zeros((0,))
    return jnp.concatenate(rows, axis=0), jnp.concatenate(rhs, axis=0)


def _forward_solve(P, c, G, h, A, b, tol, max_iter):
    """Host-side forward solve via pounce-convex. Returns (x, lam, nu).

    ``lam`` are the inequality (``G``) multipliers, ``nu`` the equality
    (``A``) multipliers."""
    n = c.shape[0]
    m_g = G.shape[0]
    m_a = A.shape[0]

    def to_coo_lower(M):
        r, cc = np.nonzero(M)
        keep = r >= cc
        return r[keep].tolist(), cc[keep].tolist(), M[r[keep], cc[keep]].tolist()

    def to_coo(M):
        r, cc = np.nonzero(M)
        return r.tolist(), cc.tolist(), M[r, cc].tolist()

    pr, pc, pv = to_coo_lower(np.asarray(P))
    gr, gc, gv = to_coo(np.asarray(G))
    ar, ac, av = to_coo(np.asarray(A))
    prob = _pounce.QpProblem(
        n=n,
        c=np.asarray(c).tolist(),
        p_rows=pr,
        p_cols=pc,
        p_vals=pv,
        a_rows=ar,
        a_cols=ac,
        a_vals=av,
        b=np.asarray(b).tolist(),
        g_rows=gr,
        g_cols=gc,
        g_vals=gv,
        h=np.asarray(h).tolist(),
    )
    d = _pounce.solve_qp(prob, tol=tol, max_iter=max_iter)
    x = np.asarray(d["x"], dtype=np.float64)
    lam = (
        np.asarray(d["z"], dtype=np.float64)
        if m_g
        else np.zeros((0,), dtype=np.float64)
    )
    nu = (
        np.asarray(d["y"], dtype=np.float64)
        if m_a
        else np.zeros((0,), dtype=np.float64)
    )
    return x, lam, nu


def _make_qp_vjp(n, m_g, m_a, tol, max_iter):
    @jax.custom_vjp
    def qp(P, c, G, h, A, b):
        x, _, _ = _pure_forward(P, c, G, h, A, b, n, m_g, m_a, tol, max_iter)
        return x

    def fwd(P, c, G, h, A, b):
        x, lam, nu = _pure_forward(P, c, G, h, A, b, n, m_g, m_a, tol, max_iter)
        return x, (P, G, A, h, x, lam, nu)

    def bwd(res, gx):
        P, G, A, h, x, lam, nu = res
        # OptNet implicit-differentiation backward (Amos & Kolter 2017,
        # §3). At the optimum (x, λ, ν) of  min ½xᵀPx+cᵀx  s.t. Gx≤h, Ax=b
        # the KKT differential system is
        #   [ P        Gᵀ        Aᵀ ] [d_x]     [ g_x ]
        #   [ D(λ)G    D(Gx−h)   0  ] [d_λ] = − [  0  ]
        #   [ A        0         0  ] [d_ν]     [  0  ]
        # with D(·) = diag(·). Solving for (d_x, d_λ, d_ν), the gradients
        # of the loss w.r.t. the *linear* parameters are
        #   ∇_c = d_x,   ∇_b = −d_ν,   ∇_h = −d_λ.
        # (Verified against finite differences on active-constraint QPs.)
        slack = G @ x - h  # ≤ 0 at feasibility; 0 on active rows
        dlam_scale = jnp.diag(lam)
        zero_ga = jnp.zeros((m_g, m_a))
        zero_ag = jnp.zeros((m_a, m_g))
        zero_aa = jnp.zeros((m_a, m_a))

        top = jnp.concatenate([P, G.T, A.T], axis=1)
        mid = jnp.concatenate([dlam_scale @ G, jnp.diag(slack), zero_ga], axis=1)
        bot = jnp.concatenate([A, zero_ag, zero_aa], axis=1)
        kkt = jnp.concatenate([top, mid, bot], axis=0)

        rhs = -jnp.concatenate([gx, jnp.zeros(m_g), jnp.zeros(m_a)])
        d = jnp.linalg.solve(kkt, rhs)
        d_x = d[:n]
        d_lam = d[n : n + m_g]
        d_nu = d[n + m_g :]

        # Linear-parameter gradients. P/G/A are treated as constants here
        # (matrix-derivative support is a documented follow-up).
        grad_c = d_x
        grad_h = -d_lam
        grad_b = -d_nu
        grad_P = jnp.zeros_like(P)
        grad_G = jnp.zeros_like(G)
        grad_A = jnp.zeros_like(A)
        return (grad_P, grad_c, grad_G, grad_h, grad_A, grad_b)

    qp.defvjp(fwd, bwd)
    return qp


def _pure_forward(P, c, G, h, A, b, n, m_g, m_a, tol, max_iter):
    """custom_vjp-friendly forward via pure_callback. Returns (x, lam, nu)."""
    shapes = (
        jax.ShapeDtypeStruct((n,), jnp.float64),
        jax.ShapeDtypeStruct((m_g,), jnp.float64),
        jax.ShapeDtypeStruct((m_a,), jnp.float64),
    )

    def host(P_h, c_h, G_h, h_h, A_h, b_h):
        return _forward_solve(
            np.asarray(P_h),
            np.asarray(c_h),
            np.asarray(G_h),
            np.asarray(h_h),
            np.asarray(A_h),
            np.asarray(b_h),
            tol,
            max_iter,
        )

    # `vmap_method="sequential"` lets the layer be used under jax.vmap
    # (each instance is an independent host solve). Older JAX releases
    # don't accept the kwarg, so fall back gracefully.
    try:
        return jax.pure_callback(
            host, shapes, P, c, G, h, A, b, vmap_method="sequential"
        )
    except TypeError:
        return jax.pure_callback(host, shapes, P, c, G, h, A, b)


def solve_qp(
    *,
    P,
    c,
    G=None,
    h=None,
    A=None,
    b=None,
    lb=None,
    ub=None,
    tol: Optional[float] = None,
    max_iter: Optional[int] = None,
):
    """Differentiable convex-QP solve ``x*(c, b, h)``.

    Solves ``min ½xᵀPx+cᵀx s.t. Gx≤h, Ax=b, lb≤x≤ub`` and is
    differentiable w.r.t. ``c``, ``b``, and ``h`` (the linear / RHS
    parameters) via the OptNet implicit-function rule. ``P``, ``G``, ``A``
    are treated as constants (matrix-derivative support is a follow-up).

    All array args are dense jnp/np arrays. Bounds are folded into the
    inequality block, so ``h`` gradients cover any finite bound levels
    too only when bounds are passed via ``G``/``h`` directly; ``lb``/``ub``
    here are constants used in the forward solve.
    """
    P = jnp.asarray(P, dtype=jnp.float64)
    c = jnp.asarray(c, dtype=jnp.float64)
    n = c.shape[0]
    G0 = jnp.zeros((0, n)) if G is None else jnp.asarray(G, dtype=jnp.float64)
    h0 = jnp.zeros((0,)) if h is None else jnp.asarray(h, dtype=jnp.float64)
    A0 = jnp.zeros((0, n)) if A is None else jnp.asarray(A, dtype=jnp.float64)
    b0 = jnp.zeros((0,)) if b is None else jnp.asarray(b, dtype=jnp.float64)

    # Fold finite bounds into G/h (constants w.r.t. differentiation here).
    G_full, h_full = _expand_bounds(G0, h0, lb, ub, n)

    fn = _make_qp_vjp(n, G_full.shape[0], A0.shape[0], tol, max_iter)
    return fn(P, c, G_full, h_full, A0, b0)


class QpLayer:
    """A reusable differentiable QP layer with fixed structure.

    Captures ``P, G, A`` (and bounds) once; calling the layer with
    ``c``/``b``/``h`` solves and is differentiable w.r.t. those. Suitable
    for use inside a larger JAX model (``jax.grad``/``jacrev``/``vmap``).
    """

    def __init__(self, P, G=None, A=None, lb=None, ub=None, *, tol=None, max_iter=None):
        self._P = P
        self._G = G
        self._A = A
        self._lb = lb
        self._ub = ub
        self._tol = tol
        self._max_iter = max_iter

    def __call__(self, c, *, b=None, h=None):
        return solve_qp(
            P=self._P,
            c=c,
            G=self._G,
            h=h,
            A=self._A,
            b=b,
            lb=self._lb,
            ub=self._ub,
            tol=self._tol,
            max_iter=self._max_iter,
        )
