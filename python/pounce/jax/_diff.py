"""Differentiate through the solver via implicit differentiation.

Setup. For a parametric NLP

    min_x  f(x, p)
    s.t.   g(x, p) = 0
           x_L <= x <= x_U,

the KKT conditions at the optimum ``x*(p)`` are

    ∇_x L(x*, λ*, p) = 0      with  L = f + λᵀ g    (active-set L)
              g(x*, p)  = 0

(plus complementarity on the bound multipliers — we treat the active
set as locally fixed; this is the standard implicit-function
assumption). Differentiating w.r.t. ``p`` and grouping into a 2×2 KKT
block,

    ⎡ H_xx   J_gxᵀ ⎤ ⎡ dx/dp ⎤     ⎡ ∂_p ∇_x L ⎤
    ⎣ J_gx     0   ⎦ ⎣ dλ/dp ⎦ = - ⎣ ∂_p g     ⎦.

For a cotangent ``v`` on ``x*``, the VJP w.r.t. ``p`` is computed by
solving the *transposed* KKT system, which is symmetric here:

    ⎡ H_xx   J_gxᵀ ⎤ ⎡ u_x ⎤   ⎡ v ⎤
    ⎣ J_gx     0   ⎦ ⎣ u_λ ⎦ = ⎣ 0 ⎦,

then    dL/dp = - u_xᵀ (∂_p ∇_x L) - u_λᵀ (∂_p g).

We assemble the dense KKT matrix from the JAX-AD Hessian and Jacobian
and solve it with ``jax.scipy.linalg.solve``. This keeps the
backward fully traced and itself differentiable (so you can take
second derivatives through the solver if you ever need to). For large
sparse problems the right move is to expose a Rust-side
sensitivity-solve via ``pounce-sensitivity``; that lands as a follow-up
once the JAX prototype is exercising the surface.

Bounds. Active variable bounds reduce dx/dp to zero on the active
coordinates. We detect activity from the optimizer's bound multipliers
``info['mult_x_L']`` / ``info['mult_x_U']`` (above
``active_tol``) and project the cotangent / right-hand-side onto the
inactive set before the KKT solve, then scatter back.

General inequalities. A two-sided constraint row ``cl[i] <= g_i(x)
<= cu[i]`` is *active* at the optimum iff (a) it is an equality
(``cl[i] == cu[i]``) or (b) ``|mult_g_i| > active_tol`` (binding at
one side). Slack inequality rows are dropped from the KKT block
(zeroed and identity-augmented on the multiplier diagonal) so the
sensitivity is taken over the active manifold only — including a
slack row as if it were ``g_i(x) = 0`` over-constrains ``dx*/dp``
and silently returns the wrong gradient (pounce#73).
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np

from ._build import _JaxProblem
from .._pounce import Problem

_ACTIVE_TOL = 1e-6


def _solve_once(
    f: Callable,
    g: Callable | None,
    p: jnp.ndarray,
    x0: jnp.ndarray,
    n: int,
    m: int,
    lb,
    ub,
    cl,
    cu,
    options: dict | None,
) -> tuple[np.ndarray, dict]:
    """Forward solve. ``p`` is closed over by ``f`` / ``g`` via partial."""

    def f_of_x(x):
        return f(x, p)

    if g is not None:
        def g_of_x(x):
            return g(x, p)
    else:
        g_of_x = None

    obj = _JaxProblem(f=f_of_x, g=g_of_x, n=n, m=m)
    problem = Problem(n=n, m=m, problem_obj=obj, lb=lb, ub=ub, cl=cl, cu=cu)
    if options:
        for k, v in options.items():
            problem.add_option(k, v)
    x_np, info = problem.solve(x0=np.asarray(x0))
    return np.asarray(x_np), info


def _make_solve_custom_vjp(
    f: Callable,
    g: Callable | None,
    n: int,
    m: int,
    lb,
    ub,
    cl,
    cu,
    options: dict | None,
):
    @jax.custom_vjp
    def solve_fn(p, x0):
        # Pure-callback to Python. The forward returns only x*; the
        # backward needs (x*, λ*, mult_x_L, mult_x_U) so we re-pack
        # them via the residual.
        x_star, _info = _pure_callback_solve(f, g, p, x0, n, m, lb, ub, cl, cu, options)
        return x_star

    def fwd(p, x0):
        x_star, info = _pure_callback_solve(f, g, p, x0, n, m, lb, ub, cl, cu, options)
        lam = jnp.asarray(info["mult_g"]) if m > 0 else jnp.zeros(0)
        mult_xL = jnp.asarray(info["mult_x_L"])
        mult_xU = jnp.asarray(info["mult_x_U"])
        return x_star, (p, x_star, lam, mult_xL, mult_xU)

    def bwd(residuals, cotangent_x):
        p, x_star, lam, mult_xL, mult_xU = residuals
        v = cotangent_x

        # Detect active variable bounds (|mult| > tol → bound binds → dx/dp = 0).
        active = (mult_xL > _ACTIVE_TOL) | (mult_xU > _ACTIVE_TOL)
        inactive = ~active

        # AD-build the Lagrangian Hessian and Jacobian at (x*, λ*, p).
        def lagrangian(x, p_):
            base = f(x, p_)
            if g is not None and m > 0:
                base = base + jnp.dot(lam, g(x, p_))
            return base

        H = jax.hessian(lagrangian, argnums=0)(x_star, p)
        # ∂_p ∇_x L  — partial Jacobian of grad-L w.r.t. p.
        grad_L_of_p = lambda p_: jax.grad(lagrangian, argnums=0)(x_star, p_)
        dgradL_dp = jax.jacrev(grad_L_of_p)(p)  # shape (n, *p_shape)

        if g is not None and m > 0:
            J = jax.jacrev(g, argnums=0)(x_star, p)
            dg_dp = jax.jacrev(lambda p_: g(x_star, p_))(p)  # (m, *p_shape)
            # Constraint-row active set: equalities are always active;
            # inequalities are active iff their multiplier is non-zero.
            # Slack inequality rows drop out of the KKT block via the
            # same identity-augment trick used for active bounds —
            # pounce#73 (without this, slack ineqs are kept as
            # equalities and the gradient is silently wrong).
            cl_arr = jnp.asarray(cl, dtype=H.dtype)
            cu_arr = jnp.asarray(cu, dtype=H.dtype)
            is_equality = cl_arr == cu_arr
            cons_active = is_equality | (jnp.abs(lam) > _ACTIVE_TOL)
            cons_inactive = ~cons_active
        else:
            J = jnp.zeros((0, n))
            dg_dp = jnp.zeros((0,) + jnp.shape(p))
            cons_inactive = jnp.zeros((0,), dtype=bool)

        # Project to inactive variables.
        idx = jnp.where(inactive, jnp.arange(n), n)  # n sentinel for masked-out
        keep = jnp.nonzero(inactive, size=n, fill_value=-1)[0]
        # We can't dynamically size arrays inside jit, so do a static
        # version: zero out rows/cols belonging to active vars, replace
        # diagonal with 1 so the system stays invertible, and zero the
        # RHS on those rows. This is the standard "augment with
        # identity on the active set" trick.
        active_mat = jnp.diag(active.astype(H.dtype))
        H_eff = jnp.where(
            active[:, None] | active[None, :], 0.0, H
        ) + active_mat
        # Zero variable-active columns AND constraint-inactive rows.
        J_eff = jnp.where(
            cons_inactive[:, None] | active[None, :], 0.0, J
        )
        v_eff = jnp.where(active, 0.0, v)

        # Assemble [[H, Jᵀ], [J, D]] u = [v; 0]   where D = diag(cons_inactive)
        # so each slack row reads `1 · u_lam[i] = 0` and drops out.
        if m > 0:
            cons_inactive_diag = jnp.diag(cons_inactive.astype(H.dtype))
            top = jnp.concatenate([H_eff, J_eff.T], axis=1)
            bot = jnp.concatenate([J_eff, cons_inactive_diag], axis=1)
            K = jnp.concatenate([top, bot], axis=0)
            rhs = jnp.concatenate([v_eff, jnp.zeros(m, dtype=H.dtype)])
            u = jnp.linalg.solve(K, rhs)
            u_x, u_lam = u[:n], u[n:]
        else:
            u_x = jnp.linalg.solve(H_eff, v_eff)
            u_lam = jnp.zeros(0)

        # Contract with the parameter sensitivities. The minus sign
        # comes from rearranging dKKT/dp = 0 into the form above.
        # u_x has shape (n,); dgradL_dp has shape (n, *p_shape).
        # u_lam has shape (m,); dg_dp has shape (m, *p_shape).
        dL_dp = -jnp.tensordot(u_x, dgradL_dp, axes=1)
        if m > 0:
            dL_dp = dL_dp - jnp.tensordot(u_lam, dg_dp, axes=1)
        # The x0 input has no sensitivity through x* (the solver is
        # deterministic at optimum); return zeros.
        return dL_dp, jnp.zeros_like(idx, dtype=jnp.float64)

    solve_fn.defvjp(fwd, bwd)
    return solve_fn


def _pure_callback_solve(f, g, p, x0, n, m, lb, ub, cl, cu, options):
    """JAX pure_callback wrapper around :func:`_solve_once`.

    Returns ``(x_star, info)`` where ``info`` is a dict of arrays.
    The shapes are static so JAX can trace through cleanly.
    """
    result_shapes = (
        jax.ShapeDtypeStruct((n,), jnp.float64),
        {
            "obj_val": jax.ShapeDtypeStruct((), jnp.float64),
            "status": jax.ShapeDtypeStruct((), jnp.int32),
            "iter_count": jax.ShapeDtypeStruct((), jnp.int32),
            "g": jax.ShapeDtypeStruct((m,), jnp.float64),
            "mult_g": jax.ShapeDtypeStruct((m,), jnp.float64),
            "mult_x_L": jax.ShapeDtypeStruct((n,), jnp.float64),
            "mult_x_U": jax.ShapeDtypeStruct((n,), jnp.float64),
        },
    )

    def host_call(p_h, x0_h):
        x_np, info = _solve_once(
            f=f, g=g,
            p=jnp.asarray(p_h),
            x0=jnp.asarray(x0_h),
            n=n, m=m, lb=lb, ub=ub, cl=cl, cu=cu,
            options=options,
        )
        info_out = {
            "obj_val": np.float64(info["obj_val"]),
            "status": np.int32(info["status"]),
            "iter_count": np.int32(info["iter_count"]),
            "g": np.asarray(info["g"], dtype=np.float64),
            "mult_g": np.asarray(info["mult_g"], dtype=np.float64),
            "mult_x_L": np.asarray(info["mult_x_L"], dtype=np.float64),
            "mult_x_U": np.asarray(info["mult_x_U"], dtype=np.float64),
        }
        return np.asarray(x_np, dtype=np.float64), info_out

    return jax.pure_callback(host_call, result_shapes, p, x0)


def solve(
    p,
    *,
    f: Callable,
    g: Callable | None = None,
    x0,
    n: int,
    m: int = 0,
    lb=None,
    ub=None,
    cl=None,
    cu=None,
    options: dict | None = None,
):
    """Parametric solve. ``x* = solve(p, f=..., g=..., x0=..., ...)``.

    Differentiable w.r.t. ``p`` via the implicit-function rule on the
    KKT system at ``x*(p)``. Not differentiable w.r.t. ``x0``.

    ``f`` and ``g`` must take ``(x, p)`` and be JAX-traceable.
    """
    fn = _make_solve_custom_vjp(f, g, n, m, lb, ub, cl, cu, options)
    return fn(p, x0)


def _solve_once_warm(
    f: Callable,
    g: Callable | None,
    p: jnp.ndarray,
    x0: jnp.ndarray,
    n: int,
    m: int,
    lb,
    ub,
    cl,
    cu,
    options: dict | None,
    lam_warm: np.ndarray,
    zL_warm: np.ndarray,
    zU_warm: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """Forward solve with user-supplied dual warm-start."""

    def f_of_x(x):
        return f(x, p)

    if g is not None:
        def g_of_x(x):
            return g(x, p)
    else:
        g_of_x = None

    obj = _JaxProblem(f=f_of_x, g=g_of_x, n=n, m=m)
    problem = Problem(n=n, m=m, problem_obj=obj, lb=lb, ub=ub, cl=cl, cu=cu)
    merged = dict(options or {})
    merged.setdefault("warm_start_init_point", "yes")
    for k, v in merged.items():
        problem.add_option(k, v)
    x_np, info = problem.solve(
        x0=np.asarray(x0),
        lagrange=np.asarray(lam_warm),
        zl=np.asarray(zL_warm),
        zu=np.asarray(zU_warm),
    )
    return (
        np.asarray(x_np, dtype=np.float64),
        np.asarray(info["mult_g"], dtype=np.float64),
        np.asarray(info["mult_x_L"], dtype=np.float64),
        np.asarray(info["mult_x_U"], dtype=np.float64),
        info,
    )


def _pure_callback_warm_solve(
    f, g, p, x0, n, m, lb, ub, cl, cu, options,
    lam_warm, zL_warm, zU_warm,
):
    """Pure-callback wrapper around :func:`_solve_once_warm`.

    Returns ``(x*, lam_out, zL_out, zU_out)`` — the four arrays the
    bwd needs (lam_out, the bound multipliers) and the warm-state
    triple the user threads into the next call.
    """
    result_shapes = (
        jax.ShapeDtypeStruct((n,), jnp.float64),
        jax.ShapeDtypeStruct((m,), jnp.float64),
        jax.ShapeDtypeStruct((n,), jnp.float64),
        jax.ShapeDtypeStruct((n,), jnp.float64),
    )

    def host_call(p_h, x0_h, lam_h, zL_h, zU_h):
        x_np, lam_out, zL_out, zU_out, _info = _solve_once_warm(
            f=f, g=g,
            p=jnp.asarray(p_h),
            x0=jnp.asarray(x0_h),
            n=n, m=m, lb=lb, ub=ub, cl=cl, cu=cu,
            options=options,
            lam_warm=lam_h, zL_warm=zL_h, zU_warm=zU_h,
        )
        return x_np, lam_out, zL_out, zU_out

    return jax.pure_callback(
        host_call, result_shapes, p, x0, lam_warm, zL_warm, zU_warm,
    )


def _make_solve_with_warm_custom_vjp(
    f: Callable,
    g: Callable | None,
    n: int,
    m: int,
    lb,
    ub,
    cl,
    cu,
    options: dict | None,
):
    @jax.custom_vjp
    def solve_fn(p, x0, lam_warm, zL_warm, zU_warm):
        x_star, lam_out, zL_out, zU_out = _pure_callback_warm_solve(
            f, g, p, x0, n, m, lb, ub, cl, cu, options,
            lam_warm, zL_warm, zU_warm,
        )
        return x_star, lam_out, zL_out, zU_out

    def fwd(p, x0, lam_warm, zL_warm, zU_warm):
        x_star, lam_out, zL_out, zU_out = _pure_callback_warm_solve(
            f, g, p, x0, n, m, lb, ub, cl, cu, options,
            lam_warm, zL_warm, zU_warm,
        )
        return (
            (x_star, lam_out, zL_out, zU_out),
            (p, x_star, lam_out, zL_out, zU_out),
        )

    def bwd(residuals, cotangents):
        p, x_star, lam, mult_xL, mult_xU = residuals
        # Only the x* cotangent contributes a gradient w.r.t. p.
        # Cotangents on (lam_out, zL_out, zU_out) are dropped — same
        # pattern existing `solve` uses for x0: warm dual outputs
        # don't carry differentiable info back to p in the implicit
        # rule (they're consequences of the active set, not inputs).
        v = cotangents[0]

        active = (mult_xL > _ACTIVE_TOL) | (mult_xU > _ACTIVE_TOL)
        inactive = ~active

        def lagrangian(x, p_):
            base = f(x, p_)
            if g is not None and m > 0:
                base = base + jnp.dot(lam, g(x, p_))
            return base

        H = jax.hessian(lagrangian, argnums=0)(x_star, p)
        grad_L_of_p = lambda p_: jax.grad(lagrangian, argnums=0)(x_star, p_)
        dgradL_dp = jax.jacrev(grad_L_of_p)(p)

        if g is not None and m > 0:
            J = jax.jacrev(g, argnums=0)(x_star, p)
            dg_dp = jax.jacrev(lambda p_: g(x_star, p_))(p)
            cl_arr = jnp.asarray(cl, dtype=H.dtype)
            cu_arr = jnp.asarray(cu, dtype=H.dtype)
            is_equality = cl_arr == cu_arr
            cons_active = is_equality | (jnp.abs(lam) > _ACTIVE_TOL)
            cons_inactive = ~cons_active
        else:
            J = jnp.zeros((0, n))
            dg_dp = jnp.zeros((0,) + jnp.shape(p))
            cons_inactive = jnp.zeros((0,), dtype=bool)

        active_mat = jnp.diag(active.astype(H.dtype))
        H_eff = jnp.where(
            active[:, None] | active[None, :], 0.0, H
        ) + active_mat
        J_eff = jnp.where(
            cons_inactive[:, None] | active[None, :], 0.0, J
        )
        v_eff = jnp.where(active, 0.0, v)

        if m > 0:
            cons_inactive_diag = jnp.diag(cons_inactive.astype(H.dtype))
            top = jnp.concatenate([H_eff, J_eff.T], axis=1)
            bot = jnp.concatenate([J_eff, cons_inactive_diag], axis=1)
            K = jnp.concatenate([top, bot], axis=0)
            rhs = jnp.concatenate([v_eff, jnp.zeros(m, dtype=H.dtype)])
            u = jnp.linalg.solve(K, rhs)
            u_x, u_lam = u[:n], u[n:]
        else:
            u_x = jnp.linalg.solve(H_eff, v_eff)
            u_lam = jnp.zeros(0)

        dL_dp = -jnp.tensordot(u_x, dgradL_dp, axes=1)
        if m > 0:
            dL_dp = dL_dp - jnp.tensordot(u_lam, dg_dp, axes=1)

        return (
            dL_dp,
            jnp.zeros((n,), dtype=jnp.float64),
            jnp.zeros((m,), dtype=jnp.float64),
            jnp.zeros((n,), dtype=jnp.float64),
            jnp.zeros((n,), dtype=jnp.float64),
        )

    solve_fn.defvjp(fwd, bwd)
    return solve_fn


def solve_with_warm(
    p,
    *,
    f: Callable,
    g: Callable | None = None,
    x0,
    n: int,
    m: int = 0,
    lb=None,
    ub=None,
    cl=None,
    cu=None,
    options: dict | None = None,
    warm_start: tuple | None = None,
):
    """Parametric solve that consumes and returns dual warm-state.

    Like :func:`solve`, but:

    * ``warm_start=(lam, zL, zU)`` seeds the solver's dual variables
      via IPOPT's ``warm_start_init_point=yes`` machinery. Pass
      ``None`` to start from zeros (still warm-starts the option,
      but with no informative duals — useful for the *first* call
      in a sequence where you want a uniform code path).
    * Returns ``(x*, (lam_out, zL_out, zU_out))`` so the caller
      can thread the dual triple into the next call.

    The forward call is differentiable w.r.t. ``p`` only — cotangents
    on the warm-output duals and the warm-input duals are dropped
    (zero), matching how :func:`solve` handles ``x0``. This is the
    implicit-function-theorem fix point: at the optimum the duals
    are a function of ``p`` and the active set, not an independent
    input feeding into ``dx*/dp``.

    Typical use::

        x0, lam, zL, zU = init_state(...)
        for p_k in trajectory:
            x_star, (lam, zL, zU) = solve_with_warm(
                p_k, f=f, g=g, x0=x0, n=n, m=m,
                lb=lb, ub=ub, cl=cl, cu=cu,
                warm_start=(lam, zL, zU),
            )
            x0 = x_star  # primal warm-start for free
    """
    if warm_start is None:
        lam_warm = jnp.zeros(m, dtype=jnp.float64)
        zL_warm = jnp.zeros(n, dtype=jnp.float64)
        zU_warm = jnp.zeros(n, dtype=jnp.float64)
    else:
        lam_warm, zL_warm, zU_warm = warm_start
        lam_warm = jnp.asarray(lam_warm, dtype=jnp.float64)
        zL_warm = jnp.asarray(zL_warm, dtype=jnp.float64)
        zU_warm = jnp.asarray(zU_warm, dtype=jnp.float64)

    fn = _make_solve_with_warm_custom_vjp(f, g, n, m, lb, ub, cl, cu, options)
    x_star, lam_out, zL_out, zU_out = fn(p, x0, lam_warm, zL_warm, zU_warm)
    return x_star, (lam_out, zL_out, zU_out)


def vmap_solve(
    p_batch,
    *,
    f: Callable,
    g: Callable | None = None,
    x0,
    n: int,
    m: int = 0,
    lb=None,
    ub=None,
    cl=None,
    cu=None,
    options: dict | None = None,
):
    """Batched solve over the leading axis of ``p_batch``.

    The pounce solver is single-threaded and stateful, so a literal
    ``jax.vmap`` of :func:`solve` would unsafely lift the pure_callback.
    This helper instead loops in Python (or, when JAX provides a
    sequential map primitive, dispatches to that), preserving
    differentiability via :func:`solve`'s ``custom_vjp``.
    """
    p_batch = jnp.asarray(p_batch)
    batch = p_batch.shape[0]

    def one(p_i):
        return solve(
            p_i, f=f, g=g, x0=x0, n=n, m=m,
            lb=lb, ub=ub, cl=cl, cu=cu, options=options,
        )

    # ``jax.lax.map`` runs sequentially under the hood (one element at
    # a time), which is exactly what we want for an impure callback.
    return jax.lax.map(one, p_batch)


def _solve_batch_threadpool(
    f, g, p_batch_np, x0_np, n, m, lb, ub, cl, cu, options, workers,
):
    """Dispatch ``B`` independent solves across a ``ThreadPoolExecutor``.

    Each worker builds its own ``Problem`` (no shared state) and runs
    ``Problem.solve``. Genuine parallelism is unlocked by the
    ``py.allow_threads`` block around ``optimize_tnlp`` in
    ``pounce-py`` — the GIL is released across the IPM iteration so
    threads actually run concurrently on the Rust side. JAX-traced
    ``f`` / ``g`` callbacks reacquire the GIL the usual way; that's
    serialized but the per-step cost is small relative to the linear
    algebra.
    """
    B = p_batch_np.shape[0]
    n_workers = workers or min(B, 8)
    x_out = np.empty((B, n), dtype=np.float64)
    lam_out = np.empty((B, m), dtype=np.float64)
    zL_out = np.empty((B, n), dtype=np.float64)
    zU_out = np.empty((B, n), dtype=np.float64)

    def one(i):
        x_np, info = _solve_once(
            f=f, g=g,
            p=jnp.asarray(p_batch_np[i]),
            x0=jnp.asarray(x0_np[i]) if x0_np.ndim == 2 else jnp.asarray(x0_np),
            n=n, m=m, lb=lb, ub=ub, cl=cl, cu=cu,
            options=options,
        )
        x_out[i] = x_np
        lam_out[i] = np.asarray(info["mult_g"], dtype=np.float64)
        zL_out[i] = np.asarray(info["mult_x_L"], dtype=np.float64)
        zU_out[i] = np.asarray(info["mult_x_U"], dtype=np.float64)

    if n_workers <= 1 or B <= 1:
        for i in range(B):
            one(i)
    else:
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            list(pool.map(one, range(B)))
    return x_out, lam_out, zL_out, zU_out


def _make_vmap_solve_parallel_custom_vjp(
    f: Callable,
    g: Callable | None,
    n: int,
    m: int,
    lb,
    ub,
    cl,
    cu,
    options: dict | None,
    workers: int | None,
):
    @jax.custom_vjp
    def solve_fn(p_batch, x0_batch):
        x_star, *_ = _pure_callback_parallel_solve(
            f, g, p_batch, x0_batch, n, m, lb, ub, cl, cu, options, workers,
        )
        return x_star

    def fwd(p_batch, x0_batch):
        x_star, lam, mult_xL, mult_xU = _pure_callback_parallel_solve(
            f, g, p_batch, x0_batch, n, m, lb, ub, cl, cu, options, workers,
        )
        return x_star, (p_batch, x_star, lam, mult_xL, mult_xU)

    def bwd_single(p, x_star, lam, mult_xL, mult_xU, v):
        active = (mult_xL > _ACTIVE_TOL) | (mult_xU > _ACTIVE_TOL)

        def lagrangian(x, p_):
            base = f(x, p_)
            if g is not None and m > 0:
                base = base + jnp.dot(lam, g(x, p_))
            return base

        H = jax.hessian(lagrangian, argnums=0)(x_star, p)
        grad_L_of_p = lambda p_: jax.grad(lagrangian, argnums=0)(x_star, p_)
        dgradL_dp = jax.jacrev(grad_L_of_p)(p)

        if g is not None and m > 0:
            J = jax.jacrev(g, argnums=0)(x_star, p)
            dg_dp = jax.jacrev(lambda p_: g(x_star, p_))(p)
            cl_arr = jnp.asarray(cl, dtype=H.dtype)
            cu_arr = jnp.asarray(cu, dtype=H.dtype)
            is_equality = cl_arr == cu_arr
            cons_active = is_equality | (jnp.abs(lam) > _ACTIVE_TOL)
            cons_inactive = ~cons_active
        else:
            J = jnp.zeros((0, n))
            dg_dp = jnp.zeros((0,) + jnp.shape(p))
            cons_inactive = jnp.zeros((0,), dtype=bool)

        active_mat = jnp.diag(active.astype(H.dtype))
        H_eff = jnp.where(
            active[:, None] | active[None, :], 0.0, H
        ) + active_mat
        J_eff = jnp.where(
            cons_inactive[:, None] | active[None, :], 0.0, J
        )
        v_eff = jnp.where(active, 0.0, v)

        if m > 0:
            cons_inactive_diag = jnp.diag(cons_inactive.astype(H.dtype))
            top = jnp.concatenate([H_eff, J_eff.T], axis=1)
            bot = jnp.concatenate([J_eff, cons_inactive_diag], axis=1)
            K = jnp.concatenate([top, bot], axis=0)
            rhs = jnp.concatenate([v_eff, jnp.zeros(m, dtype=H.dtype)])
            u = jnp.linalg.solve(K, rhs)
            u_x, u_lam = u[:n], u[n:]
        else:
            u_x = jnp.linalg.solve(H_eff, v_eff)
            u_lam = jnp.zeros(0)

        dL_dp = -jnp.tensordot(u_x, dgradL_dp, axes=1)
        if m > 0:
            dL_dp = dL_dp - jnp.tensordot(u_lam, dg_dp, axes=1)
        return dL_dp

    def bwd(residuals, cotangent_x_batch):
        p_batch, x_star_batch, lam_batch, mult_xL_batch, mult_xU_batch = residuals
        dL_dp_batch = jax.vmap(bwd_single)(
            p_batch, x_star_batch, lam_batch, mult_xL_batch, mult_xU_batch,
            cotangent_x_batch,
        )
        # x0_batch carries no gradient (matches `solve`).
        return dL_dp_batch, jnp.zeros_like(x_star_batch)

    solve_fn.defvjp(fwd, bwd)
    return solve_fn


def _pure_callback_parallel_solve(
    f, g, p_batch, x0_batch, n, m, lb, ub, cl, cu, options, workers,
):
    B = p_batch.shape[0]
    result_shapes = (
        jax.ShapeDtypeStruct((B, n), jnp.float64),
        jax.ShapeDtypeStruct((B, m), jnp.float64),
        jax.ShapeDtypeStruct((B, n), jnp.float64),
        jax.ShapeDtypeStruct((B, n), jnp.float64),
    )

    def host_call(p_h, x0_h):
        return _solve_batch_threadpool(
            f, g, np.asarray(p_h), np.asarray(x0_h),
            n, m, lb, ub, cl, cu, options, workers,
        )

    return jax.pure_callback(host_call, result_shapes, p_batch, x0_batch)


def vmap_solve_parallel(
    p_batch,
    *,
    f: Callable,
    g: Callable | None = None,
    x0,
    n: int,
    m: int = 0,
    lb=None,
    ub=None,
    cl=None,
    cu=None,
    options: dict | None = None,
    workers: int | None = None,
):
    """Parallel batched solve. Drop-in for :func:`vmap_solve`.

    Each of the ``B`` elements of ``p_batch`` is dispatched to a worker
    in a ``ThreadPoolExecutor`` of size ``workers`` (default:
    ``min(B, 8)``). Each worker owns an independent ``Problem`` so
    there's no shared state. The ``py.allow_threads`` block around
    ``optimize_tnlp`` in ``pounce-py`` releases the GIL across the
    IPM iteration, so threads actually run concurrently on the Rust
    side — the only cross-thread serialization is the Python
    callbacks for ``f`` / ``g``, which reacquire the GIL the usual
    way (typically a small fraction of total solve time for
    JAX-jitted callables).

    Differentiable w.r.t. ``p_batch`` via per-element implicit
    function theorem. The backward pass vectorizes naturally via
    ``jax.vmap`` because the KKT solve is pure JAX.

    ``x0`` may be a single ``(n,)`` vector (broadcast to all batch
    elements) or a ``(B, n)`` batch.
    """
    p_batch = jnp.asarray(p_batch)
    B = p_batch.shape[0]
    x0_arr = jnp.asarray(x0)
    if x0_arr.ndim == 1:
        x0_arr = jnp.broadcast_to(x0_arr, (B, n))
    fn = _make_vmap_solve_parallel_custom_vjp(
        f, g, n, m, lb, ub, cl, cu, options, workers,
    )
    return fn(p_batch, x0_arr)
