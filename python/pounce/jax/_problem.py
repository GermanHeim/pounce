"""Build-once, solve-many JAX problem (pounce#75).

The top-level :func:`pounce.jax.solve` / :func:`vmap_solve_parallel` /
:func:`solve_with_warm` rebuild a fresh ``_JaxProblem`` (re-JIT of
``jax.grad`` / ``jax.jacrev`` / ``jax.hessian`` plus the one-shot
random sparsity probe) and a fresh :class:`pounce.Problem` on every
call. For an iterative outer loop that solves the same-structure
problem many times — differentiable constrained layer in a training
loop, parametric sweep — that rebuild dominates wall-clock and makes
the JAX path 1–2 orders of magnitude slower than the underlying solver
(see pounce#75 for numbers).

:class:`JaxProblem` is the build-once handle: do the JIT and sparsity
probe in ``__init__``, then expose :meth:`solve`, :meth:`solve_with_warm`,
:meth:`vmap_solve`, :meth:`vmap_solve_parallel` as methods that
reuse the prebuilt state on every call.

Thread safety. ``vmap_solve_parallel`` dispatches solves across a
``ThreadPoolExecutor`` (pounce#74). The JIT-compiled JAX callables and
the sparsity pattern are immutable and thread-safe. The
:class:`pounce.Problem` instance and its bound ``problem_obj`` are
*not* — the obj closes over a mutable ``p`` that gets reset on each
solve. To avoid races, each worker thread gets its own
``(problem_obj, Problem)`` pair via a :class:`threading.local`
cache, so the per-thread build cost is paid at most once per worker
(typically ``min(B, 8)`` total) instead of ``B`` times per batch.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np

from ._build import _detect_pattern_2d, _detect_pattern_lower, _to_np
from .._pounce import Problem

_ACTIVE_TOL = 1e-6


class _ReusableJaxNlp:
    """Cyipopt-shaped problem object whose JAX callables are owned by
    a parent :class:`JaxProblem`. Closes over a mutable ``_p`` that the
    parent updates between solves so the same Rust :class:`Problem`
    instance can serve a sequence of solves at different ``p``.

    Not threadsafe on its own. Each :class:`JaxProblem` keeps one of
    these per worker thread via a ``threading.local``.
    """

    __slots__ = ("_jp", "_p")

    def __init__(self, jp: "JaxProblem"):
        self._jp = jp
        self._p = None  # set by JaxProblem before every solve

    def objective(self, x):
        return float(self._jp._f_jit(jnp.asarray(x), self._p))

    def gradient(self, x):
        return _to_np(self._jp._grad_f_jit(jnp.asarray(x), self._p))

    def constraints(self, x):
        return _to_np(self._jp._g_jit(jnp.asarray(x), self._p))

    def jacobianstructure(self):
        return (self._jp._jac_rows, self._jp._jac_cols)

    def jacobian(self, x):
        J = _to_np(self._jp._jac_g_jit(jnp.asarray(x), self._p))
        return J[self._jp._jac_rows, self._jp._jac_cols]

    def hessianstructure(self):
        return (self._jp._hess_rows, self._jp._hess_cols)

    def hessian(self, x, lam, obj_factor):
        if self._jp._m > 0:
            H = _to_np(
                self._jp._hess_lag_jit(
                    jnp.asarray(x), jnp.asarray(lam), obj_factor, self._p,
                )
            )
        else:
            H = _to_np(
                self._jp._hess_lag_jit(jnp.asarray(x), obj_factor, self._p)
            )
        return H[self._jp._hess_rows, self._jp._hess_cols]


def _bwd_single_kkt(
    f: Callable,
    g: Callable | None,
    n: int,
    m: int,
    cl,
    cu,
    p,
    x_star,
    lam,
    mult_xL,
    mult_xU,
    v,
):
    """Implicit-function-theorem VJP at a single ``(p, x*, λ*)``.

    Same logic as the bwd in :func:`pounce.jax.solve` / the per-element
    bwd in :func:`vmap_solve_parallel` — factored out so the prebuilt
    paths share one source of truth for the active-set handling
    (pounce#73 fix).
    """
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
    H_eff = jnp.where(active[:, None] | active[None, :], 0.0, H) + active_mat
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


class JaxProblem:
    """Reusable, differentiable parametric solve (pounce#75).

    Construct once with ``f(x, p)`` and ``g(x, p)``; solve many times
    at different ``p`` without re-running the JAX JIT compilation or
    the random sparsity probe.

    Parameters
    ----------
    f : callable
        Objective ``f(x, p) -> scalar``. Must be JAX-traceable.
    g : callable or None
        Constraints ``g(x, p) -> (m,)``. Required when ``m > 0``.
    n, m : int
        Variable and constraint counts.
    p_example : array-like
        Example parameter vector. Used for the one-shot sparsity probe
        (shape and dtype only — the values are discarded). Any later
        ``p`` passed to a solve method must have the same shape.
    lb, ub, cl, cu : array-like or None
        Variable and constraint bounds; same convention as
        :class:`pounce.Problem`.
    options : dict or None
        Pounce options applied once via ``add_option`` at build time.
        Options that need to vary per-solve (e.g. ``warm_start_init_point``
        flipped from ``"no"`` to ``"yes"``) are toggled internally by
        :meth:`solve_with_warm`; otherwise the same dict is in force
        for every method call.
    seed : int
        Seed for the random sparsity probe.

    Notes
    -----
    Threadsafe: :meth:`vmap_solve_parallel` dispatches across worker
    threads, and each thread keeps its own per-thread
    :class:`pounce.Problem` (via :class:`threading.local`) to avoid
    racing on the mutable ``p`` slot. The JAX callables and the
    sparsity pattern itself are immutable.
    """

    def __init__(
        self,
        *,
        f: Callable,
        g: Callable | None = None,
        n: int,
        m: int = 0,
        p_example,
        lb=None,
        ub=None,
        cl=None,
        cu=None,
        options: dict | None = None,
        seed: int = 0,
    ):
        if m > 0 and g is None:
            raise ValueError("g must be provided when m > 0")
        self._f = f
        self._g = g
        self._n = n
        self._m = m
        self._lb = lb
        self._ub = ub
        self._cl = cl
        self._cu = cu
        self._options = dict(options or {})

        # JIT-compiled derivatives over (x, p). These are stateless and
        # threadsafe — the JaxProblem is the canonical owner.
        self._f_jit = jax.jit(f)
        self._grad_f_jit = jax.jit(jax.grad(f, argnums=0))
        if g is not None and m > 0:
            self._g_jit = jax.jit(g)
            self._jac_g_jit = jax.jit(jax.jacrev(g, argnums=0))

            def lagrangian(x, lam, sigma, p):
                return sigma * f(x, p) + jnp.dot(lam, g(x, p))

            self._hess_lag_jit = jax.jit(jax.hessian(lagrangian, argnums=0))
        else:
            self._g_jit = None
            self._jac_g_jit = None

            def lagrangian_unc(x, sigma, p):
                return sigma * f(x, p)

            self._hess_lag_jit = jax.jit(jax.hessian(lagrangian_unc, argnums=0))

        # One-shot sparsity probe at random (x, p). The sparsity
        # *pattern* is assumed independent of p (true for smooth
        # pointwise compositions); see _JaxProblem for the same
        # assumption on the non-parametric path.
        p_arr = np.asarray(p_example, dtype=np.float64)
        self._p_shape = p_arr.shape
        self._p_dtype = jnp.float64
        rng = np.random.default_rng(seed)
        x_probe = jnp.asarray(rng.standard_normal(n))
        p_probe = jnp.asarray(rng.standard_normal(p_arr.shape))
        if m > 0:
            lam_probe = jnp.asarray(rng.standard_normal(m))
            jac_dense = _to_np(self._jac_g_jit(x_probe, p_probe))
            self._jac_rows, self._jac_cols = _detect_pattern_2d(jac_dense)
            hess_dense = _to_np(
                self._hess_lag_jit(x_probe, lam_probe, 1.0, p_probe)
            )
        else:
            self._jac_rows = np.zeros(0, dtype=np.int64)
            self._jac_cols = np.zeros(0, dtype=np.int64)
            hess_dense = _to_np(self._hess_lag_jit(x_probe, 1.0, p_probe))
        self._hess_rows, self._hess_cols = _detect_pattern_lower(hess_dense)

        # Per-thread (obj, Problem) cache. Built lazily on first solve
        # from the calling thread.
        self._tls = threading.local()

    # ----- internal: per-thread cached Problem -----

    def _build_problem(self) -> tuple[_ReusableJaxNlp, Problem]:
        obj = _ReusableJaxNlp(self)
        prob = Problem(
            n=self._n, m=self._m, problem_obj=obj,
            lb=self._lb, ub=self._ub, cl=self._cl, cu=self._cu,
        )
        for k, v in self._options.items():
            prob.add_option(k, v)
        return obj, prob

    def _thread_problem(self) -> tuple[_ReusableJaxNlp, Problem]:
        cached = getattr(self._tls, "pair", None)
        if cached is None:
            cached = self._build_problem()
            self._tls.pair = cached
        return cached

    def _thread_problem_warm(self) -> tuple[_ReusableJaxNlp, Problem]:
        cached = getattr(self._tls, "pair_warm", None)
        if cached is None:
            obj, prob = self._build_problem()
            # Warm-start option must be wired at build time — toggling
            # mid-life isn't reliable across the C boundary.
            prob.add_option("warm_start_init_point", "yes")
            cached = (obj, prob)
            self._tls.pair_warm = cached
        return cached

    # ----- host-side solves (called from pure_callback host_call) -----

    def _host_solve(self, p_np: np.ndarray, x0_np: np.ndarray):
        obj, prob = self._thread_problem()
        obj._p = jnp.asarray(p_np)
        return prob.solve(x0=np.asarray(x0_np, dtype=np.float64))

    def _host_solve_warm(
        self,
        p_np: np.ndarray,
        x0_np: np.ndarray,
        lam_np: np.ndarray,
        zL_np: np.ndarray,
        zU_np: np.ndarray,
    ):
        obj, prob = self._thread_problem_warm()
        obj._p = jnp.asarray(p_np)
        return prob.solve(
            x0=np.asarray(x0_np, dtype=np.float64),
            lagrange=np.asarray(lam_np, dtype=np.float64),
            zl=np.asarray(zL_np, dtype=np.float64),
            zu=np.asarray(zU_np, dtype=np.float64),
        )

    # ----- public: differentiable solve methods -----

    def solve(self, p, x0):
        """Differentiable forward solve at parameter ``p``. Returns ``x*``."""
        return self._solve_fn()(jnp.asarray(p), jnp.asarray(x0))

    def solve_with_warm(self, p, x0, warm_start: tuple | None = None):
        """Differentiable forward solve with dual warm-start (pounce#74).

        Returns ``(x*, (lam_out, zL_out, zU_out))``. Pass
        ``warm_start=None`` for an uninformed first call.
        """
        n, m = self._n, self._m
        if warm_start is None:
            lam_warm = jnp.zeros(m, dtype=jnp.float64)
            zL_warm = jnp.zeros(n, dtype=jnp.float64)
            zU_warm = jnp.zeros(n, dtype=jnp.float64)
        else:
            lam_warm, zL_warm, zU_warm = warm_start
            lam_warm = jnp.asarray(lam_warm, dtype=jnp.float64)
            zL_warm = jnp.asarray(zL_warm, dtype=jnp.float64)
            zU_warm = jnp.asarray(zU_warm, dtype=jnp.float64)
        fn = self._solve_with_warm_fn()
        x_star, lam_out, zL_out, zU_out = fn(
            jnp.asarray(p), jnp.asarray(x0), lam_warm, zL_warm, zU_warm,
        )
        return x_star, (lam_out, zL_out, zU_out)

    def vmap_solve(self, p_batch, x0):
        """Sequential batched solve over ``p_batch`` leading axis.

        Differentiable via per-element :meth:`solve`. ``x0`` may be a
        single ``(n,)`` vector (broadcast) or a ``(B, n)`` batch.
        """
        p_batch = jnp.asarray(p_batch)
        B = p_batch.shape[0]
        x0_arr = jnp.asarray(x0)
        if x0_arr.ndim == 1:
            x0_arr = jnp.broadcast_to(x0_arr, (B, self._n))

        def one(args):
            p_i, x0_i = args
            return self.solve(p_i, x0_i)

        return jax.lax.map(one, (p_batch, x0_arr))

    def vmap_solve_parallel(self, p_batch, x0, workers: int | None = None):
        """Parallel batched solve via :class:`ThreadPoolExecutor` (pounce#74).

        Each worker thread gets its own cached :class:`pounce.Problem`,
        so the build cost is paid once per worker (typically ``min(B, 8)``
        times total), not ``B`` times per batch.
        """
        p_batch = jnp.asarray(p_batch)
        B = p_batch.shape[0]
        x0_arr = jnp.asarray(x0)
        if x0_arr.ndim == 1:
            x0_arr = jnp.broadcast_to(x0_arr, (B, self._n))
        fn = self._vmap_solve_parallel_fn(workers)
        return fn(p_batch, x0_arr)

    # ----- custom_vjp factories -----

    def _solve_fn(self):
        f, g, n, m = self._f, self._g, self._n, self._m
        cl, cu = self._cl, self._cu
        jp = self

        @jax.custom_vjp
        def solve_fn(p, x0):
            x_star, _ = _pure_callback_solve(jp, p, x0)
            return x_star

        def fwd(p, x0):
            x_star, info = _pure_callback_solve(jp, p, x0)
            lam = jnp.asarray(info["mult_g"]) if m > 0 else jnp.zeros(0)
            mult_xL = jnp.asarray(info["mult_x_L"])
            mult_xU = jnp.asarray(info["mult_x_U"])
            return x_star, (p, x_star, lam, mult_xL, mult_xU)

        def bwd(residuals, v):
            p, x_star, lam, mult_xL, mult_xU = residuals
            dL_dp = _bwd_single_kkt(
                f, g, n, m, cl, cu, p, x_star, lam, mult_xL, mult_xU, v,
            )
            return dL_dp, jnp.zeros((n,), dtype=jnp.float64)

        solve_fn.defvjp(fwd, bwd)
        return solve_fn

    def _solve_with_warm_fn(self):
        f, g, n, m = self._f, self._g, self._n, self._m
        cl, cu = self._cl, self._cu
        jp = self

        @jax.custom_vjp
        def solve_fn(p, x0, lam_warm, zL_warm, zU_warm):
            return _pure_callback_warm_solve(
                jp, p, x0, lam_warm, zL_warm, zU_warm,
            )

        def fwd(p, x0, lam_warm, zL_warm, zU_warm):
            x_star, lam_out, zL_out, zU_out = _pure_callback_warm_solve(
                jp, p, x0, lam_warm, zL_warm, zU_warm,
            )
            return (
                (x_star, lam_out, zL_out, zU_out),
                (p, x_star, lam_out, zL_out, zU_out),
            )

        def bwd(residuals, cotangents):
            p, x_star, lam, mult_xL, mult_xU = residuals
            v = cotangents[0]
            dL_dp = _bwd_single_kkt(
                f, g, n, m, cl, cu, p, x_star, lam, mult_xL, mult_xU, v,
            )
            return (
                dL_dp,
                jnp.zeros((n,), dtype=jnp.float64),
                jnp.zeros((m,), dtype=jnp.float64),
                jnp.zeros((n,), dtype=jnp.float64),
                jnp.zeros((n,), dtype=jnp.float64),
            )

        solve_fn.defvjp(fwd, bwd)
        return solve_fn

    def _vmap_solve_parallel_fn(self, workers: int | None):
        f, g, n, m = self._f, self._g, self._n, self._m
        cl, cu = self._cl, self._cu
        jp = self

        @jax.custom_vjp
        def solve_fn(p_batch, x0_batch):
            x_star, *_ = _pure_callback_parallel_solve(
                jp, p_batch, x0_batch, workers,
            )
            return x_star

        def fwd(p_batch, x0_batch):
            x_star, lam, mult_xL, mult_xU = _pure_callback_parallel_solve(
                jp, p_batch, x0_batch, workers,
            )
            return x_star, (p_batch, x_star, lam, mult_xL, mult_xU)

        def bwd_single(p, x_star, lam, mult_xL, mult_xU, v):
            return _bwd_single_kkt(
                f, g, n, m, cl, cu, p, x_star, lam, mult_xL, mult_xU, v,
            )

        def bwd(residuals, cot_x_batch):
            p_batch, x_star_batch, lam_batch, mult_xL_batch, mult_xU_batch = residuals
            dL_dp_batch = jax.vmap(bwd_single)(
                p_batch, x_star_batch, lam_batch, mult_xL_batch, mult_xU_batch,
                cot_x_batch,
            )
            return dL_dp_batch, jnp.zeros_like(x_star_batch)

        solve_fn.defvjp(fwd, bwd)
        return solve_fn


# ----- pure_callback wrappers (module-level, closed over a JaxProblem) -----


def _pure_callback_solve(jp: JaxProblem, p, x0):
    n, m = jp._n, jp._m
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
        x_np, info = jp._host_solve(np.asarray(p_h), np.asarray(x0_h))
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


def _pure_callback_warm_solve(jp: JaxProblem, p, x0, lam_warm, zL_warm, zU_warm):
    n, m = jp._n, jp._m
    result_shapes = (
        jax.ShapeDtypeStruct((n,), jnp.float64),
        jax.ShapeDtypeStruct((m,), jnp.float64),
        jax.ShapeDtypeStruct((n,), jnp.float64),
        jax.ShapeDtypeStruct((n,), jnp.float64),
    )

    def host_call(p_h, x0_h, lam_h, zL_h, zU_h):
        x_np, info = jp._host_solve_warm(
            np.asarray(p_h), np.asarray(x0_h),
            np.asarray(lam_h), np.asarray(zL_h), np.asarray(zU_h),
        )
        return (
            np.asarray(x_np, dtype=np.float64),
            np.asarray(info["mult_g"], dtype=np.float64),
            np.asarray(info["mult_x_L"], dtype=np.float64),
            np.asarray(info["mult_x_U"], dtype=np.float64),
        )

    return jax.pure_callback(
        host_call, result_shapes, p, x0, lam_warm, zL_warm, zU_warm,
    )


def _pure_callback_parallel_solve(jp: JaxProblem, p_batch, x0_batch, workers):
    n, m = jp._n, jp._m
    B = p_batch.shape[0]
    result_shapes = (
        jax.ShapeDtypeStruct((B, n), jnp.float64),
        jax.ShapeDtypeStruct((B, m), jnp.float64),
        jax.ShapeDtypeStruct((B, n), jnp.float64),
        jax.ShapeDtypeStruct((B, n), jnp.float64),
    )

    def host_call(p_h, x0_h):
        p_np = np.asarray(p_h)
        x0_np = np.asarray(x0_h)
        n_workers = workers or min(B, 8)
        x_out = np.empty((B, n), dtype=np.float64)
        lam_out = np.empty((B, m), dtype=np.float64)
        zL_out = np.empty((B, n), dtype=np.float64)
        zU_out = np.empty((B, n), dtype=np.float64)

        def one(i):
            x_np, info = jp._host_solve(p_np[i], x0_np[i])
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

    return jax.pure_callback(host_call, result_shapes, p_batch, x0_batch)
