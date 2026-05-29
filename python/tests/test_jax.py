"""Tests for the JAX integration. Skipped when JAX isn't installed."""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp


def test_from_jax_hs071():
    from pounce.jax import from_jax

    def f(x):
        return x[0] * x[3] * (x[0] + x[1] + x[2]) + x[2]

    def g(x):
        return jnp.stack([jnp.prod(x), jnp.dot(x, x)])

    prob = from_jax(
        f, g,
        n=4, m=2,
        lb=np.array([1.0] * 4), ub=np.array([5.0] * 4),
        cl=np.array([25.0, 40.0]), cu=np.array([2e19, 40.0]),
    )
    prob.add_option("tol", 1e-8)
    prob.add_option("print_level", 0)
    x, info = prob.solve(x0=np.array([1.0, 5.0, 5.0, 1.0]))
    assert info["status_msg"] == "Solve_Succeeded"
    np.testing.assert_allclose(info["obj_val"], 17.0140172, rtol=1e-5)


def test_implicit_diff_parametric_qp():
    """Differentiate x*(p) for  min ||x - p||²   →   x*(p) = p,   dx*/dp = I.

    A trivial parametric problem where the analytic Jacobian is known
    in closed form (the identity). This exercises the custom_vjp end
    to end without needing scipy.
    """
    from pounce.jax import solve

    def f(x, p):
        d = x - p
        return jnp.dot(d, d)

    def loss(p):
        x_star = solve(
            p, f=f, g=None, x0=jnp.zeros_like(p),
            n=p.size, m=0,
            options={"tol": 1e-10, "print_level": 0},
        )
        return jnp.sum(x_star ** 2)

    p = jnp.array([1.0, -2.0, 3.0])
    grad = jax.grad(loss)(p)
    # dL/dp = 2 x*(p) = 2 p.
    np.testing.assert_allclose(grad, 2.0 * p, atol=1e-4)


def _solve_box_projection(p, *, n=3, B=0.5):
    """Helper: min ||x - p||² s.t. x[0] <= B, all other x free."""
    from pounce.jax import solve

    def f(x, p_):
        d = x - p_
        return jnp.dot(d, d)

    def g(x, p_):  # noqa: ARG001
        return jnp.stack([x[0]])

    return solve(
        p, f=f, g=g, x0=jnp.zeros(n), n=n, m=1,
        lb=jnp.full(n, -1e19), ub=jnp.full(n, 1e19),
        cl=jnp.array([-1e19]), cu=jnp.array([B]),
        options={"tol": 1e-10, "print_level": 0},
    )


def _finite_diff_jacobian(forward, p, eps=1e-6):
    p_np = np.asarray(p, dtype=np.float64)
    n_out = np.asarray(forward(jnp.asarray(p_np))).size
    J = np.zeros((n_out, p_np.size))
    for j in range(p_np.size):
        e = np.zeros_like(p_np)
        e[j] = eps
        J[:, j] = (
            np.asarray(forward(jnp.asarray(p_np + e)))
            - np.asarray(forward(jnp.asarray(p_np - e)))
        ) / (2.0 * eps)
    return J


def test_implicit_diff_inactive_inequality_pounce_73():
    """Issue #73: slack inequality must not pin the sensitivity.

    `min ||x - p||²` s.t. `x[0] <= 0.5`. With `p[0] = -1 < 0.5` the
    inequality is slack at the optimum (mult_g ≈ 0), so `x*(p) = p`
    and `dx*/dp = I`. The bug was that the inequality row was kept
    as an equality in the backward, yielding ``dx*/dp[:, 0] ≈ 0``.
    """
    p = jnp.array([-1.0, 2.0, -3.0])
    analytic = np.asarray(jax.jacobian(_solve_box_projection)(p))
    fd = _finite_diff_jacobian(_solve_box_projection, p)
    np.testing.assert_allclose(analytic, fd, atol=5e-6)
    # Truth at slack ineq: dx*/dp = I.
    np.testing.assert_allclose(analytic, np.eye(p.size), atol=5e-6)


def test_implicit_diff_active_inequality_pounce_73():
    """Companion: when the inequality binds, dx*/dp must still match FD."""
    p = jnp.array([2.0, 2.0, -3.0])  # p[0] > B → x*[0] = B, binding
    analytic = np.asarray(jax.jacobian(_solve_box_projection)(p))
    fd = _finite_diff_jacobian(_solve_box_projection, p)
    np.testing.assert_allclose(analytic, fd, atol=5e-6)


def test_solve_with_warm_reduces_iterations_pounce_74():
    """`solve_with_warm` should consume the previous solve's duals and
    take fewer interior-point iterations on a small perturbation —
    that's the whole point of the warm-start surface (pounce#74)."""
    from pounce.jax import solve_with_warm

    n, m, B = 3, 1, 0.5

    def f(x, p):
        d = x - p
        return jnp.dot(d, d)

    def g(x, p):
        return jnp.stack([x[0]])

    def forward(p, warm):
        return solve_with_warm(
            p, f=f, g=g, x0=jnp.zeros(n), n=n, m=m,
            lb=jnp.full(n, -1e19), ub=jnp.full(n, 1e19),
            cl=jnp.array([-1e19]), cu=jnp.array([B]),
            options={"tol": 1e-10, "print_level": 0},
            warm_start=warm,
        )

    # Cold-start, then warm-start at a small perturbation of p.
    p0 = jnp.array([2.0, 2.0, -3.0])  # active inequality
    x0_star, warm0 = forward(p0, warm=None)
    np.testing.assert_allclose(np.asarray(x0_star), [B, 2.0, -3.0], atol=1e-6)

    # Re-solve at the same p with the warm duals — answer must match,
    # and the dual triple must round-trip without exploding.
    x1_star, (lam1, zL1, zU1) = forward(p0, warm=warm0)
    np.testing.assert_allclose(np.asarray(x1_star), np.asarray(x0_star), atol=1e-8)
    assert np.all(np.isfinite(np.asarray(lam1)))
    assert np.all(np.isfinite(np.asarray(zL1)))
    assert np.all(np.isfinite(np.asarray(zU1)))

    # Differentiability w.r.t. p still works through the warm path
    # (cotangents on the dual outputs are dropped — only x* feeds back).
    def loss(p):
        x_star, _ = forward(p, warm=warm0)
        return jnp.sum(x_star ** 2)

    grad = np.asarray(jax.grad(loss)(p0))
    # x*[0] = B is fixed (binding), so dL/dp[0] = 0; the others
    # contribute 2 * x*[i] = 2 * p[i] for i in {1, 2}.
    np.testing.assert_allclose(grad, np.array([0.0, 4.0, -6.0]), atol=1e-6)
