"""Differentiable convex-QP layer (pounce.jax.solve_qp / QpLayer).

Validates the OptNet implicit-differentiation backward against finite
differences for the linear/RHS parameters (c, b, h), and checks
jacrev / vmap / QpLayer compose.
"""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from pounce.jax import QpLayer, solve_qp  # noqa: E402


def _fd(fn, x, eps=1e-6):
    x = np.asarray(x, float)
    g = np.zeros_like(x)
    for i in range(len(x)):
        xp = x.copy()
        xp[i] += eps
        xm = x.copy()
        xm[i] -= eps
        g[i] = (float(fn(jnp.array(xp))) - float(fn(jnp.array(xm)))) / (2 * eps)
    return g


P = jnp.array([[2.0, 0.0], [0.0, 2.0]])


def test_grad_c_interior():
    # Interior inequalities: gradient flows only through c.
    G = jnp.array([[1.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
    h = jnp.array([10.0, 0.0, 0.0])
    target = jnp.array([0.3, 0.4])

    def loss(c):
        return jnp.sum((solve_qp(P=P, c=c, G=G, h=h) - target) ** 2)

    c0 = jnp.array([-0.5, -0.7])
    g = jax.grad(loss)(c0)
    np.testing.assert_allclose(np.asarray(g), _fd(loss, c0), atol=1e-4)


def test_grad_h_active_inequality():
    # Active inequality x0+x1 ≤ h: gradient flows through h.
    G = jnp.array([[1.0, 1.0]])
    c0 = jnp.array([-4.0, -4.0])  # pulls past the constraint → active

    def loss(h):
        return jnp.sum(solve_qp(P=P, c=c0, G=G, h=h) ** 2)

    h0 = jnp.array([1.0])
    g = jax.grad(loss)(h0)
    np.testing.assert_allclose(np.asarray(g), _fd(loss, h0), atol=1e-4)


def test_grad_c_and_b_equality():
    A = jnp.array([[1.0, 1.0]])

    def loss_c(c):
        return jnp.sum(solve_qp(P=P, c=c, A=A, b=jnp.array([2.0])) ** 2)

    def loss_b(b):
        return jnp.sum(solve_qp(P=P, c=jnp.array([-1.0, -3.0]), A=A, b=b) ** 2)

    c0 = jnp.array([-1.0, -3.0])
    b0 = jnp.array([2.0])
    np.testing.assert_allclose(
        np.asarray(jax.grad(loss_c)(c0)), _fd(loss_c, c0), atol=1e-4
    )
    np.testing.assert_allclose(
        np.asarray(jax.grad(loss_b)(b0)), _fd(loss_b, b0), atol=1e-4
    )


def test_jacrev_of_solution():
    # Jacobian of x*(c) w.r.t. c via jacrev should be well-formed.
    G = jnp.array([[1.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
    h = jnp.array([10.0, 0.0, 0.0])
    c0 = jnp.array([-0.5, -0.7])
    J = jax.jacrev(lambda c: solve_qp(P=P, c=c, G=G, h=h))(c0)
    assert J.shape == (2, 2)
    # For an interior solution of ½·2‖x‖²+cᵀx, x* = −c/2, so dx/dc = −½I.
    np.testing.assert_allclose(np.asarray(J), -0.5 * np.eye(2), atol=1e-5)


def test_qp_layer_and_vmap():
    # QpLayer captures fixed structure; vmap over a batch of objectives.
    G = jnp.array([[1.0, 1.0]])
    layer = QpLayer(P=P, G=G)
    cs = jnp.array([[-1.0, -1.0], [-4.0, -4.0], [0.5, 0.5]])
    hs = jnp.array([[1.0], [1.0], [1.0]])
    xs = jax.vmap(lambda c, h: layer(c, h=h))(cs, hs)
    assert xs.shape == (3, 2)
    # Each row matches a direct solve.
    for i in range(3):
        xi = solve_qp(P=P, c=cs[i], G=G, h=hs[i])
        np.testing.assert_allclose(np.asarray(xs[i]), np.asarray(xi), atol=1e-5)
