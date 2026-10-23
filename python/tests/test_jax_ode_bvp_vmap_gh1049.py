"""gh#1049: vmap of a pounce.jax ODE/BVP gradient over distinct parameters.

The IFT back-solve callback runs under ``vmap_method="broadcast_all"`` so
``jax.jacobian`` at one primal factors the collocation Jacobian once for
every cotangent. Its host body collapsed the batch to lane 0, which is only
right when every lane shares the primal: under an outer ``jax.vmap`` over
the parameter, every lane was back-solved against the first problem's
Jacobian. References are per-lane unbatched gradients (and the analytic
derivative), so a self-consistently wrong batched rule cannot pass.
"""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

import pounce.jax as pj

T = jnp.linspace(0.0, 2.0, 21)
X = jnp.linspace(0.0, 1.0, 11)


def _ode_end(k):
    return pj.odeint(
        lambda t, y, th: -th[0] * y, jnp.array([1.0]), T, jnp.array([k])
    ).y[0, -1]


def _bvp_end(k):
    sol = pj.solve_bvp(
        lambda x, y, th: -th[0] * y,
        lambda ya, yb, th: jnp.array([ya[0] - 1.0]),
        X, jnp.ones((1, X.shape[0])), theta=jnp.array([k]),
    )
    return sol.y[0, -1]


def _separate(f, ks):
    return np.array([jax.grad(f)(k) for k in np.ravel(ks)]).reshape(np.shape(ks))


@pytest.mark.parametrize("f", [_ode_end, _bvp_end], ids=["ode", "bvp"])
@pytest.mark.parametrize("ks", [[0.2, 0.7, 1.3], [1.3, 0.7, 0.2]],
                         ids=["forward", "reversed"])
def test_vmap_grad_matches_separate_solves(f, ks):
    ks = jnp.array(ks)
    batch = np.array(jax.vmap(jax.grad(f))(ks))
    np.testing.assert_allclose(batch, _separate(f, ks), atol=1e-9, rtol=0)


def test_vmap_grad_ode_matches_analytic():
    ks = jnp.array([0.2, 0.7, 1.3])
    batch = np.array(jax.vmap(jax.grad(_ode_end))(ks))
    # Collocation discretization error only, not lane-0 contamination.
    np.testing.assert_allclose(batch, -2 * np.exp(-2 * np.asarray(ks)), rtol=1e-5)


def test_jit_vmap_grad_ode():
    ks = jnp.array([0.2, 0.7, 1.3])
    batch = np.array(jax.jit(jax.vmap(jax.grad(_ode_end)))(ks))
    np.testing.assert_allclose(batch, _separate(_ode_end, ks), atol=1e-9, rtol=0)


def test_nested_vmap_grad_ode():
    ks = jnp.array([[0.2, 0.7], [1.3, 0.4]])
    batch = np.array(jax.vmap(jax.vmap(jax.grad(_ode_end)))(ks))
    np.testing.assert_allclose(batch, _separate(_ode_end, ks), atol=1e-9, rtol=0)


def test_repeated_primal_lanes():
    ks = jnp.array([0.7, 0.7, 0.7])
    batch = np.array(jax.vmap(jax.grad(_ode_end))(ks))
    np.testing.assert_allclose(batch, _separate(_ode_end, ks), atol=1e-9, rtol=0)


def test_jacrev_at_fixed_primal_still_batches_cotangents():
    def traj(k):
        return pj.odeint(
            lambda t, y, th: -th[0] * y, jnp.array([1.0, 2.0]), T, jnp.array([k])
        ).y[:, -1]

    J = np.array(jax.jacrev(traj)(0.7))
    rows = np.array([jax.grad(lambda k, i=i: traj(k)[i])(0.7) for i in range(2)])
    np.testing.assert_allclose(J, rows, atol=1e-12, rtol=0)
