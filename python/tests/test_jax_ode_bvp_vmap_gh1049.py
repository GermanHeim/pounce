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


def test_vmap_grad_ode_over_initial_state():
    # y0 rides in the same callback operand as theta; distinct y0 lanes
    # are distinct primals too.
    def end(y0, k):
        return pj.odeint(lambda t, y, th: -th[0] * y * y, jnp.array([y0]), T,
                         jnp.array([k])).y[0, -1]

    y0s = jnp.array([0.5, 1.0, 2.0])
    batch = np.array(jax.vmap(jax.grad(end), in_axes=(0, None))(y0s, 0.7))
    sep = np.array([jax.grad(end)(y0, 0.7) for y0 in y0s])
    np.testing.assert_allclose(batch, sep, atol=1e-9, rtol=0)


def test_vmap_grad_bvp_with_unknown_parameters():
    # k = 1 unknown parameter: the IFT block carries a non-state row.
    # y'' = -lam*y on [0, pi], y(0)=0, y'(0)=th, y(pi)=0 -> lam = 1;
    # the solution amplitude is th, so d y'(pi) / d th = -1 exactly.
    # (y is a fixed concrete guess: a traced guess is the separate
    # np.asarray(z0) limitation the issue reports.)
    x = jnp.linspace(0.0, np.pi, 41)
    y_guess = jnp.stack([jnp.sin(x), jnp.cos(x)])

    def end(th):
        sol = pj.solve_bvp(
            lambda x, y, p, th: jnp.stack([y[1], -p[0] * y[0]]),
            lambda ya, yb, p, th: jnp.array([ya[0], yb[0], ya[1] - th[0]]),
            x, y_guess, p=jnp.array([1.0]), theta=jnp.array([th]),
        )
        return sol.y[1, -1] + sol.p[0]

    ths = jnp.array([0.5, 1.0, 2.0])
    batch = np.array(jax.vmap(jax.grad(end))(ths))
    np.testing.assert_allclose(batch, _separate(end, ths), atol=1e-9, rtol=0)
    np.testing.assert_allclose(batch, -1.0, atol=1e-6)


def test_vmap_grad_bvp_second_order():
    def end(k):
        sol = pj.solve_bvp(
            lambda x, y, th: -th[0] * y,
            lambda ya, yb, th: jnp.array([ya[0] - 1.0]),
            X, jnp.ones((1, X.shape[0])), theta=jnp.array([k]),
            method="ipm", second_order=True,
        )
        return sol.y[0, -1]

    ks = jnp.array([0.2, 0.7, 1.3])
    batch = np.array(jax.vmap(jax.grad(end))(ks))
    np.testing.assert_allclose(batch, _separate(end, ks), atol=1e-9, rtol=0)


# -- the same defect class in JaxProblem's factor-reuse back-solve ---------

def test_kkt_backsolve_groups_lanes_by_solver_id():
    """``_kkt_backsolve_pure_callback`` must not read lane 0's solver id.

    ``JaxProblem.solve_with_warm(factor_reuse=True)`` routes its backward
    through this callback. Its forward is not vmappable today, which is the
    only thing that kept the lane-0 read from returning wrong gradients;
    adding ``vmap_method="sequential"`` there reproduced a 2.87 error. Drive
    the callback directly with two distinct held factors.
    """
    from pounce.jax._problem import (
        _kkt_backsolve_pure_callback,
        _pure_callback_warm_solve,
    )

    n, m = 2, 1
    jp = pj.JaxProblem(
        f=lambda x, p: (x[0] - p[0]) ** 2 + p[1] * (x[1] - 1.0) ** 2,
        g=lambda x, p: jnp.array([x[0] + x[1]]),
        n=n, m=m, p_example=jnp.array([1.0, 1.0]),
        cl=np.array([-np.inf]), cu=np.array([1.0]),
        options={"print_level": 0, "tol": 1e-10}, factor_reuse=True,
    )
    z = jnp.zeros
    sids = []
    for p in (jnp.array([2.0, 0.5]), jnp.array([-1.0, 4.0])):
        sids.append(_pure_callback_warm_solve(jp, p, jnp.ones(n), z(m), z(n), z(n))[4])
    sids = jnp.stack(sids)
    assert sids[0] != sids[1]
    vs = jnp.array([[1.0, -2.0], [0.5, 3.0]])

    def one(sid, v):
        return _kkt_backsolve_pure_callback(jp, sid, v, n, m)

    ux_b, ug_b = jax.vmap(one)(sids, vs)
    for i in range(2):
        ux, ug = one(sids[i], vs[i])
        np.testing.assert_allclose(ux_b[i], ux, atol=1e-12, rtol=0)
        np.testing.assert_allclose(ug_b[i], ug, atol=1e-12, rtol=0)
    # jacrev-shaped batch (one id, several cotangents) and nested batching.
    ux_j, _ = jax.vmap(one, in_axes=(None, 0))(sids[1], vs)
    np.testing.assert_allclose(ux_j[0], one(sids[1], vs[0])[0], atol=1e-12, rtol=0)
    ux_n, _ = jax.vmap(jax.vmap(one, in_axes=(None, 0)), in_axes=(0, None))(sids, vs)
    np.testing.assert_allclose(ux_n[1, 0], one(sids[1], vs[0])[0], atol=1e-12, rtol=0)
