"""gh#1010: every JAX transform composes with a pounce.jax solve.

``pounce.jax.solve`` and ``JaxProblem.solve`` were ``custom_vjp``s whose
forward crossed a ``pure_callback`` with no batching rule, so ``jvp`` /
``jacfwd`` / ``hessian`` / ``vmap`` of anything calling them raised. They
are now ``custom_jvp``s with a plain-JAX (or ``custom_linear_solve``)
tangent rule and a sequentially-batched callback.

Reference values are central finite differences of the *solve*, never
another output of the sensitivity layer, so a self-consistently wrong
rule cannot pass. The model below reaches every branch the active-set
rule has: an equality row, a binding inequality (multiplier ~0.7), a
slack inequality, an active lower bound and free variables — and ``p``
enters ``f`` nonlinearly, so ``∂_p ∇_x L`` is nonzero and second order
needs the multiplier's tangent (``H`` of the Lagrangian depends on λ).
"""

import warnings

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp

import pounce.jax as pj

N, M = 4, 3
LB = np.array([-10.0, -10.0, -10.0, 0.0])
UB = np.full(N, 10.0)
CL = np.array([0.0, -np.inf, -np.inf])
CU = np.array([0.0, 10.0, 0.1])
X0 = jnp.full(N, 0.2)
P0 = jnp.array([0.9, 1.1])
OPTS = {"print_level": 0, "tol": 1e-10}
W = jnp.array([1.0, -2.0, 0.5, 3.0])
EPS = 1e-4


def f(x, p):
    t = jnp.array([p[0], p[1] ** 2, jnp.sin(p[0] * p[1]), -1.0])
    return jnp.sum((x - t) ** 2) + 0.3 * x[0] * x[1] * p[0]


def g(x, p):
    return jnp.array([
        x[0] + x[1] + x[2] + x[3] - p[0] * p[1],  # equality
        x[0] - x[1],                               # slack inequality
        x[1] * x[2] + 0.5 * x[0] ** 2,             # binding inequality
    ])


def _solve_pj(p):
    return pj.solve(p, f=f, g=g, x0=X0, n=N, m=M, lb=LB, ub=UB, cl=CL, cu=CU,
                    options=OPTS, on_failure="raise")


_JPS = {}


def _jp(factor_reuse):
    if factor_reuse not in _JPS:
        _JPS[factor_reuse] = pj.JaxProblem(
            f=f, g=g, n=N, m=M, p_example=P0, lb=LB, ub=UB, cl=CL, cu=CU,
            options=OPTS, factor_reuse=factor_reuse, on_failure="raise",
        )
    return _JPS[factor_reuse]


SURFACES = {
    "pj.solve": _solve_pj,
    "JaxProblem(factor_reuse=True)": lambda p: _jp(True).solve(p, X0),
    "JaxProblem(factor_reuse=False)": lambda p: _jp(False).solve(p, X0),
}


def _fd_jac(p):
    cols = [(np.asarray(_solve_pj(p + EPS * e)) - np.asarray(_solve_pj(p - EPS * e)))
            / (2 * EPS) for e in np.eye(2)]
    return np.stack(cols, axis=1)


def _fd_hess(p):
    s = lambda q: float(np.dot(np.asarray(_solve_pj(q)), W))
    E = np.eye(2) * EPS
    return np.array([[
        (s(p + a + b) - s(p + a - b) - s(p - a + b) + s(p - a - b)) / (4 * EPS ** 2)
        for b in E] for a in E])


@pytest.fixture(scope="module")
def reference():
    x = np.asarray(_solve_pj(P0))
    # The fixture must actually reach the branches it claims to.
    assert abs(x[3]) < 1e-6                                  # active bound
    gx = np.asarray(g(x, P0))
    assert abs(gx[2] - CU[2]) < 1e-7                         # binding row
    assert gx[1] < CU[1] - 1.0                               # slack row
    p1 = P0 + jnp.array([0.05, -0.03])
    return {"J": _fd_jac(P0), "H": _fd_hess(P0), "p1": p1, "J1": _fd_jac(p1)}


@pytest.fixture(autouse=True)
def _no_solver_warnings():
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        yield


@pytest.mark.parametrize("name", list(SURFACES))
def test_forward_and_reverse_jacobians_match_finite_differences(name, reference):
    fn = SURFACES[name]
    J_rev = np.asarray(jax.jacrev(fn)(P0))
    J_fwd = np.asarray(jax.jacfwd(fn)(P0))
    J_fwd_jit = np.asarray(jax.jit(jax.jacfwd(fn))(P0))
    np.testing.assert_allclose(J_rev, reference["J"], atol=1e-6)
    np.testing.assert_allclose(J_fwd, J_rev, atol=1e-12)
    np.testing.assert_allclose(J_fwd_jit, J_rev, atol=1e-12)
    dp = jnp.array([0.3, -1.7])
    _, tangent = jax.jvp(fn, (P0,), (dp,))
    np.testing.assert_allclose(np.asarray(tangent), J_rev @ np.asarray(dp), atol=1e-12)


@pytest.mark.parametrize("name", list(SURFACES))
def test_hessian_matches_finite_differences(name, reference):
    fn = SURFACES[name]
    loss = lambda p: jnp.dot(fn(p), W)
    H = np.asarray(jax.hessian(loss)(P0))
    np.testing.assert_allclose(H, reference["H"], atol=1e-4)
    # reverse-over-forward must agree with hessian's forward-over-reverse
    H_rf = np.asarray(jax.jacrev(jax.jacfwd(loss))(P0))
    np.testing.assert_allclose(H_rf, H, atol=1e-10)


@pytest.mark.parametrize("name", list(SURFACES))
def test_vmap_of_solve_and_of_its_jacobians(name, reference):
    fn = SURFACES[name]
    p_batch = jnp.stack([P0, reference["p1"]])
    X = np.asarray(jax.vmap(fn)(p_batch))
    np.testing.assert_allclose(X[0], np.asarray(_solve_pj(P0)), atol=1e-9)
    np.testing.assert_allclose(X[1], np.asarray(_solve_pj(reference["p1"])), atol=1e-9)
    # Two different points: on the factor-reuse path each element holds
    # its own factor, so a callback that read the first element's id for
    # the whole batch would return element 0's Jacobian twice.
    for transform in (jax.jacrev, jax.jacfwd):
        Jb = np.asarray(jax.vmap(transform(fn))(p_batch))
        np.testing.assert_allclose(Jb[0], reference["J"], atol=1e-6)
        np.testing.assert_allclose(Jb[1], reference["J1"], atol=1e-6)


def test_multiplier_tangent_matches_finite_differences(monkeypatch):
    """Second order hinges on λ̇ — check it directly, not only through H."""
    import pounce.jax._diff as D

    captured = {}
    orig = D._kkt_implicit_tangent

    def spy(*args):
        out = orig(*args)
        captured["dlam"] = out[1]
        return out

    monkeypatch.setattr(D, "_kkt_implicit_tangent", spy)
    dp = jnp.array([1.0, 0.0])
    jax.jvp(_solve_pj, (P0,), (dp,))
    monkeypatch.undo()

    def lam(p):
        _x, info = D._solve_once(f, g, p, X0, N, M, LB, UB, CL, CU, OPTS)
        return np.asarray(info["mult_g"])

    fd = (lam(P0 + EPS * dp) - lam(P0 - EPS * dp)) / (2 * EPS)
    np.testing.assert_allclose(np.asarray(captured["dlam"]), fd, atol=1e-5)
    assert abs(fd[2]) > 1e-2      # the binding row's multiplier really moves
    assert fd[1] == pytest.approx(0.0, abs=1e-8)   # the slack row's does not


def test_unconstrained_bound_only_problem_all_paths():
    """m = 0 exercises the empty constraint block through every rule."""
    def f0(x, p):
        return (jnp.sum((x - jnp.array([p[0] ** 2, -p[1], 3.0])) ** 2)
                + 0.1 * x[0] * x[1])

    lb = np.array([-5.0, 0.0, -5.0])
    ub = np.full(3, 5.0)
    x0 = jnp.full(3, 0.5)
    p0 = jnp.array([1.2, 0.7])
    fns = [lambda p: pj.solve(p, f=f0, x0=x0, n=3, lb=lb, ub=ub, options=OPTS)]
    for fr in (True, False):
        jp = pj.JaxProblem(f=f0, n=3, m=0, p_example=p0, lb=lb, ub=ub,
                           options=OPTS, factor_reuse=fr)
        fns.append(lambda p, jp=jp: jp.solve(p, x0))
    # x1 pinned at 0 (target -p1 < 0), so x0 = p0², x2 = 3.
    J_exact = np.array([[2 * p0[0], 0.0], [0.0, 0.0], [0.0, 0.0]])
    for fn in fns:
        np.testing.assert_allclose(np.asarray(jax.jacfwd(fn)(p0)), J_exact, atol=1e-7)
        np.testing.assert_allclose(np.asarray(jax.jacrev(fn)(p0)), J_exact, atol=1e-7)
        H = np.asarray(jax.hessian(lambda p, fn=fn: fn(p).sum())(p0))
        np.testing.assert_allclose(H, [[2.0, 0.0], [0.0, 0.0]], atol=1e-6)


def test_traced_bounds_keep_their_nan_contract_in_both_modes():
    """gh#740's contract, now carried by a linear tangent rule.

    A constant bound built with ``jnp`` under ``jit`` reaches the rule as
    a symbolic-zero tangent and must not poison anything; a ``p``-derived
    *active* bound must come back NaN in both modes. A ``p``-derived
    *slack* bound keeps its finite gradient in reverse mode (the NaN
    cotangent lands on a constant coordinate and is discarded).
    """
    def f0(x, p):
        return jnp.sum((x - jnp.array([p[0] ** 2, -p[1], 3.0])) ** 2)

    x0 = jnp.full(3, 0.5)
    e0 = jnp.array([1.0, 0.0])

    def make(lb_of_p):
        return lambda p: pj.solve(p, f=f0, x0=x0, n=3, lb=lb_of_p(p),
                                  ub=jnp.full(3, 5.0), options=OPTS).sum()

    const = make(lambda p: jnp.array([-5.0, 0.0, -5.0]))
    active = make(lambda p: jnp.array([-5.0, 0.0 * p[0], -5.0]))
    slack = make(lambda p: jnp.array([-5.0 + 0.0 * p[0], 0.0, -5.0]))
    p0 = jnp.array([1.2, 0.7])

    g_const = np.asarray(jax.jit(jax.grad(const))(p0))
    t_const = float(jax.jit(lambda p: jax.jvp(const, (p,), (e0,))[1])(p0))
    np.testing.assert_allclose(g_const, [2.4, 0.0], atol=1e-7)
    assert t_const == pytest.approx(2.4, abs=1e-7)

    assert np.isnan(np.asarray(jax.jit(jax.grad(active))(p0))[0])
    assert np.isnan(float(jax.jvp(active, (p0,), (e0,))[1]))

    np.testing.assert_allclose(np.asarray(jax.jit(jax.grad(slack))(p0)),
                               [2.4, 0.0], atol=1e-7)
