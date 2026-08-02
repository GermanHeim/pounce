"""information(): the un-inverted sibling of covariance() (covariance
roadmap item 2). The Lagrangian form is built by tangent recovery
(T = Zx inv(M), R = T'HT with the exact Hessian), full precision at
any barrier parameter; pinned parameters return S rather than zeros;
binding rows project; Gauss-Newton slices last."""
import warnings

import numpy as np
import pytest
import pyomo.environ as pyo

import pyomo_pounce  # noqa: F401
from pyomo_pounce import (
    covariance,
    declare_fitted,
    declare_residual,
    information,
)

N_LIN = 25
SIGMA_LIN = 0.3


def linear_data():
    rng = np.random.default_rng(42)
    x = np.linspace(0.0, 4.0, N_LIN)
    y = 1.5 - 0.7 * x + SIGMA_LIN * rng.standard_normal(N_LIN)
    X = np.column_stack([np.ones(N_LIN), x])
    return x, y, X


def linear_model(x, y):
    m = pyo.ConcreteModel()
    m.I = pyo.RangeSet(0, len(x) - 1)
    m.a = pyo.Var(initialize=0.0)
    m.b = pyo.Var(initialize=0.0)
    m.r = pyo.Var(m.I, initialize=0.0)
    m.res = pyo.Constraint(
        m.I, rule=lambda mm, i: mm.r[i] == float(y[i]) - mm.a
        - mm.b * float(x[i]))
    m.obj = pyo.Objective(expr=sum(m.r[i] ** 2 for i in m.I))
    declare_fitted(m.a)
    declare_fitted(m.b)
    declare_residual(m.r)
    return m


def test_information_is_the_reduced_hessian():
    # linear model: the observed information is exactly 2 X'X, in the
    # model's own units, no sigma^2 anywhere
    x, y, X = linear_data()
    m = linear_model(x, y)
    pyo.SolverFactory("pounce").solve(m)
    info = information(m)
    np.testing.assert_allclose(info.matrix, 2.0 * X.T @ X, rtol=1e-9)
    assert info[m.a] == pytest.approx(2.0 * N_LIN, rel=1e-9)
    ev, vecs = info.eigen()
    np.testing.assert_allclose(ev, np.linalg.eigvalsh(2.0 * X.T @ X),
                               rtol=1e-9)
    assert ev[0] < ev[1]


def test_information_inverts_covariance():
    # the roadmap's validation identity: on a well-conditioned free
    # block, covariance == 2 sigma^2 inv(information)
    x, y, X = linear_data()
    m = linear_model(x, y)
    pyo.SolverFactory("pounce").solve(m)
    info = information(m)
    cov = covariance(m, sigma_sq=SIGMA_LIN**2)
    np.testing.assert_allclose(
        cov.matrix, 2.0 * SIGMA_LIN**2 * np.linalg.inv(info.matrix),
        rtol=1e-8)


def test_gauss_newton_matches_lagrangian_on_linear():
    x, y, X = linear_data()
    m = linear_model(x, y)
    pyo.SolverFactory("pounce").solve(m)
    info_l = information(m)
    info_gn = information(m, hessian="gauss-newton")
    np.testing.assert_allclose(info_gn.matrix, info_l.matrix, rtol=1e-8)


def test_pinned_parameter_returns_s_not_zeros():
    # intercept pinned by a binding bound: its information entry is S,
    # the Schur reduction onto the pinned set, exact to the analytic
    # value (the tangent route loses nothing to the barrier weight);
    # cross blocks are zero; the free block is the plain reduced
    # Hessian entry
    x, y, X = linear_data()
    beta = np.linalg.solve(X.T @ X, X.T @ y)
    m = linear_model(x, y)
    m.a.setlb(float(beta[0]) + 0.4)      # binds, strongly active
    pyo.SolverFactory("pounce").solve(m)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        info = information(m)
    assert any("strongly active" in str(x_.message) for x_ in w)
    R = 2.0 * X.T @ X
    S_true = R[0, 0] - R[0, 1] ** 2 / R[1, 1]
    assert info[m.a] == pytest.approx(S_true, rel=1e-9)
    assert info[m.b] == pytest.approx(R[1, 1], rel=1e-9)
    assert info[m.a, m.b] == 0.0
    # Gauss-Newton forms J over ALL fitted parameters and slices last;
    # this is the test that the pinned rows actually exist to build S
    # from (a slice-first implementation loses them)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gn = information(m, hessian="gauss-newton")
    assert gn[m.a] == pytest.approx(S_true, rel=1e-9)
    assert gn[m.b] == pytest.approx(R[1, 1], rel=1e-9)
    assert gn[m.a, m.b] == 0.0


def test_binding_row_projects_information():
    # a + b <= cap binding: the free block is projected on both sides,
    # the pseudo-inverse relation to the projected covariance holds
    x, y, X = linear_data()
    beta = np.linalg.solve(X.T @ X, X.T @ y)
    cap = float(beta[0] + beta[1]) - 0.5
    m = linear_model(x, y)
    m.capcon = pyo.Constraint(expr=m.a + m.b <= cap)
    pyo.SolverFactory("pounce").solve(m)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        info = information(m)
        cov = covariance(m, sigma_sq=SIGMA_LIN**2)
    assert any("pins the fitted combination" in str(x_.message) for x_ in w)
    # info = 2 sigma^2 * pinv(cov) on the projected free block
    np.testing.assert_allclose(
        info.matrix, 2.0 * SIGMA_LIN**2 * np.linalg.pinv(cov.matrix),
        rtol=1e-6, atol=1e-9)
    u = np.array([1.0, 1.0])
    assert abs(u @ np.linalg.pinv(info.matrix) @ u) < 1e-9


def test_information_exact_under_objective_scaling():
    # data two orders larger and residuals seeded large at the start
    # point, so gradient-based scaling engages (df != 1, asserted):
    # information must still be 2 X'X in the model's own units, both
    # forms. This is the df axis of hessian_vec's natural-units
    # contract; every other fixture in this file runs at df = 1
    # (residuals initialize 0, so the starting gradient never trips
    # nlp_scaling_max_gradient)
    scale = 400.0
    rng = np.random.default_rng(7)
    x = np.linspace(0.0, 4.0, N_LIN)
    y = scale * (1.5 - 0.7 * x) \
        + (scale * SIGMA_LIN) * rng.standard_normal(N_LIN)
    X = np.column_stack([np.ones(N_LIN), x])
    m = linear_model(x, y)
    for i in m.I:
        m.r[i].set_value(400.0)
    pyo.SolverFactory("pounce").solve(m)
    session = m.__dict__["_pounce_sens"].session
    assert abs(float(session.solver.nlp_scaling["obj"]) - 1.0) > 1e-6
    R = 2.0 * X.T @ X
    np.testing.assert_allclose(information(m).matrix, R, rtol=1e-9)
    np.testing.assert_allclose(
        information(m, hessian="gauss-newton").matrix, R, rtol=1e-9)


def test_information_error_paths():
    x, y, _ = linear_data()
    m = linear_model(x, y)
    pyo.SolverFactory("pounce").solve(m)
    with pytest.raises(ValueError, match="hessian"):
        information(m, hessian="expected")
    m2 = pyo.ConcreteModel()
    m2.x = pyo.Var(initialize=1.0)
    m2.obj = pyo.Objective(expr=(m2.x - 1) ** 2)
    with pytest.raises(RuntimeError, match="no sensitivity session"):
        information(m2)
    # gauss-newton without declared residuals
    m3 = pyo.ConcreteModel()
    m3.k = pyo.Var(initialize=1.0)
    m3.obj = pyo.Objective(expr=(m3.k - 2.0) ** 2)
    declare_fitted(m3.k)
    pyo.SolverFactory("pounce").solve(m3)
    with pytest.raises(ValueError, match="residual"):
        information(m3, hessian="gauss-newton")
    # a session with residuals declared but nothing fitted (with no
    # declarations at all there is no session and the error above
    # fires first instead)
    m4 = pyo.ConcreteModel()
    m4.z = pyo.Var(initialize=0.0)
    m4.r = pyo.Var(initialize=0.0)
    m4.res = pyo.Constraint(expr=m4.r == 3.0 - m4.z)
    m4.obj = pyo.Objective(expr=m4.r ** 2)
    declare_residual(m4.r)
    pyo.SolverFactory("pounce").solve(m4)
    with pytest.raises(RuntimeError, match="no fitted parameters"):
        information(m4)


def test_indefinite_detector():
    """At a genuine minimum the reduced Hessian is PSD, so the branch
    cannot be reached by a healthy fixture; the detector is pinned
    directly instead."""
    from pyomo_pounce.sens import _indefinite
    assert not _indefinite(np.array([[2.0, 0.0], [0.0, 3.0]]))
    assert _indefinite(np.array([[2.0, 0.0], [0.0, -0.5]]))
    assert not _indefinite(np.array([[1.0, 0.0], [0.0, -1e-14]]))
    assert not _indefinite(np.zeros((0, 0)))


def test_hessian_vec_is_natural_units():
    """The new Hessian-vector product on the session: for the linear
    residual model H is 2I on the residual block and 0 on the fitted
    block, in the model's own units regardless of NLP scaling."""
    x, y, X = linear_data()
    m = linear_model(x, y)
    pyo.SolverFactory("pounce").solve(m)
    session = m.__dict__["_pounce_sens"].session
    n = int(session.nl.n)
    v = np.zeros(n)
    r0 = session.res_rows[None][0]        # a residual's primal row
    a_row = session.fit_rows[list(session.fit_rows)[0]]
    v[r0] = 1.0
    v[a_row] = 1.0
    hv = np.asarray(session.solver.hessian_vec(v))
    assert hv[r0] == pytest.approx(2.0, rel=1e-12)
    assert hv[a_row] == pytest.approx(0.0, abs=1e-12)
