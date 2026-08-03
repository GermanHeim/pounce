"""wrt= block selection (covariance roadmap item 3): both accessors
reduce onto any block of the solve's variables off the held factor,
post-solve. The declared fitted block is the default, so the no-wrt
behavior is untouched; each call re-reduces onto its own argument; a
rank-deficient block (more coordinates than the fit has degrees of
freedom, the prediction-band case) gets its marginal covariance and a
refusal from information(); strongly active variables outside the
block come back on the result as conditioned_on."""
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

N = 25
SIGMA = 0.3


def linear_data():
    rng = np.random.default_rng(42)
    x = np.linspace(0.0, 4.0, N)
    y = 1.5 - 0.7 * x + SIGMA * rng.standard_normal(N)
    X = np.column_stack([np.ones(N), x])
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


def solved():
    x, y, X = linear_data()
    m = linear_model(x, y)
    pyo.SolverFactory("pounce").solve(m)
    return m, X


def test_wrt_default_is_the_fitted_block():
    # wrt=[the fitted block, in order] must be EXACTLY the no-wrt
    # answer, both accessors: same matrix, same keys
    m, X = solved()
    cov0 = covariance(m, sigma_sq=SIGMA**2)
    cov1 = covariance(m, sigma_sq=SIGMA**2, wrt=[m.a, m.b])
    np.testing.assert_array_equal(cov0.matrix, cov1.matrix)
    info0 = information(m)
    info1 = information(m, wrt=[m.a, m.b])
    np.testing.assert_array_equal(info0.matrix, info1.matrix)
    assert cov1[m.a, m.b] == cov0[m.a, m.b]
    assert cov0.conditioned_on == () and cov1.conditioned_on == ()


def test_wrt_subblock_is_the_marginal():
    # wrt=[m.a] is the marginal over b: element 00 of the full
    # covariance, and its information is the inverse of that marginal
    # (NOT the conditional element R_aa)
    m, X = solved()
    C = SIGMA**2 * np.linalg.inv(X.T @ X)
    cov_a = covariance(m, sigma_sq=SIGMA**2, wrt=[m.a])
    assert cov_a.matrix.shape == (1, 1)
    assert cov_a[m.a] == pytest.approx(C[0, 0], rel=1e-9)
    info_a = information(m, wrt=[m.a])
    assert info_a[m.a] == pytest.approx(
        2.0 / np.linalg.inv(X.T @ X)[0, 0], rel=1e-9)
    # the sibling identity holds per block
    assert cov_a[m.a] * info_a[m.a] == pytest.approx(
        2.0 * SIGMA**2, rel=1e-9)
    # and gauss-newton profiles to the same marginal on a linear model
    gn_a = covariance(m, sigma_sq=SIGMA**2, hessian="gauss-newton",
                      wrt=[m.a])
    assert gn_a[m.a] == pytest.approx(C[0, 0], rel=1e-9)


def test_wrt_rank_deficient_block_is_the_prediction_band():
    # the residual block has 25 coordinates against 2 degrees of
    # freedom: its marginal covariance is the hat-matrix prediction
    # band sigma^2 X (X'X)^-1 X', membership handling is bypassed,
    # information() refuses toward covariance()
    m, X = solved()
    H = X @ np.linalg.solve(X.T @ X, X.T)
    cov_r = covariance(m, sigma_sq=SIGMA**2, wrt=m.r)
    np.testing.assert_allclose(cov_r.matrix, SIGMA**2 * H,
                               rtol=1e-8, atol=1e-12)
    assert cov_r[m.r[0]] == pytest.approx(SIGMA**2 * H[0, 0], rel=1e-8)
    with pytest.raises(RuntimeError, match="rank-deficient"):
        information(m, wrt=m.r)
    with pytest.raises(RuntimeError, match="rank-deficient"):
        covariance(m, sigma_sq=SIGMA**2, hessian="gauss-newton", wrt=m.r)


def test_wrt_conditioned_on_reports_the_outside_active_set():
    # a pinned by its bound, block = [b]: the block's number is the
    # value conditional on that bound (sigma^2 / (X'X)_11, the
    # fixed-intercept variance), and a comes back on conditioned_on.
    # The default block CONTAINS a, so its conditioned_on stays empty:
    # inside-block activity is membership, not conditioning.
    x, y, X = linear_data()
    beta = np.linalg.solve(X.T @ X, X.T @ y)
    m = linear_model(x, y)
    m.a.setlb(float(beta[0]) + 0.4)      # binds, strongly active
    pyo.SolverFactory("pounce").solve(m)
    cov_b = covariance(m, sigma_sq=SIGMA**2, wrt=[m.b])
    assert cov_b.conditioned_on == (m.a,)
    assert cov_b[m.b] == pytest.approx(
        SIGMA**2 / (X.T @ X)[1, 1], rel=1e-6)
    info_b = information(m, wrt=[m.b])
    assert info_b.conditioned_on == (m.a,)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cov_full = covariance(m, sigma_sq=SIGMA**2)
    assert cov_full.conditioned_on == ()


def test_wrt_accepted_forms():
    # a whole IndexedVar, a slice, a (Var, iterable) pair, and a mixed
    # list all normalize to the same coordinates
    m, X = solved()
    whole = covariance(m, sigma_sq=SIGMA**2, wrt=m.r)
    sliced = covariance(m, sigma_sq=SIGMA**2, wrt=m.r[:])
    paired = covariance(m, sigma_sq=SIGMA**2, wrt=(m.r, range(N)))
    np.testing.assert_array_equal(whole.matrix, sliced.matrix)
    np.testing.assert_array_equal(whole.matrix, paired.matrix)
    mixed = covariance(m, sigma_sq=SIGMA**2, wrt=[m.a, m.r[0]])
    assert mixed.matrix.shape == (2, 2)
    assert mixed[m.a] == pytest.approx(
        SIGMA**2 * np.linalg.inv(X.T @ X)[0, 0], rel=1e-9)


def test_wrt_derived_sigma_uses_the_fits_degrees_of_freedom():
    # sigma estimated from the declared residuals divides by n minus
    # the FITTED count (2), not the block size (1): the sub-block
    # marginal must equal the corresponding element of the default
    # answer exactly, both built from the same derived sigma
    m, X = solved()
    cov_full = covariance(m)
    cov_a = covariance(m, wrt=[m.a])
    assert cov_a[m.a] == cov_full[m.a]
    assert cov_a.sigma_sq == cov_full.sigma_sq


def test_wrt_error_paths():
    m, X = solved()
    with pytest.raises(ValueError, match="twice"):
        covariance(m, sigma_sq=SIGMA**2, wrt=[m.a, m.a])
    with pytest.raises(TypeError, match="not names"):
        covariance(m, sigma_sq=SIGMA**2, wrt="a")
    with pytest.raises(ValueError, match="empty block"):
        covariance(m, sigma_sq=SIGMA**2, wrt=[])
    m2 = pyo.ConcreteModel()
    m2.q = pyo.Var(initialize=0.0)
    with pytest.raises(ValueError, match="not a variable of the solved"):
        covariance(m, sigma_sq=SIGMA**2, wrt=[m2.q])
    # a fixed (equal-bounds) variable has no factor row to reduce onto
    x, y, _ = linear_data()
    m3 = linear_model(x, y)
    m3.dead = pyo.Var(bounds=(2.0, 2.0), initialize=2.0)
    m3.deadcon = pyo.Constraint(expr=m3.dead * m3.dead <= 1e6)
    pyo.SolverFactory("pounce").solve(m3)
    with pytest.raises(ValueError, match="removed from the solve"):
        covariance(m3, sigma_sq=SIGMA**2, wrt=[m3.dead])
