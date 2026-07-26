"""`Solver.classify_activity`: post-solve activity classification
(dev-notes/covariance-information-roadmap.md item 0, gh #362).

The scalar study model min ½x² − p·x with x ≥ 0 walks the three
regimes by moving the unconstrained minimizer: p > 0 puts it inside
(bound inactive), p < 0 outside (strongly active), p = 0 exactly on
the bound (weakly active: slack and multiplier both O(√μ), where no
fixed threshold on either alone can classify). The same geometry
written as an inequality row (x unbounded, row x ≥ 0) must classify
identically: that is the gh #362 shape, where the activity lives on a
row and never shows up in the bound multipliers.
"""

import numpy as np
import pytest

import pounce


class ScalarBound:
    """min ½x² − p·x with the bound on the variable: x ≥ 0."""

    def __init__(self, p):
        self.p = p

    def objective(self, x):
        return 0.5 * x[0] ** 2 - self.p * x[0]

    def gradient(self, x):
        return np.array([x[0] - self.p])

    def constraints(self, x):
        return np.array([])

    def jacobianstructure(self):
        empty = np.array([], dtype=np.int64)
        return empty, empty

    def jacobian(self, x):
        return np.array([])

    def hessianstructure(self):
        return np.array([0], dtype=np.int64), np.array([0], dtype=np.int64)

    def hessian(self, x, lagrange, obj_factor):
        return np.array([obj_factor])


class ScalarRow:
    """min ½x² − p·x with the bound as an inequality row: g(x) = x ≥ 0,
    the variable itself unbounded."""

    def __init__(self, p):
        self.p = p

    def objective(self, x):
        return 0.5 * x[0] ** 2 - self.p * x[0]

    def gradient(self, x):
        return np.array([x[0] - self.p])

    def constraints(self, x):
        return np.array([x[0]])

    def jacobianstructure(self):
        zero = np.array([0], dtype=np.int64)
        return zero, zero

    def jacobian(self, x):
        return np.array([1.0])

    def hessianstructure(self):
        return np.array([0], dtype=np.int64), np.array([0], dtype=np.int64)

    def hessian(self, x, lagrange, obj_factor):
        return np.array([obj_factor])


class LinearBox:
    """min −x on 0 ≤ x ≤ 1: zero curvature everywhere, so activity is
    below the identification floor no matter how hard the bound binds."""

    def objective(self, x):
        return -x[0]

    def gradient(self, x):
        return np.array([-1.0])

    def constraints(self, x):
        return np.array([])

    def jacobianstructure(self):
        empty = np.array([], dtype=np.int64)
        return empty, empty

    def jacobian(self, x):
        return np.array([])

    def hessianstructure(self):
        empty = np.array([], dtype=np.int64)
        return empty, empty

    def hessian(self, x, lagrange, obj_factor):
        return np.array([])


def _options(p):
    p.add_option("tol", 1e-10)
    p.add_option("bound_relax_factor", 0.0)
    p.add_option("print_level", 0)
    p.add_option("sb", "yes")
    return p


def _solve_bound(p):
    prob = _options(pounce.Problem(
        n=1, m=0, problem_obj=ScalarBound(p),
        lb=[0.0], ub=[1e19], cl=[], cu=[],
    ))
    solver = pounce.Solver(prob)
    _, info = solver.solve(x0=np.array([0.5]))
    assert info["status_msg"] == "Solve_Succeeded"
    return solver.classify_activity()


def _solve_row(p):
    prob = _options(pounce.Problem(
        n=1, m=1, problem_obj=ScalarRow(p),
        lb=[-1e19], ub=[1e19], cl=[0.0], cu=[1e19],
    ))
    solver = pounce.Solver(prob)
    _, info = solver.solve(x0=np.array([0.5]))
    assert info["status_msg"] == "Solve_Succeeded"
    return solver.classify_activity()


@pytest.mark.parametrize("p, status", [
    (1.0, "inactive"),
    (-1.0, "strongly_active"),
    (0.0, "weakly_active"),
])
def test_variable_bound_regimes(p, status):
    rep = _solve_bound(p)
    assert rep["var_status"] == [status]
    assert rep["var_q_sign"][0] == 1
    assert not rep["var_off_central_path"][0]
    assert not rep["var_contaminated"][0]
    assert rep["mu"] < 1e-4


def test_variable_bound_ratio_scales():
    # on the central path z·s = μ with unit curvature: r ≈ μ inactive,
    # r ≈ 1/μ strongly active, r ≈ 1 weakly active
    mu = _solve_bound(1.0)["mu"]
    assert _solve_bound(1.0)["var_ratio"][0] == pytest.approx(mu, rel=10.0)
    assert _solve_bound(-1.0)["var_ratio"][0] > 1.0 / np.sqrt(mu)
    assert _solve_bound(0.0)["var_ratio"][0] == pytest.approx(1.0, rel=0.5)


@pytest.mark.parametrize("p, status", [
    (1.0, "inactive"),
    (-1.0, "strongly_active"),
    (0.0, "weakly_active"),
])
def test_row_agrees_with_bound(p, status):
    # gh #362: the same geometry moved onto an inequality row classifies
    # identically, and the now-unbounded variable reports as such
    rep = _solve_row(p)
    assert rep["row_status"] == [status]
    assert rep["row_q_sign"][0] == 1
    assert rep["var_status"] == ["unbounded"]
    assert np.isnan(rep["var_ratio"][0])


def test_zero_curvature_is_unidentified():
    prob = _options(pounce.Problem(
        n=1, m=0, problem_obj=LinearBox(),
        lb=[0.0], ub=[1.0], cl=[], cu=[],
    ))
    solver = pounce.Solver(prob)
    _, info = solver.solve(x0=np.array([0.5]))
    assert info["status_msg"] == "Solve_Succeeded"
    rep = solver.classify_activity()
    assert rep["var_status"] == ["unidentified"]
    assert rep["var_q_sign"][0] == 0


def test_relaxed_bounds_are_refused():
    # bound_relax_factor defaults to 1e-8; the classifier's slacks and
    # complementarity products assume unperturbed bounds
    prob = pounce.Problem(
        n=1, m=0, problem_obj=ScalarBound(1.0),
        lb=[0.0], ub=[1e19], cl=[], cu=[],
    )
    prob.add_option("tol", 1e-10)
    prob.add_option("print_level", 0)
    prob.add_option("sb", "yes")
    solver = pounce.Solver(prob)
    solver.solve(x0=np.array([0.5]))
    with pytest.raises(ValueError, match="bound_relax_factor"):
        solver.classify_activity()


def test_classify_before_solve_raises():
    prob = _options(pounce.Problem(
        n=1, m=0, problem_obj=ScalarBound(1.0),
        lb=[0.0], ub=[1e19], cl=[], cu=[],
    ))
    with pytest.raises(RuntimeError, match="no converged factor"):
        pounce.Solver(prob).classify_activity()
