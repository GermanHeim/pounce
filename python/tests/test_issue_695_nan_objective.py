"""gh #695: a non-finite objective must never be reported as a successful solve.

A model whose ``objective`` returns ``NaN`` (or ``inf``) while every derivative
is finite converged to ``status = 0 Solve_Succeeded`` with ``obj_val = nan``.
A caller that gates on ``success`` and then reads ``fun`` silently received
``NaN``.

The bug was specific to the **equality-constrained** shape. That is what let it
survive gh #292, which closed the NaN-*gradient* hole and explicitly recorded
``fun``-returns-NaN as the safe contrast case — true for the shapes it
exercised, and not once an equality constraint is present. The whole shape
matrix is asserted here so a later change cannot close one column and reopen
another.

This lives in the Python suite because the defect is callback-API-only: an
``.nl`` model cannot express a NaN objective with a consistent finite gradient,
so the CLI path cannot reach it. The in-process regression test is
``crates/pounce-algorithm/tests/issue_695_nan_objective_false_success.rs``; this
one pins the reporter's own route, where the solve reaches the convergence test
*through restoration* rather than directly.
"""

from __future__ import annotations

import numpy as np
import pytest

import pounce

BIG = 2.0e19


class _NonFiniteObjective:
    """``min f(x) s.t. x0 + x1 == 1`` with ``f`` non-finite by construction.

    Every derivative is finite and exact, so each quantity the convergence test
    actually reads — gradients, residuals, complementarity — is well defined and
    the solve converges on them. Nothing in that test looks at the objective
    *value*, which is precisely why the guard has to be explicit.
    """

    def __init__(self, value: float, m: int) -> None:
        self.value = value
        self.m = m

    def objective(self, x):
        return self.value

    def gradient(self, x):
        return 2.0 * np.asarray(x, float)

    def constraints(self, x):
        return np.array([float(np.sum(x))]) if self.m else np.zeros(0)

    def jacobian(self, x):
        return np.ones(2) if self.m else np.zeros(0)

    def jacobianstructure(self):
        return (np.zeros(2, dtype=int), np.arange(2))

    def hessian(self, x, lag, of):
        return of * 2.0 * np.ones(2)

    def hessianstructure(self):
        return (np.arange(2), np.arange(2))


def _solve(value, *, m, cl=None, cu=None, lb=(-BIG, -BIG), ub=(BIG, BIG)):
    kwargs = dict(n=2, m=m, problem_obj=_NonFiniteObjective(value, m), lb=list(lb), ub=list(ub))
    if m:
        kwargs.update(cl=list(cl), cu=list(cu))
    problem = pounce.Problem(**kwargs)
    problem.add_option("print_level", 0)
    _, info = problem.solve(x0=np.array([1.0, 1.0]))
    return info


# (tag, kwargs) for every shape in the issue's matrix.
_SHAPES = [
    ("no bounds, no constraints", dict(m=0)),
    ("bounds only", dict(m=0, lb=(-10.0, -10.0), ub=(10.0, 10.0))),
    ("eq constraint only", dict(m=1, cl=(1.0,), cu=(1.0,))),
    (
        "bounds + eq constraint",
        dict(m=1, cl=(1.0,), cu=(1.0,), lb=(-10.0, -10.0), ub=(10.0, 10.0)),
    ),
    (
        "bounds + ineq constraint",
        dict(m=1, cl=(-BIG,), cu=(1.0,), lb=(-10.0, -10.0), ub=(10.0, 10.0)),
    ),
]


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
@pytest.mark.parametrize("tag,kwargs", _SHAPES, ids=[t for t, _ in _SHAPES])
def test_non_finite_objective_is_never_a_successful_solve(value, tag, kwargs):
    """``status == 0`` asserts the convergence test passed. Reporting it next to
    an objective that is not a number is self-contradictory, whatever the shape.
    """
    info = _solve(value, **kwargs)
    assert info["status"] != 0 or np.isfinite(info["obj_val"]), (
        f"{tag}: reported status={info['status']} {info['status_msg']} with "
        f"obj_val={info['obj_val']} (gh #695)"
    )


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_equality_constrained_shape_reports_invalid_number(value):
    """The column that regressed, against the status the issue's oracles agree
    on: Ipopt's ``Eval_f`` rejects a non-finite objective with
    ``Invalid_Number_Detected``, and POUNCE's own inequality-constrained shape
    already did the same.
    """
    info = _solve(value, m=1, cl=(1.0,), cu=(1.0,), lb=(-10.0, -10.0), ub=(10.0, 10.0))
    assert info["status"] == -13, (
        f"expected -13 Invalid_Number_Detected, got {info['status']} "
        f"{info['status_msg']}"
    )


def test_a_finite_objective_on_the_same_shape_still_solves():
    """The control: the guard must not be bought by rejecting good solves.

    ``min x·x s.t. x0 + x1 == 1`` has its optimum at ``x = (0.5, 0.5)``,
    ``f* = 0.5``.
    """

    class SumOfSquares(_NonFiniteObjective):
        def __init__(self):
            super().__init__(0.0, 1)

        def objective(self, x):
            return float(np.dot(x, x))

    problem = pounce.Problem(
        n=2,
        m=1,
        problem_obj=SumOfSquares(),
        lb=[-10.0, -10.0],
        ub=[10.0, 10.0],
        cl=[1.0],
        cu=[1.0],
    )
    problem.add_option("print_level", 0)
    x, info = problem.solve(x0=np.array([1.0, 1.0]))
    assert info["status"] == 0, f"{info['status']} {info['status_msg']}"
    assert info["obj_val"] == pytest.approx(0.5, abs=1e-6)
    assert np.allclose(x, [0.5, 0.5], atol=1e-6)
