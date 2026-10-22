"""gh #1041: a convex QCQP whose ball sits far from the origin.

``min x s.t. (x − 1000)² ≤ 1`` (``x* = 999``) used to reach the conic solver
as the rotated cone ``(s+1, s−1, √2·Fx)``, whose ``‖Fx‖²`` and ``2s`` are both
~2·10⁶ at the optimum while their difference *is* the constraint. The conic
IPM stalled at the infeasible ``x ≈ 1003.41`` and, under ``auto``, that
failure was returned as-is. The router now centres the cone by completing the
square, and ``auto`` re-solves a conic ``numerical_failure`` on the NLP path
as the CLI does.
"""

import warnings

import numpy as np
import pounce
import pytest
from pounce import minimize


def _offset_ball(center):
    return {
        "type": "ineq",
        "fun": lambda x: 1.0 - (x[0] - center) ** 2,
        "jac": lambda x: np.array([-2.0 * (x[0] - center)]),
    }


@pytest.mark.parametrize("selection", ["auto", "socp"])
@pytest.mark.parametrize("center", [10.0, 100.0, 1000.0])
def test_offset_ball_solves_on_the_conic_path(selection, center):
    res = minimize(
        lambda x: x[0],
        np.array([center]),
        jac=lambda x: np.array([1.0]),
        constraints=[_offset_ball(center)],
        solver_selection=selection,
    )
    assert res.info.get("solver") == "socp"
    assert res.success, res.message
    assert res.x[0] == pytest.approx(center - 1.0, abs=1e-5)


def test_coupled_offset_ellipse_matches_nlp():
    # Non-diagonal Hessian: exercises the least-squares centring branch.
    fun = lambda x: x[0] + 2.0 * x[1]
    jac = lambda x: np.array([1.0, 2.0])
    con = {
        "type": "ineq",
        "fun": lambda x: 1.0 - ((x[0] + x[1] - 2000) ** 2 + 3 * (x[0] - x[1]) ** 2),
        "jac": lambda x: (
            -np.array(
                [
                    2 * (x[0] + x[1] - 2000) + 6 * (x[0] - x[1]),
                    2 * (x[0] + x[1] - 2000) - 6 * (x[0] - x[1]),
                ]
            )
        ),
    }
    res = minimize(
        fun,
        np.array([1000.0, 1000.0]),
        jac=jac,
        constraints=[con],
        solver_selection="auto",
    )
    assert res.info.get("solver") == "socp"
    assert res.success
    assert res.fun == pytest.approx(3000.0 - np.sqrt(7.0 / 3.0), abs=1e-6)


def _failing_socp(monkeypatch):
    real = pounce.qp.solve_socp

    def fake(*args, **kwargs):
        r = real(*args, **kwargs)
        r.status = "numerical_failure"
        return r

    monkeypatch.setattr(pounce.qp, "solve_socp", fake)


def test_auto_falls_back_to_nlp_on_conic_numerical_failure(monkeypatch):
    _failing_socp(monkeypatch)
    with pytest.warns(RuntimeWarning, match="re-solved on the general NLP path"):
        res = minimize(
            lambda x: x[0],
            np.array([1000.0]),
            jac=lambda x: np.array([1.0]),
            constraints=[_offset_ball(1000.0)],
            solver_selection="auto",
        )
    assert res.info.get("solver") != "socp"
    assert res.success
    assert res.x[0] == pytest.approx(999.0, abs=1e-6)


def test_forced_socp_keeps_the_conic_verdict(monkeypatch):
    _failing_socp(monkeypatch)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        res = minimize(
            lambda x: x[0],
            np.array([1000.0]),
            jac=lambda x: np.array([1.0]),
            constraints=[_offset_ball(1000.0)],
            solver_selection="socp",
        )
    assert res.info.get("solver") == "socp"
    assert not res.success
