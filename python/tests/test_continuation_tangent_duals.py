"""`Continuation._tangent(predict_duals=True)` reads every block of the
compound KKT step through its row map, not by offset.

The model is built so each offset shortcut lands on a neighbour:

* ``x0`` is **free** and sits ahead of the lower-bounded ``x1``, so the
  ``z_l`` block (one entry per lower-bounded variable) has ``x1`` at
  position 0 -- reading it as variable 0 credits ``x0`` with ``x1``'s
  multiplier step;
* ``x2`` is **fixed** (``lb == ub``, ``make_parameter``), so the var-x
  block is one shorter than ``n`` and ``full[:n]`` takes an ``s`` entry;
* ``g0`` is an **active inequality**, whose multiplier lives in ``y_d``
  and which ``multiplier_rows`` answers ``None`` for.

Every predicted block is checked against the finite difference of the
duals of two converged solves, at ``theta`` and ``theta + dtheta``. The
model is linear in theta at its solution, so a forward difference is
exact up to the solve tolerance. The ``obj_scaling_factor`` leg checks
the step is in the same (natural) units as the reported duals.
"""

import numpy as np
import pytest

import pounce

pytestmark = pytest.mark.filterwarnings("ignore::RuntimeWarning")

INF = 1e19
FIX = 0.5
# x0 free, x1 >= 0, x2 fixed, x3 <= 1, x4 = theta (the parameter)
LB = np.array([-INF, 0.0, FIX, -INF, -INF])
UB = np.array([INF, INF, FIX, 1.0, INF])
PINS = [1]


class Model:
    """f = (x0 - 2p)^2 + (x1 + p)^2 + (x3 - 3p)^2 + x2*x4,
    g0 = x0 + x2 - 0.5 p <= 0 (active), g1 = p (the pin).

    At theta near 1: x1 = 0 with z_l = 2p, x3 = 1 with z_u = 6p - 2,
    g0 active with lambda = 3p + 1."""

    def objective(self, x):
        return ((x[0] - 2 * x[4]) ** 2 + (x[1] + x[4]) ** 2
                + (x[3] - 3 * x[4]) ** 2 + x[2] * x[4])

    def gradient(self, x):
        return np.array([
            2 * (x[0] - 2 * x[4]),
            2 * (x[1] + x[4]),
            x[4],
            2 * (x[3] - 3 * x[4]),
            -4 * (x[0] - 2 * x[4]) + 2 * (x[1] + x[4])
            - 6 * (x[3] - 3 * x[4]) + x[2],
        ])

    def constraints(self, x):
        return np.array([x[0] + x[2] - 0.5 * x[4], x[4]])

    def jacobianstructure(self):
        return (np.array([0, 0, 0, 1], dtype=np.int64),
                np.array([0, 2, 4, 4], dtype=np.int64))

    def jacobian(self, x):
        return np.array([1.0, 1.0, -0.5, 1.0])

    def hessianstructure(self):
        # lower triangle: (0,0) (1,1) (3,3) (4,0) (4,1) (4,2) (4,3) (4,4)
        return (np.array([0, 1, 3, 4, 4, 4, 4, 4], dtype=np.int64),
                np.array([0, 1, 3, 0, 1, 2, 3, 4], dtype=np.int64))

    def hessian(self, x, lagrange, obj_factor):
        h = np.array([2.0, 2.0, 2.0, -4.0, 2.0, 1.0, -6.0, 8.0 + 2.0 + 18.0])
        return obj_factor * h


def make_update(options):
    def update(theta):
        th = float(np.asarray(theta, float).ravel()[0])
        p = pounce.Problem(n=5, m=2, problem_obj=Model(), lb=LB, ub=UB,
                           cl=np.array([-INF, th]), cu=np.array([0.0, th]))
        p.add_option("tol", 1e-10)
        p.add_option("print_level", 0)
        p.add_option("sb", "yes")
        p.add_option("fixed_variable_treatment", "make_parameter")
        for k, v in options.items():
            p.add_option(k, v)
        return p
    return update


X0 = np.array([0.0, 1.0, FIX, 0.0, 1.0])


def solve(update, theta):
    prob = update(np.array([theta]))
    solver = pounce.Solver(prob)
    x, info = solver.solve(x0=X0)
    assert solver.converged
    return prob, solver, np.asarray(x, float), info


@pytest.mark.parametrize("options", [
    {},
    {"obj_scaling_factor": 7.0},
], ids=["unscaled", "obj_scaled"])
def test_predicted_duals_match_finite_differences(options):
    update = make_update(options)
    theta, dtheta = 1.0, 1e-3
    prob, solver, x0, info0 = solve(update, theta)
    _, _, x1, info1 = solve(update, theta + dtheta)

    # the fixture does what the docstring says, or the test is vacuous
    assert x0[1] == pytest.approx(0.0, abs=1e-7)
    assert x0[3] == pytest.approx(1.0, abs=1e-7)
    assert np.asarray(info0["mult_g"])[0] == pytest.approx(4.0, rel=1e-6)
    assert solver.block_dims[0] == 4  # the fixed variable has no row

    drv = pounce.Continuation(update, pins=PINS, predict_duals=True)
    dx, dlam, dzl, dzu = drv._tangent(solver, prob, np.array([dtheta]))
    assert dx is not None and dlam is not None

    fd = {
        "x": x1 - x0,
        "lambda": np.asarray(info1["mult_g"]) - np.asarray(info0["mult_g"]),
        "z_L": (np.asarray(info1["mult_x_L"])
                - np.asarray(info0["mult_x_L"])),
        "z_U": (np.asarray(info1["mult_x_U"])
                - np.asarray(info0["mult_x_U"])),
    }
    pred = {"x": dx, "lambda": dlam, "z_L": dzl, "z_U": dzu}
    # analytic steps, so a wrong FD cannot make a wrong prediction pass
    expected = {
        "x": np.array([0.5, 0.0, 0.0, 0.0, 1.0]) * dtheta,
        "lambda": np.array([3.0, -24.5]),
        "z_L": np.array([0.0, 2.0, None, 0.0, 0.0]),
        "z_U": np.array([0.0, 0.0, None, 6.0, 0.0]),
    }
    # the fixed variable's reported multiplier is reconstructed by the
    # solve, not a KKT row, so it is not compared
    free = [0, 1, 3, 4]
    for blk, idx in [("x", list(range(5))), ("lambda", [0, 1]),
                     ("z_L", free), ("z_U", free)]:
        np.testing.assert_allclose(
            pred[blk][idx], fd[blk][idx], rtol=1e-4, atol=1e-7,
            err_msg=f"block {blk}: predicted vs finite difference")
    for blk in ("lambda", "z_L", "z_U"):
        e = expected[blk]
        known = [i for i in range(e.size) if e[i] is not None]
        np.testing.assert_allclose(
            pred[blk][known], np.array(e[known], float) * dtheta,
            rtol=1e-4, atol=1e-7, err_msg=f"block {blk}: analytic")
    np.testing.assert_allclose(dx, expected["x"], rtol=1e-4, atol=1e-7)


def test_primal_only_tangent_is_user_space_length():
    """predict_duals=False takes the var-x `parametric_step`, which is
    one short here; it must come back in user space."""
    update = make_update({})
    prob, solver, _, _ = solve(update, 1.0)
    drv = pounce.Continuation(update, pins=PINS)
    dx, *_ = drv._tangent(solver, prob, np.array([1e-3]))
    np.testing.assert_allclose(dx, np.array([0.5, 0, 0, 0, 1]) * 1e-3,
                               rtol=1e-4, atol=1e-7)


def test_bound_multiplier_rows_map_through_bounded_space():
    update = make_update({})
    _, solver, _, _ = solve(update, 1.0)
    dims = solver.block_dims
    z_l_off = sum(dims[:4])
    rows = solver.bound_multiplier_rows([0, 1, 2, 3, 4])
    assert rows == [
        (None, None),               # free
        (z_l_off, None),            # first (only) lower-bounded var
        (None, None),               # fixed: removed
        (None, z_l_off + dims[4]),  # first (only) upper-bounded var
        (None, None),               # free
    ]
    with pytest.raises(ValueError):
        solver.bound_multiplier_rows([5])
