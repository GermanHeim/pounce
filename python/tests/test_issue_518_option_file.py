"""gh#518 on the Python surface: `option_file_name` is not silently dropped,
and the CLI's implicit options-file lookup does not leak into the library.

#518 was reported against the CLI — an options file that configured nothing
while the run reported success. The fix implements `option_file_name` *there*,
which leaves this surface with two ways to reproduce the same failure:

1. Options reach a file only through the CLI's resolution path. A Python
   caller who sets `option_file_name` would otherwise get exactly the reported
   bug — an option naming a whole configuration, applying none of it, and
   still returning a solution. It is refused instead.
2. The implicit `./pounce.opt` / `./ipopt.opt` lookup is deliberately *not*
   extended here. A stray file in the working directory silently steering a
   `pounce.Solver` call inside a notebook or a GAMS link would be action at a
   distance, and worse than not having the lookup. The A/B below is the test
   that keeps it that way.
"""

import os

os.environ.setdefault("RUST_LOG", "off")

import numpy as np
import pytest

import pounce


class Quadratic:
    """min (x-3)^2 over [-10, 10]; unconstrained, optimum x = 3."""

    def objective(self, x):
        return float((x[0] - 3.0) ** 2)

    def gradient(self, x):
        return np.array([2.0 * (x[0] - 3.0)])

    def constraints(self, x):
        return np.array([])

    def jacobianstructure(self):
        empty = np.array([], dtype=np.int64)
        return empty, empty

    def jacobian(self, x):
        return np.array([])

    def hessianstructure(self):
        idx = np.array([0], dtype=np.int64)
        return idx, idx

    def hessian(self, x, lagrange, obj_factor):
        return np.array([2.0 * obj_factor])


def solve(**options):
    p = pounce.Problem(
        n=1, m=0, problem_obj=Quadratic(), lb=[-10.0], ub=[10.0], cl=[], cu=[]
    )
    p.add_option("print_level", 0)
    p.add_option("sb", "yes")
    for k, v in options.items():
        p.add_option(k, v)
    return pounce.Solver(p).solve(x0=np.array([0.0]))


def test_option_file_name_is_refused_not_dropped():
    """Naming a file the library will never read must not look like it worked."""
    _, info = solve(option_file_name="tiny.opt")
    assert info["status_msg"] == "Invalid_Option"


def test_option_file_name_at_its_registered_default_still_solves():
    """`ipopt.opt` is the registered default, so setting it asks for nothing —
    the same rule that lets a generated options file spell out defaults."""
    x, info = solve(option_file_name="ipopt.opt")
    assert info["status_msg"] == "Solve_Succeeded"
    assert x[0] == pytest.approx(3.0, abs=1e-6)


def test_a_stray_options_file_does_not_steer_a_library_solve(tmp_path, monkeypatch):
    """The CLI reads `./ipopt.opt`; the library must not. Same solve, run with
    and without the file present — `max_iter 1` would be unmistakable if it
    leaked (the solve needs more than one iteration)."""
    monkeypatch.chdir(tmp_path)

    x_clean, info_clean = solve()
    assert info_clean["status_msg"] == "Solve_Succeeded"

    (tmp_path / "ipopt.opt").write_text("max_iter 1\n")
    (tmp_path / "pounce.opt").write_text("max_iter 1\n")
    x_stray, info_stray = solve()

    assert info_stray["status_msg"] == "Solve_Succeeded"
    assert x_stray[0] == pytest.approx(x_clean[0], abs=1e-12)
