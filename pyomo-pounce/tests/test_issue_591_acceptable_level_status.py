"""gh #591: an accepted (reduced-accuracy) solve must load as `status=ok`.

POUNCE used to write `Solved_To_Acceptable_Level` into the `.sol` as AMPL
`solve_result_num = 100`. Pyomo's legacy reader maps the `100..199` band to
`SolverStatus.warning` (with `TerminationCondition.optimal`) and the `0..99`
band to `SolverStatus.ok`, so an accepted POUNCE solve arrived as a warning —
Pyomo logging "Loading a SolverResults object with a warning status" — while
the equivalent IPOPT solve ("Solved To Acceptable Level.") arrived as `ok`.
POUNCE now emits IPOPT's own code, `1`.

That covers the `SolverFactory("pounce")` plugin and the route where the
POUNCE binary is driven through Pyomo's generic `ipopt` ASL interface alike,
since both read the same `.sol`. The in-process sensitivity route does not go
through a `.sol` at all, so its table is fixed here too.
"""
import pyomo.environ as pyo

import pyomo_pounce  # noqa: F401  (registers 'pounce')
from pyomo_pounce.sens import _STATUS_RESULT
from pyomo_pounce.v2 import _V2_STATUS

#: `tol` below anything reachable plus a generous `acceptable_tol` routes the
#: solve through the acceptable-level fallback (the recipe the Rust-side
#: `issue_591_acceptable_level_solved_band` and `#119` tests use).
FORCE_ACCEPTABLE = {
    "tol": 1e-30,
    "acceptable_tol": 1e-4,
    "acceptable_iter": 1,
    "print_level": 0,
}


def build():
    """A small smooth NLP with an active constraint. Deliberately not a
    quadratic: a QP is dispatched to the convex engine, whose own
    reduced-accuracy status is a different code path."""
    m = pyo.ConcreteModel()
    m.x = pyo.Var(initialize=1.0, bounds=(None, 3.0))
    m.y = pyo.Var(initialize=1.0)
    m.c = pyo.Constraint(expr=m.x + m.y >= 1.0)
    m.obj = pyo.Objective(
        expr=(m.x - 2.0) ** 2 + (m.y - 1.0) ** 2 + 0.1 * pyo.exp(m.x + m.y))
    return m


def test_acceptable_level_is_not_a_warning_in_the_sens_table():
    """The in-process (declared-parameter) route reports the same severity as
    a clean solve — an accepted solve is accepted."""
    tc, status = _STATUS_RESULT["Solved_To_Acceptable_Level"]
    assert tc is pyo.TerminationCondition.optimal
    assert status is pyo.SolverStatus.ok
    # Same severity as Solve_Succeeded; the two differ in the solver message,
    # not in whether the result loads clean.
    assert status is _STATUS_RESULT["Solve_Succeeded"][1]


def test_the_three_interfaces_agree_that_it_is_a_success():
    """v2 already treated it as a success. The point of #591 is that POUNCE's
    own interfaces must not disagree about the severity of one status."""
    v2_tc, v2_soln = _V2_STATUS["Solved_To_Acceptable_Level"]
    assert (v2_tc, v2_soln) == _V2_STATUS["Solve_Succeeded"]
    assert _STATUS_RESULT["Solved_To_Acceptable_Level"] == \
        _STATUS_RESULT["Solve_Succeeded"]


def test_sol_route_loads_an_accepted_solve_as_ok(pounce_exe):
    """End-to-end over the ordinary `.sol` route: the reported symptom.

    Driven through the `pounce_exe` fixture rather than plain resolution: the
    assertion is about the code *this* build writes, and a stale binary would
    answer for a different one."""
    m = build()
    results = pyo.SolverFactory("pounce", executable=pounce_exe).solve(
        m, options=FORCE_ACCEPTABLE)

    message = str(results.solver.message)
    # Guard the fixture: if the recipe stops reaching the acceptable-level
    # fallback, this test silently stops testing #591.
    assert "SolvedToAcceptableLevel" in message, (
        f"expected the acceptable-level fallback, got message {message!r} "
        f"with status {results.solver.status}"
    )
    assert results.solver.termination_condition is \
        pyo.TerminationCondition.optimal
    assert results.solver.status is pyo.SolverStatus.ok, (
        "an accepted solve must load without Pyomo's warning-status warning; "
        "IPOPT reports the equivalent solve as ok (gh #591)"
    )
    # The scientific distinction stays visible where it belongs: the message.
    assert "SolvedToAcceptableLevel" in message
