"""Tests for pyomo_pounce.estimate_report: what the linear step does
about the bounds."""
import numpy as np

import pytest
import pyomo.environ as pyo

import pyomo_pounce  # noqa: F401  (registers 'pounce')
from pyomo_pounce import declare_sens_param, estimate, estimate_report


def bounded(ub_y=5.0, fixed=False):
    """min (x-p)^2 + (y-2p)^2, so the unconstrained solution is x = p,
    y = 2p and the step is dx/dp = 1, dy/dp = 2. From p = 1, y reaches
    ub_y after (ub_y - 2) / 2 units of p."""
    m = pyo.ConcreteModel()
    m.p = pyo.Param(initialize=1.0, mutable=True)
    m.x = pyo.Var(bounds=(0.0, 10.0), initialize=1.0)
    m.y = pyo.Var(bounds=(-5.0, ub_y), initialize=1.0)
    expr = (m.x - m.p) ** 2 + (m.y - 2 * m.p) ** 2
    if fixed:
        # a fixed variable is removed from the solve, which shifts every
        # later factor column; the report must stay in user space
        m.f = pyo.Var(bounds=(3.0, 3.0), initialize=3.0)
        expr = expr + (m.f - 3.0) ** 2
    m.obj = pyo.Objective(expr=expr)
    declare_sens_param(m.p)
    pyo.SolverFactory("pounce").solve(m)
    return m


def with_row():
    """The same objective under x + y <= 6, which binds at p = 2 while
    both variables are still interior."""
    m = pyo.ConcreteModel()
    m.p = pyo.Param(initialize=1.0, mutable=True)
    m.x = pyo.Var(bounds=(-50.0, 50.0), initialize=1.0)
    m.y = pyo.Var(bounds=(-50.0, 50.0), initialize=2.0)
    m.c = pyo.Constraint(expr=m.x + m.y <= 6.0)
    m.obj = pyo.Objective(expr=(m.x - m.p) ** 2 + (m.y - 2 * m.p) ** 2)
    declare_sens_param(m.p)
    pyo.SolverFactory("pounce").solve(m)
    return m


def brute_force(model, param, newval):
    """Scan the unclamped step for the first variable bound reached."""
    return brute_force_multi(model, [(param, newval)])


def brute_force_multi(model, perturb):
    base = {id(v): pyo.value(v)
            for v in model.component_data_objects(pyo.Var, active=True)}
    est = estimate(model, perturb, clamp=False)
    alpha, who = float("inf"), None
    for v, x1 in est.items():
        x0 = base[id(v)]
        d = x1 - x0
        if abs(d) < 1e-14:
            continue
        b = (v.ub if v.ub is not None else np.inf) if d > 0 else (
            v.lb if v.lb is not None else -np.inf)
        if not np.isfinite(b):
            continue
        a = max((b - x0) / d, 0.0)
        if a < alpha:
            alpha, who = a, v.name
    return alpha, who


def test_step_fraction_matches_the_hand_computed_crossing():
    # y goes 2 -> 8 over p = 1 -> 4 and stops at 5, which is half way
    m = bounded()
    r = estimate_report(m, [(m.p, 4.0)])
    assert r.first == "y"
    assert r.first_kind == "variable"
    assert r.alpha == pytest.approx(0.5, abs=1e-8)


def test_step_fraction_matches_a_brute_force_scan():
    m = bounded()
    alpha, who = brute_force(m, m.p, 4.0)
    r = estimate_report(m, [(m.p, 4.0)])
    assert r.first == who
    assert r.alpha == pytest.approx(alpha, rel=1e-12)


def test_a_fixed_variable_does_not_shift_the_scan():
    m = bounded(fixed=True)
    alpha, who = brute_force(m, m.p, 4.0)
    r = estimate_report(m, [(m.p, 4.0)])
    assert r.first == who == "y"
    assert r.alpha == pytest.approx(alpha, rel=1e-12)


def test_an_interior_perturbation_crosses_nothing():
    m = bounded()
    r = estimate_report(m, [(m.p, 1.2)])
    assert r.alpha > 1.0
    assert len(r.crossed) == 0
    assert r.crossed_rows == {}
    assert r.violation == pytest.approx(0.0, abs=1e-8)


def test_crossed_reports_the_distance_past_the_bound():
    m = bounded()
    r = estimate_report(m, [(m.p, 4.0)])
    est = estimate(m, [(m.p, 4.0)], clamp=False)
    assert len(r.crossed) == 1
    assert m.y in r.crossed
    assert r.crossed[m.y] == pytest.approx(est[m.y] - m.y.ub, rel=1e-9)


def test_a_constraint_row_can_bind_before_any_variable():
    m = with_row()
    r = estimate_report(m, [(m.p, 3.0)])
    # the row sits at 3 and gains 3 per unit of p, so it reaches 6 at
    # p = 2, half way to p = 3
    assert r.first_kind == "constraint"
    assert r.alpha == pytest.approx(0.5, abs=1e-8)
    assert len(r.crossed) == 0


def test_rows_are_named_as_the_model_names_them():
    m = with_row()
    r = estimate_report(m, [(m.p, 3.0)])
    assert r.first == "c"
    assert set(r.crossed_rows) == {"c"}
    assert "c" in r.row_activity


def test_violation_matches_direct_evaluation_at_the_predicted_point():
    m = with_row()
    r = estimate_report(m, [(m.p, 3.0)])
    for v, val in estimate(m, [(m.p, 3.0)], clamp=False).items():
        v.set_value(val)
    body = pyo.value(m.x) + pyo.value(m.y)
    assert r.violation == pytest.approx(max(body - 6.0, 0.0), rel=1e-12)


def test_the_pin_row_is_not_reported_as_a_crossing():
    # the perturbation moves the pin row's right-hand side by
    # construction, so it is neither a crossing nor a violation
    m = bounded()
    r = estimate_report(m, [(m.p, 4.0)])
    assert all("paramConst" not in nm for nm in r.crossed_rows)
    assert r.violation == pytest.approx(0.0, abs=1e-8)


def test_classification_and_mu_match_the_core_classifier():
    m = bounded()
    r = estimate_report(m, [(m.p, 4.0)])
    session = m.__dict__["_pounce_sens"].session
    act = session.solver.classify_activity()
    assert r.mu == pytest.approx(float(act["mu"]), rel=1e-12)
    assert r.activity["y"] == act["var_status"][session.var_names.index("y")]
    assert set(r.activity) >= {"x", "y"}


def test_an_already_active_bound_is_not_a_crossing():
    """A variable on its bound has an O(mu) gap left and an O(mu) step
    component, and their quotient is noise. It must not set the step
    fraction: the classification is what reports it."""
    m = pyo.ConcreteModel()
    m.p = pyo.Param(initialize=4.0, mutable=True)
    m.x = pyo.Var(bounds=(0.0, 10.0), initialize=1.0)
    m.y = pyo.Var(bounds=(-5.0, 5.0), initialize=1.0)
    m.obj = pyo.Objective(expr=(m.x - m.p) ** 2 + (m.y - 2 * m.p) ** 2)
    declare_sens_param(m.p)
    pyo.SolverFactory("pounce").solve(m)
    assert pyo.value(m.y) == pytest.approx(5.0, abs=1e-6)

    r = estimate_report(m, [(m.p, 5.0)])
    assert r.activity["y"] == "strongly_active"
    assert r.first == "x"          # x runs 4 -> 5 against its bound of 10
    assert r.alpha == pytest.approx(6.0, rel=1e-6)


def test_a_saturating_control_is_named_with_its_step_fraction():
    """The case the diagnostics exist for: a setpoint move large enough
    to drive a control onto its bound, where estimate() clamps and
    reports nothing about which control or how far along the move it
    happened."""
    n, a, b, r = 6, 0.8, 0.5, 0.05
    m = pyo.ConcreteModel()
    m.k = pyo.RangeSet(0, n - 1)
    m.sp = pyo.Param(initialize=0.5, mutable=True)
    m.x = pyo.Var(pyo.RangeSet(0, n), initialize=0.0)
    m.u = pyo.Var(m.k, bounds=(-1.0, 1.0), initialize=0.0)
    m.x[0].fix(0.0)

    @m.Constraint(m.k)
    def dynamics(m, k):
        return m.x[k + 1] == a * m.x[k] + b * m.u[k]

    m.obj = pyo.Objective(
        expr=sum((m.x[k + 1] - m.sp) ** 2 for k in m.k)
        + r * sum(m.u[k] ** 2 for k in m.k))
    declare_sens_param(m.sp)
    pyo.SolverFactory("pounce").solve(m)
    assert all(abs(pyo.value(m.u[k])) < 0.999 for k in m.k)

    r_small = estimate_report(m, [(m.sp, 0.55)])
    assert r_small.alpha > 1.0
    assert len(r_small.crossed) == 0

    r_big = estimate_report(m, [(m.sp, 3.0)])
    assert r_big.first_kind == "variable"
    assert r_big.first.startswith("u[")
    assert 0.0 < r_big.alpha < 1.0
    assert r_big.crossed                     # estimate() clamps these
    assert all(v.name.startswith("u[") for v in r_big.crossed)

    # the step fraction is the fraction of the setpoint move that fits,
    # so the perturbation it admits crosses nothing
    fits = 0.5 + r_big.alpha * (3.0 - 0.5)
    assert estimate_report(m, [(m.sp, fits)]).alpha == pytest.approx(
        1.0, rel=1e-6)


def test_provenance_is_reported_on_an_ordinary_solve():
    """The three things separating the predictor from the exact value
    at the perturbed active set: the barrier parameter, whether the
    factor was regularized, and whether the solve relaxed its bounds."""
    m = bounded()
    r = estimate_report(m, [(m.p, 4.0)])
    assert np.isfinite(r.mu) and r.mu > 0.0
    assert r.bounds_relaxed is False
    assert not any(r.perturbations)     # convex model, no inertia correction


def test_a_relaxed_solve_is_reported_rather_than_raising():
    """`bound_relax_factor` lets a variable settle outside its declared
    bound, so the classifier refuses the solve. The rest of the report
    is still measured, since a caller reaches for it precisely when the
    estimate and a re-solve disagree."""
    m = pyo.ConcreteModel()
    m.p = pyo.Param(initialize=1.0, mutable=True)
    m.x = pyo.Var(bounds=(0.0, 10.0), initialize=1.0)
    m.y = pyo.Var(bounds=(-5.0, 5.0), initialize=1.0)
    m.obj = pyo.Objective(expr=(m.x - m.p) ** 2 + (m.y - 2 * m.p) ** 2)
    declare_sens_param(m.p)
    pyo.SolverFactory("pounce").solve(
        m, options={"bound_relax_factor": 1e-8})

    r = estimate_report(m, [(m.p, 4.0)])
    assert r.bounds_relaxed is True
    assert r.activity == {} and r.row_activity == {}
    assert np.isnan(r.mu)
    # the measured half still lands
    assert r.first == "y"
    assert r.alpha == pytest.approx(0.5, abs=1e-6)


def test_a_weakly_active_bound_is_classified_as_such():
    """min (x - 1)^2 with x <= 1 puts the bound on the unconstrained
    minimum, so the bound is active and its multiplier is zero: strict
    complementarity fails and the classification must say so rather
    than call it strongly active."""
    m = pyo.ConcreteModel()
    m.p = pyo.Param(initialize=1.0, mutable=True)
    m.x = pyo.Var(bounds=(None, 1.0), initialize=0.0)
    m.z = pyo.Var(bounds=(-5.0, 5.0), initialize=0.0)
    m.obj = pyo.Objective(expr=(m.x - m.p) ** 2 + (m.z - m.p) ** 2)
    declare_sens_param(m.p)
    pyo.SolverFactory("pounce").solve(m)
    # the barrier leaves an O(sqrt(mu)) gap here, not the O(mu) gap it
    # leaves at a strongly active bound
    assert pyo.value(m.x) == pytest.approx(1.0, abs=1e-3)

    r = estimate_report(m, [(m.p, 1.5)])
    assert r.activity["x"] == "weakly_active"
    # that gap is barrier residue, so it is not room the step can cross:
    # scoring it would put the first crossing at a fraction of a percent
    assert r.first != "x"
    assert r.first == "z"


def test_several_parameters_perturbed_at_once():
    m = pyo.ConcreteModel()
    m.p = pyo.Param(initialize=1.0, mutable=True)
    m.q = pyo.Param(initialize=1.0, mutable=True)
    m.x = pyo.Var(bounds=(0.0, 10.0), initialize=1.0)
    m.y = pyo.Var(bounds=(-5.0, 5.0), initialize=1.0)
    m.obj = pyo.Objective(expr=(m.x - m.p) ** 2 + (m.y - 2 * m.q) ** 2)
    declare_sens_param(m.p)
    declare_sens_param(m.q)
    pyo.SolverFactory("pounce").solve(m)

    # y tracks q alone and reaches 5 at q = 2.5, half way to q = 4
    r = estimate_report(m, [(m.p, 2.0), (m.q, 4.0)])
    assert r.first == "y"
    assert r.alpha == pytest.approx(0.5, abs=1e-6)
    alpha, who = brute_force_multi(m, [(m.p, 2.0), (m.q, 4.0)])
    assert r.first == who
    assert r.alpha == pytest.approx(alpha, rel=1e-12)


def test_every_solver_route_reports_the_same():
    """`Pounce.solve` sends a model carrying declarations down the same
    in-process sensitivity route the legacy plugin uses, so one session
    serves all three entry points and the report cannot depend on which
    one ran."""
    from pyomo.contrib.solver.common.factory import SolverFactory as SF2

    reports = []
    for solve in (lambda m: pyo.SolverFactory("pounce").solve(m),
                  lambda m: pyo.SolverFactory("pounce_v2").solve(m),
                  lambda m: SF2("pounce").solve(m)):
        m = pyo.ConcreteModel()
        m.p = pyo.Param(initialize=1.0, mutable=True)
        m.x = pyo.Var(bounds=(0.0, 10.0), initialize=1.0)
        m.y = pyo.Var(bounds=(-5.0, 5.0), initialize=1.0)
        m.obj = pyo.Objective(expr=(m.x - m.p) ** 2 + (m.y - 2 * m.p) ** 2)
        declare_sens_param(m.p)
        solve(m)
        reports.append(estimate_report(m, [(m.p, 4.0)]))

    legacy, v2, contrib = reports
    for other in (v2, contrib):
        assert other.first == legacy.first == "y"
        assert other.alpha == pytest.approx(legacy.alpha, rel=1e-12)
        assert other.mu == pytest.approx(legacy.mu, rel=1e-12)
        assert other.activity == legacy.activity


def test_no_session_is_a_clean_error():
    m = pyo.ConcreteModel()
    m.p = pyo.Param(initialize=1.0, mutable=True)
    m.x = pyo.Var(initialize=1.0)
    m.obj = pyo.Objective(expr=(m.x - m.p) ** 2)
    with pytest.raises(RuntimeError, match="no sensitivity session"):
        estimate_report(m, [(m.p, 2.0)])
