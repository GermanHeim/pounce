"""Tests for degeneracy="directional": the directional-derivative QP
at a weakly active base point, in every estimate() mode, and the
gradient() warning at a kink."""
import warnings

import pytest
import pyomo.environ as pyo

import pyomo_pounce  # noqa: F401  (registers 'pounce')
from pyomo_pounce import (
    active_set_changes,
    declare_sens_param,
    estimate,
    gradient,
)


def kink(p=0.0):
    """min (x - p)^2 with x >= 0, held at p = 0: the solution sits
    exactly on the bound with a vanishing multiplier, the canonical
    kink. The two one-sided derivatives are 1 toward positive p, where
    the bound releases, and 0 toward negative p, where it holds."""
    m = pyo.ConcreteModel()
    m.p = pyo.Param(initialize=p, mutable=True)
    m.x = pyo.Var(bounds=(0.0, 10.0), initialize=0.5)
    m.obj = pyo.Objective(expr=(m.x - m.p) ** 2)
    declare_sens_param(m.p)
    pyo.SolverFactory("pounce").solve(m, options={"tol": 1e-10})
    return m


MODES = ("linear", "fix_relax", "path")


def test_directional_is_right_on_both_sides_in_every_mode():
    m = kink()
    assert pyo.value(m.x) == pytest.approx(0.0, abs=1e-4), "on the bound"
    for mode in MODES:
        up = estimate(m, [(m.p, 1.0)], mode=mode)
        down = estimate(m, [(m.p, -1.0)], mode=mode, clamp=False)
        assert up[m.x] == pytest.approx(1.0, abs=1e-4), (
            f"mode={mode}: the releasing side's derivative is 1")
        assert down[m.x] == pytest.approx(0.0, abs=1e-4), (
            f"mode={mode}: the holding side's derivative is 0")


def test_one_sided_is_wrong_on_at_least_one_side():
    """A single linear map cannot produce both one-sided derivatives,
    1 and 0, so whatever side the thresholds lean toward, the other
    answer under mode="linear" is wrong. This is the runnable before,
    and what keeps the test above from passing vacuously."""
    m = kink()
    up = estimate(m, [(m.p, 1.0)], degeneracy="one_sided", clamp=False)
    down = estimate(m, [(m.p, -1.0)], degeneracy="one_sided", clamp=False)
    up_right = abs(up[m.x] - 1.0) < 1e-4
    down_right = abs(down[m.x] - 0.0) < 1e-4
    assert not (up_right and down_right), (
        f"one_sided cannot be right on both sides: "
        f"up {up[m.x]}, down {down[m.x]}")


def test_the_record_shows_the_kink_resolving_at_fraction_zero():
    m = kink()
    rec = active_set_changes(m, [(m.p, 1.0)])
    assert [(c.var, c.bound, c.action, c.fraction) for c in rec] == [
        (m.x, "lower", "leaves", 0.0)], f"record: {rec}"
    assert active_set_changes(m, [(m.p, -1.0)]) == [], (
        "held through the whole change, nothing to record")


def test_gradient_warns_at_a_kink_and_not_at_a_clean_point():
    m = kink()
    with pytest.warns(UserWarning, match="one-sided"):
        gradient(m.x, wrt=m.p)

    clean = pyo.ConcreteModel()
    clean.p = pyo.Param(initialize=1.0, mutable=True)
    clean.x = pyo.Var(bounds=(0.0, 10.0), initialize=1.0)
    clean.obj = pyo.Objective(expr=(clean.x - clean.p) ** 2)
    declare_sens_param(clean.p)
    pyo.SolverFactory("pounce").solve(clean, options={"tol": 1e-10})
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        gradient(clean.x, wrt=clean.p)


def test_a_clean_base_point_is_identical_under_both_settings():
    m = pyo.ConcreteModel()
    m.p = pyo.Param(initialize=1.0, mutable=True)
    m.x = pyo.Var(bounds=(0.0, 10.0), initialize=1.0)
    m.y = pyo.Var(bounds=(-50.0, 50.0), initialize=1.0)
    m.link = pyo.Constraint(expr=m.y == 2 * m.x + 1)
    m.obj = pyo.Objective(expr=(m.x - m.p) ** 2 + 0.5 * (m.y - 3 * m.p) ** 2)
    declare_sens_param(m.p)
    pyo.SolverFactory("pounce").solve(m)
    for mode in MODES:
        a = estimate(m, [(m.p, 1.5)], mode=mode)
        b = estimate(m, [(m.p, 1.5)], mode=mode, degeneracy="one_sided")
        for v in (m.x, m.y):
            assert a[v] == b[v], f"mode={mode}: clean point must be identical"


def test_an_exhausted_budget_falls_back_with_a_warning():
    """max_iter=0 leaves the QP no trials, so the decision fails and
    the estimate falls back to the one-sided step and says so."""
    m = kink()
    with pytest.warns(UserWarning, match="one-sided step"):
        fell = estimate(m, [(m.p, -1.0)], max_iter=0, clamp=False)
    plain = estimate(m, [(m.p, -1.0)], degeneracy="one_sided", max_iter=0,
                     clamp=False)
    assert fell[m.x] == plain[m.x]


def test_an_unknown_degeneracy_value_is_refused():
    m = kink()
    with pytest.raises(ValueError, match="degeneracy must be"):
        estimate(m, [(m.p, 1.0)], degeneracy="qp")
    with pytest.raises(ValueError, match="degeneracy must be"):
        active_set_changes(m, [(m.p, 1.0)], degeneracy="ignore")


def coupled_kink(p=0.0):
    """The kink coupled to an interior variable through an equality, so
    the directional decision moves more than the degenerate variable
    itself: y tracks 2x + 1 on both sides of the kink."""
    m = pyo.ConcreteModel()
    m.p = pyo.Param(initialize=p, mutable=True)
    m.x = pyo.Var(bounds=(0.0, 10.0), initialize=0.5)
    m.y = pyo.Var(bounds=(-50.0, 50.0), initialize=1.0)
    m.link = pyo.Constraint(expr=m.y == 2 * m.x + 1)
    m.obj = pyo.Objective(expr=(m.x - m.p) ** 2 + 0.1 * (m.y - 1.0) ** 2)
    declare_sens_param(m.p)
    pyo.SolverFactory("pounce").solve(m, options={"tol": 1e-10})
    return m


def test_the_coupled_variable_follows_the_decided_side():
    """The QP's decision must propagate through the equality: on the
    releasing side y moves with x, on the holding side neither moves.
    Verified against re-solves."""
    m = coupled_kink()
    assert pyo.value(m.x) == pytest.approx(0.0, abs=1e-4)

    for target, mode in ((0.5, "fix_relax"), (0.5, "path"), (-0.5, "path")):
        exact = coupled_kink(target)
        est = estimate(m, [(m.p, target)], mode=mode, clamp=False)
        assert est[m.x] == pytest.approx(pyo.value(exact.x), abs=1e-4), (
            f"target {target}, mode {mode}")
        assert est[m.y] == pytest.approx(pyo.value(exact.y), abs=1e-4), (
            f"target {target}, mode {mode}")
