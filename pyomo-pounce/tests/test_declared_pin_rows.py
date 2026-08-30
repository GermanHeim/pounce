# -*- coding: utf-8 -*-
"""A declared sens param's defining equality is pinned as written.

A model whose declared params each enter through one defining equality
solves as written: no clone, no surgery, no interface construction.
Params without that form are rewritten in place once, at declaration,
with a warning. These tests count interface constructions through a
monkeypatch, so a path that quietly resurrects the per-solve clone
fails loudly.
"""
import warnings

import pytest

import pyomo.environ as pyo

import pyomo_pounce.sens as sens_mod
from pyomo_pounce import declare_sens_param, estimate, gradient


@pytest.fixture
def si_counter(monkeypatch):
    """Count SensitivityInterface constructions inside pyomo_pounce."""
    real = sens_mod.SensitivityInterface
    calls = []

    class Counting(real):
        def __init__(self, *a, **k):
            calls.append(k.get("clone_model", a[1] if len(a) > 1 else True))
            super().__init__(*a, **k)

    Counting.get_default_block_name = real.get_default_block_name
    monkeypatch.setattr(sens_mod, "SensitivityInterface", Counting)
    return calls


def conforming_model(p0=1.0):
    """x pinned to p by its defining equality; at the optimum
    y = (20/11) x, so dx/dp = 1 and dy/dp = 20/11."""
    m = pyo.ConcreteModel()
    m.p = pyo.Param(initialize=p0, mutable=True)
    m.x = pyo.Var(initialize=p0)
    m.y = pyo.Var(initialize=0.0)
    m.pin = pyo.Constraint(expr=m.x == m.p)
    m.obj = pyo.Objective(expr=(m.y - 2 * m.x) ** 2 + 0.1 * m.y ** 2)
    return m


def test_a_conforming_model_solves_without_the_toolbox(si_counter):
    m = conforming_model()
    declare_sens_param(m.p)
    pyo.SolverFactory("pounce").solve(m)
    assert si_counter == [], "a conforming model must not be cloned"
    g = gradient(m.x, wrt=m.p)
    assert g == pytest.approx(1.0, abs=1e-6)
    est = estimate(m, [(m.p, 1.5)])
    assert est[m.x] == pytest.approx(1.5, abs=1e-6)
    assert est[m.y] == pytest.approx(1.5 * 20.0 / 11.0, abs=1e-5)


def test_repeated_solves_construct_no_interface(si_counter):
    m = conforming_model()
    declare_sens_param(m.p)
    opt = pyo.SolverFactory("pounce")
    for val in (1.0, 2.0, 0.5):
        m.p.set_value(val)
        opt.solve(m)
        assert pyo.value(m.x) == pytest.approx(val, abs=1e-6)
        est = estimate(m, [(m.p, val + 0.1)])
        assert est[m.x] == pytest.approx(val + 0.1, abs=1e-6)
    assert si_counter == []


def test_orientation_and_offset_forms_conform(si_counter):
    # p on the left, and the equality carrying a folded offset: both
    # are defining equalities, with the coefficient carrying the sign.
    m = pyo.ConcreteModel()
    m.p = pyo.Param(initialize=1.0, mutable=True)
    m.q = pyo.Param(initialize=0.5, mutable=True)
    m.x = pyo.Var(initialize=1.0)
    m.w = pyo.Var(initialize=1.5)
    m.y = pyo.Var(initialize=0.0)
    m.pin_x = pyo.Constraint(expr=m.p == m.x)
    m.pin_w = pyo.Constraint(expr=m.w - m.q == 1.0)
    m.obj = pyo.Objective(expr=(m.y - m.x - m.w) ** 2 + m.y ** 2 * 0.0
                          + (m.y - m.x - m.w) ** 2)
    declare_sens_param(m.p, m.q)
    pyo.SolverFactory("pounce").solve(m)
    assert si_counter == []
    assert gradient(m.x, wrt=m.p) == pytest.approx(1.0, abs=1e-6)
    assert gradient(m.w, wrt=m.q) == pytest.approx(1.0, abs=1e-6)
    est = estimate(m, [(m.p, 1.2), (m.q, 0.9)])
    assert est[m.x] == pytest.approx(1.2, abs=1e-6)
    assert est[m.w] == pytest.approx(1.9, abs=1e-6)
    assert est[m.y] == pytest.approx((1.2 + 1.9), abs=1e-5)


def test_a_folded_param_is_rewritten_in_place_with_a_warning(si_counter):
    m = pyo.ConcreteModel()
    m.p = pyo.Param(initialize=1.0, mutable=True)
    m.x = pyo.Var(initialize=1.0)
    m.y = pyo.Var(initialize=1.0)
    # p folded into two rows: no defining equality to pin
    m.c1 = pyo.Constraint(expr=m.x + m.p * m.y == 2.0)
    m.c2 = pyo.Constraint(expr=m.y - m.p == 0.5)
    m.obj = pyo.Objective(expr=m.x ** 2 + m.y ** 2)
    with pytest.warns(UserWarning, match="rewritten in place"):
        declare_sens_param(m.p)
    assert si_counter == [], "the rewrite is native, not the toolbox"
    assert m.component(sens_mod._DEFS) is not None
    assert m.c1.active and m.c2.active, "rows are edited, not replaced"
    pyo.SolverFactory("pounce").solve(m)
    assert si_counter == [], "the solve must not clone"
    # exact: x = 2 - p(p + 0.5), y = p + 0.5, dx/dp = -2p - 0.5
    g = gradient(m.x, wrt=m.p)
    assert g == pytest.approx(-2.5, abs=1e-5)


def test_sequential_rewrites_each_get_their_block():
    m = pyo.ConcreteModel()
    m.p = pyo.Param(initialize=1.0, mutable=True)
    m.q = pyo.Param(initialize=1.0, mutable=True)
    m.x = pyo.Var(initialize=1.0)
    m.c1 = pyo.Constraint(expr=m.x * m.p + m.x == 2.0)
    m.c2 = pyo.Constraint(expr=m.x * m.p - m.x <= 2.0)
    m.obj = pyo.Objective(expr=(m.x - m.q) ** 2 + m.q * m.x)
    with pytest.warns(UserWarning, match="rewritten in place"):
        declare_sens_param(m.p)
    with pytest.warns(UserWarning, match="rewritten in place"):
        declare_sens_param(m.q)
    pyo.SolverFactory("pounce").solve(m)
    # x(1 + p) = 2 and the stationarity of (x - q)^2 + q x in x is
    # decoupled from q at fixed x, so dx/dp = -2 / (1 + p)^2
    assert gradient(m.x, wrt=m.p) == pytest.approx(-0.5, abs=1e-5)


def test_a_param_in_the_objective_takes_the_rewrite(si_counter):
    m = pyo.ConcreteModel()
    m.p = pyo.Param(initialize=1.0, mutable=True)
    m.x = pyo.Var(initialize=1.0)
    m.obj = pyo.Objective(expr=(m.x - m.p) ** 2)
    with pytest.warns(UserWarning, match="rewritten in place"):
        declare_sens_param(m.p)
    pyo.SolverFactory("pounce").solve(m)
    assert gradient(m.x, wrt=m.p) == pytest.approx(1.0, abs=1e-5)


def test_editing_the_defining_row_after_declaration_raises():
    m = conforming_model()
    declare_sens_param(m.p)
    m.pin.deactivate()
    m.x.fix(1.0)
    with pytest.raises(RuntimeError, match="defining equality"):
        pyo.SolverFactory("pounce").solve(m)


def test_a_declared_fixed_var_tracks_re_solves(si_counter):
    # the reviewer's reproduction on #861: the pin must read the Var's
    # current value on every solve, through set_value and re-fix alike
    m = pyo.ConcreteModel()
    m.u = pyo.Var(initialize=2.0)
    m.u.fix(2.0)
    m.x = pyo.Var(initialize=0.0)
    m.c = pyo.Constraint(expr=m.x == 3.0 * m.u)
    m.obj = pyo.Objective(expr=(m.x - 1.0) ** 2)
    with pytest.warns(UserWarning, match="rewritten in place"):
        declare_sens_param(m.u)
    opt = pyo.SolverFactory("pounce")
    opt.solve(m)
    assert pyo.value(m.x) == pytest.approx(6.0, abs=1e-6)
    assert gradient(m.x, wrt=m.u) == pytest.approx(3.0, abs=1e-6)
    m.u.set_value(5.0)
    opt.solve(m)
    assert pyo.value(m.x) == pytest.approx(15.0, abs=1e-6)
    m.u.fix(7.0)
    opt.solve(m)
    assert pyo.value(m.x) == pytest.approx(21.0, abs=1e-6)
    assert gradient(m.x, wrt=m.u) == pytest.approx(3.0, abs=1e-6)
    assert si_counter == []


def test_a_raising_declaration_leaves_the_model_untouched():
    # the reviewer's reproduction on #861: validation runs before any
    # rewrite, so an illegal component in the call changes nothing
    m = pyo.ConcreteModel()
    m.p = pyo.Param(initialize=1.0, mutable=True)
    m.x = pyo.Var(initialize=1.0)
    m.z = pyo.Var(initialize=0.0)
    m.c1 = pyo.Constraint(expr=m.x * m.p + m.x == 2.0)
    m.obj = pyo.Objective(expr=(m.x - 1) ** 2 + m.z ** 2)
    with pytest.raises(ValueError, match="not fixed"):
        declare_sens_param(m.p, m.z)
    assert m.component(sens_mod._DEFS) is None
    assert sens_mod._registry(m).params == []
    assert sens_mod._registry(m).pin_records == []
    with pytest.warns(UserWarning, match="rewritten in place"):
        declare_sens_param(m.p)
    pyo.SolverFactory("pounce").solve(m)
    assert gradient(m.x, wrt=m.p) == pytest.approx(-0.5, abs=1e-5)


def test_indexed_conforming_params_solve_as_written(si_counter):
    m = pyo.ConcreteModel()
    m.p = pyo.Param(range(3), initialize=1.0, mutable=True)
    m.x = pyo.Var(range(3), initialize=1.0)
    m.y = pyo.Var(initialize=0.0)

    @m.Constraint(range(3))
    def pin(m, i):
        return m.x[i] == m.p[i]

    m.obj = pyo.Objective(
        expr=sum((m.y - m.x[i]) ** 2 for i in range(3)))
    declare_sens_param(m.p)
    pyo.SolverFactory("pounce").solve(m)
    assert si_counter == []
    assert m.component(sens_mod._DEFS) is None
    for i in range(3):
        assert gradient(m.x[i], wrt=m.p[i]) == pytest.approx(1.0, abs=1e-6)
    assert gradient(m.x[0], wrt=m.p[1]) == pytest.approx(0.0, abs=1e-8)
    est = estimate(m, [(m.p, {0: 1.3, 1: 0.7, 2: 1.0})])
    assert est[m.x[0]] == pytest.approx(1.3, abs=1e-6)
    assert est[m.x[1]] == pytest.approx(0.7, abs=1e-6)


def test_a_bound_only_param_is_rewritten_without_the_warning():
    # `bounds=(None, m.hi)` is the book's own endorsed spelling: the
    # move into a constraint is its documented mechanics, so it stays
    # quiet, and only genuinely folded params warn
    m = pyo.ConcreteModel()
    m.hi = pyo.Param(initialize=2.0, mutable=True)
    m.x = pyo.Var(initialize=1.0, bounds=(None, m.hi))
    m.obj = pyo.Objective(expr=-m.x)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        declare_sens_param(m.hi)
    assert not [w for w in caught if "rewritten" in str(w.message)]
    assert m.component(sens_mod._DEFS) is not None
    pyo.SolverFactory("pounce").solve(m)
    assert pyo.value(m.x) == pytest.approx(2.0, abs=1e-6)
    assert gradient(m.x, wrt=m.hi) == pytest.approx(1.0, abs=1e-5)


def test_declared_and_call_time_params_share_a_solve(si_counter):
    m = conforming_model()
    m.q = pyo.Param(initialize=0.5, mutable=True)
    m.w = pyo.Var(initialize=0.5)
    m.cq = pyo.Constraint(expr=m.w == 2.0 * m.q)
    declare_sens_param(m.p)
    res = sens_mod.sens_solve(m, sens_params=[m.q])
    assert str(res.solver.termination_condition) == "optimal"
    assert si_counter == [True], "the call-time param forces the clone"
    est = estimate(m, [(m.p, 1.4), (m.q, 1.0)])
    assert est[m.x] == pytest.approx(1.4, abs=1e-6)
    assert est[m.w] == pytest.approx(2.0, abs=1e-6)


def test_call_time_sens_params_still_clone(si_counter):
    m = conforming_model()
    res = sens_mod.sens_solve(m, sens_params=[m.p])
    assert str(res.solver.termination_condition) == "optimal"
    assert si_counter == [True], "call-time params keep the clone"
    assert m.component(
        sens_mod.SensitivityInterface.get_default_block_name()) is None
    est = estimate(m, [(m.p, 1.5)])
    assert est[m.x] == pytest.approx(1.5, abs=1e-6)
