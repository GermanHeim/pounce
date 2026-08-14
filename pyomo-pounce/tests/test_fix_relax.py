"""Tests for estimate(mode="fix_relax"): pin a crossing variable at its
bound and re-solve, rather than clamping it and freezing the rest."""
import warnings

import pytest
import pyomo.environ as pyo

import pyomo_pounce  # noqa: F401  (registers 'pounce')
from pyomo_pounce import declare_sens_param, estimate, estimate_report


def linked(p=1.0):
    """x tracks p against a lower bound of 0, and y is tied to x by an
    equality. Clamping x leaves y at its predictor value, which breaks
    the equality outright, so the two modes are easy to tell apart."""
    m = pyo.ConcreteModel()
    m.p = pyo.Param(initialize=p, mutable=True)
    m.x = pyo.Var(bounds=(0.0, 10.0), initialize=1.0)
    m.y = pyo.Var(bounds=(-50.0, 50.0), initialize=1.0)
    m.link = pyo.Constraint(expr=m.y == 2 * m.x + 1)
    m.obj = pyo.Objective(
        expr=(m.x - m.p) ** 2 + 0.5 * (m.y - 3 * m.p) ** 2)
    declare_sens_param(m.p)
    return m


def solved(p=1.0):
    m = linked(p)
    pyo.SolverFactory("pounce").solve(m)
    return m


def resolve_at(newval):
    """The exact answer: solve again at the perturbed parameter."""
    m = linked(newval)
    pyo.SolverFactory("pounce").solve(m)
    return pyo.value(m.x), pyo.value(m.y)


def test_fix_relax_matches_a_resolve_where_the_clamp_does_not():
    m = solved()
    exact_x, exact_y = resolve_at(-2.0)

    with pytest.warns(UserWarning, match="leaves the variable bounds"):
        lin = estimate(m, [(m.p, -2.0)])
    fix = estimate(m, [(m.p, -2.0)], mode="fix_relax")

    # both put x on its bound
    assert lin[m.x] == pytest.approx(exact_x, abs=1e-6)
    assert fix[m.x] == pytest.approx(exact_x, abs=1e-6)

    # the clamp leaves y at its predictor value, breaking y = 2x + 1
    assert abs(lin[m.y] - exact_y) > 1.0
    assert abs(lin[m.y] - (2 * lin[m.x] + 1)) > 1.0

    # the pinned re-solve lands on the re-solve and keeps the equality
    assert fix[m.y] == pytest.approx(exact_y, abs=1e-6)
    assert fix[m.y] == pytest.approx(2 * fix[m.x] + 1, abs=1e-6)


def test_the_modes_agree_when_nothing_crosses():
    m = solved()
    lin = estimate(m, [(m.p, 1.1)])
    fix = estimate(m, [(m.p, 1.1)], mode="fix_relax")
    assert estimate_report(m, [(m.p, 1.1)]).alpha > 1.0
    for v in (m.x, m.y):
        assert fix[v] == pytest.approx(lin[v], rel=1e-12)


def test_fix_relax_does_not_warn_about_clamping():
    """The pin IS the correction, so there is nothing clipped and no
    active-set change left to warn about."""
    m = solved()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        estimate(m, [(m.p, -2.0)], mode="fix_relax")


def test_the_report_and_fix_relax_agree_on_what_crosses():
    m = solved()
    r = estimate_report(m, [(m.p, -2.0)])
    assert r.first == "x"
    assert 0.0 < r.alpha < 1.0
    assert m.x in r.crossed

    # the fraction the report says fits should cross nothing, and there
    # the two modes agree
    fits = 1.0 + r.alpha * (-2.0 - 1.0)
    lin = estimate(m, [(m.p, fits)])
    fix = estimate(m, [(m.p, fits)], mode="fix_relax")
    for v in (m.x, m.y):
        assert fix[v] == pytest.approx(lin[v], abs=1e-6)


def test_an_unknown_mode_is_refused():
    m = solved()
    with pytest.raises(ValueError, match="mode must be"):
        estimate(m, [(m.p, 2.0)], mode="path")


def test_fix_relax_reaches_the_same_answer_through_every_route():
    from pyomo.contrib.solver.common.factory import SolverFactory as SF2

    out = []
    for solve in (lambda mm: pyo.SolverFactory("pounce").solve(mm),
                  lambda mm: pyo.SolverFactory("pounce_v2").solve(mm),
                  lambda mm: SF2("pounce").solve(mm)):
        m = linked()
        solve(m)
        out.append(estimate(m, [(m.p, -2.0)], mode="fix_relax"))
    first, *rest = out
    for other in rest:
        vals_a = sorted(round(v, 9) for v in first.values())
        vals_b = sorted(round(v, 9) for v in other.values())
        assert vals_a == vals_b


def scaled_linked(p=1.0):
    m = linked(p)
    m.scaling_factor = pyo.Suffix(direction=pyo.Suffix.EXPORT)
    m.scaling_factor[m.x] = 1000.0
    m.scaling_factor[m.y] = 0.001
    return m


def test_fix_relax_holds_under_user_scaling():
    """The bounds the solve carries bound the algorithm's change of
    variables, while the iterate and the step are in the model's own
    units. Comparing them directly projects onto the wrong box: before
    this was handled, the scaled model returned x = -0.525 where the
    answer is 0."""
    m = scaled_linked()
    pyo.SolverFactory("pounce").solve(
        m, options={"nlp_scaling_method": "user-scaling"})
    exact_x, exact_y = resolve_at(-2.0)

    fix = estimate(m, [(m.p, -2.0)], mode="fix_relax")
    assert fix[m.x] == pytest.approx(exact_x, abs=1e-6)
    assert fix[m.y] == pytest.approx(exact_y, abs=1e-6)
    assert fix[m.y] == pytest.approx(2 * fix[m.x] + 1, abs=1e-6)


def test_scaling_does_not_change_the_answer():
    """The same model with and without a change of variables must give
    the same estimate."""
    plain = solved()
    m = scaled_linked()
    pyo.SolverFactory("pounce").solve(
        m, options={"nlp_scaling_method": "user-scaling"})

    a = estimate(plain, [(plain.p, -2.0)], mode="fix_relax")
    b = estimate(m, [(m.p, -2.0)], mode="fix_relax")
    assert b[m.x] == pytest.approx(a[plain.x], abs=1e-6)
    assert b[m.y] == pytest.approx(a[plain.y], abs=1e-6)


def two_bounded():
    """y = 2x + 1 with both bounded below by 0, so there is one degree
    of freedom and two bounds that a large downward move wants."""
    m = pyo.ConcreteModel()
    m.p = pyo.Param(initialize=1.0, mutable=True)
    m.x = pyo.Var(bounds=(0.0, 10.0), initialize=1.0)
    m.y = pyo.Var(bounds=(0.0, 10.0), initialize=3.0)
    m.link = pyo.Constraint(expr=m.y == 2 * m.x + 1)
    m.obj = pyo.Objective(
        expr=(m.x - m.p) ** 2 + 0.5 * (m.y - 3 * m.p) ** 2)
    declare_sens_param(m.p)
    pyo.SolverFactory("pounce").solve(m)
    return m


def test_fix_relax_says_which_limit_stopped_it():
    """Pinning y at 0 forces x to -0.5 through the equality, and the
    single degree of freedom is already spent, so no step holds both.
    The warning names that limit rather than the pass budget."""
    m = two_bounded()
    with pytest.warns(UserWarning, match="degrees of freedom"):
        estimate(m, [(m.p, -5.0)], mode="fix_relax")


def test_clamp_decides_what_happens_to_what_is_left_outside():
    """`clamp` means the same thing in both modes: clip whatever is
    still outside a bound. Under fix_relax that is only what the pins
    could not hold, and clipping it costs the constraint the pins were
    solved against, which is why the caller can turn it off."""
    m = two_bounded()
    with pytest.warns(UserWarning, match="clamped"):
        clipped = estimate(m, [(m.p, -5.0)], mode="fix_relax", clamp=True)
    with pytest.warns(UserWarning, match="unclamped"):
        raw = estimate(m, [(m.p, -5.0)], mode="fix_relax", clamp=False)

    # clamped: inside the bounds, and the equality no longer holds
    assert clipped[m.x] == pytest.approx(0.0, abs=1e-9)
    assert abs(clipped[m.y] - (2 * clipped[m.x] + 1)) > 0.5

    # unclamped: outside a bound, and the equality still holds
    assert raw[m.x] < -1e-3
    assert raw[m.y] == pytest.approx(2 * raw[m.x] + 1, abs=1e-6)


def test_max_passes_is_exposed_and_bounds_the_work():
    """The cap is a budget, so a caller has to be able to set it."""
    m = solved()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        full = estimate(m, [(m.p, -2.0)], mode="fix_relax", max_passes=16)
    # one pass is enough for this model, so capping at 1 changes nothing
    one = estimate(m, [(m.p, -2.0)], mode="fix_relax", max_passes=1)
    assert one[m.x] == pytest.approx(full[m.x], rel=1e-12)

    # and a cap of zero pins nothing, leaving the plain step
    with pytest.warns(UserWarning):
        none = estimate(m, [(m.p, -2.0)], mode="fix_relax", max_passes=0,
                        clamp=False)
    assert none[m.x] < -1e-3


def test_fix_relax_is_quiet_when_it_holds_every_bound():
    m = solved()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        estimate(m, [(m.p, -2.0)], mode="fix_relax")


def test_an_absent_bound_does_not_widen_the_tolerance():
    """A variable with no upper bound reaches the check as the reader's
    +-1e19 sentinel. Scaling the tolerance by the bound rather than by
    the coordinate puts it at 1e10, and nothing is ever reported as
    outside its bounds."""
    m = pyo.ConcreteModel()
    m.p = pyo.Param(initialize=1.0, mutable=True)
    m.x = pyo.Var(bounds=(0.0, None), initialize=1.0)   # no upper bound
    m.y = pyo.Var(bounds=(-50.0, 50.0), initialize=1.0)
    m.link = pyo.Constraint(expr=m.y == 2 * m.x + 1)
    m.obj = pyo.Objective(
        expr=(m.x - m.p) ** 2 + 0.5 * (m.y - 3 * m.p) ** 2)
    declare_sens_param(m.p)
    pyo.SolverFactory("pounce").solve(m)

    # the step drives x below 0, which must still be caught
    with pytest.warns(UserWarning, match="leaves the variable bounds"):
        estimate(m, [(m.p, -2.0)])
    fix = estimate(m, [(m.p, -2.0)], mode="fix_relax")
    assert fix[m.x] == pytest.approx(0.0, abs=1e-6)
    assert fix[m.y] == pytest.approx(2 * fix[m.x] + 1, abs=1e-6)
