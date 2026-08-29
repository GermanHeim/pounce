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


def test_the_record_shows_the_kink_resolving_essentially_at_zero():
    """The departure is the walk's own release, at the fraction where
    the residual multiplier the solve left reaches zero, tiny but not
    stamped 0.0."""
    m = kink()
    rec = active_set_changes(m, [(m.p, 1.0)])
    assert [(c.var, c.bound, c.action) for c in rec] == [
        (m.x, "lower", "leaves")], f"record: {rec}"
    assert rec[0].fraction == pytest.approx(0.0, abs=1e-3)
    assert active_set_changes(m, [(m.p, -1.0)]) == [], (
        "held through the whole change, nothing to record")


def test_a_bound_inside_the_band_releases_where_its_multiplier_ends():
    """Held slightly on the active side of the kink, the bound is
    genuinely active with a small positive multiplier, and the true
    solution releases it partway through the step, where that
    multiplier reaches zero, not at the start. Deciding it at fraction
    zero instead released it early and overshot tenfold on the CSTR
    held at 75% of the breakpoint fraction, which is the defect this
    test pins."""
    eps = 1e-5
    m = pyo.ConcreteModel()
    m.p = pyo.Param(initialize=-eps, mutable=True)
    m.x = pyo.Var(bounds=(0.0, 10.0), initialize=0.5)
    m.obj = pyo.Objective(expr=(m.x - m.p) ** 2)
    declare_sens_param(m.p)
    pyo.SolverFactory("pounce").solve(m, options={"tol": 1e-8})

    # in the ambiguous band, which the gradient warning certifies
    with pytest.warns(UserWarning, match="degenerate"):
        gradient(m.x, wrt=m.p)

    rec = active_set_changes(m, [(m.p, 1.0)])
    assert [(c.var, c.bound, c.action) for c in rec] == [
        (m.x, "lower", "leaves")], f"record: {rec}"
    # The fraction is the zero crossing of the multiplier the solve
    # left, which inside the band is of order sqrt(mu) rather than
    # eps, since the residual barrier multiplier dominates the true
    # one there. What discriminates the defect is that it is strictly
    # positive: a decision stamped at fraction zero is the early
    # release this test pins.
    frac = rec[0].fraction
    assert 1e-6 < frac < 1e-2, f"release at {frac}"

    est = estimate(m, [(m.p, 1.0)], mode="path")
    assert est[m.x] == pytest.approx(1.0, abs=1e-4)


def test_a_fixed_variable_does_not_shift_the_detection():
    """The classifier reports per user variable while the factor's
    rows skip fixed variables, and the two index spaces diverge from
    the first fixed variable on. With a variable fixed by equal bounds
    sitting ahead of the kink, an unmapped index pins or releases the
    kink variable's NEIGHBOR, a plausible wrong answer, which is the
    gh#450 hazard. This model has one fixed column, one kink variable,
    and one spectator behind it."""
    m = pyo.ConcreteModel()
    m.p = pyo.Param(initialize=0.0, mutable=True)
    m.f = pyo.Var(bounds=(2.0, 2.0), initialize=2.0)
    m.x = pyo.Var(bounds=(0.0, 10.0), initialize=0.5)
    m.y = pyo.Var(bounds=(-50.0, 50.0), initialize=1.0)
    # the fixed term comes first in the expression so the NL writer
    # orders the fixed column ahead of the kink variable, which is the
    # arrangement that makes the two index spaces diverge in front of x
    m.obj = pyo.Objective(
        expr=0.1 * (m.f - 1.0) ** 3 + (m.x - m.p) ** 2
        + 0.1 * (m.y - 1.0) ** 2)
    declare_sens_param(m.p)
    pyo.SolverFactory("pounce").solve(m, options={"tol": 1e-10})
    assert pyo.value(m.x) == pytest.approx(0.0, abs=1e-4), "on the bound"

    # the fixture only guards the hazard if the two spaces diverge
    from pyomo_pounce.sens import _REG
    sess = m.__dict__[_REG].session
    assert sess.solver.block_dims[0] < len(sess.var_names), (
        "the fixed column must be out of the factor")

    with pytest.warns(UserWarning, match=r"x \(lower\)"):
        gradient(m.x, wrt=m.p)

    for mode in MODES:
        up = estimate(m, [(m.p, 1.0)], mode=mode)
        down = estimate(m, [(m.p, -1.0)], mode=mode, clamp=False)
        assert up[m.x] == pytest.approx(1.0, abs=1e-4), f"mode={mode}"
        assert down[m.x] == pytest.approx(0.0, abs=1e-4), f"mode={mode}"
        assert up[m.y] == pytest.approx(1.0, abs=1e-4), (
            f"mode={mode}: the spectator must not be touched")
        assert down[m.y] == pytest.approx(1.0, abs=1e-4), (
            f"mode={mode}: the spectator must not be touched")


def test_the_decision_is_invariant_to_the_perturbation_scale():
    """The acceptance test is relative to the direction's own norm, so
    the working set decided at a perturbation of 1e-10 is the same one
    decided at 1. An absolute tolerance accepted the all-released set
    on the holding side at tiny perturbations, reading the derivative
    as -1 instead of 0, and the barrier correction on a weak row is
    constant-sized, so before it was zeroed it left an offset of
    order sqrt(mu) that dominated tiny steps.

    The holding side is exact in every mode. On the releasing side the
    path's answer is bounded by the step rather than pinned to it,
    because the release fraction is a quotient of solve quantities and
    at a weak bound the multiplier's uncertainty equals its magnitude,
    so a step below that precision cannot be resolved against it."""
    m = kink()
    base_val = pyo.value(m.x)
    for mode in MODES:
        tiny_down = estimate(m, [(m.p, -1e-10)], mode=mode, clamp=False)
        assert tiny_down[m.x] - base_val == pytest.approx(0.0, abs=1e-12), (
            f"mode={mode}: the holding side holds at any scale")
        tiny_up = estimate(m, [(m.p, 1e-10)], mode=mode, clamp=False)
        moved = tiny_up[m.x] - base_val
        if mode == "path":
            assert -1e-12 <= moved <= 1e-10 + 1e-12, (
                f"mode={mode}: bounded by the step, got {moved}")
        else:
            assert moved == pytest.approx(1e-10, rel=1e-3), (
                f"mode={mode}: the releasing side releases at any scale")


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
    """degeneracy_iter=0 leaves the decision no back-solves, so it
    fails and the estimate falls back to the one-sided step and says
    so."""
    m = kink()
    with pytest.warns(UserWarning, match="one-sided step"):
        fell = estimate(m, [(m.p, -1.0)], degeneracy_iter=0, clamp=False)
    plain = estimate(m, [(m.p, -1.0)], degeneracy="one_sided", clamp=False)
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


def test_release_all_takes_the_released_direction_undecided():
    """degeneracy="release_all" hands back the all-released step with
    no decision: on the kink's holding side the raw step follows the
    released direction to -1, where the true derivative is 0, and the
    repair belongs to whatever runs next. Deterministic, unlike
    "one_sided", whose holding-side answer depends on which side the
    held factorization leans toward."""
    m = kink()
    down = estimate(m, [(m.p, -1.0)], degeneracy="release_all",
                    clamp=False)
    assert down[m.x] == pytest.approx(-1.0, abs=1e-4), (
        "the released direction follows p through the bound")


def test_release_all_is_repaired_by_every_mode():
    """The releasing side is right in every mode, and the holding
    side's violation is repaired by each mode's own machinery: pins
    under fix_relax, the walk under path, and the clamp under linear.
    On this one-variable fixture the clamp repair is exact; the
    neighbor coupling a clamp cannot repair needs a coupled fixture
    and is documented on estimate()."""
    m = kink()
    for mode in ("linear", "fix_relax", "path"):
        up = estimate(m, [(m.p, 1.0)], mode=mode,
                      degeneracy="release_all")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            down = estimate(m, [(m.p, -1.0)], mode=mode,
                            degeneracy="release_all")
        assert up[m.x] == pytest.approx(1.0, abs=1e-4), (
            f"mode={mode}: the releasing side's derivative is 1")
        assert down[m.x] == pytest.approx(0.0, abs=1e-4), (
            f"mode={mode}: the holding side repairs to the bound")


def test_release_all_at_a_clean_base_point_is_the_plain_step():
    """Without a weakly active bound there is nothing to release and
    every degeneracy option takes the same step."""
    m = kink(p=2.0)
    assert pyo.value(m.x) == pytest.approx(2.0, abs=1e-6), "interior"
    a = estimate(m, [(m.p, 3.0)], degeneracy="release_all")
    b = estimate(m, [(m.p, 3.0)], degeneracy="one_sided")
    assert a[m.x] == pytest.approx(b[m.x], abs=1e-12)
