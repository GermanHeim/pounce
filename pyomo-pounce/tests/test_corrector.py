"""Tests for estimate(corrector_iter=...): Newton iterations on the
barrier system after the step, against the factorization the solve left
behind."""
import warnings

import numpy as np
import pytest
import pyomo.environ as pyo

import pyomo_pounce  # noqa: F401  (registers 'pounce')
from pyomo_pounce import declare_sens_param, estimate
from pyomo_pounce.sens import (_correct, _perturbation_deltas,
                               _session_for)


def curved(p=1.0):
    """A model whose solution bends with the parameter, so the linear
    step is wrong by an amount the corrector can work on.

    The objective's coupling makes x(p) genuinely nonlinear while no
    bound is anywhere near active, which is the regime the corrector is
    for: the active set the base point settled still fits.
    """
    m = pyo.ConcreteModel()
    m.p = pyo.Param(initialize=p, mutable=True)
    m.x = pyo.Var(bounds=(-10.0, 10.0), initialize=1.0)
    m.y = pyo.Var(bounds=(-10.0, 10.0), initialize=1.0)
    m.link = pyo.Constraint(expr=m.y == pyo.exp(m.x / 4) + m.p)
    m.obj = pyo.Objective(expr=(m.x - m.p) ** 2 + (m.y - 2 * m.p) ** 4)
    declare_sens_param(m.p)
    return m


def held(p=-1.0):
    """A model whose base solve holds `x` on its lower bound.

    At `p = -1` the objective wants `x` at -0.5, so the bound takes the
    load and its multiplier is a real number rather than zero. That is
    the barrier diagonal the held factorization carries, and a
    perturbation that wants `x` off the bound is the case the corrector
    cannot reach by iterating alone.
    """
    m = pyo.ConcreteModel()
    m.p = pyo.Param(initialize=p, mutable=True)
    m.x = pyo.Var(bounds=(0.0, 10.0), initialize=0.5)
    m.y = pyo.Var(bounds=(-10.0, 10.0), initialize=0.0)
    m.link = pyo.Constraint(expr=m.y == m.x + m.p)
    m.obj = pyo.Objective(expr=(m.x - m.p) ** 2 + (m.y - m.p) ** 2)
    declare_sens_param(m.p)
    return m


def pinning():
    """A model whose base solve leaves `x` interior and whose step
    carries it onto the lower bound."""
    m = pyo.ConcreteModel()
    m.p = pyo.Param(initialize=1.0, mutable=True)
    m.x = pyo.Var(bounds=(0.0, 10.0), initialize=1.0)
    m.obj = pyo.Objective(expr=(m.x - m.p) ** 2)
    declare_sens_param(m.p)
    pyo.SolverFactory("pounce").solve(m)
    return m


def corrector_of(m, perturb, corrector_iter=8):
    """What the corrector did, from the report `estimate_report` builds."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rep = pyomo_pounce.estimate_report(
            m, perturb, corrector_iter=corrector_iter)
    return rep.corrector


def mode_corrector(m, newval, mode, corrector_iter=8):
    """The same, under a chosen mode.

    `estimate_report` reports the linear step, so a correction under
    another mode is reached through the same private path `estimate`
    itself uses.
    """
    session = _session_for(m)
    pin, deltas = _perturbation_deltas(session, [(m.p, newval)])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if mode == "linear":
            step = session.solver.parametric_step(list(pin), list(deltas))
        elif mode == "fix_relax":
            step, _ = session.solver.parametric_step_bounded(
                list(pin), list(deltas), 16)
        else:
            step, _ = session.solver.parametric_step_path(
                list(pin), list(deltas), 16)
        _, info = _correct(session, list(pin), list(deltas),
                           np.asarray(step), mode, "one_sided",
                           corrector_iter, False)
    return info


def solve_it(m):
    """Solve a model in place and return where `x` landed."""
    pyo.SolverFactory("pounce").solve(m)
    return pyo.value(m.x)


def resolve_held(newval):
    """The exact answer for `held`: solve again at the new value."""
    m = held(newval)
    pyo.SolverFactory("pounce").solve(m)
    return pyo.value(m.x), pyo.value(m.y)


def solved(p=1.0):
    m = curved(p)
    pyo.SolverFactory("pounce").solve(m)
    return m


def resolve_at(newval):
    """The exact answer: solve again at the perturbed parameter."""
    m = curved(newval)
    pyo.SolverFactory("pounce").solve(m)
    return pyo.value(m.x), pyo.value(m.y)


def estimated(m, newval, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        est = estimate(m, [(m.p, newval)], **kw)
    return est[m.x], est[m.y]


def test_the_corrector_beats_the_linear_step_where_the_solution_bends():
    m = solved()
    target = 1.6
    tx, ty = resolve_at(target)
    lx, ly = estimated(m, target)
    cx, cy = estimated(m, target, corrector_iter=8)
    lin = max(abs(lx - tx), abs(ly - ty))
    cor = max(abs(cx - tx), abs(cy - ty))
    assert lin > 1e-6, f"this step should be visibly wrong, off by {lin}"
    assert cor < lin / 10, (
        f"the corrector should beat the linear step by an order: "
        f"{lin:.3e} -> {cor:.3e}")


def test_a_zero_budget_is_the_uncorrected_step():
    m = solved()
    target = 1.6
    a = estimated(m, target)
    b = estimated(m, target, corrector_iter=0)
    assert a == pytest.approx(b, abs=1e-12)


def test_a_larger_budget_buys_accuracy():
    """Each iteration is a Newton step against the held factorization,
    so on a model whose active set does not change the error falls
    steadily with the budget rather than levelling off after one."""
    m = solved()
    target = 1.6
    tx, ty = resolve_at(target)
    budgets = (1, 2, 4, 12)
    errs = []
    for it in budgets:
        cx, cy = estimated(m, target, corrector_iter=it)
        errs.append(max(abs(cx - tx), abs(cy - ty)))
        c = corrector_of(m, [(m.p, target)], corrector_iter=it)
        assert c["iterations"] == it, (
            f"a budget of {it} should be spent in full here, "
            f"spent {c['iterations']}")
    assert all(b <= a for a, b in zip(errs, errs[1:])), (
        f"a larger budget should never lose accuracy: {errs}")
    assert errs[-1] < errs[0] * 1e-6, (
        f"the budget should buy orders, not a rounding: {errs}")


def test_it_applies_under_every_mode():
    """Each mode produces a step and the corrector refines whichever
    one ran, so asking for iterations under fix_relax or path must
    improve on that mode's own answer rather than being ignored.

    No bound changes status on this model, so the three modes agree
    here and this says nothing about which one ran. Where they
    disagree is covered by
    `test_the_modes_disagree_when_a_held_bound_must_be_released`.
    """
    m = solved()
    target = 1.6
    tx, ty = resolve_at(target)
    for mode in ("linear", "fix_relax", "path"):
        px, py = estimated(m, target, mode=mode)
        cx, cy = estimated(m, target, mode=mode, corrector_iter=8)
        plain = max(abs(px - tx), abs(py - ty))
        corr = max(abs(cx - tx), abs(cy - ty))
        assert corr < plain / 10, (
            f"mode={mode}: the corrector should improve on the mode's own "
            f"step, {plain:.3e} -> {corr:.3e}")


def test_a_correction_that_achieves_nothing_says_so():
    """A bound that has to leave the active set is what the held
    factorization cannot represent, and the caller has to be told
    rather than handed the uncorrected step as though it were
    corrected."""
    m = pyo.ConcreteModel()
    m.p = pyo.Param(initialize=1.0, mutable=True)
    m.x = pyo.Var(bounds=(0.0, 10.0), initialize=1.0)
    m.obj = pyo.Objective(expr=(m.x - m.p) ** 2)
    declare_sens_param(m.p)
    pyo.SolverFactory("pounce").solve(m)
    # p goes negative, so the solution wants x below its lower bound
    # and the bound the base point held has to take the load instead
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        corrected = estimate(m, [(m.p, -4.0)], corrector_iter=6)[m.x]
        msgs = [str(x.message) for x in w]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plain = estimate(m, [(m.p, -4.0)])[m.x]
    assert any("corrector spent" in x for x in msgs), (
        f"the corrector achieved nothing and should say so: {msgs}")
    assert corrected == pytest.approx(plain, abs=1e-12), (
        "a correction that achieves nothing should leave the estimate "
        f"where it was: {plain} -> {corrected}")


def test_estimate_report_carries_what_the_corrector_did():
    m = solved()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plain = pyomo_pounce.estimate_report(m, [(m.p, 1.6)])
        rep = pyomo_pounce.estimate_report(m, [(m.p, 1.6)], corrector_iter=8)
    assert plain.corrector is None
    c = rep.corrector
    assert c is not None
    assert 1 <= c["iterations"] <= 8
    assert c["residual"] <= c["initial_residual"]
    for key in ("stationarity", "feasibility", "complementarity",
                "active_set_changes", "released", "pinned", "converged"):
        assert key in c, f"{key} missing from {sorted(c)}"
    # the reported residual is the largest of the three blocks, and the
    # two counts are what `active_set_changes` totals
    assert c["residual"] == pytest.approx(
        max(c["stationarity"], c["feasibility"], c["complementarity"]),
        rel=1e-9), (
        f"the residual should be the largest block: {c}")
    assert c["released"] + c["pinned"] == c["active_set_changes"]
    assert all(c[k] >= 0.0 for k in
               ("stationarity", "feasibility", "complementarity"))
    # the rest of the report describes the step handed over, unchanged
    assert rep.alpha == pytest.approx(plain.alpha)
    assert rep.violation == pytest.approx(plain.violation)


def test_it_releases_a_bound_the_solve_held():
    """`fix_relax` decides that a held bound leaves the active set, and
    the corrector takes that bound out of the operator before
    iterating. Without that the held barrier diagonal keeps the
    variable where the base point put it."""
    m = held()
    base = solve_it(m)
    tx, ty = resolve_held(2.0)
    assert base < 1e-6, f"the base solve should sit on the bound, at {base}"
    assert tx > 0.5, f"the perturbed solution should be off it, at {tx}"
    cx, cy = estimated(m, 2.0, mode="fix_relax", corrector_iter=8)
    c = mode_corrector(m, 2.0, "fix_relax", 8)
    assert (c["released"], c["pinned"]) == (1, 0), (
        f"one bound should leave the active set and none join it: {c}")
    assert c["residual"] < c["initial_residual"] * 1e-6, (
        f"releasing the bound should let the iterations converge: "
        f"{c['initial_residual']:.3e} -> {c['residual']:.3e}")
    assert max(abs(cx - tx), abs(cy - ty)) < 1e-6, (
        f"corrected to ({cx}, {cy}), the re-solve is at ({tx}, {ty})")


def test_it_pins_a_bound_the_step_reaches():
    """The other direction. The solve left `x` interior, so the held
    factorization treats it as free, and the step carries it onto the
    bound. The corrector raises that bound's barrier diagonal to what
    the barrier assigns at the slack the step ends at."""
    m = pinning()
    c = corrector_of(m, [(m.p, -4.0)], corrector_iter=6)
    assert (c["released"], c["pinned"]) == (0, 1), (
        f"one bound should join the active set and none leave it: {c}")
    assert c["active_set_changes"] == 1


def test_the_modes_disagree_when_a_held_bound_must_be_released():
    """`mode="linear"` holds the active set fixed as it builds the step,
    so its endpoint keeps `x` on the bound and there is no release for
    the corrector to apply. The barrier term the base point carries
    holds `x` down, and no number of iterations moves it. `fix_relax`
    and `path` decide the release themselves and converge."""
    m = held()
    solve_it(m)
    tx, _ = resolve_held(2.0)
    lin = mode_corrector(m, 2.0, "linear", 8)
    assert lin["released"] == 0, (
        f"the linear step's endpoint shows no release: {lin}")
    assert lin["residual"] > lin["initial_residual"] * 0.99, (
        f"with the bound still in the operator there is nothing to gain: "
        f"{lin['initial_residual']:.3e} -> {lin['residual']:.3e}")
    lx, _ = estimated(m, 2.0, mode="linear", corrector_iter=8)
    assert abs(lx - tx) > 0.5, (
        f"the linear estimate should stay on the bound, at {lx}")
    for mode in ("fix_relax", "path"):
        c = mode_corrector(m, 2.0, mode, 8)
        assert c["released"] == 1, f"mode={mode} should release: {c}"
        assert c["residual"] < c["initial_residual"] * 1e-6, (
            f"mode={mode} should converge: {c['initial_residual']:.3e} -> "
            f"{c['residual']:.3e}")
        cx, _ = estimated(m, 2.0, mode=mode, corrector_iter=8)
        assert abs(cx - tx) < 1e-6, f"mode={mode} corrected to {cx}, want {tx}"


def test_a_correction_leaves_the_session_able_to_answer_again():
    """The corrector sets a trial point rather than moving the iterate,
    so the held factorization and every later estimate are unaffected.
    A step taken after a correction must match one taken before it."""
    m = solved()
    before = estimated(m, 1.6)
    estimated(m, 1.6, corrector_iter=8)
    estimated(m, 1.6, mode="fix_relax", corrector_iter=12)
    after = estimated(m, 1.6)
    assert after == pytest.approx(before, abs=1e-12), (
        f"the session answered differently after a correction: "
        f"{before} -> {after}")
