"""Tests for estimate(corrector_iter=...): Newton iterations on the
barrier system after the step, against the factorization the solve left
behind."""
import warnings

import pytest
import pyomo.environ as pyo

import pyomo_pounce  # noqa: F401  (registers 'pounce')
from pyomo_pounce import declare_sens_param, estimate


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


def test_more_iterations_do_not_make_it_worse():
    m = solved()
    target = 1.6
    tx, ty = resolve_at(target)
    errs = []
    for it in (1, 2, 4, 12):
        cx, cy = estimated(m, target, corrector_iter=it)
        errs.append(max(abs(cx - tx), abs(cy - ty)))
    assert errs[-1] <= errs[0] * 1.01, (
        f"a larger budget should not lose accuracy: {errs}")


def test_it_applies_under_every_mode():
    """Each mode produces a step and the corrector refines whichever
    one ran, so asking for iterations under fix_relax or path must
    improve on that mode's own answer rather than being ignored."""
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
