"""Tests for the 1x1 convergence test reading the `scaling_factor`
Suffix: the tolerance is measured on the row's stated scale, and an
untagged row or an unsuffixed model keeps Pyomo's absolute default."""
import math

import pytest
import pyomo.environ as pyo

import pyomo_pounce  # noqa: F401  (registers 'pounce')
from pyomo_pounce import block_initialize


ROOT2 = math.sqrt(2.0)


def hot_row(with_factor):
    """One square 1x1 block whose equation defeats the absolute test.

    The root of `1e10 * x**2 == 2e10` is sqrt(2), and no double near it
    evaluates x**2 to exactly 2, so the raw residual bottoms out near
    1e10 * 4.4e-16, about 4e-6. That is converged on the row's own
    scale and unreachable by the absolute eps=1e-8, which is the field
    failure this file pins (IDAES energy holdups near 3e7 J).
    """
    m = pyo.ConcreteModel()
    m.x = pyo.Var(initialize=1.0)
    m.c = pyo.Constraint(expr=1e10 * m.x**2 == 2e10)
    if with_factor:
        m.scaling_factor = pyo.Suffix(direction=pyo.Suffix.EXPORT)
        m.scaling_factor[m.c] = 1e-10
    return m


def test_a_tagged_hot_row_initializes():
    m = hot_row(with_factor=True)
    report = block_initialize(m)
    assert report.ok, report.failures
    assert report.blocks[0].status == "initialized"
    assert pyo.value(m.x) == pytest.approx(ROOT2, rel=1e-7)


def test_the_same_row_untagged_still_fails():
    """Without the Suffix the behavior is exactly today's: the absolute
    test cannot be met, the block fails, and the seed is restored. This
    is also what keeps the test above honest, since it proves the row
    genuinely defeats the default."""
    m = hot_row(with_factor=False)
    report = block_initialize(m)
    assert not report.ok
    assert report.blocks[0].status == "failed"
    assert pyo.value(m.x) == pytest.approx(1.0, abs=0.0), "seed restored"


def test_a_local_suffix_is_not_read():
    """A `scaling_factor` Suffix with direction=LOCAL never reaches a
    solver, so the 1x1 test does not read it either, the same policy
    the gh #483 reader applies to full-model solves."""
    m = hot_row(with_factor=False)
    m.scaling_factor = pyo.Suffix(direction=pyo.Suffix.LOCAL)
    m.scaling_factor[m.c] = 1e-10
    report = block_initialize(m)
    assert not report.ok
    assert report.blocks[0].status == "failed"


def test_an_untagged_row_keeps_the_default_next_to_a_tagged_one():
    """The factor applies per constraint, not per model: with the
    Suffix present, a row it does not tag still runs the absolute
    test."""
    m = pyo.ConcreteModel()
    m.x1 = pyo.Var(initialize=1.0)
    m.x2 = pyo.Var(initialize=1.0)
    m.c1 = pyo.Constraint(expr=1e10 * m.x1**2 == 2e10)
    m.c2 = pyo.Constraint(expr=1e10 * m.x2**2 == 2e10)
    m.scaling_factor = pyo.Suffix(direction=pyo.Suffix.EXPORT)
    m.scaling_factor[m.c1] = 1e-10
    report = block_initialize(m)

    by_con = {o.constraint: o.status for o in report.blocks}
    assert by_con[m.c1.name] == "initialized"
    assert by_con[m.c2.name] == "failed"
    assert pyo.value(m.x1) == pytest.approx(ROOT2, rel=1e-7)
    assert pyo.value(m.x2) == pytest.approx(1.0, abs=0.0), "seed restored"


def test_a_factor_on_the_indexed_container_covers_each_row():
    """Tagging the indexed Constraint tags every member, the same key
    expansion the gh #483 reader applies everywhere else."""
    m = pyo.ConcreteModel()
    m.i = pyo.Set(initialize=[1, 2])
    m.y = pyo.Var(m.i, initialize=1.0)

    @m.Constraint(m.i)
    def c(m, i):
        return 1e10 * m.y[i] ** 2 == 2e10

    m.scaling_factor = pyo.Suffix(direction=pyo.Suffix.EXPORT)
    m.scaling_factor[m.c] = 1e-10
    report = block_initialize(m)
    assert report.ok, report.failures
    for i in m.i:
        assert pyo.value(m.y[i]) == pytest.approx(ROOT2, rel=1e-7)


def test_a_well_scaled_model_is_untouched_by_a_factor_of_one():
    """A factor of 1.0 is the identity test: same eps, same outcome,
    same value as the untagged path on a row the default solves."""
    def build(tag):
        m = pyo.ConcreteModel()
        m.x = pyo.Var(initialize=1.0)
        m.c = pyo.Constraint(expr=m.x**2 == 2)
        if tag:
            m.scaling_factor = pyo.Suffix(direction=pyo.Suffix.EXPORT)
            m.scaling_factor[m.c] = 1.0
        return m

    plain, tagged = build(False), build(True)
    r_plain = block_initialize(plain)
    r_tagged = block_initialize(tagged)
    assert r_plain.ok and r_tagged.ok
    assert pyo.value(plain.x) == pyo.value(tagged.x)
