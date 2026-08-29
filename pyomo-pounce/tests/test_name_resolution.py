"""The session resolves each solver column's name to the model's
variable exactly once, when the solve loads its solution back.

Every later query reads the captured objects: routing a name through
`find_component` parses it through pyomo's component-UID lexer, and
doing that per variable per call was 2.2 s of a 3.4 s `estimate()`
call on a 62k-variable model.
"""
from unittest import mock

import pytest
import pyomo.environ as pyo
from pyomo.core.base.block import BlockData

from pyomo_pounce import declare_sens_param, estimate, gradient


def solved():
    m = pyo.ConcreteModel()
    m.p = pyo.Param(initialize=1.0, mutable=True)
    m.x = pyo.Var(range(3), initialize=1.0)

    @m.Constraint(range(3))
    def c(m, i):
        return m.x[i] == m.p * (i + 1)

    @m.Objective()
    def obj(m):
        return sum((m.x[i] - i) ** 2 for i in range(3))

    declare_sens_param(m.p)
    pyo.SolverFactory("pounce").solve(m)
    return m


def _counting():
    """Patch that counts `find_component` calls while passing them
    through unchanged."""
    real = BlockData.find_component
    return mock.patch.object(
        BlockData, "find_component", autospec=True, side_effect=real)


def test_estimate_resolves_no_names():
    """An `estimate()` call parses no component names: the session
    holds the solve's own variable objects. The parent resolved one
    name per solver column per call."""
    m = solved()
    with _counting() as fc:
        est = estimate(m, [(m.p, 1.5)])
    assert fc.call_count == 0, (
        f"estimate resolved {fc.call_count} names through find_component")
    assert est[m.x[0]] == pytest.approx(1.5)


def test_gradient_resolves_no_names():
    """`gradient(target=None)` walks every variable through the same
    captured list."""
    m = solved()
    with _counting() as fc:
        g = gradient(wrt=m.p)
    assert fc.call_count == 0, (
        f"gradient resolved {fc.call_count} names through find_component")
    assert g[m.x[0], m.p] == pytest.approx(1.0)


def test_the_solution_map_is_read_only_and_complete():
    """The returned map exposes every model variable through the
    Mapping interface and rejects item assignment: a result describes
    one estimate."""
    m = solved()
    est = estimate(m, [(m.p, 1.5)])
    assert len(est) == 3
    # component data is unhashable by design, so identity per position
    # is the membership check
    expect = [m.x[0], m.x[1], m.x[2]]
    assert all(a is b for a, b in zip(est, expect))
    assert [v for _k, v in est.items()] == [est[vd] for vd in expect]
    with pytest.raises(TypeError):
        est[m.x[0]] = 0.0
    with pytest.raises(KeyError):
        est[m.p]


def test_estimate_keys_are_the_models_own_variables():
    """The solve runs on a clone, and the returned map must be keyed by
    the ORIGINAL model's data objects, identically, not by name-alikes:
    ComponentMap hashes by identity, so membership is the identity
    test."""
    m = solved()
    est = estimate(m, [(m.p, 1.2)])
    for i in range(3):
        assert m.x[i] in est
