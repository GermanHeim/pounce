"""The session resolves each solver column's name to the model's
variable exactly once, when the solve loads its solution back.

Every later query reads the captured objects: routing a name through
`find_component` parses it through pyomo's component-UID lexer, and
doing that per variable per call accumulated 0.87 s inside one
`estimate()` call on the 62k-variable double column (N=25 Radau
collocation), timed on the method itself.
"""
import warnings
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
    # three model variables plus the substitute the in-place rewrite
    # added for the folded param, itself an ordinary model variable
    defs = m.component("_pounce_sens_defs")
    assert len(est) == 4
    # component data is unhashable by design, so identity per position
    # is the membership check
    expect = [m.x[0], m.x[1], m.x[2], defs.v[1]]
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


def crossed_row():
    """A live inequality row the step drives out of bounds, so the
    report's crossed_rows is non-empty and the row resolution runs."""
    m = pyo.ConcreteModel()
    m.p = pyo.Param(initialize=0.5, mutable=True)
    m.x = pyo.Var(initialize=0.5)

    @m.Constraint()
    def cap(m):
        return m.x <= 1.0

    @m.Objective()
    def obj(m):
        return (m.x - m.p) ** 2

    declare_sens_param(m.p)
    pyo.SolverFactory("pounce").solve(m)
    return m


def test_the_report_resolves_row_names_only_for_a_crossing():
    """A report with nothing crossed resolves no names at all, and the
    first report that carries a crossed row resolves exactly the
    solve's row names, once per session: later reports and every
    active_set_changes() call resolve nothing."""
    from pyomo_pounce import active_set_changes, estimate_report
    from pyomo_pounce.sens import _REG

    m = solved()
    with _counting() as fc:
        estimate_report(m, [(m.p, 1.5)])
    assert fc.call_count == 0, (
        f"nothing crossed, so nothing resolves: {fc.call_count}")
    with _counting() as fc:
        active_set_changes(m, [(m.p, 1.5)])
    assert fc.call_count == 0, (
        f"the record resolves nothing: {fc.call_count}")

    m = crossed_row()
    session = m.__dict__[_REG].session
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with _counting() as fc:
            rep = estimate_report(m, [(m.p, 2.0)])
        assert len(rep.crossed_rows), "the fixture must cross its row"
        assert fc.call_count == len(session.con_names), (
            f"the first crossing resolves the row names once: "
            f"{fc.call_count} of {len(session.con_names)}")
        with _counting() as fc:
            estimate_report(m, [(m.p, 2.0)])
    assert fc.call_count == 0, (
        f"a later report resolves nothing: {fc.call_count}")


def test_a_deepcopied_result_answers_for_its_own_keys():
    """The identity index is valid only for the objects in the keys
    list, so a deepcopy rebuilds it: the copy answers for its own
    copied keys and refuses the original's, the way ComponentMap's
    rehash hook behaves."""
    import copy

    m = solved()
    est = estimate(m, [(m.p, 1.5)])
    cp = copy.deepcopy(est)
    new_keys = list(cp)
    assert new_keys[0] is not m.x[0]
    for old, new in zip(est, new_keys):
        assert cp[new] == est[old]
    assert m.x[0] not in cp
    with pytest.raises(KeyError):
        cp[m.x[0]]


def test_solution_maps_compare_by_contents_without_hashing_keys():
    """Mapping's default equality round-trips through dict(self) and
    raises on unhashable component data; the identity-index equality
    returns the bool ComponentMap returned."""
    m = solved()
    a = estimate(m, [(m.p, 1.5)])
    b = estimate(m, [(m.p, 1.5)])
    c = estimate(m, [(m.p, 1.6)])
    assert a == a
    assert a == b
    assert not (a == c)
    assert a != c
    assert not (a == {})
    assert not (a == 5)
