"""Tests for pyomo_pounce.project_to_feasible and the initialize pipeline.

The projection is a real POUNCE solve, so most of these need the
binary and skip without it.
"""

import pyomo.environ as pyo
import pytest

import pyomo_pounce

pytest.importorskip("networkx", reason="initialize pipeline needs networkx")
pytest.importorskip("scipy", reason="initialize pipeline needs scipy")


@pytest.fixture(scope="module")
def solver():
    s = pyo.SolverFactory("pounce")
    if not s.available(exception_flag=False):
        pytest.skip("pounce binary not found on PATH")
    return s


def _mole_fraction_model():
    """The reviewer's example: midpoint fill gives x_i = 0.5 each, which
    violates sum(x) == 1."""
    m = pyo.ConcreteModel()
    m.x = pyo.Var([1, 2, 3], bounds=(0.0, 1.0))
    m.sum_to_one = pyo.Constraint(expr=m.x[1] + m.x[2] + m.x[3] == 1.0)
    m.obj = pyo.Objective(expr=m.x[1])
    return m


def test_project_repairs_inconsistent_fill(solver):
    m = _mole_fraction_model()
    n = pyomo_pounce.initialize_missing_values(m)  # x_i = 0.5, sum = 1.5
    assert n == 3
    cond = pyomo_pounce.project_to_feasible(m, solver=solver)
    assert cond in ("optimal", "locallyOptimal")
    total = sum(pyo.value(m.x[i]) for i in [1, 2, 3])
    assert total == pytest.approx(1.0, abs=1e-6)
    # Min-norm from equal anchors: the components stay equal.
    for i in [1, 2, 3]:
        assert pyo.value(m.x[i]) == pytest.approx(1.0 / 3.0, abs=1e-5)


def test_project_restores_objective(solver):
    m = _mole_fraction_model()
    pyomo_pounce.initialize_missing_values(m)
    assert m.obj.active
    pyomo_pounce.project_to_feasible(m, solver=solver)
    assert m.obj.active  # original objective restored
    # ... and the temporary projection objective is gone.
    objs = list(m.component_data_objects(pyo.Objective, descend_into=True))
    assert len(objs) == 1


def test_project_without_anchors_raises():
    m = _mole_fraction_model()  # nothing has a value yet
    with pytest.raises(ValueError, match="initialize_missing_values"):
        pyomo_pounce.project_to_feasible(m)


def test_initialize_pipeline_end_to_end(solver):
    # fill -> repair -> block-solve on a split with a composition sum.
    m = pyo.ConcreteModel()
    m.feed = pyo.Var(initialize=10.0)
    m.split = pyo.Var(bounds=(0.0, 1.0), initialize=0.3)
    m.out1 = pyo.Var(bounds=(0.0, None))
    m.out2 = pyo.Var(bounds=(0.0, None))
    m.x = pyo.Var([1, 2], bounds=(0.0, 1.0))
    m.bal1 = pyo.Constraint(expr=m.out1 == m.split * m.feed)
    m.bal2 = pyo.Constraint(expr=m.out2 == (1.0 - m.split) * m.feed)
    m.sum_x = pyo.Constraint(expr=m.x[1] + m.x[2] == 1.0)
    m.obj = pyo.Objective(expr=m.out1 + m.x[1])

    report = pyomo_pounce.initialize(
        m, decisions=[m.feed, m.split], solver=solver
    )
    assert report.ok, str(report)
    assert report.n_filled == 4  # out1, out2, x[1], x[2]
    assert report.projection in ("optimal", "locallyOptimal")
    assert report.block is not None and report.block.ok
    # Block solve produced the physical profile with decisions held...
    assert m.out1.value == pytest.approx(3.0, abs=1e-5)
    assert m.out2.value == pytest.approx(7.0, abs=1e-5)
    # ... and the projection made the composition consistent.
    assert pyo.value(m.x[1]) + pyo.value(m.x[2]) == pytest.approx(1.0, abs=1e-5)
    # Decisions are free again for the optimizer.
    assert not m.feed.fixed and not m.split.fixed
    assert "fill -> repair -> block-solve" in str(report)


def test_initialize_skips_projection_when_asked():
    m = pyo.ConcreteModel()
    m.x = pyo.Var(bounds=(1.0, 3.0))
    m.c = pyo.Constraint(expr=m.x <= 2.5)
    m.obj = pyo.Objective(expr=m.x)

    report = pyomo_pounce.initialize(m, project=False)
    assert report.projection is None
    assert report.n_filled == 1
    assert m.x.value == pytest.approx(2.0)  # midpoint fill, no repair


def test_initialize_pins_automatically(solver):
    # A loose integrator (M appears only as the denominator of a
    # 0 == f/M row) is identified with no user input: pinned, seeded,
    # held for the pipeline, released after.
    m = pyo.ConcreteModel()
    m.u = pyo.Var(initialize=2.0)
    m.x = pyo.Var()
    m.M = pyo.Var(bounds=(1.0, 3.0))  # no value
    m.c1 = pyo.Constraint(expr=0 == (m.x - m.u) / m.M)
    m.obj = pyo.Objective(expr=m.x)

    report = pyomo_pounce.initialize(m, solver=solver, decisions=[m.u])
    assert report.ok, str(report)
    assert report.repair is not None
    assert report.repair.pinned == [m.M]
    assert report.block.square
    assert m.M.value == pytest.approx(2.0)  # bounds-aware midpoint seed
    assert m.x.value == pytest.approx(2.0)
    assert not m.u.fixed and not m.M.fixed
    assert "spec repair" in str(report)


def test_initialize_repair_off_reports_instead(solver):
    # The strict path: nothing pruned, nothing pinned, the non-square
    # specification is reported rather than repaired.
    m = pyo.ConcreteModel()
    m.u = pyo.Var(initialize=2.0)
    m.x = pyo.Var()
    m.M = pyo.Var(bounds=(1.0, 3.0))
    m.c1 = pyo.Constraint(expr=0 == (m.x - m.u) / m.M)
    m.obj = pyo.Objective(expr=m.x)

    report = pyomo_pounce.initialize(
        m, solver=solver, decisions=[m.u], repair="off"
    )
    assert report.repair is None
    assert report.n_pinned == 0
    assert not report.block.square
    assert not m.M.fixed


def test_projection_failure_restores_values(solver):
    # projecting onto contradictory constraints fails; the point the
    # caller had must come back, so the pipeline's "continuing with the
    # unrepaired point" is literally true
    m = pyo.ConcreteModel()
    m.x = pyo.Var(initialize=5.0)
    m.c1 = pyo.Constraint(expr=m.x == 1.0)
    m.c2 = pyo.Constraint(expr=m.x == 2.0)
    m.obj = pyo.Objective(expr=m.x)

    cond = pyomo_pounce.project_to_feasible(m, solver=solver)
    assert cond not in ("optimal", "locallyOptimal")
    assert m.x.value == 5.0


def _split_model():
    m = pyo.ConcreteModel()
    m.feed = pyo.Var(initialize=10.0)
    m.split = pyo.Var(bounds=(0.0, 1.0), initialize=0.3)
    m.out1 = pyo.Var(bounds=(0.0, None))
    m.out2 = pyo.Var(bounds=(0.0, None))
    m.x = pyo.Var([1, 2], bounds=(0.0, 1.0))
    m.bal1 = pyo.Constraint(expr=m.out1 == m.split * m.feed)
    m.bal2 = pyo.Constraint(expr=m.out2 == (1.0 - m.split) * m.feed)
    m.sum_x = pyo.Constraint(expr=m.x[1] + m.x[2] == 1.0)
    m.obj = pyo.Objective(expr=m.out1 + m.x[1])
    return m


def test_initialize_walks_incidence_once(monkeypatch, solver):
    """One structural incidence walk per initialize() call (gh #444):
    the plan, analyze, and block passes filter the same graph, and
    block_initialize no longer re-plans. Counted on model-walking
    constructions only; subgraph() re-enters __init__ with a prebuilt
    graph tuple and never re-inspects constraints."""
    import pyomo.core.base.block as _block
    import pyomo.contrib.incidence_analysis.interface as _iface

    walks = []
    orig = _iface.IncidenceGraphInterface.__init__

    def counting(self, model=None, *args, **kwargs):
        if isinstance(model, _block.Block) or isinstance(
                model, _block.BlockData):
            walks.append(model)
        return orig(self, model, *args, **kwargs)

    monkeypatch.setattr(_iface.IncidenceGraphInterface, "__init__", counting)
    m = _split_model()
    report = pyomo_pounce.initialize(
        m, decisions=[m.feed, m.split], solver=solver
    )
    assert report.ok, str(report)
    assert len(walks) == 1


def test_shared_graph_matches_fresh_analysis():
    """The filtered structural view reproduces a fresh construction's
    analysis exactly (gh #444): same plan, same square decomposition,
    same block order."""
    from pyomo_pounce.block_init import (
        block_analyze,
        block_repair_plan,
        structural_incidence,
    )

    def names(comps):
        return [c.name for c in comps]

    m1 = _split_model()
    plan_fresh = block_repair_plan(
        m1, decision_candidates=[m1.feed, m1.split])
    ana_fresh = block_analyze(m1, decisions=[m1.feed, m1.split])

    m2 = _split_model()
    g = structural_incidence(m2)
    plan_shared = block_repair_plan(
        m2, decision_candidates=[m2.feed, m2.split], igraph=g)
    ana_shared = block_analyze(
        m2, decisions=[m2.feed, m2.split], igraph=g)

    assert names(plan_fresh.decisions) == names(plan_shared.decisions)
    assert names(plan_fresh.pinned) == names(plan_shared.pinned)
    assert names(plan_fresh.pruned) == names(plan_shared.pruned)
    assert ana_fresh.square == ana_shared.square
    assert [names(b) for b in ana_fresh.variable_blocks] == \
        [names(b) for b in ana_shared.variable_blocks]
    assert [names(b) for b in ana_fresh.constraint_blocks] == \
        [names(b) for b in ana_shared.constraint_blocks]


def test_shared_graph_one_factor_fixed_matches_as_sets():
    """Only one factor of the bilinear terms fixed — the arrangement
    where value substitution can reorder the linear/nonlinear split.
    The corrected contract (gh #445 review): edge sets and the square
    decomposition match a fresh build; within-row variable ORDER may
    differ, so the comparison is set-level."""
    from pyomo_pounce.block_init import (
        block_analyze,
        block_repair_plan,
        structural_incidence,
    )

    def name_set(comps):
        return {c.name for c in comps}

    m1 = _split_model()
    plan_fresh = block_repair_plan(m1, decision_candidates=[m1.split])
    ana_fresh = block_analyze(m1, decisions=[m1.split])

    m2 = _split_model()
    g = structural_incidence(m2)
    plan_shared = block_repair_plan(
        m2, decision_candidates=[m2.split], igraph=g)
    ana_shared = block_analyze(m2, decisions=[m2.split], igraph=g)

    assert name_set(plan_fresh.decisions) == name_set(plan_shared.decisions)
    assert name_set(plan_fresh.pinned) == name_set(plan_shared.pinned)
    assert name_set(plan_fresh.pruned) == name_set(plan_shared.pruned)
    assert ana_fresh.square == ana_shared.square
    assert {frozenset(name_set(b)) for b in ana_fresh.variable_blocks} == \
        {frozenset(name_set(b)) for b in ana_shared.variable_blocks}


def test_shared_graph_zero_decision_falls_back_to_fresh():
    """A decision fixed at exactly 0 cancels its bilinear partner's
    edge under value substitution; the shared view must detect the
    cancellation and fall back to a fresh build, reproducing the
    fresh diagnostics exactly rather than claiming a square system
    and failing a block (gh #445 review, finding 2a)."""
    from pyomo_pounce.block_init import block_analyze, structural_incidence

    m1 = _split_model()
    m1.split.fix(0.0)
    ana_fresh = block_analyze(m1)

    m2 = _split_model()
    g = structural_incidence(m2)
    m2.split.fix(0.0)
    ana_shared = block_analyze(m2, igraph=g)

    assert ana_fresh.square == ana_shared.square
    assert [v.name for b in ana_fresh.variable_blocks for v in b] == \
        [v.name for b in ana_shared.variable_blocks for v in b]
    assert [c.name for c in ana_fresh.underconstrained_variables] == \
        [c.name for c in ana_shared.underconstrained_variables]


def _cancelling_model():
    """`a*x - b*x` with `a` and `b` fixed EQUAL and NONZERO: the `x`
    terms cancel under value substitution, so a fresh build drops `x`
    from `c1` while the structural view keeps it -- and no fixed
    variable is itself zero."""
    m = pyo.ConcreteModel()
    m.a = pyo.Var(initialize=2.0)
    m.b = pyo.Var(initialize=2.0)
    m.x = pyo.Var(initialize=1.0)
    m.y = pyo.Var(initialize=1.0)
    m.c1 = pyo.Constraint(expr=m.y == m.a * m.x - m.b * m.x)
    m.c2 = pyo.Constraint(expr=m.y == 5.0)
    m.obj = pyo.Objective(expr=m.y)
    return m


def test_shared_graph_cancelling_nonzero_pair_falls_back_to_fresh():
    """A cancellation with no zero-valued variable in sight (gh #445
    review). Keying the guard on `value == 0.0` misses this: `a` and
    `b` are both 2.0, yet `a*x - b*x` drops `x` under substitution.

    Unguarded, the shared view keeps the phantom `x` edge, finds a
    perfect matching where a fresh build finds none, and reports a
    SQUARE system -- then solves `c1` for `x` whose derivative is
    `a - b = 0`. That is the same regression the zero-valued guard was
    added to prevent, reached by a different route: strictly worse than
    the fresh answer, which reports `c1`/`c2` overconstrained and
    breaks nothing.
    """
    from pyomo_pounce.block_init import block_analyze, structural_incidence

    m1 = _cancelling_model()
    m1.a.fix(2.0)
    m1.b.fix(2.0)
    ana_fresh = block_analyze(m1)

    m2 = _cancelling_model()
    g = structural_incidence(m2)
    m2.a.fix(2.0)
    m2.b.fix(2.0)
    ana_shared = block_analyze(m2, igraph=g)

    # the fixture is only meaningful if the fresh build really does see
    # a non-square system here
    assert ana_fresh.square is False
    assert ana_fresh.square == ana_shared.square
    assert {c.name for c in ana_fresh.overconstrained_constraints} == \
        {c.name for c in ana_shared.overconstrained_constraints}
    assert [[v.name for v in b] for b in ana_fresh.variable_blocks] == \
        [[v.name for v in b] for b in ana_shared.variable_blocks]


def test_shared_graph_nonzero_fixed_without_cancellation_stays_shared():
    """The widened guard must not send every fixed-variable model down
    the fresh-build path: a fixed nonzero that cancels nothing keeps
    the shared view, or the gh #444 speedup is gone."""
    import pyomo.contrib.incidence_analysis.interface as _iface
    from pyomo_pounce.block_init import _active_view, structural_incidence

    m = _split_model()
    g = structural_incidence(m)
    m.split.fix(0.3)                     # nonzero, cancels nothing

    walks = []
    orig = _iface.IncidenceGraphInterface.__init__

    def counting(self, model=None, *args, **kwargs):
        import pyomo.core.base.block as _block
        if isinstance(model, (_block.Block, _block.BlockData)):
            walks.append(model)
        return orig(self, model, *args, **kwargs)

    _iface.IncidenceGraphInterface.__init__ = counting
    try:
        view = _active_view(g, m)
    finally:
        _iface.IncidenceGraphInterface.__init__ = orig

    assert walks == [], "took the fresh-build fallback with no cancellation"
    assert view is not g                 # filtered, not the raw shared graph
    assert {v.name for v in view.variables} == {
        v.name for v in g.variables if not v.fixed}
