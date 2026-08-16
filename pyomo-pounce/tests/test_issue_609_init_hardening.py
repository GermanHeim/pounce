"""gh #609: scaling, option propagation, block-DAG recovery, conditioning.

Each test here targets one of the issue's acceptance criteria, and each
was measured on the parent commit (cfc11218) first -- the numbers in the
comments are those measurements, not estimates.
"""

import pyomo.environ as pyo
import pytest

import pyomo_pounce
from pyomo_pounce import InitOptions
from pyomo_pounce.block_init import OK_TERMINATIONS

pytest.importorskip("networkx", reason="initialize pipeline needs networkx")
pytest.importorskip("scipy", reason="initialize pipeline needs scipy")


@pytest.fixture(scope="module")
def solver():
    s = pyo.SolverFactory("pounce")
    if not s.available(exception_flag=False):
        pytest.skip("pounce binary not found on PATH")
    return s


# --------------------------------------------------------------------------
# 1. Scaling
# --------------------------------------------------------------------------

def _mixed_units():
    """A 1e6-unit energy balance beside a 1e-6-unit trace balance.

    Both say the same thing about the solution (a + 2b == 3, a == b, so
    a == b == 1), but Ipopt's gradient-based rule -- ``min(1, g_max /
    ||grad c||)`` -- only ever scales a row *down*. The trace row keeps
    its 1e-6 magnitude and an absolute convergence test then enforces it
    to a relative accuracy eight orders of magnitude looser than the
    energy row's.
    """
    m = pyo.ConcreteModel()
    m.a = pyo.Var(initialize=0.0, bounds=(-1e3, 1e3))
    m.b = pyo.Var(initialize=0.0, bounds=(-1e3, 1e3))
    m.energy = pyo.Constraint(expr=1e6 * m.a + 2e6 * m.b == 3e6)
    m.trace = pyo.Constraint(expr=1e-6 * m.a - 1e-6 * m.b == 0.0)
    m.obj = pyo.Objective(expr=m.a)
    return m


def test_small_magnitude_row_is_enforced_like_a_large_one(solver):
    """The scaled projection lands on the exact answer of a mixed-units
    model. On the parent commit this returned a = 0.9999999921,
    b = 1.0000000039 -- a relative residual of 1.2e-8 on the trace row
    against 1.5e-16 on the energy row."""
    m = _mixed_units()
    pyomo_pounce.initialize_missing_values(m)
    cond = pyomo_pounce.project_to_feasible(m, solver=solver)
    assert cond in ("optimal", "locallyOptimal")
    a, b = pyo.value(m.a), pyo.value(m.b)
    # Two orders tighter than the parent commit's 7.9e-9 error.
    assert a == pytest.approx(1.0, abs=1e-11)
    assert b == pytest.approx(1.0, abs=1e-11)
    assert abs(a - b) / max(1.0, abs(a)) < 1e-11


def test_scaling_none_restores_the_unscaled_merit(solver):
    """``scaling="none"`` is the pre-gh#609 behaviour, and it is still
    reachable -- the parent commit's looser answer comes back."""
    m = _mixed_units()
    pyomo_pounce.initialize_missing_values(m)
    pyomo_pounce.project_to_feasible(
        m, solver=solver, options=InitOptions(scaling="none")
    )
    a, b = pyo.value(m.a), pyo.value(m.b)
    assert abs(a - b) > 1e-11  # the defect, deliberately reproduced


def _two_scale_anchor():
    """A pressure at 1e6 Pa and a mole fraction at 1e-4, coupled so that
    each contributes 1.0 in its own units and the pair is 10% short."""
    m = pyo.ConcreteModel()
    m.P = pyo.Var(initialize=1.0e6)
    m.x = pyo.Var(initialize=1.0e-4)
    m.link = pyo.Constraint(expr=m.P / 1e6 + m.x / 1e-4 == 2.2)
    m.obj = pyo.Objective(expr=m.P)
    return m


def test_merit_shares_repair_by_relative_magnitude(solver):
    """A repair is shared in proportion to what each variable can afford.

    On the parent commit the unscaled merit moved the mole fraction by
    20% and the pressure by 1.8e-12 -- a ratio of 1.1e11 -- because an
    absolute distance says moving a 1e-4 variable is 1e10 times cheaper
    than moving a 1e6 one. Scaled, both move 10%."""
    m = _two_scale_anchor()
    cond = pyomo_pounce.project_to_feasible(m, solver=solver)
    assert cond in ("optimal", "locallyOptimal")
    rel_P = abs(pyo.value(m.P) - 1.0e6) / 1.0e6
    rel_x = abs(pyo.value(m.x) - 1.0e-4) / 1.0e-4
    assert rel_P == pytest.approx(rel_x, rel=1e-6)
    assert rel_P == pytest.approx(0.1, rel=1e-6)


def _rescalable(k):
    m = pyo.ConcreteModel()
    m.x = pyo.Var(initialize=0.0)
    m.y = pyo.Var(initialize=0.0)
    m.z = pyo.Var(initialize=0.0)
    m.big = pyo.Constraint(expr=k * 1e6 * m.x + k * 1e6 * m.y == k * 3e6)
    m.small = pyo.Constraint(expr=1e-3 * m.x - 1e-3 * m.z == 1e-3)
    m.c = pyo.Constraint(expr=m.y + m.z == 2.0)
    m.obj = pyo.Objective(expr=m.x)
    return m


@pytest.mark.parametrize("k", [1e6, 1e-6, 1e10, 1e-10])
def test_projection_is_invariant_under_row_rescaling(solver, k):
    """Acceptance criterion 1. Multiplying a row by a constant does not
    change the feasible set, so it must not change where the projection
    lands -- to within the solver's own convergence tolerance.

    Measured at the default tolerance the spread is 1.2e-9 (the solver
    stops as soon as the *scaled* residual is inside tol, and where that
    happens depends on the iterate path); tightening tol collapses it to
    2.2e-16, which is what pins this as a tolerance effect rather than a
    scaling defect."""
    opts = InitOptions(solver_options={"tol": 1e-12})

    def point(kk):
        m = _rescalable(kk)
        pyomo_pounce.initialize_missing_values(m)
        assert pyomo_pounce.project_to_feasible(
            m, solver=solver, options=opts
        ) in ("optimal", "locallyOptimal")
        return [pyo.value(m.x), pyo.value(m.y), pyo.value(m.z)]

    base = point(1.0)
    got = point(k)
    assert max(abs(a - b) for a, b in zip(base, got)) < 1e-12


def test_user_scaling_suffix_entries_win_over_the_automatic_ones(solver):
    """Scope item 1's "user scaling suffix support": an entry the model
    carries is used as given, and the Suffix is restored afterwards."""
    m = _mixed_units()
    m.scaling_factor = pyo.Suffix(direction=pyo.Suffix.EXPORT)
    m.scaling_factor[m.trace] = 1e6
    pyomo_pounce.initialize_missing_values(m)
    pyomo_pounce.project_to_feasible(m, solver=solver)
    assert pyo.value(m.a) == pytest.approx(1.0, abs=1e-11)
    # exactly one entry, unchanged, and nothing of ours left behind
    assert len(m.scaling_factor) == 1
    assert m.scaling_factor[m.trace] == 1e6


def test_projection_leaves_no_scaling_suffix_behind(solver):
    """A model that declared no Suffix must not acquire one."""
    m = _mixed_units()
    pyomo_pounce.initialize_missing_values(m)
    pyomo_pounce.project_to_feasible(m, solver=solver)
    assert m.component("scaling_factor") is None
    objs = list(m.component_data_objects(pyo.Objective, descend_into=True))
    assert len(objs) == 1 and objs[0] is m.obj


def test_a_taken_scaling_factor_name_degrades_instead_of_raising(solver):
    """`scaling_factor` already used by a non-Suffix component.

    The projection delivers its row factors through a Suffix of that
    name, so a model that already spends the name on a Param, Var or
    Block leaves nowhere to put them. That is not an error the caller
    did anything about -- the name is theirs -- so the projection runs
    unscaled, exactly as it did before gh #609, and the model comes back
    untouched. Before the guard this raised `RuntimeError` from
    `add_component` *after* the original objective had been deactivated,
    leaving the model permanently broken.
    """
    m = _mixed_units()
    m.scaling_factor = pyo.Param(initialize=1.0)
    pyomo_pounce.initialize_missing_values(m)

    cond = pyomo_pounce.project_to_feasible(m, solver=solver)

    assert cond in OK_TERMINATIONS
    assert m.obj.active
    assert m.component("_pounce_projection_objective") is None
    assert m.component("scaling_factor") is m.scaling_factor


# --------------------------------------------------------------------------
# 2. Option propagation
# --------------------------------------------------------------------------

def _two_stage_model():
    """A 1x1 feeding a genuinely coupled 2x2, so a subsystem solve runs."""
    m = pyo.ConcreteModel()
    m.a = pyo.Var(initialize=1.0)
    m.x = pyo.Var(initialize=0.3)
    m.y = pyo.Var(initialize=0.3)
    m.c0 = pyo.Constraint(expr=m.a == 4.0)
    m.c1 = pyo.Constraint(expr=m.x + m.y == m.a)
    m.c2 = pyo.Constraint(expr=m.x * m.x - m.y == 1.0)
    m.obj = pyo.Objective(expr=m.x)
    return m


def _record_stages(monkeypatch, solver, **init_kwargs):
    seen = []
    real = type(solver).solve

    def spy(self, model, **kw):
        stage = (
            "projection"
            if hasattr(model, "_pounce_projection_objective")
            else "block_subsystem"
        )
        seen.append((stage, dict(kw.get("options") or {})))
        return real(self, model, **kw)

    monkeypatch.setattr(type(solver), "solve", spy)
    m = _two_stage_model()
    pyomo_pounce.initialize(m, solver=solver, **init_kwargs)
    return seen


def test_options_reach_every_stage(monkeypatch, solver):
    """Acceptance criterion 2. On the parent commit the projection got
    ``{"max_iter": 137}`` and the block subsystem solve got ``{}`` --
    ``block_initialize`` had no ``options`` argument at all."""
    seen = _record_stages(
        monkeypatch, solver, options={"max_iter": 137, "tol": 1e-9}
    )
    stages = {s for s, _ in seen}
    assert stages == {"projection", "block_subsystem"}, seen
    for stage, opts in seen:
        assert opts.get("max_iter") == 137, (stage, opts)
        assert opts.get("tol") == 1e-9, (stage, opts)


def test_init_options_object_reaches_every_stage(monkeypatch, solver):
    seen = _record_stages(
        monkeypatch,
        solver,
        options=InitOptions(solver_options={"max_iter": 41}),
    )
    assert seen
    for stage, opts in seen:
        assert opts.get("max_iter") == 41, (stage, opts)


def test_block_initialize_takes_options_directly(monkeypatch, solver):
    seen = []
    real = type(solver).solve

    def spy(self, model, **kw):
        seen.append(dict(kw.get("options") or {}))
        return real(self, model, **kw)

    monkeypatch.setattr(type(solver), "solve", spy)
    m = _two_stage_model()
    pyomo_pounce.block_initialize(m, solver=solver, options={"max_iter": 55})
    assert seen and all(o.get("max_iter") == 55 for o in seen)


def test_bare_dict_is_solver_options_not_policy():
    """A mapping is never reinterpreted as policy -- POUNCE has a solver
    option called ``scaling``, and reading it as ``InitOptions.scaling``
    would be a silent, wrong reinterpretation."""
    opts = InitOptions.coerce({"scaling": "off", "tol": 1e-7})
    assert opts.solver_options == {"scaling": "off", "tol": 1e-7}
    assert opts.scaling == "auto"


def test_init_options_validates_and_is_immutable():
    with pytest.raises(ValueError, match="on_block_failure"):
        InitOptions(on_block_failure="explode")
    with pytest.raises(ValueError, match="cond_tol"):
        InitOptions(cond_tol=0.0)
    with pytest.raises(TypeError, match="InitOptions or a mapping"):
        InitOptions.coerce(object())
    given = {"tol": 1e-8}
    opts = InitOptions(solver_options=given)
    given["tol"] = 1.0  # must not reach a later stage of the same call
    assert opts.solver_options == {"tol": 1e-8}


# --------------------------------------------------------------------------
# 3. Block-DAG recovery
# --------------------------------------------------------------------------

def _two_branch_model():
    """Two branches sharing no variable: one fails, one must not."""
    m = pyo.ConcreteModel()
    m.p = pyo.Var(bounds=(-10.0, 0.0), initialize=-1.0)  # needs p = 2
    m.pp = pyo.Var(initialize=1.0)
    m.f1 = pyo.Constraint(expr=m.p + m.pp == 3.0)
    m.f2 = pyo.Constraint(expr=m.p - m.pp == 1.0)
    m.q = pyo.Var(initialize=0.0)
    m.r = pyo.Var(initialize=0.0)
    m.g1 = pyo.Constraint(expr=m.q == 7.0)
    m.g2 = pyo.Constraint(expr=m.r == m.q + 1.0)
    m.obj = pyo.Objective(expr=m.r)
    return m


def test_independent_branch_survives_a_failure(solver):
    """Acceptance criterion 3. On the parent commit the 2x2 failure broke
    the loop and q and r stayed at their 0.0 seeds; n_vars_initialized
    was 0."""
    m = _two_branch_model()
    report = pyomo_pounce.block_initialize(m, solver=solver)
    assert not report.ok  # the failure is still reported
    assert pyo.value(m.q) == pytest.approx(7.0)
    assert pyo.value(m.r) == pytest.approx(8.0)
    assert report.n_vars_initialized == 2
    assert pyo.value(m.p) == -1.0 and pyo.value(m.pp) == 1.0  # seeds kept


def test_dependent_blocks_are_still_skipped(solver):
    """The other half of criterion 3: a block that *does* consume the
    failed block's values must not be run on wreckage."""
    m = pyo.ConcreteModel()
    m.x = pyo.Var(bounds=(-10.0, 0.0), initialize=-1.0)
    m.y = pyo.Var(initialize=1.0)
    m.z = pyo.Var(initialize=42.0)  # downstream of the failing 2x2
    m.c1 = pyo.Constraint(expr=m.x + m.y == 3.0)
    m.c2 = pyo.Constraint(expr=m.x - m.y == 1.0)
    m.c3 = pyo.Constraint(expr=m.z == m.x * m.y)
    m.obj = pyo.Objective(expr=m.z)

    report = pyomo_pounce.block_initialize(m, solver=solver)
    assert not report.ok
    assert pyo.value(m.z) == 42.0
    skipped = report.skipped_blocks
    assert len(skipped) == 1
    assert "depends on failed block" in skipped[0].detail


def test_on_block_failure_stop_restores_the_old_behaviour(solver):
    m = _two_branch_model()
    report = pyomo_pounce.block_initialize(
        m, solver=solver, options=InitOptions(on_block_failure="stop")
    )
    assert not report.ok
    assert pyo.value(m.q) == 0.0 and pyo.value(m.r) == 0.0
    assert report.n_vars_initialized == 0
    assert all("stop" in b.detail for b in report.skipped_blocks)


def test_block_dependencies_are_reported():
    """Scope item 3's "track the block graph explicitly"."""
    m = pyo.ConcreteModel()
    m.x = pyo.Var()
    m.y = pyo.Var()
    m.z = pyo.Var()
    m.c1 = pyo.Constraint(expr=m.x == 2.0)
    m.c2 = pyo.Constraint(expr=m.y == m.x + 1.0)
    m.c3 = pyo.Constraint(expr=m.z == 5.0)  # independent branch
    m.obj = pyo.Objective(expr=m.x)

    a = pyomo_pounce.block_analyze(m)
    assert len(a.block_dependencies) == a.n_blocks == 3
    owner = {blk[0].name: i for i, blk in enumerate(a.variable_blocks)}
    assert a.block_dependencies[owner["y"]] == [owner["x"]]
    assert a.block_dependencies[owner["x"]] == []
    assert a.block_dependencies[owner["z"]] == []


def test_structured_report_accounts_for_every_block(solver):
    """Scope item 5 / acceptance criterion 5: initialized, skipped,
    failed and fallback blocks are all named."""
    m = _two_branch_model()
    report = pyomo_pounce.block_initialize(m, solver=solver)
    assert len(report.blocks) == report.n_blocks
    buckets = (
        report.initialized_blocks
        + report.fallback_blocks
        + report.failed_blocks
        + report.skipped_blocks
    )
    assert len(buckets) == len(report.blocks)
    assert [b.constraint for b in report.failed_blocks] == ["f1"]
    assert sorted(b.constraint for b in report.initialized_blocks) == [
        "g1",
        "g2",
    ]
    assert "failed" in str(report)


def test_initialize_report_exposes_the_block_record(solver):
    m = _two_stage_model()
    report = pyomo_pounce.initialize(m, solver=solver)
    assert report.ok, str(report)
    assert report.blocks and all(
        b.status == "initialized" for b in report.blocks
    )


# --------------------------------------------------------------------------
# 4. Numerical conditioning
# --------------------------------------------------------------------------

def _near_singular(eps):
    """Structurally square, numerically rank-deficient.

        u +          v == 2
        u + (1 + eps)v == 2 + eps

    Exact solution u = v = 1, but the Jacobian [[1, 1], [1, 1+eps]] has a
    condition number of about 4/eps.
    """
    m = pyo.ConcreteModel()
    m.u = pyo.Var(initialize=0.0)
    m.v = pyo.Var(initialize=0.0)
    m.w = pyo.Var(initialize=0.0)
    m.d1 = pyo.Constraint(expr=m.u + m.v == 2.0)
    m.d2 = pyo.Constraint(expr=m.u + (1.0 + eps) * m.v == 2.0 + eps)
    m.d3 = pyo.Constraint(expr=m.w == m.u + 2.0 * m.v)
    m.obj = pyo.Objective(expr=m.w)
    return m


@pytest.mark.parametrize("eps", [1e-10, 1e-14])
def test_near_singular_block_gets_a_diagnostic_and_a_fallback(solver, eps):
    """Acceptance criterion 4. On the parent commit this reported success
    and wrote u = 2, v = 0, w = 2 -- an error of 1.0 against the exact
    u = v = 1, w = 3, with nothing said about it."""
    m = _near_singular(eps)
    report = pyomo_pounce.block_initialize(m, solver=solver)
    assert report.diagnostics, str(report)
    assert "near-singular" in report.diagnostics[0]
    assert [b.constraint for b in report.fallback_blocks] == ["d1"]
    weak = report.fallback_blocks[0]
    assert weak.rcond is not None and weak.rcond < 1e-8
    # the regularized fallback returns the minimum-norm solution
    assert pyo.value(m.u) == pytest.approx(1.0, abs=1e-6)
    assert pyo.value(m.v) == pytest.approx(1.0, abs=1e-6)
    assert pyo.value(m.w) == pytest.approx(3.0, abs=1e-6)


def test_healthy_blocks_are_not_diagnosed(solver):
    """The check must not fire on a well-conditioned model, including on
    a mixed-units one -- conditioning is a numerical question, not a
    units question."""
    m = pyo.ConcreteModel()
    m.a = pyo.Var(initialize=0.5)
    m.b = pyo.Var(initialize=0.5)
    m.e = pyo.Constraint(expr=1e6 * m.a + 2e6 * m.b == 3e6)
    m.t = pyo.Constraint(expr=1e-6 * m.a - 1e-6 * m.b == 0.0)
    m.obj = pyo.Objective(expr=m.a)

    report = pyomo_pounce.block_initialize(m, solver=solver)
    assert report.ok, str(report)
    assert not report.diagnostics
    assert not report.fallback_blocks
    assert pyo.value(m.a) == pytest.approx(1.0, abs=1e-6)


def test_conditioning_off_restores_the_old_behaviour(solver):
    """``conditioning="off"`` reproduces the parent commit's answer."""
    m = _near_singular(1e-14)
    report = pyomo_pounce.block_initialize(
        m, solver=solver, options=InitOptions(conditioning="off")
    )
    assert not report.diagnostics
    assert all(b.rcond is None for b in report.blocks)
    assert abs(pyo.value(m.u) - 1.0) > 0.5  # the defect, deliberately kept


def test_fallback_off_diagnoses_without_rerouting(solver):
    m = _near_singular(1e-14)
    report = pyomo_pounce.block_initialize(
        m, solver=solver, options=InitOptions(fallback="off")
    )
    assert report.diagnostics and "fallback='off'" in report.diagnostics[0]
    assert not report.fallback_blocks


def test_coupled_fallback_solves_the_weak_block_with_its_dependents(solver):
    """``fallback="coupled"`` merges the weak block with the blocks that
    depend on it and regularizes the union.

    Its precision on the near-null direction is set by ``regularization``
    against the solver's tolerance, not by the ridge bias that governs
    the plain regularized path: measured on this fixture the error is
    7.4e-3 at the 1e-8 default and 7.6e-5 at 1e-6. Both are two to four
    orders better than the parent commit's 1.0, which is the property
    that matters -- a defined value near the minimum-norm solution rather
    than an arbitrary end of the near-null direction.
    """
    m = _near_singular(1e-14)
    report = pyomo_pounce.block_initialize(
        m,
        solver=solver,
        options=InitOptions(fallback="coupled", regularization=1e-6),
    )
    assert "coupled fallback" in report.diagnostics[0]
    assert sorted(b.constraint for b in report.fallback_blocks) == ["d1", "d3"]
    assert pyo.value(m.u) == pytest.approx(1.0, abs=1e-3)
    assert pyo.value(m.v) == pytest.approx(1.0, abs=1e-3)
    assert pyo.value(m.w) == pytest.approx(3.0, abs=1e-3)


def test_regularized_fallback_bias_scales_with_the_ridge(solver):
    """The plain regularized path's error *is* the ridge bias, so it
    falls linearly with ``regularization`` -- which is why the default is
    1e-8 and not something larger. Measured: 7.5e-9 at 1e-8, 7.5e-5 at
    1e-4."""
    errs = {}
    for lam in (1e-8, 1e-4):
        m = _near_singular(1e-14)
        pyomo_pounce.block_initialize(
            m, solver=solver, options=InitOptions(regularization=lam)
        )
        errs[lam] = abs(pyo.value(m.u) - 1.0)
    assert errs[1e-8] < 1e-7
    assert errs[1e-4] < 1e-3
    assert errs[1e-8] < errs[1e-4] / 100.0


# --------------------------------------------------------------------------
# 5. gh #444's incidence-plan caching is preserved
# --------------------------------------------------------------------------

def test_still_one_incidence_walk_per_initialize(monkeypatch, solver):
    """Acceptance criterion 5's other half. The block DAG, the
    conditioning check and the fallback all read structure, and none of
    them may pay for a second whole-model walk: the DAG comes off the
    graph already built, and the Jacobian is differentiated per block."""
    import pyomo.core.base.block as _block
    import pyomo.contrib.incidence_analysis.interface as _iface

    walks = []
    orig = _iface.IncidenceGraphInterface.__init__

    def counting(self, model=None, *args, **kwargs):
        if isinstance(model, (_block.Block, _block.BlockData)):
            walks.append(model)
        return orig(self, model, *args, **kwargs)

    monkeypatch.setattr(_iface.IncidenceGraphInterface, "__init__", counting)
    m = _two_stage_model()
    report = pyomo_pounce.initialize(m, solver=solver)
    assert report.ok, str(report)
    assert len(walks) == 1


def test_one_incidence_walk_even_when_a_block_fails(monkeypatch, solver):
    """The recovery path is the new one, so it gets its own guard:
    computing descendants must not rebuild the graph."""
    import pyomo.core.base.block as _block
    import pyomo.contrib.incidence_analysis.interface as _iface

    walks = []
    orig = _iface.IncidenceGraphInterface.__init__

    def counting(self, model=None, *args, **kwargs):
        if isinstance(model, (_block.Block, _block.BlockData)):
            walks.append(model)
        return orig(self, model, *args, **kwargs)

    monkeypatch.setattr(_iface.IncidenceGraphInterface, "__init__", counting)
    m = _two_branch_model()
    report = pyomo_pounce.initialize(m, solver=solver, project=False)
    assert not report.ok
    assert len(walks) == 1
