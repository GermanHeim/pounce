"""Tests for pyomo_pounce.sens: declared-parameter sensitivity."""
import warnings

import pytest
import pyomo.environ as pyo

import pyomo_pounce  # noqa: F401  (registers 'pounce')
from pyomo_pounce import declare_sens_param, estimate, gradient

FD_H = 1e-5


def build(p=2.0, q=1.0):
    """min (x-p)^2 + (y-q)^2 + 0.1 exp(x+y), x <= 3, one equality tying a
    third variable (for multiplier sensitivities)."""
    m = pyo.ConcreteModel()
    m.x = pyo.Var(initialize=1.0, bounds=(None, 3.0))
    m.y = pyo.Var(initialize=1.0)
    m.w = pyo.Var(initialize=0.0)
    m.p = pyo.Param(initialize=p, mutable=True)
    m.q = pyo.Param(initialize=q, mutable=True)
    m.tie = pyo.Constraint(expr=m.w == m.x + m.y)     # equality, has a dual
    m.obj = pyo.Objective(
        expr=(m.x - m.p) ** 2 + (m.y - m.q) ** 2 + 0.1 * pyo.exp(m.w)
        + 0.01 * m.w**2)
    return m


def solve_plain(m):
    m.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    pyo.SolverFactory("pounce").solve(m)
    return m


@pytest.fixture(scope="module")
def solved():
    m = build()
    declare_sens_param(m.p)
    declare_sens_param(m.q)
    pyo.SolverFactory("pounce").solve(m)
    return m


def test_gradient_matches_finite_difference(solved):
    m = solved
    for pname in ("p", "q"):
        g = gradient(m.x, wrt=getattr(m, pname))
        assert isinstance(g, float)
        hi = build(**{pname: pyo.value(getattr(m, pname)) + FD_H})
        lo = build(**{pname: pyo.value(getattr(m, pname)) - FD_H})
        solve_plain(hi)
        solve_plain(lo)
        fd = (pyo.value(hi.x) - pyo.value(lo.x)) / (2 * FD_H)
        assert g == pytest.approx(fd, abs=1e-4)


def test_multiplier_gradient_matches_finite_difference(solved):
    m = solved
    g = gradient(m.tie, wrt=m.p)
    hi = solve_plain(build(p=2.0 + FD_H))
    lo = solve_plain(build(p=2.0 - FD_H))
    fd = (hi.dual[hi.tie] - lo.dual[lo.tie]) / (2 * FD_H)
    assert abs(g) == pytest.approx(abs(fd), abs=1e-4)
    assert g == pytest.approx(fd, abs=1e-4), (
        "sign convention mismatch between parametric_step_full's y_c block "
        "and pyomo duals")


def test_estimate_matches_resolve(solved):
    m = solved
    est = estimate(m, [(m.p, 2.2), (m.q, 0.9)])
    mt = solve_plain(build(p=2.2, q=0.9))
    assert est[m.x] == pytest.approx(pyo.value(mt.x), abs=5e-3)
    assert est[m.y] == pytest.approx(pyo.value(mt.y), abs=5e-3)


def test_estimate_clamps_and_warns(solved):
    m = solved
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        est = estimate(m, [(m.p, 8.0)])       # drives x past its bound of 3
    assert any("clamped" in str(wi.message) for wi in w)
    assert est[m.x] <= 3.0 + 1e-9


def test_gradient_object_for_containers(solved):
    m = solved
    G = gradient(m.x, wrt=m.p)          # scalar -> float
    assert isinstance(G, float)
    Gall = gradient(wrt=m.p)            # all variables
    assert Gall[m.x] == pytest.approx(G)
    df_cols = gradient(wrt=m.q).to_dataframe()
    assert "x" in df_cols.index


def test_plain_solve_unaffected():
    m = build()                          # no declarations
    res = pyo.SolverFactory("pounce").solve(m)
    assert pyo.value(m.obj) == pytest.approx(1.2653, abs=1e-3)
    with pytest.raises(RuntimeError, match="no sensitivity session"):
        gradient(m.x, wrt=m.p)


def test_resolve_and_clone_are_clean():
    """Solving twice (or cloning) must not trip Pyomo's uncopyable-field
    error: the registry deepcopies its declarations but not the session."""
    import io
    import logging

    buf = io.StringIO()
    h = logging.StreamHandler(buf)
    logging.getLogger("pyomo").addHandler(h)
    try:
        m = build()
        declare_sens_param(m.p)
        pyo.SolverFactory("pounce").solve(m)
        g1 = gradient(m.x, wrt=m.p)
        m.p = 2.5
        pyo.SolverFactory("pounce").solve(m)     # re-solve re-clones
        g2 = gradient(m.x, wrt=m.p)
        clone = m.clone()
    finally:
        logging.getLogger("pyomo").removeHandler(h)
    log = buf.getvalue()
    assert "Unable to clone" not in log and "uncopyable" not in log, log
    assert g1 != g2                              # new factorization was used
    from pyomo_pounce.sens import has_declarations
    assert has_declarations(clone)               # declarations survive clone


def test_declared_solve_returns_solver_results():
    """The declared path must return the same result shape as a plain
    Pyomo solve (review #199 item 2)."""
    m = build()
    declare_sens_param(m.p)
    res = pyo.SolverFactory("pounce").solve(m)
    assert (res.solver.termination_condition
            == pyo.TerminationCondition.optimal)
    assert str(res.solver.status) == "ok"


def test_no_temp_dir_leak(tmp_path):
    """Repeated declared solves must not accumulate pounce_sens_* temp
    dirs (review #199 item 1)."""
    import glob
    import os
    import tempfile

    pattern = os.path.join(tempfile.gettempdir(), "pounce_sens_*")
    before = set(glob.glob(pattern))
    m = build()
    declare_sens_param(m.p)
    for _ in range(3):
        pyo.SolverFactory("pounce").solve(m)
    after = set(glob.glob(pattern))
    assert after == before, f"leaked: {after - before}"


def test_keyword_model_solve_uses_declared_path():
    """solve(model=m) must hit the sensitivity path too (review #199
    item 4)."""
    m = build()
    declare_sens_param(m.p)
    pyo.SolverFactory("pounce").solve(model=m)
    assert gradient(m.x, wrt=m.p) is not None


def test_explicit_sens_params_form_equals_declared():
    """solve(m, sens_params=[...]) must register and behave exactly like
    declare_sens_param."""
    m1 = build()
    declare_sens_param(m1.p)
    pyo.SolverFactory("pounce").solve(m1)
    g_declared = gradient(m1.x, wrt=m1.p)

    m2 = build()                          # no declarations
    pyo.SolverFactory("pounce").solve(m2, sens_params=[m2.p])
    g_explicit = gradient(m2.x, wrt=m2.p)
    assert g_explicit == pytest.approx(g_declared, rel=1e-9)


def test_explicit_declarations_without_model_error():
    """Passing explicit declaration kwargs with no model surfaces a clear
    error instead of silently dropping them."""
    m = build()
    with pytest.raises(ValueError, match="no model was passed"):
        pyo.SolverFactory("pounce").solve(sens_params=[m.p])


def test_tee_prints_engine_log(capfd):
    m = build()
    declare_sens_param(m.p)
    pyo.SolverFactory("pounce").solve(m, tee=True)
    out = capfd.readouterr().out
    assert "This is POUNCE version" in out    # banner
    assert "Total number of variables" in out # problem statistics
    assert "inf_pr" in out                    # iteration-table header
    assert "Number of Iterations....:" in out # summary
    assert "EXIT: Optimal Solution Found." in out


def test_no_tee_is_silent(capfd):
    m = build()
    declare_sens_param(m.p)
    pyo.SolverFactory("pounce").solve(m)
    out = capfd.readouterr().out
    assert "inf_pr" not in out and "POUNCE" not in out


def test_tee_streams_under_redirect_stdout():
    """The capture-and-stream (non-live) branch of the tee path. Unlike
    capfd, which captures fd 1 (the live path), redirect_stdout captures
    sys.stdout, so this exercises the branch that tails the engine's fd-1
    log to sys.stdout for notebooks (review #206)."""
    import contextlib
    import io
    m = build()
    declare_sens_param(m.p)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        pyo.SolverFactory("pounce").solve(m, tee=True)
    out = buf.getvalue()
    assert "This is POUNCE version" in out     # banner (via print_banner)
    assert "Total number of variables" in out  # problem statistics (engine)
    assert "inf_pr" in out                     # iteration table (engine)
    assert "Number of Iterations....:" in out  # summary (engine)


def test_results_fields_match_asl_shape():
    m = build()
    declare_sens_param(m.p)
    res = pyo.SolverFactory("pounce").solve(m)
    assert res.problem.number_of_variables > 0
    assert res.problem.number_of_constraints >= 1
    assert float(res.solver.time) >= 0.0
    assert res.solver.message.startswith("POUNCE ")
    assert "_" not in res.solver.message      # SolveSucceeded, like the .sol
    assert ": " in res.solver.message         # "POUNCE <ver>: <status>"
    assert res.solver.id == 0 and res.solver.error_rc == 0
    assert res.solver.name == "pounce (in-process sensitivity session)"
    assert res.problem.upper_bound is not None        # objective bounds restored
    assert res.problem.lower_bound == res.problem.upper_bound
    assert list(res.solver[0].keys())[-1] == "Time"   # field order parity
    assert len(res.solution) == 0             # emptied Solution block


def test_nonconvergence_returns_mapped_results():
    """An unsolvable declared model must return a results object with the
    mapped termination (like the ordinary .sol path) instead of raising,
    and must leave no session behind for the sensitivity queries."""
    m = pyo.ConcreteModel()
    m.p = pyo.Param(initialize=2.0, mutable=True)
    m.x = pyo.Var(initialize=1.0)
    m.c1 = pyo.Constraint(expr=m.x == 1.0)
    m.c2 = pyo.Constraint(expr=m.x == 3.0)     # contradictory
    m.obj = pyo.Objective(expr=(m.x - m.p) ** 2)
    declare_sens_param(m.p)
    res = pyo.SolverFactory("pounce").solve(m)
    assert res.solver.termination_condition in (
        pyo.TerminationCondition.infeasible,
        pyo.TerminationCondition.error,
        pyo.TerminationCondition.maxIterations,
    )
    assert res.solver.status != pyo.SolverStatus.ok
    with pytest.raises(RuntimeError, match="no sensitivity session"):
        gradient(m.x, wrt=m.p)


def test_failed_resolve_clears_prior_session():
    """A converged solve leaves a live session; a failed re-solve of the
    same model must drop it, or gradient() would silently answer from
    the stale factorization."""
    m = pyo.ConcreteModel()
    m.p = pyo.Param(initialize=2.0, mutable=True)
    m.x = pyo.Var(initialize=1.0)
    m.y = pyo.Var(initialize=1.0)
    m.c = pyo.Constraint(expr=m.x + m.y == m.p)
    m.obj = pyo.Objective(expr=(m.x - 1) ** 2 + (m.y - 0.5) ** 2)
    declare_sens_param(m.p)
    pyo.SolverFactory("pounce").solve(m)
    assert gradient(m.x, wrt=m.p) is not None      # session is live

    m.bad = pyo.Constraint(expr=m.x == m.x + 1.0)  # 0 == 1
    pyo.SolverFactory("pounce").solve(m)
    with pytest.raises(RuntimeError, match="no sensitivity session"):
        gradient(m.x, wrt=m.p)


def test_all_three_declarations_coexist():
    """declare_sens_param + declare_fitted + declare_residual on one
    model: one solve serves both the sensitivity and the covariance
    queries from the same held factorization."""
    import numpy as np
    from pyomo_pounce import declare_fitted, declare_residual, covariance
    rng = np.random.default_rng(7)
    tt = np.linspace(0, 3, 20)
    y = 2.0 * np.exp(-1.3 * tt) + 0.05 * rng.standard_normal(20)
    m = pyo.ConcreteModel()
    m.I = pyo.RangeSet(0, 19)
    m.shift = pyo.Param(initialize=0.0, mutable=True)
    m.A = pyo.Var(initialize=1.5)
    m.k = pyo.Var(initialize=1.0)
    m.r = pyo.Var(m.I, initialize=0.0)
    m.res = pyo.Constraint(m.I, rule=lambda mm, i: mm.r[i]
                           == float(y[i]) + mm.shift
                           - mm.A * pyo.exp(-mm.k * float(tt[i])))
    m.obj = pyo.Objective(expr=sum(m.r[i] ** 2 for i in m.I))
    declare_sens_param(m.shift)
    declare_fitted(m.A, m.k)
    declare_residual(m.r)
    pyo.SolverFactory("pounce").solve(m)

    g = gradient(m.A, wrt=m.shift)          # sensitivity family
    assert np.isfinite(g) and g != 0.0
    cov = covariance(m)                     # estimation family
    assert cov.std_err[m.A] > 0 and cov.std_err[m.k] > 0
    assert abs(cov.correlation[m.A, m.k]) < 1.0


# ── declared Params in variable bounds (jkitchin/pounce#356) ─────────────────
#
# SensitivityInterface substitutes declared Params in constraint expressions
# only, so a Param left in a bound used to be written to the NL file as a
# constant and its sensitivity read as exactly zero. sens_solve now rewrites
# such a bound as a constraint over the substituted Var. Each case below is
# a limit the objective drives the variable straight into, so dx/dp is known
# in closed form.

def _bounded(ub=None, lb=None, sense=-1.0, x0=1.0):
    m = pyo.ConcreteModel()
    m.p = pyo.Param(initialize=2.0, mutable=True)
    m.x = pyo.Var(bounds=(lb(m) if lb else None, ub(m) if ub else None),
                  initialize=x0)
    m.obj = pyo.Objective(expr=sense * m.x)
    declare_sens_param(m.p)
    pyo.SolverFactory("pounce").solve(m)
    return m


@pytest.mark.parametrize("ub, dxdp, pnew, xnew", [
    (lambda m: m.p, 1.0, 3.0, 3.0),            # bare Param
    (lambda m: 2 * m.p + 1, 2.0, 3.0, 7.0),    # expression around it
])
def test_upper_bound_on_declared_param_is_differentiable(ub, dxdp, pnew, xnew):
    m = _bounded(ub=ub)
    assert gradient(m.x, wrt=m.p) == pytest.approx(dxdp, abs=1e-6)
    assert estimate(m, [(m.p, pnew)])[m.x] == pytest.approx(xnew, abs=1e-6)


def test_lower_bound_on_declared_param_is_differentiable():
    m = _bounded(lb=lambda m: m.p, sense=1.0, x0=3.0)
    assert gradient(m.x, wrt=m.p) == pytest.approx(1.0, abs=1e-6)
    assert estimate(m, [(m.p, 3.0)])[m.x] == pytest.approx(3.0, abs=1e-6)


def test_bound_as_constraint_is_unchanged():
    """The already-working spelling keeps working and agrees with the bound."""
    m = pyo.ConcreteModel()
    m.p = pyo.Param(initialize=2.0, mutable=True)
    m.x = pyo.Var(bounds=(0, None), initialize=1.0)
    m.cap = pyo.Constraint(expr=m.x <= m.p)
    m.obj = pyo.Objective(expr=-m.x)
    declare_sens_param(m.p)
    pyo.SolverFactory("pounce").solve(m)
    assert gradient(m.x, wrt=m.p) == pytest.approx(1.0, abs=1e-6)
    assert estimate(m, [(m.p, 3.0)])[m.x] == pytest.approx(3.0, abs=1e-6)


def test_undeclared_param_in_bound_is_left_alone():
    """Only declared Params are rewritten; an undeclared one stays a bound.

    Asserted on the clone that was actually solved, via the NL bounds the
    session carries. Asserting on the original model would pass no matter
    what, since the rewrite runs on a clone and never touches it.
    """
    m = pyo.ConcreteModel()
    m.p = pyo.Param(initialize=2.0, mutable=True)
    m.cap = pyo.Param(initialize=5.0, mutable=True)   # never declared
    m.x = pyo.Var(bounds=(0, m.cap), initialize=1.0)
    m.obj = pyo.Objective(expr=-m.x)                  # drive x onto the cap
    declare_sens_param(m.p)
    pyo.SolverFactory("pounce").solve(m)

    assert m.x.ub == 5.0                              # original untouched
    session = m.__dict__["_pounce_sens"].session
    r = session.var_entry(m.x.name)
    assert session.nl.x_u[r] == pytest.approx(5.0)    # survived into the solve
    assert session.moved_bounds == {}                 # nothing was rewritten
    assert pyo.value(m.x) == pytest.approx(5.0, abs=1e-6)


def test_moved_bound_is_recorded_for_covariance():
    """A rewritten bound reads as the NL no-bound sentinel, so the value it
    had is recorded; covariance()'s projection reads that instead."""
    m = pyo.ConcreteModel()
    m.p = pyo.Param(initialize=2.0, mutable=True)
    m.x = pyo.Var(bounds=(0, m.p), initialize=1.0)
    m.obj = pyo.Objective(expr=-m.x)
    declare_sens_param(m.p)
    pyo.SolverFactory("pounce").solve(m)

    session = m.__dict__["_pounce_sens"].session
    r = session.var_entry(m.x.name)
    assert session.nl.x_u[r] >= 1e19                  # NL no-bound sentinel
    assert session.moved_bounds["x"] == (None, pytest.approx(2.0))


def test_fixed_var_bound_is_not_rewritten():
    """A fixed Var's bounds are not enforced, so turning one into a
    constraint would impose a restriction the original model never had."""
    m = pyo.ConcreteModel()
    m.p = pyo.Param(initialize=2.0, mutable=True)
    m.f = pyo.Var(bounds=(0, m.p), initialize=5.0)
    m.f.fix(5.0)                                  # legally outside its bound
    m.x = pyo.Var(bounds=(0, None), initialize=1.0)
    m.c = pyo.Constraint(expr=m.x <= m.p)
    m.obj = pyo.Objective(expr=-m.x)
    declare_sens_param(m.p)
    res = pyo.SolverFactory("pounce").solve(m)
    assert res.solver.termination_condition == pyo.TerminationCondition.optimal
    assert gradient(m.x, wrt=m.p) == pytest.approx(1.0, abs=1e-6)


def test_deactivated_block_var_is_not_rewritten():
    """Stripping a bound on a dead Block and adding an active constraint
    would pull the Var into the NL as a free column."""
    m = pyo.ConcreteModel()
    m.p = pyo.Param(initialize=2.0, mutable=True)
    m.x = pyo.Var(bounds=(0, None), initialize=1.0)
    m.c = pyo.Constraint(expr=m.x <= m.p)
    m.obj = pyo.Objective(expr=-m.x)
    m.dead = pyo.Block()
    m.dead.y = pyo.Var(bounds=(0, m.p), initialize=1.0)
    m.dead.c = pyo.Constraint(expr=m.dead.y <= 1)
    m.dead.deactivate()
    declare_sens_param(m.p)
    pyo.SolverFactory("pounce").solve(m)

    names = m.__dict__["_pounce_sens"].session.var_names
    assert not any("dead.y" in n for n in names)


def test_indexed_var_bound_on_declared_param():
    m = pyo.ConcreteModel()
    m.I = pyo.RangeSet(3)
    m.p = pyo.Param(initialize=2.0, mutable=True)
    m.x = pyo.Var(m.I, bounds=(0, m.p), initialize=1.0)
    m.obj = pyo.Objective(expr=-sum(m.x[i] for i in m.I))
    declare_sens_param(m.p)
    pyo.SolverFactory("pounce").solve(m)
    for i in m.I:
        assert gradient(m.x[i], wrt=m.p) == pytest.approx(1.0, abs=1e-6)


def test_rewritten_bound_still_projects_in_covariance():
    """A fitted Var capped by a declared Param still triggers covariance()'s
    active-bound projection, even though its NL bound is now the no-bound
    sentinel (jkitchin/pounce#357 review)."""
    from pyomo_pounce import covariance, declare_fitted, declare_residual
    m = pyo.ConcreteModel()
    m.I = pyo.RangeSet(5)
    m.cap = pyo.Param(initialize=1.0, mutable=True)
    m.A = pyo.Var(bounds=(0, m.cap), initialize=0.5)
    m.r = pyo.Var(m.I, initialize=0.0)
    m.res = pyo.Constraint(m.I, rule=lambda mm, i: mm.r[i] == 3.0 - mm.A)
    m.obj = pyo.Objective(expr=sum(m.r[i] ** 2 for i in m.I))
    declare_sens_param(m.cap)
    declare_fitted(m.A)
    declare_residual(m.r)
    pyo.SolverFactory("pounce").solve(m)

    session = m.__dict__["_pounce_sens"].session
    assert session.nl.x_u[session.var_entry(m.A.name)] >= 1e19
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        covariance(m)
    assert any("sits on its bound" in str(x.message) for x in w)
    # and the projected answer, not just the warning: identical to what the
    # same model gives with the bound left alone
    assert covariance(m).std_err[m.A] == 0.0


def test_both_bounds_rewritten_are_both_recorded():
    """A Var with both bounds on declared Params records both sides; the
    single-sided tests pass even if the merge dropped one."""
    m = pyo.ConcreteModel()
    m.lo = pyo.Param(initialize=-1.0, mutable=True)
    m.hi = pyo.Param(initialize=2.0, mutable=True)
    m.x = pyo.Var(bounds=(m.lo, m.hi), initialize=0.0)
    m.obj = pyo.Objective(expr=-m.x)
    declare_sens_param(m.lo, m.hi)
    pyo.SolverFactory("pounce").solve(m)

    session = m.__dict__["_pounce_sens"].session
    lo_rec, hi_rec = session.moved_bounds[m.x.name]
    assert lo_rec == pytest.approx(-1.0)
    assert hi_rec == pytest.approx(2.0)
    assert gradient(m.x, wrt=m.hi) == pytest.approx(1.0, abs=1e-6)
