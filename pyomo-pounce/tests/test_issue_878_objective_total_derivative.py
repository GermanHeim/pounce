"""gh#878 — ``sens_jacobian(of=<Objective>)`` returns the total derivative df/dp.

``of=`` a Var gives ``dx/dp`` and ``of=`` an equality Constraint gives
``dlambda/dp``; the objective used to be rejected outright, so there was no
route to ``df/dp`` at all.

    df/dp  =  df/dp|_x  +  sum_i (df/dx_i)(dx_i/dp)

Both halves fall out of one contraction, because ``declare_sens_param``
rewrites a declared parameter into a variable pinned by a defining equality.
``p`` is therefore an ordinary coordinate of full-x: the objective gradient
carries ``df/dp|_x`` in ``p``'s slot and the step carries ``dp/dp = 1`` there.

WHY THESE TESTS LIVE HERE AND NOT IN ``sens_invariance_legs.rs``.  gh#878's
test note asks for a row in each of the three Rust invariance legs.  The
accessor did not land in the Rust crate: ``pounce-sensitivity``'s ``Solver``
exposes no objective gradient, and the issue scopes the proposal to
``sens_jacobian``'s ``of=``, which is the Pyomo layer.  Putting the rows in the
Rust legs would test a code path that does not exist.  So the three
*dimensions* those legs defend are covered here instead, against the code that
actually runs, and each test below names the leg it stands in for.  If a Rust
``df/dp`` accessor is ever added, it needs its own rows there — these do not
cover it.

ORACLES.  Every expected value here is computed **without a solver** — in
closed form, or by hand — so none of it is a number POUNCE produced.  A step
that is self-consistently wrong is the class with the worst blast radius, and
checking POUNCE against POUNCE cannot see it.  Each fixture below carries the
derivation of its own answer.

A live central-difference re-solve through **Ipopt** runs on top of that where
Ipopt is installed, as an independent confirmation that the closed forms
describe the model POUNCE actually solved.  It is skipped, not required: the
`wheel smoke (pyomo-pounce)` CI job has no Ipopt, and a test that can only run
where a third-party binary happens to exist is not a regression guard.
"""

from __future__ import annotations

import warnings

import pytest

pyo = pytest.importorskip("pyomo.environ")

np = pytest.importorskip("numpy")

from pyomo_pounce import declare_sens_param, sens_jacobian  # noqa: E402
from pyomo_pounce import sens as _sens  # noqa: E402

TOL = 1e-7


def _have_ipopt():
    try:
        return bool(pyo.SolverFactory("ipopt").available(False))
    except Exception:
        return False


requires_ipopt = pytest.mark.skipif(
    not _have_ipopt(), reason="the live re-solve cross-check needs Ipopt"
)


def _solve(m, tol=1e-10):
    pyo.SolverFactory("pounce").solve(m, options={"tol": tol})
    return m


def _fd(build, pname, p0, h=1e-6):
    """Central difference of a re-solve, through Ipopt. Gated by
    `requires_ipopt` at every call site."""

    def f(v):
        m = build(**{pname: v})
        pyo.SolverFactory("ipopt").solve(m, options={"tol": 1e-12})
        return pyo.value(m.obj)

    return (f(p0 + h) - f(p0 - h)) / (2 * h)


def _fd_closed(f, i, args, h=1e-6):
    """Central difference of a CLOSED-FORM objective. No solver involved."""
    a, b = list(args), list(args)
    a[i] += h
    b[i] -= h
    return (f(*a) - f(*b)) / (2 * h)


def f_implicit_closed(p1, p2):
    """`implicit_only`'s optimal objective in closed form.

    It is a least-norm problem, `min |x|^2` subject to `A(p2) x = b(p1)`, so
    `x* = A'(AA')^-1 b` and `f* = b'(AA')^-1 b` — no solver, and it agrees
    with a solved value to 1e-16.
    """
    A = np.array([[6.0, 3.0, 2.0], [p2, 1.0, -1.0]])
    b = np.array([p1, 1.0])
    return float(b @ np.linalg.solve(A @ A.T, b))


def f_fixed_ahead_closed(p):
    """`fixed_ahead`'s optimal objective in closed form.

    With `u = x1/s`, `v = x2/s` and `x0` pinned at 0.75 the model is
    `min (u-p)^2 + 2(v-1)^2 + 2.25 p` subject to `u + v = 4.25`, whose
    multiplier is `lam = (4/3)(p - 3.25)`, giving `u - p = -lam/2` and
    `v - 1 = -lam/4`. Substituting collapses it to

        f*(p) = (2/3)(p - 3.25)^2 + 2.25 p      =>   df/dp = (4/3)(p - 3.25) + 2.25

    which is 7/12 at p = 2, and independent of `s` — the scaling leg's point.
    """
    return (2.0 / 3.0) * (p - 3.25) ** 2 + 2.25 * p


# --------------------------------------------------------------------------
# Fixtures, chosen to reach DIFFERENT branches of the chain rule.
# --------------------------------------------------------------------------
def implicit_only(p1=4.5, p2=1.0):
    """The parameter reaches the objective only through the solution.

    ``df/dp|_x`` is identically zero here, so this fixture exercises the
    ``sum_i (df/dx_i)(dx_i/dp)`` half and says nothing about the other one.
    Pyomo's own ``sensitivity_toolbox`` example.
    """
    m = pyo.ConcreteModel()
    m.x1 = pyo.Var(initialize=0.15)
    m.x2 = pyo.Var(initialize=0.15)
    m.x3 = pyo.Var(initialize=0.0)
    m.p1 = pyo.Param(initialize=p1, mutable=True)
    m.p2 = pyo.Param(initialize=p2, mutable=True)
    m.obj = pyo.Objective(expr=m.x1**2 + m.x2**2 + m.x3**2)
    m.c1 = pyo.Constraint(expr=6 * m.x1 + 3 * m.x2 + 2 * m.x3 - m.p1 == 0)
    m.c2 = pyo.Constraint(expr=m.p2 * m.x1 + m.x2 - m.x3 - 1 == 0)
    return m


def explicit_partial(p=2.0):
    """The parameter is in the objective, and the implicit half vanishes.

    ``min (x1 - p)^2 + 3 p^2`` subject to ``x1 + x2 == 5`` puts ``x1`` at
    ``p``, so ``df/dx1 = 2(x1 - p) = 0`` and the entire answer is the explicit
    partial: ``f* = 3 p^2``, ``df/dp = 6 p = 12``.

    This is the discriminator for the half `implicit_only` cannot see. An
    implementation that summed only ``(df/dx_i)(dx_i/dp)`` returns **0** here
    — the right shape, the wrong number, and nothing about it looks wrong.
    """
    m = pyo.ConcreteModel()
    m.x1 = pyo.Var(initialize=0.3)
    m.x2 = pyo.Var(initialize=0.3)
    m.p = pyo.Param(initialize=p, mutable=True)
    m.obj = pyo.Objective(expr=(m.x1 - m.p) ** 2 + 3 * m.p**2)
    m.c = pyo.Constraint(expr=m.x1 + m.x2 == 5)
    return m


def fixed_ahead(p=2.0, s=1.0):
    """A FIXED variable declared ahead of everything else (leg 3's shape).

    ``x0`` has ``lb == ub``, so the default ``fixed_variable_treatment =
    make_parameter`` drops it from the solve and full-x and var-x stop
    agreeing from that column on. ``df/dp`` is a scalar contracted over the
    whole step, so reading a full-x gradient against a var-x step pairs every
    ``df/dx_i`` with a NEIGHBOURING variable's sensitivity — gh#450 and gh#672
    finding 1, the same defect twice.

    ``s`` rescales the free variables for the scaling leg below.
    """
    m = pyo.ConcreteModel()
    m.x0 = pyo.Var(bounds=(0.75, 0.75), initialize=0.75)
    m.x1 = pyo.Var(initialize=0.3)
    m.x2 = pyo.Var(initialize=0.3)
    m.p = pyo.Param(initialize=p, mutable=True)
    m.obj = pyo.Objective(
        expr=(m.x1 / s - m.p) ** 2 + 2 * (m.x2 / s - 1.0) ** 2 + 3 * m.x0 * m.p
    )
    m.c = pyo.Constraint(expr=m.x1 / s + m.x2 / s + m.x0 == 5)
    return m


# --------------------------------------------------------------------------
# The accessor itself, on both branches of the chain rule.
# --------------------------------------------------------------------------
def test_the_implicit_half_matches_a_finite_difference_resolve():
    m = implicit_only()
    declare_sens_param(m.p1, m.p2)
    _solve(m)
    for pd, name, i in ((m.p1, "p1", 0), (m.p2, "p2", 1)):
        got = sens_jacobian(m.obj, wrt=pd)
        want = _fd_closed(f_implicit_closed, i, (4.5, 1.0))
        assert abs(got - want) < TOL, f"df/d{name}: {got} vs closed form {want}"


def test_the_explicit_partial_is_not_dropped():
    """The half `implicit_only` cannot see. Answer is 12; dropping the
    explicit partial gives 0."""
    m = explicit_partial()
    declare_sens_param(m.p)
    _solve(m)
    assert pyo.value(m.x1) == pytest.approx(2.0, abs=1e-7), "x1 sits at p"
    got = sens_jacobian(m.obj, wrt=m.p)
    assert got == pytest.approx(12.0, abs=1e-6), (
        f"df/dp = 6p = 12 here, got {got}. 0.0 is the answer an "
        "implementation that summed only (df/dx_i)(dx_i/dp) returns."
    )
    # and the implicit half really is zero, so the test above is not passing
    # for the wrong reason
    dx1 = sens_jacobian(m.x1, wrt=m.p)
    implicit = 2.0 * (pyo.value(m.x1) - 2.0) * dx1
    assert abs(implicit) < 1e-6, f"the implicit half should vanish, got {implicit}"


# --------------------------------------------------------------------------
# Leg 1 (scaling): df/dp is a physical quantity and must not move under a
# change of variables. `variable_scaling_sensitivity.rs` makes the same point
# about the classifier.
# --------------------------------------------------------------------------
@pytest.mark.parametrize("s", [1.0, 1e3, 1e-3])
def test_leg_scaling_the_total_derivative_is_unmoved_by_a_change_of_variables(s):
    m = fixed_ahead(s=s)
    declare_sens_param(m.p)
    _solve(m)
    got = sens_jacobian(m.obj, wrt=m.p)
    want = _fd_closed(f_fixed_ahead_closed, 0, (2.0,))
    assert abs(got - want) < 1e-6, (
        f"scale s={s:g}: df/dp = {got} against the closed form's {want}. "
        "Rescaling the variables rewrites the same problem; the derivative "
        "of the objective with respect to a parameter is not a scaled "
        "quantity and must not move."
    )


# --------------------------------------------------------------------------
# Leg 2 (perturbation magnitude): `sens_jacobian` takes no step, so the
# analogue is that the derivative agrees with the re-solve across the range of
# differencing widths where the difference is meaningful. gh#672 finding 4 put
# an absolute tolerance on a quantity that scales with the perturbation.
# --------------------------------------------------------------------------
@pytest.mark.parametrize("h", [1e-2, 1e-3, 1e-4, 1e-5, 1e-6])
def test_leg_magnitude_the_derivative_is_the_limit_of_the_difference_quotient(h):
    m = explicit_partial()
    declare_sens_param(m.p)
    _solve(m)
    got = sens_jacobian(m.obj, wrt=m.p)
    want = _fd_closed(lambda p: 3.0 * p * p, 0, (2.0,), h=h)
    assert abs(got - want) < 1e-5, f"h={h:g}: {got} vs {want}"


# --------------------------------------------------------------------------
# Leg 3 (a fixed variable ahead of the parameter). The one gh#878 called out
# as most likely to catch an error here.
# --------------------------------------------------------------------------
def test_leg_fixed_the_index_spaces_actually_diverge():
    """Without this the leg below could pass on a model where full-x and
    var-x coincide, which is every model that has no fixed variable."""
    m = fixed_ahead()
    declare_sens_param(m.p)
    _solve(m)
    session = _sens._session_for(m.p)
    rows = session._primal_row_map()
    assert any(r is None for r in rows), (
        "no variable was removed from the solve, so full-x and var-x agree "
        f"and this fixture cannot detect a mis-indexed contraction: {rows}"
    )


def test_leg_fixed_the_total_derivative_survives_a_fixed_variable_ahead_of_it():
    m = fixed_ahead()
    declare_sens_param(m.p)
    _solve(m)
    got = sens_jacobian(m.obj, wrt=m.p)
    want = _fd_closed(f_fixed_ahead_closed, 0, (2.0,))
    assert abs(got - want) < 1e-6, (
        f"df/dp = {got} against the closed form's {want}. A fixed variable ahead "
        "of the parameter shifts every later var-x column, so contracting a "
        "full-x gradient with a var-x step returns a neighbouring variable's "
        "sensitivity — plausible and wrong (gh#450, gh#672 finding 1)."
    )


# --------------------------------------------------------------------------
# Shape and refusal.
# --------------------------------------------------------------------------
def test_indexed_parameters_give_a_jacobian_with_an_objective_row():
    m = pyo.ConcreteModel()
    m.I = pyo.RangeSet(1, 2)
    m.q = pyo.Param(m.I, initialize={1: 1.0, 2: 2.0}, mutable=True)
    m.y = pyo.Var(m.I, initialize=0.5)
    m.obj = pyo.Objective(
        expr=sum((m.y[i] - m.q[i]) ** 2 for i in m.I) + m.q[1] * m.q[2]
    )
    m.c = pyo.Constraint(expr=m.y[1] + m.y[2] == 2.0)
    declare_sens_param(m.q)
    _solve(m)
    g = sens_jacobian(m.obj, wrt=m.q)
    # Worked by hand: y = (0.5, 1.5), so df/dq1 = -2(y1-q1) + q2 = 3 and
    # df/dq2 = -2(y2-q2) + q1 = 2, the implicit halves cancelling in both.
    assert g[m.obj, m.q[1]] == pytest.approx(3.0, abs=1e-7)
    assert g[m.obj, m.q[2]] == pytest.approx(2.0, abs=1e-7)
    df = g.to_dataframe()
    assert list(df.index) == ["obj"]
    assert df.loc["obj", "q[1]"] == pytest.approx(3.0, abs=1e-7)


def test_a_deactivated_objective_is_refused_by_name():
    """A script that switches formulations leaves the unused objective on the
    model. Answering for it with the solved objective's gradient, under the
    unused one's name, is the silent failure this refuses."""
    m = explicit_partial()
    m.unused = pyo.Objective(expr=m.x1)
    m.unused.deactivate()
    declare_sens_param(m.p)
    _solve(m)
    with pytest.raises(ValueError, match="not the active objective"):
        sens_jacobian(m.unused, wrt=m.p)


def test_an_undeclared_parameter_is_still_refused():
    """The objective route must not bypass the declaration check."""
    m = explicit_partial()
    m.q = pyo.Param(initialize=1.0, mutable=True)
    declare_sens_param(m.p)
    _solve(m)
    with pytest.raises(ValueError, match="declare_sens_param"):
        sens_jacobian(m.obj, wrt=m.q)


def test_the_gradient_is_evaluated_once_per_session():
    """`sens_jacobian(m.obj, wrt=<indexed param>)` asks for one column per
    member; the gradient does not depend on the column."""
    m = explicit_partial()
    declare_sens_param(m.p)
    _solve(m)
    session = _sens._session_for(m.p)
    assert session._obj_grad is None
    sens_jacobian(m.obj, wrt=m.p)
    first = session._obj_grad
    assert first is not None
    sens_jacobian(m.obj, wrt=m.p)
    assert session._obj_grad is first, "the gradient was re-evaluated"


def test_the_kink_warning_still_fires_for_an_objective_target():
    """A degenerate base point makes `df/dp` one-sided too, and the caller
    has to hear about it on this route as much as on the variable one."""
    m = pyo.ConcreteModel()
    m.p = pyo.Param(initialize=0.0, mutable=True)
    m.x = pyo.Var(bounds=(0.0, 10.0), initialize=0.5)
    m.obj = pyo.Objective(expr=(m.x - m.p) ** 2)
    declare_sens_param(m.p)
    _solve(m)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sens_jacobian(m.obj, wrt=m.p)
    assert any("degenerate" in str(w.message) for w in caught), (
        f"expected the kink warning, got {[str(w.message) for w in caught]}"
    )


# --------------------------------------------------------------------------
# The independent confirmation, where a third-party solver is available.
# --------------------------------------------------------------------------
@requires_ipopt
@pytest.mark.parametrize(
    "build, pname, p0",
    [
        (implicit_only, "p1", 4.5),
        (implicit_only, "p2", 1.0),
        (explicit_partial, "p", 2.0),
        (fixed_ahead, "p", 2.0),
    ],
)
def test_the_closed_forms_describe_the_model_pounce_actually_solved(build, pname, p0):
    """Confirms the hand-derived answers above against a live central
    difference of an **Ipopt** re-solve.

    This is what stops the closed forms from being a second opinion on my own
    algebra: if a fixture's model and its derivation ever drift apart, every
    assertion above would agree with the derivation and be wrong about the
    model. Skipped where Ipopt is absent, and nothing above depends on it.
    """
    m = build()
    declare_sens_param(getattr(m, pname))
    _solve(m)
    got = sens_jacobian(m.obj, wrt=getattr(m, pname))
    want = _fd(build, pname, p0)
    assert abs(got - want) < 1e-6, f"{build.__name__}/{pname}: {got} vs {want}"
