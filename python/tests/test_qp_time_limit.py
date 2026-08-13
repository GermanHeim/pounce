"""Wall-clock budgets on the Python convex entry points (gh #585).

``QpOptions::time_limit`` reached the CLI (``max_wall_time``) but nothing in
Python could ask a convex solve to stop at a wall-clock budget, even though
``QpResult.status`` could already come back ``"time_limit"``. These cover the
``time_limit=`` keyword on the four surfaces that take one — ``solve_qp``,
``solve_socp``, ``solve_qp_batch``, ``solve_qp_multi_rhs`` — plus the two
properties that make the option safe to hand a caller:

* a give-up at the budget is reported as ``"time_limit"``, never as a wrong
  ``"optimal"`` (the reason this waited on the #583 hardening, where a deadline
  crossing inside inertia control used to return an unsolved KKT right-hand
  side labelled ``optimal``); and
* the budget is **per instance** on the batched paths.

``time_limit=0.0`` is used wherever a *deterministic* expiry is needed: it is a
real, immediate deadline, so the solve stops before doing any work regardless of
how fast the machine is. Tests that need a solve to actually finish use a
generous budget instead of relying on timing.
"""

import numpy as np
import pytest

from pounce import _pounce
from pounce.qp import solve_qp, solve_qp_batch, solve_qp_multi_rhs, solve_socp

# A verdict, if one is reported, means the solve proved something; only a
# give-up is allowed to be relabelled by the clock.
VERDICTS = {"optimal", "optimal_inaccurate", "primal_infeasible", "dual_infeasible"}


def box_qp(c=(-3.0, -4.0)):
    """min ‖x‖² + cᵀx  s.t.  0 ≤ x ≤ 1 — the module's stock tiny QP."""
    n = len(c)
    return dict(P=np.eye(n) * 2.0, c=list(c), lb=[0.0] * n, ub=[1.0] * n)


def random_qp(n=120, m=60, seed=0):
    """A non-trivial dense convex QP: strictly convex `P`, `m` inequality rows,
    and a feasible box. Big enough that the solve is real work rather than a
    couple of arithmetic ops, so a budget has something to interrupt."""
    rng = np.random.default_rng(seed)
    M = rng.standard_normal((n, n))
    P = M.T @ M + np.eye(n)  # symmetric positive definite
    c = rng.standard_normal(n)
    G = rng.standard_normal((m, n))
    h = np.abs(rng.standard_normal(m)) + 1.0  # x = 0 is strictly feasible
    return dict(P=P, c=c, G=G, h=h, lb=[-10.0] * n, ub=[10.0] * n)


def test_zero_budget_stops_the_solve():
    assert solve_qp(**box_qp(), time_limit=0.0).status == "time_limit"


def test_none_is_unbounded():
    # The default and an explicit `None` both mean "no deadline at all" — not a
    # huge one — so the solve runs to its verdict.
    assert solve_qp(**box_qp()).status == "optimal"
    assert solve_qp(**box_qp(), time_limit=None).status == "optimal"


def test_a_generous_budget_does_not_disturb_the_answer():
    free = solve_qp(**box_qp())
    bounded = solve_qp(**box_qp(), time_limit=60.0)
    assert bounded.status == "optimal"
    np.testing.assert_allclose(bounded.x, free.x, rtol=1e-9, atol=1e-9)


def test_active_set_method_honors_the_budget():
    # The budget is a property of the convex driver, not of one engine: the
    # `pounce-qp` active-set route scopes the same deadline.
    r = solve_qp(**box_qp(), method="active-set", time_limit=0.0)
    assert r.status == "time_limit"


def test_a_cancelled_solve_is_never_a_wrong_optimal():
    """The property that makes the option safe: under a budget too small to
    finish, the status reported is a give-up, not a verdict backed by an
    iterate that is nowhere near a KKT point.

    Written as an implication rather than a hard `== "time_limit"` so it cannot
    flake on a fast machine: whatever status comes back, an `optimal` one must
    be backed by a genuinely small KKT error.
    """
    prob = random_qp()
    r = solve_qp(**prob, time_limit=1e-4)
    assert r.status in VERDICTS | {"time_limit", "iteration_limit"}
    if r.status in ("optimal", "optimal_inaccurate"):
        assert r.kkt_error is not None and r.kkt_error < 1e-3
    # And with no budget at all the same problem solves cleanly, so the
    # implication above is not passing vacuously on a broken problem.
    assert solve_qp(**prob).status == "optimal"


def test_socp_takes_a_budget():
    # min t s.t. (t, x − x*) ∈ SOC(3): a second-order cone, so this exercises
    # the conic driver rather than the orthant one.
    socp = dict(c=[1.0, 0.0, 0.0], G=-np.eye(3), h=[0.0, -2.0, 1.0])
    assert solve_socp(**socp, cones=[("soc", 3)], time_limit=0.0).status == "time_limit"
    assert solve_socp(**socp, cones=[("soc", 3)]).status == "optimal"


def test_batch_budget_is_per_instance():
    problems = [box_qp((-3.0, -4.0)), box_qp((-1.0, -2.0)), box_qp((1.0, 1.0))]

    stopped = solve_qp_batch(problems, time_limit=0.0)
    assert [r.status for r in stopped] == ["time_limit"] * len(problems)

    # Each instance opens its own deadline scope, so a budget that is ample for
    # one problem is ample for all of them however many there are — the whole
    # point of the per-instance reading (a shared clock would make *which*
    # instances survive depend on rayon's scheduling).
    solved = solve_qp_batch(problems * 4, time_limit=60.0)
    assert all(r.status == "optimal" for r in solved)
    assert len(solved) == 4 * len(problems)


def test_multi_rhs_takes_a_budget():
    cs = [[-3.0, -4.0], [-1.0, 0.5], [2.0, 2.0]]
    base = dict(P=np.eye(2) * 2.0, lb=[0.0, 0.0], ub=[1.0, 1.0])

    stopped = solve_qp_multi_rhs(**base, cs=cs, time_limit=0.0)
    assert [r.status for r in stopped] == ["time_limit"] * len(cs)

    solved = solve_qp_multi_rhs(**base, cs=cs, time_limit=60.0)
    assert all(r.status == "optimal" for r in solved)


@pytest.mark.parametrize(
    "bad",
    [-1.0, -1e-12, float("nan"), float("inf"), float("-inf"), "1.0", True, [1.0]],
)
@pytest.mark.parametrize(
    "call",
    [
        lambda **kw: solve_qp(**box_qp(), **kw),
        lambda **kw: solve_socp(
            c=[1.0], G=[[-1.0]], h=[0.0], cones=[("nonneg", 1)], **kw
        ),
        lambda **kw: solve_qp_batch([box_qp()], **kw),
        lambda **kw: solve_qp_multi_rhs(P=np.eye(2) * 2.0, cs=[[-1.0, -1.0]], **kw),
    ],
    ids=["solve_qp", "solve_socp", "solve_qp_batch", "solve_qp_multi_rhs"],
)
def test_invalid_budgets_raise_a_named_error(call, bad):
    # `inf` is rejected rather than read as "no limit": `None` is how that is
    # spelled, and silently accepting both spellings hides a typo'd budget.
    with pytest.raises(ValueError, match="time_limit"):
        call(time_limit=bad)


def test_the_binding_itself_rejects_a_bad_budget():
    # The host wrapper validates first, but `_pounce` is importable directly, so
    # the PyO3 layer must not hand a negative/non-finite value to `Duration`.
    prob = _pounce.QpProblem(n=1, c=[1.0], lb=[0.0], ub=[1.0])
    for bad in (-1.0, float("nan"), float("inf"), 1e300):
        with pytest.raises(ValueError, match="time_limit"):
            _pounce.solve_qp(prob, time_limit=bad)
