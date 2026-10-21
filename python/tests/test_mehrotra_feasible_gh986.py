"""gh#986 item 5: ``mehrotra_algorithm=yes`` on a general NLP (Mittelmann's
clnlbeam) ended ``Restoration_Failed`` after three iterations *at a feasible
point* (``inf_pr`` ~ 1e-14).

Root cause: the probing oracle's first step picks sigma ~ 0 (mu ~ 1e-11) while
the iterate's complementarity is still ~5e-2, so the iterate-quality guard
(gh#58) fires with a ratio of ~2e9 and requested restoration -- which has
nothing to repair at a feasible point and fails on entry. At a feasible
iterate the guard now recentres (LOQO mu from the iterate's own
complementarity) and restoration stays what it was for an infeasible one.

Mehrotra remains unglobalised, so on this nonconvex model it may land on a
different local solution than the default algorithm; the contract pinned here
is the one the issue asked for: no restoration from a feasible iterate, and a
verdict that is a real solve.
"""

import numpy as np
import pytest

import pounce
from pounce import NlExpr as E


def clnlbeam(ni, alpha=350.0):
    h = 1.0 / ni
    N = ni + 1
    v = E.vars(3 * N)
    t, x, u = v[:N], v[N:2 * N], v[2 * N:]
    obj = E.sum([0.5 * h * (u[i + 1] ** 2 + u[i] ** 2)
                 + 0.5 * alpha * h * (E.cos(t[i + 1]) + E.cos(t[i])) for i in range(ni)])
    cons = []
    for i in range(ni):
        cons.append(x[i + 1] - x[i] - 0.5 * h * (E.sin(t[i + 1]) + E.sin(t[i])))
        cons.append(t[i + 1] - t[i] - 0.5 * h * (u[i + 1] + u[i]))
    for end in (0, ni):
        cons += [x[end] + 0.0, t[end] + 0.0]
    s0 = 0.05 * np.cos(np.arange(N) * h * np.pi)
    x0 = np.r_[s0, s0, np.zeros(N)]
    xl = np.r_[-np.ones(N), -0.05 * np.ones(N), -1e19 * np.ones(N)]
    xu = np.r_[np.ones(N), 0.05 * np.ones(N), 1e19 * np.ones(N)]
    return pounce.build_nl_problem(
        n=3 * N, objective=obj, constraints=cons, g_l=[0.0] * len(cons),
        g_u=[0.0] * len(cons), x0=x0, x_l=xl, x_u=xu,
    )


#: The two stationary values of the beam problem that runs reach (default
#: algorithm: 344.876 at every ni measured; unglobalised Mehrotra also stops at
#: 346.496 on some ni -- measured 1000 and 2000).
_BEAM_LOCAL_VALUES = (344.876, 346.496)


def _solve(ni, **opts):
    (x, info), = pounce.solve_nlp_batch(
        [clnlbeam(ni)], options={"print_level": 0, **opts}, parallel=False,
    )
    return info


@pytest.mark.parametrize("ni", [1000, 2000])
def test_mehrotra_clnlbeam_is_not_sent_to_restoration_at_a_feasible_point(ni):
    info = _solve(ni, mehrotra_algorithm="yes")
    assert info["status_msg"] != "Restoration_Failed"
    assert info["status_msg"] == "Solve_Succeeded"
    assert info["iter_count"] > 3
    assert info["final_constr_viol"] < 1e-6
    # A genuine stationary point of the beam problem, one of the two the runs
    # reach (gh#986 review: the old 300..360 band admitted anything).
    assert min(abs(info["obj_val"] - v) for v in _BEAM_LOCAL_VALUES) < 1e-2, info["obj_val"]


def test_mehrotra_clnlbeam_ends_at_a_stationary_value_at_ni_500():
    """gh#986 review item 8 (the stale predictor) at ni = 500.

    This test used to assert that dropping the stale predictor moves ni = 500
    from the other local solution (346.4972) to the default algorithm's
    344.8762. That was true on FERAL 0.20, and it is a statement about which
    of two nearby stationary points unglobalised Mehrotra falls into, which
    is a rounding fact, not a property of the fix. FERAL has separate NEON
    and AVX2+FMA dense kernels, and on FERAL 0.21 the outcome flips with the
    architecture and the size. Measured, Mehrotra with the fix against
    Mehrotra with both ``delta_aff = None`` drops reverted:

    ===== ================== ================== ================== ==================
    ni    arm64 fixed        arm64 stale        x86_64 fixed       x86_64 stale
    ===== ================== ================== ================== ==================
    450   344.8761           346.4977           346.4977           344.8765
    500   346.4972           346.4972           346.4972           (not measured)
    600   344.8761           346.4968           344.8761           344.8761
    700   344.8761           346.4965           346.4965           344.8761
    ===== ================== ================== ================== ==================

    No size separates the two on both architectures, so no end-to-end witness
    of the fix exists here. The fix itself is pinned deterministically by
    ``accept_trial_point_drops_the_affine_step`` in
    ``crates/pounce-algorithm/src/ipopt_data.rs``. What this test keeps is
    the platform-independent contract: at ni = 500 Mehrotra solves and ends
    at one of the beam problem's two stationary values."""
    info = _solve(500, mehrotra_algorithm="yes")
    assert info["status_msg"] == "Solve_Succeeded"
    assert min(abs(info["obj_val"] - v) for v in _BEAM_LOCAL_VALUES) < 1e-2, info["obj_val"]


def test_default_algorithm_unchanged_on_clnlbeam():
    (x, info), = pounce.solve_nlp_batch(
        [clnlbeam(200)], options={"print_level": 0}, parallel=False
    )
    assert info["status_msg"] == "Solve_Succeeded"
    assert info["obj_val"] == pytest.approx(344.88, abs=0.05)
