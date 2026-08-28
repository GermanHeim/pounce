"""gh #815: a square problem's feasible point is a solved answer everywhere.

``Feasible_Point_Found`` (``ApplicationReturnStatus`` code 6) has exactly one
producer in the engine — ``min_c_1nrm.rs`` returns
``RestorationOutcome::FeasiblePointFound``, reached only through
``square_feasible_point_found`` in ``resto_inner_solver.rs``, whose first
conjunct is ``is_square_problem`` (``c.x.dim() == c.y_c.dim()``, a port of
``IpoptCalculatedQuantities::IsSquareProblem``). On a square problem the
objective is constant, so a feasible point *is* the solution and there is no
further criterion left to miss. Ipopt's own ASL driver agrees, reporting AMPL
``solve_result_num = 2``.

gh #815 fixed the ``.sol`` band, the CLI exit code and both ``pyomo_pounce``
tables. This file pins the four *Python library* surfaces that were split off
as gh #820, and pins them against the Rust number rather than against a
constant repeated here — because the whole defect gh #815 fixed was a Python
consumer disagreeing with what the same solve had just written to disk.

**What these tests are not.** They are unit-level, deliberately. No fixture in
the CLI corpus reaches this exit, and 72 generated square models (three
nonlinearity families x three start points x three row-scale spreads, under
five option sets) failed to reach it — the status needs a model larger than the
corpora carry; gh #815's own is 536x536. So there is no end-to-end arm here,
and these tests are evidence about the *mapping*, not about the engine ever
producing the status.
"""

import re
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]

#: ``ApplicationReturnStatus::FeasiblePointFound`` (`return_codes.rs`).
FEASIBLE_POINT_FOUND = 6


def _rust_solve_result_num(variant: str) -> int:
    """Parse one arm out of ``status_to_solve_result_num``.

    Parsed, not hardcoded: a hardcoded copy would let the Python surfaces and
    the ``.sol`` band drift apart silently, which is precisely the failure
    gh #815 was.
    """
    src = (_REPO / "crates/pounce-solve-report/src/lib.rs").read_text()
    body = re.search(
        r"pub fn status_to_solve_result_num\(.*?\n\}", src, re.S
    )
    assert body, "could not locate status_to_solve_result_num in Rust"
    arm = re.search(rf"^\s*{variant} => (-?\d+),", body.group(0), re.M)
    assert arm, f"no arm for {variant}; the match format changed"
    return int(arm.group(1))


def test_the_rust_sol_band_is_the_premise_these_surfaces_rest_on():
    """Every assertion below assumes the ``.sol`` route calls this solve a
    success. Check the premise first, so that if the Rust side is ever reverted
    the failure names the real cause instead of surfacing as four unrelated
    mapping tests that now look wrong."""
    code = _rust_solve_result_num("FeasiblePointFound")
    assert code == 2, (
        "Ipopt's ASL driver emits 2 for FEASIBLE_POINT_FOUND; "
        f"pounce now emits {code}"
    )
    assert 0 <= code <= 99, (
        f"code {code} is outside the solved band. Pyomo's contrib v2 reader "
        "maps 100..199 to TerminationCondition.error, which is what gh #815 "
        "was."
    )


def test_minimize_counts_a_square_feasible_point_as_success():
    """gh #815 / gh #820. ``pounce.minimize`` judged code 6 a failure, so a
    square solve that the CLI would have written to a ``.sol`` as solved came
    back from the library call with ``success=False``."""
    from pounce._minimize import _NLP_SUCCESS_STATUS

    assert FEASIBLE_POINT_FOUND in _NLP_SUCCESS_STATUS
    # The neighbours it used to be lumped with stay failures. Code 5
    # (User_Requested_Stop) is an external abort, not a solve verdict.
    for code in (2, 3, 4, 5):
        assert code not in _NLP_SUCCESS_STATUS, f"code {code} must stay a failure"


def test_curve_fit_moves_with_minimize():
    """``_curve_fit`` imports the same frozenset rather than keeping its own, so
    this is a guard against someone giving it a local copy — the two entry
    points must not be able to disagree about what a converged solve is."""
    from pounce._curve_fit import _NLP_SUCCESS_STATUS as cf
    from pounce._minimize import _NLP_SUCCESS_STATUS as m

    assert cf is m, "curve_fit must reuse minimize's set, not copy it"


def test_race_candidates_treat_a_square_feasible_point_as_finished():
    """``_starts._DONE_STATUS`` is not a success flag — it answers "would more
    budget be spent?". A square feasible point is finished either way, and
    while code 6 sat outside this set *and* outside ``_PAUSED_STATUS`` a
    candidate that solved was scored ``solve failed``, penalised, and
    eliminated at its rung."""
    from pounce._starts import _DONE_STATUS, _PAUSED_STATUS

    assert FEASIBLE_POINT_FOUND in _DONE_STATUS
    assert FEASIBLE_POINT_FOUND not in _PAUSED_STATUS, (
        "a finished solve is not a paused one; being in both would make "
        "the 'genuine failure' test in _rank_rung ambiguous"
    )


@pytest.mark.parametrize("mod", ["jax", "torch"])
def test_path_followers_accept_a_square_feasible_point_anchor(mod):
    """The differentiable path followers gate their anchor and corrector solves
    on ``_OK_STATUS`` by *name*. Both already require every general constraint
    to be an equality (``_require_equality_constraints``), which is the shape
    that reaches this status in the first place — so refusing it aborted the
    path at a converged anchor with ``anchor solve failed``."""
    pytest.importorskip(mod)
    path = pytest.importorskip(f"pounce.{mod}._path")

    assert "Feasible_Point_Found" in path._OK_STATUS
    assert "Infeasible_Problem_Detected" not in path._OK_STATUS


def test_the_status_name_matches_the_rust_spelling():
    """The two ``_OK_STATUS`` tuples key off ``upstream_name()``, not the Rust
    ``Debug`` name — ``Feasible_Point_Found``, not ``FeasiblePointFound``. A
    typo here is invisible: the comparison simply never matches and the status
    silently stays rejected."""
    src = (_REPO / "crates/pounce-nlp/src/return_codes.rs").read_text()
    assert 'Self::FeasiblePointFound => "Feasible_Point_Found"' in src, (
        "upstream_name() no longer spells the status the way _OK_STATUS "
        "expects"
    )
