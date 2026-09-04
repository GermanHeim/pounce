"""gh#776 Gate 1: the phase-changing flash, as a fast regression.

gh#776 asks the Gate 1 fixture to "become the fast CI case and the
common POUNCE/DISCOPT algebraic fixture". This is the first half. The
harness itself lives in `benchmarks/flash/` and carries the full
configuration, the report and the boundary statement; what runs here is
its asserted smoke subset, chosen to put one point in each phase regime
and one on each side of each switch.

What a failure here means
-------------------------

These are *source-level* assertions: the phase regime, the vapor
fraction and the two phase sums are compared against
`pounce.examples.flash_mpcc`, an independent Michelsen stability plus
Rachford--Rice calculation that shares only the Peng--Robinson fugacity
primitive with the model. So a red test is not "the NLP did not
converge" -- it is "the solver's answer is not the flash", which is a
different and much more serious claim. Read the failing check's number
before reaching for a tolerance.

The route is `scholtes_then_ncp`, the one Gate 0 (gh#794) established as
supported for small MPCCs, so a regression here is also the first signal
that the Gate 0 boundary has moved.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_BENCH = Path(__file__).resolve().parents[2] / "benchmarks"

jax = pytest.importorskip("jax")
pytest.importorskip("scipy")

if str(_BENCH) not in sys.path:
    sys.path.insert(0, str(_BENCH))

pytest.importorskip("flash")
from flash import path as flash_path, routes, run as flash_run  # noqa: E402
from flash.runner import cold_start, solve_route  # noqa: E402
from flash.validate import EXPECTED_OK_KEYS  # noqa: E402
from pounce.examples.flash_mpcc import (  # noqa: E402
    GATE1_FLASH,
    CORNER_TOL,
    flash,
    lower,
)


@pytest.fixture(scope="module")
def case():
    return GATE1_FLASH


@pytest.fixture(scope="module")
def solved(case):
    """One solve per smoke temperature, shared across the assertions.

    Module-scoped because the JAX callbacks compile once per case and
    each solve is a full Scholtes continuation; re-solving per assertion
    would multiply a 15-second test by the number of checks for no extra
    coverage.
    """
    route = routes.ROUTES[routes.SUPPORTED_ROUTE]
    out = {}
    for t in flash_run.SMOKE_TEMPERATURES:
        rec = solve_route(case, t, route, cold_start(case, t))
        ref = flash(t, case.pressure_pa, case.mixture)
        out[t] = (rec, ref)
    return out


@pytest.mark.parametrize("temperature_k", flash_run.SMOKE_TEMPERATURES)
def test_solver_reaches_the_flash(solved, temperature_k):
    """Status, then every source-level check the harness defines."""
    rec, ref = solved[temperature_k]
    assert rec.ok, f"{rec.status_msg} at {temperature_k} K"
    missing = EXPECTED_OK_KEYS - set(rec.validation)
    assert not missing, f"validation stopped reporting {sorted(missing)}"
    failed = [
        k
        for k, v in rec.validation.items()
        if k.endswith("_ok") and v is not None and not v
    ]
    assert not failed, f"at {temperature_k} K: {failed} (oracle says {ref.regime})"


@pytest.mark.parametrize("temperature_k", flash_run.SMOKE_TEMPERATURES)
def test_regime_and_vapor_fraction_match_the_oracle(solved, temperature_k):
    """The phase state and the amount, not just the label.

    Both are asserted because agreeing on "liquid" while disagreeing on
    the vapor fraction would mean the solver found a different incipient
    phase -- a real disagreement wearing the right label.
    """
    rec, ref = solved[temperature_k]
    assert rec.regime == ref.regime
    assert rec.beta == pytest.approx(ref.beta, abs=1e-6)
    assert rec.sum_x == pytest.approx(ref.sum_x, abs=1e-6)
    assert rec.sum_y == pytest.approx(ref.sum_y, abs=1e-6)


@pytest.mark.parametrize("temperature_k", flash_run.SMOKE_TEMPERATURES)
def test_the_finishing_solve_actually_runs(solved, temperature_k):
    """The supported route ran *both* halves, on the lowering it had to use.

    This is the one assertion that pins the degrees-of-freedom fallback,
    and without it the defect that motivated the fallback is invisible to
    this file. Measured, on a tree with the fallback removed: every other
    test here still passes, because the continuation alone already lands
    three orders inside `ORACLE_TOL` (`8.4e-10` against `1e-6`) and eight
    inside `CORNER_TOL` (`1.8e-12` against `1e-4`). The regression would
    show up only as two rows of a benchmark table quietly becoming equal
    again -- which is the same detection mechanism that missed it the
    first time.

    Both halves of the assertion earn their place. `finish_applied` alone
    stays true if the gate stops firing and the *equality* finish starts
    running, which is a real change worth failing on rather than
    absorbing: it would mean either the model stopped being square or the
    solver's degrees-of-freedom gate moved.
    """
    rec, _ = solved[temperature_k]
    assert rec.finish_applied, (
        f"the finishing solve did not run at {temperature_k} K: "
        f"{rec.finish_status_msg}"
    )
    assert rec.finish_lowering == "prod_ineq", (
        f"finished on {rec.finish_lowering!r} at {temperature_k} K; a square "
        "flash cannot run the equality finish, so this means the fallback "
        "stopped firing"
    )


def test_the_path_crosses_all_three_regimes(solved):
    """The smoke subset is not accidentally all one phase.

    Without this, every assertion above could keep passing on a fixture
    that had drifted into a single regime and stopped testing the thing
    gh#776 asked for.
    """
    regimes = {ref.regime for _, ref in solved.values()}
    assert {"liquid", "two_phase", "vapor"} <= regimes, regimes


@pytest.mark.parametrize("temperature_k", flash_run.SMOKE_TEMPERATURES)
def test_complementarity_is_reported_in_source_terms(solved, temperature_k):
    """The pair products are small *in the source model*.

    Deliberately not read off `info["final_constr_viol"]`, which for a
    Scholtes stage is a residual against `G*H <= tau` and is satisfied
    by points that are not MPCC-feasible at all. The threshold is Gate
    0's `sqrt(tol)` complementarity accuracy floor, which is numerical
    resolution rather than phase physics.
    """
    rec, _ = solved[temperature_k]
    assert rec.source["compl_max"] <= CORNER_TOL
    assert rec.source["balance_viol"] <= 1e-7
    assert rec.source["isofugacity_viol"] <= 1e-7
    assert rec.source["sign_viol"] <= 1e-9


def test_the_complementarity_pairs_are_amount_against_slack(case, solved):
    """gh#776's formulation guardrail, asserted rather than documented.

    Liquid and vapor coexist on a two-phase tray, so a pair of the form
    `L ⟂ V` would encode the wrong physics. At a genuinely two-phase
    point both amounts are positive and it is the two *slacks* that
    vanish; if some future edit made the pair `L ⟂ V`, both sides would
    be positive here and the product would be far from zero.
    """
    rec, ref = solved[300.0]
    assert 1e-3 < ref.beta < 1.0 - 1e-3, "300 K must be genuinely two-phase"
    g, h = case.pair_values(rec.x)
    assert abs(g[0]) > 1e-3 and abs(g[1]) > 1e-3, "both phase amounts are present"
    assert abs(h[0]) < 1e-8 and abs(h[1]) < 1e-8, "it is the slacks that vanish"


def test_the_oracle_answer_solves_the_model(case):
    """The cross-check itself, with no solver in it.

    This is the assertion that caught the normalization defect recorded
    in `pounce.examples.flash_mpcc` -- and it caught it *only* at the single-phase
    temperatures, because the offending term is identically zero in the
    two-phase region. It is cheap and it is the reason the rest of this
    file means anything, so it runs on every temperature rather than on
    the smoke subset.
    """
    import numpy as np

    worst, worst_t = 0.0, None
    for t in case.temperatures_k:
        t = float(t)
        ref = flash(t, case.pressure_pa, case.mixture)
        nlp = lower(case, t, "prod_eq")
        c = nlp.constraints(case.pack(ref.beta, ref.x, ref.y))
        viol = float(np.max(np.maximum(np.maximum(nlp.cl - c, c - nlp.cu), 0.0)))
        if viol > worst:
            worst, worst_t = viol, t
    assert worst <= 1e-10, f"worst {worst:.2e} at {worst_t} K"


def test_the_smoke_cli_enforces_the_same_contract(case):
    """The advertised entry point must be able to fail.

    `python -m flash.run --smoke` is what the README and the Makefile
    tell people to run, and it is asserted rather than reported -- so a
    version of it that cannot fail is worse than none. It was: the first
    implementation filtered `validation` for false `_ok` keys and called
    an empty filter a pass. On a tree with `finish_fallback` removed,
    pytest failed at all five temperatures while this exited 0 and
    printed `smoke passed`; replacing `validation` with `{}` also exited
    0.

    Running `run_smoke` itself, rather than reimplementing its checks
    here, is the point: the CLI and this suite are two entry points onto
    one claim, and they already drifted once.
    """
    assert flash_run.run_smoke(case, verbose=False) == 0


def test_the_smoke_contract_rejects_a_record_that_skipped_the_finish(case, solved):
    """...and the contract function fails on the shapes that fooled it.

    Exercised against doctored copies of a real record rather than a
    reverted tree, so the failure modes stay pinned without the suite
    needing to mutate the source. Both shapes below exited 0 before
    `smoke_contract_failures` existed.
    """
    import dataclasses

    route = routes.ROUTES[routes.SUPPORTED_ROUTE]
    good, _ = solved[300.0]
    assert not flash_run.smoke_contract_failures(good, route)

    no_finish = dataclasses.replace(
        good, finish_applied=False, finish_status_msg="Not_Enough_Degrees_Of_Freedom"
    )
    assert flash_run.smoke_contract_failures(no_finish, route)

    empty_validation = dataclasses.replace(good, validation={})
    assert flash_run.smoke_contract_failures(empty_validation, route)

    wrong_finish = dataclasses.replace(good, finish_lowering="prod_eq")
    assert flash_run.smoke_contract_failures(wrong_finish, route)


def test_the_inter_temperature_start_is_labelled_primal(case):
    """The warm leg carries a primal point, and says so.

    gh#776 asks this benchmark to compare warm-start behaviour across
    the switching points, so the label on an iteration count is part of
    the measurement. `path.traverse` deliberately carries only the
    previous temperature's primal point -- the multipliers and barrier
    parameter belong to a different problem -- and an earlier version
    recorded that seeded stage as `full` anyway, filing a primal-only
    count under a full-state label.
    """
    route = routes.ROUTES[routes.SUPPORTED_ROUTE]
    leg = flash_path.traverse(
        case, route, direction="up", start_mode="warm", temperatures=[300.0, 310.0]
    )
    first, second = sorted(leg.records, key=lambda r: r.temperature_k)
    assert first.stages[0].warm_level == "none", "the first temperature is a cold start"
    assert second.stages[0].warm_level == "primal", (
        "the inter-temperature start carries only x, so the stage must not "
        f"claim a full-state warm start (got {second.stages[0].warm_level!r})"
    )
