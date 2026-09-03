"""Source-level validation at a returned point.

Nothing in this module reads a solver status, an NLP residual, or a
lowering. That split is the same one Gate 0 made and for the same
reason: a route can converge its reformulation and still be wrong about
the model, and only a check written against the model can say so. On a
flash the gap is wider than it was on Gate 0's algebra, because the
model has physics in it that no residual sees -- a converged point can
satisfy every row and still be the trivial solution, or sit on the wrong
cubic root, or report a phase state the tangent-plane test says is
unstable.

Keys ending in ``_ok`` are pass/fail and the report renders them as
such. A failing ``_ok`` is a claim that the returned point is not the
one the physics describes; it is not automatically a solver defect, and
the report shows the number next to it so a reader can tell which.

The five checks, and what each one would catch
----------------------------------------------

``balance_ok`` / ``isofugacity_ok``
    The source rows, in the model's own units. These are the ones an
    NLP residual would also catch -- they are here so that the record
    does not have to be read against a different problem's tolerance.

``regime_matches_oracle_ok``
    The phase state agrees with `oracle.flash`. This is the check
    gh#776 asks for by name.

``beta_matches_oracle_ok`` / ``phase_sums_match_oracle_ok``
    The numbers agree, not just the label. ``Sx`` and ``Sy`` are
    included because in a single-phase regime they *are* the
    tangent-plane test's ``sum Y``: agreeing on "liquid" while
    disagreeing on ``Sy`` would mean the solver found a different
    incipient phase, which is a real disagreement wearing the right
    label.

``root_is_gibbs_optimal_ok``
    The cubic-root guard gh#776 asks for. The model picks each root by
    phase label; this checks the label against the lower-Gibbs root --
    **for the phases that are present**, which in a single-phase regime
    is one of the two. An incipient phase is deliberately not judged:
    Michelsen's trial phase probes the tangent plane at the feed rather
    than its own stability, and taking the metastable root at its own
    composition is ordinary rather than wrong. Both diagnostics are
    recorded whichever way, so the unjudged branch is visible. The
    verdict is ``None`` -- not ``True`` -- where the cubic has one real
    root, since there is nothing to have got wrong.

``not_trivial_ok``
    ``K_i = 1`` for every ``i`` solves the isofugacity rows at any
    composition and is the classic false flash answer. On this fixture
    it is also the only way the MPCC can leave ``beta`` undetermined
    (see `oracle.FlashResult.no_incipient_phase`), so it is checked at
    every point rather than only where a two-phase answer is expected.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from . import oracle as O
from . import thermo
from .spec import FlashCase

#: Source-residual tolerance. Looser than the solver's ``tol`` on
#: purpose: these are residuals of the *source* model at a point
#: converged for a *lowered* one, and the two differ by the
#: complementarity accuracy floor Gate 0 established.
SOURCE_TOL = 1e-7

#: Agreement tolerance against the oracle. `beta` and the phase sums are
#: O(1) quantities, and the oracle is polished to ~1e-14, so this is
#: dominated by the solver's own `sqrt(tol)` corner accuracy rather than
#: by the reference.
ORACLE_TOL = 1e-6


def _oracle_at(case: FlashCase, temperature_k: float) -> O.FlashResult:
    return O.flash(temperature_k, case.pressure_pa, case.mixture, case.z)


def validate(
    case: FlashCase,
    v,
    temperature_k: float,
    reference: Optional[O.FlashResult] = None,
) -> Dict[str, object]:
    """Every source-level check at ``v``, with the numbers behind them."""
    v = np.asarray(v, dtype=float)
    beta, x, y = case.unpack(v)
    ref = reference if reference is not None else _oracle_at(case, temperature_k)
    src = case.source_feasibility(v, temperature_k)
    regime = case.regime(v)

    out: Dict[str, object] = {
        "balance_ok": bool(src["balance_viol"] <= SOURCE_TOL),
        "isofugacity_ok": bool(src["isofugacity_viol"] <= SOURCE_TOL),
        "regime": regime,
        "oracle_regime": ref.regime,
        "regime_matches_oracle_ok": bool(_regime_agrees(regime, ref.regime)),
        "beta_error": float(abs(beta - ref.beta)),
        "beta_matches_oracle_ok": bool(abs(beta - ref.beta) <= ORACLE_TOL),
        "sum_x_error": float(abs(np.sum(x) - ref.sum_x)),
        "sum_y_error": float(abs(np.sum(y) - ref.sum_y)),
        "phase_sums_match_oracle_ok": bool(
            abs(np.sum(x) - ref.sum_x) <= ORACLE_TOL
            and abs(np.sum(y) - ref.sum_y) <= ORACLE_TOL
        ),
    }

    # -- the cubic-root guard, at both phase compositions -----------
    xn, yn = x / np.sum(x), y / np.sum(y)
    rd_l = thermo.root_diagnostics(
        xn, temperature_k, case.pressure_pa, case.mixture, largest=False
    )
    rd_v = thermo.root_diagnostics(
        yn, temperature_k, case.pressure_pa, case.mixture, largest=True
    )
    # Judged for the phases that are actually *present*, and merely
    # recorded for an incipient one. A present phase sitting at the
    # metastable root is a wrong answer; a trial phase doing so is
    # ordinary -- Michelsen's trial phase probes the tangent plane at
    # the feed, not its own stability, and on this path the incipient
    # vapor takes the metastable root below 240 K and the incipient
    # liquid above 340 K. Both diagnostics are kept either way, so a
    # reader can see the branch that was not judged.
    judged = []
    if regime in ("liquid", "two_phase", "bubble", "dew"):
        judged.append(rd_l["root_is_gibbs_optimal"])
    if regime in ("vapor", "two_phase", "bubble", "dew"):
        judged.append(rd_v["root_is_gibbs_optimal"])
    judged = [v for v in judged if v is not None]
    out["liquid_root"] = rd_l
    out["vapor_root"] = rd_v
    out["root_is_gibbs_optimal_ok"] = None if not judged else bool(all(judged))
    out["root_judged_for"] = regime

    # -- the trivial solution ---------------------------------------
    k = case.k_values(v)
    out["max_abs_ln_k"] = float(np.max(np.abs(np.log(k))))
    out["not_trivial_ok"] = bool(not thermo.is_trivial(k))

    # -- the supercritical record (recorded, not judged) ------------
    out["supercritical_components"] = thermo.supercritical_components(
        temperature_k, case.mixture
    )
    out["reduced_temperatures"] = thermo.reduced_temperatures(temperature_k, case.mixture)
    return out


def _regime_agrees(got: str, want: str) -> bool:
    """Regime agreement, with the switch points treated as a band.

    ``bubble`` and ``dew`` are the biactive points, and a solve that
    lands within the complementarity accuracy floor of one of them can
    legitimately report either the switch itself or the regime on
    either side of it -- the floor is ``sqrt(tol)`` wide in the pair,
    which is a real interval in temperature. Insisting on an exact label
    match there would fail correct code, which is the thing this
    harness is least allowed to do. Away from the switch points the
    match is exact.
    """
    if got == want:
        return True
    return (got, want) in {
        ("bubble", "liquid"),
        ("bubble", "two_phase"),
        ("liquid", "bubble"),
        ("two_phase", "bubble"),
        ("dew", "vapor"),
        ("dew", "two_phase"),
        ("vapor", "dew"),
        ("two_phase", "dew"),
    }
