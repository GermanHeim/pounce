"""Source-level validation, one function per benchmark class.

gh#794 asks that *each benchmark class* have expected behaviour and a
source-level validation function. `cases.py` carries the expected
behaviour; this module carries the checks, and they are class-level
rather than case-level on purpose: what makes a case a `regular` case is
a property of the point a route returns, not of the case's metadata, and
a route that returns a biactive point on a case whose class says strict
complementarity has found something the manifest did not predict.

Every check here reads the **source** MPCC at the returned point.
Nothing in this module looks at a solver status, an NLP residual, or a
lowering. That is the whole point of the split: a route can converge its
reformulation and still be wrong about the model, and only a check
written against the model can say so.

Keys ending in ``_ok`` are pass/fail and the report renders them as
such; everything else is a recorded quantity. A failing ``_ok`` is not
automatically a defect -- a local solver reaching a different local
solution will fail `strict_complementarity_ok` honestly -- it is a claim
that the returned point is not the one the class describes, which is
exactly what a reader needs to see before drawing a conclusion from the
row above it.
"""

from __future__ import annotations

from typing import Callable, Dict

import numpy as np

from .spec import ACTIVE_TOL, MpccCase, pair_activity


def _activity(case: MpccCase, x: np.ndarray):
    """`spec.pair_activity` over every pair, as two boolean arrays."""
    acts = [pair_activity(p, x, ACTIVE_TOL) for p in case.pairs]
    return (
        np.array([a for a, _ in acts], dtype=bool),
        np.array([b for _, b in acts], dtype=bool),
    )


def _regular(case: MpccCase, x: np.ndarray) -> Dict[str, object]:
    """Strict complementarity: exactly one side of every pair vanishes."""
    g, h = case.pair_values(x)
    gz, hz = _activity(case, x)
    return {
        "strict_complementarity_ok": bool(np.all(gz ^ hz)),
        "n_biactive_at_point": int(np.sum(gz & hz)),
        "n_pairs_off_both_branches": int(np.sum(~gz & ~hz)),
    }


def _biactive(case: MpccCase, x: np.ndarray) -> Dict[str, object]:
    """At least one pair with both sides at zero, as the class claims."""
    g, h = case.pair_values(x)
    gz, hz = _activity(case, x)
    both = gz & hz
    return {
        "has_biactive_pair_ok": bool(np.any(both)),
        "n_biactive_at_point": int(np.sum(both)),
        # How far the biactive pair actually is from the corner, in the
        # model's units. A route can satisfy the boolean above while
        # sitting a long way out on a badly scaled model.
        "biactive_distance": float(
            np.max(np.maximum(np.abs(g), np.abs(h))[both]) if np.any(both) else np.nan
        ),
    }


def _degenerate(case: MpccCase, x: np.ndarray) -> Dict[str, object]:
    """The returned point is MPCC-feasible, and how far off the branch it is.

    The class's defining property -- MPCC-LICQ failing, or a stationary
    point of a weaker class than S -- is established by
    `stationarity.classify`, which every record already carries. What
    this adds is the source-feasibility side of the same question: a
    degenerate case is where a route is most likely to stop just off the
    complementarity corner and report success.
    """
    s = case.source_feasibility(x)
    return {
        "source_feasible_ok": bool(
            max(s["row_viol"], s["bound_viol"], s["sign_viol"], s["compl_max"]) <= 1e-6
        ),
        "worst_source_residual": float(
            max(s["row_viol"], s["bound_viol"], s["sign_viol"], s["compl_max"])
        ),
    }


def _infeasible(case: MpccCase, x: np.ndarray) -> Dict[str, object]:
    """The returned point must NOT be source-feasible; there is no such point."""
    s = case.source_feasibility(x)
    worst = max(s["row_viol"], s["bound_viol"], s["sign_viol"], s["compl_max"])
    return {
        "correctly_not_feasible_ok": bool(worst > 1e-6),
        "worst_source_residual": float(worst),
    }


def _selector(case: MpccCase, x: np.ndarray) -> Dict[str, object]:
    """The selector committed to a branch rather than staying fractional."""
    g, h = case.pair_values(x)
    gz, hz = _activity(case, x)
    # Distance to the nearest branch, per pair: a committed selector has
    # min(|G_i|, |H_i|) = 0. The worst over pairs is what gets reported,
    # because one uncommitted pair is enough to make the answer
    # fractional.
    dist = np.minimum(np.abs(g), np.abs(h)) if g.size else np.zeros(0)
    return {
        "committed_to_a_branch_ok": bool(np.all(gz | hz)),
        "fractional_gap": float(dist.max()) if dist.size else 0.0,
    }


def _macmpec(case: MpccCase, x: np.ndarray) -> Dict[str, object]:
    """Against the pinned optimum, with the gap recorded either way.

    Signed, not absolute: a *negative* gap means the route returned an
    objective below the MPCC's optimum, which is possible only at a point
    the source model does not admit, and merging it into an absolute
    value would hide the one direction that cannot be explained away.
    """
    exp = case.expected
    if exp.obj is None:
        return {"pinned_optimum": None}
    gap = float(case.objective.value(x) - exp.obj)
    return {
        "matches_pinned_optimum_ok": bool(abs(gap) <= 1e-5 * max(1.0, abs(exp.obj))),
        "signed_objective_gap": gap,
        "below_pinned_optimum": bool(gap < -1e-6),
    }


CLASS_VALIDATORS: Dict[str, Callable[[MpccCase, np.ndarray], Dict[str, object]]] = {
    "regular": _regular,
    "biactive": _biactive,
    "degenerate": _degenerate,
    "infeasible": _infeasible,
    "selector": _selector,
    "macmpec": _macmpec,
}


def validate(case: MpccCase, x: np.ndarray) -> Dict[str, object]:
    """Class validator, then the case's own extra validators."""
    out: Dict[str, object] = {}
    fn = CLASS_VALIDATORS.get(case.klass)
    if fn is not None:
        out.update(fn(case, np.asarray(x, dtype=float)))
    return out
