"""Independent oracles: branch enumeration, and the optional CCOpt hook.

Branch enumeration
------------------

A complementarity condition ``0 <= G _|_ H >= 0`` is a disjunction: at a
feasible point each pair sits on the ``G = 0`` branch or the ``H = 0``
branch (a biactive point is on both). Fixing a branch per pair turns the
MPCC into an ordinary smooth program, and every case in this corpus has
at most two pairs, so **all** branches can be solved and the best one
taken. That is a genuine global solution of the MPCC, obtained without
POUNCE and without any complementarity machinery.

It is what makes the manifest's ``expected`` numbers checkable rather
than asserted: `selftest` re-derives them here and fails on
disagreement. It is also the only global statement anywhere in the
harness -- POUNCE is a local solver and every route record says so.

The branch programs are solved with SciPy's SLSQP from a deterministic
multi-start. Each is a small QP with affine constraints (the corpus is
quadratic by construction, `spec`), so this is not a hard ask; the
multi-start exists because SLSQP is a local method too, and a
single-start oracle would be exactly the kind of borrowed confidence
this file is meant to remove.

CCOpt
-----

gh#794 asks for a pinned CCOpt integrated-continuation comparison as an
**optional** benchmark oracle, explicitly not a required runtime
dependency. `ccopt_status` reports whether it is importable and at what
version, and the runner records that verdict in every result file so a
report can never quietly claim a comparison it did not run. No CCOpt is
vendored, pinned in any requirements file, or imported at module scope.
"""

from __future__ import annotations

import itertools
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import minimize

from .spec import MpccCase

#: The CCOpt release this comparison would be pinned to if it were run.
#: Recorded in every result file whether or not CCOpt is present, so the
#: report states which version a comparison *would* have used.
CCOPT_PIN = "ccopt==0.4.1"

_FEAS_TOL = 1e-9


def _scipy_bounds(case: MpccCase):
    return [
        (None if not np.isfinite(lo) else lo, None if not np.isfinite(hi) else hi)
        for lo, hi in zip(case.lb, case.ub)
    ]


def _constraints(case: MpccCase, branch: Tuple[int, ...]):
    """SLSQP constraint dicts for one branch assignment.

    ``branch[i] == 0`` pins ``G_i = 0`` and keeps ``H_i >= 0``;
    ``branch[i] == 1`` pins ``H_i = 0`` and keeps ``G_i >= 0``.
    """
    cons = []
    for row in case.rows:
        f = row.form
        if row.is_equality:
            cons.append(
                {
                    "type": "eq",
                    "fun": (lambda x, f=f, lo=row.lo: f.value(x) - lo),
                    "jac": (lambda x, f=f: f.grad(x)),
                }
            )
        else:
            if np.isfinite(row.hi):
                cons.append(
                    {
                        "type": "ineq",
                        "fun": (lambda x, f=f, hi=row.hi: hi - f.value(x)),
                        "jac": (lambda x, f=f: -f.grad(x)),
                    }
                )
            if np.isfinite(row.lo):
                cons.append(
                    {
                        "type": "ineq",
                        "fun": (lambda x, f=f, lo=row.lo: f.value(x) - lo),
                        "jac": (lambda x, f=f: f.grad(x)),
                    }
                )
    for i, p in enumerate(case.pairs):
        pinned, freeform = (p.G, p.H) if branch[i] == 0 else (p.H, p.G)
        cons.append(
            {
                "type": "eq",
                "fun": (lambda x, f=pinned: f.value(x)),
                "jac": (lambda x, f=pinned: f.grad(x)),
            }
        )
        cons.append(
            {
                "type": "ineq",
                "fun": (lambda x, f=freeform: f.value(x)),
                "jac": (lambda x, f=freeform: f.grad(x)),
            }
        )
    return cons


def _starts(case: MpccCase) -> List[np.ndarray]:
    """Deterministic multi-start: the case's own starts plus a fixed grid."""
    pts = [np.asarray(v, dtype=float) for v in case.starts.values()]
    rng = np.random.default_rng(794)
    lo = np.where(np.isfinite(case.lb), case.lb, -2.0)
    hi = np.where(np.isfinite(case.ub), case.ub, 2.0)
    for _ in range(12):
        pts.append(lo + (hi - lo) * rng.random(case.n))
    pts.append(np.zeros(case.n))
    return pts


def enumerate_branches(case: MpccCase, tol: float = 1e-7) -> Dict[str, object]:
    """Solve every complementarity branch; return the global optimum.

    The returned ``branches`` list is per-assignment, so a case whose
    branches tie (the selector at ``theta = 1/2``, `ctrap`'s two
    minimisers) shows the tie in the record instead of hiding it behind
    whichever one sorted first.
    """
    bounds = _scipy_bounds(case)
    results = []
    for branch in itertools.product((0, 1), repeat=case.q):
        cons = _constraints(case, branch)
        best_f, best_x = np.inf, None
        for x0 in _starts(case):
            try:
                r = minimize(
                    case.objective.value,
                    x0,
                    jac=case.objective.grad,
                    bounds=bounds,
                    constraints=cons,
                    method="SLSQP",
                    options={"maxiter": 500, "ftol": 1e-12},
                )
            except Exception:  # pragma: no cover - SLSQP internal failure
                continue
            if not r.success:
                continue
            s = case.source_feasibility(r.x)
            if max(s["row_viol"], s["bound_viol"], s["sign_viol"], s["compl_max"]) > tol:
                continue
            if r.fun < best_f - 1e-12:
                best_f, best_x = float(r.fun), np.array(r.x, dtype=float)
        results.append(
            {
                "branch": "".join("GH"[b] for b in branch),
                "feasible": best_x is not None,
                "obj": None if best_x is None else best_f,
                "x": None if best_x is None else best_x.tolist(),
            }
        )

    feas = [r for r in results if r["feasible"]]
    if not feas:
        return {"feasible": False, "obj": None, "x": None, "branches": results}
    best = min(feas, key=lambda r: r["obj"])
    ties = [r["branch"] for r in feas if r["obj"] <= best["obj"] + 1e-8]
    return {
        "feasible": True,
        "obj": best["obj"],
        "x": best["x"],
        "optimal_branches": ties,
        "unique": len(ties) == 1,
        "branches": results,
    }


def ccopt_status() -> Dict[str, object]:
    """Is the optional CCOpt comparator available, and at what version?

    Never raises and never installs anything. gh#794 wants the
    comparison pinned *if included*; the honest report of "not included,
    and here is the pin it would have used" is this dict.
    """
    try:
        import ccopt  # type: ignore  # noqa: F401
    except Exception as exc:
        return {
            "available": False,
            "pin": CCOPT_PIN,
            "reason": f"{type(exc).__name__}: {exc}",
            "comparison_run": False,
        }
    version = getattr(ccopt, "__version__", "unknown")
    pinned = CCOPT_PIN.split("==")[-1]
    return {
        "available": True,
        "pin": CCOPT_PIN,
        "version": version,
        "pin_satisfied": version == pinned,
        # The integrated-continuation comparison is only meaningful
        # against the pinned build; a different one is recorded as
        # present and not compared, rather than compared and captioned.
        "comparison_run": False,
        "reason": (
            "CCOpt is importable but the integrated-continuation adapter is "
            "not implemented in this harness; gh#794 makes the comparison "
            "optional and this records the version that was present."
        ),
    }
