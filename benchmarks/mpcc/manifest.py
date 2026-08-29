"""The manifest: what each case is, and what it is known to do.

`manifest.json` is generated from `cases.py` (``run --write-manifest``)
and committed, and `selftest` fails when the committed copy has drifted
from the code. Two sources of truth would be one too many, but a
manifest that only exists as Python is not readable by anything outside
this harness -- DiscOpt's side of gh#776 needs to read the expected
values without importing POUNCE's benchmark package.

Every ``expected`` block carries **two** numbers for the optimum: the
one derived by hand in the case's docstring, and the one
`oracle.enumerate_branches` recomputes by solving each complementarity
branch with SciPy. They are written out separately on purpose. If they
ever disagree, the manifest shows the disagreement rather than a
silently-preferred one.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List

import numpy as np

from . import cases as C
from . import routes as R
from . import stamp
from .oracle import CCOPT_PIN, enumerate_branches
from .spec import ACTIVE_TOL, CLASSES, SCALINGS
from .stationarity import classify

MANIFEST_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "manifest.json")


def _tolist(a):
    return None if a is None else [float(v) for v in np.asarray(a)]


def build(with_oracle: bool = True) -> Dict[str, object]:
    entries: List[Dict[str, object]] = []
    for name in C.REGISTRY:
        case = C.make(name)
        e = case.expected
        entry: Dict[str, object] = {
            "name": case.name,
            "class": case.klass,
            "n_variables": case.n,
            "n_rows": len(case.rows),
            "n_pairs": case.q,
            "provenance": case.provenance,
            "derivation": (C._FACTORIES[name].__doc__ or "").strip(),
            "lb": [None if not np.isfinite(v) else float(v) for v in case.lb],
            "ub": [None if not np.isfinite(v) else float(v) for v in case.ub],
            "pairs": [
                {
                    "name": p.name,
                    "G": {"a": _tolist(p.G.a), "b": p.G.b},
                    "H": {"a": _tolist(p.H.a), "b": p.H.b},
                    "branch_G_zero": p.branch_G_zero,
                    "branch_H_zero": p.branch_H_zero,
                }
                for p in case.pairs
            ],
            "starts": {k: _tolist(v) for k, v in case.starts.items()},
            "expected": {
                "feasible": e.feasible,
                "obj": e.obj,
                "x": _tolist(e.x),
                "stationarity": e.stationarity,
                "n_biactive": e.n_biactive,
                "mpcc_licq": e.mpcc_licq,
                "notes": e.notes,
            },
            "in_smoke_subset": name in C.SMOKE,
        }
        if e.x is not None:
            cl = classify(case, np.asarray(e.x, dtype=float))
            entry["classifier_at_expected_x"] = {
                "klass": cl["klass"],
                "n_biactive": cl["n_biactive"],
                "mpcc_licq": cl["mpcc_licq"],
                "residuals": cl["residuals"],
            }
        if with_oracle:
            orc = enumerate_branches(case)
            entry["oracle"] = {
                "feasible": orc["feasible"],
                "obj": orc["obj"],
                "x": orc["x"],
                "optimal_branches": orc.get("optimal_branches"),
                "unique": orc.get("unique"),
            }
        entries.append(entry)

    return {
        "schema": "pounce-mpcc-manifest/1",
        "issue": "https://github.com/jkitchin/pounce/issues/794",
        "model_data_revision": stamp.model_data_revision(),
        "classes_required": list(CLASSES),
        "classes_present": sorted({c["class"] for c in entries}),
        "smoke_subset": list(C.SMOKE),
        "active_tol": ACTIVE_TOL,
        "routes": {
            k: {
                "lowering": v.lowering,
                "warm": v.warm,
                "continuation": v.continuation,
                "options": v.options,
                "why": v.why,
            }
            for k, v in R.ROUTES.items()
        },
        "controls": R.CONTROLS,
        "base_options": R.BASE_OPTIONS,
        "tau_schedule": list(R.TAU_SCHEDULE),
        "restart_ladder": list(R.RESTART_LADDER),
        "tau_bisections": R.TAU_BISECTIONS,
        "scalings": {k: _tolist(v(3)) for k, v in SCALINGS.items()},
        "ccopt_pin": CCOPT_PIN,
        "cases": entries,
    }


def write(path: str = MANIFEST_PATH, with_oracle: bool = True) -> str:
    data = build(with_oracle=with_oracle)
    with open(path, "w") as fh:
        json.dump(data, fh, indent=2, sort_keys=False)
        fh.write("\n")
    return path


def load(path: str = MANIFEST_PATH) -> Dict[str, object]:
    with open(path) as fh:
        return json.load(fh)
