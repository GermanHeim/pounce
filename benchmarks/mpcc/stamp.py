"""Provenance stamping.

gh#794 requires every record to carry both repository commits, the
model-data revision, the environment, the scaling/tolerances, the
initial point and the configuration. The first and third of those are
the ones that are easy to fake and easy to omit, so they are computed
rather than typed:

* **Repository commits.** POUNCE's is read from git. DiscOpt is a
  cross-repository *design* dependency of gh#776 and is not a runtime
  dependency of this harness; its entry records that it was absent and
  why, rather than being left out, so a reader of a result file can
  tell "no DiscOpt comparison was run" from "a DiscOpt comparison was
  run and its commit was not recorded".

* **Model-data revision.** A SHA-256 over the modules that define the
  corpus and its lowerings. Comparing two result files whose
  ``model_data_revision`` differs is comparing two different benchmarks,
  and the report says so instead of quietly averaging them.
"""

from __future__ import annotations

import hashlib
import os
import platform
import subprocess
import sys
from typing import Dict

_HERE = os.path.dirname(os.path.abspath(__file__))

#: Files whose content defines the benchmark instances. A change to any
#: of them changes the model-data revision.
_MODEL_FILES = ("spec.py", "cases.py", "lowering.py")


def _git(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=_HERE, stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return "unknown"


def model_data_revision() -> str:
    h = hashlib.sha256()
    for name in _MODEL_FILES:
        with open(os.path.join(_HERE, name), "rb") as fh:
            h.update(fh.read())
    return h.hexdigest()[:16]


def repositories() -> Dict[str, object]:
    dirty = _git("status", "--porcelain")
    pounce = {
        "commit": _git("rev-parse", "HEAD"),
        "short": _git("rev-parse", "--short", "HEAD"),
        "describe": _git("describe", "--always", "--dirty"),
        "dirty": bool(dirty) if dirty != "unknown" else None,
    }
    try:
        import discopt  # type: ignore

        disc = {
            "present": True,
            "version": getattr(discopt, "__version__", "unknown"),
            "commit": "unknown (installed package, not a checkout)",
        }
    except Exception as exc:
        disc = {
            "present": False,
            "commit": None,
            "reason": (
                f"discopt not importable ({type(exc).__name__}); it is a "
                "cross-repository design dependency of gh#776, not a runtime "
                "dependency of this harness, and no DiscOpt comparison was run"
            ),
        }
    return {"pounce": pounce, "discopt": disc}


def environment() -> Dict[str, object]:
    try:
        import numpy

        numpy_v = numpy.__version__
    except Exception:
        numpy_v = "unknown"
    try:
        import scipy

        scipy_v = scipy.__version__
    except Exception:
        scipy_v = "unknown"
    try:
        import pounce

        pounce_v = getattr(pounce, "__version__", "unknown")
    except Exception:
        pounce_v = "not importable"
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "numpy": numpy_v,
        "scipy": scipy_v,
        "pounce": pounce_v,
    }
