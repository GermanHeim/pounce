"""Provenance stamping.

gh#776 requires every Gate 1 result to be "stamped with both repository
commits, model-data revision, scaling/tolerances, mesh, and execution
environment". The two that are easy to fake and easy to omit are
computed rather than typed.

The DiscOpt entry is the one to read carefully. Gate 1's fixture is
meant to carry a reduced DiscOpt GDP/SOS1 comparison, and that
comparison is *blocked* on DiscOpt's first-class complementarity
provenance and its local-versus-certified result contract
(jkitchin/discopt#1123). Until those land there is no comparison to
stamp, and the field says so explicitly instead of being absent --
"no DiscOpt comparison was run" and "a DiscOpt comparison was run and
its commit was not recorded" are different claims, and a result file
that cannot distinguish them is not evidence.
"""

from __future__ import annotations

import hashlib
import os
import platform
import subprocess
import sys
from typing import Dict

_HERE = os.path.dirname(os.path.abspath(__file__))

#: Files whose content defines the fixture. A change to any of them
#: changes the model-data revision, and two result files whose
#: revisions differ are two different benchmarks.
_MODEL_FILES = ("spec.py", "thermo.py", "lowering.py", "oracle.py")

#: The upstream module the thermodynamics is taken from. It is not in
#: this directory, so it is hashed by path rather than by name -- a
#: change to the Peng--Robinson layer changes this fixture's answers
#: and must change its revision.
_UPSTREAM = os.path.normpath(
    os.path.join(_HERE, "..", "..", "python", "pounce", "examples", "phase_envelope.py")
)


def _git(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=_HERE, stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return "unknown"


def model_data_revision() -> str:
    h = hashlib.sha256()
    for path in [os.path.join(_HERE, n) for n in _MODEL_FILES] + [_UPSTREAM]:
        try:
            with open(path, "rb") as fh:
                h.update(fh.read())
        except OSError:
            h.update(b"<missing>")
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
            "comparison_run": False,
            "reason": (
                "discopt is importable, but the reduced GDP/SOS1 comparison "
                "gh#776 asks for is blocked on jkitchin/discopt#1123 (first-class "
                "complementarity provenance and the local-versus-certified result "
                "contract). Nothing was compared."
            ),
        }
    except Exception as exc:
        disc = {
            "present": False,
            "commit": None,
            "comparison_run": False,
            "reason": (
                f"discopt not importable ({type(exc).__name__}); the reduced "
                "GDP/SOS1 comparison is blocked on jkitchin/discopt#1123 and no "
                "comparison was run"
            ),
        }
    return {"pounce": pounce, "discopt": disc}


def environment() -> Dict[str, object]:
    def _v(name):
        try:
            return __import__(name).__version__
        except Exception:
            return "unknown"

    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "numpy": _v("numpy"),
        "scipy": _v("scipy"),
        "jax": _v("jax"),
        "pounce": _v("pounce"),
    }


def model_stamp(case) -> Dict[str, object]:
    """The model side: mixture constants, feed, pressure, and the path.

    gh#776 asks for the "model-data revision" and the "mesh"; on a flash
    the mesh is the temperature path, so it is recorded in full rather
    than as a count. A path summarized by its endpoints cannot be
    compared against another run's.
    """
    m = case.mixture
    return {
        "case": case.name,
        "components": list(m.names),
        "critical_temperature_k": [float(v) for v in m.critical_temperature],
        "critical_pressure_pa": [float(v) for v in m.critical_pressure],
        "acentric_factor": [float(v) for v in m.acentric_factor],
        "binary_interaction": [[float(v) for v in row] for row in m.binary_interaction],
        "feed_composition": [float(v) for v in case.z],
        "pressure_pa": float(case.pressure_pa),
        "temperatures_k": [float(t) for t in case.temperatures_k],
        "equation_of_state": "Peng-Robinson, classical one-fluid mixing",
        "constants_source": m.source,
        "provenance": case.provenance,
    }
