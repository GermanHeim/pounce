"""The POUNCE configurations Gate 1 compares, and the kill switches.

`scholtes_then_ncp` is the one under test: Gate 0 (gh#794) named it the
supported route for small MPCCs, and gh#776 asks Gate 1 to use it. The
other arms are here so that "the supported route works on a phase-change
model" is a comparison rather than an assertion -- if the direct
formulation does just as well on this fixture, that is a result about
how much of Gate 0's boundary carries over, and it should be visible
rather than untested.

The options are pinned for the reason Gate 0 pinned them: an iteration
count compared across differently-converged solves measures nothing.
``bound_relax_factor = 0`` matters more here than it did there --- the
default relaxes every bound by 1e-8, which on this model means admitting
``Sy = 1 + 1e-8``, a *supersaturated* vapor. That is a source-level sign
violation dressed as a rounding convention, and it sits in the same
decimal place as the complementarity products the fixture reports.
"""

from __future__ import annotations

import dataclasses
from typing import Dict, List, Optional, Tuple

BASE_OPTIONS: Dict[str, object] = {
    "print_level": 0,
    "tol": 1e-8,
    "max_iter": 300,
    "bound_relax_factor": 0.0,
    "honor_original_bounds": "yes",
}

LOG_PRINT_LEVEL = 5


@dataclasses.dataclass(frozen=True)
class Route:
    name: str
    lowering: str
    options: Dict[str, object]
    warm: str
    continuation: bool
    why: str
    finish: Optional[str] = None


ROUTES: Dict[str, Route] = {
    "direct": Route(
        name="direct",
        lowering="prod_ineq",
        options={},
        warm="none",
        continuation=False,
        why="Direct POUNCE NLP formulation: H >= 0 with G*H <= 0.",
    ),
    "ncp_eq": Route(
        name="ncp_eq",
        lowering="prod_eq",
        options={},
        warm="none",
        continuation=False,
        why="Exact product / NCP equality, ordinary POUNCE.",
    ),
    "ncp_eq_l1": Route(
        name="ncp_eq_l1",
        lowering="prod_eq",
        options={"l1_exact_penalty_barrier": "yes"},
        warm="none",
        continuation=False,
        why="NCP equality through the opt-in l1 exact penalty-barrier wrapper.",
    ),
    "ncp_eq_l1_fallback": Route(
        name="ncp_eq_l1_fallback",
        lowering="prod_eq",
        options={"l1_fallback_on_restoration_failure": "yes"},
        warm="none",
        continuation=False,
        why="NCP equality, ordinary solve, l1 wrapper only after a restoration failure.",
    ),
    "scholtes_warm_full": Route(
        name="scholtes_warm_full",
        lowering="scholtes",
        options={},
        warm="full",
        continuation=True,
        why="Scholtes continuation with full primal/dual/barrier warm starts.",
    ),
    "scholtes_then_ncp": Route(
        name="scholtes_then_ncp",
        lowering="scholtes",
        options={},
        warm="full",
        continuation=True,
        why=(
            "Gate 0's supported route: Scholtes continuation with full warm "
            "starts to locate the branch, then one exact-product NCP-equality "
            "solve seeded from it."
        ),
        finish="prod_eq",
    ),
}

#: The route gh#776 asks Gate 1 to use.
SUPPORTED_ROUTE = "scholtes_then_ncp"

#: Gate 0's schedule, unchanged. It starts at 1.0 because a relaxation
#: that starts below a crossover never sees it; on this model the
#: crossover is the phase boundary itself, and `tau = 1` admits every
#: regime at once.
TAU_SCHEDULE: Tuple[float, ...] = (1e0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8)

RESTART_LADDER: Tuple[str, ...] = ("route", "primal", "cold_prev", "cold_x0")

CONTROLS: Dict[str, Dict[str, object]] = {
    "none": {},
    "no_acceptable": {"acceptable_iter": 0},
    "no_scaling": {"nlp_scaling_method": "none"},
    "upstream_heuristics": {
        "acceptable_progress_kappa": 0.0,
        "dual_inf_scale_kappa": 0.0,
        "obj_scale_certificate_threshold": 0.0,
    },
}

DEFAULT_CONTROLS: List[str] = ["none"]

#: Statuses that count as solved. `Solved_To_Acceptable_Level` is
#: included and the raw status is always kept, because on an MPCC the
#: difference between the two is itself a result -- that is what the
#: `no_acceptable` control is for.
OK_STATUS = (0, 1)


def options_for(route: Route, control: str, capture_log: bool = False) -> Dict[str, object]:
    opts = dict(BASE_OPTIONS)
    opts.update(route.options)
    opts.update(CONTROLS[control])
    if capture_log:
        opts["print_level"] = LOG_PRINT_LEVEL
    return opts
