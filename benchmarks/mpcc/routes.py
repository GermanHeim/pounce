"""The POUNCE configurations under comparison, and the kill switches.

Every configuration in gh#794's "required configurations" list appears
here, and nothing else does. A route is a *lowering* plus a set of
solver options plus, for the continuation routes, a warm-start level.

The base options are pinned rather than defaulted
-------------------------------------------------

``BASE_OPTIONS`` is applied to every route, because an iteration count
compared across differently-converged solves measures nothing. Two of
its entries are not conveniences:

``bound_relax_factor = 0``
    The default (1e-8) relaxes every constraint bound before the solve,
    which on this corpus means the solver is allowed to return
    ``G = -1e-8``. That is a source-level sign violation of the same
    order as the complementarity products the report exists to measure,
    and it would show up as "the route is slightly infeasible in the
    source model" on every single record.

``honor_original_bounds = yes``
    The returned point is pushed back inside the user's own bounds, so
    the ``x`` in a record is a point in the source model's feasible box
    rather than in the relaxed one.

The kill-switch controls
------------------------

gh#794 requires the kill-switch controls to be run *before* attributing
a failure to a new mechanism. Each entry in `CONTROLS` disables one
mechanism that could otherwise explain an outcome:

``no_acceptable``
    ``acceptable_iter = 0`` removes ``Solved_To_Acceptable_Level``
    entirely. An MPCC solve that stops on the acceptable-level criterion
    has stopped at a point with a *weaker* certificate, and on a
    degenerate problem that is exactly where it will stop; a route whose
    success rate collapses under this control was being carried by it.

``no_scaling``
    ``nlp_scaling_method = none``. The complementarity product row's
    entries are quadratic in the variable magnitudes, so gradient-based
    scaling behaves very differently on the ``skew`` scaling leg than on
    ``unit``. This separates "the route works" from "the scaling worked".

``upstream_heuristics``
    Sets ``acceptable_progress_kappa``, ``dual_inf_scale_kappa`` and
    ``obj_scale_certificate_threshold`` to 0, which each option's own
    documentation describes as restoring bit-for-bit upstream Ipopt
    behaviour. This is the control that answers "is this outcome a
    POUNCE-only mechanism?" -- which is the first question anyone will
    ask of a result that is about to become a POUNCE issue.

``no_presolve``
    ``presolve = no``. Only informative against ``ncp_eq_auto_l1``,
    whose whole mechanism is a presolve phase; it is the control that
    shows whether that route did anything.
"""

from __future__ import annotations

import dataclasses
from typing import Dict, List, Optional, Tuple

#: Applied to every solve in every route.
BASE_OPTIONS: Dict[str, object] = {
    "print_level": 0,
    "tol": 1e-8,
    "max_iter": 300,
    "bound_relax_factor": 0.0,
    "honor_original_bounds": "yes",
}

#: Print level used when the run captures the iteration log (see
#: `runner.capture_log`). 5 is the level that emits the standard
#: iteration table the log parser reads.
LOG_PRINT_LEVEL = 5


@dataclasses.dataclass(frozen=True)
class Route:
    name: str
    lowering: str
    options: Dict[str, object]
    #: "none" for a single solve; otherwise the warm-start level carried
    #: between continuation stages.
    warm: str
    continuation: bool
    why: str
    #: Optional lowering for one final solve after the schedule runs
    #: out, warm-started from the last accepted stage. This is what
    #: turns a continuation's `tau`-feasible answer into an
    #: MPCC-feasible one: the relaxation locates the branch, and a
    #: single exact-product solve seeded inside it drives the
    #: complementarity to zero. `None` for every other route.
    finish: Optional[str] = None


#: Warm-start levels for the continuation routes.
#:
#: ``none``    every stage is an independent cold solve from the case's
#:             own initial point -- the "explicit Scholtes continuation
#:             using independent solves with cold starts" arm. The point
#:             of it is that it is the only arm whose stages are
#:             genuinely independent, so it is the reference the warm
#:             arms' iteration counts mean something against.
#: ``primal``  the previous stage's ``x`` and nothing else. No
#:             multipliers, no barrier: the initializer has to invent a
#:             dual point the way a cold solve does.
#: ``full``    ``x``, all three multiplier blocks, and the converged
#:             ``mu`` threaded into the next stage's barrier.
WARM_LEVELS = ("none", "primal", "full")

ROUTES: Dict[str, Route] = {
    "direct": Route(
        name="direct",
        lowering="prod_ineq",
        options={},
        warm="none",
        continuation=False,
        why="Direct POUNCE NLP formulation: G,H >= 0 with G*H <= 0.",
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
    "ncp_eq_auto_l1": Route(
        name="ncp_eq_auto_l1",
        lowering="prod_eq",
        options={
            "presolve": "yes",
            "presolve_licq_check": "yes",
            "presolve_licq_action": "auto_l1",
        },
        warm="none",
        continuation=False,
        why="NCP equality with presolve's LICQ check routing degeneracy into the l1 wrapper.",
    ),
    "scholtes_cold": Route(
        name="scholtes_cold",
        lowering="scholtes",
        options={},
        warm="none",
        continuation=True,
        why="Explicit Scholtes continuation, independent solves with cold starts.",
    ),
    "scholtes_warm_primal": Route(
        name="scholtes_warm_primal",
        lowering="scholtes",
        options={},
        warm="primal",
        continuation=True,
        why="Scholtes continuation with primal-only warm starts.",
    ),
    "scholtes_warm_full": Route(
        name="scholtes_warm_full",
        lowering="scholtes",
        options={},
        warm="full",
        continuation=True,
        why="Scholtes continuation with full primal/dual/barrier warm starts.",
    ),
    # The composition, and the one gh#794 recommends as the default
    # where the branch is not known in advance. Neither half is
    # sufficient on its own: the continuation always converges but its
    # answer is only ever feasible for `G*H <= tau`, and the
    # exact-product solve returns an MPCC-feasible point but fails from
    # a cold start on the cases where a pair is biactive. Run in
    # sequence they cover each other, which is a measurement (see the
    # route summary) rather than a hope.
    # The same composition finishing on `G*H <= 0` instead of `G*H = 0`.
    # The two lowerings have **identical feasible sets** -- with
    # `G, H >= 0` the inequality is active only at `G*H = 0` -- so this
    # is not a weaker finish; it is the same finish stated without
    # adding an equality row. That distinction is invisible on this
    # corpus and decisive off it: an equality row counts against the
    # `n_x_var < n_c` degrees-of-freedom gate that
    # `application.rs` mirrors from upstream Ipopt
    # (`IpOrigIpoptNLP.cpp:299`), and a *square* source model -- which
    # every equilibrium-stage process model is -- has no slack to spare.
    # gh#776's Gate 1 flash is refused by that gate at every one of its
    # 34 temperatures, so `scholtes_then_ncp` runs only its continuation
    # half there. This arm exists to measure whether the inequality
    # finish can simply replace the equality one.
    "scholtes_then_ineq": Route(
        name="scholtes_then_ineq",
        lowering="scholtes",
        options={},
        warm="full",
        continuation=True,
        why=(
            "Scholtes continuation with full warm starts, then one direct "
            "G*H <= 0 solve seeded from it. Same feasible set as the NCP "
            "equality finish, without adding an equality row."
        ),
        finish="prod_ineq",
    ),
    "scholtes_then_ncp": Route(
        name="scholtes_then_ncp",
        lowering="scholtes",
        options={},
        warm="full",
        continuation=True,
        why=(
            "Scholtes continuation with full warm starts to locate the branch, "
            "then one exact-product NCP-equality solve seeded from it."
        ),
        finish="prod_eq",
    ),
}

#: The relaxation schedule. It starts at 1.0 rather than at a small
#: value because two cases have an analytically known crossover inside
#: `[0.25, 1]` -- `infeasible_pair` becomes infeasible below tau = 1/4
#: and `selector_theta_*` loses its fractional point there -- and a
#: schedule that starts below the crossover would never see either.
TAU_SCHEDULE: Tuple[float, ...] = (1e0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8)

#: The restart ladder, in the order it is climbed when a stage fails.
#: Each rung is a strictly weaker claim about the previous stage's
#: state, and the last rung throws that state away entirely.
#:
#: ``route``      the route's own warm level.
#: ``primal``     previous stage's x only.
#: ``cold_prev``  cold, but started from the previous stage's x.
#: ``cold_x0``    cold from the case's own initial point.
#:
#: If every rung fails, the schedule is bisected once -- the next tau is
#: the geometric mean of the last accepted tau and the failed one -- and
#: the ladder is climbed again. A second failure stops the continuation
#: and the route reports the last accepted stage, flagged.
RESTART_LADDER: Tuple[str, ...] = ("route", "primal", "cold_prev", "cold_x0")
TAU_BISECTIONS = 1

CONTROLS: Dict[str, Dict[str, object]] = {
    "none": {},
    "no_acceptable": {"acceptable_iter": 0},
    "no_scaling": {"nlp_scaling_method": "none"},
    "upstream_heuristics": {
        "acceptable_progress_kappa": 0.0,
        "dual_inf_scale_kappa": 0.0,
        "obj_scale_certificate_threshold": 0.0,
    },
    "no_presolve": {"presolve": "no"},
}

#: Controls run by default. The full matrix is every route x every
#: control, which is mostly redundant -- `no_presolve` says nothing
#: about a route that never turns presolve on. `run --controls all`
#: runs the lot anyway when a specific attribution needs it.
DEFAULT_CONTROLS: List[str] = ["none"]

#: Statuses that count as a solved stage. `Solved_To_Acceptable_Level`
#: is included, and every record keeps the raw status, because on this
#: corpus the distinction between the two is itself a result: see the
#: `no_acceptable` control.
OK_STATUS = (0, 1)


def options_for(route: Route, control: str, capture_log: bool) -> Dict[str, object]:
    opts = dict(BASE_OPTIONS)
    opts.update(route.options)
    opts.update(CONTROLS[control])
    if capture_log:
        opts["print_level"] = LOG_PRINT_LEVEL
    return opts
