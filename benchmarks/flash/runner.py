"""The measurement protocol: one solve, and one route at one temperature.

Inherited from `mpcc/runner.py` and kept deliberately parallel to it, so
Gate 0's and Gate 1's numbers are comparable field for field. The three
rules that make cells comparable are the same:

1. Every route sees the same model, the same temperature, the same
   initial point and the same stopping requirements; only the lowering
   differs, because that is the thing under comparison.
2. **Source quantities are computed from the source model, never read
   off the lowered NLP.** ``info["final_constr_viol"]`` is the lowered
   problem's residual -- for a Scholtes stage it is a residual against
   ``G*H <= tau``, satisfied by points nowhere near MPCC-feasible, and
   on this model "nowhere near MPCC-feasible" means a vapor fraction
   that does not correspond to any phase state. Every ``source_*`` field
   comes from `FlashCase.source_feasibility` at the returned point.
3. A failed stage is recorded, then retried up the restart ladder, and
   the rung that finally worked is part of the result.
"""

from __future__ import annotations

import dataclasses
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from . import routes as R
from .lowering import LoweredFlash, lower
from .spec import FlashCase


@dataclasses.dataclass
class StageRecord:
    """One solve inside a route. A non-continuation route has one."""

    index: int
    tau: Optional[float]
    status: int
    status_msg: str
    accepted: bool
    warm_level: str
    restart_level: str
    iters: int
    wall_s: float
    lowering: str = ""


@dataclasses.dataclass
class SolveRecord:
    """One ``(case, temperature, route, control, start)`` cell."""

    case: str
    temperature_k: float
    route: str
    control: str
    start: str
    lowering: str
    ok: bool
    status: int
    status_msg: str
    x: Optional[List[float]]
    beta: Optional[float]
    sum_x: Optional[float]
    sum_y: Optional[float]
    regime: Optional[str]
    # -- source-level, in the model's own units --------------------
    source: Dict[str, float] = dataclasses.field(default_factory=dict)
    validation: Dict[str, object] = dataclasses.field(default_factory=dict)
    # -- POUNCE's own NLP diagnostics, kept separate on purpose ----
    nlp: Dict[str, float] = dataclasses.field(default_factory=dict)
    iters: int = 0
    outer_stages: int = 0
    restarts: int = 0
    restoration: Dict[str, int] = dataclasses.field(default_factory=dict)
    wall_s: float = 0.0
    stages: List[StageRecord] = dataclasses.field(default_factory=list)
    error: Optional[str] = None
    #: For a route with a finishing solve: whether that solve actually
    #: ran and was accepted, and what it said if not. Surfaced as its own
    #: field rather than left in `stages` because on this fixture it is
    #: the difference between running Gate 0's supported route and
    #: running half of it -- see `finish_status` in the report. A record
    #: whose `ok` is True can have a rejected finish, and reading the
    #: first without the second is how "the supported route works here"
    #: gets said when what works is its continuation.
    finish_applied: Optional[bool] = None
    finish_status_msg: Optional[str] = None


def _build_problem(nlp: LoweredFlash, opts: Dict[str, object]):
    import pounce

    big = 1e20
    prob = pounce.Problem(
        n=nlp.n,
        m=nlp.m,
        problem_obj=nlp,
        lb=np.where(np.isfinite(nlp.lb), nlp.lb, -big),
        ub=np.where(np.isfinite(nlp.ub), nlp.ub, big),
        cl=np.where(np.isfinite(nlp.cl), nlp.cl, -big),
        cu=np.where(np.isfinite(nlp.cu), nlp.cu, big),
    )
    for k, v in opts.items():
        prob.add_option(k, v)
    return prob


def solve_once(
    nlp: LoweredFlash,
    opts: Dict[str, object],
    x0: np.ndarray,
    warm_state: Optional[dict],
    warm_level: str,
) -> Tuple[np.ndarray, dict, float]:
    """One solve. ``warm_level`` is ``none``, ``primal`` or ``full``."""
    import pounce

    prob = _build_problem(nlp, opts)
    kwargs: Dict[str, object] = {"x0": np.asarray(x0, dtype=float)}
    if warm_level != "none" and warm_state is not None:
        if warm_level == "primal":
            ws = pounce.WarmStart(x=np.asarray(warm_state["x"], dtype=float))
        else:
            ws = pounce.WarmStart(
                x=np.asarray(warm_state["x"], dtype=float),
                lagrange=warm_state.get("mult_g"),
                zl=warm_state.get("mult_x_L"),
                zu=warm_state.get("mult_x_U"),
                mu=warm_state.get("mu"),
            )
        kwargs["warm_start"] = ws
    t0 = time.perf_counter()
    x, info = prob.solve(**kwargs)
    return np.asarray(x, dtype=float), info, time.perf_counter() - t0


def _nlp_block(info: dict) -> Dict[str, float]:
    """POUNCE's own KKT diagnostics, scaled and unscaled, kept apart from
    anything source-level."""
    keys = (
        "final_kkt_error",
        "final_dual_inf",
        "final_constr_viol",
        "final_compl",
        "final_unscaled_kkt_error",
        "final_unscaled_dual_inf",
        "final_unscaled_constr_viol",
        "final_unscaled_compl",
    )
    return {k: float(info.get(k, float("nan"))) for k in keys}


def _restoration_block(info: dict) -> Dict[str, int]:
    return {
        k: int(info.get(k, 0))
        for k in ("restoration_calls", "restoration_outer_iters", "restoration_inner_iters")
    }


def _warm_from(x: np.ndarray, info: dict) -> dict:
    return {
        "x": np.array(x, dtype=float),
        "mult_g": np.array(info["mult_g"], dtype=float),
        "mult_x_L": np.array(info["mult_x_L"], dtype=float),
        "mult_x_U": np.array(info["mult_x_U"], dtype=float),
        "mu": float(info.get("mu", 0.0)) or None,
    }


def _ladder(route_warm: str) -> List[Tuple[str, str]]:
    """``(restart level, warm level)`` in ladder order.

    A cold route's ladder collapses to one rung: ``route`` already *is*
    a cold start, and climbing to a warmer rung after a failure is not a
    restart.
    """
    if route_warm == "none":
        return [("route", "none")]
    return [
        ("route", route_warm),
        ("primal", "primal"),
        ("cold_prev", "none"),
        ("cold_x0", "none"),
    ]


def solve_route(
    case: FlashCase,
    temperature_k: float,
    route: R.Route,
    x0: np.ndarray,
    *,
    control: str = "none",
    start_label: str = "cold",
    warm_state: Optional[dict] = None,
) -> SolveRecord:
    """Run one route at one temperature and produce one record.

    ``warm_state`` seeds the *first* stage from a neighbouring
    temperature's solution; the continuation's own stage-to-stage warm
    starts are internal to the route and are not that. Keeping the two
    apart is what makes "warm along the path" a separate axis from
    "warm along the relaxation schedule" -- gh#776 asks for the first,
    and Gate 0 already measured the second.
    """
    from .validate import validate

    opts = R.options_for(route, control)
    rec = SolveRecord(
        case=case.name,
        temperature_k=float(temperature_k),
        route=route.name,
        control=control,
        start=start_label,
        lowering=route.lowering,
        ok=False,
        status=-99,
        status_msg="not run",
        x=None,
        beta=None,
        sum_x=None,
        sum_y=None,
        regime=None,
    )

    x_prev = np.asarray(x0, dtype=float)
    state = warm_state
    last_x: Optional[np.ndarray] = None
    last_info: Optional[dict] = None
    t0 = time.perf_counter()

    schedule = R.TAU_SCHEDULE if route.continuation else (None,)
    try:
        for idx, tau in enumerate(schedule):
            nlp = lower(case, temperature_k, route.lowering, tau)
            solved = False
            for restart, warm in _ladder(route.warm if idx or state is not None else "none"):
                seed = x_prev if restart != "cold_x0" else np.asarray(x0, dtype=float)
                x, info, wall = solve_once(
                    nlp, opts, seed, state if warm != "none" else None, warm
                )
                status = int(info.get("status", -99))
                ok = status in R.OK_STATUS
                rec.stages.append(
                    StageRecord(
                        index=idx,
                        tau=tau,
                        status=status,
                        status_msg=str(info.get("status_msg", "")),
                        accepted=ok,
                        warm_level=warm,
                        restart_level=restart,
                        iters=int(info.get("iter_count", info.get("iterations", 0)) or 0),
                        wall_s=wall,
                        lowering=route.lowering,
                    )
                )
                rec.iters += rec.stages[-1].iters
                if restart != "route":
                    rec.restarts += 1
                if ok:
                    last_x, last_info = x, info
                    x_prev = x
                    state = _warm_from(x, info)
                    solved = True
                    break
            rec.outer_stages += 1
            if not solved:
                break

        # The finishing solve: what turns a `tau`-feasible answer into
        # an MPCC-feasible one. Without it the continuation's last stage
        # is feasible for `G*H <= 1e-8`, which on this model is a vapor
        # fraction off the branch by `sqrt(1e-8)`.
        if route.finish is not None and last_x is not None:
            nlp = lower(case, temperature_k, route.finish)
            x, info, wall = solve_once(nlp, opts, last_x, state, "full")
            status = int(info.get("status", -99))
            rec.stages.append(
                StageRecord(
                    index=len(schedule),
                    tau=None,
                    status=status,
                    status_msg=str(info.get("status_msg", "")),
                    accepted=status in R.OK_STATUS,
                    warm_level="full",
                    restart_level="finish",
                    iters=int(info.get("iter_count", info.get("iterations", 0)) or 0),
                    wall_s=wall,
                    lowering=route.finish,
                )
            )
            rec.iters += rec.stages[-1].iters
            rec.outer_stages += 1
            rec.finish_applied = status in R.OK_STATUS
            rec.finish_status_msg = str(info.get("status_msg", ""))
            if status in R.OK_STATUS:
                last_x, last_info = x, info
                rec.lowering = route.finish
    except Exception as exc:  # pragma: no cover - solver-level failure
        rec.error = f"{type(exc).__name__}: {exc}"

    rec.wall_s = time.perf_counter() - t0
    if last_x is None or last_info is None:
        rec.status_msg = rec.error or "no stage converged"
        return rec

    rec.status = int(last_info.get("status", -99))
    rec.status_msg = str(last_info.get("status_msg", ""))
    rec.ok = rec.status in R.OK_STATUS
    rec.x = [float(v) for v in last_x]
    beta, x_l, y_v = case.unpack(last_x)
    rec.beta = float(beta)
    rec.sum_x = float(np.sum(x_l))
    rec.sum_y = float(np.sum(y_v))
    rec.regime = case.regime(last_x)
    rec.source = case.source_feasibility(last_x, temperature_k)
    rec.nlp = _nlp_block(last_info)
    rec.restoration = _restoration_block(last_info)
    rec.validation = validate(case, last_x, temperature_k)
    return rec


def cold_start(case: FlashCase, temperature_k: float) -> np.ndarray:
    """The engineering cold start: Wilson ``K``, then Rachford--Rice.

    Deliberately *not* the oracle's answer. A fixture started from the
    answer it is checked against measures nothing, and gh#776 asks the
    Gate 1 fixture to compare initialization behaviour across the
    switching points -- which requires a start that is genuinely
    ignorant of the regime.

    ``beta`` is nudged inside ``[0, 1]`` because an interior-point
    method started exactly on a bound has no interior to work in. The
    nudge is one-sided and small, and it is a *start*, so it cannot
    move the answer; it can and does move the iteration count, which is
    why it is here rather than in the solver's options.
    """
    from . import oracle, thermo

    k = thermo.wilson_k(temperature_k, case.pressure_pa, case.mixture)
    beta = oracle.rachford_rice(case.z, k)
    beta = float(np.clip(beta, 1e-3, 1.0 - 1e-3))
    x = case.z / (1.0 + beta * (k - 1.0))
    y = k * x
    return case.pack(beta, x, y)
