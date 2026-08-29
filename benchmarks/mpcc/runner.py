"""The measurement protocol.

One cell of the matrix is ``(case, scaling, start, route, control)``, and
`run_cell` produces one `RouteRecord` for it. Three rules make the cells
comparable, and they are why this is a module rather than a loop in the
CLI:

1. **Every route sees the same source model, the same scaling, the same
   initial point and the same stopping requirements.** The lowering is
   the only thing that differs, and it differs because that is the thing
   under comparison. `routes.BASE_OPTIONS` pins the tolerances on every
   arm for the same reason the warm-start suite does: an iteration count
   compared across differently-converged solves measures nothing.

2. **Source quantities are computed from the source model, never read
   off the lowered NLP.** ``info["final_constr_viol"]`` is the lowered
   problem's residual -- for a Scholtes stage it is a residual against
   ``G*H <= tau``, which is satisfied by points that are nowhere near
   MPCC-feasible. Every ``source_*`` field in a record comes from
   `MpccCase.source_feasibility` evaluated at the returned point, and
   the NLP's own diagnostics are kept in a separate block.

3. **A failed stage is recorded, then retried up the ladder.** The
   restart level that finally worked is part of the result; a
   continuation that only converges because it fell back to cold starts
   at every stage is not a warm-start success, and the record has to be
   able to say so.
"""

from __future__ import annotations

import contextlib
import dataclasses
import os
import re
import sys
import tempfile
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from . import routes as R
from .lowering import LoweredNlp, lower
from .spec import MpccCase, RouteRecord, StageRecord
from .stationarity import classify


# --------------------------------------------------------------------
# iteration-log capture
# --------------------------------------------------------------------

_ITER_RE = re.compile(r"^\s*(\d+)(r?)\s+")


@contextlib.contextmanager
def capture_stdout():
    """Redirect file descriptor 1 into a temp file for the duration.

    The solver writes its iteration table from Rust, so a Python-level
    ``redirect_stdout`` does not see it; the descriptor has to move.
    Harness progress goes to stderr precisely so this can happen without
    swallowing it.
    """
    sys.stdout.flush()
    saved = os.dup(1)
    tmp = tempfile.TemporaryFile(mode="w+b")
    try:
        os.dup2(tmp.fileno(), 1)
        yield tmp
    finally:
        sys.stdout.flush()
        os.dup2(saved, 1)
        os.close(saved)


def parse_iteration_log(text: str) -> Dict[str, object]:
    """Counters gh#794 asks for "where available", read off the table.

    POUNCE prints the upstream Ipopt iteration table, ``iter objective
    inf_pr inf_du lg(mu) ||d|| lg(rg) alpha_du alpha_pr ls``. Two of the
    required counters are only visible there:

    ``inertia_corrections``  iterations whose ``lg(rg)`` column carries a
                             number rather than ``-``; that column is the
                             primal regularisation ``delta_x`` the
                             inertia correction applied.
    ``restoration_iters``    iterations whose index carries the ``r``
                             suffix.

    Filter resets are **not** in the table and POUNCE emits no message
    for them, so ``filter_resets`` is reported as ``None`` rather than
    as 0 -- an unmeasured quantity and a measured zero are different
    claims. The parser reports ``parsed: False`` if it never recognised
    a table, so a format change shows up as a missing measurement
    instead of a silent zero.
    """
    inertia = 0
    restoration = 0
    iters = 0
    seen_header = "lg(rg)" in text
    for line in text.splitlines():
        m = _ITER_RE.match(line)
        if not m:
            continue
        toks = line.split()
        if len(toks) < 10:
            continue
        iters += 1
        if m.group(2) == "r":
            restoration += 1
        rg = toks[6]
        if rg not in ("-", "--"):
            try:
                float(rg)
                inertia += 1
            except ValueError:
                pass
    return {
        "parsed": bool(seen_header and iters),
        "log_iters": iters,
        "inertia_corrections": inertia if seen_header else None,
        "restoration_iters": restoration if seen_header else None,
        "filter_resets": None,
        "filter_resets_note": "not emitted by POUNCE; unmeasured, not zero",
    }


# --------------------------------------------------------------------
# a single solve
# --------------------------------------------------------------------


def _build_problem(nlp: LoweredNlp, opts: Dict[str, object]):
    import pounce

    big = 1e20
    lb = np.where(np.isfinite(nlp.lb), nlp.lb, -big)
    ub = np.where(np.isfinite(nlp.ub), nlp.ub, big)
    cl = np.where(np.isfinite(nlp.cl), nlp.cl, -big)
    cu = np.where(np.isfinite(nlp.cu), nlp.cu, big)
    prob = pounce.Problem(
        n=nlp.n, m=nlp.m, problem_obj=nlp, lb=lb, ub=ub, cl=cl, cu=cu
    )
    for k, v in opts.items():
        prob.add_option(k, v)
    return prob


def solve_once(
    nlp: LoweredNlp,
    opts: Dict[str, object],
    x0: np.ndarray,
    warm_state: Optional[dict],
    warm_level: str,
    capture_log: bool,
) -> Tuple[np.ndarray, dict, float, Dict[str, object]]:
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

    log: Dict[str, object] = {"parsed": False}
    t0 = time.perf_counter()
    if capture_log:
        with capture_stdout() as tmp:
            x, info = prob.solve(**kwargs)
        tmp.seek(0)
        log = parse_iteration_log(tmp.read().decode("utf-8", "replace"))
    else:
        x, info = prob.solve(**kwargs)
    wall = time.perf_counter() - t0
    return np.asarray(x, dtype=float), info, wall, log


def _nlp_block(info: dict) -> Dict[str, float]:
    """POUNCE's own NLP/KKT diagnostics, scaled and unscaled, kept apart
    from anything source-level."""
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


def _warm_from_info(x: np.ndarray, info: dict) -> dict:
    return {
        "x": np.array(x, dtype=float),
        "mult_g": np.array(info["mult_g"], dtype=float),
        "mult_x_L": np.array(info["mult_x_L"], dtype=float),
        "mult_x_U": np.array(info["mult_x_U"], dtype=float),
        "mu": float(info.get("mu", 0.0)) or None,
    }


# --------------------------------------------------------------------
# the cell
# --------------------------------------------------------------------


def _ladder_levels(route_warm: str) -> List[Tuple[str, str]]:
    """``(restart level, warm level)`` pairs, in ladder order.

    A cold route's ladder collapses: ``route`` already *is* a cold
    start, so ``primal`` and ``cold_prev`` would be either the same
    solve or a warmer one, and climbing to a warmer rung after a failure
    is not a restart. It is reported as a one-rung ladder rather than
    silently sharing the warm routes' four.
    """
    if route_warm == "none":
        return [("route", "none")]
    if route_warm == "primal":
        return [("route", "primal"), ("cold_prev", "none"), ("cold_x0", "none")]
    return [
        ("route", "full"),
        ("primal", "primal"),
        ("cold_prev", "none"),
        ("cold_x0", "none"),
    ]


def run_cell(
    case: MpccCase,
    route: R.Route,
    control: str,
    start_name: str,
    scaling: str,
    scale_vec: np.ndarray,
    capture_log: bool = True,
) -> RouteRecord:
    """Run one matrix cell and build its record.

    ``case`` is already rescaled; ``scale_vec`` is kept only so the
    record can report the returned point in the *unscaled* source
    model's coordinates alongside the solved one.
    """
    opts = R.options_for(route, control, capture_log)
    x0 = case.starts[start_name]
    rec = RouteRecord(
        case=case.name,
        klass=case.klass,
        scaling=scaling,
        start=start_name,
        route=route.name,
        control=control,
        lowering=route.lowering,
        ok=False,
        status=-99,
        status_msg="not run",
        obj=None,
        x=None,
    )

    stages: List[StageRecord] = []
    warm_state: Optional[dict] = None
    prev_x = np.array(x0, dtype=float)
    best_x: Optional[np.ndarray] = None
    last_info: Optional[dict] = None
    log_totals = {"inertia_corrections": 0, "restoration_iters": 0, "parsed": True}
    t_all = time.perf_counter()

    schedule: List[Tuple[Optional[float], str]]
    if route.continuation:
        schedule = [(R.TAU_SCHEDULE[0], "initial")] + [
            (t, "geometric x0.1") for t in R.TAU_SCHEDULE[1:]
        ]
    else:
        schedule = [(None, "not a relaxation")]

    bisections_left = R.TAU_BISECTIONS
    last_accepted_tau: Optional[float] = None
    idx = 0
    si = 0
    try:
        while si < len(schedule):
            tau, tau_reason = schedule[si]
            nlp = lower(case, route.lowering, tau)
            accepted = False
            for restart_level, warm_level in _ladder_levels(route.warm):
                if restart_level == "cold_x0":
                    seed = np.array(x0, dtype=float)
                    state = None
                elif restart_level == "cold_prev":
                    seed = prev_x
                    state = None
                elif route.warm == "none":
                    # The cold continuation arm's stages must be
                    # *independent*: gh#794 asks for "independent solves
                    # with cold starts", and a stage seeded with the
                    # previous stage's x is a primal warm start whatever
                    # its multipliers say. Carrying the point over would
                    # have made this arm a second copy of
                    # `scholtes_warm_primal` and destroyed the reference
                    # the two warm arms' iteration counts are measured
                    # against.
                    seed = np.array(x0, dtype=float)
                    state = None
                else:
                    seed = prev_x
                    state = warm_state
                eff_warm = warm_level if state is not None else "none"
                mu_in = None
                if eff_warm == "full" and state is not None:
                    mu_in = state.get("mu")
                x, info, wall, log = solve_once(
                    nlp, opts, seed, state, eff_warm, capture_log
                )
                ok = int(info["status"]) in R.OK_STATUS
                if log.get("parsed"):
                    log_totals["inertia_corrections"] += log.get("inertia_corrections") or 0
                    log_totals["restoration_iters"] += log.get("restoration_iters") or 0
                else:
                    log_totals["parsed"] = False
                stages.append(
                    StageRecord(
                        index=idx,
                        tau=tau,
                        tau_reason=tau_reason,
                        status=int(info["status"]),
                        status_msg=str(info["status_msg"]),
                        accepted=ok,
                        warm_level=eff_warm,
                        restart_level=restart_level,
                        restart_reason=(
                            "first attempt"
                            if restart_level == "route"
                            else "previous rung did not converge"
                        ),
                        iters=int(info["iter_count"]),
                        mu_in=mu_in,
                        mu_out=float(info.get("mu", 0.0)),
                        mu_reason=(
                            "cold: mu_init default"
                            if eff_warm == "none"
                            else "seeded from the previous stage's converged mu"
                            if eff_warm == "full"
                            else "primal-only seed: the initializer picks mu from the "
                            "point's own residuals"
                        ),
                        wall_s=wall,
                        nlp=_nlp_block(info),
                        restoration=_restoration_block(info),
                        regime=case.regime(x),
                    )
                )
                idx += 1
                last_info = info
                if ok:
                    accepted = True
                    best_x = x
                    prev_x = x
                    warm_state = _warm_from_info(x, info)
                    last_accepted_tau = tau
                    break

            if accepted:
                si += 1
                continue

            if (
                route.continuation
                and bisections_left > 0
                and last_accepted_tau is not None
                and tau is not None
            ):
                bisections_left -= 1
                mid = float(np.sqrt(last_accepted_tau * tau))
                schedule = (
                    schedule[:si]
                    + [(mid, "bisected after a rejected stage")]
                    + schedule[si:]
                )
                continue
            break
    except Exception as exc:  # pragma: no cover - solver-side failure
        rec.error = f"{type(exc).__name__}: {exc}"

    rec.wall_s = time.perf_counter() - t_all
    rec.stages = stages
    rec.outer_stages = len({s.tau for s in stages}) if route.continuation else len(stages)
    rec.accepted_stages = sum(1 for s in stages if s.accepted)
    rec.rejected_stages = sum(1 for s in stages if not s.accepted)
    rec.restarts = sum(1 for s in stages if s.restart_level != "route")
    order = {lvl: i for i, lvl in enumerate(R.RESTART_LADDER)}
    used = [s.restart_level for s in stages]
    rec.max_restart_level = max(used, key=lambda l: order.get(l, 0)) if used else "none"
    rec.iters = sum(s.iters for s in stages)
    rec.log_counters = dict(log_totals)
    rec.log_counters["filter_resets"] = None
    rec.log_counters["filter_resets_note"] = (
        "not emitted by POUNCE; unmeasured, not zero"
    )
    if stages:
        rec.restoration = {
            k: sum(s.restoration.get(k, 0) for s in stages)
            for k in ("restoration_calls", "restoration_outer_iters", "restoration_inner_iters")
        }
        rec.regime_changes = sum(
            1
            for a, b in zip(stages, stages[1:])
            if a.regime is not None and b.regime is not None and a.regime != b.regime
        )

    if last_info is not None:
        rec.status = int(last_info["status"])
        rec.status_msg = str(last_info["status_msg"])
        rec.nlp = _nlp_block(last_info)

    if best_x is not None:
        rec.ok = True
        rec.x = [float(v) for v in best_x]
        rec.obj = float(case.objective.value(best_x))
        rec.source = case.source_feasibility(best_x)
        rec.regime = case.regime(best_x)
        rec.stationarity = classify(case, best_x)
        val: Dict[str, object] = {}
        for fn in case.validators:
            val.update(fn(case, best_x))
        rec.validation = val
        # The point in the *unscaled* source model, so records from the
        # two scaling legs can be compared without the reader doing the
        # arithmetic.
        rec.validation["x_unscaled"] = [
            float(v) for v in np.asarray(best_x) * np.asarray(scale_vec)
        ]
    elif rec.error is None:
        rec.ok = False

    return rec
