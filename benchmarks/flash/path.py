"""The temperature traversal: both directions, cold and warm.

gh#776 asks Gate 1 for two things this module owns:

* "Continue the operating parameter in both directions to expose
  branch/hysteresis artifacts."
* "Compare POUNCE solution routes and warm starts across switching
  points."

Both are about *path dependence*, and neither is visible in a single
solve. A flash at a fixed ``(T, P, z)`` has one answer; a solver that
reports a different one depending on which side it arrived from is
reporting on its own initialization, and on this model that failure has
a specific shape -- a warm start carried across a switch point seeds the
next solve inside the regime it just left, which is precisely the
starting point from which a converged-but-wrong single-phase answer is
reachable.

The four legs
-------------

``up_cold`` / ``down_cold``
    Every temperature solved from its own Wilson/Rachford--Rice start,
    independent of its neighbours. These two legs must agree point for
    point: they solve identical problems from identical starts, in a
    different order. A disagreement between them is a defect in the
    harness, not a physical result, and `hysteresis` reports it as such.

``up_warm`` / ``down_warm``
    Each temperature warm-started from the previous one's converged
    point, multipliers and barrier parameter included. These are the
    legs that can differ from each other and from the cold legs, and
    where they do, the difference *is* the measurement.

The cold legs are therefore the control and the warm legs are the
treatment, which is why all four run rather than the two that would
"cover the path".

What counts as hysteresis
--------------------------

Not "the two directions took different iteration counts" -- they will,
and that is a warm-start measurement rather than a branch artifact. The
reportable thing is a **different converged answer at the same
temperature**: a different regime label, or a ``beta`` differing by more
than the complementarity accuracy floor. `hysteresis` separates the two
and reports the iteration difference as a number rather than as a
verdict.
"""

from __future__ import annotations

import dataclasses
from typing import Dict, List, Optional, Tuple

import numpy as np

from . import routes as R
from .runner import SolveRecord, cold_start, solve_route
from .spec import FlashCase

#: The four legs, as ``(direction, start mode)``.
LEGS: Tuple[Tuple[str, str], ...] = (
    ("up", "cold"),
    ("down", "cold"),
    ("up", "warm"),
    ("down", "warm"),
)

#: Two answers at the same temperature differing by less than this are
#: the same answer. It is the complementarity accuracy floor from Gate 0
#: -- `sqrt(tol)` -- and not the solver tolerance: at a switch point the
#: pair is biactive and both sides are pinned only to `sqrt(tol)`, so a
#: threshold at `tol` would call correct code path-dependent.
SAME_ANSWER_TOL = 1e-4


@dataclasses.dataclass
class Leg:
    """One traversal: a direction, a start mode, and its records."""

    direction: str
    start_mode: str
    route: str
    control: str
    records: List[SolveRecord]

    @property
    def temperatures(self) -> List[float]:
        return [r.temperature_k for r in self.records]

    @property
    def n_failed(self) -> int:
        return sum(1 for r in self.records if not r.ok)

    @property
    def total_iters(self) -> int:
        return sum(r.iters for r in self.records)

    @property
    def regime_sequence(self) -> List[str]:
        """Regimes in *ascending temperature* order, whichever way the leg ran.

        Normalizing here rather than at the call site is deliberate: a
        descending leg's raw sequence reads ``vapor ... liquid``, and
        comparing it against an ascending one without reversing it first
        would report a "difference" on every well-behaved run.
        """
        rows = sorted(self.records, key=lambda r: r.temperature_k)
        return [r.regime or "failed" for r in rows]

    def by_temperature(self) -> Dict[float, SolveRecord]:
        return {r.temperature_k: r for r in self.records}


def traverse(
    case: FlashCase,
    route: R.Route,
    *,
    direction: str = "up",
    start_mode: str = "cold",
    control: str = "none",
    temperatures=None,
    progress=None,
) -> Leg:
    """Walk the temperature path once.

    ``start_mode="warm"`` carries the previous temperature's converged
    state -- primal, all three multiplier blocks and ``mu`` -- into the
    next solve. It is *not* carried across a failure: a failed solve has
    no state worth propagating, and propagating it anyway would turn one
    bad point into a bad tail and hide where the trouble started.
    """
    if direction not in ("up", "down"):
        raise ValueError(f"direction must be 'up' or 'down', got {direction!r}")
    if start_mode not in ("cold", "warm"):
        raise ValueError(f"start_mode must be 'cold' or 'warm', got {start_mode!r}")

    temps = np.asarray(
        case.temperatures_k if temperatures is None else temperatures, dtype=float
    )
    temps = np.sort(temps)
    if direction == "down":
        temps = temps[::-1]

    records: List[SolveRecord] = []
    warm_state: Optional[dict] = None
    prev_x: Optional[np.ndarray] = None
    for t in temps:
        t = float(t)
        if start_mode == "warm" and prev_x is not None:
            x0 = prev_x
        else:
            x0 = cold_start(case, t)
        rec = solve_route(
            case,
            t,
            route,
            x0,
            control=control,
            start_label=start_mode,
            warm_state=warm_state if start_mode == "warm" else None,
        )
        records.append(rec)
        if progress is not None:
            progress(rec)
        if rec.ok and rec.x is not None:
            prev_x = np.asarray(rec.x, dtype=float)
            # Only the primal is carried between temperatures here. The
            # multiplier/barrier state belongs to a *different* problem
            # -- a different temperature, hence different rows -- and
            # Gate 0 measured the full-state warm start where it is
            # defined, which is between the stages of one continuation.
            warm_state = {"x": prev_x}
        else:
            prev_x = None
            warm_state = None
    return Leg(direction, start_mode, route.name, control, records)


@dataclasses.dataclass
class Hysteresis:
    """What the four legs did or did not agree about."""

    #: ``(leg a, leg b) -> [(T, what differed)]``, empty when they agree.
    disagreements: Dict[str, List[Tuple[float, str]]]
    #: Iteration totals per leg. A number, not a verdict.
    iterations: Dict[str, int]
    failures: Dict[str, int]
    #: The cold legs solve identical problems from identical starts and
    #: must agree; a disagreement between them is a harness defect
    #: rather than a physical result, so it is called out separately.
    cold_legs_agree: bool

    @property
    def path_dependent(self) -> bool:
        return any(v for v in self.disagreements.values())


def _label(leg: Leg) -> str:
    return f"{leg.direction}_{leg.start_mode}"


def compare(legs: List[Leg]) -> Hysteresis:
    """Compare every pair of legs at every shared temperature."""
    disagreements: Dict[str, List[Tuple[float, str]]] = {}
    for i, a in enumerate(legs):
        for b in legs[i + 1 :]:
            key = f"{_label(a)} vs {_label(b)}"
            rows: List[Tuple[float, str]] = []
            ba = a.by_temperature()
            bb = b.by_temperature()
            for t in sorted(set(ba) & set(bb)):
                ra, rb = ba[t], bb[t]
                if ra.ok != rb.ok:
                    rows.append((t, f"one leg failed: {ra.ok} vs {rb.ok}"))
                    continue
                if not ra.ok:
                    continue
                if ra.regime != rb.regime:
                    rows.append((t, f"regime {ra.regime} vs {rb.regime}"))
                elif abs((ra.beta or 0.0) - (rb.beta or 0.0)) > SAME_ANSWER_TOL:
                    rows.append((t, f"beta {ra.beta:.6g} vs {rb.beta:.6g}"))
            disagreements[key] = rows

    cold = [lg for lg in legs if lg.start_mode == "cold"]
    cold_key = (
        f"{_label(cold[0])} vs {_label(cold[1])}" if len(cold) == 2 else None
    )
    return Hysteresis(
        disagreements=disagreements,
        iterations={_label(lg): lg.total_iters for lg in legs},
        failures={_label(lg): lg.n_failed for lg in legs},
        cold_legs_agree=(
            True if cold_key is None else not disagreements.get(cold_key)
        ),
    )
