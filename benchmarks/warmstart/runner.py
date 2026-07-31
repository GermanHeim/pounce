"""The measurement protocol.

For each (family, scale) the runner walks the parameter path once per
arm and records every step. Three rules make the arms comparable, and
they are the whole reason this file exists rather than a loop inlined
in the CLI:

1. **Every arm sees the identical parameter sequence.** For scripted
   families that is trivial. For closed-loop families the path is
   generated once by the reference arm and replayed for the others —
   otherwise small solution differences would fan out into different
   problem sequences and the iteration counts would be measuring
   different work.

2. **Warm arms start their step-0 solve cold.** There is nothing to
   warm from at k=0. Reporting that step as warm would flatter the
   arm; reporting the path total without it would hide real work. It
   is recorded as-is and marked.

3. **Every step is checked against the reference solution.** A warm
   start that converges fast to the wrong point — a different local
   minimum, or the right point on the wrong face of a degenerate
   active set — is a failure, not a win. Speed numbers for a step
   that fails the check are still recorded, but the step is flagged
   and the report counts it separately.
"""

from __future__ import annotations

import dataclasses
from typing import Dict, List, Optional, Tuple

import numpy as np

from .adapters.base import REFERENCE_ARM, SolverAdapter, is_warm
from .families import make
from .spec import SCALES, ParametricFamily, StepResult, WarmState
from .sparsity import SparseCallbacks


def _ws_distance(a, b) -> Optional[int]:
    """Hamming distance between two working sets."""
    if a is None or b is None:
        return None
    return int(np.count_nonzero(a[0] != b[0]) + np.count_nonzero(a[1] != b[1]))


def _run_path(
    adapter: SolverAdapter,
    family: ParametricFamily,
    callbacks: SparseCallbacks,
    arm: str,
    path: List[np.ndarray],
    tol: float,
) -> Tuple[List[StepResult], List[np.ndarray]]:
    """Solve every step of a scripted path with one arm."""
    warm: Optional[WarmState] = None
    prev_ws = None
    results: List[StepResult] = []
    solutions: List[np.ndarray] = []

    for k, theta in enumerate(path):
        family.set_theta(theta)
        res, nxt = adapter.solve(
            family=family,
            callbacks=callbacks,
            arm=arm,
            x0=family.cold_x0(),
            warm=warm if is_warm(arm) else None,
            step=k,
            tol=tol,
        )
        res.theta = [float(v) for v in np.atleast_1d(theta)]
        res.ws_changed = _ws_distance(prev_ws, nxt.working_set if nxt else None)
        results.append(res)
        solutions.append(nxt.x if nxt is not None else np.full(family.n, np.nan))
        prev_ws = nxt.working_set if nxt is not None else None
        warm = nxt

    return results, solutions


def _run_adaptive_reference(
    adapter: SolverAdapter,
    family: ParametricFamily,
    callbacks: SparseCallbacks,
    scale: float,
    tol: float,
) -> Tuple[List[StepResult], List[np.ndarray], List[np.ndarray]]:
    """Run the reference arm on a closed-loop family, recording the path."""
    theta = family.initial_theta(scale)
    path: List[np.ndarray] = []
    results: List[StepResult] = []
    solutions: List[np.ndarray] = []

    for k in range(family.n_steps):
        family.set_theta(theta)
        path.append(np.atleast_1d(np.asarray(theta, dtype=float)).copy())
        res, nxt = adapter.solve(
            family=family,
            callbacks=callbacks,
            arm=REFERENCE_ARM,
            x0=family.cold_x0(),
            warm=None,
            step=k,
            tol=tol,
        )
        res.theta = [float(v) for v in path[-1]]
        results.append(res)
        solutions.append(nxt.x)
        theta = family.next_theta(nxt.x)

    return results, solutions, path


def _score_correctness(
    results: List[StepResult],
    solutions: List[np.ndarray],
    ref_solutions: List[np.ndarray],
    ref_results: List[StepResult],
    obj_tol: float,
    kkt_gate: float,
    viol_gate: float,
) -> None:
    """Judge each step, and record how it compares to the reference arm.

    Deliberately *not* "did it match the reference": on a nonconvex
    family the reference arm is a baseline, not ground truth — it can
    and does converge to a worse local minimum than a warm-started arm
    finds. Judging against it would score the better answer as wrong.

    So a step is judged on its own terms first — did it converge, is
    its KKT residual actually small, is it feasible — and only then
    compared to the reference, where the *sign* of the objective
    difference is what matters:

    * worse objective (beyond ``obj_tol``) → not correct. This is the
      failure mode a warm start causes: seeded inside the wrong active
      set or the wrong basin, it converges quickly to a worse point.
    * better objective → correct, and flagged as ``better`` so the
      report can say so rather than burying it.

    ``x_err`` is recorded throughout as a diagnostic but is not a gate:
    two solves can both be optimal and differ in ``x`` when the
    solution is not unique (a degenerate face, a flat direction).
    """
    for res, x, x_ref, ref in zip(results, solutions, ref_solutions, ref_results):
        scale = 1.0 + float(np.max(np.abs(x_ref))) if x_ref.size else 1.0
        res.x_err = (
            float(np.max(np.abs(x - x_ref))) / scale if x.size else float("inf")
        )
        res.obj_err = abs(res.obj - ref.obj) / (1.0 + abs(ref.obj))

        res.converged = bool(
            res.success
            and np.isfinite(res.kkt_error)
            and res.kkt_error <= kkt_gate
            and (not np.isfinite(res.constr_viol) or res.constr_viol <= viol_gate)
        )
        margin = obj_tol * (1.0 + abs(ref.obj))
        res.better = bool(ref.success and res.obj < ref.obj - margin)
        worse = bool(ref.success and res.obj > ref.obj + margin)
        res.correct = bool(res.converged and not worse)


def run_family_scale(
    adapter: SolverAdapter,
    family_name: str,
    scale_name: str,
    arms: List[str],
    tol: float = 1e-8,
    obj_tol: float = 1e-6,
    kkt_gate: float = 1e-4,
    viol_gate: float = 1e-5,
) -> dict:
    """Run every arm over one family at one scale. Returns a JSON-able dict."""
    scale = SCALES[scale_name]
    family = make(family_name)
    callbacks = SparseCallbacks(family)

    arm_results: Dict[str, List[StepResult]] = {}
    arm_solutions: Dict[str, List[np.ndarray]] = {}
    skipped: Dict[str, str] = {}

    path = family.theta_path(scale)
    if path is None:
        # Closed-loop: the reference arm defines the path.
        res, sol, path = _run_adaptive_reference(
            adapter, family, callbacks, scale, tol
        )
        arm_results[REFERENCE_ARM] = res
        arm_solutions[REFERENCE_ARM] = sol

    for arm in arms:
        if arm in arm_results:
            continue
        if not adapter.supports(arm):
            skipped[arm] = f"{adapter.name} does not support {arm}"
            continue
        res, sol = _run_path(adapter, family, callbacks, arm, path, tol)
        arm_results[arm] = res
        arm_solutions[arm] = sol

    if REFERENCE_ARM in arm_results:
        for arm, res in arm_results.items():
            _score_correctness(
                res,
                arm_solutions[arm],
                arm_solutions[REFERENCE_ARM],
                arm_results[REFERENCE_ARM],
                obj_tol,
                kkt_gate,
                viol_gate,
            )

    return {
        "family": family_name,
        "tags": dict(family.tags),
        "n": family.n,
        "m": family.m,
        "nnz_jac": callbacks.nnz_jac,
        "nnz_hess": callbacks.nnz_hess,
        "adaptive": family.adaptive,
        "scale": scale_name,
        "scale_factor": scale,
        "n_steps": len(path),
        "arms": {
            arm: [dataclasses.asdict(r) for r in res]
            for arm, res in arm_results.items()
        },
        "skipped": skipped,
    }
