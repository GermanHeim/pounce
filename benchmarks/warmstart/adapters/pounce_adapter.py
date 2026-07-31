"""POUNCE adapter — the only file in the suite that imports pounce.

Maps the four arms onto pounce's two algorithm paths:

===========  ===================================  =========================
arm          ``algorithm``                        warm-start payload
===========  ===================================  =========================
cold-ipm     ``interior-point`` (default)         none
cold-sqp     ``active-set-sqp``                   none
warm-ipm     ``interior-point``                   ``WarmStart`` (x, λ, z, μ)
warm-sqp     ``active-set-sqp``                   working set + previous x
===========  ===================================  =========================

Two deliberate choices worth knowing when reading the numbers:

* **The ``Problem`` is rebuilt at every step.** Several families move
  their variable or constraint bounds with θ, and bounds are fixed at
  construction. Rebuilding everywhere keeps the arms symmetric, and
  costs nothing in the measurement because only the ``solve()`` call
  is timed. It also sidesteps the fact that ``WarmStart`` applies its
  enabling options through ``add_option``, which would otherwise
  persist on a reused handle and quietly contaminate a later solve.

* **Tolerances are pinned on both paths** rather than left at their
  defaults, since the two paths have separate convergence-test knobs
  and an iteration-count comparison across differently-converged
  solves means nothing. The achieved KKT error is recorded per step
  so any residual asymmetry is visible in the data instead of being
  taken on trust.
"""

from __future__ import annotations

import time
from typing import Optional, Tuple

import numpy as np

import pounce

from ..spec import ParametricFamily, StepResult, WarmState
from ..sparsity import SparseCallbacks
from .base import SolverAdapter, is_warm

# ApplicationReturnStatus values that count as a solved step.
_OK_STATUS = (0, 1)  # SolveSucceeded, SolvedToAcceptableLevel


class PounceAdapter(SolverAdapter):
    name = "pounce"

    def __init__(self, max_iter: int = 500):
        self.max_iter = max_iter

    def supports(self, arm: str) -> bool:
        return arm in ("cold-ipm", "cold-sqp", "warm-ipm", "warm-sqp")

    # -- problem construction --------------------------------------

    def _build(self, family: ParametricFamily, callbacks, arm: str, tol: float):
        b = family.bounds()
        prob = pounce.Problem(
            n=family.n,
            m=family.m,
            problem_obj=callbacks,
            lb=b.lb,
            ub=b.ub,
            cl=b.cl,
            cu=b.cu,
        )
        prob.add_option("print_level", 0)
        prob.add_option("sb", "yes")
        if arm.endswith("sqp"):
            prob.add_option("algorithm", "active-set-sqp")
            prob.add_option("sqp_print_level", 0)
            prob.add_option("sqp_tol", tol)
            prob.add_option("sqp_constr_viol_tol", 1e-6)
            prob.add_option("sqp_max_iter", self.max_iter)
        else:
            prob.add_option("tol", tol)
            prob.add_option("constr_viol_tol", 1e-6)
            prob.add_option("max_iter", self.max_iter)
        return prob

    # -- one step --------------------------------------------------

    def solve(
        self,
        family: ParametricFamily,
        callbacks: SparseCallbacks,
        arm: str,
        x0: np.ndarray,
        warm: Optional[WarmState],
        step: int,
        tol: float,
    ) -> Tuple[StepResult, Optional[WarmState]]:
        prob = self._build(family, callbacks, arm, tol)
        callbacks.reset_counts()

        # Step 0 of a warm arm has nothing to warm from: it is a cold
        # solve, and is reported as one.
        use_warm = is_warm(arm) and warm is not None

        kwargs = {}
        if use_warm:
            if arm == "warm-sqp":
                kwargs["x0"] = warm.x
                if warm.working_set is not None:
                    kwargs["working_set"] = warm.working_set
            else:
                kwargs["warm_start"] = pounce.WarmStart(
                    x=warm.x,
                    lagrange=warm.mult_g,
                    zl=warm.mult_x_L,
                    zu=warm.mult_x_U,
                    mu=warm.mu,
                )
        else:
            kwargs["x0"] = np.asarray(x0, dtype=float)

        t0 = time.perf_counter()
        x, info = prob.solve(**kwargs)
        elapsed = time.perf_counter() - t0

        status = int(info.get("status", -99))
        ws = info.get("working_set")
        n_active = None
        if ws is not None:
            n_active = int(np.count_nonzero(ws[0]) + np.count_nonzero(ws[1]))

        result = StepResult(
            step=step,
            theta=[],  # filled by the runner, which owns the path
            success=status in _OK_STATUS,
            status=status,
            status_msg=str(info.get("status_msg", "")),
            iters=int(info.get("iter_count", -1)),
            solve_time=elapsed,
            obj=float(info.get("obj_val", np.nan)),
            kkt_error=float(
                info.get("final_unscaled_kkt_error",
                         info.get("final_kkt_error", np.nan))
            ),
            constr_viol=float(
                info.get("final_unscaled_constr_viol",
                         info.get("final_constr_viol", np.nan))
            ),
            n_active=n_active,
            # Recorded as counts (possibly 0) on the SQP path and as
            # None on the IPM path, which has no QP subproblems at all.
            # Keyed off the arm rather than off the value, so a warm
            # solve that converged without solving a single QP records
            # an honest 0 instead of looking like "not measured".
            n_qp_solves=int(info.get("n_qp_solves", 0))
            if arm.endswith("sqp")
            else None,
            n_qp_ws_changes=int(info.get("n_qp_ws_changes", 0))
            if arm.endswith("sqp")
            else None,
            **callbacks.counts(),
        )

        next_warm = WarmState(
            x=np.asarray(x, dtype=float).copy(),
            mult_g=np.asarray(info.get("mult_g"), dtype=float).copy()
            if info.get("mult_g") is not None
            else None,
            mult_x_L=np.asarray(info.get("mult_x_L"), dtype=float).copy()
            if info.get("mult_x_L") is not None
            else None,
            mult_x_U=np.asarray(info.get("mult_x_U"), dtype=float).copy()
            if info.get("mult_x_U") is not None
            else None,
            mu=float(info["mu"]) if info.get("mu") else None,
            working_set=(
                (np.asarray(ws[0]).copy(), np.asarray(ws[1]).copy())
                if ws is not None
                else None
            ),
        )
        return result, next_warm
