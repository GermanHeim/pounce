"""POUNCE adapter — the only file in the suite that imports pounce.

Maps the four arms onto pounce's two algorithm paths:

===========  ===================================  =========================
arm          ``algorithm``                        warm-start payload
===========  ===================================  =========================
cold-ipm     ``interior-point`` (default)         none
cold-sqp     ``active-set-sqp``                   none
warm-ipm     ``interior-point``                   ``WarmStart`` (x, λ, z, μ)
warm-sqp     ``active-set-sqp``                   working set + previous x
cold-qp-ipm  ``pounce.solve_qp`` (pounce-convex)  none
warm-qp-ipm  ``pounce.solve_qp``                  previous ``QpResult``
===========  ===================================  =========================

Two deliberate choices worth knowing when reading the numbers:

* **The ``Problem`` is rebuilt at every step.** Several families move
  their variable or constraint bounds with θ, and bounds are fixed at
  construction. Rebuilding everywhere keeps the arms symmetric, and
  costs nothing in the measurement because only the ``solve()`` call
  is timed. It also sidesteps the fact that ``WarmStart`` applies its
  enabling options through ``add_option``, which would otherwise
  persist on a reused handle and quietly contaminate a later solve.

* **The QP arms do not go through the callback interface at all.** The
  convex solver takes matrix data, so each step assembles ``(P, c, A,
  b, G, h, lb, ub)`` from the family (see :mod:`..qpform`) and hands it
  over. The assembly is inside the timed region and routed through the
  harness's counters, because it is work a caller genuinely has to do —
  but it happens *once per step*, where the callback-driven arms
  re-evaluate every iteration. That is a real advantage of the QP path
  on a QP, not a measurement artifact, and it is why the report keeps
  these arms in their own section rather than in the headline table.
  ``check_psd`` is off: the suite's self-test already proves ``P`` is
  positive semidefinite for every family that claims to be a QP, so
  leaving the guard on would time an O(n³) eigenvalue decomposition
  that a caller who knew their problem would not pay.

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

from .. import qpform
from ..spec import ParametricFamily, StepResult, WarmState
from ..sparsity import SparseCallbacks
from .base import ARMS, QP_ARMS, SolverAdapter, is_sqp, is_warm, uses_homotopy

# ApplicationReturnStatus values that count as a solved step.
_OK_STATUS = (0, 1)  # SolveSucceeded, SolvedToAcceptableLevel

# `solve_qp` status strings that count as a solved step.
_OK_QP_STATUS = ("optimal", "optimal_inaccurate")


class PounceAdapter(SolverAdapter):
    name = "pounce"

    def __init__(self, max_iter: int = 500):
        self.max_iter = max_iter

    def supports(self, arm: str) -> bool:
        return arm in ARMS

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
        if is_sqp(arm):
            prob.add_option("algorithm", "active-set-sqp")
            prob.add_option("sqp_print_level", 0)
            prob.add_option("sqp_tol", tol)
            prob.add_option("sqp_constr_viol_tol", 1e-6)
            # The only difference between an arm and its `-hom` twin.
            prob.add_option(
                "sqp_qp_use_homotopy", "yes" if uses_homotopy(arm) else "no"
            )
            prob.add_option("sqp_max_iter", self.max_iter)
        else:
            prob.add_option("tol", tol)
            prob.add_option("constr_viol_tol", 1e-6)
            prob.add_option("max_iter", self.max_iter)
        return prob

    # -- the convex-QP path ----------------------------------------

    def _solve_qp_ipm(
        self,
        family: ParametricFamily,
        callbacks: SparseCallbacks,
        arm: str,
        warm: Optional[WarmState],
        step: int,
        tol: float,
    ) -> Tuple[StepResult, Optional[WarmState]]:
        callbacks.reset_counts()
        seed = warm.extra.get("qp") if (warm is not None and warm.extra) else None

        t0 = time.perf_counter()
        qp = qpform.extract(family, callbacks)
        res = pounce.solve_qp(
            P=qp.P,
            c=qp.c,
            A=qp.A,
            b=qp.b,
            G=qp.G,
            h=qp.h,
            lb=qp.lb,
            ub=qp.ub,
            tol=tol,
            max_iter=self.max_iter,
            warm_start=seed if is_warm(arm) else None,
            check_psd=False,
        )
        elapsed = time.perf_counter() - t0

        resid = res.residuals or {}
        result = StepResult(
            step=step,
            theta=[],
            success=res.status in _OK_QP_STATUS,
            status=0 if res.status in _OK_QP_STATUS else -1,
            status_msg=str(res.status),
            iters=int(res.iters),
            solve_time=elapsed,
            # The QP form drops the objective's constant term; add it
            # back or every objective comparison against the other arms
            # is off by a per-step offset.
            obj=float(res.obj) + qp.f0,
            kkt_error=float(resid.get("kkt_error", np.nan)),
            constr_viol=float(resid.get("primal_infeasibility", np.nan)),
            # No working set: this is an interior-point method.
            n_active=None,
            n_qp_solves=None,
            n_qp_ws_changes=None,
            **callbacks.counts(),
        )
        next_warm = WarmState(
            x=np.asarray(res.x, dtype=float).copy(),
            extra={"qp": res},
        )
        return result, next_warm

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
        if arm in QP_ARMS:
            return self._solve_qp_ipm(family, callbacks, arm, warm, step, tol)

        prob = self._build(family, callbacks, arm, tol)
        callbacks.reset_counts()

        # Step 0 of a warm arm has nothing to warm from: it is a cold
        # solve, and is reported as one.
        use_warm = is_warm(arm) and warm is not None

        kwargs = {}
        if use_warm:
            if is_sqp(arm):
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
            n_qp_solves=int(info.get("n_qp_solves", 0)) if is_sqp(arm) else None,
            n_qp_ws_changes=(
                int(info.get("n_qp_ws_changes", 0)) if is_sqp(arm) else None
            ),
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
