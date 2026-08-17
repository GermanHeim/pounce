"""Ipopt through cyipopt — the external-solver arm (pounce#611).

The suite's adapter interface was written so a second solver could be
dropped behind it without touching a family, and until now nothing had
been. The `dev-notes` list it as an open item ("The adapter interface
is exercised by the pounce adapter's three paths but has no second
solver behind it; Ipopt through cyipopt would be the obvious first
one, using the same families unchanged"), and pounce#611 makes an
external baseline an acceptance criterion.

This is that adapter, and it uses the families unchanged: cyipopt's
`Problem` takes exactly the callback object
(:class:`~..sparsity.SparseCallbacks`) the pounce adapter passes to
`pounce.Problem`, because pounce's Python surface is deliberately
cyipopt-shaped. So the two solvers are driven by the *same* Python
callables, evaluated the same number of times per iteration, and the
harness's evaluation counters mean the same thing on both sides. That
is the strongest form of the comparison available short of a common
C-level interface.

What is and is not equal
------------------------

**Equal:** the problem (same callbacks, same bounds), the primal
convergence tolerance (`tol`), the constraint-violation tolerance
(`constr_viol_tol`), the iteration cap (`max_iter`), and the
harness-side correctness gates — which are computed from the returned
point by :mod:`..kkt`, identically for both solvers, rather than read
off either solver's own status line.

**Not equal, and load-bearing:**

* **Linear solver.** This Ipopt is Debian's `coinor-libipopt` 3.11.9
  linked against MUMPS. pounce uses its own sparse LDLᵀ. HSL (MA27/MA57)
  is not redistributable and is not installed here, so the Ipopt arm is
  running on the linear solver most users without an HSL licence would
  also get — but a wall-time comparison against pounce is a comparison
  against *MUMPS-backed Ipopt*, and must be labelled that way. This is
  the single biggest caveat on any timing number in this file's arms.
* **Version.** 3.11.9 is a 2014 release, which is what the distribution
  ships. Iteration counts on modern Ipopt would differ.
* **Ipopt has no active-set or QP path**, so the `*-sqp`, `*-hom` and
  `*-qp-ipm` arms are unsupported here and are skipped with a reason
  rather than silently dropped. It also has no parametric sensitivity
  in this build (sIPOPT is a separate library, not built), so the
  predictor arms are skipped too.
* **Racing** is a pounce API, not a solver capability, so `race-*` is
  not offered here either. There is nothing stopping a future version
  from racing Ipopt starts; it is simply not what this arm measures.

Warm starting Ipopt
-------------------

Ipopt's documented warm start is `warm_start_init_point=yes` plus the
previous solve's `mult_g` / `mult_x_L` / `mult_x_U`, with the three
`warm_start_*_push` parameters reduced so the supplied point is not
shoved back into the interior before the first iteration. Those pushes
default to `1e-2`, which is large enough to discard most of what a warm
start supplies; leaving them at the default and reporting "Ipopt's warm
start barely helps" would be an artifact of the settings, not a result.
They are set to `1e-9` here and the value is recorded in the run's
metadata.

`mu_init` is carried from the previous solve as well, which is the
barrier half of the primal-dual-barrier arm the issue asks for.
"""

from __future__ import annotations

import time
from typing import Optional, Tuple

import numpy as np

from ..kkt import kkt_residual
from ..spec import ParametricFamily, StepResult, WarmState
from ..sparsity import SparseCallbacks
from .base import SolverAdapter, is_primal_only, is_warm

#: Ipopt ApplicationReturnStatus values counted as a solved step:
#: Solve_Succeeded and Solved_To_Acceptable_Level. Same two the pounce
#: adapter accepts.
_OK_STATUS = (0, 1)

#: How far Ipopt is allowed to push a supplied warm-start point back
#: into the interior. The default (1e-2) discards most of a warm start.
_WARM_PUSH = 1e-9

#: Arms this adapter can run at all.
_SUPPORTED = ("cold-ipm", "warm-ipm", "warm-ipm-primal")


class _Instrumented:
    """`SparseCallbacks` plus Ipopt's `intermediate` hook.

    cyipopt discovers callbacks by attribute, so the forwarding is
    written out rather than done with ``__getattr__`` — an interface
    where a typo silently becomes "this solver uses finite differences"
    is not one to be clever in.

    The hook exists because cyipopt's returned ``info`` has no iteration
    count, and iteration count is the suite's primary cross-arm metric.
    """

    def __init__(self, cb: SparseCallbacks):
        self._cb = cb
        self.iters = 0
        self.last_inf_pr = float("nan")
        self.last_inf_du = float("nan")
        self.last_mu = None

    def objective(self, x):
        return self._cb.objective(x)

    def gradient(self, x):
        return self._cb.gradient(x)

    def constraints(self, x):
        return self._cb.constraints(x)

    def jacobianstructure(self):
        return self._cb.jacobianstructure()

    def jacobian(self, x):
        return self._cb.jacobian(x)

    def hessianstructure(self):
        return self._cb.hessianstructure()

    def hessian(self, x, lagrange, obj_factor):
        return self._cb.hessian(x, lagrange, obj_factor)

    def intermediate(self, alg_mod, iter_count, obj_value, inf_pr, inf_du,
                     mu, d_norm, regularization_size, alpha_du, alpha_pr,
                     ls_trials):
        self.iters = int(iter_count)
        self.last_inf_pr = float(inf_pr)
        self.last_inf_du = float(inf_du)
        self.last_mu = float(mu)
        return True


class IpoptAdapter(SolverAdapter):
    name = "ipopt"

    def __init__(self, max_iter: int = 500, recentering: str = "residual"):
        self.max_iter = max_iter
        # Accepted and ignored: `warm_start_recentering` is a pounce
        # option with no Ipopt counterpart. Taking the argument keeps
        # `get_adapter` uniform; the run's metadata records that the
        # setting did not apply to this adapter.
        self.recentering = recentering

    def supports(self, arm: str) -> bool:
        return arm in _SUPPORTED

    @staticmethod
    def unsupported_reason(arm: str) -> str:
        if arm.endswith("-sqp") or arm.endswith("-hom"):
            return "Ipopt has no active-set SQP path"
        if arm.endswith("qp-ipm"):
            return "Ipopt has no dedicated matrix-form QP entry point"
        if arm.startswith("pred"):
            return "sIPOPT is not built in this Ipopt (3.11.9, Debian)"
        if arm.startswith("race"):
            return "start racing is a pounce API, not a solver capability"
        if arm == "cold-ipm-lsq":
            return "least_square_init_primal is a pounce option"
        if arm == "warm-ipm-norecenter":
            return "warm_start_recentering is a pounce option"
        return "not implemented by the Ipopt adapter"

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
        import cyipopt

        b = family.bounds()
        callbacks.reset_counts()
        inst = _Instrumented(callbacks)

        nlp = cyipopt.Problem(
            n=family.n, m=family.m, problem_obj=inst,
            lb=b.lb, ub=b.ub, cl=b.cl, cu=b.cu,
        )
        # Every setting that differs from Ipopt's default is here, and
        # each one is repeated in the report's settings table.
        for key, val in (
            ("print_level", 0),
            ("sb", "yes"),
            ("tol", tol),
            ("constr_viol_tol", 1e-6),
            ("max_iter", self.max_iter),
        ):
            nlp.add_option(key, val)

        use_warm = is_warm(arm) and warm is not None
        seeds = {}
        if use_warm and not is_primal_only(arm):
            # The complete primal-dual-barrier warm start.
            nlp.add_option("warm_start_init_point", "yes")
            nlp.add_option("warm_start_bound_push", _WARM_PUSH)
            nlp.add_option("warm_start_slack_bound_push", _WARM_PUSH)
            nlp.add_option("warm_start_mult_bound_push", _WARM_PUSH)
            if warm.mu is not None and np.isfinite(warm.mu) and warm.mu > 0:
                nlp.add_option("mu_init", float(warm.mu))
            if warm.mult_g is not None:
                seeds["lagrange"] = np.asarray(warm.mult_g, float)
            if warm.mult_x_L is not None:
                seeds["zl"] = np.asarray(warm.mult_x_L, float)
            if warm.mult_x_U is not None:
                seeds["zu"] = np.asarray(warm.mult_x_U, float)
        start = np.asarray(warm.x if use_warm else x0, dtype=float)

        t0 = time.perf_counter()
        x, info = nlp.solve(start, **seeds)
        elapsed = time.perf_counter() - t0

        status = int(info["status"])
        lam = info.get("mult_g")
        zl, zu = info.get("mult_x_L"), info.get("mult_x_U")
        # Ipopt's own residual is not comparable to pounce's, so the
        # harness recomputes both from the returned point.
        res = kkt_residual(family, callbacks, x, lam, zl, zu)

        msg = info.get("status_msg", b"")
        if isinstance(msg, bytes):
            msg = msg.decode("utf-8", "replace")

        result = StepResult(
            step=step,
            theta=[],
            success=status in _OK_STATUS,
            status=status,
            status_msg=msg.strip(),
            iters=int(inst.iters),
            solve_time=elapsed,
            init_time=0.0,
            obj=float(info["obj_val"]),
            kkt_error=float(res["kkt"]),
            constr_viol=float(res["primal"]),
            # Ipopt is interior-point only: no working set, no QP
            # subproblems. `None` rather than 0, which would read as
            # "measured, and it was zero".
            n_active=None,
            n_qp_solves=None,
            n_qp_ws_changes=None,
            **callbacks.counts(),
        )

        next_warm = WarmState(
            x=np.asarray(x, dtype=float).copy(),
            mult_g=None if lam is None else np.asarray(lam, float).copy(),
            mult_x_L=None if zl is None else np.asarray(zl, float).copy(),
            mult_x_U=None if zu is None else np.asarray(zu, float).copy(),
            # cyipopt's `info` carries no barrier parameter, so this is
            # `mu` as of the last `intermediate` callback rather than
            # the value Ipopt finished on. The two differ by at most one
            # barrier update, which is small next to what `mu_init`
            # controls -- but it is an approximation, and it is the one
            # place this arm is not reading the solver's true final state.
            mu=inst.last_mu,
            working_set=None,
        )
        return result, next_warm
