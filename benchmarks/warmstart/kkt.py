"""A KKT residual the harness computes itself, identically for every solver.

The issue makes "equal stopping criteria and clearly documented
solver-specific settings" an acceptance criterion, and the hard part of
that is not the option names — it is that two solvers reporting
"final KKT error 3e-9" are not necessarily reporting the same number.
Ipopt's `inf_du` is a scaled dual infeasibility; pounce reports both a
scaled and an unscaled error; a third solver may report a merit
function. Comparing arms across solvers on a quantity each one defines
for itself is not a comparison.

So the suite computes its own, from the returned primal point and
multipliers, using the family's callbacks:

    dual      ‖∇f(x) + Jᵀλ − z_L + z_U‖_∞
    primal    max over rows of (cl − g, g − cu, 0)  and bounds likewise
    compl     max(|z_L ⊙ (x − lb)|, |z_U ⊙ (ub − x)|, |λ ⊙ slack|)

    kkt = max(dual, primal, compl)

That is the Ipopt sign convention — ``∇f + Jᵀλ − z_L + z_U = 0``, with
``λ`` free for equality rows and signed for inequalities — and pounce's
NLP core is a port of Ipopt, so both adapters hand back multipliers in
it. The convention is *checked* rather than assumed: `selftest` compares
this function's output against the solver's own reported unscaled KKT
error on converged pounce solves, and a mismatch is a failure. If that
check passes for pounce, the same code applied to Ipopt's output is
measuring the same thing.

Complementarity is the term that gets dropped when a solver's reported
residual is really just ``max(inf_pr, inf_du)``, and it is the one that
catches a warm start that converged onto the wrong face of a degenerate
active set — so it is here on purpose, not for completeness.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

_INF = 1e19


def kkt_residual(
    family,
    callbacks,
    x: np.ndarray,
    mult_g: Optional[np.ndarray],
    mult_x_L: Optional[np.ndarray],
    mult_x_U: Optional[np.ndarray],
    count: bool = False,
) -> dict:
    """``{dual, primal, compl, kkt}`` at ``x`` with the given multipliers.

    ``count=False`` (the default) evaluates the family directly rather
    than through the counting wrapper: this is a *measurement* the
    harness makes after the solve, and charging it to the arm's
    evaluation budget would inflate every arm by a constant and make the
    counts mean something other than "what the solve cost".

    Missing multiplier blocks are treated as zero, which is what a
    solver that does not return them is implicitly claiming.
    """
    x = np.asarray(x, dtype=float).ravel()
    n, m = family.n, family.m
    if x.size != n or not np.all(np.isfinite(x)):
        return {"dual": float("inf"), "primal": float("inf"),
                "compl": float("inf"), "kkt": float("inf")}

    src = callbacks if count else family
    b = family.bounds()
    lb, ub = np.asarray(b.lb, float), np.asarray(b.ub, float)

    grad = np.asarray(src.gradient(x), dtype=float).ravel()
    lam = (np.zeros(m) if mult_g is None
           else np.asarray(mult_g, dtype=float).ravel())
    zl = (np.zeros(n) if mult_x_L is None
          else np.asarray(mult_x_L, dtype=float).ravel())
    zu = (np.zeros(n) if mult_x_U is None
          else np.asarray(mult_x_U, dtype=float).ravel())

    # -- dual feasibility ------------------------------------------
    r = grad - zl + zu
    if m:
        g = np.asarray(src.constraints(x), dtype=float).ravel()
        jac = np.asarray(family.jacobian_dense(x), dtype=float)
        r = r + jac.T @ lam
    else:
        g = np.zeros(0)
    dual = float(np.max(np.abs(r))) if r.size else 0.0

    # -- primal feasibility ----------------------------------------
    cl, cu = np.asarray(b.cl, float), np.asarray(b.cu, float)
    viol = 0.0
    if m:
        viol = float(np.max(np.maximum.reduce([cl - g, g - cu, np.zeros(m)])))
    bnd = float(np.max(np.maximum.reduce([lb - x, x - ub, np.zeros(n)])))
    primal = max(viol, bnd)

    # -- complementarity -------------------------------------------
    # Only finite bounds/rows contribute; an infinite bound has no
    # multiplier to complement against.
    terms = [0.0]
    fl, fu = lb > -_INF, ub < _INF
    if np.any(fl):
        terms.append(float(np.max(np.abs(zl[fl] * (x[fl] - lb[fl])))))
    if np.any(fu):
        terms.append(float(np.max(np.abs(zu[fu] * (ub[fu] - x[fu])))))
    if m:
        # Equality rows are complementary by definition (the slack is
        # zero and the multiplier is free), so only two-sided or
        # one-sided *inequality* rows are scored.
        ineq = cl != cu
        if np.any(ineq):
            slack = np.minimum(
                np.where(cl > -_INF, g - cl, np.inf),
                np.where(cu < _INF, cu - g, np.inf),
            )
            slack = np.where(np.isfinite(slack), slack, 0.0)
            terms.append(float(np.max(np.abs(lam[ineq] * slack[ineq]))))
    compl = max(terms)

    kkt = max(dual, primal, compl)
    return {"dual": dual, "primal": primal, "compl": compl, "kkt": kkt}
