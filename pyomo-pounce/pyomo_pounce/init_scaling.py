"""Scaling for the initialization stack's projection merit (gh #609).

Two things are scaled, for two different reasons.

**Rows.** :func:`~pyomo_pounce.project_to_feasible` hands the model's own
equalities to POUNCE as hard constraints, and the solver's default
``gradient-based`` scaling follows Ipopt's rule, ``s_i = min(1, g_max /
||grad c_i||_inf)``: it scales a large row *down* and leaves a small row
alone. A convergence test on the absolute residual then enforces a row in
units of 1e6 to a relative accuracy of 1e-22 and a row in units of 1e-6
to a relative accuracy of 1e-2 -- the "large-magnitude rows dominate
small but important constraints" of gh #609. :func:`row_factors` computes
a **two-sided** normalisation, ``s_i = 1 / ||grad c_i||_inf``, which
scales small rows up as well as large rows down, so every row is enforced
to the same relative accuracy. Measured on the module's motivating model
(a 1e6-unit energy balance beside a 1e-6-unit trace balance) that moved
the trace row's relative residual from 1.2e-8 to 0.

The factors are delivered through the model's ``scaling_factor`` Suffix
and ``nlp_scaling_method=user-scaling`` -- the same route Pyomo, AMPL and
Ipopt use, already read end-to-end by :mod:`pyomo_pounce.scaling` -- so
nothing new has to be taught to the solver, and a user entry simply wins
over the automatic one.

**Variables.** The merit was ``sum((v - v0)**2)``, an *absolute* distance.
Across mixed units that is not a distance anyone wants: a pressure at 1e6
and a mole fraction at 1e-4 are a decade apart in what "moving by 1e-5"
means, and the unscaled merit dumps the entire repair onto whichever
variable the constraint gradient happens to favour. Measured on the
module's motivating model the mole fraction absorbed 100% of the
correction (a 20% relative move) while the pressure moved 1.8e-12
relative -- a ratio of 1.1e11. :func:`variable_weights` weights each
anchor by its own magnitude, so the merit measures *relative* movement
and the repair is shared in proportion to what each variable can afford.

Both are invariant under the transformation that motivated them, which is
the property to hold on to: rescaling a row by ``k`` scales its gradient
by ``k`` and its factor by ``1/k``, so the solver sees the same scaled
row. (POUNCE's own gradient-based scaling already made the *projection*
row-rescale invariant to machine precision; what it did not do is enforce
the rows evenly, which is the defect above. Keeping the invariance is a
gh #609 acceptance criterion, so :func:`row_factors` is built not to
break it.)
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

__all__ = ["row_factors", "row_scales", "variable_weights"]

#: |v0| below this carries no usable scale -- a variable sitting at zero
#: has no magnitude to be relative to, so it keeps an absolute (unit)
#: weight and stays free to move.
SCALE_FLOOR = 1e-8

#: Widest spread we will impose between the smallest and largest anchor
#: weight. Without a cap a single anchor at 1e-8 beside one at 1e8 asks
#: the solver for a merit with a 1e32 spread; that is not scaling, it is
#: a fixed variable spelled badly.
SCALE_RANGE = 1e12

#: Row gradients below this are treated as absent rather than tiny: a
#: factor of 1/1e-30 is a way to turn a rounding error into the dominant
#: row of the problem.
GRAD_FLOOR = 1e-30


def _grad_inf_norm(con, variables) -> float:
    """``||grad c||_inf`` of a constraint body at the current point.

    Symbolic where Pyomo can do it, numeric where it cannot: a body with
    an external function or a non-differentiable node must degrade to
    "no usable gradient" (returning 0.0, i.e. an unscaled row), never
    take the pipeline down.
    """
    from pyomo.core.expr.calculus.derivatives import differentiate
    from pyomo.environ import value

    body = con.body
    try:
        grads = differentiate(body, wrt_list=list(variables))
    except Exception:  # noqa: BLE001 - fall back to differences
        return _grad_inf_norm_numeric(con, variables)
    out = 0.0
    for g in grads:
        try:
            gv = abs(float(value(g, exception=False) or 0.0))
        except Exception:  # noqa: BLE001
            return _grad_inf_norm_numeric(con, variables)
        if math.isfinite(gv):
            out = max(out, gv)
    return out


def _grad_inf_norm_numeric(con, variables) -> float:
    """Central-difference fallback for :func:`_grad_inf_norm`."""
    from pyomo.environ import value

    out = 0.0
    for v in variables:
        v0 = v.value
        if v0 is None:
            continue
        h = 1e-7 * max(1.0, abs(v0))
        try:
            v.set_value(v0 + h, skip_validation=True)
            hi = float(value(con.body, exception=False) or 0.0)
            v.set_value(v0 - h, skip_validation=True)
            lo = float(value(con.body, exception=False) or 0.0)
        except Exception:  # noqa: BLE001
            return 0.0
        finally:
            v.set_value(v0, skip_validation=True)
        g = abs(hi - lo) / (2.0 * h)
        if math.isfinite(g):
            out = max(out, g)
    return out


def row_factors(constraints, variables=None) -> Dict[object, float]:
    """Two-sided row normalisation: ``{ConstraintData: 1/||grad c||_inf}``.

    `variables` names the columns to differentiate against; omitted, each
    row is differentiated against the variables *it* mentions, which is
    the only version that stays affordable on a real model -- a whole-model
    ``wrt_list`` would differentiate every row against every column.

    Rows whose gradient is unusable (zero, non-finite, or unreadable) are
    omitted rather than given a huge factor -- an omitted row keeps the
    factor 1.0 the Suffix defaults to.
    """
    from pyomo.core.expr.visitor import identify_variables

    out = {}
    for con in constraints:
        if variables is None:
            cols = [
                v
                for v in identify_variables(con.body, include_fixed=False)
                if v.value is not None
            ]
            if not cols:
                continue
        else:
            cols = variables
        g = _grad_inf_norm(con, cols)
        if not math.isfinite(g) or g <= GRAD_FLOOR:
            continue
        out[con] = 1.0 / g
    return out


def variable_weights(anchored) -> Dict[int, float]:
    """Anchor weights for the projection merit: ``{id(VarData): w}``.

    `anchored` is the ``[(VarData, v0)]`` list the merit is built over.
    The merit is ``sum(w**2 * (v - v0)**2)``, so ``w = 1/|v0|`` makes each
    term a squared *relative* movement. Two guards keep that robust:
    :data:`SCALE_FLOOR` (an anchor at or near zero has no relative scale,
    so it keeps ``w = 1`` and stays free to move) and :data:`SCALE_RANGE`
    (the spread between the smallest and largest weight is capped, so one
    extreme anchor cannot quietly turn the merit into a fixed variable).
    """
    raw = {}
    for v, v0 in anchored:
        a = abs(float(v0))
        raw[id(v)] = a if (math.isfinite(a) and a >= SCALE_FLOOR) else 1.0
    if not raw:
        return {}
    hi = max(raw.values())
    lo = hi / SCALE_RANGE
    return {k: 1.0 / min(max(r, lo), hi) for k, r in raw.items()}


def row_scales(constraints, variables=None) -> Dict[object, float]:
    """``{ConstraintData: ||grad c||_inf}`` -- the raw row magnitudes.

    :func:`row_factors` is the reciprocal, for handing to the solver;
    this is the same quantity for callers that want to divide by it, such
    as the block conditioning check, which needs to keep an unusable row
    distinguishable from a merely small one.
    """
    from pyomo.core.expr.visitor import identify_variables

    out = {}
    for con in constraints:
        cols = (
            list(identify_variables(con.body, include_fixed=True))
            if variables is None
            else list(variables)
        )
        if not cols:
            continue
        g = _grad_inf_norm(con, cols)
        if math.isfinite(g):
            out[con] = g
    return out
