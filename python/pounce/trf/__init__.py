"""Trust-region filter optimization for glass box / black box models.

``pounce.minimize`` needs a model it can differentiate. This subpackage handles
the case where part of the model is a black box -- a CFD solve, a converged unit
model, a trained network -- that cannot be expressed as equations, while the
rest is ordinary algebra with exact derivatives.

The obvious approach, fitting a surrogate to the black box and optimizing that,
does not work: the optimum of the surrogate is not the optimum of the truth
model, and iterating "fit, optimize, refit" can converge to a point that is a
local *maximum* of the real problem. The trust-region filter method fixes this
by confining the surrogate to a region where it is provably accurate and using a
filter on ``(infeasibility, objective)`` to decide which steps to keep. It
converges to a first-order KKT point of the truth model.

Quick start
-----------
Minimize ``(z - 1)^2 + x^2`` subject to ``z = sin(x)``, treating ``sin`` as a
black box::

    import numpy as np
    from pounce.trf import trf_minimize

    # x = [x, z];  w = x[[0]] is the black-box input, y = x[[1]] its output
    res = trf_minimize(
        fun=lambda v: (v[1] - 1.0) ** 2 + v[0] ** 2,
        x0=[0.5, 0.0],
        truth_model=lambda w: np.sin(w),
        w_index=[0],
        y_index=[1],
        jac=lambda v: np.array([2 * v[0], 2 * (v[1] - 1.0)]),
        truth_jac=lambda w: np.cos(w).reshape(1, 1),
    )
    print(res.x, res.fun, res.n_truth_evals)

Supplying ``truth_jac`` is the highest-value thing you can do for performance:
it removes ``n_w`` truth-model calls per iteration and makes the surrogate
exactly first-order accurate at the base point.

Scope
-----
The truth model must be **deterministic and smooth**. Eason & Biegler assume
noise is negligible and list rigorous noise handling as an open problem, so this
is not the right tool for optimizing against physical measurements -- use a
noise-aware method such as Bayesian optimization there. The method converges to
a *local* KKT point; it is not a global optimizer.

See ``docs/src/trf.md`` for the algorithm, tuning, and worked examples.
"""

from ._filter import FilterPoint, TRFilter
from ._loop import TRFConfig, TRFIterate, TRFResult, trf_minimize
from ._surrogate import (
    Basis,
    CorrectedSurrogate,
    QuadraticBasis,
    ZeroBasis,
    central_difference_design,
    forward_difference_design,
    quadratic_design,
)

__all__ = [
    "trf_minimize",
    "TRFResult",
    "TRFConfig",
    "TRFIterate",
    "TRFilter",
    "FilterPoint",
    "Basis",
    "ZeroBasis",
    "QuadraticBasis",
    "CorrectedSurrogate",
    "forward_difference_design",
    "central_difference_design",
    "quadratic_design",
]
