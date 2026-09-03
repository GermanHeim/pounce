"""The flash MPCC -> a smooth NLP, with exact JAX derivatives.

The three lowerings are Gate 0's, unchanged, because comparing Gate 1
against Gate 0's supported route only means something if the route is
the same object:

``prod_ineq``  ``H >= 0``, ``G*H <= 0``. The direct formulation; its
               feasible points are exactly the MPCC's, since ``G, H >= 0``
               makes ``G*H <= 0`` equivalent to ``G*H = 0``.
``prod_eq``    the same with ``G*H = 0``: the exact-product / NCP
               equality form.
``scholtes``   ``G*H <= tau``, feasible for the MPCC only as
               ``tau -> 0``.

Two things differ from `mpcc/lowering.py`, and both are properties of
the model rather than choices:

* **The ``G`` sides are bounds, not rows.** ``G_V = beta`` and
  ``G_L = 1 - beta`` are already enforced exactly by ``beta in [0, 1]``,
  and adding them again as constraint rows would put two linearly
  dependent rows into every active set for no gain. An MPCC's active set
  is degenerate enough by construction (that is the theorem, not a
  defect); manufacturing more of it would make a solver failure harder
  to attribute, which is the one thing this fixture exists to avoid.
* **The rows are nonlinear.** Gate 0's corpus is quadratic with affine
  pairs so that every derivative is exact and closed-form. Here the
  isofugacity rows carry a cubic root and a logarithm, so the
  derivatives come from JAX -- exact to round-off, not finite
  differences -- and `selftest` checks them against a central difference
  anyway, because "the derivative is the first suspect" survives the
  change of technique.

Row order is fixed and is part of the contract; the runner reads
multipliers back out of ``info["mult_g"]`` positionally::

    [ balance_1..nc ] [ isofug_1..nc ] [ H_V ] [ H_L ] [ prod_V ] [ prod_L ]
"""

from __future__ import annotations

import dataclasses
from typing import Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from pounce.examples.phase_envelope import log_fugacity_coefficients

from .spec import FlashCase

jax.config.update("jax_enable_x64", True)

LOWERINGS = ("prod_ineq", "prod_eq", "scholtes")


def _residuals(v, temperature_k, case: FlashCase):
    """The MPCC's own rows at ``v``, in the order documented above.

    Written once, in JAX, and used for both the value and every
    derivative -- so a row and its gradient cannot drift apart, which is
    the single most common way a benchmark ends up measuring its own
    harness.
    """
    nc = case.nc
    beta = v[0]
    x = v[1 : 1 + nc]
    y = v[1 + nc : 1 + 2 * nc]
    sx = jnp.sum(x)
    sy = jnp.sum(y)
    xn = x / sx
    yn = y / sy
    z = jnp.asarray(case.z)

    balance = (1.0 - beta) * x + beta * y - z
    ln_phi_l = log_fugacity_coefficients(
        xn, temperature_k, case.pressure_pa, case.mixture, largest=False
    )
    ln_phi_v = log_fugacity_coefficients(
        yn, temperature_k, case.pressure_pa, case.mixture, largest=True
    )
    # Normalized *inside* phi, un-normalized in the log. See the
    # "Where the normalization goes" section of `spec.py`: carrying the
    # normalization into the log term too is a defect that is invisible
    # in the two-phase region and wrong in both single-phase ones.
    isofug = jnp.log(x) + ln_phi_l - jnp.log(y) - ln_phi_v

    h_v = 1.0 - sy
    h_l = 1.0 - sx
    prod_v = beta * h_v
    prod_l = (1.0 - beta) * h_l
    return jnp.concatenate([balance, isofug, jnp.array([h_v, h_l, prod_v, prod_l])])


#: Compiled callbacks, one set per case rather than per solve.
#:
#: **The Hessian is forward-over-forward on purpose.** `jax.hessian` is
#: `jacfwd(jacrev(.))`, and under `jax.jit` its reverse-mode half is
#: catastrophically inaccurate on this model near the cubic's
#: discriminant boundary -- where the equation of state has a double
#: root and Cardano's trigonometric branch runs `arccos` into its
#: endpoint singularity. Measured against a value-only second
#: difference, at the oracle's own point:
#:
#: =========================  ===============  ==============
#: composition                268 K and 270 K  everywhere else
#: =========================  ===============  ==============
#: `jit(jax.hessian)`         **2.1e+01**      3e-14
#: `jit(jacfwd(grad))`        **2.1e+01**      3e-14
#: `jit(jacrev(jacfwd))`      **5.8e+00**      2e-14
#: `jit(jacfwd(jacfwd))`      2.3e-14          3e-14
#: =========================  ===============  ==============
#:
#: 268 K and 270 K are the two path points straddling the bubble point
#: at 268.89 K, and they are the *only* two where it happens -- which is
#: exactly what makes it dangerous. The Jacobian is unaffected in every
#: mode, so gradients are right, KKT residuals are right, and the
#: converged answers were right: the full traversal agreed with the
#: oracle to 1e-11 at all 34 temperatures *with the wrong Hessian*. It
#: costs iterations and robustness at the one place the fixture exists
#: to test, and nothing reports it.
#:
#: Unjitting the Hessian also fixes it and costs 240 ms per call against
#: 0.15 ms, which is not available to a fixture gh#776 asks to be fast.
#: Forward-over-forward is exact and free.
#:
#: ``temperature_k`` is a *traced* argument, not baked in, so the whole
#: temperature path shares one compilation. Closing over it instead --
#: which the first draft did, via `functools.partial` -- costs a JAX
#: compilation per stage: 34 temperatures times ten continuation stages
#: is 340 of them, and the traversal spent essentially all of its wall
#: clock compiling the same function again. Compilation time is not
#: measurement, and a fixture gh#776 asks to be *fast* cannot pay it per
#: stage.
_COMPILED: Dict[int, tuple] = {}


def _compiled(case: FlashCase) -> tuple:
    key = id(case)
    hit = _COMPILED.get(key)
    if hit is not None:
        return hit

    def fn(v, t):
        return _residuals(v, t, case)

    built = (
        jax.jit(fn),
        jax.jit(jax.jacfwd(fn, argnums=0)),
        # Forward-over-forward, and NOT `jax.hessian`. See the note
        # below: reverse mode loses this model's Hessian entirely in a
        # ~1 K band around the bubble point, and costs nothing to avoid.
        jax.jit(
            jax.jacfwd(
                jax.jacfwd(lambda v, lam, t: jnp.dot(lam, fn(v, t)), argnums=0),
                argnums=0,
            )
        ),
    )
    _COMPILED[key] = built
    return built


@dataclasses.dataclass
class LoweredFlash:
    """A cyipopt-style callback object plus the bookkeeping to read it back."""

    case: FlashCase
    temperature_k: float
    lowering: str
    tau: Optional[float]
    n: int
    m: int
    lb: np.ndarray
    ub: np.ndarray
    cl: np.ndarray
    cu: np.ndarray
    balance_row0: int
    isofug_row0: int
    h_row0: int
    prod_row0: int
    _c: object = None
    _j: object = None
    _h: object = None
    _t: object = None

    # -- cyipopt-style callbacks ----------------------------------
    #
    # The objective is identically zero: this is a *square flash*, a
    # feasibility problem, and gh#776's Gate 1 asks for a flash rather
    # than a design. Giving it an objective would change which point in
    # a degenerate solution set the solver returns, and the fixture's
    # whole claim is that the point is the one the oracle computes.

    def objective(self, x):
        return 0.0

    def gradient(self, x):
        return np.zeros(self.n)

    def constraints(self, x):
        return np.asarray(
            self._c(jnp.asarray(x, dtype=jnp.float64), self._t), dtype=float
        )

    def jacobian(self, x):
        return np.asarray(
            self._j(jnp.asarray(x, dtype=jnp.float64), self._t), dtype=float
        ).reshape(-1)

    def jacobianstructure(self):
        rows = np.repeat(np.arange(self.m), self.n)
        cols = np.tile(np.arange(self.n), self.m)
        return rows, cols

    def hessianstructure(self):
        return np.tril_indices(self.n)

    def hessian(self, x, lagrange, obj_factor):
        full = np.asarray(
            self._h(
                jnp.asarray(x, dtype=jnp.float64),
                jnp.asarray(lagrange, dtype=jnp.float64),
                self._t,
            ),
            dtype=float,
        )
        r, c = np.tril_indices(self.n)
        return full[r, c]

    # -- reading a solve back -------------------------------------

    def pair_multipliers(self, mult_g) -> Tuple[np.ndarray, np.ndarray]:
        """``(mult_H, mult_prod)`` sliced out of ``info['mult_g']``.

        The *NLP* multipliers of the lowered rows. They are reported and
        they are not the MPCC's multipliers; nothing in this harness
        presents them as such.
        """
        mg = np.asarray(mult_g, dtype=float)
        return (
            mg[self.h_row0 : self.h_row0 + 2],
            mg[self.prod_row0 : self.prod_row0 + 2],
        )

    @property
    def row_names(self) -> List[str]:
        names = [f"balance_{c}" for c in self.case.mixture.names]
        names += [f"isofugacity_{c}" for c in self.case.mixture.names]
        return names + ["H_vapor", "H_liquid", "prod_vapor", "prod_liquid"]


def lower(
    case: FlashCase, temperature_k: float, lowering: str, tau: Optional[float] = None
) -> LoweredFlash:
    """Build the smooth NLP for ``case`` at ``temperature_k``.

    ``tau`` is required by ``scholtes`` and rejected by the others: a
    relaxation parameter on a non-relaxed lowering would be silently
    ignored, and a record whose ``tau`` says ``1e-8`` when nothing read
    it is worse than no field at all.
    """
    if lowering not in LOWERINGS:
        raise ValueError(f"unknown lowering {lowering!r}")
    if lowering == "scholtes":
        if tau is None:
            raise ValueError("scholtes lowering needs tau")
    elif tau is not None:
        raise ValueError(f"{lowering} lowering takes no tau")

    nc = case.nc
    n = case.n
    m = 2 * nc + 4
    balance_row0, isofug_row0, h_row0, prod_row0 = 0, nc, 2 * nc, 2 * nc + 2

    cl = np.zeros(m)
    cu = np.zeros(m)
    cl[h_row0 : h_row0 + 2] = 0.0
    cu[h_row0 : h_row0 + 2] = np.inf
    if lowering == "prod_eq":
        cl[prod_row0 : prod_row0 + 2] = 0.0
        cu[prod_row0 : prod_row0 + 2] = 0.0
    elif lowering == "prod_ineq":
        cl[prod_row0 : prod_row0 + 2] = -np.inf
        cu[prod_row0 : prod_row0 + 2] = 0.0
    else:
        cl[prod_row0 : prod_row0 + 2] = -np.inf
        cu[prod_row0 : prod_row0 + 2] = float(tau)

    c, j, h = _compiled(case)

    return LoweredFlash(
        case=case,
        temperature_k=float(temperature_k),
        lowering=lowering,
        tau=tau,
        n=n,
        m=m,
        lb=case.lb.copy(),
        ub=case.ub.copy(),
        cl=cl,
        cu=cu,
        balance_row0=balance_row0,
        isofug_row0=isofug_row0,
        h_row0=h_row0,
        prod_row0=prod_row0,
        _c=c,
        _j=j,
        _h=h,
        _t=jnp.asarray(float(temperature_k), dtype=jnp.float64),
    )


def fd_check(nlp: LoweredFlash, v: np.ndarray, h: float = 1e-6) -> dict:
    """Central-difference check of the declared derivatives.

    Gate 0's corpus is quadratic, so its equivalent check is exact to
    round-off and any tolerance above that hides something. Here the
    rows carry a cubic root and a logarithm, so the second differences
    carry genuine truncation error and the thresholds `selftest` applies
    are set to that, not tighter -- an assertion tighter than the method
    it uses would fail for a reason that has nothing to do with the
    model.
    """
    v = np.asarray(v, dtype=float)
    n = nlp.n

    def num_jac():
        out = np.zeros((nlp.m, n))
        for jx in range(n):
            e = np.zeros(n)
            e[jx] = h
            out[:, jx] = (nlp.constraints(v + e) - nlp.constraints(v - e)) / (2 * h)
        return out

    jac = nlp.jacobian(v).reshape(nlp.m, n)
    jerr = float(np.max(np.abs(num_jac() - jac)))

    rng = np.random.default_rng(776)
    lam = rng.normal(size=nlp.m)

    def lag_grad(w):
        return lam @ nlp.jacobian(w).reshape(nlp.m, n)

    num_h = np.zeros((n, n))
    for jx in range(n):
        e = np.zeros(n)
        e[jx] = h
        num_h[:, jx] = (lag_grad(v + e) - lag_grad(v - e)) / (2 * h)
    tri = nlp.hessian(v, lam, 0.0)
    full = np.zeros((n, n))
    r, c = np.tril_indices(n)
    full[r, c] = tri
    full = full + np.tril(full, -1).T
    herr = float(np.max(np.abs(full - 0.5 * (num_h + num_h.T))))
    return {"jac": jerr, "hess": herr}
