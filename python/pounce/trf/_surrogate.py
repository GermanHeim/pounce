"""Surrogate models for the trust-region filter method.

The trust-region subproblem replaces the truth model ``y = d(w)`` with a cheap
algebraic stand-in ``r_k(w)``. For the method to converge to a KKT point of the
*truth* model, ``r_k`` cannot merely fit well -- it must reproduce the truth
model's value and slope at the current base point. Matching values alone gives
feasibility but not optimality: a loop that only re-matches values can converge
to a point that is a local *maximum* of the true problem (Biegler 2024, Fig. 2a).

That requirement is met by the zero- and first-order corrections (ZOC/FOC) of
Yoshio & Biegler (2020)::

    r_k(w) = rbar(w)
             + ( d(w_k)   - rbar(w_k)   )          # ZOC -- match the value
             + ( J_d(w_k) - J_rbar(w_k) ) (w - w_k) # FOC -- match the slope

so that ``r_k(w_k) = d(w_k)`` and ``J_{r_k}(w_k) = J_d(w_k)`` hold by
construction, for *any* differentiable basis ``rbar``. This is what makes the
model kappa-fully linear automatically and lets the compatibility check and
criticality phase drop out of the algorithm entirely.

The basis ``rbar`` is therefore a free choice that affects efficiency, not
correctness. With ``rbar = 0`` (the default, and what ``pyomo.contrib.trustregion``
uses when no basis rule is supplied) the formula degenerates to a plain
linearization of the truth model at ``w_k``. A better ``rbar`` -- a low-fidelity
physical model, or a fitted local polynomial -- captures curvature the
linearization misses and can cut the iteration count substantially.

On choosing a basis
-------------------
Fit quality is a poor predictor of optimization performance. Pedrozo et al.
(2025) benchmarked five surrogate families on a CO2 pooling problem: radial
basis functions had the best R-squared of any model and needed 8 TRF iterations;
Kriging needed 2; plain global polynomials were worst on both counts. Prefer the
simplest basis that captures the local curvature, and note that Eason & Biegler
(2016) found simple linear interpolation beat Kriging on Williams-Otto by
91 truth-model calls to 3141.

References
----------
Eason & Biegler. AIChE J. 62, 3124 (2016); 64, 3934 (2018).
Yoshio, N. & Biegler, L.T. AIChE J. 67, e17054 (2021).
Biegler, L.T. Digital Chemical Engineering 13, 100197 (2024).  [Sec. 3.1]
Pedrozo et al. Comput. Chem. Eng. 200, 109199 (2025).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

__all__ = [
    "Basis",
    "ZeroBasis",
    "QuadraticBasis",
    "CorrectedSurrogate",
    "forward_difference_design",
    "central_difference_design",
    "quadratic_design",
]


# --------------------------------------------------------------------------
# Sampling designs
# --------------------------------------------------------------------------


def forward_difference_design(w0: np.ndarray, sigma: float) -> np.ndarray:
    """``n+1`` points: the base point plus one step along each coordinate.

    The minimal well-poised set for a linear model. Returns shape ``(n+1, n)``
    with ``w0`` first.
    """
    w0 = np.asarray(w0, dtype=float).ravel()
    n = w0.size
    return np.vstack([w0, w0 + sigma * np.eye(n)])


def central_difference_design(w0: np.ndarray, sigma: float) -> np.ndarray:
    """``2n+1`` points: the base point plus a symmetric pair per coordinate.

    Twice the cost of :func:`forward_difference_design` but second-order
    accurate in the gradient, which matters when the truth model is noisy or
    the step is large.
    """
    w0 = np.asarray(w0, dtype=float).ravel()
    n = w0.size
    eye = np.eye(n)
    return np.vstack([w0, w0 + sigma * eye, w0 - sigma * eye])


def quadratic_design(w0: np.ndarray, sigma: float) -> np.ndarray:
    """``(n+1)(n+2)/2`` points: the minimum needed to determine a full quadratic.

    A central design alone gives only ``2n+1`` points, which is enough for the
    constant, linear and pure-square terms but leaves every cross term
    ``dw_i*dw_j`` unidentified. Least squares will still return *an* answer for
    an underdetermined fit -- the minimum-norm one -- so the failure is silent:
    the surrogate simply has no mixed curvature and the trust-region steps are
    quietly worse. (At ``n = 1`` there are no cross terms and the two designs
    coincide, so a one-dimensional test will not catch it.)

    This adds one diagonal point per pair on top of the central design, giving
    exactly the ``(n+1)(n+2)/2`` needed.
    """
    w0 = np.asarray(w0, dtype=float).ravel()
    n = w0.size
    pts = [central_difference_design(w0, sigma)]
    if n > 1:
        cross = []
        for i in range(n):
            for j in range(i + 1, n):
                p = w0.copy()
                p[i] += sigma
                p[j] += sigma
                cross.append(p)
        pts.append(np.array(cross))
    return np.vstack(pts)


# --------------------------------------------------------------------------
# Basis protocol and built-ins
# --------------------------------------------------------------------------


@runtime_checkable
class Basis(Protocol):
    """A differentiable approximation of the truth model.

    Only ``predict`` and ``jacobian`` are used by the algorithm; ``fit`` is
    called once per iteration when the basis is data-driven. A basis that is a
    fixed analytic model (a low-fidelity physical model, say) can make ``fit``
    a no-op.
    """

    def fit(self, W: np.ndarray, Y: np.ndarray) -> "Basis":
        """Fit to samples ``W`` of shape ``(n_samples, n_w)`` and ``Y`` of
        shape ``(n_samples, n_y)``. Returns self."""
        ...

    def predict(self, w: np.ndarray) -> np.ndarray:
        """Evaluate at a single point ``w``; returns shape ``(n_y,)``."""
        ...

    def jacobian(self, w: np.ndarray) -> np.ndarray:
        """Jacobian ``d(predict)/dw`` at ``w``; returns shape ``(n_y, n_w)``."""
        ...


class ZeroBasis:
    """``rbar(w) = 0``.

    With this basis the corrected surrogate is a plain linearization of the
    truth model at the base point. This is the default and matches
    ``pyomo.contrib.trustregion``'s behaviour when no basis rule is given.
    Requires no samples beyond the base point itself.
    """

    def __init__(self, n_y: int, n_w: int) -> None:
        self.n_y = n_y
        self.n_w = n_w

    def fit(self, W: np.ndarray, Y: np.ndarray) -> "ZeroBasis":
        return self

    def predict(self, w: np.ndarray) -> np.ndarray:
        return np.zeros(self.n_y)

    def jacobian(self, w: np.ndarray) -> np.ndarray:
        return np.zeros((self.n_y, self.n_w))


# There is deliberately no ``LinearBasis`` here. Under ZOC/FOC an affine basis
# cancels *identically*, so fitting one is pure waste -- it costs n_w + 1
# truth-model evaluations per iteration to produce a surrogate bit-identical to
# the free ``ZeroBasis``.
#
# Write the correction with ``rbar(w) = a + B (w - w_ref)``. Its Jacobian is B
# everywhere, so ``J_rbar(w_k) = B`` and the basis-dependent part of
#
#     rbar(w) + (d(w_k) - rbar(w_k)) + (J_d(w_k) - J_rbar(w_k)) (w - w_k)
#
# is  ``rbar(w) - rbar(w_k) - B (w - w_k) = B(w - w_k) - B(w - w_k) = 0``,
# leaving ``r_k(w) = d(w_k) + J_d(w_k)(w - w_k)`` -- exactly the ZeroBasis
# result. Measured: value differences of 0 to 4e-16 and Jacobian differences of
# exactly 0.
#
# The general statement is that **only curvature survives the correction**.
# That is why ``QuadraticBasis`` earns its samples and an affine one never can,
# and it is the first thing to check before adding any new basis: if it is
# affine in ``w``, it cannot do anything. ``test_affine_basis_cancels_under_zoc_foc``
# locks this in.


class QuadraticBasis:
    """Least-squares quadratic model with full cross terms.

    Needs ``(n_w + 1)(n_w + 2) / 2`` samples to be determined, which grows
    quadratically -- at ``n_w = 10`` that is 66 truth-model evaluations per
    iteration. Worth it only when truth calls are cheap relative to the
    subproblem solve. Eason & Biegler (2018) found quadratic models need fewer
    *iterations* but more *evaluations* than linear ones.
    """

    def __init__(self) -> None:
        self.w_ref: np.ndarray | None = None
        self.coef: np.ndarray | None = None
        self.n_w: int = 0

    @staticmethod
    def _features(dW: np.ndarray) -> np.ndarray:
        """``[1, dw_i, dw_i*dw_j (i<=j)]`` for each row of ``dW``."""
        n = dW.shape[1]
        cols = [np.ones((dW.shape[0], 1)), dW]
        for i in range(n):
            for j in range(i, n):
                cols.append((dW[:, i] * dW[:, j]).reshape(-1, 1))
        return np.hstack(cols)

    def fit(self, W: np.ndarray, Y: np.ndarray) -> "QuadraticBasis":
        W = np.atleast_2d(np.asarray(W, dtype=float))
        Y = np.atleast_2d(np.asarray(Y, dtype=float))
        if Y.shape[0] != W.shape[0]:
            Y = Y.reshape(W.shape[0], -1)

        self.w_ref = W[0].copy()
        self.n_w = W.shape[1]
        A = self._features(W - self.w_ref)
        self.coef, *_ = np.linalg.lstsq(A, Y, rcond=None)
        return self

    def predict(self, w: np.ndarray) -> np.ndarray:
        if self.coef is None:
            raise RuntimeError("QuadraticBasis.predict called before fit")
        dw = (np.asarray(w, dtype=float) - self.w_ref).reshape(1, -1)
        return (self._features(dw) @ self.coef).ravel()

    def jacobian(self, w: np.ndarray) -> np.ndarray:
        if self.coef is None:
            raise RuntimeError("QuadraticBasis.jacobian called before fit")
        dw = np.asarray(w, dtype=float) - self.w_ref
        n = self.n_w
        n_y = self.coef.shape[1]
        J = np.zeros((n_y, n))
        # linear block occupies rows 1..n of coef
        J += self.coef[1 : 1 + n].T
        # quadratic block: d/dw_k of (dw_i * dw_j)
        row = 1 + n
        for i in range(n):
            for j in range(i, n):
                c = self.coef[row]  # shape (n_y,)
                if i == j:
                    J[:, i] += 2.0 * c * dw[i]
                else:
                    J[:, i] += c * dw[j]
                    J[:, j] += c * dw[i]
                row += 1
        return J


# --------------------------------------------------------------------------
# ZOC/FOC correction
# --------------------------------------------------------------------------


class CorrectedSurrogate:
    """A basis carrying zero- and first-order corrections at a base point.

    Wraps any :class:`Basis` so that, at the base point ``w_k``, the surrogate
    reproduces the truth model exactly in both value and slope::

        r_k(w_k)   == d(w_k)
        J_{r_k}(w_k) == J_d(w_k)

    These two identities are what the convergence theory needs; see the module
    docstring. They hold regardless of how good ``rbar`` is, which is why a
    mediocre basis is not a correctness problem -- Pedrozo et al. (2024) ran a
    CFD case to convergence with a basis whose *validation* R-squared was 0.80.
    """

    def __init__(
        self,
        basis: Basis,
        w_k: np.ndarray,
        d_k: np.ndarray,
        jac_d_k: np.ndarray,
    ) -> None:
        self.basis = basis
        self.w_k = np.asarray(w_k, dtype=float).ravel()
        self.d_k = np.atleast_1d(np.asarray(d_k, dtype=float))
        self.jac_d_k = np.atleast_2d(np.asarray(jac_d_k, dtype=float))

        self._zoc = self.d_k - np.atleast_1d(basis.predict(self.w_k))
        self._foc = self.jac_d_k - np.atleast_2d(basis.jacobian(self.w_k))

    def predict(self, w: np.ndarray) -> np.ndarray:
        w = np.asarray(w, dtype=float).ravel()
        base = np.atleast_1d(self.basis.predict(w))
        return base + self._zoc + self._foc @ (w - self.w_k)

    def jacobian(self, w: np.ndarray) -> np.ndarray:
        w = np.asarray(w, dtype=float).ravel()
        return np.atleast_2d(self.basis.jacobian(w)) + self._foc

    __call__ = predict


def finite_difference_jacobian(
    fun,
    w: np.ndarray,
    sigma: float,
    f0: np.ndarray | None = None,
    central: bool = False,
) -> tuple[np.ndarray, int]:
    """Finite-difference Jacobian of ``fun`` at ``w`` with step ``sigma``.

    The perturbation is deliberately tied to the *sampling* radius rather than
    to machine epsilon. Eason & Biegler (2016) had to inflate the step to
    ``min(0.1, 0.8*Delta)`` for a boiler model "to compensate for greater
    numerical noise in the model outputs", versus ``1e-5`` for smooth steam
    tables -- the right step size is a property of the truth model, not of the
    floating-point format.

    Returns ``(jacobian, n_truth_calls)``.
    """
    w = np.asarray(w, dtype=float).ravel()
    n = w.size
    if f0 is None:
        f0 = np.atleast_1d(np.asarray(fun(w), dtype=float))
        calls = 1
    else:
        f0 = np.atleast_1d(np.asarray(f0, dtype=float))
        calls = 0

    J = np.zeros((f0.size, n))
    for i in range(n):
        step = np.zeros(n)
        step[i] = sigma
        if central:
            fp = np.atleast_1d(np.asarray(fun(w + step), dtype=float))
            fm = np.atleast_1d(np.asarray(fun(w - step), dtype=float))
            J[:, i] = (fp - fm) / (2.0 * sigma)
            calls += 2
        else:
            fp = np.atleast_1d(np.asarray(fun(w + step), dtype=float))
            J[:, i] = (fp - f0) / sigma
            calls += 1
    return J, calls
