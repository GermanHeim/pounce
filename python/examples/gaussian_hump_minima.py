"""Find multiple local minima by adding Gaussian humps (basin flooding).

This is the molecular-dynamics-flavored idea of pushing a system out of a
minimum it has already visited by *filling that basin*: once you converge
to a minimum ``x*``, you add a repulsive Gaussian bump centered there to
the objective and re-solve. The bump raises the floor of the basin you just
found, so the next solve rolls out and settles in a *different* minimum.
Accumulate one bump per discovered minimum and you sweep out the landscape.

The same trick appears under several names across communities:

* **Metadynamics** (Laio & Parrinello, PNAS 2002) — MD adds Gaussians along
  the trajectory in collective-variable space to discourage revisiting.
* **Conformational flooding** (Grubmüller, PRE 1995) — same flooding idea.
* **Filled-function method** (Ge Renpu, Math. Prog. 1990) — global
  optimization: add a bump at a found minimum so it is no longer a minimum
  of the modified objective.
* **Tunneling method** (Levy & Montalvo, SIAM J. Sci. Stat. Comput. 1985).
* **Deflation** (Farrell, Birkisson & Funke, SISC 2015) — distinct solutions
  of nonlinear systems; multiplicative rather than additive, but same spirit.

Here the modified ("augmented") objective is

    F(x) = f(x) + sum_k  A_k * exp( -||x - x*_k||^2 / (2 * sigma_k^2) )

with one Gaussian per previously discovered minimum ``x*_k``. The gradient
(and, when available, the Hessian) of each bump is analytic, so the augmented
problem is just as smooth as the original and goes straight through
``pounce.minimize`` — bounds and constraints carry through untouched because
the bumps only touch the objective.

Run the demo:

    python gaussian_hump_minima.py

It sweeps the six-hump camel-back function and reports the minima it finds.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import numpy as np

import pounce


# --------------------------------------------------------------------------
# The Gaussian hump and its analytic derivatives.
# --------------------------------------------------------------------------
@dataclass
class GaussianHump:
    """A repulsive Gaussian bump centered on a discovered minimum.

    G(x) = A * exp(-||x - c||^2 / (2 sigma^2))
    """

    center: np.ndarray
    amplitude: float
    sigma: float

    def value(self, x: np.ndarray) -> float:
        d = x - self.center
        return float(self.amplitude * np.exp(-d @ d / (2.0 * self.sigma**2)))

    def grad(self, x: np.ndarray) -> np.ndarray:
        # dG/dx = G * (-(x - c) / sigma^2)
        d = x - self.center
        return self.value(x) * (-d / self.sigma**2)

    def hess(self, x: np.ndarray) -> np.ndarray:
        # d2G/dx2 = G * [ (x - c)(x - c)^T / sigma^4 - I / sigma^2 ]
        d = x - self.center
        g = self.value(x)
        n = x.size
        return g * (np.outer(d, d) / self.sigma**4 - np.eye(n) / self.sigma**2)


# --------------------------------------------------------------------------
# The driver.
# --------------------------------------------------------------------------
@dataclass
class MinimaSearchResult:
    minima: list[np.ndarray]
    values: list[float]
    n_solves: int
    trace: list[dict] = field(default_factory=list)

    def __len__(self) -> int:  # convenience
        return len(self.minima)


def _augment(
    fun: Callable,
    jac: Callable | None,
    hess: Callable | None,
    humps: Sequence[GaussianHump],
):
    """Wrap (fun, jac, hess) with the accumulated Gaussian humps."""

    def f_aug(x):
        return float(fun(x)) + sum(h.value(x) for h in humps)

    j_aug = None
    if jac is not None:
        def j_aug(x):
            g = np.asarray(jac(x), dtype=float).ravel().copy()
            for h in humps:
                g += h.grad(x)
            return g

    h_aug = None
    if hess is not None:
        def h_aug(x):
            H = np.asarray(hess(x), dtype=float).copy()
            for h in humps:
                H += h.hess(x)
            return H

    return f_aug, j_aug, h_aug


def find_minima(
    fun: Callable[[np.ndarray], float],
    x0: np.ndarray,
    *,
    jac: Callable | None = None,
    hess: Callable | None = None,
    bounds: Sequence | None = None,
    constraints: Sequence | dict | None = None,
    amplitude: float | None = None,
    sigma: float = 1.0,
    max_minima: int = 10,
    max_solves: int | None = None,
    dedup_tol: float = 1e-4,
    restart_jitter: float = 0.25,
    polish: bool = True,
    verify_min: bool = True,
    psd_tol: float = 1e-6,
    options: Mapping[str, Any] | None = None,
    rng: np.random.Generator | None = None,
) -> MinimaSearchResult:
    """Discover multiple local minima of ``fun`` by Gaussian basin-flooding.

    Parameters
    ----------
    fun, jac, hess, bounds, constraints, options
        Passed straight to :func:`pounce.minimize`. ``jac``/``hess`` are
        augmented analytically with the hump derivatives when supplied.
    x0
        Starting point for the first solve and the anchor for jittered
        restarts.
    amplitude
        Height of each Gaussian bump. Defaults to ``1.0`` if it cannot be
        inferred. A good rule of thumb is "a few times the typical depth of
        a basin" so the bump actually lifts the visited minimum above its
        neighbors.
    sigma
        Width of each bump, in the units of ``x``. Should be comparable to
        the spacing between minima: too small and the bump does not reach the
        ridge; too large and it floods neighbors you wanted to keep.
    max_minima
        Stop once this many distinct minima are collected.
    max_solves
        Hard cap on solver calls (defaults to ``4 * max_minima``).
    dedup_tol
        Two minima within this Euclidean distance are treated as the same.
    restart_jitter
        Std-dev of the Gaussian perturbation applied to ``x0`` when a solve
        lands on an already-known minimum (helps escape symmetric traps).
    polish
        After an augmented solve, re-solve on the *original* ``fun`` (no
        humps) so the reported point sits in the true basin rather than the
        slightly shifted flooded one.
    verify_min
        When ``hess`` is supplied, reject stationary points that are not
        genuine minima (saddles / maxima) by checking that the Hessian of
        the *original* objective is positive semidefinite. A Newton solver
        started exactly on a saddle never leaves it, so this filters those
        out. Ignored when ``hess`` is ``None``.
    psd_tol
        Smallest Hessian eigenvalue allowed by ``verify_min``.

    Returns
    -------
    MinimaSearchResult
        ``.minima`` and ``.values`` (sorted by objective) plus a ``.trace``
        of every solve for inspection.
    """
    x0 = np.asarray(x0, dtype=float)
    if rng is None:
        rng = np.random.default_rng(0)
    if amplitude is None:
        amplitude = 1.0
    if max_solves is None:
        max_solves = 4 * max_minima

    humps: list[GaussianHump] = []
    minima: list[np.ndarray] = []
    values: list[float] = []
    trace: list[dict] = []
    n_solves = 0

    def is_known(x):
        return any(np.linalg.norm(x - m) <= dedup_tol for m in minima)

    def in_bounds(x):
        if bounds is None:
            return True
        for xi, bd in zip(x, bounds):
            if bd is None:
                continue
            lo, hi = bd
            if lo is not None and xi < lo - 1e-9:
                return False
            if hi is not None and xi > hi + 1e-9:
                return False
        return True

    start = x0.copy()
    while len(minima) < max_minima and n_solves < max_solves:
        f_aug, j_aug, h_aug = _augment(fun, jac, hess, humps)
        res = pounce.minimize(
            f_aug, start, jac=j_aug, hess=h_aug,
            bounds=bounds, constraints=constraints, options=options,
        )
        n_solves += 1
        cand = np.asarray(res.x, dtype=float)

        if polish and humps:
            # Re-solve on the clean objective to settle into the true basin.
            res_p = pounce.minimize(
                fun, cand, jac=jac, hess=hess,
                bounds=bounds, constraints=constraints, options=options,
            )
            n_solves += 1
            cand = np.asarray(res_p.x, dtype=float)

        fval = float(fun(cand))
        is_min = True
        if verify_min and hess is not None:
            H = np.asarray(hess(cand), dtype=float)
            is_min = float(np.linalg.eigvalsh(0.5 * (H + H.T))[0]) >= -psd_tol
        accepted = (
            res.success and in_bounds(cand) and is_min and not is_known(cand)
        )
        trace.append({
            "solve": n_solves, "x": cand, "f": fval,
            "success": bool(res.success), "is_min": bool(is_min),
            "accepted": accepted, "n_humps": len(humps),
        })

        if accepted:
            minima.append(cand)
            values.append(fval)
            # Flood the basin we just found so the next solve avoids it.
            humps.append(GaussianHump(cand.copy(), amplitude, sigma))
            start = cand.copy()  # roll outward from the newest minimum
        else:
            # Stuck on a known basin (or a failed/out-of-bounds solve): pick a
            # fresh start. With finite bounds, sample the box uniformly so the
            # search is genuinely global (flooding alone drifts toward the
            # lower basins); otherwise jitter around x0 to break symmetry.
            if bounds is not None and all(
                b is not None and b[0] is not None and b[1] is not None
                for b in bounds
            ):
                lo = np.array([b[0] for b in bounds], dtype=float)
                hi = np.array([b[1] for b in bounds], dtype=float)
                start = lo + (hi - lo) * rng.random(x0.shape)
            else:
                start = x0 + restart_jitter * rng.standard_normal(x0.shape)

    order = np.argsort(values)
    return MinimaSearchResult(
        minima=[minima[i] for i in order],
        values=[values[i] for i in order],
        n_solves=n_solves,
        trace=trace,
    )


# --------------------------------------------------------------------------
# Demo: six-hump camel-back function (a classic multi-minima test).
# --------------------------------------------------------------------------
def _six_hump_camel():
    """f(x,y) = (4 - 2.1 x^2 + x^4/3) x^2 + x y + (-4 + 4 y^2) y^2.

    Two global minima at (+/-0.0898, -/+0.7126) with f = -1.0316, plus four
    higher local minima. A standard global-optimization torture test.
    """
    def fun(z):
        x, y = z
        return (4 - 2.1 * x**2 + x**4 / 3) * x**2 + x * y + (-4 + 4 * y**2) * y**2

    def jac(z):
        x, y = z
        dfdx = (8 - 8.4 * x**2 + 2 * x**4) * x + y
        dfdy = x + (-8 + 16 * y**2) * y
        return np.array([dfdx, dfdy])

    def hess(z):
        x, y = z
        d2x = 8 - 25.2 * x**2 + 10 * x**4
        d2y = -8 + 48 * y**2
        return np.array([[d2x, 1.0], [1.0, d2y]])

    return fun, jac, hess


def main():
    fun, jac, hess = _six_hump_camel()
    bounds = [(-2.0, 2.0), (-1.5, 1.5)]

    # Start off the central saddle at (0, 0): a Newton solver placed exactly
    # on a stationary point never leaves it.
    result = find_minima(
        fun, x0=np.array([0.5, 0.5]),
        jac=jac, hess=hess, bounds=bounds,
        amplitude=2.0, sigma=0.5,
        max_minima=6, max_solves=60, dedup_tol=1e-3,
        options={"print_level": 0, "tol": 1e-9},
    )

    print(f"Found {len(result)} distinct minima in {result.n_solves} solves:\n")
    print(f"  {'x':>10} {'y':>10} {'f(x,y)':>12}")
    for x, f in zip(result.minima, result.values):
        print(f"  {x[0]:10.4f} {x[1]:10.4f} {f:12.6f}")

    print("\n(global minimum of the six-hump camel is f = -1.031628)")


if __name__ == "__main__":
    main()
