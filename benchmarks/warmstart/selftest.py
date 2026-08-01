"""Finite-difference check of every family's derivatives.

A wrong Jacobian or Hessian in a benchmark family does not announce
itself — it shows up as "the warm-started arm needed more iterations",
which is exactly the signal the benchmark exists to measure. So this
runs before any result is believed, and in CI.

Checks, at several points along each family's path:

* ``gradient``          vs central differences of ``objective``
* ``jacobian_dense``    vs central differences of ``constraints``
* ``hessian_dense``     vs central differences of ∇ₓL, where
  ``L = obj_factor·f + λᵀg`` (the cyipopt sign convention the
  families are written to)

For any family claiming ``quadratic = True`` it additionally re-derives
the family from the extracted standard-form QP data (:mod:`qpform`) —
objective, gradient, and every constraint row — and checks that ``P`` is
symmetric positive semidefinite. A family that fails this is not a QP
and must not be handed to the convex solver, whatever it declares.

Run with ``python -m warmstart.selftest`` (no solver required).
"""

from __future__ import annotations

import sys
from typing import List

import numpy as np

from . import qpform
from .families import REGISTRY, make
from .spec import ParametricFamily
from .sparsity import SparseCallbacks

_EPS = 1e-6
_RTOL = 2e-4


def _fd_gradient(f, x):
    g = np.zeros_like(x)
    for i in range(x.size):
        h = _EPS * max(1.0, abs(x[i]))
        xp, xm = x.copy(), x.copy()
        xp[i] += h
        xm[i] -= h
        g[i] = (f(xp) - f(xm)) / (2.0 * h)
    return g


def _fd_jacobian(c, x, m):
    j = np.zeros((m, x.size))
    for i in range(x.size):
        h = _EPS * max(1.0, abs(x[i]))
        xp, xm = x.copy(), x.copy()
        xp[i] += h
        xm[i] -= h
        j[:, i] = (c(xp) - c(xm)) / (2.0 * h)
    return j


def _rel_err(a, b) -> float:
    denom = max(1.0, float(np.max(np.abs(b))) if np.size(b) else 1.0)
    return float(np.max(np.abs(a - b))) / denom if np.size(a) else 0.0


def _sample_points(family: ParametricFamily, rng) -> List[np.ndarray]:
    b = family.bounds()
    lo = np.where(b.lb > -1e19, b.lb, -2.0)
    hi = np.where(b.ub < 1e19, b.ub, 2.0)
    span = np.clip(hi - lo, 1e-3, 3.0)
    x0 = np.clip(family.cold_x0(), lo, hi)
    return [x0] + [
        np.clip(x0 + span * (rng.random(family.n) - 0.5), lo, hi)
        for _ in range(3)
    ]



def _check_sparse(family: ParametricFamily, rng, max_cols: int = 12) -> List[str]:
    """Verify a declared sparse structure without building anything dense.

    A large family cannot be checked the way the small ones are — an
    `(n, n)` finite-difference Hessian at n = 2402 is 46 MB per column.
    Instead this checks the packed values directly against central
    differences on a random subset of columns, reconstructing each
    column from the triplets. Cheap, and it tests the code path the
    solver actually calls rather than a dense twin of it.
    """
    out: List[str] = []
    jr, jc, hr, hc = (np.asarray(a) for a in family.sparse_structure())
    n, m = family.n, family.m
    x = family.cold_x0() + 0.1 * rng.standard_normal(n)
    lam = rng.standard_normal(m)

    jv = np.asarray(family.jacobian_values(x), dtype=float)
    hv = np.asarray(family.hessian_values(x, lam, 1.0), dtype=float)
    if jv.size != jr.size:
        out.append(f"jacobian_values has {jv.size} entries, structure has {jr.size}")
        return out
    if hv.size != hr.size:
        out.append(f"hessian_values has {hv.size} entries, structure has {hr.size}")
        return out
    if np.any(hr < hc):
        out.append("hessian structure is not lower-triangular")

    def lag_grad(z):
        g = 1.0 * np.asarray(family.gradient(z), dtype=float)
        v = np.asarray(family.jacobian_values(z), dtype=float)
        return g + np.bincount(jc, weights=v * lam[jr], minlength=n)

    cols = rng.choice(n, size=min(max_cols, n), replace=False)
    for j in cols:
        h = _EPS * max(1.0, abs(x[j]))
        xp, xm = x.copy(), x.copy()
        xp[j] += h
        xm[j] -= h

        if m:
            fd = (family.constraints(xp) - family.constraints(xm)) / (2.0 * h)
            col = np.zeros(m)
            sel = jc == j
            np.add.at(col, jr[sel], jv[sel])
            if _rel_err(col, fd) > _RTOL:
                out.append(f"sparse jacobian column {j} disagrees with FD")

        fd_h = (lag_grad(xp) - lag_grad(xm)) / (2.0 * h)
        # Column j of the symmetric Hessian, from the stored lower triangle:
        # entries whose column is j, plus the mirror of entries whose row is j,
        # minus the diagonal which both selections pick up.
        col = np.zeros(n)
        sel = hc == j
        np.add.at(col, hr[sel], hv[sel])
        sel = hr == j
        np.add.at(col, hc[sel], hv[sel])
        diag = (hr == j) & (hc == j)
        col[j] -= hv[diag].sum()
        if _rel_err(col, fd_h) > _RTOL:
            out.append(f"sparse hessian column {j} disagrees with FD")
    return out


def check_family(name: str, verbose: bool = True) -> List[str]:
    """Returns a list of failure messages (empty when the family is sound)."""
    rng = np.random.default_rng(1234)
    family = make(name)
    failures: List[str] = []

    path = family.theta_path(1.0)
    if path is None:
        path = [family.initial_theta(1.0)]
    thetas = [path[0], path[len(path) // 2], path[-1]]

    sparse = family.sparse_structure() is not None
    heavy = family.n > 300

    if sparse:
        for theta in thetas:
            family.set_theta(theta)
            for msg in _check_sparse(family, rng):
                failures.append(f"{name}: {msg} (θ={theta})")
    if heavy:
        # Too large to finite-difference densely, and its dense twin is
        # never called during a solve anyway. The sparse check above is
        # the real one; the dense methods are exercised at the small
        # sizes of the same family class.
        if verbose:
            status = "FAIL" if failures else "ok"
            print(
                f"  {name:<22} n={family.n:<5} m={family.m:<5} "
                f"sparse-only (n > 300)  {status}"
            )
        return failures

    worst = {"grad": 0.0, "jac": 0.0, "hess": 0.0}
    for theta in thetas:
        family.set_theta(theta)
        for x in _sample_points(family, rng):
            e = _rel_err(family.gradient(x), _fd_gradient(family.objective, x))
            worst["grad"] = max(worst["grad"], e)
            if e > _RTOL:
                failures.append(f"{name}: gradient rel-err {e:.2e} at θ={theta}")

            e = _rel_err(
                family.jacobian_dense(x),
                _fd_jacobian(family.constraints, x, family.m),
            )
            worst["jac"] = max(worst["jac"], e)
            if e > _RTOL:
                failures.append(f"{name}: jacobian rel-err {e:.2e} at θ={theta}")

            lam = rng.standard_normal(family.m)
            for obj_factor in (1.0, 0.0):

                def lag_grad(z, lam=lam, obj_factor=obj_factor):
                    return obj_factor * family.gradient(z) + family.jacobian_dense(
                        z
                    ).T @ lam

                fd = np.zeros((family.n, family.n))
                for i in range(family.n):
                    h = _EPS * max(1.0, abs(x[i]))
                    xp, xm = x.copy(), x.copy()
                    xp[i] += h
                    xm[i] -= h
                    fd[:, i] = (lag_grad(xp) - lag_grad(xm)) / (2.0 * h)
                e = _rel_err(family.hessian_dense(x, lam, obj_factor), fd)
                worst["hess"] = max(worst["hess"], e)
                if e > _RTOL:
                    failures.append(
                        f"{name}: hessian rel-err {e:.2e} "
                        f"(obj_factor={obj_factor}) at θ={theta}"
                    )

    # The sparse wrapper must agree with the dense source it came from.
    cb = SparseCallbacks(family)
    x = family.cold_x0()
    lam = rng.standard_normal(family.m)
    jr, jc = cb.jacobianstructure()
    dense_j = family.jacobian_dense(x)
    packed = cb.jacobian(x)
    if packed.size != jr.size or not np.allclose(packed, dense_j[jr, jc]):
        failures.append(f"{name}: sparse jacobian disagrees with dense")
    hr, hc = cb.hessianstructure()
    dense_h = family.hessian_dense(x, lam, 1.0)
    packed_h = cb.hessian(x, lam, 1.0)
    if packed_h.size != hr.size or not np.allclose(packed_h, dense_h[hr, hc]):
        failures.append(f"{name}: sparse hessian disagrees with dense")
    if np.any(hr < hc):
        failures.append(f"{name}: hessian structure is not lower-triangular")
    # Anything nonzero in the dense lower triangle must be in the pattern.
    covered = np.zeros_like(dense_h, dtype=bool)
    covered[hr, hc] = True
    missed = np.tril(np.abs(dense_h) > 0) & ~covered
    if missed.any():
        failures.append(
            f"{name}: {int(missed.sum())} nonzero hessian entries "
            "outside the sampled sparsity pattern"
        )

    # A small family with both paths must have them agree, which is what
    # makes the large siblings' sparse-only check trustworthy.
    if sparse:
        jr, jc, hr, hc = (np.asarray(a) for a in family.sparse_structure())
        xs = family.cold_x0()
        lam_s = rng.standard_normal(family.m)
        if family.m and not np.allclose(
            family.jacobian_values(xs), family.jacobian_dense(xs)[jr, jc]
        ):
            failures.append(f"{name}: sparse jacobian values != dense")
        if not np.allclose(
            family.hessian_values(xs, lam_s, 1.0),
            family.hessian_dense(xs, lam_s, 1.0)[hr, hc],
        ):
            failures.append(f"{name}: sparse hessian values != dense")

    # Families that claim to be QPs must survive being turned into one.
    if family.quadratic:
        for theta in thetas:
            family.set_theta(theta)
            qp = qpform.extract(family)
            for msg in qpform.verify(family, qp, rng):
                failures.append(f"{name}: QP extraction — {msg} (θ={theta})")

    if verbose:
        status = "FAIL" if failures else "ok"
        qp_note = " qp:ok" if family.quadratic and not failures else ""
        print(
            f"  {name:<22} n={family.n:<4} m={family.m:<4} "
            f"grad={worst['grad']:.1e} jac={worst['jac']:.1e} "
            f"hess={worst['hess']:.1e}  {status}{qp_note}"
        )
    return failures


def main(argv=None) -> int:
    names = argv or list(REGISTRY)
    print(f"Checking derivatives for {len(names)} families "
          f"(central differences, rtol={_RTOL:g}):")
    failures: List[str] = []
    for name in names:
        failures.extend(check_family(name))
    if failures:
        print("\nFAILURES:")
        for f in failures:
            print(f"  {f}")
        return 1
    print("\nAll families pass.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:] or None))
