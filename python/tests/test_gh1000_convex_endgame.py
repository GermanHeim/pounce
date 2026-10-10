"""gh#1000 items I.8 and I.D: the convex HSDE endgame.

Two models, each from the book, and each failing a different way on the old
driver.

* **I.D, battery MPC.** A 96-step storage QP solved closed loop. Main reported
  ``optimal_inaccurate`` on every MPC step that took the battery to a bound.
  The internal slack converged, but the true slack ``h - Gx`` sat a
  regularization-sized distance away with a multiplier of 72. The (z, z)
  regularization floor was sized by ``tol``, not ``tol_gap``, so the loop
  could not close that last 1e-9.
* **I.8, refinery blend LP.** An LP with its objective rescaled by ``10**k``.
  At small ``k`` the in-loop dual test and the verdict's per-column floor
  disagreed, and the gap test's noise floor did not count the size of the
  terms being cancelled. The run ground to the 200-iteration cap (``k = -6``
  with the contract row) or ended ``optimal_inaccurate`` (``k = -5`` at
  ``tol = 1e-10``).
"""

import numpy as np
import pytest

pytest.importorskip("scipy")
import scipy.sparse as sps  # noqa: E402

from pounce.qp import solve_qp  # noqa: E402

# ---------------------------------------------------------------- battery MPC

DT, N = 0.25, 96
EMIN, EMAX, PMAX, GMAX = 10.0, 190.0, 50.0, 100.0


def _load(t):
    return 70 + 40 * np.sin(2 * np.pi * (t - 9) / 24)


def _price(t):
    h = np.asarray(t) % 24
    return np.where((h >= 12) & (h < 20), 0.25, 0.08)


def _battery_qp(E0, p_prev, t0, lhat, w=1e-3):
    n = 2 * N + 1
    c = np.r_[np.zeros(N + 1), _price(t0 + DT * np.arange(N)) * DT]
    D = sps.eye(N) - sps.eye(N, N, -1)
    Q = sps.block_diag([sps.csr_matrix((N + 1, N + 1)), 2 * w * D.T @ D]).tocsc()
    c[N + 1] -= 2 * w * p_prev
    eye = sps.eye(n, format="csr")
    A = sps.vstack([eye[:1], sps.hstack([sps.eye(N, N + 1, 1) - sps.eye(N, N + 1), -DT * sps.eye(N)])]).tocsc()
    G = sps.vstack([eye[N + 1 :], -eye[N + 1 :]]).tocsc()
    return dict(
        P=Q,
        c=c,
        A=A,
        b=np.r_[E0, np.zeros(N)],
        G=G,
        h=np.r_[GMAX - lhat, lhat],
        lb=np.r_[np.full(N + 1, EMIN), np.full(N, -PMAX)],
        ub=np.r_[np.full(N + 1, EMAX), np.full(N, PMAX)],
    )


@pytest.mark.parametrize("noise,warm", [(0.0, True), (0.1, False)])
def test_battery_mpc_certifies_every_step(noise, warm):
    # Main: optimal_inaccurate at k = 81 and 177 (noise 0, warm), and at 80,
    # 81, 176 and 177 (noise 0.1, cold), each after 15-24 iterations.
    rng = np.random.default_rng(5)
    E, p_prev, prev = 100.0, 0.0, None
    bad = []
    for k in range(int(2 * 24 / DT)):
        t0 = k * DT
        lhat = _load(t0 + DT * np.arange(N)) * (1 + noise * rng.normal(size=N))
        lhat[0] = _load(t0)
        r = solve_qp(**_battery_qp(E, p_prev, t0, lhat), warm_start=prev if warm else None)
        if r.status != "optimal":
            bad.append((k, r.status, r.iters))
        prev, p = r, r.x[N + 1]
        E, p_prev = E + DT * p, p
    assert not bad, bad


# ------------------------------------------------------------ refinery blend

STOCKS = [
    (3000, 45, 93, 52.0, 10.0, 0.584),
    (6000, 70, 70, 11.0, 5.0, 0.664),
    (9000, 98, 98, 3.5, 1.0, 0.810),
    (12000, 92, 92, 7.0, 20.0, 0.745),
    (4000, 105, 96, 4.6, 6.0, 0.700),
]
PRODS = [(110, 20000, 91, 9.0, 10), (122, 8000, 96, 9.0, 10)]
COLS = [(i, j) for i in range(5) for j in range(2)]


def _blend(contract):
    G, h = [], []

    def row(d, b):
        r = np.zeros(len(COLS))
        for key, v in d.items():
            r[COLS.index(key)] = v
        G.append(r)
        h.append(b)

    for i, s in enumerate(STOCKS):
        row({(i, j): 1.0 for j in range(2)}, s[0])
    for j, (_, mx, ron, rvp, smax) in enumerate(PRODS):
        row({(i, j): 1.0 for i in range(5)}, mx)
        row({(i, j): -(s[2] - ron) for i, s in enumerate(STOCKS)}, 0.0)
        row({(i, j): s[3] ** 1.25 - rvp**1.25 for i, s in enumerate(STOCKS)}, 0.0)
        row({(i, j): s[5] * (s[4] - smax) for i, s in enumerate(STOCKS)}, 0.0)
    if contract:
        row({(i, 1): -1.0 for i in range(5)}, -8000.0)
    c = -np.array([PRODS[j][0] - STOCKS[i][1] for i, j in COLS], float)
    ub = np.array([PRODS[j][1] for _, j in COLS], float)
    return c, np.array(G), np.array(h), np.zeros(len(COLS)), ub


@pytest.mark.parametrize("contract", [False, True])
def test_blend_lp_certifies_at_every_objective_scale(contract):
    c, G, h, lb, ub = _blend(contract)
    ref = solve_qp(c=c, G=G, h=h, lb=lb, ub=ub, method="ipm")
    assert ref.status == "optimal"
    bad = []
    for k in range(-6, 7):
        for tol in (None, 1e-8, 1e-9, 1e-10):
            r = solve_qp(c=c * 10.0**k, G=G, h=h, lb=lb, ub=ub, tol=tol, method="ipm")
            # Measured on this branch: at most 68 iterations (k = -6,
            # tol = 1e-10, contract). The cap is 200.
            if r.status != "optimal" or r.iters > 120:
                bad.append((k, tol, r.status, r.iters))
                continue
            assert r.obj / 10.0**k == pytest.approx(ref.obj, rel=1e-8), (k, tol)
    assert not bad, bad
