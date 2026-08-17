"""Changed-structure warm starts: horizon shift and mesh prolongation.

Everything in :mod:`run` holds the problem's *shape* fixed along a
path, because that is what :class:`~.spec.ParametricFamily` requires and
what an ordinary warm start assumes. The issue asks for something the
fixed-shape runner cannot express: "horizon-shift/prolongation
transfer", where the next problem has different variables, or the same
variables meaning different things.

That is `pounce.WarmStart.transfer` / `.reindex` (pounce#621), and this
module is its benchmark. It is deliberately a separate harness rather
than another arm in `run.py`: bending the runner to permit shape changes
would weaken the invariant every other family depends on, for one arm.

The two experiments
-------------------

**Horizon shift** (`--experiment shift`). Linear MPC, solved at one
initial state after another, exactly as `mpc_horizon_*` is — but the
warm start is *shifted by one stage* before it is replayed, which is
what an MPC implementation actually does and what the suite's own notes
flag as missing:

    A shift-based warm-start arm for the closed-loop family. MPC codes
    shift the previous horizon by one step before reusing it;
    `warm-sqp` here carries the previous solution unshifted, which is
    the honest baseline for "carry the previous answer" but understates
    what an MPC implementation would do.

The shift is expressed as stable identifiers — the target problem's
slot *k* is labelled with the source's name for slot *k+1* — and
`reindex` does the gather. The final stage has no counterpart in the
source and is left unseeded, which is `reindex`'s documented behaviour
and the honest thing to do with a stage nobody has solved yet.

**Mesh prolongation** (`--experiment mesh`). The elliptic control
problem solved on a coarse mesh, then prolonged onto a mesh with twice
the resolution and solved there. Coarse node *i* sits exactly on fine
node *2i*, so the state and control transfer by injection on even nodes
and linear interpolation on odd ones. This is the classic nested-
iteration idea, measured rather than assumed.

Why the dual scaling is a separate arm
--------------------------------------

Prolonging the primal is obvious. Prolonging the multipliers is not,
because they are not mesh-independent quantities. Stationarity for this
discretization reads

    h(yᵢ − y_dᵢ) + (Jᵀλ)ᵢ = 0,     with  ∂cⱼ/∂yᵢ ~ h⁻²

so the PDE-row multipliers scale like ``h³``, the pin-row multipliers
like ``h``, and the control's bound multipliers like ``h`` (they balance
``αh·u``). Halving the mesh therefore divides the PDE-row multipliers by
8 and the others by 2. A prolongation that copies the multipliers across
unscaled is handing the solver a dual point wrong by those factors.

Both are run — `prolong-dual` with the scaling and
`prolong-dual-raw` without — because "carry the duals too" is the
obvious thing to try and the arm that shows it backfiring is worth more
than the arm that shows it working.

Output is a JSON payload of the same shape `run.py` writes, so the
composite report reads both without special-casing.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from .families import make
from .families.pde import mesh_family
from .kkt import kkt_residual
from .sparsity import SparseCallbacks

_HERE = os.path.dirname(os.path.abspath(__file__))

#: Status codes counted as solved, as elsewhere in the suite.
_OK_STATUS = (0, 1)


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=_HERE,
            stderr=subprocess.DEVNULL, text=True,
        ).strip()
    except Exception:
        return "unknown"


def _build(family, callbacks, tol: float, max_iter: int):
    import pounce

    b = family.bounds()
    prob = pounce.Problem(
        n=family.n, m=family.m, problem_obj=callbacks,
        lb=b.lb, ub=b.ub, cl=b.cl, cu=b.cu,
    )
    for key, val in (
        ("print_level", 0), ("sb", "yes"), ("tol", tol),
        ("constr_viol_tol", 1e-6), ("max_iter", max_iter),
    ):
        prob.add_option(key, val)
    return prob


def _record(family, callbacks, arm, step, x, info, elapsed, init_time) -> dict:
    res = kkt_residual(
        family, callbacks, x,
        info.get("mult_g"), info.get("mult_x_L"), info.get("mult_x_U"),
    )
    status = int(info.get("status", -99))
    return {
        "arm": arm,
        "step": step,
        "success": status in _OK_STATUS,
        "status": status,
        "status_msg": str(info.get("status_msg", "")),
        "iters": int(info.get("iter_count", -1)),
        "solve_time": elapsed,
        "init_time": init_time,
        "obj": float(info.get("obj_val", np.nan)),
        "kkt_error": float(res["kkt"]),
        "constr_viol": float(res["primal"]),
        **callbacks.counts(),
    }


# ---------------------------------------------------------------- shift


def _mpc_ids(nh: int) -> Tuple[List[str], List[str]]:
    """Stable names for one horizon's variables and constraint rows.

    Mirrors ``families/mpc_horizon.py``'s layout:
    ``z = [x₀ … x_N, u₀ … u_{N−1}]``, rows ``[pin₀, pin₁, dyn…]``.
    """
    var = [f"x{k}_{c}" for k in range(nh + 1) for c in (0, 1)]
    var += [f"u{k}" for k in range(nh)]
    con = ["pin_0", "pin_1"]
    con += [f"dyn{k}_{c}" for k in range(nh) for c in (0, 1)]
    return var, con


def _shifted_ids(nh: int) -> Tuple[List[str], List[str]]:
    """The same slots, labelled with the name of the stage *after* them.

    Slot ``k`` is given the source's name for slot ``k+1``, so
    ``reindex`` gathers stage ``k+1``'s value into stage ``k`` — the
    one-step shift. The last stage's label (``x{N+1}``, ``u{N}``) does
    not exist in the source, so `reindex` leaves it unseeded rather
    than inventing a value.
    """
    var = [f"x{k + 1}_{c}" for k in range(nh + 1) for c in (0, 1)]
    var += [f"u{k + 1}" for k in range(nh)]
    con = ["pin_0", "pin_1"]
    con += [f"dyn{k + 1}_{c}" for k in range(nh) for c in (0, 1)]
    return var, con


def run_shift(horizon: int, scale: float, tol: float, max_iter: int,
              n_steps: int) -> dict:
    """Cold vs carry-unshifted vs shift-by-one on a linear MPC path."""
    import pounce

    name = f"mpc_horizon_{horizon}"
    src_ids, src_cids = _mpc_ids(horizon)
    dst_ids, dst_cids = _shifted_ids(horizon)

    out: Dict[str, List[dict]] = {}
    for arm in ("cold", "carry", "shift"):
        family = make(name)
        cb = SparseCallbacks(family)
        path = family.theta_path(scale)[:n_steps]
        steps: List[dict] = []
        ws = None
        for k, theta in enumerate(path):
            family.set_theta(theta)
            cb.reset_counts()
            prob = _build(family, cb, tol, max_iter)

            init_t = 0.0
            kwargs = {}
            if arm != "cold" and ws is not None:
                t0 = time.perf_counter()
                try:
                    if arm == "shift":
                        seed = ws.reindex(prob, dst_ids, dst_cids)
                    else:
                        # `carry` replays the previous answer against the
                        # new problem verbatim -- the current `warm-*`
                        # behaviour, and the baseline `shift` must beat.
                        seed = ws.migrate(prob)
                    kwargs["warm_start"] = seed
                except Exception as exc:  # pragma: no cover - diagnostic
                    steps.append({"arm": arm, "step": k, "error": repr(exc)})
                    init_t = time.perf_counter() - t0
                    ws = None
                    continue
                init_t = time.perf_counter() - t0
            else:
                kwargs["x0"] = family.cold_x0()

            t0 = time.perf_counter()
            x, info = prob.solve(**kwargs)
            elapsed = time.perf_counter() - t0
            steps.append(_record(family, cb, arm, k, x, info, elapsed, init_t))
            ws = pounce.WarmStart.from_info(
                x, info, problem=prob, var_ids=src_ids, con_ids=src_cids
            )
        out[arm] = steps

    fam = make(name)
    return {
        "experiment": "shift",
        "family": name,
        "n": fam.n,
        "m": fam.m,
        "horizon": horizon,
        "scale_factor": scale,
        "n_steps": n_steps,
        "arms": out,
    }


def run_shift_closed_loop(tol: float, max_iter: int, n_steps: int) -> dict:
    """The same three arms on a genuine receding-horizon sequence.

    ``mpc_horizon_*`` walks its initial state around a *circle* at
    constant radius: consecutive problems are a rotation of one another,
    not consecutive instants of a simulation, so shifting the horizon by
    one stage is not the operation that relates them. ``nmpc_vanderpol``
    is the family whose path really is closed-loop — the next parameter
    is the state the plant reaches after applying the control the last
    solve produced — and it shares the ``[x₀ … x_N, u₀ … u_{N−1}]``
    layout, so the same identifiers apply.

    Running both is the point. If the shift helps here and not there,
    the result is about *what relates consecutive problems*, which is
    the property the whole suite is organized around; a benchmark that
    ran only the family where the shift wins would be evidence for
    nothing.

    The path is generated once by the cold arm and replayed for the
    others, exactly as :mod:`..runner` does for adaptive families, so
    every arm solves an identical sequence.
    """
    import pounce

    name = "nmpc_vanderpol"
    probe = make(name)
    nh = probe._NH
    src_ids, src_cids = _mpc_ids(nh)
    dst_ids, dst_cids = _shifted_ids(nh)

    # -- the cold arm defines the parameter sequence ----------------
    family = make(name)
    cb = SparseCallbacks(family)
    theta = family.initial_theta(1.0)
    path: List[np.ndarray] = []
    cold_steps: List[dict] = []
    for k in range(n_steps):
        family.set_theta(theta)
        path.append(np.atleast_1d(np.asarray(theta, float)).copy())
        cb.reset_counts()
        prob = _build(family, cb, tol, max_iter)
        t0 = time.perf_counter()
        x, info = prob.solve(x0=family.cold_x0())
        cold_steps.append(
            _record(family, cb, "cold", k, x, info,
                    time.perf_counter() - t0, 0.0)
        )
        theta = family.next_theta(x)

    out: Dict[str, List[dict]] = {"cold": cold_steps}
    for arm in ("carry", "shift"):
        family = make(name)
        cb = SparseCallbacks(family)
        steps: List[dict] = []
        ws = None
        for k, th in enumerate(path):
            family.set_theta(th)
            cb.reset_counts()
            prob = _build(family, cb, tol, max_iter)
            init_t = 0.0
            kwargs = {}
            if ws is not None:
                t0 = time.perf_counter()
                try:
                    seed = (ws.reindex(prob, dst_ids, dst_cids)
                            if arm == "shift" else ws.migrate(prob))
                    kwargs["warm_start"] = seed
                except Exception as exc:  # pragma: no cover - diagnostic
                    steps.append({"arm": arm, "step": k, "error": repr(exc)})
                    ws = None
                    continue
                init_t = time.perf_counter() - t0
            else:
                kwargs["x0"] = family.cold_x0()
            t0 = time.perf_counter()
            x, info = prob.solve(**kwargs)
            steps.append(_record(family, cb, arm, k, x, info,
                                 time.perf_counter() - t0, init_t))
            ws = pounce.WarmStart.from_info(
                x, info, problem=prob, var_ids=src_ids, con_ids=src_cids
            )
        out[arm] = steps

    fam = make(name)
    return {
        "experiment": "shift-closed-loop",
        "family": name,
        "n": fam.n,
        "m": fam.m,
        "horizon": nh,
        "scale_factor": 1.0,
        "n_steps": n_steps,
        "arms": out,
    }


# ----------------------------------------------------------------- mesh


def _prolong(coarse, fine, x_c: np.ndarray) -> np.ndarray:
    """Linear prolongation of a coarse iterate onto the fine mesh.

    ``N_f = 2·N_c + 1`` so ``h_f = h_c/2`` and coarse node ``i`` lands
    exactly on fine node ``2i``: injection on even nodes, midpoint
    average on odd ones, for the state and the control alike.
    """
    nc, nf = coarse._N, fine._N
    y_c, u_c = x_c[: nc + 2], x_c[nc + 2 :]
    y_f = np.empty(nf + 2)
    y_f[0::2] = y_c                       # fine 2i  <- coarse i
    y_f[1::2] = 0.5 * (y_c[:-1] + y_c[1:])  # fine 2i+1 <- midpoint

    # Controls live at interior nodes 1..N, i.e. fine index j holds the
    # control at position j·h_f. Even j = 2i takes coarse control i;
    # odd j averages its two coarse neighbours, clamped at the ends.
    u_f = np.empty(nf)
    j = np.arange(1, nf + 1)
    even = (j % 2) == 0
    idx = np.clip(j[even] // 2 - 1, 0, nc - 1)
    u_f[even] = u_c[idx]
    lo = np.clip((j[~even] - 1) // 2 - 1, 0, nc - 1)
    hi = np.clip((j[~even] + 1) // 2 - 1, 0, nc - 1)
    u_f[~even] = 0.5 * (u_c[lo] + u_c[hi])
    return np.concatenate([y_f, u_f])


def _prolong_rows(coarse, fine, lam_c: np.ndarray) -> np.ndarray:
    """Prolong the constraint multipliers (unscaled)."""
    nc, nf = coarse._N, fine._N
    pin, pde_c = lam_c[:2], lam_c[2:]
    pde_f = np.empty(nf)
    j = np.arange(1, nf + 1)
    even = (j % 2) == 0
    idx = np.clip(j[even] // 2 - 1, 0, nc - 1)
    pde_f[even] = pde_c[idx]
    lo = np.clip((j[~even] - 1) // 2 - 1, 0, nc - 1)
    hi = np.clip((j[~even] + 1) // 2 - 1, 0, nc - 1)
    pde_f[~even] = 0.5 * (pde_c[lo] + pde_c[hi])
    return np.concatenate([pin, pde_f])


def run_mesh(coarse_n: int, tol: float, max_iter: int, theta) -> dict:
    """Cold vs prolonged (primal / scaled dual / unscaled dual) on 2× mesh."""
    import pounce

    fine_n = 2 * coarse_n + 1
    coarse = mesh_family(coarse_n)()
    fine = mesh_family(fine_n)()
    theta = np.asarray(theta, dtype=float)
    coarse.set_theta(theta)
    fine.set_theta(theta)

    cb_c, cb_f = SparseCallbacks(coarse), SparseCallbacks(fine)

    # -- the coarse solve every prolonged arm starts from -----------
    cb_c.reset_counts()
    prob_c = _build(coarse, cb_c, tol, max_iter)
    t0 = time.perf_counter()
    x_c, info_c = prob_c.solve(x0=coarse.cold_x0())
    coarse_rec = _record(coarse, cb_c, "coarse", 0, x_c, info_c,
                         time.perf_counter() - t0, 0.0)

    ws_c = pounce.WarmStart.from_info(x_c, info_c, problem=prob_c)
    ratio = fine._h / coarse._h  # = 1/2 by construction

    arms: Dict[str, List[dict]] = {}
    for arm in ("cold", "prolong-primal", "prolong-dual", "prolong-dual-raw"):
        cb_f.reset_counts()
        prob_f = _build(fine, cb_f, tol, max_iter)
        init_t = 0.0
        kwargs = {}
        if arm != "cold":
            t0 = time.perf_counter()
            scale_rows = arm == "prolong-dual"

            def mapper(ctx, _arm=arm, _scale=scale_rows):
                payload = {"x": _prolong(coarse, fine, np.asarray(ctx.source.x))}
                if _arm == "prolong-primal":
                    # Duals left where they were would be the *coarse*
                    # length and fail the length check, so an explicit
                    # None hands the initializer the "you decide" case.
                    payload.update(lagrange=None, zl=None, zu=None, mu=None)
                    return payload
                lam = ctx.source.lagrange
                if lam is not None:
                    lam_f = _prolong_rows(coarse, fine, np.asarray(lam))
                    if _scale:
                        # PDE rows ~ h³, pin rows ~ h.
                        lam_f[:2] *= ratio
                        lam_f[2:] *= ratio**3
                    payload["lagrange"] = lam_f
                for key in ("zl", "zu"):
                    z = getattr(ctx.source, key)
                    if z is None:
                        continue
                    z_f = _prolong(coarse, fine, np.asarray(z))
                    if _scale:
                        z_f *= ratio  # bound multipliers balance αh·u
                    payload[key] = z_f
                return payload

            try:
                kwargs["warm_start"] = ws_c.transfer(prob_f, mapper)
            except Exception as exc:  # pragma: no cover - diagnostic
                arms[arm] = [{"arm": arm, "step": 0, "error": repr(exc)}]
                continue
            init_t = time.perf_counter() - t0
        else:
            kwargs["x0"] = fine.cold_x0()

        t0 = time.perf_counter()
        x_f, info_f = prob_f.solve(**kwargs)
        arms[arm] = [
            _record(fine, cb_f, arm, 0, x_f, info_f,
                    time.perf_counter() - t0, init_t)
        ]

    return {
        "experiment": "mesh",
        "family": f"elliptic_control_{coarse_n}->{fine_n}",
        "coarse_n": coarse_n,
        "fine_n": fine_n,
        "n": fine.n,
        "m": fine.m,
        "theta": [float(v) for v in theta],
        "coarse_solve": coarse_rec,
        "arms": arms,
    }


# ------------------------------------------------------------------ CLI


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="warmstart.transfer", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--experiment", default="all",
                   choices=("shift", "shift-cl", "mesh", "all"))
    p.add_argument("--horizons", default="10,20,40",
                   help="comma-separated MPC horizons for the shift experiment")
    p.add_argument("--meshes", default="20,40,80",
                   help="comma-separated coarse mesh sizes; each is prolonged "
                        "onto 2N+1")
    p.add_argument("--scale", type=float, default=1.0,
                   help="parameter-step multiplier for the shift path")
    p.add_argument("--steps", type=int, default=12)
    p.add_argument("--tol", type=float, default=1e-8)
    p.add_argument("--max-iter", type=int, default=500)
    p.add_argument("--out", default=os.path.join(_HERE, "transfer.json"))
    args = p.parse_args(argv)

    import pounce

    runs: List[dict] = []
    if args.experiment in ("shift", "all"):
        for h in [int(v) for v in args.horizons.split(",") if v.strip()]:
            print(f"shift  horizon={h} ... ", end="", flush=True)
            t0 = time.perf_counter()
            runs.append(run_shift(h, args.scale, args.tol, args.max_iter,
                                  args.steps))
            print(f"{time.perf_counter() - t0:.1f}s")
    if args.experiment in ("shift-cl", "all"):
        print("shift  closed-loop (nmpc_vanderpol) ... ", end="", flush=True)
        t0 = time.perf_counter()
        runs.append(run_shift_closed_loop(args.tol, args.max_iter, args.steps))
        print(f"{time.perf_counter() - t0:.1f}s")
    if args.experiment in ("mesh", "all"):
        for nc in [int(v) for v in args.meshes.split(",") if v.strip()]:
            print(f"mesh   {nc}->{2 * nc + 1} ... ", end="", flush=True)
            t0 = time.perf_counter()
            runs.append(run_mesh(nc, args.tol, args.max_iter,
                                 np.array([0.5, 0.0])))
            print(f"{time.perf_counter() - t0:.1f}s")

    payload = {
        "meta": {
            "suite": "warmstart-transfer",
            "solver": "pounce",
            "solver_version": getattr(pounce, "__version__", "unknown"),
            "git_sha": _git_sha(),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "tol": args.tol,
            "max_iter": args.max_iter,
        },
        "runs": runs,
    }
    with open(args.out, "w") as fh:
        json.dump(payload, fh, indent=1)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
