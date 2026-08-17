"""Seed-quality sweep — the measurement gh#606, gh#617 and gh#618 argue over.

``run.py`` walks a parameter path and asks "does carrying the previous
step's point forward pay?". That is the right question for continuation,
and it is the wrong one for the three issues above, every one of which is
about a *single* warm solve whose seed is deliberately of known quality.
Those measurements were taken with throwaway scripts and quoted in issue
bodies, so nobody could re-take them; this module is that harness, kept.

Two axes, crossed.

**Staleness** — how far the seed's problem is from the one being solved:

``exact``
    Solve the family at its path's far end, then re-solve *the same*
    instance from that solution. The seed is the answer. Nothing here
    should cost more than one or two iterations, and this is where a
    reconstruction that mangles a good seed shows up.
``stale``
    Solve at the *near* end of the 4x-scale path, then solve the *far*
    end seeded from it — gh#618's "mildly stale" regime, the one whose
    dual reconstruction fires without the barrier escalation that is
    supposed to pay for it.
``corrupted``
    ``exact``'s primal point, exactly, paired with multipliers that have
    had ``N(0, CORRUPTION_VAR)`` noise added — gh#617's regime. The
    primal seed is perfect and the dual seed is garbage, which is
    precisely the split the stationarity reconstruction cannot see.

**Seed content** — how much of the point the caller hands over. gh#617
asks for all four, because a rejection test that fixes ``corrupted`` at
the expense of ``full`` is not a fix:

===========  ==========================================================
``full``     ``x``, ``lagrange``, ``zl``, ``zu``
``partial``  ``x``, ``lagrange`` — the shape a frontend that keeps only
             constraint duals produces
``primal``   ``x`` alone
``cold``     no seed at all; the family's own cold start. The reference
             every warm number is read against.
===========  ==========================================================

`cold` ignores the staleness axis by construction (there is no seed to
be stale), so it is measured once and reused.

Usage::

    python -m warmstart.seedmodes --staleness corrupted --recentering residual
    python -m warmstart.seedmodes --staleness all --contents all --out r.json
    python -m warmstart.seedmodes --families nmpc_vanderpol --scales large

``--recentering none`` is the attribution control: it restores the
pre-gh#606 constants exactly, so the whole gh#606 block can be isolated
at a single commit instead of by comparing across two builds of a moving
`main`.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from typing import Dict, List, Optional

import numpy as np

import pounce

from .families import REGISTRY, make
from .sparsity import SparseCallbacks
from .spec import SCALES, ParametricFamily

_HERE = os.path.dirname(os.path.abspath(__file__))

STALENESS = ("exact", "stale", "corrupted")
CONTENTS = ("full", "partial", "primal", "cold")

#: Variance of the additive multiplier noise in the ``corrupted`` mode,
#: as gh#617 specifies it: ``N(0, 1e4)``, i.e. a standard deviation of
#: 100. Large enough to swamp every multiplier in the corpus, which is
#: the point — the seed is not merely inaccurate, it carries no signal.
CORRUPTION_VAR = 1e4

#: Fixed so a corrupted run is reproducible across builds; the whole
#: measurement is a before/after comparison and a fresh draw per build
#: would put noise in the difference.
CORRUPTION_SEED = 20606


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=_HERE,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def _build(family: ParametricFamily, callbacks, tol: float, max_iter: int):
    b = family.bounds()
    prob = pounce.Problem(
        n=family.n,
        m=family.m,
        problem_obj=callbacks,
        lb=b.lb,
        ub=b.ub,
        cl=b.cl,
        cu=b.cu,
    )
    prob.add_option("print_level", 0)
    prob.add_option("sb", "yes")
    prob.add_option("tol", tol)
    prob.add_option("constr_viol_tol", 1e-6)
    prob.add_option("max_iter", max_iter)
    return prob


def _solve_cold(family, callbacks, tol, max_iter):
    prob = _build(family, callbacks, tol, max_iter)
    x, info = prob.solve(x0=np.asarray(family.cold_x0(), dtype=float))
    return x, info


def _solve_warm(family, callbacks, tol, max_iter, ws: pounce.WarmStart):
    prob = _build(family, callbacks, tol, max_iter)
    for key, val in ws.options().items():
        prob.add_option(key, val)
    kw = ws.solve_kwargs()
    kw.pop("working_set", None)
    x, info = prob.solve(x0=ws.x, **kw)
    return x, info


def _path_ends(family: ParametricFamily, scale: float):
    """``(near, far)`` parameter points of this family's path.

    An adaptive family has no scripted path; its θ sequence depends on
    the solutions, so the near end is `initial_theta` and the far end is
    reached by walking the loop. That walk is the family's own
    definition of the path, so it is the same sequence `run.py` records.
    """
    path = family.theta_path(scale)
    if path is not None:
        return path[0], path[-1]
    theta = family.initial_theta(scale)
    near = np.atleast_1d(np.asarray(theta, dtype=float)).copy()
    callbacks = SparseCallbacks(family)
    for _ in range(family.n_steps - 1):
        family.set_theta(theta)
        x, _info = _solve_cold(family, callbacks, 1e-8, 500)
        theta = family.next_theta(np.asarray(x, dtype=float))
    return near, np.atleast_1d(np.asarray(theta, dtype=float)).copy()


def _seed_from(info, x, content: str, rng: Optional[np.random.Generator],
               recentering: Optional[str]) -> pounce.WarmStart:
    """Build the ``WarmStart`` one (content, corruption) pair asks for."""
    lam = info.get("mult_g")
    zl = info.get("mult_x_L")
    zu = info.get("mult_x_U")
    lam = None if lam is None else np.asarray(lam, dtype=float).copy()
    zl = None if zl is None else np.asarray(zl, dtype=float).copy()
    zu = None if zu is None else np.asarray(zu, dtype=float).copy()

    if rng is not None:
        sd = CORRUPTION_VAR ** 0.5
        for v in (lam, zl, zu):
            if v is not None and v.size:
                v += rng.normal(0.0, sd, size=v.shape)

    if content == "partial":
        zl = zu = None
    elif content == "primal":
        lam = zl = zu = None

    mu = info.get("mu")
    mu = float(mu) if mu is not None and float(mu) > 0.0 else None
    return pounce.WarmStart(
        x=np.asarray(x, dtype=float),
        lagrange=lam,
        zl=zl,
        zu=zu,
        mu=mu,
        recentering=recentering,
    )


def _record(info, x, seconds: float) -> dict:
    diag = info.get("warm_start") or {}
    return {
        "status": int(info.get("status", -99)),
        "status_msg": str(info.get("status_msg", "")),
        "iters": int(info.get("iter_count", -1)),
        "obj": float(info.get("obj_val", np.nan)),
        "seconds": seconds,
        "kkt": float(
            info.get("final_unscaled_kkt_error",
                     info.get("final_kkt_error", np.nan))
        ),
        "viol": float(
            info.get("final_unscaled_constr_viol",
                     info.get("final_constr_viol", np.nan))
        ),
        # gh#606's own diagnostics, so a moved iteration count can be
        # attributed to a block verdict or a barrier move rather than
        # guessed at.
        "warm_start": {k: diag[k] for k in sorted(diag)} if diag else None,
        "x": [float(v) for v in np.asarray(x, dtype=float).ravel()],
    }


def measure(
    family_name: str,
    scale_name: str,
    staleness: str,
    contents: List[str],
    recentering: Optional[str],
    tol: float = 1e-8,
    max_iter: int = 500,
) -> dict:
    """One family at one scale: the seed solve, then every content mode."""
    scale = SCALES[scale_name]
    family = make(family_name)
    callbacks = SparseCallbacks(family)
    near, far = _path_ends(family, scale)

    # The instance under test is always the far end of the path. What
    # changes with `staleness` is where the seed came from.
    seed_theta = near if staleness == "stale" else far

    family.set_theta(seed_theta)
    t0 = time.perf_counter()
    seed_x, seed_info = _solve_cold(family, callbacks, tol, max_iter)
    seed_secs = time.perf_counter() - t0

    family.set_theta(far)
    out: Dict[str, dict] = {}
    for content in contents:
        if content == "cold":
            t0 = time.perf_counter()
            x, info = _solve_cold(family, callbacks, tol, max_iter)
            out[content] = _record(info, x, time.perf_counter() - t0)
            continue
        rng = (
            np.random.default_rng(CORRUPTION_SEED)
            if staleness == "corrupted"
            else None
        )
        ws = _seed_from(seed_info, seed_x, content, rng, recentering)
        t0 = time.perf_counter()
        x, info = _solve_warm(family, callbacks, tol, max_iter, ws)
        out[content] = _record(info, x, time.perf_counter() - t0)

    return {
        "family": family_name,
        "scale": scale_name,
        "staleness": staleness,
        "seed": {
            "theta": [float(v) for v in np.atleast_1d(seed_theta)],
            "status": int(seed_info.get("status", -99)),
            "iters": int(seed_info.get("iter_count", -1)),
            "seconds": seed_secs,
        },
        "theta": [float(v) for v in np.atleast_1d(far)],
        "modes": out,
    }


def _csv(value: str, valid, what: str) -> List[str]:
    if value == "all":
        return list(valid)
    items = [v.strip() for v in value.split(",") if v.strip()]
    bad = [i for i in items if i not in valid]
    if bad:
        raise SystemExit(
            f"unknown {what}: {', '.join(bad)}\nknown: {', '.join(valid)}"
        )
    return items


def summarize(payload: dict) -> str:
    """Per-family iteration counts plus the column totals gh#617/#618 quote."""
    rows = payload["runs"]
    contents = payload["meta"]["contents"]
    width = max([len("family @ scale")] + [
        len(f"{r['family']} @ {r['scale']}") for r in rows
    ])
    lines = []
    for staleness in payload["meta"]["staleness"]:
        subset = [r for r in rows if r["staleness"] == staleness]
        if not subset:
            continue
        lines.append(f"\n## staleness = {staleness}"
                     f"   (recentering = {payload['meta']['recentering']})\n")
        head = f"{'family @ scale':<{width}}" + "".join(
            f"  {c:>9}" for c in contents
        )
        lines.append(head)
        lines.append("-" * len(head))
        totals = {c: 0 for c in contents}
        failed = {c: 0 for c in contents}
        for r in subset:
            cells = []
            for c in contents:
                m = r["modes"].get(c)
                if m is None:
                    cells.append(f"  {'-':>9}")
                    continue
                totals[c] += m["iters"]
                bad = m["status"] not in (0, 1)
                failed[c] += 1 if bad else 0
                cells.append(f"  {m['iters']:>8}{'*' if bad else ' '}")
            lines.append(f"{r['family'] + ' @ ' + r['scale']:<{width}}"
                         + "".join(cells))
        lines.append("-" * len(head))
        lines.append(f"{'TOTAL':<{width}}" + "".join(
            f"  {totals[c]:>8} " for c in contents
        ))
        if any(failed.values()):
            lines.append(f"{'not-converged':<{width}}" + "".join(
                f"  {failed[c]:>8} " for c in contents
            ))
    lines.append("\n* = status was not SolveSucceeded/SolvedToAcceptableLevel")
    return "\n".join(lines)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="warmstart.seedmodes",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--families", default="all", help="comma-separated, or 'all'")
    p.add_argument("--scales", default="large", help="comma-separated, or 'all'")
    p.add_argument("--staleness", default="all",
                   help=f"comma-separated, or 'all'; known: {', '.join(STALENESS)}")
    p.add_argument("--contents", default="all",
                   help=f"comma-separated, or 'all'; known: {', '.join(CONTENTS)}")
    p.add_argument("--recentering", default="residual",
                   choices=("residual", "none"),
                   help="warm_start_recentering. `none` is the pre-gh#606 "
                        "attribution control")
    p.add_argument("--tier", default="default",
                   choices=("default", "large", "all"))
    p.add_argument("--tol", type=float, default=1e-8)
    p.add_argument("--max-iter", type=int, default=500)
    p.add_argument("--out", default=None, help="JSON result path")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)

    families = _csv(args.families, list(REGISTRY), "family")
    if args.tier != "all" and args.families == "all":
        families = [f for f in families if REGISTRY[f].tier == args.tier]
    scales = _csv(args.scales, list(SCALES), "scale")
    staleness = _csv(args.staleness, STALENESS, "staleness")
    contents = _csv(args.contents, CONTENTS, "content")

    meta = {
        "git_sha": _git_sha(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "pounce": getattr(pounce, "__version__", "unknown"),
        "recentering": args.recentering,
        "tol": args.tol,
        "max_iter": args.max_iter,
        "corruption_var": CORRUPTION_VAR,
        "corruption_seed": CORRUPTION_SEED,
        "staleness": staleness,
        "contents": contents,
        "families": families,
        "scales": scales,
    }

    runs = []
    total = len(families) * len(scales) * len(staleness)
    done = 0
    for s in staleness:
        for family in families:
            for scale in scales:
                done += 1
                print(f"[{done}/{total}] {s} {family} @ {scale} ... ",
                      end="", flush=True)
                t0 = time.perf_counter()
                run = measure(family, scale, s, contents, args.recentering,
                              tol=args.tol, max_iter=args.max_iter)
                runs.append(run)
                print(f"{time.perf_counter() - t0:.1f}s "
                      + " ".join(f"{c}={run['modes'][c]['iters']}"
                                 for c in contents))
                if args.verbose:
                    for c in contents:
                        print(f"      {c:<8} {run['modes'][c]['warm_start']}")

    payload = {"meta": meta, "runs": runs}
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(payload, fh, indent=1)
        print(f"\nwrote {args.out}")
    print(summarize(payload))
    return 0


if __name__ == "__main__":
    sys.exit(main())
