#!/usr/bin/env python3
"""Mesh continuation on `laptime`: does a coarse solution warm-start a fine one?

The premise of a trajectory solve is that consecutive problems along a
refinement path are close, so the previous solution should be worth
something. Nothing in POUNCE can exploit that on its own: the solver sees a
flat vector and has no idea which entry is which stage's state. Mapping a
coarse solution onto a fine mesh is transcription knowledge, and therefore
the frontend's job. This script is that frontend, for one family.

The mapping is by **arc length**. `laptime` names its variables
`X[k,state]` (node states), `Xc[k,j,state]` (collocation states, `j` the
Radau point) and `U[k,ctrl]` (piecewise-constant controls), and a uniform
mesh puts each at a known fraction of the lap:

    X[k,st]     -> s = k / N
    Xc[k,j,st]  -> s = (k + radau[j-1]) / N
    U[k,c]      -> s = (k + 1/2) / N

So every quantity becomes a series in `s`, the coarse solution is
interpolated onto the fine mesh's `s` values, and the result is a starting
point. Node and collocation states of the same name are pooled into one
series, which is strictly more information than using nodes alone.

Only the **primal** point is transferred. That is the benchmark suite's
`values-ipm` arm — no multipliers, no barrier parameter — and is the
weakest of the warm starts it measures, chosen here because it needs
nothing from the `.sol` beyond `x`.

    python3 benchmarks/large_scale/mesh_continuation.py
"""

import re
import sys
import time

import numpy as np
import pounce

RADAU3 = [0.15505102572168222, 0.6449489742783178, 1.0]

_X = re.compile(r"^X\[(\d+),([^\]]+)\]$")
_XC = re.compile(r"^Xc\[(\d+),(\d+),([^\]]+)\]$")
_U = re.compile(r"^U\[(\d+),([^\]]+)\]$")


def coords(names, n_intervals):
    """(quantity, arc-length) for each variable, or None if unrecognised."""
    out = []
    for nm in names:
        m = _X.match(nm)
        if m:
            out.append((m.group(2), int(m.group(1)) / n_intervals))
            continue
        m = _XC.match(nm)
        if m:
            k, j, st = int(m.group(1)), int(m.group(2)), m.group(3)
            out.append((st, (k + RADAU3[j - 1]) / n_intervals))
            continue
        m = _U.match(nm)
        if m:
            out.append((m.group(2), (int(m.group(1)) + 0.5) / n_intervals))
            continue
        out.append(None)
    return out


def transfer(coarse_names, coarse_x, coarse_N, fine_names, fine_x0, fine_N):
    """Interpolate a coarse primal point onto the fine mesh, by arc length."""
    series = {}
    for c, v in zip(coords(coarse_names, coarse_N), coarse_x):
        if c is None:
            continue
        series.setdefault(c[0], []).append((c[1], v))
    for k in series:
        pts = sorted(series[k])
        series[k] = (np.array([p[0] for p in pts]), np.array([p[1] for p in pts]))

    out = np.array(fine_x0, dtype=float)
    hit = 0
    for i, c in enumerate(coords(fine_names, fine_N)):
        if c is None or c[0] not in series:
            continue
        s_arr, v_arr = series[c[0]]
        out[i] = float(np.interp(c[1], s_arr, v_arr))
        hit += 1
    return out, hit


def solve(prob, x0, opts, label):
    t0 = time.time()
    (x, info), = pounce.solve_nlp_batch([prob], x0s=[x0], options=opts, parallel=False)
    wall = time.time() - t0
    print(
        f"  {label:<34} {info.get('status_msg', info.get('status')):<34} "
        f"iters={info.get('iter_count', '?'):>5}  {wall:7.1f}s  "
        f"obj={info.get('obj_val', float('nan')):.9f}"
    )
    return x, info, wall


def main():
    meshes = [
        ("nl_0.08", 80),
        ("nl", 160),
        ("nl_0.32", 320),
    ]
    hess = sys.argv[1] if len(sys.argv) > 1 else "finite-difference"
    base = {"print_level": 0, "max_iter": 1200, "hessian_approximation": hess}
    # The Hessian-less case is the one this whole line of work is about, so
    # the continuation is measured there rather than on a model that has
    # second derivatives to fall back on.
    print(f"hessian_approximation={hess}\n")

    prev = None  # (names, x, N)
    for path, N in meshes:
        nlfile = f"benchmarks/large_scale/{path}/laptime.nl"
        try:
            prob = pounce.read_nl(nlfile)
        except Exception as exc:  # noqa: BLE001
            print(f"{path}: cannot read ({exc})")
            continue
        names = list(prob.var_names)
        print(f"N={N}  n={prob.n}  m={prob.m}  ({nlfile})")

        x_cold, info_cold, t_cold = solve(prob, None, base, "cold (.nl start)")

        if prev is not None:
            x_warm0, hit = transfer(prev[0], prev[1], prev[2], names, prob.x0, N)
            print(f"  transferred {hit}/{prob.n} entries from N={prev[2]}")
            x_warm, info_warm, t_warm = solve(prob, x_warm0, base, "warm (interpolated coarse)")
            ic, iw = info_cold.get("iter_count"), info_warm.get("iter_count")
            if ic and iw:
                print(f"  --> iterations {ic} -> {iw} ({ic / iw:.2f}x), "
                      f"wall {t_cold:.1f}s -> {t_warm:.1f}s ({t_cold / max(t_warm, 1e-9):.2f}x)")
            prev = (names, x_warm, N)
        else:
            prev = (names, x_cold, N)
        print()


if __name__ == "__main__":
    main()
