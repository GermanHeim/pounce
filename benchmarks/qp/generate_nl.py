#!/usr/bin/env python3
"""Generate the Maros-Meszaros convex-QP suite as AMPL ``.nl`` files.

The Maros-Meszaros test set is the standard collection of ~138 convex
quadratic programs

    minimize    1/2 xᵀ P x + qᵀ x + r
    subject to  l_c <= C x <= u_c        (general linear rows)
                lb  <=  x  <= ub          (variable bounds)

curated by Maros & Meszaros (1997), *A Repository of Convex Quadratic
Programming Problems*, Imperial College tech. report DOC 97/6. They are
deliberately hard QPs (ill-conditioned, degenerate, large) and serve as
the canonical QP benchmark, analogous to the NETLIB set for LP.

The original distribution is QPS (MPS + a quadratic-objective section) on
Csaba Meszaros' FTP. That FTP is effectively dead (redirect loop / 404),
so this script pulls the well-known faithful mirror maintained by the
``qpsolvers`` project:

    https://github.com/qpsolvers/maros_meszaros_qpbenchmark  (data/*.mat)

Those ``.mat`` files were produced from the original SIF problems by
``sif2mat.m`` in ``proxqp_benchmark``; each is a MATLAB v5 file (read with
``scipy.io.loadmat``) holding the QP in the documented form

    P  (n×n sparse)   Hessian of the (1/2 xᵀPx) term
    q  (n)            linear objective coefficients
    r  (scalar)       constant objective offset
    A  (m×n sparse)   == vstack([C, eye(n)]) — the LAST n rows are the
                      variable-bound identity block, the rest are the
                      general linear rows C
    l, u (m)          double-sided bounds   l <= A x <= u
    n, m              dimensions

with the infinity constant set to 1e20. This is exactly the convention
used by ``qpbenchmark``'s own loader (``maros_meszaros.py``,
``load_problem_from_mat_file`` / ``convert_problem_from_double_sided``);
we reproduce it here and additionally keep the constant offset ``r`` so
the emitted objective matches the published Maros-Meszaros optimal value.

For each problem we build a Pyomo ``ConcreteModel`` with the exact same
math and write it with ``model.write('<NAME>.nl', format='nl')`` so the
suite runs through the standard ``.nl`` driver
(``benchmarks/scripts/run_nl_bench.sh``) like every other suite — no
compiled harness, no FFI.

Usage:
    python3 generate_nl.py                 # fetch (if needed) + convert all
    python3 generate_nl.py --out-dir nl    # output dir for .nl (default ./nl)
    python3 generate_nl.py --data-dir data # cache dir for .mat (default ./data)
    python3 generate_nl.py --no-fetch      # use already-downloaded .mat only
    python3 generate_nl.py QPTEST HS21     # only the named problems

The downloaded ``.mat`` files (under ``--data-dir``) and the generated
``.nl`` are regenerated locally and not tracked in git.
"""

from __future__ import annotations

import argparse
import os
import sys
import urllib.request

import numpy as np
import scipy.io as sio
import scipy.sparse as sp

from pyomo.environ import (
    ConcreteModel,
    Var,
    Objective,
    Constraint,
    Reals,
    minimize,
)

# Mirror of the Maros-Meszaros set as MATLAB .mat (qpsolvers project).
GITHUB_API = (
    "https://api.github.com/repos/qpsolvers/"
    "maros_meszaros_qpbenchmark/git/trees/main?recursive=1"
)
RAW_BASE = (
    "https://raw.githubusercontent.com/qpsolvers/"
    "maros_meszaros_qpbenchmark/main/"
)

INF = 1e20  # the .mat infinity sentinel (proxqp_benchmark sif2mat.m)


def list_mat_files() -> list[str]:
    """Return the names (e.g. ``QPTEST``) of every .mat in the mirror."""
    import json

    with urllib.request.urlopen(GITHUB_API, timeout=60) as resp:
        tree = json.load(resp)["tree"]
    names = [
        os.path.basename(t["path"])[:-4]
        for t in tree
        if t["path"].startswith("data/") and t["path"].endswith(".mat")
    ]
    return sorted(names)


def fetch_mat(name: str, data_dir: str) -> str:
    """Download data/<name>.mat into data_dir if absent; return its path."""
    path = os.path.join(data_dir, name + ".mat")
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return path
    url = RAW_BASE + "data/" + name + ".mat"
    with urllib.request.urlopen(url, timeout=120) as resp:
        blob = resp.read()
    with open(path, "wb") as f:
        f.write(blob)
    return path


def load_qp(path: str):
    """Parse one Maros-Meszaros .mat into (P, q, r, C, lc, uc, lb, ub).

    Mirrors qpbenchmark's load_problem_from_mat_file: A == [C; eye(n)], the
    last n rows being the variable-bound block. P is the FULL Hessian (so
    the objective is 1/2 xᵀ P x + qᵀ x + r). Infinities are the 1e20
    sentinel.
    """
    d = sio.loadmat(path)
    P = d["P"].astype(float).tocsc()
    q = d["q"].T.flatten().astype(float)
    r = float(np.asarray(d["r"]).flatten()[0]) if "r" in d else 0.0
    A = d["A"].astype(float).tocsr()
    l = d["l"].T.flatten().astype(float)
    u = d["u"].T.flatten().astype(float)
    n = int(np.asarray(d["n"]).flatten()[0])
    m = int(np.asarray(d["m"]).flatten()[0])
    assert A.shape == (m, n), f"{path}: A {A.shape} != ({m},{n})"

    # Variable bounds are the last n rows (the eye(n) block); the general
    # linear rows C are everything above them.
    lb = l[-n:].copy()
    ub = u[-n:].copy()
    C = A[: m - n].tocsr()
    lc = l[: m - n].copy()
    uc = u[: m - n].copy()
    return P, q, r, C, lc, uc, lb, ub


def build_model(name, P, q, r, C, lc, uc, lb, ub) -> ConcreteModel:
    """Build a Pyomo model for min 1/2 xᵀPx + qᵀx + r s.t. lc<=Cx<=uc, lb<=x<=ub."""
    n = q.shape[0]
    model = ConcreteModel(name=name)
    idx = list(range(n))

    def bounds(_m, i):
        lo = lb[i]
        hi = ub[i]
        return (
            None if lo <= -INF else float(lo),
            None if hi >= INF else float(hi),
        )

    def start(_m, i):
        # Clamp 0 into [lb, ub] so the initial point is feasible wrt bounds.
        lo = lb[i] if lb[i] > -INF else -INF
        hi = ub[i] if ub[i] < INF else INF
        return float(min(max(0.0, lo), hi))

    model.x = Var(idx, domain=Reals, bounds=bounds, initialize=start)
    x = model.x

    # Objective: 1/2 xᵀ P x + qᵀ x + r. P is the full (symmetric) Hessian,
    # so 1/2 P gives each unordered pair once with the right coefficient.
    P = P.tocoo()
    quad = 0.0
    for i, j, v in zip(P.row, P.col, P.data):
        if v != 0.0:
            quad += 0.5 * float(v) * x[int(i)] * x[int(j)]
    lin = sum(float(q[i]) * x[i] for i in idx if q[i] != 0.0)
    model.obj = Objective(expr=quad + lin + r, sense=minimize)

    # General linear rows lc <= C x <= uc, skipping empty rows.
    C = C.tocsr()
    rows = []
    for ri in range(C.shape[0]):
        s, e = C.indptr[ri], C.indptr[ri + 1]
        cols = C.indices[s:e]
        vals = C.data[s:e]
        lo, hi = lc[ri], uc[ri]
        if len(cols) == 0:
            continue
        rows.append((ri, cols, vals, lo, hi))

    if rows:
        model.row_idx = list(r[0] for r in rows)
        rowmap = {r[0]: r for r in rows}

        def con(_m, ri):
            _, cols, vals, lo, hi = rowmap[ri]
            expr = sum(float(v) * x[int(c)] for c, v in zip(cols, vals))
            lo_inf = lo <= -INF
            hi_inf = hi >= INF
            if not lo_inf and not hi_inf and abs(hi - lo) < 1e-12:
                return expr == float(lo)
            if lo_inf and hi_inf:
                return Constraint.Skip
            if lo_inf:
                return expr <= float(hi)
            if hi_inf:
                return float(lo) <= expr
            return (float(lo), expr, float(hi))

        model.c = Constraint(model.row_idx, rule=con)

    return model


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    here = os.path.dirname(os.path.abspath(__file__))
    p.add_argument("problems", nargs="*",
                   help="problem names to convert (default: all in the mirror)")
    p.add_argument("--out-dir", default=os.path.join(here, "nl"),
                   help="output directory for .nl files (default: ./nl)")
    p.add_argument("--data-dir", default=os.path.join(here, "data"),
                   help="cache directory for downloaded .mat (default: ./data)")
    p.add_argument("--no-fetch", action="store_true",
                   help="do not download; use .mat already in --data-dir")
    args = p.parse_args(argv)

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.data_dir, exist_ok=True)

    if args.problems:
        names = args.problems
    elif args.no_fetch:
        names = sorted(
            f[:-4] for f in os.listdir(args.data_dir) if f.endswith(".mat")
        )
    else:
        print("listing mirror ...")
        names = list_mat_files()
    print(f"{len(names)} problem(s) to process")

    ok = 0
    failed = []
    for name in names:
        try:
            if args.no_fetch:
                path = os.path.join(args.data_dir, name + ".mat")
                if not os.path.exists(path):
                    raise FileNotFoundError(path)
            else:
                path = fetch_mat(name, args.data_dir)
            P, q, r, C, lc, uc, lb, ub = load_qp(path)
            model = build_model(name, P, q, r, C, lc, uc, lb, ub)
            out = os.path.join(args.out_dir, name + ".nl")
            model.write(out, format="nl",
                        io_options={"symbolic_solver_labels": True})
            nvars = q.shape[0]
            ncons = C.shape[0]
            print(f"  wrote {name}.nl  (n={nvars}, general rows={ncons})")
            ok += 1
        except Exception as exc:  # noqa: BLE001 - report and continue
            failed.append((name, repr(exc)))
            print(f"  FAILED {name}: {exc!r}", file=sys.stderr)

    print(f"\n{ok}/{len(names)} converted; {len(failed)} failed")
    for name, err in failed:
        print(f"  - {name}: {err}")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
