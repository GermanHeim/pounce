#!/usr/bin/env python3
"""Fixed-budget vs adaptive successive-halving racing (pounce#610).

The harness behind ``race_starts``' acceptance criterion 4: on a
multi-basin benchmark set, does the halving ladder *match or improve* the
fixed-budget answer at materially lower total evaluations?

It measures the whole workflow the docs prescribe — race, then continue
the winner warm at full effort — and counts every call into the user
callables, for every candidate including the ones the race discards. That
last part is why the counting is done here rather than read off the
result: the pre-#610 policy threw away every candidate outside ``top``
and so could not report its own spend.

Needs no benchmark data: the models are closures, so this runs anywhere
pounce imports.

    python benchmarks/scripts/race_starts_bench.py
    python benchmarks/scripts/race_starts_bench.py --grid 8:10 16:20 27:40
    python benchmarks/scripts/race_starts_bench.py --json out.json
"""

from __future__ import annotations

import argparse
import json
import math

import numpy as np

import pounce


# ---------------------------------------------------------------------------
# Counting
# ---------------------------------------------------------------------------
class Counter:
    """Counts calls into the user callables, across every candidate."""

    def __init__(self):
        self.n = 0

    def wrap(self, fn):
        if fn is None:
            return None

        def inner(x):
            self.n += 1
            return fn(x)

        return inner

    def wrap_cons(self, cons):
        if not cons:
            return cons
        out = []
        for c in cons:
            d = dict(c)
            d["fun"] = self.wrap(d["fun"])
            if d.get("jac") is not None:
                d["jac"] = self.wrap(d["jac"])
            out.append(d)
        return out


# ---------------------------------------------------------------------------
# The benchmark set: multi-basin models where *which* start the full solve
# continues from decides the answer.
# ---------------------------------------------------------------------------
def double_well():
    def f(x):
        return float((x[0] ** 2 - 1.0) ** 2 + 0.25 * (x[0] + 1.0))

    def g(x):
        return np.array([4.0 * x[0] * (x[0] ** 2 - 1.0) + 0.25])

    return "double_well", f, g, [(-3.0, 3.0)], None


def himmelblau_disc():
    def f(x):
        a = x[0] ** 2 + x[1] - 11.0
        b = x[0] + x[1] ** 2 - 7.0
        return float(a * a + b * b + 0.7 * x[0] + 0.4 * x[1])

    def g(x):
        a = x[0] ** 2 + x[1] - 11.0
        b = x[0] + x[1] ** 2 - 7.0
        return np.array([4.0 * a * x[0] + 2.0 * b + 0.7,
                         2.0 * a + 4.0 * b * x[1] + 0.4])

    cons = [dict(type="ineq",
                 fun=lambda x: np.array([26.0 - x[0] ** 2 - x[1] ** 2]),
                 jac=lambda x: np.array([[-2.0 * x[0], -2.0 * x[1]]]))]
    return ("himmelblau_disc", f, g, [(-5.0, 5.0), (-5.0, 5.0)], cons)


def six_hump_camel():
    def f(x):
        u, v = x[0], x[1]
        return float((4.0 - 2.1 * u ** 2 + u ** 4 / 3.0) * u ** 2
                     + u * v + (-4.0 + 4.0 * v ** 2) * v ** 2)

    def g(x):
        u, v = x[0], x[1]
        return np.array([(8.0 - 8.4 * u ** 2 + 2.0 * u ** 4) * u + v,
                         u + (-8.0 + 16.0 * v ** 2) * v])

    cons = [dict(type="ineq",
                 fun=lambda x: np.array([x[0] + x[1] + 2.0]),
                 jac=lambda x: np.array([[1.0, 1.0]]))]
    return ("six_hump_camel", f, g, [(-2.0, 2.0), (-1.5, 1.5)], cons)


def rastrigin_eq():
    def f(x):
        return float(20.0 + sum(xi ** 2 - 10.0 * math.cos(2.0 * math.pi * xi)
                                for xi in x))

    def g(x):
        return np.array([2.0 * xi + 20.0 * math.pi * math.sin(2.0 * math.pi * xi)
                         for xi in x])

    cons = [dict(type="eq",
                 fun=lambda x: np.array([x[0] + x[1] - 1.0]),
                 jac=lambda x: np.array([[1.0, 1.0]]))]
    return ("rastrigin_eq", f, g, [(-4.0, 4.0), (-4.0, 4.0)], cons)


def hs71():
    def f(x):
        return float(x[0] * x[3] * (x[0] + x[1] + x[2]) + x[2])

    def g(x):
        return np.array([x[0] * x[3] + x[3] * (x[0] + x[1] + x[2]),
                         x[0] * x[3], x[0] * x[3] + 1.0,
                         x[0] * (x[0] + x[1] + x[2])])

    cons = [
        dict(type="ineq",
             fun=lambda x: np.array([x[0] * x[1] * x[2] * x[3] - 25.0]),
             jac=lambda x: np.array([[x[1] * x[2] * x[3], x[0] * x[2] * x[3],
                                      x[0] * x[1] * x[3], x[0] * x[1] * x[2]]])),
        dict(type="eq",
             fun=lambda x: np.array([sum(xi ** 2 for xi in x) - 40.0]),
             jac=lambda x: np.array([[2.0 * x[0], 2.0 * x[1], 2.0 * x[2],
                                      2.0 * x[3]]])),
    ]
    return "hs71", f, g, [(1.0, 5.0)] * 4, cons


def deceptive_circle():
    """The adversarial model: off the constraint manifold the objective is
    unbounded below, so the starts with the best early objective are the
    ones headed for the wrong basin."""

    def f(x):
        return float(x[0] ** 3 - 3.0 * x[0] * x[1] ** 2 + 0.3 * x[1])

    def g(x):
        return np.array([3.0 * x[0] ** 2 - 3.0 * x[1] ** 2,
                         -6.0 * x[0] * x[1] + 0.3])

    cons = [dict(type="eq",
                 fun=lambda x: np.array([x[0] ** 2 + x[1] ** 2 - 4.0]),
                 jac=lambda x: np.array([[2.0 * x[0], 2.0 * x[1]]]))]
    return ("deceptive_circle", f, g, [(-3.0, 3.0), (-3.0, 3.0)], cons)


SUITE = [double_well, himmelblau_disc, six_hump_camel, rastrigin_eq, hs71,
         deceptive_circle]


# ---------------------------------------------------------------------------
def run_case(builder, n_starts, iters, policy, seed=0):
    name, f, g, bounds, cons = builder()
    starts = pounce.generate_starts(n_starts, bounds=bounds, seed=seed)
    c = Counter()
    best, rep = pounce.race_starts(
        c.wrap(f), starts, jac=c.wrap(g), bounds=bounds,
        constraints=c.wrap_cons(cons), iters=iters, policy=policy,
        return_report=True,
    )
    race_evals = c.n
    w = best[0]
    ws = pounce.WarmStart.from_info(w.x, w.info)
    fin = pounce.minimize(c.wrap(f), w.x, jac=c.wrap(g), bounds=bounds,
                          constraints=c.wrap_cons(cons), warm_start=ws)
    viol = float(fin.info.get("final_constr_viol", 0.0))
    return {
        "case": name, "policy": policy, "n_starts": n_starts, "iters": iters,
        "race_user_evals": race_evals, "total_user_evals": c.n,
        "race_solver_evals": rep.total_evals,
        "race_iters": rep.total_iters, "rungs": rep.n_rounds,
        "resumes": rep.n_resumes,
        "obj": float(fin.fun), "viol": viol,
        "feasible": bool(viol <= 1e-6),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--grid", nargs="*", default=["8:10", "16:20", "27:40"],
                    help="n_starts:iters pairs (default: 8:10 16:20 27:40)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", help="also write the raw rows here")
    args = ap.parse_args()
    grid = [tuple(int(v) for v in s.split(":")) for s in args.grid]

    rows = []
    print(f"{'case':<18}{'n':>4}{'it':>4}  "
          f"{'fixed evals':>12}{'halving evals':>14}{'Δ%':>7}  "
          f"{'fixed iters':>12}{'halving iters':>14}  quality")
    grand = {"fixed": 0, "halving": 0}
    regressions = []
    for n_starts, iters in grid:
        for builder in SUITE:
            got = {p: run_case(builder, n_starts, iters, p, args.seed)
                   for p in ("fixed", "halving")}
            rows.extend(got.values())
            fx, hv = got["fixed"], got["halving"]
            grand["fixed"] += fx["total_user_evals"]
            grand["halving"] += hv["total_user_evals"]
            worse = (not hv["feasible"] and fx["feasible"]) or (
                hv["obj"] > fx["obj"] + 1e-6 * max(1.0, abs(fx["obj"])))
            if worse:
                regressions.append((fx["case"], n_starts, iters,
                                    fx["obj"], hv["obj"]))
            delta = 100.0 * (fx["total_user_evals"] - hv["total_user_evals"]) \
                / max(1, fx["total_user_evals"])
            print(f"{fx['case']:<18}{n_starts:>4}{iters:>4}  "
                  f"{fx['total_user_evals']:>12}{hv['total_user_evals']:>14}"
                  f"{delta:>+7.1f}  "
                  f"{fx['race_iters']:>12}{hv['race_iters']:>14}  "
                  f"{'WORSE' if worse else 'same-or-better'}")

    saved = 100.0 * (grand["fixed"] - grand["halving"]) / max(1, grand["fixed"])
    print()
    print(f"TOTAL user evaluations: fixed={grand['fixed']} "
          f"halving={grand['halving']}  ({saved:+.1f}%)")
    print(f"quality regressions: {len(regressions)}")
    for r in regressions:
        print(f"  {r[0]} n={r[1]} iters={r[2]}: fixed {r[3]:.8g} -> "
              f"halving {r[4]:.8g}")
    if args.json:
        with open(args.json, "w") as fh:
            json.dump({"rows": rows, "totals": grand,
                       "regressions": regressions}, fh, indent=1)


if __name__ == "__main__":
    main()
