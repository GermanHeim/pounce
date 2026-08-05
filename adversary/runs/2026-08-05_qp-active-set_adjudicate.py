"""Independent adjudicator for the elastic-certificate fuzz.

Re-decides feasibility of every generated instance with
``scipy.optimize.linprog`` (HiGHS), which has never heard of pounce. Its
job is to catch a bug in the *generator* before any conclusion is drawn
about the solver: a witness that satisfies my own arithmetic but not an
independent LP solver means my instances are wrong, not pounce.

Feasibility of a QP's constraint set is a pure LP question — the
objective plays no part — so this is an exact reformulation, not an
approximation:

    find x  s.t.  bl <= A x <= bu,  xl <= x <= xu

Reports, per instance: the constructive truth, scipy's verdict, and the
conditioning that decides whether a disagreement is a real defect or an
arithmetically unreachable tolerance.

Usage:  python 2026-08-05_qp-active-set_adjudicate.py <instances.jsonl> [seed ...]
"""

import json
import sys

import numpy as np
from scipy.optimize import linprog

INF = 1e19


def adjudicate(rec):
    n, m = rec["n"], rec["m"]
    A = np.array(rec["a"], dtype=float).reshape(m, n)
    bl = np.array(rec["bl"], dtype=float)
    bu = np.array(rec["bu"], dtype=float)
    xl = np.array(rec["xl"], dtype=float)
    xu = np.array(rec["xu"], dtype=float)

    lb = np.where(bl <= -INF, -np.inf, bl)
    ub = np.where(bu >= INF, np.inf, bu)

    # Pure feasibility: zero objective. Two-sided rows go in as a pair of
    # one-sided `A_ub` rows, which every linprog version accepts.
    ub_rows = []
    ub_rhs = []
    eq_rows = []
    eq_rhs = []
    for i in range(m):
        if bl[i] == bu[i]:
            eq_rows.append(A[i])
            eq_rhs.append(bl[i])
            continue
        if bu[i] < INF:
            ub_rows.append(A[i])
            ub_rhs.append(bu[i])
        if bl[i] > -INF:
            ub_rows.append(-A[i])
            ub_rhs.append(-bl[i])

    return linprog(
        c=np.zeros(n),
        A_ub=np.array(ub_rows) if ub_rows else None,
        b_ub=np.array(ub_rhs) if ub_rows else None,
        A_eq=np.array(eq_rows) if eq_rows else None,
        b_eq=np.array(eq_rhs) if eq_rows else None,
        bounds=list(zip(xl, xu)),
        method="highs",
    )


def conditioning(rec):
    n, m = rec["n"], rec["m"]
    A = np.array(rec["a"], dtype=float).reshape(m, n)
    bl = np.array(rec["bl"], dtype=float)
    bu = np.array(rec["bu"], dtype=float)
    finite = np.concatenate(
        [bl[np.abs(bl) < INF], bu[np.abs(bu) < INF]]
    )
    row_norms = np.linalg.norm(A, np.inf, axis=1)
    # How tight is the tightest two-sided/equality row, relative to its
    # own magnitude? A row of scale 1e6 asked to hold to an absolute
    # 1e-9 is asking for 1e-15 relative — the f64 noise floor.
    tightest_rel = np.inf
    for i in range(m):
        if abs(bl[i]) < INF and abs(bu[i]) < INF:
            width = bu[i] - bl[i]
            scale = max(row_norms[i], abs(bl[i]), 1.0)
            tightest_rel = min(tightest_rel, width / scale)
    return {
        "max_|A|": float(np.abs(A).max()) if A.size else 0.0,
        "row_scale_spread": float(row_norms.max() / max(row_norms.min(), 1e-300))
        if m
        else 1.0,
        "max_|b|": float(np.abs(finite).max()) if finite.size else 0.0,
        "tightest_row_rel_width": float(tightest_rel),
        "cond(A)": float(np.linalg.cond(A)) if m and n else float("nan"),
    }


def main():
    path = sys.argv[1]
    wanted = set(int(s) for s in sys.argv[2:])

    agree = disagree = 0
    rows = []
    with open(path) as fh:
        for line in fh:
            rec = json.loads(line)
            if wanted and rec["seed"] not in wanted:
                continue
            res = adjudicate(rec)
            scipy_feasible = res.status == 0
            truth_feasible = rec["truth"] == "Feasible"
            ok = scipy_feasible == truth_feasible
            agree += ok
            disagree += not ok
            if wanted or not ok:
                rows.append((rec, scipy_feasible, ok, res))

    for rec, scipy_feasible, ok, res in rows:
        print(f"--- seed={rec['seed']} kind={rec['kind']}")
        print(f"    constructive truth : {rec['truth']}  ({rec['proof']})")
        print(f"    scipy/HiGHS        : {'FEASIBLE' if scipy_feasible else 'INFEASIBLE'}"
              f"  (status={res.status}, {res.message.strip()[:60]})")
        print(f"    generator agrees   : {'yes' if ok else 'NO — generator is wrong'}")
        c = conditioning(rec)
        print("    conditioning       : " + ", ".join(f"{k}={v:.3e}" for k, v in c.items()))
        if scipy_feasible and res.x is not None:
            A = np.array(rec["a"], float).reshape(rec["m"], rec["n"])
            ax = A @ res.x
            bl = np.array(rec["bl"], float)
            bu = np.array(rec["bu"], float)
            slack_lo = np.where(bl > -INF, ax - bl, np.inf)
            slack_hi = np.where(bu < INF, bu - ax, np.inf)
            print(f"    scipy point slack  : min={min(slack_lo.min(), slack_hi.min()):.3e}")

    print()
    print(f"instances adjudicated: {agree + disagree}")
    print(f"generator confirmed by scipy: {agree}")
    print(f"generator contradicted by scipy: {disagree}")
    print("VERDICT: " + ("GENERATOR SOUND" if disagree == 0 else "GENERATOR BUG"))
    return 0 if disagree == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
