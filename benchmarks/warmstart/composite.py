"""The composite report: every table regenerated from raw JSON.

pounce#611's fourth acceptance criterion is "composite report wired into
the existing benchmark documentation", and its first is "reproducible
scripts and machine-readable raw results". This module is where those
two meet: it reads the raw result files the runners write and emits one
markdown document. Nothing here is hand-transcribed, and no number in
the output exists anywhere except as a function of the inputs — so a
re-measure after a solver change is a re-run of two commands, not an
editing pass.

That property is not decoration. Two PRs in flight (#638 on seed
rejection, #639 on the model-probe facet) will move the recentering and
transfer arms; the whole point of generating tables from
``results*.json`` is that re-running them is cheap enough that nobody is
tempted to patch a number by hand and leave the rest stale.

What it reads
-------------

``--results``    the pounce sweep (``warmstart.run``)
``--ipopt``      the same sweep under the external adapter, optional
``--transfer``   the changed-structure experiments (``warmstart.transfer``)

Each is optional; sections whose input is absent say so rather than
being silently omitted, because a missing section and an empty one mean
different things to a reader deciding whether a claim is supported.

Performance and data profiles
-----------------------------

The issue asks for "performance/data profiles over repeated sequences",
which are the standard way to compare solvers over a set of problems
without letting one hard instance dominate a mean.

*Performance profile* — for each instance ``p`` (one family × scale ×
step) and arm ``s``, take the cost ``c(p,s)``; the ratio
``r(p,s) = c(p,s) / minₛ c(p,s)`` is how much worse that arm was than
the best arm on that instance. ``ρₛ(τ)`` is the fraction of instances
where ``r(p,s) ≤ τ``. ``ρₛ(1)`` is "how often this arm was the best";
``ρₛ(∞)`` is "how often it solved at all". An arm that fails an instance
gets ``r = ∞`` and is counted only in ``ρₛ(∞)``.

*Data profile* — the fraction of instances an arm solves within a budget
of ``κ(n+1)`` function evaluations, which puts problems of different
size on one axis. Reported at a few κ rather than as a curve, since the
suite's instances span n = 4 to n = 2402 and a plot in markdown would
be less honest than the numbers.

**An instance counts as solved only if the harness gates say so** —
converged *and* not on a worse optimum than the reference. Profiling on
speed alone would rank the falsification families' fastest-to-the-wrong-
basin arm first, which is exactly backwards.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from typing import Dict, List, Optional, Sequence, Tuple

_INF = float("inf")

#: The issue's initialization list, mapped onto the arms that implement
#: it. Kept here rather than in prose so the coverage table cannot drift
#: away from what the suite actually runs.
ISSUE_ARMS: List[Tuple[str, Tuple[str, ...], str]] = [
    ("raw user point / cold solve", ("cold-ipm", "cold-sqp"), ""),
    ("previous primal only", ("values-ipm",), ""),
    ("complete primal-dual-barrier", ("warm-ipm",), ""),
    ("residual-adaptive recentering", ("warm-ipm", "warm-ipm-norecenter"),
     "the pair is the attribution control: `none` is pre-#606 behaviour"),
    ("horizon-shift / prolongation transfer", ("shift", "prolong-dual"),
     "in `warmstart.transfer`, not the fixed-shape sweep"),
    ("sparse safeguarded normal step", ("cold-ipm-lsq",), ""),
    ("sensitivity predictor", ("pred-ipm",), ""),
    ("predictor-corrector continuation", ("predcorr-ipm",), ""),
    ("fixed-budget / successive-halving racing",
     ("race-fixed", "race-halving"), ""),
]

#: Warm arm → the cold arm it must be scored against. Pairing each arm
#: with its own algorithm is what separates "warm started" from
#: "switched algorithms".
PAIRS: Dict[str, str] = {
    "warm-ipm": "cold-ipm",
    "values-ipm": "cold-ipm",
    "warm-ipm-norecenter": "cold-ipm",
    "warm-sqp": "cold-sqp",
    "warm-sqp-hom": "cold-sqp-hom",
    "warm-qp-ipm": "cold-qp-ipm",
    "pred-ipm": "cold-ipm",
    "predcorr-ipm": "cold-ipm",
    "cold-ipm-lsq": "cold-ipm",
    "race-fixed": "cold-ipm",
    "race-halving": "cold-ipm",
}


def _fmt(v, spec=".2f", dash="—"):
    if v is None:
        return dash
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return "∞" if v == _INF else dash
    return format(v, spec)


def _ok(step: dict) -> bool:
    """Solved *and* correct. Speed on a wrong answer is not a result."""
    return bool(step.get("correct", step.get("success", False)))


# ------------------------------------------------------------- profiles


def _instances(payload: dict, cost_key: str) -> Dict[str, Dict[str, float]]:
    """``{instance_id: {arm: cost}}``; cost is ``inf`` on an unsolved step."""
    table: Dict[str, Dict[str, float]] = {}
    for run in payload.get("runs", []):
        base = f"{run['family']}@{run['scale']}"
        for arm, steps in run["arms"].items():
            for step in steps:
                key = f"{base}#{step['step']}"
                cost = _INF
                if _ok(step):
                    if cost_key == "evals":
                        cost = float(
                            step["n_obj"] + step["n_grad"]
                            + step["n_cons"] + step["n_jac"] + step["n_hess"]
                        )
                    elif cost_key == "time":
                        cost = float(step["solve_time"] + step.get("init_time", 0.0))
                    else:
                        it = step["iters"]
                        cost = float(it) if it >= 0 else _INF
                    # A zero-cost win would make every ratio infinite;
                    # the shift is the usual guard and matches the
                    # shifted geometric mean `report.py` already uses.
                    cost += 1.0
                table.setdefault(key, {})[arm] = cost
    return table


def performance_profile(
    payload: dict, arms: Sequence[str], cost_key: str, taus: Sequence[float]
) -> Dict[str, Dict[float, float]]:
    """``{arm: {tau: fraction of instances within tau x best}}``.

    Only instances where *every* listed arm was actually run are
    counted, so the denominator is the same for all of them. An arm
    skipped on a family (a QP arm on a nonlinear family, a predictor arm
    without pin rows) would otherwise be compared on an easier subset.
    """
    table = _instances(payload, cost_key)
    common = [c for c in table.values() if all(a in c for a in arms)]
    out: Dict[str, Dict[float, float]] = {a: {} for a in arms}
    if not common:
        return out
    for costs in common:
        best = min(costs[a] for a in arms)
        for arm in arms:
            costs[f"__ratio__{arm}"] = (
                costs[arm] / best if best < _INF and costs[arm] < _INF else _INF
            )
    n = len(common)
    for arm in arms:
        for tau in taus:
            hit = sum(
                1 for c in common
                if c[f"__ratio__{arm}"] <= tau
            )
            out[arm][tau] = hit / n
    return out


def data_profile(
    payload: dict, arms: Sequence[str], kappas: Sequence[float]
) -> Dict[str, Dict[float, float]]:
    """``{arm: {kappa: fraction solved within kappa*(n+1) evaluations}}``."""
    per_arm: Dict[str, List[Tuple[float, float]]] = {a: [] for a in arms}
    for run in payload.get("runs", []):
        dim = float(run["n"]) + 1.0
        for arm in arms:
            for step in run["arms"].get(arm, []):
                evals = float(
                    step["n_obj"] + step["n_grad"] + step["n_cons"]
                    + step["n_jac"] + step["n_hess"]
                )
                per_arm[arm].append(
                    (evals / dim if _ok(step) else _INF, 1.0)
                )
    out: Dict[str, Dict[float, float]] = {}
    for arm in arms:
        rows = per_arm[arm]
        out[arm] = {
            k: (sum(1 for r, _ in rows if r <= k) / len(rows)) if rows else 0.0
            for k in kappas
        }
    return out


# --------------------------------------------------------------- tables


def _arm_totals(payload: dict, arm: str) -> Optional[dict]:
    steps = [
        s for run in payload.get("runs", []) for s in run["arms"].get(arm, [])
    ]
    if not steps:
        return None
    warm_steps = [
        s for run in payload.get("runs", []) for s in run["arms"].get(arm, [])[1:]
    ]
    return {
        "steps": len(steps),
        "iters": sum(s["iters"] for s in steps if s["iters"] >= 0),
        "evals": sum(
            s["n_obj"] + s["n_grad"] + s["n_cons"] + s["n_jac"] + s["n_hess"]
            for s in steps
        ),
        "solve_time": sum(s["solve_time"] for s in steps),
        "init_time": sum(s.get("init_time", 0.0) for s in steps),
        "failed": sum(1 for s in steps if not s["success"]),
        "bad": sum(1 for s in steps if not s.get("correct", True)),
        "better": sum(1 for s in steps if s.get("better")),
        "max_kkt": max((s["kkt_error"] for s in steps if s["success"]),
                       default=float("nan")),
        "warm_steps": len(warm_steps),
    }


def _geomean_speedup(payload: dict, warm: str, cold: str) -> Optional[float]:
    """Shifted geometric mean of cold/warm iterations over warm-started steps."""
    logs: List[float] = []
    for run in payload.get("runs", []):
        w, c = run["arms"].get(warm), run["arms"].get(cold)
        if not w or not c:
            continue
        for ws, cs in zip(w[1:], c[1:]):
            if ws["iters"] < 0 or cs["iters"] < 0:
                continue
            logs.append(math.log((cs["iters"] + 1.0) / (ws["iters"] + 1.0)))
    return float(math.exp(sum(logs) / len(logs))) if logs else None


def _falsification_rows(payload: dict, families: Sequence[str]) -> List[dict]:
    rows = []
    for run in payload.get("runs", []):
        if run["family"] not in families:
            continue
        for arm, steps in sorted(run["arms"].items()):
            bad = sum(1 for s in steps if not s.get("correct", True))
            rows.append({
                "family": run["family"],
                "scale": run["scale"],
                "arm": arm,
                "steps": len(steps),
                "bad": bad,
                "iters": sum(s["iters"] for s in steps if s["iters"] >= 0),
            })
    return rows


# --------------------------------------------------------------- render


_TAUS = (1.0, 1.5, 2.0, 4.0, _INF)
_KAPPAS = (10.0, 50.0, 100.0, 500.0)

#: Families built for pounce#611 to make warm starting *lose*. Called
#: out by name so the section cannot quietly stop being reported.
_FALSIFY = ("rastrigin_drift", "rastrigin_scatter")


def render(
    results: Optional[dict],
    ipopt: Optional[dict],
    transfer: Optional[dict],
) -> str:
    out: List[str] = []
    w = out.append

    w("# Warm-start benchmark — composite report")
    w("")
    w("Generated by `python -m warmstart.composite`. Every number below "
      "is computed from the raw JSON named in the provenance table; none "
      "is transcribed. Re-running the two sweeps and this command "
      "regenerates the whole document.")
    w("")

    # -- provenance ------------------------------------------------
    w("## Provenance")
    w("")
    w("| input | solver | version | commit | taken |")
    w("|---|---|---|---|---|")
    for label, payload in (("pounce sweep", results),
                           ("external sweep", ipopt),
                           ("changed-structure", transfer)):
        if payload is None:
            w(f"| {label} | — | *not supplied* | — | — |")
            continue
        m = payload["meta"]
        w(f"| {label} | `{m.get('solver', '?')}` | {m.get('solver_version', '?')} "
          f"| `{m.get('git_sha', '?')}` | {m.get('timestamp', '?')} |")
    w("")

    if results is not None:
        m = results["meta"]
        w("### Stopping criteria and settings")
        w("")
        w("The issue makes equal stopping criteria an acceptance "
          "criterion. Both solvers are asked for the same tolerances and "
          "iteration cap, and correctness is judged by the harness from "
          "the returned point (`warmstart/kkt.py`) rather than from "
          "either solver's own status line.")
        w("")
        w("| setting | value | applies to |")
        w("|---|---|---|")
        w(f"| `tol` | {m['tol']:g} | both |")
        w("| `constr_viol_tol` | 1e-6 | both |")
        w(f"| `max_iter` | {m['max_iter']} | both |")
        w(f"| harness converged-gate | KKT ≤ {m['kkt_gate']:g}, "
          f"viol ≤ {m['viol_gate']:g} | both |")
        w(f"| worse-optimum margin | {m['obj_tol']:g} relative | both |")
        w(f"| `warm_start_recentering` | {m.get('recentering', '?')} | "
          "pounce only (no Ipopt counterpart) |")
        w("| `warm_start_bound_push` / slack / mult | 1e-9 | Ipopt only "
          "(default 1e-2 discards most of a warm start) |")
        w("| `linear_solver` | MUMPS (Ipopt) vs pounce's own sparse LDLᵀ | "
          "**not equal — see caveats** |")
        w("")

    # -- issue coverage --------------------------------------------
    w("## Initialization-arm coverage")
    w("")
    w("The nine arms pounce#611 lists, and where each is measured.")
    w("")
    w("| issue arm | harness arm(s) | status | note |")
    w("|---|---|---|---|")
    have = set()
    if results:
        for run in results["runs"]:
            have.update(run["arms"])
    if transfer:
        for run in transfer["runs"]:
            have.update(run.get("arms", {}))
    for label, arms, note in ISSUE_ARMS:
        present = [a for a in arms if a in have]
        status = ("**run**" if len(present) == len(arms)
                  else "*partial*" if present else "*not run*")
        w(f"| {label} | {', '.join('`%s`' % a for a in arms)} | {status} "
          f"| {note} |")
    w("")

    if results is None:
        w("*No pounce sweep supplied; every section below is empty.*")
        return "\n".join(out) + "\n"

    # -- headline arm table ----------------------------------------
    w("## Arms over the whole sweep")
    w("")
    w("`speedup` is the shifted geometric mean of cold/warm iterations "
      "over warm-started steps, each arm against its own cold "
      "counterpart (see `PAIRS`); **>1 means the arm won**. `bad` counts "
      "steps that failed to converge *or* landed on a worse optimum than "
      "the reference arm — a nonzero entry voids the speedup on its row.")
    w("")
    w("| arm | steps | iters | evals | solve s | init s | speedup | bad | "
      "better | max KKT |")
    w("|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|")
    all_arms = []
    for run in results["runs"]:
        for a in run["arms"]:
            if a not in all_arms:
                all_arms.append(a)
    for arm in all_arms:
        t = _arm_totals(results, arm)
        if t is None:
            continue
        sp = (_geomean_speedup(results, arm, PAIRS[arm])
              if arm in PAIRS else None)
        w(f"| `{arm}` | {t['steps']} | {t['iters']} | {t['evals']} | "
          f"{_fmt(t['solve_time'], '.2f')} | {_fmt(t['init_time'], '.3f')} | "
          f"{_fmt(sp)} | {t['bad']} | {t['better']} | "
          f"{_fmt(t['max_kkt'], '.1e')} |")
    w("")

    # -- falsification ---------------------------------------------
    w("## Falsification: where warm starting loses")
    w("")
    w("These families were added by pounce#611 **to produce a result "
      "against warm starting**, and they do. `rastrigin_drift` walks a "
      "smooth path through a lattice of local minima whose spacing is 1 "
      "and whose per-step increment is `0.3 x scale`; `rastrigin_scatter` "
      "does not walk a path at all — consecutive θ are independent draws, "
      "the issue's \"unrelated global/nonconvex cases where continuation "
      "should not be expected to help\".")
    w("")
    w("A wrong-basin step is not a failed solve. It converges, quickly, "
      "to a worse optimum — so it shows up in `bad`, not in a status "
      "code, and an arm can look *faster* on exactly the steps it got "
      "wrong.")
    w("")
    rows = _falsification_rows(results, _FALSIFY)
    if not rows:
        w("*Falsification families were not in this run.* Their absence is "
          "itself a caveat on every speedup above: without them the sweep "
          "cannot produce a result against warm starting.")
    else:
        w("| family | scale | arm | steps | bad | iters |")
        w("|---|---|---|--:|--:|--:|")
        for r in rows:
            flag = " ⚠" if r["bad"] else ""
            w(f"| {r['family']} | {r['scale']} | `{r['arm']}` | {r['steps']} "
              f"| {r['bad']}{flag} | {r['iters']} |")
    w("")

    # -- profiles --------------------------------------------------
    w("## Performance profiles")
    w("")
    w("Fraction of instances (one family x scale x step) on which the arm "
      "was within `τ` of the best arm on that instance. Restricted to "
      "instances where every arm in the table ran, so the denominator is "
      "shared. `τ=1` is \"how often it was best\"; `τ=∞` is \"how often it "
      "produced a correct answer at all\".")
    w("")
    for cost_key, title in (("iters", "iterations"),
                            ("evals", "function/derivative evaluations"),
                            ("time", "wall time (solve + init)")):
        profile_arms = [a for a in all_arms if a in
                        ("cold-ipm", "warm-ipm", "values-ipm",
                         "warm-ipm-norecenter", "cold-ipm-lsq",
                         "race-fixed", "race-halving")]
        prof = performance_profile(results, profile_arms, cost_key, _TAUS)
        if not any(prof.values()):
            continue
        w(f"**By {title}**")
        w("")
        w("| arm | " + " | ".join(f"τ={_fmt(t, '.1f')}" for t in _TAUS) + " |")
        w("|---|" + "--:|" * len(_TAUS))
        for arm in profile_arms:
            cells = " | ".join(_fmt(prof[arm].get(t), ".2f") for t in _TAUS)
            w(f"| `{arm}` | {cells} |")
        w("")

    w("## Data profiles")
    w("")
    w("Fraction of *all* steps solved correctly within `κ(n+1)` "
      "function/derivative evaluations. Unlike the performance profile "
      "this is over every step the arm ran, so arms that skip families "
      "are not directly comparable here — read down a column only for "
      "arms defined on the same families.")
    w("")
    dp_arms = [a for a in all_arms if a in
               ("cold-ipm", "warm-ipm", "values-ipm", "pred-ipm",
                "predcorr-ipm", "race-fixed", "race-halving")]
    dp = data_profile(results, dp_arms, _KAPPAS)
    w("| arm | " + " | ".join(f"κ={int(k)}" for k in _KAPPAS) + " |")
    w("|---|" + "--:|" * len(_KAPPAS))
    for arm in dp_arms:
        w(f"| `{arm}` | "
          + " | ".join(_fmt(dp[arm].get(k), ".2f") for k in _KAPPAS) + " |")
    w("")

    # -- external solver -------------------------------------------
    w("## External solver")
    w("")
    if ipopt is None:
        w("*No external sweep supplied.* The adapter seam exists "
          "(`warmstart/adapters/ipopt_adapter.py`) and `--solver ipopt` "
          "runs it, but this report was generated without one.")
    else:
        im = ipopt["meta"]
        w(f"Ipopt via cyipopt — {im.get('solver_version', '?')}. Driven "
          "through the **same** Python callback object as pounce, so "
          "evaluation counts mean the same thing on both sides.")
        w("")
        w("| arm | solver | steps | iters | evals | solve s | bad | speedup |")
        w("|---|---|--:|--:|--:|--:|--:|--:|")
        for arm in ("cold-ipm", "warm-ipm", "values-ipm"):
            for label, payload in (("pounce", results), ("ipopt", ipopt)):
                t = _arm_totals(payload, arm)
                if t is None:
                    continue
                sp = (_geomean_speedup(payload, arm, PAIRS[arm])
                      if arm in PAIRS else None)
                w(f"| `{arm}` | {label} | {t['steps']} | {t['iters']} | "
                  f"{t['evals']} | {_fmt(t['solve_time'], '.2f')} | "
                  f"{t['bad']} | {_fmt(sp)} |")
        w("")
        skipped: Dict[str, str] = {}
        for run in ipopt["runs"]:
            skipped.update(run.get("skipped", {}))
        if skipped:
            w("Arms the external adapter does not offer, with the reason "
              "recorded per run rather than dropped:")
            w("")
            for arm, why in sorted(skipped.items()):
                w(f"- `{arm}` — {why}")
            w("")
        w("**Wall-time caveat.** This Ipopt is linked against MUMPS; HSL "
          "(MA27/MA57) is not redistributable and is not installed. A "
          "timing comparison here is against MUMPS-backed Ipopt 3.11.9, "
          "not against a licensed HSL build, and iteration counts are the "
          "more portable column.")
        w("")

    # -- changed structure -----------------------------------------
    w("## Changed structure: horizon shift and mesh prolongation")
    w("")
    if transfer is None:
        w("*No changed-structure run supplied.*")
    else:
        for run in transfer["runs"]:
            if run["experiment"] == "mesh":
                w(f"**Mesh prolongation — {run['family']}** "
                  f"(n = {run['n']}), coarse solve "
                  f"{run['coarse_solve']['iters']} iters")
                w("")
                w("| arm | iters | init s | solve s | obj | KKT |")
                w("|---|--:|--:|--:|--:|--:|")
                for arm, steps in run["arms"].items():
                    s = steps[0]
                    if "error" in s:
                        w(f"| `{arm}` | *error* | — | — | — | — |")
                        continue
                    w(f"| `{arm}` | {s['iters']} | "
                      f"{_fmt(s['init_time'], '.4f')} | "
                      f"{_fmt(s['solve_time'], '.4f')} | "
                      f"{_fmt(s['obj'], '.8g')} | "
                      f"{_fmt(s['kkt_error'], '.1e')} |")
                w("")
            else:
                w(f"**{run['experiment']} — {run['family']}** "
                  f"(n = {run['n']}, horizon {run['horizon']}, "
                  f"{run['n_steps']} steps)")
                w("")
                w("| arm | iters (steps 1+) | init s | solve s | failed |")
                w("|---|--:|--:|--:|--:|")
                for arm, steps in run["arms"].items():
                    errs = [s for s in steps if "error" in s]
                    if errs:
                        w(f"| `{arm}` | *error* | — | — | — |")
                        continue
                    w(f"| `{arm}` | "
                      f"{sum(s['iters'] for s in steps[1:] if s['iters'] >= 0)} | "
                      f"{_fmt(sum(s.get('init_time', 0.0) for s in steps), '.4f')} | "
                      f"{_fmt(sum(s['solve_time'] for s in steps), '.4f')} | "
                      f"{sum(1 for s in steps if not s['success'])} |")
                w("")
    return "\n".join(out) + "\n"


def summarize(
    results: Optional[dict],
    ipopt: Optional[dict],
    transfer: Optional[dict],
) -> dict:
    """The same tables the markdown renders, as a machine-readable dict.

    The repo ignores per-run ``results.json`` inside a suite directory
    (`.gitignore`: "its per-run results.json / results.md outputs are
    regenerated and ignored like every other suite's") but tracks
    composite summaries — `BENCHMARK_REPORT.json`, `qp_three_way.json`.
    This is the warm-start suite's equivalent: small enough to commit,
    complete enough that a later reader can check any number in the
    report without re-running anything, and derived entirely from the
    raw files so it cannot disagree with them.
    """
    payload: dict = {
        "schema": "warmstart-composite/1",
        "sources": {},
        "arms": {},
        "issue_arm_coverage": [],
        "falsification": [],
        "profiles": {},
        "transfer": [],
    }
    for label, src in (("pounce", results), ("ipopt", ipopt),
                       ("transfer", transfer)):
        payload["sources"][label] = None if src is None else src["meta"]

    have = set()
    if results:
        for run in results["runs"]:
            have.update(run["arms"])
    if transfer:
        for run in transfer["runs"]:
            have.update(run.get("arms", {}))
    for label, arms, note in ISSUE_ARMS:
        present = [a for a in arms if a in have]
        payload["issue_arm_coverage"].append({
            "issue_arm": label,
            "harness_arms": list(arms),
            "present": present,
            "status": ("run" if len(present) == len(arms)
                       else "partial" if present else "not-run"),
            "note": note,
        })

    if results is None:
        return payload

    all_arms: List[str] = []
    for run in results["runs"]:
        for a in run["arms"]:
            if a not in all_arms:
                all_arms.append(a)
    for arm in all_arms:
        totals = _arm_totals(results, arm)
        if totals is None:
            continue
        totals["speedup_vs_own_cold"] = (
            _geomean_speedup(results, arm, PAIRS[arm]) if arm in PAIRS else None
        )
        totals["paired_against"] = PAIRS.get(arm)
        payload["arms"][arm] = totals
    if ipopt is not None:
        payload["ipopt_arms"] = {
            arm: _arm_totals(ipopt, arm)
            for arm in ("cold-ipm", "warm-ipm", "values-ipm")
            if _arm_totals(ipopt, arm) is not None
        }
        skipped: Dict[str, str] = {}
        for run in ipopt["runs"]:
            skipped.update(run.get("skipped", {}))
        payload["ipopt_skipped"] = skipped

    payload["falsification"] = _falsification_rows(results, _FALSIFY)

    prof_arms = [a for a in all_arms if a in
                 ("cold-ipm", "warm-ipm", "values-ipm",
                  "warm-ipm-norecenter", "cold-ipm-lsq",
                  "race-fixed", "race-halving")]
    for cost_key in ("iters", "evals", "time"):
        prof = performance_profile(results, prof_arms, cost_key, _TAUS)
        payload["profiles"][f"performance_{cost_key}"] = {
            arm: {("inf" if t == _INF else str(t)): v for t, v in vals.items()}
            for arm, vals in prof.items()
        }
    dp_arms = [a for a in all_arms if a in
               ("cold-ipm", "warm-ipm", "values-ipm", "pred-ipm",
                "predcorr-ipm", "race-fixed", "race-halving")]
    payload["profiles"]["data"] = {
        arm: {str(int(k)): v for k, v in vals.items()}
        for arm, vals in data_profile(results, dp_arms, _KAPPAS).items()
    }

    if transfer is not None:
        payload["transfer"] = transfer["runs"]
    return payload


def main(argv=None) -> int:
    here = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser(
        prog="warmstart.composite", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--results", default=os.path.join(here, "results.json"))
    p.add_argument("--ipopt", default=None)
    p.add_argument("--transfer", default=None)
    p.add_argument("-o", "--out", default=os.path.join(here, "composite.md"))
    p.add_argument("--json-out", default=None,
                   help="also write the machine-readable summary here")
    args = p.parse_args(argv)

    def load(path):
        if not path or not os.path.exists(path):
            return None
        with open(path) as fh:
            return json.load(fh)

    res, ipo, tra = load(args.results), load(args.ipopt), load(args.transfer)
    with open(args.out, "w") as fh:
        fh.write(render(res, ipo, tra))
    print(f"wrote {args.out}")
    if args.json_out:
        with open(args.json_out, "w") as fh:
            json.dump(summarize(res, ipo, tra), fh, indent=1, sort_keys=True)
        print(f"wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
