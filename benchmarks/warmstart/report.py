"""Render a results JSON into markdown.

The headline number is the **shifted geometric mean iteration speedup**
of a warm arm over its own cold counterpart (``warm-sqp`` vs
``cold-sqp``, ``warm-ipm`` vs ``cold-ipm``), taken over the steps that
actually warm-start — step 0 has nothing to warm from and is excluded
from the ratio, though it is still in the totals.

Pairing each warm arm with its *own* algorithm is deliberate: it
isolates the warm start. Comparing ``warm-sqp`` against ``cold-ipm``
mixes the warm start with the algorithm switch, so that column is
reported too, but separately.

Two columns matter as much as the speedup:

``worse``
    Steps where the warm arm took strictly more iterations than the
    cold one. Warm starting is not free — a start inside the wrong
    active set can cost more than starting fresh — and a benchmark
    that only reports the mean hides it.
``bad``
    Steps that failed to converge or converged somewhere other than
    the reference solution. Any nonzero entry here voids the speedup
    on that row; a fast wrong answer is not a result.
"""

from __future__ import annotations

import math
import statistics
from typing import List, Optional

_SHIFT = 1.0  # shifted geometric mean, the usual guard against 0-iteration steps


def _geomean_ratio(cold: List[int], warm: List[int]) -> Optional[float]:
    """Shifted geometric mean of cold/warm iteration counts (>1 = warm wins)."""
    pairs = [(c, w) for c, w in zip(cold, warm) if c >= 0 and w >= 0]
    if not pairs:
        return None
    logs = [math.log((c + _SHIFT) / (w + _SHIFT)) for c, w in pairs]
    return float(math.exp(sum(logs) / len(logs)))


def _fmt(v, spec=".2f", dash="—"):
    return dash if v is None else format(v, spec)


def _arm_summary(steps: List[dict]) -> dict:
    iters = [s["iters"] for s in steps]
    return {
        "steps": len(steps),
        "qp_solves": sum(s["n_qp_solves"] or 0 for s in steps),
        "qp_ws_changes": sum(s["n_qp_ws_changes"] or 0 for s in steps),
        "iters_total": sum(i for i in iters if i >= 0),
        "iters_median": statistics.median(iters) if iters else 0,
        "time_total": sum(s["solve_time"] for s in steps),
        "obj_evals": sum(s["n_obj"] for s in steps),
        "hess_evals": sum(s["n_hess"] for s in steps),
        "failed": sum(1 for s in steps if not s["success"]),
        "bad": sum(1 for s in steps if not s.get("correct", True)),
        "max_kkt": max((s["kkt_error"] for s in steps), default=float("nan")),
        "better": sum(1 for s in steps if s.get("better")),
        "n_active": [s["n_active"] for s in steps if s["n_active"] is not None],
        "ws_changed": [
            s["ws_changed"] for s in steps if s.get("ws_changed") is not None
        ],
    }


def _pair_stats(run: dict, warm_arm: str, cold_arm: str) -> Optional[dict]:
    arms = run["arms"]
    if warm_arm not in arms or cold_arm not in arms:
        return None
    warm, cold = arms[warm_arm], arms[cold_arm]
    # Step 0 of a warm arm is a cold solve — excluded from the ratio.
    w_it = [s["iters"] for s in warm[1:]]
    c_it = [s["iters"] for s in cold[1:]]
    w_qp = [s["n_qp_ws_changes"] for s in warm[1:] if s["n_qp_ws_changes"] is not None]
    c_qp = [s["n_qp_ws_changes"] for s in cold[1:] if s["n_qp_ws_changes"] is not None]
    return {
        "speedup": _geomean_ratio(c_it, w_it),
        "qp_speedup": _geomean_ratio(c_qp, w_qp) if len(w_qp) == len(c_qp) else None,
        "qp_warm": sum(w_qp),
        "qp_cold": sum(c_qp),
        # `worse` tracks whichever metric this pairing is judged on:
        # inner active-set work when the path has QP subproblems (the
        # SQP pairing), outer iterations otherwise (the IPM pairing).
        "worse": (
            sum(1 for c, w in zip(c_qp, w_qp) if w > c)
            if len(w_qp) == len(c_qp) and w_qp
            else sum(1 for c, w in zip(c_it, w_it) if w > c)
        ),
        "compared": len(w_it),
        "iters_warm": sum(w_it),
        "iters_cold": sum(c_it),
    }


def render(payload: dict) -> str:
    meta = payload["meta"]
    runs = payload["runs"]
    out: List[str] = []
    w = out.append

    w("# Warm-start benchmark")
    w("")
    w(f"- solver: **{meta['solver']}** {meta.get('solver_version', '')}")
    w(f"- git: `{meta.get('git_sha', '?')}`  ")
    w(f"- run: {meta.get('timestamp', '?')} on {meta.get('platform', '?')}")
    w(
        f"- tol: {meta['tol']:g} · converged-gate: KKT ≤ {meta['kkt_gate']:g}, "
        f"viol ≤ {meta['viol_gate']:g} · worse-optimum margin: "
        f"{meta['obj_tol']:g} · max_iter: {meta['max_iter']}"
    )
    w("")
    w(
        "Both speedups are shifted geometric means of cold/warm counts over "
        "the warm-started steps (step 0 excluded — nothing to warm from), so "
        "**>1 means warm starting won**."
    )
    w("")
    w(
        "- **SQP outer** — outer SQP iterations. On a family whose "
        "subproblem is already a QP this is 1 either way, so the column is "
        "flat by construction and says nothing.\n"
        "- **SQP active-set work** — active-set changes (adds + drops) "
        "inside the QP subproblems, with the raw cold→warm totals in "
        "parentheses. This is where a working-set warm start actually "
        "pays, and it is the column to read.\n"
        "- **worse** — steps where the warm arm cost *more* than the cold "
        "one on the metric in the column to its left. Warm starting is not "
        "free: a start inside the wrong active set can cost more than "
        "starting fresh.\n"
        "- **bad** — steps that did not converge (status, KKT residual or "
        "feasibility), or converged to a *worse* optimum than the reference "
        "arm. **A row with a nonzero `bad` has no valid speedup.** Finding "
        "a *better* optimum than the reference is not counted bad — on the "
        "nonconvex families that happens, and it is listed separately."
    )
    w("")

    # -- headline ---------------------------------------------------
    w("## Iteration speedup from warm starting")
    w("")
    w(
        "| family | regime | scale | SQP outer | SQP active-set work | "
        "worse | bad | IPM cold→warm | worse | bad |"
    )
    w("|---|---|---|--:|--:|--:|--:|--:|--:|--:|")
    for run in runs:
        sqp = _pair_stats(run, "warm-sqp", "cold-sqp")
        ipm = _pair_stats(run, "warm-ipm", "cold-ipm")
        cross = _pair_stats(run, "warm-sqp", "cold-ipm")
        arms = run["arms"]
        bad_sqp = sum(
            1 for s in arms.get("warm-sqp", []) if not s.get("correct", True)
        )
        bad_ipm = sum(
            1 for s in arms.get("warm-ipm", []) if not s.get("correct", True)
        )
        qp_detail = (
            f" ({sqp['qp_cold']}→{sqp['qp_warm']})"
            if sqp and sqp.get("qp_speedup") is not None
            else ""
        )
        w(
            f"| `{run['family']}` | {run['tags'].get('regime', '')} "
            f"| {run['scale']} "
            f"| {_fmt(sqp and sqp['speedup'], '.2f')}× "
            f"| {_fmt(sqp and sqp.get('qp_speedup'), '.2f')}×{qp_detail} "
            f"| {sqp['worse'] if sqp else '—'} | {bad_sqp} "
            f"| {_fmt(ipm and ipm['speedup'], '.2f')}× "
            f"| {ipm['worse'] if ipm else '—'} | {bad_ipm} |"
        )
    w("")

    # -- active-set behavior ---------------------------------------
    w("## Active-set behavior along each path")
    w("")
    w(
        "From the SQP arms' returned working sets. `churn` is the mean "
        "Hamming distance between consecutive steps' working sets — the "
        "property that decides whether warm starting can pay at all."
    )
    w("")
    w("| family | scale | n | m | mean &#124;A&#124; | churn/step | max churn |")
    w("|---|---|--:|--:|--:|--:|--:|")
    for run in runs:
        steps = run["arms"].get("cold-sqp") or run["arms"].get("warm-sqp")
        if not steps:
            continue
        s = _arm_summary(steps)
        churn = s["ws_changed"]
        w(
            f"| `{run['family']}` | {run['scale']} | {run['n']} | {run['m']} "
            f"| {_fmt(statistics.mean(s['n_active']) if s['n_active'] else None, '.1f')} "
            f"| {_fmt(statistics.mean(churn) if churn else None, '.2f')} "
            f"| {max(churn) if churn else '—'} |"
        )
    w("")

    # -- per-arm detail --------------------------------------------
    w("## Per-arm totals")
    w("")
    w(
        "Wall time includes the Python callback round trip, which dominates "
        "at these problem sizes — read iterations and evaluation counts as "
        "the primary measurements and time only as a cross-check."
    )
    w("")
    for run in runs:
        w(
            f"### `{run['family']}` @ {run['scale']} "
            f"({', '.join(f'{k}={v}' for k, v in run['tags'].items())}; "
            f"n={run['n']}, m={run['m']}, {run['n_steps']} steps)"
        )
        w("")
        w(
            "| arm | Σ iters | median | Σ QP solves | Σ QP active-set changes "
            "| Σ time (s) | f-evals | failed | bad | better | max KKT |"
        )
        w("|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|")
        for arm in (
            "cold-ipm", "warm-ipm", "cold-sqp", "warm-sqp",
            "cold-sqp-hom", "warm-sqp-hom", "cold-qp-ipm", "warm-qp-ipm",
        ):
            steps = run["arms"].get(arm)
            if not steps:
                continue
            s = _arm_summary(steps)
            qp = "—" if not s["qp_solves"] else str(s["qp_solves"])
            qp_ws = "—" if not s["qp_solves"] else str(s["qp_ws_changes"])
            w(
                f"| {arm} | {s['iters_total']} | {s['iters_median']:g} "
                f"| {qp} | {qp_ws} "
                f"| {s['time_total']:.3f} | {s['obj_evals']} "
                f"| {s['failed']} | {s['bad']} | {s['better']} "
                f"| {s['max_kkt']:.1e} |"
            )
        for arm, why in run.get("skipped", {}).items():
            w(f"| {arm} | *skipped* | | | | | | | | | {why} |")
        w("")

    # -- parametric homotopy vs the conventional inner solve ---------
    # The cold pair is what this section is about; the warm pair is a
    # secondary column. A narrowed `--arms` — `cold-sqp,cold-sqp-hom` is
    # the reproduction in dev-notes/warm-start-benchmark.md — leaves the
    # warm twins absent, so they are looked up defensively below rather
    # than indexed.
    hom_runs = [
        r
        for r in runs
        if "cold-sqp-hom" in r["arms"] and "cold-sqp" in r["arms"]
    ]
    if hom_runs:
        w("## Parametric homotopy vs the conventional inner QP")
        w("")
        w(
            "`-hom` arms set `sqp_qp_use_homotopy`, which replaces the inner "
            "QP's **cold** phase-1/phase-2 solve with the §4.2 parametric "
            "homotopy (box-only relaxation, then the row bounds tightened "
            "along `t ∈ [0,1]`). Everything else about the arm is identical to "
            "its twin, so the difference is that one option. Warm inner QPs "
            "mostly skip the cold path, which is why the warm columns move "
            "less than the cold ones."
        )
        w("")
        w(
            "| family | scale | cold: conventional → homotopy | ratio "
            "| warm: conventional → homotopy | ratio |"
        )
        w("|---|---|--:|--:|--:|--:|")
        tot = {"cc": 0, "ch": 0, "wc": 0, "wh": 0}
        for run in hom_runs:
            a = run["arms"]

            def ws(arm):
                if arm not in a:
                    return None
                return sum(s["n_qp_ws_changes"] or 0 for s in a[arm])

            cc, ch, wc, wh = (
                ws("cold-sqp"), ws("cold-sqp-hom"), ws("warm-sqp"), ws("warm-sqp-hom")
            )
            for k, v in (("cc", cc), ("ch", ch), ("wc", wc), ("wh", wh)):
                if v is not None:
                    tot[k] += v
            w(
                f"| `{run['family']}` | {run['scale']} "
                f"| {_fmt(cc, 'd')} → {_fmt(ch, 'd')} "
                f"| {_fmt(cc / ch if ch else None, '.2f')}× "
                f"| {_fmt(wc, 'd')} → {_fmt(wh, 'd')} "
                f"| {_fmt(wc / wh if wc is not None and wh else None, '.2f')}× |"
            )
        w(
            f"| **total** | | **{tot['cc']} → {tot['ch']}** "
            f"| **{_fmt(tot['cc'] / tot['ch'] if tot['ch'] else None, '.2f')}×** "
            f"| **{_fmt(tot['wc'], 'd')} → {_fmt(tot['wh'], 'd')}** "
            f"| **{_fmt(tot['wc'] / tot['wh'] if tot['wh'] else None, '.2f')}×** |"
            if tot["ch"]
            else ""
        )
        w("")
        w(
            "Above 1.00× the homotopy did less inner active-set work; below "
            "it, more. A flat 1.00× means the option changed nothing on that "
            "family — its inner QPs never take the cold path far enough for "
            "the homotopy to matter."
        )
        w("")

    # -- three-way on the QP-shaped families ------------------------
    qp_runs = [r for r in runs if "cold-qp-ipm" in r["arms"]]
    if qp_runs:
        w("## Dedicated convex QP solver vs the two general paths")
        w("")
        w(
            "Only for families whose instances are literally QPs — the "
            "others skip these arms with a reason. Read wall time with the "
            "asymmetry in mind: the QP arms are handed the problem in "
            "matrix form once per step, while the general paths call back "
            "into the model at every iteration. Iteration counts are not "
            "comparable across *methods* either (an active-set pivot and "
            "an interior-point step are different units of work); the "
            "column that compares like with like is each arm against "
            "itself, cold vs warm."
        )
        w("")
        w(
            "| family | scale | qp-ipm cold→warm iters | nlp-ipm cold→warm "
            "| sqp cold→warm (QP active-set) | qp-ipm time (ms) cold→warm "
            "| fastest warm arm |"
        )
        w("|---|---|--:|--:|--:|--:|---|")
        for run in qp_runs:
            a = run["arms"]
            def tot(arm, key="iters"):
                return sum(s[key] for s in a[arm]) if arm in a else None
            def ms(arm):
                return sum(s["solve_time"] for s in a[arm]) * 1e3 if arm in a else None
            qp_pair = _pair_stats(run, "warm-qp-ipm", "cold-qp-ipm")
            sqp_pair = _pair_stats(run, "warm-sqp", "cold-sqp")
            warm_times = {
                arm: ms(arm)
                for arm in ("warm-ipm", "warm-sqp", "warm-qp-ipm")
                if ms(arm) is not None
            }
            fastest = min(warm_times, key=warm_times.get) if warm_times else "—"
            w(
                f"| `{run['family']}` | {run['scale']} "
                f"| {tot('cold-qp-ipm')}→{tot('warm-qp-ipm')} "
                f"({_fmt(qp_pair and qp_pair['speedup'], '.2f')}×) "
                f"| {tot('cold-ipm')}→{tot('warm-ipm')} "
                f"| {sqp_pair['qp_cold'] if sqp_pair else '—'}→"
                f"{sqp_pair['qp_warm'] if sqp_pair else '—'} "
                f"({_fmt(sqp_pair and sqp_pair.get('qp_speedup'), '.2f')}×) "
                f"| {_fmt(ms('cold-qp-ipm'), '.1f')}→{_fmt(ms('warm-qp-ipm'), '.1f')} "
                f"| {fastest} |"
            )
        w("")

    # -- regressions ------------------------------------------------
    regressions = []
    for run in runs:
        for warm_arm, cold_arm in (("warm-sqp", "cold-sqp"), ("warm-ipm", "cold-ipm")):
            arms = run["arms"]
            if warm_arm not in arms or cold_arm not in arms:
                continue
            for cw, ww in zip(arms[cold_arm][1:], arms[warm_arm][1:]):
                key = (
                    "n_qp_ws_changes"
                    if ww.get("n_qp_ws_changes") is not None
                    and cw.get("n_qp_ws_changes") is not None
                    else "iters"
                )
                if ww[key] > cw[key]:
                    regressions.append(
                        (run["family"], run["scale"], warm_arm, ww["step"],
                         "QP active-set changes" if key != "iters" else "outer iters",
                         cw[key], ww[key], ww.get("ws_changed"))
                    )
    w("## Where warm starting cost more than it saved")
    w("")
    if not regressions:
        w("No step in this run was slower warm than cold.")
    else:
        w("| family | scale | arm | step | metric | cold | warm | churn |")
        w("|---|---|---|--:|---|--:|--:|--:|")
        for r in regressions:
            w(
                f"| `{r[0]}` | {r[1]} | {r[2]} | {r[3]} | {r[4]} | {r[5]} "
                f"| {r[6]} | {r[7] if r[7] is not None else '—'} |"
            )
    w("")

    # -- correctness ------------------------------------------------
    bad_rows = []
    for run in runs:
        for arm, steps in run["arms"].items():
            for s in steps:
                if not s.get("correct", True):
                    reason = (
                        "did not converge"
                        if not s.get("converged", True)
                        else "worse optimum"
                    )
                    bad_rows.append(
                        (run["family"], run["scale"], arm, s["step"],
                         reason, s["status_msg"], s["kkt_error"],
                         s.get("obj_err"))
                    )
    w("## Correctness failures")
    w("")
    if not bad_rows:
        w(
            "Every step of every arm converged (KKT ≤ "
            f"{meta['kkt_gate']:g}, violation ≤ {meta['viol_gate']:g}) and "
            "none landed on a worse optimum than the reference arm."
        )
    else:
        w("| family | scale | arm | step | why | status | KKT | Δobj rel |")
        w("|---|---|---|--:|---|---|--:|--:|")
        for r in bad_rows:
            w(
                f"| `{r[0]}` | {r[1]} | {r[2]} | {r[3]} | {r[4]} | {r[5]} "
                f"| {_fmt(r[6], '.1e')} | {_fmt(r[7], '.2e')} |"
            )
    w("")
    return "\n".join(out)


def main(argv=None) -> int:
    import argparse
    import json
    import os

    p = argparse.ArgumentParser(description="render a warm-start results JSON")
    here = os.path.dirname(os.path.abspath(__file__))
    p.add_argument("results", nargs="?", default=os.path.join(here, "results.json"))
    p.add_argument("-o", "--out", default=None)
    args = p.parse_args(argv)

    with open(args.results) as fh:
        payload = json.load(fh)
    text = render(payload)
    if args.out:
        with open(args.out, "w") as fh:
            fh.write(text)
        print(f"wrote {args.out}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
