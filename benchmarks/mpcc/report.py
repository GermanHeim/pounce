"""Markdown rendering of a result file.

The report is organised around the two questions gh#794 asks it to
answer -- which route is supported, and where the boundary is -- and
around one rule: **a source-level number and an NLP number never share
a column.** POUNCE's ``final_constr_viol`` on a Scholtes stage is a
residual against ``G*H <= tau``; putting it next to the source
complementarity product in the same table would invite exactly the
conflation the gate exists to prevent.

The triage table is mechanical and says so. It applies four rules to
each observation and labels the *candidate* owner; assigning ownership
for real needs the kill-switch controls and a reproducer, which is what
the rules' `evidence` column points at.
"""

from __future__ import annotations

import io
from typing import Dict, List, Optional

from . import cases as C
from . import routes as R


def _fmt(v, prec=3, dash="--"):
    if v is None:
        return dash
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, float):
        if v != v:
            return "nan"
        if v == 0:
            return "0"
        return f"{v:.{prec}e}"
    return str(v)


def _worst_source(rec) -> float:
    s = rec["source"]
    if not s:
        return float("nan")
    return max(s["row_viol"], s["bound_viol"], s["sign_viol"], s["compl_max"])


def triage(
    rec,
    case_expected: dict,
    siblings: List[dict],
    tau_min: Optional[float] = None,
    feas_tol: Optional[float] = None,
) -> Dict[str, str]:
    """Mechanical candidate-owner label for one observation.

    Four rules, in the order they are applied:

    1. The MPCC is infeasible or has no S-stationary point, and the
       route reported that faithfully -> **source formulation**: there
       is nothing for a solver to do differently.
    2. The route reported success at a point that is not source-feasible
       -> **lowering/harness**: the reformulation, its schedule, or this
       harness is at fault before the solver is.
    3. The route failed where another route solved the same case from
       the same start -> **POUNCE candidate**: a difference between two
       configurations of the same solver on the same model is the only
       shape of evidence that can become a POUNCE issue, and it still
       needs the kill-switch controls to survive.
    3b. The route converged, but the record's own scaled and unscaled
       KKT errors are three orders apart and the point is not stationary
       in the model's units -> **scaling**. POUNCE converged the problem
       it internally scaled, and reports the gap itself in
       ``final_unscaled_*`` (gh#173); the stationarity classifier,
       working in the user's units, is agreeing with that figure, not
       contradicting it. This rule comes before the stationarity rules
       below because otherwise every such cell is misfiled as a solver
       defect.
    3c. The route reported success at an objective below the MPCC's
       optimum, at a point whose source residual is *inside* the
       solver's own feasibility tolerance -> **complementarity tolerance
       floor**. `G*H` is quadratically flat at the corner, so a residual
       of `eps` permits an excursion of order `sqrt(eps)` along the pair
       and the objective follows it. Nothing is hidden and no solver
       setting removes it; only a tighter `tol` moves it.
    4. The route reported success at an objective **below** the MPCC's
       own optimum, at a point that is *not* inside the feasibility
       tolerance. That can only happen at a point the source model
       does not admit, and which lowering it came through decides what
       it means: through ``scholtes`` it is the **relaxation limit**,
       the method working as designed and the reason a relaxed
       objective is not an MPCC objective; through an exact-product
       lowering it is a **POUNCE candidate**, because that lowering's
       feasible set *is* the MPCC's.
    4b. A ``scholtes`` route whose returned point still carries
       complementarity at the schedule floor -> **relaxation limit**
       again, and for a sharper reason than 4: the point is not
       MPCC-feasible at all, so *no* MPCC stationarity class applies to
       it and the classifier's "none" is a statement about the
       relaxation rather than about the solver.
    5. Everything else that failed -> **unassigned**, pending a
       reproducer. gh#794's issue-splitting rule says an unassigned gap
       does not become a feature issue.
    """
    if not rec["ok"]:
        solved = [
            s
            for s in siblings
            if s["ok"] and s["start"] == rec["start"] and s["scaling"] == rec["scaling"]
        ]
        if case_expected.get("feasible") is False:
            return {
                "owner": "source formulation",
                "why": "the MPCC is infeasible; a failure here is the correct answer",
                "evidence": "manifest expected.feasible = false",
            }
        if solved:
            return {
                "owner": "POUNCE candidate",
                "why": f"{solved[0]['route']} solved the same case from the same start",
                "evidence": "re-run with --controls all before filing",
            }
        return {
            "owner": "unassigned",
            "why": "no configuration solved this cell",
            "evidence": "needs a minimal reproducer before it becomes an issue",
        }
    worst = _worst_source(rec)
    if worst > 1e-5 and rec["lowering"] != "scholtes":
        return {
            "owner": "lowering/harness",
            "why": f"reported success at source residual {worst:.1e}",
            "evidence": "the exact-product lowering's feasible set is the MPCC's",
        }
    fstar = case_expected.get("obj")
    if (
        fstar is not None
        and rec["obj"] is not None
        and rec["obj"] < fstar - 1e-6
        and feas_tol is not None
        and worst <= 100.0 * feas_tol
    ):
        # The point satisfies the complementarity condition to the
        # tolerance the solve was asked for, and the objective is still
        # below the optimum. That is not a residual being hidden -- the
        # residual is reported and is inside tol -- it is the geometry of
        # a complementarity constraint: `G*H` is quadratically flat at
        # the corner, so a residual of `eps` buys an excursion of order
        # `sqrt(eps)` along the pair, and the objective follows.
        #
        # Measured on this corpus: `ralph1` at compl 2.6e-09 is 5.07e-05
        # below f*, against sqrt(2.6e-09) = 5.1e-05. `ralph2`, whose
        # relaxed optimum is linear in the residual rather than square
        # root, sits at -4.25e-10 against a residual of 2.13e-10.
        #
        # It is the single most important number for Gate 1: at
        # tol = 1e-8 an MPCC objective is only good to about 1e-4, and
        # no solver setting changes that.
        gap = fstar - rec["obj"]
        return {
            "owner": "complementarity tolerance floor",
            "why": (
                f"objective {gap:.1e} below f* at a point whose source residual is "
                f"{worst:.1e}, inside tol -- the sqrt-flatness of G*H at the corner, "
                "not a hidden violation"
            ),
            "evidence": "tighten tol to move it; nothing else will",
        }
    if fstar is not None and rec["obj"] is not None and rec["obj"] < fstar - 1e-6:
        if rec["lowering"] == "scholtes":
            return {
                "owner": "relaxation limit",
                "why": (
                    f"objective {rec['obj']:.3e} is below the MPCC optimum "
                    f"{fstar:.3e}: the point satisfies G*H <= tau, not G*H = 0"
                ),
                "evidence": f"source compl_max {worst:.1e} at the schedule floor",
            }
        return {
            "owner": "POUNCE candidate",
            "why": (
                f"reported {rec['status_msg']} at objective {rec['obj']:.3e}, below "
                f"the MPCC optimum {fstar:.3e}, through a lowering whose feasible "
                "set is the MPCC's"
            ),
            "evidence": f"source compl_max {worst:.1e}; re-run with --controls all",
        }
    klass = (rec.get("stationarity") or {}).get("klass")
    if (
        rec["lowering"] == "scholtes"
        and tau_min
        and rec["source"].get("compl_max", 0.0) > 0.01 * tau_min
    ):
        return {
            "owner": "relaxation limit",
            "why": (
                f"source complementarity {rec['source']['compl_max']:.1e} sits at the "
                f"schedule floor tau={tau_min:g}: the point is feasible for "
                "G*H <= tau and not for the MPCC, so no MPCC stationarity class "
                "applies to it"
            ),
            "evidence": "drive the schedule further, or use an exact-product lowering",
        }
    nlp = rec.get("nlp") or {}
    sc, un = nlp.get("final_kkt_error"), nlp.get("final_unscaled_kkt_error")
    if (
        klass in ("C", "W", "none")
        and sc is not None
        and un is not None
        and sc == sc
        and un == un
        and un > 1e3 * max(sc, 1e-300)
    ):
        return {
            "owner": "scaling",
            "why": (
                f"converged on the internally scaled NLP (kkt {sc:.1e}) while its "
                f"own unscaled KKT error is {un:.1e}; the MPCC stationarity "
                "residual is measured in the user's units and says the same thing"
            ),
            "evidence": "the record's nlp block; re-run with --controls no_scaling",
        }
    if klass in ("C", "W", "none") and case_expected.get("stationarity") in ("C", "W", "M"):
        return {
            "owner": "source formulation",
            "why": f"the MPCC's own solution is only {case_expected['stationarity']}-stationary",
            "evidence": "manifest expected.stationarity",
        }
    if klass in ("C", "W", "none"):
        # Same corner effect as rule 3c, read off the class instead of
        # the objective. Inside the feasibility tolerance the pair sits
        # `sqrt(tol)` from the corner and the MPCC multipliers it
        # generates are of that size; S and C differ only in their
        # *signs*, so at a multiplier of 3e-05 against a corner band of
        # 1e-04 the class is not resolved by the data. Measured:
        # `ralph2` under the ℓ₁ routes stops at G = H = 1.46e-05 with
        # nu = w = -2.9e-05, which reads C — correctly, and without
        # meaning the solver did anything wrong.
        mults = (rec.get("stationarity") or {}).get("multipliers") or {}
        pair_mult = max(
            (abs(v) for k, v in mults.items() if k.startswith(("nu[", "w["))),
            default=0.0,
        )
        if (
            feas_tol is not None
            and worst <= 100.0 * feas_tol
            and pair_mult <= feas_tol**0.5
        ):
            return {
                "owner": "complementarity tolerance floor",
                "why": (
                    f"class {klass} rests on MPCC multipliers of size {pair_mult:.1e} at a "
                    f"source residual of {worst:.1e}; inside the sqrt(tol) corner band "
                    "their signs are not resolved by the data"
                ),
                "evidence": "tighten tol to resolve the class; nothing else will",
            }
        return {
            "owner": "POUNCE candidate",
            "why": f"stopped at a {klass}-stationary point where the MPCC has an S-stationary one",
            "evidence": "re-run with --controls all before filing",
        }
    return {"owner": "-", "why": "converged to an S/M-stationary point", "evidence": "-"}


def render(payload: dict) -> str:
    out = io.StringIO()
    w = out.write
    st = payload["stamp"]
    cfg = payload["config"]
    recs = payload["records"]

    w("# MPCC benchmark (gh#794, Gate 0 of gh#776)\n\n")
    w(
        "Every number below is stamped with the commits, corpus revision and\n"
        "configuration in the next table. Source-level quantities and POUNCE's\n"
        "own NLP diagnostics are reported in separate columns throughout: an\n"
        "MPCC lowering's NLP residual is a residual of a different problem, and\n"
        "reading one as the other is the specific mistake this gate exists to\n"
        "prevent.\n\n"
    )

    w("## Provenance\n\n")
    w("| field | value |\n|---|---|\n")
    p = st["repositories"]["pounce"]
    w(f"| pounce commit | `{p.get('describe', p.get('commit'))}` |\n")
    d = st["repositories"]["discopt"]
    w(
        f"| discopt | {'present, ' + str(d.get('version')) if d['present'] else 'absent -- ' + d.get('reason', '')} |\n"
    )
    w(f"| model-data revision | `{st['model_data_revision']}` |\n")
    for k, v in st["environment"].items():
        w(f"| {k} | {v} |\n")
    cc = st.get("ccopt", {})
    w(
        f"| CCOpt oracle | available: {cc.get('available')}, pin `{cc.get('pin')}`, "
        f"comparison run: {cc.get('comparison_run')} |\n"
    )
    w(f"| mode | {cfg['mode']} |\n")
    w(f"| base options | `{cfg['base_options']}` |\n")
    w(f"| tau schedule | `{cfg['tau_schedule']}` |\n")
    w(f"| restart ladder | `{cfg['restart_ladder']}` (+{cfg['tau_bisections']} bisection) |\n")
    w(f"| started (UTC) | {st['started_utc']} |\n")
    w(f"| wall | {st['wall_s']:.1f} s |\n\n")

    if not cc.get("comparison_run"):
        w(
            f"> **CCOpt**: no integrated-continuation comparison was run. "
            f"{cc.get('reason','')} The pin a comparison would use is "
            f"`{cc.get('pin')}`.\n\n"
        )

    from . import manifest as M

    try:
        man = {c["name"]: c for c in M.load()["cases"]}
    except Exception:
        man = {}

    # ---- headline: which route is supported ----
    w("## Route summary\n\n")
    w(
        "Solved counts the cells that returned `Solve_Succeeded` or\n"
        "`Solved_To_Acceptable_Level` **and** left the source model feasible to\n"
        "1e-5; a cell that converged the NLP but not the MPCC is counted as\n"
        "unsolved here and appears in the triage table below. The infeasible\n"
        "case is excluded from the counts and reported on its own, because\n"
        "'did not solve it' is the right answer there.\n\n"
    )
    w(
        "`at f*` is the column that matters: it counts the cells that reached\n"
        "the MPCC's own global optimum, which the branch-enumeration oracle\n"
        "knows independently. `below f*` counts cells that returned an\n"
        "objective *lower* than the optimum -- possible only at a point the\n"
        "source model does not admit, and the reason a relaxed objective must\n"
        "never be quoted as an MPCC objective.\n\n"
    )
    w(
        "| route | lowering | solved | at f* | below f* | of | median iters | "
        "median stages | restarts | source compl (max) |\n"
    )
    w("|---|---|---:|---:|---:|---:|---:|---:|---:|---|\n")
    for rname in cfg["routes"]:
        cells = [
            r
            for r in recs
            if r["route"] == rname and r["case"] != "infeasible_pair" and r["control"] == "none"
        ]
        if not cells:
            continue
        good = [r for r in cells if r["ok"] and _worst_source(r) <= 1e-5]
        at_star = below = 0
        for r in cells:
            fstar = (man.get(r["case"], {}).get("expected") or {}).get("obj")
            if fstar is None or r["obj"] is None:
                continue
            if r["obj"] < fstar - 1e-6:
                below += 1
            elif abs(r["obj"] - fstar) <= 1e-5 * max(1.0, abs(fstar)):
                at_star += 1
        iters = sorted(r["iters"] for r in cells)
        stages = sorted(r["outer_stages"] for r in cells)
        med_i = iters[len(iters) // 2] if iters else 0
        med_s = stages[len(stages) // 2] if stages else 0
        restarts = sum(r["restarts"] for r in cells)
        compl = max((r["source"].get("compl_max", 0.0) for r in good), default=None)
        w(
            f"| `{rname}` | {R.ROUTES[rname].lowering} | {len(good)} | {at_star} | "
            f"{below} | {len(cells)} | {med_i} | {med_s} | {restarts} | {_fmt(compl)} |\n"
        )
    w("\n")

    # ---- per case ----
    w("## Per case\n\n")
    by_class: Dict[str, List[str]] = {}
    for name in cfg["cases"]:
        by_class.setdefault(C.make(name).klass, []).append(name)

    for klass in sorted(by_class):
        w(f"### class: {klass}\n\n")
        for name in by_class[klass]:
            entry = man.get(name, {})
            exp = entry.get("expected", {})
            w(f"#### `{name}`\n\n")
            if exp:
                w(
                    f"Expected: feasible={exp.get('feasible')}, "
                    f"f*={_fmt(exp.get('obj'))}, stationarity={exp.get('stationarity')}, "
                    f"biactive pairs={exp.get('n_biactive')}, "
                    f"MPCC-LICQ={exp.get('mpcc_licq')}. "
                )
                orc = entry.get("oracle")
                if orc:
                    w(
                        f"Branch-enumeration oracle: feasible={orc.get('feasible')}, "
                        f"f*={_fmt(orc.get('obj'))}"
                    )
                    if orc.get("optimal_branches"):
                        w(f", optimal branches {orc['optimal_branches']}")
                    w(".")
                w("\n\n")
                if exp.get("notes"):
                    w(f"> {exp['notes']}\n\n")
            w(
                "| scaling | start | route | control | status | src f | src compl | "
                "src viol | MPCC class | LICQ | regime | iters | stages | acc/rej | "
                "restarts | NLP kkt (scaled) | NLP kkt (unscaled) | restoration | inertia |\n"
            )
            w("|---|---|---|---|---|---|---|---|---|---|---|---:|---:|---:|---:|---|---|---:|---:|\n")
            for r in [x for x in recs if x["case"] == name]:
                s = r["source"] or {}
                stn = r.get("stationarity") or {}
                nlp = r.get("nlp") or {}
                lc = r.get("log_counters") or {}
                w(
                    f"| {r['scaling']} | {r['start']} | `{r['route']}` | {r['control']} | "
                    f"{r['status_msg']} | {_fmt(r['obj'], 6)} | "
                    f"{_fmt(s.get('compl_max'))} | "
                    f"{_fmt(max(s.get('row_viol', 0.0), s.get('bound_viol', 0.0), s.get('sign_viol', 0.0)))} | "
                    f"{stn.get('klass','--')} | {_fmt(stn.get('mpcc_licq'))} | "
                    f"{','.join(r['regime']) if r.get('regime') else '--'} | "
                    f"{r['iters']} | {r['outer_stages']} | "
                    f"{r['accepted_stages']}/{r['rejected_stages']} | {r['restarts']} | "
                    f"{_fmt(nlp.get('final_kkt_error'))} | "
                    f"{_fmt(nlp.get('final_unscaled_kkt_error'))} | "
                    f"{(r.get('restoration') or {}).get('restoration_calls', '--')} | "
                    f"{lc.get('inertia_corrections', '--')} |\n"
                )
            w("\n")
            vals = [r for r in recs if r["case"] == name and r.get("validation")]
            keys = sorted({k for r in vals for k in r["validation"] if k != "x_unscaled"})
            if keys:
                w("Source-level validation:\n\n")
                w("| scaling | start | route | " + " | ".join(keys) + " |\n")
                w("|---" * (3 + len(keys)) + "|\n")
                for r in vals:
                    row = " | ".join(_fmt(r["validation"].get(k)) for k in keys)
                    w(
                        f"| {r['scaling']} | {r['start']} | `{r['route']}` | {row} |\n"
                    )
                w("\n")

    # ---- sensitivity legs ----
    w("## Sensitivity to initialization and scaling\n\n")
    w(
        "A route whose verdict moves between the `unit` and `skew` scaling legs\n"
        "is reporting on the scaling, not on the MPCC; a route whose verdict\n"
        "moves between starts is reporting a local solver's basin, which is\n"
        "expected and is why the column exists rather than being averaged away.\n\n"
    )
    w("| case | route | starts solved | scalings solved | distinct objectives |\n")
    w("|---|---|---|---|---|\n")
    for name in cfg["cases"]:
        for rname in cfg["routes"]:
            cells = [
                r
                for r in recs
                if r["case"] == name and r["route"] == rname and r["control"] == "none"
            ]
            if not cells:
                continue
            starts = {r["start"] for r in cells}
            ok_starts = {r["start"] for r in cells if r["ok"]}
            scal = {r["scaling"] for r in cells}
            ok_scal = {r["scaling"] for r in cells if r["ok"]}
            objs = sorted({round(r["obj"], 6) for r in cells if r["obj"] is not None})
            w(
                f"| `{name}` | `{rname}` | {len(ok_starts)}/{len(starts)} | "
                f"{len(ok_scal)}/{len(scal)} | {objs} |\n"
            )
    w("\n")

    # ---- kill-switch controls ----
    controls = [c for c in cfg["controls"] if c != "none"]
    if controls:
        w("## Kill-switch controls\n\n")
        w(
            "Each control disables one mechanism that could otherwise explain an\n"
            "outcome. A cell whose verdict changes under a control was being\n"
            "carried by that mechanism, and the attribution has to say so.\n\n"
        )
        w("| case | start | route | " + " | ".join(["none"] + controls) + " |\n")
        w("|---" * (3 + 1 + len(controls)) + "|\n")
        seen = set()
        for r in recs:
            key = (r["case"], r["scaling"], r["start"], r["route"])
            if r["scaling"] != "unit" or key in seen:
                continue
            seen.add(key)
            row = []
            changed = False
            base = None
            for ctl in ["none"] + controls:
                m = [
                    x
                    for x in recs
                    if (x["case"], x["scaling"], x["start"], x["route"]) == key
                    and x["control"] == ctl
                ]
                if not m:
                    row.append("--")
                    continue
                v = m[0]["status_msg"] if not m[0]["ok"] else f"ok/{m[0]['iters']}"
                if base is None:
                    base = m[0]["ok"]
                elif m[0]["ok"] != base:
                    changed = True
                row.append(v)
            if changed:
                w(f"| `{r['case']}` | {r['start']} | `{r['route']}` | " + " | ".join(row) + " |\n")
        w("\nOnly rows whose *solved/not solved* verdict moves under a control are listed.\n\n")

    # ---- triage ----
    w("## Mechanical ownership triage\n\n")
    w(
        "Applied by rule, not by judgement. A `POUNCE candidate` row is not a\n"
        "POUNCE issue: gh#794's issue-splitting rule requires a minimal source\n"
        "model, commit-stamped baseline and comparator measurements,\n"
        "kill-switch evidence excluding existing mechanisms, and a measurable\n"
        "acceptance criterion before one is filed.\n\n"
    )
    w("| case | scaling | start | route | owner | why |\n|---|---|---|---|---|---|\n")
    counts: Dict[str, int] = {}
    for r in recs:
        if r["control"] != "none":
            continue
        entry = man.get(r["case"], {}).get("expected", {})
        sibs = [x for x in recs if x["case"] == r["case"] and x["control"] == "none"]
        t = triage(
            r,
            entry,
            sibs,
            tau_min=min(cfg["tau_schedule"]) if cfg.get("tau_schedule") else None,
            feas_tol=cfg.get("base_options", {}).get("tol"),
        )
        counts[t["owner"]] = counts.get(t["owner"], 0) + 1
        if t["owner"] == "-":
            continue
        w(
            f"| `{r['case']}` | {r['scaling']} | {r['start']} | `{r['route']}` | "
            f"**{t['owner']}** | {t['why']} |\n"
        )
    w("\n")
    w("| owner | observations |\n|---|---:|\n")
    for k in sorted(counts):
        w(f"| {'converged, nothing to assign' if k == '-' else k} | {counts[k]} |\n")
    w("\n")
    return out.getvalue()


def write(payload: dict, path: str) -> str:
    with open(path, "w") as fh:
        fh.write(render(payload))
    return path


def main(argv=None) -> int:  # pragma: no cover - thin CLI
    import argparse
    import json

    ap = argparse.ArgumentParser(prog="mpcc.report")
    ap.add_argument("results")
    ap.add_argument("-o", "--out", required=True)
    a = ap.parse_args(argv)
    with open(a.results) as fh:
        payload = json.load(fh)
    print(write(payload, a.out))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
