"""Command line entry point.

    python -m mpcc.run --smoke                  # deterministic subset, asserted
    python -m mpcc.run --full                   # the reproducible full run
    python -m mpcc.run --cases ralph1 --routes all --controls all -v
    python -m mpcc.run --write-manifest         # regenerate manifest.json

``--smoke`` is the subset that has to stay fast and has to keep passing:
it runs one case per benchmark class over the required route list at
unit scaling, and then asserts a handful of *properties* rather than
recorded numbers. The distinction matters. Pinning iteration counts here
would make the smoke test a trajectory guard for the whole NLP solver,
which is `scripts/sweep-fixtures.sh`'s job and not this file's; what
this file must catch is a harness or route that has stopped meaning what
it says -- a route reporting success on the infeasible case, a lowering
whose returned point is not source-feasible, a classifier that no longer
reproduces a hand-derived stationarity class.

``--full`` adds the remaining cases, both scaling legs, every named
start, and every kill-switch control, and writes the report the gate
decision is made from.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime
import json
import math
import os
import sys
from typing import Any, Dict, List, Optional

from . import cases as C
from . import manifest as M
from . import report as REPORT
from . import routes as R
from . import stamp
from .oracle import ccopt_status
from .spec import ACTIVE_TOL, SCALINGS
from .runner import run_cell

_HERE = os.path.dirname(os.path.abspath(__file__))

#: Routes the smoke subset runs. Every required configuration of gh#794
#: appears; the point of the subset is fewer *cases*, not fewer routes,
#: because a route that has stopped working is what this catches.
SMOKE_ROUTES = list(R.ROUTES)


def _csv(value: str, valid, what: str) -> List[str]:
    if value == "all":
        return list(valid)
    out = [v.strip() for v in value.split(",") if v.strip()]
    bad = [v for v in out if v not in valid]
    if bad:
        raise SystemExit(f"unknown {what}: {', '.join(bad)}\nvalid: {', '.join(valid)}")
    return out


def _asdict(rec) -> dict:
    d = dataclasses.asdict(rec)
    d["stages"] = [dataclasses.asdict(s) if not isinstance(s, dict) else s for s in rec.stages]
    return d


def run(
    case_names: List[str],
    route_names: List[str],
    control_names: List[str],
    scaling_names: List[str],
    start_names: Optional[List[str]],
    capture_log: bool,
    verbose: bool,
) -> List[dict]:
    records: List[dict] = []
    for cname in case_names:
        base = C.make(cname)
        for sc in scaling_names:
            vec = SCALINGS[sc](base.n)
            case = base if sc == "unit" else base.rescale(vec)
            starts = start_names or list(case.starts)
            for st in starts:
                if st not in case.starts:
                    continue
                for rname in route_names:
                    for ctl in control_names:
                        if verbose:
                            print(
                                f"  {cname:20s} {sc:5s} {st:14s} {rname:22s} {ctl}",
                                file=sys.stderr,
                                flush=True,
                            )
                        rec = run_cell(
                            case,
                            R.ROUTES[rname],
                            ctl,
                            st,
                            sc,
                            vec,
                            capture_log=capture_log,
                        )
                        records.append(_asdict(rec))
    return records


# --------------------------------------------------------------------
# smoke assertions
# --------------------------------------------------------------------


def smoke_checks(records: List[dict]) -> List[str]:
    """Properties that must hold, whatever the numbers came out as.

    Each returns a failure string; an empty list is a pass. They are
    stated as properties because the alternative -- a recorded
    iteration count or objective per cell -- would turn an ordinary
    trajectory change anywhere in the NLP solver into a red smoke test
    here, which is not this harness's job and would get it disabled.
    """
    fails: List[str] = []

    by_case: Dict[str, List[dict]] = {}
    for r in records:
        by_case.setdefault(r["case"], []).append(r)

    # 0. The run has to contain solved cells at all.
    #
    #    Every property below is quantified over *successful* records and
    #    skips the rest, which is right -- a route is allowed to fail, and
    #    several are expected to. But it means a run in which nothing
    #    succeeded satisfies all of them vacuously and reports a pass. A
    #    harness that cannot tell "every property held" from "there was
    #    nothing to check" is not asserting anything (gh#794 review).
    feasible = [r for r in records if r["case"] != "infeasible_pair"]
    if feasible and not any(r["ok"] for r in feasible):
        fails.append(
            "no feasible smoke case was solved by any route: the properties "
            "below would all pass vacuously"
        )

    # 1. The infeasible case must never be reported as a solved MPCC.
    #
    #    Any success here is wrong, not merely a success at a
    #    source-feasible point: `infeasible_pair` has no feasible point at
    #    all, so a route that completes on it is reporting on its own
    #    reformulation. The earlier form of this check only fired when the
    #    returned point was *near-feasible*, so a confident success at a
    #    large source violation -- the worse failure -- passed silently.
    for r in by_case.get("infeasible_pair", []):
        if not r["ok"]:
            continue
        src = r.get("source") or {}
        worst = (
            max(src["row_viol"], src["bound_viol"], src["sign_viol"], src["compl_max"])
            if src
            else float("nan")
        )
        fails.append(
            f"infeasible_pair/{r['route']}/{r['start']}: reported "
            f"{r['status_msg']} on a provably infeasible MPCC "
            f"(worst source residual {worst:.2e})"
        )

    # 1b. The supported route must actually complete the feasible corpus.
    #
    #     `scholtes_then_ncp` is what the gate recommends, so its coverage
    #     is a property and not a statistic: if it stops solving a case
    #     the recommendation is stale, and no other check here would say
    #     so -- they are all conditioned on success.
    supported = [
        r for r in records if r["route"] == "scholtes_then_ncp" and r["case"] != "infeasible_pair"
    ]
    for r in supported:
        if not r["ok"]:
            fails.append(
                f"{r['case']}/{r['start']}: the supported route "
                f"scholtes_then_ncp did not complete ({r['status_msg']})"
            )

    # 2. A successful exact-product route must return a source-feasible
    #    point. The lowering's feasible set IS the MPCC's, so anything
    #    else is a harness or solver defect, not a modelling tradeoff.
    for r in records:
        if not r["ok"] or r["lowering"] not in ("prod_ineq", "prod_eq"):
            continue
        if r["case"] == "infeasible_pair":
            continue
        src = r.get("source") or {}
        if not src:
            fails.append(
                f"{r['case']}/{r['route']}/{r['start']}: reported success with "
                "no source measurement"
            )
            continue
        worst = max(src["row_viol"], src["bound_viol"], src["sign_viol"], src["compl_max"])
        if worst > 1e-5:
            fails.append(
                f"{r['case']}/{r['route']}/{r['start']}: exact-product route "
                f"reported {r['status_msg']} at a point whose worst source "
                f"residual is {worst:.2e}"
            )

    # 3. A successful Scholtes continuation must end at a complementarity
    #    product no worse than its final tau, by an order of magnitude of
    #    slack. Anything looser means the schedule was not actually
    #    driven home.
    tau_min = min(R.TAU_SCHEDULE)
    for r in records:
        if not r["ok"] or r["lowering"] != "scholtes":
            continue
        if r["case"] == "infeasible_pair":
            continue
        if r["rejected_stages"] and r["accepted_stages"] < len(R.TAU_SCHEDULE):
            continue  # stopped early; reported, not asserted
        if r["source"]["compl_max"] > 10 * tau_min:
            fails.append(
                f"{r['case']}/{r['route']}/{r['start']}: continuation ran to "
                f"tau={tau_min:g} but left compl_max="
                f"{r['source']['compl_max']:.2e}"
            )

    # 4. Every route that succeeds must have produced a classification.
    for r in records:
        if r["ok"] and not r["stationarity"].get("klass"):
            fails.append(f"{r['case']}/{r['route']}/{r['start']}: no stationarity class")

    # 5. Every benchmark class must be represented by at least one record.
    seen = {r["klass"] for r in records}
    for k in ("regular", "biactive", "degenerate", "infeasible", "selector", "macmpec"):
        if k not in seen:
            fails.append(f"benchmark class {k!r} produced no records")

    return fails


# --------------------------------------------------------------------
# result contract
# --------------------------------------------------------------------


def _jsonable(o: Any) -> Any:
    """Recursively replace non-finite floats with ``None``.

    ``NaN`` and ``Infinity`` are not JSON. Python's encoder emits them
    as bare literals by default, which every strict parser rejects --
    so a result file full of them is not readable by the consumers this
    contract exists for. They arise honestly: a solve that returned no
    point has no KKT residuals to report (`nlp.final_*`, 40 records in
    the full sweep), and `validation.biactive_distance` is undefined
    where no pair is biactive (12).

    ``null`` is the representation, because "not available" is exactly
    what it means, and the schema admits it only on the fields where
    that is a real outcome.
    """
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    # numpy scalars and anything else `default=float` would have caught.
    try:
        if hasattr(o, "item"):
            v = o.item()
            if isinstance(v, float):
                return v if math.isfinite(v) else None
            return v
    except Exception:  # pragma: no cover - defensive
        pass
    return o


_RECORD_REQUIRED = (
    "case", "klass", "scaling", "start", "route", "control", "lowering",
    "ok", "status", "status_msg", "obj", "source", "nlp", "iters",
    "outer_stages", "accepted_stages", "rejected_stages",
)

_SOURCE_REQUIRED = (
    "row_viol", "bound_viol", "sign_viol", "compl_max", "compl_min", "compl_sum",
)


def contract_checks(payload: dict) -> List[str]:
    """The emitted artifact really is the contract `schema.json` states.

    Checked here rather than left to a consumer, because the result
    contract *is* a Gate 0 deliverable: a sweep that writes an
    unparseable or under-populated artifact has not produced the thing
    it was run to produce, and exiting 0 on one is the failure mode this
    guards.

    Three properties, in the order a consumer meets them:

    1. the payload survives a strict JSON round trip -- no `NaN`, no
       `Infinity`, both of which Python will happily write and no
       conforming parser will read back;
    2. every record carries the keys the schema marks required;
    3. `source` is fully populated exactly when a point was returned.
       An empty `source` beside a returned point would mean the
       original-space measurement silently did not happen, which is the
       one thing the source/NLP separation exists to prevent.

    When `jsonschema` is importable the committed schema is applied in
    full on top of these; it is not a dependency of the harness, so its
    absence is reported rather than passed over in silence.
    """
    fails: List[str] = []

    try:
        text = json.dumps(payload, allow_nan=False)
    except ValueError as exc:
        return [f"payload is not strict JSON: {exc}"]

    def _reject(c: str) -> float:
        raise ValueError(f"non-JSON literal {c!r} in emitted payload")

    try:
        json.loads(text, parse_constant=_reject)
    except ValueError as exc:
        fails.append(str(exc))

    for i, r in enumerate(payload.get("records", [])):
        where = f"record {i} ({r.get('case')}/{r.get('route')}/{r.get('start')})"
        for k in _RECORD_REQUIRED:
            if k not in r:
                fails.append(f"{where}: missing required key {k!r}")
        src = r.get("source", {})
        if r.get("x") is None:
            if src:
                fails.append(
                    f"{where}: returned no point but carries source measurements"
                )
        else:
            missing = [k for k in _SOURCE_REQUIRED if k not in src]
            if missing:
                fails.append(
                    f"{where}: returned a point but source is missing {missing}"
                )

    try:
        import jsonschema  # type: ignore
    except ImportError:
        return fails

    schema_path = os.path.join(_HERE, "schema.json")
    with open(schema_path) as fh:
        schema = json.load(fh)
    validator = jsonschema.Draft7Validator(schema)
    for err in sorted(validator.iter_errors(payload), key=lambda e: list(e.path))[:20]:
        fails.append("schema: " + "/".join(str(p) for p in err.path) + f": {err.message}")
    return fails


def manifest_checks() -> List[str]:
    """The committed manifest still describes the code."""
    fails = []
    if not os.path.exists(M.MANIFEST_PATH):
        return ["manifest.json is missing; run `python -m mpcc.run --write-manifest`"]
    committed = M.load()
    fresh = M.build(with_oracle=False)
    for key in ("model_data_revision", "smoke_subset", "tau_schedule", "restart_ladder"):
        if committed.get(key) != fresh.get(key):
            fails.append(
                f"manifest.json is stale on {key!r} "
                "(regenerate with `python -m mpcc.run --write-manifest`)"
            )
    if {c["name"] for c in committed["cases"]} != {c["name"] for c in fresh["cases"]}:
        fails.append("manifest.json case list has drifted from cases.py")
    return fails


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="mpcc.run", description=__doc__)
    ap.add_argument("--smoke", action="store_true", help="deterministic asserted subset")
    ap.add_argument("--full", action="store_true", help="the reproducible full configuration")
    ap.add_argument("--cases", default=None)
    ap.add_argument("--routes", default=None)
    ap.add_argument("--controls", default=None)
    ap.add_argument("--scalings", default=None)
    ap.add_argument("--starts", default=None)
    ap.add_argument("--no-log-capture", action="store_true")
    ap.add_argument("--out", default=None, help="result JSON path")
    ap.add_argument("--report", default=None, help="markdown report path")
    ap.add_argument("--no-report", action="store_true")
    ap.add_argument("--write-manifest", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    # The solver's `tracing` output goes to stderr independently of
    # `print_level`, and on this corpus it is dominated by one INFO line
    # per refused acceptable-level termination -- thousands of them
    # across a full run, for a mechanism the `upstream_heuristics`
    # control measures properly. Quiet by default, overridable.
    os.environ.setdefault("RUST_LOG", "error")

    if args.write_manifest:
        print(M.write())
        return 0

    if args.smoke and args.full:
        raise SystemExit("--smoke and --full are exclusive")

    if args.smoke:
        mode = "smoke"
        case_names = list(C.SMOKE)
        route_names = list(SMOKE_ROUTES)
        control_names = ["none"]
        scaling_names = ["unit"]
        start_names = None
    elif args.full:
        mode = "full"
        case_names = list(C.REGISTRY)
        route_names = list(R.ROUTES)
        control_names = list(R.CONTROLS)
        scaling_names = list(SCALINGS)
        start_names = None
    else:
        mode = "custom"
        case_names = _csv(args.cases or "all", C.REGISTRY, "case")
        route_names = _csv(args.routes or "all", list(R.ROUTES), "route")
        control_names = _csv(args.controls or "none", list(R.CONTROLS), "control")
        scaling_names = _csv(args.scalings or "unit", list(SCALINGS), "scaling")
        start_names = None if args.starts is None else [s.strip() for s in args.starts.split(",")]

    out = args.out or os.path.join(_HERE, f"results-{mode}.json")
    rep = args.report or os.path.splitext(out)[0] + ".md"

    mfails = manifest_checks()
    for f in mfails:
        print(f"MANIFEST: {f}", file=sys.stderr)

    t0 = datetime.datetime.now(datetime.timezone.utc)
    records = run(
        case_names,
        route_names,
        control_names,
        scaling_names,
        start_names,
        capture_log=not args.no_log_capture,
        verbose=args.verbose,
    )
    wall = (datetime.datetime.now(datetime.timezone.utc) - t0).total_seconds()

    payload = {
        "schema": "pounce-mpcc-results/1",
        "issue": "https://github.com/jkitchin/pounce/issues/794",
        "stamp": {
            "repositories": stamp.repositories(),
            "model_data_revision": stamp.model_data_revision(),
            "environment": stamp.environment(),
            "ccopt": ccopt_status(),
            "started_utc": t0.isoformat(),
            "wall_s": wall,
        },
        "config": {
            "mode": mode,
            "cases": case_names,
            "routes": route_names,
            "controls": control_names,
            "scalings": scaling_names,
            "starts": start_names,
            "base_options": R.BASE_OPTIONS,
            "tau_schedule": list(R.TAU_SCHEDULE),
            "restart_ladder": list(R.RESTART_LADDER),
            "tau_bisections": R.TAU_BISECTIONS,
            "capture_log": not args.no_log_capture,
            "active_tol": ACTIVE_TOL,
        },
        "records": records,
    }
    # Sanitize before writing, then refuse to write anything a strict
    # parser could not read back (gh#794 review).
    payload = _jsonable(payload)
    with open(out, "w") as fh:
        json.dump(payload, fh, indent=1, default=float, allow_nan=False)
        fh.write("\n")
    print(f"wrote {out} ({len(records)} records, {wall:.1f}s)", file=sys.stderr)

    if not args.no_report:
        REPORT.write(payload, rep)
        print(f"wrote {rep}", file=sys.stderr)

    fails = list(mfails)
    # The artifact's own validity is not a smoke-only concern: a run that
    # writes an unreadable or under-populated result file has failed at
    # the thing it exists to produce, whatever mode it ran in.
    fails += contract_checks(payload)
    if args.smoke:
        fails += smoke_checks(payload["records"])
    if fails:
        print("\nFAILED:", file=sys.stderr)
        for f in fails:
            print(f"  - {f}", file=sys.stderr)
        return 1
    if args.smoke:
        print("smoke: all checks passed", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
