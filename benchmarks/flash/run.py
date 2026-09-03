"""The Gate 1 CLI: run the legs, write the result file, render the report.

    python -m flash.selftest          # no solver needed; gate on everything else
    python -m flash.run --smoke       # one leg, five temperatures (~20 s)
    python -m flash.run --full        # every leg, every route, writes results

Modes
-----

``--smoke`` is the deterministic asserted subset and the shape the CI
fixture drives: the supported route, one ascending cold leg, and five
temperatures chosen to put one point in each regime and one on each side
of a switch. It asserts rather than reports, so a regression is a
non-zero exit rather than a number in a file nobody reads.

``--full`` is the gh#776 configuration: every route in `routes.ROUTES`
on the supported route's leg set, all four legs for the supported route,
and the whole path. It writes ``results-full.json`` (against
``schema.json``) and ``results-full.md`` beside this module. Both are
regenerated per run and gitignored; ``schema.json`` is tracked.

Every record keeps source-level quantities apart from POUNCE's own NLP
diagnostics, and every file carries the stamp `stamp.py` builds --
including the explicit statement that no DiscOpt comparison was run and
why.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time
from typing import Dict, List

import numpy as np

from . import oracle, path, routes as R, spec, stamp
from .runner import SolveRecord, cold_start, solve_route

_HERE = os.path.dirname(os.path.abspath(__file__))

SCHEMA_ID = "pounce-flash-results/1"

#: The smoke subset: one point per regime, one either side of a switch.
#: Chosen from the path rather than invented, so the smoke run is a
#: strict subset of the full one and their records are comparable.
SMOKE_TEMPERATURES = (250.0, 268.0, 300.0, 324.0, 350.0)


def _jsonable(obj):
    if dataclasses.is_dataclass(obj):
        return {k: _jsonable(v) for k, v in dataclasses.asdict(obj).items()}
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [_jsonable(v) for v in obj.tolist()]
    # bool first: Python's `bool` is a subclass of `int`, so an `int`
    # branch above this one silently writes `true` as `1` -- which is
    # valid JSON, passes a loose reader, and fails the schema's
    # {"type": "boolean"}. It shipped `cold_legs_agree: 1` once.
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return v if np.isfinite(v) else None
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    return obj


def _oracle_row(case: spec.FlashCase, t: float) -> Dict[str, object]:
    ref = oracle.flash(t, case.pressure_pa, case.mixture)
    return {
        "temperature_k": t,
        "regime": ref.regime,
        "beta": ref.beta,
        "sum_x": ref.sum_x,
        "sum_y": ref.sum_y,
        "k": [float(v) for v in ref.k],
        "residual": ref.residual,
        "trivial": ref.trivial,
        "ambiguous": ref.ambiguous,
        "no_incipient_phase": ref.no_incipient_phase,
        "vapor_trial_sum": ref.vapor_trial.sum_y,
        "liquid_trial_sum": ref.liquid_trial.sum_y,
        "vapor_trial_stationary_points": ref.vapor_trial.stationary_points,
        "liquid_trial_stationary_points": ref.liquid_trial.stationary_points,
    }


def _all_ok(rec: SolveRecord) -> bool:
    """Every ``_ok`` key in the record's validation block, plus the status."""
    if not rec.ok:
        return False
    return all(
        v for k, v in rec.validation.items() if k.endswith("_ok") and v is not None
    )


def run_smoke(case: spec.FlashCase, verbose: bool = True) -> int:
    """The asserted subset. Returns a process exit code."""
    route = R.ROUTES[R.SUPPORTED_ROUTE]
    failures: List[str] = []
    print(
        f"flash smoke: {case.name}, route {route.name}, "
        f"{len(SMOKE_TEMPERATURES)} temperatures"
    )
    for t in SMOKE_TEMPERATURES:
        rec = solve_route(case, t, route, cold_start(case, t))
        ref = oracle.flash(t, case.pressure_pa, case.mixture)
        ok = _all_ok(rec)
        if not ok:
            bad = [
                k
                for k, v in rec.validation.items()
                if k.endswith("_ok") and v is not None and not v
            ]
            failures.append(f"T={t}: {rec.status_msg or 'failed'} {bad}")
        if verbose:
            print(
                f"  [{'ok  ' if ok else 'FAIL'}] T={t:6.1f} K  "
                f"{(rec.regime or 'failed'):10s} (oracle {ref.regime:10s})  "
                f"beta={rec.beta if rec.beta is None else round(rec.beta, 8)}  "
                f"compl={rec.source.get('compl_max', float('nan')):.1e}  "
                f"iters={rec.iters}"
            )
    if failures:
        print("\nFAILED:")
        for f in failures:
            print(f"  {f}")
        return 1
    print("\nsmoke passed")
    return 0


def run_full(case: spec.FlashCase, verbose: bool = False) -> Dict[str, object]:
    """Every leg for the supported route, plus every route on one leg."""
    started = time.time()
    legs: List[path.Leg] = []
    supported = R.ROUTES[R.SUPPORTED_ROUTE]

    def progress(rec: SolveRecord) -> None:
        if verbose:
            print(
                f"    T={rec.temperature_k:6.1f} {(rec.regime or 'failed'):10s} "
                f"iters={rec.iters}",
                file=sys.stderr,
            )

    for direction, start_mode in path.LEGS:
        if verbose:
            print(f"  leg {direction}/{start_mode} ({supported.name})", file=sys.stderr)
        legs.append(
            path.traverse(
                case,
                supported,
                direction=direction,
                start_mode=start_mode,
                progress=progress,
            )
        )
    hyst = path.compare(legs)

    # Every other route, on the ascending cold leg only. The point of
    # the comparison is "how much of Gate 0's route boundary carries
    # over to a phase-change model", and that is answered by one leg;
    # running four of each would quadruple the cost for a repetition.
    route_legs: List[path.Leg] = []
    for name, route in R.ROUTES.items():
        if name == R.SUPPORTED_ROUTE:
            continue
        if verbose:
            print(f"  route {name} (up/cold)", file=sys.stderr)
        route_legs.append(
            path.traverse(case, route, direction="up", start_mode="cold", progress=progress)
        )

    oracle_rows = [_oracle_row(case, float(t)) for t in case.temperatures_k]
    switches = oracle.bubble_and_dew(case)

    return {
        "schema": SCHEMA_ID,
        "issue": "gh#776 Gate 1",
        "stamp": {
            "repositories": stamp.repositories(),
            "model_data_revision": stamp.model_data_revision(),
            "environment": stamp.environment(),
            "model": stamp.model_stamp(case),
            "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)),
            "wall_s": time.time() - started,
        },
        "config": {
            "base_options": _jsonable(R.BASE_OPTIONS),
            "supported_route": R.SUPPORTED_ROUTE,
            "tau_schedule": list(R.TAU_SCHEDULE),
            "controls": list(R.DEFAULT_CONTROLS),
            "routes": {k: v.why for k, v in R.ROUTES.items()},
            "legs": [f"{d}_{s}" for d, s in path.LEGS],
        },
        "oracle": {
            "rows": _jsonable(oracle_rows),
            "switch_temperatures_k": _jsonable(switches),
            "method": (
                "Michelsen tangent-plane stability with a multistart, per phase "
                "label; Rachford-Rice with a Newton polish for the two-phase "
                "solve. Shares only `log_fugacity_coefficients` with the model."
            ),
        },
        # `path_dependent` is a property rather than a field, so
        # `dataclasses.asdict` does not reach it; a result file without
        # it is missing the one verdict the four legs exist to produce.
        "hysteresis": dict(_jsonable(hyst), path_dependent=bool(hyst.path_dependent)),
        "legs": _jsonable(
            [
                {
                    "direction": lg.direction,
                    "start_mode": lg.start_mode,
                    "route": lg.route,
                    "control": lg.control,
                    "records": lg.records,
                }
                for lg in legs + route_legs
            ]
        ),
    }


# --------------------------------------------------------------------
# markdown
# --------------------------------------------------------------------


def render(result: Dict[str, object]) -> str:
    st = result["stamp"]
    out: List[str] = []
    a = out.append
    a("# MPCC Gate 1 -- phase-changing flash\n")
    a(f"> gh#776 Gate 1. POUNCE `{st['repositories']['pounce']['describe']}`, "
      f"model-data revision `{st['model_data_revision']}`.\n")
    disc = st["repositories"]["discopt"]
    a(f"> DiscOpt comparison: **not run**. {disc['reason']}\n")

    m = st["model"]
    a("\n## The fixture\n")
    a(f"- {' / '.join(m['components'])}, feed {m['feed_composition']}, "
      f"{m['pressure_pa'] / 1e5:.1f} bar, {m['equation_of_state']}.")
    a(f"- {len(m['temperatures_k'])} temperatures, "
      f"{min(m['temperatures_k'])}-{max(m['temperatures_k'])} K.")
    sw = result["oracle"]["switch_temperatures_k"]
    if sw:
        a("- Switch points located by the oracle: "
          + ", ".join(f"{k.replace('_k', '')} = {v:.4f} K" for k, v in sw.items())
          + ".")

    a("\n## Path traversal (supported route)\n")
    h = result["hysteresis"]
    a("| leg | failed | iterations |")
    a("|---|---:|---:|")
    for k in h["iterations"]:
        a(f"| `{k}` | {h['failures'][k]} | {h['iterations'][k]} |")
    a("")
    a(f"- Cold legs agree: **{h['cold_legs_agree']}** "
      "(they solve identical problems from identical starts; a disagreement "
      "would be a harness defect, not a physical result).")
    path_dependent = bool(h.get("path_dependent", any(h["disagreements"].values())))
    a(f"- Path-dependent answer anywhere: **{path_dependent}**.")
    for key, rows in h["disagreements"].items():
        if rows:
            a(f"  - `{key}`: {rows}")

    a("\n## Agreement with the independent oracle\n")
    a("Source-level only. Nothing in this table is an NLP residual.\n")
    a("| route | leg | points | solved | all checks pass | worst |beta - beta_oracle| | worst |G*H| |")
    a("|---|---|---:|---:|---:|---:|---:|")
    for leg in result["legs"]:
        recs = leg["records"]
        solved = sum(1 for r in recs if r["ok"])
        okall = sum(1 for r in recs if _record_all_ok(r))
        wb = max((r["validation"].get("beta_error", 0.0) or 0.0) for r in recs if r["ok"]) if solved else float("nan")
        wc = max((r["source"].get("compl_max", 0.0) or 0.0) for r in recs if r["ok"]) if solved else float("nan")
        a(f"| `{leg['route']}` | {leg['direction']}/{leg['start_mode']} | {len(recs)} | "
          f"{solved} | {okall} | {wb:.1e} | {wc:.1e} |")

    a("\n## Did the finishing solve run?\n")
    a("Only routes with a `finish` lowering have one. A route whose "
      "records are `ok` while its finish was rejected ran the "
      "*continuation half* of its definition, and the table says so "
      "rather than the summary above implying otherwise.\n")
    a("| route | leg | finish accepted | rejected | reason |")
    a("|---|---|---:|---:|---|")
    any_finish = False
    for leg in result["legs"]:
        recs = [r for r in leg["records"] if r.get("finish_applied") is not None]
        if not recs:
            continue
        any_finish = True
        yes = sum(1 for r in recs if r["finish_applied"])
        no = len(recs) - yes
        reasons = sorted({r["finish_status_msg"] for r in recs if not r["finish_applied"]})
        a(f"| `{leg['route']}` | {leg['direction']}/{leg['start_mode']} | {yes} | {no} | "
          f"{'; '.join(reasons) if reasons else '-'} |")
    if not any_finish:
        a("| - | - | - | - | no route in this run defines a finishing solve |")

    a("\n## Regime sequence\n")
    a("Ascending temperature. `L` liquid, `T` two-phase, `V` vapor.\n")
    a("```")
    a("oracle    " + "".join(r["regime"][0].upper() for r in result["oracle"]["rows"]))
    for leg in result["legs"]:
        rows = sorted(leg["records"], key=lambda r: r["temperature_k"])
        a(f"{leg['route'][:9]:9s} " + "".join((r["regime"] or "?")[0].upper() for r in rows)
          + f"   ({leg['direction']}/{leg['start_mode']})")
    a("```")

    a("\n## What this fixture is not evidence about\n")
    a("- **More than one flash.** Two components, five variables, one "
      "equilibrium stage. Nothing here bounds a tray, a column, or a "
      "dynamic transcription; gh#776 gates those on this result and not "
      "the other way round.")
    a("- **The DiscOpt half of Gate 1.** The reduced GDP/SOS1 regime "
      "cross-validation is blocked on jkitchin/discopt#1123 and was not run.")
    a("- **Supercritical mixture states.** The path stays far from the "
      "mixture critical point. Ethane is above its own `Tc` over the top "
      "third of it, which is ordinary and is not the same thing.")
    a("- **Retrograde or multi-root regions.** The path crosses each "
      "regime exactly once, by construction and by assertion "
      "(`selftest.check_regime_coverage`).")
    return "\n".join(out) + "\n"


def _record_all_ok(rec: dict) -> bool:
    if not rec.get("ok"):
        return False
    return all(
        v for k, v in (rec.get("validation") or {}).items()
        if k.endswith("_ok") and v is not None
    )


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="gh#776 Gate 1 flash fixture")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--smoke", action="store_true", help="asserted subset (~20 s)")
    g.add_argument("--full", action="store_true", help="the full configuration")
    p.add_argument("-v", "--verbose", action="store_true")
    p.add_argument("-o", "--output", default=None, help="result file (default beside this module)")
    args = p.parse_args(argv)

    case = spec.GATE1_FLASH
    if args.smoke:
        return run_smoke(case, verbose=True)

    result = run_full(case, verbose=args.verbose)
    out = args.output or os.path.join(_HERE, "results-full.json")
    with open(out, "w") as fh:
        json.dump(_jsonable(result), fh, indent=1)
    md = os.path.splitext(out)[0] + ".md"
    with open(md, "w") as fh:
        fh.write(render(result))
    print(f"wrote {out}\nwrote {md}")
    h = result["hysteresis"]
    return 0 if h["cold_legs_agree"] and not h.get("path_dependent") else 1


if __name__ == "__main__":
    sys.exit(main())
