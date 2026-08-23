#!/usr/bin/env python3
"""Assert that the GAMS solver-link smoke check actually exercised the link.

`make -C benchmarks gams-bench` exists for one reason: to prove the GAMS
solver-link path (GMO/GEV -> pounce) still works.  Nothing else in the
benchmark sweep touches it.  A liveness check that can pass without running
anything is worse than no check, because the report it leaves behind is
stamped with the current commit and the current date and reads as a fresh
result.

That is what gh#747 found.  Two failures compounded:

  * the vendored `runsolver` Makefile keys `rungams` off the per-instance
    trace CSVs, not off the pounce build, so a rebuilt solver left them
    satisfied -- `make[4]: Nothing to be done for 'rungams'` -- and the
    report was regenerated from months-old traces;

  * the pip GAMS link never reported `resUsed`, so every POUNCE row carried
    `NA` in the `SolverTime` column, and `nlpbench_report.py` -- which drops
    an instance from the head-to-head when either side has no time --
    printed `10/10 solved` for both solvers and `both solved: 0` in the same
    report, without complaint.

The link fix is in `python/pounce/gams/link.py`; the re-run is forced by the
`gams-bench` target.  This script is the part that refuses to call any of it
a pass by accident.  `nlpbench_report.py` and the `runsolver` Makefile live
in `gams/nlpbench/`, a private GAMS-licensed clone that is not in this
repository (`.gitignore`), so they cannot carry the guarantee themselves.

Stdlib only.  Exit 0 on success, 1 on a failed assertion, 2 on bad usage.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

# GAMS trace (traceopt=3) column order.  Only the few this script reads are
# named; the rest are positional padding.
COL_INSTANCE = 0
COL_SOLVER = 2
COL_MODEL_STATUS = 13
COL_SOLVER_STATUS = 14
COL_SOLVER_TIME = 17
N_COLUMNS = 22

# Same predicate `nlpbench_report.py` uses: ModelStatus 1 Optimal / 2 Locally
# Optimal, SolverStatus 1 Normal / 2 IterLim.
SOLVED_MODEL_STATUS = {1, 2}
SOLVED_SOLVER_STATUS = {1, 2}


class Failure(Exception):
    """A liveness assertion that did not hold."""


def _num(cell: str):
    cell = cell.strip()
    if cell == "" or cell.upper() == "NA":
        return None
    try:
        return float(cell)
    except ValueError:
        return None


def read_trace(path: Path) -> dict[str, dict]:
    """Parse a GAMS trace CSV into {instance: {...}}."""
    if not path.exists():
        raise Failure(f"{path} does not exist -- the suite did not produce a trace")
    rows: dict[str, dict] = {}
    with path.open() as f:
        for raw in csv.reader(f):
            if not raw or raw[0].startswith("*"):
                continue
            raw = raw + [""] * (N_COLUMNS - len(raw))
            ms = _num(raw[COL_MODEL_STATUS])
            ss = _num(raw[COL_SOLVER_STATUS])
            if ms is None or ss is None:
                continue
            rows[raw[COL_INSTANCE].strip()] = {
                "solver": raw[COL_SOLVER].strip().lower(),
                "solved": int(ms) in SOLVED_MODEL_STATUS
                and int(ss) in SOLVED_SOLVER_STATUS,
                "time": _num(raw[COL_SOLVER_TIME]),
                "model_status": int(ms),
                "solver_status": int(ss),
            }
    if not rows:
        raise Failure(f"{path} has no data rows -- the suite solved nothing")
    return rows


def check(
    traces: dict[str, Path],
    fresh: set[str],
    newer_than: Path | None,
    min_common: int,
) -> list[str]:
    """Run every assertion, returning the notes to print on success."""
    notes: list[str] = []

    # 1. Freshness.  A trace older than the stamp was not produced by this
    #    run -- which is the whole of gh#747's first defect.
    if newer_than is not None:
        cutoff = newer_than.stat().st_mtime
        for name in sorted(fresh):
            mtime = traces[name].stat().st_mtime
            if mtime < cutoff:
                raise Failure(
                    f"{traces[name]} was not rewritten by this run "
                    f"(mtime {mtime:.0f} < {cutoff:.0f}); the solve was skipped "
                    f"and the report would be built from stale traces"
                )
        notes.append(f"{len(fresh)} trace(s) rewritten by this run")

    parsed = {name: read_trace(path) for name, path in traces.items()}
    for name, rows in parsed.items():
        notes.append(f"{name}: {len(rows)} instances, "
                     f"{sum(r['solved'] for r in rows.values())} solved")

    # 2. Every row must carry a solver time.  A missing time is silently
    #    dropped from the head-to-head, which is gh#747's second defect.
    for name, rows in parsed.items():
        missing = sorted(i for i, r in rows.items() if r["time"] is None)
        if missing:
            raise Failure(
                f"{name} reported no SolverTime for {len(missing)} instance(s) "
                f"(e.g. {', '.join(missing[:3])}); the head-to-head silently "
                f"drops these, so the report would contradict itself"
            )

    # 3. The head-to-head must actually join.
    if len(parsed) >= 2:
        names = sorted(parsed)
        common = set(parsed[names[0]])
        for name in names[1:]:
            common &= set(parsed[name])
        if len(common) < min_common:
            raise Failure(
                f"only {len(common)} instance(s) common to {', '.join(names)}; "
                f"expected at least {min_common}"
            )
        both = sorted(i for i in common if all(parsed[n][i]["solved"] for n in names))
        if not both:
            raise Failure(
                f"no instance was solved by all of {', '.join(names)}; "
                f"the solver link is not working"
            )
        notes.append(f"{len(common)} common instances, {len(both)} solved by all")
    else:
        rows = next(iter(parsed.values()))
        solved = sum(r["solved"] for r in rows.values())
        if solved < min_common:
            raise Failure(
                f"only {solved} instance(s) solved; expected at least {min_common}"
            )

    return notes


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--trace",
        action="append",
        required=True,
        metavar="NAME=PATH",
        help="trace CSV to check, tagged with a solver name (repeatable)",
    )
    p.add_argument(
        "--fresh",
        action="append",
        default=[],
        metavar="NAME",
        help="trace that must be newer than --newer-than (repeatable). "
        "The ipopt reference is deliberately cached, so only the pounce "
        "trace is required to be fresh.",
    )
    p.add_argument(
        "--newer-than",
        metavar="PATH",
        help="stamp file; --fresh traces must postdate it",
    )
    p.add_argument("--min-common", type=int, default=1)
    args = p.parse_args()

    traces: dict[str, Path] = {}
    for spec in args.trace:
        if "=" not in spec:
            p.error(f"--trace expects NAME=PATH, got {spec!r}")
        name, path = spec.split("=", 1)
        traces[name] = Path(path)

    fresh = set(args.fresh)
    unknown = fresh - set(traces)
    if unknown:
        p.error(f"--fresh names no such trace: {', '.join(sorted(unknown))}")

    stamp = Path(args.newer_than) if args.newer_than else None
    if stamp is not None and not stamp.exists():
        p.error(f"--newer-than {stamp} does not exist")

    try:
        notes = check(traces, fresh, stamp, args.min_common)
    except Failure as exc:
        print(f"gams smoke check FAILED: {exc}", file=sys.stderr)
        return 1

    for note in notes:
        print(f"  {note}")
    print("gams smoke check passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
