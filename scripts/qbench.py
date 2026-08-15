#!/usr/bin/env python3
"""Before/after measurement harness for the quadratic-structure work (gh #588).

Every phase of that series is justified by a number, so the number has to be
produced the same way each time. This runs a fixed instance set through a given
`pounce` binary and records, per instance:

  * routing        -- the `Problem class:` line, i.e. whether the recognizer's
                      answer survived to the solver
  * outcome        -- status, iteration count, objective
  * cost           -- wall clock, peak RSS
  * attribution    -- the `print_timing_statistics` phase breakdown, which is
                      how we tell an evaluation-bound instance from a
                      factorization-bound one
  * work counters  -- num_hess_evals / num_constr_jac_evals, and the linear
                      solver's n_factors / n_pattern_reuse

`compare` then diffs two runs. The point of separating them is that the "before"
run must be taken with the *parent* binary, not reconstructed afterwards from
memory.

Usage
-----
    qbench.py run   <pounce-binary> <out.json> [--set NAME] [--timeout SECS]
    qbench.py compare <before.json> <after.json> [--md]

Instances come from $POUNCE_BENCH_DATA (default
~/Dropbox/projects/pounce-bench-data).

Sets
----
    quad     the qcqp family + qssp180/nql180   (the target of gh #588)
    control  problems that must NOT move        (regression guard)
    qp       convex QPs incl. EXDATA            (the widest-applying lever)
    all      everything above
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

BENCH = Path(
    os.environ.get(
        "POUNCE_BENCH_DATA", Path.home() / "Dropbox/projects/pounce-bench-data"
    )
)

# (name, relative .nl path, extra solver options)
QUAD = [
    ("qcqp500-3c", "mittelmann/nl/qcqp500-3c.nl", []),
    ("qcqp500-3nc", "mittelmann/nl/qcqp500-3nc.nl", []),
    ("qcqp750-2c", "mittelmann/nl/qcqp750-2c.nl", []),
    ("qcqp750-2nc", "mittelmann/nl/qcqp750-2nc.nl", []),
    ("qcqp1000-1nc", "mittelmann/nl/qcqp1000-1nc.nl", []),
    ("qcqp1000-2c", "mittelmann/nl/qcqp1000-2c.nl", []),
    ("qcqp1000-2nc", "mittelmann/nl/qcqp1000-2nc.nl", []),
    ("qcqp1500-1c", "mittelmann/nl/qcqp1500-1c.nl", []),
    ("qcqp1500-1nc", "mittelmann/nl/qcqp1500-1nc.nl", []),
    ("qssp180", "mittelmann/nl/qssp180.nl", []),
    ("nql180", "mittelmann/nl/nql180.nl", []),
]

# Deliberately *not* quadratic, or quadratic-but-already-routed. If a phase
# claims to be inert outside its target class, these are where that shows.
CONTROL = [
    ("clnlbeam", "mittelmann/nl/clnlbeam.nl", []),
    ("bearing_400", "mittelmann/nl/bearing_400.nl", []),
    ("dirichlet120", "mittelmann/nl/dirichlet120.nl", []),
    ("camshape_6400", "mittelmann/nl/camshape_6400.nl", []),
    ("marine_1600", "mittelmann/nl/marine_1600.nl", []),
]

QP = [
    ("EXDATA", "qp/nl/EXDATA.nl", []),
    ("EXDATA-nlp", "qp/nl/EXDATA.nl", ["solver_selection=nlp"]),
    ("CVXQP1_L", "qp/nl/CVXQP1_L.nl", []),
    ("AUG2DC", "qp/nl/AUG2DC.nl", []),
    ("BOYD1", "qp/nl/BOYD1.nl", []),
]

SETS = {"quad": QUAD, "control": CONTROL, "qp": QP, "all": QUAD + CONTROL + QP}

CLASS_RE = re.compile(r"^Problem class:\s*(.+?)\.?$", re.M)
TIMING_RE = re.compile(r"^\s*(\w[\w ]*?)\.+:\s+([\d.]+)s\s*$", re.M)

# The phase timers we actually reason about. Others are recorded but these are
# the ones the report prints, because they are what discriminate the regimes.
KEY_TIMERS = [
    "OverallAlgorithm",
    "LinearSystemFactorization",
    "LinearSystemBackSolve",
    "TotalFunctionEvaluations",
    "LagrangianHessianEvaluations",
    "ConstraintEvaluations",
    "ConstraintJacobianEvaluations",
    "ComputeSearchDirection",
]


def _rss_mb(ru_maxrss: int) -> float:
    """ru_maxrss to MB. macOS reports bytes, Linux kilobytes."""
    return ru_maxrss / (1024 * 1024) if sys.platform == "darwin" else ru_maxrss / 1024


def _run_capture(cmd: list[str], timeout: int) -> tuple[str, float, bool]:
    """Run `cmd`, returning (stdout, peak_rss_mb, timed_out).

    Deliberately not `subprocess.run`: `resource.getrusage(RUSAGE_CHILDREN)`
    is a high-water mark across *every* child this process has ever reaped, so
    reading it after each solve reports the largest instance seen so far for all
    the rest. `os.wait4` gives the rusage of this one child, which is the number
    the memory claim in the design note actually rests on.
    """
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True
    )
    timed_out = False

    def _kill() -> None:
        nonlocal timed_out
        timed_out = True
        proc.kill()

    timer = threading.Timer(timeout, _kill)
    timer.start()
    try:
        out = proc.stdout.read()
        _, status, ru = os.wait4(proc.pid, 0)
    finally:
        timer.cancel()
        proc.stdout.close()
        proc.returncode = os.waitstatus_to_exitcode(status) if not timed_out else -9
    return out, _rss_mb(ru.ru_maxrss), timed_out


def run_one(binary: str, name: str, nl: Path, extra: list[str], timeout: int) -> dict:
    if not nl.exists():
        return {"name": name, "error": f"missing: {nl}"}

    out_json = Path(f"/tmp/qbench-{os.getpid()}-{name}.json")
    cmd = [
        binary,
        str(nl),
        "--no-sol",
        "--json-output",
        str(out_json),
        "timing_statistics=yes",
        "print_timing_statistics=yes",
        *extra,
    ]

    t0 = time.monotonic()
    stdout, rss, timed_out = _run_capture(cmd, timeout)
    wall = time.monotonic() - t0

    rec: dict = {
        "name": name,
        "wall_secs": round(wall, 3),
        "peak_rss_mb": round(rss, 1),
        "timed_out": timed_out,
    }

    m = CLASS_RE.search(stdout)
    if m:
        line = m.group(1).strip()
        cls, _, solver = line.partition(". Selected solver:")
        rec["problem_class"] = cls.strip()
        if solver:
            rec["selected_solver"] = solver.strip()

    timers = {k: float(v) for k, v in TIMING_RE.findall(stdout)}
    if timers:
        rec["timing"] = {k: timers[k] for k in KEY_TIMERS if k in timers}
        overall = timers.get("OverallAlgorithm") or 0.0
        if overall > 0:
            rec["pct"] = {
                k: round(100 * timers[k] / overall, 1)
                for k in ("LinearSystemFactorization", "TotalFunctionEvaluations",
                          "LagrangianHessianEvaluations")
                if k in timers
            }

    if out_json.exists():
        try:
            d = json.loads(out_json.read_text())
            st = d.get("statistics", {})
            ls = d.get("linear_solver", {})
            sol = d.get("solution", {})
            rec.update(
                status=sol.get("status"),
                objective=sol.get("objective"),
                iters=st.get("iteration_count"),
                num_hess_evals=st.get("num_hess_evals"),
                num_constr_jac_evals=st.get("num_constr_jac_evals"),
                num_constr_evals=st.get("num_constr_evals"),
                final_constr_viol=st.get("final_constr_viol"),
                linear_solver=ls.get("solver_name"),
                n_factors=ls.get("n_factors"),
                n_pattern_reuse=ls.get("n_pattern_reuse"),
                last_nnz_l=ls.get("last_nnz_l"),
                max_fill_ratio=ls.get("max_fill_ratio"),
            )
        except (json.JSONDecodeError, OSError) as e:
            rec["json_error"] = str(e)
        finally:
            out_json.unlink(missing_ok=True)
    elif not timed_out:
        rec["error"] = "no JSON report written"

    return rec


def cmd_run(args) -> int:
    instances = SETS[args.set]
    results = []
    for name, rel, extra in instances:
        print(f"  {name:16s} ", end="", flush=True)
        rec = run_one(args.binary, name, BENCH / rel, extra, args.timeout)
        results.append(rec)
        if "error" in rec:
            print(f"SKIP ({rec['error']})")
        else:
            print(
                f"{rec.get('status', 'TIMEOUT'):28s} "
                f"iters={str(rec.get('iters', '-')):>5s} "
                f"{rec['wall_secs']:8.2f}s  {rec['peak_rss_mb']:8.0f} MB"
            )

    payload = {
        "binary": args.binary,
        "set": args.set,
        "bench_data": str(BENCH),
        "git_describe": git_describe(),
        "results": results,
    }
    Path(args.out).write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {args.out}")
    return 0


def git_describe() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def _fmt_ratio(before, after, lower_is_better=True) -> str:
    if before in (None, 0) or after in (None, 0):
        return "-"
    r = before / after if lower_is_better else after / before
    if 0.98 < r < 1.02:
        return "="
    return f"{r:.2f}x"


def cmd_compare(args) -> int:
    b = {r["name"]: r for r in json.loads(Path(args.before).read_text())["results"]}
    a = {r["name"]: r for r in json.loads(Path(args.after).read_text())["results"]}

    rows = []
    for name in [n for n in b if n in a]:
        rb, ra = b[name], a[name]
        if "error" in rb or "error" in ra:
            continue
        rows.append(
            {
                "name": name,
                "class_before": rb.get("problem_class", "?"),
                "class_after": ra.get("problem_class", "?"),
                "status_before": rb.get("status", "TIMEOUT"),
                "status_after": ra.get("status", "TIMEOUT"),
                "iters_before": rb.get("iters"),
                "iters_after": ra.get("iters"),
                "wall_before": rb.get("wall_secs"),
                "wall_after": ra.get("wall_secs"),
                "rss_before": rb.get("peak_rss_mb"),
                "rss_after": ra.get("peak_rss_mb"),
                "hess_before": rb.get("num_hess_evals"),
                "hess_after": ra.get("num_hess_evals"),
                "speedup": _fmt_ratio(rb.get("wall_secs"), ra.get("wall_secs")),
                "rss_ratio": _fmt_ratio(rb.get("peak_rss_mb"), ra.get("peak_rss_mb")),
            }
        )

    if args.md:
        print("| instance | class | status | iters | wall (s) | peak RSS (MB) | speedup |")
        print("|---|---|---|---|---|---|---|")
        for r in rows:
            cls = (
                r["class_before"]
                if r["class_before"] == r["class_after"]
                else f"**{r['class_before']} → {r['class_after']}**"
            )
            st = (
                r["status_before"]
                if r["status_before"] == r["status_after"]
                else f"**{r['status_before']} → {r['status_after']}**"
            )
            print(
                f"| {r['name']} | {cls} | {st} | "
                f"{r['iters_before']} → {r['iters_after']} | "
                f"{r['wall_before']} → {r['wall_after']} | "
                f"{r['rss_before']:.0f} → {r['rss_after']:.0f} | "
                f"{r['speedup']} |"
            )
    else:
        for r in rows:
            print(json.dumps(r))

    # Correctness guard: a phase that changes a status or an objective has to
    # say so out loud rather than be read off a speedup column.
    moved = [r for r in rows if r["status_before"] != r["status_after"]]
    if moved:
        print("\n!! STATUS CHANGES:", file=sys.stderr)
        for r in moved:
            print(
                f"   {r['name']}: {r['status_before']} -> {r['status_after']}",
                file=sys.stderr,
            )
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run")
    r.add_argument("binary")
    r.add_argument("out")
    r.add_argument("--set", default="quad", choices=sorted(SETS))
    r.add_argument("--timeout", type=int, default=1800)
    r.set_defaults(func=cmd_run)

    c = sub.add_parser("compare")
    c.add_argument("before")
    c.add_argument("after")
    c.add_argument("--md", action="store_true", help="emit a markdown table")
    c.set_defaults(func=cmd_compare)

    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
