#!/usr/bin/env python3
"""Derive per-problem reference status for the Vanderbei CUTE suite.

Robert Vanderbei tabulates SNOPT / NITRO / LOQO results for the CUTE-in-AMPL
collection at https://vanderbei.princeton.edu/cute_table.pdf. This script
parses that table and classifies every problem in our `nl/` suite by what the
table says is known about it:

    optimum      at least one reference solver reported a finite optimal
                 objective  -> a feasible solution is known to exist
                 (this is the "expected-solvable" set)
    hard         in the table, but all three solvers failed only via time /
                 iteration limit or error -> feasible-or-unknown but hard;
                 a POUNCE failure here is shared with the commercial solvers
    unbounded    a reference solver reported (Unb)
    infeasible   a reference solver reported (Inf) / (P/D I)
    untabulated  not present in cute_table.pdf -> no reference datum

For `optimum` problems the finite reference objective(s) are recorded so a
report can flag a POUNCE objective that disagrees with the literature.

Output: cute_table_status.json next to this script. Regenerate with:

    python3 cute_table_status.py            # downloads the PDF, needs poppler's pdftotext
    python3 cute_table_status.py --txt cute_table.txt   # parse a pre-extracted text layer

Requires `pdftotext` (poppler) when downloading; the resulting JSON is the
durable artifact and is committed, so regeneration is only needed if the
upstream table or our suite changes.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
URL = "https://vanderbei.princeton.edu/cute_table.pdf"
OUT = os.path.join(HERE, "cute_table_status.json")
NL_DIR = os.path.join(HERE, "nl")

FLOAT_RE = re.compile(r'^[+-]?\d+(\.\d+)?[Ee][+-]?\d+$|^[+-]?\d+\.\d+$|^[+-]?\d+$')
INT_RE = re.compile(r'^\d+$')
FLAG_RE = re.compile(r'^[a-zA-Z]{1,2}$')   # trailing single-letter marker column


def get_text(txt_path: str | None) -> str:
    if txt_path:
        with open(txt_path) as f:
            return f.read()
    with tempfile.TemporaryDirectory() as d:
        pdf = os.path.join(d, "cute_table.pdf")
        txt = os.path.join(d, "cute_table.txt")
        urllib.request.urlretrieve(URL, pdf)
        try:
            subprocess.run(["pdftotext", "-layout", pdf, txt], check=True)
        except (OSError, subprocess.CalledProcessError) as e:
            sys.exit(f"pdftotext failed ({e}); install poppler or pass --txt")
        with open(txt) as f:
            return f.read()


def parse_table(text: str) -> dict:
    """name -> {'n','m','objs':[snopt,nitro,loqo]} for every data row."""
    table = {}
    for line in text.splitlines():
        t = line.split()
        if len(t) < 5 or not (INT_RE.match(t[1]) and INT_RE.match(t[2])):
            continue
        body = t[:-1] if (FLAG_RE.match(t[-1]) and not FLOAT_RE.match(t[-1])) else t
        table[t[0].lower()] = {'n': int(t[1]), 'm': int(t[2]), 'objs': body[-3:]}
    return table


def classify(objs: list[str]) -> tuple[str, list]:
    """Return (status, [finite-or-None per solver]). 'optimum' wins whenever
    any solver got a finite objective."""
    finite = [float(o) if FLOAT_RE.match(o) else None for o in objs]
    if any(v is not None for v in finite):
        return "optimum", finite
    blob = " ".join(objs)
    if "Unb" in blob:
        return "unbounded", finite
    if "Inf" in blob or "P/D" in blob:
        return "infeasible", finite
    return "hard", finite


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--txt", help="pre-extracted pdftotext -layout output")
    args = ap.parse_args(argv)

    table = parse_table(get_text(args.txt))
    ours = sorted(f[:-3].lower() for f in os.listdir(NL_DIR) if f.endswith(".nl"))

    problems = {}
    counts = {"optimum": 0, "hard": 0, "infeasible": 0, "unbounded": 0, "untabulated": 0}
    for name in ours:
        row = table.get(name)
        if row is None:
            problems[name] = {"status": "untabulated"}
            counts["untabulated"] += 1
            continue
        status, finite = classify(row["objs"])
        counts[status] += 1
        entry = {"status": status, "n": row["n"], "m": row["m"]}
        if status == "optimum":
            snopt, nitro, loqo = finite
            entry["solvers"] = {"snopt": snopt, "nitro": nitro, "loqo": loqo}
            # LOQO is Vanderbei's own solver; prefer it, else any finite value.
            ref = loqo if loqo is not None else next(v for v in (nitro, snopt) if v is not None)
            entry["ref_obj"] = ref
            vals = [v for v in finite if v is not None]
            spread = max(vals) - min(vals)
            scale = max(1.0, max(abs(v) for v in vals))
            entry["solvers_agree"] = spread <= 1e-3 * scale
        else:
            entry["markers"] = " ".join(row["objs"])
        problems[name] = entry

    out = {
        "_meta": {
            "source": URL,
            "description": (
                "Reference status per problem for the Vanderbei CUTE-in-AMPL "
                "suite, derived from cute_table.pdf (SNOPT/NITRO/LOQO). "
                "status: optimum (finite reference optimum -> expected-solvable) | "
                "hard (in table, all solvers hit time/iter limit) | "
                "infeasible | unbounded | untabulated (not in the table). "
                "'optimum' wins whenever any solver reported a finite objective."
            ),
            "suite_size": len(ours),
            "counts": counts,
        },
        "problems": problems,
    }
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"wrote {OUT}")
    print("  " + ", ".join(f"{k}={v}" for k, v in counts.items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
