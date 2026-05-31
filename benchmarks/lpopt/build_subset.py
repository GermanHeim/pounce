#!/usr/bin/env python3
"""Fetch a curated, pounce-tractable subset of Hans Mittelmann's lpopt LP
benchmark (https://plato.asu.edu/ftp/lpopt.html) and convert it to .nl.

Instances are pulled as bzip2'd standard MPS from the plato lptestset
mirror (https://plato.asu.edu/ftp/lptestset/), decompressed, size-screened,
and converted to AMPL .nl via mps_to_nl.convert (HiGHS parse -> Pyomo ->
.nl). Anything above the size cap is deferred (the full lpopt set runs to
~30M rows, far beyond what pounce's IPM can handle).

The lpopt page also lists 16 undisclosed instances (unavailable) and
several multi-million-row LPs; this script deliberately covers only the
smaller disclosed instances. Add names to CANDIDATES (with their plato
filename) and/or raise the caps to extend coverage.

Usage:
    build_subset.py            # fetch+convert all under the cap
    build_subset.py qap15 ex10 # only the named lpopt instances
"""
import os
import sys
import subprocess
import bz2

import highspy

import mps_to_nl

HERE = os.path.dirname(os.path.abspath(__file__))
NL_DIR = os.path.join(HERE, "nl")
MPS_DIR = os.path.join(HERE, "mps")
BASE = "https://plato.asu.edu/ftp/lptestset"

# Size caps -- pounce/feral is sparse but this is still a dense-ish IPM.
MAX_VARS = 200_000
MAX_CONS = 200_000
MAX_NNZ = 2_000_000

# lpopt instance name -> filename under plato lptestset/. Only the smaller
# disclosed instances; the 18-70M-compressed ones (square41, scpm1, s82,
# set-cover-model) and the multi-million-row LPs are intentionally omitted.
CANDIDATES = {
    "qap15": "qap15.mps.bz2",
    "supportcase10": "supportcase10.mps.bz2",
    "irish-electricity": "irish-electricity.mps.bz2",
    "ex10": "ex10.mps.bz2",
    "datt256": "datt256_lp.mps.bz2",
    "graph40-40": "graph40-40.mps.bz2",
    "s100": "s100.mps.bz2",
    "savsched1": "savsched1.mps.bz2",
    "woodlands09": "woodlands09.mps.bz2",
}


def fetch(fname, dest):
    url = f"{BASE}/{fname}"
    subprocess.run(["curl", "-fsSL", url, "-o", dest], check=True)


def decompress(bz2_path, mps_path):
    with bz2.open(bz2_path, "rb") as f, open(mps_path, "wb") as o:
        while True:
            chunk = f.read(1 << 20)
            if not chunk:
                break
            o.write(chunk)


def dims(mps_path):
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    h.readModel(mps_path)
    lp = h.getModel().lp_
    return lp.num_col_, lp.num_row_, len(list(lp.a_matrix_.value_))


def main():
    os.makedirs(NL_DIR, exist_ok=True)
    os.makedirs(MPS_DIR, exist_ok=True)
    names = sys.argv[1:] or list(CANDIDATES)
    rows = []
    for name in names:
        if name not in CANDIDATES:
            print(f"SKIP {name}: not in CANDIDATES map")
            continue
        fname = CANDIDATES[name]
        bz2_path = os.path.join(MPS_DIR, fname)
        mps_path = bz2_path[:-4]  # drop .bz2
        nl_path = os.path.join(NL_DIR, f"{name}.nl")
        try:
            if not os.path.exists(bz2_path):
                fetch(fname, bz2_path)
            decompress(bz2_path, mps_path)
            n, m, nnz = dims(mps_path)
            if n > MAX_VARS or m > MAX_CONS or nnz > MAX_NNZ:
                print(f"DEFER {name}: n={n} m={m} nnz={nnz} exceeds cap")
                rows.append((name, n, m, nnz, "deferred(too large)"))
                continue
            mps_to_nl.convert(mps_path, nl_path)
            print(f"OK    {name}: n={n} m={m} nnz={nnz} -> {nl_path}")
            rows.append((name, n, m, nnz, "converted"))
        except Exception as e:  # noqa: BLE001 - want to continue the batch
            print(f"FAIL  {name}: {e}")
            rows.append((name, "?", "?", "?", f"failed: {e}"))
        finally:
            # keep .nl + .bz2 (reproducible); drop the bulky decompressed .mps
            if os.path.exists(mps_path):
                os.remove(mps_path)

    print("\n=== manifest ===")
    print(f"{'instance':22} {'n':>10} {'m':>10} {'nnz':>12}  status")
    for name, n, m, nnz, st in rows:
        print(f"{name:22} {str(n):>10} {str(m):>10} {str(nnz):>12}  {st}")
    converted = sum(1 for r in rows if r[4] == "converted")
    print(f"\nconverted {converted}/{len(rows)} into {NL_DIR}")


if __name__ == "__main__":
    main()
