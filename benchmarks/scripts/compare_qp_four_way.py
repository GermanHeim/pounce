#!/usr/bin/env python3
"""Four-way head-to-head on the Maros-Meszaros QP benchmark:

    pounce-convex QP-IPM  vs  Clarabel  vs  pounce NLP (filter-IPM)
                          vs  pounce active-set QP

graded against the published ground-truth optima
(``pounce-bench-data/qp/Maros-Meszaros-answers.json``, DOC 97/6).

Each .mat instance is solved by all four on the *same* source problem:
  - the three pounce engines run live on one freshly generated .nl
    (solver_selection={qp-ipm,nlp,qp-active-set}),
  - Clarabel runs in-process on the matrices.

The active-set column exists because that engine had almost no benchmark
exposure: before this it was graded on 3 hand-constructed problems in the
adversarial battery, while the IPM path was graded on all 138 here. It is the
engine behind `algorithm=active-set-sqp`, and the one that matters for
warm-started sequences (MPC, B&B nodes, homotopy), so degeneracy-rich published
sets are exactly where it should be held to ground truth.
Every objective is compared to the ground-truth OPT; a solve is "correct" when
|obj-opt| <= atol + rtol*max(|obj|,|opt|).

Reuses the assembly/runner helpers in compare_pounce_clarabel.py.

Usage:
  python3 benchmarks/scripts/compare_qp_four_way.py [--limit N]
        [--time-limit SECS] [--rtol R] [--atol A]
Out:
  benchmarks/qp_four_way.json   per-problem records
  benchmarks/qp_four_way.md     side-by-side report
"""
import argparse
import glob
import importlib.util
import json
import math
import os
import tempfile
import time

def _bench_root():
    """Corpus root via the shared resolver (local mirror preferred)."""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from bench_data import bench_data_root

    return bench_data_root()


HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.dirname(HERE)
ROOT = os.path.dirname(BENCH)

# Reuse the existing comparison module (matrix assembly, Clarabel runner,
# .nl generation, pounce runner).
_spec = importlib.util.spec_from_file_location(
    "cmp_pc", os.path.join(HERE, "compare_pounce_clarabel.py"))
cmp_pc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cmp_pc)

GROUND_TRUTH = os.path.expanduser(
    str(_bench_root() / "qp" / "Maros-Meszaros-answers.json"))

POUNCE_OK = cmp_pc.POUNCE_OK            # {"SolveSucceeded","SolvedToAcceptableLevel"}
CLARABEL_OK = cmp_pc.CLARABEL_OK        # {"Solved","AlmostSolved"}


def load_ground_truth():
    with open(GROUND_TRUTH) as fh:
        doc = json.load(fh)
    # key by lowercase name (matches .mat basename.lower())
    return {k.lower(): v["opt"] for k, v in doc["problems"].items()}


def rel_err(obj, opt):
    if obj is None or opt is None:
        return None
    return abs(obj - opt) / max(abs(opt), abs(obj), 1e-10)


def correct(obj, opt, rtol, atol):
    if obj is None or opt is None:
        return False
    return abs(obj - opt) <= atol + rtol * max(abs(obj), abs(opt))


def run(limit, time_limit, rtol, atol):
    gt = load_ground_truth()
    srcs = sorted(glob.glob(os.path.join(BENCH, "qp", "data", "*.mat")),
                  key=os.path.getsize)
    if limit:
        srcs = srcs[:limit]

    rows = []
    hdr = (f"{'problem':<14}{'opt':>14} | "
           f"{'qp-ipm':>11}{'re':>9} | {'clarabel':>11}{'re':>9} | "
           f"{'nlp':>11}{'re':>9} | {'qp-as':>11}{'re':>9} | t(qp/cl/nlp/as)")
    print(f"=== QP four-way ({len(srcs)} problems, time-limit {time_limit:g}s, "
          f"correct: |Δ|<= {atol:g}+{rtol:g}·max) ===")
    print(hdr)
    print("-" * len(hdr))

    for p in srcs:
        name = os.path.basename(p)[:-4]
        key = name.lower()
        opt = gt.get(key)

        # pounce QP-IPM and NLP both run on one generated .nl.
        try:
            with tempfile.NamedTemporaryFile(suffix=".nl", delete=False) as tf:
                nl = tf.name
            cmp_pc.gen_nl_qp(p, nl)
            qp = cmp_pc.run_pounce(nl, "qp-ipm", time_limit)
            nlp = cmp_pc.run_pounce(nl, "nlp", time_limit)
            qpas = cmp_pc.run_pounce(nl, "qp-active-set", time_limit)
            os.path.exists(nl) and os.unlink(nl)
        except Exception as e:
            err = {"status": f"GenError:{type(e).__name__}", "objective": None,
                   "iterations": None, "solve_time": None}
            qp, nlp, qpas = dict(err), dict(err), dict(err)

        # Clarabel on matrices.
        try:
            P, q, G, b, cones, n, m, off = cmp_pc.load_qp(p)
            cl = cmp_pc.solve_clarabel(P, q, G, b, cones, off, time_limit)
        except Exception as e:
            cl = {"status": f"LoadError:{type(e).__name__}", "objective": None,
                  "iterations": None, "solve_time": None, "wall": None}
            n = m = None

        def grade(rec, ok_set):
            o = rec.get("objective")
            solved = rec.get("status") in ok_set
            re_ = rel_err(o, opt)
            return {
                "status": rec.get("status"),
                "objective": o,
                "iterations": rec.get("iterations"),
                "solve_time": rec.get("solve_time"),
                "rel_err": re_,
                "solved": solved,
                "correct": solved and correct(o, opt, rtol, atol),
            }

        row = {
            "name": name, "n": n, "m": m, "opt": opt,
            "qp_ipm": grade(qp, POUNCE_OK),
            "clarabel": grade(cl, CLARABEL_OK),
            "nlp": grade(nlp, POUNCE_OK),
            "qp_active_set": grade(qpas, POUNCE_OK),
        }
        rows.append(row)

        def cell(g):
            o = g["objective"]
            os_ = f"{o:.4e}" if o is not None else g["status"][:11]
            re_ = g["rel_err"]
            rs = f"{re_:.1e}" if re_ is not None else "  n/a"
            mark = "✓" if g["correct"] else ("·" if g["solved"] else "✗")
            return f"{os_:>11}{rs:>8}{mark}"

        ts = lambda g: g["solve_time"] if g["solve_time"] is not None else float("nan")
        opts = f"{opt:.4e}" if opt is not None else "n/a"
        print(f"{name:<14}{opts:>14} | {cell(row['qp_ipm'])} | "
              f"{cell(row['clarabel'])} | {cell(row['nlp'])} | "
              f"{cell(row['qp_active_set'])} | "
              f"{ts(row['qp_ipm']):.2f}/{ts(row['clarabel']):.2f}/"
              f"{ts(row['nlp']):.2f}/{ts(row['qp_active_set']):.2f}",
              # Flush per problem. Redirected to a file, stdout is
              # block-buffered, so progress lags a full 8KB (~55 problems)
              # behind reality — which reads exactly like a hang on the tail
              # of a long sweep, and cost one killed run to learn.
              flush=True)

    return rows


def _binary_provenance():
    """Which binary produced this report — the first line of `pounce --about`.

    Recorded because a benchmark report without it is unfalsifiable: you
    cannot tell a real regression from a stale binary, which is exactly how
    the previous report's NLP column drifted unnoticed.
    """
    import subprocess
    try:
        out = subprocess.run([cmp_pc.POUNCE_BIN, "--about"], capture_output=True,
                             text=True, timeout=30).stdout.strip().splitlines()
        return out[0] if out else "unknown build"
    except Exception:
        return "unknown build"


def geomean(xs):
    xs = [x for x in xs if x is not None and x > 0]
    return math.exp(sum(map(math.log, xs)) / len(xs)) if xs else None


def summarize(rows, rtol, atol):
    N = len(rows)
    have_gt = [r for r in rows if r["opt"] is not None]
    out = ["# POUNCE-QP vs Clarabel vs POUNCE-NLP vs POUNCE active-set — "
           "Maros-Meszaros QP benchmark",
           "",
           f"{N} problems; {len(have_gt)} with ground-truth optima "
           "(DOC 97/6, BPMPD reference). A solve is **correct** when "
           f"`|obj-opt| <= {atol:g} + {rtol:g}·max(|obj|,|opt|)`.",
           "",
           f"Produced by **{_binary_provenance()}**. Numbers here are only "
           "comparable to another run of the same binary: the previous "
           "committed report was six weeks stale, and the NLP column moved "
           "129/138 → 105/138 across that gap on nothing but binary drift. "
           "Rebuild before regenerating (`cargo build --release --bin pounce`).",
           ""]

    def block(key, label):
        solved = [r for r in rows if r[key]["solved"]]
        cor = [r for r in have_gt if r[key]["correct"]]
        # wrong = solved (by its own status) but objective wrong vs ground truth
        wrong = [r for r in have_gt if r[key]["solved"] and not r[key]["correct"]]
        res = [r[key]["rel_err"] for r in cor if r[key]["rel_err"] is not None]
        med = sorted(res)[len(res) // 2] if res else None
        L = [f"### {label}",
             f"- Solved (own status): **{len(solved)}/{N}**",
             f"- Correct vs ground truth: **{len(cor)}/{len(have_gt)}**",
             f"- Solved-but-wrong (status OK, obj off): **{len(wrong)}**",
             (f"- Median rel-err on correct solves: {med:.1e}" if med is not None else ""),
             ]
        if wrong:
            L.append("- Wrong objectives: " + ", ".join(
                f"{r['name']}(re={r[key]['rel_err']:.1e})" for r in wrong[:20])
                + (" …" if len(wrong) > 20 else ""))
        L.append("")
        return "\n".join(x for x in L if x)

    out.append(block("qp_ipm", "pounce QP-IPM (solver_selection=qp-ipm)"))
    out.append(block("clarabel", "Clarabel"))
    out.append(block("nlp", "pounce NLP (solver_selection=nlp)"))
    out.append(block("qp_active_set",
                     "pounce active-set QP (solver_selection=qp-active-set)"))

    # Speed: geomean over the set where ALL THREE produced a correct solve.
    # `is not None`, not truthiness: a solve that finishes in under a
    # millisecond reports 0.0, which is falsy, so the old test silently
    # dropped the fastest solves and biased the geomean upward. It excluded
    # every one of the active-set engine's correct solves.
    def _timed(r, *keys):
        return all(r[k]["solve_time"] is not None for k in keys)

    allc = [r for r in have_gt
            if r["qp_ipm"]["correct"] and r["clarabel"]["correct"] and r["nlp"]["correct"]
            and _timed(r, "qp_ipm", "clarabel", "nlp")]
    if allc:
        gq = geomean([r["qp_ipm"]["solve_time"] for r in allc])
        gc = geomean([r["clarabel"]["solve_time"] for r in allc])
        gn = geomean([r["nlp"]["solve_time"] for r in allc])
        out += [f"### Speed (geomean over {len(allc)} all-three-correct problems)",
                f"- pounce QP-IPM : {gq:.3f}s",
                f"- Clarabel      : {gc:.3f}s",
                f"- pounce NLP    : {gn:.3f}s",
                f"- QP-IPM vs Clarabel: {gq/gc:.2f}×  "
                f"(Clarabel {'faster' if gq>gc else 'slower'})",
                f"- QP-IPM vs NLP     : {gn/gq:.2f}×  "
                f"(QP-IPM {'faster' if gn>gq else 'slower'})",
                "",
                "Basis note: membership is now `solve_time is not None` "
                "rather than a truthiness test. The old form dropped any solve "
                "finishing in under a millisecond (0.0 is falsy), which biased "
                "the geomean upward and excluded *every* correct active-set "
                "solve. Counts here may therefore differ slightly from reports "
                "generated before that fix. The active-set engine is timed "
                "separately below, on its own pairwise basis, so adding it "
                "does not move these three figures.",
                ""]

    # Active-set vs QP-IPM, on the set where BOTH are correct. A separate
    # pairwise basis rather than an all-four intersection: the active-set
    # engine is the newcomer here, and folding it into the shared set would
    # silently move the three long-standing numbers above.
    both = [r for r in have_gt
            if r["qp_ipm"]["correct"] and r["qp_active_set"]["correct"]
            and _timed(r, "qp_ipm", "qp_active_set")
            and r["qp_active_set"]["solve_time"] > 0
            and r["qp_ipm"]["solve_time"] > 0]
    if not both:
        # Do not synthesize a number here. The active-set path currently
        # writes `total_wallclock_time_secs: 0.0` into `--json-output` while
        # actually taking real time (measured: 0.568s on AUG2D, where qp-ipm
        # correctly reports 0.678s). Falling back to the harness's own
        # wall-clock would include process startup and so compare a different
        # quantity than the figure reported for the other engines — an
        # apples-to-oranges number is worse than an absent one.
        out += ["### Speed — active-set vs QP-IPM",
                "",
                "**Not reported: the active-set path does not populate "
                "`total_wallclock_time_secs` in `--json-output`** (it emits "
                "`0.0` regardless of actual runtime, while `qp-ipm` on the "
                "same problem reports correctly). Any geomean built on that "
                "field would read as \"instantaneous\" and be meaningless. "
                "Tracked separately; this section will populate once the "
                "field is correct.",
                ""]
    if both:
        gi = geomean([r["qp_ipm"]["solve_time"] for r in both])
        ga = geomean([r["qp_active_set"]["solve_time"] for r in both])
        out += [f"### Speed — active-set vs QP-IPM "
                f"(geomean over {len(both)} both-correct problems)",
                f"- pounce QP-IPM     : {gi:.3f}s",
                f"- pounce active-set : {ga:.3f}s",
                f"- ratio: {ga/gi:.2f}×  "
                f"(active-set {'slower' if ga>gi else 'faster'})",
                "",
                "Cold-start timings. The active-set engine's design point is "
                "*warm*-started sequences, where it carries the working set "
                "across solves — a single cold solve is the case it is least "
                "suited to, so read this as a floor, not a verdict.",
                ""]

    # Where ground truth discriminates: pounce-QP correct but another solver wrong.
    def st(g):
        if g["correct"]:
            return "✓"
        if g["solved"]:
            return f"off re={g['rel_err']:.1e}" if g["rel_err"] is not None else "off"
        return g["status"] or "—"

    disc = [r for r in have_gt if r["qp_ipm"]["correct"]
            and (not r["clarabel"]["correct"] or not r["nlp"]["correct"]
                 or not r["qp_active_set"]["correct"])]
    if disc:
        out.append("### Problems where pounce-QP is correct but another solver is not")
        out.append("")
        out.append("| problem | opt | clarabel | nlp | qp-active-set |")
        out.append("|---|---|---|---|---|")
        for r in disc:
            out.append(f"| {r['name']} | {r['opt']:.6g} | "
                       f"{st(r['clarabel'])} | {st(r['nlp'])} | "
                       f"{st(r['qp_active_set'])} |")
        out.append("")

    # The reason this column was added: where does the active-set engine
    # disagree with ground truth? Listed on its own so it is not buried in the
    # table above, and so a "solved but wrong" is visibly distinct from a
    # clean failure to solve — the first is the dangerous kind.
    as_bad = [r for r in have_gt if not r["qp_active_set"]["correct"]]
    if as_bad:
        wrong = [r for r in as_bad if r["qp_active_set"]["solved"]]
        out.append(f"### Active-set QP: {len(as_bad)} of {len(have_gt)} "
                   "not matching ground truth")
        out.append("")
        out.append(f"Of these, **{len(wrong)}** report a successful status with "
                   "a wrong objective (the dangerous kind); the rest fail "
                   "visibly.")
        out.append("")
        out.append("| problem | n | m | opt | status | objective | rel-err |")
        out.append("|---|---|---|---|---|---|---|")
        for r in sorted(as_bad, key=lambda r: (not r["qp_active_set"]["solved"],
                                               r["name"])):
            g = r["qp_active_set"]
            o = f"{g['objective']:.6g}" if g["objective"] is not None else "—"
            re_ = f"{g['rel_err']:.1e}" if g["rel_err"] is not None else "—"
            out.append(f"| {r['name']} | {r['n'] if r['n'] is not None else '—'} "
                       f"| {r['m'] if r['m'] is not None else '—'} "
                       f"| {r['opt']:.6g} | {g['status'] or '—'} | {o} | {re_} |")
        out.append("")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--time-limit", type=float, default=120.0)
    ap.add_argument("--rtol", type=float, default=1e-4)
    ap.add_argument("--atol", type=float, default=1e-5)
    args = ap.parse_args()

    rows = run(args.limit, args.time_limit, args.rtol, args.atol)
    jpath = os.path.join(BENCH, "qp_four_way.json")
    with open(jpath, "w") as fh:
        json.dump(rows, fh, indent=2)
    md = summarize(rows, args.rtol, args.atol)
    mpath = os.path.join(BENCH, "qp_four_way.md")
    with open(mpath, "w") as fh:
        fh.write(md + "\n")
    print("\n" + md)
    print(f"\nwrote {os.path.relpath(jpath, ROOT)} and {os.path.relpath(mpath, ROOT)}")


if __name__ == "__main__":
    main()
