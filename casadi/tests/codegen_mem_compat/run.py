#!/usr/bin/env python3
"""Compile the plugin's codegen memory request against every CasADi shape (gh#782).

`PounceInterface::codegen_needs_mem()` cannot be marked `override`: CasADi
3.7 and earlier declare no such virtual, and `override` there is a hard
error. So the compiler does not check that it binds to anything, and the
consequence of it not binding is silent -- CasADi stops emitting the
`<name>_mem` array while the plugin's generated bodies keep referring to
it, and the failure surfaces as a C compile error in generated code, on
whichever CasADi the user happens to have.

That is exactly the state CasADi 3.8 introduced, and the state the plugin
was in when 3.8.0 shipped. On any one machine only one of the shapes below
is compiled, so this compiles all of them against mock declarations, with
no CasADi installed and nothing linked.

    python3 run.py                 # every compiler found on PATH
    python3 run.py --cxx g++       # just this one

The declaration under test is extracted from the plugin source rather than
copied, so this test follows edits to the plugin.
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
PROBE = os.path.join(HERE, "probe.cpp")
PLUGIN = os.path.normpath(os.path.join(HERE, "..", "..",
                                       "casadi_nlpsol_pounce.cpp"))

# (label, defines, expected base_call)
#
# `base_call` is what CasADi's CodeGenerator sees, asking through a
# base-class pointer. `true` is the only answer that emits the memory
# array.
CASES = [
    ("casadi <= 3.7 (no such virtual)", [], "absent"),
    ("casadi >= 3.8 (virtual present)", ["MOCK_HAS_NEEDS_MEM"], "true"),
    ("signature drift", ["MOCK_NEEDS_MEM_MISMATCH"], "false"),
]

# The plugin builds as C++17 (see the Makefile). The declaration uses
# nothing past C++11, and a downstream tree may be on an older standard.
STANDARDS = ["c++11", "c++14", "c++17"]


def extract_member():
    """The plugin's own declaration of the memory request, verbatim.

    Deliberately strict: this test is worth nothing if it silently falls
    back to a copy, so anything it cannot identify is an error naming the
    file rather than a default.
    """
    with open(PLUGIN, encoding="utf-8") as fh:
        lines = fh.readlines()

    hits = [ln for ln in lines
            if re.search(r"\bbool\s+codegen_needs_mem\s*\(", ln)]
    if len(hits) != 1:
        raise SystemExit(
            f"error: expected exactly one declaration of codegen_needs_mem "
            f"in {PLUGIN}, found {len(hits)}. If it moved or grew a body "
            f"across several lines, teach this extractor about it -- do not "
            f"copy it here.")

    decl = hits[0]
    if "{" not in decl or "}" not in decl:
        raise SystemExit(
            f"error: the declaration in {PLUGIN} is not a one-line inline "
            f"definition; this extractor cannot carry it verbatim.")
    return decl


def compilers(requested):
    if requested:
        return list(requested)
    found = [c for c in ("g++", "clang++") if shutil.which(c)]
    # MSVC builds the Windows wheel. It is on PATH only inside a developer
    # shell.
    if shutil.which("cl"):
        found.append("cl")
    return found


def command(cxx, std, defines, incdir, out, is_msvc):
    if is_msvc:
        # /W4 without /WX: an unrelated warning in a future toolset should
        # not fail a test about virtual binding.
        return (["cl", "/nologo", "/EHsc", "/W4", f"/std:{std}",
                 f"/I{incdir}"]
                + [f"/D{d}" for d in defines]
                + [PROBE, f"/Fe:{out}", f"/Fo:{out}.obj"])
    return ([cxx, f"-std={std}", "-Wall", "-Wextra", "-Werror", "-I", incdir]
            + [f"-D{d}" for d in defines] + [PROBE, "-o", out])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cxx", action="append", default=[],
                    help="compiler to use (repeatable); default: all found")
    args = ap.parse_args()

    found = compilers(args.cxx)
    if not found:
        print("error: no C++ compiler found (looked for g++, clang++, cl)",
              file=sys.stderr)
        return 2

    member = extract_member()
    print(f"declaration under test, from {os.path.relpath(PLUGIN, HERE)}:")
    print(f"    {member.strip()}\n")

    failures = 0
    with tempfile.TemporaryDirectory() as tmp:
        with open(os.path.join(tmp, "plugin_member.inc"), "w",
                  encoding="utf-8") as fh:
            fh.write(member)

        for cxx in found:
            # exact match: "clang++" also begins with "cl"
            is_msvc = os.path.basename(cxx).lower() in ("cl", "cl.exe")
            for std in STANDARDS:
                if is_msvc and std == "c++11":
                    continue  # MSVC has no /std:c++11; c++14 is its floor
                for label, defines, expect in CASES:
                    out = os.path.join(tmp, "probe.exe")
                    for stale in (out, out + ".obj"):
                        if os.path.exists(stale):
                            os.remove(stale)
                    build = subprocess.run(
                        command(cxx, std, defines, tmp, out, is_msvc),
                        capture_output=True, text=True, cwd=tmp)
                    what = f"{cxx:8s} {std:6s} {label:33s}"

                    if build.returncode != 0:
                        # The `casadi <= 3.7` case failing to compile is the
                        # signature this test was written for: it is what an
                        # `override` on the member looks like.
                        print(f"{what} FAIL: did not compile")
                        print((build.stderr or build.stdout).strip()[:2000])
                        failures += 1
                        continue

                    run = subprocess.run([out], capture_output=True, text=True)
                    got = {}
                    for line in run.stdout.splitlines():
                        if "=" in line:
                            k, v = line.split("=", 1)
                            got[k.strip()] = v.strip()

                    if run.returncode != 0 or got.get("base_call") != expect:
                        print(f"{what} FAIL: expected base_call={expect}, got "
                              f"{got.get('base_call') or '(nothing)'}")
                        print((run.stdout + run.stderr).strip()[:2000])
                        failures += 1
                    elif got.get("direct_call") != "true":
                        print(f"{what} FAIL: the member itself answered "
                              f"{got.get('direct_call')}, not true")
                        failures += 1
                    else:
                        print(f"{what} ok (base_call={expect})")

    print()
    if failures:
        print(f"{failures} failure(s)")
        return 1
    print("the codegen memory request binds on every CasADi shape")
    return 0


if __name__ == "__main__":
    sys.exit(main())
