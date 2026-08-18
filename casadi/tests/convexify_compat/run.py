#!/usr/bin/env python3
"""Compile the convexify name shim against every CasADi spelling (gh#668).

`convexify_compat.hpp` has two overloads and any one CasADi declares the
helper under exactly one name, so a build against a real CasADi compiles
one of them and leaves the other unchecked. The wheel ships plugins for
casadi 3.6 and 3.7 -- both of which take the *fallback* -- while CI builds
against whatever release is current, so the branch nobody compiles is the
branch most users get. This compiles both, against mock declarations, with
no CasADi installed and nothing linked.

    python3 run.py                 # every compiler found on PATH
    python3 run.py --cxx g++       # just this one

Runs the matrix (declaration state x compiler x -std), checks that the
expected overload is the one that ran, and checks the negative case: with
neither name declared the shim must fail to compile rather than resolve to
something else.
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
PROBE = os.path.join(HERE, "probe.cpp")

# (label, defines, expected selection or None for "must not compile")
CASES = [
    ("casadi <= 3.7.2 (unprefixed)", ["MOCK_OLD_NAME"], "OLD"),
    ("casadi > 3.7.2 (prefixed)", ["MOCK_NEW_NAME"], "NEW"),
    ("both spellings declared", ["MOCK_OLD_NAME", "MOCK_NEW_NAME"], "NEW"),
    ("neither spelling declared", [], None),
]

# The plugin is built as C++17 (see the Makefile), but the shim uses nothing
# past C++11 and a user building the source into their own tree may be on an
# older standard, so hold it to that.
STANDARDS = ["c++11", "c++14", "c++17"]


def compilers(requested):
    if requested:
        return [c for c in requested]
    found = [c for c in ("g++", "clang++") if shutil.which(c)]
    # MSVC builds the Windows wheel, and it is the compiler whose two-phase
    # lookup is likeliest to disagree about a dependent name that is not
    # declared. It is only on PATH inside a developer shell.
    if shutil.which("cl"):
        found.append("cl")
    return found


def command(cxx, std, defines, out, is_msvc):
    if is_msvc:
        # /W4 without /WX: an unrelated MSVC warning in a future toolset
        # should not fail a test about name lookup.
        cmd = ["cl", "/nologo", "/EHsc", "/W4", f"/std:{std}", PROBE,
               f"/Fe:{out}", f"/Fo:{out}.obj"]
        return cmd[:-3] + [f"/D{d}" for d in defines] + cmd[-3:]
    return ([cxx, f"-std={std}", "-Wall", "-Wextra", "-Werror"]
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

    failures = 0
    with tempfile.TemporaryDirectory() as tmp:
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
                        command(cxx, std, defines, out, is_msvc),
                        capture_output=True, text=True, cwd=tmp)
                    what = f"{cxx:8s} {std:6s} {label:30s}"

                    if expect is None:
                        if build.returncode == 0:
                            print(f"{what} FAIL: compiled with neither helper "
                                  "declared; the shim is resolving to "
                                  "something it should not see")
                            failures += 1
                        else:
                            print(f"{what} ok (rejected, as it must be)")
                        continue

                    if build.returncode != 0:
                        print(f"{what} FAIL: did not compile")
                        print((build.stderr or build.stdout).strip()[:2000])
                        failures += 1
                        continue

                    run = subprocess.run([out], capture_output=True, text=True)
                    got = ""
                    for line in run.stdout.splitlines():
                        if line.startswith("selected="):
                            got = line.split("=", 1)[1].strip()
                    if run.returncode != 0 or got != expect:
                        print(f"{what} FAIL: expected {expect}, got "
                              f"{got or '(nothing)'}")
                        print((run.stdout + run.stderr).strip()[:2000])
                        failures += 1
                    else:
                        print(f"{what} ok (selected {got})")

    print()
    if failures:
        print(f"{failures} failure(s)")
        return 1
    print("convexify name shim resolves correctly on every spelling")
    return 0


if __name__ == "__main__":
    sys.exit(main())
