#!/usr/bin/env bash
# Combined Rust + Python coverage for pounce.
#
# WHY THIS EXISTS
# ---------------
# `cargo llvm-cov --workspace` instruments and runs only the *Rust* test suite.
# Large parts of pounce are exercised solely through the Python extension
# (`pounce._pounce`) or through the CLI driven by pytest/pyomo. Those paths show
# up as 0% in a Rust-only report, which makes the report actively misleading as
# a "what is under-tested?" signal: it invents gaps that are in fact covered.
#
# `cargo llvm-cov report` cannot fix this on its own — it has no `--object`
# flag, so it can never attribute the extension module's profile data. This
# script therefore drives `llvm-profdata` / `llvm-cov` directly and passes every
# instrumented artifact (Rust test binaries + CLI + the installed .so) as an
# explicit `-object`.
#
# THE ONE RULE: build everything under instrumentation FIRST, then run, then
# report. Rebuilding any artifact between profiling and reporting changes its
# coverage-mapping hash and silently yields a 0% report.
#
# Usage:
#   scripts/coverage-combined.sh              # full run, summary to stdout
#   scripts/coverage-combined.sh --quick      # skip the slow pytest suite
#   COV_OUT=/tmp/cov scripts/coverage-combined.sh
#
# Outputs (under $COV_OUT, default target/coverage-combined):
#   summary.txt   per-file table, all sources
#   core.txt      numerical-core files ranked by uncovered regions
#   lcov.info     for editor/CI consumption
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO" || exit 1
COV_OUT="${COV_OUT:-$REPO/target/coverage-combined}"
PROFDIR="$COV_OUT/profraw"
QUICK=0
[ "${1:-}" = "--quick" ] && QUICK=1

LLVMBIN="$(dirname "$(rustc --print target-libdir)")/bin"
for t in llvm-profdata llvm-cov; do
  [ -x "$LLVMBIN/$t" ] || { echo "FATAL: $t not found in $LLVMBIN"; echo "  rustup component add llvm-tools-preview"; exit 1; }
done

rm -rf "$COV_OUT"; mkdir -p "$PROFDIR"
export RUSTFLAGS="-C instrument-coverage ${RUSTFLAGS:-}"
export LLVM_PROFILE_FILE="$PROFDIR/pounce-%p-%m.profraw"

echo "==> [1/5] building instrumented Rust test binaries"
# --no-run so nothing executes before every artifact exists.
cargo test --workspace --no-run --message-format=json 2>/dev/null \
  | python3 -c '
import json,sys
for line in sys.stdin:
    try: m=json.loads(line)
    except ValueError: continue
    exe=m.get("executable")
    if exe: print(exe)
' | sort -u > "$COV_OUT/objects.txt"
echo "    $(wc -l < "$COV_OUT/objects.txt") test binaries"

echo "==> [2/5] building instrumented CLI + Python extension"
cargo build --workspace >/dev/null 2>&1
# Absolute path: cargo's JSON already reports absolute executables, so a
# relative entry here would hand llvm-cov the same object twice.
[ -x target/debug/pounce ] && echo "$REPO/target/debug/pounce" >> "$COV_OUT/objects.txt"
(cd python && maturin develop --release 2>&1 | tail -1)
SO="$(python -c 'import pounce._pounce as m; print(m.__file__)')"
echo "$SO" >> "$COV_OUT/objects.txt"
echo "    extension: $SO"

# Run a pytest suite, surfacing which tests failed. Piping straight to
# `tail -1` keeps the output short but hides the FAILED lines, which turns a
# single failure into a full re-run to find out what broke.
run_pytest() {
  local label="$1"; shift
  local log="$COV_OUT/pytest-$label.log"
  "$@" > "$log" 2>&1
  grep -E '^(FAILED|ERROR)' "$log" | sed 's/^/    /'
  tail -1 "$log" | sed 's/^/    /'
}

echo "==> [3/5] running Rust tests"
cargo test --workspace >/dev/null 2>&1
echo "==> [4/5] running Python + pyomo suites"
if [ "$QUICK" = "1" ]; then
  (cd python && run_pytest python python -m pytest tests/ -q -x -k "problem or minimize")
else
  (cd python && run_pytest python python -m pytest tests/ -q)
  run_pytest pyomo python -m pytest pyomo-pounce/tests -q
fi

echo "==> [5/5] merging $(find "$PROFDIR" -name '*.profraw' | wc -l | tr -d ' ') profraw files and reporting"
"$LLVMBIN/llvm-profdata" merge -sparse "$PROFDIR"/*.profraw -o "$COV_OUT/merged.profdata" || exit 1

OBJARGS=(); while read -r o; do [ -e "$o" ] && OBJARGS+=(-object "$o"); done < "$COV_OUT/objects.txt"
IGNORE='(/\.cargo/registry|/rustc/|/library/std|/tests?/|_test\.rs$)'

"$LLVMBIN/llvm-cov" report "${OBJARGS[@]}" \
  -instr-profile="$COV_OUT/merged.profdata" \
  -ignore-filename-regex="$IGNORE" > "$COV_OUT/summary.txt" 2>/dev/null

"$LLVMBIN/llvm-cov" export "${OBJARGS[@]}" \
  -instr-profile="$COV_OUT/merged.profdata" \
  -ignore-filename-regex="$IGNORE" -format=lcov > "$COV_OUT/lcov.info" 2>/dev/null

# Numerical core only: where a coverage gap can hide a silently-wrong answer.
# Diagnostics/dump/binaries are deliberately excluded — low coverage there is
# real but cannot corrupt a solve.
python3 - "$COV_OUT" <<'PY'
import sys, os
out = sys.argv[1]
CORE = ('pounce-algorithm','pounce-qp','pounce-convex','pounce-nlp','pounce-linsol',
        'pounce-feral','pounce-restoration','pounce-presolve','pounce-l1penalty',
        'pounce-sensitivity','pounce-linalg','pounce-common','pounce-nl')
SKIP = ('debug.rs','iter_dump.rs','iterate_dump.rs','console.rs','output.rs')
# Examples/benches/bins are not shipped solve paths: 0% there is expected and
# would otherwise dominate the ranking and crowd out real gaps.
SKIPDIR = ('/examples/','/benches/','/bin/')
rows=[]
for l in open(os.path.join(out,'summary.txt')):
    p=l.split()
    if len(p)<10 or 'crates/' not in p[0]: continue
    try: rows.append((p[0], int(p[1]), int(p[2]), float(p[3].rstrip('%'))))
    except ValueError: pass
core=[r for r in rows
      if any(f'/{c}/' in r[0] for c in CORE)
      and not any(r[0].endswith(s) for s in SKIP)
      and not any(d in r[0] for d in SKIPDIR)]
with open(os.path.join(out,'core.txt'),'w') as f:
    if core:
        tot=sum(r[1] for r in core); mis=sum(r[2] for r in core)
        f.write(f"numerical core: {100*(1-mis/tot):.1f}% region coverage "
                f"({mis} uncovered of {tot}) across {len(core)} files\n\n")
        f.write(f"{'file':64s} {'regions':>8s} {'uncov':>7s} {'cov%':>7s}\n")
        for r in sorted(core, key=lambda r:-r[2])[:30]:
            f.write(f"{r[0].replace('crates/',''):64s} {r[1]:8d} {r[2]:7d} {r[3]:6.1f}%\n")
    else:
        f.write("no core rows parsed - check summary.txt\n")
print(open(os.path.join(out,'core.txt')).read())
PY

echo "wrote: $COV_OUT/{summary.txt,core.txt,lcov.info}"
echo
echo "NOTE: this leaves python/pounce/_pounce.abi3.so built WITH instrumentation"
echo "      (slower, and timing-sensitive tests may fail). Restore with:"
echo "      cd python && maturin develop --release"
