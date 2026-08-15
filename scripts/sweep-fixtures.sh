#!/usr/bin/env bash
# Trajectory sweep over the CLI fixture corpus.
#
# Runs every fixture in crates/pounce-cli/tests/fixtures and records status,
# objective AND iteration count, one line per model, sorted and diffable.
#
# WHY THIS EXISTS (pounce gh#592, and gh#544 before it). The CLI test suite
# asserts *status* and *objective*. It does not assert trajectory length, and
# a change to the step computation can leave both of those untouched while
# taking four times as many iterations to get there. That is exactly what
# gh#544 did to pooling_rt2stp -- 206 -> 812 iterations -- and nothing in the
# suite could see it; it surfaced three days before the 0.10.0 release as a
# wall-clock timeout, was misattributed, and the cap was raised. The defect it
# was a symptom of shipped, and came back as gh#592.
#
# So: any change that reroutes WHICH correction the solver reaches for, or
# reorders/rescales the steps it takes, needs this sweep. "It cannot produce a
# wrong answer" is not the relevant safety property -- trajectory changes are
# invisible to the answer.
#
# Usage:
#   scripts/sweep-fixtures.sh <pounce-binary> <outfile> [extra solver opts...]
#
#   git stash && cargo build --release && cp target/release/pounce /tmp/p-base
#   git stash pop && cargo build --release
#   scripts/sweep-fixtures.sh /tmp/p-base            /tmp/base.txt
#   scripts/sweep-fixtures.sh target/release/pounce  /tmp/new.txt
#   diff /tmp/base.txt /tmp/new.txt
#
# An empty diff is the expected result for a change that is not meant to move
# the corpus. Every line that does move should be explainable before merge --
# not after a user reports it.
set -uo pipefail

BIN="${1:?usage: sweep-fixtures.sh <pounce-binary> <outfile> [opts...]}"
OUT="${2:?usage: sweep-fixtures.sh <pounce-binary> <outfile> [opts...]}"
shift 2

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FX="$ROOT/crates/pounce-cli/tests/fixtures"
[ -d "$FX" ] || { echo "no fixture dir at $FX" >&2; exit 2; }

W=$(mktemp -d)
trap 'rm -rf "$W"' EXIT
: > "$OUT"

for f in "$FX"/*.nl; do
  n=$(basename "$f" .nl)
  # A fixture that hangs must not hang the sweep; it shows up as NO_JSON.
  timeout 300 "$BIN" "$f" "$W/$n.sol" --json-output "$W/$n.json" "$@" \
    >/dev/null 2>&1
  if [ -f "$W/$n.json" ]; then
    python3 - "$W/$n.json" "$n" >> "$OUT" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
s = d.get("solution", {})
st = d.get("statistics", {})
obj = s.get("objective")
print("%-40s %-32s it=%-6s obj=%s" % (
    sys.argv[2],
    s.get("status"),
    st.get("iteration_count", "?"),
    "%.10g" % obj if obj is not None else "none",
))
PY
  else
    printf '%-40s NO_JSON\n' "$n" >> "$OUT"
  fi
done

sort -o "$OUT" "$OUT"
echo "swept $(wc -l < "$OUT" | tr -d ' ') fixtures -> $OUT"
