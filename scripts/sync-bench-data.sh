#!/usr/bin/env bash
# Mirror the benchmark corpus out of Dropbox into a local, unsynced directory.
#
# WHY THIS EXISTS
# ---------------
# The corpus lives in a Dropbox folder. A benchmark run reads ~4.5 GB from it
# *and writes a `.sol` beside every input* (1397 of them at last count — that is
# the documented layout, see benchmarks/README.md). Both halves put a shared,
# externally-scheduled daemon inside the measurement path: the indexer wakes on
# its own schedule, for reasons unrelated to the run, and every timing panel
# then measures Dropbox as well as the solver.
#
# Mirroring to a plain local directory removes that variable. Nothing about the
# corpus is Dropbox-specific — it is read-only input data plus regenerable
# scratch.
#
# Usage:
#   scripts/sync-bench-data.sh            # sync, then verify counts
#   scripts/sync-bench-data.sh --check    # dry run: report what would change
#   BENCH_SRC=... BENCH_DST=... scripts/sync-bench-data.sh
#
# After a successful sync, point the harnesses at the mirror:
#   export POUNCE_BENCH_DATA="$HOME/projects/pounce-bench-data"
# (scripts/bench_data_root.sh prefers it automatically — see that file.)
set -uo pipefail

SRC="${BENCH_SRC:-$HOME/Dropbox/projects/pounce-bench-data}"
DST="${BENCH_DST:-$HOME/projects/pounce-bench-data}"
CHECK=0
[ "${1:-}" = "--check" ] && CHECK=1

[ -d "$SRC" ] || { echo "FATAL: source corpus not found: $SRC"; exit 1; }

# Excluded from the mirror:
#   *.sol      — per-run solver scratch the harness rewrites on every run
#                (368 MB). Inputs, not outputs, are what must be mirrored.
#   .DS_Store  — Finder droppings.
# Deliberately NOT excluded: Maros-Meszaros-answers.{json,pdf}. Those are
# *oracle* files the QP comparison checks against; dropping them would make
# that harness silently unable to verify anything.
EXCLUDES=(--exclude "*.sol" --exclude ".DS_Store")

# macOS ships openrsync / rsync 2.6.9, which has no `--info=`. Passing one makes
# rsync print usage and exit in a way that looks like success while leaving an
# EMPTY mirror — so the flag is not used here, and the file counts below are the
# real check. Never trust the exit code alone.
echo "==> source: $SRC"
echo "==> mirror: $DST"
rsync --version 2>&1 | head -1 | sed 's/^/    rsync: /'

count_inputs() { find "$1" \( -name '*.nl' -o -name '*.cbf' \) -type f 2>/dev/null | wc -l | tr -d ' '; }
count_all()    { find "$1" -type f ! -name '*.sol' ! -name '.DS_Store' 2>/dev/null | wc -l | tr -d ' '; }

SRC_INPUTS="$(count_inputs "$SRC")"
SRC_ALL="$(count_all "$SRC")"
echo "==> source holds $SRC_INPUTS problem inputs (.nl/.cbf), $SRC_ALL mirrored files total"
if [ "$SRC_INPUTS" -eq 0 ]; then
  echo "FATAL: source contains zero problem inputs — refusing to build a vacuous mirror."
  exit 1
fi

if [ "$CHECK" = "1" ]; then
  echo "==> --check: dry run, nothing will be written"
  rsync -a --dry-run --delete "${EXCLUDES[@]}" "$SRC/" "$DST/" | tail -20
  if [ -d "$DST" ]; then
    echo "==> current mirror: $(count_inputs "$DST") inputs, $(count_all "$DST") files"
  else
    echo "==> mirror does not exist yet"
  fi
  exit 0
fi

mkdir -p "$DST"
rsync -a --delete "${EXCLUDES[@]}" "$SRC/" "$DST/"
RC=$?

# Verification. A silent partial mirror combined with a resolver that prefers it
# is the worst outcome available here — every benchmark would pass vacuously
# while looking healthy. Counts, not the exit code, decide success.
DST_INPUTS="$(count_inputs "$DST")"
DST_ALL="$(count_all "$DST")"
echo "==> mirror holds $DST_INPUTS problem inputs, $DST_ALL files total"

if [ "$SRC_INPUTS" != "$DST_INPUTS" ] || [ "$SRC_ALL" != "$DST_ALL" ]; then
  echo "FATAL: mirror is incomplete (rsync exit $RC)."
  echo "       inputs: source=$SRC_INPUTS mirror=$DST_INPUTS"
  echo "       files : source=$SRC_ALL mirror=$DST_ALL"
  echo "       The mirror is left in place for inspection but MUST NOT be used;"
  echo "       unset POUNCE_BENCH_DATA or re-run this script until counts match."
  exit 1
fi

# Sentinel: the resolver refuses a root without it, so a half-synced tree can
# never shadow a good one. Written last, only once the counts agree.
date -u +"%Y-%m-%dT%H:%M:%SZ" > "$DST/.pounce-bench-data"
echo "    wrote sentinel $DST/.pounce-bench-data"

echo "==> OK — $DST_INPUTS inputs verified against source ($(du -sh "$DST" | cut -f1))"
echo
echo "Point the harnesses at it:"
echo "    export POUNCE_BENCH_DATA=\"$DST\""
