#!/usr/bin/env bash
# Shell counterpart of benchmarks/bench_data.py — resolve the corpus root,
# preferring a local mirror over Dropbox. Source it, then use $BENCH_DATA_ROOT:
#
#   . "$(git rev-parse --show-toplevel)/scripts/bench_data_root.sh"
#   ls "$BENCH_DATA_ROOT/vanderbei/nl"
#
# Order: $POUNCE_BENCH_DATA -> ~/projects/pounce-bench-data -> ~/Dropbox/...
# A root only counts if it actually holds the corpus; an empty or half-synced
# tree that shadows a good one would make every benchmark silently vacuous.

_bench_looks_like_corpus() {
    [ -d "$1" ] || return 1
    [ -f "$1/.pounce-bench-data" ] && return 0
    for probe in vanderbei/nl mittelmann/nl qp/nl lp/nl; do
        [ -d "$1/$probe" ] || continue
        [ -n "$(find "$1/$probe" -maxdepth 1 -name '*.nl' -print -quit 2>/dev/null)" ] && return 0
    done
    return 1
}

bench_data_root() {
    for cand in "${POUNCE_BENCH_DATA:-}" "$HOME/projects/pounce-bench-data" \
                "$HOME/Dropbox/projects/pounce-bench-data"; do
        [ -n "$cand" ] || continue
        if _bench_looks_like_corpus "$cand"; then echo "$cand"; return 0; fi
    done
    return 1
}

# True when the resolved corpus sits in a sync folder — its daemon's I/O lands
# in any timing measured against it.
bench_corpus_is_synced() {
    case "${1:-$BENCH_DATA_ROOT}" in
        *Dropbox*|*"Google Drive"*|*OneDrive*|*iCloud*) return 0 ;;
        *) return 1 ;;
    esac
}

bench_warn_if_synced() {
    if bench_corpus_is_synced "${1:-$BENCH_DATA_ROOT}"; then
        echo "WARNING: benchmark corpus is inside a sync folder (${1:-$BENCH_DATA_ROOT});" >&2
        echo "         timings include the sync daemon's I/O. Run scripts/sync-bench-data.sh" >&2
        echo "         and export POUNCE_BENCH_DATA to the mirror." >&2
        return 0
    fi
    return 1
}

BENCH_DATA_ROOT="$(bench_data_root)" || {
    echo "FATAL: no benchmark corpus found (tried \$POUNCE_BENCH_DATA," >&2
    echo "       ~/projects/pounce-bench-data, ~/Dropbox/projects/pounce-bench-data)." >&2
    echo "       Build one with scripts/sync-bench-data.sh" >&2
}
export BENCH_DATA_ROOT
