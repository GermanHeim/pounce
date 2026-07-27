"""Resolve the benchmark corpus root, preferring a local mirror over Dropbox.

Why this exists
---------------
The corpus historically lives in a Dropbox folder. A benchmark run reads ~4.5 GB
from it *and* writes a ``.sol`` beside every input (the documented layout — see
``benchmarks/README.md``). Both halves put a shared, externally-scheduled daemon
inside the measurement path, so a timing panel measures Dropbox's indexer as
well as the solver.

``scripts/sync-bench-data.sh`` builds an unsynced mirror; this module is the one
place that decides which root to use.

Resolution order
----------------
1. ``$POUNCE_BENCH_DATA``     — explicit override, always wins if valid
2. ``~/projects/pounce-bench-data``  — the local mirror
3. ``~/Dropbox/projects/pounce-bench-data`` — original, so a fresh checkout works

**A root only counts if it actually holds the corpus.** An empty or half-synced
tree that shadows a good one makes every benchmark silently vacuous — the worst
outcome available here, because it looks like success. ``_looks_like_corpus``
is what prevents it.
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path

#: Written by ``scripts/sync-bench-data.sh`` only after its file counts match
#: the source, so its presence means "this mirror was verified complete".
SENTINEL = ".pounce-bench-data"

#: Suites used as a fallback liveness probe for roots without a sentinel (the
#: Dropbox original never had one). Any single hit is enough.
_PROBE_DIRS = ("vanderbei/nl", "mittelmann/nl", "qp/nl", "lp/nl")

#: Path prefixes treated as "inside a sync folder".
_SYNCED_MARKERS = ("Dropbox", "Google Drive", "OneDrive", "iCloud")


def _looks_like_corpus(root: Path) -> bool:
    """True if `root` actually contains problem inputs.

    Checked cheaply: the verified-mirror sentinel, else at least one known
    suite directory holding at least one ``.nl``. A recursive count would be
    correct too but walks 4.5 GB on every call.
    """
    if not root.is_dir():
        return False
    if (root / SENTINEL).is_file():
        return True
    for probe in _PROBE_DIRS:
        d = root / probe
        if d.is_dir() and any(d.glob("*.nl")):
            return True
    return False


def candidate_roots() -> list[Path]:
    """The resolution order, before validity filtering."""
    roots: list[Path] = []
    env = os.environ.get("POUNCE_BENCH_DATA")
    if env:
        roots.append(Path(env).expanduser())
    roots.append(Path.home() / "projects" / "pounce-bench-data")
    roots.append(Path.home() / "Dropbox" / "projects" / "pounce-bench-data")
    return roots


def bench_data_root(required: bool = True) -> Path | None:
    """Return the first candidate root that actually holds the corpus.

    Raises ``FileNotFoundError`` when nothing valid is found and `required`,
    rather than returning a path that would make every downstream glob empty.
    """
    tried = candidate_roots()
    for root in tried:
        if _looks_like_corpus(root):
            return root
    if not required:
        return None
    listed = "\n  ".join(str(p) for p in tried)
    raise FileNotFoundError(
        "no benchmark corpus found. Tried:\n  "
        + listed
        + "\n\nBuild the local mirror with:\n"
        "    scripts/sync-bench-data.sh\n"
        "or point POUNCE_BENCH_DATA at an existing corpus."
    )


def corpus_is_synced(root: Path | None = None) -> bool:
    """True if the resolved corpus sits inside a file-sync folder.

    A timing harness should refuse or warn on this: the sync daemon wakes on
    its own schedule and its I/O lands in the measurement.
    """
    if root is None:
        root = bench_data_root(required=False)
    if root is None:
        return False
    return any(marker in str(root) for marker in _SYNCED_MARKERS)


def warn_if_synced(root: Path | None = None) -> bool:
    """Warn when timing off a synced corpus. Returns True if a warning fired."""
    if root is None:
        root = bench_data_root(required=False)
    if root is not None and corpus_is_synced(root):
        warnings.warn(
            f"benchmark corpus is inside a sync folder ({root}); timings will "
            "include the sync daemon's I/O and are not comparable across runs. "
            "Build a local mirror with scripts/sync-bench-data.sh and export "
            "POUNCE_BENCH_DATA to it.",
            RuntimeWarning,
            stacklevel=2,
        )
        return True
    return False


if __name__ == "__main__":  # pragma: no cover - manual probe
    root = bench_data_root(required=False)
    print(f"resolved root : {root}")
    print(f"synced folder : {corpus_is_synced(root)}")
    if root is not None:
        n = sum(1 for _ in root.rglob("*.nl"))
        print(f"problem inputs: {n} .nl")
        if n == 0:
            raise SystemExit("FAIL: resolved a root with zero inputs")
