"""Console-script shim for the bundled `pounce` CLI binary.

The wheel ships the Rust `pounce` binary inside `pounce/bin/`. The
`[project.scripts]` entry in pyproject.toml points at `main` below,
which transfers control to that binary. The Python interpreter
disappears from the process tree on Unix (os.execv); on Windows we
fall back to a subprocess + propagated exit code because os.execv
there spawns a child and returns control to the shell, which breaks
signal handling.

`maturin develop` builds only the extension module, so in a source
checkout `pounce/bin/` is empty and the shim has nothing to exec. That
used to be the end of it: every CLI invocation failed, `--version`
included, and anything driving the CLI (Pyomo through NL/SOL, the
benchmark harness) failed with it or silently resolved some other
`pounce` earlier on PATH (gh #816). So when the package is being
imported *out of the source tree*, the shim now falls back to the
binary cargo already built — `target/release/pounce` or
`target/debug/pounce` — and says on stderr which one it picked.

The fallback is deliberately loud and deliberately narrow. Loud,
because a dev binary is not the wheel's binary and the difference has
shipped wrong answers before (gh #315: a stale build returned flipped
duals and the version string could not tell it apart). Narrow in two
ways: it is reached only when there is no bundled binary at all, which
for a published wheel is never — the wheel always carries one, and
`resolve_binary` returns it before looking any further — and even then
it engages only where `Cargo.toml` and `crates/pounce-cli/` sit above
this file. So a released install cannot quietly run something out of
somebody's build directory.
"""

import os
import subprocess
import sys
from pathlib import Path

# Origin labels for `resolve_binary`, shared with pyomo-pounce's
# binary check so both name the same three cases the same way.
BUNDLED = "bundled"
CHECKOUT = "checkout"


def _exe_name() -> str:
    return "pounce.exe" if sys.platform == "win32" else "pounce"


def _bundled_binary() -> Path:
    """Where the wheel puts the CLI binary. May or may not exist."""
    return Path(__file__).parent / "bin" / _exe_name()


def _repo_root() -> "Path | None":
    """The POUNCE source checkout this file lives in, or None.

    Recognized by the two things an installed wheel never has above the
    package directory: a workspace `Cargo.toml` and `crates/pounce-cli/`.
    Requiring both keeps an unrelated Rust project that happens to sit
    above a site-packages tree from being mistaken for this one.
    """
    for parent in Path(__file__).resolve().parents:
        if (parent / "Cargo.toml").is_file() and (
            parent / "crates" / "pounce-cli" / "Cargo.toml"
        ).is_file():
            return parent
    return None


def _checkout_binary() -> "Path | None":
    """The cargo-built CLI of the source checkout this file lives in.

    Returns the most recently built of `target/release/pounce` and
    `target/debug/pounce` (honoring `CARGO_TARGET_DIR`), or None when
    this is not a source checkout or nothing has been built yet.

    Newest-wins rather than release-always-wins: a developer who has
    just run `cargo build` and is testing a fix should get *that*
    binary, not a release artifact from three weeks ago. The caller
    prints the path it chose, so the choice is always visible.
    """
    root = _repo_root()
    if root is None:
        return None
    target = os.environ.get("CARGO_TARGET_DIR") or (root / "target")
    name = _exe_name()
    target = Path(target)
    built = [
        p
        for p in (target / "release" / name, target / "debug" / name)
        if p.is_file() and os.access(p, os.X_OK)
    ]
    if not built:
        return None
    return max(built, key=lambda p: p.stat().st_mtime)


def resolve_binary():
    """`(path, origin)` for the CLI binary this install should run.

    `origin` is `BUNDLED` for the wheel's own binary and `CHECKOUT` for
    a cargo build picked up from the surrounding source tree; both are
    `None` when neither exists.
    """
    bundled = _bundled_binary()
    if bundled.is_file():
        return bundled, BUNDLED
    checkout = _checkout_binary()
    if checkout is not None:
        return checkout, CHECKOUT
    return None, None


def _not_found_message(bundled: Path) -> str:
    root = _repo_root()
    if root is None:
        return (
            f"pounce: bundled CLI binary not found at {bundled}.\n"
            "This usually means the package was installed with "
            "`maturin develop`, which builds only the Python extension.\n"
            "Reinstall the published wheel:\n"
            "    pip install --force-reinstall pounce-solver\n"
        )
    target = Path(os.environ.get("CARGO_TARGET_DIR") or (root / "target"))
    return (
        f"pounce: no CLI binary found.\n"
        f"  wheel-bundled : {bundled} (missing)\n"
        f"  cargo build   : {target}/{{release,debug}}/{_exe_name()} "
        f"(missing)\n"
        "`maturin develop` builds only the Python extension, so a source\n"
        "checkout has no CLI binary until you build one:\n"
        f"    make -C {root} dev     # build the CLI and stage it here\n"
        "or, for the binary alone:\n"
        "    cargo build --release --bin pounce\n"
        "`pyomo_pounce.check_binary()` reports which binary a Pyomo solve\n"
        "would run.\n"
    )


# Argv forms whose output is machine-read: Pyomo's ASL layer runs
# `pounce -v` and regex-scans the merged stdout+stderr for the first
# `N.N[.N]` it finds (`pyomo.opt.base.solvers._extract_version`). A
# fallback notice naming a path like `/opt/py3.11/...` would be read as
# the solver's version, so these queries answer with nothing but their
# own output.
_QUIET_ARGV = frozenset({"-v", "-V", "--version", "--about", "-h", "--help"})


def _fallback_notice(binary: Path, argv) -> str:
    if any(a in _QUIET_ARGV for a in argv):
        return ""
    return (
        f"pounce: this install has no wheel-bundled CLI binary; running the "
        f"cargo build at {binary}.\n"
        f"        `make dev` from the repo root stages it into the package, "
        f"which is what a wheel ships.\n"
    )


def main() -> int:
    binary, origin = resolve_binary()
    if binary is None:
        sys.stderr.write(_not_found_message(_bundled_binary()))
        return 1

    if origin == CHECKOUT:
        # Never silent: this is a build-directory binary standing in for
        # the wheel's, and which build it is has mattered (gh #315).
        sys.stderr.write(_fallback_notice(binary, sys.argv[1:]))

    args = [str(binary), *sys.argv[1:]]
    if sys.platform == "win32":
        completed = subprocess.run(args)
        return completed.returncode
    os.execv(str(binary), args)


if __name__ == "__main__":
    raise SystemExit(main())
