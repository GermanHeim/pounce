"""Console-script shim for the bundled `pounce` CLI binary.

The wheel ships the Rust `pounce` binary inside `pounce/bin/`. The
`[project.scripts]` entry in pyproject.toml points at `main` below,
which transfers control to that binary. The Python interpreter
disappears from the process tree on Unix (os.execv); on Windows we
fall back to a subprocess + propagated exit code because os.execv
there spawns a child and returns control to the shell, which breaks
signal handling.

If the binary is missing — typically because the user installed via
`maturin develop` (which builds only the extension module) instead of
a published wheel — print a focused error explaining how to recover.
"""

import os
import subprocess
import sys
from pathlib import Path


def _bundled_binary() -> Path:
    name = "pounce.exe" if sys.platform == "win32" else "pounce"
    return Path(__file__).parent / "bin" / name


def main() -> int:
    binary = _bundled_binary()
    if not binary.is_file():
        sys.stderr.write(
            f"pounce: bundled CLI binary not found at {binary}.\n"
            "This usually means the package was installed with "
            "`maturin develop`, which builds only the Python extension.\n"
            "For local development, run the CLI via cargo instead:\n"
            "    cargo run --release --bin pounce -- <args>\n"
            "or install the published wheel:\n"
            "    pip install --force-reinstall pounce-solver\n"
        )
        return 1

    args = [str(binary), *sys.argv[1:]]
    if sys.platform == "win32":
        completed = subprocess.run(args)
        return completed.returncode
    os.execv(str(binary), args)


if __name__ == "__main__":
    raise SystemExit(main())
