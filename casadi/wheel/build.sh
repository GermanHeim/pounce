#!/usr/bin/env bash
# Stage the plugin for the *installed* casadi and build a wheel.
#
#   ./build.sh                      # wheel for the casadi in this environment
#   POUNCE_CASADI_STAGE_ONLY=1 ./build.sh    # stage, do not build the wheel
#   POUNCE_CASADI_PLAT_NAME=manylinux_2_28_x86_64 ./build.sh
#
# A release build runs this once per (casadi minor x platform). Within one
# platform the runs share this tree and `_plugins/<minor>/` accumulates, so
# the shape is: stage every minor with POUNCE_CASADI_STAGE_ONLY=1, then one
# final run to build the single wheel that carries them all.
#
# The wheel is tagged `py3-none-<platform>` -- see setup.py for why that is
# neither `any` nor `cp311`. The casadi axis is not in the tag: casadi is a
# runtime dependency selected on `casadi.__version__` at import, and no wheel
# tag can express it.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="$here/../.."

# Probe casadi from a neutral directory. `python3 -c` puts the current
# directory first on sys.path and this repository has a `casadi/` directory
# of its own, so running build.sh from the repo root otherwise imports the
# source tree as a namespace package -- which does not fail, it succeeds and
# then dies on `AttributeError: module 'casadi' has no attribute
# '__version__'`, which reads like a broken casadi install.
probe=$(cd / && python3 - <<'PY'
import sys

try:
    import casadi
except ImportError:
    sys.exit(
        "error: casadi is not installed in this environment.\n"
        "       pip install 'casadi>=3.7,<3.8' and re-run."
    )
version = getattr(casadi, "__version__", None)
if version is None:
    sys.exit(f"error: imported casadi from {casadi.__file__!r}, which has no "
             "__version__; that is not the casadi package.")
print(version, ".".join(version.split(".")[:2]))
PY
)
ver=${probe%% *}
minor=${probe##* }
echo "building pounce-casadi for casadi $ver"

cargo build --release -p pounce-cinterface --manifest-path "$root/Cargo.toml"
make -C "$here/.." "$@"

dest="$here/pounce_casadi/_plugins/$minor"
mkdir -p "$dest"
cp "$here/../libcasadi_nlpsol_pounce."* "$dest/"

# The shared library only. The old glob also matched the sibling `.d`
# depfile and `.rlib`, which are build artefacts -- the `.rlib` alone is
# 1.3 MB of Rust static archive shipped to every user for nothing.
copied=0
for f in "$root/target/release/libpounce_cinterface.so" \
         "$root/target/release/libpounce_cinterface.dylib" \
         "$root/target/release/pounce_cinterface.dll"; do
  if [ -f "$f" ]; then cp "$f" "$dest/"; copied=1; fi
done
if [ "$copied" = 0 ]; then
  # Fatal, not a warning: the plugin resolves the solver beside itself, so
  # a wheel staged without it installs and then fails at import. That is
  # the failure mode this script exists to prevent, and it is worse when
  # the artefact is already published.
  echo "error: no libpounce_cinterface shared library in $root/target/release" >&2
  exit 1
fi

if [ -n "${POUNCE_CASADI_STAGE_ONLY:-}" ]; then
  echo "staged $dest (POUNCE_CASADI_STAGE_ONLY set; no wheel built)"
  exit 0
fi

rm -rf "$here/dist"
python3 -m pip wheel --no-deps -w "$here/dist" "$here"

# Assert the tag rather than trust it. A wheel tagged `any` installs on
# every platform and carries exactly one platform's shared libraries, so
# `pip install` succeeds and `import pounce_casadi` is where the user finds
# out. If setup.py stops being picked up -- a build frontend change, a
# setuptools change, someone deleting it as redundant next to pyproject.toml
# -- this is what says so, here, instead of in a bug report.
wheel=$(ls "$here"/dist/pounce_casadi-*.whl)
case "$(basename "$wheel")" in
  *-none-any.whl)
    echo "error: built $(basename "$wheel"), which claims to install on every" >&2
    echo "       platform while carrying one platform's binaries. setup.py's" >&2
    echo "       platform tagging did not take effect." >&2
    exit 1;;
esac
echo "wheel written to $here/dist: $(basename "$wheel")"
