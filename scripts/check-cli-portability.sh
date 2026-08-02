#!/usr/bin/env bash
# Assert that a bundled Linux `pounce` CLI can actually run everywhere the
# wheel claims it can.
#
# WHY THIS EXISTS (#452). The wheels ship the CLI as an opaque file inside
# pounce/bin/. auditwheel never looks at it — as far as the wheel format is
# concerned it is data, not a linked object — so a wheel can advertise
# manylinux2014 (glibc 2.17) while the binary beside it demands something far
# newer. That is exactly what shipped in 0.9.0: the CLI was built on the
# release runner (Ubuntu 24.04, glibc 2.39) rather than inside the manylinux
# container maturin builds the extension module in, so it refused to start on
# Debian 12, RHEL 8/9, and most cluster images with
#
#   pounce/bin/pounce: /lib/x86_64-linux-gnu/libc.so.6:
#     version `GLIBC_2.39' not found
#
# Exactly two symbols caused it — pidfd_spawnp and pidfd_getpid, from Rust
# std's process-spawn path. Nothing in POUNCE's numerics. Everything else the
# binary needed topped out at glibc 2.34.
#
# The existing wheel-smoke job could not catch this: it runs on the same host
# that built the binary, the one platform where the floor is invisible by
# construction. So check the property on the artifact instead of inferring it
# from a successful run.
#
# Usage:
#   scripts/check-cli-portability.sh <binary> [max-glibc]
#
# max-glibc defaults to 2.17, the manylinux2014 floor that python/pyproject
# builds against. Raise it only by also raising the wheel's manylinux tag —
# they are two halves of the same promise.

set -euo pipefail

BIN="${1:?usage: check-cli-portability.sh <binary> [max-glibc]}"
MAX="${2:-2.17}"

[[ -f "$BIN" ]] || { echo "check-cli-portability: no such file: $BIN" >&2; exit 1; }
command -v objdump >/dev/null 2>&1 || {
  echo "check-cli-portability: objdump not found (install binutils)" >&2
  exit 1
}

echo "== $BIN =="
file "$BIN" || true

# Versioned glibc symbol references are what actually break at exec time: the
# dynamic loader refuses to start a binary asking for a symbol version the
# host's libc does not define. The highest one referenced IS the floor.
vers="$(objdump -T "$BIN" 2>/dev/null | grep -o 'GLIBC_[0-9.]*' | sort -uV || true)"

if [[ -z "$vers" ]]; then
  # A static build (e.g. musl) references no versioned symbols at all. Not
  # what we produce today, but it trivially satisfies any floor.
  echo "  OK — no versioned glibc symbols (static; no runtime floor)"
  echo "check-cli-portability: OK"
  exit 0
fi

highest="$(echo "$vers" | tail -1)"
highest_num="${highest#GLIBC_}"

echo "  glibc versions referenced: $(echo "$vers" | tr '\n' ' ')"
echo "  highest (the runtime floor): $highest_num"
echo "  allowed maximum:             $MAX"

# sort -V puts the larger version last; if that is not $MAX, the binary needs
# something newer than we promise.
if [[ "$(printf '%s\n%s\n' "$highest_num" "$MAX" | sort -V | tail -1)" != "$MAX" ]]; then
  echo
  echo "  FAIL — the bundled CLI requires glibc $highest_num but the wheel"
  echo "         promises $MAX. It will not start on anything older."
  echo "         Symbols forcing the floor:"
  objdump -T "$BIN" | grep "GLIBC_${highest_num}\b" | awk '{print $NF}' | sort -u \
    | while IFS= read -r s; do printf '           %s\n' "$s"; done
  echo
  echo "check-cli-portability: FAILED — the CLI must be built INSIDE the" >&2
  echo "  manylinux container, not on the runner host. See #452." >&2
  exit 1
fi

echo "  OK — floor is within the manylinux2014 promise"
echo "check-cli-portability: OK"
