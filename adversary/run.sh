#!/usr/bin/env bash
# Reproduce the 2026-08-05 adversary run end to end.
#
# Everything is deterministic given the seed: the generators use an
# in-tree splitmix64, so there is no `rand` dependency and no machine
# variation. Re-running this reproduces the reports bit for bit.
set -uo pipefail
cd "$(dirname "$0")/fuzz"

SEED=${SEED:-20260805}
cargo build --release || exit 2

echo "### QP elastic-certificate fuzz"
./target/release/adversary-fuzz qp "${QP_N:-400}" "$SEED"
qp_rc=$?

echo
echo "### Independent adjudication (scipy/HiGHS) of the generated instances"
python3 ../runs/2026-08-05_qp-active-set_adjudicate.py instances.jsonl
adj_rc=$?

echo
echo "### C warm-start answer-transparency fuzz"
./target/release/adversary-fuzz warmstart "${WS_N:-300}" "$SEED"
ws_rc=$?

echo
echo "qp=$qp_rc adjudicator=$adj_rc warmstart=$ws_rc"
echo "(adjudicator must be 0: a nonzero there means the GENERATOR is wrong,"
echo " and nothing the other two report can be trusted until it is fixed.)"
