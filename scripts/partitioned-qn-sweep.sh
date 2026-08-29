#!/usr/bin/env bash
# Measure the partitioned quasi-Newton Hessian against the exact and
# limited-memory paths on the `laptime` direct-collocation family
# (benchmarks/large_scale). Emits one TSV row per (model, leg).
#
#   scripts/partitioned-qn-sweep.sh <binary> <nl-file> [nl-file ...]
#
# Legs are the three Hessian sources plus the two partitioned update
# formulas, each at its own default barrier strategy — which is what a
# user actually gets, since `hessian_approximation=limited-memory`
# switches `mu_strategy` to adaptive on its own (application.rs, the
# gh#746 branch) while exact and partitioned stay monotone. The
# `*-monotone` legs pin the strategy so the Hessian is the only thing
# that moves.
set -uo pipefail

BIN=${1:?usage: partitioned-qn-sweep.sh <binary> <nl-file> [...]}
shift
MAX_ITER=${MAX_ITER:-1000}

legs=(
  "exact|hessian_approximation=exact"
  "lbfgs|hessian_approximation=limited-memory"
  "lbfgs-monotone|hessian_approximation=limited-memory mu_strategy=monotone"
  "partitioned-sr1|hessian_approximation=partitioned"
  "partitioned-bfgs|hessian_approximation=partitioned partitioned_update_type=bfgs"
  "partitioned-sr1-adaptive|hessian_approximation=partitioned mu_strategy=adaptive"
)

printf 'model\tleg\tstatus\titers\twall_s\tobjective\n'
for nl in "$@"; do
  name=$(basename "$nl" .nl)
  for leg in "${legs[@]}"; do
    tag=${leg%%|*}
    opts=${leg#*|}
    log=$(mktemp)
    start=$(date +%s.%N)
    # shellcheck disable=SC2086
    "$BIN" "$nl" max_iter="$MAX_ITER" print_level=5 $opts >"$log" 2>&1
    end=$(date +%s.%N)
    wall=$(awk -v a="$start" -v b="$end" 'BEGIN{printf "%.2f", b-a}')
    iters=$(grep -m1 'Number of Iterations' "$log" | awk -F: '{gsub(/[ .]/,"",$2); print $2}')
    obj=$(grep -m1 'Objective\.' "$log" | awk '{print $2}')
    status=$(grep -m1 '^EXIT' "$log" | sed 's/^EXIT: //; s/\.$//')
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$name" "$tag" "${status:-NO_EXIT_LINE}" "${iters:-NA}" "$wall" "${obj:-NA}"
    rm -f "$log"
  done
done
