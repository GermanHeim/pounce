#!/usr/bin/env bash
# Dual-solver .nl benchmark driver.
#
# Solves every *.nl under a suite directory with both pounce and
# ipopt-ma57 (the AMPL solver protocol), writes a single combined
# results JSON in the cutest-style schema:
#
#   [{"solver":"pounce|ipopt", "name":..., "n":..., "m":...,
#     "status":..., "objective":..., "iterations":..., "solve_time":...}, ...]
#
# Usage:
#   run_nl_bench.sh <suite_name> <nl_dir> <results_json> \
#                   <pounce_bin> <ipopt_bin> <time_limit_seconds>
#
# Suites (electrolyte, grid, water, gas, cho) drive this from their
# Makefile target. Output feeds benchmark_report.py's
# load_domain_results() loader.

set -u

SUITE="$1"
NL_DIR="$2"
RESULT="$3"
POUNCE_BIN="$4"
IPOPT_BIN="$5"
TIMELIMIT="${6:-300}"

LOGDIR="$(dirname "$RESULT")/logs/${SUITE}"
mkdir -p "$LOGDIR" "$(dirname "$RESULT")"

# Locate binaries
for b in "$POUNCE_BIN" "$IPOPT_BIN"; do
  if [ ! -x "$b" ] && ! command -v "$b" >/dev/null 2>&1; then
    echo "run_nl_bench.sh: binary not found or not executable: $b" >&2
    exit 2
  fi
done

shopt -s nullglob
nl_files=("$NL_DIR"/*.nl)
total=${#nl_files[@]}
if [ "$total" -eq 0 ]; then
  echo "run_nl_bench.sh: no .nl files under $NL_DIR" >&2
  exit 2
fi

# Helpers ------------------------------------------------------------

# Parse n, m from line 2 of an AMPL .nl file: "nvar ncon ... ".
parse_nm() {
  local nl="$1"
  awk 'NR==2 {gsub(/[\t#].*/,""); print $1, $2; exit}' "$nl"
}

# Map ipopt's free-form termination message → cutest-style status label.
ipopt_status_from_log() {
  local log="$1"
  if grep -q "Optimal Solution Found" "$log"; then echo "Solve_Succeeded"; return; fi
  if grep -q "Solved To Acceptable Level" "$log"; then echo "Solved_To_Acceptable_Level"; return; fi
  if grep -q "Maximum Number of Iterations Exceeded" "$log"; then echo "Maximum_Iterations_Exceeded"; return; fi
  if grep -q "Maximum CPU time exceeded" "$log"; then echo "Maximum_CpuTime_Exceeded"; return; fi
  if grep -q "Converged to a point of local infeasibility" "$log"; then echo "Infeasible_Problem_Detected"; return; fi
  if grep -q "Restoration Failed" "$log"; then echo "Restoration_Failed"; return; fi
  if grep -q "Search Direction is becoming Too Small" "$log"; then echo "Search_Direction_Becomes_Too_Small"; return; fi
  if grep -q "Diverging Iterates" "$log"; then echo "Diverging_Iterates"; return; fi
  if grep -q "Invalid number" "$log"; then echo "Invalid_Number_Detected"; return; fi
  echo "Unknown_Error"
}

# pounce CLI prints `Status: Solve_Succeeded` (or similar Status: <X>)
# at the end. Fall back to log scraping if we can't find it.
pounce_status_from_log() {
  local log="$1"
  local s
  s=$(grep -oE '^[Ss]tatus:[[:space:]]+\w+' "$log" | tail -1 | awk '{print $2}')
  if [ -n "$s" ]; then echo "$s"; return; fi
  # Pounce mirrors Ipopt's stdout for the common cases
  ipopt_status_from_log "$log"
}

# Extract iter count and objective from solver stdout (both use Ipopt's
# "Number of Iterations....: N" and "Objective...........: V" lines).
extract_iters() { grep -oE 'Number of Iterations[. :]+[0-9]+' "$1" | tail -1 | grep -oE '[0-9]+$'; }
extract_obj() {
  # Prefer the "Objective..." line; fall back to "Final objective value: V".
  local v
  v=$(grep -oE 'Objective[. :]+[-+0-9.eE]+' "$1" | tail -1 | grep -oE '[-+0-9.eE]+$')
  if [ -n "$v" ]; then echo "$v"; return; fi
  grep -oE 'Final objective[. :]+[-+0-9.eE]+' "$1" | tail -1 | grep -oE '[-+0-9.eE]+$'
}

# Run one solver on one .nl. $1=label, $2=binary, $3=nl path, $4=ampl_protocol
# Emits one JSON object on stdout (no trailing comma).
run_one() {
  local label="$1" bin="$2" nl="$3" ampl_protocol="$4"
  local problem nm n m start end elapsed log rc
  problem=$(basename "$nl" .nl)
  nm=$(parse_nm "$nl"); n=${nm%% *}; m=${nm##* }
  log="${LOGDIR}/${problem}.${label}.log"

  start=$(python3 -c 'import time; print(time.time())')
  if [ "$ampl_protocol" = "yes" ]; then
    timeout "$TIMELIMIT" "$bin" "$nl" -AMPL max_cpu_time="$TIMELIMIT" > "$log" 2>&1
    rc=$?
  else
    timeout "$TIMELIMIT" "$bin" "$nl" > "$log" 2>&1
    rc=$?
  fi
  end=$(python3 -c 'import time; print(time.time())')
  elapsed=$(python3 -c "print(f'{$end - $start:.6f}')")

  local status
  if [ "$rc" = "124" ]; then
    status="Maximum_CpuTime_Exceeded"
  elif [ "$rc" -ne 0 ]; then
    # Try log-scrape first; many real status outcomes still produce
    # non-zero rc (Infeasible_Problem_Detected, etc.).
    if [ "$ampl_protocol" = "yes" ]; then
      status=$(ipopt_status_from_log "$log")
    else
      status=$(pounce_status_from_log "$log")
    fi
    if [ -z "$status" ] || [ "$status" = "Unknown_Error" ]; then
      status="Solver_Error"
    fi
  else
    if [ "$ampl_protocol" = "yes" ]; then
      status=$(ipopt_status_from_log "$log")
    else
      status=$(pounce_status_from_log "$log")
    fi
  fi

  local obj iter
  obj=$(extract_obj "$log"); obj=${obj:-null}
  iter=$(extract_iters "$log"); iter=${iter:-0}

  # JSON solver label: "ipopt" for the AMPL-protocol invocation (so the
  # report's load_domain_results() finds it under the canonical key).
  local solver_label
  case "$label" in
    pounce) solver_label="pounce" ;;
    ipopt*) solver_label="ipopt" ;;
    *) solver_label="$label" ;;
  esac

  printf '  {"solver":"%s","name":"%s","n":%s,"m":%s,"status":"%s","objective":%s,"iterations":%s,"solve_time":%s}' \
    "$solver_label" "$problem" "$n" "$m" "$status" "$obj" "$iter" "$elapsed"
}

# Main loop ----------------------------------------------------------

echo "[" > "$RESULT"
first=1
i=0
for nl in "${nl_files[@]}"; do
  i=$((i+1))
  problem=$(basename "$nl" .nl)
  printf "[%2d/%d] %-30s " "$i" "$total" "$problem"

  # pounce
  printf "pounce..."
  if [ $first -eq 0 ]; then echo "," >> "$RESULT"; fi; first=0
  run_one pounce "$POUNCE_BIN" "$nl" no >> "$RESULT"

  # ipopt-ma57 (AMPL protocol)
  printf " ipopt..."
  echo "," >> "$RESULT"
  run_one ipopt-ma57 "$IPOPT_BIN" "$nl" yes >> "$RESULT"

  printf " done\n"
done

echo "" >> "$RESULT"
echo "]" >> "$RESULT"
echo "wrote $RESULT"
