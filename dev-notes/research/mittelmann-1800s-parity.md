# Mittelmann at 1800 s — the like-for-like POUNCE arm (gh#581)

**Status: measurement record.** This note documents the run that closed
[#581](https://github.com/jkitchin/pounce/issues/581) and the code change that
went with it. It records what was measured, not a design proposal.

## 1. What #581 asked for

The mittelmann suite compared a **300 s** POUNCE arm against an **1800 s** Ipopt
reference. The reference had been regenerated at 1800 s
(`ipopt_ma57.provenance.json`, suite override) because `max_cpu_time` counts CPU
summed across threads and unpinned multithreaded runs burned a 300 s budget in
~90 s wall. That fix was correct, but the POUNCE arm has no time-limit flag — it
is wrapped in `timeout $BENCH_TIMELIMIT`, default 300 — so a plain sweep left the
two columns on different clocks.

The ask: rerun the POUNCE arm at 1800 s so the suite is like-for-like.

## 2. Setup

| | |
|---|---|
| POUNCE | `5dc835f5`, clean tree, `cargo build --release -p pounce-cli`, sha256 `702f586f…` |
| limit | 1800 s wall (`BENCH_TIMELIMIT=1800`) |
| threads | `OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=VECLIB_MAXIMUM_THREADS=RAYON_NUM_THREADS=1` |
| corpus | `~/projects/pounce-bench-data` — the **local mirror**, not Dropbox |
| driver | `benchmarks/scripts/run_nl_bench.sh`, mode `pounce` |
| host | kitchin.lan.local.cmu.edu, macOS 26.5.2 arm64, 14 cores |

Two setup points that are load-bearing:

**The corpus was moved off Dropbox first.** `POUNCE_BENCH_DATA` pointed at
`~/Dropbox/projects/pounce-bench-data`, and `scripts/bench_data_root.sh` warns
that timings taken there include the sync daemon's I/O. `scripts/sync-bench-data.sh`
mirrors it to a plain local directory; the mirror was verified at 1570 inputs /
4.1 GB before the run. The previously published mittelmann numbers were taken
with Dropbox in the path.

**`make -C benchmarks mittelmann-rerun` does not work on a fresh checkout.** It
depends on `mittelmann-translate`, which delegates to
`benchmarks/mittelmann/Makefile` — a file `.gitignore:44` (`/benchmarks/*/*`)
excludes and that has never been committed (`git log` on the path is empty). It
fails with ``No rule to make target `fetch` ``. The `.nl` already existed, so the
run invoked `run_nl_bench.sh` directly with the make rule's exact arguments.
Tracked separately; it is not specific to this run.

## 3. Result

**47/47 `Solve_Succeeded`.** Total 54.1 min. Six instances changed status
relative to the previously published arm:

| instance | published | this run |
|---|---|---|
| `WM_CFy` | `Maximum_CpuTime_Exceeded`, 99 it | Optimal, 1533.6 s, 556 it |
| `robot_a` | `Maximum_CpuTime_Exceeded`, 1449 it | Optimal, 23.0 s, 189 it |
| `robot_b` | `Maximum_CpuTime_Exceeded`, 1426 it | Optimal, 26.6 s, 271 it |
| `robot_c` | `Maximum_CpuTime_Exceeded`, 1512 it | Optimal, 26.4 s, 222 it |
| `NARX_CFy` | `Solved_To_Acceptable_Level`, 564 it | Optimal, 162.1 s, 400 it |
| `qcqp1000-1nc` | `Solved_To_Acceptable_Level`, 187 it | Optimal, 16.1 s, 177 it |

For the mittelmann suite the report now reads **0 Ipopt-exclusive solves, 6
POUNCE-exclusive**, 41 both-Optimal, 40/41 objectives matching within 0.01 %.
That is the outcome #581 predicted for its headline number.

## 4. What this run does NOT establish

#581 predicted that **only `WM_CFy`** could change, because only that instance
was decided by the limit gap. Six changed. The issue's reasoning was not wrong;
the comparison is confounded.

The previously published `pounce.json` was produced by
`perf/feral-refine-target @ 96846891-dirty` — a different, dirty branch — per the
provenance table of the report generated 2026-08-22. So published-vs-this-run
differs in **four** ways at once: binary, time limit, corpus location, and thread
pinning.

So the run was repeated with the **same binary at 300 s**, everything else
identical, which isolates the limit exactly. That control settles it:

| instance | 300 s | 1800 s | attributable to |
|---|---|---|---|
| `WM_CFy` | killed, 98 it | Optimal, 1533.6 s, 556 it | **the limit** |
| `nql180` | killed, 48 it | Optimal, 329.8 s, 52 it | **the limit** |
| `robot_a/b/c` | Optimal, 23–27 s | Optimal, 23–27 s | the binary |
| `NARX_CFy` | Optimal, 166.0 s | Optimal, 162.1 s | the binary |
| `qcqp1000-1nc` | Optimal, 16.4 s | Optimal, 16.1 s | the binary |

**Exactly two instances are decided by the time limit.** Everything else solves
at 300 s under this binary, with iteration counts identical to the 1800 s run.
#581's reasoning was sound; it was the published column that had moved.

**Do not cite the `robot_a/b/c` improvement as an outcome of #581** — it is a
solver change, and it is the more interesting result of the two.

`nql180` deserves a footnote: it is limit-attributable only *because* the binary
made it ~6x slower (54.5 s published → 329.8 s here), pushing it past a 300 s cap
it used to clear comfortably. The raised limit rescued an instance a solver
change had broken. See the `nql180` entry below.

## 4a. Determinism and measurement noise, as a byproduct

The control doubles as two checks worth recording.

**Determinism.** All 45 instances that finished under 300 s in both arms took the
*identical* iteration count. Under pinned threads this solver is reproducible
run-to-run on this host.

**Noise floor** (n = 45, same binary, same iteration counts, wall clock only):
median **0.9 %**, mean 1.3 %, max 12.5 %. The single outlier is `arki0003`, a
1.97 s problem where 12.5 % is 0.24 s absolute. Any future claim on this suite
that a change is worth less than ~2 % on a multi-second instance is measuring the
machine, not the solver — and on sub-2 s instances noise alone exceeds 10 %.

Two differences run the other way and are worth attention on their own merits,
independent of which binary produced which column:

- `clnlbeam`: 483 → 2465 iterations (36.9 s → 100.0 s). Ipopt takes 543
  iterations on the same instance, so this is a ~4.5x iteration disadvantage.
- `nql180`: 34 → 52 iterations but 54.5 s → 329.8 s. The 6x wall-clock change
  far exceeds the iteration change, so per-iteration cost moved as well. This is
  the instance that now needs the raised limit to finish at all.

Whether those are regressions on `main` or simply the perf branch's gains being
absent from it is **not established here**, and settling it requires building
`96846891` and rerunning. Recorded rather than asserted.

Both are well outside the 0.9 % noise floor measured in §4a, so they are real
movements rather than measurement scatter — the open question is their direction
of causation, not their existence.

## 5. The code change

`benchmark_report.py:time_limit_note()` stated:

> its limit is not stamped in the results and is inferred here from the longest
> run that was killed

That was already untrue — `run_nl_bench.sh` writes the limit it wrapped the
solver in to `<suite>/pounce.env.json`, and `read_env_stamp()` was already
loading that file for the threading note. The inference also had a second
failure mode that this run triggers directly: **it goes blind exactly when a
suite stops timing out**, because a run with no kills leaves nothing to infer
from. A suite at genuine parity would have printed no POUNCE limit at all.

The note now prefers the stamp and falls back to the inference only for results
that predate it. When the two limits agree it says so explicitly and stops
looking for instances "decided by the gap", because at equal limits there is no
gap to decide anything:

> POUNCE ran this suite at the same 1800s (recorded in `pounce.env.json`), so
> both columns are held to the same clock here.

A POUNCE timeout under equal limits is then a genuine capability difference and
is reported by the suite's ordinary timeout accounting, not by the limits note.

## 6. Why `BENCHMARK_REPORT.md` is not regenerated in this change

Only `benchmarks/mittelmann/pounce.json` exists in this checkout; every other
suite's results are gitignored and absent. Running `benchmark_report.py` here
therefore rewrites the committed report — which covers 1 326 models across
eleven suites — as a 47-model mittelmann-only file, deleting ~440 lines and
rewriting `figures/profile_*.png`. That would be a substantial regression to the
repo's records in exchange for one suite's update.

The committed report stays as generated on 2026-08-22. The mittelmann-only
render produced by this run is kept as evidence outside the repo, alongside the
raw arm. The committed report will pick up the parity at the next full sweep,
and the `time_limit_note` fix means that sweep will state the POUNCE limit from
the stamp rather than guessing at it.

## 7. Artifacts

`benchmarks/mittelmann/pounce.json` is gitignored by design, so the durable copy
of this arm — with its `pounce.env.json` stamp (`timelimit: 1800`, four thread
vars pinned to 1) and the mittelmann-only report render — lives at
`~/projects/research/mittelmann-option-tuning/baseline/canonical/`, together with
the previously published arm at `baseline/prior-300s/` and a same-binary 300 s
control at `baseline/control-300s/` that isolates the time limit from the binary
change described in §4.
