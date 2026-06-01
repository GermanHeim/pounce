# CBLIB suite — conic (exponential / power cone) tier

The **conic** benchmark tier: instances from the Conic Benchmark Library
(CBLIB, <https://cblib.zib.de>) in Conic Benchmark Format (`.cbf`). Unlike
every other suite here — which is `.nl`-driven through the main `pounce`
NLP binary — these are *conic programs* (geometric programs and power-cone
models) solved through POUNCE's convex conic driver (`pounce-convex`'s
non-symmetric HSDE path) via the dedicated `pounce_cblib` binary.

Each instance is recorded in the same schema as the other suites —
`{solver, name, n, m, status, objective, iterations, solve_time}` — in
`cblib/pounce.json`, so it merges into the composite `BENCHMARK_REPORT.md`.

## What runs

By default the runner solves the small instances **vendored with the
repo** (under `crates/pounce-cli/tests/data/cblib/`), so it works offline:

| Instance | Class | Cones |
|---|---|---|
| `demb761`, `beck751`, `fang88` | geometric programs (Demberg / Beck / Fang) | exponential |
| `pow3_synthetic` | hand-authored power-cone problem | power (`POWCONES`) |

These are also the cross-check tests in
`crates/pounce-cli/tests/cblib_vs_nlp.rs`, where each conic solve is
validated against an **independent** smooth-NLP solve (the two agree on the
objective to ~1e-8). Published CBLIB reference objectives are unavailable
(the solution files 404), so that conic-vs-NLP cross-check *is* the
correctness reference.

## Running

```sh
python3 benchmarks/cblib/run_cblib.py            # vendored instances
python3 benchmarks/cblib/run_cblib.py --detail full   # + per-iteration trace
python3 benchmarks/cblib/run_cblib.py --dir /path/to/cblib   # more instances
```

`--dir` points at a folder of additional `.cbf` files — e.g. a local CBLIB
checkout. The reader supports the cone kinds `F`/`L=`/`L+`/`L-`/`EXP`/`Q`
and the 3-D power cone (`POWCONES` / `@k:POW`); instances using PSD
(`DCOORD`), rotated SOC (`QR`), or dual power cones are skipped with a
clear error. The large power-cone instances (`2013_fir*`, ~120 MB) are not
vendored; fetch them into a `--dir` to include them.

The underlying `pounce_cblib <file.cbf> --json-output <out>` emits a full
`pounce.solve-report/v1` JSON (the same schema the `.nl` path writes, with
an input descriptor of kind `cbf-file`); the runner projects each into the
suite record schema.
