# Electrical Grid Suite (AC Optimal Power Flow)

AC Optimal Power Flow (AC OPF) is the canonical nonconvex NLP of the power
systems community: minimise generation cost subject to nonlinear AC power
flow equations, voltage limits, line thermal limits, and generator output
bounds. Problems come from the MATPOWER test case library (Zimmerman,
Murillo-Sanchez, Thomas, "MATPOWER: Steady-State Operations, Planning, and
Analysis Tools for Power Systems Research and Education", IEEE Trans. Power
Syst. 2011). POUNCE uses the polar form with explicit bus voltage magnitude
and angle variables.

Problems currently range from case3_lmbd (3 buses, 12 variables) to
case30_ieee (30 buses, 72 variables, 142 constraints). These are good
stress tests for dense KKT paths because the constraint Jacobian is
moderately dense and the Hessian has many non-convex blocks from the
voltage–angle coupling.

This suite lives under `benchmarks/grid/`. See `benchmarks/README.md` for
an overview of all suites. Like the other suites, problems are solved
through the AMPL `.nl` interface by the shared NL driver
(`benchmarks/scripts/run_nl_bench.sh`), which runs both POUNCE and Ipopt
(linked against MA57) and feeds the composite `benchmarks/BENCHMARK_REPORT.md`.

## Contents

- `nl/` — the AMPL `.nl` files solved by the harness (one per MATPOWER case)
- `grid_nl_export.py` — regenerates the `.nl` files from the MATPOWER cases
- `pounce.json` — latest POUNCE per-problem results
- `ipopt_ma57.json` — Ipopt/MA57 reference results
- `grid_benchmark_report.md` — per-problem analysis
- `problems.rs` — **retired** pure-Rust `NlpProblem` definitions, kept for
  reference; no longer compiled (the suite is now `.nl`-driven)

## How to run

From the repo root:

```bash
make -C benchmarks grid-run     # writes benchmarks/grid/pounce.json
make -C benchmarks grid-rerun   # force a fresh run
```

Or solve a single case directly:

```bash
pounce benchmarks/grid/nl/case30_ieee.nl print_level=5
```

## Re-exporting

If the MATPOWER cases or export logic change, regenerate the `.nl` files:

```bash
python benchmarks/grid/grid_nl_export.py
```
