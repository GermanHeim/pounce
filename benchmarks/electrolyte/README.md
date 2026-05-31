# Electrolyte Thermodynamics Suite

Gibbs free energy minimisation for aqueous electrolyte systems. Given a set
of species and their reference-state thermodynamic data, find the
equilibrium composition subject to element-balance (mass conservation) and
charge-neutrality constraints. These problems are notoriously ill-conditioned
because species concentrations span many orders of magnitude and the
logarithmic activity terms blow up near zero concentration.

The problems cover mixed acid-base, hydrolysis, and
activity-coefficient-fitting systems drawn from classic chemical
engineering thermodynamics textbook examples and POUNCE's own regression
set.

This suite lives under `benchmarks/electrolyte/`. See `benchmarks/README.md`
for an overview of all suites. Like the other suites, problems are solved
through the AMPL `.nl` interface by the shared NL driver
(`benchmarks/scripts/run_nl_bench.sh`), which runs both POUNCE and Ipopt
(linked against MA57) and feeds the composite `benchmarks/BENCHMARK_REPORT.md`.

## Contents

- `nl/` — the AMPL `.nl` files solved by the harness (one per system)
- `electrolyte_nl_export.py` — regenerates the `.nl` files from the system
  definitions
- `pounce.json` — latest POUNCE per-problem results
- `ipopt_ma57.json` — Ipopt/MA57 reference results
- `electrolyte_benchmark_report.md` — per-problem analysis
- `problems.rs` — **retired** pure-Rust `NlpProblem` definitions, kept for
  reference; no longer compiled (the suite is now `.nl`-driven)

## How to run

From the repo root:

```bash
make -C benchmarks electrolyte-run     # writes benchmarks/electrolyte/pounce.json
make -C benchmarks electrolyte-rerun   # force a fresh run
```

Or solve a single system directly:

```bash
pounce benchmarks/electrolyte/nl/butanol_water_lle.nl print_level=5
```

## Re-exporting

If the system definitions or export logic change, regenerate the `.nl` files:

```bash
python benchmarks/electrolyte/electrolyte_nl_export.py
```
