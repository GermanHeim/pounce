# Electrolyte Thermodynamics Suite

Gibbs free energy minimisation for aqueous electrolyte systems. Given a set
of species and their reference-state thermodynamic data, find the
equilibrium composition subject to element-balance (mass conservation) and
charge-neutrality constraints. These problems are notoriously ill-conditioned
because species concentrations span many orders of magnitude and the
logarithmic activity terms blow up near zero concentration.

The 13 problems in `problems.rs` cover mixed acid-base, hydrolysis, and
activity-coefficient-fitting systems drawn from classic chemical
engineering thermodynamics textbook examples and POUNCE's own regression
set.

## Contents

- `problems.rs` — `NlpProblem` definitions for all 13 systems; compiled
  into the `pounce-domain-bench` crate as a `#[path]` module
- `electrolyte_results.json` — latest POUNCE per-problem results
- `electrolyte_benchmark_report.md` — per-problem analysis

## Prerequisites

A stable Rust toolchain. Nothing else — the harness is pure Rust and uses
POUNCE's default FERAL linear-solver backend.

## How to run

From the repo root:

```bash
make electrolyte-run
```

or directly:

```bash
cargo run --release --bin electrolyte_suite
```

Set `RESULTS_FILE=<path>` to override the output location.

## Output

- `electrolyte_results.json` — POUNCE per-problem results (status,
  objective, iterations, wall time)
- `electrolyte_stderr.txt` — solver chatter (gitignored)
