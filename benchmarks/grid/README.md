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

The suite name is "grid" rather than "opf" so that the directory layout
makes its purpose obvious at a glance — every problem here is an electrical
grid optimization.

## Contents

- `problems.rs` — MATPOWER test cases wrapped as `NlpProblem` instances;
  compiled into the `pounce-domain-bench` crate as a `#[path]` module
- `grid_results.json` — latest POUNCE per-problem results
- `grid_benchmark_report.md` — per-problem analysis

## Prerequisites

A stable Rust toolchain. Nothing else — the harness is pure Rust and uses
POUNCE's default FERAL linear-solver backend.

## How to run

From the repo root:

```bash
make grid-run
```

or directly:

```bash
cargo run --release --bin grid_suite
```

Set `RESULTS_FILE=<path>` to override the output location.

## Output

- `grid_results.json` — POUNCE per-problem results (status, objective,
  iterations, wall time)
- `grid_stderr.txt` — solver chatter (gitignored)
