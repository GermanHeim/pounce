# pounce-solve-report

Machine-readable `pounce.solve-report/v1` JSON writer for
[POUNCE](https://github.com/jkitchin/pounce). Bundles the same payload
AMPL's `.sol` carries (status, primal, dual, suffixes) with
FAIR-aligned provenance metadata (solver identity, input descriptor,
timestamp) and per-iteration history when requested.

Used by `pounce-cli` (CLI `--json` / `--json-detail`),
`pounce-cinterface` (`IpoptWriteSolveReport`), and the GAMS link.
Pure Rust; depends only on `serde` + `serde_json` plus three POUNCE
support crates.

## Schema versioning

The current schema tag is `pounce.solve-report/v1`. Breaking changes
bump the major version (`v2`, …). Adding fields without removing or
renaming existing ones is non-breaking — JSON consumers should
tolerate unknown fields.

## Detail levels

- **`ReportDetail::Summary`** (default) — FAIR metadata, problem
  dimensions, final solution, aggregate statistics. Equivalent to a
  `.sol` plus provenance. Use this for production logs.
- **`ReportDetail::Full`** — adds per-iteration history (when
  captured via `IpoptApplication::enable_iter_history`) and any
  `solution.suffixes` (e.g. sIPOPT sensitivity outputs, reduced-
  Hessian blocks). Use for debug captures and post-mortem analysis.

## Quick example

```rust
use pounce_solve_report::{ReportBuilder, ReportDetail, write_report_file};

let report = ReportBuilder::new()
    .with_detail(ReportDetail::Summary)
    .with_solver_identity("pounce", env!("CARGO_PKG_VERSION"))
    .with_input_path("model.nl")
    .with_problem_dimensions(n, m, /* n_eq */ 0)
    .with_status(status)
    .with_solution(&x, &lambda, &z_l, &z_u, objective)
    .with_statistics(&stats)
    .build();

write_report_file(std::path::Path::new("model.solve-report.json"), &report)?;
```

## FAIR provenance

The metadata block (`fair`) records solver identity, input
descriptor, build features, and a UTC timestamp so each report is
self-describing.

Wilkinson, M. D. et al. *The FAIR Guiding Principles for scientific
data management and stewardship.* Scientific Data 3, 160018 (2016).
DOI [10.1038/sdata.2016.18](https://doi.org/10.1038/sdata.2016.18).

## License

EPL-2.0. See [LICENSE](../../LICENSE) at the repo root.
