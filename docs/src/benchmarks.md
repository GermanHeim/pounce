# Benchmarks

The `benchmarks/` directory contains comparison harnesses that run
POUNCE against upstream Ipopt across several test suites: the Vanderbei
CUTE-in-AMPL collection, Mittelmann ampl-nlp, CHO parameter estimation,
GasLib pipelines, water-network design, electrolyte thermodynamics,
AC optimal power flow, and large-scale synthetic NLPs. Every suite is
`.nl`-driven — a directory of AMPL `.nl` files solved by both `pounce`
and `ipopt`.

Common targets:

```sh
make benchmark              # full sweep: every suite + composite report
make benchmark-report       # regenerate benchmarks/BENCHMARK_REPORT.md
make benchmark-cho          # one suite at a time
make benchmark-gas
make benchmark-water
make benchmark-mittelmann
make benchmark-vanderbei    # Vanderbei CUTE-in-AMPL collection (733 problems)
```

One suite is deliberately not `.nl`-driven:
[the warm-start benchmark](warm-start-benchmark.md) measures the cost of
solving a *sequence* of related problems, cold versus warm, across all
three of POUNCE's solve paths. Carrying a working set between solves
needs an in-process handle, so it runs through the Python API instead of
the CLI, and it reports on its own rather than into the composite
report.

The benchmark inputs themselves — the `.nl` problem files — and the
per-run logs and JSON results are regenerated locally and not tracked in
the repository. See
[`benchmarks/README.md`](https://github.com/jkitchin/pounce/blob/main/benchmarks/README.md)
for the full list and per-suite details.
