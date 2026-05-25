# Benchmarks

The `benchmarks/` directory contains comparison harnesses that run
POUNCE against upstream Ipopt across several test suites:
Hock-Schittkowski, CUTEst, Mittelmann ampl-nlp, CHO parameter
estimation, GasLib pipelines, water-network design, and large-scale
synthetic NLPs.

Common targets:

```sh
make benchmark              # full sweep: every suite + composite report
make benchmark-report       # regenerate benchmarks/BENCHMARK_REPORT.md
make benchmark-cho          # one suite at a time
make benchmark-gas
make benchmark-water
make benchmark-mittelmann
make benchmark-cutest       # CUTEst (requires `make -C benchmarks cutest-prepare`)
```

The benchmark inputs themselves — large `.nl` exports, compiled SIF
problem libraries, per-run logs, and JSON results — are regenerated
locally and not tracked in the repository. See
[`benchmarks/README.md`](https://github.com/jkitchin/pounce/blob/main/benchmarks/README.md)
for the full list and per-suite details.
