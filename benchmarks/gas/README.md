# Gas Pipeline Network NLP Benchmarks

Benchmark problems from [Sakshi21299/gas_networks](https://github.com/Sakshi21299/gas_networks),
exported as AMPL NL files. See [issue #12](https://github.com/jkitchin/ripopt/issues/12).

This suite lives under `benchmarks/gas/`. See `benchmarks/README.md` for an
overview of all benchmark suites. Unlike the HS / CUTEst / domain benchmarks,
the gas suite does **not** feed into the composite `BENCHMARK_REPORT.md`:
the problems are solved one-at-a-time via the AMPL `.nl` interface, solver
output is written to per-problem `.sol` files, and results are inspected
manually rather than aggregated into a JSON summary.

## Problems

| File | Network | Phase | n | m | Ipopt status | Ipopt obj | Ipopt time | Ipopt iters |
|------|---------|-------|---|---|-------------|-----------|------------|-------------|
| `gaslib11_steady.nl` | GasLib-11 | steady | 204 | 200 | Optimal | 3.286e-2 | 0.006s | ~20 |
| `gaslib11_dynamic.nl` | GasLib-11 | 24h dynamic | 2,542 | 2,492 | Optimal | 1.699e0 | 0.033s | ~20 |
| `gaslib40_steady.nl` | GasLib-40 | steady | 1,694 | 1,682 | Optimal | 1.290e0 | 0.085s | ~100 |
| `gaslib40_dynamic.nl` | GasLib-40 | 24h dynamic | 21,058 | 20,908 | LocalInfeasibility | 5.535e1 | 10.9s | ~340 |

All problems are equality-constrained (no inequalities). The NLPs arise from
finite-volume PDE discretization of the Euler equations on a pipe network.

## Usage

```bash
# Test with ipopt
ipopt benchmarks/gas/gaslib11_steady.nl

# Test with ripopt
ripopt benchmarks/gas/gaslib11_steady.nl print_level=5
```

Or run the full suite via the benchmarks Makefile:

```bash
make -C benchmarks gas-run
```

## Re-exporting

If the gas_net package or data changes, regenerate the NL files:

```bash
python benchmarks/gas/export_nl.py
```

Requires `gas_net` installed (`pip install -e path/to/gas_networks`).
