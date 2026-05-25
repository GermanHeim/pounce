# Water Distribution Network NLP Benchmarks

Water distribution network design instances from
[MINLPLib](https://www.minlplib.org), downloaded directly as AMPL NL files.

This suite lives under `benchmarks/water/`. Like the gas suite, it does **not**
feed into the composite `BENCHMARK_REPORT.md`: problems are solved one-at-a-time
via the AMPL `.nl` interface, solver output is written to per-problem `.sol`
files, and results are inspected manually.

## Problems

All instances are water network design problems. The underlying physics
(Hazen-Williams head-loss, mass balance, pressure constraints) produces
signomial / nonconvex nonlinear constraints. `water.nl` is the only pure NLP;
the others include binary pipe-choice variables that both Ipopt and ripopt
treat as continuous (MINLPLib stores them as a relaxation in `.nl`).

| File | Type | n | m | Ipopt status | Ipopt obj | Ipopt iters |
|------|------|---|---|--------------|-----------|-------------|
| `water.nl` | NLP | 41 | 25 | Optimal | 1.0012e3 | 102 |
| `waterx.nl` | MBNLP (14 bin, relaxed) | 70 | 54 | Optimal | 9.5088e2 | 139 |
| `water3.nl` | MBNLP (28 bin, relaxed) | 195 | 137 | Optimal | 2.4904e2 | 184 |
| `water4.nl` | MBNLP (relaxed) | — | — | Optimal | 3.0302e2 | 137 |
| `waterz.nl` | MBNLP (relaxed) | — | — | Optimal | 2.0751e2 | 419 |
| `watersbp.nl` | MBNLP (relaxed) | — | — | Optimal | 2.0726e2 | 154 |

The best-known primal bound for `water.nl` on MINLPLib is **963.1343**, which
ripopt matches; the local minimum Ipopt reports (1001.16) is a different
stationary point.

## Sources

- Small instances (`water`, `waterx`, `water3`, `water4`): GAMS Model Library,
  based on Drud & Rosenborg (1973), *Dimensioning Water Distribution Networks*,
  Technical University of Denmark.
- Canonical water-network-design paper:
  Bragalli, D'Ambrosio, Lee, Lodi, Toth (2012),
  [*On the optimal design of water distribution networks: a practical MINLP
  approach*](https://link.springer.com/article/10.1007/s11081-011-9141-7),
  *Optim. Eng.* 13, 219–246. The Bragalli `waternd_*` instances on MINLPLib
  are distributed only as `.osil` / `.gms` / `.py`, not `.nl`, so they are
  not included here.

## Usage

```bash
# Test with system ipopt
ipopt benchmarks/water/water.nl

# Test with ripopt
cargo run --bin ripopt --release -- benchmarks/water/water.nl
```

Or run the full suite via the benchmarks Makefile:

```bash
make -C benchmarks water-run
```

## Re-downloading

```bash
cd benchmarks/water
for f in water waterx water3 water4 waterz watersbp; do
    curl -sSO "https://www.minlplib.org/nl/$f.nl"
done
```
