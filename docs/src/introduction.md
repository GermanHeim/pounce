# Introduction

POUNCE is a pure-Rust port of the [Ipopt](https://github.com/coin-or/Ipopt)
interior-point nonlinear programming solver. It solves problems of the
form

```text
min  f(x)
s.t. g_L <= g(x) <= g_U
     x_L <=   x  <= x_U
```

where `f` and `g` are twice-continuously-differentiable.

The algorithm, console output, and option semantics follow upstream
Ipopt closely enough that anyone used to reading `ipopt` logs can drop
in `pounce` without relearning where the numbers live.

## Pure Rust by default

The default build is pure Rust — no Fortran, no HSL, no system BLAS
required. The bundled FERAL backend provides a sparse symmetric LDLᵀ
factorization. The HSL MA57 backend is available behind the optional
`ma57` feature for users who already have `libcoinhsl` installed (see
[Installation](installation.md)).

## Status

Work in progress. The algorithm-side core, NLP interface, line search,
filter, barrier update, KKT solve, restoration phase, AMPL `.nl`
reader, and CLI are in place and solve a wide range of NLPs from the
standard test suites (Hock-Schittkowski, CUTEst, Mittelmann ampl-nlp,
CHO parameter estimation, gas/water network design). The C ABI shim
(`pounce-cinterface`) is scaffolded so existing PyIpopt / cyipopt /
JuMP / AMPL clients can link against it; full coverage lands
incrementally.

## License

EPL-2.0, the same license as upstream Ipopt.

## Where to go next

- [Installation](installation.md) — build and install POUNCE.
- [Quick Start](quick-start.md) — solve your first problem.
- [Running Solves](cli.md) — the command-line driver in depth.
- [Acknowledgments](acknowledgments.md) — the papers behind the
  algorithm.
