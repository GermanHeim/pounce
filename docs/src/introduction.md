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

The default build is pure Rust — no Fortran, no commercial solver, no system BLAS
required. The bundled FERAL backend provides a sparse symmetric LDLᵀ
factorization. The HSL MA57 backend is available behind the optional
`ma57` feature for users who have a license for `libcoinhsl` and have it installed (see
[Installation](installation.md)).

## Status

Production-ready for the core IPM workflow. The algorithm-side core,
NLP interface, line search, filter, barrier update (monotone +
Mehrotra adaptive), KKT solve, restoration phase, AMPL `.nl` reader,
the C ABI (`pounce-cinterface`), the Python wrapper (`pounce-solver`),
and the CLI all solve a wide range of NLPs from the standard test
suites (Hock-Schittkowski, CUTEst, Mittelmann ampl-nlp, CHO parameter
estimation, gas/water network design). Sensitivity analysis (sIPOPT
port), reduced-Hessian computation, the auxiliary-equality + FBBT
presolve, and the active-set SQP path are all wired in and available
behind option keys. Existing PyIpopt / cyipopt / JuMP / AMPL clients
link against `libpounce_cinterface` in place of `libipopt`
unchanged.

## License

EPL-2.0, the same license as upstream Ipopt.

## Where to go next

- [Installation](installation.md) — build and install POUNCE.
- [Quick Start](quick-start.md) — solve your first problem.
- [Running Solves](cli.md) — the command-line driver in depth.
- [Acknowledgments](acknowledgments.md) — the papers behind the
  algorithm.
