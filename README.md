# POUNCE

![POUNCE](logos/pounce_A_pounce.png)

POUNCE is a pure-Rust port of the [Ipopt](https://github.com/coin-or/Ipopt)
interior-point nonlinear programming solver. It solves problems of the
form

```
min  f(x)
s.t. g_L <= g(x) <= g_U
     x_L <=   x  <= x_U
```

where `f` and `g` are twice-continuously-differentiable. The algorithm,
console output, and option semantics follow upstream Ipopt closely enough
that anyone used to reading `ipopt` logs can drop in `pounce` without
relearning where the numbers live.

The default build is pure Rust — no Fortran, no HSL, no system BLAS
required. The bundled [FERAL](crates/pounce-feral) backend provides a
sparse symmetric LDLᵀ factorization. The HSL MA57 backend is available
behind the optional `ma57` feature for users who have `libcoinhsl`
installed.

License: EPL-2.0 (same as upstream Ipopt).

## Status

Work in progress. The algorithm-side core, NLP interface, line search,
filter, barrier update, KKT solve, restoration phase, AMPL `.nl` reader,
and CLI are in place and solve a wide range of NLPs from the standard
test suites (Hock-Schittkowski, CUTEst, Mittelmann ampl-nlp, CHO
parameter estimation, gas/water network design). The C ABI shim
(`pounce-cinterface`) is scaffolded so existing PyIpopt / cyipopt / JuMP
/ AMPL clients can link against it; full coverage lands incrementally.

See `benchmarks/` for the comparison harness against upstream Ipopt.

## Workspace layout

| Crate                                          | Purpose |
|------------------------------------------------|---------|
| [`pounce-common`](crates/pounce-common)        | Types, exceptions, journalist, options, tagged objects, cached results (Ipopt `src/Common`). |
| [`pounce-linalg`](crates/pounce-linalg)        | BLAS-1, dense/compound vectors and matrices, triplet storage, CSC conversion (Ipopt `src/LinAlg`). |
| [`pounce-linsol`](crates/pounce-linsol)        | Symmetric linear-solver trait layer — no FFI; backends plug in below. |
| [`pounce-feral`](crates/pounce-feral)          | Pure-Rust sparse symmetric LDLᵀ backend. Default. |
| [`pounce-hsl`](crates/pounce-hsl)              | MA57 backend via `libcoinhsl` (optional, behind `ma57` feature). |
| [`pounce-nlp`](crates/pounce-nlp)              | TNLP trait, TNLPAdapter, `IpoptApplication` entry point (Ipopt `src/Interfaces`). |
| [`pounce-algorithm`](crates/pounce-algorithm)  | IteratesVector, IpoptData, calculated quantities, KKT, line search, mu update, conv check, main loop (Ipopt `src/Algorithm`). |
| [`pounce-restoration`](crates/pounce-restoration) | Restoration phase (Ipopt `Algorithm/Resto*`). |
| [`pounce-cinterface`](crates/pounce-cinterface) | C ABI shim — `IpoptCreate` / `IpoptSolve` / `IpoptFreeProblem`. |
| [`pounce-cli`](crates/pounce-cli)              | The `pounce` command-line driver. |

## Build

Prerequisites: a stable Rust toolchain. Nothing else for the default
build.

```sh
make            # release build of the workspace
make test       # run all tests
make clippy     # lint
make doc        # rustdoc
```

To build with the HSL MA57 backend (requires `libcoinhsl` discoverable
by the linker):

```sh
cargo build -p pounce-cli --release --features ma57
```

## Install

```sh
make install                # installs to $HOME/.local
sudo make install PREFIX=/usr/local   # or system-wide
```

This drops the `pounce` binary into `$PREFIX/bin` and the
`libpounce_cinterface` shared library into `$PREFIX/lib`. Make sure
`$HOME/.local/bin` is on your `PATH`.

## Usage

Solve an AMPL `.nl` file:

```sh
pounce problem.nl
pounce problem.nl print_level=8 max_iter=500 tol=1e-10
pounce problem.nl linear_solver=ma57       # with --features ma57
```

Trailing `KEY=VALUE` pairs follow the same syntax and semantics as the
upstream Ipopt CLI; they override values loaded from `--options-file`.

List available built-in test problems:

```sh
pounce --list-problems
pounce --problem hs071
```

Full help:

```sh
pounce --help
```

## Benchmarks

`benchmarks/` contains comparison harnesses against upstream Ipopt
across several suites (Hock-Schittkowski, CUTEst, Mittelmann ampl-nlp,
CHO, gas pipelines, water networks, large-scale synthetic NLPs).
Common targets:

```sh
make bench-cho          # CHO parameter-estimation
make bench-gas          # GasLib pipelines
make bench-water        # Water network design
make bench-mittelmann   # Mittelmann ampl-nlp
make bench-cutest       # CUTEst (requires one-time `make bench-cutest-prepare`)
```

See `benchmarks/README.md` for the full list and per-suite details.
