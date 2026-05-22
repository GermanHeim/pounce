# Installation

## Prerequisites

A stable Rust toolchain. Nothing else is needed for the default
pure-Rust build. Install Rust via [rustup](https://rustup.rs/):

```sh
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source "$HOME/.cargo/env"
```

Verify the install:

```sh
rustc --version && cargo --version
```

## Build

From the repository root:

```sh
make            # release build of the workspace
make test       # run all tests
make clippy     # lint
make doc        # rustdoc for the Rust API
```

## Install

```sh
make install                          # installs to $HOME/.local
sudo make install PREFIX=/usr/local   # or system-wide
```

This drops the `pounce` binary into `$PREFIX/bin` and the
`libpounce_cinterface` shared library into `$PREFIX/lib`. Make sure
`$HOME/.local/bin` is on your `PATH`, then verify:

```sh
pounce --version
```

## HSL MA57 backend (optional)

The default FERAL backend needs no external libraries. To build with
the HSL MA57 linear solver instead, you need a CoinHSL install whose
`lib/` directory holds `libcoinhsl`. Point the `COINHSL_DIR`
environment variable at it and build with the `ma57` feature:

```sh
export COINHSL_DIR=/path/to/CoinHSL
cargo build -p pounce-cli --release --features ma57
```

Build CoinHSL from <https://www.hsl.rl.ac.uk/ipopt/>. MA57 is
primarily useful for benchmarking against upstream Ipopt; the FERAL
backend is the supported default for everyday use, and a build without
`--features ma57` never touches `COINHSL_DIR`.

## Using POUNCE as a Rust library

The workspace is a set of library crates (see
[Algorithm & Workspace](algorithm.md) for the layout). To browse the
Rust API, build and open the rustdoc:

```sh
make doc        # generates target/doc
```
