# Installation

Three routes, in the order most people want them:

| | Command | When |
|---|---|---|
| **pip** | `pip install pounce-solver` | you just want to solve something |
| **container** | `docker pull ghcr.io/jkitchin/pounce` | clusters, or nothing installed on the host |
| **source** | `make && make install` | developing POUNCE, or you want the `ma57` backend |

## With pip

```sh
pip install pounce-solver
```

Prebuilt wheels for Linux, macOS, and Windows (CPython 3.9+). No Rust
toolchain is involved. This installs both interfaces at once:

```sh
pounce problem.nl                       # the CLI
python -c "import pounce; print(pounce.__version__)"
```

For Pyomo models:

```sh
pip install pyomo-pounce
```

```python
import pyomo.environ as pyo
results = pyo.SolverFactory("pounce").solve(model)
```

Optional extras — none needed for a normal solve:

```sh
pip install "pounce-solver[jax]"     # pounce.jax autodiff frontend
pip install "pounce-solver[torch]"   # pounce.torch autodiff frontend
pip install "pounce-solver[viz]"     # debugger plots (pounce-dbg-viz)
pip install "pounce-solver[gams]"    # GAMS solver link — see gams.md
```

### If the CLI will not start (`GLIBC_2.39 not found`)

Releases up to and including 0.9.0 bundled a Linux CLI built against a
newer glibc than the wheel advertised, so `pounce` fails to exec on older
distributions (Debian 12, Ubuntu 22.04, RHEL/Rocky/Alma 8 and 9, and most
HPC images) with:

```
pounce/bin/pounce: /lib/x86_64-linux-gnu/libc.so.6: version `GLIBC_2.39' not found
```

`import pounce` still works; only the CLI (and therefore Pyomo, which
shells out to it) is affected. This is fixed for releases after 0.9.0. If
you hit it, use the [container](docker.md) or build from source below.

## With a container

No toolchain and nothing installed on the host:

```sh
docker run --rm -v "$PWD:/work" ghcr.io/jkitchin/pounce:latest problem.nl
apptainer pull pounce.sif docker://ghcr.io/jkitchin/pounce:0.9.0
```

Both images carry the CLI, the Python API, and the Pyomo plugin. See
[Docker & Containers](docker.md) for tags, bind mounts, and a Slurm
example.

## From source

### Prerequisites

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

### Build

From the repository root:

```sh
make            # release build of the workspace
make test       # run all tests
make clippy     # lint
make doc        # rustdoc for the Rust API
```

### Install

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

### HSL MA57 backend (optional)

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
