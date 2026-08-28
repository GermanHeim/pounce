# pounce-hsl

HSL MA57 backend for POUNCE. Port of Ipopt's
`IpMa57TSolverInterface.{hpp,cpp}`. Implements
[`SparseSymLinearSolverInterface`](../pounce-linsol) by linking against
`libcoinhsl` at runtime.

Off by default. Enable with `--features ma57` on `pounce-cli` or
`pounce-algorithm`; the option `linear_solver = ma57` then resolves to
this backend instead of falling back to FERAL.

## Prerequisites

1. A **CoinHSL install** — build from
   [HSL for IPOPT](https://www.hsl.rl.ac.uk/ipopt/) or use a
   precompiled drop. Its `lib/` must contain `libcoinhsl.{dylib,a}`.
2. Set **`COINHSL_DIR`** to that install when building with
   `--features ma57`:

   ```sh
   COINHSL_DIR=/path/to/CoinHSL cargo build -p pounce-cli --release --features ma57
   ```

`build.rs` reads `COINHSL_DIR`, adds its `lib/` to the link search
path, and embeds an `-rpath` so `libcoinhsl` and its transitive
dependencies (`libopenblas`, `libmetis`, `libgfortran`, `libgomp`)
resolve at runtime — no `DYLD_LIBRARY_PATH` / `LD_LIBRARY_PATH` needed.

The `-rpath` reaches downstream binaries by travelling as build-script
metadata. `cargo:rustc-link-arg` applies only to targets in the package
that emits it, and this crate builds no binary, so on its own it covered
nothing outside `pounce-hsl`'s own tests — an `--features ma57` build of
the `pounce` CLI linked cleanly and then died at process start with
`Library not loaded: @rpath/libcoinhsl.dylib ... no LC_RPATH's found`
([issue #811](https://github.com/jkitchin/pounce/issues/811)). The
`links = "coinhsl"` declaration in `Cargo.toml` — which also prevents an
accidental double-link — lets `build.rs` emit `cargo:rpath=<dir>`, which
cargo hands to the build script of every *direct* dependent as
`DEP_COINHSL_RPATH`. `crates/pounce-cli/build.rs` and
`crates/pounce-algorithm/build.rs` read it and re-emit the flag against
their own targets.

**A package that grows an `ma57` feature and produces a binary or a
linked test must do the same**, or it inherits the same silent failure.
Guard: `crates/pounce-cli/tests/ma57_binary_starts.rs`.

## Why MA57?

MA57 is the canonical sparse symmetric indefinite Bunch-Kaufman
factorization used by Ipopt for its KKT solves. It handles the
indefiniteness inherent to the augmented system, reports inertia, and
supports the `increase_quality` / `pivtol` escalation that the IPM
needs when the system is nearly singular. FERAL provides the same
contract in pure Rust; MA57 is generally faster on large problems.

## License

EPL-2.0 for the wrapper. The HSL routines themselves are governed by
their own [HSL license](https://www.hsl.rl.ac.uk/licencing.html);
this crate does not bundle them.
