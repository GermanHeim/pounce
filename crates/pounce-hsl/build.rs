//! Tells cargo where to find `libcoinhsl.dylib` at link- and run-time.
//!
//! Set the env var `COINHSL_DIR` to a CoinHSL install whose `lib/`
//! holds `libcoinhsl.{dylib,a}`. Only consulted when the `ma57`
//! feature is enabled — this crate is left out of the default build.
//!
//! `libcoinhsl.dylib` itself depends on `libopenblas`, `libmetis`,
//! `libgfortran.5`, `libgomp.1`, all of which live next to it under
//! `@rpath`. A single `-rpath` linker arg is enough to satisfy all of
//! them at runtime.

use std::env;
use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-env-changed=COINHSL_DIR");

    let Ok(coinhsl_dir) = env::var("COINHSL_DIR").map(PathBuf::from) else {
        // No CoinHSL on this machine — compile pounce-hsl as a regular
        // rlib without emitting any link directives. Downstream crates
        // only pull pounce-hsl into a final binary when their `ma57`
        // feature is enabled, and *that* path needs CoinHSL; if a
        // downstream selects `ma57` here without COINHSL_DIR set, the
        // linker will fail with a clear "library not found: coinhsl"
        // error. The common `cargo build --workspace` (no `ma57`
        // feature) just compiles this crate as an unlinked rlib.
        println!(
            "cargo:warning=COINHSL_DIR not set; pounce-hsl compiled without link directives. \
             Selecting the `ma57` feature in a downstream crate without setting COINHSL_DIR will \
             fail at link time. Build CoinHSL from https://www.hsl.rl.ac.uk/ipopt/ and set \
             COINHSL_DIR to its install root to enable MA57."
        );
        return;
    };

    let lib_dir = coinhsl_dir.join("lib");
    assert!(
        lib_dir.is_dir(),
        "COINHSL lib directory not found: {}",
        lib_dir.display(),
    );

    let Some(lib_dir_str) = lib_dir.to_str() else {
        panic!("COINHSL lib path is not valid UTF-8: {}", lib_dir.display());
    };
    println!("cargo:rustc-link-search=native={lib_dir_str}");
    println!("cargo:rustc-link-lib=dylib=coinhsl");
    // Explicit -lopenblas so `openblas_set_num_threads` resolves at
    // link time. macOS two-level namespace will not pull the symbol
    // transitively through libcoinhsl. The dylib lives in the same
    // lib_dir, so the search path above already finds it.
    println!("cargo:rustc-link-lib=dylib=openblas");
    // libcoinhsl.dylib's @rpath dependencies live in the same lib
    // directory, so this single rpath resolves all of them.
    //
    // Two emissions, because they reach different things.
    //
    // `rustc-link-arg` applies only to targets in *this* package — so
    // it covers pounce-hsl's own integration tests and nothing else.
    // It does not reach the `pounce` binary, and that was gh#811: an
    // `ma57` build linked, then died at process start with
    // "Library not loaded: @rpath/libcoinhsl.dylib ... no LC_RPATH's
    // found". There is no propagating spelling of a linker argument,
    // so a downstream package that produces a binary has to emit its
    // own.
    println!("cargo:rustc-link-arg=-Wl,-rpath,{lib_dir_str}");
    // The propagating half. This crate declares `links = "coinhsl"`,
    // so cargo hands `cargo:rpath=<dir>` to the build script of every
    // package that depends on pounce-hsl *directly*, as the env var
    // `DEP_COINHSL_RPATH`. `crates/pounce-cli/build.rs` reads it and
    // re-emits the `-rpath` against its own targets. A new package
    // that grows an `ma57` feature and produces a binary or a test
    // must do the same; `crates/pounce-cli/tests/ma57_binary_starts.rs`
    // is the guard that the CLI's copy still works.
    println!("cargo:rpath={lib_dir_str}");
}
