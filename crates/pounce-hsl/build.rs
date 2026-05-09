//! Tells cargo where to find `libcoinhsl.dylib` at link- and run-time.
//!
//! Default search path is the precompiled CoinHSL drop the user
//! supplied; override with the env var `COINHSL_DIR` (which must
//! contain `lib/libcoinhsl.{dylib,a}`).
//!
//! `libcoinhsl.dylib` itself depends on `libopenblas`, `libmetis`,
//! `libgfortran.5`, `libgomp.1`, all of which live next to it under
//! `@rpath`. A single `-rpath` linker arg is enough to satisfy all of
//! them at runtime.

use std::env;
use std::path::PathBuf;

const DEFAULT_COINHSL_DIR: &str =
    "/Users/jkitchin/Dropbox/projects/CoinHSL.v2023.11.17.aarch64-apple-darwin-libgfortran5";

fn main() {
    println!("cargo:rerun-if-env-changed=COINHSL_DIR");

    let coinhsl_dir = env::var("COINHSL_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(DEFAULT_COINHSL_DIR));

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
    // libcoinhsl.dylib's @rpath dependencies live in the same lib
    // directory, so this single rpath resolves all of them.
    println!("cargo:rustc-link-arg=-Wl,-rpath,{lib_dir_str}");
}
