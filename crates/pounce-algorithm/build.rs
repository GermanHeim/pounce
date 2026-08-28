//! Re-emits the CoinHSL `-rpath` against this package's test binaries.
//!
//! `crates/pounce-hsl/build.rs` knows the CoinHSL lib directory, but
//! `cargo:rustc-link-arg` applies only to targets in the package that
//! emitted it, and pounce-hsl is a library with no binary of its own —
//! so the flag reached nothing and an MA57-linked executable died at
//! process start with "Library not loaded: @rpath/libcoinhsl.dylib ...
//! no LC_RPATH's found" (gh#811). There is no propagating spelling of a
//! linker argument, so the directory travels as build-script metadata:
//! pounce-hsl declares `links = "coinhsl"` and emits `cargo:rpath=<dir>`,
//! which cargo hands to the build script of each direct dependent as
//! `DEP_COINHSL_RPATH`.
//!
//! This crate ships no binary; what needs the rpath is
//! `tests/ma57_through_t_sym_solver.rs` and
//! `tests/ma57_options_reach_the_backend.rs`, which link the dylib and
//! run under `cargo test -p pounce-algorithm --features ma57`. The env
//! var exists only when pounce-hsl is in the graph, i.e. under that
//! feature; a default build emits nothing here.
//!
//! Kept in sync with `crates/pounce-cli/build.rs`, which does the same
//! for the `pounce` binaries.

fn main() {
    println!("cargo:rerun-if-env-changed=DEP_COINHSL_RPATH");
    let Ok(rpath) = std::env::var("DEP_COINHSL_RPATH") else {
        return;
    };
    if rpath.is_empty() {
        return;
    }
    println!("cargo:rustc-link-arg=-Wl,-rpath,{rpath}");
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos") {
        // Header padding so a packaging step can rewrite the rpath with
        // `install_name_tool` afterwards; a release link otherwise leaves
        // no room and the tool refuses.
        println!("cargo:rustc-link-arg=-Wl,-headerpad_max_install_names");
    }
}
