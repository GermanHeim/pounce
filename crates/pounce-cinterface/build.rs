//! Re-emits the CoinHSL `-rpath` against this package's test binaries.
//!
//! The third copy of the same three lines, and the third package to need
//! them: `cargo:rustc-link-arg` applies only to targets in the package
//! that emitted it, so `crates/pounce-hsl/build.rs`'s copy reaches
//! nothing downstream and an MA57-linked executable dies at process
//! start with "Library not loaded: @rpath/libcoinhsl.dylib" (gh#811).
//! The directory travels as build-script metadata instead: pounce-hsl
//! declares `links = "coinhsl"` and emits `cargo:rpath=<dir>`, which
//! cargo hands to the build script of each **direct** dependent as
//! `DEP_COINHSL_RPATH` — hence the optional `pounce-hsl` dependency in
//! `Cargo.toml`, which exists to make this crate a direct dependent and
//! for no other reason. `pounce-cli` carries the same one.
//!
//! What needs it here is the crate's own test binaries, which under
//! `ma57` link the dylib. `cargo test -p pounce-cinterface --features
//! ma57` aborted at process start before this existed, on `main` and on
//! every branch — CI never saw it because the CI job runs `cargo check
//! -p pounce-hsl`, which neither links nor runs anything, and every
//! other job passes `--exclude pounce-hsl`. So the whole MA57 link path
//! is exercised only by someone with a CoinHSL install.
//!
//! This crate matters for that path more than its position in the
//! dependency graph suggests: `libpounce_cinterface` is what the CasADi
//! plugin loads, so a `linear_solver=ma57` build reaches a user through
//! here.
//!
//! Kept in sync with `crates/pounce-algorithm/build.rs` and
//! `crates/pounce-cli/build.rs`, which do the same for their own targets.

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
