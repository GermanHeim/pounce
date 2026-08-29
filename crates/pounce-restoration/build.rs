//! Re-emits the CoinHSL `-rpath` against this package's test binaries.
//!
//! `cargo:rustc-link-arg` applies only to targets in the package that
//! emitted it, so `crates/pounce-hsl/build.rs`'s copy reaches nothing
//! downstream and an MA57-linked target dies at process start with
//! "Library not loaded: @rpath/libcoinhsl.dylib" (gh#811). The directory
//! travels as build-script metadata instead: pounce-hsl declares
//! `links = "coinhsl"` and emits `cargo:rpath=<dir>`, which cargo hands
//! to each **direct** dependent's build script as `DEP_COINHSL_RPATH` —
//! hence the optional `pounce-hsl` dependency in `Cargo.toml`, which is
//! there to make this crate a direct dependent and for nothing else.
//!
//! Kept in sync with the identical copies in `pounce-algorithm`,
//! `pounce-cli`, `pounce-cinterface` and `pounce-py`.

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
