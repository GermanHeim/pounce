//! Stamps build-time metadata into the `pounce` binary so `--about`
//! can print version/build/git/rustc info without runtime introspection,
//! and — under `--features ma57` — emits the CoinHSL `-rpath` that the
//! `pounce` binary needs to start.
//!
//! The metadata half is best-effort: missing git or `date` just becomes
//! "unknown" in the output. The rpath half is not optional; see
//! [`emit_coinhsl_rpath`].

use std::process::Command;

fn main() {
    emit_coinhsl_rpath();
    // Re-stamp when HEAD moves; otherwise the SHA in the binary is stale.
    // The .git/HEAD path is relative to this crate's manifest dir.
    println!("cargo:rerun-if-changed=../../.git/HEAD");
    println!("cargo:rerun-if-changed=../../.git/index");
    println!("cargo:rerun-if-env-changed=POUNCE_BUILD_GIT");
    println!("cargo:rerun-if-env-changed=PROFILE");
    println!("cargo:rerun-if-env-changed=TARGET");
    println!("cargo:rerun-if-env-changed=HOST");
    println!("cargo:rerun-if-env-changed=SOURCE_DATE_EPOCH");

    // POUNCE_BUILD_GIT lets a caller supply the revision when git itself is
    // not reachable from the build. The Docker source build is the motivating
    // case: .git is kept out of the build context (~90M of history the
    // compile does not need), so without the override every image would
    // report "unknown" and you could not tell which commit you were running.
    // Same escape hatch as SOURCE_DATE_EPOCH below — an explicit value wins,
    // otherwise fall back to interrogating git.
    let git = std::env::var("POUNCE_BUILD_GIT")
        .ok()
        .filter(|s| !s.trim().is_empty())
        .unwrap_or_else(|| {
            let git_sha =
                run("git", &["rev-parse", "--short=8", "HEAD"]).unwrap_or_else(|| "unknown".into());
            let dirty = run("git", &["status", "--porcelain"])
                .map(|s| if s.is_empty() { "" } else { "+dirty" })
                .unwrap_or("");
            format!("{git_sha}{dirty}")
        });

    // UTC ISO-8601 timestamp. Honor SOURCE_DATE_EPOCH for reproducible
    // builds; otherwise fall back to `date -u` at compile time.
    let build_time = std::env::var("SOURCE_DATE_EPOCH")
        .ok()
        .and_then(|s| run("date", &["-u", "-r", &s, "+%Y-%m-%dT%H:%M:%SZ"]))
        .or_else(|| run("date", &["-u", "+%Y-%m-%dT%H:%M:%SZ"]))
        .unwrap_or_else(|| "unknown".into());

    let rustc = std::env::var("RUSTC").unwrap_or_else(|_| "rustc".into());
    let rustc_version = run(&rustc, &["--version"]).unwrap_or_else(|| "rustc unknown".into());

    println!("cargo:rustc-env=POUNCE_BUILD_GIT={git}");
    println!("cargo:rustc-env=POUNCE_BUILD_TIME={build_time}");
    println!("cargo:rustc-env=POUNCE_BUILD_RUSTC={rustc_version}");
    println!(
        "cargo:rustc-env=POUNCE_BUILD_PROFILE={}",
        std::env::var("PROFILE").unwrap_or_default()
    );
    println!(
        "cargo:rustc-env=POUNCE_BUILD_TARGET={}",
        std::env::var("TARGET").unwrap_or_default()
    );
    println!(
        "cargo:rustc-env=POUNCE_BUILD_HOST={}",
        std::env::var("HOST").unwrap_or_default()
    );
}

fn run(cmd: &str, args: &[&str]) -> Option<String> {
    let out = Command::new(cmd).args(args).output().ok()?;
    if !out.status.success() {
        return None;
    }
    let s = String::from_utf8(out.stdout).ok()?.trim().to_string();
    Some(s)
}

/// Re-emit the CoinHSL `-rpath` against *this* package's targets.
///
/// `crates/pounce-hsl/build.rs` knows the CoinHSL lib directory, but
/// `cargo:rustc-link-arg` applies only to targets in the package that
/// emitted it, and pounce-hsl is a library with no binary of its own —
/// so the flag reached nothing, and an `ma57` build of `pounce` died at
/// process start with "Library not loaded: @rpath/libcoinhsl.dylib ...
/// no LC_RPATH's found" (gh#811). There is no propagating spelling of a
/// linker argument, so the directory travels as build-script metadata
/// instead: pounce-hsl declares `links = "coinhsl"` and emits
/// `cargo:rpath=<dir>`, which cargo hands to the build script of each
/// direct dependent as `DEP_COINHSL_RPATH`.
///
/// The var is present only when pounce-hsl is in the graph, i.e. under
/// `--features ma57`; a default build sees nothing here and emits
/// nothing.
///
/// `rustc-link-arg` (not `-bins`) on purpose: the integration tests in
/// `tests/` link the same dylib and have to start too, and
/// `ma57_binary_starts.rs` is one of them.
fn emit_coinhsl_rpath() {
    println!("cargo:rerun-if-env-changed=DEP_COINHSL_RPATH");
    let Ok(rpath) = std::env::var("DEP_COINHSL_RPATH") else {
        return;
    };
    if rpath.is_empty() {
        return;
    }
    println!("cargo:rustc-link-arg=-Wl,-rpath,{rpath}");
    // Leave header padding so a packaging step can rewrite the rpath
    // afterwards with `install_name_tool`. Without it a release link
    // leaves no room and `install_name_tool -add_rpath` refuses with
    // "larger updated load commands do not fit (the program must be
    // relinked)" — which is why gh#811 could not be patched in place.
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos") {
        println!("cargo:rustc-link-arg=-Wl,-headerpad_max_install_names");
    }
}
