//! No production code may construct an MA57 backend with hard-coded
//! defaults.
//!
//! This is the CI-runnable half of the gh#825 guard, and it exists
//! because the other half cannot run here. Asserting that the `ma57_*`
//! options reach the backend requires an MA57 backend, which requires
//! linking CoinHSL, which is licensed and not available to CI — so
//! `ma57_options_reach_the_backend.rs` is `#![cfg(feature = "ma57")]`
//! and never executes on a pull request. Without something that does,
//! the fix is protected in CI only by the fact that
//! `default_backend_factory` now *demands* a `Ma57Config`; a caller can
//! still satisfy the compiler with `Ma57Config::default()` and put the
//! defect straight back, silently, exactly as before.
//!
//! `Ma57SolverInterface::new()` is the defect's signature. It
//! hard-codes `Options::defaults()`, its own doc comment says it is for
//! tests — and it was the only path production code had. Every one of
//! the three production construction sites called it, so all nine
//! `ma57_*` options were registered, documented, accepted and dropped.
//!
//! What this test can and cannot see:
//!
//! * It reads source text. It cannot tell whether the options a call
//!   site passes are the *right* ones (the resto sites read the
//!   `"resto."` prefix; nothing here checks that) — only that the site
//!   is not asking for defaults outright.
//! * It scans `src/`, never `tests/`, and truncates each file at its
//!   first `#[cfg(test)]`. `pounce-hsl`'s own unit tests legitimately
//!   call `::new()` and live inside `src/ma57.rs`.
//! * It skips line comments, several of which name the constructor in
//!   order to warn against it — including the one on the constructor.
//! * It is deliberately not feature-gated: the source is on disk in
//!   every build, which is the whole point.

use std::path::{Path, PathBuf};

fn workspace_root() -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.pop(); // crates/
    p.pop(); // workspace root
    p
}

/// Every `.rs` file under `crates/*/src/`.
fn production_sources(root: &Path) -> Vec<PathBuf> {
    fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
        let Ok(entries) = std::fs::read_dir(dir) else {
            return;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                walk(&path, out);
            } else if path.extension().is_some_and(|e| e == "rs") {
                out.push(path);
            }
        }
    }
    let mut out = Vec::new();
    let crates = root.join("crates");
    let entries =
        std::fs::read_dir(&crates).unwrap_or_else(|e| panic!("read {}: {e}", crates.display()));
    for entry in entries.flatten() {
        let src = entry.path().join("src");
        if src.is_dir() {
            walk(&src, &mut out);
        }
    }
    assert!(
        out.len() > 100,
        "expected to find the workspace's sources under {}; found {}",
        crates.display(),
        out.len()
    );
    out
}

/// The file's text with everything from its first `#[cfg(test)]` onward
/// removed, so an in-file unit-test module does not count as production.
fn production_text(text: &str) -> &str {
    match text.find("#[cfg(test)]") {
        Some(i) => &text[..i],
        None => text,
    }
}

#[test]
fn no_production_site_builds_ma57_with_default_options() {
    let root = workspace_root();
    let mut offenders = Vec::new();
    for path in production_sources(&root) {
        let Ok(text) = std::fs::read_to_string(&path) else {
            continue;
        };
        for (lineno, line) in production_text(&text).lines().enumerate() {
            // Comments name the constructor freely — several of them
            // explain why not to call it, including the one on the
            // constructor itself.
            if line.trim_start().starts_with("//") {
                continue;
            }
            if line.contains("Ma57SolverInterface::new()") {
                offenders.push(format!(
                    "{}:{}: {}",
                    path.strip_prefix(&root).unwrap_or(&path).display(),
                    lineno + 1,
                    line.trim()
                ));
            }
        }
    }
    assert!(
        offenders.is_empty(),
        "`Ma57SolverInterface::new()` hard-codes `Options::defaults()`, so an MA57 backend \
         built this way silently discards every `ma57_*` option the user set — that is gh#825. \
         Production code has an `OptionsList`; build the backend from it, via \
         `ma57_config_from_options(options, prefix)` threaded into `default_backend_factory` \
         (prefix \"\" for the main IPM, \"resto.\" for the restoration sub-IPM), or via \
         `Ma57SolverInterface::from_options_list` directly.\n  {}",
        offenders.join("\n  ")
    );
}

/// The guard is only worth anything if the string it looks for is the
/// string the code would actually contain. If `Ma57SolverInterface` is
/// renamed, or the constructor is, the test above starts passing
/// vacuously — so pin that the needle still occurs somewhere in the
/// tree (in `pounce-hsl`'s own tests, which are allowed to use it).
#[test]
fn the_needle_still_exists() {
    let root = workspace_root();
    let ma57 = root.join("crates/pounce-hsl/src/ma57.rs");
    let text =
        std::fs::read_to_string(&ma57).unwrap_or_else(|e| panic!("read {}: {e}", ma57.display()));
    let called = text
        .lines()
        .any(|l| !l.trim_start().starts_with("//") && l.contains("Ma57SolverInterface::new()"));
    assert!(
        called,
        "no `Ma57SolverInterface::new()` anywhere in {} — if the constructor was renamed or \
         removed, update the needle in this file's sibling test, which is otherwise now \
         passing vacuously.",
        ma57.display()
    );
}
