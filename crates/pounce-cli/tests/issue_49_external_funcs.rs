//! Integration test for issue #49: AMPL imported (external) functions.
//!
//! Drives the `pounce` binary on `idaes_helmholtz.nl`, a real-world
//! fixture from the IDAES general-Helmholtz example that calls three
//! externally-defined AMPL functions (`vf_hp`, `h_liq_hp`, `h_vap_hp`).
//! With `AMPLFUNC` pointing at the installed Helmholtz dylib, pounce
//! must (a) parse the F/funcall tokens, (b) load the library via
//! `funcadd_ASL`, (c) build a tape whose `TapeOp::Funcall` nodes
//! evaluate through the live ABI, and (d) drive the IPM to a clean
//! `EXIT: Optimal Solution Found`.
//!
//! The test is skipped when the Helmholtz dylib isn't installed
//! locally — this is an integration-with-real-library check, not a
//! ripopt-style minimal unit test.

use std::path::PathBuf;
use std::process::Command;

fn pounce_exe() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_pounce"))
}

fn fixture_path() -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("tests");
    p.push("fixtures_issue_49");
    p.push("idaes_helmholtz.nl");
    p
}

fn helmholtz_dylib() -> Option<PathBuf> {
    let home = std::env::var_os("HOME")?;
    let dylib = PathBuf::from(home).join(".idaes/bin/general_helmholtz_external.dylib");
    if dylib.exists() {
        Some(dylib)
    } else {
        None
    }
}

#[test]
fn pounce_solves_idaes_helmholtz_with_external_functions() {
    let dylib = match helmholtz_dylib() {
        Some(p) => p,
        None => {
            eprintln!("skipping: ~/.idaes/bin/general_helmholtz_external.dylib not installed");
            return;
        }
    };

    let out = Command::new(pounce_exe())
        .env("AMPLFUNC", dylib.as_os_str())
        .arg(fixture_path())
        .arg("max_iter=50")
        .arg("print_level=0")
        .output()
        .expect("spawn pounce binary");

    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);

    assert!(
        out.status.success(),
        "pounce exited non-zero:\n--- stdout ---\n{stdout}\n--- stderr ---\n{stderr}"
    );
    assert!(
        stdout.contains("Optimal Solution Found"),
        "expected optimal exit, got:\n--- stdout ---\n{stdout}\n--- stderr ---\n{stderr}"
    );
}

#[test]
fn pounce_rejects_external_function_problem_without_amplfunc() {
    let out = Command::new(pounce_exe())
        .env_remove("AMPLFUNC")
        .arg(fixture_path())
        .output()
        .expect("spawn pounce binary");

    assert!(!out.status.success(), "pounce should fail without AMPLFUNC");
    let combined = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(
        combined.contains("AMPLFUNC") || combined.to_lowercase().contains("external function"),
        "error should mention AMPLFUNC or external functions, got: {combined}"
    );
}
