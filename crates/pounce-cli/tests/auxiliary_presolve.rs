//! End-to-end CLI integration test for the auxiliary-equality
//! preprocessing pass (issue #53).
//!
//! Drives the built `pounce` binary against the small
//! `parametric.nl` fixture with `presolve_auxiliary=yes` /
//! `presolve_auxiliary_diagnostics=yes` and verifies:
//!
//! 1. The binary doesn't panic.
//! 2. With diagnostics on, the stderr output contains the
//!    `auxiliary-preprocessing:` header line emitted by the
//!    diagnostics `Display` impl.
//! 3. Turning `presolve_auxiliary=no` produces the same final
//!    objective as a baseline solve (the pass really is a no-op
//!    when off).
//!
//! The real acceptance criteria for issue #53 — zero IPM iterations
//! on `tutorial_flow_density.nl` and the `gaslib11_steady.nl`
//! reduction — require vendoring those fixtures from ripopt; see
//! `crates/pounce-cli/tests/fixtures/aux_presolve/README.md` and
//! `benchmarks/preprocessing/README.md`.

use std::path::PathBuf;
use std::process::Command;

use pounce_cli::solve_report::SolveReport;

fn pounce_exe() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_pounce"))
}

fn fixture_nl() -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("tests");
    p.push("fixtures");
    p.push("parametric.nl");
    p
}

fn tmp_json(suffix: &str) -> PathBuf {
    let mut p = std::env::temp_dir();
    p.push(format!(
        "pounce_aux_{}_{suffix}.json",
        std::process::id()
    ));
    p
}

#[test]
fn presolve_auxiliary_yes_does_not_panic() {
    let output = Command::new(pounce_exe())
        .arg(fixture_nl())
        .arg("presolve=yes")
        .arg("presolve_auxiliary=yes")
        .output()
        .expect("spawn pounce");
    // Exit code 134 is SIGABRT (panic). 0 is success, 1 is a clean
    // solver-fail; either is acceptable. The point is no panic.
    let code = output.status.code().unwrap_or(-1);
    assert!(
        code == 0 || code == 1,
        "pounce exited with code {code} (stderr: {})",
        String::from_utf8_lossy(&output.stderr)
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        !stderr.contains("panicked at"),
        "pounce panicked: {stderr}"
    );
}

#[test]
fn presolve_auxiliary_diagnostics_yes_emits_summary() {
    let output = Command::new(pounce_exe())
        .arg(fixture_nl())
        .arg("presolve=yes")
        .arg("presolve_auxiliary=yes")
        .arg("presolve_auxiliary_diagnostics=yes")
        .output()
        .expect("spawn pounce");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("auxiliary-preprocessing:"),
        "expected diagnostics header in stderr, got:\n{stderr}"
    );
}

/// Sanity check: `presolve_auxiliary=no` and the baseline (no
/// presolve at all) produce the same final objective to within a
/// reasonable tolerance. This guards against accidental side effects
/// of the wrapper.
#[test]
fn presolve_auxiliary_no_unchanged_from_baseline() {
    let baseline_json = tmp_json("baseline");
    let aux_off_json = tmp_json("aux_off");

    let baseline_status = Command::new(pounce_exe())
        .arg(fixture_nl())
        .arg("--json-output")
        .arg(&baseline_json)
        .arg("--json-detail")
        .arg("summary")
        .status()
        .expect("spawn pounce baseline");
    assert!(baseline_status.success());

    let aux_off_status = Command::new(pounce_exe())
        .arg(fixture_nl())
        .arg("--json-output")
        .arg(&aux_off_json)
        .arg("--json-detail")
        .arg("summary")
        .arg("presolve=yes")
        .arg("presolve_auxiliary=no")
        .status()
        .expect("spawn pounce aux-off");
    assert!(aux_off_status.success());

    let baseline_text = std::fs::read_to_string(&baseline_json).unwrap();
    let baseline: SolveReport = serde_json::from_str(&baseline_text).unwrap();
    let aux_off_text = std::fs::read_to_string(&aux_off_json).unwrap();
    let aux_off: SolveReport = serde_json::from_str(&aux_off_text).unwrap();

    let bobj = baseline.statistics.final_objective;
    let aobj = aux_off.statistics.final_objective;
    assert!(
        (bobj - aobj).abs() < 1e-6,
        "baseline obj {bobj} vs aux-off obj {aobj}"
    );

    let _ = std::fs::remove_file(&baseline_json);
    let _ = std::fs::remove_file(&aux_off_json);
}
