//! Regression test for pounce#24: the `pounce` binary must not panic
//! when `l1_fallback_on_restoration_failure=yes` or
//! `l1_exact_penalty_barrier=yes` triggers more than one inner-IPM
//! solve.
//!
//! Before the fix, `pounce-cli/src/main.rs` wired the restoration
//! phase via `app.set_restoration_factory(factory)` — a one-shot
//! closure that panicked with
//! `"restoration factory invoked more than once"` on the second call.
//! The fix routes through
//! `app.set_restoration_factory_provider(...)` (the multi-pass
//! provider from pounce#10 Phase 3), which mints a fresh factory per
//! inner solve.
//!
//! This test drives the built `pounce` binary against the
//! parametric_cpp fixture with the l1 fallback enabled and verifies:
//!   1. the process exits non-fatally (no panic ⇒ no exit code 134),
//!   2. either with success (`0`) or a clean solver-fail code (`1`),
//!   3. no `restoration factory invoked more than once` message in stderr.
//!
//! parametric_cpp converges cleanly on a normal solve so the fallback
//! path isn't actually taken — but the fact that the binary built,
//! linked, and ran end-to-end without the panic is the regression
//! signal. For coverage of the actual fallback path, the Mittelmann
//! `qcqp750-2nc` test in pounce#21 will be the integration check once
//! that issue's investigation lands.

use std::path::PathBuf;
use std::process::Command;

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

#[test]
fn l1_fallback_flag_does_not_panic_in_cli() {
    let output = Command::new(pounce_exe())
        .arg(fixture_nl())
        .arg("l1_fallback_on_restoration_failure=yes")
        .output()
        .expect("spawn pounce");

    // Exit code 134 / 139 / negative signal = SIGABRT (panic on macOS
    // = signal 6 = exit 134) or other crash. We accept 0
    // (Solve_Succeeded) and 1 (any non-success terminal status).
    let code = output.status.code();
    assert!(
        matches!(code, Some(0) | Some(1)),
        "unexpected exit: {:?} stderr={}",
        output.status,
        String::from_utf8_lossy(&output.stderr),
    );

    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        !stderr.contains("restoration factory invoked more than once"),
        "panic message in stderr (pounce#24 regression):\n{stderr}",
    );
}

#[test]
fn l1_exact_penalty_barrier_flag_does_not_panic_in_cli() {
    let output = Command::new(pounce_exe())
        .arg(fixture_nl())
        .arg("l1_exact_penalty_barrier=yes")
        .output()
        .expect("spawn pounce");

    let code = output.status.code();
    assert!(
        matches!(code, Some(0) | Some(1)),
        "unexpected exit: {:?} stderr={}",
        output.status,
        String::from_utf8_lossy(&output.stderr),
    );

    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        !stderr.contains("restoration factory invoked more than once"),
        "panic message in stderr (pounce#24 regression):\n{stderr}",
    );
}
