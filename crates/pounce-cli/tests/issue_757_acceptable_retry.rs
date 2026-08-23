//! gh #757 — the default-on μ-strategy retry takes
//! `Solved_To_Acceptable_Level` when the caller left the convergence
//! configuration alone.
//!
//! pounce#748 turned `mu_strategy_fallback` on by default but narrowed
//! its trigger to `Maximum_Iterations_Exceeded`, refusing
//! `Solved_To_Acceptable_Level` outright. Two of the three reasons it
//! gave are properties of a *caller-modified* configuration rather than
//! of the status: retrying launders a downgrade the caller armed an
//! option to produce, and it can hand back the other run's point. Both
//! are addressed by deferring to `TERMINATION_POLICY_OPTIONS` instead of
//! to the status alone, which is what these tests pin.
//!
//! The motivating model is `cho_parmest`, which lives in the external
//! benchmark corpus and so cannot be a fixture here; `csfi2` reproduces
//! the same promotion on an in-repo model.

use std::process::Command;

fn pounce_exe() -> String {
    let mut p = std::path::PathBuf::from(env!("CARGO_BIN_EXE_pounce"));
    p.set_extension(std::env::consts::EXE_EXTENSION);
    p.to_string_lossy().into_owned()
}

fn fixture(name: &str) -> String {
    format!("{}/tests/fixtures/{name}", env!("CARGO_MANIFEST_DIR"))
}

/// `(exit line, iteration count)` of the *last* solve the run reported.
fn solve(extra: &[&str]) -> (String, u32) {
    let mut cmd = Command::new(pounce_exe());
    cmd.arg(fixture("csfi2.nl")).arg("--no-sol");
    for o in extra {
        cmd.arg(o);
    }
    let out = cmd.output().expect("spawn pounce");
    let text = String::from_utf8_lossy(&out.stdout).into_owned();
    let exit = text
        .lines()
        .filter(|l| l.contains("EXIT:"))
        .next_back()
        .unwrap_or_default()
        .trim()
        .to_string();
    let iters = text
        .lines()
        .filter(|l| l.contains("Number of Iterations"))
        .next_back()
        .and_then(|l| l.rsplit(' ').next().and_then(|v| v.parse().ok()))
        .unwrap_or(0);
    (exit, iters)
}

/// Stock options: the acceptable-level exit is POUNCE's own stall, so it
/// earns one retry under the flipped schedule — and that retry certifies.
#[test]
fn stock_options_retry_an_acceptable_level_exit() {
    let (exit, iters) = solve(&[]);
    assert!(
        exit.contains("Optimal Solution Found"),
        "expected the retry to promote csfi2 to a certificate, got {exit:?}",
    );
    assert!(
        iters > 0 && iters < 30,
        "expected the promoting (adaptive) attempt's iteration count, got {iters}",
    );
}

/// `mu_strategy_fallback=no` restores upstream's single solve.
#[test]
fn an_explicit_no_still_declines() {
    let (exit, _) = solve(&["mu_strategy_fallback=no"]);
    assert!(
        exit.contains("Solved To Acceptable Level"),
        "an explicit no must not retry, got {exit:?}",
    );
}

/// A caller-armed termination guard means the downgrade may be the
/// signal the caller asked for, so the retry stands down — this is the
/// pounce#748 laundering objection, preserved.
#[test]
fn a_caller_set_termination_option_declines_the_retry() {
    for opt in [
        "kkt_fidelity_tol=1e-14",
        "acceptable_iter=3",
        "dual_diverging_streak=1",
        "resto_decline_deferrals=0",
        "tol=1e-9",
    ] {
        let (exit, _) = solve(&[opt]);
        assert!(
            !exit.contains("Optimal Solution Found"),
            "{opt} must suppress the default-on retry, got {exit:?}",
        );
    }
}

/// The caller can always override the guard back on.
#[test]
fn an_explicit_yes_overrides_a_caller_set_termination_option() {
    let (exit, _) = solve(&["kkt_fidelity_tol=1e-14", "mu_strategy_fallback=yes"]);
    assert!(
        exit.contains("Optimal Solution Found") || exit.contains("Solved To Acceptable Level"),
        "an explicit yes must still run the retry path, got {exit:?}",
    );
}
