//! Regression test for pounce#439: `required_infeasibility_reduction`
//! was registered as a user option (and documented with upstream's help
//! text) but nothing read it — the κ_resto the restoration sub-solve's
//! early-exit guard runs with was hardcoded to upstream's `0.9` default
//! in `run_inner_resto`. Setting the option produced no error, no
//! warning, and no effect.
//!
//! This pins the whole chain end-to-end through the binary: options list
//! → `AlgorithmBuilder::resto` → `RestoAlgorithmBuilder` →
//! `RestoConvCheck::kappa_resto`. The unit tests in `pounce-restoration`
//! cover the links individually; only a real solve proves they are
//! actually joined up at the callsite.
//!
//! `pooling_rt2stp.nl` is used because it enters restoration with a
//! non-square original NLP, so the guard is live and the square-problem
//! override (`IpRestoMinC_1Nrm.cpp:157-163`, which forces κ to 0) does
//! not mask the user's value.

use std::path::PathBuf;
use std::process::Command;

fn pounce_exe() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_pounce"))
}

fn fixture_nl() -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("tests");
    p.push("fixtures");
    p.push("pooling_rt2stp.nl");
    p
}

/// Run the fixture and return every distinct `kappa_resto=` value the
/// restoration guard logged. `POUNCE_DBG_RESTO_KAPPA` gates the trace;
/// the guard only logs when it is live (`kappa_resto > 0`).
fn logged_kappas(extra_args: &[&str]) -> Vec<String> {
    let mut cmd = Command::new(pounce_exe());
    cmd.arg(fixture_nl())
        .arg("print_level=0")
        .arg("--no-sol")
        .args(extra_args)
        .env("POUNCE_DBG_RESTO_KAPPA", "1")
        .env("RUST_LOG", "pounce::restoration=debug")
        .env("NO_COLOR", "1");
    let output = cmd.output().expect("spawn pounce");
    let stderr = String::from_utf8_lossy(&output.stderr);

    let mut seen: Vec<String> = stderr
        .lines()
        .filter_map(|l| l.split("kappa_resto=").nth(1))
        .map(|rest| {
            rest.split_whitespace()
                .next()
                .unwrap_or_default()
                .to_string()
        })
        .collect();
    seen.sort();
    seen.dedup();
    seen
}

#[test]
fn default_kappa_resto_is_upstreams_default() {
    let kappas = logged_kappas(&[]);
    assert_eq!(
        kappas,
        vec!["9.000e-1".to_string()],
        "with the option unset the guard must run at upstream's 0.9 \
         default (behavior before #439 must be unchanged)",
    );
}

#[test]
fn user_required_infeasibility_reduction_reaches_the_guard() {
    let kappas = logged_kappas(&["required_infeasibility_reduction=0.05"]);
    assert_eq!(
        kappas,
        vec!["5.000e-2".to_string()],
        "pounce#439: setting `required_infeasibility_reduction` must \
         change the κ_resto the restoration guard runs with, not be a \
         silent no-op",
    );
}

#[test]
fn zero_required_infeasibility_reduction_disables_the_guard() {
    // Upstream treats κ_resto == 0 as "no reduction requirement" — the
    // sub-solve runs to its own convergence instead of exiting early.
    // `RestoConvCheck` skips the whole block (and therefore the trace)
    // in that case.
    let kappas = logged_kappas(&["required_infeasibility_reduction=0.0"]);
    assert!(
        kappas.is_empty(),
        "kappa 0 must disable the reduction guard entirely, got {kappas:?}",
    );
}
