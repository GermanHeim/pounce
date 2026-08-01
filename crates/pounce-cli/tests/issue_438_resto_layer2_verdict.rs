//! Regression test for pounce#438: the restoration convergence check had
//! no layer 2.
//!
//! `IpRestoConvCheck::CheckConvergence` is two layers. Layer 1 — the
//! `kappa_resto` reduction guard, the square-problem fast path, and the
//! outer-filter acceptance test — answers *can the trial point leave
//! restoration?*. Layer 2 (`IpRestoConvCheck.cpp:200-240`) answers the
//! complementary question, *has the restoration sub-problem itself
//! converged?*, and renders one of four verdicts when it has.
//!
//! Pounce ported layer 1 and omitted layer 2, so a restoration whose
//! sub-problem had provably done everything it could was indistinguishable
//! from one still making progress: it reported the sub-problem's own
//! status and let the outer re-enter, or — when the kappa target was out
//! of reach, which it is at every iteration once restoration moves the
//! iterate away from a nearly-feasible entry point — ground on until an
//! iteration cap turned "restoration cannot succeed" into "the solve ran
//! out of iterations".
//!
//! This pins the verdict end-to-end through the binary, because the unit
//! tests in `pounce-restoration` cover [`resto_orig_verdict`]'s arms
//! individually and only a real solve proves the query is wired to the
//! nested IPM's convergence check at all.
//!
//! `issue_372_infeasible_bounds.nl` (`0 <= x <= 0.6` with `x >= 0.7`) is
//! used because it is a one-inequality contradiction: restoration drives
//! the sub-problem to its KKT point in ten inner iterations at an
//! original-NLP violation of `1e-1`, which is exactly layer 2's
//! locally-infeasible arm and nothing else.

use std::path::PathBuf;
use std::process::Command;

fn pounce_exe() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_pounce"))
}

fn fixture_nl(name: &str) -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("tests");
    p.push("fixtures");
    p.push(name);
    p
}

/// Run a fixture with the layer-2 trace enabled and return `(stdout, stderr)`.
fn run_with_layer2_trace(fixture: &str) -> (String, String) {
    let output = Command::new(pounce_exe())
        .arg(fixture_nl(fixture))
        .arg("--no-sol")
        .env("POUNCE_DBG_RESTO_LAYER2", "1")
        .env("RUST_LOG", "pounce::restoration=debug")
        .env("NO_COLOR", "1")
        .output()
        .expect("spawn pounce");
    (
        String::from_utf8_lossy(&output.stdout).into_owned(),
        String::from_utf8_lossy(&output.stderr).into_owned(),
    )
}

/// Every distinct layer-2 verdict the run logged, in first-seen order.
fn logged_verdicts(stderr: &str) -> Vec<String> {
    let mut seen: Vec<String> = Vec::new();
    for line in stderr.lines() {
        let Some(rest) = line.split("[PN_RESTO_LAYER2] verdict=").nth(1) else {
            continue;
        };
        let v = rest
            .split_whitespace()
            .next()
            .unwrap_or_default()
            .to_string();
        if !seen.contains(&v) {
            seen.push(v);
        }
    }
    seen
}

/// The core of #438: layer 2 is reached, and on a model whose restoration
/// sub-problem converges at a violated point it renders the
/// locally-infeasible verdict rather than no verdict at all.
#[test]
fn sub_problem_convergence_renders_a_locally_infeasible_verdict() {
    let (_, stderr) = run_with_layer2_trace("issue_372_infeasible_bounds.nl");
    assert_eq!(
        logged_verdicts(&stderr),
        vec!["LocallyInfeasible".to_string()],
        "pounce#438: once the restoration sub-problem converges, layer 2 \
         must render a verdict; a violation of 1e-1 against tol=1e-8 is \
         upstream's `LOCALLY_INFEASIBLE` arm \
         (`IpRestoConvCheck.cpp:240`).\n--- stderr ---\n{stderr}",
    );
}

/// The verdict must reach the user as the status it means. Before #438
/// this model was diagnosed by a post-hoc heuristic in
/// `resto_inner_solver` that reconstructed the same conclusion from the
/// inner solver's terminal status and KKT residual (the `tiny_step` gate,
/// added for gh #372); the verdict now supplies it directly, and the
/// user-facing status is unchanged.
#[test]
fn locally_infeasible_verdict_surfaces_as_the_infeasibility_status() {
    let (stdout, stderr) = run_with_layer2_trace("issue_372_infeasible_bounds.nl");
    let combined = format!("{stdout}{stderr}");
    assert!(
        combined.contains("local infeasibility"),
        "expected the local-infeasibility exit status, got:\n{combined}",
    );
}

/// Layer 2 must stay silent while restoration is still working — it is
/// gated on the sub-problem's *own* convergence check, not on elapsed
/// iterations. `pooling_rt2stp.nl` enters restoration, leaves it via
/// layer 1, and solves; no verdict should ever be rendered.
#[test]
fn no_verdict_while_the_sub_problem_is_still_making_progress() {
    let (stdout, stderr) = run_with_layer2_trace("pooling_rt2stp.nl");
    assert!(
        logged_verdicts(&stderr).is_empty(),
        "layer 2 fired on a restoration that left via layer 1: {:?}",
        logged_verdicts(&stderr),
    );
    assert!(
        format!("{stdout}{stderr}").contains("Optimal Solution Found"),
        "fixture is expected to solve; layer 2 must not have changed that",
    );
}
