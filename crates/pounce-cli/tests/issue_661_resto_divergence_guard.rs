//! Regression test for pounce#661: a *diverging* restoration could be
//! reported as a converged point of local infeasibility.
//!
//! `run_inner_resto` renders the locally-infeasible verdict from six
//! gates. One of them — layer 2 — is a verdict the restoration
//! sub-solve's own convergence check issued at a point it certified.
//! The other five *reconstruct* "the sub-solve stalled at a point it
//! could not improve on" after the fact, from a terminal status plus a
//! KKT residual, and each then tests only that the recovered violation
//! is *large*.
//!
//! Large is a different claim from stalled, and the two come apart in
//! the worst direction: a restoration that is actively blowing up
//! satisfies the size test more emphatically the further it diverges.
//! On `pooling_rt2stp.nl` under `mehrotra_algorithm=yes` the
//! `step_failure` gate fired at an original-NLP violation of `7.35e5`
//! after restoration was *entered* at `6.96e0` — feasibility made
//! 105,700x worse — and the solver told the user the model may be
//! infeasible. It is not: the same model solves at default options.
//!
//! The fix adds the premise those gates already claim in their comments
//! but never tested — the recovered point must not be dramatically
//! worse than where restoration started. `Restoration_Failed`, the
//! solver admitting *it* failed, is the honest answer for a blow-up;
//! `Infeasible_Problem_Detected` is an affirmative claim about the
//! user's model and must not rest on a diverging trajectory.
//!
//! Note on provenance: #619 did not introduce this. Its change of
//! starting point only made the inner explode at iteration 32 rather
//! than 19, and the gate carries an `iter >= 30` floor — identical
//! divergence on either side of #619, opposite verdict, decided by an
//! iteration count.
//!
//! Scope. These are behavioural tests only — they assert what the user is
//! told, not how the guard reached it. The guard's decision rule is a pure
//! function, `diverged_from_restoration_entry`, unit-tested directly in
//! `pounce-restoration::resto_inner_solver`; that is where the plateau /
//! blow-up discrimination, the floor waiver and the boundaries are pinned,
//! and `pounce_algorithm::inf_pr_floor` is where the floor evidence the
//! waiver reads is measured and tested.
//! An earlier revision of this file asserted the mechanism end-to-end by
//! parsing the `POUNCE_DBG_RESTO_LOCINF` trace, and it did not survive
//! contact with a second platform: on x86_64-linux this restoration
//! explodes at inner iteration 15 rather than 32, so it never clears the
//! `step_failure` gate's `iter >= 30` floor and the trace legitimately
//! shows no gate firing at all. *Which* gate a trajectory reaches is not a
//! property this fix owns, and asserting it made the test a trajectory
//! detector rather than a regression test. Whether these end-to-end cases
//! are non-vacuous on a given platform therefore varies — they are
//! provably non-vacuous on aarch64-darwin, where a baseline binary prints
//! the false verdict — which is the other reason the mechanism is pinned
//! in unit tests.

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

/// Run a fixture with the locally-infeasible gate trace enabled.
fn run_with_locinf_trace(fixture: &str, opts: &[&str]) -> String {
    let output = Command::new(pounce_exe())
        .arg(fixture_nl(fixture))
        .arg("--no-sol")
        .args(opts)
        .env("POUNCE_DBG_RESTO_LOCINF", "1")
        .env("RUST_LOG", "pounce::restoration=debug")
        .env("NO_COLOR", "1")
        .output()
        .expect("spawn pounce");
    format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    )
}

/// The user-facing half of #661: a model that solves at default options
/// must never be reported as possibly infeasible just because one
/// option cascade sends its restoration divergent.
#[test]
fn a_diverging_restoration_is_not_reported_as_an_infeasible_model() {
    let combined = run_with_locinf_trace("pooling_rt2stp.nl", &["mehrotra_algorithm=yes"]);
    assert!(
        !combined.contains("local infeasibility"),
        "pounce#661: `pooling_rt2stp.nl` is feasible — it solves at \
         default options. Under `mehrotra_algorithm=yes` its restoration \
         diverges, and a diverging restoration must exit \
         `Restoration_Failed`, not claim the model may be \
         infeasible.\n--- output ---\n{combined}",
    );
}

/// A second, independent instance of the same defect, on a different
/// model. `hs71_obj1e8` is plain HS071 with its objective scaled by 1e8;
/// it solves in 11 iterations at default options. Under the same option
/// cascade its restoration was entered at `1.04e2` and rendered the
/// verdict at `6.79e9`, while still *reducing* the original violation
/// (`4.48e1` -> `2.25e1`) shortly before it diverged — a sub-solve making
/// progress, not one out of room.
#[test]
fn a_second_feasible_model_is_not_reported_infeasible_either() {
    let combined = run_with_locinf_trace("hs71_obj1e8.nl", &["mehrotra_algorithm=yes"]);
    assert!(
        !combined.contains("local infeasibility"),
        "pounce#661: `hs71_obj1e8.nl` is feasible — it solves in 11 \
         iterations at default options — so no option cascade may lead the \
         solver to report it as possibly infeasible.\n--- output \
         ---\n{combined}",
    );
}

/// The guard must not cost genuine detection. A one-inequality
/// contradiction (`0 <= x <= 0.6` with `x >= 0.7`) is infeasible, its
/// restoration converges rather than diverges, and it must still be
/// diagnosed as such.
#[test]
fn a_genuinely_infeasible_model_is_still_diagnosed() {
    let combined = run_with_locinf_trace("issue_372_infeasible_bounds.nl", &[]);
    assert!(
        combined.contains("local infeasibility"),
        "the divergence guard must not suppress a real \
         local-infeasibility diagnosis\n--- output ---\n{combined}",
    );
}
