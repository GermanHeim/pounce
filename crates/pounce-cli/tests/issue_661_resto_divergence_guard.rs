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

/// Parse the `key=value` fields of every `[PN_RESTO_LOCINF]` line.
fn locinf_lines(combined: &str) -> Vec<Vec<(String, String)>> {
    combined
        .lines()
        .filter_map(|l| l.split("[PN_RESTO_LOCINF] ").nth(1))
        .map(|rest| {
            rest.split_whitespace()
                .filter_map(|tok| {
                    let (k, v) = tok.split_once('=')?;
                    Some((k.to_string(), v.to_string()))
                })
                .collect()
        })
        .collect()
}

fn field<'a>(line: &'a [(String, String)], key: &str) -> Option<&'a str> {
    line.iter().find(|(k, _)| k == key).map(|(_, v)| v.as_str())
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

/// The mechanism, pinned rather than inferred from the exit line: the
/// gate does fire on its own terms, and the divergence guard is what
/// withholds the verdict.
#[test]
fn the_guard_is_what_withholds_the_verdict_on_the_diverging_call() {
    let combined = run_with_locinf_trace("pooling_rt2stp.nl", &["mehrotra_algorithm=yes"]);
    let lines = locinf_lines(&combined);
    assert!(
        !lines.is_empty(),
        "expected the gate trace to be emitted\n--- output ---\n{combined}",
    );

    let guarded: Vec<_> = lines
        .iter()
        .filter(|l| field(l, "diverged_from_entry") == Some("true"))
        .collect();
    assert!(
        !guarded.is_empty(),
        "pounce#661: expected at least one restoration call on this model \
         to be recognised as diverged from its entry violation\n--- output \
         ---\n{combined}",
    );

    for line in &guarded {
        // A gate did want to render the verdict — otherwise the guard is
        // not the thing being tested and this test would silently stop
        // covering the regression.
        let any_reconstructed_gate = ["strict", "alt", "cycle", "step_fail", "tiny_step"]
            .iter()
            .any(|g| field(line, g) == Some("true"));
        assert!(
            any_reconstructed_gate,
            "expected a reconstructed gate to have fired on the diverged \
             call, so the guard is what suppresses it: {line:?}",
        );
        assert_eq!(
            field(line, "loc_inf"),
            Some("false"),
            "pounce#661: the divergence guard must withhold the verdict on \
             a call whose recovered violation is far worse than the one \
             restoration was entered at: {line:?}",
        );
    }
}

/// The guard defers to `RESTO_STALL_EVIDENCE_ITERS`. A sub-solve that
/// spent 1016 inner iterations pinned at the violation it entered
/// restoration at — `1.04e-2`, to the digit, which is the infeasibility
/// gap this fixture is built around — and only then blew up over its
/// last three has demonstrated exactly the floor the gates look for. The
/// large final ratio describes those three iterations, not the run, and
/// the verdict must survive it.
///
/// This is the discriminator between the two shapes: `pooling_rt2stp`
/// and `hs71_obj1e8` exit after ~30 inner iterations with no plateau at
/// all, and `hs71_obj1e8` was still *reducing* the original violation
/// (`4.48e1` -> `2.25e1`) shortly before it diverged.
#[test]
fn a_long_stalled_sub_solve_keeps_its_verdict_despite_a_terminal_blow_up() {
    let combined = run_with_locinf_trace(
        "issue_508_infeasible_gap_1em2.nl",
        &["mehrotra_algorithm=yes"],
    );
    assert!(
        combined.contains("local infeasibility"),
        "pounce#661: this model is infeasible by a constructed 1e-2 gap,          and its restoration stalls at that gap for 1000+ inner iterations          before a terminal blow-up. The divergence guard must not read          those last few iterations as grounds to withhold the          verdict.\n--- output ---\n{combined}",
    );

    let lines = locinf_lines(&combined);
    let long_stalls: Vec<_> = lines
        .iter()
        .filter(|l| {
            field(l, "iter")
                .and_then(|v| v.parse::<i64>().ok())
                .is_some_and(|n| n >= 1000)
        })
        .collect();
    assert!(
        !long_stalls.is_empty(),
        "expected a restoration call that burned the stall-evidence \
         budget\n--- output ---\n{combined}",
    );
    for line in &long_stalls {
        assert_eq!(
            field(line, "diverged_from_entry"),
            Some("false"),
            "the divergence guard must stand down once the sub-solve has \
             burned the same iteration budget the `cycle` gate accepts as \
             stall evidence on its own: {line:?}",
        );
    }
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
