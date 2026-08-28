//! gh #818 — a line-search failure at an *already feasible* point is not a
//! restoration problem.
//!
//! When the backtracking line search cannot accept any trial step, either
//! the **point** is bad — infeasible, and the restoration phase is exactly
//! the right tool — or the **direction** is, because `W` is a quasi-Newton
//! model carrying curvature the iterate has left behind. Upstream has one
//! answer for both, because restoration is the only fallback it has.
//!
//! At a feasible point that answer is a no-op. The restoration NLP minimizes
//! the constraint violation and there is none to minimize, so it wanders at
//! `theta ~ 1e-13` and reports `Restoration_Failed`. Measured on `deb7` under
//! `limited-memory`: the solve stalls with `inf_pr ~ 1e-12` and `inf_du ~ 1e5`,
//! enters restoration at a point feasible to 8e-13, and spends 340 of its 1242
//! iterations there. On an unconstrained model `theta` is identically zero, so
//! restoration cannot move at all.
//!
//! `IpoptAlgorithm::try_reanchor_before_restoration` puts one rung in front of
//! the hand-off: drop every curvature pair but the newest and retry. It is not
//! a feasibility gate on restoration — that was tried and rejected before (see
//! the `constr_viol_tol` paragraph in `IpoptAlgorithm::invoke_restoration`) —
//! and it runs *after* the acceptable-point decline, so a reportable point is
//! still reported.
//!
//! **The rung ships off.** `limited_memory_ls_failure_restarts` defaults to
//! `0` — upstream's unconditional hand-off — because measured on top of the
//! safeguarded interpolation that actually fixes gh #818, it does not pay for
//! itself over the fixture corpus: `deb7` 715 -> 610 and
//! `issue_508_infeasible_gap_1em4` 79 -> 76 in its favour, against `eigena2`
//! 91 -> 98, `pooling_rt2stp` 295 -> 307 and `infeasible_square_scaled_1em4`
//! 24 -> 26, the last two being models the shipped configuration leaves
//! exactly where `main` had them. An earlier draft of this file measured the
//! rung against `main` instead and read all five as wins.
//!
//! Every test here therefore sets the option explicitly on both sides. Both
//! directions are pinned deliberately: the two wins are why the code stays in
//! the tree and stays documented, and the two costs are why it is not the
//! default. A change that moves any of the four is a change to that decision.
//!
//! **What this file is not evidence about.** It pins that the rung is wired,
//! fires, is switchable, and which way it moves four named models. It does not
//! pin the corpus — that is `scripts/sweep-fixtures.sh` — and it says nothing
//! about the exact-Hessian arm, which has no curvature history to re-anchor.
//! Note also that setting the option at all, `0` included, opts the solve out
//! of the `Solved_To_Acceptable_Level` re-solve ladder
//! (`TERMINATION_POLICY_OPTIONS`), so every cell below runs in a regime the
//! shipped default never enters.

use pounce_solve_report::SolveReport;
use std::path::PathBuf;
use std::process::Command;

fn pounce_exe() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_pounce"))
}

fn fixture_named(name: &str) -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("tests");
    p.push("fixtures");
    p.push(name);
    p
}

/// Solve `fixture` on the limited-memory arm with `restarts` re-anchors
/// allowed, returning its JSON report. The CLI's exit code is not asserted:
/// `eigena2` fails on this arm either way, and the point of the test is the
/// iteration count, not the status.
fn solve(fixture: &str, tag: &str, restarts: u32) -> SolveReport {
    let json = std::env::temp_dir().join(format!("pounce_issue_818_{tag}.json"));
    let _ = std::fs::remove_file(&json);
    let out = Command::new(pounce_exe())
        .arg(fixture_named(fixture))
        .arg("--no-sol")
        .arg("--json-output")
        .arg(&json)
        .arg("hessian_approximation=limited-memory")
        .arg(format!("limited_memory_ls_failure_restarts={restarts}"))
        .output()
        .expect("spawn pounce");
    let text = std::fs::read_to_string(&json).unwrap_or_else(|e| {
        panic!(
            "JSON report should be written for {fixture} ({tag}): {e}\n{}{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        )
    });
    let _ = std::fs::remove_file(&json);
    serde_json::from_str(&text).expect("deserialize report")
}

fn iters(r: &SolveReport) -> i32 {
    r.statistics.iteration_count
}

/// Run `fixture` and return `(how many times the rung fired, the status line)`.
///
/// The re-anchor is not in the JSON report — it is one `tracing` line at debug
/// level, which is the only place the *decision* is observable rather than its
/// downstream effect on a trajectory. `issue_438_resto_layer2_verdict.rs` reads
/// the restoration layer's verdict the same way.
fn solve_counting_reanchors(fixture: &str, tag: &str, restarts: u32) -> (usize, String) {
    let out = Command::new(pounce_exe())
        .arg(fixture_named(fixture))
        .arg("--no-sol")
        .arg("hessian_approximation=limited-memory")
        .arg(format!("limited_memory_ls_failure_restarts={restarts}"))
        .env("RUST_LOG", "pounce::algorithm=debug")
        .env("NO_COLOR", "1")
        .output()
        .unwrap_or_else(|e| panic!("spawn pounce for {fixture} ({tag}): {e}"));
    let combined = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    let fires = combined
        .lines()
        .filter(|l| l.contains("re-anchoring the limited-memory Hessian"))
        .count();
    (fires, combined)
}

/// `issue_508_infeasible_gap_1em4` is the cheap, unambiguous win: the rung
/// fires, the solve reaches the *same* verdict — the problem really is
/// infeasible — and gets there in fewer iterations (79 -> 76).
///
/// The verdict assertion is the load-bearing half. A rung that reached the
/// answer faster by giving up on a genuine restoration would show here as a
/// changed status, which is precisely the failure mode the ordering against
/// the acceptable-point decline is meant to prevent.
#[test]
fn reanchor_shortens_the_infeasibility_certificate_without_changing_it() {
    let off = solve("issue_508_infeasible_gap_1em4.nl", "gap_off", 0);
    let on = solve("issue_508_infeasible_gap_1em4.nl", "gap_on", 1);

    assert_eq!(
        off.solution.status, on.solution.status,
        "the rung must not change the verdict, only how long it takes"
    );
    assert!(
        matches!(
            off.solution.status,
            pounce_nlp::return_codes::ApplicationReturnStatus::InfeasibleProblemDetected
        ),
        "expected an infeasibility certificate, got {:?}",
        off.solution.status
    );
    assert!(
        iters(&on) < iters(&off),
        "the rung fired but cost iterations: {} with, {} without",
        iters(&on),
        iters(&off)
    );
}

/// `deb7` is the stall the rung was designed from, and the reason the code
/// stays in the tree. What is pinned here is that the rung **fires** on it,
/// and that firing does not manufacture a success.
///
/// **The iteration count is deliberately not asserted, because it is not
/// portable.** An earlier revision of this test asserted that the rung
/// shortens `deb7` (measured 715 without / 610 with). That ordering does not
/// reproduce: on Linux, at this commit, `deb7` runs 455 iterations without
/// the rung and 729 with — identical under `--release` and under the
/// unoptimised profile CI actually builds, so it is not an optimisation
/// artefact, and CI reported exactly those two numbers. A stalling solve's
/// trajectory depends on the host's linear algebra, so any strict inequality
/// on its length is a test that passes where it was measured and fails
/// elsewhere.
///
/// What survives that is the mechanism: the re-anchor either happens or it
/// does not, and that is a property of the code rather than of the arithmetic
/// it runs on. `limited_memory_ls_failure_restarts` bounds it at one per
/// solve, so "fires exactly once with the budget, never without it" is exact.
///
/// The second assertion is the safety half. `deb7` does not solve on this arm
/// under any setting measured, so a `Solve_Succeeded` appearing here would be
/// the rung turning a stall into a spurious success — a wrong answer, and the
/// one outcome that would matter more than any iteration count.
#[test]
fn the_rung_fires_on_the_model_it_was_designed_from() {
    let (off_fires, off_status) = solve_counting_reanchors("deb7.nl", "deb7_off", 0);
    let (on_fires, on_status) = solve_counting_reanchors("deb7.nl", "deb7_on", 1);

    assert_eq!(
        off_fires, 0,
        "the default budget is 0, so nothing may re-anchor; saw {off_fires}"
    );
    assert_eq!(
        on_fires, 1,
        "a budget of 1 must be spent exactly once on deb7 — the model the rung \
         was designed from; saw {on_fires}"
    );

    for (label, status) in [("off", &off_status), ("on", &on_status)] {
        assert!(
            !status.contains("Solve_Succeeded"),
            "deb7 does not solve on the limited-memory arm; a success with the \
             rung {label} means the rung manufactured one ({status})"
        );
    }
}

/// The other half of the measurement, and the one that decided the default.
/// On `eigena2` and `infeasible_square_scaled_1em4` the rung *costs*
/// iterations to the same verdict, and both of those models are ones the
/// shipped configuration leaves exactly where `main` had them — so switching
/// the rung on would introduce two regressions rather than inherit them.
///
/// This is pinned in the direction it was measured. If it ever goes red, the
/// rung has become cheaper than it was, and the case for leaving it off is
/// the thing to re-examine — not this assertion.
#[test]
fn the_rung_costs_iterations_on_the_two_models_that_kept_it_off() {
    for (fixture, tag) in [
        ("eigena2.nl", "eigena2"),
        ("infeasible_square_scaled_1em4.nl", "infeas_sq"),
    ] {
        let off = solve(fixture, &format!("{tag}_cost_off"), 0);
        let on = solve(fixture, &format!("{tag}_cost_on"), 1);
        assert_eq!(
            off.solution.status, on.solution.status,
            "{fixture}: the rung must not change the verdict either way"
        );
        assert!(
            iters(&on) > iters(&off),
            "{fixture}: the rung was measured as a cost here ({} on against \
             {} off) and the default is off because of it; a reversal means \
             that decision needs re-measuring",
            iters(&on),
            iters(&off)
        );
    }
}

/// `0` is the default and is upstream's unconditional hand-off; `1` has to
/// actually reach the algorithm. Without this the option could be registered
/// and ignored — the gh #677 failure mode, where
/// `limited_memory_initialization` was parsed, validated, and read nowhere.
#[test]
fn the_option_reaches_the_algorithm() {
    let off = solve("eigena2.nl", "eigena2_zero", 0);
    let on = solve("eigena2.nl", "eigena2_one", 1);
    assert_ne!(
        iters(&off),
        iters(&on),
        "limited_memory_ls_failure_restarts=1 took the same {} iterations as \
         the default 0 — the option is parsed but not reaching the algorithm",
        iters(&off)
    );
}
