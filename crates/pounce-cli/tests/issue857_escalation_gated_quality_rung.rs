//! The losing direction of `feral_increase_quality` now recovers itself
//! (gh#857, rung 4 of the second-opinion ladder).
//!
//! # The shape of the bug
//!
//! `feral_increase_quality` is on by default and is genuinely two-sided: it
//! buys accuracy and 15–25% of the iterations on several fixture-legs, and it
//! loses whole solves on others. Its own option text has said so since gh#850,
//! and the documented remedy for the losing side was for the user to notice,
//! read the option, and re-run. `square_flowsheet_resto` is the fixture that
//! measures both halves:
//!
//! ```text
//!   exact  escalating   Restoration_Failed/131   (rung 3 rescues it: 54, tot 185)
//!   exact  rung off     SolveSucceeded/99
//!   lbfgs  escalating   MaximumIterations/3000   (nothing rescued it)
//!   lbfgs  rung off     SolveSucceeded/178
//! ```
//!
//! The lbfgs leg is the one with no recovery, and it is not a rare path — the
//! Python frontend and the CasADi plugin both select `limited-memory`
//! automatically when no exact Lagrangian Hessian is available.
//!
//! # Why the gate is a measurement
//!
//! The rung opens on `Restoration_Failed` **or**
//! `Maximum_Iterations_Exceeded`, and only when the failing solve's
//! `quality_escalations` count is at least 1.
//!
//! The second half is what makes the first half affordable. A budget exit
//! normally opens no ladder at all, on the sound reasoning that the answer to
//! running out of iterations is more iterations; naming a trigger for every
//! `Maximum_Iterations_Exceeded` would put an extra solve on every capped run
//! in the corpus. The escalation count is what says which capped runs are
//! candidates — and it is not visible anywhere else, which is why gh#857 had
//! to add it before it could add this. `a_budget_exit_that_never_escalated_…`
//! below is the branch that keeps the cost at zero, and it is the test, not a
//! duplicate of the one above it.
//!
//! It is a gate at `>= 1` and deliberately not a threshold: `deb7` escalates
//! exactly as many times as this fixture's base solve and *gains* by it. The
//! count cannot separate the two; the verdict can.
//!
//! # Appended, not prepended
//!
//! On a `Restoration_Failed` the gh#815 displacement rung runs first and
//! promotes first. `square_flowsheet_resto`'s exact leg is exactly that case,
//! so it reaches the same answer in the same 185 iterations it did before —
//! `the_exact_leg_is_unchanged_because_the_new_rung_is_last` is the pin.

use std::path::PathBuf;
use std::process::Command;

fn pounce_exe() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_pounce"))
}

fn fixture(name: &str) -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("tests");
    p.push("fixtures");
    p.push(name);
    p
}

struct Run {
    out: String,
    /// The reported iteration count, which on a promoted run is the promoted
    /// rung's — the same field the sweep's `it=` column reads.
    iterations: i64,
}

fn solve(model: &str, extra: &[&str]) -> Run {
    let tag: String = format!("{model}_{}", extra.join("_"))
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .collect();
    let dir = std::env::temp_dir();
    let sol = dir.join(format!("pounce_857g_{tag}.sol"));
    let json = dir.join(format!("pounce_857g_{tag}.json"));
    let out = Command::new(pounce_exe())
        .arg(fixture(model))
        .arg(&sol)
        .arg("--json-output")
        .arg(&json)
        .args(extra)
        .output()
        .expect("run pounce");
    let text = std::fs::read_to_string(&json).expect("json report written");
    let key = "\"iteration_count\":";
    let at = text.find(key).expect("iteration_count in report");
    let digits: String = text[at + key.len()..]
        .trim_start()
        .chars()
        .take_while(|c| c.is_ascii_digit() || *c == '-')
        .collect();
    Run {
        out: format!(
            "{}{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        ),
        iterations: digits.parse().expect("numeric iteration_count"),
    }
}

/// Every second-opinion narration line, which is what a user actually sees.
///
/// Not simply every `pounce:` line: the CLI prefixes its ordinary chatter the
/// same way (`pounce: wrote …`), so a bare prefix filter is never empty and
/// the "no ladder ran" assertions below would have failed on a correct
/// build — as this file's first draft did.
fn ladder_lines(out: &str) -> Vec<&str> {
    out.lines()
        .filter(|l| l.starts_with("pounce:"))
        .filter(|l| {
            l.contains("re-solving") || l.contains("re-solve") || l.contains("keeping the original")
        })
        .collect()
}

/// The fix. A capped solve that escalated is re-solved without the
/// escalation, and that re-solve converges.
#[test]
fn an_escalating_budget_exit_recovers_by_undoing_the_escalation() {
    let run = solve(
        "square_flowsheet_resto.nl",
        &["hessian_approximation=limited-memory"],
    );
    assert!(
        run.out.contains("Maximum Number of Iterations Exceeded"),
        "the base solve should still hit the cap — this rung recovers a \
         failure, it does not prevent one:\n{}",
        run.out
    );
    let lines = ladder_lines(&run.out);
    assert!(
        lines.iter().any(|l| l.contains(
            "iteration limit after a factorization escalation — re-solving \
             along 1 different trajectory"
        )),
        "the ladder should open on the budget exit, with exactly one rung — \
         the displacement rung must not join it:\n{lines:#?}"
    );
    assert!(
        lines
            .iter()
            .any(|l| l
                .contains("feral_increase_quality=no re-solve recovered the problem — promoting")),
        "{lines:#?}"
    );
    assert!(
        run.out.contains("EXIT: Optimal Solution Found"),
        "{}",
        run.out
    );
    assert_eq!(
        run.iterations, 178,
        "the promoted rung is the un-escalated solve, which converges in 178 \
         iterations — the same trajectory a user gets by setting \
         feral_increase_quality=no by hand:\n{}",
        run.out
    );
}

/// The rung's own enable, and the evidence that the recovery above is this
/// rung and not something else: turn it off and the pre-gh#857 verdict comes
/// straight back.
#[test]
fn the_rung_can_be_turned_off_and_the_old_verdict_returns() {
    let run = solve(
        "square_flowsheet_resto.nl",
        &[
            "hessian_approximation=limited-memory",
            "feral_increase_quality_retry=no",
        ],
    );
    assert!(
        run.out.contains("Maximum Number of Iterations Exceeded"),
        "{}",
        run.out
    );
    assert!(
        ladder_lines(&run.out).is_empty(),
        "with the rung disabled the ladder has nothing to run, and must not \
         narrate:\n{}",
        run.out
    );
    assert_eq!(run.iterations, 3000);
}

/// **The other branch of the gate**, and the reason `for_status` naming a
/// trigger for every `Maximum_Iterations_Exceeded` costs nothing.
///
/// The fixture is `deb7`, capped at 100 iterations. Three things have to be
/// true at once for this to be evidence and they are all measured, not
/// assumed:
///
/// - it reaches the **NLP arm**, so the ladder code actually runs (42 of the
///   79 CLI fixtures never touch it, and a control on the convex arm would
///   pass while proving nothing — the rule from CLAUDE.md);
/// - it exits `Maximum_Iterations_Exceeded`, so `for_status` names the new
///   trigger;
/// - and it has escalated **zero** times by iteration 100, so rung 4's gate
///   is the only thing standing between it and an extra solve.
///
/// `deb7` is the fixture rather than a never-escalating one because it *does*
/// escalate — once by iteration 110 and twice by 120 — so the escalation path
/// is reachable on this exact model under these exact options, and the count
/// is provably the operative difference rather than an accident of which
/// model was picked. `the_same_count_buys_a_solve_here_and_costs_one_there`
/// in `issue857_quality_escalations_are_reported.rs` measures the other end
/// of the same run.
///
/// The matching *one* branch is on `square_flowsheet_resto`, not here, and
/// that is forced rather than chosen: no cap on `deb7` produces a surviving
/// budget exit that escalated, because the μ-strategy stall retry recovers
/// it to `Optimal` at 108 from `max_iter=110` on — past the first escalation
/// and before the cap can hold. Conversely no cap on `square_flowsheet_resto`
/// produces a budget exit that has *not* escalated: even at `max_iter=5` its
/// internal retry escalates twice. One fixture per branch is the most this
/// corpus allows.
///
/// Without this branch the change would add a second full solve to every
/// capped run in the corpus, and
/// `an_escalating_budget_exit_recovers_by_undoing_the_escalation` would pass
/// just the same.
#[test]
fn a_budget_exit_that_never_escalated_is_left_alone() {
    let run = solve("deb7.nl", &["max_iter=100"]);
    assert!(
        run.out.contains("Maximum Number of Iterations Exceeded"),
        "{}",
        run.out
    );
    assert!(
        run.out.contains("Selected solver: NLP filter"),
        "the control must reach the arm the ladder lives on, or it is \
         evidence about nothing:\n{}",
        run.out
    );
    assert!(
        !run.out
            .contains("Number of linear solver quality escalations"),
        "deb7 has not escalated by iteration 100; if this line appeared the \
         cap has drifted past the first escalation and the test is no longer \
         exercising the zero branch:\n{}",
        run.out
    );
    assert!(
        ladder_lines(&run.out).is_empty(),
        "a budget exit with no escalation has no hypothesis for the ladder \
         to test and must not pay for a solve:\n{}",
        run.out
    );
    assert_eq!(run.iterations, 100);
}

/// The rung is dropped when the baseline already ran with the escalation
/// off, where it would re-run the solve that just failed.
#[test]
fn a_baseline_that_already_declined_the_escalation_gets_no_rung() {
    let run = solve(
        "square_flowsheet_resto.nl",
        &[
            "hessian_approximation=limited-memory",
            "feral_increase_quality=no",
        ],
    );
    // This combination happens to converge, so the ladder never opens at all;
    // the assertion that matters is that the answer is the un-escalated one
    // and nothing was re-solved to get it.
    assert!(
        run.out.contains("EXIT: Optimal Solution Found"),
        "{}",
        run.out
    );
    assert_eq!(run.iterations, 178);
    assert!(ladder_lines(&run.out).is_empty(), "{}", run.out);
}

/// Appended, not prepended — the ordering claim, measured.
///
/// The exact leg fails with `Restoration_Failed` after escalating twice, so
/// both rung 3 and rung 4 are open. Rung 3 runs first and promotes, so the
/// leg reaches the same answer at the same cost it did before gh#857: 131 in
/// the base solve plus 54 in the rung. Rung 4 is announced and never run.
#[test]
fn the_exact_leg_is_unchanged_because_the_new_rung_is_last() {
    let run = solve("square_flowsheet_resto.nl", &[]);
    let lines = ladder_lines(&run.out);
    assert!(
        lines.iter().any(|l| l.contains(
            "restoration failure — re-solving along 2 different trajectories \
             before believing it (second-opinion ladder: \
             start_point_perturbation=1e-2, feral_increase_quality=no)"
        )),
        "both rungs should be open, in this order:\n{lines:#?}"
    );
    assert!(
        lines.iter().any(|l| l
            .contains("start_point_perturbation=1e-2 re-solve recovered the problem — promoting")),
        "{lines:#?}"
    );
    assert!(
        !lines
            .iter()
            .any(|l| l.contains("re-solving with feral_increase_quality=no")),
        "rung 3 promoted, so rung 4 must never have run — an extra solve \
         here is the cost this ordering exists to avoid:\n{lines:#?}"
    );
    assert_eq!(
        run.iterations, 54,
        "the promoted rung's own count, unchanged from before gh#857"
    );
}
