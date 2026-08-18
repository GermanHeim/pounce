//! gh#678 — `alpha_red_factor` reaches the line search.
//!
//! The option was registered (bounded (0,1), default 0.5), accepted
//! without complaint, and never read: `LineSearchOptions` had no such
//! field, `application.rs` never asked the `OptionsList` for it, and
//! `alg_builder.rs` never assigned it. The field on
//! `BacktrackingLineSearch` is genuinely consumed — it scales alpha at
//! every backtracking step — so it just kept its `new()` default of
//! 0.5 forever. Asking the line search to backtrack by 0.2 got you no
//! error, no warning, and a bit-identical trajectory.
//!
//! Per gh#551 the deliverable for this class of bug is a test that
//! proves the option *changes behaviour*, not one that proves the field
//! is assigned — the latter is what the wiring tests in
//! `pounce-algorithm/tests/algorithm_options_wiring.rs` do, and on its
//! own it would have passed just as happily against a value nothing
//! downstream used. So this file drives the real CLI.
//!
//! Two claims, and both matter:
//!
//! 1. **The option is live.** Distinct values must produce distinct
//!    trajectories on a fixture that actually backtracks. `hs71_obj1e8`,
//!    the fixture in the gh#678 report, is *not* such a fixture — it
//!    accepts nearly every trial step at full alpha (13 objective
//!    evaluations for 11 iterations), so it reads identical across the
//!    whole legal range even after the fix. It demonstrated the bug; it
//!    cannot demonstrate the repair. `hs13_bigstart` backtracks hard
//!    (85 objective evaluations for 29 iterations at the default).
//!
//! 2. **The default did not move.** The registered default (0.5) equals
//!    the hard-coded one, so wiring the option must leave an unset run
//!    bit-for-bit where it was. That is the entire safety argument for a
//!    trajectory change, and CLAUDE.md is explicit that it has to be
//!    demonstrated rather than asserted. `scripts/sweep-fixtures.sh`
//!    over all 57 fixtures diffed empty against the parent commit, and
//!    an explicit `alpha_red_factor=0.5` sweep also diffed empty against
//!    the *pre-fix* binary. The same sweep at 0.2 moved 12 fixtures,
//!    which is what says the empty diffs mean "default preserved" rather
//!    than "still unwired". This test pins the fixture-sized version of
//!    both halves so a future retune has to notice.

use std::path::PathBuf;
use std::process::Command;

use pounce_nlp::return_codes::ApplicationReturnStatus;
use pounce_solve_report::SolveReport;

fn pounce_exe() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_pounce"))
}

fn fixture(name: &str) -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("tests");
    p.push("fixtures");
    p.push(format!("{name}.nl"));
    p
}

fn tmp_path(tag: &str, ext: &str) -> PathBuf {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let seq = COUNTER.fetch_add(1, Ordering::Relaxed);
    let mut p = std::env::temp_dir();
    p.push(format!(
        "pounce_gh678_{}_{seq}_{tag}.{ext}",
        std::process::id()
    ));
    p
}

/// Solve `model`, optionally with an explicit `alpha_red_factor`.
/// `None` leaves the option unset so the registry default applies.
fn solve(model: &str, alpha_red_factor: Option<&str>) -> SolveReport {
    let tag = format!("{model}_{}", alpha_red_factor.unwrap_or("unset"));
    let json = tmp_path(&tag, "json");
    // Explicit, so a solved fixture does not drop a `.sol` beside the
    // `.nl` in the source tree.
    let sol = tmp_path(&tag, "sol");
    let mut cmd = Command::new(pounce_exe());
    cmd.arg(fixture(model))
        .arg("--sol-output")
        .arg(&sol)
        .arg("--json-output")
        .arg(&json)
        .arg("--json-detail")
        .arg("summary")
        .arg("print_level=0");
    if let Some(v) = alpha_red_factor {
        cmd.arg(format!("alpha_red_factor={v}"));
    }
    let out = cmd.output().expect("spawn pounce");
    let text = std::fs::read_to_string(&json).unwrap_or_else(|e| {
        panic!(
            "no report for {model} @ alpha_red_factor={:?} (exit {:?}, {e}); \
             stderr:\n{}",
            alpha_red_factor,
            out.status.code(),
            String::from_utf8_lossy(&out.stderr),
        )
    });
    let _ = std::fs::remove_file(&json);
    let _ = std::fs::remove_file(&sol);
    serde_json::from_str(&text).expect("parse SolveReport JSON")
}

/// `(iterations, objective evaluations)`. The evaluation count is the
/// more direct witness of the two: `alpha_red_factor` sets how fast the
/// backtracking loop shrinks alpha, so it changes how many trial points
/// get evaluated per iteration whether or not the iteration count moves.
fn trajectory(r: &SolveReport) -> (usize, usize) {
    (
        r.statistics.iteration_count as usize,
        r.statistics.num_obj_evals as usize,
    )
}

/// Claim 1: the option reaches the line search.
///
/// No absolute counts are asserted — those are the most
/// platform-sensitive numbers in a sweep, and nothing here rests on a
/// particular one. What is asserted is that the trajectories *differ*,
/// which is exactly the property that was false before gh#678 and is
/// unfalsifiable by any amount of grepping for the option name.
#[test]
fn alpha_red_factor_changes_the_trajectory() {
    let slow = solve("hs13_bigstart", Some("0.2"));
    let default = solve("hs13_bigstart", None);
    let fast = solve("hs13_bigstart", Some("0.8"));

    // Measured on x86_64-unknown-linux-gnu at the fixing commit:
    // 0.2 -> (31, 64), unset -> (29, 85), 0.8 -> (26, 39).
    assert_ne!(
        trajectory(&slow),
        trajectory(&default),
        "alpha_red_factor=0.2 produced the default trajectory; the \
         option is not reaching BacktrackingLineSearch (this is the \
         gh#678 regression)",
    );
    assert_ne!(
        trajectory(&fast),
        trajectory(&default),
        "alpha_red_factor=0.8 produced the default trajectory; the \
         option is not reaching BacktrackingLineSearch (this is the \
         gh#678 regression)",
    );
    assert_ne!(
        trajectory(&slow),
        trajectory(&fast),
        "alpha_red_factor=0.2 and =0.8 produced the same trajectory",
    );

    // A knob that only scales the backtracking step must not change
    // which optimum is found. hs13 violates the constraint
    // qualification at its solution, so the tolerance is loose on
    // purpose: the three runs stop at slightly different points on a
    // flat approach, not at different answers.
    for (label, r) in [("0.2", &slow), ("unset", &default), ("0.8", &fast)] {
        assert_eq!(
            r.solution.status,
            ApplicationReturnStatus::SolveSucceeded,
            "hs13_bigstart @ alpha_red_factor={label} did not converge",
        );
        let obj = r.solution.objective;
        assert!(
            (obj - 1.0).abs() < 2e-2,
            "hs13_bigstart @ alpha_red_factor={label} reached {obj}, \
             not the expected optimum near 1.0",
        );
    }
}

/// Claim 2: wiring the option left the default run exactly where it
/// was.
///
/// The registered default and the `BacktrackingLineSearch::new`
/// default are both 0.5, so an unset run and an explicit `=0.5` run
/// must be indistinguishable — same iteration count, same evaluation
/// count, same objective to the last bit. If someone ever changes one
/// of the two defaults without the other, this fails, and the corpus
/// sweep that was clean at merge stops meaning anything.
#[test]
fn unset_is_bit_identical_to_the_registered_default() {
    for model in ["hs13_bigstart", "csfi2", "eigenb2"] {
        let default = solve(model, None);
        let explicit = solve(model, Some("0.5"));
        assert_eq!(
            trajectory(&default),
            trajectory(&explicit),
            "{model}: unset and alpha_red_factor=0.5 took different \
             trajectories, so the registered default no longer matches \
             the hard-coded one",
        );
        assert_eq!(
            default.solution.objective, explicit.solution.objective,
            "{model}: unset and alpha_red_factor=0.5 reached different \
             objectives",
        );
        assert_eq!(default.solution.status, explicit.solution.status);
    }
}
