//! gh#616 — the corpus cost of the gh#605 least-square-init safeguard,
//! pinned so it is visible to the suite instead of living in a PR body.
//!
//! Under `least_square_init_primal=yes` (off by default), gh#605's
//! safeguard is a broad win — ten fixtures reach the same answer in
//! fewer iterations, several dramatically — and it costs two tolerance
//! downgrades: `csfi2` and `eigenb2` finish at
//! `SolvedToAcceptableLevel` where the unsafeguarded step reached
//! `SolveSucceeded`.
//!
//! gh#616 decided that cost is acceptable and that the accept test must
//! **not** be tightened to chase it. The reasoning is in
//! `docs/src/initialization.md` and the algebra is pinned in
//! `pounce-algorithm/tests/issue_616_ls_init_accept_test.rs`. What this
//! file pins is the measurement, because the failure mode CLAUDE.md
//! warns about is a recorded cost that nothing re-reads: gh#544's
//! `pooling_rt2stp` 206 → 812 sat in a commit message through a
//! release. If someone later retunes the accept test and these statuses
//! move, that is a decision being reversed and it should have to be
//! noticed.
//!
//! Every assertion here is on **status and objective**, plus three
//! iteration-count *relations* (two `==`, one `!=`) that each carry a
//! mechanism claim about which arm of the safeguard ran. No absolute
//! iteration count is asserted: those are the most platform-sensitive
//! numbers in a sweep, and none of gh#616's conclusions rests on a
//! particular one.
//!
//! The measured counts, for the record, under
//! `least_square_init_primal=yes` against `=no` on the parent commit
//! `a44f4e8b`: `csfi2` 35/35, `eigena2` 65/26, `eigenb2` 57/67, `deb7`
//! 202/154, `pooling_rt2stp` 81/298, `unbounded_cubic` 290/290.

use std::path::PathBuf;
use std::process::Command;

use pounce_cli::solve_report::SolveReport;

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
        "pounce_gh616_{}_{seq}_{tag}.{ext}",
        std::process::id()
    ));
    p
}

/// Solve `model` and return its report. `ls_init` picks the route:
/// `true` sets `least_square_init_primal=yes`, `false` leaves the
/// default (`no`).
///
/// The exit status is deliberately not asserted: `unbounded_cubic` is
/// meant to end at `DivergingIterates`, which the CLI reports with a
/// nonzero exit, and that verdict is exactly what one of the tests
/// below is checking. The report file being written and parseable is
/// the success condition here.
fn solve(model: &str, ls_init: bool) -> SolveReport {
    let json = tmp_path(model, "json");
    // Explicit, so a fixture that solves does not drop a `.sol` beside
    // the `.nl` in the source tree.
    let sol = tmp_path(model, "sol");
    let mut cmd = Command::new(pounce_exe());
    cmd.arg(fixture(model))
        .arg("--sol-output")
        .arg(&sol)
        .arg("--json-output")
        .arg(&json)
        .arg("--json-detail")
        .arg("summary")
        .arg("print_level=0");
    if ls_init {
        cmd.arg("least_square_init_primal=yes");
    }
    let out = cmd.output().expect("spawn pounce");
    let text = std::fs::read_to_string(&json).unwrap_or_else(|e| {
        panic!(
            "no report for {model} (exit {:?}, {e}); stderr:\n{}",
            out.status.code(),
            String::from_utf8_lossy(&out.stderr),
        )
    });
    let _ = std::fs::remove_file(&json);
    let _ = std::fs::remove_file(&sol);
    serde_json::from_str(&text).expect("parse SolveReport JSON")
}

/// The attribution channel gh#616 added, because taking this
/// measurement without it meant editing `init/default.rs` to print the
/// report and rebuilding the workspace — for every hypothesis. The
/// accessor `IpoptApplication::least_square_init_report` is not
/// reachable from the CLI, and the fixture sweep runs the CLI.
///
/// One line per solve on the `pounce::algorithm` target at `debug`,
/// carrying the same fields as the accessor. Silent at the default log
/// level: this must not become per-iteration noise on a normal run.
#[test]
fn the_safeguard_decision_is_attributable_from_the_cli() {
    let sol = tmp_path("eigenb2_log", "sol");
    let run = |rust_log: Option<&str>| -> String {
        let mut cmd = Command::new(pounce_exe());
        cmd.arg(fixture("eigenb2"))
            .arg("--sol-output")
            .arg(&sol)
            .arg("print_level=0")
            .arg("least_square_init_primal=yes");
        match rust_log {
            Some(v) => cmd.env("RUST_LOG", v),
            None => cmd.env_remove("RUST_LOG"),
        };
        let out = cmd.output().expect("spawn pounce");
        String::from_utf8_lossy(&out.stderr).into_owned()
    };

    let verbose = run(Some("pounce::algorithm=debug"));
    let _ = std::fs::remove_file(&sol);
    assert!(
        verbose.contains("least_square_init_primal safeguard decision"),
        "RUST_LOG=pounce::algorithm=debug did not surface the \
         safeguard's decision; stderr was:\n{verbose}",
    );
    // The decision itself, not just the headline — this is what makes a
    // moving fixture attributable to an arm of the safeguard rather
    // than guessed at.
    for field in [
        "violation_initial",
        "violation_final",
        "alpha",
        "rejected_trials",
        "termination",
    ] {
        assert!(
            verbose.contains(field),
            "attribution line is missing `{field}`:\n{verbose}",
        );
    }
    assert!(
        verbose.contains("accepted"),
        "eigenb2 accepts a backtracked step, so the line should say so:\
         \n{verbose}",
    );

    let quiet = run(None);
    let _ = std::fs::remove_file(&sol);
    assert!(
        !quiet.contains("least_square_init_primal safeguard decision"),
        "the safeguard decision must stay off the default log:\n{quiet}",
    );
}

fn status_of(r: &SolveReport) -> String {
    format!("{:?}", r.solution.status)
}

/// The two downgrades, named. Each is `SolvedToAcceptableLevel` under
/// `least_square_init_primal=yes` — a tolerance-legal answer at the
/// right objective, not a wrong one.
///
/// `csfi2` reaches an objective bit-identical to the one the
/// unsafeguarded step reached (55.0176045); `eigenb2` lands at
/// 1.599999991 against 1.6.
#[test]
fn the_two_accepted_downgrades_are_still_where_gh616_measured_them() {
    let csfi2 = solve("csfi2", true);
    assert_eq!(
        status_of(&csfi2),
        "SolvedToAcceptableLevel",
        "csfi2 under least_square_init_primal=yes; gh#616 accepted this \
         downgrade deliberately — if it moved, say which change moved it",
    );
    assert!(
        (csfi2.solution.objective - 55.0176045).abs() < 1e-5,
        "csfi2 objective drifted: {}",
        csfi2.solution.objective,
    );

    let eigenb2 = solve("eigenb2", true);
    assert_eq!(
        status_of(&eigenb2),
        "SolvedToAcceptableLevel",
        "eigenb2 under least_square_init_primal=yes; see gh#616",
    );
    assert!(
        (eigenb2.solution.objective - 1.6).abs() < 1e-6,
        "eigenb2 objective drifted: {}",
        eigenb2.solution.objective,
    );
}

/// The fact that decides gh#616: `eigena2` and `eigenb2` hand the
/// safeguard **bit-identical** numbers — `theta_0 = 1.0`, accepted
/// `theta = 0.2500000062500001`, `alpha = 0.5`, one rejected trial —
/// and come out differently. `eigena2` improves; `eigenb2` downgrades.
///
/// Measured against the unsafeguarded step, `eigena2` goes 78 → 65
/// iterations at the same `SolveSucceeded` / 82.5, while `eigenb2` goes
/// 55 → 57 and drops a tolerance band.
///
/// No criterion computed from the safeguard's own inputs can separate
/// them, which is why gh#616 did not tighten the accept test. If this
/// assertion ever fails because `eigena2` also downgraded, the argument
/// is unaffected but the cost is bigger; if it fails because `eigenb2`
/// recovered, something changed the trajectory and gh#616's conclusion
/// should be re-derived rather than assumed.
#[test]
fn eigena2_and_eigenb2_disagree_on_identical_safeguard_numbers() {
    let eigena2 = solve("eigena2", true);
    assert_eq!(
        status_of(&eigena2),
        "SolveSucceeded",
        "eigena2 accepts the same alpha = 0.5 step as eigenb2 and \
         converges to full tolerance — the pair is gh#616's evidence \
         that the outcome is not a property of the accept test",
    );
    assert!(
        (eigena2.solution.objective - 82.5).abs() < 1e-6,
        "eigena2 objective drifted: {}",
        eigena2.solution.objective,
    );

    // The pair, side by side, on the route where the safeguard runs.
    // Both accept at alpha = 0.5 off theta_0 = 1.0; one keeps full
    // tolerance and the other does not. No assertion about which is
    // faster — the point is that identical safeguard inputs are
    // compatible with both outcomes, so the accept test is not the
    // thing that decides them.
    let eigenb2 = solve("eigenb2", true);
    assert_ne!(
        status_of(&eigena2),
        status_of(&eigenb2),
        "gh#616 rests on these two disagreeing on identical safeguard \
         numbers; if they now agree, re-derive the conclusion instead \
         of deleting this test",
    );
}

/// `csfi2` is not a case the accept test could ever have rescued: the
/// safeguard **declines** there — all four trials are worse than
/// `theta_0 = 1508.554...` — so `least_square_init_primal=yes` gives
/// back exactly what `=no` gives, to the bit.
///
/// That is the whole reason no tightening reaches `csfi2`: a tighter
/// test still declines, and the only thing that reaches its old
/// `SolveSucceeded` is accepting a step that makes the violation worse,
/// which is what gh#605 exists to prevent.
#[test]
fn csfi2_declines_the_step_so_the_option_changes_nothing_there() {
    let on = solve("csfi2", true);
    let off = solve("csfi2", false);
    assert_eq!(
        status_of(&on),
        status_of(&off),
        "csfi2 declines every trial, so the two routes must agree",
    );
    assert_eq!(
        on.statistics.iteration_count, off.statistics.iteration_count,
        "csfi2 declines every trial, so the two routes must take the \
         same trajectory",
    );
    assert_eq!(
        on.solution.objective, off.solution.objective,
        "csfi2 declines every trial, so the two routes must reach the \
         same objective",
    );
}

/// But "declined" is **not** "never asked", and that surprises people.
///
/// Declining restores the user's `x` exactly. It does not restore the
/// solver's state: computing the direction already drove the first
/// factorization through the augmented-system solver, on the `W = 0`
/// least-square matrix rather than on the first real KKT matrix.
/// gh#616 isolated this by forcing a decline on either side of that
/// call — declining *before* it is bit-identical to
/// `least_square_init_primal=no` everywhere, declining *after* it is
/// bit-identical to the real safeguard.
///
/// `pooling_rt2stp` is where it shows: same objective, same status, and
/// a large iteration difference in the *declined* route's favour. This
/// test pins that the two routes differ, not by how much — the
/// direction is the mechanism claim, the magnitude is a platform
/// detail.
#[test]
fn a_declined_step_is_not_the_same_as_never_asking() {
    let on = solve("pooling_rt2stp", true);
    let off = solve("pooling_rt2stp", false);

    assert_eq!(status_of(&on), "SolveSucceeded");
    assert_eq!(status_of(&off), "SolveSucceeded");
    assert!(
        (on.solution.objective - off.solution.objective).abs() < 1e-3,
        "pooling_rt2stp declines the step, so both routes should land on \
         the same local optimum: {} on, {} off",
        on.solution.objective,
        off.solution.objective,
    );
    assert_ne!(
        on.statistics.iteration_count, off.statistics.iteration_count,
        "pooling_rt2stp declines every trial, yet the trajectories are \
         known to differ — the augmented-system solve that computed the \
         rejected direction leaves the linear solver's state seeded \
         differently. If these ever agree, the carry-over described in \
         docs/src/initialization.md is gone and that section is stale",
    );
}

/// `unbounded_cubic` starts feasible (`theta_0 = 0`), so the safeguard
/// short-circuits before computing any direction: no violation can be
/// improved on zero. The unsafeguarded path took a step anyway and
/// diverged in 91 iterations against 290 here.
///
/// This is the third, independent arm of the safeguard — neither
/// `csfi2`'s decline nor `eigenb2`'s backtrack — and the issue asked
/// specifically whether the three share a mechanism. They do not. The
/// status is `DivergingIterates` either way: the model is unbounded and
/// the verdict is right on both routes.
#[test]
fn a_feasible_start_short_circuits_and_stays_diverging() {
    let on = solve("unbounded_cubic", true);
    let off = solve("unbounded_cubic", false);
    assert_eq!(
        status_of(&on),
        "DivergingIterates",
        "unbounded_cubic is unbounded; the least-square route must not \
         change that verdict",
    );
    assert_eq!(status_of(&off), "DivergingIterates");
    // theta_0 = 0 means the block returns before the augmented-system
    // solve, so unlike the declined case above this really is a no-op.
    assert_eq!(
        on.statistics.iteration_count, off.statistics.iteration_count,
        "a feasible start short-circuits before the augmented-system \
         solve, so the two routes must be identical",
    );
}

/// The headline that keeps the decision honest: turning the option on
/// does not lose a model. The solved-or-acceptable set is the same size
/// on both routes — gh#605 moved two fixtures between the two solved
/// statuses, it did not drop any.
#[test]
fn no_fixture_stops_solving_when_the_option_is_turned_on() {
    for model in [
        "csfi2",
        "eigena2",
        "eigenb2",
        "deb7",
        "pooling_rt2stp",
        "hs71_obj1e8",
        "user_scaling_suffix",
    ] {
        let on = status_of(&solve(model, true));
        let off = status_of(&solve(model, false));
        let solved = |s: &str| s == "SolveSucceeded" || s == "SolvedToAcceptableLevel";
        assert_eq!(
            solved(&on),
            solved(&off),
            "{model}: solved-or-acceptable must not depend on \
             least_square_init_primal (on = {on}, off = {off})",
        );
    }
}
