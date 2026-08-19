//! gh#616 — the corpus cost of the gh#605 least-square-init safeguard,
//! pinned so it is visible to the suite instead of living in a PR body.
//!
//! Under `least_square_init_primal=yes` (off by default), gh#605's
//! safeguard is a broad win — ten fixtures reach the same answer in
//! fewer iterations, several dramatically — and it cost two tolerance
//! downgrades when gh#616 measured it: `csfi2` and `eigenb2` finished
//! at `SolvedToAcceptableLevel` where the unsafeguarded step reached
//! `SolveSucceeded`.
//!
//! **For one release, one of those two was gone, and nobody went looking
//! for it** (gh#681). gh#588's Q4 evaluates a recognized degree-≤2 row
//! from its constant matrix instead of rebuilding an AD tape every
//! iteration, which reassociates the sums in `eval_g` and `eval_jac_g` —
//! `quad_evaluator_differential.rs` declares those two comparisons
//! non-bitwise in advance, for exactly this reason. `eigenb2` sat close
//! enough to the accept band for the reassociation to carry it across,
//! reaching `SolveSucceeded` in 54 iterations where the tape stalled at
//! `SolvedToAcceptableLevel` in 57. `csfi2` does not move at all, to the
//! bit — the safeguard declines there, so no evaluator change can reach
//! it.
//!
//! gh#702 then compensated that same sum, because the uncompensated
//! association was costing a 1500-variable QCQP 28 extra iterations, and
//! `eigenb2` came back to `SolvedToAcceptableLevel`. So the count is two
//! again, as gh#616 measured it. The lesson is not that either verdict is
//! right: it is that on these two models the status is **downstream
//! round-off**, and the tests below are written to make that visible
//! rather than to defend a particular one.
//!
//! That is a trajectory change and it is pinned as one: the tests below
//! assert both legs, so the fast path's verdict and the tape's are each
//! held in place and a future move is attributable to one of them.
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

use std::collections::BTreeMap;
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
    solve_with_env(model, ls_init, &[])
}

/// `solve`, plus environment for the child process. The only caller
/// that needs it sets `POUNCE_DBG_NO_QUAD=1` to force the AD tape,
/// which is how a fixture that moved under gh#588's Q4 is made to say
/// so instead of being asserted around.
fn solve_with_env(model: &str, ls_init: bool, env: &[(&str, &str)]) -> SolveReport {
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
    for (k, v) in env {
        cmd.env(k, v);
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

/// Strip ANSI SGR escapes. `tracing_subscriber`'s fmt layer colours
/// its field names whether or not stderr is a terminal, so a captured
/// attribution line holds `\x1b[3mviolation_initial\x1b[0m\x1b[2m=\x1b[0m1.0`
/// where the eye reads `violation_initial=1.0`.
fn strip_ansi(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut it = s.chars().peekable();
    while let Some(c) = it.next() {
        if c != '\u{1b}' {
            out.push(c);
            continue;
        }
        if it.peek() == Some(&'[') {
            it.next();
            // CSI runs to the first byte in @..=~.
            for c2 in it.by_ref() {
                if ('\u{40}'..='\u{7e}').contains(&c2) {
                    break;
                }
            }
        }
    }
    out
}

/// The safeguard's attribution line for `model` under
/// `least_square_init_primal=yes`, parsed to field -> value.
///
/// This is the channel gh#616 added, used for the thing it was added
/// for: comparing two models' safeguard decisions without editing
/// `init/default.rs` and rebuilding.
fn safeguard_decision(model: &str) -> BTreeMap<String, String> {
    let sol = tmp_path(model, "sol");
    let out = Command::new(pounce_exe())
        .arg(fixture(model))
        .arg("--sol-output")
        .arg(&sol)
        .arg("print_level=0")
        .arg("least_square_init_primal=yes")
        .env("RUST_LOG", "pounce::algorithm=debug")
        .output()
        .expect("spawn pounce");
    let _ = std::fs::remove_file(&sol);
    let stderr = strip_ansi(&String::from_utf8_lossy(&out.stderr));
    let tail = stderr
        .lines()
        .find_map(|l| l.split_once("safeguard decision"))
        .unwrap_or_else(|| panic!("no safeguard attribution line for {model}; stderr:\n{stderr}"))
        .1;
    // Values are `key=value`, and a value containing spaces is quoted:
    // a declining model reports `termination="no improvement"`, which a
    // plain whitespace split silently truncates to `"no`.
    let mut out = BTreeMap::new();
    let mut pending: Option<(String, String)> = None;
    for tok in tail.split_whitespace() {
        match pending.take() {
            Some((k, mut v)) => {
                v.push(' ');
                v.push_str(tok);
                if tok.ends_with('"') {
                    out.insert(k, v.trim_matches('"').to_string());
                } else {
                    pending = Some((k, v));
                }
            }
            None => {
                let Some((k, v)) = tok.split_once('=') else {
                    continue;
                };
                if v.starts_with('"') && !(v.len() > 1 && v.ends_with('"')) {
                    pending = Some((k.to_string(), v.to_string()));
                } else {
                    out.insert(k.to_string(), v.trim_matches('"').to_string());
                }
            }
        }
    }
    if let Some((k, v)) = pending {
        out.insert(k, v.trim_matches('"').to_string());
    }
    out
}

/// What the safeguard costs, measured on the route users actually get.
///
/// gh#616 recorded two downgrades. On this branch there is one.
///
/// `csfi2` still reaches 55.0176045 at `SolvedToAcceptableLevel`, bit
/// for bit what `=no` reaches, because the safeguard declines every
/// trial there — a decline is not a step, so there is no trajectory for
/// an evaluator change to perturb.
///
/// `eigenb2` downgrades too, which is what gh#616 measured in the first
/// place. It spent one release not doing so: gh#588's Q4 reassociated
/// `eval_g` and nudged this model — which sat close enough to the accept
/// band to be nudged — across it. That was never a better safeguard, and
/// the version of this test written against it said so.
///
/// gh#702 compensated the same sum and put `eigenb2` back where gh#616
/// found it. Three associations of one dot product, three answers:
///
/// | route                       | status                    | iters | objective          |
/// |-----------------------------|---------------------------|-------|--------------------|
/// | tape (`POUNCE_DBG_NO_QUAD`) | `SolvedToAcceptableLevel` |    57 | 1.59999999134715   |
/// | Q4, uncompensated           | `SolveSucceeded`          |    54 | 1.599999999992518  |
/// | Q4, compensated (gh#702)    | `SolvedToAcceptableLevel` |    48 | 1.599999996403372  |
///
/// Two of the three agree with gh#616, and the odd one out is the one
/// nobody designed. Do not read the recovery as progress if it comes
/// back — check which association produced it.
///
/// The `POUNCE_DBG_NO_QUAD=1` leg no longer discriminates, since both
/// evaluator routes now downgrade. It stays as a cross-check: the two
/// routes agreeing on this model is the normal state, and Q4's window
/// was the exception.
#[test]
fn the_safeguards_measured_cost_is_csfi2_and_eigenb2() {
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
        "gh#616 measured this downgrade and gh#702 restored it. \
         `SolveSucceeded` here means something reassociated `eval_g` \
         again and landed on the lucky side of the accept band — that \
         is what gh#588's Q4 did, and it is not a fix",
    );
    assert!(
        (eigenb2.solution.objective - 1.6).abs() < 1e-6,
        "eigenb2 objective drifted: {}",
        eigenb2.solution.objective,
    );

    let tape = solve_with_env("eigenb2", true, &[("POUNCE_DBG_NO_QUAD", "1")]);
    assert_eq!(
        status_of(&tape),
        "SolvedToAcceptableLevel",
        "with the fast path off, eigenb2 must still reproduce gh#616's \
         measurement. Since gh#702 the fast path reproduces it too, so \
         the two routes agree here; a disagreement means one of them \
         moved and the diff says which",
    );
    assert!(
        (tape.solution.objective - 1.6).abs() < 1e-6,
        "eigenb2 objective drifted on the tape route: {}",
        tape.solution.objective,
    );
}

/// The fact that decides gh#616: `eigena2` and `eigenb2` hand the
/// safeguard **the same numbers**, and it takes the same decision on
/// both — `theta_0 = 1.0`, one rejected trial, accepted at
/// `alpha = 0.5` on a step of norm 3.2596011939729705.
///
/// gh#616 read that fact off a status *disagreement*: identical
/// decisions, different outcomes, therefore no criterion computed from
/// the safeguard's own inputs separates them, therefore tightening the
/// accept test could not rescue `eigenb2` without also reaching
/// `eigena2`, which needed no rescuing. That disagreement has since
/// opened and closed twice, so this test asserts the premise
/// **directly**, off the attribution channel, instead of inferring it
/// from the outcomes.
///
/// The conclusion now rests on three independent reassociations of one
/// dot product rather than one. Each moves the `violation_final` the
/// safeguard reports by a few ulps, each moves **both models by the
/// same amount**, and nothing else the accept test reads changes at all:
///
/// | route                    | `violation_final`    | `eigena2` | `eigenb2` |
/// |--------------------------|----------------------|-----------|-----------|
/// | tape                     | 0.2500000062500001   | succeeded | acceptable|
/// | Q4, uncompensated        | 0.2500000062500003   | succeeded | succeeded |
/// | Q4, compensated (gh#702) | 0.25000000624999996  | acceptable| acceptable|
///
/// Three associations, three verdicts on the pair — agreeing, agreeing,
/// and disagreeing — off inputs that stay bit-for-bit identical between
/// the two models every time. So neither model's status was ever a
/// property of the accept test: both are decided downstream, by where
/// the iteration after the safeguard lands relative to the acceptable
/// band, and one reassociated sum in `eval_g` is enough to move either.
/// An accept test tightened to chase `eigenb2` would have been tuned
/// against round-off — gh#616's conclusion, now re-derived twice.
///
/// The compensated row is the one that costs something: `eigena2` takes
/// 127 iterations there against 51 on the uncompensated fast path and
/// 65 on the tape, for an objective still correct to 82.50000000000348.
/// That is tracked as its own defect (gh#706) and it is a cost, not a
/// wash. The question there is not how to get 51 back — that number was
/// luck — but why a last-ulp change moves this model 76 iterations.
///
/// The absolute values pinned below are the ones the accept test reads
/// (`violation_initial`, `alpha`, `rejected_trials`, `termination`).
/// `violation_final` and `step_norm` are compared **between** the two
/// models only, never against a literal: cross-model equality is the
/// durable fact, and the last two digits of a reported violation are
/// the kind of number Q4 has already been shown to move.
#[test]
fn eigena2_and_eigenb2_hand_the_safeguard_identical_numbers() {
    let a = safeguard_decision("eigena2");
    let b = safeguard_decision("eigenb2");
    assert!(!a.is_empty(), "eigena2 attribution parsed to nothing");

    for field in [
        "violation_initial",
        "violation_final",
        "alpha",
        "step_norm",
        "rejected_trials",
        "termination",
    ] {
        assert!(
            a.contains_key(field),
            "eigena2 attribution has no `{field}`: {a:?}",
        );
        assert_eq!(
            a.get(field),
            b.get(field),
            "gh#616 rests on eigena2 and eigenb2 handing the safeguard \
             identical numbers, and `{field}` now differs. Re-derive the \
             conclusion in docs/src/initialization.md instead of editing \
             this list",
        );
    }

    // The inputs the accept test actually reads, as literals.
    let at = |k: &str| a.get(k).map(String::as_str);
    assert_eq!(at("violation_initial"), Some("1.0"));
    assert_eq!(at("alpha"), Some("0.5"));
    assert_eq!(at("rejected_trials"), Some("1"));
    assert_eq!(at("termination"), Some("accepted"));

    // Same decision, and — since gh#702 — the same verdict again, at
    // the other status. Pinned so that the pair moving is a finding
    // rather than a surprise; the premise above is what gh#616 rests
    // on, and these two are downstream round-off.
    assert_eq!(
        status_of(&solve("eigena2", true)),
        "SolvedToAcceptableLevel",
        "eigena2 accepts the alpha = 0.5 step; where it lands relative \
         to the acceptable band is decided by round-off downstream, and \
         gh#702's compensated sum puts it inside rather than through",
    );
    assert_eq!(
        status_of(&solve("eigenb2", true)),
        "SolvedToAcceptableLevel",
        "see the test above — this is gh#616's originally measured \
         downgrade, restored by gh#702",
    );
    assert!(
        (solve("eigena2", true).solution.objective - 82.5).abs() < 1e-6,
        "eigena2 objective drifted",
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
