//! pounce#748 — what the default-on μ-strategy retry is allowed to
//! trigger on.
//!
//! #748 turned `mu_strategy_fallback` on by default, to pay for #746's
//! `dirichlet120` casualty. The retry re-runs a solve under the other
//! `mu_strategy` and keeps the second answer only if it reaches
//! `Solve_Succeeded`. Since #138 it had triggered on two statuses:
//! `Maximum_Iterations_Exceeded` and `Solved_To_Acceptable_Level`.
//!
//! Carrying both into the default broke five test targets, all the same
//! way, and the fixture sweep did not see any of it — the sweep runs
//! default options only, and every one of these needs a non-default
//! option to provoke the downgrade the retry then erased:
//!
//! ```text
//!   optimize_hs71::hs071_kkt_fidelity_tol_downgrades_succeeded
//!   masked_certificate_fuzz::the_earliest_refusal_is_the_one_restored_not_the_strictest
//!   issue_616_ls_init_downgrades::the_safeguards_measured_cost_is_now_csfi2_alone
//!   issue_250_dual_guard_never_worse::dual_guard_diversion_does_not_return_a_worse_point
//!   issue_534_resto_decline_progress::a_lost_deferral_costs_a_bounded_number_of_iterations
//! ```
//!
//! `Solved_To_Acceptable_Level` is not a failure — it is a converged
//! answer at the acceptable tolerance — and retrying it by default is
//! wrong three separate ways:
//!
//! 1. It doubles the cost of a solve that already succeeded. "One extra
//!    solve on a run that had already failed" does not describe this.
//! 2. It launders downgrades the caller induced *deliberately* — a tight
//!    `kkt_fidelity_tol`, a certificate veto, `least_square_init_primal`
//!    — so the signal those options exist to produce never arrives.
//! 3. The retry returns the other run's **point**, not just its status,
//!    so it can hand back a different local solution. On
//!    `autocorr_bern55-06` with the dual-divergence guard enabled it
//!    swapped -2304.0000278 for -2320.0000298.
//!
//! So the default-on retry triggers on `Maximum_Iterations_Exceeded`
//! alone, and an explicit `mu_strategy_fallback=yes` keeps the historical
//! pair. `dirichlet120` stalls at max-iterations, so #748's motivating
//! case is recovered either way.
//!
//! The cost of the narrowing, measured: the three fixture-legs the flip
//! had gained go back to where main had them, and they were the only
//! three that had moved. Against main the PR is now a no-op on all 142
//! fixture-legs.
//!
//! ```text
//!   exact csfi2           SolveSucceeded 21  -> SolvedToAcceptableLevel 35
//!   lbfgs eigenb2         SolvedToAcceptableLevel 41 -> SolvedToAcceptableLevel 69
//!   lbfgs pooling_rt2stp  SolveSucceeded 295 -> SolvedToAcceptableLevel 362
//! ```
//!
//! All three remain available to anyone who asks for them by name, which
//! is what the second test below pins.

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

fn solve(model: &str, opts: &[&str]) -> SolveReport {
    // Keep `=` out of the temp-file names: the option strings are
    // `KEY=VALUE`, and a path carrying one does not survive the round
    // trip through the CLI.
    let slug: String = opts
        .join("_")
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .collect();
    let tag = format!("{}_{}_{}", model, std::process::id(), slug);
    let mut json_path = std::env::temp_dir();
    json_path.push(format!("pounce_issue748_{tag}.json"));
    let mut sol_path = std::env::temp_dir();
    sol_path.push(format!("pounce_issue748_{tag}.sol"));

    let mut cmd = Command::new(pounce_exe());
    cmd.arg(fixture(model))
        .arg(&sol_path)
        .arg("--json-output")
        .arg(&json_path);
    for o in opts {
        cmd.arg(o);
    }
    let _ = cmd.status().expect("spawn pounce");
    let text = std::fs::read_to_string(&json_path).expect("read json report");
    let _ = std::fs::remove_file(&json_path);
    let _ = std::fs::remove_file(&sol_path);
    serde_json::from_str(&text).expect("parse SolveReport JSON")
}

/// `csfi2` finishes `Solved_To_Acceptable_Level` in 35 iterations, and the
/// opposite μ strategy reaches `Solve_Succeeded` in 21. That is a genuine
/// improvement and it is exactly what the default must **not** reach for
/// on its own: taking it means every downgraded solve silently pays for a
/// second one, and every deliberate downgrade elsewhere in the suite gets
/// erased. Status only — the iteration count is the platform-sensitive
/// half.
#[test]
fn the_default_retry_leaves_an_acceptable_level_solve_alone() {
    let r = solve("csfi2", &[]);
    assert_eq!(
        format!("{:?}", r.solution.status),
        "SolvedToAcceptableLevel",
        "with no options set, csfi2 must come back downgraded. If this now \
         reads SolveSucceeded the default-on retry has widened back to \
         Solved_To_Acceptable_Level (pounce#748) — see this file's header \
         for the five targets that breaks",
    );
}

/// The historical trigger pair is still there for anyone who names it.
/// The retry keeps the second answer only on `Solve_Succeeded`, so asking
/// for it cannot make the status worse — only the wall clock.
#[test]
fn an_explicit_opt_in_still_retries_and_promotes() {
    let r = solve("csfi2", &["mu_strategy_fallback=yes"]);
    assert_eq!(
        format!("{:?}", r.solution.status),
        "SolveSucceeded",
        "mu_strategy_fallback=yes must still retry a Solved_To_Acceptable_Level \
         solve under the other mu_strategy and promote it (pounce#138, #748)",
    );
}

/// An explicit `mu_strategy` stands the automatic retry down — #748's
/// condition, pinned end to end rather than at the predicate. Retrying
/// under the other schedule recovers a solve that stalled on a strategy
/// POUNCE chose; it is not licence to override one the caller named, and
/// without this every controlled comparison that pins `mu_strategy` —
/// this repository's own benchmark arms included — would silently run
/// both arms.
#[test]
fn naming_a_strategy_stands_the_automatic_retry_down() {
    let r = solve("csfi2", &["mu_strategy=monotone"]);
    assert_eq!(
        format!("{:?}", r.solution.status),
        "SolvedToAcceptableLevel",
        "an explicit mu_strategy must suppress the automatic retry (pounce#748)",
    );
}
