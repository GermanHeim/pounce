//! A convex QCQP whose conic solve returns no verified KKT point must be
//! re-solved on the general NLP path, not reported as a failure.
//!
//! Fixture `airport.nl` (n=84, m=42, from Vanderbei's CUTE-in-AMPL set)
//! classifies as a convex QCQP, so `solver_selection=auto` routes it to the
//! conic (SOCP) interior-point solver. That solver *stalls*: it stops at 31
//! iterations — identically at `max_iter=200` and `max_iter=1000`, so this is
//! a lack of progress and not a budget — with a complementarity of ~9.5e-4,
//! about five orders above tolerance. Dual infeasibility (2.4e-10), constraint
//! violation (1.1e-8) and bound violation (0) are all converged, and the
//! objective agrees with Ipopt-MA57's to nine significant figures, so the
//! point is essentially the optimum. The post-solve verification is right to
//! refuse to certify it — but refusing was previously *terminal*, and the
//! problem was reported `Restoration_Failed` (`solve_result_num` 500) even
//! though the NLP filter line-search solves it in 15 iterations, matching
//! Ipopt exactly.
//!
//! That is the same defect shape as gh #413: a specialized fast path displaced
//! a general one, and when the fast path failed there was no fallback left.
//! The dispatcher already routes *large* convex QCQPs to the NLP path before
//! solving, on the reasoning (`dispatch.rs`) that "a convex QCQP is still a
//! valid NLP, so the fallback is sound" — this applies the same reasoning
//! after the fact.
//!
//! The `socp` arm below is the other half of the contract, and the reason the
//! fallback is gated on `auto`: when the user names the engine, its verdict
//! must stand. Silently answering from a different solver would hide exactly
//! the stall this test documents.

use std::path::PathBuf;
use std::process::Command;

fn pounce_exe() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_pounce"))
}

fn fixture() -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("tests");
    p.push("fixtures");
    p.push("airport.nl");
    p
}

fn run(args: &[&str]) -> (String, String, Option<i32>) {
    let out = Command::new(pounce_exe())
        .arg(fixture())
        .arg("--no-sol")
        .args(args)
        .output()
        .expect("spawn pounce");
    (
        String::from_utf8_lossy(&out.stdout).into_owned(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
        out.status.code(),
    )
}

/// The headline contract: under `auto` the stalled conic solve is not the
/// final word. Previously `Restoration_Failed`, exit 1.
#[test]
fn unverified_conic_solve_is_re_solved_on_the_nlp_path() {
    let (stdout, stderr, code) = run(&["solver_selection=auto"]);
    assert_eq!(
        code,
        Some(0),
        "must exit 0; stdout=\n{stdout}\nstderr=\n{stderr}"
    );
    let lower = stdout.to_lowercase();
    assert!(
        lower.contains("optimal solution found"),
        "must report an optimum after falling back; stdout=\n{stdout}"
    );
    assert!(
        !lower.contains("numerical failure") && !lower.contains("restoration"),
        "the conic failure must not be the reported verdict; stdout=\n{stdout}"
    );
}

/// The reroute must be announced. A solve that silently answers from a
/// different engine than the banner named is worse than a slow one — anyone
/// comparing engines would be reading the wrong solver's numbers.
#[test]
fn the_fallback_says_so() {
    let (_stdout, stderr, _) = run(&["solver_selection=auto"]);
    let lower = stderr.to_lowercase();
    assert!(
        lower.contains("no verified kkt point") && lower.contains("nlp"),
        "the reroute must be explained on stderr; stderr=\n{stderr}"
    );
}

/// Exactly one verdict. The fallback decision is taken before the conic path
/// prints its status line, writes the `.sol` or emits the JSON report, so a
/// rerouted solve must not leave a stray conic result behind for a log scraper
/// (or the benchmark harness) to pick up.
#[test]
fn a_rerouted_solve_reports_one_status_not_two() {
    let (stdout, _stderr, _) = run(&["solver_selection=auto"]);
    let conic_lines = stdout
        .lines()
        .filter(|l| l.contains("conic IPM, pounce-convex"))
        .count();
    assert_eq!(
        conic_lines, 0,
        "a rerouted solve must not also print the conic status line; stdout=\n{stdout}"
    );
}

/// An explicitly requested engine keeps its verdict: no reroute, no note. This
/// is what makes the stall observable, and it is how the conic result stays
/// available to anyone debugging the conic solver itself.
#[test]
fn an_explicitly_selected_conic_solve_is_not_rerouted() {
    let (stdout, stderr, _code) = run(&["solver_selection=socp"]);
    assert!(
        stdout.contains("conic IPM, pounce-convex"),
        "solver_selection=socp must report the conic engine's own result; stdout=\n{stdout}"
    );
    assert!(
        stdout.to_lowercase().contains("numerical failure"),
        "the conic verdict must stand when the engine was named; stdout=\n{stdout}"
    );
    assert!(
        !stderr.to_lowercase().contains("re-solved"),
        "an explicit selection must not be rerouted; stderr=\n{stderr}"
    );
}

/// `max_iter=0` must still stop without a solve. The zero-iteration contract
/// (pounce#186) returns `IterationLimit`, which is deliberately *not* a
/// fallback trigger — rerouting it would run a full NLP solve for a request
/// that asked for no iterations at all.
#[test]
fn zero_iteration_contract_is_not_a_fallback_trigger() {
    let (stdout, stderr, _) = run(&["solver_selection=auto", "max_iter=0"]);
    assert!(
        stdout.contains("Maximum iterations exceeded"),
        "max_iter=0 must report the iteration limit; stdout=\n{stdout}"
    );
    assert!(
        !stderr.to_lowercase().contains("re-solved"),
        "max_iter=0 must not trigger the NLP fallback; stderr=\n{stderr}"
    );
}
