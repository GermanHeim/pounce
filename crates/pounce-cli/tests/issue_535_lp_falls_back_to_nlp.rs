//! gh #535: an LP the convex interior-point method cannot certify must be
//! re-solved on the general NLP path under `auto`, not reported as the last
//! word.
//!
//! The models that motivate this are NETLIB `gen`/`gen1`. `auto` classifies
//! them LP and routes them to the convex IPM, which exhausts its 200-iteration
//! budget in 190.8 s and exits `Solved_To_Acceptable_Level` with a primal
//! residual of 1.374e-7 against `tol = 1e-8`; `solver_selection=nlp` solves the
//! same model in 19 iterations and 0.982 s to `Solve_Succeeded`, matching
//! Ipopt-3.14.20/MA57's objective to four figures. They are highly degenerate
//! and rank-deficient, strict complementarity fails, and a pure IPM cannot
//! certify the vertex (gh #133); crossover was built to close that and does
//! not. So the router — not the crossover engine — is the lever, and this is
//! the same defect shape `socp_unverified_falls_back_to_nlp.rs` documents for
//! the conic path: a specialized fast path displaced a general one, and when
//! the fast path failed there was no fallback left.
//!
//! `gen.nl` is n=2560 / m=769 / 63085 Jacobian nonzeros and takes three
//! minutes to fail, so it is not a repo fixture. `lp_afiro.nl` under a
//! tolerance no double-precision solver can reach reproduces the *trigger*
//! exactly and deterministically: the convex IPM's KKT error floors at ~5.7e-14
//! and it burns all 199 iterations, which is the `IterationLimit` half of the
//! contract. The gating tests below — a named engine, a user-set budget, a
//! certified solve, a non-LP class — are the half that keeps the fallback from
//! firing anywhere it should not.

use std::path::PathBuf;
use std::process::Command;

/// Well below the ~5.7e-14 KKT error the convex IPM floors at on `afiro`, so
/// the solve provably cannot certify — on any machine, without depending on a
/// fixture that happens to sit on the accuracy knife-edge.
const UNREACHABLE_TOL: &str = "tol=1e-20";

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

fn run_on(name: &str, args: &[&str]) -> (String, String, Option<i32>) {
    let out = Command::new(pounce_exe())
        .arg(fixture(name))
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

fn run(args: &[&str]) -> (String, String, Option<i32>) {
    run_on("lp_afiro.nl", args)
}

/// The convex status line, which a rerouted solve must not print.
fn convex_verdict_lines(stdout: &str) -> Vec<&str> {
    stdout
        .lines()
        .filter(|l| l.contains("IPM, pounce-convex"))
        .collect()
}

/// The headline contract: an LP the convex path could not certify continues on
/// the NLP path instead of ending there.
#[test]
fn an_uncertified_lp_is_re_solved_on_the_nlp_path() {
    let (stdout, stderr, _code) = run(&["solver_selection=auto", UNREACHABLE_TOL]);
    assert!(
        stderr
            .to_lowercase()
            .contains("did not certify a kkt point"),
        "the convex solve must decline; stderr=\n{stderr}"
    );
    assert!(
        stdout.contains("EXIT:"),
        "the NLP path must run and report; stdout=\n{stdout}\nstderr=\n{stderr}"
    );
}

/// The reroute must be announced, and must name where the solve went. A run
/// that silently answers from a different engine than the routing banner named
/// misleads anyone comparing the two.
#[test]
fn the_fallback_says_so() {
    let (_stdout, stderr, _) = run(&["solver_selection=auto", UNREACHABLE_TOL]);
    let lower = stderr.to_lowercase();
    assert!(
        lower.contains("nlp interior-point path"),
        "the reroute must say where the solve went; stderr=\n{stderr}"
    );
    assert!(
        lower.contains("solver_selection=qp-ipm"),
        "the note must say how to see the convex result instead; stderr=\n{stderr}"
    );
}

/// Exactly one verdict. The decision is taken above the convex status line, the
/// `.sol` write and the JSON report, so a rerouted solve leaves no stray convex
/// result for a log scraper (or the benchmark harness) to pick up.
#[test]
fn a_rerouted_solve_reports_one_status_not_two() {
    let (stdout, stderr, _) = run(&["solver_selection=auto", UNREACHABLE_TOL]);
    assert!(
        convex_verdict_lines(&stdout).is_empty(),
        "a rerouted solve must not also print the convex status line; \
         stdout=\n{stdout}\nstderr=\n{stderr}"
    );
}

/// …and nothing else from the discarded attempt either. `afiro` reduces under
/// the convex path's presolve, which used to print its summary before the
/// solve, so a rerouted run left one line of a solve that never reported. The
/// summary is now held back until the convex path is known to be the one that
/// reports — a rerouted run's stdout is the NLP solve's and nothing else.
#[test]
fn a_rerouted_solve_leaves_no_trace_of_the_discarded_attempt() {
    let (rerouted, stderr, _) = run(&["solver_selection=auto", UNREACHABLE_TOL]);
    assert!(
        stderr.to_lowercase().contains("did not certify"),
        "precondition: this run must reroute; stderr=\n{stderr}"
    );
    assert!(
        !rerouted.contains("Presolve:"),
        "the declined convex attempt must not leave its presolve summary \
         behind; stdout=\n{rerouted}"
    );

    // The other half: held back, not lost. A convex solve that *does* report
    // still says what presolve did.
    let (kept, _, _) = run(&["solver_selection=auto"]);
    assert!(
        kept.contains("Presolve:"),
        "a convex solve that reports must still print its presolve summary; \
         stdout=\n{kept}"
    );
}

/// The common case is untouched: an LP the convex IPM certifies is still
/// answered by the convex IPM, with no second solve. This is the 369-of-371
/// case in the LP suite and the reason the fallback triggers on a failure to
/// certify rather than on a class.
#[test]
fn a_certified_lp_still_answers_from_the_convex_path() {
    let (stdout, stderr, code) = run(&["solver_selection=auto"]);
    assert_eq!(code, Some(0), "stdout=\n{stdout}\nstderr=\n{stderr}");
    assert!(
        stdout.contains("Optimal Solution Found"),
        "stdout=\n{stdout}"
    );
    assert_eq!(
        convex_verdict_lines(&stdout).len(),
        1,
        "the convex engine must report its own result; stdout=\n{stdout}"
    );
    assert!(
        !stderr.to_lowercase().contains("did not certify"),
        "a certified solve must not reroute; stderr=\n{stderr}"
    );
}

/// A named engine keeps its verdict. This is what makes the convex stall
/// observable at all, and how the convex result stays available to anyone
/// working on the convex solver itself.
#[test]
fn an_explicitly_selected_convex_solve_is_not_rerouted() {
    for sel in ["solver_selection=qp-ipm", "solver_selection=lp-ipm"] {
        let (stdout, stderr, _code) = run(&[sel, UNREACHABLE_TOL]);
        assert_eq!(
            convex_verdict_lines(&stdout).len(),
            1,
            "{sel} must report the convex engine's own result; stdout=\n{stdout}"
        );
        assert!(
            stdout.contains("Maximum iterations exceeded"),
            "{sel}: the convex verdict must stand; stdout=\n{stdout}"
        );
        assert!(
            !stderr.to_lowercase().contains("did not certify"),
            "{sel} must not be rerouted; stderr=\n{stderr}"
        );
    }
}

/// A user-set iteration budget is the question being asked, so its answer
/// stands. `max_iter=0` is the sharp end of this: the zero-iteration contract
/// (pounce#186) must stop without a solve, and rerouting would launch a full
/// NLP solve for a request that asked for no iterations at all.
#[test]
fn a_user_set_iteration_budget_is_not_a_fallback_trigger() {
    for budget in ["max_iter=0", "max_iter=5"] {
        let (stdout, stderr, _) = run(&["solver_selection=auto", budget]);
        assert!(
            stdout.contains("Maximum iterations exceeded"),
            "{budget} must report the iteration limit; stdout=\n{stdout}"
        );
        assert!(
            !stderr.to_lowercase().contains("did not certify"),
            "{budget} must not trigger the NLP fallback; stderr=\n{stderr}"
        );
    }
}

/// The fallback is scoped to `P = 0`. A convex QP is a different (and
/// unmeasured) population — it keeps the engine chosen for it whatever the
/// tolerance.
#[test]
fn a_convex_qp_is_not_rerouted() {
    let (stdout, stderr, _) = run_on("convex_qp.nl", &["solver_selection=auto", UNREACHABLE_TOL]);
    assert!(
        stdout.contains("Problem class: convex QP"),
        "fixture must classify as a convex QP; stdout=\n{stdout}"
    );
    assert_eq!(
        convex_verdict_lines(&stdout).len(),
        1,
        "a convex QP must report from the convex engine; stdout=\n{stdout}"
    );
    assert!(
        !stderr.to_lowercase().contains("did not certify"),
        "a convex QP must not reroute; stderr=\n{stderr}"
    );
}
