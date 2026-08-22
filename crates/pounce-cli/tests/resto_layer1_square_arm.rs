//! Layer 1's square-problem arm — `IpRestoConvCheck.cpp:174`.
//!
//! `IpRestoConvCheck::CheckConvergence` decides in two layers. Layer 1
//! asks "can this restoration iterate leave restoration?" and has four
//! arms; layer 2 (pounce#438) only runs when layer 1 answered `CONTINUE`,
//! and renders a terminal verdict once the sub-problem has converged in
//! its own right.
//!
//! One of layer 1's arms was never ported to the live path. On a **square**
//! NLP — as many equality constraints as variables — there is nothing to
//! optimise, so upstream releases any iterate feasible to
//! `min(tol, constr_viol_tol)` immediately:
//!
//! ```text
//! else if( orig_ip_cq->IsSquareProblem()
//!          && orig_trial_inf_pr <= Min(orig_ip_data->tol(), orig_constr_viol_tol_) )
//! {
//!    status = CONVERGED;
//! }
//! ```
//!
//! Plain `CONVERGED` — which `MinC_1NrmRestorationPhase` turns into
//! `resto_status == SUCCESS`, hence `retval = 0`, hence *the outer phase
//! resumes from the recovered point and goes on optimising*.
//!
//! `RestoConvCheck` (the scalar core) has this arm. `RestoConvCheckAdapter`
//! — the implementation actually wired into the restoration inner solve —
//! did not, so a square restoration that reached feasibility fell through
//! to layer 2. Layer 2 has a square arm of its own, but it is the
//! `CONVERGED_TO_ACCEPTABLE_POINT` one at line 222, which
//! `resto_inner_solver`'s square gate converts to `FeasiblePointFound` and
//! which **terminates the solve**. The answer was right and the status was
//! not: the solver stopped at `Feasible Point Found` three iterations in,
//! where reference Ipopt runs the outer phase to `Optimal Solution Found`.
//!
//! The two assertions below are the status and the mechanism. The mechanism
//! one matters independently: layer 2 must log *no verdict at all*, because
//! upstream's arm sits above the `status == CONTINUE` guard and so layer 2
//! never sees such a point. A future change that reached the right status
//! by adding another layer-2 arm would pass the first assertion and fail
//! this one, which is the distinction worth keeping.

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

/// Run a fixture with the layer-2 trace enabled and return `(stdout, stderr)`.
fn run_with_layer2_trace(fixture: &str) -> (String, String) {
    let output = Command::new(pounce_exe())
        .arg(fixture_nl(fixture))
        .arg("--no-sol")
        .env("POUNCE_DBG_RESTO_LAYER2", "1")
        .env("RUST_LOG", "pounce::restoration=debug")
        .env("NO_COLOR", "1")
        .output()
        .expect("spawn pounce");
    (
        String::from_utf8_lossy(&output.stdout).into_owned(),
        String::from_utf8_lossy(&output.stderr).into_owned(),
    )
}

/// `eigmaxa` is square (101 variables, 101 equality constraints) and enters
/// restoration at iteration 3. The restoration drives the original NLP's
/// violation to ~7.5e-15, far under `min(tol, constr_viol_tol) = 1e-8`, so
/// layer 1's square arm applies and the outer phase resumes.
///
/// Reference Ipopt 3.14.19 reaches `Optimal Solution Found` in 21
/// iterations on this model. No iteration count is asserted here — those
/// are the most platform-sensitive numbers in a run — only that the outer
/// phase ran on rather than the solve stopping at the recovered point.
#[test]
fn square_restoration_reaching_feasibility_returns_to_the_outer_phase() {
    let (stdout, stderr) = run_with_layer2_trace("eigmaxa.nl");

    assert!(
        stdout.contains("EXIT: Optimal Solution Found."),
        "a square NLP whose restoration reaches feasibility must hand the \
         point back to the outer phase (`IpRestoConvCheck.cpp:174` -> \
         `resto_status == SUCCESS` -> `retval = 0`), not terminate at it. \
         Terminating reports `Feasible Point Found` with the correct \
         objective, which is why the corpus' status-and-objective \
         assertions cannot see this.\n--- stdout ---\n{stdout}",
    );

    assert!(
        !stderr.contains("[PN_RESTO_LAYER2]"),
        "layer 2 must never be consulted for this point: upstream's square \
         arm sits above the `status == CONTINUE` guard, so a feasible \
         square iterate is released by layer 1. A layer-2 verdict here \
         means the arm is being emulated one layer too low.\n\
         --- stderr ---\n{stderr}",
    );

    // The recovered point is the optimum, not merely a feasible one.
    assert!(
        stdout.contains("-1.0000000000"),
        "eigmaxa's optimal objective is -1; got:\n{stdout}",
    );
}
