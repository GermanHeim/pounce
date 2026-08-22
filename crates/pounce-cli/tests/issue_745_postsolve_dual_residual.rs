//! Convex postsolve must not manufacture a KKT residual (gh #745).
//!
//! The LP arm exited `Solve_Succeeded` on netlib `problem` while reporting
//! `Dual infeasibility 1.25e-3` and `Complementarity 1.17e-4` — three to five
//! orders above the `1e-8` the termination test had just certified. The
//! filed diagnosis blamed the termination test. It is not: the *reduced*
//! solve converges to `dual = 9.4e-12`. Both numbers are created by
//! `Presolve::postsolve` on the way back out to the original problem space,
//! and the residuals the summary block prints are recomputed there.
//!
//! Two independent defects, both pinned here:
//!
//! 1. Postsolve **re-derives** each variable's bound multipliers instead of
//!    carrying the reduced solve's through — it has to, since most restored
//!    columns were eliminated and never had one — using a hard `at_bound`
//!    classification with a `1e-6` window. A variable `1.55e-6` off a zero
//!    lower bound with `z_lb = 1.25e-3` fell outside the window, lost the
//!    multiplier, and reported its whole reduced cost as dual infeasibility.
//!    Their product is `1.9e-9`, i.e. exactly the barrier complementarity, so
//!    the pair was a perfectly good certificate.
//!
//! 2. An eliminated column's primal is *computed* — back-substituted out of
//!    its consumed row, or evaluated as `x = α·y + β` — so it can land a few
//!    ulps outside its own box. Column 0 sat `1.17e-11` below a lower bound
//!    whose multiplier is `1e7`; that product is the `1.17e-4` of
//!    complementarity.
//!
//! `problem` is 46 columns and 12 rows and needs both a free-column singleton
//! and an aggregation to reproduce, which is why it is checked in whole
//! rather than reduced to a synthetic fixture.

use std::path::PathBuf;
use std::process::Command;

/// The four residual lines of the summary block, unscaled column.
struct Residuals {
    objective: f64,
    dual: f64,
    primal: f64,
    complementarity: f64,
}

fn solve(fixture: &str, tag: &str) -> Residuals {
    let dir = std::env::temp_dir().join(format!("pounce_issue_745_{tag}"));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("scratch dir");
    let mut src = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    src.push("tests/fixtures");
    src.push(fixture);
    std::fs::copy(&src, dir.join("m.nl")).expect("copy fixture");

    let out = Command::new(PathBuf::from(env!("CARGO_BIN_EXE_pounce")))
        .current_dir(&dir)
        .arg("m.nl")
        .arg("--no-sol")
        .output()
        .expect("run pounce");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("pounce-convex"),
        "{fixture} no longer routes to the convex solver:\n{stdout}"
    );
    assert!(
        stdout.contains("Optimal Solution Found"),
        "{fixture} did not solve:\n{stdout}"
    );

    // The summary prints `(scaled)  (unscaled)`; take the unscaled column,
    // which is the last field on each line.
    let field = |prefix: &str| -> f64 {
        let line = stdout
            .lines()
            .find(|l| l.starts_with(prefix))
            .unwrap_or_else(|| panic!("no `{prefix}` line in:\n{stdout}"));
        line.split_whitespace()
            .next_back()
            .expect("value")
            .parse()
            .expect("parse residual")
    };
    Residuals {
        objective: field("Objective..."),
        dual: field("Dual infeasibility"),
        primal: field("Constraint violation"),
        complementarity: field("Complementarity"),
    }
}

/// Ipopt/MA57 on the same model. The pre-fix answer, `-1.6001171170`, is
/// wrong in the *fourth* decimal — a converged solve reported as such.
const IPOPT_OBJ: f64 = -1.5996991454255909;

/// What the solver claims when it says `Solve_Succeeded`: `tol` is `1e-8`,
/// and every residual in the summary must actually be at that scale. `1e-6`
/// leaves room for the postsolve arithmetic without leaving room for either
/// defect — the smaller of the two was `1.17e-4`.
const CERTIFIED: f64 = 1e-6;

#[test]
fn postsolve_does_not_invent_a_dual_residual() {
    let r = solve("issue745_netlib_problem.nl", "problem");
    assert!(
        r.dual < CERTIFIED,
        "postsolve dropped a bound multiplier: dual infeasibility {:.6e} on a \
         solve reported Optimal (gh #745 saw 1.2486e-3)",
        r.dual
    );
    assert!(
        r.complementarity < CERTIFIED,
        "postsolve left a primal outside its box while pinned to it: \
         complementarity {:.6e} on a solve reported Optimal (gh #745 saw \
         1.1718e-4)",
        r.complementarity
    );
    assert!(
        r.primal < CERTIFIED,
        "constraint violation {:.6e}",
        r.primal
    );
    // The wrong duals came with a wrong point. Ipopt's own answer is the
    // check that the repair moved *towards* the optimum, not merely towards
    // a smaller residual.
    let rel = (r.objective - IPOPT_OBJ).abs() / IPOPT_OBJ.abs();
    assert!(
        rel < 1e-3,
        "objective {:.10e} vs Ipopt/MA57 {IPOPT_OBJ:.10e} (rel {rel:.2e})",
        r.objective
    );
}
