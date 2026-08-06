//! A `.nl` model whose variable box is **empty** (`x_l > x_u`) reports
//! `primal infeasible` on every convex route, at every crossing width
//! (gh #491).
//!
//! Three things had to change before this held, and each of them was
//! observable from here:
//!
//! 1. `qp_extract` emitted variable bounds as `G` rows and left the solver's
//!    box empty. The empty-box screen reads `lb`/`ub`, so a reversed bound
//!    arrived as a pair of contradictory rows instead — an infeasibility that
//!    has to be *certified* numerically rather than seen.
//! 2. Presolve could not fold that pair back into a box: the leave-one-out
//!    activity for a singleton row's own column computed `−∞ − (−∞) = NaN`,
//!    and its bound-tightening disjointness rule let only the first of the two
//!    rows be a source anyway.
//! 3. With neither of those, the verdict was left entirely to the iteration —
//!    which managed it at most widths but not around `1e-8`, where it returned
//!    `Numerical failure` at a `NaN` iterate, and printed `obj=NaN` on the
//!    summary line.
//!
//! Both widths are checked because the failure was *non-monotone*: `1e0` was
//! reported correctly while `1e-8` was not, so a single-width test would have
//! passed throughout.

use std::path::PathBuf;
use std::process::Command;

fn pounce_exe() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_pounce"))
}

/// Copy a fixture to a scratch dir and solve it, returning
/// `(exit code, stdout)`.
fn run(fixture_name: &str, tag: &str, opts: &[&str]) -> (Option<i32>, String) {
    let dir = std::env::temp_dir().join(format!("pounce_crossed_{tag}"));
    std::fs::create_dir_all(&dir).expect("scratch dir");
    let nl = dir.join(fixture_name);
    let mut fixture = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    fixture.push("tests/fixtures");
    fixture.push(fixture_name);
    std::fs::copy(&fixture, &nl).expect("copy fixture");

    let out = Command::new(pounce_exe())
        .arg(&nl)
        .args(opts)
        .output()
        .expect("run pounce");
    let _ = std::fs::remove_dir_all(&dir);
    (
        out.status.code(),
        String::from_utf8_lossy(&out.stdout).into_owned(),
    )
}

/// The routes a convex-classified model can take. `auto` is what a user gets
/// by default; the two explicit selections are the engines it chooses
/// between, and the whole point is that they agree.
const ROUTES: [&str; 3] = ["auto", "qp-active-set", "qp-ipm"];

/// `min (x−3)²  s.t.  1 ≤ x ≤ 0` — an empty box by a wide margin.
#[test]
fn a_reversed_variable_bound_is_primal_infeasible_on_every_route() {
    for (i, route) in ROUTES.iter().enumerate() {
        let (code, stdout) = run(
            "crossed_bound_qp.nl",
            &format!("wide{i}"),
            &[&format!("solver_selection={route}")],
        );
        assert!(
            stdout.contains("primal infeasible"),
            "route {route}: expected a primal-infeasible verdict, got:\n{stdout}"
        );
        assert!(
            !stdout.contains("NaN"),
            "route {route}: a verdict must not carry NaN:\n{stdout}"
        );
        assert_eq!(code, Some(1), "route {route}: infeasible exits 1");
    }
}

/// The same model with the box crossed by `1e-8` — wider than the tolerance
/// presolve absorbs, narrow enough that the interior-point iteration could
/// neither converge nor certify. This is the case that used to print
/// `Numerical failure (no verified KKT point).  obj=NaN`.
#[test]
fn a_narrowly_reversed_bound_reaches_the_same_verdict() {
    for (i, route) in ROUTES.iter().enumerate() {
        let (code, stdout) = run(
            "narrow_crossed_bound_qp.nl",
            &format!("narrow{i}"),
            &[&format!("solver_selection={route}")],
        );
        assert!(
            stdout.contains("primal infeasible"),
            "route {route}: expected a primal-infeasible verdict, got:\n{stdout}"
        );
        assert!(
            !stdout.contains("NaN"),
            "route {route}: a verdict must not carry NaN:\n{stdout}"
        );
        assert_eq!(code, Some(1), "route {route}: infeasible exits 1");
    }
}

/// The guard against over-reach: an ordinary box still solves, and a
/// *fixed* variable (`x_l == x_u`) is a legitimate model, not an empty box.
#[test]
fn ordinary_and_fixed_boxes_are_untouched() {
    for (i, route) in ROUTES.iter().enumerate() {
        let (code, stdout) = run(
            "boxed_qp_min.nl",
            &format!("ok{i}"),
            &[&format!("solver_selection={route}")],
        );
        assert!(
            stdout.contains("Optimal Solution Found"),
            "route {route}: `0 ≤ x ≤ 1` must still solve:\n{stdout}"
        );
        assert_eq!(code, Some(0), "route {route}");

        // `x = 0.5` exactly: `lb == ub` is the boundary case the screen must
        // *not* claim, and the objective's own minimum is elsewhere, so a
        // dropped bound would show up in the answer.
        let (code, stdout) = run(
            "fixed_var_qp.nl",
            &format!("fixed{i}"),
            &[&format!("solver_selection={route}")],
        );
        assert!(
            stdout.contains("Optimal Solution Found"),
            "route {route}: a fixed variable is feasible:\n{stdout}"
        );
        assert!(
            stdout.contains("obj=6.25"),
            "route {route}: `(0.5 − 3)² = 6.25`, so the bound really held:\n{stdout}"
        );
        assert_eq!(code, Some(0), "route {route}");
    }
}
