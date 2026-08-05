//! A negative `obj_scaling_factor` must not reach the convex solver
//! (gh#483 follow-up).
//!
//! `obj_scaling_factor` is upstream's documented spelling for
//! maximization: the IPM minimizes `factor·f`, so a negative factor
//! maximizes `f`. The convex solvers in `pounce-convex` equilibrate
//! internally and never read the option — and every LP / convex-QP model
//! routes to them by default. So `obj_scaling_factor=-1` on
//! `min (x−3)²  s.t.  x ∈ [0, 1]` returned `x = 1`: the *minimizer* of
//! the objective the user asked to maximize, reported as
//! `Optimal Solution Found` with nothing said about it. Solving the same
//! file with `solver_selection=nlp` gave the right answer, `x = 0`,
//! which is what made it a routing bug rather than a solver bug.
//!
//! The fix follows the routing bargain already used for the #196
//! post-optimal request: under `auto` decline the fast path and use the
//! general NLP interior-point solver, which honors the option. Under an
//! explicit convex `solver_selection` the choice is *refused*, not
//! warned about — for #196 the fast path merely skipped extra work, but
//! here it would answer the wrong question.
//!
//! A **positive** factor is a different case and deliberately untouched:
//! it only rescales conditioning, and the convex path already reports
//! natural units, so both paths agree. Pinned below so the guard cannot
//! quietly widen into "any non-default factor reroutes".

use std::path::PathBuf;
use std::process::Command;

fn pounce_exe() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_pounce"))
}

/// `min (x − 3)²  s.t.  x ∈ [0, 1]` (plus a trivial `x ≥ 0` row so the
/// file has a constraint). Classifies as a convex QP, so `auto` routes
/// it to `pounce-convex` by default. Minimizer `x = 1`; maximizer over
/// the same box `x = 0`.
fn run(tag: &str, opts: &[&str]) -> (Option<i32>, String, String, Option<f64>) {
    let dir = std::env::temp_dir().join(format!("pounce_objsense_{tag}"));
    std::fs::create_dir_all(&dir).expect("scratch dir");
    let nl = dir.join("boxed_qp_min.nl");
    let mut fixture = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    fixture.push("tests/fixtures/boxed_qp_min.nl");
    std::fs::copy(&fixture, &nl).expect("copy fixture");

    let out = Command::new(pounce_exe())
        .arg(&nl)
        .args(opts)
        .arg("print_level=0")
        .output()
        .expect("run pounce");
    // `.sol` layout: message, blank, "Options", 4 header ints, then the
    // m dual values and the n primal values. m = n = 1 here, so the
    // solution is the last numeric line.
    let x = std::fs::read_to_string(nl.with_extension("sol"))
        .ok()
        .and_then(|s| {
            s.lines()
                .filter_map(|l| l.trim().parse::<f64>().ok())
                .next_back()
        });
    let _ = std::fs::remove_dir_all(&dir);
    (
        out.status.code(),
        String::from_utf8_lossy(&out.stdout).into_owned(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
        x,
    )
}

/// Baseline: no option, convex route, the minimizer.
#[test]
fn default_route_minimizes() {
    let (code, out, err, x) = run("baseline", &[]);
    assert_eq!(code, Some(0), "stdout:\n{out}\nstderr:\n{err}");
    assert!(out.contains("convex QP interior-point"), "stdout:\n{out}");
    let x = x.expect("no solution in .sol");
    assert!(
        (x - 1.0).abs() < 1e-6,
        "expected the minimizer x=1, got {x}"
    );
}

/// `obj_scaling_factor=-1` under `auto`: reroute to the NLP path and
/// return the *maximizer*. Pre-fix this stayed on the convex route and
/// returned `x = 1`.
#[test]
fn negative_obj_scaling_reroutes_and_maximizes() {
    let (code, out, err, x) = run("maximize", &["obj_scaling_factor=-1"]);
    assert_eq!(code, Some(0), "stdout:\n{out}\nstderr:\n{err}");
    assert!(
        out.contains("NLP filter line-search"),
        "should have declined the convex fast path; stdout:\n{out}",
    );
    assert!(
        err.contains("cannot express"),
        "the reroute must say why; stderr:\n{err}",
    );
    let x = x.expect("no solution in .sol");
    assert!(
        x.abs() < 1e-6,
        "maximizing (x-3)^2 over [0,1] gives x=0; got {x} \
         (x=1 means the objective sense was dropped again)",
    );
}

/// Same answer when the NLP path is asked for directly — the reroute
/// must agree with the engine it reroutes to, not merely differ from
/// the convex one.
#[test]
fn forced_nlp_path_agrees_with_the_reroute() {
    let (code, _out, err, x) = run(
        "maximize_nlp",
        &["obj_scaling_factor=-1", "solver_selection=nlp"],
    );
    assert_eq!(code, Some(0), "stderr:\n{err}");
    assert!(x.expect("no solution in .sol").abs() < 1e-6);
}

/// A forced convex solver plus a maximize request is refused: there is
/// no honest outcome, so exiting beats reporting the wrong optimum.
#[test]
fn negative_obj_scaling_with_a_forced_convex_solver_is_refused() {
    let (code, out, err, _x) = run(
        "forced",
        &["obj_scaling_factor=-1", "solver_selection=qp-ipm"],
    );
    assert_eq!(code, Some(2), "stdout:\n{out}\nstderr:\n{err}");
    assert!(
        err.contains("minimizer of the objective you asked to maximize"),
        "the refusal must name the consequence; stderr:\n{err}",
    );
}

/// A positive factor only rescales conditioning; the convex path
/// reports natural units either way, so it keeps the fast path and the
/// same answer.
#[test]
fn positive_obj_scaling_keeps_the_convex_route() {
    let (code, out, err, x) = run("positive", &["obj_scaling_factor=100"]);
    assert_eq!(code, Some(0), "stdout:\n{out}\nstderr:\n{err}");
    assert!(
        out.contains("convex QP interior-point"),
        "a positive factor must not reroute; stdout:\n{out}",
    );
    let x = x.expect("no solution in .sol");
    assert!((x - 1.0).abs() < 1e-6, "expected x=1, got {x}");
}
