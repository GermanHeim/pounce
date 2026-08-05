//! `derivative_test` produced no test and no output (gh#483 follow-up).
//!
//! All five `derivative_test*` options were registered and none was ever
//! read, so `derivative_test=first-order` ran nothing and printed
//! nothing. That is the worst shape an unimplemented option can take: a
//! *checker* that silently checks nothing reports success by omission —
//! a user with a hand-written `eval_grad_f` turns it on, sees no
//! complaints, and concludes the gradient is right.
//!
//! The checker's own correctness (does it catch a wrong gradient, a
//! wrong Jacobian entry, a missing sparsity entry?) is pinned by unit
//! tests in `pounce-nlp/src/derivative_test.rs`, against TNLPs with
//! deliberately corrupted derivatives. A `.nl` model cannot express
//! that — its derivatives come from pounce's own AD — so what is tested
//! here is the wiring: that the option reaches the engine, on **both**
//! solver routes, and that it stays off by default.

use std::path::PathBuf;
use std::process::Command;

fn pounce_exe() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_pounce"))
}

/// Run `fixture_name` with `opts`; returns `(success, stderr)`. The
/// report goes to stderr so it survives `print_level=0` and leaves
/// `--json-output`'s stdout clean.
fn run(fixture_name: &str, tag: &str, opts: &[&str]) -> (bool, String) {
    let dir = std::env::temp_dir().join(format!("pounce_derivtest_{tag}"));
    std::fs::create_dir_all(&dir).expect("scratch dir");
    let nl = dir.join(fixture_name);
    let mut fixture = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    fixture.push("tests/fixtures");
    fixture.push(fixture_name);
    std::fs::copy(&fixture, &nl).expect("copy fixture");
    let out = Command::new(pounce_exe())
        .arg(&nl)
        .args(opts)
        .arg("print_level=0")
        .output()
        .expect("run pounce");
    let _ = std::fs::remove_dir_all(&dir);
    (
        out.status.success(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
    )
}

/// Off by default — the option's registered default is `none` and a
/// plain run must be unchanged.
#[test]
fn no_report_by_default() {
    let (ok, err) = run("user_scaling_suffix.nl", "default", &[]);
    assert!(ok, "stderr:\n{err}");
    assert!(
        !err.contains("Derivative checker"),
        "an unrequested run must print nothing; stderr:\n{err}",
    );
}

/// `first-order` reports on the NLP interior-point route. Pre-fix this
/// printed nothing at all.
#[test]
fn first_order_reports_on_the_nlp_route() {
    let (ok, err) = run(
        "user_scaling_suffix.nl",
        "first",
        &["solver_selection=nlp", "derivative_test=first-order"],
    );
    assert!(
        ok,
        "the check is advisory and must not fail the solve; {err}"
    );
    assert!(
        err.contains("Derivative checker: first derivatives"),
        "stderr:\n{err}",
    );
    assert!(
        err.contains("No suspicious derivatives found"),
        "pounce's own AD derivatives must come back clean; stderr:\n{err}",
    );
}

/// The convex dispatch never reaches `optimize_tnlp`, so it needs — and
/// has — its own call. A model classifying as a convex QP must still be
/// checked; the test is about the model, not the engine.
#[test]
fn first_order_reports_on_the_convex_route() {
    let (ok, err) = run(
        "boxed_qp_min.nl",
        "convex",
        &["derivative_test=first-order"],
    );
    assert!(ok, "stderr:\n{err}");
    assert!(
        err.contains("Derivative checker"),
        "the convex route must run the check too; stderr:\n{err}",
    );
}

/// It fires exactly once — the two call sites are mutually exclusive,
/// not stacked.
#[test]
fn the_report_is_not_emitted_twice() {
    for (tag, fixture, sel) in [
        ("once_nlp", "user_scaling_suffix.nl", "solver_selection=nlp"),
        ("once_qp", "boxed_qp_min.nl", "solver_selection=auto"),
    ] {
        let (_, err) = run(fixture, tag, &[sel, "derivative_test=first-order"]);
        assert_eq!(
            err.matches("Derivative checker").count(),
            1,
            "{tag}: stderr:\n{err}",
        );
    }
}

/// `second-order` covers the Hessian as well, and `derivative_test_print_all`
/// turns the summary into a full listing.
#[test]
fn second_order_and_print_all_reach_the_engine() {
    let (ok, err) = run(
        "user_scaling_suffix.nl",
        "second",
        &[
            "solver_selection=nlp",
            "derivative_test=second-order",
            "derivative_test_print_all=yes",
        ],
    );
    assert!(ok, "stderr:\n{err}");
    assert!(
        err.contains("first and second derivatives"),
        "stderr:\n{err}",
    );
    assert!(
        err.contains("grad_f["),
        "print_all must list entries; {err}"
    );
    assert!(err.contains("h_obj["), "…including the Hessian; {err}");
}

/// `only-second-order` skips the first-order pass, so no `grad_f`
/// comparisons appear even under `print_all`.
#[test]
fn only_second_order_skips_the_first_order_pass() {
    let (ok, err) = run(
        "user_scaling_suffix.nl",
        "onlysecond",
        &[
            "solver_selection=nlp",
            "derivative_test=only-second-order",
            "derivative_test_print_all=yes",
        ],
    );
    assert!(ok, "stderr:\n{err}");
    assert!(err.contains("h_obj["), "stderr:\n{err}");
    assert!(
        !err.contains("grad_f["),
        "the first-order pass must be skipped; stderr:\n{err}",
    );
}
