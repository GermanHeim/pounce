//! A constant parked in a row's `C<i>` expression segment must not change
//! what the model *is* (issue #492).
//!
//! The `.nl` format lets a writer leave a constant on the left-hand side:
//! `x0 + x1 + 3 <= 6` rather than `x0 + x1 <= 3`. That constant arrives as
//! the row's *nonlinear-part* expression, and POUNCE read every non-empty
//! expression segment as "this row is nonlinear". The reader now folds a
//! constant row body into the row bounds at parse instead. Every fixture
//! here carries a constant the equivalent committed fixture writes into
//! the bound by hand, and must behave exactly as if it had been.
//!
//! Two consequences are pinned, and they bite on different shapes:
//!
//! * **Presolve.** The linear-equality reduction (Phase 6, #490) only
//!   consumes rows tagged linear, so it declined *any* row with a constant
//!   body — including a bare literal. That is
//!   `linear_eq_aggregation_row_constant.nl` below.
//! * **Routing.** The problem classifier lowers each row body to a
//!   degree-≤2 polynomial, which already tolerates a bare literal (it
//!   drops the constant and `qp_extract` re-derives it as a `const_shift`).
//!   What it cannot lower is a constant it has to *compute* — `sqrt(9)`,
//!   an `exp`, a `min` — so a model that is otherwise a plain LP
//!   classified NLP and never reached `pounce-convex`. That is
//!   `lp_row_constant_expr.nl` below.

use std::path::PathBuf;
use std::process::Command;

fn pounce_exe() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_pounce"))
}

struct Run {
    code: Option<i32>,
    stdout: String,
    stderr: String,
    /// The `.sol` value block, in `.nl` order: duals first, then primals.
    /// Empty when the run produced no `.sol`.
    values: Vec<f64>,
}

/// Copy `fixture` into a scratch directory and solve it, AMPL-style.
/// `n_values` is the number of trailing numeric entries to keep — the
/// model's rows plus its columns.
fn run(tag: &str, fixture: &str, n_values: usize, opts: &[&str]) -> Run {
    let dir = std::env::temp_dir().join(format!("pounce_issue_492_{tag}"));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("scratch dir");
    let mut src = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    src.push("tests/fixtures");
    src.push(fixture);
    std::fs::copy(&src, dir.join("m.nl")).expect("copy fixture");

    let mut cmd = Command::new(pounce_exe());
    cmd.current_dir(&dir).arg("m").arg("-AMPL");
    for o in opts {
        cmd.arg(o);
    }
    let out = cmd.output().expect("run pounce");
    let values = match std::fs::read_to_string(dir.join("m.sol")) {
        Ok(sol) => {
            let numeric: Vec<f64> = sol
                .lines()
                .take_while(|l| !l.starts_with("objno"))
                .filter_map(|l| l.trim().parse::<f64>().ok())
                .collect();
            assert!(
                numeric.len() >= n_values,
                "expected at least {n_values} entries in the .sol body, got {numeric:?}"
            );
            numeric[numeric.len() - n_values..].to_vec()
        }
        Err(_) => Vec::new(),
    };
    Run {
        code: out.status.code(),
        stdout: String::from_utf8_lossy(&out.stdout).into_owned(),
        stderr: String::from_utf8_lossy(&out.stderr).into_owned(),
        values,
    }
}

// ---------------------------------------------------------------------
// Routing: an LP with a computed row constant is an LP.
//
// Both LP fixtures are `min −x0 − 2·x1` subject to `x0 + x1 + 3 <= 6` over
// `x0, x1 ∈ [0, 3]`, so the folded row is `x0 + x1 <= 3` and the optimum
// is `x1 = 3`, `x0 = 0`, `obj = −6`. They differ only in how the `3` is
// written: `lp_row_constant.nl` uses the literal `n3`,
// `lp_row_constant_expr.nl` uses `sqrt(9)`.
// ---------------------------------------------------------------------

/// A constant the classifier's polynomial walk cannot lower used to make
/// the whole model NLP. The fold replaces it with the identity zero before
/// the classifier ever sees it.
#[test]
fn a_computed_row_constant_no_longer_forces_an_lp_onto_the_nlp_route() {
    let r = run("auto_expr", "lp_row_constant_expr.nl", 3, &[]);
    assert_eq!(r.code, Some(0), "stderr:\n{}", r.stderr);
    assert!(
        r.stdout.contains("Problem class: LP"),
        "expected the classifier to say LP:\n{}",
        r.stdout
    );
    assert!(
        r.stdout.contains("pounce-convex"),
        "an LP must reach the convex route:\n{}",
        r.stdout
    );
}

/// The same thing said as an error the user would actually hit: a forced
/// solver that does not match the detected class is a hard exit-2 failure
/// naming both, so before the fold this run failed with "NLP … lp-ipm"
/// instead of solving.
#[test]
fn the_forced_lp_solver_accepts_a_model_with_a_computed_row_constant() {
    let r = run(
        "forced_expr",
        "lp_row_constant_expr.nl",
        3,
        &["solver_selection=lp-ipm"],
    );
    assert_eq!(
        r.code,
        Some(0),
        "forcing lp-ipm on an LP must solve, not error\nstdout:\n{}\nstderr:\n{}",
        r.stdout,
        r.stderr
    );
}

/// Rerouting must not solve a different problem. The fold moves the bound
/// with the body, so the answer is the one `x0 + x1 + 3 <= 6` describes —
/// not the `x0 + x1 <= 6` it would describe if the constant were dropped.
///
/// Both spellings of the constant are checked. The literal form already
/// routed correctly before the fold (`qp_extract` re-derived the shift),
/// so it is here as a no-regression pin on the realistic shape rather than
/// as a reproduction.
#[test]
fn the_folded_row_is_the_row_the_model_wrote() {
    for (tag, fixture) in [
        ("optimum_literal", "lp_row_constant.nl"),
        ("optimum_expr", "lp_row_constant_expr.nl"),
    ] {
        let r = run(tag, fixture, 3, &[]);
        assert_eq!(r.code, Some(0), "{fixture} stderr:\n{}", r.stderr);
        // One row then two columns, in `.nl` order.
        let (x0, x1) = (r.values[1], r.values[2]);
        assert!(
            (x0 - 0.0).abs() < 1e-6 && (x1 - 3.0).abs() < 1e-6,
            "{fixture}: expected (x0, x1) = (0, 3), got ({x0}, {x1}); \
             a dropped row constant would give x0 + x1 = 6"
        );
    }
}

// ---------------------------------------------------------------------
// Presolve Phase 6 reaches a row whose constant lived in `C<i>`.
// ---------------------------------------------------------------------

/// `linear_eq_aggregation_row_constant.nl` is the committed
/// `linear_eq_aggregation.nl` with its first row written `x0 − 2·x1 + 3 =
/// 3` instead of `x0 − 2·x1 = 0`. Same feasible set, same optimum — and
/// now the same reduction, where before the pass declined the row outright
/// because `get_constraints_linearity` tagged it NonLinear.
///
/// A bare literal is enough to reproduce this one; unlike the classifier,
/// the reduction's eligibility test has no polynomial walk to fall back on.
#[test]
fn phase_6_reduces_a_row_whose_constant_lived_in_the_expression_segment() {
    let r = run(
        "phase6",
        "linear_eq_aggregation_row_constant.nl",
        5,
        &["presolve=yes", "presolve_linear_eq_reduction=yes"],
    );
    assert_eq!(r.code, Some(0), "stderr:\n{}", r.stderr);
    assert!(
        r.stdout.contains("eliminated 1 columns"),
        "the reduction declined the row-constant equality:\n{}",
        r.stdout
    );
    assert!(
        r.stdout.contains("dropped 1 rows"),
        "the consumed row was not dropped:\n{}",
        r.stdout
    );
    assert!(r.stdout.contains("Optimal Solution Found"), "{}", r.stdout);
}

/// The fold is a change of bookkeeping, not of the problem: solving the
/// row-constant fixture through the newly-reachable reduction must land on
/// the same point and the same duals as a bare solve of the hand-folded
/// fixture. This is the property that keeps a `.sol` readable by AMPL —
/// the multipliers are reported against rows whose residual the fold left
/// alone — and it is what would break if the shift had the wrong sign or
/// hit the wrong bound.
#[test]
fn the_row_constant_fixture_agrees_with_the_hand_folded_one() {
    let folded = run("hand_folded", "linear_eq_aggregation.nl", 5, &[]);
    let offset = run(
        "offset",
        "linear_eq_aggregation_row_constant.nl",
        5,
        &["presolve=yes", "presolve_linear_eq_reduction=yes"],
    );
    assert_eq!(folded.code, Some(0), "stderr:\n{}", folded.stderr);
    assert_eq!(offset.code, Some(0), "stderr:\n{}", offset.stderr);
    assert_eq!(
        folded.values.len(),
        offset.values.len(),
        "the .sol body changed length: {:?} vs {:?}",
        folded.values,
        offset.values
    );
    for (k, (f, o)) in folded.values.iter().zip(offset.values.iter()).enumerate() {
        assert!(
            (f - o).abs() < 1e-6,
            ".sol entry {k} diverged: hand-folded={f} row-constant={o}"
        );
    }
    // Entries 0..2 are the duals. The consumed row's multiplier must come
    // back recovered, not zeroed.
    assert!(
        offset.values[0].abs() > 1e-6,
        "the consumed row came back with a zero multiplier: {:?}",
        offset.values
    );
}
