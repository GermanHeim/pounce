//! A model whose only nonlinearity cancelled must not be **routed** as an
//! LP (gh #685, part 2).
//!
//! Part 1 (`issue_685_cancelled_quadratic_evaluation`) closed the
//! evaluation half: a row whose quadratic coefficients cancelled in the
//! recognizer's own floating-point arithmetic is no longer handed to Q4's
//! constant-structure evaluator in place of its tape. This is the routing
//! half, and it is the larger of the two — it is a *classification*
//! decision, and `dispatch`'s LP fast path is what turns a wrong
//! classification into a wrong answer rather than a slow one.
//!
//! The shape: `2⁵³·x₀² + x₀² − 2⁵³·x₀²` folds to an empty quadratic map, so
//! `classify_inner`'s "purely linear after all" arm took the row, no row
//! was quadratic, the objective was linear, and the model classified **LP**
//! — the reproduction printed
//!
//! ```text
//! pounce: problem class LP — every nonlinear part expanded to a
//!         linear (or constant) polynomial
//! Objective: -1.0000000000000000e+06
//! ```
//!
//! `qp_extract` then folded the row's (empty) linear part into `G` and the
//! constraint left the model altogether: `min −x₀` with nothing holding it
//! walks to its `10⁶` bound and reports `Optimal`.
//!
//! The fix is the same gate as part 1's and in the same place it belongs —
//! `NlBody::analyze_quadratic_full` refuses a form that dropped a term, so
//! every consumer that *reads coefficients out* (the classifier, both
//! extractors) refuses with it. The classifier then names the finding
//! rather than reporting it as "not a degree-2 polynomial", which is what
//! the two new `ClassReason` variants are for.
//!
//! The refusal was at first conservative in a way worth knowing about: it
//! keyed off "a term was dropped", not off "the arithmetic that dropped
//! it was lossy", so an exact `x − x` lost the LP path along with the
//! catastrophic cases. gh #687 sharpened the flag to the inexact *fold*,
//! so exact cancellations keep their fast path and only the lossy ones —
//! the shape asserted below — route NLP.
//!
//! What is asserted here is the routing, not a numerical optimum. A body
//! whose coefficients cancel in the recognizer's arithmetic cancels in the
//! tape's too, so the NLP route does not compute the mathematical
//! `x₀² + x₁²` either; it computes what the row means to this solver, which
//! is an ill-conditioned thing to hand a line search. The defect gh #685
//! reports is the six-order-of-magnitude gap and the confident `Optimal`
//! beside it, and that is what goes away.

use std::path::PathBuf;
use std::process::Command;

use pounce_cli::dispatch::{ClassReason, ProblemClass, classify_problem_explained};
use pounce_cli::nl_reader::{NlProblem, parse_nl_text_with_quadratic};

// ---------------------------------------------------------------------
// Models
// ---------------------------------------------------------------------

/// `2⁵³·x₀² + x₀² − 2⁵³·x₀²`, folded front to back: `2⁵³`, then `2⁵³ + 1`
/// rounds back to `2⁵³`, then `2⁵³ − 2⁵³` is exactly `0`. The body is
/// `x₀²`; the stored form is nothing at all.
///
/// The middle term is spelled `x₀^2` where the outer two are `x₀·x₀` so
/// that the tape does not hash-cons all three onto one node — same
/// construction, and same reason, as gh #683's reproducer.
fn cancelling_body() -> String {
    let big = (1u64 << 53) as f64;
    format!(
        "o54\n3\n\
         o2\nn{big:.1}\no2\nv0\nv0\n\
         o5\nv0\nn2\n\
         o2\nn{neg:.1}\no2\nv0\nv0\n",
        neg = -big,
        big = big,
    )
}

/// `(10⁻²⁰⁰·x₀)·(10⁻²⁰⁰·x₀)`: one monomial, degree 2, whose coefficient
/// `10⁻⁴⁰⁰` is not representable and flushes to zero on the multiply.
fn underflowing_body() -> &'static str {
    "o2\no2\nn1e-200\nv0\no2\nn1e-200\nv0\n"
}

/// An honest `x₀²`, to check the gate did not simply close the QP route.
fn ordinary_body() -> &'static str {
    "o5\nv0\nn2\n"
}

/// The reported reproduction: `min −x₀` subject to `body ≤ 5`, with
/// `0 ≤ x₀ ≤ 10⁶`. One variable, one row, and *nothing else nonlinear* —
/// which is the point. Every other model in this pair of test files keeps a
/// `sin` around to hold the classifier on the NLP route; this one has none,
/// so the class is decided entirely by what the recognizer made of row 0.
fn con_model(body: &str) -> String {
    format!(
        "g3 0 1 0\n\
         1 1 1 0 0\n\
         1 0\n\
         0 0\n\
         1 0 0\n\
         0 0 0 1\n\
         0 0 0 0 0\n\
         1 1\n\
         0 0\n\
         0 0 0 0 0\n\
         C0\n{body}\
         O0 0\n\
         n0\n\
         x1\n\
         0 1.0\n\
         r\n\
         1 5.0\n\
         b\n\
         0 0.0 1000000.0\n\
         k0\n\
         J0 1\n\
         0 0\n\
         G0 1\n\
         0 -1.0\n"
    )
}

/// The same defect on the objective side: `min body` subject to a linear
/// row `x₀ ≥ 1`, `0 ≤ x₀ ≤ 10⁶`. A cancelled objective classified LP by the
/// same route — an empty Hessian read as "it was effectively linear".
fn obj_model(body: &str) -> String {
    format!(
        "g3 0 1 0\n\
         1 1 1 0 0\n\
         0 1\n\
         0 0\n\
         0 1 0\n\
         0 0 0 1\n\
         0 0 0 0 0\n\
         1 1\n\
         0 0\n\
         0 0 0 0 0\n\
         C0\n\
         n0\n\
         O0 0\n{body}\
         x1\n\
         0 1.0\n\
         r\n\
         2 1.0\n\
         b\n\
         0 0.0 1000000.0\n\
         k0\n\
         J0 1\n\
         0 1.0\n\
         G0 1\n\
         0 0\n"
    )
}

/// Parse with parse-time quadratic recognition on and off. The classifier
/// reaches its form from a `Quad` body one way and re-derives it from a
/// `Tree` the other, and the gate has to be on both arms or it is on
/// neither.
fn both_paths(txt: &str) -> [(&'static str, NlProblem); 2] {
    [
        (
            "recognizing",
            parse_nl_text_with_quadratic(txt, true).expect("parse (recognizing)"),
        ),
        (
            "trees",
            parse_nl_text_with_quadratic(txt, false).expect("parse (trees)"),
        ),
    ]
}

// ---------------------------------------------------------------------
// The routing decision
// ---------------------------------------------------------------------

/// The regression: a row that lost a term is not called linear, so the
/// model does not route LP.
///
/// Before the fix all four of these classified `Lp` with
/// `NonlinearPartsCancelled`.
#[test]
fn a_row_that_dropped_a_term_does_not_route_lp() {
    for (what, txt) in [
        ("cancellation", con_model(&cancelling_body())),
        ("underflow", con_model(underflowing_body())),
    ] {
        for (path, prob) in both_paths(&txt) {
            let (class, reason) = classify_problem_explained(&prob);
            assert_eq!(
                class,
                ProblemClass::Nlp,
                "{what} ({path}): a model whose only row lost its quadratic \
                 term was routed to the convex path",
            );
            assert_eq!(
                reason,
                ClassReason::ConstraintTermsDropped { row: 0 },
                "{what} ({path}): routed NLP, but for the wrong stated reason",
            );
        }
    }
}

/// The same on the objective side.
#[test]
fn an_objective_that_dropped_a_term_does_not_route_lp() {
    for (what, txt) in [
        ("cancellation", obj_model(&cancelling_body())),
        ("underflow", obj_model(underflowing_body())),
    ] {
        for (path, prob) in both_paths(&txt) {
            let (class, reason) = classify_problem_explained(&prob);
            assert_eq!(
                class,
                ProblemClass::Nlp,
                "{what} ({path}): a model whose objective lost its quadratic \
                 term was routed to the convex path",
            );
            assert_eq!(
                reason,
                ClassReason::ObjectiveTermsDropped,
                "{what} ({path}): routed NLP, but for the wrong stated reason",
            );
        }
    }
}

/// The gate is on the drop, not on the shape: an honest quadratic still
/// reaches the convex path, on both the constraint and the objective side.
/// Without this the "fix" is just a switch that turns the QP route off.
#[test]
fn an_ordinary_quadratic_still_routes_convex() {
    for (path, prob) in both_paths(&con_model(ordinary_body())) {
        assert_eq!(
            classify_problem_explained(&prob).0,
            ProblemClass::ConvexQcqp,
            "{path}: an honest `x₀² ≤ 5` lost the conic route",
        );
    }
    for (path, prob) in both_paths(&obj_model(ordinary_body())) {
        assert_eq!(
            classify_problem_explained(&prob).0,
            ProblemClass::ConvexQp,
            "{path}: an honest `min x₀²` lost the convex-QP route",
        );
    }
}

// ---------------------------------------------------------------------
// The reported symptom
// ---------------------------------------------------------------------

/// End to end, through the CLI: the reproduction no longer reports the
/// bound.
///
/// The two things asserted are the two things gh #685 shows: the routing
/// log says NLP where it said LP, and the reported objective is nowhere
/// near `x₀`'s `10⁶` ceiling — the row constrains the model again instead
/// of having vanished out of the extracted LP. The optimum itself is not
/// asserted, for the reason in the module header: on either route this row
/// is what floating point makes of it, not `x₀²`.
#[test]
fn the_cli_no_longer_reports_the_bound() {
    let dir = std::env::temp_dir().join("pounce_issue_685_routing");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("scratch dir");
    let path = dir.join("cancelling.nl");
    std::fs::write(&path, con_model(&cancelling_body())).expect("write model");

    // A cancelling row is ill-conditioned on the tape as well, so cap the
    // iterations rather than wait out a thrash; the routing decision this
    // is watching is made before iteration 1.
    let out = Command::new(PathBuf::from(env!("CARGO_BIN_EXE_pounce")))
        .arg(&path)
        .arg("max_iter=200")
        .env("POUNCE_DBG_CLASSIFY", "1")
        .output()
        .expect("run pounce");
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr),
    );

    let class = text
        .lines()
        .find(|l| l.contains("pounce: problem class"))
        .unwrap_or_else(|| panic!("no classification log in:\n{text}"));
    assert!(
        class.contains("problem class NLP"),
        "the reproduction still routes to the convex path: {class}",
    );

    let obj = objective(&text).unwrap_or_else(|| panic!("no objective in:\n{text}"));
    assert!(
        obj > -1.0e3,
        "the reproduction still walks x₀ to its 10⁶ bound: objective {obj}",
    );
}

/// The unscaled objective from the end-of-run summary block.
fn objective(text: &str) -> Option<f64> {
    let line = text
        .lines()
        .find(|l| l.trim_start().starts_with("Objective."))?;
    line.split_whitespace()
        .filter_map(|t| t.parse::<f64>().ok())
        .next_back()
}
