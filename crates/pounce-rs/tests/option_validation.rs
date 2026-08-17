//! Rejected solver options must be reported, not silently dropped (gh#649).
//!
//! `OptionsList` already validates every option against the registry —
//! unknown name, wrong value type, out-of-range or unregistered choice. The
//! builder used to discard that `Result`, so a misspelled option left the
//! default quietly in effect and the solve looked like it had honoured the
//! request. These tests pin the reporting, and pin that valid options still
//! reach the solver.

use pounce_rs::builder::{Nlp, NlpError, Problem};

/// Rosenbrock: unconstrained, converges in well under 100 iterations.
struct Rosenbrock;

impl Problem for Rosenbrock {
    fn objective(&self, x: &[f64]) -> f64 {
        (1.0 - x[0]).powi(2) + 100.0 * (x[1] - x[0] * x[0]).powi(2)
    }
}

fn nlp() -> Nlp<Rosenbrock> {
    Nlp::new(Rosenbrock).x0(&[-1.2, 1.0])
}

#[test]
fn misspelled_string_option_is_reported() {
    // "mu_stratgey" is the transposed spelling of "mu_strategy" from the
    // issue report.
    let err = nlp()
        .option_str("mu_stratgey", "adaptive")
        .try_solve()
        .expect_err("misspelled option should be rejected");

    match err {
        NlpError::InvalidOption { tag, value, reason } => {
            assert_eq!(tag, "mu_stratgey");
            assert_eq!(value, "adaptive");
            assert!(
                reason.contains("Unknown option"),
                "reason should name the failure, got: {reason}"
            );
        }
        other => panic!("expected InvalidOption, got {other:?}"),
    }
}

#[test]
fn misspelled_numeric_option_is_reported() {
    let err = nlp()
        .option_num("tolerence", 1e-8)
        .try_solve()
        .expect_err("misspelled option should be rejected");
    assert!(matches!(err, NlpError::InvalidOption { ref tag, .. } if tag == "tolerence"));
}

#[test]
fn misspelled_integer_option_is_reported() {
    let err = nlp()
        .option_int("max_iterations", 500)
        .try_solve()
        .expect_err("misspelled option should be rejected");
    assert!(matches!(err, NlpError::InvalidOption { ref tag, .. } if tag == "max_iterations"));
}

#[test]
fn wrong_value_type_for_a_real_option_is_reported() {
    // "max_iter" exists but is an integer option, not a string one.
    let err = nlp()
        .option_str("max_iter", "500")
        .try_solve()
        .expect_err("wrong-typed option should be rejected");
    match err {
        NlpError::InvalidOption { tag, reason, .. } => {
            assert_eq!(tag, "max_iter");
            assert!(
                reason.contains("not a string option"),
                "reason should name the type mismatch, got: {reason}"
            );
        }
        other => panic!("expected InvalidOption, got {other:?}"),
    }
}

#[test]
fn unregistered_choice_for_a_real_option_is_reported() {
    // "mu_strategy" is real; "aggressive" is not one of its choices.
    let err = nlp()
        .option_str("mu_strategy", "aggressive")
        .try_solve()
        .expect_err("unregistered choice should be rejected");
    assert!(matches!(err, NlpError::InvalidOption { ref tag, .. } if tag == "mu_strategy"));
}

#[test]
fn out_of_range_numeric_value_is_reported() {
    // "tol" is registered with a strict lower bound of 0.
    let err = nlp()
        .option_num("tol", -1.0)
        .try_solve()
        .expect_err("out-of-range value should be rejected");
    assert!(matches!(err, NlpError::InvalidOption { ref tag, .. } if tag == "tol"));
}

#[test]
fn missing_variable_count_is_reported_not_panicked() {
    let err = Nlp::new(Rosenbrock)
        .try_solve()
        .expect_err("unknown variable count should be an error");
    assert!(matches!(err, NlpError::UnknownVariableCount));
}

#[test]
#[should_panic(expected = "option mu_stratgey=adaptive rejected")]
fn solve_panics_on_a_rejected_option() {
    let _ = nlp().option_str("mu_stratgey", "adaptive").solve();
}

#[test]
fn valid_options_still_reach_the_solver() {
    let sol = nlp()
        .option_num("tol", 1e-10)
        .option_int("print_level", 0)
        .option_str("mu_strategy", "adaptive")
        .try_solve()
        .expect("valid options should be accepted");

    assert!(sol.success, "status: {:?}", sol.status);
    assert!((sol.x[0] - 1.0).abs() < 1e-4);
    assert!((sol.x[1] - 1.0).abs() < 1e-4);
}

/// The regression proper: an accepted option must *do* something. Before the
/// fix a bogus name and a real one were indistinguishable — both ran to
/// convergence. `max_iter=3` has to stop the solve short.
#[test]
fn an_accepted_option_actually_takes_effect() {
    let capped = nlp()
        .option_int("max_iter", 3)
        .option_int("print_level", 0)
        .try_solve()
        .expect("max_iter is a valid option");
    assert!(
        !capped.success,
        "max_iter=3 should not reach optimality on Rosenbrock, got {:?}",
        capped.status
    );
    assert!(capped.stats.iteration_count <= 3);

    let uncapped = nlp()
        .option_int("print_level", 0)
        .try_solve()
        .expect("no options is valid");
    assert!(uncapped.success, "status: {:?}", uncapped.status);
    assert!(
        uncapped.stats.iteration_count > 3,
        "baseline should take more than the capped 3 iterations"
    );
}
