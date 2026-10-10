//! gh#1000 (I.20) — a square problem's withheld infeasibility certificate is
//! bounded, and the verdict does not depend on where `max_iter` falls.
//!
//! The fixtures are the optimization book's circle–parabola pair (chapter
//! I.20), square: two variables, two equalities.
//!
//! ```text
//!   min (x1-1)² + (x2+2)²   s.t.   x1² + x2² = 1,   x2 - x1² = t
//! ```
//!
//! At `t = 1.5` the parabola's vertex is above the circle, so the model is
//! infeasible; the least ℓ1 violation is 0.5, at `(0, 1)`. At `t = 0.5` it is
//! feasible.
//!
//! Before the fix, `t = 1.5` spent its whole 3000-iteration budget in a
//! period-2 orbit: restoration settled at `(0, 1)` and certified it locally
//! infeasible 124 times, every certificate withheld by the square-problem
//! carve-out, while each accepted main-phase step reset the cycle detectors.
//! The verdict it finally printed came from a stall gate reading the
//! restoration's *absolute* iteration counter, so it depended on where the cap
//! landed: `Infeasible_Problem_Detected` at the default 3000,
//! `Maximum_Iterations_Exceeded` at 500, 1000 or 2000.
//!
//! `t = 0.5` is the control that sets the budget's lower bound: restoration
//! certifies the spurious `(0, -1)` eleven times in a row before the main
//! phase escapes to the solution. See
//! `pounce_restoration::min_c_1nrm::SQUARE_CERTIFICATE_REPEATS`.
//!
//! Every second-opinion rung is switched off, so these assert the primary
//! solve's own verdict rather than whichever re-solve the ladder kept.

use std::path::PathBuf;
use std::process::Command;

fn fixture(name: &str) -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("tests");
    p.push("fixtures");
    p.push(name);
    p
}

/// `(EXIT line, first reported iteration count)` of one solve.
fn solve(model: &str, extra: &[&str]) -> (String, i64) {
    let out = Command::new(env!("CARGO_BIN_EXE_pounce"))
        .arg(fixture(model))
        .args([
            "feral_infeasibility_scaling_retry=no",
            "infeasibility_mu_strategy_retry=no",
            "infeasibility_perturbed_start_retry=no",
            "feral_increase_quality_retry=no",
        ])
        .args(extra)
        .output()
        .expect("spawn pounce");
    let text = String::from_utf8_lossy(&out.stdout);
    let exit = text
        .lines()
        .rfind(|l| l.starts_with("EXIT:"))
        .unwrap_or_else(|| panic!("no EXIT line:\n{text}"))
        .to_string();
    let iters = text
        .lines()
        .find_map(|l| l.strip_prefix("Number of Iterations....:"))
        .unwrap_or_else(|| panic!("no iteration count:\n{text}"))
        .trim()
        .parse()
        .expect("iteration count parses");
    (exit, iters)
}

const INFEASIBLE: &str =
    "EXIT: Converged to a point of local infeasibility. Problem may be infeasible.";
const MAXITER: &str = "EXIT: Maximum Number of Iterations Exceeded.";

/// The repeated certificate is honoured long before the default cap. Measured
/// 594 iterations against 3000 before the fix; the bound leaves room for
/// honest drift without letting the orbit back in.
#[test]
fn an_infeasible_square_model_is_diagnosed_well_inside_the_cap() {
    let (exit, iters) = solve("circle_parabola_t1p5.nl", &[]);
    assert_eq!(exit, INFEASIBLE);
    assert!(
        iters < 1000,
        "the withheld certificate must be honoured, not ground to max_iter: {iters} iterations"
    );
}

/// Once the cap is above the point where the certificate is honoured, moving
/// it changes nothing: same verdict, same iteration count. Below it the solve
/// ends at exactly the cap with `Maximum_Iterations_Exceeded` — not
/// `Restoration_Failed`, and not a verdict a truncated restoration
/// reconstructed from its terminal status.
#[test]
fn the_verdict_does_not_depend_on_where_max_iter_falls() {
    let (_, reference) = solve("circle_parabola_t1p5.nl", &[]);
    for cap in ["1000", "2000", "5000"] {
        let (exit, iters) = solve("circle_parabola_t1p5.nl", &[&format!("max_iter={cap}")]);
        assert_eq!(exit, INFEASIBLE, "max_iter={cap}");
        assert_eq!(iters, reference, "max_iter={cap}");
    }
    let cap = reference / 2;
    let (exit, iters) = solve("circle_parabola_t1p5.nl", &[&format!("max_iter={cap}")]);
    assert_eq!(exit, MAXITER, "max_iter={cap}");
    assert_eq!(iters, cap, "a restoration must not overrun max_iter");
}

/// The feasible control: the budget must not turn a solvable square model
/// into the infeasibility verdict Ipopt (wrongly) gives on it.
#[test]
fn the_feasible_control_still_solves() {
    let (exit, _) = solve("circle_parabola_t0p5.nl", &[]);
    assert_eq!(exit, "EXIT: Optimal Solution Found.");
}
