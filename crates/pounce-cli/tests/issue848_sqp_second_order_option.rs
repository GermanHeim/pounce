//! gh #848 — `sqp_qp_certify_second_order` is a real switch, end to end.
//!
//! The second-order check that stops the active-set engine certifying a
//! saddle point lives in `pounce-qp`. It is on for every standalone QP solve
//! and **off** for the SQP's step subproblem, because those two callers ask
//! the engine different questions — see `QpOptions::sqp_subproblem`, and
//! gh #856 for what turning it on here needs first.
//!
//! Off-by-default makes the switch matter more, not less. It is a
//! *trajectory* change on every route that hands the engine an indefinite
//! Hessian, so it needs a user-reachable switch — and a switch nothing reads
//! is worse than none, because its documentation describes behaviour that does
//! not exist. That is gh #677 (`limited_memory_initialization` was registered
//! and never read) and the `sqp_qp_use_homotopy` no-op found while writing the
//! warm-start benchmark. Both were invisible to
//! `convex_option_readers_match_the_registry`, which pins that a value the
//! registry accepts is never rejected by a reader — a different claim from
//! "setting it changes the answer".
//!
//! So this file asserts that setting the option changes what the solver does.
//! It used to assert that specifically as a *different objective*, and the
//! reasoning for preferring objectives to iteration counts is below and still
//! sound. gh #873 took that away: repairing gh #856's escape at the KKT point
//! means the arm now reaches `−6752.25` on `nonconvex_two_escapes.nl` with the
//! option off as well as on, and over both legs of the corpus — 180
//! fixture-legs — **no** fixture separates the two settings by objective,
//! while exactly three separate them by iteration count, all on the exact
//! leg. Two independent guards catching the
//! same models is the right outcome; the cost is this file's discriminator,
//! and `the_switch_is_still_read_end_to_end` records what replaced it and how
//! much weaker the replacement is. `pounce-qp`'s own
//! `issue848_second_order_certification::the_check_can_be_switched_off_and_then_the_saddle_comes_back`
//! pins the engine-level behaviour; what is only reachable from here is the
//! plumbing between the CLI option registry and that reader.
//!
//! ## Which fixture can carry which claim
//!
//! `nonconvex_qp.nl` is `min x₀·x₁ s.t. x₀ + x₁ = 2, 0 ≤ x ≤ 4` (gh #797). On
//! the feasible segment `f(x₀) = x₀(2 − x₀)` is concave, so the interior
//! stationary point `(1, 1)` is the constrained **maximum** at `obj = 1` and
//! the minimum `obj = 0` sits at either endpoint. With the check on, the arm
//! reaches the minimum — that claim is portable and is
//! `turning_it_on_stops_the_sqp_arm_certifying_the_constrained_maximum`.
//!
//! What is **not** portable is the *default's* answer on that fixture, and the
//! first version of this file asserted it. All three of `(1, 1)`, `(0, 2)` and
//! `(2, 0)` satisfy first-order KKT, so which one an active-set method returns
//! is decided by the working-set path and not by the specification — and the
//! model is exactly symmetric under swapping `x₀` and `x₁`, so that path turns
//! on a tie. Both architectures break it, and they break it differently:
//! macOS/arm64 returns the maximum `1` (20 of 20 runs, debug and release);
//! ubuntu-latest/x86-64 returns `0`, which is how the assertion failed in
//! CI run 33287729052. `nonconvex_two_escapes.py`'s own docstring names the
//! mechanism from the other side — "exactly the symmetry that makes
//! `nonconvex_qp.nl` converge onto its constrained maximum". A tie is not a
//! defect manifestation you can pin; the record of what the default costs
//! lives in prose here and on gh #856, not in an assertion that is true on
//! half the machines that run it.
//!
//! `nonconvex_two_escapes.nl` (gh #805) carries the switch-is-real claim
//! instead, because its default answer is reached by exact cancellation
//! rather than by a tie. The model is even in `x₁`, so `∂f/∂x₁` vanishes
//! identically on `x₁ = 0`, and `∂f/∂x₀` vanishes identically at `x₀ = 0` —
//! both in exact arithmetic and in floating point, where the products are
//! zero bit-for-bit. From the start `(0, 1)` the arm is driven onto `A =
//! (0, 0)`, `f = 0`, in one iteration and stops there: a genuine first-order
//! point, and the pre-#797 answer. With the check on it reaches the global
//! minimum at the corner, `f = −6752.25`. That is a gap of nearly four
//! orders of magnitude, so the test asserts the *strict improvement* rather
//! than either endpoint's exact value — a switch that stops being read fails
//! it, and a tie broken the other way on some future target does not.

use std::path::PathBuf;
use std::process::Command;

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

/// `(objective, iteration count)`. The iteration count is what
/// `the_switch_is_still_read_end_to_end` needs; see its doc comment for why
/// the objective alone can no longer carry that claim.
fn solve(args: &[&str]) -> (f64, u32) {
    let out = Command::new(pounce_exe())
        .args(args)
        .output()
        .expect("spawn pounce");
    let combined = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    assert_eq!(out.status.code(), Some(0), "must solve:\n{combined}");
    let iters = combined
        .lines()
        .find(|l| l.trim_start().starts_with("Number of Iterations"))
        .and_then(|l| l.split_whitespace().last())
        .and_then(|t| t.parse().ok())
        .unwrap_or_else(|| panic!("no iteration count in:\n{combined}"));
    (objective(args), iters)
}

fn objective(args: &[&str]) -> f64 {
    let out = Command::new(pounce_exe())
        .args(args)
        .output()
        .expect("spawn pounce");
    let combined = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    assert_eq!(out.status.code(), Some(0), "must solve:\n{combined}");
    let line = combined
        .lines()
        .find(|l| l.starts_with("Objective"))
        .unwrap_or_else(|| panic!("no objective line in:\n{combined}"));
    line.split_whitespace()
        .nth(1)
        .and_then(|t| t.parse().ok())
        .unwrap_or_else(|| panic!("unparseable objective line {line:?}"))
}

#[test]
fn turning_it_on_stops_the_sqp_arm_certifying_the_constrained_maximum() {
    let f = fixture("nonconvex_qp.nl");
    let f = f.to_str().unwrap();
    let on = objective(&[
        f,
        "--no-sol",
        "algorithm=active-set-sqp",
        "sqp_qp_certify_second_order=yes",
    ]);
    assert!(
        on.abs() < 1e-6,
        "with the check on the SQP must reach the minimum 0, not the interior \
         stationary point 1; got {on}"
    );
}

/// The option is plumbed from the CLI registry through to the engine, and
/// setting it changes what the solver does — gh #677's lesson, asserted
/// rather than assumed.
///
/// # Why this no longer asserts an objective
///
/// It used to, and the objective was the right thing to assert: on
/// `nonconvex_two_escapes.nl` the default stopped at the ridge point `A`
/// (`f = 0`) and turning the check on reached the global `−6752.25`, a gap of
/// nearly four orders of magnitude that no tie-break could counterfeit.
///
/// gh #873 removed that gap, by fixing the *other* guard. gh #856's
/// second-order escape at the KKT point had been finding the negative
/// curvature on that fixture and then discarding it; with
/// `exhibit_better_point` repaired, the arm reaches `−6752.25` on that model
/// whether this option is on or off. Two independent guards now catch it,
/// which is the outcome to want — but it costs this test its discriminator,
/// and pretending otherwise by loosening the old assertion until it passed
/// would have left gh #677's shape undetectable here.
///
/// Measured at that commit with `scripts/sweep-fixtures.sh …
/// sqp_qp_certify_second_order=no|yes`, over both legs — 180 fixture-legs:
/// **zero** differ in objective, and exactly **three** differ in iteration
/// count — `nonconvex_qp` 3 → 1, `nonconvex_two_escapes` 5 → 4,
/// `nonconvex_qcqp` 6 → 8. All three are on the **exact** leg; the
/// limited-memory leg is unmoved everywhere, which is why the assertions
/// below do not pass `hessian_approximation`. So the option is still read,
/// and what it still moves is the trajectory.
///
/// # What is asserted instead, and why it is not brittle
///
/// That *at least one* of those three fixtures responds to the option, in
/// objective or in iteration count. An option that is registered and never
/// read — gh #677 exactly — makes all three identical in both, and fails
/// this. A single platform breaking a single fixture's count the same way on
/// both settings does not, because the other two still have to agree as well.
///
/// This is deliberately weaker than the claim the module docs argue for, and
/// the difference is worth stating plainly rather than burying: an
/// iteration-count difference is evidence the value reaches a reader, not
/// evidence that it still buys a better answer anywhere. On this corpus it
/// does not, and `pounce-qp`'s own
/// `issue848_second_order_certification::the_check_can_be_switched_off_and_then_the_saddle_comes_back`
/// remains the test that the engine-level behaviour is real. If a future
/// fixture separates the two settings by objective again, that assertion
/// belongs back here.
#[test]
fn the_switch_is_still_read_end_to_end() {
    let mut responded = Vec::new();
    for name in [
        "nonconvex_two_escapes.nl",
        "nonconvex_qp.nl",
        "nonconvex_qcqp.nl",
    ] {
        let f = fixture(name);
        let f = f.to_str().unwrap();
        let off = solve(&[f, "--no-sol", "algorithm=active-set-sqp"]);
        let on = solve(&[
            f,
            "--no-sol",
            "algorithm=active-set-sqp",
            "sqp_qp_certify_second_order=yes",
        ]);
        if (off.0 - on.0).abs() > 1e-9 || off.1 != on.1 {
            responded.push(format!("{name}: off={off:?} on={on:?}"));
        }
    }
    assert!(
        !responded.is_empty(),
        "sqp_qp_certify_second_order changed neither the objective nor the          iteration count on any of the three fixtures it is known to move.          That is gh #677's shape: an option the registry accepts and no          reader consults."
    );
}

/// The claim the fixture above lost, kept where it is still true: with the
/// check on, the arm reaches the corner minimum and not the ridge point.
///
/// This does not compare the two settings, so gh #873 making the default
/// agree with it does not weaken it — it pins the *answer*, which is what a
/// user of the option cares about, and it fails if the escape ladder ever
/// stops reaching `C`.
#[test]
fn with_the_check_on_the_arm_reaches_the_corner_minimum() {
    let f = fixture("nonconvex_two_escapes.nl");
    let f = f.to_str().unwrap();
    let on = objective(&[
        f,
        "--no-sol",
        "algorithm=active-set-sqp",
        "sqp_qp_certify_second_order=yes",
    ]);
    assert!(
        (on + 6752.25).abs() < 1e-3,
        "with the check on the arm should reach the global minimum at the \
         corner, f = -6752.25; got {on}. A different value means the escape \
         ladder changed — see nonconvex_two_escapes.py, which documents which \
         answer each rung gives."
    );
}
