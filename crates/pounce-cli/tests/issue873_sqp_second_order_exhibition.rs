//! gh #873: the active-set SQP arm's second-order escape finds the negative
//! curvature and then throws it away.
//!
//! gh #856 gave that arm an escape so it would stop certifying constrained
//! maxima. On the repo's **own** fixture for the class,
//! `nonconvex_two_escapes.nl`, it did not fire — the arm returned `0.0`, the
//! point the fixture's generator labels a *maximum*, and reported
//! `EXIT: Optimal Solution Found.`, while the default NLP arm on the same file
//! walked its documented ladder to the global minimum `−6752.25`. pounce
//! disagreed with itself across two arms on a file it ships.
//!
//! # The branch these fixtures reach
//!
//! `issue856_sqp_second_order.rs`'s fixtures reach *escape-succeeds* and
//! *no-negative-curvature*. Every defect below lives in the third branch —
//! **curvature is found and the exhibition declines it** — which had no
//! fixture at all. Per CLAUDE.md's rule, the fixture for the other branch is
//! the test, not a duplicate of the first: on `nonconvex_two_escapes` the
//! escape *did* find `ev[0] = −0.9` against a threshold of `−2e-5` and `d = e₀`,
//! every time, and then discarded it.
//!
//! # Four independent mechanisms, each sufficient alone
//!
//! The fourth — `a_refutation_may_not_buy_its_improvement_with_infeasibility`
//! — was found by `scripts/sweep-fixtures.sh`, not by this file, and only
//! *after* fixing the first three made the ray searchable. Its own doc
//! comment carries the numbers. It is here as a standing reminder that the
//! suite asserts status and objective, so a change that keeps both and moves
//! only the trajectory is invisible to every test in this file.
//!
//! 1. **The ray was never searched.** `exhibit_better_point` left its
//!    backtracking loop on *feasibility alone*, so `alpha` never halved past
//!    the first feasible trial — and `alpha` starts at the far wall. The
//!    profile along `d` is not monotone: on that fixture the quartic term puts
//!    `f = +1.8` at the wall and `−0.225` in the interior of the same ray, so
//!    testing only the wall declines a direction that descends. The issue
//!    pins the mechanism causally — varying only the box half-width `B`, the
//!    verdict flips at exactly `B = √2`, the root of `0.225·B⁴ − 0.45·B²`.
//!
//! 2. **The first acceptable point was returned, not the best.** Covered by
//!    `the_exhibition_returns_the_best_sign_not_the_first` below, which is
//!    also the regression test for a defect introduced *while fixing 3*.
//!
//! 3. **Absolute thresholds on scale-dependent quantities** — the same class as
//!    gh #872, in three places: the curvature threshold's `h_scale…max(1.0)`,
//!    the acceptance bar's additive `1e-10`, and the active-set test's use of
//!    `constr_viol_tol` as an absolute distance in `x` units.
//!
//! All three floors are **lowered only** (`.min(1.0)`), so no solve that
//! succeeds today becomes newly refutable, and every refutation still has to
//! exhibit a strictly better feasible point before it changes any answer.
//!
//! # Deliberately not fixed here
//!
//! The issue's **D3**: the exhibition walks a straight tangent and tests the
//! *objective*, so a maximum whose negative curvature lives entirely in the
//! `λ·∇²c` term is structurally un-refutable at any tolerance. That is a gap in
//! gh #856's coverage rather than a regression — the issue records that
//! `ipopt` 3.14.19 returns `0.0` on the same family, and a local solver is
//! entitled to stop at a genuine KKT point. Closing it needs a curved
//! exhibition, not a threshold.
//!
//! Also unfixed, and worth knowing when reading the issue's table: this arm
//! caps escapes with its own `MAX_SECOND_ORDER_ESCAPES = 8` and never reads
//! `neg_curv_escapes`. That option is the NLP arm's. It is why the SQP column
//! was *constant* across the option — evidence the escape was not firing, but
//! not the reason it was not.

use std::path::PathBuf;
use std::process::Command;

fn run(name: &str, extra: &[&str]) -> String {
    let mut fx = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    fx.push("tests");
    fx.push("fixtures");
    fx.push(name);
    let sol = std::env::temp_dir().join(format!("pounce_873_{name}.sol"));
    let out = Command::new(PathBuf::from(env!("CARGO_BIN_EXE_pounce")))
        .arg(fx)
        .arg(&sol)
        .args(extra)
        .output()
        .expect("run pounce");
    String::from_utf8_lossy(&out.stdout).into_owned()
}

/// The **unscaled** objective — the second number on the line, not the first.
/// The scaled column is what the solver worked in and does not have to equal
/// the value the user asked about; reading it is how the ladder on
/// `nonconvex_two_escapes` was misread once already while writing this file.
fn objective(stdout: &str) -> Option<f64> {
    stdout
        .lines()
        .find(|l| l.trim_start().starts_with("Objective."))
        .and_then(|l| l.split_whitespace().nth(2))
        .and_then(|v| v.parse().ok())
}

fn solved(stdout: &str) -> bool {
    stdout.contains("EXIT: Optimal Solution Found.")
}

const SQP: &str = "algorithm=active-set-sqp";

/// Both legs. The Python frontend and the CasADi plugin both select
/// `limited-memory` on their own whenever no exact Lagrangian Hessian is
/// available, so a fix that only holds on the exact leg is half a fix — and
/// gh #856's own escape was originally gated on `SqpHessianSource::Exact`,
/// which left the L-BFGS leg certifying the same maximum.
const LEGS: [&[&str]; 2] = [&[SQP], &[SQP, "hessian_approximation=limited-memory"]];

/// D1, on the repo's own fixture. Its generator names the three stationary
/// points:
///
/// ```text
/// A = (0, 0)      f =       0     W = diag(−0.9, +2000)   <-- a MAXIMUM along x₀
/// B = (±1, 0)     f =  −0.225     W = diag(+1.8,  −0.9)   <-- a saddle
/// C = (±2, ±1.5)  f = −6752.25    the global minimum, at the corner
/// ```
///
/// The arm used to return A. Searching the ray *and* scoring both signs now
/// takes it to C directly.
#[test]
fn the_sqp_arm_no_longer_certifies_the_documented_maximum() {
    for leg in LEGS {
        let out = run("nonconvex_two_escapes.nl", leg);
        let obj = objective(&out).expect("an objective line");
        assert!(solved(&out), "{leg:?}: expected a solve\n{out}");
        assert!(
            obj < -6752.0,
            "{leg:?}: the SQP arm returned {obj}; point A (0.0) is the \
             fixture's documented maximum and C is −6752.25\n{out}"
        );
    }
}

/// D1's other half, and the regression test for a defect introduced while
/// fixing D2's acceptance bar.
///
/// `min −x₀² + x₁²` with `x₀ ∈ [−2, 5e−2]`. The curvature direction is `±e₀`
/// and the two signs are worth wildly different amounts: `+e₀` reaches
/// `f = −2.5e−3` at the near wall, `−e₀` reaches the global `f = −4`.
/// `exhibit_better_point` scans `[+1, −1]` in that order.
///
/// Before gh #873 this returned `−2.5e−3`. It reached `−4` only for a *narrow*
/// band of `g` where the absolute `1e-10` acceptance bar happened to reject the
/// near-wall improvement and let the loop fall through to the other sign — a
/// coincidence, not a design, and one that inverts the moment the bar is made
/// scale-relative. Scoring every trial and returning the best removes the
/// dependence on both the sign order and the bar.
#[test]
fn the_exhibition_returns_the_best_sign_not_the_first() {
    for leg in LEGS {
        let out = run("sqp_exhibit_signs.nl", leg);
        let obj = objective(&out).expect("an objective line");
        assert!(solved(&out), "{leg:?}: expected a solve\n{out}");
        assert!(
            (obj - (-4.0)).abs() < 1e-9,
            "{leg:?}: expected the global −4 from the `−e₀` sign, got {obj}; \
             −2.5e−3 is the near wall, i.e. the first acceptable point rather \
             than the best one\n{out}"
        );
    }
}

/// D2, the active-set tolerance. `sqp_exhibit_units_s1.nl` and
/// `sqp_exhibit_units_s1em2.nl` are the same model in two systems of units,
/// `u = S·x`; the optimum is `−4` in both.
///
/// The bound `x₀ ≤ 5e−6` is genuinely inactive at the solution, but the active
/// set compared its distance against `constr_viol_tol`, an absolute distance in
/// `x` units. At `S = 1e−2` that distance is `5e−8`, the bound was frozen into
/// the working set, and the only direction of negative curvature the model has
/// was closed — so the arm certified the constrained maximum `0.0`.
///
/// The `S = 1` leg is the control: it was correct before and must stay correct,
/// so a failure here reads as "the answer moved with the units" rather than
/// "the fixture is wrong".
#[test]
fn the_active_set_does_not_freeze_a_bound_that_a_change_of_units_moved() {
    for leg in LEGS {
        for fx in ["sqp_exhibit_units_s1.nl", "sqp_exhibit_units_s1em2.nl"] {
            let out = run(fx, leg);
            let obj = objective(&out).expect("an objective line");
            assert!(solved(&out), "{fx} {leg:?}: expected a solve\n{out}");
            assert!(
                (obj - (-4.0)).abs() < 1e-9,
                "{fx} {leg:?}: expected −4 at every scale, got {obj}\n{out}"
            );
        }
    }
}

/// D2, the two floors that are absolute in the objective's own units:
/// `ev[0] >= −1e-8·h_scale` with `h_scale…max(1.0)`, and the acceptance bar's
/// additive `1e-10`.
///
/// `min k·x₀x₁ s.t. x₀ + x₁ = 2` has reduced Hessian `−k` on the null space of
/// its equality — as indefinite at `k = 1e−20` as at `k = 1`. The true minimum
/// is `−63k` at the corner `(9, −7)`; the stationary point `(1, 1)` is the
/// constrained *maximum*, worth `+k`. Before the fix the arm returned `+k`.
#[test]
fn an_indefinite_model_below_the_absolute_floors_is_still_refuted() {
    const K: f64 = 1e-20;
    for leg in LEGS {
        let out = run("sqp_tiny_objective_k1em20.nl", leg);
        let obj = objective(&out).expect("an objective line");
        assert!(solved(&out), "{leg:?}: expected a solve\n{out}");
        assert!(
            obj < -60.0 * K,
            "{leg:?}: expected ≈ −63k = {:e}, got {obj}; +{K:e} is the \
             constrained maximum\n{out}",
            -63.0 * K
        );
    }
}

/// The paired negative control, and the lesson gh #872's `units_qp_convex.nl`
/// taught: lowering a floor is only a fix if what sits below it is still judged
/// on its merits. Without this, "refute everything at tiny scale" passes every
/// other test in this file.
///
/// `min k(x₀² + x₁²) s.t. x₀ + x₁ = 2` is strictly convex at every `k`, with
/// minimum `2k` at `(1, 1)` and reduced Hessian `+4k`. Because the curvature
/// threshold now scales with `k` rather than being floored at unit scale, that
/// `+4k` is still read as positive at `k = 1e−20` instead of being swamped —
/// and the point is certified, not refuted. Measured identical on the baseline
/// binary and after the fix, on both legs.
#[test]
fn a_convex_model_at_the_same_tiny_scale_is_not_newly_refuted() {
    const K: f64 = 1e-20;
    for leg in LEGS {
        let out = run("sqp_tiny_objective_convex.nl", leg);
        let obj = objective(&out).expect("an objective line");
        assert!(solved(&out), "{leg:?}: expected a solve\n{out}");
        assert!(
            (obj - 2.0 * K).abs() < 1e-3 * K,
            "{leg:?}: expected the true minimum 2k = {:e}, got {obj}; anything \
             below it is a refutation of a point that is genuinely optimal\n{out}",
            2.0 * K
        );
    }
}

/// The **fourth** branch, which none of the fixtures above reaches and which
/// the fixture sweep — not the test suite — is what found.
///
/// Curvature is found, a better point exists, and it is better only because it
/// is *less feasible*. `exhibit_better_point` admitted any trial inside
/// `constr_viol_tol`, which is the bar for calling a solve converged, not the
/// bar a refutation of local optimality has to clear. Measured on this fixture
/// while fixing D1: the KKT point is feasible to `2.2e-16`, and the accepted
/// trials violated the rows by `6e-7` — nine orders worse, legal under
/// `constr_viol_tol = 1e-6` — to gain `4.4e-7` of objective. The arm then
/// restored feasibility, returned to the same point, and did it again, eight
/// times, until `MAX_SECOND_ORDER_ESCAPES` capped it.
///
/// The answer never changed: `0.87189754866737` before and after, identical to
/// 15 significant figures. Only the iteration count moved, 15 → 45. That is
/// gh #544's shape exactly — the right answer, slowly — and the reason
/// CLAUDE.md requires the sweep on a trajectory change: the suite asserts
/// status and objective, so nothing here would have seen it.
///
/// `cresc4` is a curved-constraint model, which is why it is the one that
/// exposes this: a straight tangent probe leaves a curved feasible set at
/// order `α²`, so *every* trial along the ray is slightly infeasible and the
/// only question is whether the bar admits it.
///
/// The ceiling is deliberately loose. Pinning 15 exactly would fail on any
/// benign retuning of a step rule, which is not what this test is about; 45 is
/// what the defect produced, and anything at or above 30 means the escape is
/// cycling again.
#[test]
fn a_refutation_may_not_buy_its_improvement_with_infeasibility() {
    let out = run("cresc4.nl", &[SQP, "hessian_approximation=limited-memory"]);
    assert!(solved(&out), "expected a solve\n{out}");
    let obj = objective(&out).expect("an objective line");
    assert!(
        (obj - 0.871_897_548_667_377_6).abs() < 1e-12,
        "cresc4's optimum moved: {obj}\n{out}"
    );
    let iters: u32 = out
        .lines()
        .find(|l| l.trim_start().starts_with("Number of Iterations"))
        .and_then(|l| l.split_whitespace().last())
        .and_then(|v| v.parse().ok())
        .expect("an iteration count");
    assert!(
        iters < 30,
        "cresc4 took {iters} iterations for an unchanged objective; the \
         second-order escape is buying objective with constraint violation \
         again (it was 15, and 45 with the defect)\n{out}"
    );
}
