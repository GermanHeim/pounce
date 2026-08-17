//! gh#616 — what the safeguarded least-square initializer's accept
//! test can and cannot express.
//!
//! gh#605 gave the initializer a backtracking safeguard: the min-norm
//! solution of the *linearized* constraints is only accepted when it
//! actually reduces the true nonlinear violation. Over the 57 CLI
//! fixtures under `least_square_init_primal=yes` that is a broad win,
//! and it costs two tolerance downgrades — `csfi2` and `eigenb2` finish
//! at `SolvedToAcceptableLevel` where the unsafeguarded step reached
//! `SolveSucceeded`.
//!
//! gh#616 asked whether the accept test needs a tighter criterion to
//! recover them. It does not, and cannot. The corpus evidence is pinned
//! in `pounce-cli/tests/issue_616_ls_init_downgrades.rs`; what is pinned
//! *here* is the algebra that makes the corpus result a general
//! statement rather than a coincidence of two models.
//!
//! [`DefaultIterateInitializer::accepts_trial`] reads exactly four
//! numbers: `theta_0`, the trial's `theta`, `alpha`, and the tunable
//! `eta` (`least_square_init_accept_ratio`). The measured decisions on
//! the two fixtures that matter are:
//!
//! | fixture | `theta_0` | `theta` | `alpha` | outcome vs the unsafeguarded step |
//! |---|---|---|---|---|
//! | `eigena2` | 1.0 | 0.2500000062500001 | 0.5 | 78 → 65 iterations, `SolveSucceeded` both |
//! | `eigenb2` | 1.0 | 0.2500000062500001 | 0.5 | 55 → 57 iterations, `SolveSucceeded` → `SolvedToAcceptableLevel` |
//!
//! Bit-identical inputs, opposite outcomes. That is what the tests
//! below turn into assertions.

use pounce_algorithm::init::default::DefaultIterateInitializer as Init;
use pounce_common::types::Number;

/// The safeguard decision measured on both `eigena2` and `eigenb2`
/// under `least_square_init_primal=yes` (`RUST_LOG=pounce::algorithm=debug`).
const THETA_0: Number = 1.0;
const THETA_ACCEPTED: Number = 0.2500000062500001;
const ALPHA_ACCEPTED: Number = 0.5;

/// The shipped default for `least_square_init_accept_ratio`.
const ETA_DEFAULT: Number = 1e-2;

/// `eta` is meaningful only on `(0, 1]`: at `eta > 1` the `alpha = 1`
/// trial would have to reach a *negative* violation, so the full step
/// becomes unreachable no matter how good it is, and the option stops
/// being a safeguard ratio. Sampled densely across that interval.
fn meaningful_etas() -> Vec<Number> {
    let mut v = vec![1e-12, 1e-6, ETA_DEFAULT];
    for k in 1..=100 {
        v.push(k as Number / 100.0);
    }
    v
}

/// The floor the safeguard promises, independent of `eta`: a trial that
/// does not strictly reduce the true violation is never accepted. This
/// is gh#605's contract, and it is what makes `csfi2` unreachable —
/// every one of its four trials is worse than `theta_0`, so no setting
/// of `eta` can produce an acceptance there.
#[test]
fn no_eta_accepts_a_trial_that_does_not_reduce_the_violation() {
    for &eta in &meaningful_etas() {
        for &alpha in &[1.0, 0.5, 0.25, 0.125] {
            // Exactly equal, and strictly worse: `csfi2`'s four trials
            // are all of the second kind (theta_0 = 1508.554..., every
            // trial above it).
            assert!(
                !Init::accepts_trial(THETA_0, THETA_0, alpha, eta),
                "eta = {eta}, alpha = {alpha} accepted a trial that did \
                 not move the violation at all",
            );
            assert!(
                !Init::accepts_trial(THETA_0, THETA_0 * 1.001, alpha, eta),
                "eta = {eta}, alpha = {alpha} accepted a trial that made \
                 the violation WORSE",
            );
        }
    }
}

/// The gh#616 conclusion for `eigenb2`, stated as a property: retuning
/// `least_square_init_accept_ratio` is not a route to its old
/// `SolveSucceeded`, because no meaningful `eta` rejects the step it
/// took.
///
/// The arithmetic: acceptance is `theta_0 - theta >= eta * alpha *
/// theta_0`, i.e. `0.75 >= eta * 0.5`, i.e. `eta <= 1.5`. The whole
/// meaningful range of `eta` sits below that.
#[test]
fn no_eta_rejects_the_eigenb2_step() {
    for &eta in &meaningful_etas() {
        assert!(
            Init::accepts_trial(THETA_0, THETA_ACCEPTED, ALPHA_ACCEPTED, eta),
            "eta = {eta} rejected the trial that eigenb2 (and eigena2) \
             accepted; if this ever becomes reachable, gh#616's \
             conclusion that eta cannot separate them needs re-deriving",
        );
    }
}

/// And the reason a "prefer the untouched point when the improvement is
/// marginal" band cannot be the answer either: `eigenb2`'s accepted step
/// is not marginal. It cuts the violation by 4x, which is the *median*
/// of the sixteen accepted steps in the corpus — the same ratio as
/// `airport` (103.6 → 25.96), `cresc4` (1715.3 → 437.2) and both
/// `issue_508_infeasible_gap_*` fixtures, every one of which is a win.
/// A band drawn tight enough to exclude `eigenb2` excludes those too.
#[test]
fn the_eigenb2_step_is_a_median_sized_reduction_not_a_marginal_one() {
    let ratio = THETA_ACCEPTED / THETA_0;
    assert!(
        (ratio - 0.25).abs() < 1e-8,
        "eigenb2's accepted step cut the violation to {ratio} of theta_0; \
         gh#616's argument assumes 1/4",
    );
    // The corpus's accepted-step ratios, measured under
    // `least_square_init_primal=yes`. `eigenb2` is not in the marginal
    // tail: `user_scaling_bad_var_suffix` (0.889) and
    // `user_scaling_suffix` (0.875) are, and both are wins as well.
    let corpus_ratios: &[Number] = &[
        0.0,    // hs13_bigstart
        5.0e-9, // nonconvex_qp
        5.2e-6, // jit1_node
        0.01,   // linear_eq_collapsed_box
        0.037,  // infeasible_equalities
        0.194,  // parametric
        0.25,   // eigena2, eigenb2, issue_508_infeasible_gap_1em4
        0.2505, // airport
        0.2548, // cresc4
        0.2624, // issue_508_infeasible_gap_1em2
        0.5382, // hs71_obj1e8
        0.875,  // user_scaling_suffix, user_scaling_var_suffix
        0.8889, // user_scaling_bad_var_suffix
    ];
    let stricter = corpus_ratios.iter().filter(|&&r| r < ratio).count();
    assert!(
        stricter >= 6,
        "at least six accepted corpus steps reduce the violation by more \
         than eigenb2's does; got {stricter}. A rejection band placed \
         above eigenb2's ratio would take the rest of the tail with it, \
         which is the measurement gh#616 rests on",
    );
}

/// Guard on the direction of the knob, so a future reader does not have
/// to re-derive which way it tightens: larger `eta` demands more, and
/// the trials it starts rejecting are the *short* ones, where the linear
/// model predicted little. `eigenb2`'s `alpha = 0.5` step is nowhere
/// near that frontier.
#[test]
fn eta_tightens_toward_short_steps_first() {
    // A step that removes 1% of the violation at alpha = 1: rejected as
    // soon as eta exceeds 0.01.
    assert!(Init::accepts_trial(1.0, 0.99, 1.0, 0.009));
    assert!(!Init::accepts_trial(1.0, 0.99, 1.0, 0.011));
    // The same 1% reduction bought with alpha = 1/8 survives eight times
    // more `eta`, because the model predicted eight times less.
    assert!(Init::accepts_trial(1.0, 0.99, 0.125, 0.079));
    assert!(!Init::accepts_trial(1.0, 0.99, 0.125, 0.081));
}

/// A non-finite trial violation is never accepted, whatever `eta` says.
/// `theta = NaN` fails every comparison anyway; this pins that the
/// predicate does not accidentally invert somewhere and let one through.
#[test]
fn non_finite_trials_are_never_accepted() {
    for &eta in &meaningful_etas() {
        assert!(!Init::accepts_trial(THETA_0, Number::NAN, 1.0, eta));
        assert!(!Init::accepts_trial(THETA_0, Number::INFINITY, 1.0, eta));
    }
}
