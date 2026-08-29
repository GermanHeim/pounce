//! `ma57_batched_backsolve` must stay registered, and must stay `no`.
//!
//! The option lets MA57 affirm
//! `SparseSymLinearSolverInterface::multi_solve_matches_single_solve`,
//! which is what admits `LowRankAugSystemSolver`'s SMW columns to a
//! single blocked back-substitution instead of one traversal of the
//! factor per column. MA57's blocked path is **not** bit-identical to
//! the per-column one — measured at about one ulp on gh#809's review
//! model, and enough to move the trajectory and the final iteration
//! count — so the option is a permission the user grants, not a
//! capability that should arrive switched on.
//!
//! Two properties, and neither is checkable by the tests that live
//! next to the backend:
//!
//! * The **registered default** is `no`. `pounce-hsl`'s own reader
//!   falls back to `false` when the option is absent, but that fallback
//!   is dead code in production — the registry always answers first —
//!   so a registry that defaulted to `yes` would silently switch every
//!   MA57 run onto a different trajectory while every fallback in the
//!   reader still read `false`. That is the gh#677 shape exactly:
//!   registered with one default, read with another, nothing comparing
//!   the two.
//! * The option is **registered at all**. Unregistered reads as unset,
//!   unset reads as `false`, and the feature would go quietly
//!   unreachable rather than fail — which is gh#825's failure mode with
//!   the sign flipped.
//!
//! This runs in CI, which the rest of the MA57 coverage cannot: CoinHSL
//! is licensed and cannot be linked here, so
//! `ma57_options_reach_the_backend.rs` is `#![cfg(feature = "ma57")]`
//! and `pounce-hsl`'s tests are excluded from every CI job. Registration
//! is deliberately unconditional on the `ma57` cargo feature — the
//! registry is built the same way in every build so that
//! `--print-options` is not build-dependent — which is what makes this
//! assertion possible here.

use pounce_algorithm::IpoptApplication;

fn app() -> IpoptApplication {
    let mut app = IpoptApplication::new();
    app.initialize().expect("registry initializes");
    app
}

/// Unset, with the registry attached, the option reads `false`.
///
/// Read through `OptionsList` rather than off the `RegisteredOption`
/// because that is the path production takes: `Ma57Options::from_options_list`
/// asks the same question of the same object.
#[test]
fn the_registered_default_is_no() {
    let app = app();
    let (v, _) = app
        .options()
        .get_bool_value("ma57_batched_backsolve", "")
        .expect("ma57_batched_backsolve is registered, so an unset read succeeds");
    assert!(
        !v,
        "ma57_batched_backsolve must default to `no`: turning it on is a \
         trajectory change (about one ulp per batched back-solve, enough to \
         move the final iteration count), not a free speed-up"
    );
}

/// And the restoration sub-solve inherits that default rather than
/// carrying one of its own.
///
/// The `resto.` prefix is a real facility for this family
/// (`ma57_options_reach_the_backend.rs::the_resto_prefix_selects_its_own_values`),
/// so "off by default" has to hold at both prefixes or the restoration
/// IPM batches while the main one does not.
#[test]
fn the_resto_prefix_defaults_to_no_too() {
    let app = app();
    let (v, _) = app
        .options()
        .get_bool_value("ma57_batched_backsolve", "resto.")
        .expect("the prefixed read falls back to the registered default");
    assert!(!v);
}

/// Setting it is what turns it on — the other branch, so that a
/// hard-coded `false` reader would not pass the two tests above.
///
/// Without this, a reader that ignored the option entirely and returned
/// `false` unconditionally satisfies everything else here.
#[test]
fn setting_it_is_visible_in_the_options_list() {
    let mut app = app();
    app.options_mut()
        .read_from_str("ma57_batched_backsolve yes", true)
        .expect("`yes` is a legal bool");
    let (v, _) = app
        .options()
        .get_bool_value("ma57_batched_backsolve", "")
        .expect("registered");
    assert!(v);
}
