//! The `ma57_*` options must reach the MA57 backend the factory builds.
//!
//! `pounce_hsl::ma57::Options::from_options_list` has always been unit
//! tested, so the reader was covered and looked live. Nothing tested that
//! the reader was *reached*, and it was not: every production
//! construction of `Ma57SolverInterface` called `::new()`, which
//! hard-codes `Options::defaults()`. All nine options were registered,
//! documented, accepted — and discarded (gh#825). The symptom was that
//! there was no symptom: two solves whose `ma57_*` blocks spanned eight
//! orders of magnitude in `ma57_pivtol` and swapped the elimination
//! ordering printed identical iteration logs and an objective identical
//! to all seventeen digits.
//!
//! So this test asserts the property the missing one would have: build
//! the factory the way the application builds it, ask it for an MA57
//! backend, and read the settings back off the live object.
//!
//! Two things it is deliberately *not*:
//!
//! * Not a re-test of the reader. Every expected value below is written
//!   out by hand rather than obtained from `from_options_list`, because
//!   a comparison of the reader against itself passes just as happily
//!   when the factory ignores it.
//! * Not a check that the values are *upstream's*. That is
//!   `upstream_options.rs`'s job; the registry's defaults and the
//!   reader's fallbacks are compared in `default_options_are_the_registrys`
//!   below, which is the gh#677 shape (registered with one default, read
//!   with another) applied to this option family.
//!
//! Runs only under `--features ma57`, which is the only build in which
//! MA57 exists. CI cannot link CoinHSL, so this test does not run there;
//! `no_production_site_builds_ma57_with_default_options` is the half that
//! does, and it is why that one exists.
#![cfg(feature = "ma57")]
#![allow(clippy::unwrap_used)]

use pounce_algorithm::IpoptApplication;
use pounce_algorithm::alg_builder::LinearSolverChoice;
use pounce_algorithm::application::{
    Ma57Config, default_backend_factory, default_backend_factory_with_sink,
    feral_config_from_options, ma57_config_from_options,
};
use pounce_common::OptionsList;
use pounce_hsl::{Ma57Options, Ma57SolverInterface};
use pounce_linsol::SparseSymLinearSolverInterface;
use pounce_linsol::summary::LinearSolverSummary;
use std::sync::{Arc, Mutex};

/// An options list with the registry attached, as the application has.
///
/// The registry matters: without it `get_integer_value` on an unset
/// option is an `Err` and the reader's own fallback answers, while with
/// it the *registered* default answers. Production always has the
/// registry, so a test without one would exercise a branch no user
/// reaches — and would miss a registry/reader default mismatch entirely.
fn app() -> IpoptApplication {
    let mut app = IpoptApplication::new();
    app.initialize().expect("registry initializes");
    app
}

fn set(app: &mut IpoptApplication, assignment: &str) {
    app.options_mut()
        .read_from_str(assignment, true)
        .unwrap_or_else(|e| panic!("{assignment}: {e:?}"));
}

/// The settings of the MA57 backend behind a factory's trait object.
/// `Ma57Options` is `Copy`, so they outlive the boxed backend.
fn options_of(backend: Box<dyn SparseSymLinearSolverInterface>) -> Ma57Options {
    *backend
        .as_any()
        .expect("the MA57 backend opts into the downcast seam")
        .downcast_ref::<Ma57SolverInterface>()
        .expect("linear_solver=ma57 builds an Ma57SolverInterface")
        .options()
}

/// Build an MA57 backend the way the application does, and read its
/// settings back.
fn ma57_options_from(options: &OptionsList, prefix: &str) -> Ma57Options {
    let mut factory = default_backend_factory(
        feral_config_from_options(options),
        ma57_config_from_options(options, prefix),
    );
    options_of(factory(LinearSolverChoice::Ma57))
}

/// The nine values used throughout, chosen so every one differs from
/// both the registry default and its neighbours — a factory that crossed
/// two fields, or that honoured some options and not others, fails on the
/// specific field rather than passing by coincidence.
struct Probe;
impl Probe {
    const ASSIGNMENTS: [&'static str; 9] = [
        "ma57_print_level 3",
        "ma57_pivtol 0.5",
        "ma57_pivtolmax 0.75",
        "ma57_pre_alloc 5.0",
        "ma57_pivot_order 2",
        "ma57_automatic_scaling yes",
        "ma57_block_size 128",
        "ma57_node_amalgamation 1",
        "ma57_small_pivot_flag 1",
    ];

    fn assert_arrived(o: &Ma57Options) {
        assert_eq!(o.print_level(), 3, "ma57_print_level");
        assert_eq!(o.pivtol(), 0.5, "ma57_pivtol");
        assert_eq!(o.pivtolmax(), 0.75, "ma57_pivtolmax");
        assert_eq!(o.pre_alloc(), 5.0, "ma57_pre_alloc");
        assert_eq!(o.pivot_order(), 2, "ma57_pivot_order");
        assert!(o.automatic_scaling(), "ma57_automatic_scaling");
        assert_eq!(o.block_size(), 128, "ma57_block_size");
        assert_eq!(o.node_amalgamation(), 1, "ma57_node_amalgamation");
        assert_eq!(o.small_pivot_flag(), 1, "ma57_small_pivot_flag");
    }
}

/// The headline: all nine options, set at the main-IPM prefix, arrive.
///
/// Before gh#825 every assertion here read the registry default instead.
#[test]
fn every_ma57_option_reaches_the_backend() {
    let mut app = app();
    for a in Probe::ASSIGNMENTS {
        set(&mut app, a);
    }
    Probe::assert_arrived(&ma57_options_from(app.options(), ""));
}

/// The same through `default_backend_factory_with_sink`, which is the
/// factory the application actually installs (`application.rs`, the
/// `linear_backend_factory` fallback) — the plain one is what the
/// restoration sub-IPM's callers mint. Both had the defect; a fix to one
/// is not a fix to the other.
#[test]
fn the_sink_factory_reaches_the_backend_too() {
    let mut app = app();
    for a in Probe::ASSIGNMENTS {
        set(&mut app, a);
    }
    let sink = Arc::new(Mutex::new(LinearSolverSummary::default()));
    let mut factory = default_backend_factory_with_sink(
        feral_config_from_options(app.options()),
        ma57_config_from_options(app.options(), ""),
        sink,
    );
    Probe::assert_arrived(&options_of(factory(LinearSolverChoice::Ma57)));
}

/// The `"resto."` prefix is a real facility, not decoration.
///
/// `from_options_list` has always taken a prefix, mirroring upstream's
/// `Ma57TSolverInterface::InitializeImpl(options, prefix)`, and with no
/// callers it did nothing. The restoration sub-IPM builds its own backend
/// through its own `InnerBackendFactoryFactory`, so the two prefixes must
/// land on genuinely independent backends.
///
/// Note which branch each half takes: the un-prefixed read must see the
/// *main* value even though a `resto.` value is also set, and vice versa.
/// A prefix that were ignored would pass a test that set only one of them.
#[test]
fn the_resto_prefix_selects_its_own_values() {
    let mut app = app();
    set(&mut app, "ma57_pivtol 0.5");
    set(&mut app, "ma57_pivot_order 2");
    set(&mut app, "resto.ma57_pivtol 0.125");
    set(&mut app, "resto.ma57_pivot_order 3");

    let main = ma57_options_from(app.options(), "");
    assert_eq!(main.pivtol(), 0.5, "main IPM keeps its own pivtol");
    assert_eq!(main.pivot_order(), 2);

    let resto = ma57_options_from(app.options(), "resto.");
    assert_eq!(resto.pivtol(), 0.125, "resto. overrides");
    assert_eq!(resto.pivot_order(), 3);
}

/// An option set only at the main prefix still reaches the restoration
/// backend, because upstream's prefixed lookup falls back to the
/// un-prefixed value. The previous test pins the override; this pins the
/// inheritance, which is the other branch of the same rule.
#[test]
fn the_resto_backend_inherits_unprefixed_values() {
    let mut app = app();
    set(&mut app, "ma57_pivtol 0.5");
    let resto = ma57_options_from(app.options(), "resto.");
    assert_eq!(resto.pivtol(), 0.5);
}

/// With nothing set, the backend must carry exactly what it carried
/// before gh#825 — otherwise the fix is a trajectory change on every
/// existing MA57 run.
///
/// This is the gh#677 shape: `limited_memory_initialization` was
/// registered with one default and read with another, and nothing
/// compared the two. Here the reader's fallbacks are dead code in
/// production (the registry always answers first), so a divergence would
/// show up only as a silently moved default — which is what this asserts
/// against.
#[test]
fn default_options_are_the_registrys() {
    let app = app();
    let o = ma57_options_from(app.options(), "");
    assert_eq!(o.print_level(), 0);
    assert_eq!(o.pivtol(), 1e-8);
    assert_eq!(o.pivtolmax(), 1e-4);
    assert_eq!(o.pre_alloc(), 1.05);
    assert_eq!(o.pivot_order(), 5);
    assert!(!o.automatic_scaling());
    assert_eq!(o.block_size(), 16);
    assert_eq!(o.node_amalgamation(), 16);
    assert_eq!(o.small_pivot_flag(), 0);
    // And identical to the no-OptionsList constructor, which is what
    // `::new()` used to hand production.
    assert_eq!(o, Ma57Options::defaults());
}

/// The `ma57_pivtolmax` reader has two branches, and the *value* half of
/// each is checked here; the refusal half is
/// `ma57_pivtol_bracket.rs`, which needs no HSL and so runs in CI.
///
/// Upstream (`IpMa57TSolverInterface.cpp:311-320`) lifts the registered
/// default to `ma57_pivtol` when the option is unset, and takes an
/// explicitly set value verbatim. pounce used to apply the lift
/// unconditionally, which silently rewrote an explicit value that sat
/// below `ma57_pivtol`.
#[test]
fn an_unset_pivtolmax_is_lifted_to_pivtol() {
    let mut app = app();
    set(&mut app, "ma57_pivtol 0.5");
    let o = ma57_options_from(app.options(), "");
    assert_eq!(o.pivtol(), 0.5);
    assert_eq!(
        o.pivtolmax(),
        0.5,
        "the 1e-4 default must be lifted to ma57_pivtol, not left under it"
    );
}

/// The other branch: set explicitly, taken verbatim. `0.75` is above
/// `ma57_pivtol` so the pair is legal and the lift would be a no-op —
/// what this pins is that a *legal* explicit value is not touched.
#[test]
fn an_explicit_pivtolmax_is_taken_verbatim() {
    let mut app = app();
    set(&mut app, "ma57_pivtol 0.25");
    set(&mut app, "ma57_pivtolmax 0.75");
    let o = ma57_options_from(app.options(), "");
    assert_eq!(o.pivtolmax(), 0.75);
}

/// A contradictory explicit pair reaches the reader **verbatim**, not
/// clamped.
///
/// This is the test that separates the two candidate rules. Everywhere
/// else they agree: `ma57_pivtol_bracket.rs` refuses this pair inside
/// `optimize_tnlp`, so no solve can carry it, and every legal pair is
/// untouched by a clamp. Without this assertion, restoring the old
/// unconditional `max(pivtolmax, pivtol)` leaves the whole suite green.
///
/// Verbatim is the right answer for a layering reason. This reader has
/// no error channel — it returns `Options`, not `Result` — so a clamp
/// here is a silent rewrite of what the user wrote, which is the exact
/// behaviour being removed. The refusal belongs to the layer that can
/// deliver a verdict, and `IpoptApplication::optimize_tnlp` is reached
/// by every solve entry point pounce has.
///
/// What a caller who bypasses the application and builds a backend
/// straight from an `OptionsList` gets: a `pivtolmax` under `pivtol`,
/// which makes `increase_quality` lower the tolerance once and then
/// report exhausted. Useless, not dangerous — and it is the answer to
/// the question they asked.
#[test]
fn a_contradictory_explicit_pivtolmax_is_not_silently_clamped() {
    let mut app = app();
    set(&mut app, "ma57_pivtol 0.5");
    set(&mut app, "ma57_pivtolmax 1e-9");
    let o = ma57_options_from(app.options(), "");
    assert_eq!(
        o.pivtolmax(),
        1e-9,
        "the reader must report what the user wrote; refusing the pair is \
         `optimize_tnlp`'s job, and clamping it here is how the contradiction \
         used to disappear without a diagnostic"
    );
    assert_eq!(o.pivtol(), 0.5);
}

/// With neither set, the lift is what produces the registry default —
/// `max(1e-4, 1e-8)`. Pins that the two branches agree on the default
/// path, which is what keeps an existing MA57 run bit-identical.
#[test]
fn the_default_pair_is_unmoved_by_the_lift() {
    let app = app();
    let o = ma57_options_from(app.options(), "");
    assert_eq!(o.pivtol(), 1e-8);
    assert_eq!(o.pivtolmax(), 1e-4);
}

/// `Ma57Config::default()` is the empty config, and a factory given one
/// must still produce a *working* backend rather than, say, zeros in
/// ICNTL. The two test call sites in the tree pass it (they never select
/// MA57), so it has to be the upstream defaults and not junk.
#[test]
fn the_empty_config_is_the_upstream_defaults() {
    let mut factory = default_backend_factory(
        feral_config_from_options(&OptionsList::default()),
        Ma57Config::default(),
    );
    assert_eq!(
        options_of(factory(LinearSolverChoice::Ma57)),
        Ma57Options::defaults()
    );
}
