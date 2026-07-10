//! Wiring tests for algorithmic tuning constants that were registered
//! but never read (#191), so a user override was silently dropped and the
//! solver ran with the hard-coded struct default.
//!
//! Like `mu_options_wiring.rs`, these are pure wiring tests: they assert a
//! `OptionsList` set turns into the corresponding field on
//! `AlgorithmBuilder`. The numerics themselves are exercised by the
//! upstream-mirroring unit tests in `crate::ipopt_alg` /
//! `crate::ipopt_cq`.

use pounce_algorithm::alg_builder::AlgorithmBuilder;
use pounce_algorithm::application::IpoptApplication;

fn builder_from(setup: impl FnOnce(&mut IpoptApplication)) -> AlgorithmBuilder {
    let mut app = IpoptApplication::new();
    setup(&mut app);
    app.algorithm_builder_from_options()
}

#[test]
fn kappa_sigma_default_matches_registered() {
    // Untouched option ⇒ builder carries the upstream default 1e10.
    let b = builder_from(|_| {});
    assert!((b.kappa_sigma - 1e10).abs() <= 1e10 * 1e-12);
}

#[test]
fn kappa_sigma_override_flows_through() {
    let b = builder_from(|app| {
        app.options_mut()
            .set_numeric_value("kappa_sigma", 0.5, true, false)
            .unwrap();
    });
    // The documented `< 1` "disable the correction" value must survive.
    assert!((b.kappa_sigma - 0.5).abs() < 1e-12);
}

#[test]
fn kappa_d_default_matches_registered() {
    let b = builder_from(|_| {});
    assert!((b.kappa_d - 1e-5).abs() < 1e-17);
}

#[test]
fn kappa_d_override_flows_through() {
    let b = builder_from(|app| {
        app.options_mut()
            .set_numeric_value("kappa_d", 0.0, true, false)
            .unwrap();
    });
    assert_eq!(b.kappa_d, 0.0);
}
