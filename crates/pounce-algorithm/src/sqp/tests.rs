//! Scaffolding-level tests for the SQP module. Phase 5b commit 1
//! only verifies that the types compile, the defaults are sane,
//! and `AlgorithmChoice::ActiveSetSqp` is wired into the builder
//! enum. End-to-end optimize tests land in later commits.

use crate::alg_builder::AlgorithmChoice;
use crate::sqp::iterates::SqpIterates;
use crate::sqp::options::{SqpGlobalization, SqpHessianSource, SqpOptions};

#[test]
fn algorithm_choice_default_is_interior_point() {
    assert_eq!(AlgorithmChoice::default(), AlgorithmChoice::InteriorPoint);
}

#[test]
fn sqp_options_default_matches_design_note() {
    let opts = SqpOptions::default();
    assert_eq!(opts.globalization, SqpGlobalization::Filter);
    assert_eq!(opts.hessian, SqpHessianSource::Exact);
    assert_eq!(opts.max_iter, 200);
    assert!((opts.tol - 1e-8).abs() < f64::EPSILON);
    assert!((opts.l1_penalty - 1.0).abs() < f64::EPSILON);
}

#[test]
fn sqp_iterates_cold_has_expected_lengths_and_no_working_set() {
    let it = SqpIterates::cold(5, 3);
    assert_eq!(it.n(), 5);
    assert_eq!(it.m(), 3);
    assert_eq!(it.x, vec![0.0; 5]);
    assert_eq!(it.lambda_g, vec![0.0; 3]);
    assert_eq!(it.lambda_x, vec![0.0; 5]);
    assert!(it.working.is_none());
}
