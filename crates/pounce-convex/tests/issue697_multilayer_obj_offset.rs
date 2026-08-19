//! gh #697 — `Presolve::obj_offset` must be the *whole* reduction's objective
//! constant, not just the top object's own.
//!
//! An iterated presolve returns a wrapper whose layers live in a chain; the
//! wrapper itself substitutes nothing. The aggregate offset used to be a
//! stored field on that wrapper, initialized to `0.0` and never filled in, so
//! every multi-layer reduction reported "no offset" — and the CLI, which
//! feeds this value into the solver's `obj_constant`, silently dropped the
//! presolve contribution on exactly the reductions that had one.
//!
//! The defining property, layer by layer, is
//! `obj_original(x) = obj_reduced(x_reduced) + Σ offsetₖ`, so that is what
//! these tests assert — against an independently computed objective, not
//! against a restatement of the implementation.

use pounce_convex::presolve::{PresolveOutcome, presolve};
use pounce_convex::{QpOptions, QpProblem, QpStatus, Triplet, solve_qp_ipm};
use pounce_feral::FeralSolverInterface;
use pounce_linsol::SparseSymLinearSolverInterface;

fn backend() -> Box<dyn SparseSymLinearSolverInterface> {
    Box::new(FeralSolverInterface::new())
}

/// `0.5·xᵀPx + cᵀx`, with `p_lower` the lower triangle: the diagonal carries
/// the `0.5`, each off-diagonal entry stands for both symmetric halves.
fn objective(prob: &QpProblem, x: &[f64]) -> f64 {
    let mut v = 0.0;
    for t in &prob.p_lower {
        v += if t.row == t.col {
            0.5 * t.val * x[t.row] * x[t.col]
        } else {
            t.val * x[t.row] * x[t.col]
        };
    }
    for (i, &ci) in prob.c.iter().enumerate() {
        v += ci * x[i];
    }
    v
}

/// A cascade of equality rows: `x3 = 4` is a singleton, which turns
/// `x2 + x3 = 7` into one, which turns `x1 + x2 = 5` into one. No single pass
/// can take all three, so the fixpoint runs more than one layer — and each
/// fixed variable carries a linear and a quadratic term into the objective.
fn cascade() -> QpProblem {
    QpProblem {
        n: 4,
        p_lower: vec![
            Triplet::new(0, 0, 2.0),
            Triplet::new(1, 1, 2.0),
            Triplet::new(2, 2, 2.0),
            Triplet::new(3, 3, 2.0),
        ],
        c: vec![-2.0, 1.0, 1.0, 1.0],
        a: vec![
            Triplet::new(0, 3, 1.0), // x3 = 4
            Triplet::new(1, 2, 1.0), // x2 + x3 = 7
            Triplet::new(1, 3, 1.0),
            Triplet::new(2, 1, 1.0), // x1 + x2 = 5
            Triplet::new(2, 2, 1.0),
        ],
        b: vec![4.0, 7.0, 5.0],
        g: vec![Triplet::new(0, 0, 1.0), Triplet::new(0, 1, 1.0)],
        h: vec![10.0], // slack at the optimum; keeps x0 in the reduced problem
        lb: vec![],
        ub: vec![],
    }
}

#[test]
fn multi_layer_presolve_reports_the_summed_objective_offset() {
    let prob = cascade();
    let PresolveOutcome::Reduced(ps) = presolve(&prob) else {
        panic!("cascade should reduce, not prove infeasible or unbounded");
    };

    // The premise of the bug: this reduction really is multi-layer.
    let rounds = ps.stats().rounds;
    assert!(
        rounds >= 2,
        "expected an iterated presolve, got {rounds} layer(s)"
    );

    let red = solve_qp_ipm(&ps.reduced, &QpOptions::default(), backend);
    assert_eq!(red.status, QpStatus::Optimal);
    let full = ps.postsolve(&red);
    assert_eq!(full.status, QpStatus::Optimal);

    let want = objective(&prob, &full.x) - objective(&ps.reduced, &red.x);
    // x3=4, x2=3, x1=2 → (1·4 + 4²) + (1·3 + 3²) + (1·2 + 2²) = 38.
    assert!(
        (want - 38.0).abs() < 1e-6,
        "fixture drifted: expected a 38.0 constant, measured {want}"
    );
    assert!(
        (ps.obj_offset() - want).abs() < 1e-6,
        "obj_offset() = {}, but the reduction moved {want} into the objective",
        ps.obj_offset()
    );
}

/// The single-pass path is the one that already worked; it must keep working.
/// One singleton row, one layer, one offset — read through the same accessor.
#[test]
fn single_layer_presolve_still_reports_its_own_offset() {
    let prob = QpProblem {
        n: 2,
        p_lower: vec![Triplet::new(0, 0, 2.0), Triplet::new(1, 1, 2.0)],
        c: vec![-2.0, 1.0],
        a: vec![Triplet::new(0, 1, 1.0)], // x1 = 4
        b: vec![4.0],
        g: vec![],
        h: vec![],
        lb: vec![],
        ub: vec![],
    };
    let PresolveOutcome::Reduced(ps) = presolve(&prob) else {
        panic!("singleton row should reduce");
    };
    assert_eq!(ps.stats().rounds, 1, "fixture should be a single layer");
    // x1 = 4 → 1·4 + 0.5·2·4² = 20.
    assert!(
        (ps.obj_offset() - 20.0).abs() < 1e-9,
        "obj_offset() = {}",
        ps.obj_offset()
    );
}

/// A presolve that finds nothing reports no offset — the accessor must not
/// invent one for the passthrough layer.
#[test]
fn no_op_presolve_reports_zero_offset() {
    let prob = QpProblem {
        n: 2,
        p_lower: vec![Triplet::new(0, 0, 2.0), Triplet::new(1, 1, 2.0)],
        c: vec![-2.0, -3.0],
        a: vec![],
        b: vec![],
        g: vec![],
        h: vec![],
        lb: vec![],
        ub: vec![],
    };
    let PresolveOutcome::Reduced(ps) = presolve(&prob) else {
        panic!("unconstrained strictly convex QP should not be infeasible");
    };
    assert_eq!(ps.obj_offset(), 0.0);
}
