//! gh #877 F-3, end to end through the builder: a wrong answer reported as
//! `SolveSucceeded`.
//!
//! The engine-level pins live in
//! `crates/pounce-presolve/tests/issue877_fbbt_unsound_tightening.rs`. This
//! file exists because the severity of F-3 is not "FBBT returns a bad box" —
//! it is that the bad box reaches the IPM, the IPM converges on it, and the
//! caller is handed the wrong optimum with `status = SolveSucceeded`,
//! `infeasibility_witness: None`, and no diagnostic anywhere. That is only
//! visible from outside the presolve crate.
//!
//! ```text
//! minimize x   s.t.  g(x) = x ∈ [-10, 10],   x ∈ [-10, 10]
//! exact optimum: x = -10
//!
//! before:  fbbt=no   SolveSucceeded  x = -10.000000098702795
//!          fbbt=yes  SolveSucceeded  x =  -9.990002698385514e-9   ← wrong
//! ```
//!
//! The tape restates `constraints()` exactly at its root, so the builder's
//! two-point value check passes. What it carries is a dead `Ln(0)` slot that
//! no path from the root reaches — routine output from a producer that emits
//! one tape per row out of a shared CSE pool, which is the usage the tape
//! format is sold on.
//!
//! Not evidence about: the `.nl` path (its translator emits reachable slots
//! only, which is why this was unreachable before hand-written tapes were
//! accepted), or any shipping default — `presolve` and `presolve_fbbt` are
//! both `no` unless asked for, as they are here.

use pounce_nlp::expression_provider::{FbbtOp, FbbtTape};
use pounce_rs::prelude::*;

/// `min x` s.t. `g(x) = x ∈ [-10, 10]`, `x ∈ [-10, 10]`. Optimum `x = -10`.
struct DeadSlot;

impl Problem for DeadSlot {
    fn objective(&self, x: &[f64]) -> f64 {
        x[0]
    }
    fn gradient(&self, _x: &[f64], g: &mut [f64]) -> bool {
        g[0] = 1.0;
        true
    }
    fn n_constraints(&self) -> usize {
        1
    }
    fn constraints(&self, x: &[f64], out: &mut [f64]) {
        out[0] = x[0];
    }
    fn jacobian(&self, _x: &[f64], j: &mut [f64]) -> bool {
        j[0] = 1.0;
        true
    }
    fn constraint_expression(&self, _i: usize) -> Option<FbbtTape> {
        Some(FbbtTape {
            ops: vec![
                FbbtOp::Var(0),     // 0
                FbbtOp::Ln(0),      // 1  dead: nothing downstream reads it
                FbbtOp::Const(0.0), // 2
                FbbtOp::Add(0, 2),  // 3  root = x + 0 = x  (restates g exactly)
            ],
        })
    }
}

fn solve(fbbt: bool) -> pounce_rs::NlpSolution {
    let mut nlp = Nlp::new(DeadSlot)
        .var_bounds(&[-10.0], &[10.0])
        .constraint_bounds(&[-10.0], &[10.0])
        .x0(&[0.0])
        .option_int("print_level", 0);
    if fbbt {
        nlp = nlp
            .option_str("presolve", "yes")
            .option_str("presolve_fbbt", "yes");
    }
    nlp.try_solve().expect("solve")
}

#[test]
fn a_dead_tape_slot_does_not_move_the_reported_optimum() {
    let off = solve(false);
    let on = solve(true);

    assert!(off.success, "baseline solve failed: {:?}", off.status);
    assert!(on.success, "fbbt solve failed: {:?}", on.status);

    // The bug was silent: both runs report success, and only the number is
    // wrong. Asserting the status alone would have passed on the defect —
    // which is the whole reason this test asserts the answer.
    assert!(
        (off.x[0] - on.x[0]).abs() < 1e-6,
        "FBBT moved the optimum: fbbt=no x = {}, fbbt=yes x = {} (pre-fix: -9.99e-9)",
        off.x[0],
        on.x[0]
    );
    assert!(
        (on.x[0] + 10.0).abs() < 1e-6,
        "x should be at the lower bound -10, got {}",
        on.x[0]
    );
}

#[test]
fn the_dead_slot_produces_no_tightening_and_no_witness() {
    let on = solve(true);
    let report = on
        .fbbt_report
        .expect("presolve_fbbt=yes must produce a report");
    assert_eq!(
        report.bound_updates, 0,
        "nothing about this model is tightenable; report = {report:?}"
    );
    assert_eq!(report.infeasibility_witness, None);
    assert_eq!(report.total_tightening, 0.0);
}
