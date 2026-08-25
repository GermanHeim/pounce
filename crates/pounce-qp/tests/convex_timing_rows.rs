//! gh #767: the active-set engine's linear algebra charges the convex path's
//! timing rows.
//!
//! The convex driver's `print_timing_statistics` report splits the solve into
//! symbolic factorization / numeric factorization / back-solve, and it draws
//! those three rows from whichever linear-algebra layer actually ran. The
//! interior-point engines go through `pounce_linsol::Factorization`; the
//! parametric active-set engine goes through this crate's [`LinearSolver`]
//! instead, and an uninstrumented one would report a solve it spent every
//! second of as costing nothing in the linear system — the same "0% for every
//! phase" report the issue is about, one engine down.
//!
//! Asserted on the raw wall-clock totals, not the printed rows: `Instant`
//! resolves nanoseconds, so any real work is strictly positive even when the
//! report's three-decimal row rounds it to `0.000s`.

use pounce_common::timing::{ConvexTimingScope, ConvexTimingStatistics};
use pounce_qp::LinearSolver;
use pounce_qp::kkt::KktTriplet;
use std::rc::Rc;

/// `[[4, 1], [1, 3]]` in 1-based lower-triangle triplets — symmetric positive
/// definite, so the factor succeeds and `resolve` has something to reuse.
fn spd_2x2() -> KktTriplet {
    KktTriplet {
        dim: 2,
        irn: vec![1, 2, 2],
        jcn: vec![1, 1, 2],
        vals: vec![4.0, 1.0, 3.0],
    }
}

#[test]
fn the_active_set_linear_solver_charges_each_convex_timing_row() {
    let stats = Rc::new(ConvexTimingStatistics::new());
    let scope = ConvexTimingScope::open(&stats);

    let mut solver = LinearSolver::new(Box::new(pounce_feral::FeralSolverInterface::new()));
    let kkt = spd_2x2();

    let mut rhs = vec![1.0, 2.0];
    solver
        .factorize_and_solve(&kkt, &mut rhs, None)
        .expect("SPD KKT factors");
    assert!(
        stats
            .linear_system_symbolic_factorization
            .total_wallclock_time()
            > 0.0,
        "the structure pass must charge the symbolic row"
    );
    assert!(
        stats.linear_system_factorization.total_wallclock_time() > 0.0,
        "the numeric factor must charge the factorization row"
    );
    assert_eq!(
        stats.linear_system_back_solve.total_wallclock_time(),
        0.0,
        "`factorize_and_solve`'s substitution is part of the factor call, not a \
         separate back-solve — no `resolve` has run yet"
    );

    let after_factor = stats.linear_system_factorization.total_wallclock_time();
    let mut rhs2 = vec![3.0, 4.0];
    solver.resolve(&mut rhs2).expect("cached factor resolves");
    assert!(
        stats.linear_system_back_solve.total_wallclock_time() > 0.0,
        "`resolve` must charge the back-solve row"
    );
    assert_eq!(
        stats.linear_system_factorization.total_wallclock_time(),
        after_factor,
        "reusing the cached factor must not be charged as a factorization"
    );

    // Outside a scope nothing is charged: this type is also reached from the
    // NLP-side SQP subproblem reader, which keeps its own timers.
    drop(scope);
    let before = stats.linear_system_back_solve.total_wallclock_time();
    solver.resolve(&mut [5.0, 6.0]).expect("resolve");
    assert_eq!(
        stats.linear_system_back_solve.total_wallclock_time(),
        before,
        "a closed scope must record nothing"
    );
}
