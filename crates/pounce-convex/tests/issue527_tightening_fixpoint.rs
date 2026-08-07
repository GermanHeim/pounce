//! gh #527 — the fixpoint must actually reach a fixpoint, and must say so.
//!
//! `presolve` documents itself as iterating the reduction passes to a
//! **fixpoint**. On netlib `bore3d` it never reached one: it exited on the
//! `MAX_ROUNDS` layer cap on every solve, at every cap tried (32, 64, 200),
//! which meant a defensive constant — not the algorithm — was choosing which
//! of two different reduced problems the solver was handed, and nothing
//! anywhere recorded that it had.
//!
//! The mechanism is that bound tightening is the one reduction that consumes
//! nothing. A fixing removes its column and an aggregation removes its row,
//! so those can fire at most `n + m` times however long the loop runs.
//! Narrowing a box leaves the column *and* the row in place, so a pair of
//! rows that mutually imply ever-tighter bounds on each other's variables
//! fires every round forever, converging toward a limit it never reaches.
//!
//! The obvious guard — refuse an improvement that is too small — does not
//! work, and it is worth being precise about *why*, because the absolute and
//! the relative version fail on different models and neither failure is
//! visible from the other.
//!
//! - An **absolute** floor already exists (`BOUND_FEAS_TOL`). It does stop a
//!   cascade whose bounds collapse toward *zero* — `bore3d`'s #523 cascade
//!   died on it once its boxes reached `1e-9`. It cannot stop one converging
//!   to a limit of `1e3`, where the improvements stay far above any floor
//!   anyone would set. That is the model below, and the case these tests pin.
//! - A **relative** floor fails on `bore3d` itself, whose real cascade shrinks
//!   each bound to `3.887e-2` of its previous value — a 96% relative
//!   improvement, every round, forever. No relative threshold stops that.
//!
//! The model below is *not* evidence about relative thresholds: its relative
//! step is 0.1%, which a relative floor could well catch. It is a
//! self-contained stand-in for the mechanism, since `bore3d` needs a corpus
//! fixture that is not vendored. The relative half of the argument rests on
//! the measured `bore3d` ratio, recorded in `presolve.rs`.
//!
//! `MAX_BOX_REFINEMENTS` bounds the *number* of refinements instead of judging
//! their size, so neither failure mode applies to it and no scale-dependent
//! constant enters.

use pounce_convex::presolve::{FixpointExit, PresolveOutcome, presolve, solve_with_presolve};
use pounce_convex::{QpOptions, QpProblem, QpSolution, QpStatus, Triplet, solve_qp_ipm};
use pounce_feral::FeralSolverInterface;
use pounce_linsol::SparseSymLinearSolverInterface;

fn backend() -> Box<dyn SparseSymLinearSolverInterface> {
    Box::new(FeralSolverInterface::new())
}

/// Two nonnegative variables whose rows imply each other's upper bound with
/// a contraction factor just under one:
///
/// ```text
///   x₀ − 0.999 x₁ ≤ 1        ⇒  ub(x₀) ← 1 + 0.999 · ub(x₁)
///   x₁ − 0.999 x₀ ≤ 1        ⇒  ub(x₁) ← 1 + 0.999 · ub(x₀)
/// ```
///
/// Starting from `ub = 1e6`, that is `u ← 1 + 0.999u`, converging to
/// `1/(1 − 0.999) = 1000` — from above, monotonically, by a strictly positive
/// amount every single round. It needs some 28,000 rounds to shave the last
/// improvement below `BOUND_FEAS_TOL`, so before #527 the loop ran until the
/// layer cap stopped it, whatever the cap was.
///
/// The two rows overlap on both columns, so the disjoint-source rule lets
/// only one of them fire per round and they alternate — one refinement per
/// round, which is the shape the refinement budget counts.
///
/// `x₂` carries the objective (`min x₂` subject to `x₂ ≥ 10`) so the reduced
/// problem is a real solve with a known answer rather than an empty one.
fn contracting_pair() -> QpProblem {
    QpProblem {
        n: 3,
        p_lower: vec![],
        c: vec![0.0, 0.0, 1.0],
        a: vec![],
        b: vec![],
        g: vec![
            Triplet::new(0, 0, 1.0),
            Triplet::new(0, 1, -0.999),
            Triplet::new(1, 1, 1.0),
            Triplet::new(1, 0, -0.999),
            Triplet::new(2, 2, -1.0),
        ],
        h: vec![1.0, 1.0, -10.0],
        lb: vec![0.0, 0.0, f64::NEG_INFINITY],
        ub: vec![1e6, 1e6, f64::INFINITY],
    }
}

/// The regression: this presolve stops because it ran out of *reductions*,
/// not because it ran out of *layers*.
#[test]
fn a_contracting_bound_cascade_reaches_a_fixpoint() {
    let prob = contracting_pair();
    let PresolveOutcome::Reduced(ps) = presolve(&prob) else {
        panic!("feasible bounded problem");
    };
    let st = ps.stats();
    assert_eq!(
        st.exit,
        FixpointExit::Fixpoint,
        "the loop stopped on the layer cap after {} layers, not at a \
         fixpoint — the cap is deciding the reduction again (stats={st:?})",
        st.rounds,
    );
    // The cascade is real: it did tighten, repeatedly, before settling.
    assert!(
        st.tightened_bounds >= 2,
        "expected the cascade to run at all, stats={st:?}"
    );
    // And it settled because the *budget* ran out, not because the loop was
    // let run long enough to exhaust a 28,000-round cascade some other way.
    // Two box sides at `MAX_BOX_REFINEMENTS` each, one refinement per round.
    assert!(
        st.rounds <= 32,
        "expected the budget to end this in a couple of dozen layers, \
         stats={st:?}"
    );
}

/// Running out of refinements costs at most a *looser* box, never a wrong
/// one — the bounds already derived stay, and the answer does not move.
#[test]
fn the_refinement_budget_does_not_change_the_answer() {
    let prob = contracting_pair();
    let with = solve_with_presolve(&prob, |r| solve_qp_ipm(r, &QpOptions::default(), backend));
    let without: QpSolution = solve_qp_ipm(&prob, &QpOptions::default(), backend);
    assert_eq!(with.status, QpStatus::Optimal, "presolved solve");
    assert_eq!(without.status, QpStatus::Optimal, "unpresolved solve");
    // min x₂ s.t. x₂ ≥ 10, with x₀/x₁ free to sit anywhere feasible.
    assert!(
        (with.obj - 10.0).abs() < 1e-6,
        "presolved objective {} != 10",
        with.obj
    );
    assert!(
        (with.obj - without.obj).abs() < 1e-6,
        "presolve moved the optimum: {} vs {}",
        with.obj,
        without.obj
    );
    // Every bound the cascade derived is an over-approximation of the true
    // limit (1000), so the recovered point must respect the *original* box.
    for i in 0..prob.n {
        assert!(
            with.x[i] >= prob.lb_of(i) - 1e-6 && with.x[i] <= prob.ub_of(i) + 1e-6,
            "x[{i}] = {} outside its original box",
            with.x[i]
        );
    }
}

/// A cascade whose limit is far from zero is the case the existing absolute
/// floor cannot catch. Pinning the numbers so a future "just add a minimum
/// improvement threshold" change has to confront them.
///
/// Note what this does and does not establish. The *absolute* floor demonstrably
/// fails here — tens of thousands of rounds before it bites. The *relative*
/// step is 0.1%, so a relative floor might well stop this particular cascade;
/// the evidence that a relative floor fails is `bore3d`'s measured 96%, which
/// needs the corpus and so lives in `presolve.rs` rather than here.
#[test]
fn the_absolute_floor_would_take_tens_of_thousands_of_rounds_here() {
    // The propagation is `u ← 1 + 0.999·u` from `u = 1e6`, limit 1000.
    let (mut u, limit) = (1e6_f64, 1000.0_f64);
    let mut rounds = 0;
    loop {
        let next = 1.0 + 0.999 * u;
        let improvement = u - next;
        if improvement <= 1e-9 {
            break;
        }
        // The step is a *constant* fraction of the remaining distance to the
        // limit — and that distance is precisely what no runtime test can
        // see, since the limit is what propagation has not reached yet.
        let of_gap = improvement / (u - limit);
        assert!(
            (of_gap - 0.001).abs() < 1e-5,
            "expected a constant fraction of the gap, got {of_gap}"
        );
        // Measured against the bound itself — the only quantity a threshold
        // could actually compare — it opens at 0.1% and *decays* toward zero
        // as the bound approaches its limit. Recorded because it is the
        // honest weakness of this model as an argument about relative
        // thresholds: one would eventually catch this cascade. It is the
        // absolute floor that provably does not, which is what this test
        // asserts below.
        if rounds == 0 {
            let of_bound = improvement / u;
            assert!(
                (of_bound - 0.001).abs() < 1e-5,
                "expected the opening step to be ~0.1% of the bound, got {of_bound}"
            );
        }
        u = next;
        rounds += 1;
        assert!(rounds < 100_000, "guard");
    }
    assert!(
        rounds > 25_000,
        "expected the absolute floor to take tens of thousands of rounds to \
         bite, got {rounds}"
    );
}

/// The catalog's other reductions must still cascade across rounds — the
/// budget is on box refinement only. A chain of singleton equalities needs
/// one round per link, and all of them must still fire.
#[test]
fn structural_reductions_still_cascade_across_rounds() {
    // x₀ = 1; x₁ = x₀ + 1; x₂ = x₁ + 1; … each row becomes a singleton only
    // after the previous one has been fixed and substituted out.
    let k = 20;
    let n = k + 1;
    let mut a = vec![Triplet::new(0, 0, 1.0)];
    let mut b = vec![1.0];
    for i in 1..n {
        a.push(Triplet::new(i, i, 1.0));
        a.push(Triplet::new(i, i - 1, -1.0));
        b.push(1.0);
    }
    let prob = QpProblem {
        n,
        p_lower: vec![],
        c: vec![0.0; n],
        a,
        b,
        g: vec![],
        h: vec![],
        lb: vec![],
        ub: vec![],
    };
    let PresolveOutcome::Reduced(ps) = presolve(&prob) else {
        panic!("feasible problem");
    };
    let st = ps.stats();
    assert_eq!(st.exit, FixpointExit::Fixpoint, "stats={st:?}");
    assert_eq!(
        st.reduced_vars, 0,
        "every variable is determined; stats={st:?}"
    );
}
