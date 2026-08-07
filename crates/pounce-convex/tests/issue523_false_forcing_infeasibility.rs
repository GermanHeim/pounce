//! gh #523 — a near-forcing row must not pin, and an infeasibility verdict
//! must survive a re-derivation without the speculative fixings.
//!
//! Netlib `bore3d` and its Maros-Mészáros quadratic twin `QBORE3D` (both
//! n=315, m=233, both feasible, both solved to the Ipopt-MA57 reference by
//! POUNCE's NLP path) came back from the convex path as
//! `Infeasible_Problem_Detected` at iteration 0, in five milliseconds, with
//! no diagnostic.
//!
//! The chain: bound tightening propagated a group of nonnegative variables'
//! upper bounds geometrically toward their true limit of zero, round after
//! round. By round 21 those boxes were ~1e-8 wide, and one equality row's
//! activity range `[-6.3e-10, 4.9e-8]` came within `ACTIVITY_TOL` of its
//! right-hand side `0` at the *min* vertex. The forcing reduction read that
//! as "this row can hold only at that vertex" and pinned all six of its
//! variables to bounds — including two, with coefficients `-1.14e-1` and
//! `-5.7e-3`, to *upper* bounds `4.8e-9` and `1.5e-8` away from zero. A gap
//! of `6.3e-10` had licensed a displacement of `1.5e-8`. Substituted into the
//! next row those two appeared in, the residual was `2.0e-8` against a
//! tolerance of `1.0e-9`, and presolve called a feasible problem infeasible.
//!
//! The first test below is that geometry, distilled to the two rows that
//! matter. The rest guard the two halves of the fix: forcing must still fire
//! when a row really is forcing, and an infeasibility verdict must still be
//! reached — and now be *explained* — when the problem really is infeasible.

use pounce_convex::presolve::{PresolveOutcome, presolve, solve_with_presolve};
use pounce_convex::{QpOptions, QpProblem, QpSolution, QpStatus, Triplet, solve_qp_ipm};
use pounce_feral::FeralSolverInterface;
use pounce_linsol::SparseSymLinearSolverInterface;

fn backend() -> Box<dyn SparseSymLinearSolverInterface> {
    Box::new(FeralSolverInterface::new())
}

fn with_presolve(prob: &QpProblem) -> QpSolution {
    solve_with_presolve(prob, |r| solve_qp_ipm(r, &QpOptions::default(), backend))
}

fn without_presolve(prob: &QpProblem) -> QpSolution {
    solve_qp_ipm(prob, &QpOptions::default(), backend)
}

/// The `bore3d` geometry: two equality rows over six nonnegative variables
/// whose boxes bound tightening has already narrowed to ~1e-8, plus a
/// seventh variable carrying the objective so the solve is not trivial.
///
/// Row A is the near-forcing one — activity range `[-6.29e-10, 4.87e-8]`
/// against `b = 0`, so its min vertex is `6.29e-10` short of the right-hand
/// side, inside `ACTIVITY_TOL`. Pinning there puts `x3` at `4.76e-9` and `x4`
/// at `1.52e-8`. Row B is the row that then cannot hold: it wants those five
/// to sum to `x5`, and `x5`'s pin is `0`.
///
/// The whole system is satisfied at the origin, and `x6 = 10` is the optimum.
fn near_forcing_pair() -> QpProblem {
    QpProblem {
        n: 7,
        p_lower: vec![],
        c: vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        a: vec![
            // Row A: 8.17e-2 x0 + 5.32e-2 x1 + 8.3e-3 x2
            //        − 1.14e-1 x3 − 5.7e-3 x4 + x5 = 0
            Triplet::new(0, 0, 8.17e-2),
            Triplet::new(0, 1, 5.32e-2),
            Triplet::new(0, 2, 8.3e-3),
            Triplet::new(0, 3, -1.14e-1),
            Triplet::new(0, 4, -5.7e-3),
            Triplet::new(0, 5, 1.0),
            // Row B: x0 + x1 + x2 + x3 + x4 − x5 = 0
            Triplet::new(1, 0, 1.0),
            Triplet::new(1, 1, 1.0),
            Triplet::new(1, 2, 1.0),
            Triplet::new(1, 3, 1.0),
            Triplet::new(1, 4, 1.0),
            Triplet::new(1, 5, -1.0),
        ],
        b: vec![0.0, 0.0],
        // x6 ≥ 10, as a row so the objective survives presolve as a solve.
        g: vec![Triplet::new(0, 6, -1.0)],
        h: vec![-10.0],
        lb: vec![0.0; 7],
        ub: vec![
            1.980622733188442e-7,
            3.041670625967964e-7,
            2.206593781384032e-8,
            4.759319920632226e-9,
            1.5170332247015222e-8,
            1.618168773014957e-8,
            100.0,
        ],
    }
}

#[test]
fn near_forcing_row_does_not_falsify_a_feasible_lp() {
    let prob = near_forcing_pair();

    // The origin (with x6 = 10) satisfies every row and every box, so any
    // infeasibility verdict here is false by construction.
    match presolve(&prob) {
        PresolveOutcome::Reduced(ps) => assert!(
            ps.discarded_infeasibility().is_none(),
            "presolve should not even reach an infeasibility claim, got {:?}",
            ps.discarded_infeasibility().map(|t| t.to_string())
        ),
        PresolveOutcome::Infeasible(t) => {
            panic!("feasible LP reported primal infeasible by presolve: {t}")
        }
        PresolveOutcome::Unbounded => panic!("bounded LP reported unbounded"),
    }

    let sol = with_presolve(&prob);
    let bare = without_presolve(&prob);
    assert_eq!(
        sol.status,
        QpStatus::Optimal,
        "presolved solve must succeed"
    );
    assert_eq!(bare.status, QpStatus::Optimal, "bare solve must succeed");
    assert!(
        (sol.obj - bare.obj).abs() <= 1e-6 * (1.0 + bare.obj.abs()),
        "presolved objective {} != bare objective {}",
        sol.obj,
        bare.obj
    );
    assert!(
        (sol.obj - 10.0).abs() < 1e-6,
        "optimum is x6 = 10, got {}",
        sol.obj
    );
}

/// The pin-tightness guard must not cost the reduction its normal case: a row
/// whose activity range touches its right-hand side *exactly* is still
/// forcing, and still fixes every variable in it.
#[test]
fn an_exactly_touching_row_still_forces() {
    // x0 + x1 = 0 with x ∈ [0, 1]²: min activity is exactly 0, so both
    // variables are pinned to their lower bounds — a real forcing row.
    let prob = QpProblem {
        n: 3,
        p_lower: vec![],
        c: vec![1.0, 1.0, 1.0],
        a: vec![
            Triplet::new(0, 0, 1.0),
            Triplet::new(0, 1, 1.0),
            Triplet::new(1, 2, 1.0),
            Triplet::new(1, 0, 1.0),
        ],
        b: vec![0.0, 0.5],
        g: vec![],
        h: vec![],
        lb: vec![0.0; 3],
        ub: vec![1.0; 3],
    };
    match presolve(&prob) {
        PresolveOutcome::Reduced(ps) => assert!(
            ps.stats().forcing_rows >= 1,
            "an exactly-touching row must still be recognized as forcing"
        ),
        other => panic!(
            "expected Reduced, got {}",
            match other {
                PresolveOutcome::Infeasible(t) => format!("Infeasible ({t})"),
                _ => "Unbounded".to_string(),
            }
        ),
    }
    let sol = with_presolve(&prob);
    assert_eq!(sol.status, QpStatus::Optimal);
    assert!(
        (sol.obj - 0.5).abs() < 1e-6,
        "optimum is 0.5, got {}",
        sol.obj
    );
}

/// A real infeasibility is still proved in presolve — and now says which
/// screen proved it and on what, which is the whole diagnostic a caller gets
/// for a verdict reached in zero iterations.
#[test]
fn a_real_infeasibility_is_reported_with_its_trigger() {
    // x0 + x1 = 5 with x ∈ [0, 1]²: the activity range is [0, 2].
    let prob = QpProblem {
        n: 2,
        p_lower: vec![],
        c: vec![0.0, 0.0],
        a: vec![Triplet::new(0, 0, 1.0), Triplet::new(0, 1, 1.0)],
        b: vec![5.0],
        g: vec![],
        h: vec![],
        lb: vec![0.0, 0.0],
        ub: vec![1.0, 1.0],
    };
    match presolve(&prob) {
        PresolveOutcome::Infeasible(t) => {
            assert!(!t.screen.is_empty(), "the screen must be named");
            assert!(
                t.detail.contains("row 0"),
                "the trigger must name the row it tripped on, got {t}"
            );
        }
        _ => panic!("an out-of-range equality must presolve to Infeasible"),
    }
    assert_eq!(with_presolve(&prob).status, QpStatus::PrimalInfeasible);
}

/// The guard's cost, stated as a test: when a *genuine* infeasibility is only
/// reachable through a forcing pin, the re-derivation has to find it another
/// way — here domain propagation does, by shrinking the same box the pin
/// would have chosen — so the verdict survives.
///
/// This is what keeps the guard from being a blanket "never report
/// infeasible": it withholds only the fixings, and everything that merely
/// narrows or screens is still there to confirm.
#[test]
fn a_genuine_forcing_infeasibility_is_still_confirmed() {
    // x0 + x1 = 0 forces x0 = x1 = 0 (both nonnegative); x0 + 2·x1 = 1 then
    // cannot hold.
    let prob = QpProblem {
        n: 2,
        p_lower: vec![],
        c: vec![0.0, 0.0],
        a: vec![
            Triplet::new(0, 0, 1.0),
            Triplet::new(0, 1, 1.0),
            Triplet::new(1, 0, 1.0),
            Triplet::new(1, 1, 2.0),
        ],
        b: vec![0.0, 1.0],
        g: vec![],
        h: vec![],
        lb: vec![0.0, 0.0],
        ub: vec![1.0, 1.0],
    };
    match presolve(&prob) {
        PresolveOutcome::Infeasible(_) => {}
        PresolveOutcome::Reduced(_) => {
            // Acceptable only if the solver itself still says infeasible —
            // the guard may cost the fast path, never the answer.
            assert_eq!(
                with_presolve(&prob).status,
                QpStatus::PrimalInfeasible,
                "downgraded verdict must still come out infeasible"
            );
        }
        PresolveOutcome::Unbounded => panic!("not unbounded"),
    }
}
