//! Doubleton-equality aggregation (gh #494), end to end.
//!
//! `presolve.rs`'s contract is that a presolved-then-postsolved solve
//! yields a valid primal–dual point of the **original** problem, so that
//! is what these assert: full-space stationarity carrying the bound
//! multipliers, primal feasibility of every row including the ones the
//! aggregation consumed, dual signs, and complementarity — plus agreement
//! with a bare solve on the objective.
//!
//! The dual is the part worth the scrutiny. Planning transfers an
//! eliminated column's box onto its survivor, so a reduced solve reports
//! the bound force on a variable that may not declare that bound at all;
//! `aggregate::postsolve` re-attributes it. `alias_bound_multiplier_stays_
//! on_the_bounded_column` is the direct pin on that, and every `assert_kkt`
//! below would fail if the re-attribution were dropped.

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

/// Full-space KKT of the *original* problem, bound multipliers included:
/// `Px + c + Aᵀy + Gᵀz − z_lb + z_ub = 0`, `Ax = b`, `Gx ≤ h`, `z ≥ 0`,
/// `z_lb, z_ub ≥ 0`, and both complementarities.
fn assert_kkt(prob: &QpProblem, sol: &QpSolution, tol: f64) {
    let n = prob.n;
    let mut g = prob.c.clone();
    prob.p_mul(&sol.x, &mut g);
    prob.at_mul(&sol.y, &mut g);
    prob.gt_mul(&sol.z, &mut g);
    for i in 0..n {
        let stat = g[i] - sol.z_lb[i] + sol.z_ub[i];
        assert!(stat.abs() < tol, "stationarity[{i}] = {stat}");
        assert!(
            sol.z_lb[i] > -tol && sol.z_ub[i] > -tol,
            "bound dual sign [{i}]: {} / {}",
            sol.z_lb[i],
            sol.z_ub[i]
        );
        // A bound multiplier on a bound the problem never declared is not
        // a dual of this problem at all — the failure mode the survivor's
        // transferred box would otherwise produce.
        if prob.lb_of(i) <= f64::NEG_INFINITY {
            assert!(sol.z_lb[i].abs() < tol, "z_lb on unbounded [{i}]");
        }
        if prob.ub_of(i) >= f64::INFINITY {
            assert!(sol.z_ub[i].abs() < tol, "z_ub on unbounded [{i}]");
        }
        assert!(
            sol.x[i] >= prob.lb_of(i) - tol && sol.x[i] <= prob.ub_of(i) + tol,
            "box [{i}]: {} in [{}, {}]",
            sol.x[i],
            prob.lb_of(i),
            prob.ub_of(i)
        );
        if prob.lb_of(i) > f64::NEG_INFINITY {
            assert!(
                (sol.z_lb[i] * (sol.x[i] - prob.lb_of(i))).abs() < 1e-4,
                "lb complementarity [{i}]"
            );
        }
        if prob.ub_of(i) < f64::INFINITY {
            assert!(
                (sol.z_ub[i] * (prob.ub_of(i) - sol.x[i])).abs() < 1e-4,
                "ub complementarity [{i}]"
            );
        }
    }
    let mut ax = vec![0.0; prob.m_eq()];
    prob.a_mul(&sol.x, &mut ax);
    for (i, (&axi, &bi)) in ax.iter().zip(&prob.b).enumerate() {
        assert!((axi - bi).abs() < tol, "Ax=b row {i}: {axi} vs {bi}");
    }
    let mut gx = vec![0.0; prob.m_ineq()];
    prob.g_mul(&sol.x, &mut gx);
    for i in 0..prob.m_ineq() {
        let slack = prob.h[i] - gx[i];
        assert!(slack > -tol, "Gx≤h row {i}: slack {slack}");
        assert!(sol.z[i] > -tol, "z[{i}] = {}", sol.z[i]);
        assert!(
            (sol.z[i] * slack).abs() < 1e-4,
            "ineq complementarity row {i}"
        );
    }
}

/// Reduced `(columns, rows)`, asserting the aggregation is what did it.
///
/// Worth the assertion rather than inferring it from the sizes: several of
/// these shapes are also within reach of the pre-existing catalog — a free
/// column singleton in particular eats an alias row whenever one of its two
/// variables is unbounded and absent from `P` and `G`. A fixture that
/// quietly took *that* path would look like a passing test of this feature
/// while pinning none of it.
fn reduced_size(prob: &QpProblem) -> (usize, usize) {
    match presolve(prob) {
        PresolveOutcome::Reduced(ps) => {
            assert!(
                ps.stats().aggregated_vars > 0,
                "the aggregation did not fire on this fixture"
            );
            (ps.reduced.n, ps.reduced.m_eq() + ps.reduced.m_ineq())
        }
        other => panic!(
            "expected a reduction, got {}",
            match other {
                PresolveOutcome::Infeasible(_) => "Infeasible",
                PresolveOutcome::Unbounded => "Unbounded",
                PresolveOutcome::Reduced(_) => unreachable!(),
            }
        ),
    }
}

/// `with_presolve`, first asserting the aggregation fired.
fn with_aggregation(prob: &QpProblem) -> QpSolution {
    let _ = reduced_size(prob);
    with_presolve(prob)
}

// --- the shape the issue is about ---

/// An alias-heavy LP of the gh #487 shape: a chain of arc equalities
/// `x_i = x_{i+1}` over a block, with the objective and the real
/// constraints touching only the ends. Before this reduction the whole
/// chain reached the solver; the columns must actually drop.
#[test]
fn alias_chain_lp_columns_drop() {
    const BLOCKS: usize = 12;
    const CHAIN: usize = 8;
    let n = BLOCKS * CHAIN;
    let mut a = Vec::new();
    let mut b = Vec::new();
    for blk in 0..BLOCKS {
        for k in 0..CHAIN - 1 {
            let (i, j) = (blk * CHAIN + k, blk * CHAIN + k + 1);
            let r = b.len();
            a.push(Triplet::new(r, i, 1.0));
            a.push(Triplet::new(r, j, -1.0));
            b.push(0.0);
        }
    }
    // One real inequality per block on the chain's head, and a cost on
    // every column so no column is free/empty (which the pre-existing
    // catalog would have removed on its own).
    let mut g = Vec::new();
    let mut h = Vec::new();
    for blk in 0..BLOCKS {
        g.push(Triplet::new(blk, blk * CHAIN, -1.0));
        h.push(-1.0); // x_head ≥ 1
    }
    let prob = QpProblem {
        n,
        p_lower: (0..n).map(|i| Triplet::new(i, i, 2.0)).collect(),
        c: vec![0.0; n],
        a,
        b,
        g,
        h,
        lb: vec![],
        ub: vec![],
    };

    let (red_n, red_rows) = reduced_size(&prob);
    assert_eq!(red_n, BLOCKS, "one survivor per chain, got {red_n}");
    // Zero, not `BLOCKS`: the per-block "real inequality" `-x_head <= -1` is
    // itself a *singleton* row, and a singleton row is a bound — it folds into
    // the box as `x_head >= 1` and is then redundant as a row (gh #491). So the
    // alias rows this test is named for are all consumed, and so is every row
    // that survived them. The column count above is the aggregation's own
    // claim and is unchanged; the KKT and objective checks below are what
    // prove nothing was lost with them.
    assert_eq!(red_rows, 0, "every row consumed, got {red_rows}");

    let sol = with_presolve(&prob);
    assert_eq!(sol.status, QpStatus::Optimal);
    assert_kkt(&prob, &sol, 1e-6);
    // Each chain sits at 1, so the objective is BLOCKS·CHAIN·(0.5·2·1²).
    let want = (BLOCKS * CHAIN) as f64;
    assert!((sol.obj - want).abs() < 1e-6, "obj {} vs {want}", sol.obj);
}

/// The dual half of the same story, in the smallest instance that shows
/// it — and the direct pin on the re-attribution.
///
/// `x0 − 2·x1 = 0` with `x0 ≥ 1` and `x1 ∈ [−10, 10]`, minimizing `x0`.
/// Planning eliminates `x0` and transfers `≥ 1` onto `x1` as `≥ 0.5`, so
/// the reduced solve stops at `x1 = 0.5` reporting a bound multiplier
/// there. But `0.5` is strictly interior to `x1`'s **declared** box, so
/// that multiplier is not a dual of this problem at all: the force belongs
/// to `x0`, at `1`, scaled by the substitution's `α = 2`.
///
/// Every constant here is load-bearing. `x1` is boxed (not free) so the
/// free-column-singleton reduction cannot claim the row instead; `α ≠ 1`
/// so a recovery that forgot to scale by it would be wrong rather than
/// merely lucky; and `x1`'s own box is slack at the optimum so the
/// survivor genuinely cannot carry the multiplier itself.
#[test]
fn alias_bound_multiplier_stays_on_the_bounded_column() {
    let prob = QpProblem {
        n: 2,
        p_lower: vec![],
        c: vec![1.0, 0.0],
        a: vec![Triplet::new(0, 0, 1.0), Triplet::new(0, 1, -2.0)],
        b: vec![0.0],
        g: vec![],
        h: vec![],
        lb: vec![1.0, -10.0],
        ub: vec![f64::INFINITY, 10.0],
    };
    let sol = with_aggregation(&prob);
    assert_eq!(sol.status, QpStatus::Optimal);
    assert_kkt(&prob, &sol, 1e-6);
    assert!(
        (sol.x[0] - 1.0).abs() < 1e-6 && (sol.x[1] - 0.5).abs() < 1e-6,
        "{:?}",
        sol.x
    );
    assert!(
        (sol.z_lb[0] - 1.0).abs() < 1e-6,
        "the bound force belongs to x0, at unit scale: z_lb = {:?}",
        sol.z_lb
    );
    assert!(
        sol.z_lb[1].abs() < 1e-6 && sol.z_ub[1].abs() < 1e-6,
        "x1 sits interior to its declared box and cannot carry a bound \
         multiplier: {:?} / {:?}",
        sol.z_lb,
        sol.z_ub
    );
    assert!(
        sol.y[0].abs() < 1e-6,
        "consumed row multiplier should be 0 here, got {}",
        sol.y[0]
    );
}

/// The re-attribution, on a fixture where nothing else can do it for us.
///
/// Getting here takes some care. The catalog's bound-tightening pass runs
/// *before* the aggregation and, left to itself, propagates the eliminated
/// column's bound onto its partner anyway — after which the survivor is on
/// a bound of the aggregation layer's own input and simply keeps the
/// multiplier. So the alias row here is denied that: `x1 + x2 + x3 = 3` is
/// accepted as a tightening source first and claims `x1`, which makes the
/// alias row `x0 − 2·x1 = 0` fail the disjoint-source rule and go
/// untightened. The transferred `x0 ≥ 1` therefore reaches `x1` only
/// through the aggregation, and at the optimum `x1 = 0.5` sits strictly
/// inside its own declared `[−10, 3]`.
///
/// Hand-solved: `x2 = x3 = 1.25`, `x1 = 0.5`, `x0 = 1`, with `λ = −2.5` on
/// the sum row, `ν = −1.25` on the alias row, and the whole bound force
/// `z_lb[x0] = 8.75` on `x0` — the column that actually declares it.
#[test]
fn re_attribution_when_tightening_cannot_pre_empt_it() {
    let prob = QpProblem {
        n: 4,
        p_lower: vec![Triplet::new(2, 2, 2.0), Triplet::new(3, 3, 2.0)],
        c: vec![10.0, 0.0, 0.0, 0.0],
        a: vec![
            Triplet::new(0, 1, 1.0),
            Triplet::new(0, 2, 1.0),
            Triplet::new(0, 3, 1.0), // x1 + x2 + x3 = 3
            Triplet::new(1, 0, 1.0),
            Triplet::new(1, 1, -2.0), // x0 = 2·x1
        ],
        b: vec![3.0, 0.0],
        g: vec![],
        h: vec![],
        lb: vec![1.0, -10.0, 0.0, 0.0],
        ub: vec![f64::INFINITY, 10.0, 10.0, 10.0],
    };
    let sol = with_aggregation(&prob);
    assert_eq!(sol.status, QpStatus::Optimal);
    assert_kkt(&prob, &sol, 1e-5);
    for (i, want) in [1.0, 0.5, 1.25, 1.25].iter().enumerate() {
        assert!((sol.x[i] - want).abs() < 1e-5, "x[{i}] = {}", sol.x[i]);
    }
    assert!((sol.y[0] + 2.5).abs() < 1e-5, "y = {:?}", sol.y);
    assert!((sol.y[1] + 1.25).abs() < 1e-5, "y = {:?}", sol.y);
    assert!(
        (sol.z_lb[0] - 8.75).abs() < 1e-5,
        "the whole bound force belongs to x0: z_lb = {:?}",
        sol.z_lb
    );
    assert!(
        sol.z_lb[1].abs() < 1e-5 && sol.z_ub[1].abs() < 1e-5,
        "x1 is interior to its declared box: {:?} / {:?}",
        sol.z_lb,
        sol.z_ub
    );
}

/// The same re-attribution with the substitution's sign reversed, so the
/// eliminated column's *lower* bound arrives as an *upper* bound on the
/// survivor. `x0 + 2·x1 = 0`, `x0 ≥ 1`, `x1 ∈ [−10, 10]`, minimizing `x0`.
#[test]
fn re_attribution_across_a_sign_flip() {
    let prob = QpProblem {
        n: 2,
        p_lower: vec![],
        c: vec![1.0, 0.0],
        a: vec![Triplet::new(0, 0, 1.0), Triplet::new(0, 1, 2.0)],
        b: vec![0.0],
        g: vec![],
        h: vec![],
        lb: vec![1.0, -10.0],
        ub: vec![f64::INFINITY, 10.0],
    };
    let sol = with_aggregation(&prob);
    assert_eq!(sol.status, QpStatus::Optimal);
    assert_kkt(&prob, &sol, 1e-6);
    assert!(
        (sol.x[0] - 1.0).abs() < 1e-6 && (sol.x[1] + 0.5).abs() < 1e-6,
        "{:?}",
        sol.x
    );
    assert!((sol.z_lb[0] - 1.0).abs() < 1e-6, "z_lb = {:?}", sol.z_lb);
    assert!(
        sol.z_lb[1].abs() < 1e-6 && sol.z_ub[1].abs() < 1e-6,
        "{:?} / {:?}",
        sol.z_lb,
        sol.z_ub
    );
}

/// When the survivor *is* on one of its own declared bounds, it keeps the
/// multiplier — the same attribution a solve without this pass produces.
/// `x0 − x1 = 0` with both boxed at `[2, 5]` and a cost pushing to the
/// lower bound.
#[test]
fn a_survivor_on_its_own_bound_keeps_the_multiplier() {
    let prob = QpProblem {
        n: 2,
        p_lower: vec![],
        c: vec![1.0, 1.0],
        a: vec![Triplet::new(0, 0, 1.0), Triplet::new(0, 1, -1.0)],
        b: vec![0.0],
        g: vec![],
        h: vec![],
        lb: vec![2.0, 2.0],
        ub: vec![5.0, 5.0],
    };
    let sol = with_aggregation(&prob);
    assert_eq!(sol.status, QpStatus::Optimal);
    assert_kkt(&prob, &sol, 1e-6);
    assert!(
        (sol.x[0] - 2.0).abs() < 1e-6 && (sol.x[1] - 2.0).abs() < 1e-6,
        "{:?}",
        sol.x
    );
    // Both columns are at their own lower bound; the total force is 2 and
    // however it splits, stationarity (checked above) must hold.
    let total = sol.z_lb[0] + sol.z_lb[1];
    assert!((total - 2.0).abs() < 1e-5, "z_lb = {:?}", sol.z_lb);
}

/// Both signs of the aggregation coefficient, with the bound on the
/// eliminated side either way round, so the `α < 0` branch of the bound
/// transfer *and* of the multiplier re-attribution are both exercised.
#[test]
fn both_signs_of_the_coefficient() {
    for (a0, a1, bnd) in [
        (1.0, -1.0, 0.0),
        (1.0, 1.0, 0.0),
        (2.0, -3.0, 1.0),
        (2.0, 3.0, -1.0),
        (-2.0, 3.0, 1.0),
    ] {
        let prob = QpProblem {
            n: 2,
            p_lower: vec![Triplet::new(0, 0, 2.0), Triplet::new(1, 1, 1.0)],
            c: vec![1.0, -1.0],
            a: vec![Triplet::new(0, 0, a0), Triplet::new(0, 1, a1)],
            b: vec![bnd],
            g: vec![],
            h: vec![],
            lb: vec![-2.0, -3.0],
            ub: vec![2.0, 3.0],
        };
        let (red_n, _) = reduced_size(&prob);
        assert_eq!(red_n, 1, "coeffs ({a0}, {a1})");
        let sol = with_presolve(&prob);
        let bare = without_presolve(&prob);
        assert_eq!(sol.status, QpStatus::Optimal, "coeffs ({a0}, {a1})");
        assert_kkt(&prob, &sol, 1e-6);
        assert!(
            (sol.obj - bare.obj).abs() < 1e-6,
            "coeffs ({a0}, {a1}): {} vs bare {}",
            sol.obj,
            bare.obj
        );
    }
}

/// The substitution is a congruence `P' = MᵀPM`, so a convex QP stays
/// convex — worth a test rather than an assumption, since the whole
/// LP/QP dispatch upstream rests on the classification made *before* the
/// reduction runs. Checked where it matters: the reduced problem solves
/// to the same optimum, and its Hessian is PSD on a spanning set of
/// directions.
#[test]
fn convexity_survives_the_substitution() {
    // A genuinely coupled PSD Hessian: P = LᵀL with L = [[1,1,0],[0,1,1]],
    // then two alias rows fold x1 and x2 onto x0's cluster.
    let prob = QpProblem {
        n: 4,
        p_lower: vec![
            Triplet::new(0, 0, 1.0),
            Triplet::new(1, 0, 1.0),
            Triplet::new(1, 1, 2.0),
            Triplet::new(2, 1, 1.0),
            Triplet::new(2, 2, 1.0),
            Triplet::new(3, 3, 3.0),
        ],
        c: vec![1.0, -2.0, 0.5, 1.0],
        a: vec![
            Triplet::new(0, 0, 1.0),
            Triplet::new(0, 1, -2.0), // x0 = 2·x1
            Triplet::new(1, 2, 1.0),
            Triplet::new(1, 3, 1.5), // x2 = −1.5·x3
        ],
        b: vec![0.0, 0.0],
        g: vec![],
        h: vec![],
        lb: vec![],
        ub: vec![],
    };

    let ps = match presolve(&prob) {
        PresolveOutcome::Reduced(ps) => ps,
        _ => panic!("expected a reduction"),
    };
    assert_eq!(ps.reduced.n, 2);
    let red = &ps.reduced;
    // yᵀP'y ≥ 0 over a spanning sweep of directions.
    for k in 0..16 {
        let t = (k as f64) * std::f64::consts::PI / 8.0;
        let d = [t.cos(), t.sin()];
        let mut pd = vec![0.0; red.n];
        red.p_mul(&d, &mut pd);
        let quad: f64 = (0..red.n).map(|i| d[i] * pd[i]).sum();
        assert!(quad > -1e-12, "reduced Hessian indefinite: dᵀP'd = {quad}");
    }

    let sol = with_presolve(&prob);
    let bare = without_presolve(&prob);
    assert_eq!(sol.status, QpStatus::Optimal);
    assert_kkt(&prob, &sol, 1e-6);
    assert!(
        (sol.obj - bare.obj).abs() < 1e-6,
        "{} vs bare {}",
        sol.obj,
        bare.obj
    );
}

/// A QP whose Hessian couples an eliminated column to a *kept* one: the
/// coupling has to move into the survivor's linear term, and the
/// consumed row's multiplier has to be recovered against the full `Px`.
#[test]
fn hessian_coupling_across_the_substitution() {
    let prob = QpProblem {
        n: 4,
        p_lower: vec![
            Triplet::new(0, 0, 2.0),
            Triplet::new(2, 0, 1.0), // couples x0 (eliminated) to x2 (kept)
            Triplet::new(2, 2, 2.0),
            Triplet::new(1, 1, 2.0),
            Triplet::new(3, 3, 2.0),
        ],
        c: vec![1.0, 0.5, -1.0, 0.25],
        a: vec![
            Triplet::new(0, 0, 1.0),
            Triplet::new(0, 1, -3.0), // x0 = 3·x1 + 6
            // Three distinct clusters once the alias is folded, so this
            // row survives and the substitution has to rewrite it —
            // including moving x0's offset of 6 into its right-hand side.
            Triplet::new(1, 0, 1.0),
            Triplet::new(1, 2, 1.0),
            Triplet::new(1, 3, 1.0),
        ],
        b: vec![6.0, 2.0],
        g: vec![],
        h: vec![],
        lb: vec![],
        ub: vec![],
    };
    let (red_n, _) = reduced_size(&prob);
    assert_eq!(red_n, 3);
    let sol = with_presolve(&prob);
    let bare = without_presolve(&prob);
    assert_eq!(sol.status, QpStatus::Optimal);
    assert_kkt(&prob, &sol, 1e-6);
    assert!(
        (sol.obj - bare.obj).abs() < 1e-6,
        "{} vs bare {}",
        sol.obj,
        bare.obj
    );
    for i in 0..prob.n {
        assert!(
            (sol.x[i] - bare.x[i]).abs() < 1e-5,
            "x[{i}] {} vs bare {}",
            sol.x[i],
            bare.x[i]
        );
    }
}

/// An inequality row over eliminated columns is rewritten, not consumed,
/// and its multiplier comes back on the original row.
#[test]
fn inequalities_are_rewritten_and_keep_their_duals() {
    let prob = QpProblem {
        n: 3,
        p_lower: vec![Triplet::new(2, 2, 2.0)],
        c: vec![1.0, 1.0, 0.0],
        a: vec![Triplet::new(0, 0, 1.0), Triplet::new(0, 1, -1.0)], // x0 = x1
        b: vec![0.0],
        g: vec![
            Triplet::new(0, 0, -1.0),
            Triplet::new(0, 2, -1.0), // −x0 − x2 ≤ −2
        ],
        h: vec![-2.0],
        // Boxed, so `x1` is not a *free* column singleton and the row is
        // the aggregation's to consume rather than the older reduction's.
        lb: vec![-5.0, -5.0, f64::NEG_INFINITY],
        ub: vec![5.0, 5.0, f64::INFINITY],
    };
    let (red_n, red_rows) = reduced_size(&prob);
    assert_eq!(red_n, 2);
    assert_eq!(red_rows, 1, "the inequality survives");
    let sol = with_presolve(&prob);
    assert_eq!(sol.status, QpStatus::Optimal);
    assert_kkt(&prob, &sol, 1e-6);
    assert!(sol.z[0] > 1e-6, "the inequality is active: z = {:?}", sol.z);
}

/// Failing closed: a contradictory alias system is not called infeasible
/// by the elimination pass. Either the rest of the catalog reaches that
/// verdict on its own, or the model is handed to the solver whole — what
/// must not happen is the aggregation inventing the answer.
#[test]
fn contradictory_aliases_are_not_this_passs_verdict() {
    let prob = QpProblem {
        n: 3,
        p_lower: vec![Triplet::new(2, 2, 2.0)],
        c: vec![1.0, 1.0, 0.0],
        a: vec![
            Triplet::new(0, 0, 1.0),
            Triplet::new(0, 1, -1.0),
            Triplet::new(1, 0, 1.0),
            Triplet::new(1, 1, -1.0),
            Triplet::new(2, 2, 1.0),
        ],
        b: vec![0.0, 1.0, 3.0],
        g: vec![],
        h: vec![],
        lb: vec![],
        ub: vec![],
    };
    // The pass declines; the surviving verdict is whatever the rest of the
    // stack (or the solver) concludes, and either way it is a verdict about
    // a model that still has its contradictory rows in it.
    match presolve(&prob) {
        PresolveOutcome::Reduced(ps) => {
            // The alias rows must still be there for someone else to judge.
            assert!(ps.reduced.m_eq() >= 2, "{}", ps.reduced.m_eq());
        }
        PresolveOutcome::Infeasible(_) => {
            // Reached by the *duplicate-row* reduction, which is entitled
            // to it: two identical rows with different right-hand sides.
        }
        PresolveOutcome::Unbounded => panic!("not unbounded"),
    }
}

/// Rows that collapse to `0 = 0` under the accumulated substitutions are
/// redundant and go, without their multiplier being invented.
#[test]
fn redundant_rows_collapse() {
    let prob = QpProblem {
        n: 3,
        p_lower: (0..3).map(|i| Triplet::new(i, i, 2.0)).collect(),
        c: vec![-1.0, -1.0, -1.0],
        a: vec![
            Triplet::new(0, 0, 1.0),
            Triplet::new(0, 1, -1.0), // x0 = x1
            Triplet::new(1, 1, 1.0),
            Triplet::new(1, 2, -1.0), // x1 = x2
            // x0 − x2 = 0: implied by the two above, so `0 = 0` once
            // substituted.
            Triplet::new(2, 0, 1.0),
            Triplet::new(2, 2, -1.0),
        ],
        b: vec![0.0, 0.0, 0.0],
        g: vec![],
        h: vec![],
        lb: vec![],
        ub: vec![],
    };
    let (red_n, red_rows) = reduced_size(&prob);
    assert_eq!(red_n, 1);
    assert_eq!(red_rows, 0, "all three rows gone");
    let sol = with_presolve(&prob);
    assert_eq!(sol.status, QpStatus::Optimal);
    assert_kkt(&prob, &sol, 1e-6);
}

/// Randomized roundtrip over alias-linked QPs with boxes on both sides of
/// each link, which is where the bound transfer and the multiplier
/// re-attribution actually get exercised in combination.
#[test]
fn randomized_roundtrip() {
    // Deterministic LCG; the point is coverage of sign/box combinations,
    // not statistical rigour.
    let mut state: u64 = 0x5eed_1234_abcd_ef01;
    let mut next = move || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 33) as f64) / ((1u64 << 31) as f64) // [0, 1)
    };

    let mut aggregated = 0usize;
    let mut checked = 0usize;
    for case in 0..400 {
        let n = 4 + (case % 4);
        // Hessian: a positive diagonal plus an occasional off-diagonal
        // small enough to stay diagonally dominant, so the problem is
        // convex by construction and the congruence has something to bite
        // on.
        let mut p_lower: Vec<Triplet> = Vec::new();
        for i in 0..n {
            p_lower.push(Triplet::new(i, i, 2.0 + next()));
        }
        if next() < 0.5 && n >= 3 {
            p_lower.push(Triplet::new(2, 0, 0.5 * next()));
        }
        let c: Vec<f64> = (0..n).map(|_| 2.0 * next() - 1.0).collect();
        // One or two alias rows with coefficients and offsets of either
        // sign, plus — half the time — a wider equality row that should
        // survive and be rewritten.
        let mut a = Vec::new();
        let mut b = Vec::new();
        let links = 1 + (case % 2);
        for k in 0..links {
            let (i, j) = (2 * k, 2 * k + 1);
            if j >= n {
                break;
            }
            let r = b.len();
            let a0 = if next() < 0.5 { 1.0 } else { -2.0 };
            let a1 = if next() < 0.5 { 1.5 } else { -1.0 };
            a.push(Triplet::new(r, i, a0));
            a.push(Triplet::new(r, j, a1));
            b.push(2.0 * next() - 1.0);
        }
        if b.is_empty() {
            continue;
        }
        if next() < 0.5 && n >= 4 {
            let r = b.len();
            for j in [0usize, 2, 3] {
                a.push(Triplet::new(r, j, 1.0 + next()));
            }
            b.push(2.0 * next() - 1.0);
        }
        // An inequality over the same columns, so `Gᵀz` is in the gradient
        // the sweep has to work against.
        let (g, h) = if next() < 0.5 {
            (
                (0..n).map(|j| Triplet::new(0, j, next() - 0.5)).collect(),
                vec![1.0 + next()],
            )
        } else {
            (Vec::new(), Vec::new())
        };
        // Boxes: two-sided, one-sided either way, or absent — the four
        // shapes the bound transfer and the re-attribution have to cope
        // with.
        let mut lb = vec![0.0; n];
        let mut ub = vec![0.0; n];
        for j in 0..n {
            let (lo, hi) = (-1.0 - next(), next());
            match ((next() * 4.0) as usize).min(3) {
                0 => (lb[j], ub[j]) = (lo, hi),
                1 => (lb[j], ub[j]) = (lo, f64::INFINITY),
                2 => (lb[j], ub[j]) = (f64::NEG_INFINITY, hi),
                _ => (lb[j], ub[j]) = (f64::NEG_INFINITY, f64::INFINITY),
            }
        }
        let prob = QpProblem {
            n,
            p_lower,
            c,
            a,
            b,
            g,
            h,
            lb,
            ub,
        };
        if let PresolveOutcome::Reduced(ps) = presolve(&prob) {
            if ps.stats().aggregated_vars > 0 {
                aggregated += 1;
            }
        }
        let sol = with_presolve(&prob);
        if sol.status != QpStatus::Optimal {
            continue; // an empty box from the random draw; not this test's business
        }
        checked += 1;
        assert_kkt(&prob, &sol, 1e-5);
        let bare = without_presolve(&prob);
        if bare.status == QpStatus::Optimal {
            assert!(
                (sol.obj - bare.obj).abs() < 1e-5 * (1.0 + bare.obj.abs()),
                "case {case}: {} vs bare {}",
                sol.obj,
                bare.obj
            );
        }
    }
    // A probe that stopped generating aggregatable instances would pass
    // while testing nothing; hold it to a floor.
    assert!(aggregated > 200, "only {aggregated} instances aggregated");
    assert!(checked > 200, "only {checked} instances solved");
}
