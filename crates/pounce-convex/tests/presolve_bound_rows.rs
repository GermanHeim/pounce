//! Column-shaped reductions must still fire when the box arrived as *rows*.
//!
//! Presolve keys its column-shaped reductions — activity redundancy, forcing
//! rows, dominated columns, bound tightening — on `QpProblem::lb`/`ub`. A
//! caller that hands its bounds over as single-variable `G` rows instead
//! leaves that box empty, so those reductions would read `±∞` for every
//! variable and reach nothing (gh #500). Presolve now folds such a row back
//! into the box and drops it, so both entry paths reach the same reductions.
//!
//! The rest of the suite builds its boxes natively, which is the *other* side
//! of that seam — see the comment at `presolve_reductions.rs`'s activity-bound
//! section, "need the variable box". These tests come at it from the row side:
//! each one lowers a native box into `G` rows via `lower_box_to_rows` and
//! asserts the reduction still fires, the primal still matches, and the dual
//! is KKT-valid *for the row-form problem* — which is what would break if the
//! fold were ever reordered to run after the reductions that depend on it.

use pounce_convex::presolve::{PresolveOutcome, presolve, solve_with_presolve};
use pounce_convex::{QpOptions, QpProblem, QpSolution, QpStatus, Triplet, solve_qp_ipm};
use pounce_feral::FeralSolverInterface;
use pounce_linsol::SparseSymLinearSolverInterface;

fn backend() -> Box<dyn SparseSymLinearSolverInterface> {
    Box::new(FeralSolverInterface::new())
}

fn direct(prob: &QpProblem) -> QpSolution {
    solve_qp_ipm(prob, &QpOptions::default(), backend)
}

fn with_presolve(prob: &QpProblem) -> QpSolution {
    solve_with_presolve(prob, |r| solve_qp_ipm(r, &QpOptions::default(), backend))
}

/// Lower every finite box bound into a `G` row (`x ≤ ub`, `−x ≤ −lb`),
/// leaving the box empty — the shape a caller produces when it treats bounds
/// as ordinary constraints.
fn lower_box_to_rows(p: &QpProblem) -> QpProblem {
    let (mut g, mut h) = (p.g.clone(), p.h.clone());
    for i in 0..p.n {
        if p.ub_of(i).is_finite() {
            g.push(Triplet::new(h.len(), i, 1.0));
            h.push(p.ub_of(i));
        }
        if p.lb_of(i).is_finite() {
            g.push(Triplet::new(h.len(), i, -1.0));
            h.push(-p.lb_of(i));
        }
    }
    QpProblem {
        n: p.n,
        p_lower: p.p_lower.clone(),
        c: p.c.clone(),
        a: p.a.clone(),
        b: p.b.clone(),
        g,
        h,
        lb: Vec::new(),
        ub: Vec::new(),
    }
}

fn reduced(prob: &QpProblem) -> pounce_convex::presolve::Presolve {
    match presolve(prob) {
        PresolveOutcome::Reduced(ps) => ps,
        PresolveOutcome::Infeasible(_) => panic!("expected Reduced, got Infeasible"),
        PresolveOutcome::Unbounded => panic!("expected Reduced, got Unbounded"),
    }
}

/// KKT validity of `sol` for `prob`, to tolerance `tol` — including the
/// inequality multipliers' sign and complementarity, which is where a folded
/// bound row's dual has to land.
fn assert_kkt(prob: &QpProblem, sol: &QpSolution, tol: f64) {
    let n = prob.n;
    let mut g = prob.c.clone();
    prob.p_mul(&sol.x, &mut g);
    prob.at_mul(&sol.y, &mut g);
    prob.gt_mul(&sol.z, &mut g);
    for i in 0..n {
        let stat = g[i] + sol.z_ub[i] - sol.z_lb[i];
        assert!(stat.abs() < tol, "stationarity[{i}] = {stat}");
        assert!(
            sol.x[i] >= prob.lb_of(i) - tol && sol.x[i] <= prob.ub_of(i) + tol,
            "box [{i}]: {}",
            sol.x[i]
        );
    }
    let mut gx = vec![0.0; prob.m_ineq()];
    prob.g_mul(&sol.x, &mut gx);
    for i in 0..prob.m_ineq() {
        let slack = prob.h[i] - gx[i];
        assert!(slack > -tol, "Gx≤h row {i}: slack {slack}");
        assert!(sol.z[i] > -tol, "z[{i}] = {} < 0", sol.z[i]);
        assert!(
            (sol.z[i] * slack).abs() < 1e-4,
            "complementarity row {i}: z={} slack={slack}",
            sol.z[i]
        );
    }
    let mut ax = vec![0.0; prob.m_eq()];
    prob.a_mul(&sol.x, &mut ax);
    for (i, (&axi, &bi)) in ax.iter().zip(&prob.b).enumerate() {
        assert!((axi - bi).abs() < tol, "Ax=b row {i}: {axi} vs {bi}");
    }
}

// --- the two entry paths reach the same reductions ---

/// `x0` is absent from `P` and `A`, appears in one `≤` row with `+1`, and has
/// `c0 > 0` — a dominated column, optimal at its lower bound. With the bounds
/// as rows the sign test also sees the `−1` of the `−x0 ≤ 0` bound row, so
/// the reduction cannot fire even in principle until that row is folded away.
#[test]
fn dominated_column_fires_with_bounds_as_rows() {
    // min x0 − 2·x1 + x1²  s.t.  x0 + x1 ≤ 5,  x0, x1 ∈ [0, 3].
    let prob = QpProblem {
        n: 2,
        p_lower: vec![Triplet::new(1, 1, 2.0)],
        c: vec![1.0, -2.0],
        a: vec![],
        b: vec![],
        g: vec![Triplet::new(0, 0, 1.0), Triplet::new(0, 1, 1.0)],
        h: vec![5.0],
        lb: vec![0.0, 0.0],
        ub: vec![3.0, 3.0],
    };
    let rows = lower_box_to_rows(&prob);
    assert_eq!(reduced(&prob).stats().dominated_cols, 1);
    assert_eq!(reduced(&rows).stats().dominated_cols, 1, "gh #500");
    // Same model, so the same solution — through either entry path.
    let a = with_presolve(&prob);
    let b = with_presolve(&rows);
    assert_eq!(b.status, QpStatus::Optimal);
    for i in 0..prob.n {
        assert!(
            (a.x[i] - b.x[i]).abs() < 1e-5,
            "x[{i}]: {} vs {}",
            a.x[i],
            b.x[i]
        );
    }
    assert_kkt(&rows, &b, 1e-5);
}

/// A row whose activity range touches its right-hand side pins every variable
/// in it to a bound. That range is `(−∞, +∞)` without a box, so the reduction
/// is guarded off until the bound rows fold in.
#[test]
fn forcing_row_fires_with_bounds_as_rows() {
    // min x0+x1+x2 s.t. x0+x1 ≤ 0, x2 ≤ 5, x ∈ [0,1]³.
    // min-activity of row 0 is 0 = h ⇒ forcing: x0 = x1 = 0.
    let prob = QpProblem {
        n: 3,
        p_lower: vec![],
        c: vec![1.0, 1.0, 1.0],
        a: vec![],
        b: vec![],
        g: vec![
            Triplet::new(0, 0, 1.0),
            Triplet::new(0, 1, 1.0),
            Triplet::new(1, 2, 1.0),
        ],
        h: vec![0.0, 5.0],
        lb: vec![0.0, 0.0, 0.0],
        ub: vec![1.0, 1.0, 1.0],
    };
    let rows = lower_box_to_rows(&prob);
    assert_eq!(reduced(&prob).stats().forcing_rows, 1);
    assert_eq!(reduced(&rows).stats().forcing_rows, 1, "gh #500");
    let sol = with_presolve(&rows);
    assert_eq!(sol.status, QpStatus::Optimal);
    for (i, v) in sol.x.iter().enumerate() {
        assert!(v.abs() < 1e-5, "x[{i}] = {v}, all pinned to 0");
    }
    assert_kkt(&rows, &sol, 1e-5);
}

/// A row the box already implies is redundant should be dropped. Without a
/// box its activity range is unbounded and nothing fires.
#[test]
fn activity_redundant_row_dropped_with_bounds_as_rows() {
    // min x0²+x1²−x0−x1 s.t. x0+x1 ≤ 100, x ∈ [0,3]² — the row cannot bind.
    let prob = QpProblem {
        n: 2,
        p_lower: vec![Triplet::new(0, 0, 2.0), Triplet::new(1, 1, 2.0)],
        c: vec![-1.0, -1.0],
        a: vec![],
        b: vec![],
        g: vec![Triplet::new(0, 0, 1.0), Triplet::new(0, 1, 1.0)],
        h: vec![100.0],
        lb: vec![0.0, 0.0],
        ub: vec![3.0, 3.0],
    };
    let rows = lower_box_to_rows(&prob);
    assert_eq!(reduced(&prob).reduced.m_ineq(), 0);
    assert_eq!(reduced(&rows).reduced.m_ineq(), 0, "gh #500");
    let sol = with_presolve(&rows);
    assert_eq!(sol.status, QpStatus::Optimal);
    assert!(
        sol.z.iter().all(|z| z.abs() < 1e-6),
        "no row binds: {:?}",
        sol.z
    );
    assert_kkt(&rows, &sol, 1e-5);
}

// --- randomized KKT roundtrips over row-form models ---

/// Tiny deterministic LCG, so the sweep is reproducible.
struct Rng(u64);
impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0
    }
    fn unif(&mut self, lo: f64, hi: f64) -> f64 {
        let u = (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
        lo + (hi - lo) * u
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
}

/// Many random boxed QPs, each solved twice — once with its native box and
/// once with the box lowered to `G` rows. The row-form solve must reach the
/// same primal and produce a KKT-valid dual *of the row-form problem*, with
/// every bound's multiplier on its row. This is the only randomized presolve
/// roundtrip in the suite; the hand-built cases above pin named reductions,
/// this one covers the combinations nobody thought to write down.
#[test]
fn randomized_row_form_roundtrip() {
    let mut rng = Rng(0x5006_0500);
    let mut solved = 0;
    for case in 0..200 {
        let n = 2 + rng.below(3); // 2..4 variables
        let m = 1 + rng.below(3); // 1..3 general rows
        // A convex diagonal Hessian keeps the optimum unique (so the two
        // solves' primals must agree), with some purely linear columns so the
        // dominated-column reduction has something to bite on.
        let mut p_lower: Vec<Triplet> = Vec::new();
        for i in 0..n {
            if rng.unif(0.0, 1.0) < 0.7 {
                p_lower.push(Triplet::new(i, i, rng.unif(0.5, 3.0)));
            }
        }
        let c: Vec<f64> = (0..n).map(|_| rng.unif(-4.0, 4.0)).collect();
        let mut g = Vec::new();
        let mut h = Vec::new();
        for row in 0..m {
            for col in 0..n {
                if rng.unif(0.0, 1.0) < 0.6 {
                    g.push(Triplet::new(row, col, rng.unif(-2.0, 2.0)));
                }
            }
            h.push(rng.unif(-1.0, 6.0));
        }
        // Sometimes an equality row, and sometimes a *singleton* one — which
        // fixes a variable and can leave a general row with a single live
        // column, i.e. a bound row born mid-presolve.
        let mut a = Vec::new();
        let mut b = Vec::new();
        let lb: Vec<f64> = (0..n).map(|_| rng.unif(-3.0, 0.0)).collect();
        let ub: Vec<f64> = (0..n).map(|i| lb[i] + rng.unif(0.5, 5.0)).collect();
        if rng.unif(0.0, 1.0) < 0.5 {
            let col = rng.below(n);
            if rng.unif(0.0, 1.0) < 0.5 {
                // x_col = v, inside its box so the draw stays feasible.
                a.push(Triplet::new(0, col, 1.0));
                b.push(rng.unif(lb[col], ub[col]));
            } else {
                let other = (col + 1) % n;
                a.push(Triplet::new(0, col, 1.0));
                a.push(Triplet::new(0, other, rng.unif(-2.0, 2.0)));
                b.push(rng.unif(-1.0, 1.0));
            }
        }
        let boxed = QpProblem {
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
        let rows = lower_box_to_rows(&boxed);

        let d = direct(&rows);
        if d.status != QpStatus::Optimal {
            continue; // infeasible / unbounded draw — not this test's subject
        }
        solved += 1;
        let sol = with_presolve(&rows);
        assert_eq!(sol.status, QpStatus::Optimal, "case {case}");
        assert_kkt(&rows, &sol, 1e-4);
        for i in 0..n {
            assert!(
                (sol.x[i] - d.x[i]).abs() < 1e-4,
                "case {case} x[{i}]: {} vs direct {}",
                sol.x[i],
                d.x[i]
            );
        }
        // The reductions the box unlocks must not change the objective.
        assert!(
            (sol.obj - d.obj).abs() < 1e-4,
            "case {case} obj: {} vs {}",
            sol.obj,
            d.obj
        );
    }
    // The draw is random but seeded: 174 of the 200 land Optimal today. Assert
    // a floor so a future change to the generator (or to what counts as
    // feasible) cannot quietly turn this sweep into a no-op.
    assert!(
        solved > 150,
        "only {solved}/200 draws exercised the roundtrip"
    );
}
