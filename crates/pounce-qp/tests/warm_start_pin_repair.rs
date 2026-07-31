//! #428 — a warm start must survive the active set *moving*.
//!
//! `solve_with_working_set` pins the hinted active rows to their new boundary
//! values and hands the resulting primal to `solve`. Once the true active set
//! has drifted, that pinned point holds a row which should have been released,
//! so it overshoots some *other* row — and `solve`'s warm-start admission
//! pre-check then threw the entire hint away for a cold l1-elastic phase-1
//! whose recovery re-solve starts from `WorkingSet::cold`. A hint wrong by one
//! entry cost exactly as much as one wrong by hundreds: on a parametric MPC
//! sweep, roughly one working-set change per constraint row, which past
//! `m > max_iter` stops producing an answer at all.
//!
//! The fix repairs the hint instead of discarding it — the violated rows are
//! known, so they are pinned too and the |A| − 1 entries the hint got right
//! are kept. Nothing about the pre-check's tolerance changes; it is simply
//! handed a feasible point. `stats.used_phase1` is the tell these tests key
//! on: `true` means the hint was discarded and phase-1 rebuilt the answer from
//! scratch, which is the bug.

use pounce_linalg::triplet::{GenTMatrix, GenTMatrixSpace, SymTMatrix, SymTMatrixSpace};
use pounce_qp::working_set::{BoundStatus, ConsStatus, WorkingSet};
use pounce_qp::{
    HessianInertia, ParametricActiveSetSolver, QpOptions, QpProblem, QpSolver, QpStatus,
};
use std::rc::Rc;

const NEG_INF: f64 = -1e20;
const POS_INF: f64 = 1e20;

fn solver() -> ParametricActiveSetSolver {
    ParametricActiveSetSolver::new(Box::new(pounce_feral::FeralSolverInterface::new()))
}

// ---------------------------------------------------------------------------
// Analytic case: the hint is wrong by exactly one entry — the case the old
// all-or-nothing recovery handled worst, and the common one in a sweep.
//
//   min ½‖x − (2, 0.1)‖²   s.t.   r₀: x₁ + x₂ ≤ 1,   r₁: x₁ ≤ 0.6
//
// Optimum: x* = (0.6, 0.1), with r₁ active and r₀ slack (0.7 < 1).
// Hint: r₀ active, r₁ inactive — one entry wrong in each direction.
// Pinning r₀ alone gives (1.45, −0.45), which violates r₁ by 0.85, so the
// pre-check rejected it and the whole solve fell through to phase-1.
// ---------------------------------------------------------------------------
#[test]
fn hint_wrong_by_one_entry_is_repaired_not_discarded() {
    let h_space = SymTMatrixSpace::new(2, vec![1, 2], vec![1, 2]);
    let mut h = SymTMatrix::new(Rc::clone(&h_space));
    h.set_values(&[1.0, 1.0]);

    let a_space = GenTMatrixSpace::new(2, 2, vec![1, 1, 2], vec![1, 2, 1]);
    let mut a = GenTMatrix::new(Rc::clone(&a_space));
    a.set_values(&[1.0, 1.0, 1.0]);

    let g = [-2.0, -0.1];
    let bl = [NEG_INF, NEG_INF];
    let bu = [1.0, 0.6];
    let xl = [NEG_INF, NEG_INF];
    let xu = [POS_INF, POS_INF];

    let qp = QpProblem {
        n: 2,
        m: 2,
        h: &h,
        g: &g,
        a: &a,
        bl: &bl,
        bu: &bu,
        xl: &xl,
        xu: &xu,
        hessian_inertia: HessianInertia::Psd,
    };

    let working = WorkingSet {
        constraints: vec![ConsStatus::AtUpper, ConsStatus::Inactive],
        bounds: vec![BoundStatus::Inactive; 2],
    };

    let mut s = solver();
    let sol = s
        .solve_with_working_set(&qp, &working, &QpOptions::default())
        .expect("warm solve");

    assert_eq!(sol.status, QpStatus::Optimal, "status = {:?}", sol.status);
    assert!((sol.x[0] - 0.6).abs() < 1e-9, "x[0] = {}", sol.x[0]);
    assert!((sol.x[1] - 0.1).abs() < 1e-9, "x[1] = {}", sol.x[1]);
    assert!(
        !sol.stats.used_phase1,
        "a hint wrong by one entry must be repaired, not thrown away for an \
         l1-elastic phase-1"
    );
    assert!(
        sol.stats.n_working_set_changes <= 2,
        "repaired hint should need a pivot or two to release r₀, took {}",
        sol.stats.n_working_set_changes
    );
}

// ---------------------------------------------------------------------------
// Parametric MPC — the shape the issue was measured on.
//
// Linear-quadratic double integrator over `horizon` steps: variables
// `[p₀ v₀ u₀ | p₁ v₁ u₁ | … | p_N v_N]` (n = 3N+2), initial condition plus
// dynamics as equality rows (m = 2N+2), controls box-bounded. The parameter is
// the angle θ of the initial state on a circle; stepping θ changes which
// controls saturate, i.e. moves the active set by a few entries.
// ---------------------------------------------------------------------------

const DT: f64 = 0.1;
const UMAX: f64 = 0.5;
const RADIUS: f64 = 1.0;

struct Mpc {
    n: usize,
    m: usize,
    h: SymTMatrix,
    g: Vec<f64>,
    a: GenTMatrix,
    bl: Vec<f64>,
    bu: Vec<f64>,
    xl: Vec<f64>,
    xu: Vec<f64>,
}

impl Mpc {
    fn qp(&self) -> QpProblem<'_> {
        QpProblem {
            n: self.n,
            m: self.m,
            h: &self.h,
            g: &self.g,
            a: &self.a,
            bl: &self.bl,
            bu: &self.bu,
            xl: &self.xl,
            xu: &self.xu,
            hessian_inertia: HessianInertia::Psd,
        }
    }
}

fn mpc(horizon: usize, theta: f64) -> Mpc {
    let n = 3 * horizon + 2;
    let m = 2 * horizon + 2;
    let p = |k: usize| 3 * k;
    let v = |k: usize| 3 * k + 1;
    let u = |k: usize| 3 * k + 2;

    // H: diagonal state / control weights.
    let mut hi: Vec<i32> = Vec::new();
    let mut hj: Vec<i32> = Vec::new();
    let mut hv: Vec<f64> = Vec::new();
    let diag = |idx: usize, w: f64, hi: &mut Vec<i32>, hj: &mut Vec<i32>, hv: &mut Vec<f64>| {
        hi.push(idx as i32 + 1);
        hj.push(idx as i32 + 1);
        hv.push(w);
    };
    for k in 0..=horizon {
        diag(p(k), 1.0, &mut hi, &mut hj, &mut hv);
        diag(v(k), 1.0, &mut hi, &mut hj, &mut hv);
        if k < horizon {
            diag(u(k), 0.05, &mut hi, &mut hj, &mut hv);
        }
    }
    let h_space = SymTMatrixSpace::new(n as i32, hi, hj);
    let mut h = SymTMatrix::new(Rc::clone(&h_space));
    h.set_values(&hv);

    // A: x₀ = x_init(θ), then x_{k+1} = A_d x_k + B_d u_k.
    let mut ai: Vec<i32> = Vec::new();
    let mut aj: Vec<i32> = Vec::new();
    let mut av: Vec<f64> = Vec::new();
    let push =
        |r: usize, c: usize, val: f64, ai: &mut Vec<i32>, aj: &mut Vec<i32>, av: &mut Vec<f64>| {
            ai.push(r as i32 + 1);
            aj.push(c as i32 + 1);
            av.push(val);
        };
    push(0, p(0), 1.0, &mut ai, &mut aj, &mut av);
    push(1, v(0), 1.0, &mut ai, &mut aj, &mut av);
    for k in 0..horizon {
        let rp = 2 + 2 * k;
        let rv = 3 + 2 * k;
        push(rp, p(k + 1), 1.0, &mut ai, &mut aj, &mut av);
        push(rp, p(k), -1.0, &mut ai, &mut aj, &mut av);
        push(rp, v(k), -DT, &mut ai, &mut aj, &mut av);
        push(rp, u(k), -0.5 * DT * DT, &mut ai, &mut aj, &mut av);
        push(rv, v(k + 1), 1.0, &mut ai, &mut aj, &mut av);
        push(rv, v(k), -1.0, &mut ai, &mut aj, &mut av);
        push(rv, u(k), -DT, &mut ai, &mut aj, &mut av);
    }
    let a_space = GenTMatrixSpace::new(m as i32, n as i32, ai, aj);
    let mut a = GenTMatrix::new(Rc::clone(&a_space));
    a.set_values(&av);

    let mut bl = vec![0.0; m];
    let mut bu = vec![0.0; m];
    bl[0] = RADIUS * theta.cos();
    bu[0] = bl[0];
    bl[1] = RADIUS * theta.sin();
    bu[1] = bl[1];

    let mut xl = vec![NEG_INF; n];
    let mut xu = vec![POS_INF; n];
    for k in 0..horizon {
        xl[u(k)] = -UMAX;
        xu[u(k)] = UMAX;
    }

    Mpc {
        n,
        m,
        h,
        g: vec![0.0; n],
        a,
        bl,
        bu,
        xl,
        xu,
    }
}

/// A hint carried one parameter step must cost far less than solving cold — at
/// every horizon, so the cost cannot be growing with `m`.
///
/// Before the repair, `warm` matched `cold` pivot for pivot at every horizon
/// here (10/15/19/20 against 10/15/19/20): the hint was discarded whole and
/// phase-1 rebuilt the answer. The repaired path lands on the same optimum to
/// ~1e-13 in 0 pivots — and, unlike raising `feas_tol` until the pre-check
/// admits the hint, does not trade that against a looser acceptance test.
#[test]
fn parameter_step_keeps_the_warm_start_at_every_horizon() {
    let opts = QpOptions {
        max_iter: 20_000,
        ..QpOptions::default()
    };
    for horizon in [10usize, 20, 40, 100] {
        let base = mpc(horizon, 0.3);
        let next = mpc(horizon, 0.35);

        let mut s = solver();
        let hint = s.solve(&base.qp(), None, &opts).expect("cold base solve");
        assert_eq!(hint.status, QpStatus::Optimal);

        let mut sc = solver();
        let cold = sc.solve(&next.qp(), None, &opts).expect("cold solve");
        assert_eq!(cold.status, QpStatus::Optimal);

        let mut sw = solver();
        let warm = sw
            .solve_with_working_set(&next.qp(), &hint.working, &opts)
            .expect("warm solve");

        assert_eq!(warm.status, QpStatus::Optimal, "N = {horizon}");
        assert!(
            !warm.stats.used_phase1,
            "N = {horizon}: a one-step-old hint was discarded for phase-1"
        );
        assert!(
            warm.stats.n_working_set_changes + 4 < cold.stats.n_working_set_changes,
            "N = {horizon}: warm took {} pivots against cold's {} — the hint bought nothing",
            warm.stats.n_working_set_changes,
            cold.stats.n_working_set_changes,
        );
        assert!(
            (warm.obj - cold.obj).abs() < 1e-9,
            "N = {horizon}: warm obj {} != cold obj {}",
            warm.obj,
            cold.obj,
        );
    }
}

/// The whole sweep, at the **default** iteration budget: every step must reach
/// the cold solve's optimum for no more pivots than cold spends. The repair is
/// allowed to decline a hint — it does, on the larger steps here, where too
/// much has moved — but declining has to land back on the old elastic
/// recovery and its answer, never on a worse one.
#[test]
fn a_full_parameter_sweep_matches_cold_at_the_default_budget() {
    let horizon = 40;
    let reference = QpOptions {
        max_iter: 20_000,
        ..QpOptions::default()
    };
    let base = mpc(horizon, 0.3);
    let mut s = solver();
    let hint = s
        .solve(&base.qp(), None, &reference)
        .expect("cold base solve");

    for step in [-0.2f64, -0.05, -0.01, 0.01, 0.05, 0.2, 0.7] {
        let next = mpc(horizon, 0.3 + step);

        let mut sc = solver();
        let cold = sc.solve(&next.qp(), None, &reference).expect("cold solve");
        assert_eq!(cold.status, QpStatus::Optimal, "step {step}");

        let mut sw = solver();
        let warm = sw
            .solve_with_working_set(&next.qp(), &hint.working, &QpOptions::default())
            .expect("warm solve");

        assert_eq!(warm.status, QpStatus::Optimal, "step {step}");
        assert!(
            (warm.obj - cold.obj).abs() < 1e-9,
            "step {step}: warm obj {} != cold obj {}",
            warm.obj,
            cold.obj,
        );
        assert!(
            warm.stats.n_working_set_changes <= cold.stats.n_working_set_changes,
            "step {step}: warm took {} pivots against cold's {}",
            warm.stats.n_working_set_changes,
            cold.stats.n_working_set_changes,
        );
    }
}

/// A hint that is badly wrong is what the admission pre-check was written for
/// (a degenerate crossover vertex violating hundreds of inactive rows), and it
/// must keep going to l1-elastic phase-1. The repair's budget scales with how
/// much of the hint is violated, so it declines here: one row hinted active,
/// twelve rows violated by the point pinning it produces.
///
///   min ½‖x‖² − ½·Σ_{i<12} x_i   over x ∈ R¹⁰⁰
///   s.t.  r₀: Σ_{i<12} x_i ≤ 24,   rᵢ₊₁: x_i ≤ 1  (i < 12)
///
/// The optimum is interior — x_i = 0.5 for i < 12, 0 elsewhere, no row active.
/// The hint claims r₀ is active at 24, which pins x_i = 2 and overshoots all
/// twelve single-variable rows at once.
#[test]
fn a_badly_wrong_hint_still_falls_through_to_elastic() {
    const N: usize = 100;
    const K: usize = 12;
    let m = K + 1;

    let h_space =
        SymTMatrixSpace::new(N as i32, (1..=N as i32).collect(), (1..=N as i32).collect());
    let mut h = SymTMatrix::new(Rc::clone(&h_space));
    h.set_values(&vec![1.0; N]);

    // Row 0: Σ_{i<K} x_i ≤ 24. Rows 1..=K: x_i ≤ 1.
    let mut ai: Vec<i32> = Vec::new();
    let mut aj: Vec<i32> = Vec::new();
    for i in 0..K {
        ai.push(1);
        aj.push(i as i32 + 1);
    }
    for i in 0..K {
        ai.push(i as i32 + 2);
        aj.push(i as i32 + 1);
    }
    let a_space = GenTMatrixSpace::new(m as i32, N as i32, ai, aj);
    let mut a = GenTMatrix::new(Rc::clone(&a_space));
    a.set_values(&vec![1.0; 2 * K]);

    let mut g = vec![0.0; N];
    for gi in g.iter_mut().take(K) {
        *gi = -0.5;
    }
    let bl = vec![NEG_INF; m];
    let mut bu = vec![1.0; m];
    bu[0] = 24.0;
    let xl = vec![NEG_INF; N];
    let xu = vec![POS_INF; N];

    let qp = QpProblem {
        n: N,
        m,
        h: &h,
        g: &g,
        a: &a,
        bl: &bl,
        bu: &bu,
        xl: &xl,
        xu: &xu,
        hessian_inertia: HessianInertia::Psd,
    };

    let mut working = WorkingSet::cold(N, m);
    working.constraints[0] = ConsStatus::AtUpper;

    let mut s = solver();
    let sol = s
        .solve_with_working_set(&qp, &working, &QpOptions::default())
        .expect("warm solve from a hopeless hint");

    assert_eq!(sol.status, QpStatus::Optimal, "status = {:?}", sol.status);
    assert!(
        sol.stats.used_phase1,
        "a hint this wrong must still be declined and recovered through \
         l1-elastic phase-1"
    );
    for (i, &xi) in sol.x.iter().enumerate() {
        let want = if i < K { 0.5 } else { 0.0 };
        assert!((xi - want).abs() < 1e-8, "x[{i}] = {xi}, want {want}");
    }
}
