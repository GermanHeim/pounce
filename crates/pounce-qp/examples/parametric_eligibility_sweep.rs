//! Measurement harness for gh #602 — what does `solve_parametric` actually do
//! when the caller changes something its eligibility guard does not check?
//!
//! `solve_parametric_scoped` admits a warm parametric solve on two conditions:
//! the two problems have the same `(n, m)`, and the same `H`. The homotopy it
//! then runs interpolates only `g` and the **row** bounds `(bl, bu)`. Three
//! things are therefore free to differ between `qp_prev` and `qp_new` without
//! the path modelling the difference at all: the constraint matrix `A`, the
//! variable bounds `(xl, xu)`, and the `hessian_inertia` declaration.
//!
//! #602 asks whether those should be added to the guard. That is a *cost*
//! question, not a correctness one (the path is a predictor and the corrector
//! re-solves against the true problem), and #434 established that no guard in
//! this area gets chosen without per-problem data. This example is the
//! instrument for the synthetic half of that data; the real half is the
//! Maros-Mészáros sweep in `pounce-convex/examples/homotopy_sweep.rs`.
//!
//! Three routes are timed on the *same* target QP, so the columns are directly
//! comparable:
//!
//! * `homotopy` — `solve_parametric`, i.e. what ships today.
//! * `cold`     — `solve(qp_new, None)`, i.e. what a stricter guard would fall
//!                back to.
//! * `ws-only`  — `solve_with_working_set(qp_new, sol_prev.working)`, i.e. the
//!                route the SQP driver already takes, and the fallback that is
//!                available to a stricter guard but is not what #602 proposes.
//!
//! Run:
//!
//! ```text
//! cargo run -p pounce-qp --example parametric_eligibility_sweep
//! POUNCE_HOMOTOPY_DEBUG=1 cargo run -p pounce-qp --example parametric_eligibility_sweep
//! ```
//!
//! With `POUNCE_HOMOTOPY_DEBUG` set, each row is preceded by three `[hom]`
//! summary lines (previous solve, warm path, cold solve); the middle one's
//! "handoff x has max target violation" is how far off-manifold the warm path
//! ended, which is the quantity that grows with the unmodelled change.

use pounce_common::types::{NLP_LOWER_BOUND_INF, NLP_UPPER_BOUND_INF};
use pounce_linalg::triplet::{GenTMatrix, GenTMatrixSpace, SymTMatrix, SymTMatrixSpace};
use pounce_qp::{
    HessianInertia, ParametricActiveSetSolver, QpOptions, QpProblem, QpSolution, QpSolver, QpStatus,
};
use std::rc::Rc;

const N: usize = 30;
const M: usize = 20;

fn backend() -> Box<pounce_feral::FeralSolverInterface> {
    Box::new(pounce_feral::FeralSolverInterface::new())
}

/// Deterministic pseudo-random in `[-1, 1]`, so the whole sweep is reproducible
/// without pulling in an RNG dependency.
fn pr(k: usize) -> f64 {
    let s = ((k as u64)
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407))
        >> 33;
    ((s % 2000) as f64) / 1000.0 - 1.0
}

/// What moves between the previous problem and the new one.
///
/// `dh` is the one the guard *rejects* on; the rest it admits. Splitting them
/// out this way keeps the two questions #602 raises separable: what the guard
/// lets through (`da`, `xu_cap`, and the inertia argument to `row`), and what
/// it costs to be turned away (`dh`).
#[derive(Clone, Copy, Default)]
struct Change {
    /// Relative perturbation of the Hessian diagonal. Non-zero ⇒ `same_h` is
    /// false ⇒ the parametric guard declines and takes the fallback branch.
    dh: f64,
    /// Relative perturbation of every entry of `A`; structure fixed.
    da: f64,
    /// Shift applied to `g`.
    dg: f64,
    /// Shift applied to the row upper bounds.
    db: f64,
    /// Finite cap for the variable box, which is `+inf` above by default.
    xu_cap: Option<f64>,
}

/// One member of the parametric family.
struct Data {
    h: SymTMatrix,
    a: GenTMatrix,
    g: Vec<f64>,
    bl: Vec<f64>,
    bu: Vec<f64>,
    xl: Vec<f64>,
    xu: Vec<f64>,
}

fn data(c: Change) -> Data {
    let Change {
        dh,
        da,
        dg,
        db,
        xu_cap,
    } = c;

    // H = diag(1 + i/n): positive definite, so the box relaxation the cold arm
    // starts from is bounded and every route is comparable on the same footing.
    let (mut hi, mut hj, mut hv) = (Vec::new(), Vec::new(), Vec::new());
    for i in 0..N {
        hi.push((i + 1) as i32);
        hj.push((i + 1) as i32);
        hv.push((1.0 + (i as f64) / (N as f64)) * (1.0 + dh * pr(i + 907)));
    }
    let hs = SymTMatrixSpace::new(N as i32, hi, hj);
    let mut h = SymTMatrix::new(Rc::clone(&hs));
    h.set_values(&hv);

    // Four nonzeros per row, each scaled by its own `da`-sized perturbation —
    // the shape a relinearization produces, rather than a uniform rescale.
    let (mut ai, mut aj, mut av) = (Vec::new(), Vec::new(), Vec::new());
    for i in 0..M {
        for k in 0..4 {
            let col = (i * 7 + k * 5) % N;
            ai.push((i + 1) as i32);
            aj.push((col + 1) as i32);
            av.push((0.5 + pr(i * 13 + k).abs()) * (1.0 + da * pr(i * 31 + k + 7)));
        }
    }
    let asp = GenTMatrixSpace::new(M as i32, N as i32, ai, aj);
    let mut a = GenTMatrix::new(Rc::clone(&asp));
    a.set_values(&av);

    let g: Vec<f64> = (0..N)
        .map(|j| -2.0 - pr(j).abs() + dg * pr(j + 501))
        .collect();
    let bl = vec![NLP_LOWER_BOUND_INF; M];
    let bu: Vec<f64> = (0..M)
        .map(|i| 1.0 + 0.5 * pr(i + 101).abs() + db * pr(i + 601))
        .collect();
    let xu = match xu_cap {
        Some(c) => (0..N)
            .map(|j| if j % 3 == 0 { c } else { 10.0 * c })
            .collect(),
        None => vec![NLP_UPPER_BOUND_INF; N],
    };
    Data {
        h,
        a,
        g,
        bl,
        bu,
        xl: vec![0.0; N],
        xu,
    }
}

fn qp(d: &Data, inertia: HessianInertia) -> QpProblem<'_> {
    QpProblem {
        n: N,
        m: M,
        h: &d.h,
        g: &d.g,
        a: &d.a,
        bl: &d.bl,
        bu: &d.bu,
        xl: &d.xl,
        xu: &d.xu,
        hessian_inertia: inertia,
    }
}

fn cell(s: &QpSolution) -> String {
    format!(
        "{:?} chg={:>3} {:>6.1}ms",
        s.status,
        s.stats.n_working_set_changes,
        s.stats.time.as_secs_f64() * 1e3
    )
}

fn row(label: &str, prev: &Data, new: &Data, inertia_new: HessianInertia) {
    let opts = QpOptions {
        use_homotopy: true,
        ..QpOptions::default()
    };
    let mut s = ParametricActiveSetSolver::new(backend());
    let q_prev = qp(prev, HessianInertia::Psd);
    let sol_prev = s.solve(&q_prev, None, &opts).expect("previous solve");
    assert_eq!(
        sol_prev.status,
        QpStatus::Optimal,
        "{label}: previous solve"
    );

    let q_new = qp(new, inertia_new);
    let warm = s
        .solve_parametric(&q_prev, &sol_prev, &q_new, &opts)
        .expect("parametric solve");
    let cold = ParametricActiveSetSolver::new(backend())
        .solve(&q_new, None, &opts)
        .expect("cold solve");
    let ws = ParametricActiveSetSolver::new(backend())
        .solve_with_working_set(&q_new, &sol_prev.working, &opts)
        .expect("working-set solve");

    // The property that matters most: every route must land on the same answer.
    // A route being slow is a cost question; a route being wrong is not.
    let dobj = (warm.obj - cold.obj).abs();
    let dx = warm
        .x
        .iter()
        .zip(cold.x.iter())
        .fold(0.0_f64, |a, (u, v)| a.max((u - v).abs()));

    println!(
        "{label:<24} {:>24} {:>24} {:>24}  |dx|={dx:.1e} dobj={dobj:.1e}",
        cell(&warm),
        cell(&cold),
        cell(&ws)
    );
}

fn main() {
    /// The small `g` / `b` movement every row carries, so the differences
    /// between rows are attributable to the quantity named in the label.
    const BASE: Change = Change {
        dh: 0.0,
        da: 0.0,
        dg: 0.15,
        db: 0.2,
        xu_cap: None,
    };

    let prev = data(Change::default());
    println!(
        "{:<24} {:>24} {:>24} {:>24}",
        "change from prev", "homotopy", "cold", "ws-only"
    );

    println!("\n-- only the interpolated quantities move (guard admits; path models it) --");
    row("g+b small", &prev, &data(BASE), HessianInertia::Psd);
    row(
        "g+b large",
        &prev,
        &data(Change {
            dg: 1.50,
            db: 1.0,
            ..BASE
        }),
        HessianInertia::Psd,
    );

    println!("\n-- A moves too (guard admits; path does NOT model it) --");
    for da in [0.02, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.80, 1.00] {
        row(
            &format!("A{da:.2} + g+b small"),
            &prev,
            &data(Change { da, ..BASE }),
            HessianInertia::Psd,
        );
    }

    println!("\n-- variable bounds tighten (guard admits; path never moves or tests them) --");
    for c in [2.0, 0.5, 0.1] {
        row(
            &format!("xu={c} + g+b small"),
            &prev,
            &data(Change {
                xu_cap: Some(c),
                ..BASE
            }),
            HessianInertia::Psd,
        );
    }

    println!("\n-- inertia declaration changes, H bit-identical (guard admits) --");
    row(
        "inertia -> Indefinite",
        &prev,
        &data(BASE),
        HessianInertia::Indefinite,
    );
    row(
        "inertia -> Unknown",
        &prev,
        &data(BASE),
        HessianInertia::Unknown,
    );

    // The fallback branch. `H` differs, so `same_h` is false and
    // `solve_parametric` never reaches the homotopy at all — the `homotopy`
    // column below is really "whatever the ineligible branch does". Compare it
    // against `ws-only`, which is the working set the caller already handed us.
    println!("\n-- H moves: the guard REJECTS, so this measures the fallback --");
    for dh in [0.01, 0.05, 0.10, 0.25, 0.50] {
        row(
            &format!("H{dh:.2} + g+b small"),
            &prev,
            &data(Change { dh, ..BASE }),
            HessianInertia::Psd,
        );
        row(
            &format!("H{dh:.2} + A0.1 + g+b"),
            &prev,
            &data(Change {
                dh,
                da: 0.10,
                ..BASE
            }),
            HessianInertia::Psd,
        );
    }
}
