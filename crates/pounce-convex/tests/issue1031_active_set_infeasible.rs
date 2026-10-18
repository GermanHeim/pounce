//! gh#1031 — the active-set engine never proved a node LP infeasible.
//!
//! Three fac2 OA-master nodes (the gh#1029 LPs with every free binary in
//! columns 54–65 fixed) are infeasible; HiGHS and the IPM agree, the IPM in 21
//! iterations. `method="active-set"` ended them `IterationLimit` after 2,422 and
//! 4,280 iterations, and the third ran to its 60 s time limit and reported zero
//! iterations.
//!
//! Two defects. The simplex phase-1 that seeds the engine *stopped* at its own
//! optimum with infeasibility left over — a Farkas certificate in hand — and
//! was read as "no seed", handing the problem to the engine's cold l1-elastic
//! phase-1, which does not terminate here. Phase-1's multipliers are now
//! checked as a certificate and, when they pass, are the answer. And
//! `pounce-qp`'s `time_limit_solution` hard-coded `n_working_set_changes: 0`,
//! dropping the iterations its caller had counted.
//!
//! The certificate is re-checked below with arithmetic written out here, not
//! through the crate's own checker, so a checker that passed everything would
//! not pass this test.

use std::time::Duration;

use pounce_convex::{
    ActiveSetOverrides, ActiveSetQp, HessianInertia, QpOptions, QpProblem, QpStatus, Triplet,
    engine_options, solve_qp_active_set, solve_qp_ipm,
};
use pounce_feral::FeralSolverInterface;
use pounce_linsol::SparseSymLinearSolverInterface;
use pounce_qp::{ParametricActiveSetSolver, QpSolver, QpStatus as EngineStatus};

fn backend() -> Box<dyn SparseSymLinearSolverInterface> {
    Box::new(FeralSolverInterface::new())
}

/// Parses a `tests/data/issue1029_node_lp_*.txt` file: `n`, the vectors `c
/// lb ub h b`, then one `G`/`A` triplet per line.
fn load(text: &str) -> QpProblem {
    let mut prob = QpProblem {
        n: 0,
        p_lower: vec![],
        c: vec![],
        a: vec![],
        b: vec![],
        g: vec![],
        h: vec![],
        lb: vec![],
        ub: vec![],
    };
    let nums = |it: std::str::SplitWhitespace| -> Vec<f64> {
        it.map(|t| t.parse().expect("float")).collect()
    };
    for line in text.lines().filter(|l| !l.starts_with('#')) {
        let mut it = line.split_whitespace();
        match it.next() {
            Some("n") => prob.n = it.next().unwrap().parse().unwrap(),
            Some("c") => prob.c = nums(it),
            Some("lb") => prob.lb = nums(it),
            Some("ub") => prob.ub = nums(it),
            Some("h") => prob.h = nums(it),
            Some("b") => prob.b = nums(it),
            Some(m @ ("G" | "A")) => {
                let row = it.next().unwrap().parse().unwrap();
                let col = it.next().unwrap().parse().unwrap();
                let val = it.next().unwrap().parse().unwrap();
                let t = Triplet::new(row, col, val);
                if m == "G" {
                    prob.g.push(t)
                } else {
                    prob.a.push(t)
                }
            }
            _ => {}
        }
    }
    prob
}

/// A gh#1029 LP with the binaries in columns 54..66 fixed to `bits`.
fn node(text: &str, bits: [u8; 12]) -> QpProblem {
    let mut prob = load(text);
    for (k, &b) in bits.iter().enumerate() {
        prob.lb[54 + k] = f64::from(b);
        prob.ub[54 + k] = f64::from(b);
    }
    prob
}

fn node_a() -> QpProblem {
    node(
        include_str!("data/issue1029_node_lp_m4.txt"),
        [1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1],
    )
}
fn node_b() -> QpProblem {
    node(
        include_str!("data/issue1029_node_lp_m3.txt"),
        [0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1],
    )
}
fn node_c() -> QpProblem {
    node(
        include_str!("data/issue1029_node_lp_m4.txt"),
        [1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1],
    )
}

/// `(y, z ≥ 0)` proves `{Ax = b, Gx ≤ h, lb ≤ x ≤ ub}` empty when
/// `min_box (Aᵀy + Gᵀz)ᵀx > bᵀy + hᵀz`: every feasible `x` has
/// `(Aᵀy + Gᵀz)ᵀx ≤ bᵀy + hᵀz`. Returns the margin, relative to the larger
/// side.
fn farkas_margin(prob: &QpProblem, y: &[f64], z: &[f64]) -> f64 {
    assert_eq!((y.len(), z.len()), (prob.b.len(), prob.h.len()));
    let zmax = z.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    assert!(
        z.iter().all(|&v| v >= -1e-12 * zmax),
        "z has a negative entry"
    );
    let mut q = vec![0.0; prob.n];
    for t in &prob.a {
        q[t.col] += t.val * y[t.row];
    }
    for t in &prob.g {
        q[t.col] += t.val * z[t.row];
    }
    let v: f64 = prob.b.iter().zip(y).map(|(b, y)| b * y).sum::<f64>()
        + prob.h.iter().zip(z).map(|(h, z)| h * z).sum::<f64>();
    // A free column's `qⱼ` is zero only up to round-off (`3.5e-23` on node A's
    // objective column against multipliers near 1); anything above a few ulps
    // of the largest multiplier would make `L = −∞` and prove nothing.
    let dual = y.iter().chain(z).fold(0.0_f64, |m, v| m.max(v.abs()));
    let mut l = 0.0;
    for (j, &qj) in q.iter().enumerate() {
        let bound = if qj > 0.0 { prob.lb[j] } else { prob.ub[j] };
        if bound.abs() < 1e19 {
            l += qj * bound;
        } else {
            assert!(
                qj.abs() <= 1e-14 * dual,
                "col {j}: q={qj:e} on an infinite bound"
            );
        }
    }
    (l - v) / l.abs().max(v.abs())
}

fn certified(name: &str, prob: &QpProblem) {
    // The IPM is the reference verdict on all three.
    let ipm = solve_qp_ipm(prob, &QpOptions::default(), backend);
    assert_eq!(ipm.status, QpStatus::PrimalInfeasible, "{name}: ipm");

    let opts = QpOptions {
        time_limit: Some(Duration::from_secs(60)),
        ..QpOptions::default()
    };
    let t = std::time::Instant::now();
    let sol = solve_qp_active_set(prob, &opts, &ActiveSetOverrides::default(), &mut backend);
    assert_eq!(
        sol.status,
        QpStatus::PrimalInfeasible,
        "{name}: active-set iters={} after {:?}",
        sol.iters,
        t.elapsed()
    );
    let margin = farkas_margin(prob, &sol.y, &sol.z);
    assert!(margin > 1e-6, "{name}: Farkas margin {margin:e}");
}

#[test]
fn node_a_is_certified_infeasible() {
    certified("A", &node_a());
}

#[test]
fn node_b_is_certified_infeasible() {
    certified("B", &node_b());
}

#[test]
fn node_c_is_certified_infeasible() {
    certified("C", &node_c());
}

/// The engine, cold and unseeded — the route the driver used to fall into —
/// stopped by its deadline still reports the iterations it ran.
#[test]
fn a_time_limited_engine_solve_reports_its_iterations() {
    let prob = node_c();
    let native = ActiveSetQp::from_convex(&prob);
    let mut qopts = engine_options(
        &QpOptions::default(),
        &ActiveSetOverrides::default(),
        native.n(),
        native.m(),
        HessianInertia::Psd,
    );
    qopts.use_homotopy = false;
    qopts.time_limit = Some(Duration::from_secs(2));
    let mut solver = ParametricActiveSetSolver::new(backend());
    let sol = solver
        .solve(&native.problem(), None, &qopts)
        .expect("soft outcome");
    assert_eq!(sol.status, EngineStatus::TimeLimit);
    assert!(
        sol.stats.n_working_set_changes > 0,
        "TimeLimit after {:?} with zero iterations reported",
        sol.stats.time
    );
}
