//! Warm starting the convex-QP IPM across a sequence of nearby problems.
//!
//! A common pattern (parametric / receding-horizon / training-loop
//! solving) is to solve a sequence of QPs that differ only slightly. Each
//! solve's solution is a good warm start for the next. This example
//! solves a path of perturbed problems cold vs. warm and prints the
//! per-solve iteration counts and the total.
//!
//! Run: `cargo run -p pounce-convex --example warm_start`

use pounce_convex::{
    solve_qp_ipm, solve_qp_ipm_warm, QpOptions, QpProblem, QpWarmStart, Triplet,
};
use pounce_feral::FeralSolverInterface;
use pounce_linsol::SparseSymLinearSolverInterface;

fn backend() -> Box<dyn SparseSymLinearSolverInterface> {
    Box::new(FeralSolverInterface::new())
}

/// `min ½·2‖x‖² + cᵀx s.t. Σx ≤ cap` (P = 2I, one capacity row).
fn capped_qp(c: &[f64], cap: f64) -> QpProblem {
    let n = c.len();
    QpProblem {
        n,
        p_lower: (0..n).map(|i| Triplet::new(i, i, 2.0)).collect(),
        c: c.to_vec(),
        a: vec![],
        b: vec![],
        g: (0..n).map(|i| Triplet::new(0, i, 1.0)).collect(),
        h: vec![cap],
        lb: vec![],
        ub: vec![],
    }
}

fn main() {
    let opts = QpOptions::default();
    let n = 40;
    let base_c: Vec<f64> = (0..n).map(|i| -1.0 - (i as f64) * 0.05).collect();

    // A path of 8 problems, each a small perturbation of the previous.
    let steps = 8;
    let mut cold_total = 0usize;
    let mut warm_total = 0usize;

    // Seed the warm path with the first cold solve.
    let mut prev = solve_qp_ipm(&capped_qp(&base_c, 5.0), &opts, backend);

    println!("{:<6} {:>10} {:>10}", "step", "cold_iters", "warm_iters");
    for k in 0..steps {
        let scale = 1.0 + 0.02 * (k as f64 + 1.0);
        let c: Vec<f64> = base_c.iter().map(|v| v * scale).collect();
        let cap = 5.0 + 0.1 * (k as f64 + 1.0);
        let prob = capped_qp(&c, cap);

        let cold = solve_qp_ipm(&prob, &opts, backend);
        let warm = solve_qp_ipm_warm(&prob, &opts, &QpWarmStart::from_solution(&prev), backend);

        println!("{:<6} {:>10} {:>10}", k, cold.iters, warm.iters);
        cold_total += cold.iters;
        warm_total += warm.iters;
        prev = warm; // chain: next warm start is this solution
    }

    println!(
        "\ntotal iters: cold={cold_total} warm={warm_total} \
         ({:.0}% fewer with warm start)",
        100.0 * (cold_total as f64 - warm_total as f64) / cold_total as f64
    );
}
