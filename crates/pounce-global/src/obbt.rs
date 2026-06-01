//! Optimization-based bound tightening (OBBT).
//!
//! FBBT tightens a box by interval propagation through each constraint; OBBT
//! tightens it by *optimizing*: minimize and maximize each variable over the
//! whole relaxation (the same polyhedral outer approximation used for the
//! lower bound), optionally with an incumbent-cutoff row `objective ≤ ub`. The
//! relaxation contains every truly feasible point, so its min/max of `x_i` are
//! valid new bounds — usually much tighter than FBBT's, at the cost of `2n` LP
//! solves per pass. This is the single biggest box-reducer in commercial
//! global solvers; it is gated by frequency in the driver because of that cost.

use crate::problem::GlobalProblem;
use crate::relax::build_relaxation;
use pounce_convex::{solve_qp_ipm, QpOptions, QpStatus, Triplet};
use pounce_linsol::SparseSymLinearSolverInterface;

/// Tighten `[lo, hi]` in place by OBBT. `cutoff`, when set, adds the row
/// `objective ≤ cutoff` (the incumbent), which lets OBBT exploit that no
/// improving point exceeds the incumbent. Returns `false` if the relaxation is
/// infeasible over the box (the node can then be pruned).
/// `parallel` runs each pass's `2n` min/max solves on a thread pool. They are
/// independent (all use the same pass-start relaxation), so the result is
/// identical to the serial sweep — only faster.
#[allow(clippy::too_many_arguments)]
pub(crate) fn tighten<F>(
    prob: &GlobalProblem,
    lo: &mut [f64],
    hi: &mut [f64],
    cutoff: Option<f64>,
    passes: usize,
    parallel: bool,
    opts: &QpOptions,
    make_backend: &F,
) -> bool
where
    F: Fn() -> Box<dyn SparseSymLinearSolverInterface> + Sync,
{
    use rayon::prelude::*;

    let n = prob.n_vars;
    for _ in 0..passes {
        let relax = build_relaxation(prob, lo, hi, true);
        if relax.trivially_infeasible {
            return false;
        }
        let mut qp = relax.qp;
        if let (Some(cut), Some(oc)) = (cutoff, relax.obj_col) {
            let row = qp.h.len();
            qp.g.push(Triplet::new(row, oc, 1.0));
            qp.h.push(cut);
        }

        // Minimize and maximize x_i over the (fixed, pass-start) relaxation.
        // `(min x_i, max x_i)` for variable `i`, each `None` if not optimal.
        let solve_var = |i: usize| -> (Option<f64>, Option<f64>) {
            let mut mk = || make_backend();
            let mut q = qp.clone();
            // Optimize x_i *alone* — the cloned relaxation carries the objective
            // cost, which must be zeroed first.
            q.c.iter_mut().for_each(|c| *c = 0.0);
            q.c[i] = 1.0;
            let smin = solve_qp_ipm(&q, opts, &mut mk);
            let min = (smin.status == QpStatus::Optimal).then_some(smin.obj);
            q.c[i] = -1.0;
            let smax = solve_qp_ipm(&q, opts, &mut mk);
            let max = (smax.status == QpStatus::Optimal).then_some(-smax.obj);
            (min, max)
        };
        let results: Vec<(Option<f64>, Option<f64>)> = if parallel {
            (0..n).into_par_iter().map(solve_var).collect()
        } else {
            (0..n).map(solve_var).collect()
        };

        let mut improved = false;
        for (i, (min, max)) in results.into_iter().enumerate() {
            if let Some(v) = min {
                if v > lo[i] + 1e-9 {
                    lo[i] = v.min(hi[i]);
                    improved = true;
                }
            }
            if let Some(v) = max {
                if v < hi[i] - 1e-9 {
                    hi[i] = v.max(lo[i]);
                    improved = true;
                }
            }
            if lo[i] > hi[i] + 1e-9 {
                return false; // box collapsed → infeasible
            }
        }
        if !improved {
            break;
        }
    }
    true
}
