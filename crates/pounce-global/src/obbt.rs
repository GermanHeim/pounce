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
pub(crate) fn tighten<F>(
    prob: &GlobalProblem,
    lo: &mut [f64],
    hi: &mut [f64],
    cutoff: Option<f64>,
    passes: usize,
    opts: &QpOptions,
    make_backend: &mut F,
) -> bool
where
    F: FnMut() -> Box<dyn SparseSymLinearSolverInterface>,
{
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

        let mut improved = false;
        for i in 0..n {
            // Solve over the (fixed, pass-start) relaxation — valid since it is
            // an outer approximation regardless of in-pass bound updates.
            qp.c.iter_mut().for_each(|c| *c = 0.0);
            qp.c[i] = 1.0; // minimize x_i
            let s = solve_qp_ipm(&qp, opts, &mut *make_backend);
            if s.status == QpStatus::Optimal && s.obj > lo[i] + 1e-9 {
                lo[i] = s.obj.min(hi[i]);
                improved = true;
            }
            qp.c[i] = -1.0; // maximize x_i (minimize −x_i)
            let s = solve_qp_ipm(&qp, opts, &mut *make_backend);
            if s.status == QpStatus::Optimal && -s.obj < hi[i] - 1e-9 {
                hi[i] = (-s.obj).max(lo[i]);
                improved = true;
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
