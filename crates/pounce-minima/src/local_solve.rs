//! The seam between the search and the local solver.
//!
//! [`find_minima`](crate::find_minima) never sees a solver application: it
//! hands each start to a [`LocalSolve`] and reads back a [`LocalOutcome`].
//! [`IpoptLocalSolve`] is the implementation over a reused
//! [`IpoptApplication`], which is what the CLI uses; a frontend with its own
//! solve path (the Python bindings) implements the trait over that instead.

use crate::capture_tnlp::{CaptureTnlp, FinalSlot};
use pounce_algorithm::application::{ConvergedCallback, IpoptApplication};
use pounce_common::types::Number;
use pounce_nlp::return_codes::ApplicationReturnStatus;
use pounce_nlp::tnlp::TNLP;
use std::cell::RefCell;
use std::rc::Rc;

/// Which Hessian a local solve should use. The search asks for
/// [`QuasiNewton`](Self::QuasiNewton) on the augmented penalty / tunnel
/// objectives, which decline an exact Hessian, and [`Exact`](Self::Exact) on
/// clean and polish solves.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HessianMode {
    Exact,
    QuasiNewton,
}

/// The solver's own unscaled solve-quality numbers, read by the `kkt_tol`
/// veto (gh#983). Each is `NaN` when the solve path does not compute it,
/// which the veto treats as "no evidence" rather than as a rejection.
#[derive(Clone, Copy, Debug)]
pub struct SolveQuality {
    /// Unscaled dual infeasibility (stationarity residual) at the returned
    /// point: `SolveStatistics::final_unscaled_dual_inf`.
    pub unscaled_dual_inf: Number,
    /// Magnitude of the terms that residual is assembled from:
    /// `SolveStatistics::final_unscaled_dual_scale`.
    pub unscaled_dual_scale: Number,
    /// `max |∇f|` at the solve's starting point:
    /// `SolveStatistics::start_obj_grad_max`.
    pub start_obj_grad_max: Number,
}

impl Default for SolveQuality {
    fn default() -> Self {
        Self {
            unscaled_dual_inf: Number::NAN,
            unscaled_dual_scale: Number::NAN,
            start_obj_grad_max: Number::NAN,
        }
    }
}

/// One local solve's result.
#[derive(Clone, Debug)]
pub struct LocalOutcome {
    /// The solve converged (`Solve_Succeeded` or `Solved_To_Acceptable_Level`)
    /// and produced a point. The search applies its own acceptance tests on
    /// top of this; it is not a verdict that the point is a minimum.
    pub converged: bool,
    /// The returned primal, full length — a candidate only when `converged`;
    /// otherwise whatever point the solve ended at, or empty if none.
    pub x: Vec<Number>,
    /// Constraint duals in the user's `∇f + λ·∇g + z = 0` convention, length
    /// `m` (empty if the solve path cannot report them).
    pub lambda: Vec<Number>,
    pub quality: SolveQuality,
}

/// Solves one problem from the start it carries.
///
/// `problem` is already seeded with the start (`get_starting_point` returns
/// it) and, for the repulsion strategies, already augmented. Calls are
/// sequential and the search counts them against its own budget.
pub trait LocalSolve {
    fn solve(&mut self, problem: Rc<RefCell<dyn TNLP>>, mode: HessianMode) -> LocalOutcome;
}

/// [`LocalSolve`] over one reused [`IpoptApplication`] (no rebuild per start).
///
/// Each solve sets `hessian_approximation` for its [`HessianMode`] — the IPM
/// rereads the option every solve — and reads the answer from the
/// `finalize_solution` payload, through a wrapper around the problem: the
/// reported `x` (after `honor_original_bounds`, when set) and the user-space
/// `lambda`, the same numbers a plain solve's `.sol` carries.
///
/// It used to read them from the application's on-converged hook instead,
/// which sees the algorithm's iterate: the un-projected `x`, so
/// `honor_original_bounds=yes` was ignored, and — when a later attempt lost
/// and an earlier answer was restored from the floor (the mu fallback,
/// pounce#870, on by default, or the gh#884 retry) — the *discarded*
/// attempt's point, while the status described the winner's. The floor
/// restore re-delivers the winner through `finalize_solution`, so the payload
/// is right in both cases.
///
/// Holding the application mutably for its lifetime also keeps the caller's
/// own on-converged callback, if any, out of the search's solves; it is put
/// back when this is dropped.
pub struct IpoptLocalSolve<'a> {
    app: &'a mut IpoptApplication,
    previous_hook: Option<ConvergedCallback>,
}

impl<'a> IpoptLocalSolve<'a> {
    pub fn new(app: &'a mut IpoptApplication) -> Self {
        let previous_hook = app.take_on_converged();
        Self { app, previous_hook }
    }
}

impl LocalSolve for IpoptLocalSolve<'_> {
    fn solve(&mut self, problem: Rc<RefCell<dyn TNLP>>, mode: HessianMode) -> LocalOutcome {
        let line = match mode {
            HessianMode::Exact => "hessian_approximation exact\n",
            HessianMode::QuasiNewton => "hessian_approximation limited-memory\n",
        };
        let _ = self.app.options_mut().read_from_str(line, true);
        let slot: FinalSlot = Rc::new(RefCell::new(None));
        let wrapped: Rc<RefCell<dyn TNLP>> =
            Rc::new(RefCell::new(CaptureTnlp::new(problem, Rc::clone(&slot))));
        let status = self.app.optimize_tnlp(wrapped);
        let succeeded = matches!(
            status,
            ApplicationReturnStatus::SolveSucceeded
                | ApplicationReturnStatus::SolvedToAcceptableLevel
        );
        let stats = self.app.statistics();
        let quality = SolveQuality {
            unscaled_dual_inf: stats.final_unscaled_dual_inf,
            unscaled_dual_scale: stats.final_unscaled_dual_scale,
            start_obj_grad_max: stats.start_obj_grad_max,
        };
        match slot.borrow_mut().take() {
            Some((x, lambda)) => LocalOutcome {
                converged: succeeded,
                x,
                lambda,
                quality,
            },
            None => LocalOutcome {
                converged: false,
                x: Vec::new(),
                lambda: Vec::new(),
                quality,
            },
        }
    }
}

impl Drop for IpoptLocalSolve<'_> {
    fn drop(&mut self) {
        // Put back whatever the caller had registered.
        if let Some(cb) = self.previous_hook.take() {
            self.app.set_on_converged(cb);
        }
    }
}
