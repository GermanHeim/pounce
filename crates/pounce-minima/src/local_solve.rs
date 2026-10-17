//! The seam between the search and the local solver.
//!
//! [`find_minima`](crate::find_minima) never sees a solver application: it
//! hands each start to a [`LocalSolve`] and reads back a [`LocalOutcome`].
//! [`IpoptLocalSolve`] is the implementation over a reused
//! [`IpoptApplication`], which is what the CLI uses; a frontend with its own
//! solve path (the Python bindings) implements the trait over that instead.

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
    /// The converged primal, full length. Empty when `converged` is false.
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

/// On-converged capture slot: the lifted full-length primal `x` plus the
/// base-problem constraint duals `lambda` of the most recent solve.
type SolveCapture = Rc<RefCell<Option<(Vec<Number>, Vec<Number>)>>>;

/// [`LocalSolve`] over one reused [`IpoptApplication`] (no rebuild per start).
///
/// Each solve sets `hessian_approximation` for its [`HessianMode`] — the IPM
/// rereads the option every solve — and captures the converged point through
/// the application's on-converged hook. Construction installs that hook; any
/// callback the caller had registered is put back when this is dropped.
pub struct IpoptLocalSolve<'a> {
    app: &'a mut IpoptApplication,
    capture: SolveCapture,
    previous_hook: Option<ConvergedCallback>,
}

impl<'a> IpoptLocalSolve<'a> {
    pub fn new(app: &'a mut IpoptApplication) -> Self {
        // Capture the converged primal AND the base-problem constraint duals
        // of each solve via the on-converged hook. `lambda` uses the same
        // `finalize_solution_lambda` convention as the main NLP `.sol` path
        // (c/d split inversion + unscaling), so the `.sol` duals match a plain
        // solve. An nlp that does not expose user-facing duals returns an
        // empty vec; that (or any length mismatch) falls back to zeros where
        // the duals are stored.
        let capture: SolveCapture = Rc::new(RefCell::new(None));
        let previous_hook = app.take_on_converged();
        let cap = Rc::clone(&capture);
        // NOTE (gh#884 follow-up): `set_on_converged` fires once per
        // *attempt*. When a later attempt loses and an earlier one's answer
        // is replayed through the three-sink floor -- the mu fallback
        // (pounce#870, on by DEFAULT) or the gh#884 dual-divergence retry --
        // the converged KKT state this closure reads belongs to the
        // DISCARDED attempt, while the status, objective and statistics
        // reported alongside it are the winner's.
        // `IpoptApplication::answer_restored_from_floor()` reports that this
        // happened; the CLI's main path consults it and re-reads the point
        // from the `finalize_solution` payload. This site does not, because
        // what it needs is the factorization and the KKT state, which the
        // payload does not carry and which cannot be rewound. Pre-existing
        // and unfixed: a sensitivity result taken across a floored solve
        // describes the attempt that lost.
        app.set_on_converged(Box::new(move |data, _cq, nlp, _pd| {
            if let Some(curr) = data.borrow().curr.clone() {
                let nlp_ref = nlp.borrow();
                let x = nlp_ref.lift_x_to_full(&*curr.x);
                let lambda = nlp_ref.finalize_solution_lambda(&*curr.y_c, &*curr.y_d);
                *cap.borrow_mut() = Some((x, lambda));
            }
        }));
        Self {
            app,
            capture,
            previous_hook,
        }
    }
}

impl LocalSolve for IpoptLocalSolve<'_> {
    fn solve(&mut self, problem: Rc<RefCell<dyn TNLP>>, mode: HessianMode) -> LocalOutcome {
        let line = match mode {
            HessianMode::Exact => "hessian_approximation exact\n",
            HessianMode::QuasiNewton => "hessian_approximation limited-memory\n",
        };
        let _ = self.app.options_mut().read_from_str(line, true);
        *self.capture.borrow_mut() = None;
        let status = self.app.optimize_tnlp(problem);
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
        match self.capture.borrow_mut().take() {
            Some((x, lambda)) if succeeded => LocalOutcome {
                converged: true,
                x,
                lambda,
                quality,
            },
            _ => LocalOutcome {
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
        // Put back whatever the caller had registered; dropping our own
        // capture hook with it.
        let _ = self.app.take_on_converged();
        if let Some(cb) = self.previous_hook.take() {
            self.app.set_on_converged(cb);
        }
    }
}
