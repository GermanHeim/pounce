//! `Solver` — value-typed session API that holds an `IpoptApplication`,
//! its TNLP, and the converged KKT factor between calls.
//!
//! This is Phase 3a of the factor-reuse work tracked in
//! [pounce#16](https://github.com/jkitchin/pounce/issues/16). It is
//! the public surface for callers who want to:
//!
//! 1. Run a normal IPM solve, then
//! 2. Issue many cheap operations against the converged factor
//!    (`kkt_solve`, `parametric_step`) without going through the
//!    [`set_on_converged`] callback shape that [`crate::SensSolve`]
//!    requires.
//!
//! [`set_on_converged`]: pounce_algorithm::IpoptApplication::set_on_converged
//!
//! # Usage
//!
//! ```ignore
//! use pounce_sensitivity::Solver;
//! use std::cell::RefCell;
//! use std::rc::Rc;
//!
//! let app = make_configured_app();
//! let tnlp: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(MyTnlp));
//! let mut solver = Solver::new(app, tnlp);
//!
//! let status = solver.solve();
//! assert!(solver.converged().is_some());
//!
//! // Issue any number of back-solves against the same factor:
//! let dim = solver.kkt_dim().unwrap();
//! let mut lhs = vec![0.0; dim];
//! let rhs = vec![1.0; dim];
//! solver.kkt_solve(&rhs, &mut lhs).unwrap();
//!
//! // Parametric step with respect to a set of pinned equality
//! // constraints (same interpretation as [`crate::SensSolve`]):
//! let dx = solver.parametric_step(&[2, 3], &[-0.5, 0.0]).unwrap();
//! ```
//!
//! # Scope of Phase 3a
//!
//! - **In**: `solve()`, `converged()`, `kkt_solve()`, `parametric_step()`,
//!   `block_dims()` / `kkt_dim()`.
//! - **Deferred to Phase 3b**: `resolve()` (warm-start that reuses the
//!   linear backend pool), `compute_reduced_hessian()` on the Solver
//!   (currently only available through [`crate::SensSolve`]), and the
//!   `parametric_mpc` / `sensitivity_session` example binaries.

use std::cell::{Ref, RefCell};
use std::rc::Rc;

use pounce_algorithm::application::IpoptApplication;
use pounce_common::types::{Index, Number};
use pounce_nlp::TNLP;
use pounce_nlp::return_codes::ApplicationReturnStatus;

use crate::PdSensBacksolver;
use crate::activity::ActivityReport;
use crate::backsolver::SensBacksolver;
use crate::schur_data::IndexSchurData;
use crate::sens_app::{SensApplication, SensOptions};
use crate::vec_util::dense_to_vec;

/// Sign of the barrier correction term, set from a comparison
/// against sIPOPT rather than derived.
const BARRIER_SIGN: Number = -1.0;

/// Errors returned by post-convergence operations on [`Solver`].
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum SolverError {
    /// The solver has not yet converged, or the last solve failed
    /// before producing a usable KKT factor.
    NotConverged,
    /// An input slice's length did not match the KKT dimension or the
    /// parameter count.
    BadShape {
        /// Human description of the mismatched buffer.
        what: &'static str,
        /// Length the caller passed.
        got: usize,
        /// Length expected.
        expected: usize,
    },
    /// The underlying back-solve failed (singular factor, numerical
    /// breakdown).
    BacksolveFailed,
    /// The underlying [`SensApplication`] step failed (e.g. row mapping
    /// invalid for the current problem).
    SensComputationFailed(String),
    /// An option the requested computation depends on holds an
    /// incompatible value; the message names the option and the value
    /// required.
    BadOptions(String),
}

/// State captured at convergence: the user-visible iterate plus the
/// `PdSensBacksolver` that wraps the converged KKT factor.
///
/// Read this via [`Solver::converged`].
pub struct ConvergedState {
    /// IPM return status of the most recent solve.
    pub status: ApplicationReturnStatus,
    /// Final primal iterate `x*` (length `n_x`), in the user's own
    /// units: a `user-scaling` change of variables is undone here, so
    /// this is `x`, never the algorithm's `x̃ = d ⊙ x` (gh#486).
    pub x: Vec<Number>,
    /// Final objective value `f(x*)`.
    pub obj_val: Number,
    /// `bound_relax_factor` **as the solve that produced this state
    /// ran with it**, not as the application's options read today.
    /// The bounds were relaxed (or not) once, during this solve; a
    /// later `set_numeric_value` cannot change what the held slacks
    /// were measured against, so post-solve calls whose validity
    /// depends on unrelaxed bounds must guard on this value. See
    /// [`Solver::classify_activity`].
    pub bound_relax_factor: Number,
    /// Converged KKT-factor wrapper. Owns `Rc` handles to the
    /// `PdFullSpaceSolver`, the IpoptData / Cq, and the NLP, so it
    /// outlives the IPM call frame.
    backsolver: PdSensBacksolver,
}

impl ConvergedState {
    /// Block dimensions of the compound KKT vector in
    /// `(x, s, y_c, y_d, z_l, z_u, v_l, v_u)` order.
    pub fn block_dims(&self) -> [usize; 8] {
        self.backsolver.block_dims()
    }

    /// Total dimension of the compound KKT vector (sum of `block_dims`).
    pub fn kkt_dim(&self) -> usize {
        self.backsolver.dim()
    }
}

/// Session-style solver: holds an [`IpoptApplication`], its TNLP, and
/// the converged factor between calls.
pub struct Solver {
    app: IpoptApplication,
    tnlp: Rc<RefCell<dyn TNLP>>,
    /// Side channel populated by the `on_converged` callback installed
    /// in [`Self::solve`]. The `RefCell<Option<…>>` shape mirrors the
    /// pattern in [`crate::convenience`] (the callback closure needs
    /// shared mutable access; the `Option` is `None` before the first
    /// solve and gets overwritten on each call).
    state: Rc<RefCell<Option<ConvergedState>>>,
}

impl Solver {
    /// Build a new session. The `app` should already have its options
    /// configured and `initialize()` called.
    pub fn new(app: IpoptApplication, tnlp: Rc<RefCell<dyn TNLP>>) -> Self {
        Self {
            app,
            tnlp,
            state: Rc::new(RefCell::new(None)),
        }
    }

    /// Borrow the underlying `IpoptApplication` (e.g. to read its
    /// options table after a solve). Mutation between `solve` calls is
    /// supported via [`Self::app_mut`].
    pub fn app(&self) -> &IpoptApplication {
        &self.app
    }

    /// Mutable borrow of the underlying `IpoptApplication`. Useful for
    /// reconfiguring options before a follow-up `solve()`. Note that
    /// changing options that affect the KKT linear system between
    /// calls will invalidate the cached factor; the next `solve()`
    /// rebuilds it.
    pub fn app_mut(&mut self) -> &mut IpoptApplication {
        &mut self.app
    }

    /// Run the IPM to convergence. On a successful solve the
    /// [`ConvergedState`] (including the KKT backsolver) is stashed
    /// inside the `Solver` and accessible via [`Self::converged`].
    ///
    /// Each call to `solve()` overwrites the previous converged
    /// state; the previously held factor is dropped.
    pub fn solve(&mut self) -> ApplicationReturnStatus {
        // Clear any previous state so a failed re-solve doesn't leave
        // a stale factor visible.
        self.state.borrow_mut().take();

        // Snapshot the options this solve will run under, before it
        // runs. `bound_relax_factor` is consumed once, when the NLP
        // relaxes its bounds; reading it back at query time would
        // describe the application's options rather than the state
        // being queried. The registry supplies its own default when
        // the option is unset, so no second copy of the default lives
        // here.
        let brf = self
            .app
            .options()
            .get_numeric_value("bound_relax_factor", "")
            .map(|(v, _)| v)
            .expect("bound_relax_factor is a registered core option");

        let state_cb = Rc::clone(&self.state);
        self.app
            .set_on_converged(Box::new(move |data, cq, nlp, pd| {
                let curr = match data.borrow().curr.clone() {
                    Some(c) => c,
                    None => return,
                };
                let backsolver = match PdSensBacksolver::new(data, cq, nlp, Rc::clone(&pd)) {
                    Ok(b) => b,
                    Err(e) => {
                        // No session state is stored, so post-solve
                        // calls will report NotConverged; at least say
                        // why on stderr rather than failing silently.
                        eprintln!("pounce: Solver could not capture the KKT factor: {e}");
                        return;
                    }
                };
                // The algorithm's iterate is `x̃ = d ⊙ x` when the
                // solve ran under a change of variables (gh#486): this
                // capture reads the iterate, not the
                // `finalize_solution` payload, so it undoes the
                // substitution itself. The backsolver already read the
                // factors off the NLP, in this same var-x space.
                let mut x = dense_to_vec(&*curr.x);
                if let Some(d) = backsolver.variable_scaling() {
                    debug_assert_eq!(x.len(), d.len());
                    for (xi, &di) in x.iter_mut().zip(d.iter()) {
                        *xi /= di;
                    }
                }
                let obj_val = cq.borrow_mut().curr_f();
                // Status is overwritten with the real value after
                // optimize_tnlp returns.
                *state_cb.borrow_mut() = Some(ConvergedState {
                    status: ApplicationReturnStatus::InternalError,
                    x,
                    obj_val,
                    bound_relax_factor: brf,
                    backsolver,
                });
            }));

        let status = crate::optimize_tnlp_for_sensitivity(&mut self.app, Rc::clone(&self.tnlp));
        if let Some(s) = self.state.borrow_mut().as_mut() {
            s.status = status;
        }
        status
    }

    /// Borrow the converged state, if a successful solve has been
    /// run. Returns `None` if no solve has run or if the most recent
    /// solve failed before reaching convergence.
    pub fn converged(&self) -> Option<Ref<'_, ConvergedState>> {
        let r = self.state.borrow();
        r.as_ref()?;
        Some(Ref::map(r, |o| {
            o.as_ref()
                .unwrap_or_else(|| unreachable!("checked is_some above"))
        }))
    }

    /// Total dimension of the compound KKT vector (sum of
    /// `block_dims`). Returns `None` if no converged factor is held.
    pub fn kkt_dim(&self) -> Option<usize> {
        self.converged().map(|c| c.kkt_dim())
    }

    /// Block dimensions of the compound KKT vector in
    /// `(x, s, y_c, y_d, z_l, z_u, v_l, v_u)` order. Returns `None` if
    /// no converged factor is held.
    pub fn block_dims(&self) -> Option<[usize; 8]> {
        self.converged().map(|c| c.block_dims())
    }

    /// Classify every bounded variable and every finite-bounded
    /// inequality row of the converged solve by activity: see
    /// [`crate::activity`] and
    /// `dev-notes/covariance-information-roadmap.md` item 0 (gh #362).
    ///
    /// Requires the held solve to have run with `bound_relax_factor=0`
    /// (the Ipopt default is `1e-8`): with relaxed bounds the solver's
    /// slacks are measured against perturbed bounds, and the
    /// complementarity products the classifier reads no longer track
    /// `μ`.
    ///
    /// The guard reads
    /// [`ConvergedState::bound_relax_factor`] — the value that solve
    /// ran under — not the application's current options. Setting the
    /// option after the fact neither unlocks a state whose bounds were
    /// relaxed nor invalidates one whose bounds were not; re-solve to
    /// change the answer.
    pub fn classify_activity(&self) -> Result<ActivityReport, SolverError> {
        let state = self.state.borrow();
        let state = state.as_ref().ok_or(SolverError::NotConverged)?;
        let brf = state.bound_relax_factor;
        if brf != 0.0 {
            return Err(SolverError::BadOptions(format!(
                "classify_activity requires bound_relax_factor=0, but the \
                 held solve ran with {brf:e}: relaxed bounds shift the \
                 slacks the classifier reads. Set the option and solve() \
                 again — changing it now does not re-measure the slacks."
            )));
        }
        Ok(crate::activity::compute(&state.backsolver))
    }

    /// The gradient of user constraint row `user_row` at the converged
    /// iterate, in user variable order (length `n_full_x`) and in
    /// **natural (unscaled) units**: the internal Jacobian row carries
    /// the solver's per-row `c_scale`/`d_scale`, which is divided out
    /// here, so this is the gradient of the row as the user wrote it.
    /// Equality and inequality rows alike; entries for fixed
    /// (`make_parameter`-removed) variables are 0 because the solve
    /// dropped their columns. Errors on an out-of-range row.
    ///
    /// Serves the covariance roadmap's item 1: a binding row's normal
    /// restricted to the fitted block is the projection direction.
    pub fn row_normal(&self, user_row: usize) -> Result<Vec<Number>, SolverError> {
        let state = self.state.borrow();
        let state = state.as_ref().ok_or(SolverError::NotConverged)?;
        crate::activity::row_normal(&state.backsolver, user_row).map_err(|m| {
            SolverError::BadShape {
                what: "row_normal constraint index",
                got: user_row,
                expected: m,
            }
        })
    }

    /// The exact Lagrangian Hessian times a user-space vector, in
    /// user variable order and natural units (see
    /// [`crate::activity::hessian_vec`]). Errors on a length mismatch.
    pub fn hessian_vec(&self, v: &[Number]) -> Result<Vec<Number>, SolverError> {
        let state = self.state.borrow();
        let state = state.as_ref().ok_or(SolverError::NotConverged)?;
        crate::activity::hessian_vec(&state.backsolver, v).map_err(|n| SolverError::BadShape {
            what: "hessian_vec vector length",
            got: v.len(),
            expected: n,
        })
    }

    /// Solve `K · lhs = rhs` against the converged KKT factor. Both
    /// slices must have length `kkt_dim()`; the layout is the flat
    /// `x || s || y_c || y_d || z_l || z_u || v_l || v_u` packing.
    ///
    /// `K` here is the **natural-units** (unscaled) KKT matrix: when
    /// the IPM solved with active NLP scaling, the backsolver scales
    /// the RHS/solution (all eight blocks, including the z/v
    /// bound-multiplier rows) so callers pass and receive data in the
    /// user's own units (pounce#128) — see
    /// [`crate::PdSensBacksolver::solve`]. For the raw scaled-space
    /// back-solve use [`Self::kkt_solve_scaled`].
    pub fn kkt_solve(&self, rhs: &[Number], lhs: &mut [Number]) -> Result<(), SolverError> {
        self.kkt_solve_impl(rhs, lhs, false)
    }

    /// [`Self::kkt_solve`] without the natural-units conjugation: the
    /// back-solve runs against the factor exactly as the IPM holds it
    /// (the solver's internal scaled space). Identical to `kkt_solve`
    /// when no NLP scaling is active. "Scaled space" includes a
    /// `user-scaling` change of variables (gh#486), so on such a solve
    /// the `x` and `z` blocks here are in the substituted coordinates
    /// `x̃ = d ⊙ x`, not the model's.
    pub fn kkt_solve_scaled(&self, rhs: &[Number], lhs: &mut [Number]) -> Result<(), SolverError> {
        self.kkt_solve_impl(rhs, lhs, true)
    }

    fn kkt_solve_impl(
        &self,
        rhs: &[Number],
        lhs: &mut [Number],
        scaled: bool,
    ) -> Result<(), SolverError> {
        let state = self.state.borrow();
        let state = state.as_ref().ok_or(SolverError::NotConverged)?;
        let total = state.backsolver.dim();
        if rhs.len() != total {
            return Err(SolverError::BadShape {
                what: "rhs",
                got: rhs.len(),
                expected: total,
            });
        }
        if lhs.len() != total {
            return Err(SolverError::BadShape {
                what: "lhs",
                got: lhs.len(),
                expected: total,
            });
        }
        let ok = if scaled {
            state.backsolver.solve_scaled_space(rhs, lhs)
        } else {
            state.backsolver.solve(rhs, lhs)
        };
        if ok {
            Ok(())
        } else {
            Err(SolverError::BacksolveFailed)
        }
    }

    /// Batched-RHS back-solve. `rhs_flat` and `lhs_flat` are row-major
    /// `(n_rhs, kkt_dim)` buffers; each row is solved against the
    /// same converged factor. Equivalent in result to looping
    /// [`Self::kkt_solve`] but reuses one `IteratesVector` for the
    /// RHS and one for the result across all `n_rhs` calls — see
    /// [`crate::algorithm_backsolver::PdSensBacksolver::solve_many`].
    pub fn kkt_solve_many(
        &self,
        rhs_flat: &[Number],
        lhs_flat: &mut [Number],
        n_rhs: usize,
    ) -> Result<(), SolverError> {
        self.kkt_solve_many_impl(rhs_flat, lhs_flat, n_rhs, false)
    }

    /// [`Self::kkt_solve_many`] without the natural-units
    /// conjugation (the batched sibling of [`Self::kkt_solve_scaled`]).
    pub fn kkt_solve_many_scaled(
        &self,
        rhs_flat: &[Number],
        lhs_flat: &mut [Number],
        n_rhs: usize,
    ) -> Result<(), SolverError> {
        self.kkt_solve_many_impl(rhs_flat, lhs_flat, n_rhs, true)
    }

    fn kkt_solve_many_impl(
        &self,
        rhs_flat: &[Number],
        lhs_flat: &mut [Number],
        n_rhs: usize,
        scaled: bool,
    ) -> Result<(), SolverError> {
        let state = self.state.borrow();
        let state = state.as_ref().ok_or(SolverError::NotConverged)?;
        let total = state.backsolver.dim();
        let expected = n_rhs * total;
        if rhs_flat.len() != expected {
            return Err(SolverError::BadShape {
                what: "rhs",
                got: rhs_flat.len(),
                expected,
            });
        }
        if lhs_flat.len() != expected {
            return Err(SolverError::BadShape {
                what: "lhs",
                got: lhs_flat.len(),
                expected,
            });
        }
        let ok = if scaled {
            state
                .backsolver
                .solve_many_scaled_space(rhs_flat, lhs_flat, n_rhs)
        } else {
            state.backsolver.solve_many(rhs_flat, lhs_flat, n_rhs)
        };
        if ok {
            Ok(())
        } else {
            Err(SolverError::BacksolveFailed)
        }
    }

    /// First-order parametric step `Δx ≈ ∂x*/∂p · Δp` for a set of
    /// pinned equality constraints. `pin_constraint_indices` are
    /// 0-based indices into the user's `g(x)`; `deltas` is the
    /// perturbation `Δp` (same length).
    ///
    /// Returns the `n_x`-long primal step. For the full KKT-space
    /// step, use [`Self::kkt_solve`] directly.
    pub fn parametric_step(
        &self,
        pin_constraint_indices: &[Index],
        deltas: &[Number],
    ) -> Result<Vec<Number>, SolverError> {
        if pin_constraint_indices.len() != deltas.len() {
            return Err(SolverError::BadShape {
                what: "deltas",
                got: deltas.len(),
                expected: pin_constraint_indices.len(),
            });
        }
        let state = self.state.borrow();
        let state = state.as_ref().ok_or(SolverError::NotConverged)?;

        // Map user g-indices to y_c rows through the NLP's c/d-split
        // permutation (pounce#128; matches `convenience.rs`).
        let dims = state.backsolver.block_dims();
        let n_x = dims[0];
        let param_rows = state
            .backsolver
            .map_pin_g_to_kkt_rows(pin_constraint_indices)
            .map_err(SolverError::SensComputationFailed)?;
        let signs = vec![1; pin_constraint_indices.len()];
        let a_data = IndexSchurData::from_parts(param_rows, signs)
            .map_err(|e| SolverError::SensComputationFailed(format!("{e:?}")))?;

        let opts = SensOptions {
            run_sens: true,
            ..SensOptions::default()
        };
        let sens_app = SensApplication::new(a_data, state.backsolver.clone(), opts);
        let n_full = state.backsolver.dim();
        let mut dx_full = vec![0.0; n_full];
        if !sens_app.parametric_step(deltas, &mut dx_full) {
            return Err(SolverError::SensComputationFailed(
                "SensApplication::parametric_step failed".into(),
            ));
        }
        // carry the step from the barrier problem's solution toward the
        // original problem's (the paper's equation 11)
        let corr = self.barrier_correction(state)?;
        for (d, c) in dx_full.iter_mut().zip(corr.iter()) {
            *d += *c * BARRIER_SIGN;
        }
        dx_full.truncate(n_x);
        Ok(dx_full)
        // NOTE: parametric_step_full below applies the same correction,
        // so the two agree on their shared block.
    }

    /// The barrier correction of the parametric step: the paper's
    /// equation 11 term, which carries the step from the solution of
    /// the barrier problem at `mu > 0` toward the one at `mu = 0`.
    ///
    /// [`Self::parametric_step`] is taken against a factorization held
    /// at the final `mu`, so it estimates where the BARRIER problem's
    /// solution moves, not where the original problem's does. The two
    /// differ by `O(mu)`, which is negligible at a tight tolerance and
    /// is not at a loose one. Measured against sIPOPT on a nonlinear
    /// model, the uncorrected step agrees to 2e-9 at `tol = 1e-8` and
    /// differs by 9e-6 at `tol = 1e-3`.
    ///
    /// The term is one more backsolve against the same factor, with
    /// `mu` in the complementarity rows, which are the bound multiplier
    /// blocks of the compound vector.
    ///
    /// Returns the correction over the whole compound vector, to be
    /// added to the step.
    fn barrier_correction(&self, state: &ConvergedState) -> Result<Vec<Number>, SolverError> {
        let dims = state.backsolver.block_dims();
        let n_full = state.backsolver.dim();
        let mu = {
            let (data, _, _) = state.backsolver.activity_handles();
            let d = data.borrow();
            d.curr_mu
        };
        // z_l, z_u, v_l, v_u: the rows carrying the complementarity
        // conditions, which are the ones the barrier perturbs
        let start = dims[0] + dims[1] + dims[2] + dims[3];
        let end = start + dims[4] + dims[5] + dims[6] + dims[7];
        let mut rhs = vec![0.0; n_full];
        for r in rhs.iter_mut().take(end).skip(start) {
            *r = mu;
        }
        let mut corr = vec![0.0; n_full];
        if !state.backsolver.solve(&rhs, &mut corr) {
            return Err(SolverError::BacksolveFailed);
        }
        Ok(corr)
    }

    /// Parametric step with the bounds respected by pinning, not by
    /// clamping. Returns the `n_x`-long primal step and the variables
    /// pinned to reach it.
    ///
    /// [`Self::parametric_step`] answers where the linear predictor
    /// points, which can be outside the box. Clamping a coordinate
    /// back to its bound leaves every other coordinate at its
    /// predictor value, so the answer is feasible but no longer
    /// consistent with the KKT relations. This instead adds a row
    /// pinning the offending coordinate at the bound and re-solves, so
    /// the others move to stay consistent under the pin, which is the
    /// refinement upstream runs under `sens_boundcheck`.
    ///
    /// Each pass augments the held factorization with the pin rows and
    /// takes the Schur complement over them. The factorization itself
    /// is never rebuilt, but the Schur complement is rebuilt each pass,
    /// so pass `k` costs one dense `k × k` solve and `k + 1`
    /// back-solves and the total grows quadratically in the number of
    /// pins. The default `max_passes` of 16 is 136 back-solves.
    ///
    /// What counts as outside a bound is taken from the solve rather
    /// than from the caller: it was willing to leave a converged point
    /// `bound_relax_factor` outside its bound, so anything within that
    /// is on the bound. An unrelaxed solve gets a roundoff floor.
    ///
    /// Passes stop when nothing is outside its bound by that much, at
    /// `max_passes`, or when a pin cannot be achieved because the pins
    /// have exhausted the problem's degrees of freedom. None of those
    /// is an error: the step returned is the last one computed, and the
    /// returned pin list says how far the refinement got. `max_passes`
    /// is a budget, since each pass costs a dense `k × k` solve and the
    /// point of the refinement is to stay cheaper than a re-solve.
    pub fn parametric_step_bounded(
        &self,
        pin_constraint_indices: &[Index],
        deltas: &[Number],
        max_passes: usize,
    ) -> Result<(Vec<Number>, Vec<Index>), SolverError> {
        let dx_full = self.parametric_step_full(pin_constraint_indices, deltas)?;
        let state = self.state.borrow();
        let state = state.as_ref().ok_or(SolverError::NotConverged)?;
        let n_x = state.backsolver.block_dims()[0];

        // Expanded once, before any re-solve: reading the compressed
        // form means borrowing the NLP, and `run_sens_step` below
        // re-borrows it.
        let (mut lo, mut hi) = {
            let (_, _, nlp) = state.backsolver.activity_handles();
            let nl = nlp.borrow();
            crate::boundcheck::expand_bounds(n_x, &nl.px_l(), &nl.px_u(), nl.x_l(), nl.x_u())
        };
        // Those bounds bound the algorithm's `x̃ = d ⊙ x`, while
        // `state.x` and the step are both in the model's own units
        // (gh#486 stage 3). Undo the change of variables on the bounds
        // so all three agree, rather than projecting onto the wrong box.
        // A negative factor reflects the interval, so the sides swap.
        // `variable_scaling`, not `variable_scaling_full`: `lo` / `hi`
        // are var-x length, and the two index spaces diverge from the
        // first fixed variable on.
        if let Some(d) = state.backsolver.variable_scaling() {
            for i in 0..n_x {
                let di = d[i];
                if di == 0.0 || di == 1.0 {
                    continue;
                }
                let (a, b) = (lo[i] / di, hi[i] / di);
                lo[i] = a.min(b);
                hi[i] = a.max(b);
            }
        }
        let x_curr = &state.x[..n_x];

        // What counts as outside a bound is the solve's own answer: it
        // was willing to leave a converged point `bound_relax_factor`
        // outside, so anything within that is on the bound, not past
        // it. A floor keeps an unrelaxed solve from pinning on
        // roundoff.
        let eps = state.bound_relax_factor.abs().max(1e-9);
        // The bound multipliers at the base point, with the compound
        // row each one occupies, so the refinement can tell when the
        // step drives one negative and release that bound.
        let mults = {
            let dims = state.backsolver.block_dims();
            let (z_l_off, z_u_off) = (
                dims[0] + dims[1] + dims[2] + dims[3],
                dims[0] + dims[1] + dims[2] + dims[3] + dims[4],
            );
            let (data, _, _) = state.backsolver.activity_handles();
            let d = data.borrow();
            let curr = d.curr.as_ref().ok_or(SolverError::NotConverged)?;
            let mut out = Vec::new();
            for (off, v) in [(z_l_off, &curr.z_l), (z_u_off, &curr.z_u)] {
                for (k, &base) in crate::vec_util::dense_to_vec(&**v).iter().enumerate() {
                    out.push(crate::boundcheck::BoundMultiplier { row: off + k, base });
                }
            }
            out
        };
        let (dx, pinned) = crate::boundcheck::refine_step_onto_bounds(
            &state.backsolver,
            &dx_full,
            x_curr,
            &lo,
            &hi,
            &mults,
            eps,
            max_passes,
        )
        .map_err(SolverError::SensComputationFailed)?;
        Ok((
            dx[..n_x].to_vec(),
            pinned.into_iter().map(|p| p as Index).collect(),
        ))
    }

    /// Full KKT-space parametric step for a set of pinned equality
    /// constraints: the same computation as [`Self::parametric_step`],
    /// returned WITHOUT truncating to the primal block. The layout is
    /// the compound KKT vector `(x, s, y_c, y_d, z_l, z_u, v_l, v_u)`;
    /// use [`Self::block_dims`] for the block sizes and
    /// [`Self::g_multiplier_rows`] to locate a constraint's multiplier
    /// row. This exposes the multiplier sensitivities `∂λ*/∂p`
    /// alongside the primal step.
    pub fn parametric_step_full(
        &self,
        pin_constraint_indices: &[Index],
        deltas: &[Number],
    ) -> Result<Vec<Number>, SolverError> {
        if pin_constraint_indices.len() != deltas.len() {
            return Err(SolverError::BadShape {
                what: "deltas",
                got: deltas.len(),
                expected: pin_constraint_indices.len(),
            });
        }
        let state = self.state.borrow();
        let state = state.as_ref().ok_or(SolverError::NotConverged)?;

        let param_rows = state
            .backsolver
            .map_pin_g_to_kkt_rows(pin_constraint_indices)
            .map_err(SolverError::SensComputationFailed)?;
        let signs = vec![1; pin_constraint_indices.len()];
        let a_data = IndexSchurData::from_parts(param_rows, signs)
            .map_err(|e| SolverError::SensComputationFailed(format!("{e:?}")))?;

        let opts = SensOptions {
            run_sens: true,
            ..SensOptions::default()
        };
        let sens_app = SensApplication::new(a_data, state.backsolver.clone(), opts);
        let n_full = state.backsolver.dim();
        let mut dx_full = vec![0.0; n_full];
        if !sens_app.parametric_step(deltas, &mut dx_full) {
            return Err(SolverError::SensComputationFailed(
                "SensApplication::parametric_step failed".into(),
            ));
        }
        let corr = self.barrier_correction(state)?;
        for (d, c) in dx_full.iter_mut().zip(corr.iter()) {
            *d += *c * BARRIER_SIGN;
        }
        Ok(dx_full)
    }

    /// Flat rows of the compound KKT vector holding the equality
    /// multipliers `y_c` for the given 0-based **full-g** constraint
    /// indices. `None` for inequalities (their multipliers live in the
    /// `y_d` block; mapping those is not exposed here). Row `r` of a
    /// [`Self::parametric_step_full`] result is then `∂λ_g/∂p · Δp`.
    pub fn g_multiplier_rows(
        &self,
        g_indices: &[Index],
    ) -> Result<Vec<Option<Index>>, SolverError> {
        let state = self.state.borrow();
        let state = state.as_ref().ok_or(SolverError::NotConverged)?;
        let dims = state.backsolver.block_dims();
        let y_c_offset = (dims[0] + dims[1]) as Index;
        Ok(g_indices
            .iter()
            .map(|&g| {
                state
                    .backsolver
                    .full_g_to_c_block(g)
                    .map(|pos| y_c_offset + pos)
            })
            .collect())
    }

    /// Flat rows of the compound KKT vector holding the primal values
    /// `x` for the given 0-based **full-x** variable indices. `None`
    /// where the solve removed the column (`x_l == x_u` under
    /// `fixed_variable_treatment = make_parameter`), which has no row
    /// in the factor at all.
    ///
    /// The `x` counterpart of [`Self::g_multiplier_rows`], and needed
    /// for the same reason: a caller holding user-space indices — from
    /// the `.col` file, from [`Self::classify_activity`], from
    /// [`Self::row_normal`] — cannot index the factor with them
    /// directly. Row `r` of a [`Self::parametric_step_full`] result is
    /// then `∂x/∂p · Δp` for that variable, and `e_r` is the unit
    /// vector selecting its column in a [`Self::kkt_solve`].
    pub fn x_primal_rows(&self, x_indices: &[Index]) -> Result<Vec<Option<Index>>, SolverError> {
        let state = self.state.borrow();
        let state = state.as_ref().ok_or(SolverError::NotConverged)?;
        let n_full = state.backsolver.n_full_x();
        // out of range must not masquerade as "removed as fixed": the
        // NLP map returns None for both, and the caller's whole reason
        // for asking is that it cannot tell the spaces apart itself
        if let Some(&bad) = x_indices.iter().find(|&&i| i < 0 || i >= n_full) {
            return Err(SolverError::BadShape {
                what: "x_primal_rows variable index",
                got: bad as usize,
                expected: n_full as usize,
            });
        }
        // the x block starts at flat index 0, so the var-x position IS
        // the KKT row; the offset stays explicit for the day it is not
        Ok(x_indices
            .iter()
            .map(|&i| state.backsolver.full_x_to_var_x(i))
            .collect())
    }

    /// The user TNLP's variable count: the length of a full-x report
    /// and the domain of [`Self::x_primal_rows`].
    pub fn n_full_x(&self) -> Result<usize, SolverError> {
        let state = self.state.borrow();
        let state = state.as_ref().ok_or(SolverError::NotConverged)?;
        Ok(state.backsolver.n_full_x() as usize)
    }

    /// Reduced Hessian `H_R = obj_scal · B K⁻¹ Bᵀ` over the pinned
    /// equality-constraint rows, where `B` selects the
    /// `pin_constraint_indices` rows of the y_c block and `K` is the
    /// **natural-units** (unscaled) KKT matrix — active NLP scaling
    /// is undone by the backsolver, so `−inv(H_R)` is directly the
    /// parameter covariance regardless of `nlp_scaling_method`
    /// (pounce#128). `obj_scal` survives as a plain extra multiplier
    /// (default 1.0); it is no longer needed to recover natural units.
    /// Returns the `n²`-long column-major dense matrix
    /// (`n = pin_constraint_indices.len()`).
    ///
    /// Equivalent to [`crate::SensSolve::with_reduced_hessian`] but
    /// usable post-hoc on a held `Solver`. For the solver-space
    /// (pre-#128) value use [`Self::compute_reduced_hessian_scaled`];
    /// the factors themselves are exposed via [`Self::nlp_scaling`] /
    /// [`Self::pin_g_scaling`].
    pub fn compute_reduced_hessian(
        &self,
        pin_constraint_indices: &[Index],
        obj_scal: Number,
    ) -> Result<Vec<Number>, SolverError> {
        let state = self.state.borrow();
        let state = state.as_ref().ok_or(SolverError::NotConverged)?;
        let n = pin_constraint_indices.len();
        let param_rows = state
            .backsolver
            .map_pin_g_to_kkt_rows(pin_constraint_indices)
            .map_err(SolverError::SensComputationFailed)?;
        let signs = vec![1; n];
        let a_data = IndexSchurData::from_parts(param_rows, signs)
            .map_err(|e| SolverError::SensComputationFailed(format!("{e:?}")))?;
        let opts = SensOptions {
            compute_red_hessian: true,
            obj_scal,
            ..SensOptions::default()
        };
        let mut sens_app = SensApplication::new(a_data, state.backsolver.clone(), opts);
        let mut hr = vec![0.0; n * n];
        if !sens_app.compute_reduced_hessian(&mut hr) {
            return Err(SolverError::SensComputationFailed(
                "SensApplication::compute_reduced_hessian failed".into(),
            ));
        }
        Ok(hr)
    }

    /// The reduced Hessian as the solver's internal **scaled** space
    /// sees it — the value [`Self::compute_reduced_hessian`] returned
    /// before pounce#128: `H̃_ij = (df / (dc_i·dc_j)) · H_ij`.
    /// Identical to `compute_reduced_hessian` when no NLP scaling is
    /// active.
    pub fn compute_reduced_hessian_scaled(
        &self,
        pin_constraint_indices: &[Index],
        obj_scal: Number,
    ) -> Result<Vec<Number>, SolverError> {
        let mut hr = self.compute_reduced_hessian(pin_constraint_indices, obj_scal)?;
        let state = self.state.borrow();
        let state = state.as_ref().ok_or(SolverError::NotConverged)?;
        let df = state.backsolver.obj_scaling_factor();
        let dc = state
            .backsolver
            .pin_c_scales(pin_constraint_indices)
            .map_err(SolverError::SensComputationFailed)?;
        crate::reduced_hessian::scale_to_solver_space(&mut hr, df, &dc);
        Ok(hr)
    }

    /// Effective NLP scaling the IPM applied on the most recent
    /// converged solve: `(obj_scaling_factor, c_scale, d_scale)`.
    /// `(1.0, None, None)` ⇔ no scaling was active. The vectors are
    /// per-row factors over the algorithm's equality (`c`) and
    /// inequality (`d`) blocks.
    pub fn nlp_scaling(
        &self,
    ) -> Result<(Number, Option<Vec<Number>>, Option<Vec<Number>>), SolverError> {
        let state = self.state.borrow();
        let state = state.as_ref().ok_or(SolverError::NotConverged)?;
        Ok(state.backsolver.nlp_scaling())
    }

    /// The per-variable `user-scaling` factors `d` the held solve ran
    /// under (gh#486), in the user TNLP's **full-x** space, or `None`
    /// when the solve applied no change of variables.
    ///
    /// Every accessor on this type already reports natural units, so
    /// this is diagnostic rather than a correction a caller has to
    /// apply — it answers "was this solve conditioned, and by how
    /// much", the x-axis counterpart of [`Self::nlp_scaling`].
    pub fn variable_scaling(&self) -> Result<Option<Vec<Number>>, SolverError> {
        let state = self.state.borrow();
        let state = state.as_ref().ok_or(SolverError::NotConverged)?;
        Ok(state.backsolver.variable_scaling_full().map(|d| d.to_vec()))
    }

    /// Inertia-correction perturbations `(δ_x, δ_s, δ_c, δ_d)` baked
    /// into the held KKT factor. All zero ⇔ the final factorization
    /// was unregularized and the natural-units back-solves invert the
    /// exact KKT matrix — see
    /// [`crate::PdSensBacksolver::kkt_perturbations`].
    pub fn kkt_perturbations(&self) -> Result<[Number; 4], SolverError> {
        let state = self.state.borrow();
        let state = state.as_ref().ok_or(SolverError::NotConverged)?;
        Ok(state.backsolver.kkt_perturbations())
    }

    /// Per-pin equality-row scaling factors `dc_i` (1.0 entries when
    /// no constraint scaling is active), ordered like
    /// `pin_constraint_indices`.
    pub fn pin_g_scaling(
        &self,
        pin_constraint_indices: &[Index],
    ) -> Result<Vec<Number>, SolverError> {
        let state = self.state.borrow();
        let state = state.as_ref().ok_or(SolverError::NotConverged)?;
        state
            .backsolver
            .pin_c_scales(pin_constraint_indices)
            .map_err(SolverError::SensComputationFailed)
    }
}
