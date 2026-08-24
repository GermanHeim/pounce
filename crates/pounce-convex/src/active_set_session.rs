//! [`ActiveSetSession`] — a persistent handle over the convex active-set
//! driver, for callers that solve a *family* of QPs rather than one.
//!
//! # Why this exists
//!
//! [`solve_qp_active_set`](crate::active_set::solve_qp_active_set) is a free
//! function over a [`QpProblem`]: it translates to `pounce-qp` form, solves,
//! verifies, and drops everything it built. That is the right shape for a
//! one-shot solve and the wrong one for every other workload the engine is
//! good at — MPC, scenario sweeps, parametric continuation, an outer loop that
//! re-poses the same QP with a moved `c`, `b` or `h`. Two things were lost at
//! each call:
//!
//! * **The translation.** It lived in locals inside the driver, so it was
//!   unreachable, and a frontend that wanted the active-set engine over a
//!   convex `QpProblem` had to restate the whole map — the `+1` index shift,
//!   the `[A_eq ; G]` row stacking, the `±1e19` free-bound convention, and,
//!   most dangerously, the **dual sign transform** on the way back, which a
//!   reimplementation can get wrong without the answer ever looking wrong.
//!   It is now [`ActiveSetQp`], owned by the session (gh #769).
//! * **The previous solve.** `pounce-qp` is a *parametric* active-set engine:
//!   its headline capability is [`QpSolver::solve_parametric`], which traces a
//!   homotopy from a solved neighbouring QP instead of starting over. That
//!   needs the previous problem *and* its solution in the engine's own
//!   coordinates — precisely what the free function throws away, so every
//!   solve through it was cold.
//!
//! # What a session keeps, and what it does not
//!
//! It keeps the last `(problem, solution)` pair **in `pounce-qp` coordinates**,
//! and only when that solve is one worth tracing from (see
//! [`ActiveSetSession::solve`]). It does **not** keep a
//! [`ParametricActiveSetSolver`] across calls: the engine's cached factor
//! belongs to a KKT system the next problem re-factors anyway, and a solver
//! rebuilt per call is exactly what the cold driver does — so the warm and
//! cold legs cannot drift into being two different solvers. The backend
//! factory is called per solve, as the cold driver already calls it.
//!
//! # Everything else is the cold driver's behaviour, unchanged
//!
//! A session is not a second implementation of the driver. The screen, the
//! standard engine defaults, the unscaled → Ruiz → simplex-seeded retry
//! ladder, the certificate re-derivation and the status banding are all
//! reached by calling into [`crate::active_set`] itself, so a session solve
//! that cannot reuse anything is bit-identical to the free function
//! (`session_cold_matches_free_function` asserts it). What the session adds
//! around that is presolve/postsolve — which the CLI had open-coded, leaving
//! every other frontend to either restate it or silently solve the unreduced
//! problem — and the parametric attempt.
//!
//! # The reuse rule
//!
//! One attempt, then the cold ladder:
//!
//! 1. A previous pair exists, the engine called it `Optimal`, the driver
//!    agreed, and the new problem has the same shape ⇒ run
//!    [`QpSolver::solve_parametric`]. That call applies its **own** guards
//!    (identical `H`, unchanged equality/fixed topology) and falls back
//!    internally — first to the previous working set, then cold — so the
//!    session deliberately does not restate them. In particular it does not
//!    guard on `A` or the variable box: gh #602 measured that guard and
//!    declined it, and re-adding it here would quietly overrule that
//!    measurement (`dev-notes/issue-602-parametric-eligibility.md`).
//! 2. The warm answer is verified against the **original** problem by the same
//!    [`verify_status`](crate::active_set) the cold path uses — a warm start
//!    is a hint, never a reason to accept a weaker proof — and is reported
//!    only if it is conclusive.
//! 3. Otherwise the full cold ladder runs and owns the answer. The cost of a
//!    rejected reuse is one solve, which is the same bet `solve_parametric`
//!    documents for its own fallbacks.
//!
//! Reuse never changes what may be *reported*: a warm solve reaches the user
//! only through the identical verification the cold one does.
//!
//! # Frontend responsibilities this does not take over
//!
//! `max_iter = 0` (AMPL's "take no iterations") is a *frontend* semantic and
//! must be enforced above a session: presolve can solve a trivial problem
//! outright, with no iteration to cap. The CLI does that today, and keeps
//! doing it.

use pounce_linsol::SparseSymLinearSolverInterface;
use pounce_qp::{
    ActiveSetOverrides, ParametricActiveSetSolver, QpSolver, QpStatus as ActiveSetStatus,
};

use crate::active_set::{
    ActiveSetQp, NativeSolve, back_translate, empty_solution, engine_options, is_conclusive,
    is_solved, solve_qp_active_set_attempt, verify_status,
};
use crate::ipm::{QpOptions, finite_or_failed};
use crate::presolve::{PresolveOutcome, PresolveStats, presolve};
use crate::qp::{BoxScreen, QpProblem, QpSolution, QpStatus, screen_variable_box};

/// A persistent active-set solve over convex [`QpProblem`]s.
///
/// Construct once, [`solve`](Self::solve) many times. Each solve reports a
/// [`QpSolution`] in the coordinates of the problem it was handed — presolve
/// and any reuse are internal, and neither is visible in the answer.
///
/// ```no_run
/// # use pounce_convex::{ActiveSetSession, QpProblem};
/// # fn demo(backend: impl FnMut() -> Box<dyn pounce_linsol::SparseSymLinearSolverInterface> + 'static,
/// #         problems: Vec<QpProblem>) {
/// let mut session = ActiveSetSession::new(backend);
/// for prob in &problems {
///     let sol = session.solve(prob);
///     println!("{:?} obj={} ({:?})", sol.status, sol.obj, session.last_reuse());
/// }
/// # }
/// ```
pub struct ActiveSetSession {
    make_backend: Box<dyn FnMut() -> Box<dyn SparseSymLinearSolverInterface>>,
    opts: QpOptions,
    engine: ActiveSetOverrides,
    presolve: bool,
    /// The last solve worth tracing from, in `pounce-qp` coordinates.
    prev: Option<NativeSolve>,
    last_reuse: Reuse,
    last_presolve: PresolveNote,
    stats: SessionStats,
}

/// Where the answer the session just reported came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Reuse {
    /// No solve ran — presolve or a screen concluded the problem on its own.
    NoSolve,
    /// Nothing was eligible to reuse; the cold driver produced the answer.
    Cold,
    /// A parametric solve from the previous problem produced the answer.
    Parametric,
    /// A parametric solve ran, its verdict did not stand up against the
    /// original problem, and the cold driver produced the reported answer.
    /// The reuse cost one solve and changed nothing about what was reported.
    ParametricRejected,
}

/// What presolve did to the problem the session was handed.
///
/// A presolve verdict arrives with no iteration behind it, so when it is wrong
/// it is the most expensive failure the solver has (gh #523). The trigger is
/// carried out here for the same reason the CLI prints it: so the claim is
/// auditable from whatever frontend is driving the session.
#[derive(Debug, Clone, PartialEq)]
pub enum PresolveNote {
    /// Presolve is off for this session, or no solve has run yet.
    Disabled,
    /// The problem was reduced and the reduced problem solved.
    Reduced {
        stats: PresolveStats,
        /// A screen claimed infeasibility and the re-derivation without the
        /// speculative fixings did not reproduce it, so presolve solved on
        /// (gh #523). Names the screen that misfired.
        discarded_infeasibility: Option<String>,
    },
    /// Presolve proved the problem primal-infeasible; no solve ran.
    Infeasible { trigger: String },
    /// Presolve proved the problem unbounded below; no solve ran.
    Unbounded,
}

/// Running counts over the session's lifetime.
///
/// Reuse is a *performance* claim, and a claim nothing can measure is one
/// nobody can check. These are what a caller tuning a sweep — or a benchmark
/// asking whether the warm path is engaging at all — reads.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SessionStats {
    /// Calls to [`ActiveSetSession::solve`] / [`ActiveSetSession::solve_cold`].
    pub solves: usize,
    /// Solves that ran a parametric attempt.
    pub parametric_attempts: usize,
    /// Parametric attempts whose verdict stood up and was reported.
    pub parametric_accepted: usize,
    /// Solves that ran the cold ladder (including after a rejected attempt).
    pub cold_solves: usize,
}

impl ActiveSetSession {
    /// Open a session over a linear-solver backend factory.
    ///
    /// The factory is called at least once per solve — the engine may need
    /// more than one backend instance over a single solve, which is why the
    /// driver takes a factory rather than a backend.
    pub fn new<F>(make_backend: F) -> Self
    where
        F: FnMut() -> Box<dyn SparseSymLinearSolverInterface> + 'static,
    {
        ActiveSetSession {
            make_backend: Box::new(make_backend),
            opts: QpOptions::default(),
            engine: ActiveSetOverrides::default(),
            presolve: true,
            prev: None,
            last_reuse: Reuse::Cold,
            last_presolve: PresolveNote::Disabled,
            stats: SessionStats::default(),
        }
    }

    /// Set the convex solve options. `time_limit` is **per
    /// [`solve`](Self::solve) call**: the session opens one deadline scope per
    /// solve, covering presolve, the parametric attempt and the cold ladder
    /// alike. This matches how the batch entry points read the same option.
    #[must_use]
    pub fn with_options(mut self, opts: QpOptions) -> Self {
        self.opts = opts;
        self
    }

    /// Set the inner-engine overrides (the `sqp_qp_*` family).
    #[must_use]
    pub fn with_engine_overrides(mut self, engine: ActiveSetOverrides) -> Self {
        self.engine = engine;
        self
    }

    /// Run convex presolve (and postsolve) around each solve. On by default.
    ///
    /// Off is for callers that need the reported iterate to be in the
    /// coordinates of the problem *as posed* with nothing eliminated — a
    /// debugger, say — and for a parametric family where the reduction is not
    /// stable across members, since a reduction that changes shape from one
    /// member to the next makes every solve cold.
    #[must_use]
    pub fn with_presolve(mut self, on: bool) -> Self {
        self.presolve = on;
        self
    }

    /// Replace the options between solves.
    pub fn set_options(&mut self, opts: QpOptions) {
        self.opts = opts;
    }

    /// The options this session solves under.
    pub fn options(&self) -> &QpOptions {
        &self.opts
    }

    /// Where the last reported answer came from.
    pub fn last_reuse(&self) -> Reuse {
        self.last_reuse
    }

    /// What presolve did on the last solve.
    pub fn last_presolve(&self) -> &PresolveNote {
        &self.last_presolve
    }

    /// Running counts — see [`SessionStats`].
    pub fn stats(&self) -> SessionStats {
        self.stats
    }

    /// Forget the previous solve, so the next one starts cold.
    ///
    /// Reuse is never *unsafe* — the verification gate is the same one the
    /// cold path passes through — so this is a performance control, for a
    /// caller that knows the next problem is unrelated to the last and would
    /// rather not spend a solve finding out.
    pub fn reset(&mut self) {
        self.prev = None;
    }

    /// Solve `prob`, reusing the previous solve when it is eligible.
    pub fn solve(&mut self, prob: &QpProblem) -> QpSolution {
        self.solve_inner(prob, true)
    }

    /// Solve `prob` without attempting reuse, and make its result the base for
    /// the next [`solve`](Self::solve).
    ///
    /// Identical to [`solve_qp_active_set`](crate::active_set::solve_qp_active_set)
    /// under this session's options, plus presolve/postsolve when enabled.
    pub fn solve_cold(&mut self, prob: &QpProblem) -> QpSolution {
        self.solve_inner(prob, false)
    }

    fn solve_inner(&mut self, prob: &QpProblem, allow_reuse: bool) -> QpSolution {
        self.stats.solves += 1;
        // One deadline scope per solve. `with_deadline` is re-entrant — an
        // inner scope defers to an outer one — so the driver's own scope,
        // opened a call below, measures against this clock rather than
        // restarting it, and the retry ladder cannot spend the budget twice.
        let limit = self.opts.time_limit;
        crate::deadline::with_deadline(limit, || self.solve_scoped(prob, allow_reuse))
    }

    fn solve_scoped(&mut self, prob: &QpProblem, allow_reuse: bool) -> QpSolution {
        if !self.presolve {
            self.last_presolve = PresolveNote::Disabled;
            return self.solve_engine(prob, 0.0, allow_reuse);
        }
        match presolve(prob) {
            PresolveOutcome::Reduced(ps) => {
                self.last_presolve = PresolveNote::Reduced {
                    stats: ps.stats(),
                    discarded_infeasibility: ps
                        .discarded_infeasibility()
                        .map(|trigger| trigger.to_string()),
                };
                // The reduced problem differs from `prob` by `ps.obj_offset()`,
                // so the constant that makes the solver's objective
                // commensurate with the caller's carries that offset too
                // (gh #689: `obj_constant` normalizes the convergence test, and
                // an offset left out of it normalizes by the wrong magnitude).
                let red = self.solve_engine(&ps.reduced, ps.obj_offset(), allow_reuse);
                ps.postsolve(&red)
            }
            PresolveOutcome::Infeasible(trigger) => {
                self.last_presolve = PresolveNote::Infeasible {
                    trigger: trigger.to_string(),
                };
                self.conclude_without_solving(prob, QpStatus::PrimalInfeasible)
            }
            PresolveOutcome::Unbounded => {
                self.last_presolve = PresolveNote::Unbounded;
                self.conclude_without_solving(prob, QpStatus::DualInfeasible)
            }
        }
    }

    /// A verdict reached with no engine solve behind it. There is no pair to
    /// trace the next problem from, so the session goes cold rather than
    /// carrying a pair that describes some earlier problem.
    fn conclude_without_solving(&mut self, prob: &QpProblem, status: QpStatus) -> QpSolution {
        self.prev = None;
        self.last_reuse = Reuse::NoSolve;
        empty_solution(prob.n, prob.m_eq(), prob.m_ineq(), status)
    }

    fn solve_engine(&mut self, prob: &QpProblem, obj_offset: f64, allow_reuse: bool) -> QpSolution {
        let opts = QpOptions {
            obj_constant: self.opts.obj_constant + obj_offset,
            ..self.opts
        };
        // Overwritten by `try_parametric` when an attempt actually runs, and
        // again below when one is accepted; a solve that never attempts reuse
        // leaves it here.
        self.last_reuse = Reuse::Cold;
        if allow_reuse && let Some(sol) = self.try_parametric(prob, &opts) {
            self.last_reuse = Reuse::Parametric;
            return sol;
        }
        self.stats.cold_solves += 1;
        let engine = self.engine;
        let mut mk: &mut dyn FnMut() -> Box<dyn SparseSymLinearSolverInterface> =
            &mut *self.make_backend;
        let att = solve_qp_active_set_attempt(prob, &opts, &engine, &mut mk);
        self.remember(att.native, att.sol.status);
        att.sol
    }

    /// One parametric attempt. `None` means "not eligible, or its verdict did
    /// not stand up" — in both cases the caller runs the cold ladder, and in
    /// the second the stats record that the attempt happened.
    fn try_parametric(&mut self, prob: &QpProblem, opts: &QpOptions) -> Option<QpSolution> {
        // Taken, not borrowed: the backend factory below needs `&mut self`, and
        // a pair that turns out not to be reusable has no second chance — the
        // cold solve about to run replaces it.
        let prev = self.prev.take()?;

        // Screen the variable box exactly as the cold driver does at its own
        // entry (gh #295, gh #491): an empty box panicked the engine, and a
        // hairline crossing is repaired rather than rejected. An empty box is
        // handed on to the cold path rather than concluded here, so the
        // certified `PrimalInfeasible` is produced in exactly one place.
        let snapped;
        let prob = match screen_variable_box(prob) {
            BoxScreen::Feasible => prob,
            BoxScreen::Empty => return None,
            BoxScreen::Snapped(p) => {
                snapped = p;
                &snapped
            }
        };

        let native = ActiveSetQp::from_convex(prob);
        // The only guard the session applies. A shape change makes the pair
        // dimensionally meaningless, and `solve_parametric` would answer it
        // with a cold solve of its own — one that has neither the Ruiz retry
        // nor the simplex seed, i.e. strictly worse than the ladder below.
        // Every *other* eligibility question (identical `H`, unchanged
        // equality/fixed topology, and the `A` / box guards gh #602 measured
        // and declined) belongs to `solve_parametric`, which applies it with
        // the fallbacks it was measured against.
        if native.n() != prev.qp.n() || native.m() != prev.qp.m() {
            return None;
        }
        if crate::deadline::expired() {
            return None;
        }

        self.stats.parametric_attempts += 1;
        let qopts = engine_options(opts, &self.engine, native.n(), native.m());
        let mut solver = ParametricActiveSetSolver::new((self.make_backend)());
        let qsol =
            match solver.solve_parametric(&prev.qp.problem(), &prev.sol, &native.problem(), &qopts)
            {
                Ok(q) => q,
                // A hard error is a numerical failure, not a verdict — and never
                // an infeasibility claim. Fall through to the cold ladder, which
                // may well succeed on its own seed.
                Err(_) => return None,
            };

        // Verified against the problem as posed, by the same gate the cold
        // path runs. A warm start is a hint about *where* the answer is; it
        // earns nothing about whether the point returned is one.
        let mut sol = back_translate(prob, &qsol);
        sol.status = verify_status(qsol.status, qsol.unbounded_ray.as_deref(), &sol, prob, opts);
        let sol = finite_or_failed(prob, sol);
        // Same policy as the cold driver: a deadline crossing observed after
        // the solve returned relabels a give-up status, never a verdict.
        let sol = if crate::deadline::expired() {
            crate::ipm::mark_timed_out(sol)
        } else {
            sol
        };
        if !is_conclusive(sol.status) {
            // The attempt ran and cost a solve; say so rather than reporting
            // this as a cold solve that never tried.
            self.last_reuse = Reuse::ParametricRejected;
            return None;
        }
        self.stats.parametric_accepted += 1;
        let status = sol.status;
        self.remember(
            Some(NativeSolve {
                qp: native,
                sol: qsol,
            }),
            status,
        );
        Some(sol)
    }

    /// Keep the engine-side pair only when the next solve could honestly trace
    /// from it.
    ///
    /// Both halves of the test are load-bearing. [`QpSolver::solve_parametric`]
    /// requires `sol_prev.status == Optimal` — it starts the path *on* the
    /// previous solution manifold, and a `MaxIter` or `TimeLimit` iterate is
    /// not on it. And the driver's own verdict has to agree: the engine
    /// claiming `Optimal` is exactly the claim
    /// [`verify_status`](crate::active_set) exists to re-derive rather than
    /// propagate (`QSC205` returns `Optimal` at the wrong objective), so a
    /// point the driver refused to report is not one to warm-start from
    /// either. Together they also cover the
    /// [`finite_or_failed`](crate::ipm) gate: a solution that gate replaced
    /// never carries a solved status, so the pair behind it is dropped here.
    fn remember(&mut self, native: Option<NativeSolve>, reported: QpStatus) {
        self.prev =
            native.filter(|ns| ns.sol.status == ActiveSetStatus::Optimal && is_solved(reported));
    }
}
