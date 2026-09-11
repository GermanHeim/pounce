//! Warm-start-aware presolve sessions: use presolve *and* warm starts.
//!
//! A session retains the presolve transformation across re-solves and maps
//! every original-space warm point into the reduced space the solver sees,
//! so persistent callers (oximo, MPC loops) no longer pick one or the other.
//!
//! ```
//! use std::cell::RefCell;
//! use std::rc::Rc;
//! use pounce_rs::prelude::*;
//! use pounce_rs::session::TnlpPresolveSession;
//! # struct P;
//! # impl TNLP for P {
//! #     fn get_nlp_info(&mut self) -> Option<NlpInfo> {
//! #         Some(NlpInfo { n: 1, m: 0, nnz_jac_g: 0, nnz_h_lag: 0, index_style: IndexStyle::C })
//! #     }
//! #     fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
//! #         b.x_l[0] = -10.0; b.x_u[0] = 10.0; true
//! #     }
//! #     fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool { sp.x[0] = 0.0; true }
//! #     fn eval_f(&mut self, x: &[Number], _n: bool) -> Option<Number> { Some((x[0] - 1.0).powi(2)) }
//! #     fn eval_grad_f(&mut self, x: &[Number], _n: bool, g: &mut [Number]) -> bool {
//! #         g[0] = 2.0 * (x[0] - 1.0); true
//! #     }
//! #     fn eval_g(&mut self, _x: &[Number], _n: bool, _g: &mut [Number]) -> bool { true }
//! #     fn eval_jac_g(&mut self, _x: Option<&[Number]>, _n: bool, _m: SparsityRequest<'_>) -> bool { true }
//! #     fn finalize_solution(&mut self, _s: Solution<'_>, _d: &IpoptData, _q: &IpoptCq) {}
//! # }
//!
//! let inner: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(P));
//! let mut session = TnlpPresolveSession::new(Rc::clone(&inner))?;
//! session.set_option_str("presolve", "yes")?;
//! session.set_option_str("hessian_approximation", "limited-memory")?;
//! let first = session.solve_cold()?;
//! assert!(first.success);
//! // Mutate the problem in place, then re-solve warm through presolve.
//! let second = session.solve_warm_last()?;
//! assert!(second.success);
//! # Ok::<(), pounce_rs::session::SessionError>(())
//! ```
//!
//! ## How it works
//!
//! The session owns the [`IpoptApplication`] and an injector TNLP between
//! the user's problem and the presolve wrapper:
//!
//! ```text
//! user TNLP → WarmInjector → PresolveTnlp → [LinearEqElimTnlp] → solver
//! ```
//!
//! The injector serves the staged warm point from `get_starting_point`
//! (falling back on length mismatch) and records the `finalize_solution`
//! payload — forwarded in original space — as the next seed. Both wrappers
//! already project what the injector serves, so the seed flows through
//! whatever reduction is live.
//!
//! ## Cache + validate
//!
//! The transformation derives from live problem data, so the session
//! fingerprints it before each solve (see
//! [`pounce_presolve::warm::PresolveFingerprint`]): match reuses the
//! wrapper, mismatch rebuilds it and maps the seed through the fresh
//! transform. FBBT-tape swaps and nonlinear data under auxiliary Phase 0
//! are invisible to the fingerprint — call
//! [`TnlpPresolveSession::invalidate`] after those.
//!
//! ## Scope notes
//!
//! * No second-opinion ladder: failures return as-is, so `status`/`stats`
//!   always belong to the staged seed.
//! * No iteration capture unless [`TnlpPresolveSession::enable_iter_history`]
//!   is set.

use std::cell::RefCell;
use std::rc::Rc;

use pounce_algorithm::application::IpoptApplication;
use pounce_common::options_list::OptionsList;
use pounce_common::types::Number;
use pounce_nlp::constant_derivatives::DerivativeProofs;
use pounce_nlp::return_codes::ApplicationReturnStatus;
use pounce_nlp::solve_statistics::SolveStatistics;
use pounce_nlp::tnlp::{
    BoundsInfo, IpoptCq, IpoptData, IterStats, Linearity, MetaData, NlpInfo, ScalingRequest,
    Solution as TnlpSolution, SparsityRequest, StartingPoint, TNLP,
};
use pounce_presolve::warm::{
    PresolveFingerprint, WarmPoint, WarmProjectionReport, compute_fingerprint, project_warm_point,
};
use pounce_presolve::{LinearEqElimTnlp, PresolveOptions, PresolveTnlp};

/// Why a session solve could not be started.
///
/// A solve that *runs* but fails still returns [`SessionSolution`] with
/// `success == false`, mirroring [`crate::builder::NlpError`].
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum SessionError {
    /// `IpoptApplication::initialize` failed.
    Initialize(String),
    /// Presolve options unreadable from the option table.
    PresolveOptions(String),
    /// A `set_option_*` value rejected by the registry.
    InvalidOption {
        /// Option name as passed.
        tag: String,
        /// Rejected value, rendered.
        value: String,
        /// Registry's explanation.
        reason: String,
    },
}

impl std::fmt::Display for SessionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Initialize(msg) => write!(f, "IpoptApplication::initialize failed: {msg}"),
            Self::PresolveOptions(msg) => write!(f, "presolve setup failed: {msg}"),
            Self::InvalidOption { tag, value, reason } => {
                write!(f, "option {tag}={value} rejected: {reason}")
            }
        }
    }
}

impl std::error::Error for SessionError {}

/// One session solve, in **original** problem space.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct SessionSolution {
    /// Solver status; `success` is the convenient boolean.
    pub status: ApplicationReturnStatus,
    /// `true` for `SolveSucceeded` / `SolvedToAcceptableLevel`.
    pub success: bool,
    /// Optimal variables (length `n`; empty if the solve never finalized).
    pub x: Vec<Number>,
    /// Objective at the solution.
    pub objective: Number,
    /// Constraint multipliers (length `m`, original row order; dropped rows
    /// recovered via KKT stationarity where applicable).
    pub lambda: Vec<Number>,
    /// Constraint values at the solution (length `m`).
    pub g: Vec<Number>,
    /// Lower-bound multipliers (length `n`).
    pub z_l: Vec<Number>,
    /// Upper-bound multipliers (length `n`).
    pub z_u: Vec<Number>,
    /// Per-solve statistics (wall time, iterations, `final_mu`, …).
    pub stats: SolveStatistics,
    /// Whether the retained transformation was reused (`false` = rebuilt;
    /// always `false` with `presolve=no`).
    pub presolve_reused: bool,
    /// What the warm-point projection did, when a seed was staged and a
    /// presolve map was live to describe it.
    pub warm_report: Option<WarmProjectionReport>,
}

impl SessionSolution {
    /// This solution as the next solve's warm start, threading `final_mu`.
    pub fn warm_point(&self) -> WarmPoint {
        WarmPoint {
            x: self.x.clone(),
            lambda: self.lambda.clone(),
            z_l: self.z_l.clone(),
            z_u: self.z_u.clone(),
            mu: Some(self.stats.final_mu),
        }
    }
}

/// Floor for `mu_init` threading, mirroring `batch.rs`.
const WARM_MU_FLOOR: Number = 1e-9;
/// Resuming above the cold default walks back out to the central path.
const WARM_MU_CEILING: Number = 0.1;

/// Last `finalize_solution` payload.
#[derive(Debug, Clone, Default)]
struct CapturedSolution {
    x: Vec<Number>,
    lambda: Vec<Number>,
    z_l: Vec<Number>,
    z_u: Vec<Number>,
    g: Vec<Number>,
    obj: Number,
}

/// Transparent decorator between the user's problem and presolve: serves a
/// staged original-space seed, records the (original-space) solution as the
/// next seed. Length mismatches fall back to the inner TNLP, mirroring the
/// batch warm-start contract in `pounce-algorithm/src/batch.rs`.
struct WarmInjector {
    inner: Rc<RefCell<dyn TNLP>>,
    warm: Option<WarmPoint>,
    captured: Option<CapturedSolution>,
}

impl WarmInjector {
    fn set_warm(&mut self, warm: Option<WarmPoint>) {
        self.warm = warm;
    }

    fn take_capture(&mut self) -> Option<CapturedSolution> {
        self.captured.take()
    }

    fn dims_ok(&self, sp: &StartingPoint<'_>) -> bool {
        match &self.warm {
            Some(w) => {
                w.x.len() == sp.x.len()
                    && (!sp.init_lambda || w.lambda.len() == sp.lambda.len())
                    && (!sp.init_z || (w.z_l.len() == sp.z_l.len() && w.z_u.len() == sp.z_u.len()))
            }
            None => false,
        }
    }
}

impl TNLP for WarmInjector {
    fn get_nlp_info(&mut self) -> Option<NlpInfo> {
        self.inner.borrow_mut().get_nlp_info()
    }

    fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
        self.inner.borrow_mut().get_bounds_info(b)
    }

    fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
        // Misshaped seeds fall back to the inner point (cold for this solve).
        if !self.dims_ok(&sp) {
            return self.inner.borrow_mut().get_starting_point(sp);
        }
        // `dims_ok` is false for `None`, so this is always `Some`.
        let warm = match &self.warm {
            Some(w) => w,
            None => return self.inner.borrow_mut().get_starting_point(sp),
        };
        if sp.init_x {
            sp.x.copy_from_slice(&warm.x);
        }
        if sp.init_z {
            sp.z_l.copy_from_slice(&warm.z_l);
            sp.z_u.copy_from_slice(&warm.z_u);
        }
        if sp.init_lambda {
            sp.lambda.copy_from_slice(&warm.lambda);
        }
        true
    }

    fn eval_f(&mut self, x: &[Number], new_x: bool) -> Option<Number> {
        self.inner.borrow_mut().eval_f(x, new_x)
    }

    fn eval_grad_f(&mut self, x: &[Number], new_x: bool, grad_f: &mut [Number]) -> bool {
        self.inner.borrow_mut().eval_grad_f(x, new_x, grad_f)
    }

    fn eval_g(&mut self, x: &[Number], new_x: bool, g: &mut [Number]) -> bool {
        self.inner.borrow_mut().eval_g(x, new_x, g)
    }

    fn eval_jac_g(&mut self, x: Option<&[Number]>, new_x: bool, mode: SparsityRequest<'_>) -> bool {
        self.inner.borrow_mut().eval_jac_g(x, new_x, mode)
    }

    fn eval_h(
        &mut self,
        x: Option<&[Number]>,
        new_x: bool,
        obj_factor: Number,
        lambda: Option<&[Number]>,
        new_lambda: bool,
        mode: SparsityRequest<'_>,
    ) -> bool {
        self.inner
            .borrow_mut()
            .eval_h(x, new_x, obj_factor, lambda, new_lambda, mode)
    }

    fn finalize_solution(&mut self, sol: TnlpSolution<'_>, ip_data: &IpoptData, ip_cq: &IpoptCq) {
        self.captured = Some(CapturedSolution {
            x: sol.x.to_vec(),
            lambda: sol.lambda.to_vec(),
            z_l: sol.z_l.to_vec(),
            z_u: sol.z_u.to_vec(),
            g: sol.g.to_vec(),
            obj: sol.obj_value,
        });
        self.inner
            .borrow_mut()
            .finalize_solution(sol, ip_data, ip_cq);
    }

    fn get_var_con_metadata(&mut self, var: &mut MetaData, con: &mut MetaData) -> bool {
        self.inner.borrow_mut().get_var_con_metadata(var, con)
    }

    fn get_scaling_parameters(&mut self, req: ScalingRequest<'_>) -> bool {
        self.inner.borrow_mut().get_scaling_parameters(req)
    }

    fn get_variables_linearity(&mut self, types: &mut [Linearity]) -> bool {
        self.inner.borrow_mut().get_variables_linearity(types)
    }

    fn get_objective_variables_linearity(&mut self, types: &mut [Linearity]) -> bool {
        self.inner
            .borrow_mut()
            .get_objective_variables_linearity(types)
    }

    fn get_constraints_linearity(&mut self, types: &mut [Linearity]) -> bool {
        self.inner.borrow_mut().get_constraints_linearity(types)
    }

    fn get_number_of_nonlinear_variables(&mut self) -> pounce_common::types::Index {
        self.inner.borrow_mut().get_number_of_nonlinear_variables()
    }

    fn get_list_of_nonlinear_variables(
        &mut self,
        pos_nonlin_vars: &mut [pounce_common::types::Index],
    ) -> bool {
        self.inner
            .borrow_mut()
            .get_list_of_nonlinear_variables(pos_nonlin_vars)
    }

    fn derivative_proofs(&mut self) -> DerivativeProofs {
        // Transparent decorator, unlike presolve: forward the inner answer.
        self.inner.borrow_mut().derivative_proofs()
    }

    fn intermediate_callback(
        &mut self,
        stats: IterStats,
        ip_data: &IpoptData,
        ip_cq: &IpoptCq,
    ) -> bool {
        self.inner
            .borrow_mut()
            .intermediate_callback(stats, ip_data, ip_cq)
    }

    fn finalize_metadata(&mut self, var: &MetaData, con: &MetaData) {
        self.inner.borrow_mut().finalize_metadata(var, con);
    }

    fn is_presolve_wrapper(&self) -> bool {
        self.inner.borrow().is_presolve_wrapper()
    }

    fn scaling_factors(&self) -> Option<Vec<Number>> {
        self.inner.borrow().scaling_factors()
    }

    fn presolve_infeasibility_proof(&self) -> Option<pounce_nlp::tnlp::InfeasibilityProof> {
        self.inner.borrow().presolve_infeasibility_proof()
    }
}

/// Persistent solve handle over one problem: presolve stays on across
/// re-solves, and every re-solve can seed from the last solution.
///
/// Owns the [`IpoptApplication`]; the TNLP stays caller-owned and is mutated
/// in place between solves. Shape changes are picked up via the fingerprint;
/// FBBT-tape swaps need [`Self::invalidate`].
pub struct TnlpPresolveSession {
    app: IpoptApplication,
    inner: Rc<RefCell<dyn TNLP>>,
    injector: Rc<RefCell<WarmInjector>>,
    presolve: Option<Rc<RefCell<PresolveTnlp>>>,
    elim: Option<Rc<RefCell<LinearEqElimTnlp>>>,
    outer: Rc<RefCell<dyn TNLP>>,
    fingerprint: Option<PresolveFingerprint>,
    last: Option<SessionSolution>,
    /// Held guard keeping iteration capture live once enabled.
    #[allow(dead_code)]
    iter_scope: Option<pounce_observability::CollectorScope>,
}

impl TnlpPresolveSession {
    /// Open a session over `inner`. Marks `presolve_already_applied` (the
    /// session wraps itself) and sets `warm_start_init_point=yes` so staged
    /// duals are consumed.
    pub fn new(inner: Rc<RefCell<dyn TNLP>>) -> Result<Self, SessionError> {
        let mut app = IpoptApplication::new();
        app.initialize()
            .map_err(|e| SessionError::Initialize(e.message))?;
        app.set_presolve_already_applied(true);
        let _ = app
            .options_mut()
            .set_string_value("warm_start_init_point", "yes", true, false);
        let injector = Rc::new(RefCell::new(WarmInjector {
            inner: Rc::clone(&inner),
            warm: None,
            captured: None,
        }));
        let outer = Rc::clone(&injector) as Rc<RefCell<dyn TNLP>>;
        Ok(Self {
            app,
            inner,
            injector,
            presolve: None,
            elim: None,
            outer,
            fingerprint: None,
            last: None,
            iter_scope: None,
        })
    }

    /// Solver tolerances, `presolve=yes`, … — re-read before every solve, so
    /// toggling `presolve` between solves just works.
    pub fn options_mut(&mut self) -> &mut OptionsList {
        self.app.options_mut()
    }

    /// Record the per-iteration trajectory into `stats.iterations`.
    pub fn enable_iter_history(&mut self) {
        self.app.enable_iter_history();
        if self.iter_scope.is_none() {
            self.iter_scope = Some(crate::collector_scope());
        }
    }

    /// A numeric option (`("tol", 1e-8)`); rejections surface as
    /// [`SessionError::InvalidOption`] instead of passing silently.
    pub fn set_option_num(&mut self, tag: &str, value: Number) -> Result<(), SessionError> {
        self.app
            .options_mut()
            .set_numeric_value(tag, value, true, true)
            .map_err(|e| SessionError::InvalidOption {
                tag: tag.to_string(),
                value: value.to_string(),
                reason: e.message,
            })?;
        Ok(())
    }

    /// An integer option (`("max_iter", 500)`).
    pub fn set_option_int(&mut self, tag: &str, value: i32) -> Result<(), SessionError> {
        self.app
            .options_mut()
            .set_integer_value(tag, value, true, true)
            .map_err(|e| SessionError::InvalidOption {
                tag: tag.to_string(),
                value: value.to_string(),
                reason: e.message,
            })?;
        Ok(())
    }

    /// A string option (`("presolve", "yes")`).
    pub fn set_option_str(&mut self, tag: &str, value: &str) -> Result<(), SessionError> {
        self.app
            .options_mut()
            .set_string_value(tag, value, true, true)
            .map_err(|e| SessionError::InvalidOption {
                tag: tag.to_string(),
                value: value.to_string(),
                reason: e.message,
            })?;
        Ok(())
    }

    /// Drop the retained transformation and last solution; the next solve
    /// rebuilds from scratch. Needed after fingerprint-invisible changes
    /// (FBBT tapes, nonlinear data under `presolve_auxiliary=yes`).
    pub fn invalidate(&mut self) {
        self.presolve = None;
        self.elim = None;
        self.fingerprint = None;
        self.last = None;
        self.outer = Rc::clone(&self.injector) as Rc<RefCell<dyn TNLP>>;
    }

    /// The last outcome, if any solve has run.
    pub fn last(&self) -> Option<&SessionSolution> {
        self.last.as_ref()
    }

    /// Solve cold (the TNLP's own starting point). Presolve still applies.
    pub fn solve_cold(&mut self) -> Result<SessionSolution, SessionError> {
        self.run(None)
    }

    /// Solve warm from `warm` (original space) through the live reduction.
    /// A misshaped seed falls back to cold for this solve.
    pub fn solve_warm(&mut self, warm: &WarmPoint) -> Result<SessionSolution, SessionError> {
        self.run(Some(warm.clone()))
    }

    /// Solve warm from the last solution. Cold when no solve has run yet.
    pub fn solve_warm_last(&mut self) -> Result<SessionSolution, SessionError> {
        let warm = self.last.as_ref().map(SessionSolution::warm_point);
        self.run(warm)
    }

    /// Rebuild-or-reuse the wrapper chain; returns whether it was reused.
    fn ensure_wrapper(&mut self, opts: &PresolveOptions) -> Result<bool, SessionError> {
        if !opts.enabled {
            self.presolve = None;
            self.elim = None;
            self.fingerprint = None;
            self.outer = Rc::clone(&self.injector) as Rc<RefCell<dyn TNLP>>;
            return Ok(false);
        }
        let fp = compute_fingerprint(&self.inner, opts);
        let reuse = match (&self.presolve, &self.fingerprint, &fp) {
            (Some(_), Some(a), Some(b)) => a == b,
            _ => false,
        };
        if !reuse {
            let dyn_injector = Rc::clone(&self.injector) as Rc<RefCell<dyn TNLP>>;
            let ps = Rc::new(RefCell::new(PresolveTnlp::new(dyn_injector, *opts)));
            let outer: Rc<RefCell<dyn TNLP>> = if opts.linear_eq_reduction {
                let elim = Rc::new(RefCell::new(LinearEqElimTnlp::new(
                    Rc::clone(&ps) as Rc<RefCell<dyn TNLP>>,
                    *opts,
                )));
                let dyn_elim = Rc::clone(&elim) as Rc<RefCell<dyn TNLP>>;
                self.elim = Some(elim);
                dyn_elim
            } else {
                self.elim = None;
                Rc::clone(&ps) as Rc<RefCell<dyn TNLP>>
            };
            self.presolve = Some(ps);
            self.outer = outer;
            self.fingerprint = fp;
        }
        Ok(reuse)
    }

    fn run(&mut self, warm: Option<WarmPoint>) -> Result<SessionSolution, SessionError> {
        let opts = PresolveOptions::from_options_list(self.app.options())
            .map_err(|e| SessionError::PresolveOptions(e.message))?;
        let reused = self.ensure_wrapper(&opts)?;

        // Explicit caller `mu` wins, else the last `final_mu`, clamped —
        // unless the caller set `mu_init` explicitly (the `batch.rs` rule).
        let user_set_mu = matches!(
            self.app.options().get_numeric_value("mu_init", ""),
            Ok((_, true))
        );
        if !user_set_mu {
            if let Some(w) = &warm {
                let mu =
                    w.mu.or_else(|| self.last.as_ref().map(|l| l.stats.final_mu))
                        .unwrap_or(WARM_MU_CEILING)
                        .clamp(WARM_MU_FLOOR, WARM_MU_CEILING);
                let _ = self
                    .app
                    .options_mut()
                    .set_numeric_value("mu_init", mu, true, false);
            } else {
                let _ = self.app.options_mut().set_numeric_value(
                    "mu_init",
                    WARM_MU_CEILING,
                    true,
                    false,
                );
            }
        }

        self.injector.borrow_mut().set_warm(warm.clone());
        let status = self.app.optimize_tnlp(Rc::clone(&self.outer));
        let stats = self.app.statistics();
        self.injector.borrow_mut().set_warm(None);

        let captured = self
            .injector
            .borrow_mut()
            .take_capture()
            .unwrap_or_default();
        let success = matches!(
            status,
            ApplicationReturnStatus::SolveSucceeded
                | ApplicationReturnStatus::SolvedToAcceptableLevel
        );

        // Read the map after the solve so the report matches the transform.
        let warm_report = match (&warm, &self.presolve) {
            (Some(w), Some(ps)) => {
                let mut ps = ps.borrow_mut();
                ps.transformation()
                    .as_ref()
                    .and_then(|map| project_warm_point(map, w))
                    .map(|p| p.report)
            }
            _ => None,
        };

        let sol = SessionSolution {
            status,
            success,
            x: captured.x,
            objective: captured.obj,
            lambda: captured.lambda,
            g: captured.g,
            z_l: captured.z_l,
            z_u: captured.z_u,
            stats,
            presolve_reused: reused,
            warm_report,
        };
        self.last = Some(sol.clone());
        Ok(sol)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use pounce_common::types::Index;
    use pounce_nlp::tnlp::{IndexStyle, NlpInfo};
    use std::cell::RefCell as StdRefCell;

    /// min (x-2)^2 s.t. x + y == 3, y <= 10 (redundant in the box),
    /// 0 <= x,y <= 5. Optimum (2, 1). Params behind a shared handle so tests
    /// mutate the problem between solves.
    #[derive(Debug, Clone)]
    struct Params {
        x_l: Vec<f64>,
        x_u: Vec<f64>,
        x0: Vec<f64>,
    }

    struct Mini {
        params: Rc<StdRefCell<Params>>,
    }

    impl TNLP for Mini {
        fn get_nlp_info(&mut self) -> Option<NlpInfo> {
            Some(NlpInfo {
                n: 2,
                m: 2,
                nnz_jac_g: 4,
                nnz_h_lag: 0,
                index_style: IndexStyle::C,
            })
        }
        fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
            let p = self.params.borrow();
            b.x_l.copy_from_slice(&p.x_l);
            b.x_u.copy_from_slice(&p.x_u);
            b.g_l.copy_from_slice(&[3.0, -2.0e19]);
            b.g_u.copy_from_slice(&[3.0, 10.0]);
            true
        }
        fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
            let p = self.params.borrow();
            sp.x.copy_from_slice(&p.x0);
            true
        }
        fn eval_f(&mut self, x: &[Number], _new_x: bool) -> Option<Number> {
            Some((x[0] - 2.0).powi(2))
        }
        fn eval_grad_f(&mut self, x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
            g[0] = 2.0 * (x[0] - 2.0);
            g[1] = 0.0;
            true
        }
        fn eval_g(&mut self, x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
            g[0] = x[0] + x[1];
            g[1] = x[1];
            true
        }
        fn eval_jac_g(
            &mut self,
            _x: Option<&[Number]>,
            _new_x: bool,
            mode: SparsityRequest<'_>,
        ) -> bool {
            match mode {
                SparsityRequest::Structure { irow, jcol } => {
                    irow.copy_from_slice(&[0, 0, 1, 1]);
                    jcol.copy_from_slice(&[0, 1, 0, 1]);
                }
                SparsityRequest::Values { values } => {
                    values.copy_from_slice(&[1.0, 1.0, 0.0, 1.0]);
                }
            }
            true
        }
        fn get_constraints_linearity(&mut self, types: &mut [Linearity]) -> bool {
            types.fill(Linearity::Linear);
            true
        }
        fn finalize_solution(&mut self, _s: TnlpSolution<'_>, _d: &IpoptData, _q: &IpoptCq) {}
    }

    fn mini_session(presolve: bool) -> (TnlpPresolveSession, Rc<StdRefCell<Params>>) {
        let params = Rc::new(StdRefCell::new(Params {
            x_l: vec![0.0, 0.0],
            x_u: vec![5.0, 5.0],
            x0: vec![0.0, 0.0],
        }));
        let inner: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(Mini {
            params: Rc::clone(&params),
        }));
        let mut s = TnlpPresolveSession::new(inner).expect("session");
        s.set_option_str("hessian_approximation", "limited-memory")
            .expect("opt");
        s.set_option_int("print_level", 0).expect("opt");
        if presolve {
            s.set_option_str("presolve", "yes").expect("opt");
        }
        (s, params)
    }

    fn assert_optimum(sol: &SessionSolution) {
        assert!(sol.success, "status = {:?}", sol.status);
        assert_eq!(sol.x.len(), 2);
        assert_eq!(sol.lambda.len(), 2);
        assert!((sol.x[0] - 2.0).abs() < 1e-4, "x = {:?}", sol.x);
        assert!((sol.x[1] - 1.0).abs() < 1e-4, "x = {:?}", sol.x);
    }

    #[test]
    fn cold_then_warm_through_presolve() {
        let (mut s, _params) = mini_session(true);
        let first = s.solve_cold().expect("cold");
        assert_optimum(&first);
        assert!(!first.presolve_reused, "first solve builds the wrapper");
        assert!(first.warm_report.is_none(), "cold solve stages nothing");

        // Unchanged data reuses the wrapper; the seed flows through it.
        let second = s.solve_warm_last().expect("warm");
        assert_optimum(&second);
        assert!(second.presolve_reused, "identical data reuses the map");
        let rep = second.warm_report.expect("warm report");
        // Dropped y<=10 row shows up as dropped dual mass.
        assert_eq!(rep.n_dropped_rows, 1, "report = {rep:?}");
    }

    #[test]
    fn bound_change_rebuilds_and_stays_warm() {
        let (mut s, params) = mini_session(true);
        let first = s.solve_cold().expect("cold");
        assert_optimum(&first);

        // Moved box: rebuild, still warm, same optimum.
        params.borrow_mut().x_l = vec![0.0, 0.5];
        let second = s.solve_warm_last().expect("warm");
        assert_optimum(&second);
        assert!(
            !second.presolve_reused,
            "a bound change rebuilds the transformation"
        );
        assert!(second.warm_report.is_some());
    }

    #[test]
    fn presolve_off_still_warms() {
        let (mut s, _params) = mini_session(false);
        let first = s.solve_cold().expect("cold");
        assert_optimum(&first);
        assert!(!first.presolve_reused);
        let second = s.solve_warm_last().expect("warm");
        assert_optimum(&second);
        assert!(!second.presolve_reused);
        assert!(second.warm_report.is_none());
    }

    #[test]
    fn misshaped_warm_falls_back_to_cold() {
        let (mut s, _params) = mini_session(true);
        let bad = WarmPoint {
            x: vec![0.0], // wrong n
            lambda: vec![0.0, 0.0],
            z_l: vec![0.0, 0.0],
            z_u: vec![0.0, 0.0],
            mu: None,
        };
        let sol = s.solve_warm(&bad).expect("fallback");
        assert_optimum(&sol);
    }

    #[test]
    fn invalid_option_is_reported() {
        let (mut s, _params) = mini_session(false);
        let err = s.set_option_str("presolve_licq_action", "bogus");
        assert!(matches!(err, Err(SessionError::InvalidOption { .. })));
    }

    #[test]
    fn index_types_line_up() {
        let want: Index = 2;
        let (mut s, _params) = mini_session(true);
        let sol = s.solve_cold().expect("cold");
        assert_eq!(sol.x.len() as Index, want);
    }
}
