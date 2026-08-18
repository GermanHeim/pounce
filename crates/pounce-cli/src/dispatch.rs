//! Solver routing for the LP/QP/QCQP dispatch.
//!
//! See `dev-notes/lp-qp-routing.md`. This module sits between problem
//! loading and the call to `optimize_tnlp`. It does three things:
//!
//! 1. **Classify** the parsed problem into a [`ProblemClass`] by walking
//!    the nonlinear expression trees the `.nl` reader already produced.
//! 2. **Resolve** that class against the user's `solver_selection`
//!    option into a [`SolverChoice`].
//! 3. **Dispatch** to the chosen solver (in `main.rs`).
//!
//! All solvers are wired: `auto` routes an LP/convex-QP to `pounce-convex`'s
//! interior-point solver, a convex QCQP to the same crate's conic (SOCP)
//! driver, and everything else to the existing filter-IPM (`Nlp`).
//!
//! ## Classification
//!
//! The `.nl` format has no dedicated quadratic section: each row's
//! linear part lives in the `G`/`J` coefficient segments (already split
//! out into [`NlProblem::obj_linear`] / [`NlProblem::con_linear`]),
//! while any higher-order term — including a QP's quadratic terms — is
//! written into the nonlinear expression tree as `Mul`/`Pow` nodes. So:
//!
//! - no nonlinear parts at all → **LP**;
//! - all nonlinear parts are degree-2 polynomials → **QP** family
//!   (convex / nonconvex / QCQP split by curvature);
//! - anything else (transcendental, higher degree) → **NLP**.
//!
//! ### Conservative fallback (correctness guard)
//!
//! Misclassifying an indefinite or non-quadratic problem *into* a convex
//! solver would return a spurious KKT point as if globally optimal.
//! Whenever the walk cannot *prove* the stronger class, the classifier
//! falls back to the more general one, ultimately `Nlp`. The convexity
//! (PSD) test uses a tolerance and routes "inconclusive within
//! tolerance" to the safe side, never to the convex path.

use crate::nl_reader::NlProblem;
use pounce_common::types::{lower_bound_present, upper_bound_present};

/// Tolerance for the smallest-eigenvalue sign test in the convexity
/// check. A Hessian eigenvalue below `-PSD_TOL` is treated as a genuine
/// negative direction (nonconvex); within `±PSD_TOL` it is treated as
/// zero. Scaled tolerances would be better once we have problem scaling
/// in this path; a fixed absolute tolerance is adequate here and errs
/// toward the safe (more general) class.
const PSD_TOL: f64 = 1e-9;

/// Budget on the **structural** cost of putting a convex QCQP into conic form,
/// above which it is routed to the general NLP solver instead.
///
/// This is one of **two** independent guards on the conic path, and the split
/// is the point. The `n · m` budget it replaces was silently doing two jobs:
/// bounding the reformulation, and bounding the conic solve itself. Only the
/// first was ever explained, and only the first has been fixed.
///
/// The reformulation cost genuinely used to scale with the problem's width —
/// [`crate::qp_extract::extract_socp_with_map`] built a dense `n×n` Hessian and
/// an `n`-column factor *per quadratic row*. It no longer does: rows are
/// factored on their own support, and a diagonal row is factored in `O(k)`. So
/// the model is now the actual work performed:
///
/// ```text
/// Σ_rows  k³  if the row's Hessian has off-diagonal entries
///         k   if it is diagonal (one √d per entry; no factorization)
/// ```
///
/// where `k` is the number of variables that row couples. Units are
/// floating-point operations, so the budget is a real time bound rather than a
/// dimensionless guess. Measured: `qssp180` costs 1.96e5 flops and `nql180`
/// 6.48e4 — both three orders of magnitude inside this budget, where the old
/// proxy scored them at `n · m` ≈ 1e10 and rejected them.
///
/// **The value is deliberately conservative.** `2e7` sits just above `256³`,
/// the per-row width the previous guard allowed, so every problem that routed
/// to NLP for *reformulation* reasons still does — including the
/// `qcqp1000-*`/`qcqp1500-*` rows, which Q0 measured solving well on the NLP
/// path. Raising it to admit the dense thousand-variable rows is a separate
/// experiment, now cheap to run because the `k³` term states its price.
const SOCP_REFORM_FLOP_BUDGET: u128 = 20_000_000;

/// The second guard: an **empirical** cap on the size of problem handed to the
/// conic solve, independent of how cheap the reformulation is.
///
/// This exists because measurement, not theory, says so. With the extractor
/// fixed, `qssp180` and `nql180` reformulate almost for free and the conic
/// solver reaches an accurate optimum on both — and is *slower* doing it:
///
/// ```text
///            NLP filter-IPM        conic IPM
/// qssp180    47.1 s / 27 it        178.5 s / 73 it     3.8x slower
/// nql180     57.8 s / 36 it        156.5 s / 83 it     2.7x slower
/// ```
///
/// Both reach `Optimal` with KKT error ≤ 1.5e-12, so this is a performance
/// choice and not a soundness one. The conic path takes roughly 2.5x the
/// iterations at this scale; until that is understood, the larger problems keep
/// the solver that measurably wins.
///
/// `1e8` reproduces the routing the `n · m` proxy happened to produce for these
/// two instances — the right answer for a reason that was never stated. It is a
/// placeholder for a real conic-solve cost model, and it is deliberately the
/// *only* thing still keyed to `n · m`.
const SOCP_SOLVE_SIZE_CAP: u64 = 100_000_000;

/// The mathematical class of a loaded problem, from most to least
/// specialized. See the module docs and `dev-notes/lp-qp-routing.md`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProblemClass {
    /// Linear objective, linear constraints.
    Lp,
    /// Convex quadratic objective, linear constraints (Hessian PSD).
    ConvexQp,
    /// Convex quadratic objective and/or convex quadratic constraints.
    /// SOCP-representable; routes to the conic (SOCP) interior-point solver.
    ConvexQcqp,
    /// Quadratic but with an indefinite Hessian somewhere. Falls through
    /// to the NLP solver for a local minimum.
    NonconvexQp,
    /// General nonlinear (transcendental terms, higher-degree
    /// polynomials, or anything the classifier cannot prove quadratic).
    Nlp,
}

impl ProblemClass {
    /// Human-readable name for diagnostics and the
    /// forced-solver-mismatch error message.
    pub fn name(self) -> &'static str {
        match self {
            ProblemClass::Lp => "LP",
            ProblemClass::ConvexQp => "convex QP",
            ProblemClass::ConvexQcqp => "convex QCQP",
            ProblemClass::NonconvexQp => "nonconvex QP",
            ProblemClass::Nlp => "NLP",
        }
    }
}

/// Why [`classify_problem`] reached the class it did.
///
/// Every arm is a place the classifier *stops* — either because it proved a
/// class or because it could not, and fell back to the more general one. The
/// distinction matters most for a convex QCQP: four different findings
/// (a nonconvex row, a nonconvex *sense*, an unaffordable reformulation, an
/// oversized conic solve) all route to `Nlp`, and a user staring at
/// `Problem class: NLP` on a model they know is a QCQP has so far had no way
/// to tell which. `POUNCE_DBG_CLASSIFY=1` prints it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClassReason {
    /// Neither the objective nor any row carries a nonlinear part.
    NoNonlinearParts,
    /// Nonlinear parts exist but all of them expanded to nothing of
    /// degree 2 or higher — the model is linear after expansion.
    NonlinearPartsCancelled,
    /// The objective's nonlinear part is not a degree-2 polynomial.
    ObjectiveNotQuadratic,
    /// Row `row`'s nonlinear part is not a degree-2 polynomial.
    ConstraintNotQuadratic { row: usize },
    /// The sense-adjusted objective Hessian has a negative eigenvalue.
    ObjectiveHessianIndefinite,
    /// Row `row` is quadratic with a PSD Hessian, but its bound sense
    /// (`>=`, `=`, or two-sided) carves a nonconvex feasible set.
    ConstraintSenseNonconvex { row: usize },
    /// Row `row`'s quadratic Hessian is not PSD.
    ConstraintHessianIndefinite { row: usize },
    /// Convex QCQP, but the cone reformulation exceeds
    /// [`SOCP_REFORM_FLOP_BUDGET`].
    QcqpReformTooCostly { flops: u128 },
    /// Convex QCQP whose reformulation is affordable, but whose conic
    /// solve exceeds [`SOCP_SOLVE_SIZE_CAP`].
    QcqpTooLargeToSolve { size: u64 },
    /// Convex quadratic objective, linear constraints.
    ConvexQuadraticObjective,
    /// Convex QCQP inside both guards — the conic path is taken.
    ConvexQcqpWithinBudgets { flops: u128, size: u64 },
}

impl ClassReason {
    /// One-line explanation, for the `POUNCE_DBG_CLASSIFY` log.
    pub fn explain(self) -> String {
        match self {
            ClassReason::NoNonlinearParts => {
                "no nonlinear part in the objective or any row".to_string()
            }
            ClassReason::NonlinearPartsCancelled => {
                "every nonlinear part expanded to a linear (or constant) polynomial".to_string()
            }
            ClassReason::ObjectiveNotQuadratic => {
                "the objective's nonlinear part is not a degree-2 polynomial".to_string()
            }
            ClassReason::ConstraintNotQuadratic { row } => {
                format!("row {row}'s nonlinear part is not a degree-2 polynomial")
            }
            ClassReason::ObjectiveHessianIndefinite => {
                "the objective Hessian (sense-adjusted for minimization) is not PSD".to_string()
            }
            ClassReason::ConstraintSenseNonconvex { row } => format!(
                "row {row} is a convex quadratic but its bound sense (>=, =, or \
                 two-sided) makes the feasible set nonconvex"
            ),
            ClassReason::ConstraintHessianIndefinite { row } => {
                format!("row {row}'s quadratic Hessian is not PSD")
            }
            ClassReason::QcqpReformTooCostly { flops } => format!(
                "convex QCQP downgraded: cone reformulation costs {flops} flops \
                 (budget {SOCP_REFORM_FLOP_BUDGET})"
            ),
            ClassReason::QcqpTooLargeToSolve { size } => format!(
                "convex QCQP downgraded: conic solve size n·m = {size} \
                 (cap {SOCP_SOLVE_SIZE_CAP})"
            ),
            ClassReason::ConvexQuadraticObjective => {
                "convex quadratic objective, linear rows".to_string()
            }
            ClassReason::ConvexQcqpWithinBudgets { flops, size } => format!(
                "convex QCQP inside both guards: reformulation {flops} flops \
                 (budget {SOCP_REFORM_FLOP_BUDGET}), conic solve size n·m = {size} \
                 (cap {SOCP_SOLVE_SIZE_CAP})"
            ),
        }
    }
}

/// The resolved solver to dispatch to, after combining a
/// [`ProblemClass`] with the `solver_selection` option.
///
/// `auto` resolves an LP/convex-QP to [`SolverChoice::LpIpm`]/[`SolverChoice::QpIpm`],
/// a convex QCQP to [`SolverChoice::SocpIpm`], and everything else to
/// [`SolverChoice::Nlp`]; a forced `solver_selection` can pin any of them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SolverChoice {
    /// The existing Wächter-Biegler filter-IPM.
    Nlp,
    /// LP interior-point in `pounce-convex`.
    LpIpm,
    /// Convex-QP interior-point in `pounce-convex`.
    QpIpm,
    /// Conic (SOCP) IPM in `pounce-convex`: convex QCQP, reformulated to
    /// second-order cones.
    SocpIpm,
    /// Active-set QP in `pounce-qp` (parallel track).
    QpActiveSet,
}

impl SolverChoice {
    /// Human-readable description of the dispatched solver, for the
    /// banner-level "Solving as …" log line. Names the algorithm and the
    /// crate that implements it so a reader can tell which of pounce's
    /// solvers actually ran.
    pub fn describe(self) -> &'static str {
        match self {
            SolverChoice::Nlp => "NLP filter line-search interior-point (pounce-nlp)",
            SolverChoice::LpIpm => "LP interior-point (pounce-convex)",
            SolverChoice::QpIpm => "convex QP interior-point (pounce-convex)",
            SolverChoice::SocpIpm => "convex QCQP conic interior-point (pounce-convex)",
            SolverChoice::QpActiveSet => "active-set QP (pounce-qp)",
        }
    }
}

/// Parsed `solver_selection` option value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SolverSelection {
    /// Pick the most specialized solver matching the class. Default.
    Auto,
    /// Force the NLP solver regardless of class (current behavior).
    Nlp,
    /// Force IPM-LP; error if the problem is not an LP.
    LpIpm,
    /// Force IPM-QP; error if the problem is not LP/convex-QP.
    QpIpm,
    /// Force the conic (SOCP) IPM; error if the problem is not a convex
    /// LP / QP / QCQP (all of which the conic solver handles).
    Socp,
    /// Force active-set QP; error if the problem is not LP/convex-QP.
    QpActiveSet,
}

impl SolverSelection {
    /// Parse the `solver_selection` option string. Returns `None` for an
    /// unrecognized value so the caller can surface a tidy error.
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "auto" => Some(SolverSelection::Auto),
            "nlp" => Some(SolverSelection::Nlp),
            "lp-ipm" => Some(SolverSelection::LpIpm),
            "qp-ipm" => Some(SolverSelection::QpIpm),
            "socp" => Some(SolverSelection::Socp),
            "qp-active-set" => Some(SolverSelection::QpActiveSet),
            _ => None,
        }
    }

    /// The accepted values, for error messages and option registration.
    pub const VALUES: &'static [&'static str] =
        &["auto", "nlp", "lp-ipm", "qp-ipm", "socp", "qp-active-set"];
}

/// Classify a parsed `.nl` problem.
///
/// Works off the already-split linear / nonlinear representation in
/// [`NlProblem`]: a row contributes to the class only through its
/// nonlinear `Expr` (the linear part is, by construction, linear). The
/// classifier is deliberately conservative — see the module docs.
pub fn classify_problem(prob: &NlProblem) -> ProblemClass {
    classify_problem_explained(prob).0
}

/// [`classify_problem`], plus the finding that produced the class.
///
/// Also the single place the `POUNCE_DBG_CLASSIFY=1` routing log is
/// emitted, so every caller — including the tests — sees the same line.
pub fn classify_problem_explained(prob: &NlProblem) -> (ProblemClass, ClassReason) {
    let verdict = classify_inner(prob);
    if std::env::var_os("POUNCE_DBG_CLASSIFY").is_some() {
        eprintln!(
            "pounce: problem class {} — {} [{}]",
            verdict.0.name(),
            verdict.1.explain(),
            header_census(prob)
        );
    }
    verdict
}

/// The `.nl` header's nonlinearity census next to what the parsed trees
/// actually say, for the `POUNCE_DBG_CLASSIFY` line.
///
/// The two are allowed to differ in exactly one direction — the header can
/// over-state, because `parse_nl_text` folds a constant `C` body into the
/// row bounds after AMPL took its census (`gh #492`). A header that
/// *under*-states is a non-conforming writer, and the note says so rather
/// than letting the discrepancy pass silently; nothing in the classifier
/// trusts the header, so it is a diagnostic, not a guard.
fn header_census(prob: &NlProblem) -> String {
    let tree_rows = prob
        .con_nonlinear
        .iter()
        .filter(|e| !is_trivially_zero(e))
        .count();
    let tree_obj = usize::from(!is_trivially_zero(&prob.obj_nonlinear));
    let Some(c) = prob.nl_counts else {
        return format!("no .nl header census; trees: nl_rows={tree_rows} nl_obj={tree_obj}");
    };
    let flag = if c.nl_cons < tree_rows || c.nl_objs < tree_obj {
        " — HEADER UNDER-STATES the trees; writer is non-conforming"
    } else {
        ""
    };
    format!(
        "header nlc={} nlo={} nlvc={} nlvo={} nlvb={} ({} of {} vars nonlinear); \
         trees: nl_rows={tree_rows} nl_obj={tree_obj}{flag}",
        c.nl_cons,
        c.nl_objs,
        c.nl_vars_cons,
        c.nl_vars_objs,
        c.nl_vars_both,
        c.nonlinear_vars(),
        prob.n,
    )
}

fn classify_inner(prob: &NlProblem) -> (ProblemClass, ClassReason) {
    // Fast path: no nonlinear parts anywhere ⇒ LP.
    //
    // This deliberately stays a walk over the parsed rows rather than the
    // O(1) header read (`nlc == 0 && nlo == 0`) the design note proposed.
    // Two reasons, both found by writing it out: the test is already O(1)
    // *per row* — `is_trivially_zero` matches the root node, it does not
    // walk an `Expr` — so the header saves a pointer compare per row and
    // nothing else; and the header answer is the writer's claim, while this
    // one is a fact about the trees pounce will evaluate. Trusting the
    // claim would route a model to the LP solver on the strength of a
    // header field, which is not a trade this classifier makes anywhere
    // else. The header is logged beside the verdict instead.
    let obj_nl = !is_trivially_zero(&prob.obj_nonlinear);
    let cons_nl = prob.con_nonlinear.iter().any(|e| !is_trivially_zero(e));
    if !obj_nl && !cons_nl {
        return (ProblemClass::Lp, ClassReason::NoNonlinearParts);
    }

    // Objective curvature.
    let obj_quad = match analyze_quadratic(&prob.obj_nonlinear) {
        Some(q) => q,
        // Objective has a non-quadratic nonlinear term ⇒ NLP.
        None => return (ProblemClass::Nlp, ClassReason::ObjectiveNotQuadratic),
    };

    // Constraint curvature. A quadratic constraint makes this a QCQP;
    // any non-quadratic constraint term makes the whole problem NLP.
    let mut any_quadratic_constraint = false;
    for (row, c) in prob.con_nonlinear.iter().enumerate() {
        if is_trivially_zero(c) {
            continue;
        }
        match analyze_quadratic(c) {
            Some(q) if q.is_empty() => {} // purely linear after all
            Some(_) => any_quadratic_constraint = true,
            None => {
                return (
                    ProblemClass::Nlp,
                    ClassReason::ConstraintNotQuadratic { row },
                );
            }
        }
    }

    // Objective Hessian definiteness, as the *minimizer* sees it. A
    // `maximize` problem is internally negated to a minimization, so a
    // concave-up (PSD-Hessian) maximize is a nonconvex minimize. Test the
    // sense-adjusted Hessian, not the raw one, or maximize-of-convex slips
    // through to the convex IPM and produces a wrong (max/saddle) answer.
    if !obj_quad.is_empty() {
        let effective: QuadHessian = if prob.minimize {
            obj_quad.clone()
        } else {
            obj_quad.iter().map(|(k, v)| (*k, -v)).collect()
        };
        if !hessian_is_psd(&effective, prob.n) {
            return (
                ProblemClass::NonconvexQp,
                ClassReason::ObjectiveHessianIndefinite,
            );
        }
    }

    if any_quadratic_constraint {
        // Convex QCQP requires every quadratic constraint to be convex *as a
        // feasible set*, not merely to have a PSD Hessian. A quadratic
        // `g(x) = ½xᵀQx + … ` carves a convex region only when it is a
        // one-sided **upper** bound `g(x) ≤ g_u` *and* `Q ⪰ 0`. The other
        // senses are nonconvex even with a PSD Hessian:
        //   - `g(x) ≥ g_l` (finite lower bound): the super-level set of a
        //     convex function is nonconvex;
        //   - a quadratic equality `g(x) = c`;
        //   - a two-sided range `g_l ≤ g(x) ≤ g_u` (includes the `≥` side).
        // This sense test matters now that ConvexQcqp is dispatched to the
        // conic solver (it is SOC-representable only in the convex case); a
        // misclassified nonconvex row would return a spurious "optimum".
        // Anything not provably convex falls back to NLP (sound: the
        // filter-IPM finds a local minimum either way).
        let mut reform_flops: u128 = 0;
        for (row, c) in prob.con_nonlinear.iter().enumerate() {
            if is_trivially_zero(c) {
                continue;
            }
            match analyze_quadratic(c) {
                Some(q) if q.is_empty() => {} // purely linear after all
                Some(q) => {
                    let lo = prob.g_l[row];
                    let hi = prob.g_u[row];
                    // Presence is directional (gh #401). The symmetric
                    // `|v| < 1e19` test this used to run called a row with a
                    // real bound past the *opposite* sentinel — `g(x) >= 5e20`
                    // arrives as `g_l = 5e20`, `g_u = 1e19` — free on both
                    // sides, and `continue` below then dropped a real
                    // constraint from the convexity decision.
                    let lo_present = lower_bound_present(lo);
                    let hi_present = upper_bound_present(hi);
                    let vacuous = !lo_present && !hi_present;
                    let upper_only = hi_present && !lo_present;
                    if vacuous {
                        // Free row: imposes nothing, so it cannot make the
                        // problem nonconvex. Ignore it.
                        continue;
                    }
                    // Convexity (cheap sparse certificate) gates the QCQP
                    // class; the per-row coupling guard then gates the *conic*
                    // path: a convex but heavily-coupled constraint Hessian is
                    // ruinous to put in SOC form, so route the whole QCQP to
                    // NLP (which solves it soundly) rather than burn the budget
                    // in the reformulation — the mittelmann `qcqp1000-*` rows.
                    if !upper_only {
                        return (
                            ProblemClass::Nlp,
                            ClassReason::ConstraintSenseNonconvex { row },
                        );
                    }
                    if !hessian_is_psd(&q, prob.n) {
                        return (
                            ProblemClass::Nlp,
                            ClassReason::ConstraintHessianIndefinite { row },
                        );
                    }
                    reform_flops = reform_flops.saturating_add(socp_reform_flops(&q));
                }
                None => {
                    return (
                        ProblemClass::Nlp,
                        ClassReason::ConstraintNotQuadratic { row },
                    );
                }
            }
        }
        // Two independent guards, for two different costs. A convex QCQP whose
        // *reformulation* is too expensive falls back to NLP (see
        // `SOCP_REFORM_FLOP_BUDGET`); so does one whose reformulation is cheap
        // but whose *conic solve* is measurably slower than the filter-IPM at
        // that scale (see `SOCP_SOLVE_SIZE_CAP`). Both fall back to a solver
        // that answers the same question soundly, so either is a performance
        // decision only.
        let solve_size = (prob.n as u64).saturating_mul(prob.m as u64);
        let too_costly_to_reform = reform_flops > SOCP_REFORM_FLOP_BUDGET;
        let too_large_to_solve = solve_size > SOCP_SOLVE_SIZE_CAP;
        if std::env::var_os("POUNCE_DBG_SOCP_COST").is_some() {
            eprintln!(
                "pounce: QCQP conic reformulation cost {reform_flops} flops \
                 (budget {SOCP_REFORM_FLOP_BUDGET}), conic solve size n·m \
                 {solve_size} (cap {SOCP_SOLVE_SIZE_CAP}) → {}",
                if too_costly_to_reform || too_large_to_solve {
                    "NLP"
                } else {
                    "ConvexQcqp"
                }
            );
        }
        if too_costly_to_reform {
            return (
                ProblemClass::Nlp,
                ClassReason::QcqpReformTooCostly {
                    flops: reform_flops,
                },
            );
        }
        if too_large_to_solve {
            return (
                ProblemClass::Nlp,
                ClassReason::QcqpTooLargeToSolve { size: solve_size },
            );
        }
        return (
            ProblemClass::ConvexQcqp,
            ClassReason::ConvexQcqpWithinBudgets {
                flops: reform_flops,
                size: solve_size,
            },
        );
    }

    // Quadratic (or linear) convex objective with linear constraints.
    if obj_quad.is_empty() {
        // Objective nonlinear part collapsed to nothing quadratic and no
        // constraints are quadratic — it was effectively linear.
        (ProblemClass::Lp, ClassReason::NonlinearPartsCancelled)
    } else {
        (
            ProblemClass::ConvexQp,
            ClassReason::ConvexQuadraticObjective,
        )
    }
}

/// Resolve a [`ProblemClass`] and a [`SolverSelection`] into the solver
/// to dispatch to, or an error string when a forced selection does not
/// match the detected class.
///
/// `auto` routes LP / convex QP to the convex IPM (`QpIpm`) and convex
/// QCQP to the conic IPM (`SocpIpm`); nonconvex QP and general NLP resolve
/// to `Nlp`. A forced selection that does not match the detected class is
/// rejected with a clear message. (`QpActiveSet` is accepted for LP / convex
/// QP and dispatched to the active-set SQP engine — see `main.rs`.)
pub fn resolve_solver(
    class: ProblemClass,
    selection: SolverSelection,
) -> Result<SolverChoice, String> {
    use ProblemClass as P;
    use SolverSelection as S;

    // Is this class within the convex-QP family (LP or convex QP)?
    let is_lp = class == P::Lp;
    let is_convex_qp = matches!(class, P::Lp | P::ConvexQp);
    // The conic solver handles the whole convex cone family: LP, convex QP,
    // and (reformulated to second-order cones) convex QCQP.
    let is_conic = matches!(class, P::Lp | P::ConvexQp | P::ConvexQcqp);

    match selection {
        // `auto`: route LP and convex QP to the specialized convex IPM
        // (`pounce-convex`) and convex QCQP to the same crate's conic
        // (SOCP) IPM; nonconvex QP and general NLP fall through to the NLP
        // filter-IPM. LP is solved by the same QP IPM (P = 0), so it
        // resolves to `QpIpm` rather than a distinct LP entry point.
        S::Auto => match class {
            P::Lp | P::ConvexQp => Ok(SolverChoice::QpIpm),
            P::ConvexQcqp => Ok(SolverChoice::SocpIpm),
            _ => Ok(SolverChoice::Nlp),
        },
        S::Nlp => Ok(SolverChoice::Nlp),
        S::LpIpm => {
            if is_lp {
                Ok(SolverChoice::LpIpm)
            } else {
                Err(mismatch_msg(class, "lp-ipm", "an LP"))
            }
        }
        S::QpIpm => {
            if is_convex_qp {
                Ok(SolverChoice::QpIpm)
            } else {
                Err(mismatch_msg(class, "qp-ipm", "an LP or convex QP"))
            }
        }
        S::Socp => {
            if is_conic {
                Ok(SolverChoice::SocpIpm)
            } else {
                Err(mismatch_msg(class, "socp", "a convex LP, QP, or QCQP"))
            }
        }
        S::QpActiveSet => {
            if is_convex_qp {
                Ok(SolverChoice::QpActiveSet)
            } else {
                Err(mismatch_msg(class, "qp-active-set", "an LP or convex QP"))
            }
        }
    }
}

fn mismatch_msg(class: ProblemClass, forced: &str, expected: &str) -> String {
    format!(
        "problem class {} does not match forced solver {} (expected {})",
        class.name(),
        forced,
        expected
    )
}

// ---------------------------------------------------------------------
// Quadratic-form analysis
// ---------------------------------------------------------------------

// The recognizer itself lives in `pounce-nl` (`nl_quadratic`), next to the
// `Expr` DAG it walks, so that the consumers that are not this binary can
// use it — see that module's docs. What stays here is the *routing*: which
// `ProblemClass` a recognized form implies, and which guards a QCQP has to
// clear to reach the conic path. These re-exports keep the call sites in
// `qp_extract` and in the tests below spelled as they were.
pub(crate) use pounce_nl::nl_quadratic::{
    QuadHessian, analyze_quadratic, analyze_quadratic_full, is_trivially_zero,
};

// ---------------------------------------------------------------------
// PSD test
// ---------------------------------------------------------------------

/// Number of distinct variables that couple inside a quadratic form — the
/// dimension `k` of the matrix that would be factored.
fn hessian_active_vars(h: &QuadHessian) -> usize {
    let mut active: Vec<usize> = Vec::with_capacity(2 * h.len());
    for (i, j) in h.keys() {
        active.push(*i);
        active.push(*j);
    }
    active.sort_unstable();
    active.dedup();
    active.len()
}

/// Flops [`crate::qp_extract::socp_factor_rows`] will spend putting one
/// quadratic row into cone form.
///
/// A diagonal Hessian takes the `O(k)` path — one `√d` per entry, no
/// factorization — which is why the very large diagonal QCQPs are cheap to
/// reformulate despite their width. Anything with an off-diagonal entry gets a
/// pivoted Cholesky on a dense `k×k`, i.e. `O(k³)`.
fn socp_reform_flops(h: &QuadHessian) -> u128 {
    let k = hessian_active_vars(h) as u128;
    if h.keys().any(|(i, j)| i != j) {
        k.saturating_mul(k).saturating_mul(k)
    } else {
        k
    }
}

/// Is the (symmetric, sparse) Hessian positive semidefinite?
///
/// A purely diagonal Hessian is settled in `O(nnz)` by sign — its
/// eigenvalues *are* its diagonal entries — with no factorization at all;
/// this keeps large separable / least-squares QPs cheap. A *coupled*
/// Hessian is certified by a sparse symmetric factorization (see
/// [`coupled_hessian_is_psd`]): feral's LDLᵀ reports the matrix inertia in
/// roughly `O(nnz · fill)`, so even the large but sparse coupled Hessians of
/// the CVXQP family (n ≈ 1000) are classified in well under the solve cost —
/// no dense `k×k` allocation and no `O(k³)` eigensolve. Returns `true` only
/// when the smallest eigenvalue is `≥ -PSD_TOL`; an indefinite or
/// inconclusive result returns `false`, routing to the safe (more general)
/// class.
fn hessian_is_psd(h: &QuadHessian, _n: usize) -> bool {
    if h.is_empty() {
        return true; // zero matrix is PSD (the linear case)
    }
    // Fast path: a diagonal Hessian is PSD iff every diagonal entry is
    // `≥ -PSD_TOL`. No factorization — essential for large but separable
    // objectives, where the answer is trivial.
    if h.keys().all(|(i, j)| i == j) {
        return h.values().all(|v| *v >= -PSD_TOL);
    }
    coupled_hessian_is_psd(h)
}

/// PSD certificate for a *coupled* Hessian via a sparse symmetric
/// factorization.
///
/// The test is positive-definiteness of the `ε`-shifted matrix `H + ε·I`
/// with `ε = PSD_TOL`. A genuinely-PSD `H` (smallest eigenvalue `λ_min ≥ 0`,
/// even a singular one) becomes strictly positive definite after the shift,
/// so feral factors it with no negative pivots (`inertia.negative == 0`); a
/// truly indefinite `H` with `λ_min < -PSD_TOL` keeps a strictly-negative
/// shifted eigenvalue and yields `negative > 0`. The `negative == 0` test on
/// the shifted matrix is therefore exactly `λ_min ≥ -PSD_TOL` — the same
/// tolerance the dense path used — and it scales to large sparse Hessians
/// because the factorization cost tracks the nonzero/fill count, not a dense
/// `k³`.
///
/// The Hessian is compressed to its active variable set so the factored
/// dimension is `k` (the number of distinct variables in the form). The
/// [`QuadHessian`] is upper-triangular (`i ≤ j`); feral wants the lower
/// triangle (`row ≥ col`), so each entry `(i, j)` is emitted at
/// `(row = j, col = i)`. Every active diagonal is seeded with `ε` (the shift;
/// `from_triplets` sums it with any diagonal entry already in `H`), which
/// also guarantees no structurally empty column. A non-`Success`
/// factorization (singular/fatal — should not occur given the strictly-PD
/// shift, but possible on a pathological form) is treated conservatively as
/// not-provably-PSD.
fn coupled_hessian_is_psd(h: &QuadHessian) -> bool {
    use feral::{CscMatrix, FactorStatus, Solver};

    // Compress to the active variable set so the factored dimension is `k`.
    let mut active: Vec<usize> = Vec::with_capacity(2 * h.len());
    for (i, j) in h.keys() {
        active.push(*i);
        active.push(*j);
    }
    active.sort_unstable();
    active.dedup();
    let k = active.len();
    let idx = |v: usize| active.binary_search(&v).unwrap();

    // Lower-triangle triplets: H's entry (i ≤ j) maps to (row = j, col = i).
    // Capacity covers H's nonzeros plus one ε-shift per active diagonal.
    let mut rows: Vec<usize> = Vec::with_capacity(h.len() + k);
    let mut cols: Vec<usize> = Vec::with_capacity(h.len() + k);
    let mut vals: Vec<f64> = Vec::with_capacity(h.len() + k);
    for ((i, j), v) in h {
        let (ri, rj) = (idx(*i), idx(*j));
        // i ≤ j by the upper-tri convention, so rj ≥ ri ⇒ lower triangle.
        rows.push(rj);
        cols.push(ri);
        vals.push(*v);
    }
    // εI shift: seed every active diagonal (summed with H's own diagonal).
    for d in 0..k {
        rows.push(d);
        cols.push(d);
        vals.push(PSD_TOL);
    }

    let mat = match CscMatrix::from_triplets(k, &rows, &cols, &vals) {
        Ok(m) => m,
        Err(_) => return false, // malformed ⇒ be conservative
    };
    let mut solver = Solver::new();
    match solver.factor(&mat, None) {
        FactorStatus::Success => {
            // PD ⟺ no negative pivots in the LDLᵀ of the ε-shifted matrix.
            solver.inertia().map(|i| i.negative == 0).unwrap_or(false)
        }
        // Singular / wrong-inertia / fatal: cannot certify ⇒ safe fallback.
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nl_reader::{BinOp, Expr, UnaryOp, parse_nl_text};

    // --- SolverSelection parsing ---

    #[test]
    fn parse_selection_values() {
        assert_eq!(SolverSelection::parse("auto"), Some(SolverSelection::Auto));
        assert_eq!(SolverSelection::parse("nlp"), Some(SolverSelection::Nlp));
        assert_eq!(
            SolverSelection::parse("lp-ipm"),
            Some(SolverSelection::LpIpm)
        );
        assert_eq!(
            SolverSelection::parse("qp-ipm"),
            Some(SolverSelection::QpIpm)
        );
        assert_eq!(
            SolverSelection::parse("qp-active-set"),
            Some(SolverSelection::QpActiveSet)
        );
        assert_eq!(SolverSelection::parse("lp-simplex"), None);
        assert_eq!(SolverSelection::parse("bogus"), None);
    }

    // --- resolve_solver: auto routes LP/convex-QP to the convex IPM,
    // everything else to NLP ---

    #[test]
    fn auto_routes_convex_qp_family_to_qp_ipm() {
        assert_eq!(
            resolve_solver(ProblemClass::Lp, SolverSelection::Auto),
            Ok(SolverChoice::QpIpm),
            "auto should route LP to the convex IPM (P=0)"
        );
        assert_eq!(
            resolve_solver(ProblemClass::ConvexQp, SolverSelection::Auto),
            Ok(SolverChoice::QpIpm),
            "auto should route convex QP to the convex IPM"
        );
    }

    #[test]
    fn auto_routes_convex_qcqp_to_socp() {
        assert_eq!(
            resolve_solver(ProblemClass::ConvexQcqp, SolverSelection::Auto),
            Ok(SolverChoice::SocpIpm),
            "auto should route convex QCQP to the conic IPM"
        );
    }

    #[test]
    fn auto_routes_nonconvex_to_nlp() {
        for class in [ProblemClass::NonconvexQp, ProblemClass::Nlp] {
            assert_eq!(
                resolve_solver(class, SolverSelection::Auto),
                Ok(SolverChoice::Nlp),
                "auto must resolve to Nlp for {:?}",
                class
            );
        }
    }

    #[test]
    fn forced_socp_accepts_convex_cone_family_only() {
        for class in [
            ProblemClass::Lp,
            ProblemClass::ConvexQp,
            ProblemClass::ConvexQcqp,
        ] {
            assert_eq!(
                resolve_solver(class, SolverSelection::Socp),
                Ok(SolverChoice::SocpIpm),
                "socp should accept {:?}",
                class
            );
        }
        assert!(resolve_solver(ProblemClass::NonconvexQp, SolverSelection::Socp).is_err());
        assert!(resolve_solver(ProblemClass::Nlp, SolverSelection::Socp).is_err());
    }

    #[test]
    fn forced_nlp_always_ok() {
        assert_eq!(
            resolve_solver(ProblemClass::ConvexQp, SolverSelection::Nlp),
            Ok(SolverChoice::Nlp)
        );
    }

    #[test]
    fn forced_lp_on_nlp_errors() {
        let err = resolve_solver(ProblemClass::Nlp, SolverSelection::LpIpm).unwrap_err();
        assert!(err.contains("NLP"), "msg should name detected class: {err}");
        assert!(
            err.contains("lp-ipm"),
            "msg should name forced solver: {err}"
        );
    }

    #[test]
    fn forced_lp_on_lp_ok() {
        assert_eq!(
            resolve_solver(ProblemClass::Lp, SolverSelection::LpIpm),
            Ok(SolverChoice::LpIpm)
        );
    }

    #[test]
    fn forced_qp_accepts_lp_and_convex_qp_only() {
        assert_eq!(
            resolve_solver(ProblemClass::Lp, SolverSelection::QpIpm),
            Ok(SolverChoice::QpIpm)
        );
        assert_eq!(
            resolve_solver(ProblemClass::ConvexQp, SolverSelection::QpIpm),
            Ok(SolverChoice::QpIpm)
        );
        assert!(resolve_solver(ProblemClass::NonconvexQp, SolverSelection::QpIpm).is_err());
        assert!(resolve_solver(ProblemClass::Nlp, SolverSelection::QpIpm).is_err());
    }

    // --- Poly / quadratic analysis unit tests ---

    #[test]
    fn poly_of_quadratic_diagonal() {
        // (x0 - 1)^2  =>  x0^2 - 2 x0 + 1
        let e = Expr::Binary(
            BinOp::Pow,
            Box::new(Expr::Binary(
                BinOp::Sub,
                Box::new(Expr::Var(0)),
                Box::new(Expr::Const(1.0)),
            )),
            Box::new(Expr::Const(2.0)),
        );
        let h = analyze_quadratic(&e).expect("degree-2 polynomial");
        // d²/dx0² (x0²) = 2
        assert_eq!(h.get(&(0, 0)), Some(&2.0));
    }

    #[test]
    fn poly_rejects_transcendental() {
        // sin(x0) is not polynomial.
        let e = Expr::Unary(UnaryOp::Sin, Box::new(Expr::Var(0)));
        assert!(analyze_quadratic(&e).is_none());
    }

    #[test]
    fn poly_rejects_cubic() {
        // x0^3
        let e = Expr::Binary(
            BinOp::Pow,
            Box::new(Expr::Var(0)),
            Box::new(Expr::Const(3.0)),
        );
        assert!(analyze_quadratic(&e).is_none());
    }

    #[test]
    fn cross_term_hessian() {
        // x0 * x1  =>  H[0,1] = 1
        let e = Expr::Binary(BinOp::Mul, Box::new(Expr::Var(0)), Box::new(Expr::Var(1)));
        let h = analyze_quadratic(&e).expect("degree-2");
        assert_eq!(h.get(&(0, 1)), Some(&1.0));
    }

    #[test]
    fn large_quadratic_sum_lowers_without_quadratic_blowup() {
        // Regression guard for the `solver_selection=auto` classifier hang
        // (mittelmann QCQP/bearing_400/qssp180 emitted zero iterations and
        // burned the full CPU budget). A quadratic expressed as a large
        // `Sum` of monomials must lower in O(N log N): the recognizer used
        // to re-scan the whole accumulated polynomial for zeros on every
        // merged item, so an N-monomial sum was O(N²) and spun for >300 s
        // before the solver started (Ipopt solved the same problems in
        // seconds). Build a 5000-term sum of distinct squares and confirm
        // the full diagonal Hessian is recovered.
        //
        // That fix covered the n-ary `Sum` node only. The same quadratic
        // written as a chain of binary `Add`s — which is what a `.nl` writer
        // emitting `o0` produces — kept the re-scan until Q3 moved the
        // per-merge zero check onto the merged terms; see
        // `wide_diagonal_convex_qcqp_keeps_conic` for that shape.
        const N: usize = 5000;
        let terms: Vec<Expr> = (0..N)
            .map(|i| Expr::Binary(BinOp::Mul, Box::new(Expr::Var(i)), Box::new(Expr::Var(i))))
            .collect();
        let e = Expr::Sum(terms);
        let h = analyze_quadratic(&e).expect("degree-2 sum of squares is a QP");
        assert_eq!(h.len(), N, "every xᵢ² contributes one diagonal entry");
        assert_eq!(h.get(&(0, 0)), Some(&2.0));
        assert_eq!(h.get(&(N - 1, N - 1)), Some(&2.0));
    }

    // --- PSD test ---

    #[test]
    fn psd_accepts_convex_separable() {
        // diag(2, 4): both eigenvalues positive.
        let mut h = QuadHessian::new();
        h.insert((0, 0), 2.0);
        h.insert((1, 1), 4.0);
        assert!(hessian_is_psd(&h, 2));
    }

    #[test]
    fn psd_rejects_indefinite() {
        // [[0,1],[1,0]] has eigenvalues ±1.
        let mut h = QuadHessian::new();
        h.insert((0, 1), 1.0);
        assert!(!hessian_is_psd(&h, 2));
    }

    #[test]
    fn psd_accepts_psd_with_zero_eigenvalue() {
        // [[1,1],[1,1]] is PSD (eigenvalues 0 and 2).
        let mut h = QuadHessian::new();
        h.insert((0, 0), 1.0);
        h.insert((0, 1), 1.0);
        h.insert((1, 1), 1.0);
        assert!(hessian_is_psd(&h, 2));
    }

    // --- A1: ±PSD_TOL boundary of the convexity test (silent-misroute guard) ---

    /// The safety-critical case: a *real* negative direction — even a small
    /// one, well beyond `PSD_TOL` — must read non-PSD so an indefinite QP
    /// routes to NLP, never to the convex IPM (which would return a spurious
    /// "optimal" at a saddle/maximum).
    #[test]
    fn psd_rejects_small_but_real_negative_curvature() {
        // diag(2, −1e-3): min eigenvalue −1e-3 ≪ −PSD_TOL.
        let mut h = QuadHessian::new();
        h.insert((0, 0), 2.0);
        h.insert((1, 1), -1e-3);
        assert!(
            !hessian_is_psd(&h, 2),
            "a −1e-3 eigenvalue must read indefinite, not be rounded to PSD"
        );
    }

    /// Pin the threshold at exactly `±PSD_TOL` (1e-9). Within the band the
    /// test rounds a tiny negative eigenvalue to PSD **by design**: a
    /// genuinely semidefinite Hessian whose smallest eigenvalue computes as a
    /// tiny negative (Jacobi roundoff) must not be misread as nonconvex. The
    /// band is far below the error of solving a convex QP with that much
    /// curvature, so it is the sound tradeoff — see the A1 Finding in
    /// `dev-notes/pr70-hardening.md`. (1×1 Hessians are returned exactly, so
    /// this is deterministic.)
    #[test]
    fn psd_threshold_is_psd_tol() {
        let mut just_inside = QuadHessian::new();
        just_inside.insert((0, 0), -1e-10); // |λ| < PSD_TOL ⇒ treated as zero
        assert!(
            hessian_is_psd(&just_inside, 1),
            "−1e-10 is within tolerance and must round to PSD"
        );

        let mut just_outside = QuadHessian::new();
        just_outside.insert((0, 0), -1e-7); // |λ| > PSD_TOL ⇒ genuine negative
        assert!(
            !hessian_is_psd(&just_outside, 1),
            "−1e-7 is beyond tolerance and must read indefinite"
        );
    }

    // --- Sparse-factorization PSD certificate (CVXQP family) ---

    /// A large *diagonal* Hessian must take the O(nnz) sign fast path — no
    /// factorization at all — and read PSD. This is the large separable /
    /// least-squares QP shape (AUG2D, LISWET, …) that stays on the convex
    /// fast path.
    #[test]
    fn large_diagonal_hessian_is_cheap_and_psd() {
        let n = 50_000;
        let mut h = QuadHessian::new();
        for i in 0..n {
            h.insert((i, i), 2.0);
        }
        assert!(
            hessian_is_psd(&h, n),
            "diag(2,…,2) is PSD and must be settled by the O(nnz) sign path"
        );
    }

    /// A large *coupled* convex Hessian (off-diagonal terms over many
    /// variables) is the CVXQP-family shape that the old dense-Jacobi cap
    /// refused to certify (routing it to NLP). The sparse-factorization
    /// certificate now proves it PSD cheaply, so it reaches the convex
    /// solver. This is the regression fix.
    #[test]
    fn large_coupled_convex_hessian_is_certified_psd() {
        let k = 1_000;
        let mut h = QuadHessian::new();
        // Diagonally dominant tridiagonal: SPD. 2 on the diagonal, 0.1 on
        // the off-diagonal coupling chain ⇒ strictly diagonally dominant.
        for i in 0..k {
            h.insert((i, i), 2.0);
        }
        for i in 0..(k - 1) {
            h.insert((i, i + 1), 0.1);
        }
        assert!(
            hessian_is_psd(&h, k),
            "a diagonally-dominant coupled Hessian over {k} vars must be \
             certified PSD by the sparse factorization (CVXQP regression)"
        );
    }

    /// The sparse certificate must still *reject* a large coupled Hessian
    /// that is genuinely indefinite — size does not buy a free pass.
    #[test]
    fn large_coupled_indefinite_hessian_is_rejected() {
        let k = 1_000;
        let mut h = QuadHessian::new();
        for i in 0..k {
            h.insert((i, i), 2.0);
        }
        for i in 0..(k - 1) {
            h.insert((i, i + 1), 0.1);
        }
        // Flip one diagonal strongly negative ⇒ an indefinite direction.
        h.insert((0, 0), -5.0);
        assert!(
            !hessian_is_psd(&h, k),
            "a coupled Hessian with a strong negative-curvature direction \
             must be rejected regardless of size"
        );
    }

    /// A *small* coupled Hessian is certified by the same sparse path.
    #[test]
    fn small_coupled_hessian_is_certified_psd() {
        // [[2, 1], [1, 2]] — eigenvalues 1 and 3, PSD.
        let mut h = QuadHessian::new();
        h.insert((0, 0), 2.0);
        h.insert((0, 1), 1.0);
        h.insert((1, 1), 2.0);
        assert!(hessian_is_psd(&h, 2));
    }

    // --- End-to-end classify_problem on parsed .nl text ---

    /// Minimal `g`-format `.nl` text builder is overkill; instead use the
    /// reader's own fixtures via parse_nl_text on hand-written stubs.
    /// These cover the header LP fast-path and the AST walk.

    #[test]
    fn classify_pure_lp() {
        // minimize x0 + x1 s.t. x0 + x1 <= 1, no nonlinear parts.
        // Build an NlProblem directly for a hermetic test.
        let prob = NlProblem {
            n: 2,
            m: 1,
            num_obj: 1,
            minimize: true,
            obj_nonlinear: Expr::Const(0.0),
            obj_linear: vec![(0, 1.0), (1, 1.0)],
            obj_constant: 0.0,
            con_nonlinear: vec![Expr::Const(0.0)],
            con_linear: vec![vec![(0, 1.0), (1, 1.0)]],
            x_l: vec![0.0, 0.0],
            x_u: vec![f64::INFINITY, f64::INFINITY],
            g_l: vec![f64::NEG_INFINITY],
            g_u: vec![1.0],
            x0: vec![0.0, 0.0],
            lambda0: vec![0.0],
            suffixes: Default::default(),
            imported_funcs: Vec::new(),
            ampl_options: Vec::new(),
            nl_counts: None,
            var_names: Vec::new(),
            con_names: Vec::new(),
        };
        assert_eq!(classify_problem(&prob), ProblemClass::Lp);
    }

    #[test]
    fn classify_convex_qp() {
        // minimize x0^2 + x1^2 s.t. linear; convex (H = diag(2,2)).
        let obj = Expr::Binary(
            BinOp::Add,
            Box::new(Expr::Binary(
                BinOp::Pow,
                Box::new(Expr::Var(0)),
                Box::new(Expr::Const(2.0)),
            )),
            Box::new(Expr::Binary(
                BinOp::Pow,
                Box::new(Expr::Var(1)),
                Box::new(Expr::Const(2.0)),
            )),
        );
        let prob = qp_stub(obj, vec![Expr::Const(0.0)]);
        assert_eq!(classify_problem(&prob), ProblemClass::ConvexQp);
    }

    /// **gh #401.** A quadratic row whose real bound lies past the *opposite*
    /// sentinel must not be waved through as a free row.
    ///
    /// `x0² + x1² >= 5e20` arrives as `g_l = 5e20` (real), `g_u = 1e19`
    /// (the absent-upper sentinel). The symmetric `|v| < 1e19` test called
    /// *both* sides infinite, so `vacuous` was true and the row was skipped
    /// with "Free row: imposes nothing" — and the model then classified as a
    /// convex QCQP and went to the conic solver as if the constraint were not
    /// there. It is a reverse-convex row: the honest answer is NLP.
    #[test]
    fn a_quadratic_row_bounded_past_the_sentinel_is_not_vacuous() {
        let con = Expr::Binary(
            BinOp::Add,
            Box::new(Expr::Binary(
                BinOp::Pow,
                Box::new(Expr::Var(0)),
                Box::new(Expr::Const(2.0)),
            )),
            Box::new(Expr::Binary(
                BinOp::Pow,
                Box::new(Expr::Var(1)),
                Box::new(Expr::Const(2.0)),
            )),
        );
        let mut prob = qp_stub(Expr::Const(0.0), vec![con]);
        prob.obj_linear = vec![(0, 1.0)];
        prob.g_l = vec![5e20]; // real lower bound
        prob.g_u = vec![1e19]; // absent-upper sentinel
        assert_eq!(
            classify_problem(&prob),
            ProblemClass::Nlp,
            "a `>=` quadratic row is reverse-convex and must route to NLP; \
             treating it as a free row sent the model to the conic solver \
             with the constraint silently dropped"
        );
    }

    #[test]
    fn classify_nonconvex_qp() {
        // minimize x0 * x1 (indefinite Hessian) s.t. linear.
        let obj = Expr::Binary(BinOp::Mul, Box::new(Expr::Var(0)), Box::new(Expr::Var(1)));
        let prob = qp_stub(obj, vec![Expr::Const(0.0)]);
        assert_eq!(classify_problem(&prob), ProblemClass::NonconvexQp);
    }

    #[test]
    fn classify_nlp_from_transcendental_objective() {
        let obj = Expr::Unary(UnaryOp::Exp, Box::new(Expr::Var(0)));
        let prob = qp_stub(obj, vec![Expr::Const(0.0)]);
        assert_eq!(classify_problem(&prob), ProblemClass::Nlp);
    }

    /// Regression: a `maximize` of a PSD-Hessian objective is a *concave*
    /// maximization ⇒ nonconvex minimization. The convexity test must run
    /// on the sense-adjusted Hessian, or this slips through to the convex
    /// IPM and returns a wrong (maximum/saddle) answer.
    #[test]
    fn classify_maximize_psd_objective_is_nonconvex() {
        // maximize x0^2 + x1^2 (H = diag(2,2), PSD) — concave max.
        let obj = Expr::Binary(
            BinOp::Add,
            Box::new(Expr::Binary(
                BinOp::Pow,
                Box::new(Expr::Var(0)),
                Box::new(Expr::Const(2.0)),
            )),
            Box::new(Expr::Binary(
                BinOp::Pow,
                Box::new(Expr::Var(1)),
                Box::new(Expr::Const(2.0)),
            )),
        );
        let mut prob = qp_stub(obj, vec![Expr::Const(0.0)]);
        prob.minimize = false;
        assert_eq!(classify_problem(&prob), ProblemClass::NonconvexQp);
    }

    /// Mirror: `maximize` of a concave (NSD-Hessian) objective is a convex
    /// minimization once negated, so it is a legitimate `ConvexQp`.
    #[test]
    fn classify_maximize_concave_objective_is_convex() {
        // maximize −(x0^2 + x1^2) (H = diag(−2,−2)); negated ⇒ PSD.
        let neg_sq = |v: usize| {
            Expr::Unary(
                UnaryOp::Neg,
                Box::new(Expr::Binary(
                    BinOp::Pow,
                    Box::new(Expr::Var(v)),
                    Box::new(Expr::Const(2.0)),
                )),
            )
        };
        let obj = Expr::Binary(BinOp::Add, Box::new(neg_sq(0)), Box::new(neg_sq(1)));
        let mut prob = qp_stub(obj, vec![Expr::Const(0.0)]);
        prob.minimize = false;
        assert_eq!(classify_problem(&prob), ProblemClass::ConvexQp);
    }

    #[test]
    fn classify_convex_qcqp() {
        // convex quadratic objective + a convex quadratic constraint.
        let obj = Expr::Binary(
            BinOp::Pow,
            Box::new(Expr::Var(0)),
            Box::new(Expr::Const(2.0)),
        );
        let con = Expr::Binary(
            BinOp::Add,
            Box::new(Expr::Binary(
                BinOp::Pow,
                Box::new(Expr::Var(0)),
                Box::new(Expr::Const(2.0)),
            )),
            Box::new(Expr::Binary(
                BinOp::Pow,
                Box::new(Expr::Var(1)),
                Box::new(Expr::Const(2.0)),
            )),
        );
        let prob = qp_stub(obj, vec![con]);
        assert_eq!(classify_problem(&prob), ProblemClass::ConvexQcqp);
    }

    /// Build a convex QCQP (linear objective + one convex quadratic
    /// constraint `x0² ≤ 1`) at an arbitrary declared `n`/`m`, padding the
    /// extra constraints with trivially-zero rows. Used to exercise the
    /// two routing caps (`SOCP_REFORM_FLOP_BUDGET`, `SOCP_SOLVE_SIZE_CAP`)
    /// without allocating `n×n` data.
    fn convex_qcqp_at_size(n: usize, m: usize) -> NlProblem {
        let mut con_nonlinear = vec![Expr::Const(0.0); m];
        con_nonlinear[0] = Expr::Binary(
            BinOp::Pow,
            Box::new(Expr::Var(0)),
            Box::new(Expr::Const(2.0)),
        );
        let g_l = vec![f64::NEG_INFINITY; m];
        let mut g_u = vec![f64::INFINITY; m];
        g_u[0] = 1.0; // upper-only bound ⇒ convex feasible set
        NlProblem {
            n,
            m,
            num_obj: 1,
            minimize: true,
            obj_nonlinear: Expr::Const(0.0),
            obj_linear: vec![(0, 1.0)],
            obj_constant: 0.0,
            con_nonlinear,
            con_linear: vec![vec![]; m],
            x_l: vec![f64::NEG_INFINITY; n],
            x_u: vec![f64::INFINITY; n],
            g_l,
            g_u,
            x0: vec![0.0; n],
            lambda0: vec![0.0; m],
            suffixes: Default::default(),
            imported_funcs: Vec::new(),
            ampl_options: Vec::new(),
            nl_counts: None,
            var_names: Vec::new(),
            con_names: Vec::new(),
        }
    }

    /// A convex QCQP small enough to keep the conic path (n·m ≤ budget).
    #[test]
    fn small_convex_qcqp_routes_to_conic() {
        let prob = convex_qcqp_at_size(100, 100); // n·m = 1e4 ≪ budget
        assert_eq!(classify_problem(&prob), ProblemClass::ConvexQcqp);
    }

    /// The two guards are independent, and this problem shows why both are
    /// needed. Its one quadratic row is `x0² ≤ 1` — a single variable,
    /// diagonal, one flop to put in cone form — so the *reformulation* guard
    /// passes easily. It is still routed to NLP, by `SOCP_SOLVE_SIZE_CAP`,
    /// because at 1e4 × 1e4 the conic solve itself measured slower than the
    /// filter-IPM on exactly this shape (`nql180`, `qssp180`).
    ///
    /// Keeping the two apart matters: the old single `n·m` proxy conflated
    /// them, so fixing the extractor's cost premise silently moved a *routing*
    /// decision that measurement says should not move.
    #[test]
    fn large_qcqp_cheap_to_reform_still_falls_back_on_solve_size() {
        let prob = convex_qcqp_at_size(10_001, 10_001);
        assert!((prob.n as u64) * (prob.m as u64) > SOCP_SOLVE_SIZE_CAP);
        assert_eq!(classify_problem(&prob), ProblemClass::Nlp);
    }

    /// The same shape just *under* the solve-size cap keeps the conic path —
    /// confirming the fallback above is the size cap talking and not the
    /// reformulation budget rejecting a one-variable diagonal row.
    #[test]
    fn large_qcqp_under_the_solve_size_cap_keeps_conic() {
        let prob = convex_qcqp_at_size(10_000, 9_000);
        assert!((prob.n as u64) * (prob.m as u64) <= SOCP_SOLVE_SIZE_CAP);
        assert_eq!(classify_problem(&prob), ProblemClass::ConvexQcqp);
    }

    /// Four different findings route a QCQP to `Nlp`, and until now the log
    /// said only "NLP" for all of them. Each guard must name itself.
    #[test]
    fn qcqp_downgrades_say_which_guard_fired() {
        // Solve-size cap: one diagonal row, trivial to reform.
        let (class, reason) = classify_problem_explained(&convex_qcqp_at_size(10_001, 10_001));
        assert_eq!(class, ProblemClass::Nlp);
        assert!(
            matches!(reason, ClassReason::QcqpTooLargeToSolve { size } if size == 10_001 * 10_001),
            "expected the solve-size cap, got {reason:?}"
        );

        // Reformulation budget: one dense row coupling 400 variables, k³ ≫ 2e7.
        let (class, reason) = classify_problem_explained(&coupled_convex_qcqp_with_k_vars(400));
        assert_eq!(class, ProblemClass::Nlp);
        assert!(
            matches!(reason, ClassReason::QcqpReformTooCostly { .. }),
            "expected the reformulation budget, got {reason:?}"
        );

        // Inside both guards: the reason carries the numbers that decided it.
        let (class, reason) = classify_problem_explained(&convex_qcqp_at_size(100, 100));
        assert_eq!(class, ProblemClass::ConvexQcqp);
        assert!(
            matches!(
                reason,
                ClassReason::ConvexQcqpWithinBudgets { size, .. } if size == 10_000
            ),
            "expected a within-budget QCQP, got {reason:?}"
        );
    }

    /// A quadratic row that is convex as a *function* but bounded from
    /// below carves a nonconvex set. That is a different finding from a row
    /// whose Hessian is indefinite, and the two used to be one `return`.
    #[test]
    fn nonconvex_sense_and_indefinite_hessian_are_distinct_reasons() {
        // x0² ≥ 1 — PSD Hessian, nonconvex feasible set.
        let mut prob = convex_qcqp_at_size(10, 10);
        prob.g_l[0] = 1.0;
        prob.g_u[0] = f64::INFINITY;
        let (class, reason) = classify_problem_explained(&prob);
        assert_eq!(class, ProblemClass::Nlp);
        assert_eq!(reason, ClassReason::ConstraintSenseNonconvex { row: 0 });

        // −x0² ≤ 1 — upper-bounded, but the Hessian is negative definite.
        let mut prob = convex_qcqp_at_size(10, 10);
        prob.con_nonlinear[0] = Expr::Unary(
            UnaryOp::Neg,
            Box::new(Expr::Binary(
                BinOp::Pow,
                Box::new(Expr::Var(0)),
                Box::new(Expr::Const(2.0)),
            )),
        );
        let (class, reason) = classify_problem_explained(&prob);
        assert_eq!(class, ProblemClass::Nlp);
        assert_eq!(reason, ClassReason::ConstraintHessianIndefinite { row: 0 });
    }

    /// The LP fast path reports *why* it is an LP, and the two ways of
    /// getting there are told apart: nothing nonlinear at all, versus
    /// nonlinear parts that expanded to nothing quadratic.
    #[test]
    fn lp_reasons_distinguish_absent_from_cancelled() {
        let prob = qp_stub(Expr::Const(0.0), vec![Expr::Const(0.0)]);
        assert_eq!(
            classify_problem_explained(&prob),
            (ProblemClass::Lp, ClassReason::NoNonlinearParts)
        );

        // x0 − x0 in the objective's nonlinear part: present, but degree 0
        // once expanded.
        let cancels = Expr::Binary(BinOp::Sub, Box::new(Expr::Var(0)), Box::new(Expr::Var(0)));
        let prob = qp_stub(cancels, vec![Expr::Const(0.0)]);
        assert_eq!(
            classify_problem_explained(&prob),
            (ProblemClass::Lp, ClassReason::NonlinearPartsCancelled)
        );
    }

    /// Build a convex QCQP whose single quadratic constraint `(Σ xᵢ)² ≤ 1`
    /// couples all `k` variables (a dense rank-1 PSD Hessian over `k` vars),
    /// with `n = k`, `m = 1`. Exercises the per-row conic-reformulation guard
    /// independently of the `n·m` budget.
    fn coupled_convex_qcqp_with_k_vars(k: usize) -> NlProblem {
        // sum = x0 + x1 + … + x_{k-1}
        let mut sum = Expr::Var(0);
        for i in 1..k {
            sum = Expr::Binary(BinOp::Add, Box::new(sum), Box::new(Expr::Var(i)));
        }
        // constraint (Σ xᵢ)² ≤ 1 — convex feasible set, Hessian = 2·(all-ones),
        // PSD (rank 1) and fully coupled across all k variables.
        let con = Expr::Binary(BinOp::Pow, Box::new(sum), Box::new(Expr::Const(2.0)));
        NlProblem {
            n: k,
            m: 1,
            num_obj: 1,
            minimize: true,
            obj_nonlinear: Expr::Const(0.0),
            obj_linear: vec![(0, 1.0)],
            obj_constant: 0.0,
            con_nonlinear: vec![con],
            con_linear: vec![vec![]],
            x_l: vec![f64::NEG_INFINITY; k],
            x_u: vec![f64::INFINITY; k],
            g_l: vec![f64::NEG_INFINITY],
            g_u: vec![1.0],
            x0: vec![0.0; k],
            lambda0: vec![0.0],
            suffixes: Default::default(),
            imported_funcs: Vec::new(),
            ampl_options: Vec::new(),
            nl_counts: None,
            var_names: Vec::new(),
            con_names: Vec::new(),
        }
    }

    /// Build a convex QCQP whose single quadratic constraint `Σ xᵢ² ≤ 1` is
    /// **separable** — a diagonal Hessian over `k` variables, no coupling.
    /// This is the `qssp180`/`nql180` shape: very wide, trivially factorable.
    fn separable_convex_qcqp_with_k_vars(k: usize) -> NlProblem {
        let sq = |i: usize| {
            Expr::Binary(
                BinOp::Pow,
                Box::new(Expr::Var(i)),
                Box::new(Expr::Const(2.0)),
            )
        };
        let mut con = sq(0);
        for i in 1..k {
            con = Expr::Binary(BinOp::Add, Box::new(con), Box::new(sq(i)));
        }
        NlProblem {
            n: k,
            m: 1,
            num_obj: 1,
            minimize: true,
            obj_nonlinear: Expr::Const(0.0),
            obj_linear: vec![(0, 1.0)],
            obj_constant: 0.0,
            con_nonlinear: vec![con],
            con_linear: vec![vec![]],
            x_l: vec![f64::NEG_INFINITY; k],
            x_u: vec![f64::INFINITY; k],
            g_l: vec![f64::NEG_INFINITY],
            g_u: vec![1.0],
            x0: vec![0.0; k],
            lambda0: vec![0.0],
            suffixes: Default::default(),
            imported_funcs: Vec::new(),
            ampl_options: Vec::new(),
            nl_counts: None,
            var_names: Vec::new(),
            con_names: Vec::new(),
        }
    }

    /// The converse: a *small* problem can still be too expensive to
    /// reformulate. This one is `n·m = 300`, but its single row couples all
    /// 300 variables densely, so the pivoted Cholesky costs `300³ = 2.7e7`
    /// flops — over budget. Route to NLP. This is the mittelmann `qcqp1000-*`
    /// shape (few constraints, ~1000-var coupled rows), and Q0 measured those
    /// solving well on the NLP path.
    #[test]
    fn heavily_coupled_convex_qcqp_falls_back_to_nlp() {
        let k = 300;
        let prob = coupled_convex_qcqp_with_k_vars(k);
        assert!((k as u128).pow(3) > SOCP_REFORM_FLOP_BUDGET);
        assert_eq!(classify_problem(&prob), ProblemClass::Nlp);
    }

    /// The companion to the guard: a convex QCQP whose dense row is narrow
    /// enough to factor inside the budget keeps the conic path.
    /// `250³ = 1.56e7 ≤ 2e7`.
    #[test]
    fn lightly_coupled_convex_qcqp_keeps_conic() {
        let k = 250;
        let prob = coupled_convex_qcqp_with_k_vars(k);
        assert!((k as u128).pow(3) <= SOCP_REFORM_FLOP_BUDGET);
        assert_eq!(classify_problem(&prob), ProblemClass::ConvexQcqp);
    }

    /// A diagonal row is factored in `O(k)`, so width alone must not push a
    /// separable QCQP off the conic path — 100 000 uncoupled variables cost
    /// 100 000 flops, where the same width densely coupled would cost 1e15.
    ///
    /// `k` was held at 1 000 by an unrelated defect rather than by the cost
    /// model: the constraint is a left-deep `Add` tree, the recognizer walked
    /// it recursively, and a sum of a few thousand squares overflowed the
    /// stack during classification (Q1 found this; a left-deep `o0` chain is
    /// what a `.nl` writer emits for a long sum). Q3 made the recognizer
    /// iterative, so the cost model can now be tested at a width that means
    /// something. See `pounce_nl::nl_quadratic` for the depth tests
    /// themselves.
    ///
    /// The problem is leaked rather than dropped: `Expr`'s derived `Drop` is
    /// still recursive and would overflow tearing a tree this deep down. That
    /// is a real remaining defect on the same shape — the Python bindings
    /// work around it with a big-stack worker thread (pounce#472) — and it is
    /// not this test's subject.
    #[test]
    fn wide_diagonal_convex_qcqp_keeps_conic() {
        let k = 100_000;
        let prob = separable_convex_qcqp_with_k_vars(k);
        assert_eq!(classify_problem(&prob), ProblemClass::ConvexQcqp);
        std::mem::forget(prob);
    }

    /// Classification mirror of the boundary guard: a QP whose only
    /// curvature is a genuine (beyond-tolerance) negative direction is
    /// `NonconvexQp`, so `auto` routes it to NLP rather than the convex IPM.
    /// `minimize −x0²` is concave for a minimizer ⇒ indefinite.
    #[test]
    fn classify_concave_minimize_is_nonconvex() {
        let obj = Expr::Unary(
            UnaryOp::Neg,
            Box::new(Expr::Binary(
                BinOp::Pow,
                Box::new(Expr::Var(0)),
                Box::new(Expr::Const(2.0)),
            )),
        );
        let prob = qp_stub(obj, vec![Expr::Const(0.0)]);
        assert_eq!(classify_problem(&prob), ProblemClass::NonconvexQp);
    }

    /// Conservative QCQP guard: a convex quadratic objective with an
    /// *indefinite* quadratic constraint must fall back to NLP — never be
    /// called `ConvexQcqp` and handed to the conic path, which would treat a
    /// nonconvex feasible region as convex.
    #[test]
    fn classify_qcqp_with_indefinite_constraint_falls_back_to_nlp() {
        // obj x0² (convex); constraint x0·x1 (indefinite Hessian).
        let obj = Expr::Binary(
            BinOp::Pow,
            Box::new(Expr::Var(0)),
            Box::new(Expr::Const(2.0)),
        );
        let con = Expr::Binary(BinOp::Mul, Box::new(Expr::Var(0)), Box::new(Expr::Var(1)));
        let prob = qp_stub(obj, vec![con]);
        assert_eq!(classify_problem(&prob), ProblemClass::Nlp);
    }

    /// Sense guard: a PSD-Hessian quadratic constraint is convex only as an
    /// **upper** bound. With a finite *lower* bound (`g(x) ≥ g_l`) the
    /// feasible set is the nonconvex super-level set, so it must fall back to
    /// NLP — never be routed to the conic solver as if convex.
    #[test]
    fn classify_psd_quadratic_with_lower_bound_is_nonconvex() {
        let obj = Expr::Binary(
            BinOp::Pow,
            Box::new(Expr::Var(0)),
            Box::new(Expr::Const(2.0)),
        );
        let con = Expr::Binary(
            BinOp::Add,
            Box::new(Expr::Binary(
                BinOp::Pow,
                Box::new(Expr::Var(0)),
                Box::new(Expr::Const(2.0)),
            )),
            Box::new(Expr::Binary(
                BinOp::Pow,
                Box::new(Expr::Var(1)),
                Box::new(Expr::Const(2.0)),
            )),
        );
        let mut prob = qp_stub(obj, vec![con]);
        // g(x) ≥ 1  (finite lower, infinite upper) — convex function, but the
        // ≥ side is a nonconvex region.
        prob.g_l = vec![1.0];
        prob.g_u = vec![f64::INFINITY];
        assert_eq!(classify_problem(&prob), ProblemClass::Nlp);
    }

    /// Sense guard: a quadratic *equality* (`g(x) = c`) is nonconvex even
    /// with a PSD Hessian, so it must fall back to NLP, not ConvexQcqp.
    #[test]
    fn classify_quadratic_equality_is_nonconvex() {
        let obj = Expr::Const(0.0);
        let con = Expr::Binary(
            BinOp::Pow,
            Box::new(Expr::Var(0)),
            Box::new(Expr::Const(2.0)),
        );
        let mut prob = qp_stub(obj, vec![con]);
        prob.g_l = vec![1.0];
        prob.g_u = vec![1.0]; // x0² = 1 — nonconvex.
        assert_eq!(classify_problem(&prob), ProblemClass::Nlp);
    }

    /// A nonlinear objective expression whose quadratic part algebraically
    /// cancels has an empty Hessian ⇒ classify as `Lp`, not a spurious QP
    /// (which would otherwise route a linear problem to the QP IPM).
    #[test]
    fn classify_cancelling_quadratic_objective_is_lp() {
        // x0² − x0²  ≡ 0: the degree-2 terms cancel in the polynomial walk.
        let sq = || {
            Expr::Binary(
                BinOp::Pow,
                Box::new(Expr::Var(0)),
                Box::new(Expr::Const(2.0)),
            )
        };
        let obj = Expr::Binary(BinOp::Sub, Box::new(sq()), Box::new(sq()));
        let prob = qp_stub(obj, vec![Expr::Const(0.0)]);
        assert_eq!(classify_problem(&prob), ProblemClass::Lp);
    }

    #[test]
    fn classify_nlp_from_transcendental_constraint() {
        let obj = Expr::Binary(
            BinOp::Pow,
            Box::new(Expr::Var(0)),
            Box::new(Expr::Const(2.0)),
        );
        let con = Expr::Unary(UnaryOp::Log, Box::new(Expr::Var(1)));
        let prob = qp_stub(obj, vec![con]);
        assert_eq!(classify_problem(&prob), ProblemClass::Nlp);
    }

    /// Build a 2-var, 1-con problem stub with the given nonlinear
    /// objective and per-constraint nonlinear parts. Linear parts and
    /// bounds are filled with benign defaults.
    fn qp_stub(obj_nonlinear: Expr, con_nonlinear: Vec<Expr>) -> NlProblem {
        let m = con_nonlinear.len();
        NlProblem {
            n: 2,
            m,
            num_obj: 1,
            minimize: true,
            obj_nonlinear,
            obj_linear: vec![],
            obj_constant: 0.0,
            con_nonlinear,
            con_linear: vec![vec![]; m],
            x_l: vec![f64::NEG_INFINITY; 2],
            x_u: vec![f64::INFINITY; 2],
            g_l: vec![f64::NEG_INFINITY; m],
            g_u: vec![0.0; m],
            x0: vec![0.0; 2],
            lambda0: vec![0.0; m],
            suffixes: Default::default(),
            imported_funcs: Vec::new(),
            ampl_options: Vec::new(),
            nl_counts: None,
            var_names: Vec::new(),
            con_names: Vec::new(),
        }
    }

    // Keep parse_nl_text reachable for a future header-fast-path test
    // against a committed .nl fixture.
    #[allow(dead_code)]
    fn _parse(txt: &str) -> NlProblem {
        parse_nl_text(txt).expect("valid .nl")
    }

    /// **gh #492.** `min −x0 − 2·x1  s.t.  x0 + x1 + 3 <= 6, x ∈ [0,3]²`,
    /// with the `3` written into the row's expression segment. `body` is
    /// the `C0` token stream for that constant.
    fn lp_with_row_constant(body: &str) -> NlProblem {
        let nl = format!(
            "g3 1 1 0
 2 1 1 0 0
 1 0 0 0 0 0
 0 0
 1 0 0
 0 0 0 1
 0 0 0 0 0
 2 2
 0 0
 0 0 0 0 0
C0
{body}
O0 0
n0
r
1 6.0
b
0 0 3
0 0 3
k1
1
J0 2
0 1
1 1
G0 2
0 -1
1 -2
"
        );
        parse_nl_text(&nl).expect("valid .nl")
    }

    /// The classifier's fast path asks `is_trivially_zero` of every
    /// `con_nonlinear` entry, which is an *identity* test — it cannot tell
    /// "this row has a nonlinear part" from "this row's part is the
    /// constant 3". A bare literal survived that anyway, because the
    /// fallback polynomial walk lowers `Const` and finds no quadratic
    /// term; what it does not do is keep the constant, so the row's `+3`
    /// lived on only as `qp_extract`'s `const_shift`. After the parse-time
    /// fold the bound carries it and the fast path is exact.
    #[test]
    fn a_literal_row_constant_classifies_lp_and_moves_the_bound() {
        let prob = lp_with_row_constant("n3");
        assert_eq!(classify_problem(&prob), ProblemClass::Lp);
        // `x0 + x1 + 3 <= 6` is `x0 + x1 <= 3`.
        assert!((prob.g_u[0] - 3.0).abs() < 1e-12, "g_u = {}", prob.g_u[0]);
        assert!(
            matches!(prob.con_nonlinear[0], Expr::Const(c) if c == 0.0),
            "the row body should be the identity zero: {:?}",
            prob.con_nonlinear[0]
        );
    }

    /// The case the polynomial walk cannot rescue: a constant it has to
    /// *compute*. `sqrt(9)` is not a degree-≤2 polynomial in any variable,
    /// so `analyze_quadratic` returns `None` and the row made the whole
    /// model NLP — an LP that never reached the convex route. The fold
    /// settles it at parse, where the value is known.
    #[test]
    fn a_computed_row_constant_does_not_make_an_lp_classify_nlp() {
        let prob = lp_with_row_constant("o39\nn9");
        assert_eq!(classify_problem(&prob), ProblemClass::Lp);
        assert!((prob.g_u[0] - 3.0).abs() < 1e-12, "g_u = {}", prob.g_u[0]);
    }
}
