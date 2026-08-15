//! Regression tests for gh #590: a feasible NLP whose every constraint row
//! sits at or below its own floating-point resolution was refused a
//! certificate, and — in the released 0.10.0 — convicted of local
//! infeasibility outright.
//!
//! Reported against LyoPRONTO's pseudosteady-limit continuation study.
//! `pounce-solver 0.10.0` returned `Infeasible_Problem_Detected` on the
//! Problem 1 `f = 0.02` rung, a known-feasible OCP that Ipopt 3.14.16 solves;
//! through Pyomo that status is indistinguishable from a genuine infeasibility
//! proof. The model is written in Landau coordinates, so its conduction rows
//! carry `1/(H − S)²` and reach magnitudes near `1e8`. At the point POUNCE
//! stalled on, the measurements were:
//!
//! ```text
//!   scaled KKT error        4.29e-10   (tol = 1e-6)
//!   unscaled constr_viol    1.62e-02   (constr_viol_tol = 1e-6)
//!   rows above their floor  none
//! ```
//!
//! One ulp of a row at `1e8` scaled through `‖x‖` is `~1e-2`, so that
//! violation is the quantum the rows are measured in, not a distance from
//! feasibility. Ipopt lands on the same point with `8.06e-3` and calls it
//! `Solved To Acceptable Level`; which side of `1e-2` a run falls on is
//! arithmetic luck.
//!
//! gh #528 built the per-row noise floor and gave it to the strict
//! **aggregate**, deliberately leaving the per-component `constr_viol` test on
//! the raw residual. That left the gate incoherent — an unfloored component
//! can veto a certificate the floored aggregate has already granted, which is
//! exactly what happened here — and left the rapid-infeasibility detector's
//! absolute arm free to convict on a quantum. This fix extends the same floor
//! to both, in both cases only when *no* row rises above its own floor.
//!
//! The family below is gh #528's, restated at the scales where the quantum
//! clears the issue's `constr_viol_tol = 1e-6`, and it is **scale-equivariant**:
//! mapping `b → σ·b` and `ub → σ·ub` maps the LP onto itself under `x → σ·x`,
//! so `f*(σ) = σ·f*(1)` exactly. Any disagreement is the magnitude of the data
//! changing the answer, which is the defect.

use pounce_algorithm::application::IpoptApplication;
use pounce_common::types::Number;
use pounce_nlp::return_codes::ApplicationReturnStatus;
use pounce_nlp::tnlp::{
    BoundsInfo, IndexStyle, IpoptCq, IpoptData, NlpInfo, Solution, SparsityRequest, StartingPoint,
    TNLP,
};
use std::cell::RefCell;
use std::rc::Rc;

/// `min cᵀx` s.t. `A x >= b`, `0 <= x <= ub`, with `A > 0` and `c > 0`
/// entrywise — so the feasible set is a nonempty subset of a compact box and
/// the LP is bounded by construction. Dense constant Jacobian, zero Hessian.
struct DenseLp {
    n: usize,
    m: usize,
    a: Vec<Number>, // row-major, m x n
    b: Vec<Number>,
    c: Vec<Number>,
    ub: Number,
    x0: Number,
    solution: Option<(Number, Vec<Number>)>,
}

/// Deterministic stand-in for the issue's `numpy.random.default_rng` draws:
/// a 64-bit LCG mapped onto `[lo, hi)`. The instance only has to be a
/// well-posed LP with `O(1)` matrix entries and a right-hand side at `scale`;
/// nothing depends on reproducing numpy's exact stream.
struct Lcg(u64);

impl Lcg {
    fn uniform(&mut self, lo: Number, hi: Number) -> Number {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let u = ((self.0 >> 11) as Number) / ((1u64 << 53) as Number);
        lo + u * (hi - lo)
    }
}

impl DenseLp {
    /// The issue's generator at a given data `scale`: `A ∈ [0.5, 5)`,
    /// `b = scale · [0.5, 2) · n/4`, `c ∈ [1, 10)`, `ub = 10·scale`, started
    /// from the box midpoint.
    ///
    /// The start is the box midpoint rather than the issue's `min(ub/2, 1e3)`
    /// so that it too rides the `x → σ·x` map and the family stays exactly
    /// self-similar; the reported exit reproduces from either (verified across
    /// the issue's full sweep at starts of `ub/2` and `ub/20`).
    fn new(seed: u64, n: usize, m: usize, scale: Number) -> Self {
        let mut rng = Lcg(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1));
        let a = (0..m * n).map(|_| rng.uniform(0.5, 5.0)).collect();
        let b = (0..m)
            .map(|_| scale * rng.uniform(0.5, 2.0) * (n as Number) * 0.25)
            .collect();
        let c = (0..n).map(|_| rng.uniform(1.0, 10.0)).collect();
        DenseLp {
            n,
            m,
            a,
            b,
            c,
            ub: 10.0 * scale,
            x0: 5.0 * scale,
            solution: None,
        }
    }

    /// The same LP with every magnitude divided by `scale` — the `σ = 1`
    /// member of the family this instance belongs to. `A` and `c` are shared
    /// verbatim, so the two differ only in the magnitude of the data, and the
    /// starting point maps across with everything else.
    fn at_unit_scale(&self, scale: Number) -> Self {
        DenseLp {
            n: self.n,
            m: self.m,
            a: self.a.clone(),
            b: self.b.iter().map(|v| v / scale).collect(),
            c: self.c.clone(),
            ub: self.ub / scale,
            x0: self.x0 / scale,
            solution: None,
        }
    }
}

impl TNLP for DenseLp {
    fn get_nlp_info(&mut self) -> Option<NlpInfo> {
        Some(NlpInfo {
            n: self.n as i32,
            m: self.m as i32,
            nnz_jac_g: (self.m * self.n) as i32,
            nnz_h_lag: 0,
            index_style: IndexStyle::C,
        })
    }

    fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
        for j in 0..self.n {
            b.x_l[j] = 0.0;
            b.x_u[j] = self.ub;
        }
        for i in 0..self.m {
            b.g_l[i] = self.b[i];
            b.g_u[i] = 2e19; // +inf sentinel
        }
        true
    }

    fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
        for j in 0..self.n {
            sp.x[j] = self.x0;
        }
        true
    }

    fn eval_f(&mut self, x: &[Number], _new_x: bool) -> Option<Number> {
        Some((0..self.n).map(|j| self.c[j] * x[j]).sum())
    }

    fn eval_grad_f(&mut self, _x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
        g[..self.n].copy_from_slice(&self.c);
        true
    }

    fn eval_g(&mut self, x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
        for (i, gi) in g.iter_mut().enumerate().take(self.m) {
            *gi = (0..self.n).map(|j| self.a[i * self.n + j] * x[j]).sum();
        }
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
                for i in 0..self.m {
                    for j in 0..self.n {
                        irow[i * self.n + j] = i as i32;
                        jcol[i * self.n + j] = j as i32;
                    }
                }
            }
            SparsityRequest::Values { values } => {
                values[..self.m * self.n].copy_from_slice(&self.a);
            }
        }
        true
    }

    fn eval_h(
        &mut self,
        _x: Option<&[Number]>,
        _new_x: bool,
        _obj_factor: Number,
        _lambda: Option<&[Number]>,
        _new_lambda: bool,
        _mode: SparsityRequest<'_>,
    ) -> bool {
        true
    }

    fn finalize_solution(&mut self, sol: Solution<'_>, _d: &IpoptData, _q: &IpoptCq) {
        self.solution = Some((sol.obj_value, sol.x.to_vec()));
    }
}

/// As [`solve`], with arbitrary numeric options applied first.
fn solve_with_options(
    lp: DenseLp,
    options: &[(&str, Number)],
) -> (ApplicationReturnStatus, Number, Vec<Number>) {
    let mut app = IpoptApplication::new();
    app.options_mut()
        .set_integer_value("print_level", 0, true, false)
        .unwrap();
    for &(name, value) in options {
        app.options_mut()
            .set_numeric_value(name, value, true, false)
            .unwrap();
    }
    // LyoPRONTO's own string options, alongside the numeric ones. The issue is
    // reported under this exact configuration, and `gradient-based` scaling is
    // load-bearing for it: it is what drives the *scaled* residual to `1e-10`
    // while the unscaled one stays at the rows' own quantum, which is the gap
    // between the aggregate and the component gate this fix closes.
    for (name, value) in [
        ("mu_strategy", "adaptive"),
        ("nlp_scaling_method", "gradient-based"),
    ] {
        app.options_mut()
            .set_string_value(name, value, true, false)
            .unwrap();
    }
    app.initialize().unwrap();
    let inst = Rc::new(RefCell::new(lp));
    let tnlp: Rc<RefCell<dyn TNLP>> = inst.clone();
    let status = app.optimize_tnlp(tnlp);
    let (obj, x) = inst
        .borrow()
        .solution
        .clone()
        .unwrap_or((Number::NAN, vec![]));
    (status, obj, x)
}

/// The option set the issue reports under: LyoPRONTO's own, whose
/// `constr_viol_tol = 1e-6` is two decades tighter than the default and so
/// sits far below the quantum at these data scales.
fn issue_options() -> Vec<(&'static str, Number)> {
    vec![
        ("tol", 1e-6),
        ("constr_viol_tol", 1e-6),
        ("acceptable_tol", 1e-3),
    ]
}

/// The issue proper. Every LP here is feasible and bounded, and every one is
/// the `σ = 1` LP with its data multiplied through by `σ` — so a solver whose
/// verdict depends on the magnitude of the data is the only way the two can
/// disagree.
///
/// Before the fix this grid of 12 returned 7 `Solve_Succeeded`, 4
/// `Solved_To_Acceptable_Level` and 1 `Search_Direction_Becomes_Too_Small`,
/// every one of them holding the right optimum. The five degraded verdicts are
/// the same defect as the reported one, one rung less severe: the strict gate
/// refusing a point it cannot fault, then the solve running on at a point it
/// cannot improve.
#[test]
fn a_residual_below_every_row_floor_does_not_cost_the_certificate() {
    let (n, m) = (20usize, 10usize);
    for seed in 0..6u64 {
        // `at_unit_scale` divides out the same factor `new` multiplied in, so
        // every scale in the sweep shares one reference LP and one reference
        // objective — which is what makes the cross-scale comparison below
        // exact rather than approximate.
        let (ref_status, ref_obj, _) = solve_with_options(
            DenseLp::new(seed, n, m, 1e10).at_unit_scale(1e10),
            &issue_options(),
        );
        assert_eq!(
            ref_status,
            ApplicationReturnStatus::SolveSucceeded,
            "the σ=1 member of the family is the reference and must solve (seed={seed})",
        );

        let mut deviations = Vec::new();
        for scale in [1e10, 1e11] {
            let (status, obj, x) =
                solve_with_options(DenseLp::new(seed, n, m, scale), &issue_options());
            assert_eq!(
                status,
                ApplicationReturnStatus::SolveSucceeded,
                "gh #590: feasible bounded LP at data scale {scale:e} (seed={seed}) \
                 exited {status:?} with obj={obj:e}",
            );
            assert!(
                x.iter().all(|v| v.is_finite()),
                "non-finite component in the returned point (seed={seed} scale={scale:e})",
            );

            // Loose against `tol`, not against the arithmetic: both solves stop
            // at `tol = 1e-6`, so the objective is only pinned to about that,
            // and this bar is one order looser again. The tight statement is
            // the cross-scale one below.
            let expected = scale * ref_obj;
            let deviation = (obj - expected).abs() / expected.abs();
            assert!(
                deviation <= 1e-5,
                "objective is not scale-equivariant: got {obj:e}, expected \
                 {scale:e}·{ref_obj:e} = {expected:e} (relative {deviation:e}, \
                 seed={seed} scale={scale:e})",
            );
            deviations.push(deviation);
        }

        // The claim the issue is actually about. A deviation from the σ=1
        // reference is expected — both solves stopped at a tolerance — but it
        // must be the *same* deviation at every data scale. If the magnitude of
        // the data were still reaching the verdict, the decade between these two
        // members would show up here; measured, it does not move the tenth
        // significant figure.
        let spread = (deviations[0] - deviations[1]).abs();
        assert!(
            spread <= 1e-8 * deviations[0].max(1e-12),
            "the deviation from the σ=1 optimum moved with the data scale \
             ({:e} at 1e10 vs {:e} at 1e11, seed={seed}) — the magnitude of the \
             data is still reaching the answer",
            deviations[0],
            deviations[1],
        );
    }
}

/// `ROW_NOISE_KAPPA` from `ipopt_cq.rs`, restated here because it is private.
const ROW_NOISE_KAPPA: Number = 64.0;

/// Direction guard for the component gate. The relaxation is confined to the
/// case where *no* row rises above its own floor; one resolvable row anywhere
/// and the raw `constr_viol <= constr_viol_tol` comparison stands. Here every
/// row is short by `10 · ROW_NOISE_KAPPA · eps · |b_i|` — one decade above the
/// quantum, and the closest a violation can get to the floor while still being
/// one the solver is obliged to see.
///
/// `bound_relax_factor` is pinned to `0` for the reason gh #528's companion
/// test documents: at its default and `ub = 1e11` the solver widens every
/// variable bound by ~1e3 before the first iteration, which absorbs this
/// shortfall and makes the model genuinely feasible as posed.
#[test]
fn a_violation_just_above_the_quantum_is_still_refused() {
    let mut lp = DenseLp::new(0, 20, 10, 1e10);
    let mut smallest_shortfall = Number::INFINITY;
    for i in 0..lp.m {
        let reach: Number = (0..lp.n).map(|j| lp.a[i * lp.n + j] * lp.ub).sum();
        let shortfall = 10.0 * ROW_NOISE_KAPPA * Number::EPSILON * reach;
        lp.b[i] = reach + shortfall;
        smallest_shortfall = smallest_shortfall.min(shortfall);
    }
    // The premise: above the quantum, and above the tolerance in play.
    assert!(
        smallest_shortfall > 1e-6,
        "shortfall {smallest_shortfall:e} must clear the issue's constr_viol_tol",
    );

    let mut options = issue_options();
    options.push(("bound_relax_factor", 0.0));
    let (status, _obj, _x) = solve_with_options(lp, &options);
    assert!(
        refused(status),
        "an LP short of its right-hand side by ~10 ulp per row must not certify \
         (got {status:?})",
    );
}

/// Direction guard for the infeasibility verdict. The detector's absolute arm
/// now needs the violation to be something the model's arithmetic can resolve,
/// and a model short of its right-hand side by `~1e10` per row is not close to
/// that boundary — the verdict must survive the change untouched.
#[test]
fn a_genuinely_infeasible_large_scale_model_is_still_refused() {
    let mut lp = DenseLp::new(0, 20, 10, 1e10);
    for i in 0..lp.m {
        let reach: Number = (0..lp.n).map(|j| lp.a[i * lp.n + j] * lp.ub).sum();
        lp.b[i] = 2.0 * reach;
    }
    let (status, _obj, _x) = solve_with_options(lp, &issue_options());
    assert!(
        refused(status),
        "an LP short of its right-hand side by ~1e10 per row must never certify \
         (got {status:?})",
    );
}

/// The opt-out reaches both new gates. `primal_noise_floor_kappa = 0` is
/// documented as restoring upstream's bare-absolute primal term; if the grid
/// above still certified with the floor switched off, this fix would not be
/// what is carrying it.
#[test]
fn switching_the_floor_off_loses_the_certificates_again() {
    let mut options = issue_options();
    options.push(("primal_noise_floor_kappa", 0.0));
    let mut lost = 0;
    for scale in [1e10, 1e11] {
        for seed in 0..6u64 {
            let lp = DenseLp::new(seed, 20, 10, scale);
            let (status, _obj, _x) = solve_with_options(lp, &options);
            if status != ApplicationReturnStatus::SolveSucceeded {
                lost += 1;
            }
        }
    }
    assert!(
        lost > 0,
        "with the floor switched off, gh #590's grid must lose certificates \
         again — none of the 12 did, so the option is not reaching the gates",
    );
}

/// A verdict that is not a certificate *and* not a crash. `assert_ne!` against
/// `Solve_Succeeded` alone would pass equally on `Invalid_Number_Detected` or
/// an internal error, which is a solver falling over rather than refusing.
fn refused(status: ApplicationReturnStatus) -> bool {
    matches!(
        status,
        ApplicationReturnStatus::InfeasibleProblemDetected
            | ApplicationReturnStatus::RestorationFailed
            | ApplicationReturnStatus::SearchDirectionBecomesTooSmall
            | ApplicationReturnStatus::MaximumIterationsExceeded
    )
}
