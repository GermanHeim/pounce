//! Regression tests for gh #528: a feasible, bounded LP whose right-hand
//! sides reach `~1e7` and above exits `Search_Direction_Becomes_Too_Small`
//! while holding the correct optimum.
//!
//! The KKT error POUNCE (and upstream Ipopt) compares against `tol` is
//!
//! ```text
//!   max( ‖∇L‖_∞ / s_d , max(‖c‖_∞, ‖d − s‖_∞) , ‖compl‖_∞ / s_c )
//! ```
//!
//! The dual and complementarity terms are normalised; the primal one is a bare
//! absolute residual. But `c_i` and `d_i − s_i` are *differences of quantities
//! the row's own size*, so they are quantised in units of `eps ·` that
//! magnitude: at `|b| ~ 1e8` the smallest nonzero value `‖d − s‖_∞` can take is
//! one ulp, `1.5e-8`, already above the default `tol = 1e-8`. `nlp_err <= tol`
//! then stops being a statement about the iterate and becomes a bet on the
//! residual landing on an exact `0` rather than on one ulp. Iterates that lose
//! the bet keep the solve running at a point it cannot improve until the search
//! direction collapses — the reported exit, with the LP's optimum in hand to
//! eight significant figures.
//!
//! The fix makes the strict gate judge the primal term against the finest
//! residual each row can represent (`IpoptCalculatedQuantities::
//! curr_primal_infeasibility_above_noise`), leaving `constr_viol_tol` to bound
//! what is admitted in absolute terms.
//!
//! The LP family below is the issue's, restated so it needs no reference
//! solver: it is **scale-equivariant**. Mapping `b → σ·b` and `ub → σ·ub` maps
//! the LP onto itself under `x → σ·x`, so the optimum must satisfy
//! `f*(σ) = σ·f*(1)` exactly. `σ = 1` is the regime POUNCE has always handled,
//! which makes it the reference: any disagreement is the magnitude of the data
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

/// Solve and return `(status, objective, x)`.
fn solve(lp: DenseLp) -> (ApplicationReturnStatus, Number, Vec<Number>) {
    solve_with(lp, None)
}

/// As [`solve`], optionally pinning `bound_relax_factor`. Only the boundary
/// test needs this: at `ub = 1e9` the default `1e-8` relaxes every variable
/// bound by ~10, which at that scale absorbs violations far larger than
/// anything the noise floor forgives, and would mask what that test is asking.
fn solve_with(
    lp: DenseLp,
    bound_relax_factor: Option<Number>,
) -> (ApplicationReturnStatus, Number, Vec<Number>) {
    match bound_relax_factor {
        Some(f) => solve_with_options(lp, &[("bound_relax_factor", f)]),
        None => solve_with_options(lp, &[]),
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

/// The issue proper. Every one of these LPs is feasible and bounded, and every
/// one is the `σ = 1` LP with its data multiplied through by `σ` — so a solver
/// whose verdict depends on the magnitude of `b` is the only way the two can
/// disagree.
///
/// Before the fix these 24 runs returned 12 `Solve_Succeeded`, 9
/// `Search_Direction_Becomes_Too_Small` and 3 `Solved_To_Acceptable_Level`,
/// every one of them holding the right optimum — all 12 lost certificates at
/// `σ ∈ {1e7, 1e8}`, none at `1e6`. The scatter is the point: which side of the
/// ulp the primal residual lands on is arithmetic luck, so the same LP written
/// larger loses its certificate for no reason the model can express.
#[test]
fn a_large_right_hand_side_does_not_cost_the_certificate() {
    for scale in [1e6, 1e7, 1e8] {
        for &(n, m) in &[(20usize, 10usize), (40, 20)] {
            for seed in 0..4u64 {
                let lp = DenseLp::new(seed, n, m, scale);
                let (ref_status, ref_obj, _) = solve(lp.at_unit_scale(scale));
                assert_eq!(
                    ref_status,
                    ApplicationReturnStatus::SolveSucceeded,
                    "the σ=1 member of the family is the reference and must solve \
                     (seed={seed} n={n} scale={scale:e})",
                );

                let (status, obj, x) = solve(DenseLp::new(seed, n, m, scale));
                assert_eq!(
                    status,
                    ApplicationReturnStatus::SolveSucceeded,
                    "gh #528: feasible bounded LP at data scale {scale:e} \
                     (seed={seed} n={n}) exited {status:?} with obj={obj:e}",
                );
                let expected = scale * ref_obj;
                assert!(
                    (obj - expected).abs() <= 1e-6 * expected.abs(),
                    "objective is not scale-equivariant: got {obj:e}, expected \
                     {scale:e}·{ref_obj:e} = {expected:e} (seed={seed} n={n})",
                );
                assert!(
                    x.iter().all(|v| v.is_finite()),
                    "non-finite component in the returned point (seed={seed} n={n} \
                     scale={scale:e})",
                );
            }
        }
    }
}

/// Direction guard. The noise floor may only forgive a residual the arithmetic
/// could not have made smaller — never a real violation. This LP demands
/// `Σⱼ aᵢⱼ xⱼ >= b` with `b` set to twice what the box can supply, so every row
/// is short by `~1e8`: astronomically above any floor, and the solve must not
/// come back `Solve_Succeeded`.
#[test]
fn an_infeasible_large_scale_row_is_still_refused() {
    let mut lp = DenseLp::new(0, 20, 10, 1e8);
    // Largest attainable row value is `Σⱼ aᵢⱼ · ub`; ask for twice that.
    for i in 0..lp.m {
        let reach: Number = (0..lp.n).map(|j| lp.a[i * lp.n + j] * lp.ub).sum();
        lp.b[i] = 2.0 * reach;
    }
    let (status, _obj, _x) = solve(lp);
    assert!(
        refused(status),
        "an LP short of its right-hand side by ~1e8 per row must never certify \
         (got {status:?})",
    );
}

/// `ROW_NOISE_KAPPA` from `ipopt_cq.rs`, restated here because it is private.
/// If the two ever drift apart this test stops probing the boundary it claims
/// to — the assertion on `shortfall` below is what would catch that.
const ROW_NOISE_KAPPA: Number = 64.0;

/// The **near** side of the guard above, which leaves thirteen decades between
/// the shortfall and the floor and so would pass against a floor a trillion
/// times too loose. Here every row is short by only
/// `10 · ROW_NOISE_KAPPA · eps · |b_i|` — one decade above the quantum the
/// floor forgives, and the closest a violation can get to the floor while
/// still being a violation the solver is obliged to see.
///
/// What refuses it is the containment this fix rests on: `passes_component_tols`
/// tests `constr_viol` against `constr_viol_tol` (default `1e-4`) on the full,
/// **unfloored** residual, so nothing above that tolerance can certify however
/// large `|b|` grows and however loose `64·eps·|b|` gets. The floor only ever
/// narrows what the aggregate admits *inside* that band.
///
/// gh #590 later gave that component test the floor as well, but only for the
/// case where *no* row rises above its own — which is the one case this fixture
/// is built to exclude. Every row here is short by ten times its quantum, so
/// `curr_primal_infeasibility_above_noise` reports them all, the raw comparison
/// stays in force, and the refusal is the same one for the same reason.
///
/// `bound_relax_factor` is pinned to `0` here, and that is the whole reason
/// this test needs a knob the others don't. At its default of `1e-8` and
/// `ub = 1e9` the solver already widens every variable bound by ~10 before the
/// first iteration, which lets `Σⱼ aᵢⱼ xⱼ` overshoot `reach` by far more than
/// this shortfall — the model is then genuinely feasible as posed and
/// certifying it is correct (verified: it returns `Solve_Succeeded` at the
/// default, on this build and before the fix alike). That relaxation is
/// pre-existing, unrelated to gh #528, and six orders of magnitude *coarser*
/// than the quantum this fix forgives; switching it off is what leaves the
/// noise floor as the only thing that could wrongly admit this point.
#[test]
fn a_violation_just_above_the_noise_quantum_is_still_refused() {
    let mut lp = DenseLp::new(0, 20, 10, 1e8);
    // Largest attainable row value is `Σⱼ aᵢⱼ · ub`; ask for a hair more.
    let mut smallest_shortfall = Number::INFINITY;
    for i in 0..lp.m {
        let reach: Number = (0..lp.n).map(|j| lp.a[i * lp.n + j] * lp.ub).sum();
        let shortfall = 10.0 * ROW_NOISE_KAPPA * Number::EPSILON * reach;
        lp.b[i] = reach + shortfall;
        smallest_shortfall = smallest_shortfall.min(shortfall);
    }
    // The premise: above the quantum, and above the user's own feasibility
    // tolerance. Both have to hold for this to be testing what it says.
    assert!(
        smallest_shortfall > 1e-4,
        "shortfall {smallest_shortfall:e} must clear the default constr_viol_tol",
    );

    let (a, b, ub, n, m) = (lp.a.clone(), lp.b.clone(), lp.ub, lp.n, lp.m);
    let (status, _obj, x) = solve_with(lp, Some(0.0));
    assert!(
        refused(status),
        "an LP short of its right-hand side by ~10 ulp per row must not certify \
         (got {status:?})",
    );

    // …and refused for the *right* reason: the point it came back with is one
    // the box cannot make feasible, not a solver that fell over on the way.
    // Every returned `x` is inside the box, and the worst row is still short.
    assert_eq!(x.len(), n, "no iterate came back at all");
    let worst = (0..m)
        .map(|i| b[i] - (0..n).map(|j| a[i * n + j] * x[j]).sum::<Number>())
        .fold(Number::NEG_INFINITY, Number::max);
    assert!(
        x.iter().all(|&v| (-1e-6..=ub + 1e-6).contains(&v)),
        "the returned point left the box, so the refusal says nothing about feasibility",
    );
    assert!(
        worst > 1e-4,
        "the returned point should still violate a row by more than \
         constr_viol_tol; worst shortfall was {worst:e}",
    );
}

/// The near side of the boundary, where `constr_viol_tol` is *not* what does
/// the refusing. The two guards above leave the shortfall above that tolerance,
/// so the component gate rejects them and they say nothing about the floor.
/// Here the shortfall sits an order of magnitude *under* `constr_viol_tol` and
/// an order of magnitude *over* the quantum — the narrow band in which the
/// aggregate is the only tolerance still in play. The data scale lands
/// `64·eps·|b|` near `1e-6`, so `10·quantum ≈ 1e-5` against
/// `constr_viol_tol = 1e-4`.
///
/// **What this measured, which is more than it set out to.** The point is
/// refused, and not by the noise floor: re-running it with `ROW_NOISE_KAPPA`
/// raised a *thousandfold* — a floor of `8.8e-4`, which silences this row's
/// residual outright — still refuses. On a model with no feasible point the
/// filter and the restoration phase reach a verdict on their own criteria,
/// and the convergence gate the floor feeds is never the deciding vote. So the
/// floor cannot buy an infeasible model a certificate at any kappa; it only
/// ever participates at a point the rest of the algorithm already believes is
/// converged. That is a stronger containment than the `constr_viol_tol`
/// argument alone, and it is why the floor's own boundary is pinned by the
/// unit tests on `amax_above_floor` and
/// `curr_primal_infeasibility_above_noise` rather than through a solve.
///
/// The mirror case is deliberately not asserted: an LP made infeasible by
/// *less* than a quantum comes back `Restoration_Failed` for the same reason,
/// and it is not the situation gh #528 describes. The case the floor exists to
/// admit is a *feasible* LP whose residual at the optimum is quantisation
/// noise — `a_large_right_hand_side_does_not_cost_the_certificate`.
#[test]
fn the_floor_refuses_a_violation_just_above_the_quantum() {
    // `reach ≈ n · mean(a) · ub ≈ 55 · ub`; at `ub = 10·scale` this puts the
    // per-row magnitude near `7e7`, where the quantum is ~`1e-6`.
    let shortfall_at = |multiple: Number| {
        let mut lp = DenseLp::new(0, 20, 10, 1.3e5);
        let mut smallest = Number::INFINITY;
        for i in 0..lp.m {
            let reach: Number = (0..lp.n).map(|j| lp.a[i * lp.n + j] * lp.ub).sum();
            let quantum = ROW_NOISE_KAPPA * Number::EPSILON * reach;
            lp.b[i] = reach + multiple * quantum;
            smallest = smallest.min(multiple * quantum);
        }
        (lp, smallest)
    };

    // Premises, asserted rather than assumed: the coarse shortfall has to sit
    // strictly between the quantum and the user's feasibility tolerance, or
    // this test is measuring one of the other gates again.
    let (lp_above, above) = shortfall_at(10.0);
    assert!(
        (1e-8..1e-4).contains(&above),
        "shortfall {above:e} must clear tol and stay under constr_viol_tol",
    );
    // Above the quantum: a real violation the arithmetic could have resolved,
    // and `constr_viol_tol` is not what refuses it.
    let (status, _obj, _x) = solve_with(lp_above, Some(0.0));
    assert!(
        refused(status),
        "a violation {above:e} — above the row's quantum and below \
         constr_viol_tol — must not certify (got {status:?})",
    );
}

/// The two aggregates really do diverge on this family, which is what makes
/// the extra summary line fire — and what makes it necessary. A solve that
/// reports `Overall NLP error` above `tol` beside `EXIT: Optimal Solution
/// Found` has to account for the gap somewhere, and `final_kkt_error_above_noise`
/// is the number the strict gate actually tested.
#[test]
fn the_reported_aggregate_and_the_tested_one_diverge_at_large_scale() {
    let mut app = IpoptApplication::new();
    app.options_mut()
        .set_integer_value("print_level", 0, true, false)
        .unwrap();
    app.initialize().unwrap();
    let inst = Rc::new(RefCell::new(DenseLp::new(0, 20, 10, 1e8)));
    let tnlp: Rc<RefCell<dyn TNLP>> = inst.clone();
    let status = app.optimize_tnlp(tnlp);
    assert_eq!(status, ApplicationReturnStatus::SolveSucceeded);

    let stats = app.statistics();
    assert!(
        stats.final_kkt_error_above_noise < stats.final_kkt_error,
        "the floored aggregate should be strictly smaller here (raw {:e}, \
         floored {:e}) — otherwise the summary line never fires and the gate \
         had nothing to forgive",
        stats.final_kkt_error,
        stats.final_kkt_error_above_noise,
    );
    // And the raw one is what makes the line worth printing: it sits above the
    // default `tol` on a solve that legitimately certified.
    assert!(
        stats.final_kkt_error > 1e-8,
        "raw error {:e} was expected above tol on this model",
        stats.final_kkt_error,
    );
}

/// The opt-out is real, not decorative: `primal_noise_floor_kappa = 0` puts
/// every floor at zero, so the strict gate is upstream Ipopt's bare-absolute
/// primal term again and the issue's LPs go back to losing their certificates.
///
/// This is the one test that would still pass if the fix were reverted, and
/// that is the point — it pins the *option*, so a later change that quietly
/// ignores it fails here. The pairing with
/// `a_large_right_hand_side_does_not_cost_the_certificate`, which runs the same
/// family at the default kappa, is what makes it meaningful: same LPs, same
/// build, opposite verdicts, the option the only difference.
#[test]
fn switching_the_noise_floor_off_restores_the_old_verdicts() {
    let mut lost = 0;
    for seed in 0..4u64 {
        let lp = DenseLp::new(seed, 20, 10, 1e7);
        let (status, _obj, _x) = solve_with_options(lp, &[("primal_noise_floor_kappa", 0.0)]);
        if status != ApplicationReturnStatus::SolveSucceeded {
            lost += 1;
        }
    }
    assert!(
        lost > 0,
        "with the floor switched off, gh #528's LPs must lose certificates \
         again — none of the 4 did, so the option is not reaching the gate",
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
