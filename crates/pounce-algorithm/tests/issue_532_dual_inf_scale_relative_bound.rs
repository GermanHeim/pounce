//! Regression tests for gh #532: `dual_inf_tol` is a bare absolute bound on a
//! quantity the aggregate KKT error normalises, so a solve that is stationary
//! relative to the size of the gradients involved is refused a certificate
//! whenever those gradients are large.
//!
//! The aggregate the strict gate compares against `tol` is
//!
//! ```text
//!   max( ‖∇L‖_∞ / s_d , max(‖c‖_∞, ‖d − s‖_∞) , ‖compl‖_∞ / s_c )
//! ```
//!
//! and its dual term carries `s_d`, which grows with the mean magnitude of the
//! multipliers. On Vanderbei's `orthrds2` — the issue's model — `s_d ≈ 1.6e10`
//! with `‖∇L‖_∞ = 89.7`, so the aggregate's dual term is `5.6e-09`,
//! comfortably inside the default `tol = 1e-8`. The per-component gate then
//! tested that same `‖∇L‖_∞` against `dual_inf_tol`, default `1.0`, and refused:
//! one quantity, two standards, ten orders of magnitude apart. `orthrds2` exited
//! `Solved_To_Acceptable_Level` holding the answer, and relaxing that one
//! constant (`dual_inf_tol=1e3`) was sufficient to turn it into
//! `Optimal Solution Found` at the same objective.
//!
//! The family below states the asymmetry in its purest form. Multiplying an
//! objective by a positive constant `w` changes nothing about a problem — same
//! feasible set, same solution set, same active set, same Newton step — but it
//! multiplies `∇f`, every multiplier, `s_d` and `‖∇L‖_∞` by `w`. A verdict that
//! moves under that map is a verdict about the units the user chose rather than
//! about the iterate.
//!
//! These QPs solve in a single iteration at `w = 1`. At `w = 1e17` the residual
//! `‖∇L‖_∞` cannot be smaller than the `eps · w ~ 10` its own arithmetic
//! quantises to, so the run met `1.0` with a residual it had no way to shrink:
//! before the fix it ground out the full 15-iteration acceptable-level streak
//! and returned `Solved_To_Acceptable_Level` at a point it reached on
//! iteration 1.

use pounce_algorithm::application::IpoptApplication;
use pounce_common::types::Number;
use pounce_nlp::return_codes::ApplicationReturnStatus;
use pounce_nlp::tnlp::{
    BoundsInfo, IndexStyle, IpoptCq, IpoptData, NlpInfo, Solution, SparsityRequest, StartingPoint,
    TNLP,
};
use std::cell::RefCell;
use std::rc::Rc;

/// `min w·½‖x − p‖²` s.t. `A x = b`, `x` free — a strictly convex QP with
/// linear equality constraints, so it has a unique solution `x*` that does not
/// depend on `w`, and no variable bounds, so there are no bound multipliers and
/// the complementarity term is identically zero (as it is on `orthrds2`). `w`
/// multiplies the objective and nothing else.
struct WeightedQp {
    n: usize,
    m: usize,
    a: Vec<Number>, // row-major, m x n
    b: Vec<Number>,
    p: Vec<Number>,
    w: Number,
    solution: Option<(Number, Vec<Number>)>,
}

/// Deterministic stand-in for a random draw: a 64-bit LCG mapped onto
/// `[lo, hi)`. Nothing depends on the exact stream — only that the instances
/// are well-posed QPs with `O(1)` data and full-rank `A`.
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

impl WeightedQp {
    fn new(seed: u64, n: usize, m: usize, w: Number) -> Self {
        let mut rng = Lcg(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1));
        let a = (0..m * n).map(|_| rng.uniform(-2.0, 2.0)).collect();
        let b = (0..m).map(|_| rng.uniform(0.5, 2.0)).collect();
        let p = (0..n).map(|_| rng.uniform(-1.0, 1.0)).collect();
        WeightedQp {
            n,
            m,
            a,
            b,
            p,
            w,
            solution: None,
        }
    }

    /// The same QP with the objective weight reset to `1` — the reference
    /// member of the family. Every other datum is shared verbatim, so the two
    /// differ only by a positive factor on `f`, and `x*` is common to both.
    fn at_unit_weight(&self) -> Self {
        WeightedQp {
            n: self.n,
            m: self.m,
            a: self.a.clone(),
            b: self.b.clone(),
            p: self.p.clone(),
            w: 1.0,
            solution: None,
        }
    }
}

impl TNLP for WeightedQp {
    fn get_nlp_info(&mut self) -> Option<NlpInfo> {
        Some(NlpInfo {
            n: self.n as i32,
            m: self.m as i32,
            nnz_jac_g: (self.m * self.n) as i32,
            nnz_h_lag: self.n as i32,
            index_style: IndexStyle::C,
        })
    }

    fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
        for j in 0..self.n {
            b.x_l[j] = -2e19; // -inf sentinel
            b.x_u[j] = 2e19; // +inf sentinel
        }
        for i in 0..self.m {
            b.g_l[i] = self.b[i];
            b.g_u[i] = self.b[i];
        }
        true
    }

    fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
        for j in 0..self.n {
            sp.x[j] = 0.5;
        }
        true
    }

    fn eval_f(&mut self, x: &[Number], _new_x: bool) -> Option<Number> {
        Some(
            self.w
                * 0.5
                * (0..self.n)
                    .map(|j| (x[j] - self.p[j]).powi(2))
                    .sum::<Number>(),
        )
    }

    fn eval_grad_f(&mut self, x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
        for j in 0..self.n {
            g[j] = self.w * (x[j] - self.p[j]);
        }
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
        obj_factor: Number,
        _lambda: Option<&[Number]>,
        _new_lambda: bool,
        mode: SparsityRequest<'_>,
    ) -> bool {
        // ∇²L = w·I (the constraints are linear).
        match mode {
            SparsityRequest::Structure { irow, jcol } => {
                for j in 0..self.n {
                    irow[j] = j as i32;
                    jcol[j] = j as i32;
                }
            }
            SparsityRequest::Values { values } => {
                values[..self.n].fill(obj_factor * self.w);
            }
        }
        true
    }

    fn finalize_solution(&mut self, sol: Solution<'_>, _d: &IpoptData, _q: &IpoptCq) {
        self.solution = Some((sol.obj_value, sol.x.to_vec()));
    }
}

/// Solve with objective scaling switched off, so that `s_d` — not
/// `nlp_scaling`'s `df` — is the only thing between the aggregate and the
/// component gate, and the gh #200 masked-scale veto (which keys on a clamped
/// `df`) stays out of the measurement. This is `orthrds2`'s own regime: its
/// report prints the scaled and unscaled dual infeasibility as the same
/// `8.9669e+01`.
fn solve_with_options(
    qp: WeightedQp,
    options: &[(&str, Number)],
) -> (ApplicationReturnStatus, Number, Vec<Number>) {
    let mut app = IpoptApplication::new();
    app.options_mut()
        .set_integer_value("print_level", 0, true, false)
        .unwrap();
    app.options_mut()
        .set_string_value("nlp_scaling_method", "none", true, false)
        .unwrap();
    for &(name, value) in options {
        app.options_mut()
            .set_numeric_value(name, value, true, false)
            .unwrap();
    }
    app.initialize().unwrap();
    let inst = Rc::new(RefCell::new(qp));
    let tnlp: Rc<RefCell<dyn TNLP>> = inst.clone();
    let status = app.optimize_tnlp(tnlp);
    let (obj, x) = inst
        .borrow()
        .solution
        .clone()
        .unwrap_or((Number::NAN, vec![]));
    (status, obj, x)
}

fn solve(qp: WeightedQp) -> (ApplicationReturnStatus, Number, Vec<Number>) {
    solve_with_options(qp, &[])
}

/// The issue proper. Each QP is the `w = 1` QP with a positive constant on its
/// objective, so a solver whose verdict depends on the magnitude of the
/// multipliers is the only way the two can disagree.
///
/// Before the fix every run at `w ∈ {1e17, 1e18}` returned
/// `Solved_To_Acceptable_Level` — the strict gate refused a point whose
/// aggregate KKT error was `~1e-13`, four decades inside `tol`, because
/// `‖∇L‖_∞` had grown past `1.0` with the objective weight.
#[test]
fn an_objective_weight_does_not_cost_the_certificate() {
    for w in [1e14, 1e16, 1e17, 1e18] {
        for &(n, m) in &[(8usize, 3usize), (20, 8)] {
            for seed in 0..3u64 {
                let qp = WeightedQp::new(seed, n, m, w);
                let (ref_status, ref_obj, ref_x) = solve(qp.at_unit_weight());
                assert_eq!(
                    ref_status,
                    ApplicationReturnStatus::SolveSucceeded,
                    "the w=1 member of the family is the reference and must solve \
                     (seed={seed} n={n} w={w:e})",
                );

                let (status, obj, x) = solve(WeightedQp::new(seed, n, m, w));
                assert_eq!(
                    status,
                    ApplicationReturnStatus::SolveSucceeded,
                    "gh #532: the same QP with its objective multiplied by {w:e} \
                     exited {status:?} with obj={obj:e} (seed={seed} n={n})",
                );
                let expected = w * ref_obj;
                assert!(
                    (obj - expected).abs() <= 1e-6 * expected.abs(),
                    "objective is not equivariant under the objective weight: got \
                     {obj:e}, expected {w:e}·{ref_obj:e} = {expected:e} \
                     (seed={seed} n={n})",
                );
                // The solution itself is weight-independent, so the certificate
                // is being granted at the same point the reference returns —
                // not at some other point the relaxation happened to admit.
                assert_eq!(x.len(), ref_x.len());
                for (j, (got, want)) in x.iter().zip(&ref_x).enumerate() {
                    assert!(
                        (got - want).abs() <= 1e-6 * want.abs().max(1.0),
                        "x[{j}] = {got:e} at w={w:e}, but the weight-free solution \
                         has {want:e} (seed={seed} n={n})",
                    );
                }
            }
        }
    }
}

/// The opt-out is real, not decorative: `dual_inf_scale_kappa = 0` puts the
/// bare absolute `dual_inf_tol` back and the reported behaviour returns. This
/// is also what pins the diagnosis — nothing else about these solves had to
/// change.
#[test]
fn the_kappa_zero_escape_hatch_restores_the_reported_behaviour() {
    let (status, obj, _) = solve_with_options(
        WeightedQp::new(0, 8, 3, 1e18),
        &[("dual_inf_scale_kappa", 0.0)],
    );
    assert_eq!(
        status,
        ApplicationReturnStatus::SolvedToAcceptableLevel,
        "with the floor disabled the strict gate must refuse again exactly as \
         reported (obj={obj:e})",
    );
    // And raising `dual_inf_tol` by hand is the issue's own workaround, which
    // must reach the same verdict the floor now reaches on its own.
    let (status, _, _) = solve_with_options(
        WeightedQp::new(0, 8, 3, 1e18),
        &[("dual_inf_scale_kappa", 0.0), ("dual_inf_tol", 1e6)],
    );
    assert_eq!(status, ApplicationReturnStatus::SolveSucceeded);
}

/// Direction guard. `min −exp(x)` s.t. `x >= 0` is unbounded below and has no
/// KKT point: the iterates run away and `‖∇L‖_∞ = |−exp(x)|` runs away with
/// them, reaching `8.8e+47` (the case called out at `ipopt_alg.rs`'s
/// restoration-entry guard). Its dual scale runs away by exactly the same
/// factor — nothing cancels, because there is no multiplier to meet `∇f` — so
/// the relative floor is `tol` times a number the residual is `1e8` times
/// larger than, and the point stays refused. A relative rule that forgave this
/// would be forgiving non-stationarity itself.
#[test]
fn a_runaway_gradient_is_still_refused() {
    struct Runaway {
        status_x: Option<Number>,
    }

    impl TNLP for Runaway {
        fn get_nlp_info(&mut self) -> Option<NlpInfo> {
            Some(NlpInfo {
                n: 1,
                m: 0,
                nnz_jac_g: 0,
                nnz_h_lag: 1,
                index_style: IndexStyle::C,
            })
        }
        fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
            b.x_l[0] = 0.0;
            b.x_u[0] = 2e19;
            true
        }
        fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
            sp.x[0] = 1.0;
            true
        }
        fn eval_f(&mut self, x: &[Number], _new_x: bool) -> Option<Number> {
            Some(-x[0].exp())
        }
        fn eval_grad_f(&mut self, x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
            g[0] = -x[0].exp();
            true
        }
        fn eval_g(&mut self, _x: &[Number], _new_x: bool, _g: &mut [Number]) -> bool {
            true
        }
        fn eval_jac_g(
            &mut self,
            _x: Option<&[Number]>,
            _new_x: bool,
            _mode: SparsityRequest<'_>,
        ) -> bool {
            true
        }
        fn eval_h(
            &mut self,
            x: Option<&[Number]>,
            _new_x: bool,
            obj_factor: Number,
            _lambda: Option<&[Number]>,
            _new_lambda: bool,
            mode: SparsityRequest<'_>,
        ) -> bool {
            match mode {
                SparsityRequest::Structure { irow, jcol } => {
                    irow[0] = 0;
                    jcol[0] = 0;
                }
                SparsityRequest::Values { values } => {
                    values[0] = -obj_factor * x.unwrap()[0].exp();
                }
            }
            true
        }
        fn finalize_solution(&mut self, sol: Solution<'_>, _d: &IpoptData, _q: &IpoptCq) {
            self.status_x = Some(sol.x[0]);
        }
    }

    let mut app = IpoptApplication::new();
    app.options_mut()
        .set_integer_value("print_level", 0, true, false)
        .unwrap();
    app.initialize().unwrap();
    let inst = Rc::new(RefCell::new(Runaway { status_x: None }));
    let tnlp: Rc<RefCell<dyn TNLP>> = inst.clone();
    let status = app.optimize_tnlp(tnlp);
    assert_ne!(
        status,
        ApplicationReturnStatus::SolveSucceeded,
        "an unbounded problem with a runaway Lagrangian gradient must never \
         certify optimality",
    );
}
