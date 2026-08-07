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
    let mut app = IpoptApplication::new();
    app.options_mut()
        .set_integer_value("print_level", 0, true, false)
        .unwrap();
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
    assert_ne!(
        status,
        ApplicationReturnStatus::SolveSucceeded,
        "an LP short of its right-hand side by ~1e8 per row must never certify",
    );
}
