//! gh#884: a biactive complementarity pair makes the exact-product
//! lowering's multipliers diverge, and the run fails at the exact primal
//! solution.
//!
//! **These are characterization tests, not a fix.** gh#884 is open. They
//! pin the mechanism as measured so that a proposed fix has something to
//! move, and — more importantly — one of them pins the *trap*: the
//! obvious remedy makes a different model report success at an objective
//! below its true optimum. If you are here because you are fixing
//! gh#884, read `dev-notes/mpcc-biactive-dual-divergence.md` first; three
//! plausible approaches are already measured and ruled out there.
//!
//! The fixture is `qpec_small` from `benchmarks/mpcc/` under the exact
//! product (`ncp_eq` / `prod_eq`) lowering:
//!
//! ```text
//! min (x-1)^2 + (y1-1)^2 + y2^2
//! s.t. 0 <= x <= 2
//!      0 <= y1  _|_  (2 y1 - 1 - x) >= 0
//!      0 <= y2  _|_  (2 y2 - 1 + x) >= 0
//! ```
//!
//! with `f* = 0` at `(1, 1, 0)`, where pair 1 is strict and **pair 2 is
//! biactive** (`G2 = H2 = 0`). There `grad f` is exactly `0` and the
//! product row's gradient is exactly `(0, 0, 0)`, so `lambda = 0`
//! certifies stationarity with residual `0` — the answer exists and is
//! reachable.
//!
//! What actually happens: the solver reaches the point, and the
//! multipliers on the *linearly dependent* rows run away in near-
//! cancelling pairs. Measured at the returned iterate — rows 2 and 5 both
//! restrict only `y2`, and rows 1 and 4 are likewise parallel:
//!
//! | row | `|grad|_inf` | `lambda` |
//! |---|---:|---:|
//! | 1 (`H1 >= 0`) | `2.0` | `-2.089e10` |
//! | 4 (`G1·H1 = 0`) | `2.0` | `+2.089e10` |
//! | 2 (`G2 >= 0`) | `1.0` | `-7.283e11` |
//! | 5 (`G2·H2 = 0`) | `2.3e-4` | `+1.737e15` |
//!
//! `-7.283e11 + 1.737e15 · 2.3e-4` is the reported `3.253e11` residual.
//!
//! | test | what it pins |
//! |---|---|
//! | `the_primal_converges_while_the_dual_diverges` | the mechanism: exact point, feasible to `2.2e-16`, objective at `f*`, and an unscaled KKT error of `3.3e11` |
//! | `the_hessian_sparsity_pattern_does_not_change_the_outcome` | gh#884's stated prerequisite hypothesis, **refuted** |
//! | `dual_regularization_reaches_the_answer_honestly` | a genuine solve exists at `9.96e-8` unscaled — so this is a POUNCE gap, not a property of the reformulation |
//! | `always_on_dual_regularization_reports_success_below_the_true_optimum` | why the previous test is **not** a licence to change the default |

use std::cell::RefCell;
use std::rc::Rc;

use pounce_algorithm::application::IpoptApplication;
use pounce_common::types::{Index, Number};
use pounce_nlp::return_codes::ApplicationReturnStatus;
use pounce_nlp::tnlp::{
    BoundsInfo, IndexStyle, IpoptCq, IpoptData, NlpInfo, Solution, SparsityRequest, StartingPoint,
    TNLP,
};

const INF: Number = 2e19;

/// What `finalize_solution` hands back, so a test can recompute the
/// stationarity residual itself rather than trusting a reported one.
#[derive(Clone, Default)]
struct Solved {
    x: Vec<Number>,
    lambda: Vec<Number>,
}

/// `qpec_small` under the exact-product lowering. Rows, in order:
/// `0: G1 = y1`, `1: H1 = 2y1 − 1 − x`, `2: G2 = y2`, `3: H2 = 2y2 − 1 + x`,
/// `4: G1·H1`, `5: G2·H2`, the last two equalities at `0`.
struct QpecSmallProdEq {
    captured: Rc<RefCell<Option<Solved>>>,
    /// The MPCC harness hands back a *dense* lower triangle (6 nonzeros)
    /// where the structurally correct pattern has 5 — `(2, 1)` is
    /// identically zero. gh#884 names that difference as the leading
    /// hypothesis for its two paths' different exits, so it is a switch.
    dense_hessian: bool,
}

impl QpecSmallProdEq {
    fn new(dense_hessian: bool) -> (Self, Rc<RefCell<Option<Solved>>>) {
        let captured = Rc::new(RefCell::new(None));
        (
            Self {
                captured: Rc::clone(&captured),
                dense_hessian,
            },
            captured,
        )
    }
}

/// Violation of the model's own rows and bounds at `x`, in the model's
/// units, computed independently of the solver.
fn true_violation(x: &[Number]) -> Number {
    let g = [x[1], 2.0 * x[1] - 1.0 - x[0], x[2], 2.0 * x[2] - 1.0 + x[0]];
    let mut v: Number = 0.0;
    for gi in &g {
        v = v.max((-gi).max(0.0));
    }
    v = v.max((g[0] * g[1]).abs());
    v = v.max((g[2] * g[3]).abs());
    v = v.max((-x[0]).max(0.0));
    v.max((x[0] - 2.0).max(0.0))
}

impl TNLP for QpecSmallProdEq {
    fn get_nlp_info(&mut self) -> Option<NlpInfo> {
        Some(NlpInfo {
            n: 3,
            m: 6,
            nnz_jac_g: 18,
            nnz_h_lag: if self.dense_hessian { 6 } else { 5 },
            index_style: IndexStyle::C,
        })
    }

    fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
        b.x_l[0] = 0.0;
        b.x_u[0] = 2.0;
        b.x_l[1] = -INF;
        b.x_u[1] = INF;
        b.x_l[2] = -INF;
        b.x_u[2] = INF;
        for i in 0..4 {
            b.g_l[i] = 0.0;
            b.g_u[i] = INF;
        }
        for i in 4..6 {
            b.g_l[i] = 0.0;
            b.g_u[i] = 0.0;
        }
        true
    }

    fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
        sp.x[0] = 0.0;
        sp.x[1] = 0.5;
        sp.x[2] = 0.5;
        true
    }

    fn eval_f(&mut self, x: &[Number], _new_x: bool) -> Option<Number> {
        Some((x[0] - 1.0).powi(2) + (x[1] - 1.0).powi(2) + x[2] * x[2])
    }

    fn eval_grad_f(&mut self, x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
        g[0] = 2.0 * (x[0] - 1.0);
        g[1] = 2.0 * (x[1] - 1.0);
        g[2] = 2.0 * x[2];
        true
    }

    fn eval_g(&mut self, x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
        let g1 = x[1];
        let h1 = 2.0 * x[1] - 1.0 - x[0];
        let g2 = x[2];
        let h2 = 2.0 * x[2] - 1.0 + x[0];
        g[0] = g1;
        g[1] = h1;
        g[2] = g2;
        g[3] = h2;
        g[4] = g1 * h1;
        g[5] = g2 * h2;
        true
    }

    fn eval_jac_g(
        &mut self,
        x: Option<&[Number]>,
        _new_x: bool,
        mode: SparsityRequest<'_>,
    ) -> bool {
        match mode {
            SparsityRequest::Structure { irow, jcol } => {
                let mut k = 0;
                for i in 0..6 {
                    for j in 0..3 {
                        irow[k] = i as Index;
                        jcol[k] = j as Index;
                        k += 1;
                    }
                }
                true
            }
            SparsityRequest::Values { values } => {
                let Some(x) = x else { return false };
                let rows = jac_rows(x);
                for (i, row) in rows.iter().enumerate() {
                    values[3 * i..3 * i + 3].copy_from_slice(row);
                }
                true
            }
        }
    }

    fn eval_h(
        &mut self,
        _x: Option<&[Number]>,
        _new_x: bool,
        obj_factor: Number,
        lambda: Option<&[Number]>,
        _new_lambda: bool,
        mode: SparsityRequest<'_>,
    ) -> bool {
        let dense = self.dense_hessian;
        match mode {
            SparsityRequest::Structure { irow, jcol } => {
                let pattern: &[(Index, Index)] = if dense {
                    &[(0, 0), (1, 0), (1, 1), (2, 0), (2, 1), (2, 2)]
                } else {
                    &[(0, 0), (1, 0), (1, 1), (2, 0), (2, 2)]
                };
                for (k, (i, j)) in pattern.iter().enumerate() {
                    irow[k] = *i;
                    jcol[k] = *j;
                }
                true
            }
            SparsityRequest::Values { values } => {
                let l4 = lambda.and_then(|l| l.get(4).copied()).unwrap_or(0.0);
                let l5 = lambda.and_then(|l| l.get(5).copied()).unwrap_or(0.0);
                // ∇²f = diag(2,2,2); row 4 adds ∂²/∂y1² = 4, ∂²/∂x∂y1 = −1;
                // row 5 adds ∂²/∂y2² = 4, ∂²/∂x∂y2 = +1. `(2,1)` is
                // identically zero, which is what makes it optional.
                values[0] = 2.0 * obj_factor;
                values[1] = -l4;
                values[2] = 2.0 * obj_factor + 4.0 * l4;
                values[3] = l5;
                if dense {
                    values[4] = 0.0;
                    values[5] = 2.0 * obj_factor + 4.0 * l5;
                } else {
                    values[4] = 2.0 * obj_factor + 4.0 * l5;
                }
                true
            }
        }
    }

    fn finalize_solution(&mut self, sol: Solution<'_>, _ip_data: &IpoptData, _ip_cq: &IpoptCq) {
        *self.captured.borrow_mut() = Some(Solved {
            x: sol.x.to_vec(),
            lambda: sol.lambda.to_vec(),
        });
    }
}

/// The six constraint gradients at `x`.
fn jac_rows(x: &[Number]) -> [[Number; 3]; 6] {
    [
        [0.0, 1.0, 0.0],
        [-1.0, 2.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 2.0],
        [-x[1], 4.0 * x[1] - 1.0 - x[0], 0.0],
        [x[2], 0.0, 4.0 * x[2] - 1.0 + x[0]],
    ]
}

/// `ralph1` under the `direct` lowering (`G·H <= 0`):
/// `min 2x − y  s.t.  x >= 0, G = y >= 0, H = y − x >= 0, G·H <= 0`,
/// with `f* = 0` at the origin.
///
/// The origin is M-stationary but **not** S-stationary, and NLP KKT is
/// S-stationarity, so no sign-feasible multiplier exists there and
/// failing is the correct outcome. gh#884 makes that an explicit
/// acceptance criterion: a fix that greens this has over-fired.
struct Ralph1Direct {
    captured: Rc<RefCell<Option<Solved>>>,
}

impl Ralph1Direct {
    fn new() -> (Self, Rc<RefCell<Option<Solved>>>) {
        let captured = Rc::new(RefCell::new(None));
        (
            Self {
                captured: Rc::clone(&captured),
            },
            captured,
        )
    }
}

impl TNLP for Ralph1Direct {
    fn get_nlp_info(&mut self) -> Option<NlpInfo> {
        Some(NlpInfo {
            n: 2,
            m: 3,
            nnz_jac_g: 6,
            nnz_h_lag: 3,
            index_style: IndexStyle::C,
        })
    }
    fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
        b.x_l[0] = 0.0;
        b.x_l[1] = -INF;
        b.x_u[0] = INF;
        b.x_u[1] = INF;
        b.g_l[0] = 0.0;
        b.g_u[0] = INF;
        b.g_l[1] = 0.0;
        b.g_u[1] = INF;
        // the `direct` lowering: `G·H <= 0`
        b.g_l[2] = -INF;
        b.g_u[2] = 0.0;
        true
    }
    fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
        sp.x[0] = 0.0;
        sp.x[1] = 0.0;
        true
    }
    fn eval_f(&mut self, x: &[Number], _n: bool) -> Option<Number> {
        Some(2.0 * x[0] - x[1])
    }
    fn eval_grad_f(&mut self, _x: &[Number], _n: bool, g: &mut [Number]) -> bool {
        g[0] = 2.0;
        g[1] = -1.0;
        true
    }
    fn eval_g(&mut self, x: &[Number], _n: bool, g: &mut [Number]) -> bool {
        g[0] = x[1];
        g[1] = x[1] - x[0];
        g[2] = x[1] * (x[1] - x[0]);
        true
    }
    fn eval_jac_g(&mut self, x: Option<&[Number]>, _n: bool, m: SparsityRequest<'_>) -> bool {
        match m {
            SparsityRequest::Structure { irow, jcol } => {
                let mut k = 0;
                for i in 0..3 {
                    for j in 0..2 {
                        irow[k] = i as Index;
                        jcol[k] = j as Index;
                        k += 1;
                    }
                }
                true
            }
            SparsityRequest::Values { values } => {
                let Some(x) = x else { return false };
                values[0] = 0.0;
                values[1] = 1.0;
                values[2] = -1.0;
                values[3] = 1.0;
                values[4] = -x[1];
                values[5] = 2.0 * x[1] - x[0];
                true
            }
        }
    }
    fn eval_h(
        &mut self,
        _x: Option<&[Number]>,
        _n: bool,
        _obj_factor: Number,
        lambda: Option<&[Number]>,
        _nl: bool,
        m: SparsityRequest<'_>,
    ) -> bool {
        match m {
            SparsityRequest::Structure { irow, jcol } => {
                irow[0] = 0;
                jcol[0] = 0;
                irow[1] = 1;
                jcol[1] = 0;
                irow[2] = 1;
                jcol[2] = 1;
                true
            }
            SparsityRequest::Values { values } => {
                let l2 = lambda.and_then(|l| l.get(2).copied()).unwrap_or(0.0);
                values[0] = 0.0;
                values[1] = -l2;
                values[2] = 2.0 * l2;
                true
            }
        }
    }
    fn finalize_solution(&mut self, sol: Solution<'_>, _d: &IpoptData, _c: &IpoptCq) {
        *self.captured.borrow_mut() = Some(Solved {
            x: sol.x.to_vec(),
            lambda: sol.lambda.to_vec(),
        });
    }
}

/// The `.nl`-path options gh#884 measures under.
fn app(always_cd: bool, max_iter: i32) -> IpoptApplication {
    let mut app = IpoptApplication::new();
    {
        let o = app.options_mut();
        let _ = o.set_string_value("sb", "yes", true, false);
        let _ = o.set_integer_value("print_level", 0, true, false);
        let _ = o.set_numeric_value("tol", 1e-8, true, false);
        let _ = o.set_numeric_value("bound_relax_factor", 0.0, true, false);
        let _ = o.set_string_value("honor_original_bounds", "yes", true, false);
        let _ = o.set_integer_value("max_iter", max_iter, true, false);
        if always_cd {
            let _ = o.set_string_value("perturb_always_cd", "yes", true, false);
        }
    }
    app.initialize().expect("initialize");
    app
}

/// The mechanism: the primal reaches the answer and the dual leaves.
///
/// Every number here was measured, not chosen. The point is feasible to
/// `2.2e-16` and sits at the true optimum, and the run still fails —
/// carrying an unscaled KKT error of `3.3e11` produced entirely by
/// multipliers on linearly dependent rows.
#[test]
fn the_primal_converges_while_the_dual_diverges() {
    let (tnlp, captured) = QpecSmallProdEq::new(false);
    let mut a = app(false, 300);
    let status = a.optimize_tnlp(Rc::new(RefCell::new(tnlp)));
    let sol = captured.borrow().clone().expect("finalize_solution ran");
    let s = a.statistics();

    // The primal is at the answer.
    assert!(
        true_violation(&sol.x) <= 1e-12,
        "expected a feasible point, got violation {:.3e} at {:?}",
        true_violation(&sol.x),
        sol.x,
    );
    for (i, want) in [1.0, 1.0, 0.0].iter().enumerate() {
        assert!(
            (sol.x[i] - want).abs() <= 1e-3,
            "x[{i}] = {:.6e} is not at the optimum {want}",
            sol.x[i],
        );
    }
    assert!(
        s.final_objective.abs() <= 1e-6,
        "objective {:.3e} is not at f* = 0",
        s.final_objective,
    );

    // And the run fails anyway, on a dual residual that is enormous in
    // the model's own units.
    assert_ne!(
        status,
        ApplicationReturnStatus::SolveSucceeded,
        "gh#884 is fixed — update this characterization test and the dev-note",
    );
    assert!(
        s.final_unscaled_kkt_error > 1e6,
        "expected the diverged dual gh#884 describes, got {:.3e}",
        s.final_unscaled_kkt_error,
    );

    // The residual is carried by rows whose gradients are parallel: rows
    // 2 and 5 both restrict only `y2`, and their multipliers are huge
    // and opposing. This is the part a fix has to address.
    let rows = jac_rows(&sol.x);
    let contribution =
        |i: usize| sol.lambda[i].abs() * rows[i].iter().fold(0.0_f64, |a, v| a.max(v.abs()));
    assert!(
        contribution(2) > 1e6 && contribution(5) > 1e6,
        "expected the parallel pair (rows 2 and 5) to carry the residual, \
         got {:.3e} and {:.3e}",
        contribution(2),
        contribution(5),
    );
}

/// gh#884's stated prerequisite hypothesis, **refuted**.
///
/// The issue records that the same model exits differently through the
/// MPCC harness's Python path (`Error_In_Step_Computation`, 118 iters)
/// than through `.nl`/CLI (`Solved_To_Acceptable_Level`, 41), and names
/// the Hessian sparsity difference — a dense lower triangle's 6 nonzeros
/// against the structural 5 — as the leading hypothesis, explicitly not
/// established.
///
/// It is not the cause: both patterns produce bit-identical iterates,
/// the same iteration count and the same status. Whatever separates the
/// two paths is somewhere else.
#[test]
fn the_hessian_sparsity_pattern_does_not_change_the_outcome() {
    let mut seen = Vec::new();
    for dense in [false, true] {
        let (tnlp, captured) = QpecSmallProdEq::new(dense);
        let mut a = app(false, 300);
        let status = a.optimize_tnlp(Rc::new(RefCell::new(tnlp)));
        let sol = captured.borrow().clone().expect("finalize_solution ran");
        let s = a.statistics();
        seen.push((status, s.iteration_count, sol.x.clone()));
    }
    assert_eq!(
        seen[0].0, seen[1].0,
        "status differs between the 5- and 6-nonzero Hessian patterns",
    );
    assert_eq!(
        seen[0].1, seen[1].1,
        "iteration count differs between the 5- and 6-nonzero Hessian patterns",
    );
    assert_eq!(
        seen[0].2, seen[1].2,
        "the returned iterate differs between the 5- and 6-nonzero Hessian \
         patterns — bit-identical is the measured result",
    );
}

/// A genuine solve exists, so this is a POUNCE gap and not a property of
/// the reformulation.
///
/// With dual regularization on from the start the same model converges
/// to an **unscaled** KKT error of `~1e-7` — honestly, not by
/// normalising the residual away. That is the evidence that gh#884 has
/// something to fix.
#[test]
fn dual_regularization_reaches_the_answer_honestly() {
    let (tnlp, captured) = QpecSmallProdEq::new(false);
    let mut a = app(true, 300);
    let status = a.optimize_tnlp(Rc::new(RefCell::new(tnlp)));
    let sol = captured.borrow().clone().expect("finalize_solution ran");
    let s = a.statistics();

    assert_eq!(
        status,
        ApplicationReturnStatus::SolveSucceeded,
        "dual regularization no longer reaches the answer on qpec_small",
    );
    assert!(
        s.final_unscaled_kkt_error <= 1e-6,
        "converged, but not in the model's own units: unscaled KKT {:.3e}",
        s.final_unscaled_kkt_error,
    );
    assert!(
        true_violation(&sol.x) <= 1e-9,
        "converged to an infeasible point: violation {:.3e}",
        true_violation(&sol.x),
    );
}

/// **The trap.** The remedy above must not become the default.
///
/// On `ralph1` under the `direct` lowering the origin is M-stationary but
/// not S-stationary, so no sign-feasible multiplier exists and the
/// correct outcome is a failure — which is what POUNCE reports today.
/// Turn dual regularization on always and the same model instead reports
/// plain `Solve_Succeeded` at an objective **below** `f* = 0`, at a point
/// the MPCC does not contain.
///
/// So "fix gh#884 by flipping `perturb_always_cd`" trades one wrong
/// answer for a worse one: gh#884's failure is honest, this is a silent
/// success below the true optimum. Measured, not argued — and it is
/// exactly the acceptance criterion gh#884 states ("a fix that greens all
/// eight has over-fired").
#[test]
fn always_on_dual_regularization_reports_success_below_the_true_optimum() {
    // Today, with the default, the failure is honest.
    let (tnlp, _c) = Ralph1Direct::new();
    let mut a = app(false, 3000);
    let baseline = a.optimize_tnlp(Rc::new(RefCell::new(tnlp)));
    assert_ne!(
        baseline,
        ApplicationReturnStatus::SolveSucceeded,
        "ralph1/direct is expected to fail at the default: no sign-feasible \
         multiplier exists at the origin",
    );

    // With dual regularization always on, it does not.
    let (tnlp, captured) = Ralph1Direct::new();
    let mut a = app(true, 3000);
    let status = a.optimize_tnlp(Rc::new(RefCell::new(tnlp)));
    let sol = captured.borrow().clone().expect("finalize_solution ran");
    let s = a.statistics();

    // `f* = 0`; anything materially below it is outside the MPCC.
    assert!(
        s.final_objective < -1e-6,
        "the trap this test exists for has moved: expected an objective below \
         f* = 0, got {:.6e} at {:?} (status {status:?}). Re-measure before \
         relaxing this — the point of the test is that the objective is \
         reported below the true optimum.",
        s.final_objective,
        sol.x,
    );
    assert_eq!(
        status,
        ApplicationReturnStatus::SolveSucceeded,
        "expected the measured plain success below f*, got {status:?}",
    );
}
