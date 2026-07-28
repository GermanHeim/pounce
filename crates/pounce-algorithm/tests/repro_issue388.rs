//! Regression tests for issue #388: on an unbounded LP the active-set SQP
//! selector (`solver_selection = qp-active-set`) reported `Internal_Error`
//! (AMPL `solve_result_num=500`, "the solver broke") where every IPM
//! selector reported `Diverging_Iterates` (`300`, "your model is
//! unbounded") on the byte-identical model.
//!
//! The unboundedness was in fact detected — the inner QP said so in as many
//! words — but the SQP driver folded that correct diagnosis into
//! `QpFailure(LinearSolverFailure("QP subproblem returned status
//! unbounded"))`, which the application layer mapped to `Internal_Error`.
//! It was also mislabeled a linear-solver failure with no linear solver
//! having failed.
//!
//! An unbounded *step* QP is, on its own, only a statement about the
//! linearization: on a nonconvex NLP the constraints can curve back and the
//! objective turn around, so the fix does not blanket-map it to
//! unboundedness. `pounce-qp` now returns the certified recession ray
//! alongside the verdict and the SQP driver re-tests that ray against the
//! true `f` and `c` before claiming anything. Hence the two halves of this
//! file: the unbounded shapes must reach `DivergingIterates`, and the
//! bounded controls must never do so.

use pounce_algorithm::application::IpoptApplication;
use pounce_common::types::Number;
use pounce_nlp::return_codes::ApplicationReturnStatus;
use pounce_nlp::tnlp::{
    BoundsInfo, IndexStyle, IpoptCq, IpoptData, NlpInfo, Solution, SparsityRequest, StartingPoint,
    TNLP,
};
use std::cell::RefCell;
use std::rc::Rc;

const INF: Number = 2e19; // > the 1e19 infinity sentinel

/// A dense linear program `min cᵀx  s.t.  gₗ ≤ A x ≤ gᵤ,  xₗ ≤ x ≤ xᵤ`.
struct DenseLp {
    n: usize,
    m: usize,
    c: Vec<Number>,
    a: Vec<Number>, // row-major m×n
    g_l: Vec<Number>,
    g_u: Vec<Number>,
    x_l: Vec<Number>,
    x_u: Vec<Number>,
    x0: Vec<Number>,
    final_obj: Option<Number>,
    final_x: Option<Vec<Number>>,
}

impl DenseLp {
    fn a(&self, i: usize, j: usize) -> Number {
        self.a[i * self.n + j]
    }
}

impl TNLP for DenseLp {
    fn get_nlp_info(&mut self) -> Option<NlpInfo> {
        Some(NlpInfo {
            n: self.n as i32,
            m: self.m as i32,
            nnz_jac_g: (self.m * self.n) as i32,
            nnz_h_lag: 0, // linear objective + constraints ⇒ zero Hessian
            index_style: IndexStyle::C,
        })
    }

    fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
        b.x_l.copy_from_slice(&self.x_l);
        b.x_u.copy_from_slice(&self.x_u);
        b.g_l.copy_from_slice(&self.g_l);
        b.g_u.copy_from_slice(&self.g_u);
        true
    }

    fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
        sp.x.copy_from_slice(&self.x0);
        true
    }

    fn eval_f(&mut self, x: &[Number], _new_x: bool) -> Option<Number> {
        Some((0..self.n).map(|j| self.c[j] * x[j]).sum())
    }

    fn eval_grad_f(&mut self, _x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
        g.copy_from_slice(&self.c);
        true
    }

    fn eval_g(&mut self, x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
        for i in 0..self.m {
            g[i] = (0..self.n).map(|j| self.a(i, j) * x[j]).sum();
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
                let mut k = 0;
                for i in 0..self.m {
                    for j in 0..self.n {
                        irow[k] = i as i32;
                        jcol[k] = j as i32;
                        k += 1;
                    }
                }
            }
            SparsityRequest::Values { values } => {
                values.copy_from_slice(&self.a);
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
        // Zero Hessian (LP): no structure entries, nothing to fill.
        true
    }

    fn finalize_solution(&mut self, sol: Solution<'_>, _d: &IpoptData, _q: &IpoptCq) {
        self.final_obj = Some(sol.obj_value);
        self.final_x = Some(sol.x.to_vec());
    }
}

/// A nonconvex NLP with a *linearization* that is unbounded below at the
/// starting point but a *true* objective that is bounded: `min −x  s.t.
/// x² ≤ 1`. At `x = 0` the linearized constraint `0 + 0·p ≤ 1` does not
/// block the descent ray `p = +1` at all, so the step QP is unbounded —
/// yet the feasible set is `[−1, 1]` and the optimum is `f = −1` at
/// `x = 1`. This is the shape that must NOT be reported unbounded.
struct CurvedBounded {
    final_obj: Option<Number>,
}

impl TNLP for CurvedBounded {
    fn get_nlp_info(&mut self) -> Option<NlpInfo> {
        Some(NlpInfo {
            n: 1,
            m: 1,
            nnz_jac_g: 1,
            nnz_h_lag: 1,
            index_style: IndexStyle::C,
        })
    }

    fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
        b.x_l[0] = -INF;
        b.x_u[0] = INF;
        b.g_l[0] = -INF;
        b.g_u[0] = 1.0;
        true
    }

    fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
        sp.x[0] = 0.0;
        true
    }

    fn eval_f(&mut self, x: &[Number], _new_x: bool) -> Option<Number> {
        Some(-x[0])
    }

    fn eval_grad_f(&mut self, _x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
        g[0] = -1.0;
        true
    }

    fn eval_g(&mut self, x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
        g[0] = x[0] * x[0];
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
                irow[0] = 0;
                jcol[0] = 0;
            }
            SparsityRequest::Values { values } => {
                values[0] = 2.0 * x.expect("values mode supplies x")[0];
            }
        }
        true
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
        match mode {
            SparsityRequest::Structure { irow, jcol } => {
                irow[0] = 0;
                jcol[0] = 0;
            }
            SparsityRequest::Values { values } => {
                // ∇²L = obj_factor·∇²f + λ·∇²g = 0 + 2λ.
                values[0] = 2.0 * lambda.map_or(0.0, |l| l[0]) + 0.0 * obj_factor;
            }
        }
        true
    }

    fn finalize_solution(&mut self, sol: Solution<'_>, _d: &IpoptData, _q: &IpoptCq) {
        self.final_obj = Some(sol.obj_value);
    }
}

/// Solve through the active-set SQP engine, selected exactly the way the
/// issue's reproduction does (`solver_selection=qp-active-set`).
fn solve_active_set(tnlp: Rc<RefCell<dyn TNLP>>) -> (ApplicationReturnStatus, Number) {
    let mut app = IpoptApplication::new();
    app.options_mut()
        .set_string_value("solver_selection", "qp-active-set", true, false)
        .unwrap();
    app.options_mut()
        .set_integer_value("print_level", 0, true, false)
        .unwrap();
    app.initialize().unwrap();
    let status = app.optimize_tnlp(tnlp);
    let obj = app.statistics().final_objective;
    (status, obj)
}

/// Solve through the default (IPM) path — the cross-selector oracle the
/// issue leans on: every other selector answers `DivergingIterates` on
/// these models.
fn solve_default(tnlp: Rc<RefCell<dyn TNLP>>) -> ApplicationReturnStatus {
    let mut app = IpoptApplication::new();
    app.options_mut()
        .set_integer_value("print_level", 0, true, false)
        .unwrap();
    app.initialize().unwrap();
    app.optimize_tnlp(tnlp)
}

/// The issue's minimal reproduction: `min −x  s.t.  x ≥ 0`, one variable,
/// one constraint, `x` unbounded above. Must report `DivergingIterates`
/// (`solve_result_num=300`), never `InternalError` (`500`).
#[test]
fn unbounded_lp_reports_diverging_not_internal_error() {
    let inst = Rc::new(RefCell::new(DenseLp {
        n: 1,
        m: 1,
        c: vec![-1.0],
        a: vec![1.0],
        g_l: vec![0.0],
        g_u: vec![INF],
        x_l: vec![-INF],
        x_u: vec![INF],
        x0: vec![0.0],
        final_obj: None,
        final_x: None,
    }));
    let (status, _obj) = solve_active_set(Rc::clone(&inst) as Rc<RefCell<dyn TNLP>>);
    assert!(
        !matches!(status, ApplicationReturnStatus::InternalError),
        "an unbounded LP must not be reported as a solver internal error \
         (issue #388); got {status:?}",
    );
    assert!(
        matches!(status, ApplicationReturnStatus::DivergingIterates),
        "qp-active-set must report DivergingIterates (solve_result_num=300) \
         on min −x s.t. x ≥ 0, matching every other selector (issue #388); \
         got {status:?}",
    );
    // The issue also notes the failing path emitted a null objective and no
    // `x` block, because it bailed out before `finalize_solution`. Returning
    // a status through the normal result path restores both.
    assert!(
        inst.borrow().final_obj.is_some() && inst.borrow().final_x.is_some(),
        "the unbounded verdict must still reach finalize_solution so the \
         report carries an objective and an x block (issue #388)",
    );
}

/// The same model on the default IPM path — the cross-selector oracle. The
/// two selectors must agree.
#[test]
fn unbounded_lp_agrees_across_selectors() {
    let make = || {
        Rc::new(RefCell::new(DenseLp {
            n: 1,
            m: 1,
            c: vec![-1.0],
            a: vec![1.0],
            g_l: vec![0.0],
            g_u: vec![INF],
            x_l: vec![-INF],
            x_u: vec![INF],
            x0: vec![0.0],
            final_obj: None,
            final_x: None,
        })) as Rc<RefCell<dyn TNLP>>
    };
    let (sqp, _) = solve_active_set(make());
    let ipm = solve_default(make());
    assert_eq!(
        format!("{sqp:?}"),
        format!("{ipm:?}"),
        "qp-active-set and the IPM path must return the same status on the \
         same unbounded LP (issue #388)",
    );
}

/// A recession ray in an equality null space over free variables
/// (`min −x₀  s.t.  x₀ − x₁ = 0`, ray `d = (1, 1)`) — the #285 shape, now
/// through the active-set selector.
#[test]
fn unbounded_null_aeq_ray_reports_diverging() {
    let inst = Rc::new(RefCell::new(DenseLp {
        n: 2,
        m: 1,
        c: vec![-1.0, 0.0],
        a: vec![1.0, -1.0],
        g_l: vec![0.0],
        g_u: vec![0.0],
        x_l: vec![-INF, -INF],
        x_u: vec![INF, INF],
        x0: vec![0.0, 0.0],
        final_obj: None,
        final_x: None,
    }));
    let (status, _obj) = solve_active_set(inst as Rc<RefCell<dyn TNLP>>);
    assert!(
        matches!(status, ApplicationReturnStatus::DivergingIterates),
        "a recession ray in null(A_eq) over free variables must report \
         DivergingIterates on the active-set path too (issue #388); got \
         {status:?}",
    );
}

/// Bounded control: the identical LP with the ray blocked by a finite
/// variable bound. Must solve, and must never claim unboundedness.
#[test]
fn bounded_lp_still_solves() {
    let inst = Rc::new(RefCell::new(DenseLp {
        n: 1,
        m: 1,
        c: vec![-1.0],
        a: vec![1.0],
        g_l: vec![0.0],
        g_u: vec![INF],
        x_l: vec![-INF],
        x_u: vec![3.0],
        x0: vec![0.0],
        final_obj: None,
        final_x: None,
    }));
    let (status, obj) = solve_active_set(Rc::clone(&inst) as Rc<RefCell<dyn TNLP>>);
    assert!(
        !matches!(status, ApplicationReturnStatus::DivergingIterates),
        "a bounded LP must never report DivergingIterates (issue #388); got \
         {status:?}",
    );
    assert!(
        matches!(status, ApplicationReturnStatus::SolveSucceeded),
        "min −x s.t. x ≥ 0, x ≤ 3 must solve; got {status:?}",
    );
    assert!(
        (obj - (-3.0)).abs() < 1e-6,
        "expected objective −3 at x = 3; got {obj}",
    );
}

/// Infeasible control: `x ≥ 1` and `x ≤ 0` via a two-row LP. The issue
/// records `qp-active-set` already answering `200` here; the unbounded
/// branch must not have disturbed it.
#[test]
fn infeasible_lp_unchanged() {
    let inst = Rc::new(RefCell::new(DenseLp {
        n: 1,
        m: 2,
        c: vec![-1.0],
        a: vec![1.0, 1.0],
        g_l: vec![1.0, -INF],
        g_u: vec![INF, 0.0],
        x_l: vec![-INF],
        x_u: vec![INF],
        x0: vec![0.0],
        final_obj: None,
        final_x: None,
    }));
    let (status, _obj) = solve_active_set(inst as Rc<RefCell<dyn TNLP>>);
    assert!(
        !matches!(
            status,
            ApplicationReturnStatus::DivergingIterates | ApplicationReturnStatus::SolveSucceeded
        ),
        "an infeasible LP must be reported neither unbounded nor solved \
         (issue #388); got {status:?}",
    );
}

/// The false-positive guard, and the reason the fix verifies the ray rather
/// than trusting the QP's verdict: `min −x  s.t.  x² ≤ 1` has an unbounded
/// step QP at `x = 0` (the linearized constraint does not block `p = +1`)
/// but a bounded feasible set `[−1, 1]`. Reporting this unbounded would
/// tell a modeler their bounded model diverges.
#[test]
fn curved_constraint_never_falsely_unbounded() {
    let inst = Rc::new(RefCell::new(CurvedBounded { final_obj: None }));
    let (status, _obj) = solve_active_set(inst as Rc<RefCell<dyn TNLP>>);
    assert!(
        !matches!(status, ApplicationReturnStatus::DivergingIterates),
        "min −x s.t. x² ≤ 1 is bounded (optimum −1 at x = 1) and must never \
         be reported unbounded, however its step QP behaves (issue #388); \
         got {status:?}",
    );
    assert!(
        !matches!(status, ApplicationReturnStatus::InternalError),
        "and it must not be an internal error either; got {status:?}",
    );
}
