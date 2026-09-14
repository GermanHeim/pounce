//! gh#937 — the reduced Hessian's sign convention, pinned at the public
//! API instead of in passing.
//!
//! `Solver::compute_reduced_hessian` and `SensSolve::with_reduced_hessian`
//! return `−H_R`, not `H_R`: the pin rows map into the `y_c` multiplier
//! block, whose diagonal block of `K⁻¹` is `−(A H⁻¹ Aᵀ)⁻¹`, and the
//! `−obj_scal` residual factor of `compute_reduced_hessian` leaves the
//! minus in place. That is deliberate — `−inv` of the returned matrix is
//! the covariance — but before gh#937 it was asserted only inside
//! `crossover_sigma_downstream.rs` and `crossover_sigma_frame.rs`, where
//! it is incidental to what those files are testing. A caller had to read
//! the test suite to learn it, and the way it bites is silent: an
//! identifiability read takes the leading (smallest) eigenvalues as the
//! soft directions and gets the **stiffest** ones, from unit-norm,
//! sign-pinned, entirely plausible-looking vectors.
//!
//! The fixture is the smallest model that can tell the three candidate
//! conventions apart:
//!
//! ```text
//! min x0² + x1² + x0·x1   s.t.  x0 = p0,  x1 = p1
//! ```
//!
//! Every variable is pinned, so the null space of the active constraints
//! is `{0}` and the reduced Hessian is the objective Hessian itself,
//! `H = [[2, 1], [1, 2]]`. There is no projection to get wrong, so the
//! returned matrix is `+H`, `−H` or `H⁻¹` and nothing else — and those
//! three differ in **magnitude** as well as sign (`H`'s entries are
//! `2, 1`; `H⁻¹`'s are `2/3, 1/3`), so an assertion that a negation
//! satisfies cannot be satisfied by an inversion. `H`'s eigenvalues are
//! `1` (along `[1, −1]`) and `3` (along `[1, 1]`), far enough apart that
//! an ordering claim about them is not a roundoff claim.
//!
//! The runnable narration of the same check is
//! `examples/rh_orientation_check.rs`.
//!
//! **Mutation table** — each row is a change to the convention that must
//! turn a test here red:
//!
//! | change | what goes red |
//! | --- | --- |
//! | drop the `−` from `compute_reduced_hessian`'s `factor` (`pounce-sens-core`) | [`the_pin_path_returns_the_negated_reduced_hessian`], [`the_ascending_spectrum_runs_stiffest_first`] |
//! | return `H_R⁻¹` (select x rows rather than `y_c` rows) | [`the_pin_path_returns_the_negated_reduced_hessian`] on magnitude, before sign |
//! | let `obj_scal` flip or absorb the sign | [`obj_scal_scales_the_magnitude_and_leaves_the_sign`] |
//! | make `compute_reduced_hessian_scaled` negate independently of `df` | [`the_solver_space_value_does_not_inherit_the_orientation`] |
//! | let a negative `obj_scaling_factor` reach the natural-units value | [`the_solver_space_value_does_not_inherit_the_orientation`] |
//! | drift `SensSolve`'s field away from the session API | [`the_builder_and_the_session_api_agree_on_the_sign`] |

use std::cell::RefCell;
use std::rc::Rc;

use pounce_algorithm::application::IpoptApplication;
use pounce_common::types::{Index, Number};
use pounce_nlp::return_codes::ApplicationReturnStatus;
use pounce_nlp::tnlp::{
    BoundsInfo, IndexStyle, IpoptCq, IpoptData, NlpInfo, Solution, SparsityRequest, StartingPoint,
    TNLP,
};
use pounce_sensitivity::{SensSolve, Solver};

/// `min x0² + x1² + x0·x1`  s.t.  `g0: x0 = p0`, `g1: x1 = p1`.
///
/// The objective Hessian is `H = [[2, 1], [1, 2]]`. Both variables are
/// pinned, so the null space of the active constraints is `{0}` and the
/// reduced Hessian over the two pin rows is `H` itself — which is the
/// point: there is no projection to get wrong, so whatever the accessor
/// returns is `±H` or `H⁻¹` and nothing else.
struct PinnedQuadratic {
    p0: Number,
    p1: Number,
}

impl TNLP for PinnedQuadratic {
    fn get_nlp_info(&mut self) -> Option<NlpInfo> {
        Some(NlpInfo {
            n: 2,
            m: 2,
            nnz_jac_g: 2,
            nnz_h_lag: 3,
            index_style: IndexStyle::C,
        })
    }

    fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
        for k in 0..2 {
            b.x_l[k] = -1.0e19;
            b.x_u[k] = 1.0e19;
        }
        b.g_l[0] = self.p0;
        b.g_u[0] = self.p0;
        b.g_l[1] = self.p1;
        b.g_u[1] = self.p1;
        true
    }

    fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
        sp.x[0] = self.p0;
        sp.x[1] = self.p1;
        true
    }

    fn eval_f(&mut self, x: &[Number], _new_x: bool) -> Option<Number> {
        Some(x[0] * x[0] + x[1] * x[1] + x[0] * x[1])
    }

    fn eval_grad_f(&mut self, x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
        g[0] = 2.0 * x[0] + x[1];
        g[1] = 2.0 * x[1] + x[0];
        true
    }

    fn eval_g(&mut self, x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
        g[0] = x[0];
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
                irow.copy_from_slice(&[0 as Index, 1 as Index]);
                jcol.copy_from_slice(&[0 as Index, 1 as Index]);
            }
            SparsityRequest::Values { values } => values.copy_from_slice(&[1.0, 1.0]),
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
        match mode {
            SparsityRequest::Structure { irow, jcol } => {
                // lower triangle of [[2, 1], [1, 2]]
                irow.copy_from_slice(&[0 as Index, 1 as Index, 1 as Index]);
                jcol.copy_from_slice(&[0 as Index, 0 as Index, 1 as Index]);
            }
            SparsityRequest::Values { values } => {
                values.copy_from_slice(&[2.0 * obj_factor, obj_factor, 2.0 * obj_factor]);
            }
        }
        true
    }

    fn finalize_solution(&mut self, _sol: Solution<'_>, _d: &IpoptData, _q: &IpoptCq) {}
}

/// The true reduced Hessian `H_R = H`, column-major.
const H: [Number; 4] = [2.0, 1.0, 1.0, 2.0];

/// `H⁻¹`, column-major — the third candidate, kept explicit so the
/// assertions discriminate on magnitude and not only on sign.
const H_INV: [Number; 4] = [2.0 / 3.0, -1.0 / 3.0, -1.0 / 3.0, 2.0 / 3.0];

const TOL: Number = 1e-7;

fn make_app() -> IpoptApplication {
    let mut app = IpoptApplication::new();
    app.options_mut()
        .set_integer_value("print_level", 0, true, false)
        .unwrap();
    app.options_mut()
        .set_string_value("sb", "yes", true, false)
        .unwrap();
    app.initialize().unwrap();
    app
}

fn converged_solver() -> Solver {
    converged_solver_with(None)
}

/// `obj_scaling` of `Some(-1.0)` declares a maximization, which is the
/// only way in this fixture to give `df` a sign.
fn converged_solver_with(obj_scaling: Option<Number>) -> Solver {
    let tnlp: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(PinnedQuadratic { p0: 1.0, p1: 2.0 }));
    let mut app = make_app();
    if let Some(v) = obj_scaling {
        app.options_mut()
            .set_numeric_value("obj_scaling_factor", v, true, false)
            .unwrap();
    }
    let mut solver = Solver::new(app, tnlp);
    let status = solver.solve();
    assert!(
        matches!(
            status,
            ApplicationReturnStatus::SolveSucceeded
                | ApplicationReturnStatus::SolvedToAcceptableLevel
        ),
        "solve failed: {status:?}"
    );
    solver
}

fn max_abs_diff(got: &[Number], want: &[Number; 4]) -> Number {
    (0..4)
        .map(|k| (got[k] - want[k]).abs())
        .fold(0.0 as Number, Number::max)
}

#[test]
fn the_pin_path_returns_the_negated_reduced_hessian() {
    let solver = converged_solver();
    let hr = solver
        .compute_reduced_hessian(&[0, 1], 1.0)
        .expect("reduced Hessian");
    assert_eq!(hr.len(), 4);

    let neg_h = [-H[0], -H[1], -H[2], -H[3]];
    assert!(
        max_abs_diff(&hr, &neg_h) < TOL,
        "expected −H_R = {neg_h:?}, got {hr:?}"
    );
    // The two conventions this is NOT, spelled out so a failure names the
    // one that was substituted rather than just printing a mismatch.
    assert!(
        max_abs_diff(&hr, &H) > 1.0,
        "returned +H_R = {hr:?}; the convention is −H_R (gh#937)"
    );
    assert!(
        max_abs_diff(&hr, &H_INV) > 1.0,
        "returned H_R⁻¹ = {hr:?}; the pin rows select the y_c block, whose \
         diagonal block of K⁻¹ is −(A H⁻¹ Aᵀ)⁻¹ — an inverse of an inverse, \
         i.e. ±H_R itself, not a submatrix of K⁻¹ (gh#937)"
    );
}

#[test]
fn the_ascending_spectrum_runs_stiffest_first() {
    let solver = converged_solver();
    let (hr, vals, vecs) = solver
        .compute_reduced_hessian_eigen(&[0, 1], 1.0)
        .expect("reduced Hessian eigendecomposition");

    // Same matrix the non-eigen entry point returns.
    assert!(
        max_abs_diff(&hr, &[-H[0], -H[1], -H[2], -H[3]]) < TOL,
        "the eigen entry point returned a different matrix: {hr:?}"
    );

    // Ascending on −H_R ⇒ most negative first ⇒ the STIFFEST mode first.
    assert!(
        (vals[0] - (-3.0)).abs() < TOL && (vals[1] - (-1.0)).abs() < TOL,
        "expected [−3, −1] (ascending on −H_R), got {vals:?}"
    );
    assert!(
        vals[0] < vals[1],
        "`symmetric_eigen` must report ascending order: {vals:?}"
    );

    // …and the leading column is the curvature-3 direction [1, 1]/√2,
    // NOT the soft [1, −1]/√2 an unqualified "smallest eigenvalue first"
    // read would expect. This is the gh#937 foot-gun as an assertion.
    let s = 1.0 / (2.0 as Number).sqrt();
    assert!(
        (vecs[0].abs() - s).abs() < TOL
            && (vecs[1].abs() - s).abs() < TOL
            && vecs[0] * vecs[1] > 0.0,
        "leading eigenvector should be the stiff ±[1, 1]/√2, got [{}, {}]",
        vecs[0],
        vecs[1]
    );
    assert!(
        vecs[2] * vecs[3] < 0.0,
        "trailing eigenvector should be the soft ±[1, −1]/√2, got [{}, {}]",
        vecs[2],
        vecs[3]
    );
}

#[test]
fn obj_scal_scales_the_magnitude_and_leaves_the_sign() {
    let solver = converged_solver();
    let hr = solver
        .compute_reduced_hessian(&[0, 1], 2.5)
        .expect("reduced Hessian");
    let want = [-2.5 * H[0], -2.5 * H[1], -2.5 * H[2], -2.5 * H[3]];
    assert!(
        max_abs_diff(&hr, &want) < TOL,
        "obj_scal is a plain positive multiplier on −H_R: expected {want:?}, got {hr:?}"
    );
}

#[test]
fn the_solver_space_value_does_not_inherit_the_orientation() {
    // The natural-units value is `−H_R` whatever the scaling. The
    // solver-space one is that times `df / (dc_i·dc_j)`, and `df` is
    // NEGATIVE under a declared maximization — so the two entry points
    // disagree about orientation there, on purpose. Both branches are
    // asserted, because a fixture that only ever ran at `df = 1` would
    // report the two as interchangeable.
    //
    // No constraint scaling is active on this model (`dc = 1`), so `df`
    // is the whole factor and the flip is exactly the sign of `df`.
    for (obj_scaling, want_scaled_sign) in [(None, -1.0), (Some(-1.0), 1.0)] {
        let solver = converged_solver_with(obj_scaling);
        let hr = solver
            .compute_reduced_hessian(&[0, 1], 1.0)
            .expect("reduced Hessian");
        let scaled = solver
            .compute_reduced_hessian_scaled(&[0, 1], 1.0)
            .expect("solver-space reduced Hessian");
        let (df, dc, _) = solver.nlp_scaling().expect("nlp scaling");

        // The natural-units value is unmoved by the objective's sign.
        assert!(
            max_abs_diff(&hr, &[-H[0], -H[1], -H[2], -H[3]]) < TOL,
            "obj_scaling_factor={obj_scaling:?}: natural units must stay −H_R, got {hr:?}"
        );
        assert!(
            dc.is_none(),
            "fixture assumption broken: constraint scaling is active ({dc:?}), \
             so `df` is no longer the whole factor"
        );
        assert!(
            (df - obj_scaling.unwrap_or(1.0)).abs() < TOL,
            "expected df = {:?}, got {df}",
            obj_scaling.unwrap_or(1.0)
        );

        // …while the solver-space value carries `df`'s sign.
        let want = [
            want_scaled_sign * H[0],
            want_scaled_sign * H[1],
            want_scaled_sign * H[2],
            want_scaled_sign * H[3],
        ];
        assert!(
            max_abs_diff(&scaled, &want) < TOL,
            "obj_scaling_factor={obj_scaling:?}: expected solver-space {want:?}, \
             got {scaled:?}"
        );
    }
}

#[test]
fn the_builder_and_the_session_api_agree_on_the_sign() {
    let mut app = make_app();
    let tnlp: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(PinnedQuadratic { p0: 1.0, p1: 2.0 }));
    let result = SensSolve::new(vec![0, 1])
        .with_reduced_hessian_eigen()
        .run(&mut app, tnlp);
    assert!(
        matches!(
            result.status,
            ApplicationReturnStatus::SolveSucceeded
                | ApplicationReturnStatus::SolvedToAcceptableLevel
        ),
        "solve failed: {:?}",
        result.status
    );
    let hr = result
        .reduced_hessian
        .expect("with_reduced_hessian_eigen populates reduced_hessian");
    assert!(
        max_abs_diff(&hr, &[-H[0], -H[1], -H[2], -H[3]]) < TOL,
        "the one-shot builder reports a different orientation than the \
         session API: {hr:?}"
    );
    let vals = result
        .reduced_hessian_eigenvalues
        .expect("eigenvalues populated");
    assert!(
        (vals[0] - (-3.0)).abs() < TOL && (vals[1] - (-1.0)).abs() < TOL,
        "expected [−3, −1], got {vals:?}"
    );
}
