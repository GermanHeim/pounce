//! gh #231 — `PdSearchDirCalc` must not panic when the Mehrotra corrector's
//! affine step is missing.
//!
//! Under `mehrotra_algorithm=yes` the predictor (affine) step is produced by the
//! adaptive-μ oracle and stashed in `IpoptData::delta_aff`. Several legitimate
//! paths leave it unset when the main search-direction solve runs: the affine
//! solve can fail and fall back to the LOQO oracle (which computes no predictor),
//! the probing iterate-quality guard can return early without one, or an early
//! iteration may precede the first affine step. Previously the main-step RHS
//! assembly did `unwrap_or_else(|| panic!(...))` on `delta_aff`, so those states
//! took the whole process down — uncatchable from the C API, the Python
//! bindings, or the GAMS link.
//!
//! The minimal reproducer from the issue: a separable quartic with a single
//! upper-bounded inequality, started far from the optimum with a large target
//! coefficient, which is exactly the regime where the zero/ill-conditioned
//! affine solve leaves `delta_aff` empty. The only property under test is that
//! the solve returns *a status* instead of panicking.

use pounce_algorithm::application::IpoptApplication;
use pounce_common::types::Number;
use pounce_nlp::return_codes::ApplicationReturnStatus;
use pounce_nlp::tnlp::{
    BoundsInfo, IndexStyle, IpoptCq, IpoptData, NlpInfo, Solution, SparsityRequest, StartingPoint,
    TNLP,
};
use std::cell::RefCell;
use std::rc::Rc;

/// `min (x0 − a0)⁴ + (x1 − a1)⁴  s.t.  x0 ≤ a0`, started at the origin.
///
/// With `a = [100, 137]` this is the issue's `a0 ≥ 1e2` panicking family.
struct QuarticWithCap {
    a: [Number; 2],
}

impl TNLP for QuarticWithCap {
    fn get_nlp_info(&mut self) -> Option<NlpInfo> {
        Some(NlpInfo {
            n: 2,
            m: 1,
            nnz_jac_g: 1,
            nnz_h_lag: 2,
            index_style: IndexStyle::C,
        })
    }

    fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
        b.x_l.copy_from_slice(&[-2.0e19; 2]);
        b.x_u.copy_from_slice(&[2.0e19; 2]);
        // g(x) = x0 ≤ a0.
        b.g_l[0] = -2.0e19;
        b.g_u[0] = self.a[0];
        true
    }

    fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
        sp.x.copy_from_slice(&[0.0, 0.0]);
        true
    }

    fn eval_f(&mut self, x: &[Number], _new_x: bool) -> Option<Number> {
        Some((x[0] - self.a[0]).powi(4) + (x[1] - self.a[1]).powi(4))
    }

    fn eval_grad_f(&mut self, x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
        g[0] = 4.0 * (x[0] - self.a[0]).powi(3);
        g[1] = 4.0 * (x[1] - self.a[1]).powi(3);
        true
    }

    fn eval_g(&mut self, x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
        g[0] = x[0];
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
                irow[0] = 0;
                jcol[0] = 0;
            }
            SparsityRequest::Values { values } => {
                values[0] = 1.0;
            }
        }
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
                for i in 0..2 {
                    irow[i] = i as i32;
                    jcol[i] = i as i32;
                }
            }
            SparsityRequest::Values { values } => {
                // g is linear, so the Lagrangian Hessian is the objective's.
                let x = x.expect("eval_h(Values) without x");
                values[0] = obj_factor * 12.0 * (x[0] - self.a[0]).powi(2);
                values[1] = obj_factor * 12.0 * (x[1] - self.a[1]).powi(2);
            }
        }
        true
    }

    fn finalize_solution(&mut self, _sol: Solution<'_>, _d: &IpoptData, _q: &IpoptCq) {}
}

fn solve_mehrotra(a: [Number; 2]) -> ApplicationReturnStatus {
    let mut app = IpoptApplication::new();
    app.options_mut()
        .set_string_value("mehrotra_algorithm", "yes", true, false)
        .unwrap();
    // Keep the run short — we only care that assembling the search direction
    // does not panic, not that this pathological problem converges.
    app.options_mut()
        .set_integer_value("max_iter", 50, true, false)
        .unwrap();
    app.initialize().unwrap();
    let tnlp: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(QuarticWithCap { a }));
    app.optimize_tnlp(tnlp)
}

/// The regression: before the fix this panicked with
/// "PdSearchDirCalc: delta_aff missing for Mehrotra". Now it returns a status.
#[test]
fn missing_affine_step_returns_a_status_instead_of_panicking() {
    let status = solve_mehrotra([100.0, 137.0]);
    eprintln!("gh#231 reproducer finished with {status:?}");
    // Any terminal status is acceptable; the point is that the library returned
    // control to the caller rather than aborting the process. `Internal_Error`
    // (which is what an unwinding panic would map to if it were caught) must not
    // appear, since the fix is a graceful fallback, not a caught panic.
    assert!(
        !matches!(status, ApplicationReturnStatus::InternalError),
        "solve should not report an internal error, got {status:?}"
    );
}

/// The issue notes the panic is fine at small `a0` and fires for `a0 ≥ 1e2`.
/// The small-coefficient case must keep working (and should actually solve).
#[test]
fn small_coefficient_case_still_solves() {
    let status = solve_mehrotra([1.0, 1.0]);
    eprintln!("small-coefficient control finished with {status:?}");
    assert!(
        matches!(
            status,
            ApplicationReturnStatus::SolveSucceeded
                | ApplicationReturnStatus::SolvedToAcceptableLevel
        ),
        "the well-behaved case should still converge, got {status:?}"
    );
}
