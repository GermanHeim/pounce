//! Lifting the live iterate through a presolve layer.
//!
//! The algorithm installs the iterate it hands the intermediate callback
//! ([`pounce_nlp::live_iterate`]) in the space of the problem it solves —
//! the reduced one, when presolve is in the stack. Each presolve wrapper
//! lifts it one layer on its way through `intermediate_callback`, so the
//! user reads their own variables and rows.
//!
//! The lift *is* the wrapper's own `finalize_solution`, run against a
//! stand-in inner TNLP that forwards every evaluation and keeps the
//! `Solution` it is handed instead of finalizing. Reusing the code rather
//! than restating it is the point: the iterate read at the last fire is the
//! solution `finalize_solution` reports because it is computed by the same
//! function — eliminated columns restored, dropped rows re-evaluated, their
//! multipliers recovered — and a later change to that recovery reaches the
//! live iterate without anyone having to remember it.

use std::cell::RefCell;
use std::rc::Rc;

use pounce_common::types::Number;
use pounce_nlp::alg_types::SolverReturn;
use pounce_nlp::live_iterate::{self, UserIterate};
use pounce_nlp::tnlp::{
    BoundsInfo, IpoptCq, IpoptData, NlpInfo, Solution, SparsityRequest, StartingPoint, TNLP,
};

/// Forwards the evaluations a `finalize_solution` makes of its inner TNLP,
/// and captures the solution it would have handed on.
struct Capture {
    inner: Rc<RefCell<dyn TNLP>>,
    captured: Option<UserIterate>,
}

impl TNLP for Capture {
    fn get_nlp_info(&mut self) -> Option<NlpInfo> {
        self.inner.borrow_mut().get_nlp_info()
    }
    fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
        self.inner.borrow_mut().get_bounds_info(b)
    }
    fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
        self.inner.borrow_mut().get_starting_point(sp)
    }
    fn eval_f(&mut self, x: &[Number], new_x: bool) -> Option<Number> {
        self.inner.borrow_mut().eval_f(x, new_x)
    }
    fn eval_grad_f(&mut self, x: &[Number], new_x: bool, g: &mut [Number]) -> bool {
        self.inner.borrow_mut().eval_grad_f(x, new_x, g)
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
    fn finalize_solution(&mut self, sol: Solution<'_>, _d: &IpoptData, _q: &IpoptCq) {
        self.captured = Some(UserIterate {
            x: sol.x.to_vec(),
            z_l: sol.z_l.to_vec(),
            z_u: sol.z_u.to_vec(),
            g: sol.g.to_vec(),
            lambda: sol.lambda.to_vec(),
        });
    }
}

/// The installed live iterate lifted through `this` — `this`'s own
/// `finalize_solution` with `inner` (reached through `slot`) swapped for a
/// [`Capture`] for the duration. `None` when no iterate is installed or it
/// is not shaped like `this`'s problem (`outer_dims`), in which case the
/// caller forwards the callback with the iterate untouched.
///
/// The caller saves and restores whatever state its `finalize_solution`
/// records about the *final* solution, which this run must not overwrite.
pub(crate) fn lift<T: TNLP>(
    this: &mut T,
    slot: fn(&mut T) -> &mut Rc<RefCell<dyn TNLP>>,
    outer_dims: (usize, usize),
) -> Option<UserIterate> {
    let it = live_iterate::current_user_iterate()?;
    let (n, m) = outer_dims;
    if it.x.len() != n
        || it.z_l.len() != n
        || it.z_u.len() != n
        || it.g.len() != m
        || it.lambda.len() != m
    {
        return None;
    }
    let original = Rc::clone(slot(this));
    let capture = Rc::new(RefCell::new(Capture {
        inner: Rc::clone(&original),
        captured: None,
    }));
    let as_dyn: Rc<RefCell<dyn TNLP>> = capture.clone();
    *slot(this) = as_dyn;
    this.finalize_solution(
        Solution {
            // Unread by the lift; the solve has not ended.
            status: SolverReturn::Success,
            x: &it.x,
            z_l: &it.z_l,
            z_u: &it.z_u,
            g: &it.g,
            lambda: &it.lambda,
            obj_value: Number::NAN,
        },
        &IpoptData::default(),
        &IpoptCq::default(),
    );
    *slot(this) = original;
    capture.borrow_mut().captured.take()
}
