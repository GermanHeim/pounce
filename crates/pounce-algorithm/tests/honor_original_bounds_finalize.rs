//! `honor_original_bounds` at the library boundary — the `x` handed to
//! `TNLP::finalize_solution` (gh#483 follow-up).
//!
//! The CLI reads its `.sol` primal from the `on_converged` hook, but
//! every other consumer — `pounce-py`'s `Problem.solve`, the C interface,
//! any Rust `TNLP` — reads it from `finalize_solution`. Those are two
//! separate lifts of the same iterate, so the projection has to hold on
//! both or the option means different things depending on who is asking.
//! This test pins the `finalize_solution` side; `pounce-cli/tests/
//! honor_original_bounds.rs` pins the `.sol` side.
//!
//! Problem: `min (x − 3)²` over `x ∈ [0, 1]` (`m = 0`). The optimum pins
//! the upper bound, and `bound_relax_factor` (default `1e-8`) widens the
//! box first, so the converged iterate lands just outside it.

use pounce_algorithm::application::IpoptApplication;
use pounce_common::types::{Index, Number};
use pounce_nlp::return_codes::ApplicationReturnStatus;
use pounce_nlp::tnlp::{
    BoundsInfo, IndexStyle, IpoptCq, IpoptData, NlpInfo, Solution, SparsityRequest, StartingPoint,
    TNLP,
};
use std::cell::RefCell;
use std::rc::Rc;

#[derive(Default)]
struct BoundPinned {
    final_x: Option<Vec<Number>>,
}

impl TNLP for BoundPinned {
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
        b.x_u[0] = 1.0;
        true
    }

    fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
        sp.x[0] = 0.5;
        true
    }

    fn eval_f(&mut self, x: &[Number], _new_x: bool) -> Option<Number> {
        Some((x[0] - 3.0).powi(2))
    }

    fn eval_grad_f(&mut self, x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
        g[0] = 2.0 * (x[0] - 3.0);
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
        _x: Option<&[Number]>,
        _new_x: bool,
        obj_factor: Number,
        _lambda: Option<&[Number]>,
        _new_lambda: bool,
        mode: SparsityRequest<'_>,
    ) -> bool {
        match mode {
            SparsityRequest::Structure { irow, jcol } => {
                irow[0] = 0 as Index;
                jcol[0] = 0 as Index;
            }
            SparsityRequest::Values { values } => values[0] = 2.0 * obj_factor,
        }
        true
    }

    fn finalize_solution(&mut self, sol: Solution<'_>, _d: &IpoptData, _q: &IpoptCq) {
        self.final_x = Some(sol.x.to_vec());
    }
}

fn solve(honor: Option<bool>) -> f64 {
    let mut app = IpoptApplication::new();
    if let Some(v) = honor {
        app.options_mut()
            .set_string_value(
                "honor_original_bounds",
                if v { "yes" } else { "no" },
                true,
                false,
            )
            .unwrap();
    }
    app.options_mut()
        .set_integer_value("print_level", 0, true, false)
        .unwrap();
    app.initialize().unwrap();

    let concrete = Rc::new(RefCell::new(BoundPinned::default()));
    let tnlp: Rc<RefCell<dyn TNLP>> = Rc::clone(&concrete) as _;
    let status = app.optimize_tnlp(tnlp);
    assert!(
        matches!(
            status,
            ApplicationReturnStatus::SolveSucceeded
                | ApplicationReturnStatus::SolvedToAcceptableLevel
        ),
        "unexpected status {status:?}",
    );
    let x = concrete
        .borrow()
        .final_x
        .clone()
        .expect("finalize_solution never ran");
    x[0]
}

/// Unset and explicit-`no` behave identically, and both report the
/// relaxed point — upstream's default, unchanged by the fix.
#[test]
fn default_hands_finalize_the_unprojected_point() {
    let unset = solve(None);
    let explicit_no = solve(Some(false));
    assert!(unset > 1.0, "expected x past the bound, got {unset}");
    assert!((unset - 1.0).abs() < 1e-6, "…but only by the relaxation");
    assert_eq!(unset, explicit_no, "unset and `no` must agree");
}

/// `yes` projects: the TNLP is handed a point inside its own bounds.
/// Pre-fix the option was never read and this equalled the default.
#[test]
fn opting_in_hands_finalize_a_point_inside_the_bounds() {
    let x = solve(Some(true));
    assert!(x <= 1.0, "x = {x} is outside the declared bound 1");
    assert_eq!(x, 1.0, "an active bound should project exactly onto it");
}
