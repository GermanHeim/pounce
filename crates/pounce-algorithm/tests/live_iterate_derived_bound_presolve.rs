//! The live-iterate lift and the derived-bound dual recovery together.
//!
//! Each presolve layer lifts the live iterate by running its own
//! `finalize_solution` (see `pounce-presolve/src/live_lift.rs`), and that
//! function now also re-solves the original problem's multipliers when a
//! bound presolve derived is active (`dual_recovery.rs`, M24). This model
//! reaches both: `x0 >= 4` and `x0 = 2 x1` make the elimination derive
//! `x1 >= 2`, which is the active bound of the reduced problem.
//!
//! Two things must hold. The iterate read at the last callback fire is the
//! solution `finalize_solution` reports, so the recovery runs inside the
//! lift as well. And the final duals equal `presolve=no`'s, so no state a
//! lift recorded is left where the final solution's belongs.
//!
//! Model: `min x0^2 + x1^2  s.t.  x0 - 2 x1 = 0,  x0 >= 4`, so `x* = (4, 2)`
//! with a unique multiplier pair.

use pounce_algorithm::application::IpoptApplication;
use pounce_algorithm::intermediate::with_current_user_iterate;
use pounce_common::types::Number;
use pounce_nlp::return_codes::ApplicationReturnStatus;
use pounce_nlp::tnlp::{
    BoundsInfo, IndexStyle, IpoptCq, IpoptData, IterStats, Linearity, NlpInfo, Solution,
    SparsityRequest, StartingPoint, TNLP,
};
use std::cell::RefCell;
use std::rc::Rc;

#[derive(Clone, Debug, Default)]
struct Point {
    x: Vec<Number>,
    z_l: Vec<Number>,
    z_u: Vec<Number>,
    lambda: Vec<Number>,
}

#[derive(Default)]
struct Derived {
    fires: usize,
    last: Option<Point>,
    out: Option<Point>,
}

impl TNLP for Derived {
    fn get_nlp_info(&mut self) -> Option<NlpInfo> {
        Some(NlpInfo {
            n: 2,
            m: 1,
            nnz_jac_g: 2,
            nnz_h_lag: 2,
            index_style: IndexStyle::C,
        })
    }
    fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
        b.x_l.copy_from_slice(&[4.0, -1.0e19]);
        b.x_u.copy_from_slice(&[1.0e19, 1.0e19]);
        b.g_l[0] = 0.0;
        b.g_u[0] = 0.0;
        true
    }
    fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
        sp.x.copy_from_slice(&[5.0, 3.0]);
        true
    }
    fn get_constraints_linearity(&mut self, t: &mut [Linearity]) -> bool {
        t[0] = Linearity::Linear;
        true
    }
    fn eval_f(&mut self, x: &[Number], _n: bool) -> Option<Number> {
        Some(x[0] * x[0] + x[1] * x[1])
    }
    fn eval_grad_f(&mut self, x: &[Number], _n: bool, g: &mut [Number]) -> bool {
        g[0] = 2.0 * x[0];
        g[1] = 2.0 * x[1];
        true
    }
    fn eval_g(&mut self, x: &[Number], _n: bool, g: &mut [Number]) -> bool {
        g[0] = x[0] - 2.0 * x[1];
        true
    }
    fn eval_jac_g(&mut self, _x: Option<&[Number]>, _n: bool, mode: SparsityRequest<'_>) -> bool {
        match mode {
            SparsityRequest::Structure { irow, jcol } => {
                irow.copy_from_slice(&[0, 0]);
                jcol.copy_from_slice(&[0, 1]);
            }
            SparsityRequest::Values { values } => values.copy_from_slice(&[1.0, -2.0]),
        }
        true
    }
    fn eval_h(
        &mut self,
        _x: Option<&[Number]>,
        _n: bool,
        obj_factor: Number,
        _l: Option<&[Number]>,
        _nl: bool,
        mode: SparsityRequest<'_>,
    ) -> bool {
        match mode {
            SparsityRequest::Structure { irow, jcol } => {
                irow.copy_from_slice(&[0, 1]);
                jcol.copy_from_slice(&[0, 1]);
            }
            SparsityRequest::Values { values } => {
                values.copy_from_slice(&[2.0 * obj_factor, 2.0 * obj_factor])
            }
        }
        true
    }
    fn wants_current_iterate(&self) -> bool {
        true
    }
    fn intermediate_callback(&mut self, _st: IterStats, _d: &IpoptData, _q: &IpoptCq) -> bool {
        self.fires += 1;
        self.last = with_current_user_iterate(|u| Point {
            x: u.x.clone(),
            z_l: u.z_l.clone(),
            z_u: u.z_u.clone(),
            lambda: u.lambda.clone(),
        });
        true
    }
    fn finalize_solution(&mut self, s: Solution<'_>, _d: &IpoptData, _q: &IpoptCq) {
        self.out = Some(Point {
            x: s.x.to_vec(),
            z_l: s.z_l.to_vec(),
            z_u: s.z_u.to_vec(),
            lambda: s.lambda.to_vec(),
        });
    }
}

fn solve(algorithm: &str, presolve: bool) -> Derived {
    let p = Rc::new(RefCell::new(Derived::default()));
    let mut a = IpoptApplication::new();
    a.initialize().unwrap();
    let o = a.options_mut();
    o.set_integer_value("print_level", 0, true, false).unwrap();
    o.set_string_value("algorithm", algorithm, true, false)
        .unwrap();
    o.set_string_value("presolve", if presolve { "yes" } else { "no" }, true, false)
        .unwrap();
    if presolve {
        o.set_string_value("presolve_linear_eq_reduction", "yes", true, false)
            .unwrap();
    }
    assert_eq!(
        a.optimize_tnlp(p.clone()),
        ApplicationReturnStatus::SolveSucceeded,
        "{algorithm} presolve={presolve}"
    );
    Rc::try_unwrap(p).ok().unwrap().into_inner()
}

#[track_caller]
fn assert_close(what: &str, got: &[Number], want: &[Number], tol: Number) {
    assert_eq!(got.len(), want.len(), "{what}: length {got:?} vs {want:?}");
    for (k, (&g, &w)) in got.iter().zip(want).enumerate() {
        assert!(
            (g - w).abs() <= tol * w.abs().max(1.0),
            "{what}[{k}]: {g} vs {w} ({got:?} vs {want:?})"
        );
    }
}

fn check(algorithm: &str) {
    let off = solve(algorithm, false);
    let on = solve(algorithm, true);
    let want = off.out.expect("presolve=no finalize");
    let fin = on.out.expect("presolve=yes finalize");
    assert_close("x*", &want.x, &[4.0, 2.0], 1e-6);
    // The final record: the recovery's answer, not a lift's leftover.
    assert_close("lambda vs presolve=no", &fin.lambda, &want.lambda, 1e-5);
    assert_close("z_L vs presolve=no", &fin.z_l, &want.z_l, 1e-5);
    assert_close("z_U vs presolve=no", &fin.z_u, &want.z_u, 1e-5);
    assert!(
        want.z_l[0] > 1.0,
        "the declared x0 >= 4 must carry the multiplier, got {:?}",
        want.z_l
    );
    // The live iterate at the last fire is that same solution.
    assert!(on.fires > 0, "{algorithm}: the callback never fired");
    let last = on.last.expect("a live iterate at the last fire");
    assert_close("live x", &last.x, &fin.x, 1e-9);
    assert_close("live lambda", &last.lambda, &fin.lambda, 1e-9);
    assert_close("live z_L", &last.z_l, &fin.z_l, 1e-9);
    assert_close("live z_U", &last.z_u, &fin.z_u, 1e-9);
}

#[test]
fn ipm_live_iterate_and_final_duals_with_an_active_derived_bound() {
    check("interior-point");
}

#[test]
fn sqp_live_iterate_and_final_duals_with_an_active_derived_bound() {
    check("active-set-sqp");
}
