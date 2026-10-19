//! The live iterate at the last intermediate-callback fire is the solution
//! `finalize_solution` reports, on both arms, under every layer that sits
//! between the algorithm and the user's TNLP.
//!
//! Upstream `TNLP::get_curr_iterate` (Ipopt 3.14, `IpTNLP.cpp`) reports the
//! iterate in the user's space: fixed variables reinserted (`ResortX`), and
//! their bound multipliers recovered by `TNLPAdapter::ResortBoundMultipliers`
//! as `z_L − z_U = ∇f + Jᵀλ` under `make_parameter`. The interior-point arm
//! installed no materialised iterate at all, so `with_current_user_iterate`
//! read `None` there; the C API's lazy reader left those multipliers at `0`.
//!
//! Model: `min K·[(x0-3)² + (x1-3)² + (x2-1)² + (x3-4)²]`
//! s.t. `g0 = x0 + x1 + x3 ≤ 3.5`, `g1 = x0 − x2 = 0`,
//! `x0` free, `x1 ∈ [-10, 1]` (upper bound active), `x2 ≥ 0`,
//! `x3 = 1` fixed (`x_L = x_U`), so `make_parameter` removes it and its
//! stationarity row `2K(1-4) + λ0` is carried by a bound multiplier.

use pounce_algorithm::application::IpoptApplication;
use pounce_algorithm::intermediate::with_current_user_iterate;
use pounce_common::types::Number;
use pounce_nlp::return_codes::ApplicationReturnStatus;
use pounce_nlp::tnlp::{
    BoundsInfo, IndexStyle, IpoptCq, IpoptData, IterStats, NlpInfo, ScalingRequest, Solution,
    SparsityRequest, StartingPoint, TNLP,
};
use std::cell::RefCell;
use std::rc::Rc;

const D: [Number; 4] = [2.0, 0.5, 4.0, 3.0];

#[derive(Clone, Debug, Default, PartialEq)]
struct Point {
    x: Vec<Number>,
    z_l: Vec<Number>,
    z_u: Vec<Number>,
    g: Vec<Number>,
    lambda: Vec<Number>,
}

#[derive(Default)]
struct Model {
    /// Objective multiplier: 1000 makes gradient-based scaling engage.
    k: Number,
    x_scaling: bool,
    fires: usize,
    last: Option<Point>,
    out: Option<Point>,
}

impl TNLP for Model {
    fn get_nlp_info(&mut self) -> Option<NlpInfo> {
        Some(NlpInfo {
            n: 4,
            m: 2,
            nnz_jac_g: 5,
            nnz_h_lag: 4,
            index_style: IndexStyle::C,
        })
    }
    fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
        b.x_l.copy_from_slice(&[-1e19, -10.0, 0.0, 1.0]);
        b.x_u.copy_from_slice(&[1e19, 1.0, 1e19, 1.0]);
        b.g_l.copy_from_slice(&[-1e19, 0.0]);
        b.g_u.copy_from_slice(&[3.5, 0.0]);
        true
    }
    fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
        sp.x.copy_from_slice(&[0.0, 0.0, 0.5, 1.0]);
        true
    }
    fn get_scaling_parameters(&mut self, r: ScalingRequest<'_>) -> bool {
        *r.obj_scaling = 3.0;
        *r.use_g_scaling = true;
        r.g_scaling.copy_from_slice(&[0.01, 5.0]);
        *r.use_x_scaling = self.x_scaling;
        if self.x_scaling {
            r.x_scaling.copy_from_slice(&D);
        }
        true
    }
    fn eval_f(&mut self, x: &[Number], _n: bool) -> Option<Number> {
        Some(
            self.k
                * ((x[0] - 3.0).powi(2)
                    + (x[1] - 3.0).powi(2)
                    + (x[2] - 1.0).powi(2)
                    + (x[3] - 4.0).powi(2)),
        )
    }
    fn eval_grad_f(&mut self, x: &[Number], _n: bool, g: &mut [Number]) -> bool {
        g[0] = 2.0 * self.k * (x[0] - 3.0);
        g[1] = 2.0 * self.k * (x[1] - 3.0);
        g[2] = 2.0 * self.k * (x[2] - 1.0);
        g[3] = 2.0 * self.k * (x[3] - 4.0);
        true
    }
    fn eval_g(&mut self, x: &[Number], _n: bool, g: &mut [Number]) -> bool {
        g[0] = x[0] + x[1] + x[3];
        g[1] = x[0] - x[2];
        true
    }
    fn eval_jac_g(&mut self, _x: Option<&[Number]>, _n: bool, mode: SparsityRequest<'_>) -> bool {
        match mode {
            SparsityRequest::Structure { irow, jcol } => {
                irow.copy_from_slice(&[0, 0, 0, 1, 1]);
                jcol.copy_from_slice(&[0, 1, 3, 0, 2]);
            }
            SparsityRequest::Values { values } => {
                values.copy_from_slice(&[1.0, 1.0, 1.0, 1.0, -1.0]);
            }
        }
        true
    }
    fn eval_h(
        &mut self,
        _x: Option<&[Number]>,
        _n: bool,
        of: Number,
        _l: Option<&[Number]>,
        _nl: bool,
        mode: SparsityRequest<'_>,
    ) -> bool {
        match mode {
            SparsityRequest::Structure { irow, jcol } => {
                irow.copy_from_slice(&[0, 1, 2, 3]);
                jcol.copy_from_slice(&[0, 1, 2, 3]);
            }
            SparsityRequest::Values { values } => values.copy_from_slice(&[2.0 * self.k * of; 4]),
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
            g: u.g.clone(),
            lambda: u.lambda.clone(),
        });
        true
    }
    fn finalize_solution(&mut self, s: Solution<'_>, _d: &IpoptData, _q: &IpoptCq) {
        self.out = Some(Point {
            x: s.x.to_vec(),
            z_l: s.z_l.to_vec(),
            z_u: s.z_u.to_vec(),
            g: s.g.to_vec(),
            lambda: s.lambda.to_vec(),
        });
    }
}

#[derive(Clone, Copy, Debug)]
enum Scaling {
    /// `gradient-based` on a model whose gradient exceeds
    /// `nlp_scaling_max_gradient`, times `obj_scaling_factor = 3`.
    GradientBased,
    /// `user-scaling` with non-unit per-variable factors (`ScalingTnlp`).
    UserX,
}

fn solve(algorithm: &str, scaling: Scaling) -> Model {
    let model = match scaling {
        Scaling::GradientBased => Model {
            k: 1000.0,
            ..Default::default()
        },
        Scaling::UserX => Model {
            k: 1.0,
            x_scaling: true,
            ..Default::default()
        },
    };
    let p = Rc::new(RefCell::new(model));
    let mut a = IpoptApplication::new();
    a.initialize().unwrap();
    let o = a.options_mut();
    o.set_integer_value("print_level", 0, true, false).unwrap();
    o.set_string_value("algorithm", algorithm, true, false)
        .unwrap();
    match scaling {
        Scaling::GradientBased => {
            o.set_string_value("nlp_scaling_method", "gradient-based", true, false)
                .unwrap();
            o.set_numeric_value("obj_scaling_factor", 3.0, true, false)
                .unwrap();
        }
        Scaling::UserX => {
            o.set_string_value("nlp_scaling_method", "user-scaling", true, false)
                .unwrap();
        }
    }
    assert_eq!(
        a.optimize_tnlp(p.clone()),
        ApplicationReturnStatus::SolveSucceeded,
        "{algorithm} {scaling:?}"
    );
    if let Scaling::UserX = scaling {
        assert!(a.variable_scaling().is_some(), "x factors not applied");
    }
    Rc::try_unwrap(p).ok().unwrap().into_inner()
}

fn assert_close(what: &str, got: &[Number], want: &[Number]) {
    assert_eq!(
        got.len(),
        want.len(),
        "{what}: length ({got:?} vs {want:?})"
    );
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!(
            (g - w).abs() <= 1e-9 * (1.0 + w.abs()),
            "{what}[{i}]: live iterate {g:e}, finalize {w:e} ({got:?} vs {want:?})"
        );
    }
}

fn check(algorithm: &str, scaling: Scaling) {
    let m = solve(algorithm, scaling);
    assert!(m.fires > 1, "{algorithm} {scaling:?}: {} fires", m.fires);
    let last = m
        .last
        .unwrap_or_else(|| panic!("{algorithm} {scaling:?}: no live iterate at the last fire"));
    let out = m.out.expect("finalize_solution");
    // Informative only if the fixed variable's multiplier is not zero:
    // `z_L − z_U = 2K(1 − 4) + λ0`, negative, so z_U carries it.
    assert!(
        out.z_u[3] > 1e-3,
        "{algorithm} {scaling:?}: fixed x3 carries no multiplier: {out:?}"
    );
    assert!(out.z_u[1] > 1e-3, "x1's upper bound inactive: {out:?}");
    assert_close("x", &last.x, &out.x);
    assert_close("z_l", &last.z_l, &out.z_l);
    assert_close("z_u", &last.z_u, &out.z_u);
    assert_close("g", &last.g, &out.g);
    assert_close("lambda", &last.lambda, &out.lambda);
}

#[test]
fn ipm_live_iterate_equals_finalize_under_gradient_scaling_with_a_fixed_variable() {
    check("interior-point", Scaling::GradientBased);
}

#[test]
fn ipm_live_iterate_equals_finalize_under_user_x_scaling_with_a_fixed_variable() {
    check("interior-point", Scaling::UserX);
}

#[test]
fn sqp_live_iterate_equals_finalize_under_gradient_scaling_with_a_fixed_variable() {
    check("active-set-sqp", Scaling::GradientBased);
}

#[test]
fn sqp_live_iterate_equals_finalize_under_user_x_scaling_with_a_fixed_variable() {
    check("active-set-sqp", Scaling::UserX);
}

/// A TNLP that does not ask for the live iterate pays nothing for it, and
/// reads none, on the interior-point arm (the gate every Python solve,
/// whose bridge fires the callback unconditionally, relies on).
#[test]
fn ipm_installs_no_iterate_for_a_tnlp_that_does_not_want_one() {
    struct NoWant(Model);
    impl TNLP for NoWant {
        fn get_nlp_info(&mut self) -> Option<NlpInfo> {
            self.0.get_nlp_info()
        }
        fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
            self.0.get_bounds_info(b)
        }
        fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
            self.0.get_starting_point(sp)
        }
        fn eval_f(&mut self, x: &[Number], n: bool) -> Option<Number> {
            self.0.eval_f(x, n)
        }
        fn eval_grad_f(&mut self, x: &[Number], n: bool, g: &mut [Number]) -> bool {
            self.0.eval_grad_f(x, n, g)
        }
        fn eval_g(&mut self, x: &[Number], n: bool, g: &mut [Number]) -> bool {
            self.0.eval_g(x, n, g)
        }
        fn eval_jac_g(&mut self, x: Option<&[Number]>, n: bool, m: SparsityRequest<'_>) -> bool {
            self.0.eval_jac_g(x, n, m)
        }
        fn eval_h(
            &mut self,
            x: Option<&[Number]>,
            n: bool,
            of: Number,
            l: Option<&[Number]>,
            nl: bool,
            m: SparsityRequest<'_>,
        ) -> bool {
            self.0.eval_h(x, n, of, l, nl, m)
        }
        fn intermediate_callback(&mut self, st: IterStats, d: &IpoptData, q: &IpoptCq) -> bool {
            self.0.intermediate_callback(st, d, q)
        }
        fn finalize_solution(&mut self, s: Solution<'_>, d: &IpoptData, q: &IpoptCq) {
            self.0.finalize_solution(s, d, q)
        }
    }
    let p = Rc::new(RefCell::new(NoWant(Model {
        k: 1.0,
        ..Default::default()
    })));
    let mut a = IpoptApplication::new();
    a.initialize().unwrap();
    a.options_mut()
        .set_integer_value("print_level", 0, true, false)
        .unwrap();
    assert_eq!(
        a.optimize_tnlp(p.clone()),
        ApplicationReturnStatus::SolveSucceeded
    );
    let m = Rc::try_unwrap(p).ok().unwrap().into_inner().0;
    assert!(m.fires > 1);
    assert!(m.last.is_none(), "an iterate was installed: {:?}", m.last);
}
