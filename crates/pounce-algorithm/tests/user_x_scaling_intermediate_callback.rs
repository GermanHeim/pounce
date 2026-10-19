//! Per-variable `user-scaling` and the intermediate callback.
//!
//! Non-unit `x` factors install a `ScalingTnlp` around the user's problem
//! (`x̃ = d ⊙ x`, gh#486). That wrapper did not forward
//! `intermediate_callback` (nor `wants_intermediate_callback`), so the
//! trait default answered for the user's problem and their callback never
//! fired, on either arm. Separately, the active-set SQP arm's user iterate
//! (`with_current_user_iterate`, what `GetIpoptCurrentIterate` reads there)
//! was lifted from the algorithm's iterate without undoing the
//! substitution, so it reported `x̃` and `z̃` where `finalize_solution`
//! reports `x` and `z`.
//!
//! Model as in `sqp_warm_start_user_space.rs`: `g0` an active inequality
//! ahead of the equality `g1`, `x0` free ahead of `x1`, whose upper bound
//! is active, so a factor on each variable moves something.

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

const D: [Number; 3] = [2.0, 0.5, 4.0];

#[derive(Clone, Debug, Default)]
struct Point {
    x: Vec<Number>,
    z_l: Vec<Number>,
    z_u: Vec<Number>,
    lambda: Vec<Number>,
}

/// `min (x0-3)² + (x1-3)² + (x2-1)²`
/// s.t. `g0 = x0 + x1 ≤ 2.5`, `g1 = x0 − x2 = 0`,
/// `x0` free, `x1 ∈ [-10, 1]`, `x2 ≥ 0`.
#[derive(Default)]
struct Model {
    fires: usize,
    /// The SQP arm's user iterate at the most recent fire.
    last: Option<Point>,
    out: Option<Point>,
}

impl TNLP for Model {
    fn get_nlp_info(&mut self) -> Option<NlpInfo> {
        Some(NlpInfo {
            n: 3,
            m: 2,
            nnz_jac_g: 4,
            nnz_h_lag: 3,
            index_style: IndexStyle::C,
        })
    }
    fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
        b.x_l.copy_from_slice(&[-1e19, -10.0, 0.0]);
        b.x_u.copy_from_slice(&[1e19, 1.0, 1e19]);
        b.g_l.copy_from_slice(&[-1e19, 0.0]);
        b.g_u.copy_from_slice(&[2.5, 0.0]);
        true
    }
    fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
        sp.x.copy_from_slice(&[0.0, 0.0, 0.5]);
        true
    }
    fn get_scaling_parameters(&mut self, r: ScalingRequest<'_>) -> bool {
        *r.obj_scaling = 3.0;
        *r.use_g_scaling = true;
        r.g_scaling.copy_from_slice(&[0.01, 5.0]);
        *r.use_x_scaling = true;
        r.x_scaling.copy_from_slice(&D);
        true
    }
    fn eval_f(&mut self, x: &[Number], _n: bool) -> Option<Number> {
        Some((x[0] - 3.0).powi(2) + (x[1] - 3.0).powi(2) + (x[2] - 1.0).powi(2))
    }
    fn eval_grad_f(&mut self, x: &[Number], _n: bool, g: &mut [Number]) -> bool {
        g[0] = 2.0 * (x[0] - 3.0);
        g[1] = 2.0 * (x[1] - 3.0);
        g[2] = 2.0 * (x[2] - 1.0);
        true
    }
    fn eval_g(&mut self, x: &[Number], _n: bool, g: &mut [Number]) -> bool {
        g[0] = x[0] + x[1];
        g[1] = x[0] - x[2];
        true
    }
    fn eval_jac_g(&mut self, _x: Option<&[Number]>, _n: bool, mode: SparsityRequest<'_>) -> bool {
        match mode {
            SparsityRequest::Structure { irow, jcol } => {
                irow.copy_from_slice(&[0, 0, 1, 1]);
                jcol.copy_from_slice(&[0, 1, 0, 2]);
            }
            SparsityRequest::Values { values } => {
                values.copy_from_slice(&[1.0, 1.0, 1.0, -1.0]);
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
                irow.copy_from_slice(&[0, 1, 2]);
                jcol.copy_from_slice(&[0, 1, 2]);
            }
            SparsityRequest::Values { values } => values.copy_from_slice(&[2.0 * of; 3]),
        }
        true
    }
    // The SQP arm, like the interior-point one, materialises the live
    // iterate only for a TNLP that reads it.
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

fn solve(algorithm: &str) -> Model {
    let p = Rc::new(RefCell::new(Model::default()));
    let mut a = IpoptApplication::new();
    a.initialize().unwrap();
    let o = a.options_mut();
    o.set_integer_value("print_level", 0, true, false).unwrap();
    o.set_string_value("nlp_scaling_method", "user-scaling", true, false)
        .unwrap();
    o.set_string_value("algorithm", algorithm, true, false)
        .unwrap();
    assert_eq!(
        a.optimize_tnlp(p.clone()),
        ApplicationReturnStatus::SolveSucceeded,
        "{algorithm}"
    );
    assert_eq!(
        a.variable_scaling().as_deref(),
        Some(&D[..]),
        "x factors applied"
    );
    Rc::try_unwrap(p).ok().unwrap().into_inner()
}

fn assert_close(what: &str, got: &[Number], want: &[Number]) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!(
            (g - w).abs() <= 1e-9 * (1.0 + w.abs()),
            "{what}[{i}]: callback iterate {g:e}, finalize {w:e} ({got:?} vs {want:?})"
        );
    }
}

#[test]
fn the_callback_fires_under_x_scaling_on_the_interior_point_arm() {
    let m = solve("interior-point");
    assert!(m.fires > 1, "intermediate_callback fired {} times", m.fires);
}

#[test]
fn the_callback_fires_under_x_scaling_on_the_sqp_arm() {
    let m = solve("active-set-sqp");
    assert!(m.fires > 1, "intermediate_callback fired {} times", m.fires);
}

/// The last fire is the converged iterate, so the SQP user iterate there
/// must be what `finalize_solution` reports — `x`, not `d ⊙ x`, and the
/// bound multipliers in the same units.
#[test]
fn the_sqp_user_iterate_is_unscaled_like_finalize() {
    let m = solve("active-set-sqp");
    let last = m.last.expect("a user iterate at the last fire");
    let out = m.out.expect("finalize_solution");
    // Informative only if x1's upper bound binds and the x factors are
    // distinguishable from 1 on the reported point.
    assert!(out.z_u[1] > 1e-3, "x1's upper bound inactive: {out:?}");
    assert!(out.x[0].abs() > 1e-3 && out.x[2].abs() > 1e-3, "{out:?}");
    assert_close("x", &last.x, &out.x);
    assert_close("z_l", &last.z_l, &out.z_l);
    assert_close("z_u", &last.z_u, &out.z_u);
    assert_close("lambda", &last.lambda, &out.lambda);
}
