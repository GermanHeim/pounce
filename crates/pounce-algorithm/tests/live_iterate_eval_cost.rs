//! What materialising the live iterate costs in user evaluations.
//!
//! A TNLP that asks for the live iterate (`wants_current_iterate`) gets one
//! built before every intermediate-callback fire. Everything it needs — `∇f`
//! and `J` for the fixed-variable bound multipliers and the Lagrangian
//! gradient, `g` for rows presolve dropped — the algorithm already evaluated
//! at that point, so the iterate must cost no evaluation the solve did not
//! already make. Measured against the same solve with a TNLP that does not
//! ask: identical iteration count and final `x` (nothing in the solve
//! moved), and identical `eval_grad_f` / `eval_g` / `eval_jac_g` counts.
//!
//! Model: the one in `live_iterate_matches_finalize.rs`, with `x3` fixed
//! (`x_L = x_U`, removed by `make_parameter`) or free in `[-10, 10]`, and
//! presolve (`g2` implied by the box and dropped, `g1` eliminating a column)
//! on or off.

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

#[derive(Default, Debug, Clone, Copy, PartialEq, Eq)]
struct Counts {
    grad_f: usize,
    g: usize,
    jac_vals: usize,
}

#[derive(Default)]
struct Counting {
    fixed: bool,
    want: bool,
    counts: Counts,
    fires: usize,
    reads: usize,
    iters: Index,
    x: Vec<Number>,
}

type Index = i32;

impl TNLP for Counting {
    fn get_nlp_info(&mut self) -> Option<NlpInfo> {
        Some(NlpInfo {
            n: 4,
            m: 3,
            nnz_jac_g: 7,
            nnz_h_lag: 4,
            index_style: IndexStyle::C,
        })
    }
    fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
        let (l3, u3) = if self.fixed {
            (1.0, 1.0)
        } else {
            (-10.0, 10.0)
        };
        b.x_l.copy_from_slice(&[-1e19, -10.0, 0.0, l3]);
        b.x_u.copy_from_slice(&[1e19, 1.0, 1e19, u3]);
        b.g_l.copy_from_slice(&[-1e19, 0.0, -1e19]);
        b.g_u.copy_from_slice(&[3.5, 0.0, 10.0]);
        true
    }
    fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
        sp.x.copy_from_slice(&[0.0, 0.0, 0.5, 1.0]);
        true
    }
    fn eval_f(&mut self, x: &[Number], _n: bool) -> Option<Number> {
        Some(
            (x[0] - 3.0).powi(2)
                + (x[1] - 3.0).powi(2)
                + (x[2] - 1.0).powi(2)
                + (x[3] - 4.0).powi(2),
        )
    }
    fn eval_grad_f(&mut self, x: &[Number], _n: bool, g: &mut [Number]) -> bool {
        self.counts.grad_f += 1;
        g[0] = 2.0 * (x[0] - 3.0);
        g[1] = 2.0 * (x[1] - 3.0);
        g[2] = 2.0 * (x[2] - 1.0);
        g[3] = 2.0 * (x[3] - 4.0);
        true
    }
    fn eval_g(&mut self, x: &[Number], _n: bool, g: &mut [Number]) -> bool {
        self.counts.g += 1;
        g[0] = x[0] + x[1] + x[3];
        g[1] = x[0] - x[2];
        g[2] = x[1] + x[3];
        true
    }
    fn eval_jac_g(&mut self, _x: Option<&[Number]>, _n: bool, mode: SparsityRequest<'_>) -> bool {
        match mode {
            SparsityRequest::Structure { irow, jcol } => {
                irow.copy_from_slice(&[0, 0, 0, 1, 1, 2, 2]);
                jcol.copy_from_slice(&[0, 1, 3, 0, 2, 1, 3]);
            }
            SparsityRequest::Values { values } => {
                self.counts.jac_vals += 1;
                values.copy_from_slice(&[1.0, 1.0, 1.0, 1.0, -1.0, 1.0, 1.0]);
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
            SparsityRequest::Values { values } => values.copy_from_slice(&[2.0 * of; 4]),
        }
        true
    }
    fn get_constraints_linearity(&mut self, t: &mut [Linearity]) -> bool {
        t.fill(Linearity::Linear);
        true
    }
    fn get_variables_linearity(&mut self, t: &mut [Linearity]) -> bool {
        t.fill(Linearity::NonLinear);
        true
    }
    fn wants_current_iterate(&self) -> bool {
        self.want
    }
    fn intermediate_callback(&mut self, st: IterStats, _d: &IpoptData, _q: &IpoptCq) -> bool {
        self.fires += 1;
        self.iters = st.iter;
        if with_current_user_iterate(|u| u.x.len()).is_some() {
            self.reads += 1;
        }
        true
    }
    fn finalize_solution(&mut self, s: Solution<'_>, _d: &IpoptData, _q: &IpoptCq) {
        self.x = s.x.to_vec();
    }
}

fn run(algorithm: &str, fixed: bool, presolve: bool, want: bool) -> Counting {
    let p = Rc::new(RefCell::new(Counting {
        fixed,
        want,
        ..Default::default()
    }));
    let mut a = IpoptApplication::new();
    a.initialize().unwrap();
    let o = a.options_mut();
    o.set_integer_value("print_level", 0, true, false).unwrap();
    o.set_string_value("algorithm", algorithm, true, false)
        .unwrap();
    if presolve {
        o.set_string_value("presolve", "yes", true, false).unwrap();
        o.set_string_value("presolve_linear_eq_reduction", "yes", true, false)
            .unwrap();
    }
    assert_eq!(
        a.optimize_tnlp(p.clone()),
        ApplicationReturnStatus::SolveSucceeded
    );
    Rc::try_unwrap(p).ok().unwrap().into_inner()
}

fn check(algorithm: &str, fixed: bool, presolve: bool) {
    let off = run(algorithm, fixed, presolve, false);
    let on = run(algorithm, fixed, presolve, true);
    eprintln!(
        "EVALCOST {algorithm} fixed={fixed} presolve={presolve}: fires {} iters {} | \
         no-iterate {:?} | iterate {:?}",
        on.fires, on.iters, off.counts, on.counts
    );
    assert!(
        on.fires > 1 && on.reads == on.fires,
        "{} of {} fires read an iterate",
        on.reads,
        on.fires
    );
    assert_eq!(
        on.iters, off.iters,
        "the live iterate moved the iteration count"
    );
    assert_eq!(on.x, off.x, "the live iterate moved the final x");
    assert_eq!(
        on.counts, off.counts,
        "the live iterate cost evaluations the solve did not make ({} fires)",
        on.fires
    );
}

#[test]
fn ipm_free() {
    check("interior-point", false, false);
}
#[test]
fn ipm_fixed() {
    check("interior-point", true, false);
}
#[test]
fn ipm_free_presolve() {
    check("interior-point", false, true);
}
#[test]
fn ipm_fixed_presolve() {
    check("interior-point", true, true);
}
#[test]
fn sqp_free() {
    check("active-set-sqp", false, false);
}
#[test]
fn sqp_fixed() {
    check("active-set-sqp", true, false);
}
#[test]
fn sqp_free_presolve() {
    check("active-set-sqp", false, true);
}
#[test]
fn sqp_fixed_presolve() {
    check("active-set-sqp", true, true);
}
