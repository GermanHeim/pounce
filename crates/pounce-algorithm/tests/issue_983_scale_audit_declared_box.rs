//! The gh#983 scale audit may not ship a point outside the declared box --
//! checked through `IpoptApplication::optimize_tnlp`, the entry point the C
//! interface and the Python frontend drive, and on the point the user TNLP's
//! `finalize_solution` receives. The CLI test of the same name checks the
//! `.sol` / JSON report on the `.nl` transcription of this model
//! (`crates/pounce-cli/tests/fixtures/scale_audit_bound_active.nl`).
//!
//! `min 1e-3·((y−1)² − x₁ − x₂ − x₃)` s.t. `y³ + Σxᵢ ≥ 1.5`, `xᵢ ∈ [0, 1]`,
//! `y ∈ [−10, 10]`: optimum `xᵢ = 1`, `y = 1`, `f* = −3e-3`, which is also the
//! infimum of `f` over the declared box. The small objective sends the audit
//! into an up-scaled re-solve, whose larger bound multipliers shrink each
//! bound's slack below the `bound_relax_factor` widening; before the fix the
//! promoted re-solve returned `xᵢ = 1 + 4.97e-9` and `f < f*`.

use pounce_algorithm::application::IpoptApplication;
use pounce_common::types::Number;
use pounce_nlp::return_codes::ApplicationReturnStatus;
use pounce_nlp::solve_statistics::SolveStatistics;
use pounce_nlp::tnlp::{
    BoundsInfo, IndexStyle, IpoptCq, IpoptData, NlpInfo, Solution, SparsityRequest, StartingPoint,
    TNLP,
};
use std::cell::RefCell;
use std::rc::Rc;

const SCALE: Number = 1e-3;
const F_STAR: Number = -3e-3;

/// Variables `[y, x1, x2, x3]`, the order the `.nl` transcription has.
#[derive(Default)]
struct BoundActive {
    x: Vec<Number>,
    obj: Number,
}

impl TNLP for BoundActive {
    fn get_nlp_info(&mut self) -> Option<NlpInfo> {
        Some(NlpInfo {
            n: 4,
            m: 1,
            nnz_jac_g: 4,
            nnz_h_lag: 1,
            index_style: IndexStyle::C,
        })
    }
    fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
        b.x_l.copy_from_slice(&[-10.0, 0.0, 0.0, 0.0]);
        b.x_u.copy_from_slice(&[10.0, 1.0, 1.0, 1.0]);
        b.g_l[0] = 1.5;
        b.g_u[0] = 2e19;
        true
    }
    fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
        sp.x.copy_from_slice(&[0.0, 0.5, 0.5, 0.5]);
        true
    }
    fn eval_f(&mut self, x: &[Number], _n: bool) -> Option<Number> {
        Some(SCALE * ((x[0] - 1.0).powi(2) - x[1] - x[2] - x[3]))
    }
    fn eval_grad_f(&mut self, x: &[Number], _n: bool, g: &mut [Number]) -> bool {
        g.copy_from_slice(&[SCALE * 2.0 * (x[0] - 1.0), -SCALE, -SCALE, -SCALE]);
        true
    }
    fn eval_g(&mut self, x: &[Number], _n: bool, g: &mut [Number]) -> bool {
        g[0] = x[0].powi(3) + x[1] + x[2] + x[3];
        true
    }
    fn eval_jac_g(&mut self, x: Option<&[Number]>, _n: bool, mode: SparsityRequest<'_>) -> bool {
        match mode {
            SparsityRequest::Structure { irow, jcol } => {
                for k in 0..4 {
                    irow[k] = 0;
                    jcol[k] = k as i32;
                }
            }
            SparsityRequest::Values { values: v } => {
                let x = x.unwrap();
                v.copy_from_slice(&[3.0 * x[0] * x[0], 1.0, 1.0, 1.0]);
            }
        }
        true
    }
    fn eval_h(
        &mut self,
        x: Option<&[Number]>,
        _n: bool,
        obj_factor: Number,
        lambda: Option<&[Number]>,
        _nl: bool,
        mode: SparsityRequest<'_>,
    ) -> bool {
        match mode {
            SparsityRequest::Structure { irow, jcol } => {
                irow[0] = 0;
                jcol[0] = 0;
            }
            SparsityRequest::Values { values: v } => {
                let x = x.unwrap();
                let l = lambda.unwrap();
                v[0] = obj_factor * SCALE * 2.0 + l[0] * 6.0 * x[0];
            }
        }
        true
    }
    fn finalize_solution(&mut self, s: Solution<'_>, _d: &IpoptData, _q: &IpoptCq) {
        self.x = s.x.to_vec();
        self.obj = s.obj_value;
    }
}

fn solve(opts: &[(&str, Number)]) -> (ApplicationReturnStatus, SolveStatistics, BoundActive) {
    let mut app = IpoptApplication::new();
    app.options_mut()
        .set_integer_value("print_level", 0, true, false)
        .unwrap();
    for &(k, v) in opts {
        app.options_mut()
            .set_numeric_value(k, v, true, false)
            .unwrap();
    }
    app.initialize().unwrap();
    let tnlp = Rc::new(RefCell::new(BoundActive::default()));
    let status = app.optimize_tnlp(tnlp.clone());
    let st = app.statistics();
    let model = std::mem::take(&mut *tnlp.borrow_mut());
    (status, st, model)
}

fn assert_inside(st: &SolveStatistics, m: &BoundActive) {
    assert_eq!(m.x.len(), 4);
    assert!((-10.0..=10.0).contains(&m.x[0]), "y = {}", m.x[0]);
    for (i, &xi) in m.x[1..].iter().enumerate() {
        assert!(
            (0.0..=1.0).contains(&xi),
            "finalize_solution received x{} = {xi:.17e}, outside [0, 1]",
            i + 1
        );
    }
    assert!(
        m.obj >= F_STAR,
        "objective {:.12e} below f* {F_STAR:e}",
        m.obj
    );
    assert_eq!(st.final_declared_box_viol, 0.0);
    // The reported objective is the one at the point actually returned.
    assert_eq!(st.final_objective, m.obj);
}

#[test]
fn the_promoted_re_solve_hands_finalize_solution_a_point_inside_the_box() {
    let (status, st, m) = solve(&[]);
    assert_eq!(status, ApplicationReturnStatus::SolveSucceeded);
    // The audit re-solved and promoted: its point is the better one.
    assert!(
        m.obj - F_STAR < 1e-10,
        "expected the promoted re-solve's objective, got {:.12e}",
        m.obj
    );
    assert!(st.warnings.is_empty(), "{:?}", st.warnings);
    assert_inside(&st, &m);
}

#[test]
fn a_named_widening_keeps_the_base_attempt_and_its_box_measurement() {
    let (status, st, m) = solve(&[("bound_relax_factor", 1e-8)]);
    assert_eq!(status, ApplicationReturnStatus::SolveSucceeded);
    assert!(
        st.warnings
            .iter()
            .any(|w| w.contains("outside the declared variable bounds")),
        "{:?}",
        st.warnings
    );
    assert_inside(&st, &m);
}
