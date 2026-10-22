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
//!
//! Also here, because they need an `IpoptApplication` the test can reuse:
//! the re-solve's temporary configuration (the box flag, `max_iter`,
//! `obj_scaling_factor`, the CPU / wall budget) is put back on every exit --
//! after a promotion, after a decline, and after a panic inside the re-solve,
//! which `pounce-cinterface`'s `ffi_guard` catches while keeping the
//! application for the next `IpoptSolve`.

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

/// Variables `[y, x1, x2, x3]`, the order the `.nl` transcription has, with
/// the controls in `[0, ub]` and the objective multiplied by `scale`.
struct BoundActive {
    scale: Number,
    ub: Number,
    /// Panic on the first objective evaluation after a `finalize_solution`,
    /// i.e. inside the scale audit's re-solve.
    panic_in_retry: bool,
    finalized: usize,
    x: Vec<Number>,
    obj: Number,
}

impl Default for BoundActive {
    fn default() -> Self {
        Self {
            scale: SCALE,
            ub: 1.0,
            panic_in_retry: false,
            finalized: 0,
            x: Vec::new(),
            obj: Number::NAN,
        }
    }
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
        b.x_u.copy_from_slice(&[10.0, self.ub, self.ub, self.ub]);
        b.g_l[0] = 1.5;
        b.g_u[0] = 2e19;
        true
    }
    fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
        let h = 0.5 * self.ub;
        sp.x.copy_from_slice(&[0.0, h, h, h]);
        true
    }
    fn eval_f(&mut self, x: &[Number], _n: bool) -> Option<Number> {
        if self.panic_in_retry && self.finalized >= 1 {
            panic!("injected: a panic inside the scale audit's re-solve");
        }
        Some(self.scale * ((x[0] - 1.0).powi(2) - x[1] - x[2] - x[3]))
    }
    fn eval_grad_f(&mut self, x: &[Number], _n: bool, g: &mut [Number]) -> bool {
        let s = self.scale;
        g.copy_from_slice(&[s * 2.0 * (x[0] - 1.0), -s, -s, -s]);
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
                v[0] = obj_factor * self.scale * 2.0 + l[0] * 6.0 * x[0];
            }
        }
        true
    }
    fn finalize_solution(&mut self, s: Solution<'_>, _d: &IpoptData, _q: &IpoptCq) {
        self.finalized += 1;
        self.x = s.x.to_vec();
        self.obj = s.obj_value;
    }
}

fn new_app(opts: &[(&str, Number)]) -> IpoptApplication {
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
    app
}

fn solve_on(
    app: &mut IpoptApplication,
    model: BoundActive,
) -> (ApplicationReturnStatus, SolveStatistics, BoundActive) {
    let tnlp = Rc::new(RefCell::new(model));
    let status = app.optimize_tnlp(tnlp.clone());
    let st = app.statistics();
    let model = std::mem::take(&mut *tnlp.borrow_mut());
    (status, st, model)
}

fn solve(opts: &[(&str, Number)]) -> (ApplicationReturnStatus, SolveStatistics, BoundActive) {
    solve_on(&mut new_app(opts), BoundActive::default())
}

fn assert_inside(st: &SolveStatistics, m: &BoundActive) {
    assert_eq!(m.x.len(), 4);
    assert!((-10.0..=10.0).contains(&m.x[0]), "y = {}", m.x[0]);
    for (i, &xi) in m.x[1..].iter().enumerate() {
        assert!(
            (0.0..=m.ub).contains(&xi),
            "finalize_solution received x{} = {xi:.17e}, outside [0, {}]",
            i + 1,
            m.ub
        );
    }
    let f_star = m.scale * (-3.0 * m.ub);
    assert!(
        m.obj >= f_star,
        "objective {:.12e} below f* {f_star:e}",
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

/// Bounds of magnitude `1e4` and `1e6`, where one ulp of the bound
/// (`1.8e-12`, `1.2e-10`) exceeds any absolute floor that still tells `1e-8`
/// from noise on a unit bound. The default arm promotes inside the box; the
/// named-factor arm declines on a genuine excess (`2.5e-10` and `2.5e-12`
/// relative to the bound), which the relative comparison still sees.
#[test]
fn large_bounds_promote_inside_the_box_and_decline_only_on_genuine_excess() {
    for ub in [1e4, 1e6] {
        let model = || BoundActive {
            ub,
            ..BoundActive::default()
        };
        let (status, st, m) = solve_on(&mut new_app(&[]), model());
        assert_eq!(status, ApplicationReturnStatus::SolveSucceeded, "ub {ub}");
        assert!(st.warnings.is_empty(), "ub {ub}: {:?}", st.warnings);
        assert_inside(&st, &m);
        let (_, st, m) = solve_on(&mut new_app(&[("bound_relax_factor", 1e-8)]), model());
        assert!(
            st.warnings
                .iter()
                .any(|w| w.contains("outside the declared variable bounds")),
            "ub {ub}: {:?}",
            st.warnings
        );
        // Declined: the base attempt's point, whatever its own widening use.
        assert_eq!(st.final_objective, m.obj, "ub {ub}");
    }
}

/// What the next solve on the same application must see, whatever the
/// previous one did: the caller's options exactly, and the variable box
/// widened again. The probe is an un-audited, bound-active solve at unit
/// objective scale, which lands on its widened bound (`~7.5e-9` outside its
/// box) only if the widening is back.
fn assert_next_solve_is_clean(app: &mut IpoptApplication, after: &str) {
    let o = app.options();
    assert_eq!(
        o.get_integer_value("max_iter", "").unwrap(),
        (500, true),
        "{after}: max_iter"
    );
    assert_eq!(
        o.get_numeric_value("max_cpu_time", "").unwrap(),
        (1000.0, true),
        "{after}: max_cpu_time"
    );
    assert!(
        !o.get_numeric_value("obj_scaling_factor", "").unwrap().1,
        "{after}: obj_scaling_factor left caller-set"
    );
    assert!(
        !o.get_numeric_value("max_wall_time", "").unwrap().1,
        "{after}: max_wall_time left caller-set"
    );
    let probe = BoundActive {
        scale: 1.0,
        ..BoundActive::default()
    };
    let (status, st, _) = solve_on(app, probe);
    assert_eq!(status, ApplicationReturnStatus::SolveSucceeded, "{after}");
    assert!(
        st.final_declared_box_viol > 1e-9,
        "{after}: the next solve ran with the variable box un-widened \
         (declared-box violation {:e})",
        st.final_declared_box_viol
    );
}

#[test]
fn the_next_solve_after_a_promoted_or_declined_re_solve_is_unaffected() {
    for (after, relax) in [("promotion", None), ("decline", Some(1e-8))] {
        let mut opts = vec![("max_cpu_time", 1000.0)];
        if let Some(r) = relax {
            opts.push(("bound_relax_factor", r));
        }
        let mut app = new_app(&opts);
        app.options_mut()
            .set_integer_value("max_iter", 500, true, false)
            .unwrap();
        let (status, st, _) = solve_on(&mut app, BoundActive::default());
        assert_eq!(status, ApplicationReturnStatus::SolveSucceeded, "{after}");
        assert_eq!(
            st.warnings.is_empty(),
            relax.is_none(),
            "{after}: {:?}",
            st.warnings
        );
        assert_next_solve_is_clean(&mut app, after);
    }
}

/// The same after a panic inside the re-solve, caught the way
/// `pounce-cinterface`'s `ffi_guard` catches it.
#[test]
fn a_panic_inside_the_re_solve_does_not_leak_its_configuration() {
    let mut app = new_app(&[("max_cpu_time", 1000.0)]);
    app.options_mut()
        .set_integer_value("max_iter", 500, true, false)
        .unwrap();
    let model = BoundActive {
        panic_in_retry: true,
        ..BoundActive::default()
    };
    let tnlp = Rc::new(RefCell::new(model));
    let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        app.optimize_tnlp(tnlp.clone())
    }));
    assert!(caught.is_err(), "the injected panic did not fire");
    assert_eq!(
        tnlp.borrow().finalized,
        1,
        "the panic must land inside the re-solve, after the base attempt"
    );
    assert_next_solve_is_clean(&mut app, "panic");
}
