//! `find_minima` driven directly, through the real IPM and through scripted
//! `LocalSolve`s (gh#966).
//!
//! Each test names the branch its fixture reaches, because a green test is
//! only evidence about the branch it executes:
//!
//! | test | branch |
//! |---|---|
//! | `every_method_returns_only_genuine_camel_minima` | acceptance, all six strategy loops |
//! | `multistart_reaches_the_target` | `Stop::TargetReached` |
//! | `a_single_minimum_converges` | `Stop::Converged` via patience after accepts |
//! | `the_solve_budget_stops_the_search` | `Stop::BudgetExhausted` |
//! | `mlsl_sample_cap_stops_the_search` | `Stop::SampleCapReached` |
//! | `kkt_tol_vetoes_a_converged_non_stationary_point` | the gh#983 veto, and `kkt_tol: None` |
//! | `failed_solves_exhaust_patience` | a non-converged solve → reject → patience |
//! | `ipopt_local_solve_puts_the_callers_hook_back` | `IpoptLocalSolve`'s `Drop` |
//! | `honor_original_bounds_is_honoured` | the answer read from `finalize_solution`, projected |

use pounce_algorithm::application::IpoptApplication;
use pounce_common::types::Number;
use pounce_minima::{
    HessianMode, IpoptLocalSolve, LocalOutcome, LocalSolve, Method, MinimaOptions, SolveQuality,
    Stop, find_minima,
};
use pounce_nlp::tnlp::{
    BoundsInfo, IndexStyle, IpoptCq, IpoptData, NlpInfo, Solution, SparsityRequest, StartingPoint,
    TNLP,
};
use std::cell::{Cell, RefCell};
use std::rc::Rc;

/// An unconstrained two-variable model on a box, from plain functions.
struct Box2 {
    f: fn(&[Number]) -> Number,
    g: fn(&[Number]) -> [Number; 2],
    /// Lower triangle `[h00, h10, h11]`.
    h: fn(&[Number]) -> [Number; 3],
    lo: [Number; 2],
    hi: [Number; 2],
    x0: [Number; 2],
}

impl TNLP for Box2 {
    fn get_nlp_info(&mut self) -> Option<NlpInfo> {
        Some(NlpInfo {
            n: 2,
            m: 0,
            nnz_jac_g: 0,
            nnz_h_lag: 3,
            index_style: IndexStyle::C,
        })
    }
    fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
        b.x_l.copy_from_slice(&self.lo);
        b.x_u.copy_from_slice(&self.hi);
        true
    }
    fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
        sp.x.copy_from_slice(&self.x0);
        true
    }
    fn eval_f(&mut self, x: &[Number], _new_x: bool) -> Option<Number> {
        Some((self.f)(x))
    }
    fn eval_grad_f(&mut self, x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
        g.copy_from_slice(&(self.g)(x));
        true
    }
    fn eval_g(&mut self, _x: &[Number], _new_x: bool, _g: &mut [Number]) -> bool {
        true
    }
    fn eval_jac_g(&mut self, _x: Option<&[Number]>, _new_x: bool, _m: SparsityRequest<'_>) -> bool {
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
                irow.copy_from_slice(&[0, 1, 1]);
                jcol.copy_from_slice(&[0, 0, 1]);
            }
            SparsityRequest::Values { values } => {
                let h = (self.h)(x.unwrap());
                for (v, hk) in values.iter_mut().zip(h) {
                    *v = obj_factor * hk;
                }
            }
        }
        true
    }
    fn finalize_solution(&mut self, _sol: Solution<'_>, _d: &IpoptData, _q: &IpoptCq) {}
}

/// Six-hump camel on [-3,3]x[-2,2]: six minima in three symmetric pairs.
fn camel() -> Rc<RefCell<dyn TNLP>> {
    Rc::new(RefCell::new(Box2 {
        f: |z| {
            let (x, y) = (z[0], z[1]);
            4.0 * x * x - 2.1 * x.powi(4) + x.powi(6) / 3.0 + x * y - 4.0 * y * y + 4.0 * y.powi(4)
        },
        g: |z| {
            let (x, y) = (z[0], z[1]);
            [
                8.0 * x - 8.4 * x.powi(3) + 2.0 * x.powi(5) + y,
                x - 8.0 * y + 16.0 * y.powi(3),
            ]
        },
        h: |z| {
            let (x, y) = (z[0], z[1]);
            [
                8.0 - 25.2 * x * x + 10.0 * x.powi(4),
                1.0,
                -8.0 + 48.0 * y * y,
            ]
        },
        lo: [-3.0, -2.0],
        hi: [3.0, 2.0],
        x0: [0.5, 0.5],
    }))
}

const CAMEL_MINIMA: [Number; 3] = [-1.031_628_453_5, -0.215_463_824_4, 2.104_250_310_3];

/// (x-1)^2 + (y+0.5)^2 on [-2,2]^2: one minimum.
fn bowl() -> Rc<RefCell<dyn TNLP>> {
    Rc::new(RefCell::new(Box2 {
        f: |z| (z[0] - 1.0).powi(2) + (z[1] + 0.5).powi(2),
        g: |z| [2.0 * (z[0] - 1.0), 2.0 * (z[1] + 0.5)],
        h: |_| [2.0, 0.0, 2.0],
        lo: [-2.0, -2.0],
        hi: [2.0, 2.0],
        x0: [0.0, 0.0],
    }))
}

fn quiet_app() -> IpoptApplication {
    let mut app = IpoptApplication::new();
    app.options_mut()
        .read_from_str("print_level 0\nsb yes\n", true)
        .unwrap();
    app.initialize().unwrap();
    app
}

fn opts(method: Method, n_minima: usize, max_solves: usize) -> MinimaOptions {
    MinimaOptions {
        method,
        n_minima,
        max_solves: Some(max_solves),
        seed: 0,
        ..MinimaOptions::default()
    }
}

fn run(base: Rc<RefCell<dyn TNLP>>, cfg: &MinimaOptions) -> pounce_minima::MinimaResult {
    let mut app = quiet_app();
    let mut solver = IpoptLocalSolve::new(&mut app);
    find_minima(&base, &mut solver, cfg).expect("search starts")
}

#[test]
fn every_method_returns_only_genuine_camel_minima() {
    for method in [
        Method::Multistart,
        Method::Mlsl,
        Method::Basinhopping,
        Method::Flooding,
        Method::Deflation,
        Method::Tunneling,
    ] {
        let base = camel();
        let r = run(Rc::clone(&base), &opts(method, 6, 30));
        assert!(!r.minima.is_empty(), "{method:?}: nothing found");
        for mn in &r.minima {
            assert!(
                CAMEL_MINIMA.iter().any(|v| (mn.objective - v).abs() < 1e-6),
                "{method:?}: {} is not a camel minimum value",
                mn.objective
            );
            let mut g = [0.0; 2];
            base.borrow_mut().eval_grad_f(&mn.x, true, &mut g);
            assert!(g.iter().all(|v| v.abs() < 1e-5), "{method:?}: grad {g:?}");
        }
        for w in r.minima.windows(2) {
            assert!(w[0].objective <= w[1].objective, "{method:?}: not ranked");
        }
        assert_eq!(r.n_kkt_rejected, 0, "{method:?}");
    }
}

#[test]
fn multistart_reaches_the_target() {
    let r = run(camel(), &opts(Method::Multistart, 6, 30));
    assert_eq!(r.stop, Stop::TargetReached);
    assert_eq!(r.minima.len(), 6);
    assert_eq!(r.scale, vec![6.0, 4.0]);
}

#[test]
fn a_single_minimum_converges() {
    let r = run(bowl(), &opts(Method::Multistart, 3, 30));
    assert_eq!(r.stop, Stop::Converged);
    assert_eq!(r.minima.len(), 1);
    assert!((r.minima[0].x[0] - 1.0).abs() < 1e-6);
    assert!((r.minima[0].x[1] + 0.5).abs() < 1e-6);
    // One accept, then `patience` (default 8) empty solves.
    assert_eq!(r.n_solves, 1 + MinimaOptions::default().patience);
}

#[test]
fn the_solve_budget_stops_the_search() {
    let r = run(camel(), &opts(Method::Multistart, 6, 3));
    assert_eq!(r.stop, Stop::BudgetExhausted);
    assert_eq!(r.n_solves, 3);
}

#[test]
fn mlsl_sample_cap_stops_the_search() {
    // A huge critical radius rejects every sample, so no solve after the
    // first is spent and only the sample cap can stop the run.
    let cfg = MinimaOptions {
        gamma: Some(100.0),
        samples_per_round: Some(4),
        ..opts(Method::Mlsl, 6, 5)
    };
    let r = run(camel(), &cfg);
    assert_eq!(r.stop, Stop::SampleCapReached);
    assert_eq!(r.n_solves, 1);
}

/// A `LocalSolve` that returns each problem's own start as its answer, with
/// scripted convergence and residual.
struct Scripted {
    converged: bool,
    dual_inf: Number,
    calls: usize,
}

impl LocalSolve for Scripted {
    fn solve(&mut self, problem: Rc<RefCell<dyn TNLP>>, _mode: HessianMode) -> LocalOutcome {
        self.calls += 1;
        let mut x = vec![0.0; 2];
        let (mut zl, mut zu, mut lam) = (vec![0.0; 2], vec![0.0; 2], vec![]);
        problem.borrow_mut().get_starting_point(StartingPoint {
            init_x: true,
            x: &mut x,
            init_z: false,
            z_l: &mut zl,
            z_u: &mut zu,
            init_lambda: false,
            lambda: &mut lam,
        });
        LocalOutcome {
            converged: self.converged,
            x: if self.converged { x } else { Vec::new() },
            lambda: Vec::new(),
            quality: SolveQuality {
                unscaled_dual_inf: self.dual_inf,
                unscaled_dual_scale: 1.0,
                start_obj_grad_max: Number::NAN,
            },
        }
    }
}

#[test]
fn kkt_tol_vetoes_a_converged_non_stationary_point() {
    // Every "solution" is the start, with residual 0.5 against a bound of
    // 1e-4 * max(1, |grad f(x0)| = 2, 1) = 2e-4.
    let cfg = opts(Method::Multistart, 3, 6);
    let mut solver = Scripted {
        converged: true,
        dual_inf: 0.5,
        calls: 0,
    };
    let r = find_minima(&bowl(), &mut solver, &cfg).unwrap();
    assert!(r.minima.is_empty());
    assert_eq!(r.n_kkt_rejected, r.n_solves);
    assert_eq!(r.n_solves, 6);

    // The same answers with the veto off are archived (the bowl is convex,
    // so the saddle check passes them).
    let cfg = MinimaOptions {
        kkt_tol: None,
        ..cfg
    };
    let mut solver = Scripted {
        converged: true,
        dual_inf: 0.5,
        calls: 0,
    };
    let r = find_minima(&bowl(), &mut solver, &cfg).unwrap();
    assert_eq!(r.n_kkt_rejected, 0);
    assert!(!r.minima.is_empty());
}

#[test]
fn failed_solves_exhaust_patience() {
    let cfg = opts(Method::Multistart, 3, 100);
    let mut solver = Scripted {
        converged: false,
        dual_inf: Number::NAN,
        calls: 0,
    };
    let r = find_minima(&bowl(), &mut solver, &cfg).unwrap();
    assert_eq!(r.stop, Stop::Converged);
    assert!(r.minima.is_empty());
    assert_eq!(r.n_solves, cfg.patience);
    // No dual-recovery solves: nothing was accepted.
    assert_eq!(solver.calls, cfg.patience);
}

#[test]
fn ipopt_local_solve_puts_the_callers_hook_back() {
    let fired = Rc::new(Cell::new(0usize));
    let mut app = quiet_app();
    let f = Rc::clone(&fired);
    app.set_on_converged(Box::new(move |_, _, _, _| f.set(f.get() + 1)));
    {
        let mut solver = IpoptLocalSolve::new(&mut app);
        let _ = find_minima(&bowl(), &mut solver, &opts(Method::Multistart, 1, 2)).unwrap();
    }
    // The caller's hook did not see the search's solves...
    assert_eq!(fired.get(), 0);
    // ...and is back in place for the caller's own.
    let _ = app.optimize_tnlp(bowl());
    assert_eq!(fired.get(), 1);
}

/// (x+1)^2 + (y-0.3)^2 on [0,2]x[-1,1]: the minimum binds x >= 0.
fn bound_active() -> Rc<RefCell<dyn TNLP>> {
    Rc::new(RefCell::new(Box2 {
        f: |z| (z[0] + 1.0).powi(2) + (z[1] - 0.3).powi(2),
        g: |z| [2.0 * (z[0] + 1.0), 2.0 * (z[1] - 0.3)],
        h: |_| [2.0, 0.0, 2.0],
        lo: [0.0, -1.0],
        hi: [2.0, 1.0],
        x0: [1.0, 0.0],
    }))
}

fn bound_active_x(extra_options: &str) -> Number {
    let mut app = IpoptApplication::new();
    app.options_mut()
        .read_from_str(&format!("print_level 0\nsb yes\n{extra_options}"), true)
        .unwrap();
    app.initialize().unwrap();
    let mut solver = IpoptLocalSolve::new(&mut app);
    let r = find_minima(
        &bound_active(),
        &mut solver,
        &opts(Method::Multistart, 1, 4),
    )
    .unwrap();
    assert_eq!(r.minima.len(), 1);
    r.minima[0].x[0]
}

/// The answer is the `finalize_solution` payload, so `honor_original_bounds`
/// is honoured. Reading the algorithm's iterate (the on-converged hook this
/// replaced) returned the relaxed point either way.
#[test]
fn honor_original_bounds_is_honoured() {
    // The premise: without the option the reported point sits past the bound
    // by the bound relaxation, so this fixture reaches the projection branch.
    let relaxed = bound_active_x("");
    assert!(relaxed < 0.0 && relaxed > -1e-6, "relaxed x = {relaxed:e}");
    let projected = bound_active_x("honor_original_bounds yes\n");
    assert_eq!(projected, 0.0);
}
