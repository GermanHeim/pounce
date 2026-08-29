//! The finite-difference Hessian must not be the updater the restoration
//! sub-IPM runs.
//!
//! `hessian_approximation=finite-difference` recovers `∇²ₓₓL` by probing
//! the analytic Jacobian, and its pattern is the *original* NLP's: the
//! objective clique names positions in the orig `x_var` space, and the
//! declared block it reads is the orig NLP's too. The restoration
//! sub-NLP's primal is the five-block `[orig | n_c | p_c | n_d | p_d]`
//! compound, which those indices do not describe — the same mismatch
//! that already scoped the partitioned Hessian out of restoration
//! (`resto_inner_solver.rs`), and for the first of the two reasons stated
//! there.
//!
//! What made it look safe is the *second* reason: unlike the partitioned
//! updater, the FD one never calls `eval_h`, so a model with no second
//! derivatives is genuinely fine. That is true and not sufficient.
//!
//! The fixture is a nonconvex equality pair with bounds that block the
//! Newton direction, so the solve enters feasibility restoration. Before
//! the fix this returned `RestorationFailed`; every other Hessian mode
//! solved it. Found while wiring the CasADi plugin's
//! `finite-difference` path (gh#823 review).

use pounce_algorithm::application::{
    IpoptApplication, Ma57Config, default_backend_factory, feral_config_from_options,
};
use pounce_common::types::Number;
use pounce_nlp::ApplicationReturnStatus;
use pounce_nlp::tnlp::{
    BoundsInfo, IndexStyle, IpoptCq, IpoptData, NlpInfo, Solution, SparsityRequest, StartingPoint,
    TNLP,
};
use pounce_restoration::resto_alg_builder::RestoAlgorithmBuilder;
use pounce_restoration::resto_inner_solver::{
    InnerBackendFactoryFactory, make_default_restoration_factory_provider,
};
use std::cell::RefCell;
use std::rc::Rc;

/// min x0 + x1 + x2  s.t.  ‖x‖² = 1,  sin(5·x0) + x1³ = 0.9,  x ∈ [-1, 1]³
///
/// Declares **no** Hessian (`nnz_h_lag = 0`), which is the model class
/// the mode exists for and which forces the Jacobian-derived pattern.
#[derive(Default)]
struct Kink;

impl TNLP for Kink {
    fn get_nlp_info(&mut self) -> Option<NlpInfo> {
        Some(NlpInfo {
            n: 3,
            m: 2,
            nnz_jac_g: 6,
            nnz_h_lag: 0,
            index_style: IndexStyle::C,
        })
    }

    fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
        b.x_l.copy_from_slice(&[-1.0; 3]);
        b.x_u.copy_from_slice(&[1.0; 3]);
        b.g_l.copy_from_slice(&[0.0, 0.0]);
        b.g_u.copy_from_slice(&[0.0, 0.0]);
        true
    }

    fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
        // Hard against the upper bound in every coordinate, so the first
        // Newton direction is blocked and the solve drops into
        // restoration rather than stepping away.
        sp.x.copy_from_slice(&[0.99, 0.99, 0.99]);
        true
    }

    fn eval_f(&mut self, x: &[Number], _new_x: bool) -> Option<Number> {
        Some(x[0] + x[1] + x[2])
    }

    fn eval_grad_f(&mut self, _x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
        g.copy_from_slice(&[1.0, 1.0, 1.0]);
        true
    }

    fn eval_g(&mut self, x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
        g[0] = x[0] * x[0] + x[1] * x[1] + x[2] * x[2] - 1.0;
        g[1] = (5.0 * x[0]).sin() + x[1] * x[1] * x[1] - 0.9;
        true
    }

    fn eval_jac_g(
        &mut self,
        x: Option<&[Number]>,
        _new_x: bool,
        mode: SparsityRequest<'_>,
    ) -> bool {
        match mode {
            SparsityRequest::Structure { irow, jcol } => {
                irow.copy_from_slice(&[0, 0, 0, 1, 1, 1]);
                jcol.copy_from_slice(&[0, 1, 2, 0, 1, 2]);
            }
            SparsityRequest::Values { values } => {
                let x = x.expect("eval_jac_g(Values) without x");
                values[0] = 2.0 * x[0];
                values[1] = 2.0 * x[1];
                values[2] = 2.0 * x[2];
                values[3] = 5.0 * (5.0 * x[0]).cos();
                values[4] = 3.0 * x[1] * x[1];
                values[5] = 0.0;
            }
        }
        true
    }

    /// No second derivatives, exactly as an FMU- or CasADi-`Callback`-backed
    /// model reports them.
    fn eval_h(
        &mut self,
        _x: Option<&[Number]>,
        _new_x: bool,
        _obj_factor: Number,
        _lambda: Option<&[Number]>,
        _new_lambda: bool,
        _mode: SparsityRequest<'_>,
    ) -> bool {
        false
    }

    fn finalize_solution(&mut self, _sol: Solution<'_>, _d: &IpoptData, _q: &IpoptCq) {}
}

fn solve_with(hessian_approximation: &str) -> (ApplicationReturnStatus, Vec<Number>, i32) {
    let mut app = IpoptApplication::new();
    app.options_mut()
        .set_string_value("hessian_approximation", hessian_approximation, true, true)
        .unwrap();
    app.options_mut()
        .set_integer_value("print_level", 0, true, true)
        .unwrap();
    app.options_mut()
        .set_integer_value("max_iter", 500, true, true)
        .unwrap();
    app.initialize().unwrap();

    let feral_cfg = feral_config_from_options(app.options());
    let bff_mint = move || -> InnerBackendFactoryFactory {
        let feral_cfg = feral_cfg.clone();
        Box::new(move || default_backend_factory(feral_cfg.clone(), Ma57Config::default()))
    };
    let resto_provider = make_default_restoration_factory_provider(
        RestoAlgorithmBuilder::new(),
        app.algorithm_builder_from_options(),
        bff_mint,
    );
    app.set_restoration_factory_provider(resto_provider);

    let solution = Rc::new(RefCell::new(Vec::<Number>::new()));

    struct Capture {
        inner: Kink,
        out: Rc<RefCell<Vec<Number>>>,
    }
    impl TNLP for Capture {
        fn get_nlp_info(&mut self) -> Option<NlpInfo> {
            self.inner.get_nlp_info()
        }
        fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
            self.inner.get_bounds_info(b)
        }
        fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
            self.inner.get_starting_point(sp)
        }
        fn eval_f(&mut self, x: &[Number], n: bool) -> Option<Number> {
            self.inner.eval_f(x, n)
        }
        fn eval_grad_f(&mut self, x: &[Number], n: bool, g: &mut [Number]) -> bool {
            self.inner.eval_grad_f(x, n, g)
        }
        fn eval_g(&mut self, x: &[Number], n: bool, g: &mut [Number]) -> bool {
            self.inner.eval_g(x, n, g)
        }
        fn eval_jac_g(&mut self, x: Option<&[Number]>, n: bool, m: SparsityRequest<'_>) -> bool {
            self.inner.eval_jac_g(x, n, m)
        }
        fn eval_h(
            &mut self,
            x: Option<&[Number]>,
            n: bool,
            o: Number,
            l: Option<&[Number]>,
            nl: bool,
            m: SparsityRequest<'_>,
        ) -> bool {
            self.inner.eval_h(x, n, o, l, nl, m)
        }
        fn finalize_solution(&mut self, sol: Solution<'_>, _d: &IpoptData, _q: &IpoptCq) {
            *self.out.borrow_mut() = sol.x.to_vec();
        }
    }

    let tnlp: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(Capture {
        inner: Kink,
        out: Rc::clone(&solution),
    }));
    let status = app.optimize_tnlp(tnlp);
    let x = solution.borrow().clone();
    let resto_calls = app.statistics().restoration_calls;
    (status, x, resto_calls)
}

/// The residual of the two equalities, so the assertion is feasibility
/// rather than a shared objective: the model is nonconvex and the
/// Hessian modes legitimately land on different local solutions.
fn violation(x: &[Number]) -> Number {
    let g0 = x[0] * x[0] + x[1] * x[1] + x[2] * x[2] - 1.0;
    let g1 = (5.0 * x[0]).sin() + x[1] * x[1] * x[1] - 0.9;
    g0.abs().max(g1.abs())
}

#[test]
fn finite_difference_survives_restoration() {
    let (status, x, resto_calls) = solve_with("finite-difference");
    // Guards the test against going vacuous: if the fixture ever stops
    // entering restoration, the FD updater is never built for the sub-IPM
    // and this would pass on the broken code too.
    assert!(
        resto_calls > 0,
        "the fixture no longer reaches restoration, so this test proves nothing"
    );
    assert_eq!(
        status,
        ApplicationReturnStatus::SolveSucceeded,
        "finite-difference reported {status:?}. Restoration runs the \
         limited-memory updater for this mode; without that scoping the \
         FD updater carries the original NLP's pattern into the five-block \
         compound primal and this returns RestorationFailed, on a model \
         limited-memory solves."
    );
    assert!(
        violation(&x) < 1e-8,
        "answer is infeasible: |g|inf = {:.3e}, x = {x:?}",
        violation(&x)
    );
    assert!(
        x.iter().all(|&v| (-1.0 - 1e-8..=1.0 + 1e-8).contains(&v)),
        "answer left its bounds: x = {x:?}"
    );
}

/// The control: the fixture is solvable, and solvable *through*
/// restoration, by the other updater that needs no second derivatives.
///
/// `exact` is deliberately not in this list. The model declares
/// `nnz_h_lag = 0` and refuses `eval_h`, which is the whole point of the
/// fixture — under `exact` there is no curvature to read and the run has
/// no meaning, rather than a meaningful failure.
#[test]
fn limited_memory_solves_this_fixture_through_restoration() {
    let (status, x, resto_calls) = solve_with("limited-memory");
    assert_eq!(
        status,
        ApplicationReturnStatus::SolveSucceeded,
        "limited-memory reported {status:?}"
    );
    assert!(
        violation(&x) < 1e-8,
        "limited-memory answer is infeasible: |g|inf = {:.3e}",
        violation(&x)
    );
    assert!(resto_calls > 0, "the fixture no longer reaches restoration");
}
