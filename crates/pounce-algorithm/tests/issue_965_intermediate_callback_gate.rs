//! gh#965 — the intermediate-callback fire built an `IterStats` every
//! iteration even when the TNLP's callback was the no-op default. On a
//! 53-iteration AC-OPF solve that was 4.6% of the run, nearly all of it
//! `inf_du` assembling ∇L for a payload nobody read.
//!
//! `TNLP::wants_intermediate_callback` now gates the fire. These tests pin
//! both halves of its contract: a TNLP that opts out is never called and its
//! solve is unchanged, and one that keeps the default is still called on
//! every iteration — directly and through a forwarding decorator, since a
//! decorator that drops the method would silently opt every caller in or out.
use pounce_algorithm::application::IpoptApplication;
use pounce_common::types::Number;
use pounce_nlp::return_codes::ApplicationReturnStatus;
use pounce_nlp::seeded_tnlp::SeededTnlp;
use pounce_nlp::tnlp::{
    BoundsInfo, IndexStyle, IpoptCq, IpoptData, IterStats, NlpInfo, Solution, SparsityRequest,
    StartingPoint, TNLP,
};
use std::cell::RefCell;
use std::rc::Rc;

/// `min (x0 - 1)² + (x1 - 2)²  s.t.  x0 + x1 <= 2`, optimum `(0.5, 1.5)`.
/// The row is active, so `inf_du` carries a real `Jᵀy` term.
struct Probe {
    wants: bool,
    calls: usize,
    last_iter: Option<i32>,
}

impl TNLP for Probe {
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
        b.x_l.fill(-2.0e19);
        b.x_u.fill(2.0e19);
        b.g_l[0] = -2.0e19;
        b.g_u[0] = 2.0;
        true
    }
    fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
        sp.x.fill(0.0);
        true
    }
    fn eval_f(&mut self, x: &[Number], _n: bool) -> Option<Number> {
        Some((x[0] - 1.0).powi(2) + (x[1] - 2.0).powi(2))
    }
    fn eval_grad_f(&mut self, x: &[Number], _n: bool, g: &mut [Number]) -> bool {
        g[0] = 2.0 * (x[0] - 1.0);
        g[1] = 2.0 * (x[1] - 2.0);
        true
    }
    fn eval_g(&mut self, x: &[Number], _n: bool, g: &mut [Number]) -> bool {
        g[0] = x[0] + x[1];
        true
    }
    fn eval_jac_g(&mut self, _x: Option<&[Number]>, _n: bool, m: SparsityRequest<'_>) -> bool {
        match m {
            SparsityRequest::Structure { irow, jcol } => {
                irow.copy_from_slice(&[0, 0]);
                jcol.copy_from_slice(&[0, 1]);
            }
            SparsityRequest::Values { values } => values.copy_from_slice(&[1.0, 1.0]),
        }
        true
    }
    fn eval_h(
        &mut self,
        _x: Option<&[Number]>,
        _n: bool,
        obj: Number,
        _l: Option<&[Number]>,
        _nl: bool,
        m: SparsityRequest<'_>,
    ) -> bool {
        match m {
            SparsityRequest::Structure { irow, jcol } => {
                irow.copy_from_slice(&[0, 1]);
                jcol.copy_from_slice(&[0, 1]);
            }
            SparsityRequest::Values { values } => values.copy_from_slice(&[2.0 * obj, 2.0 * obj]),
        }
        true
    }
    fn wants_intermediate_callback(&self) -> bool {
        self.wants
    }
    fn intermediate_callback(&mut self, st: IterStats, _d: &IpoptData, _q: &IpoptCq) -> bool {
        self.calls += 1;
        self.last_iter = Some(st.iter);
        true
    }
    fn finalize_solution(&mut self, _s: Solution<'_>, _d: &IpoptData, _q: &IpoptCq) {}
}

struct Run {
    status: ApplicationReturnStatus,
    iters: i32,
    calls: usize,
    last_iter: Option<i32>,
}

fn run(wants: bool, decorated: bool) -> Run {
    let mut app = IpoptApplication::new();
    app.initialize().unwrap();
    let probe = Rc::new(RefCell::new(Probe {
        wants,
        calls: 0,
        last_iter: None,
    }));
    let inner: Rc<RefCell<dyn TNLP>> = Rc::clone(&probe) as _;
    let tnlp: Rc<RefCell<dyn TNLP>> = if decorated {
        Rc::new(RefCell::new(SeededTnlp::new(inner, vec![0.0, 0.0])))
    } else {
        inner
    };
    let status = app.optimize_tnlp(tnlp);
    let p = probe.borrow();
    Run {
        status,
        iters: app.statistics().iteration_count,
        calls: p.calls,
        last_iter: p.last_iter,
    }
}

#[test]
fn a_tnlp_that_wants_the_callback_is_called_every_iteration() {
    for decorated in [false, true] {
        let r = run(true, decorated);
        assert!(
            matches!(r.status, ApplicationReturnStatus::SolveSucceeded),
            "decorated={decorated}: {:?}",
            r.status
        );
        // Iteration 0 plus one per iteration.
        assert_eq!(r.calls as i32, r.iters + 1, "decorated={decorated}");
        assert_eq!(r.last_iter, Some(r.iters), "decorated={decorated}");
    }
}

#[test]
fn a_tnlp_that_opts_out_is_never_called_and_solves_identically() {
    for decorated in [false, true] {
        let on = run(true, decorated);
        let off = run(false, decorated);
        assert_eq!(off.calls, 0, "decorated={decorated}: the gate did not hold");
        assert_eq!(off.status, on.status, "decorated={decorated}");
        assert_eq!(off.iters, on.iters, "decorated={decorated}");
    }
}
