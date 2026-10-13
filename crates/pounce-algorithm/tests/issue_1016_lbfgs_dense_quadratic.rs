//! gh #1016 — L-BFGS grinds to `max_iter` on a dense, strictly convex quadratic.
//!
//! `min ½xᵀQx − bᵀx` with `Q = AAᵀ + 1e-2·I` dense, `n ∈ {10, 20, 40}`, no
//! Hessian supplied (so `hessian_approximation=limited-memory`). Exact-Hessian
//! solves it in one iteration; the limited-memory arm used to spend all 3000.
//!
//! **Mechanism.** Near the solution `x` is large in the weak directions and
//! `Q` is not small, so evaluating `f` cancels terms of order `1e3` and carries
//! round-off of about `5e-12` — seven orders above `10·eps·|f|`, the slack
//! `compare_le` gives the Armijo test. Once the true decrease per step falls
//! below that, no backtracked trial passes Armijo until `alpha` is so small
//! that `x + alpha·d == x`. That null step is "accepted", yields no curvature
//! pair (`s = 0`), the L-BFGS history is discarded after two skips, and the
//! next direction is again `-g` against a stiff `Q`: a cycle.
//!
//! The fix recognises the null step, and from then on also accepts a trial
//! whose `phi` is within a relative `1e-6` of the start and whose slope along
//! the step satisfies the Hager–Zhang approximate-Wolfe conditions. The slope
//! comes from the gradient, which has no such floor.
//!
//! The `Q` here comes from a fixed LCG rather than numpy's generator, so the
//! conditioning is of the same class as the report (`cond ~ 1e3..1e4`), not
//! identical to it.
use pounce_algorithm::application::IpoptApplication;
use pounce_common::types::Number;
use pounce_nlp::return_codes::ApplicationReturnStatus;
use pounce_nlp::tnlp::{
    BoundsInfo, IndexStyle, IpoptCq, IpoptData, IterStats, NlpInfo, Solution, SparsityRequest,
    StartingPoint, TNLP,
};
use std::cell::RefCell;
use std::rc::Rc;

struct Quad {
    n: usize,
    q: Vec<f64>,
    b: Vec<f64>,
    iters: Option<usize>,
    final_grad_norm: f64,
}

fn gauss(s: &mut u64) -> f64 {
    let mut u = || {
        *s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (((*s >> 11) as f64) + 0.5) / (1u64 << 53) as f64
    };
    let (a, b) = (u(), u());
    (-2.0 * a.ln()).sqrt() * (2.0 * std::f64::consts::PI * b).cos()
}

impl Quad {
    fn new(n: usize) -> Self {
        let mut s = 12345u64;
        let a: Vec<f64> = (0..n * n).map(|_| gauss(&mut s)).collect();
        let mut q = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                q[i * n + j] = (0..n).map(|k| a[i * n + k] * a[j * n + k]).sum::<f64>();
            }
            q[i * n + i] += 1e-2;
        }
        let b = (0..n).map(|_| gauss(&mut s)).collect();
        Quad {
            n,
            q,
            b,
            iters: None,
            final_grad_norm: f64::NAN,
        }
    }
}

impl TNLP for Quad {
    fn get_nlp_info(&mut self) -> Option<NlpInfo> {
        Some(NlpInfo {
            n: self.n as i32,
            m: 0,
            nnz_jac_g: 0,
            nnz_h_lag: 0,
            index_style: IndexStyle::C,
        })
    }
    fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
        b.x_l.fill(-2.0e19);
        b.x_u.fill(2.0e19);
        true
    }
    fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
        sp.x.fill(0.0);
        true
    }
    fn eval_f(&mut self, x: &[Number], _n: bool) -> Option<Number> {
        let n = self.n;
        let mut f = 0.0;
        for i in 0..n {
            let qx: f64 = (0..n).map(|j| self.q[i * n + j] * x[j]).sum();
            f += 0.5 * x[i] * qx - self.b[i] * x[i];
        }
        Some(f)
    }
    fn eval_grad_f(&mut self, x: &[Number], _n: bool, g: &mut [Number]) -> bool {
        let n = self.n;
        for i in 0..n {
            g[i] = (0..n).map(|j| self.q[i * n + j] * x[j]).sum::<f64>() - self.b[i];
        }
        true
    }
    fn eval_g(&mut self, _x: &[Number], _n: bool, _g: &mut [Number]) -> bool {
        true
    }
    fn eval_jac_g(&mut self, _x: Option<&[Number]>, _n: bool, _m: SparsityRequest<'_>) -> bool {
        true
    }
    fn eval_h(
        &mut self,
        _x: Option<&[Number]>,
        _n: bool,
        _o: Number,
        _l: Option<&[Number]>,
        _nl: bool,
        _m: SparsityRequest<'_>,
    ) -> bool {
        false
    }
    fn intermediate_callback(&mut self, st: IterStats, _d: &IpoptData, _q: &IpoptCq) -> bool {
        self.iters = Some(st.iter as usize);
        true
    }
    fn finalize_solution(&mut self, s: Solution<'_>, _d: &IpoptData, _q: &IpoptCq) {
        let n = self.n;
        let mut g = vec![0.0; n];
        self.eval_grad_f(s.x, true, &mut g);
        self.final_grad_norm = g.iter().map(|v| v * v).sum::<f64>().sqrt();
    }
}

fn run(n: usize) -> (ApplicationReturnStatus, usize, f64) {
    let mut app = IpoptApplication::new();
    app.initialize().unwrap();
    app.options_mut()
        .set_string_value("hessian_approximation", "limited-memory", true, true)
        .unwrap();
    let t = Rc::new(RefCell::new(Quad::new(n)));
    let tnlp: Rc<RefCell<dyn TNLP>> = Rc::clone(&t) as _;
    let st = app.optimize_tnlp(tnlp);
    let it = t.borrow().iters.unwrap_or(0);
    let g = t.borrow().final_grad_norm;
    (st, it, g)
}

#[test]
fn dense_quadratic_lbfgs_terminates_early() {
    for n in [10, 20, 40] {
        let (st, it, g) = run(n);
        assert!(
            matches!(st, ApplicationReturnStatus::SolveSucceeded),
            "n={n}: {st:?}"
        );
        assert!(it < 1000, "n={n}: {it} iterations");
        assert!(g < 1e-6, "n={n}: |grad| = {g:e}");
    }
}
