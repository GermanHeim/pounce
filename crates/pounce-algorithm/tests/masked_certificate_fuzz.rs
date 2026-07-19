//! Adversarial fuzz over the gh #200 masked-certificate veto.
//!
//! The hand-written tests in `masked_certificate_veto.rs` check the cases the
//! fix was designed around, which is exactly what makes them weak evidence: a
//! fix can pass every test written by the person who wrote the fix and still be
//! wrong off that path. This file instead generates problems designed to *break*
//! the mechanism and asserts invariants that must hold for all of them.
//!
//! The generator sweeps the parameters that drive the pathology rather than
//! random noise: the exponent (how fast the gradient vanishes near the
//! minimum), the offset magnitude (how large the initial gradient is, hence how
//! extreme `obj_scale` becomes), conditioning spread, dimension, start point,
//! and convexity. Non-convex instances are included deliberately — the veto
//! makes a run travel further, and travelling further is how a solver finds a
//! *different* stationary point.
//!
//! The invariants are stated against the opt-out (`obj_scale_certificate_threshold
//! = 0`), which is the pre-fix behaviour, so every one of them is a statement of
//! the form "the fix did not make this worse".

use pounce_algorithm::application::IpoptApplication;
use pounce_common::types::Number;
use pounce_nlp::return_codes::ApplicationReturnStatus;
use pounce_nlp::tnlp::{
    BoundsInfo, IndexStyle, IpoptCq, IpoptData, NlpInfo, Solution, SparsityRequest, StartingPoint,
    TNLP,
};
use std::cell::RefCell;
use std::rc::Rc;

/// Deterministic xorshift64*, so any failure is reproducible from its seed.
struct Rng(u64);
impl Rng {
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn unit(&mut self) -> Number {
        (self.next_u64() >> 11) as Number / (1u64 << 53) as Number
    }
    fn pick<T: Copy>(&mut self, xs: &[T]) -> T {
        xs[(self.next_u64() % xs.len() as u64) as usize]
    }
}

/// `f(x) = Σᵢ cᵢ·(xᵢ − aᵢ)^p − wᵢ·xᵢ²`
///
/// `p` even and ≥ 4 gives a gradient that vanishes super-linearly at the
/// minimum, which combined with a large `a` is the scaling pathology. `w > 0`
/// adds a concave term, making the problem non-convex with several stationary
/// points — the case where "keep iterating" could plausibly land somewhere
/// worse rather than better.
#[derive(Clone)]
struct Spec {
    n: usize,
    p: i32,
    a: Vec<Number>,
    c: Vec<Number>,
    w: Vec<Number>,
    x0: Number,
    /// Dense rows of a linear constraint block `A x {=,<=} b`. Linear keeps the
    /// Lagrangian Hessian free of multiplier terms while still exercising the
    /// machinery the unconstrained fuzz cannot reach at all: constraint
    /// violation, equality/inequality multipliers, barrier complementarity,
    /// the filter line search, and restoration.
    arows: Vec<Vec<Number>>,
    brhs: Vec<Number>,
    /// `true` → equalities (`= b`), `false` → inequalities (`<= b`).
    eq: bool,
}

struct Problem(Spec, Rc<RefCell<Vec<Number>>>);

impl TNLP for Problem {
    fn get_nlp_info(&mut self) -> Option<NlpInfo> {
        Some(NlpInfo {
            n: self.0.n as i32,
            m: self.0.arows.len() as i32,
            nnz_jac_g: (self.0.arows.len() * self.0.n) as i32,
            nnz_h_lag: self.0.n as i32,
            index_style: IndexStyle::C,
        })
    }
    fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
        for v in b.x_l.iter_mut() {
            *v = -2.0e19;
        }
        for v in b.x_u.iter_mut() {
            *v = 2.0e19;
        }
        let s = &self.0;
        for (k, rhs) in s.brhs.iter().enumerate() {
            b.g_u[k] = *rhs;
            b.g_l[k] = if s.eq { *rhs } else { -2.0e19 };
        }
        true
    }
    fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
        for v in sp.x.iter_mut() {
            *v = self.0.x0;
        }
        true
    }
    fn eval_f(&mut self, x: &[Number], _n: bool) -> Option<Number> {
        let s = &self.0;
        Some(
            (0..s.n)
                .map(|i| s.c[i] * (x[i] - s.a[i]).powi(s.p) - s.w[i] * x[i] * x[i])
                .sum(),
        )
    }
    fn eval_grad_f(&mut self, x: &[Number], _n: bool, g: &mut [Number]) -> bool {
        let s = &self.0;
        for i in 0..s.n {
            g[i] = s.c[i] * s.p as Number * (x[i] - s.a[i]).powi(s.p - 1) - 2.0 * s.w[i] * x[i];
        }
        true
    }
    fn eval_g(&mut self, x: &[Number], _n: bool, g: &mut [Number]) -> bool {
        for (k, row) in self.0.arows.iter().enumerate() {
            g[k] = row.iter().zip(x).map(|(a, xi)| a * xi).sum();
        }
        true
    }
    fn eval_jac_g(&mut self, _x: Option<&[Number]>, _n: bool, mode: SparsityRequest<'_>) -> bool {
        let s = &self.0;
        match mode {
            SparsityRequest::Structure { irow, jcol } => {
                let mut t = 0;
                for k in 0..s.arows.len() {
                    for j in 0..s.n {
                        irow[t] = k as i32;
                        jcol[t] = j as i32;
                        t += 1;
                    }
                }
            }
            SparsityRequest::Values { values } => {
                let mut t = 0;
                for row in &s.arows {
                    for a in row {
                        values[t] = *a;
                        t += 1;
                    }
                }
            }
        }
        true
    }
    fn eval_h(
        &mut self,
        x: Option<&[Number]>,
        _n: bool,
        obj_factor: Number,
        _l: Option<&[Number]>,
        _nl: bool,
        mode: SparsityRequest<'_>,
    ) -> bool {
        let s = &self.0;
        match mode {
            SparsityRequest::Structure { irow, jcol } => {
                for i in 0..s.n {
                    irow[i] = i as i32;
                    jcol[i] = i as i32;
                }
            }
            SparsityRequest::Values { values } => {
                let x = x.expect("eval_h(Values) without x");
                for i in 0..s.n {
                    let pp = s.p as Number;
                    values[i] = obj_factor
                        * (s.c[i] * pp * (pp - 1.0) * (x[i] - s.a[i]).powi(s.p - 2) - 2.0 * s.w[i]);
                }
            }
        }
        true
    }
    fn finalize_solution(&mut self, s: Solution<'_>, _d: &IpoptData, _q: &IpoptCq) {
        *self.1.borrow_mut() = s.x.to_vec();
    }
}

struct Outcome {
    status: ApplicationReturnStatus,
    obj: Number,
    iters: usize,
}

fn run(spec: &Spec, threshold: Option<Number>, max_cpu: Option<Number>) -> Outcome {
    let mut app = IpoptApplication::new();
    if let Some(t) = threshold {
        app.options_mut()
            .set_numeric_value("obj_scale_certificate_threshold", t, true, false)
            .unwrap();
    }
    if let Some(t) = max_cpu {
        app.options_mut()
            .set_numeric_value("max_cpu_time", t, true, false)
            .unwrap();
    }
    // Keep the fuzz fast; the pathology shows up well inside this budget.
    app.options_mut()
        .set_integer_value("max_iter", 300, true, false)
        .unwrap();
    app.options_mut()
        .set_integer_value("print_level", 0, true, false)
        .unwrap();
    app.initialize().unwrap();
    let tnlp: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(Problem(
        spec.clone(),
        Rc::new(RefCell::new(Vec::new())),
    )));
    let status = app.optimize_tnlp(tnlp);
    let s = app.statistics();
    Outcome {
        status,
        obj: s.final_objective,
        iters: s.iteration_count as usize,
    }
}

fn run_capped(spec: &Spec, threshold: Option<Number>, max_iter: i32) -> Outcome {
    let mut app = IpoptApplication::new();
    if let Some(t) = threshold {
        app.options_mut()
            .set_numeric_value("obj_scale_certificate_threshold", t, true, false)
            .unwrap();
    }
    app.options_mut()
        .set_integer_value("max_iter", max_iter, true, false)
        .unwrap();
    app.options_mut()
        .set_integer_value("print_level", 0, true, false)
        .unwrap();
    app.initialize().unwrap();
    let tnlp: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(Problem(
        spec.clone(),
        Rc::new(RefCell::new(Vec::new())),
    )));
    let status = app.optimize_tnlp(tnlp);
    let s = app.statistics();
    Outcome {
        status,
        obj: s.final_objective,
        iters: s.iteration_count as usize,
    }
}

fn succeeded(s: ApplicationReturnStatus) -> bool {
    matches!(s, ApplicationReturnStatus::SolveSucceeded)
}

fn gen_spec(rng: &mut Rng) -> Spec {
    let n = rng.pick(&[2usize, 5, 20, 200]);
    let p = rng.pick(&[2i32, 4, 6, 8]);
    // Offset magnitude drives the initial gradient, hence obj_scale. The large
    // end pins it at the 1e-8 floor; the small end leaves scaling ordinary, so
    // the sweep covers both sides of the threshold.
    let amag = rng.pick(&[1.0, 10.0, 1e3, 1e5]);
    // Conditioning spread across coordinates.
    let cspread = rng.pick(&[1.0, 1e3, 1e6]);
    // Non-convexity: 0 for convex, positive adds a concave well.
    let wmag = rng.pick(&[0.0, 0.0, 1.0, 100.0]);
    let x0 = rng.pick(&[0.0, 2.0, -50.0]);
    // Constraint block: none, equalities, or inequalities. Kept small relative
    // to `n` so the problems stay solvable.
    let m = rng.pick(&[0usize, 0, 1, 3]).min(n.saturating_sub(1));
    let eq = rng.pick(&[true, false]);
    let a: Vec<Number> = (0..n).map(|_| (rng.unit() * 2.0 - 1.0) * amag).collect();
    let arows: Vec<Vec<Number>> = (0..m)
        .map(|_| (0..n).map(|_| rng.unit() * 2.0 - 1.0).collect())
        .collect();
    // Route each row near the unconstrained minimum so the feasible set is
    // non-empty and the constraints sometimes bind.
    let brhs = arows
        .iter()
        .map(|row| {
            let at_min: Number = row.iter().zip(&a).map(|(r, ai)| r * ai).sum();
            at_min + (rng.unit() * 2.0 - 1.0) * amag.sqrt()
        })
        .collect();
    Spec {
        n,
        p,
        a,
        c: (0..n).map(|_| 1.0 + rng.unit() * (cspread - 1.0)).collect(),
        w: (0..n).map(|_| rng.unit() * wmag).collect(),
        x0,
        arows,
        brhs,
        eq,
    }
}

/// The core guarantee, stated adversarially: enabling the veto must never turn
/// a successful solve into a failure, and must never return a worse point.
///
/// Both halves have teeth. The first is what the two-site fallback got wrong
/// (a run held back by the veto could exit through an unwired path); the second
/// is what the last-acceptable snapshot got wrong (the restored point could
/// drift away from the one that was refused).
#[test]
fn veto_never_degrades_status_or_objective() {
    let mut rng = Rng(0x5EED_2000);
    let (mut cases, mut improved, mut vetoed_paths) = (0, 0, 0);
    for case in 0..240 {
        let spec = gen_spec(&mut rng);
        let base = run(&spec, Some(0.0), None);
        let veto = run(&spec, None, None);
        cases += 1;

        if succeeded(base.status) {
            assert!(
                succeeded(veto.status),
                "case {case} (n={} p={} x0={}): baseline succeeded but veto gave {:?}",
                spec.n,
                spec.p,
                spec.x0,
                veto.status
            );
            // Minimization: never return a worse objective. The slack is
            // relative and tiny — this is meant to catch real regressions, not
            // last-bit noise.
            let slack = 1e-9 * base.obj.abs().max(1.0);
            assert!(
                veto.obj <= base.obj + slack,
                "case {case} (n={} p={} amag~{:.0e} w={}): veto objective {:.12e} is WORSE than \
                 baseline {:.12e}",
                spec.n,
                spec.p,
                spec.a.iter().fold(0.0_f64, |m, v| m.max(v.abs())),
                spec.w.iter().fold(0.0_f64, |m, v| m.max(*v)),
                veto.obj,
                base.obj
            );
            if veto.obj < base.obj - slack {
                improved += 1;
            }
        }
        if veto.iters > base.iters {
            vetoed_paths += 1;
        }
    }
    // Guard the premise: if the generator stopped producing problems where the
    // veto actually engages, every assertion above would pass vacuously.
    assert!(
        vetoed_paths >= 10,
        "only {vetoed_paths}/{cases} cases engaged the veto — the fuzz is not exercising it"
    );
    eprintln!("fuzz: {cases} cases, veto engaged on {vetoed_paths}, improved {improved}");
}

/// The paths the original two-site fallback silently missed.
///
/// The veto spends extra iterations by design, so anything that bounds the run
/// can fire *because* of it — and before this was fixed only two of sixteen
/// termination sites restored the refused certificate. Here the run is cut off
/// at exactly the iteration count the baseline needed, which guarantees the
/// veto run cannot finish naturally and must exit through the cap instead.
///
/// A CPU-time budget is the other such bound and flows through the same
/// post-loop hook, but it is deliberately not fuzzed: a threshold tight enough
/// to cut the veto run but loose enough to spare the baseline is a race, and a
/// flaky test here would be worse than none. (An earlier revision of this file
/// did exactly that and "failed" only because the budget was so small the veto
/// never fired at all — the test was wrong, not the code.)
#[test]
fn an_exit_forced_before_the_veto_finishes_still_yields_the_refused_certificate() {
    let mut rng = Rng(0xC0DE_2000);
    let (mut checked, mut forced) = (0, 0);
    for case in 0..80 {
        let spec = gen_spec(&mut rng);
        let base = run(&spec, Some(0.0), None);
        if !succeeded(base.status) || base.iters == 0 {
            continue;
        }
        let veto_free = run(&spec, None, None);
        // Only interesting where the veto actually made the run longer; that is
        // precisely the population that a cap can now cut off.
        if veto_free.iters <= base.iters {
            continue;
        }
        forced += 1;

        // Cap at the baseline's own iteration count: the veto run provably
        // cannot converge within it.
        let capped = run_capped(&spec, None, base.iters as i32);
        checked += 1;
        assert!(
            !matches!(
                capped.status,
                ApplicationReturnStatus::MaximumIterationsExceeded
            ),
            "case {case}: a veto cut short at {} iters surfaced MaximumIterationsExceeded \
             where the baseline succeeded",
            base.iters
        );
        let slack = 1e-9 * base.obj.abs().max(1.0);
        assert!(
            capped.obj <= base.obj + slack,
            "case {case}: cut-short veto objective {:.12e} is worse than the refused \
             certificate {:.12e}",
            capped.obj,
            base.obj
        );
    }
    assert!(
        forced >= 10 && checked >= 10,
        "only {checked} cases exercised a forced exit — the fuzz is not reaching this path"
    );
    eprintln!("forced-exit fuzz: {checked} cases checked");
}

/// The opt-out must be inert: with the mechanism disabled, results must not
/// depend on it at all, and repeated runs must agree bit-for-bit.
#[test]
fn opt_out_is_inert_and_the_solver_stays_deterministic() {
    let mut rng = Rng(0xDEAD_2000);
    for case in 0..40 {
        let spec = gen_spec(&mut rng);
        let a = run(&spec, Some(0.0), None);
        let b = run(&spec, Some(0.0), None);
        assert_eq!(
            format!("{:?}", a.status),
            format!("{:?}", b.status),
            "case {case}: opt-out is non-deterministic"
        );
        assert!(
            (a.obj - b.obj).abs() <= 0.0 || a.obj.to_bits() == b.obj.to_bits(),
            "case {case}: opt-out objective differs between runs: {} vs {}",
            a.obj,
            b.obj
        );
        // And the veto run is itself reproducible.
        let c = run(&spec, None, None);
        let d = run(&spec, None, None);
        assert_eq!(
            format!("{:?}", c.status),
            format!("{:?}", d.status),
            "case {case}: veto run is non-deterministic"
        );
        assert!(
            c.obj.to_bits() == d.obj.to_bits(),
            "case {case}: veto objective differs between runs: {} vs {}",
            c.obj,
            d.obj
        );
    }
}

/// Evaluate the objective directly, independent of the solver's own bookkeeping.
fn eval_obj(spec: &Spec, x: &[Number]) -> Number {
    (0..spec.n)
        .map(|i| spec.c[i] * (x[i] - spec.a[i]).powi(spec.p) - spec.w[i] * x[i] * x[i])
        .sum()
}

/// The returned point must be the point the reported objective describes.
///
/// This is the failure mode a statistics-only test cannot see: the veto's
/// fallback rewrites the iterate *after* the solve loop has ended, so if the
/// restore and the reported statistics were drawn at different moments a caller
/// would receive an `x` that does not correspond to the objective it was handed
/// — a silent, and much nastier, kind of wrong answer than a bad status.
#[test]
fn the_returned_point_matches_the_reported_objective() {
    let mut rng = Rng(0xF00D_2000);
    let mut checked = 0;
    for case in 0..120 {
        let spec = gen_spec(&mut rng);
        let seen = Rc::new(RefCell::new(Vec::new()));
        let mut app = IpoptApplication::new();
        app.options_mut()
            .set_integer_value("max_iter", 300, true, false)
            .unwrap();
        app.options_mut()
            .set_integer_value("print_level", 0, true, false)
            .unwrap();
        app.initialize().unwrap();
        let tnlp: Rc<RefCell<dyn TNLP>> =
            Rc::new(RefCell::new(Problem(spec.clone(), Rc::clone(&seen))));
        let status = app.optimize_tnlp(tnlp);
        let reported = app.statistics().final_objective;
        let x = seen.borrow().clone();
        if x.len() != spec.n || !reported.is_finite() {
            continue;
        }
        checked += 1;
        let direct = eval_obj(&spec, &x);
        let scale = reported.abs().max(direct.abs()).max(1.0);
        assert!(
            (direct - reported).abs() <= 1e-6 * scale,
            "case {case} ({status:?}): returned x evaluates to {direct:.12e} but the reported \
             objective is {reported:.12e}"
        );
    }
    assert!(
        checked >= 60,
        "only {checked} cases produced a usable solution vector"
    );
    eprintln!("solution-consistency fuzz: {checked} cases checked");
}

/// Veto state must not leak between solves on a reused application object.
///
/// `vetoed_iterate` holds a full iterate snapshot. If it survived into a second
/// solve — a different problem, possibly a different dimension — restoring it
/// would at best return a stranger's answer and at worst corrupt the iterate.
/// Reusing one `IpoptApplication` across solves is ordinary in warm-start and
/// parametric workflows, and every other test here builds a fresh app, so
/// nothing else would catch this.
#[test]
fn veto_state_does_not_leak_across_solves_on_a_reused_application() {
    let mut rng = Rng(0xBEEF_2000);
    let mut app = IpoptApplication::new();
    app.options_mut()
        .set_integer_value("max_iter", 300, true, false)
        .unwrap();
    app.options_mut()
        .set_integer_value("print_level", 0, true, false)
        .unwrap();
    app.initialize().unwrap();

    for case in 0..40 {
        let spec = gen_spec(&mut rng);
        // Reference: same problem on a pristine application.
        let fresh = run(&spec, None, None);

        let seen = Rc::new(RefCell::new(Vec::new()));
        let tnlp: Rc<RefCell<dyn TNLP>> =
            Rc::new(RefCell::new(Problem(spec.clone(), Rc::clone(&seen))));
        let status = app.optimize_tnlp(tnlp);
        let obj = app.statistics().final_objective;

        assert_eq!(
            format!("{status:?}"),
            format!("{:?}", fresh.status),
            "case {case}: reused application gave a different status than a fresh one"
        );
        let scale = obj.abs().max(fresh.obj.abs()).max(1.0);
        assert!(
            (obj - fresh.obj).abs() <= 1e-9 * scale,
            "case {case}: reused application gave {obj:.12e}, fresh gave {:.12e} — state leaked",
            fresh.obj
        );
    }
}
