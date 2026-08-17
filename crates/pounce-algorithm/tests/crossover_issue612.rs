//! gh#612 — crossover: hand a converged IPM iterate to the active-set path
//! to identify an *exact* active set.
//!
//! # What these tests measure, and what they deliberately do not
//!
//! The tempting assertions here — "did it still solve", "was it fast" — are
//! both useless for this feature. Crossover runs only on a solve that has
//! already converged, so "did it solve" was answered before it started; and
//! it is a strict addition of work, so "was it faster" is a foregone no.
//!
//! The property that actually discriminates is **failure of strict
//! complementarity**, and the measurement is how far the returned point sits
//! from the constraints that are active at it. An interior method cannot put
//! an iterate on a constraint: the fraction-to-boundary rule holds it
//! `O(√μ)` away, which at `μ ≈ 1e-9` is a distance of about `1e-5` — five
//! orders of magnitude larger than the `1e-8` tolerance the solve reports
//! converged at, and far too large for a downstream tolerance test to call
//! the constraint active with any confidence. That gap is the AMBIGUOUS
//! class in `docs/src/sensitivity.md`, and closing it is the whole feature.
//!
//! So each degenerate case below asserts the distance drops from `> 1e-7`
//! (measured on the same problem with crossover off) to `< 1e-10`, on a
//! problem whose exact solution is known analytically.

use pounce_algorithm::application::IpoptApplication;
use pounce_common::types::Number;
use pounce_nlp::return_codes::ApplicationReturnStatus;
use pounce_nlp::tnlp::{
    BoundsInfo, IndexStyle, IpoptCq, IpoptData, NlpInfo, Solution, SparsityRequest, StartingPoint,
    TNLP,
};
use std::cell::RefCell;
use std::rc::Rc;

const INF: Number = 2.0e19;

/// Captures the point handed to `finalize_solution`, so a test can measure
/// the returned solution rather than infer it from the objective.
#[derive(Default, Clone)]
struct Captured {
    x: Vec<Number>,
    g: Vec<Number>,
    obj: Number,
}

// ─────────────────────────────────────────────────────────────────
// Problem A — a weakly active *bound*.
//
//   min (x₀ − 1)² + (x₁ − ½)²   s.t.  0 ≤ x ≤ 1
//
// x* = (1, ½). The upper bound on x₀ is active with multiplier exactly
// zero — the unconstrained minimum sits precisely on it — so strict
// complementarity fails there. x₁ is strictly interior, the control: a
// correct crossover must leave it alone.
//
// The barrier's own stationarity for x₀ is `2(x₀−1) + μ/(1−x₀) = 0`, i.e.
// `1 − x₀ = √(μ/2)`; at the converged `μ ≈ 1e-9` that is ≈ 2e-5. No
// tolerance test on the interior iterate can call this bound active.
// ─────────────────────────────────────────────────────────────────
struct WeakBound;

impl TNLP for WeakBound {
    fn get_nlp_info(&mut self) -> Option<NlpInfo> {
        Some(NlpInfo {
            n: 2,
            m: 0,
            nnz_jac_g: 0,
            nnz_h_lag: 2,
            index_style: IndexStyle::C,
        })
    }
    fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
        b.x_l.copy_from_slice(&[0.0, 0.0]);
        b.x_u.copy_from_slice(&[1.0, 1.0]);
        true
    }
    fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
        sp.x.copy_from_slice(&[0.5, 0.2]);
        true
    }
    fn eval_f(&mut self, x: &[Number], _n: bool) -> Option<Number> {
        Some((x[0] - 1.0).powi(2) + (x[1] - 0.5).powi(2))
    }
    fn eval_grad_f(&mut self, x: &[Number], _n: bool, g: &mut [Number]) -> bool {
        g[0] = 2.0 * (x[0] - 1.0);
        g[1] = 2.0 * (x[1] - 0.5);
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
        obj_factor: Number,
        _lambda: Option<&[Number]>,
        _nl: bool,
        mode: SparsityRequest<'_>,
    ) -> bool {
        match mode {
            SparsityRequest::Structure { irow, jcol } => {
                irow.copy_from_slice(&[0, 1]);
                jcol.copy_from_slice(&[0, 1]);
            }
            SparsityRequest::Values { values } => {
                values[0] = 2.0 * obj_factor;
                values[1] = 2.0 * obj_factor;
            }
        }
        true
    }
    fn finalize_solution(&mut self, _s: Solution<'_>, _d: &IpoptData, _q: &IpoptCq) {}
}

// ─────────────────────────────────────────────────────────────────
// Problem B — a weakly active *constraint row*.
//
//   min (x₀ − 1)² + x₁²   s.t.  x₀ + x₁ ≤ 1
//
// x* = (1, 0), where the row holds with equality and multiplier zero: the
// unconstrained minimum lies exactly on the constraint. Same degeneracy as
// problem A, moved from the bound block to the row block — which is the
// block whose multiplier sign convention gh#612 also had to correct.
// ─────────────────────────────────────────────────────────────────
struct WeakRow {
    seen: Rc<RefCell<Captured>>,
}

impl TNLP for WeakRow {
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
        b.x_l.copy_from_slice(&[-INF, -INF]);
        b.x_u.copy_from_slice(&[INF, INF]);
        b.g_l[0] = -INF;
        b.g_u[0] = 1.0;
        true
    }
    fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
        sp.x.copy_from_slice(&[0.0, 0.0]);
        true
    }
    fn eval_f(&mut self, x: &[Number], _n: bool) -> Option<Number> {
        Some((x[0] - 1.0).powi(2) + x[1] * x[1])
    }
    fn eval_grad_f(&mut self, x: &[Number], _n: bool, g: &mut [Number]) -> bool {
        g[0] = 2.0 * (x[0] - 1.0);
        g[1] = 2.0 * x[1];
        true
    }
    fn eval_g(&mut self, x: &[Number], _n: bool, g: &mut [Number]) -> bool {
        g[0] = x[0] + x[1];
        true
    }
    fn eval_jac_g(&mut self, _x: Option<&[Number]>, _n: bool, mode: SparsityRequest<'_>) -> bool {
        match mode {
            SparsityRequest::Structure { irow, jcol } => {
                irow.copy_from_slice(&[0, 0]);
                jcol.copy_from_slice(&[0, 1]);
            }
            SparsityRequest::Values { values } => {
                values[0] = 1.0;
                values[1] = 1.0;
            }
        }
        true
    }
    fn eval_h(
        &mut self,
        _x: Option<&[Number]>,
        _n: bool,
        obj_factor: Number,
        _lambda: Option<&[Number]>,
        _nl: bool,
        mode: SparsityRequest<'_>,
    ) -> bool {
        match mode {
            SparsityRequest::Structure { irow, jcol } => {
                irow.copy_from_slice(&[0, 1]);
                jcol.copy_from_slice(&[0, 1]);
            }
            SparsityRequest::Values { values } => {
                values[0] = 2.0 * obj_factor;
                values[1] = 2.0 * obj_factor;
            }
        }
        true
    }
    fn finalize_solution(&mut self, s: Solution<'_>, _d: &IpoptData, _q: &IpoptCq) {
        *self.seen.borrow_mut() = Captured {
            x: s.x.to_vec(),
            g: s.g.to_vec(),
            obj: s.obj_value,
        };
    }
}

/// Solve `WeakBound`, optionally with crossover, returning
/// `(status, x, crossover accepted?)`.
fn solve_weak_bound(crossover: bool) -> (ApplicationReturnStatus, Vec<Number>, Option<bool>) {
    let seen = Rc::new(RefCell::new(Captured::default()));
    // `WeakBound` has no constraints, so capture the point through the
    // application's own converged-callback-free path: re-derive x from the
    // objective is not possible here, so wrap it.
    struct Capture {
        inner: WeakBound,
        seen: Rc<RefCell<Captured>>,
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
            obj_factor: Number,
            lambda: Option<&[Number]>,
            nl: bool,
            mode: SparsityRequest<'_>,
        ) -> bool {
            self.inner.eval_h(x, n, obj_factor, lambda, nl, mode)
        }
        fn finalize_solution(&mut self, s: Solution<'_>, _d: &IpoptData, _q: &IpoptCq) {
            self.seen.borrow_mut().x = s.x.to_vec();
            self.seen.borrow_mut().obj = s.obj_value;
        }
    }

    let mut app = IpoptApplication::new();
    if crossover {
        app.options_mut()
            .set_string_value("crossover", "yes", true, false)
            .unwrap();
    }
    app.initialize().unwrap();
    let tnlp: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(Capture {
        inner: WeakBound,
        seen: Rc::clone(&seen),
    }));
    let status = app.optimize_tnlp(tnlp);
    let accepted = app.crossover_report().map(|r| r.accepted());
    let x = seen.borrow().x.clone();
    (status, x, accepted)
}

fn solve_weak_row(crossover: bool) -> (ApplicationReturnStatus, Captured, Option<bool>) {
    let seen = Rc::new(RefCell::new(Captured::default()));
    let mut app = IpoptApplication::new();
    if crossover {
        app.options_mut()
            .set_string_value("crossover", "yes", true, false)
            .unwrap();
    }
    app.initialize().unwrap();
    let tnlp: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(WeakRow {
        seen: Rc::clone(&seen),
    }));
    let status = app.optimize_tnlp(tnlp);
    let accepted = app.crossover_report().map(|r| r.accepted());
    let out = seen.borrow().clone();
    (status, out, accepted)
}

fn succeeded(s: ApplicationReturnStatus) -> bool {
    matches!(
        s,
        ApplicationReturnStatus::SolveSucceeded | ApplicationReturnStatus::SolvedToAcceptableLevel
    )
}

/// The baseline the feature is measured against: without crossover, the
/// converged interior iterate is `O(√μ)` from the weakly active bound.
///
/// This is not a bug in the interior method — it is what an interior method
/// is. The test exists so the "after" number below has something honest to
/// be compared to, and so that a future change which happens to land the IPM
/// on the bound by accident does not let the crossover assertion pass
/// vacuously.
#[test]
fn interior_solve_leaves_the_weakly_active_bound_visibly_slack() {
    let (status, x, report) = solve_weak_bound(false);
    assert!(succeeded(status), "baseline solve failed: {status:?}");
    assert_eq!(report, None, "crossover ran when the option was not set");
    let gap = (1.0 - x[0]).abs();
    assert!(
        gap > 1e-7,
        "expected the interior iterate to stand off the bound, but 1 − x₀ = {gap:e}; \
         if the IPM now lands on the bound the crossover test below proves nothing"
    );
}

/// The feature: crossover closes that gap by five-plus orders of magnitude,
/// putting the iterate *on* the bound.
#[test]
fn crossover_puts_the_weakly_active_bound_exactly_on_its_bound() {
    let (status, x, report) = solve_weak_bound(true);
    assert!(succeeded(status), "crossover solve failed: {status:?}");
    assert_eq!(report, Some(true), "crossover did not accept");
    let gap = (1.0 - x[0]).abs();
    assert!(
        gap < 1e-10,
        "x₀ should sit on its upper bound after crossover, but 1 − x₀ = {gap:e}"
    );
    // On the bound, not past it. The interior solve reports `x₀` a little
    // *outside* the declared box because `bound_relax_factor` widened it
    // (`docs/src/options.md`, "Bound relaxation"), which is what
    // `honor_original_bounds` exists to paper over. Crossover pivots against
    // the declared bounds, so the point it returns is already inside them.
    assert!(
        x[0] <= 1.0 + 1e-12,
        "crossover put x₀ outside the declared box: x₀ = {:.17e}",
        x[0]
    );
    // The strictly interior variable is the control: identifying the active
    // set must not disturb a component that was never near a bound.
    assert!(
        (x[1] - 0.5).abs() < 1e-6,
        "interior variable moved: x₁ = {}",
        x[1]
    );
}

#[test]
fn interior_solve_leaves_the_weakly_active_row_visibly_slack() {
    let (status, out, report) = solve_weak_row(false);
    assert!(succeeded(status), "baseline solve failed: {status:?}");
    assert_eq!(report, None);
    let gap = (1.0 - out.g[0]).abs();
    assert!(
        gap > 1e-7,
        "expected the interior iterate to stand off the row, but 1 − g = {gap:e}"
    );
}

/// The row-block twin of the bound test above. Worth having separately:
/// rows and bounds carry *opposite* multiplier sign conventions, and the
/// row half of the active-set estimate was inverted until gh#612 — a
/// bound-only test would have passed throughout.
#[test]
fn crossover_puts_the_weakly_active_row_exactly_on_its_bound() {
    let (status, out, report) = solve_weak_row(true);
    assert!(succeeded(status), "crossover solve failed: {status:?}");
    assert_eq!(report, Some(true), "crossover did not accept");
    let gap = (1.0 - out.g[0]).abs();
    assert!(
        gap < 1e-10,
        "the active row should hold with equality after crossover, but 1 − g = {gap:e}"
    );
    assert!(
        (out.x[0] - 1.0).abs() < 1e-6 && out.x[1].abs() < 1e-6,
        "x moved away from the known optimum (1, 0): {:?}",
        out.x
    );
    assert!(out.obj.abs() < 1e-12, "objective should be 0: {}", out.obj);
}

/// Crossover publishes the identified set as the SQP warm-start output —
/// the IPM → SQP handoff the active-set path never had (`docs/src/
/// active-set-sqp.md` could only warm-start from a previous *SQP* solve).
#[test]
fn crossover_publishes_the_identified_set_for_a_following_sqp_solve() {
    let mut app = IpoptApplication::new();
    app.options_mut()
        .set_string_value("crossover", "yes", true, false)
        .unwrap();
    app.initialize().unwrap();
    let seen = Rc::new(RefCell::new(Captured::default()));
    let tnlp: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(WeakRow {
        seen: Rc::clone(&seen),
    }));
    assert!(succeeded(app.optimize_tnlp(tnlp)));

    let ws = app
        .last_sqp_working_set()
        .expect("crossover should publish a working set")
        .clone();
    assert_eq!(ws.constraints.len(), 1);
    assert_eq!(
        ws.constraints[0],
        pounce_qp::ConsStatus::AtUpper,
        "the row is at its upper bound at (1, 0); the published set says otherwise"
    );
}

/// HS14: `min (x₁−2)² + (x₂−1)²  s.t.  x₁ − 2x₂ + 1 = 0, x₁²/4 + x₂² ≤ 1`.
/// `f* ≈ 1.3934649` at `x* ≈ (0.8228756, 0.9114378)`, where the inequality is
/// active with a multiplier bounded away from zero — strict complementarity
/// holds, so this is the case crossover must leave alone.
struct Hs14;
impl TNLP for Hs14 {
    fn get_nlp_info(&mut self) -> Option<NlpInfo> {
        Some(NlpInfo {
            n: 2,
            m: 2,
            nnz_jac_g: 4,
            nnz_h_lag: 3,
            index_style: IndexStyle::C,
        })
    }
    fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
        b.x_l.copy_from_slice(&[-INF; 2]);
        b.x_u.copy_from_slice(&[INF; 2]);
        b.g_l[0] = 0.0;
        b.g_u[0] = 0.0;
        b.g_l[1] = -INF;
        b.g_u[1] = 0.0;
        true
    }
    fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
        sp.x.copy_from_slice(&[2.0, 2.0]);
        true
    }
    fn eval_f(&mut self, x: &[Number], _n: bool) -> Option<Number> {
        Some((x[0] - 2.0).powi(2) + (x[1] - 1.0).powi(2))
    }
    fn eval_grad_f(&mut self, x: &[Number], _n: bool, g: &mut [Number]) -> bool {
        g[0] = 2.0 * (x[0] - 2.0);
        g[1] = 2.0 * (x[1] - 1.0);
        true
    }
    fn eval_g(&mut self, x: &[Number], _n: bool, g: &mut [Number]) -> bool {
        g[0] = x[0] - 2.0 * x[1] + 1.0;
        g[1] = x[0] * x[0] / 4.0 + x[1] * x[1] - 1.0;
        true
    }
    fn eval_jac_g(&mut self, x: Option<&[Number]>, _n: bool, mode: SparsityRequest<'_>) -> bool {
        match mode {
            SparsityRequest::Structure { irow, jcol } => {
                irow.copy_from_slice(&[0, 0, 1, 1]);
                jcol.copy_from_slice(&[0, 1, 0, 1]);
            }
            SparsityRequest::Values { values } => {
                let x = x.expect("values without x");
                values[0] = 1.0;
                values[1] = -2.0;
                values[2] = 0.5 * x[0];
                values[3] = 2.0 * x[1];
            }
        }
        true
    }
    fn eval_h(
        &mut self,
        _x: Option<&[Number]>,
        _n: bool,
        obj_factor: Number,
        lambda: Option<&[Number]>,
        _nl: bool,
        mode: SparsityRequest<'_>,
    ) -> bool {
        match mode {
            SparsityRequest::Structure { irow, jcol } => {
                irow.copy_from_slice(&[0, 1, 1]);
                jcol.copy_from_slice(&[0, 0, 1]);
            }
            SparsityRequest::Values { values } => {
                let lam = lambda.expect("values without lambda");
                values[0] = obj_factor * 2.0 + lam[1] * 0.5;
                values[1] = 0.0;
                values[2] = obj_factor * 2.0 + lam[1] * 2.0;
            }
        }
        true
    }
    fn finalize_solution(&mut self, _s: Solution<'_>, _d: &IpoptData, _q: &IpoptCq) {}
}

/// Where crossover should show nothing: a problem satisfying strict
/// complementarity. HS14's inequality is active with a multiplier bounded
/// away from zero, so the interior iterate's active set is already
/// unambiguous and crossover has nothing to correct — it must return the
/// same optimum rather than wander off it.
#[test]
fn crossover_is_a_no_op_on_a_strictly_complementary_solution() {
    let f_star = 1.393_464_91;
    let mut app = IpoptApplication::new();
    app.options_mut()
        .set_string_value("crossover", "yes", true, false)
        .unwrap();
    app.initialize().unwrap();
    let tnlp: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(Hs14));
    let status = app.optimize_tnlp(tnlp);
    assert!(succeeded(status), "HS14 with crossover failed: {status:?}");
    let stats = app.statistics();
    assert!(
        (stats.final_objective - f_star).abs() < 1e-6,
        "crossover moved a strictly complementary optimum: f = {} (expected {f_star})",
        stats.final_objective
    );
}

/// gh#646 — an accepted crossover must not make the solve's *reported*
/// residuals worse.
///
/// The trap this pins is that crossover and the interior iteration measure
/// against different bounds. `bound_relax_factor` (default `1e-8`) widens
/// every bound before the solve, and the interior iterate never touches even
/// the widened one, so the difference is invisible — until crossover puts the
/// point *exactly* on a declared bound, which is `1e-8` inside the relaxed
/// one. The four `s·z` complementarity blocks then read `|multiplier| · 1e-8`
/// rather than zero: on HS14 that took a converged solve from `2.5e-9` to
/// `1.8e-8`, i.e. across `tol`, so `Overall NLP error` printed above the
/// tolerance the run had converged at and `kkt_fidelity_tol` would downgrade
/// `Solve_Succeeded` on a strictly *better* point.
///
/// Asserting "not worse than the interior solve" rather than a fixed number
/// is deliberate: the defect is a relative one — a comparison between two
/// runs of the same problem — and a threshold would drift with `tol`.
#[test]
fn crossover_does_not_inflate_the_reported_complementarity() {
    let solve = |crossover: bool| {
        let mut app = IpoptApplication::new();
        app.options_mut()
            .set_string_value(
                "crossover",
                if crossover { "yes" } else { "no" },
                true,
                false,
            )
            .unwrap();
        app.initialize().unwrap();
        let tnlp: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(Hs14));
        let status = app.optimize_tnlp(tnlp);
        assert!(
            succeeded(status),
            "HS14 failed (crossover={crossover}): {status:?}"
        );
        let crossed = app
            .crossover_report()
            .is_some_and(pounce_algorithm::crossover::CrossoverReport::accepted);
        (app.statistics().clone(), crossed)
    };

    let (interior, _) = solve(false);
    let (crossed, accepted) = solve(true);
    assert!(
        accepted,
        "HS14 crossover was expected to run and be accepted; without that this \
         test is not measuring anything"
    );

    // The reported complementarity is the term the relaxation lands on.
    assert!(
        crossed.final_compl <= interior.final_compl,
        "crossover inflated the reported complementarity: {:.6e} (interior) → {:.6e}",
        interior.final_compl,
        crossed.final_compl
    );
    assert!(
        crossed.final_unscaled_compl <= interior.final_unscaled_compl,
        "crossover inflated the reported unscaled complementarity: {:.6e} → {:.6e}",
        interior.final_unscaled_compl,
        crossed.final_unscaled_compl
    );

    // ...and therefore neither aggregate crosses `tol`, which is what made
    // the summary and the `kkt_fidelity_tol` gate disagree with the status.
    let tol = 1e-8;
    assert!(
        crossed.final_kkt_error <= tol,
        "Overall NLP error above tol after crossover: {:.6e}",
        crossed.final_kkt_error
    );
    assert!(
        crossed.final_unscaled_kkt_error <= tol,
        "unscaled KKT error above tol after crossover — `kkt_fidelity_tol` \
         would downgrade this status: {:.6e}",
        crossed.final_unscaled_kkt_error
    );

    // The other two terms were already better; guard against a "fix" that
    // trades them away for the complementarity number.
    assert!(
        crossed.final_dual_inf <= interior.final_dual_inf.max(tol),
        "dual infeasibility regressed: {:.6e} → {:.6e}",
        interior.final_dual_inf,
        crossed.final_dual_inf
    );
    assert!(
        crossed.final_constr_viol <= interior.final_constr_viol.max(tol),
        "constraint violation regressed: {:.6e} → {:.6e}",
        interior.final_constr_viol,
        crossed.final_constr_viol
    );
}
