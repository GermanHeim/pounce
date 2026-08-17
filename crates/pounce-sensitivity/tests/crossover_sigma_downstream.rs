//! What crossover's barrier diagonal `Σ = z/s` does to a *downstream*
//! reduced Hessian (gh#653).
//!
//! gh#612's PR left open "what does `Σ = V/S` do at exact-zero slack",
//! on the assumption that crossover — which puts the iterate *on* a
//! bound — drives `Σ` toward infinity and degrades whatever the
//! sensitivity path factorizes. Measurement inverted that. `Σ` is
//! finite, and a larger `Σ` is a *more* accurate answer, because `Σ` is
//! the stiffness with which the barrier pins the bounded variable and
//! the reduced Hessian's residual error is the `O(1/Σ)` leftover of
//! that pin being finite.
//!
//! # Fixture
//!
//! `min ½xᵀQx − qᵀx` over `x = (a, b, w)`, with `a` and `b` held by
//! equality rows (the pins the reduced Hessian is taken over) and `w`
//! capped by an upper bound that binds with multiplier `z ≈ 4.5` —
//! strictly complementary, so the barrier has no excuse. `Q` is
//! non-diagonal, which is what makes the measurement discriminating:
//!
//! ```text
//! w pinned at its bound : H_R = Q_ab                = [[4, 1], [1, 3]]
//! w free (eliminated)   : H_R = Q_ab − q_w q_wᵀ/Q_ww = [[3.2, .6], [.6, 2.8]]
//! ```
//!
//! Those differ by `O(1)`, so drift toward the second is unmissable.
//! The solver returns the first plus an error of exactly `Q_aw²/Σ_w`
//! — the bound block's Schur complement, matched to every printed
//! digit at both ends of a 300× range in `Σ`.
//!
//! `compute_reduced_hessian` returns `−H_R` under its sign convention
//! (`−inv(H_R)` is the covariance), so every assertion here negates it.

use std::cell::RefCell;
use std::rc::Rc;

use pounce_algorithm::application::IpoptApplication;
use pounce_common::types::{Index, Number};
use pounce_nlp::tnlp::{
    BoundsInfo, IndexStyle, IpoptCq, IpoptData, Linearity, NlpInfo, Solution, SparsityRequest,
    StartingPoint, TNLP,
};
use pounce_sensitivity::Solver;

const Q: [[Number; 3]; 3] = [[4.0, 1.0, 2.0], [1.0, 3.0, 1.0], [2.0, 1.0, 5.0]];
/// Linear term; only `w` carries one, and it is what pushes `w` into
/// its cap hard enough to leave a multiplier of `4.5`.
const QV: [Number; 3] = [0.0, 0.0, 10.0];
const A_PIN: Number = 1.0;
const B_PIN: Number = 1.0;
const W_CAP: Number = 0.5;

/// `H_R` when the bound pins `w` exactly: the `(a, b)` block of `Q`.
const H_PINNED: [Number; 4] = [Q[0][0], Q[1][0], Q[0][1], Q[1][1]];

/// The `O(1/Σ)` error constant: the reduced Hessian's residual is
/// `Q_aw²/Σ_w`, with `Q_aw = Q[0][2] = 2`.
const ERR_NUMERATOR: Number = Q[0][2] * Q[0][2];

struct Fixture;

impl TNLP for Fixture {
    fn get_nlp_info(&mut self) -> Option<NlpInfo> {
        Some(NlpInfo {
            n: 3,
            m: 2,
            nnz_jac_g: 2,
            nnz_h_lag: 6,
            index_style: IndexStyle::C,
        })
    }

    fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
        b.x_l[0] = -1.0e19;
        b.x_u[0] = 1.0e19;
        b.x_l[1] = -1.0e19;
        b.x_u[1] = 1.0e19;
        b.x_l[2] = -1.0e19;
        b.x_u[2] = W_CAP;
        b.g_l[0] = A_PIN;
        b.g_u[0] = A_PIN;
        b.g_l[1] = B_PIN;
        b.g_u[1] = B_PIN;
        true
    }

    fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
        sp.x.fill(0.0);
        true
    }

    fn get_constraints_linearity(&mut self, t: &mut [Linearity]) -> bool {
        t.fill(Linearity::Linear);
        true
    }

    fn eval_f(&mut self, x: &[Number], _new_x: bool) -> Option<Number> {
        let mut f = 0.0;
        for i in 0..3 {
            for j in 0..3 {
                f += 0.5 * x[i] * Q[i][j] * x[j];
            }
            f -= QV[i] * x[i];
        }
        Some(f)
    }

    fn eval_grad_f(&mut self, x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
        for i in 0..3 {
            g[i] = -QV[i];
            for j in 0..3 {
                g[i] += Q[i][j] * x[j];
            }
        }
        true
    }

    fn eval_g(&mut self, x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
        g[0] = x[0];
        g[1] = x[1];
        true
    }

    fn eval_jac_g(
        &mut self,
        _x: Option<&[Number]>,
        _new_x: bool,
        mode: SparsityRequest<'_>,
    ) -> bool {
        match mode {
            SparsityRequest::Structure { irow, jcol } => {
                irow.copy_from_slice(&[0 as Index, 1]);
                jcol.copy_from_slice(&[0 as Index, 1]);
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
        _new_x: bool,
        obj_factor: Number,
        _lambda: Option<&[Number]>,
        _new_lambda: bool,
        mode: SparsityRequest<'_>,
    ) -> bool {
        // Lower triangle of Q.
        let rs: [Index; 6] = [0, 1, 1, 2, 2, 2];
        let cs: [Index; 6] = [0, 0, 1, 0, 1, 2];
        match mode {
            SparsityRequest::Structure { irow, jcol } => {
                irow.copy_from_slice(&rs);
                jcol.copy_from_slice(&cs);
            }
            SparsityRequest::Values { values } => {
                for k in 0..6 {
                    values[k] = obj_factor * Q[rs[k] as usize][cs[k] as usize];
                }
            }
        }
        true
    }

    fn finalize_solution(&mut self, _s: Solution<'_>, _d: &IpoptData, _q: &IpoptCq) {}
}

fn app(crossover: bool, bound_relax_factor: Number) -> IpoptApplication {
    let mut a = IpoptApplication::new();
    a.options_mut()
        .set_integer_value("print_level", 0, true, false)
        .unwrap();
    a.options_mut()
        .set_string_value("sb", "yes", true, false)
        .unwrap();
    a.options_mut()
        .set_numeric_value("bound_relax_factor", bound_relax_factor, true, false)
        .unwrap();
    if crossover {
        a.options_mut()
            .set_string_value("crossover", "yes", true, false)
            .unwrap();
    }
    a.initialize().unwrap();
    a
}

/// `|H_R − Q_ab|_inf` for one option combination, plus the solver it
/// came from so the caller can read `Σ` off the same held state.
fn reduced_hessian_error(crossover: bool, brf: Number) -> (Number, Solver) {
    let tnlp: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(Fixture));
    let mut solver = Solver::new(app(crossover, brf), tnlp);
    solver.solve();
    let hr = solver
        .compute_reduced_hessian(&[0, 1], 1.0)
        .expect("reduced Hessian over the two pin rows");
    let err = (0..4)
        .map(|k| (-hr[k] - H_PINNED[k]).abs())
        .fold(0.0, Number::max);
    (err, solver)
}

/// Crossover run against the bounds as declared — the combination the
/// sensitivity docs steer users into, since `classify_activity`
/// *requires* `bound_relax_factor = 0` — makes the downstream reduced
/// Hessian **more** accurate, not less. This is gh#653's central
/// finding and the one that inverted its premise.
#[test]
fn crossover_at_declared_bounds_sharpens_the_reduced_hessian() {
    let (err_off, _) = reduced_hessian_error(false, 0.0);
    let (err_on, _) = reduced_hessian_error(true, 0.0);

    // Both must land on the pinned answer rather than the free one;
    // the two differ by 0.8 in the (0,0) entry, so anything near that
    // means the bound stopped pinning altogether.
    assert!(
        err_off < 1e-6 && err_on < 1e-6,
        "both runs must reproduce the w-pinned reduced Hessian, not the \
         free-w one (errors {err_off:e} / {err_on:e} against a 0.8 gap)",
    );

    // Measured 306x at the time of writing (4.95e-10 -> 1.62e-12). The
    // assertion is a wide lower bound: the point is the *direction*,
    // which is what the issue got backwards.
    assert!(
        err_on * 50.0 < err_off,
        "crossover at declared bounds must sharpen the reduced Hessian by \
         a wide margin: |err| went {err_off:e} -> {err_on:e}",
    );
}

/// The residual is the bound block's Schur complement, `Q_aw²/Σ_w`, so
/// it falls exactly as `Σ` rises. Pinning the *law* rather than two
/// numbers is what makes the previous test's margin explainable
/// instead of anecdotal — and it is the reason a growing `Σ` is a
/// benefit rather than the conditioning hazard gh#653 assumed.
#[test]
fn the_reduced_hessian_error_tracks_one_over_sigma() {
    for crossover in [false, true] {
        let (err, solver) = reduced_hessian_error(crossover, 0.0);
        // `classify_activity` is the public read of the barrier
        // diagonal, and it is available here precisely because these
        // runs use bound_relax_factor = 0.
        let report = solver
            .classify_activity()
            .expect("bound_relax_factor = 0, so the classifier accepts");
        let sigma = report.var_sigma[2];
        let predicted = ERR_NUMERATOR / sigma;
        assert!(
            (err - predicted).abs() <= 0.05 * predicted,
            "crossover={crossover}: |H_R| error {err:e} should be \
             Q_aw^2/Sigma = {predicted:e} (Sigma = {sigma:e})",
        );
    }
}

/// The other half of the measurement, and a defect rather than a
/// property: with the bound relaxation left at its default, crossover
/// makes the same reduced Hessian **worse** than not crossing over at
/// all.
///
/// Crossover parks the iterate exactly on the *declared* bound, which
/// is a full `δ = bound_relax_factor` inside the *live* relaxed one.
/// The slack the barrier then sees is `δ` rather than the `μ/z` an
/// interior iterate would have carried, so `Σ = z/δ` instead of
/// `z²/μ`, and the bound is pinned **less** stiffly. The degradation
/// factor is `z·δ/μ` — it grows with the bound's multiplier.
///
/// This is gh#646's frame mismatch reaching the numerics; gh#647 fixed
/// only the reporting half. Tracked as gh#654. The assertion records the
/// regression in executable form so it cannot drift, or be fixed,
/// unnoticed.
#[test]
fn crossover_under_bound_relaxation_loosens_the_reduced_hessian() {
    let (err_off, _) = reduced_hessian_error(false, 1e-8);
    let (err_on, _) = reduced_hessian_error(true, 1e-8);

    // Measured 18x at the time of writing (4.95e-10 -> 8.89e-9), rising
    // toward ~400x as the bound multiplier grows.
    assert!(
        err_on > err_off * 5.0,
        "known gh#654 defect: crossover under a nonzero bound_relax_factor \
         should LOOSEN the pin (errors {err_off:e} -> {err_on:e}). If this \
         now fails, the frame mismatch was fixed — delete the test and say \
         so, do not relax the bound.",
    );

    // And it is strictly worse than the same crossover run against
    // declared bounds: the relaxation, not crossover, is the cause.
    let (err_declared, _) = reduced_hessian_error(true, 0.0);
    assert!(
        err_declared < err_on,
        "crossover at declared bounds ({err_declared:e}) must beat crossover \
         under relaxation ({err_on:e})",
    );
}
