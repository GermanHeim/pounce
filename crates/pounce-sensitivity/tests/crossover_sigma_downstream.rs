//! Crossover, `bound_relax_factor`, and the reduced Hessian read off the
//! held factor (gh#654, split out of gh#653; same root cause as gh#646).
//!
//! `bound_relax_factor` (default `1e-8`) widens every bound by `δ` before
//! the interior solve. Crossover then parks the iterate exactly on the
//! **declared** bound, which is a full `δ` inside the live relaxed one. The
//! barrier therefore sees a slack of exactly `δ` at every active bound,
//! where an interior iterate would have carried `μ/z`. Since the barrier
//! diagonal is `Σ = z/s`:
//!
//! ```text
//! no crossover     Σ = z / (μ/z) = z²/μ
//! crossover        Σ = z / δ
//! ```
//!
//! and crossover **loosens** the pin whenever `z·δ/μ > 1` — the normal case,
//! because `δ` is capped at `constr_viol_tol` and `μ` ends near
//! `tol/(barrier_tol_factor+1)`. `Σ` is the stiffness with which the barrier
//! holds a bounded variable, and any reduced Hessian read off the held KKT
//! factor carries a residual error of exactly `O(1/Σ)` — the leftover of that
//! pin being finite. A looser pin is a less accurate covariance.
//!
//! # The fixture
//!
//! `min ½·xᵀQx − qᵀx` over `(a, b, w)`. `a` and `b` are held by the two
//! equality rows, which are the pin rows the reduced Hessian is taken over;
//! `w` is capped by an upper bound that binds with multiplier `z`. `Q` is
//! non-diagonal, so the reduced Hessian over the pins differs by `O(1)`
//! between the bound-pinned answer (`Q_ab`) and the one that lets `w` move
//! (`Q_ab − Q_aw·Q_ww⁻¹·Q_wa`). The measured error below is
//! `‖H_R − Q_ab‖_∞`: how much of the free answer is still leaking through a
//! pin that is finite rather than exact.
//!
//! # What is asserted
//!
//! Three directions, each a comparison between runs rather than a fixed
//! threshold — the effect is a ratio and a fixed bar would drift with `tol`:
//!
//! 1. crossover under `bound_relax_factor = 0` is *more* accurate than no
//!    crossover — the pin tightens from the `μ/z` standoff to the point's
//!    own distance from the bound, which is nothing;
//! 2. crossover under the **default** relaxation is also more accurate than
//!    no crossover. This is the gh#654 defect in executable form: before the
//!    fix it was `18x` **worse** at `z = 4.5`, and the degradation grew with
//!    the bound multiplier exactly as `z·δ/μ` predicts;
//! 3. the two crossover runs agree. That is the fix's actual claim: `Σ` is
//!    now read in the frame crossover solved in, so whether the bounds were
//!    relaxed no longer changes the answer.
//!
//! Measured on this fixture, before → after:
//!
//! ```text
//!    z     no crossover    crossover, δ=1e-8        crossover, δ=0
//!   4.5      6.07e-11    1.09e-09 → 8.88e-16    1.99e-13 → 8.88e-16
//!  94.5      1.38e-13    5.19e-11 → 8.88e-16    9.99e-15 → 8.88e-16
//! 994.5      1.24e-14    4.93e-12 → 0           8.88e-16 → 0
//! ```
//!
//! Both crossover columns now agree entry for entry, and both sit at the
//! roundoff of the answer itself: with the point *on* its bound the pin is
//! as exact as double precision expresses, so the `O(1/Σ)` leak is gone
//! rather than merely smaller.
//!
//! A fourth test guards the plumbing rather than the number: the batched
//! back-solve must reach the same corrected system the single-RHS one does,
//! which it can only do by declining its two cached tiers.

use std::cell::RefCell;
use std::rc::Rc;

use pounce_algorithm::application::IpoptApplication;
use pounce_common::types::{Index, Number};
use pounce_nlp::return_codes::ApplicationReturnStatus;
use pounce_nlp::tnlp::{
    BoundsInfo, IndexStyle, IpoptCq, IpoptData, Linearity, NlpInfo, Solution, SparsityRequest,
    StartingPoint, TNLP,
};
use pounce_sensitivity::Solver;

/// Objective Hessian, `(a, b, w)` order. Non-diagonal in the `w` column:
/// that coupling is the whole experiment, since it is what makes the
/// bound-pinned reduced Hessian differ from the free one.
const Q: [[Number; 3]; 3] = [[2.0, 0.3, 0.7], [0.3, 3.0, 0.5], [0.7, 0.5, 4.0]];

/// Where the pin rows hold `a` and `b`.
const A_PIN: Number = 0.5;
const B_PIN: Number = 0.25;

/// `w`'s upper bound.
const W_CAP: Number = 1.0;

/// The reduced Hessian over `(a, b)` when `w` is held at its bound: the
/// leading 2×2 block of `Q`, column-major, in the sign convention
/// `compute_reduced_hessian` reports pin rows in (`H_R = B·K⁻¹·Bᵀ`,
/// which carries the augmented system's leading minus on a multiplier
/// row). Letting `w` move instead would give
/// `−(Q_ab − Q_aw·Q_ww⁻¹·Q_wa)`, an `O(0.1)` different matrix — that gap
/// is what a finite pin leaks a fraction of.
const Q_AB: [Number; 4] = [-Q[0][0], -Q[0][1], -Q[1][0], -Q[1][1]];

/// `min ½·xᵀQx − qᵀx` s.t. `a = A_PIN`, `b = B_PIN`, `w ≤ W_CAP`.
///
/// `q_w` is derived from the bound multiplier the caller wants: with `a`
/// and `b` pinned and `w` at its cap, the `w` row of the stationarity
/// condition reads `(Qx)_w − q_w + z = 0`.
struct CappedQp {
    q_w: Number,
}

impl CappedQp {
    /// A fixture whose `w` bound binds with multiplier `z`.
    fn with_bound_multiplier(z: Number) -> Self {
        let qx_w = Q[2][0] * A_PIN + Q[2][1] * B_PIN + Q[2][2] * W_CAP;
        Self { q_w: qx_w + z }
    }

    fn q(&self) -> [Number; 3] {
        [0.0, 0.0, self.q_w]
    }
}

impl TNLP for CappedQp {
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
        sp.x[0] = 0.0;
        sp.x[1] = 0.0;
        sp.x[2] = 0.0;
        true
    }

    fn get_constraints_linearity(&mut self, types: &mut [Linearity]) -> bool {
        types.fill(Linearity::Linear);
        true
    }

    fn eval_f(&mut self, x: &[Number], _new_x: bool) -> Option<Number> {
        let q = self.q();
        let mut f = 0.0;
        for i in 0..3 {
            f -= q[i] * x[i];
            for j in 0..3 {
                f += 0.5 * Q[i][j] * x[i] * x[j];
            }
        }
        Some(f)
    }

    fn eval_grad_f(&mut self, x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
        let q = self.q();
        for i in 0..3 {
            g[i] = -q[i];
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
                irow[0] = 0;
                jcol[0] = 0;
                irow[1] = 1;
                jcol[1] = 1;
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
        // Dense lower triangle of Q; the constraints are linear so they
        // contribute nothing.
        let mut k = 0;
        match mode {
            SparsityRequest::Structure { irow, jcol } => {
                for i in 0..3 {
                    for j in 0..=i {
                        irow[k] = i as Index;
                        jcol[k] = j as Index;
                        k += 1;
                    }
                }
            }
            SparsityRequest::Values { values } => {
                for (i, row) in Q.iter().enumerate() {
                    for &q in row.iter().take(i + 1) {
                        values[k] = obj_factor * q;
                        k += 1;
                    }
                }
            }
        }
        true
    }

    fn finalize_solution(&mut self, _s: Solution<'_>, _d: &IpoptData, _c: &IpoptCq) {}
}

/// Solve the fixture under one `(crossover, bound_relax_factor)` combination
/// and return `‖H_R − Q_ab‖_∞`, the residual of the bound's pin being finite
/// rather than exact.
fn reduced_hessian_error(z: Number, crossover: bool, brf: Number) -> Number {
    let mut app = IpoptApplication::new();
    let opts = app.options_mut();
    opts.set_integer_value("print_level", 0, true, false)
        .unwrap();
    opts.set_string_value("sb", "yes", true, false).unwrap();
    opts.set_string_value(
        "crossover",
        if crossover { "yes" } else { "no" },
        true,
        false,
    )
    .unwrap();
    opts.set_numeric_value("bound_relax_factor", brf, true, false)
        .unwrap();
    app.initialize().unwrap();

    let tnlp: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(CappedQp::with_bound_multiplier(z)));
    let mut solver = Solver::new(app, tnlp);
    let status = solver.solve();
    assert!(
        matches!(
            status,
            ApplicationReturnStatus::SolveSucceeded
                | ApplicationReturnStatus::SolvedToAcceptableLevel
        ),
        "fixture must converge (crossover={crossover}, brf={brf:e}); got {status:?}",
    );
    let accepted = solver
        .app()
        .crossover_report()
        .is_some_and(pounce_algorithm::crossover::CrossoverReport::accepted);
    assert_eq!(
        accepted, crossover,
        "the run must actually cross over when asked to (brf={brf:e})",
    );

    let hr = solver
        .compute_reduced_hessian(&[0, 1], 1.0)
        .expect("reduced Hessian over the two pin rows");
    hr.iter()
        .zip(Q_AB.iter())
        .fold(0.0, |acc: Number, (&h, &q)| acc.max((h - q).abs()))
}

#[test]
fn crossover_without_bound_relaxation_tightens_the_reduced_hessian() {
    let z = 4.5;
    let interior = reduced_hessian_error(z, false, 0.0);
    let crossed = reduced_hessian_error(z, true, 0.0);
    assert!(
        crossed < interior,
        "crossover at bound_relax_factor=0 must tighten the pin: \
         interior={interior:e}, crossed={crossed:e}",
    );
}

/// gh#654. The defect: under the **default** relaxation the same crossover
/// loosened the pin instead, by a factor that tracked the bound multiplier
/// (`18x` worse at `z = 4.5`, `376x` at `z = 94.5`, `396x` at `z = 994.5`,
/// off the table in the module doc).
#[test]
fn crossover_under_bound_relaxation_does_not_loosen_the_reduced_hessian() {
    for z in [4.5, 94.5, 994.5] {
        let interior = reduced_hessian_error(z, false, 1e-8);
        let crossed = reduced_hessian_error(z, true, 1e-8);
        assert!(
            crossed <= interior,
            "z={z}: crossover under bound_relax_factor=1e-8 must not be less \
             accurate than no crossover at all: interior={interior:e}, \
             crossed={crossed:e} ({:.1}x worse)",
            crossed / interior,
        );
    }
}

/// The fix's claim, stated directly: whether the bounds were relaxed no
/// longer changes what a crossed-over solve reports downstream, because `Σ`
/// is measured in the frame crossover solved in.
#[test]
fn a_crossed_over_solve_reads_the_same_whether_or_not_bounds_were_relaxed() {
    let z = 4.5;
    let relaxed = reduced_hessian_error(z, true, 1e-8);
    let unrelaxed = reduced_hessian_error(z, true, 0.0);
    let scale = relaxed.max(unrelaxed).max(1e-300);
    assert!(
        (relaxed - unrelaxed).abs() <= 0.05 * scale,
        "the two crossed-over runs must agree: relaxed={relaxed:e}, \
         unrelaxed={unrelaxed:e}",
    );
}

/// The batched back-solve has to answer in the same frame as the single-RHS
/// one.
///
/// It has two cached tiers that assemble their elimination from the
/// calculated `Σ` and fire against whatever factor the previous solve left
/// behind. On a crossed-over solve neither is the corrected system, and the
/// tag check does not save them: on the *first* call after convergence the
/// cached tags are the algorithm's own final solve, which used exactly the
/// diagonal being corrected. So the batched call here is deliberately the
/// first thing asked of the held factor.
#[test]
fn the_batched_back_solve_agrees_with_the_single_rhs_one_after_crossover() {
    let mut app = IpoptApplication::new();
    let opts = app.options_mut();
    opts.set_integer_value("print_level", 0, true, false)
        .unwrap();
    opts.set_string_value("sb", "yes", true, false).unwrap();
    opts.set_string_value("crossover", "yes", true, false)
        .unwrap();
    // The default relaxation: the combination gh#654 is about.
    opts.set_numeric_value("bound_relax_factor", 1e-8, true, false)
        .unwrap();
    app.initialize().unwrap();

    let tnlp: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(CappedQp::with_bound_multiplier(4.5)));
    let mut solver = Solver::new(app, tnlp);
    solver.solve();
    assert!(
        solver
            .app()
            .crossover_report()
            .is_some_and(pounce_algorithm::crossover::CrossoverReport::accepted),
        "fixture must cross over",
    );
    let dim = solver.kkt_dim().expect("converged");

    // Three unit-ish right-hand sides spread across the compound vector.
    let n_rhs = 3;
    let mut rhs_flat = vec![0.0; n_rhs * dim];
    for (k, row) in rhs_flat.chunks_mut(dim).enumerate() {
        row[k % dim] = 1.0;
        row[(k + dim / 2) % dim] = -0.5;
    }
    let mut batched = vec![0.0; n_rhs * dim];
    solver
        .kkt_solve_many(&rhs_flat, &mut batched, n_rhs)
        .expect("batched back-solve");

    for k in 0..n_rhs {
        let mut one = vec![0.0; dim];
        solver
            .kkt_solve(&rhs_flat[k * dim..(k + 1) * dim], &mut one)
            .expect("single back-solve");
        for (i, (&b, &s)) in batched[k * dim..(k + 1) * dim]
            .iter()
            .zip(one.iter())
            .enumerate()
        {
            let scale = b.abs().max(s.abs()).max(1.0);
            assert!(
                (b - s).abs() <= 1e-9 * scale,
                "rhs {k}, row {i}: batched={b:e} vs single={s:e}",
            );
        }
    }
}
