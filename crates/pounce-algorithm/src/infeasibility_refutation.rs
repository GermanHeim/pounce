//! Refuting an infeasibility verdict with a point that is actually feasible.
//!
//! # Why this exists
//!
//! `Infeasible_Problem_Detected` (AMPL `solve_result_num` 200, Pyomo
//! `TerminationCondition.infeasible`) is the most consequential thing POUNCE
//! can say: a caller told a feasible model is infeasible has no signal that
//! anything went wrong. It fails silently and confidently, which is worse than
//! an error.
//!
//! The numerical paths that produce that verdict — the restoration gates, the
//! outer cycle detector, the SQP infeasible-subproblem exit, the ℓ₁ wrapper's
//! uncollapsed-slack certificate — all reason from a *local* argument: the
//! feasibility sub-problem stopped making progress at a point whose violation
//! is bounded away from zero. That is evidence, not proof, and gh #379 is what
//! it looks like when the evidence is wrong. On seed 294 of the
//! feasible-by-construction property sweep
//! (`pyomo-pounce/tests/test_infeasibility_no_false_positives.py`) the solver
//! *starts* at a point that satisfies every row exactly, walks away from it
//! (the model carries `±1e30` row coefficients, so the barrier's slack
//! initialization moves the scaled slack far from what `x` can follow), burns
//! the restoration budget, and reports the model infeasible.
//!
//! A concrete feasible point settles the question outright. If some `x` inside
//! the variable box satisfies every constraint, the feasible set is not empty,
//! whatever a local argument concluded. So: before any numerical path is
//! allowed to say "infeasible", try to refute it.
//!
//! # Which point
//!
//! The model's own starting point, clamped into the variable box. Deliberately
//! *only* that one, unlike the presolve-side refutation
//! (`pounce_presolve::witness_refutes_infeasibility`), which also samples the
//! box midpoint and corners.
//!
//! The two are answering different questions. Presolve claims a *proof* over
//! the whole box from interval arithmetic, so probing the box is exactly the
//! right counter-evidence. Here the claim is numerical and the refutation runs
//! on every solve that ends in the infeasible band; widening it to sampled
//! points would change verdicts on models this change cannot be validated
//! against (the benchmark corpus is not in the tree). The starting point needs
//! no such justification: a modeller who hands the solver a feasible point and
//! is told the model is infeasible has been given a wrong answer under any
//! reading.
//!
//! # Direction
//!
//! One-directional, like its presolve twin: this can only ever *withdraw* a
//! verdict, never create one. A model with no feasible point cannot produce a
//! witness, so a genuinely infeasible model is untouched — and any failure to
//! evaluate (`eval_g` returning false, a non-finite value, a missing starting
//! point) simply declines to refute.

use pounce_common::tolerance::is_significant;
use pounce_common::types::Number;
use pounce_nlp::tnlp::{BoundsInfo, StartingPoint, TNLP};
use std::cell::RefCell;
use std::rc::Rc;

/// A point that satisfies every constraint and bound, disproving a candidate
/// infeasibility verdict.
#[derive(Debug, Clone)]
pub struct FeasibleWitness {
    /// The witnessing point, in the user's variable order.
    pub x: Vec<Number>,
    /// Largest constraint violation at `x`, for the diagnostic message. Below
    /// `tol` times the row's own magnitude by construction.
    pub max_violation: Number,
}

/// Try to refute a candidate infeasibility verdict using the model's starting
/// point.
///
/// Returns the witness when the starting point (clamped into the variable box)
/// satisfies every constraint; `None` when it does not, or when the model
/// cannot be evaluated. `None` means "no refutation", never "infeasible".
///
/// `lower_bound_inf` / `upper_bound_inf` are the solver's
/// `nlp_lower_bound_inf` / `nlp_upper_bound_inf` sentinels: a bound at or
/// beyond them is treated as absent, both for clamping and for sizing a row's
/// magnitude. Letting the sentinel (~1e19) inform the scale would put the
/// slack around 1e11 and make every row look satisfied — the same trap the
/// presolve refutation documents.
pub fn starting_point_refutes_infeasibility(
    tnlp: &Rc<RefCell<dyn TNLP>>,
    lower_bound_inf: Number,
    upper_bound_inf: Number,
    tol: Number,
) -> Option<FeasibleWitness> {
    let info = tnlp.borrow_mut().get_nlp_info()?;
    let n = info.n.max(0) as usize;
    let m = info.m.max(0) as usize;
    if n == 0 {
        return None;
    }

    let mut x_l = vec![0.0; n];
    let mut x_u = vec![0.0; n];
    let mut g_l = vec![0.0; m];
    let mut g_u = vec![0.0; m];
    if !tnlp.borrow_mut().get_bounds_info(BoundsInfo {
        x_l: &mut x_l,
        x_u: &mut x_u,
        g_l: &mut g_l,
        g_u: &mut g_u,
    }) {
        return None;
    }

    let mut x = vec![0.0; n];
    let mut z_l = vec![0.0; n];
    let mut z_u = vec![0.0; n];
    let mut lambda = vec![0.0; m];
    let have_x0 = tnlp.borrow_mut().get_starting_point(StartingPoint {
        init_x: true,
        x: &mut x,
        init_z: false,
        z_l: &mut z_l,
        z_u: &mut z_u,
        init_lambda: false,
        lambda: &mut lambda,
    });
    if !have_x0 || x.iter().any(|v| !v.is_finite()) {
        return None;
    }

    // Clamp into the box. The solver does this to `x0` itself before iterating,
    // so a point outside the declared bounds is not a witness as given — but the
    // clamped point still is one if it satisfies the rows, since it is inside
    // the box by construction.
    for j in 0..n {
        // A crossed box (`x_l > x_u`) cannot be clamped into; that is presolve's
        // territory and not something to refute from.
        if x_l[j].is_finite() && x_u[j].is_finite() && x_l[j] > x_u[j] {
            return None;
        }
        if x_l[j].is_finite() && x_l[j] > lower_bound_inf && x[j] < x_l[j] {
            x[j] = x_l[j];
        }
        if x_u[j].is_finite() && x_u[j] < upper_bound_inf && x[j] > x_u[j] {
            x[j] = x_u[j];
        }
    }

    // No constraints: the box alone defines the feasible set, and a point
    // inside a non-crossed box is a witness.
    let mut g = vec![0.0; m];
    if m > 0 && !tnlp.borrow_mut().eval_g(&x, true, &mut g) {
        return None;
    }

    let mut max_violation: Number = 0.0;
    for i in 0..m {
        let v = g[i];
        if !v.is_finite() {
            return None;
        }
        // Only *finite* bounds inform a row's magnitude — see the doc comment.
        let finite_mag = |b: Number, is_lower: bool| -> Number {
            let absent = if is_lower {
                b <= lower_bound_inf
            } else {
                b >= upper_bound_inf
            };
            if b.is_finite() && !absent {
                b.abs()
            } else {
                0.0
            }
        };
        let scale = v
            .abs()
            .max(finite_mag(g_l[i], true))
            .max(finite_mag(g_u[i], false));
        let lo_viol = if g_l[i].is_finite() && g_l[i] > lower_bound_inf {
            g_l[i] - v
        } else {
            0.0
        };
        let hi_viol = if g_u[i].is_finite() && g_u[i] < upper_bound_inf {
            v - g_u[i]
        } else {
            0.0
        };
        let viol = lo_viol.max(hi_viol).max(0.0);
        // Pure relative — `tol * scale`, via `is_significant` — and *not* the
        // clamped accepting form `is_negligible`.
        //
        // The clamped form is right when the question is "did the solver
        // converge well enough to call this feasible", because a solver
        // converges to absolute residuals. It is wrong here, where the question
        // is "is this residual real, or evaluation noise on a row of this
        // magnitude". The clamp reinstates an absolute floor for `scale < 1`,
        // and that is precisely the down-scaled direction this gate must not be
        // fooled in.
        //
        // Measured, not assumed. With `is_negligible` the scale-invariance
        // harness (`pyomo-pounce/tests/test_scale_invariance.py`) regressed on
        // three genuinely infeasible models at row scalings `1e-12 … 1e-8`:
        // `x >= 2` over `x ∈ [0, 1]`, multiplied through by `1e-12`, has a
        // violation of `2e-12` against a row magnitude of `2e-12` — a full unit
        // violation — but `tol * max(scale, 1)` is `1e-8`, so the starting point
        // read as a witness and a correct infeasibility verdict was withdrawn.
        // `tol * scale` is `2e-20` and the verdict stands.
        if is_significant(viol, scale, tol) {
            return None;
        }
        max_violation = max_violation.max(viol);
    }

    Some(FeasibleWitness { x, max_violation })
}

#[cfg(test)]
mod tests {
    use super::*;
    use pounce_nlp::tnlp::{IndexStyle, IpoptCq, IpoptData, NlpInfo, Solution, SparsityRequest};

    const LO_INF: Number = -1e19;
    const UP_INF: Number = 1e19;

    /// `min (x-x0)^2` over `x ∈ [lo, hi]` subject to one row `a·x ∈ [g_l, g_u]`.
    struct OneRow {
        x0: Vec<Number>,
        lo: Vec<Number>,
        hi: Vec<Number>,
        a: Vec<Number>,
        g_l: Number,
        g_u: Number,
        eval_g_ok: bool,
        have_x0: bool,
    }

    impl OneRow {
        fn new(
            x0: Vec<Number>,
            lo: Vec<Number>,
            hi: Vec<Number>,
            a: Vec<Number>,
            g_l: Number,
            g_u: Number,
        ) -> Self {
            Self {
                x0,
                lo,
                hi,
                a,
                g_l,
                g_u,
                eval_g_ok: true,
                have_x0: true,
            }
        }
    }

    impl TNLP for OneRow {
        fn get_nlp_info(&mut self) -> Option<NlpInfo> {
            Some(NlpInfo {
                n: self.x0.len() as i32,
                m: 1,
                nnz_jac_g: self.a.len() as i32,
                nnz_h_lag: 0,
                index_style: IndexStyle::C,
            })
        }
        fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
            b.x_l.copy_from_slice(&self.lo);
            b.x_u.copy_from_slice(&self.hi);
            b.g_l[0] = self.g_l;
            b.g_u[0] = self.g_u;
            true
        }
        fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
            if !self.have_x0 {
                return false;
            }
            sp.x.copy_from_slice(&self.x0);
            true
        }
        fn eval_f(&mut self, _x: &[Number], _new_x: bool) -> Option<Number> {
            Some(0.0)
        }
        fn eval_grad_f(&mut self, _x: &[Number], _new_x: bool, grad_f: &mut [Number]) -> bool {
            grad_f.fill(0.0);
            true
        }
        fn eval_g(&mut self, x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
            if !self.eval_g_ok {
                return false;
            }
            g[0] = self.a.iter().zip(x).map(|(a, v)| a * v).sum();
            true
        }
        fn eval_jac_g(
            &mut self,
            _x: Option<&[Number]>,
            _new_x: bool,
            _mode: SparsityRequest<'_>,
        ) -> bool {
            true
        }
        fn finalize_solution(&mut self, _s: Solution<'_>, _d: &IpoptData, _c: &IpoptCq) {}
    }

    fn refute(t: OneRow) -> Option<FeasibleWitness> {
        let rc: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(t));
        starting_point_refutes_infeasibility(&rc, LO_INF, UP_INF, 1e-8)
    }

    /// gh #379 seed 294 in miniature: the modeller's own starting point sits
    /// exactly on a row whose coefficients are `±1e30`.
    #[test]
    fn extreme_scale_starting_point_refutes() {
        let w = refute(OneRow::new(
            vec![5e5, 5e5],
            vec![0.0, 0.0],
            vec![1e6, 1e6],
            vec![-1e30, 1e30],
            -1e-6,
            UP_INF,
        ))
        .expect("x0 satisfies the row exactly — the verdict must be withdrawn");
        assert_eq!(w.x, vec![5e5, 5e5]);
        assert_eq!(w.max_violation, 0.0);
    }

    /// The whole point of the gate: a model with no feasible point cannot
    /// produce a witness, so a correct verdict survives.
    #[test]
    fn genuinely_infeasible_model_is_not_refuted() {
        // x ∈ [0, 0.6] with the row x >= 0.7 — gh #372's reproducer.
        assert!(
            refute(OneRow::new(
                vec![0.3],
                vec![0.0],
                vec![0.6],
                vec![1.0],
                0.7,
                UP_INF
            ))
            .is_none()
        );
    }

    /// The verdict must not depend on how the model is written. `x >= 2` over
    /// `x ∈ [0, 1]` is empty at every row scaling, so no scaling may produce a
    /// witness.
    ///
    /// This is the case that rules out the clamped `is_negligible` form: at
    /// `s = 1e-12` the violation and the row magnitude are both `2e-12`, which
    /// `tol * max(scale, 1)` calls negligible and `tol * scale` does not. The
    /// scale-invariance harness caught it on three models at once.
    #[test]
    fn a_down_scaled_infeasible_row_is_never_refuted() {
        for k in -12..=12 {
            let s = 10f64.powi(k);
            assert!(
                refute(OneRow::new(
                    vec![0.5],
                    vec![0.0],
                    vec![1.0],
                    vec![s],
                    2.0 * s,
                    UP_INF
                ))
                .is_none(),
                "`x >= 2` over `x ∈ [0, 1]` is empty at row scaling 10^{k} too"
            );
        }
    }

    /// The feasible twin of the sweep above: a witness stays a witness at every
    /// row scaling.
    #[test]
    fn a_scaled_feasible_row_is_refuted_at_every_scale() {
        for k in -12..=12 {
            let s = 10f64.powi(k);
            assert!(
                refute(OneRow::new(
                    vec![0.5],
                    vec![0.0],
                    vec![1.0],
                    vec![s],
                    0.25 * s,
                    UP_INF
                ))
                .is_some(),
                "`x >= 0.25` at `x = 0.5` holds at row scaling 10^{k} too"
            );
        }
    }

    /// A starting point outside the box is clamped, and the clamped point is a
    /// witness on its own terms.
    #[test]
    fn starting_point_is_clamped_into_the_box() {
        let w = refute(OneRow::new(
            vec![5.0],
            vec![0.0],
            vec![1.0],
            vec![1.0],
            0.0,
            2.0,
        ))
        .expect("clamped to x = 1, which satisfies 0 <= x <= 2");
        assert_eq!(w.x, vec![1.0]);
    }

    /// Clamping must not manufacture a witness for a row the clamped point
    /// violates.
    #[test]
    fn clamping_does_not_manufacture_a_witness() {
        assert!(
            refute(OneRow::new(
                vec![5.0],
                vec![0.0],
                vec![1.0],
                vec![1.0],
                3.0,
                UP_INF
            ))
            .is_none(),
            "clamped to x = 1, which violates x >= 3"
        );
    }

    /// Failing to evaluate declines to refute — it never asserts infeasibility.
    #[test]
    fn unevaluable_model_declines_to_refute() {
        let mut t = OneRow::new(vec![0.5], vec![0.0], vec![1.0], vec![1.0], 0.0, 2.0);
        t.eval_g_ok = false;
        assert!(refute(t).is_none());

        let mut t = OneRow::new(vec![0.5], vec![0.0], vec![1.0], vec![1.0], 0.0, 2.0);
        t.have_x0 = false;
        assert!(refute(t).is_none());

        let t = OneRow::new(vec![Number::NAN], vec![0.0], vec![1.0], vec![1.0], 0.0, 2.0);
        assert!(refute(t).is_none());
    }

    /// An absent bound must not set the row's magnitude from the ~1e19
    /// sentinel — that would make every row look satisfied.
    #[test]
    fn infinite_bound_sentinel_does_not_inflate_the_scale() {
        // Row value 1.0 against `g <= 0.5`: a real violation of 0.5, on a row
        // whose only finite bound is 0.5. The absent lower bound is the
        // sentinel and must contribute nothing.
        assert!(
            refute(OneRow::new(
                vec![1.0],
                vec![0.0],
                vec![2.0],
                vec![1.0],
                LO_INF,
                0.5
            ))
            .is_none()
        );
    }

    /// A crossed box is presolve's business, not something to refute from.
    #[test]
    fn crossed_box_declines_to_refute() {
        assert!(
            refute(OneRow::new(
                vec![0.5],
                vec![1.0],
                vec![0.0],
                vec![1.0],
                LO_INF,
                UP_INF
            ))
            .is_none()
        );
    }
}
