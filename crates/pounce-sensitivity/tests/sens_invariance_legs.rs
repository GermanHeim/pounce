//! Three invariance legs for the sensitivity layer, over one fixture
//! carrying a genuine kink.
//!
//! Each leg exists because a shipped defect got through a corpus that
//! was uniform in exactly the dimension the defect lived in. The legs
//! sweep that dimension the way `sweep-fixtures.sh` sweeps the exact
//! and L-BFGS Hessians.
//!
//! 1. **Scaling.** The corrector mixed the algorithm's scaled frame
//!    with the model's own units, and the mix is invisible at unit
//!    scaling -- "which coincide only at unit scaling, which is every
//!    fixture it had" (`205bb67`). The same gap reaches *membership*
//!    rules: under `x̃ = d ⊙ x` the barrier diagonal carries `d^-2`, so
//!    a threshold compared against a bare `Sigma` calls the same bound
//!    a different kind of bound in different units.
//!    `variable_scaling_sensitivity.rs` already runs this leg for
//!    `classify_activity`, but its fixture has no weakly active bound,
//!    so the degeneracy surface is untested there.
//!
//! 2. **Perturbation magnitude.** gh#672 finding 4: an acceptance
//!    tolerance was absolute on a quantity that scales with the
//!    perturbation, so a step of `1e-10` cleared feasibility
//!    everywhere and the holding side's derivative read `-1` instead
//!    of `0`. The leg sweeps `delta` over eight orders on both sides.
//!    It runs over TWO fixtures, because the rule that engages a weak
//!    row branches on the classifier's verdict: the certified
//!    `WEAKLY_ACTIVE` kink below, and a coupled kink that lands in
//!    `AMBIGUOUS` ([`CoupledKinkTnlp`]). A rule exact in one class and
//!    length-based in the other is invisible to a corpus carrying only
//!    the first -- which is the same shape as every entry above.
//!
//! 3. **Fixed variable ahead of the kink.** gh#672 finding 1, the
//!    gh#450 hazard: full-x and var-x diverge from the first
//!    `make_parameter`-removed variable on, and reading one as the
//!    other returns a NEIGHBORING variable's answer -- plausible and
//!    wrong. The leg puts a fixed variable in front of the kink and
//!    requires every var-x answer to be unmoved by it.
//!
//! # What the legs compare
//!
//! Not `dx / delta`. The parametric step is affine in `delta`, not
//! linear: it carries a base-point term of order `mu`, because the
//! converged iterate sits that far off the exact solution and the step
//! corrects it (see [`the_step_is_affine_in_delta`], which pins the
//! size of that term). Dividing by `delta` therefore inflates it as
//! `delta` shrinks -- at `1e-10` it is the whole answer -- so the
//! invariant is the *slope*, taken as a difference quotient between
//! two perturbations on the same side of the kink. The constant
//! cancels and what remains is the one-sided derivative, which is what
//! every leg here asserts.

use std::cell::RefCell;
use std::rc::Rc;

use pounce_algorithm::application::IpoptApplication;
use pounce_common::types::{Index, Number};
use pounce_nlp::TNLP;
use pounce_nlp::return_codes::ApplicationReturnStatus;
use pounce_nlp::tnlp::{
    BoundsInfo, IndexStyle, IpoptCq, IpoptData, NlpInfo, ScalingRequest, Solution, SparsityRequest,
    StartingPoint,
};
use pounce_sensitivity::Solver;

/// Coupling between the pin and the kink variable. The one-sided
/// derivative on the leaving side is exactly this.
const A: Number = 1.10;
/// Where the interior variable sits. Far enough from either bound that
/// no membership rule should ever reach it.
const W_STAR: Number = 2.0;
/// The value the leading fixed variable is pinned to.
const FIXED_AT: Number = 1.5;
/// Back-solve budget for the directional decision. The fixture engages
/// at most one weak row, so this is generous.
const DEGENERACY_ITER: usize = 16;

/// ```text
/// min  0.5 k^2 - A p k + 0.5 (w - W_STAR)^2  [+ 0.5 (xf - 0.7)^2]
/// s.t. p = 0,   0 <= k <= 10,   0 <= w <= 10  [, xf == FIXED_AT]
/// ```
///
/// At `p = 0` the kink variable `k` sits at its lower bound with a
/// multiplier that vanishes with `mu`: a genuine kink. Moving the pin
/// up lets `k` follow at `A`; moving it down would drive `k` through
/// its bound, so `k` holds and the derivative is `0`. `w` is interior
/// and decoupled from the pin -- it must never be called weak and must
/// never move.
///
/// `leading_fixed` prepends `xf`, whose equal bounds make
/// `fixed_variable_treatment=make_parameter` (the default) remove its
/// column, so full-x and var-x diverge in front of everything
/// interesting.
struct KinkTnlp {
    /// Per-variable factors to report, or `None` to decline scaling.
    x_scaling: Option<Vec<Number>>,
    leading_fixed: bool,
}

impl KinkTnlp {
    fn new(x_scaling: Option<Vec<Number>>, leading_fixed: bool) -> Self {
        Self {
            x_scaling,
            leading_fixed,
        }
    }

    /// full-x offset of the logical block: 1 when a fixed variable
    /// leads, 0 otherwise.
    fn off(&self) -> usize {
        usize::from(self.leading_fixed)
    }
}

impl TNLP for KinkTnlp {
    fn get_nlp_info(&mut self) -> Option<NlpInfo> {
        Some(NlpInfo {
            n: (3 + self.off()) as Index,
            m: 1,
            nnz_jac_g: 1,
            nnz_h_lag: (3 + self.off()) as Index,
            index_style: IndexStyle::C,
        })
    }

    fn get_scaling_parameters(&mut self, req: ScalingRequest<'_>) -> bool {
        let Some(d) = self.x_scaling.as_ref() else {
            return false;
        };
        *req.obj_scaling = 1.0;
        *req.use_x_scaling = true;
        req.x_scaling.copy_from_slice(d);
        *req.use_g_scaling = false;
        true
    }

    fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
        let o = self.off();
        if self.leading_fixed {
            b.x_l[0] = FIXED_AT;
            b.x_u[0] = FIXED_AT;
        }
        b.x_l[o] = 0.0;
        b.x_u[o] = 10.0;
        b.x_l[o + 1] = 0.0;
        b.x_u[o + 1] = 10.0;
        b.x_l[o + 2] = -1.0e19;
        b.x_u[o + 2] = 1.0e19;
        b.g_l[0] = 0.0;
        b.g_u[0] = 0.0;
        true
    }

    fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
        let o = self.off();
        if self.leading_fixed {
            sp.x[0] = FIXED_AT;
        }
        sp.x[o] = 0.3;
        sp.x[o + 1] = 0.5;
        sp.x[o + 2] = 0.0;
        true
    }

    fn eval_f(&mut self, x: &[Number], _new_x: bool) -> Option<Number> {
        let o = self.off();
        let (k, w, p) = (x[o], x[o + 1], x[o + 2]);
        let mut f = 0.5 * k * k - A * p * k + 0.5 * (w - W_STAR) * (w - W_STAR);
        if self.leading_fixed {
            f += 0.5 * (x[0] - 0.7) * (x[0] - 0.7);
        }
        Some(f)
    }

    fn eval_grad_f(&mut self, x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
        let o = self.off();
        let (k, w, p) = (x[o], x[o + 1], x[o + 2]);
        if self.leading_fixed {
            g[0] = x[0] - 0.7;
        }
        g[o] = k - A * p;
        g[o + 1] = w - W_STAR;
        g[o + 2] = -A * k;
        true
    }

    fn eval_g(&mut self, x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
        g[0] = x[self.off() + 2];
        true
    }

    fn eval_jac_g(&mut self, _x: Option<&[Number]>, _nx: bool, mode: SparsityRequest<'_>) -> bool {
        match mode {
            SparsityRequest::Structure { irow, jcol } => {
                irow[0] = 0;
                jcol[0] = (self.off() + 2) as Index;
            }
            SparsityRequest::Values { values } => values[0] = 1.0,
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
        let o = self.off() as Index;
        match mode {
            SparsityRequest::Structure { irow, jcol } => {
                // lower triangle: (k,k), (w,w), (p,k) [, (xf,xf)]
                let mut rs: Vec<Index> = vec![o, o + 1, o + 2];
                let mut cs: Vec<Index> = vec![o, o + 1, o];
                if self.leading_fixed {
                    rs.push(0);
                    cs.push(0);
                }
                irow.copy_from_slice(&rs);
                jcol.copy_from_slice(&cs);
            }
            SparsityRequest::Values { values } => {
                values[0] = obj_factor;
                values[1] = obj_factor;
                values[2] = -obj_factor * A;
                if self.leading_fixed {
                    values[3] = obj_factor;
                }
            }
        }
        true
    }

    fn finalize_solution(&mut self, _s: Solution<'_>, _d: &IpoptData, _q: &IpoptCq) {}
}

/// Both arms of every leg run under `user-scaling`, so the ONLY
/// difference between them is whether the TNLP hands factors back --
/// the option, the objective factor and the row factors are identical.
/// `weakly_active_bounds` refuses a relaxed-bound solve, so the relax
/// factor is off.
fn solved(x_scaling: Option<Vec<Number>>, leading_fixed: bool) -> Solver {
    let mut app = IpoptApplication::new();
    app.options_mut()
        .set_integer_value("print_level", 0, true, false)
        .unwrap();
    app.options_mut()
        .set_string_value("sb", "yes", true, false)
        .unwrap();
    app.options_mut()
        .set_string_value("nlp_scaling_method", "user-scaling", true, false)
        .unwrap();
    app.options_mut()
        .set_numeric_value("tol", 1e-8, true, false)
        .unwrap();
    app.options_mut()
        .set_numeric_value("bound_relax_factor", 0.0, true, false)
        .unwrap();
    app.initialize().unwrap();

    let tnlp: Rc<RefCell<dyn TNLP>> =
        Rc::new(RefCell::new(KinkTnlp::new(x_scaling, leading_fixed)));
    let mut solver = Solver::new(app, tnlp);
    let status = solver.solve();
    assert!(
        matches!(
            status,
            ApplicationReturnStatus::SolveSucceeded
                | ApplicationReturnStatus::SolvedToAcceptableLevel
        ),
        "base solve failed: {status:?}",
    );
    solver
}

/// The weak set as `(var_row, lower)` pairs, sorted so the comparison
/// does not depend on the order the rows happen to be walked in.
fn weak_set(s: &Solver) -> Vec<(usize, bool)> {
    let mut v: Vec<(usize, bool)> = s
        .weakly_active_bounds()
        .expect("weak set")
        .iter()
        .map(|b| (b.var_row, b.lower))
        .collect();
    v.sort();
    v
}

/// `parametric_step_directional` over the single pin, at `delta`.
fn step(s: &Solver, delta: Number) -> Vec<Number> {
    let (d, _held, _work) = s
        .parametric_step_directional(&[0], &[delta], DEGENERACY_ITER)
        .unwrap_or_else(|e| panic!("directional step at delta={delta:e}: {e:?}"));
    d
}

/// The one-sided derivative on `sign`'s side, as the difference
/// quotient between two perturbations there. See the module docs: the
/// step is affine in `delta`, so this cancels the base-point constant
/// and `step(delta) / delta` would not.
fn derivative(s: &Solver, sign: Number) -> Vec<Number> {
    slope(s, sign * 1.0e-3, sign * 1.0e-6)
}

fn slope(s: &Solver, d_hi: Number, d_lo: Number) -> Vec<Number> {
    let hi = step(s, d_hi);
    let lo = step(s, d_lo);
    hi.iter()
        .zip(lo.iter())
        .map(|(&a, &b)| (a - b) / (d_hi - d_lo))
        .collect()
}

/// The exact one-sided derivative of this fixture, in var-x order
/// `[k, w, p]`. The pin moves `p` at unit rate; `k` follows at `A`
/// where that is feasible and holds at its bound where it is not; `w`
/// is decoupled.
fn exact_derivative(sign: Number) -> [Number; 3] {
    [if sign > 0.0 { A } else { 0.0 }, 0.0, 1.0]
}

#[track_caller]
fn assert_close(what: &str, got: &[Number], want: &[Number], tol: Number) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    for (k, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
        let err = (g - w).abs() / w.abs().max(1.0);
        assert!(
            err < tol,
            "{what}[{k}]: got {g:e}, want {w:e}, rel err {err:e} not < {tol:e}"
        );
    }
}

// ---------------------------------------------------------------
// Preconditions -- without these every leg below is vacuous
// ---------------------------------------------------------------

/// The fixture has to actually carry a kink. Asserted once and named
/// so a failure here reads as "the fixture stopped being degenerate"
/// rather than as a defect in a leg.
#[test]
fn the_fixture_carries_a_kink_and_an_untouchable_interior_variable() {
    let s = solved(None, false);
    let x = s.converged().expect("converged").x.clone();
    assert!(x[0].abs() < 1e-3, "k sits on its lower bound, got {}", x[0]);
    assert!(
        (x[1] - W_STAR).abs() < 1e-6,
        "w sits at W_STAR, got {}",
        x[1]
    );

    let weak = weak_set(&s);
    assert!(
        weak.contains(&(0, true)),
        "k's lower bound is the kink and must be weak: {weak:?}"
    );
    assert!(
        !weak.iter().any(|&(r, _)| r == 1),
        "w is {W_STAR} from either bound and must never be weak: {weak:?}"
    );

    // The two sides must genuinely disagree, or the magnitude leg
    // would pass on any linear map.
    assert_close(
        "derivative up",
        &derivative(&s, 1.0),
        &exact_derivative(1.0),
        1e-6,
    );
    assert_close(
        "derivative down",
        &derivative(&s, -1.0),
        &exact_derivative(-1.0),
        1e-6,
    );
}

/// The step is `J·delta + c`, not `J·delta`. `c` is the correction of
/// the base point's own barrier displacement -- order `mu`, the same
/// vector at every `delta` -- which is why the legs compare slopes and
/// not ratios. Pinned here so that if `c` ever stops being negligible,
/// or stops being constant, this test says so rather than a leg
/// failing for a reason its name does not describe.
#[test]
fn the_step_is_affine_in_delta() {
    let s = solved(None, false);
    let j = exact_derivative(1.0);

    let mut constants = Vec::new();
    for &delta in &[1.0e-2, 1.0e-4, 1.0e-7, 1.0e-10] {
        let d = step(s_ref(&s), delta);
        let c: Vec<Number> = d
            .iter()
            .zip(j.iter())
            .map(|(&v, &jj)| v - jj * delta)
            .collect();
        for (i, &ci) in c.iter().enumerate() {
            assert!(
                ci.abs() < 1e-6,
                "the base-point term must stay of order mu: c[{i}] = {ci:e} at delta={delta:e}"
            );
        }
        constants.push(c);
    }
    let first = &constants[0];
    for (n, c) in constants.iter().enumerate().skip(1) {
        assert_close(&format!("base-point term at magnitude {n}"), c, first, 1e-6);
    }
}

/// Borrow helper: keeps the loop above readable without cloning the
/// solver.
fn s_ref(s: &Solver) -> &Solver {
    s
}

// ---------------------------------------------------------------
// Leg 1 -- scaling
// ---------------------------------------------------------------

/// Factors spanning five orders, mixed above and below one, with the
/// kink variable's factor well away from 1: under `x̃ = d ⊙ x` the
/// barrier diagonal carries `d^-2`, so a rule comparing a bare `Sigma`
/// against a fixed band sees this kink as a different kind of bound
/// than the unscaled arm does.
const D_PLAIN: [Number; 3] = [1.0e-2, 5.0, 1.0e3];
const D_FIXED: [Number; 4] = [2.0, 1.0e-2, 5.0, 1.0e3];

#[test]
fn leg_scaling_the_weak_set_is_unmoved_by_the_change_of_variables() {
    let plain = solved(None, false);
    let scaled = solved(Some(D_PLAIN.to_vec()), false);

    let want = weak_set(&plain);
    assert!(
        !want.is_empty(),
        "precondition: the unscaled arm must find the kink"
    );
    assert_eq!(
        weak_set(&scaled),
        want,
        "membership is a fact about the bound, not about the units the \
         model is written in: Sigma carries d^-2, so a threshold on a \
         bare Sigma moves here"
    );
}

#[test]
fn leg_scaling_the_directional_derivative_is_unmoved_by_the_change_of_variables() {
    let plain = solved(None, false);
    let scaled = solved(Some(D_PLAIN.to_vec()), false);

    for sign in [1.0, -1.0] {
        let exact = exact_derivative(sign);
        // Both arms are checked against the exact derivative as well,
        // so a shared error cannot pass the pair off as agreement.
        let want = derivative(&plain, sign);
        let got = derivative(&scaled, sign);
        assert_close(
            &format!("plain derivative (sign {sign})"),
            &want,
            &exact,
            1e-6,
        );
        assert_close(
            &format!("scaled derivative (sign {sign})"),
            &got,
            &exact,
            1e-6,
        );
        assert_close(
            &format!("derivative parity (sign {sign})"),
            &got,
            &want,
            1e-6,
        );
    }
}

// ---------------------------------------------------------------
// Leg 2 -- perturbation magnitude
// ---------------------------------------------------------------

/// Eight orders. An absolute tolerance anywhere on the path shows up
/// as one end of this range disagreeing with the other.
const DELTAS: [Number; 4] = [1.0e-2, 1.0e-4, 1.0e-7, 1.0e-10];

fn magnitude_sweep(s: &Solver, what: &str) {
    for sign in [1.0, -1.0] {
        let exact = exact_derivative(sign);
        for w in DELTAS.windows(2) {
            let (hi, lo) = (sign * w[0], sign * w[1]);
            assert_close(
                &format!("{what}: slope over [{lo:e}, {hi:e}]"),
                &slope(s, hi, lo),
                &exact,
                1e-6,
            );
        }
        // and the full span, so a defect that only shows between
        // distant magnitudes is not stepped over
        let (hi, lo) = (sign * DELTAS[0], sign * DELTAS[DELTAS.len() - 1]);
        assert_close(
            &format!("{what}: slope over the full span [{lo:e}, {hi:e}]"),
            &slope(s, hi, lo),
            &exact,
            1e-6,
        );
    }
}

#[test]
fn leg_magnitude_the_directional_derivative_does_not_depend_on_the_step_size() {
    magnitude_sweep(&solved(None, false), "plain");
}

#[test]
fn leg_magnitude_holds_under_the_change_of_variables_too() {
    // The legs compose: gh#672 finding 4 was an absolute tolerance on
    // a perturbation-scaled quantity, and a scaled frame moves the
    // scale such a tolerance is implicitly calibrated against.
    magnitude_sweep(&solved(Some(D_PLAIN.to_vec()), false), "scaled");
}

// ---------------------------------------------------------------
// Leg 3 -- a fixed variable ahead of the kink
// ---------------------------------------------------------------

#[test]
fn leg_fixed_the_index_spaces_actually_diverge() {
    // Without this the leg proves nothing: `make_parameter` has to
    // have removed the leading column, so that full-x is one longer
    // than var-x and every var-x row of interest is shifted.
    let plain = solved(None, false);
    let fixed = solved(None, true);

    assert_eq!(plain.n_full_x().expect("n_full_x"), 3);
    assert_eq!(fixed.n_full_x().expect("n_full_x"), 4);
    assert_eq!(
        plain.x_primal_rows(&[0, 1, 2]).expect("plain rows"),
        vec![Some(0), Some(1), Some(2)],
        "with nothing removed the two spaces coincide"
    );
    assert_eq!(
        fixed.x_primal_rows(&[0, 1, 2, 3]).expect("fixed rows"),
        vec![None, Some(0), Some(1), Some(2)],
        "the fixed variable has no row at all, and the kink's full-x \
         index 1 is var-x row 0: reading the one as the other lands on \
         the NEIGHBORING variable"
    );
}

#[test]
fn leg_fixed_the_weak_set_is_unmoved_by_a_fixed_variable_ahead_of_the_kink() {
    let plain = solved(None, false);
    let fixed = solved(None, true);

    let want = weak_set(&plain);
    assert!(
        !want.is_empty(),
        "precondition: the plain arm must find the kink"
    );
    assert_eq!(
        weak_set(&fixed),
        want,
        "the weak set is var-x indexed, so removing a column ahead of \
         the kink must not move it (gh#450, gh#672 finding 1)"
    );
}

#[test]
fn leg_fixed_the_directional_derivative_is_unmoved_by_a_fixed_variable_ahead_of_the_kink() {
    let plain = solved(None, false);
    let fixed = solved(None, true);

    for sign in [1.0, -1.0] {
        let exact = exact_derivative(sign);
        let want = derivative(&plain, sign);
        let got = derivative(&fixed, sign);
        assert_close(
            &format!("plain derivative (sign {sign})"),
            &want,
            &exact,
            1e-6,
        );
        assert_close(
            &format!("fixed derivative (sign {sign})"),
            &got,
            &exact,
            1e-6,
        );
        assert_close(
            &format!("derivative parity (sign {sign})"),
            &got,
            &want,
            1e-6,
        );
    }
}

// ---------------------------------------------------------------
// The corners -- where a defect surviving each leg alone shows up
// ---------------------------------------------------------------

#[test]
fn the_legs_compose_at_the_fixed_and_scaled_corner() {
    let plain = solved(None, false);
    let both = solved(Some(D_FIXED.to_vec()), true);

    assert_eq!(
        weak_set(&both),
        weak_set(&plain),
        "weak set at the fixed-and-scaled corner"
    );
    for sign in [1.0, -1.0] {
        assert_close(
            &format!("fixed+scaled derivative (sign {sign})"),
            &derivative(&both, sign),
            &exact_derivative(sign),
            1e-6,
        );
    }
    magnitude_sweep(&both, "fixed+scaled");
}

// ---------------------------------------------------------------
// Leg 2, second fixture: the magnitude sweep over an AMBIGUOUS kink
// ---------------------------------------------------------------
//
// The engagement rule that decides a weak row branches on the
// classifier's verdict, so the magnitude leg has to run in BOTH
// classes. The fixture above is certified `WEAKLY_ACTIVE`; a rule
// that treats the certified class exactly and the ambiguous class by
// a length is invisible to it.
//
// A genuine kink lands in the ambiguous class whenever it is coupled.
// `classify` divides `sigma` by the Hessian's DIAGONAL `H_ii`, but the
// multiplier at a kink is generated by the curvature *reduced* along
// that coordinate. Eliminating a free partner `y` from
// `[[h, c], [c, m]]` leaves
//
//     reduced = h - c^2/m,   sigma = reduced,   ratio = reduced / h
//
// so the ratio is 1 only when the coordinate is DECOUPLED. Drive
// `c^2/(h*m)` toward 1 and the ratio falls below the band edge at
// `1e-1`: a genuine kink, classified `AMBIGUOUS`. On a collocation
// model -- the kind that motivated the degeneracy work -- strong
// coupling between neighbouring coordinates is the normal case, not a
// corner.

/// Reduced curvature along `k` in [`CoupledKinkTnlp`]. Chosen so the
/// classifier's ratio is `1e-2`, a decade inside the ambiguous class.
const RHO: Number = 1.0e-2;
/// Diagonal curvature of both coordinates in [`CoupledKinkTnlp`].
const H_DIAG: Number = 1.0;
const M_DIAG: Number = 1.0;

/// Cross term giving reduced curvature [`RHO`]: `h - c^2/m = rho`.
fn cross() -> Number {
    (M_DIAG * (H_DIAG - RHO)).sqrt()
}

/// ```text
/// min  0.5*h*k^2 + c*k*y + 0.5*m*y^2 - A*p*k
/// s.t. p = 0,  0 <= k <= 10,  y free
/// ```
///
/// At `p = 0` the reduced gradient at `k = 0` is zero and the
/// multiplier vanishes with `mu`: a kink by construction, exactly as
/// in [`KinkTnlp`], but coupled. Moving the pin up lets `k` follow at
/// `A/RHO`; moving it down would drive `k` through its bound, so `k`
/// holds and the derivative is `0` -- at EVERY step size.
struct CoupledKinkTnlp;

impl TNLP for CoupledKinkTnlp {
    fn get_nlp_info(&mut self) -> Option<NlpInfo> {
        Some(NlpInfo {
            n: 3,
            m: 1,
            nnz_jac_g: 1,
            nnz_h_lag: 4,
            index_style: IndexStyle::C,
        })
    }

    fn get_scaling_parameters(&mut self, _req: ScalingRequest<'_>) -> bool {
        false
    }

    fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
        b.x_l[0] = 0.0;
        b.x_u[0] = 10.0;
        b.x_l[1] = -1.0e19;
        b.x_u[1] = 1.0e19;
        b.x_l[2] = -1.0e19;
        b.x_u[2] = 1.0e19;
        b.g_l[0] = 0.0;
        b.g_u[0] = 0.0;
        true
    }

    fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
        sp.x[0] = 0.3;
        sp.x[1] = 0.0;
        sp.x[2] = 0.0;
        true
    }

    fn eval_f(&mut self, x: &[Number], _new_x: bool) -> Option<Number> {
        let (k, y, p) = (x[0], x[1], x[2]);
        Some(0.5 * H_DIAG * k * k + cross() * k * y + 0.5 * M_DIAG * y * y - A * p * k)
    }

    fn eval_grad_f(&mut self, x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
        let (k, y, p) = (x[0], x[1], x[2]);
        g[0] = H_DIAG * k + cross() * y - A * p;
        g[1] = cross() * k + M_DIAG * y;
        g[2] = -A * k;
        true
    }

    fn eval_g(&mut self, x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
        g[0] = x[2];
        true
    }

    fn eval_jac_g(&mut self, _x: Option<&[Number]>, _nx: bool, mode: SparsityRequest<'_>) -> bool {
        match mode {
            SparsityRequest::Structure { irow, jcol } => {
                irow[0] = 0;
                jcol[0] = 2;
            }
            SparsityRequest::Values { values } => values[0] = 1.0,
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
        match mode {
            SparsityRequest::Structure { irow, jcol } => {
                // lower triangle: (k,k), (y,k), (y,y), (p,k)
                irow.copy_from_slice(&[0 as Index, 1, 1, 2]);
                jcol.copy_from_slice(&[0 as Index, 0, 1, 0]);
            }
            SparsityRequest::Values { values } => {
                values[0] = obj_factor * H_DIAG;
                values[1] = obj_factor * cross();
                values[2] = obj_factor * M_DIAG;
                values[3] = -obj_factor * A;
            }
        }
        true
    }

    fn finalize_solution(&mut self, _s: Solution<'_>, _d: &IpoptData, _q: &IpoptCq) {}
}

fn solved_coupled() -> Solver {
    let mut app = IpoptApplication::new();
    app.options_mut()
        .set_integer_value("print_level", 0, true, false)
        .unwrap();
    app.options_mut()
        .set_string_value("sb", "yes", true, false)
        .unwrap();
    app.options_mut()
        .set_numeric_value("tol", 1e-8, true, false)
        .unwrap();
    app.options_mut()
        .set_numeric_value("bound_relax_factor", 0.0, true, false)
        .unwrap();
    app.initialize().unwrap();

    let tnlp: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(CoupledKinkTnlp));
    let mut solver = Solver::new(app, tnlp);
    let status = solver.solve();
    assert!(
        matches!(
            status,
            ApplicationReturnStatus::SolveSucceeded
                | ApplicationReturnStatus::SolvedToAcceptableLevel
        ),
        "coupled base solve failed: {status:?}",
    );
    solver
}

/// `k` follows the pin at `A/RHO` on the leaving side and holds at `0`
/// on the other; `y` follows `k` at `-c/m`; the pin moves at `1`.
fn exact_coupled(sign: Number) -> [Number; 3] {
    if sign > 0.0 {
        let dk = A / RHO;
        [dk, -(cross() / M_DIAG) * dk, 1.0]
    } else {
        [0.0, 0.0, 1.0]
    }
}

/// The precondition the leg rests on, asserted rather than assumed:
/// the coupled fixture's kink really is a kink, really is weak, and
/// really is in the AMBIGUOUS class rather than the certified one. If
/// a classifier change ever moves it, this fails rather than letting
/// the leg pass vacuously against the wrong branch.
#[test]
fn the_coupled_fixture_carries_an_ambiguous_kink() {
    use pounce_sensitivity::activity::{AMBIGUOUS, WEAKLY_ACTIVE};

    let s = solved_coupled();
    let report = s.classify_activity().expect("activity report");

    assert_eq!(
        report.var_status[0], AMBIGUOUS,
        "the coupled kink must land in the AMBIGUOUS class (got {}, WEAKLY_ACTIVE is {}); \
         the ratio is {:e}",
        report.var_status[0], WEAKLY_ACTIVE, report.var_ratio[0]
    );
    // The ratio is `reduced/diagonal` up to the barrier's own finite
    // `mu`; the point is the decade it sits in, not the last digit.
    assert!(
        (report.var_ratio[0] - RHO).abs() < 1e-3 * RHO,
        "the ratio is reduced/diagonal = {RHO:e}, got {:e}",
        report.var_ratio[0]
    );
    assert!(
        weak_set(&s).contains(&(0, true)),
        "the coupled kink's lower bound must still be weak: {:?}",
        weak_set(&s)
    );
    // The two sides must genuinely disagree, or the leg would pass on
    // any linear map.
    assert_close(
        "coupled derivative up",
        &slope(&s, 1.0e-3, 1.0e-6),
        &exact_coupled(1.0),
        1e-6,
    );
}

/// Leg 2 over the ambiguous class. The holding side's derivative is
/// `0` at every step size; a rule that engages the row only once the
/// step exceeds a fixed base-point length reads the LEAVING side's
/// answer below that length instead, which is a first-order error and
/// a step straight through the bound.
///
/// This is gh#672 finding 4's shape: a length compared against a
/// quantity that scales with the perturbation. The length being
/// *measured* rather than *chosen* does not change that -- it is still
/// fixed while the step shrinks.
#[test]
fn leg_magnitude_an_ambiguous_kink_decides_the_same_way_at_every_step_size() {
    let s = solved_coupled();
    for sign in [1.0, -1.0] {
        let exact = exact_coupled(sign);
        for w in DELTAS.windows(2) {
            let (hi, lo) = (sign * w[0], sign * w[1]);
            assert_close(
                &format!("ambiguous kink: slope over [{lo:e}, {hi:e}]"),
                &slope(&s, hi, lo),
                &exact,
                1e-6,
            );
        }
        let (hi, lo) = (sign * DELTAS[0], sign * DELTAS[DELTAS.len() - 1]);
        assert_close(
            &format!("ambiguous kink: slope over the full span [{lo:e}, {hi:e}]"),
            &slope(&s, hi, lo),
            &exact,
            1e-6,
        );
    }
}
