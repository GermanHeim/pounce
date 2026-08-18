//! The third state: a hint POUNCE can neither prove nor disprove
//! (gh #588, phase Q6).
//!
//! `pounce-cli`'s corpus tests cover the two states an `.nl` model can
//! reach — a proof of constancy (reuse without being asked) and a proof of
//! variation (a warning, and the user's option refused). Neither is
//! reachable from a *callback* TNLP: the C interface, the Python bridge and
//! both GAMS links hand POUNCE numbers, so every proof comes back
//! `Unknown`, and what the user asserts is all there is.
//!
//! Upstream Ipopt behaves this way for every model. POUNCE behaves this way
//! only here, which makes this the one place the divergence *does not*
//! apply — and therefore the one worth an end-to-end test, because
//! "unproved" quietly turning into "refused" would break every front end
//! the note says the win is actually on.
//!
//! The model below is a QP written as callbacks: a quadratic objective over
//! a linear equality and a linear inequality. `∇²L` really is constant, so
//! the assertion is true, and the counter proves POUNCE stopped asking for
//! it — with the answer unchanged.

use pounce_algorithm::application::IpoptApplication;
use pounce_common::types::Number;
use pounce_nlp::return_codes::ApplicationReturnStatus;
use pounce_nlp::tnlp::{
    BoundsInfo, IndexStyle, IpoptCq, IpoptData, NlpInfo, Solution, SparsityRequest, StartingPoint,
    TNLP,
};
use std::cell::RefCell;
use std::rc::Rc;

/// `min x0² + 2x1² + x2²  s.t.  x0 + x1 + x2 == 3,  x0 − x1 <= 1`, as
/// callbacks and nothing else. POUNCE sees no algebra here, so
/// `derivative_proofs` keeps its declining default.
#[derive(Default)]
struct CallbackQp {
    h_calls: usize,
    jac_calls: usize,
    final_obj: Option<Number>,
    final_x: Option<Vec<Number>>,
}

impl TNLP for CallbackQp {
    fn get_nlp_info(&mut self) -> Option<NlpInfo> {
        Some(NlpInfo {
            n: 3,
            m: 2,
            nnz_jac_g: 5,
            nnz_h_lag: 3,
            index_style: IndexStyle::C,
        })
    }

    fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
        b.x_l.copy_from_slice(&[-10.0; 3]);
        b.x_u.copy_from_slice(&[10.0; 3]);
        b.g_l.copy_from_slice(&[3.0, -2.0e19]);
        b.g_u.copy_from_slice(&[3.0, 1.0]);
        true
    }

    fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
        sp.x.copy_from_slice(&[0.0, 0.0, 0.0]);
        true
    }

    fn eval_f(&mut self, x: &[Number], _: bool) -> Option<Number> {
        Some(x[0] * x[0] + 2.0 * x[1] * x[1] + x[2] * x[2])
    }

    fn eval_grad_f(&mut self, x: &[Number], _: bool, g: &mut [Number]) -> bool {
        g[0] = 2.0 * x[0];
        g[1] = 4.0 * x[1];
        g[2] = 2.0 * x[2];
        true
    }

    fn eval_g(&mut self, x: &[Number], _: bool, g: &mut [Number]) -> bool {
        g[0] = x[0] + x[1] + x[2];
        g[1] = x[0] - x[1];
        true
    }

    fn eval_jac_g(&mut self, _x: Option<&[Number]>, _: bool, mode: SparsityRequest<'_>) -> bool {
        match mode {
            SparsityRequest::Structure { irow, jcol } => {
                irow.copy_from_slice(&[0, 0, 0, 1, 1]);
                jcol.copy_from_slice(&[0, 1, 2, 0, 1]);
            }
            SparsityRequest::Values { values } => {
                self.jac_calls += 1;
                values.copy_from_slice(&[1.0, 1.0, 1.0, 1.0, -1.0]);
            }
        }
        true
    }

    fn eval_h(
        &mut self,
        _x: Option<&[Number]>,
        _: bool,
        obj_factor: Number,
        _lambda: Option<&[Number]>,
        _: bool,
        mode: SparsityRequest<'_>,
    ) -> bool {
        match mode {
            SparsityRequest::Structure { irow, jcol } => {
                irow.copy_from_slice(&[0, 1, 2]);
                jcol.copy_from_slice(&[0, 1, 2]);
            }
            SparsityRequest::Values { values } => {
                self.h_calls += 1;
                values.copy_from_slice(&[2.0 * obj_factor, 4.0 * obj_factor, 2.0 * obj_factor]);
            }
        }
        true
    }

    fn finalize_solution(&mut self, sol: Solution<'_>, _: &IpoptData, _: &IpoptCq) {
        self.final_obj = Some(sol.obj_value);
        self.final_x = Some(sol.x.to_vec());
    }
}

fn solve(hints: &[(&str, &str)]) -> (ApplicationReturnStatus, usize, usize, Number, Vec<Number>) {
    let tnlp = Rc::new(RefCell::new(CallbackQp::default()));
    let mut app = IpoptApplication::new();
    app.options_mut()
        .set_integer_value("print_level", 0, true, false)
        .expect("print_level");
    for (k, v) in hints {
        app.options_mut()
            .set_string_value(k, v, true, false)
            .unwrap_or_else(|_| panic!("set {k}"));
    }
    let status = app.optimize_tnlp(Rc::clone(&tnlp) as Rc<RefCell<dyn TNLP>>);
    let t = tnlp.borrow();
    (
        status,
        t.h_calls,
        t.jac_calls,
        t.final_obj.expect("objective"),
        t.final_x.clone().expect("x"),
    )
}

/// Baseline: no hint, no proof, so every iterate asks for `∇²L` and the
/// Jacobian again. This is the number the assertion below has to beat, and
/// pinning it here means the test cannot pass vacuously on a model that
/// happened to converge in one step.
#[test]
fn without_a_hint_a_callback_model_is_re_evaluated_every_iterate() {
    let (status, h, jac, _, _) = solve(&[]);
    assert_eq!(status, ApplicationReturnStatus::SolveSucceeded);
    assert!(h > 1, "expected repeated Hessian evaluations, got {h}");
    assert!(jac > 1, "expected repeated Jacobian evaluations, got {jac}");
}

/// The divergence does **not** apply where there is no proof. POUNCE has no
/// way to check this model's algebra, so the user's assertion stands and the
/// derivative is evaluated once — upstream's contract, and the case the
/// design note says the win is actually on.
#[test]
fn an_unprovable_hint_is_honoured_on_trust() {
    let (base_status, base_h, base_jac, base_obj, base_x) = solve(&[]);
    let (status, h, jac, obj, x) = solve(&[
        ("hessian_constant", "yes"),
        ("jac_c_constant", "yes"),
        ("jac_d_constant", "yes"),
    ]);
    assert_eq!(status, base_status);
    assert_eq!(h, 1, "`hessian_constant=yes` must stop the re-evaluation");
    // Two, not one: gradient-based scaling (the default
    // `nlp_scaling_method`) asks the user TNLP for `∇g` at the starting
    // point directly, before the NLP object and its caches exist. That
    // probe is outside anything Q6 touches; the *algorithm* then evaluates
    // once and reuses it for every iterate.
    assert_eq!(jac, 2, "`jac_*_constant=yes` must stop the re-evaluation");
    assert!(
        base_h > h && base_jac > jac,
        "hint made no difference: {base_h}/{base_jac} -> {h}/{jac}"
    );

    // The hint is true for this model, so honouring it must not move the
    // answer by a single bit — a reused derivative is the *same* matrix,
    // not an approximation of it.
    assert_eq!(obj.to_bits(), base_obj.to_bits(), "objective moved");
    for (a, b) in x.iter().zip(base_x.iter()) {
        assert_eq!(a.to_bits(), b.to_bits(), "solution moved");
    }
}

/// A hint set to its registered default asks for nothing and must not
/// engage the reuse — the same rule the refusal table follows for every
/// other option (`unimplemented_options::set_to_a_non_default`).
#[test]
fn writing_the_default_explicitly_asserts_nothing() {
    let (_, h, jac, _, _) = solve(&[("hessian_constant", "no"), ("jac_c_constant", "no")]);
    assert!(h > 1, "an explicit `no` must not enable reuse, got {h}");
    assert!(jac > 1, "an explicit `no` must not enable reuse, got {jac}");
}
