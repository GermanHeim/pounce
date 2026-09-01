//! gh#884: a biactive complementarity pair makes the exact-product
//! lowering's multipliers diverge, and the run fails next to the primal
//! solution.
//!
//! Numbers in this file were measured on `87402274`.
//!
//! **gh#884 is open, and these are invariants rather than a description
//! of it.** Each one is written so that it stays correct after a fix: it
//! fails when the solver gets *worse*, never when it gets better. In
//! particular none of them asserts that the current failure persists —
//! a test that pinned the bug would go red on a genuine fix, which is
//! backwards, and would train the next reader to expect red here.
//!
//! The measurements — the runaway multipliers, the `mu` trace, and three
//! approaches ruled out — live in
//! `dev-notes/mpcc-biactive-dual-divergence.md`. That is where history
//! belongs; this file holds contracts. If you are fixing gh#884, read the
//! note first: the obvious remedy is measured and rejected there.
//!
//! The fixture is `qpec_small` from `benchmarks/mpcc/` under the exact
//! product (`ncp_eq` / `prod_eq`) lowering:
//!
//! ```text
//! min (x-1)^2 + (y1-1)^2 + y2^2
//! s.t. 0 <= x <= 2
//!      0 <= y1  _|_  (2 y1 - 1 - x) >= 0
//!      0 <= y2  _|_  (2 y2 - 1 + x) >= 0
//! ```
//!
//! with `f* = 0` at `(1, 1, 0)`, where pair 1 is strict and **pair 2 is
//! biactive** (`G2 = H2 = 0`). There `grad f` is exactly `0` and the
//! product row's gradient is exactly `(0, 0, 0)`, so `lambda = 0`
//! certifies stationarity with residual `0` **at that exact point** — so
//! the answer exists and is reachable.
//!
//! What happens instead: the solve reaches `(1.0002321, 1.0001161,
//! 2.67e-15)`, feasible to `2.2e-16` with `f = 6.73e-8`, and stops there.
//! That is *near* the optimum, not at it — `lambda = 0` does not certify
//! the returned iterate, where it leaves an objective-gradient residual of
//! `4.64e-4`. Meanwhile the multipliers on the *linearly dependent* rows
//! run away in near-cancelling pairs; rows 2 and 5 both restrict only
//! `y2`, and rows 1 and 4 are likewise parallel:
//!
//! | row | `|grad|_inf` | `lambda` |
//! |---|---:|---:|
//! | 1 (`H1 >= 0`) | `2.0` | `-2.089e10` |
//! | 4 (`G1·H1 = 0`) | `2.0` | `+2.089e10` |
//! | 2 (`G2 >= 0`) | `1.0` | `-7.283e11` |
//! | 5 (`G2·H2 = 0`) | `2.3e-4` | `+1.737e15` |
//!
//! `-7.283e11 + 1.737e15 · 2.3e-4` is the `3.253e11` residual it fails
//! on. That table is measurement, not an assertion — see the note.
//!
//! **The asymmetry between the two fixtures is the point.** `qpec_small`
//! failing is the *bug*, so nothing here asserts that it fails; what is
//! asserted is that the point it returns is the optimum, and that it
//! never claims a success it cannot back. `ralph1` failing is *correct* —
//! no sign-feasible multiplier exists at its origin — so that one is
//! asserted directly, and it is what catches a `perturb_always_cd`
//! default flip.
//!
//! | test | what it pins | red when |
//! |---|---|---|
//! | `qpec_small_returns_the_optimum_and_never_claims_a_false_success` | the returned point is feasible and at `f*`; any claimed success must hold in the model's own units | a verdict-only "fix" reports success at a `1e11` residual |
//! | `a_structurally_zero_hessian_entry_does_not_change_the_solve` | declaring an identically-zero Hessian entry is a no-op (and refutes gh#884's stated prerequisite hypothesis) | the declared sparsity pattern starts changing the answer |
//! | `dual_regularization_reaches_the_optimum_honestly` | an honest solve of this model exists at `9.96e-8` unscaled, so gh#884 is a POUNCE gap | that configuration stops reaching the answer |
//! | `ralph1_must_not_claim_success_where_no_multiplier_certifies_it` | a model with no sign-feasible multiplier must not report success, and never below `f*` | `perturb_always_cd` is turned on by default — measured: plain `Solve_Succeeded` at `-2.71e-5` |

use std::cell::RefCell;
use std::rc::Rc;

use pounce_algorithm::application::IpoptApplication;
use pounce_common::types::{Index, Number};
use pounce_nlp::return_codes::ApplicationReturnStatus;
use pounce_nlp::tnlp::{
    BoundsInfo, IndexStyle, IpoptCq, IpoptData, NlpInfo, Solution, SparsityRequest, StartingPoint,
    TNLP,
};

const INF: Number = 2e19;

/// What `finalize_solution` hands back, so a test can measure the
/// returned point for itself rather than trusting a reported number.
///
/// The multipliers are deliberately *not* carried here. The runaway
/// values in the module table above are a measurement of behaviour
/// gh#884 is open about, and asserting on them would pin the bug; the
/// table and `dev-notes/mpcc-biactive-dual-divergence.md` are their home.
#[derive(Clone, Default)]
struct Solved {
    x: Vec<Number>,
}

/// `qpec_small` under the exact-product lowering. Rows, in order:
/// `0: G1 = y1`, `1: H1 = 2y1 − 1 − x`, `2: G2 = y2`, `3: H2 = 2y2 − 1 + x`,
/// `4: G1·H1`, `5: G2·H2`, the last two equalities at `0`.
struct QpecSmallProdEq {
    captured: Rc<RefCell<Option<Solved>>>,
    /// The MPCC harness hands back a *dense* lower triangle (6 nonzeros)
    /// where the structurally correct pattern has 5 — `(2, 1)` is
    /// identically zero. gh#884 names that difference as the leading
    /// hypothesis for its two paths' different exits, so it is a switch.
    dense_hessian: bool,
}

impl QpecSmallProdEq {
    fn new(dense_hessian: bool) -> (Self, Rc<RefCell<Option<Solved>>>) {
        let captured = Rc::new(RefCell::new(None));
        (
            Self {
                captured: Rc::clone(&captured),
                dense_hessian,
            },
            captured,
        )
    }
}

/// Violation of the model's own rows and bounds at `x`, in the model's
/// units, computed independently of the solver.
fn true_violation(x: &[Number]) -> Number {
    let g = [x[1], 2.0 * x[1] - 1.0 - x[0], x[2], 2.0 * x[2] - 1.0 + x[0]];
    let mut v: Number = 0.0;
    for gi in &g {
        v = v.max((-gi).max(0.0));
    }
    v = v.max((g[0] * g[1]).abs());
    v = v.max((g[2] * g[3]).abs());
    v = v.max((-x[0]).max(0.0));
    v.max((x[0] - 2.0).max(0.0))
}

impl TNLP for QpecSmallProdEq {
    fn get_nlp_info(&mut self) -> Option<NlpInfo> {
        Some(NlpInfo {
            n: 3,
            m: 6,
            nnz_jac_g: 18,
            nnz_h_lag: if self.dense_hessian { 6 } else { 5 },
            index_style: IndexStyle::C,
        })
    }

    fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
        b.x_l[0] = 0.0;
        b.x_u[0] = 2.0;
        b.x_l[1] = -INF;
        b.x_u[1] = INF;
        b.x_l[2] = -INF;
        b.x_u[2] = INF;
        for i in 0..4 {
            b.g_l[i] = 0.0;
            b.g_u[i] = INF;
        }
        for i in 4..6 {
            b.g_l[i] = 0.0;
            b.g_u[i] = 0.0;
        }
        true
    }

    fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
        sp.x[0] = 0.0;
        sp.x[1] = 0.5;
        sp.x[2] = 0.5;
        true
    }

    fn eval_f(&mut self, x: &[Number], _new_x: bool) -> Option<Number> {
        Some((x[0] - 1.0).powi(2) + (x[1] - 1.0).powi(2) + x[2] * x[2])
    }

    fn eval_grad_f(&mut self, x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
        g[0] = 2.0 * (x[0] - 1.0);
        g[1] = 2.0 * (x[1] - 1.0);
        g[2] = 2.0 * x[2];
        true
    }

    fn eval_g(&mut self, x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
        let g1 = x[1];
        let h1 = 2.0 * x[1] - 1.0 - x[0];
        let g2 = x[2];
        let h2 = 2.0 * x[2] - 1.0 + x[0];
        g[0] = g1;
        g[1] = h1;
        g[2] = g2;
        g[3] = h2;
        g[4] = g1 * h1;
        g[5] = g2 * h2;
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
                let mut k = 0;
                for i in 0..6 {
                    for j in 0..3 {
                        irow[k] = i as Index;
                        jcol[k] = j as Index;
                        k += 1;
                    }
                }
                true
            }
            SparsityRequest::Values { values } => {
                let Some(x) = x else { return false };
                let rows = jac_rows(x);
                for (i, row) in rows.iter().enumerate() {
                    values[3 * i..3 * i + 3].copy_from_slice(row);
                }
                true
            }
        }
    }

    fn eval_h(
        &mut self,
        _x: Option<&[Number]>,
        _new_x: bool,
        obj_factor: Number,
        lambda: Option<&[Number]>,
        _new_lambda: bool,
        mode: SparsityRequest<'_>,
    ) -> bool {
        let dense = self.dense_hessian;
        match mode {
            SparsityRequest::Structure { irow, jcol } => {
                let pattern: &[(Index, Index)] = if dense {
                    &[(0, 0), (1, 0), (1, 1), (2, 0), (2, 1), (2, 2)]
                } else {
                    &[(0, 0), (1, 0), (1, 1), (2, 0), (2, 2)]
                };
                for (k, (i, j)) in pattern.iter().enumerate() {
                    irow[k] = *i;
                    jcol[k] = *j;
                }
                true
            }
            SparsityRequest::Values { values } => {
                let l4 = lambda.and_then(|l| l.get(4).copied()).unwrap_or(0.0);
                let l5 = lambda.and_then(|l| l.get(5).copied()).unwrap_or(0.0);
                // ∇²f = diag(2,2,2); row 4 adds ∂²/∂y1² = 4, ∂²/∂x∂y1 = −1;
                // row 5 adds ∂²/∂y2² = 4, ∂²/∂x∂y2 = +1. `(2,1)` is
                // identically zero, which is what makes it optional.
                values[0] = 2.0 * obj_factor;
                values[1] = -l4;
                values[2] = 2.0 * obj_factor + 4.0 * l4;
                values[3] = l5;
                if dense {
                    values[4] = 0.0;
                    values[5] = 2.0 * obj_factor + 4.0 * l5;
                } else {
                    values[4] = 2.0 * obj_factor + 4.0 * l5;
                }
                true
            }
        }
    }

    fn finalize_solution(&mut self, sol: Solution<'_>, _ip_data: &IpoptData, _ip_cq: &IpoptCq) {
        *self.captured.borrow_mut() = Some(Solved { x: sol.x.to_vec() });
    }
}

/// The six constraint gradients at `x`.
fn jac_rows(x: &[Number]) -> [[Number; 3]; 6] {
    [
        [0.0, 1.0, 0.0],
        [-1.0, 2.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 2.0],
        [-x[1], 4.0 * x[1] - 1.0 - x[0], 0.0],
        [x[2], 0.0, 4.0 * x[2] - 1.0 + x[0]],
    ]
}

/// `ralph1` under the `direct` lowering (`G·H <= 0`):
/// `min 2x − y  s.t.  x >= 0, G = y >= 0, H = y − x >= 0, G·H <= 0`,
/// with `f* = 0` at the origin.
///
/// The origin is M-stationary but **not** S-stationary, and NLP KKT is
/// S-stationarity, so no sign-feasible multiplier exists there and
/// failing is the correct outcome. gh#884 makes that an explicit
/// acceptance criterion: a fix that greens this has over-fired.
struct Ralph1Direct {
    captured: Rc<RefCell<Option<Solved>>>,
}

impl Ralph1Direct {
    fn new() -> (Self, Rc<RefCell<Option<Solved>>>) {
        let captured = Rc::new(RefCell::new(None));
        (
            Self {
                captured: Rc::clone(&captured),
            },
            captured,
        )
    }
}

impl TNLP for Ralph1Direct {
    fn get_nlp_info(&mut self) -> Option<NlpInfo> {
        Some(NlpInfo {
            n: 2,
            m: 3,
            nnz_jac_g: 6,
            nnz_h_lag: 3,
            index_style: IndexStyle::C,
        })
    }
    fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
        b.x_l[0] = 0.0;
        b.x_l[1] = -INF;
        b.x_u[0] = INF;
        b.x_u[1] = INF;
        b.g_l[0] = 0.0;
        b.g_u[0] = INF;
        b.g_l[1] = 0.0;
        b.g_u[1] = INF;
        // the `direct` lowering: `G·H <= 0`
        b.g_l[2] = -INF;
        b.g_u[2] = 0.0;
        true
    }
    fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
        sp.x[0] = 0.0;
        sp.x[1] = 0.0;
        true
    }
    fn eval_f(&mut self, x: &[Number], _n: bool) -> Option<Number> {
        Some(2.0 * x[0] - x[1])
    }
    fn eval_grad_f(&mut self, _x: &[Number], _n: bool, g: &mut [Number]) -> bool {
        g[0] = 2.0;
        g[1] = -1.0;
        true
    }
    fn eval_g(&mut self, x: &[Number], _n: bool, g: &mut [Number]) -> bool {
        g[0] = x[1];
        g[1] = x[1] - x[0];
        g[2] = x[1] * (x[1] - x[0]);
        true
    }
    fn eval_jac_g(&mut self, x: Option<&[Number]>, _n: bool, m: SparsityRequest<'_>) -> bool {
        match m {
            SparsityRequest::Structure { irow, jcol } => {
                let mut k = 0;
                for i in 0..3 {
                    for j in 0..2 {
                        irow[k] = i as Index;
                        jcol[k] = j as Index;
                        k += 1;
                    }
                }
                true
            }
            SparsityRequest::Values { values } => {
                let Some(x) = x else { return false };
                values[0] = 0.0;
                values[1] = 1.0;
                values[2] = -1.0;
                values[3] = 1.0;
                values[4] = -x[1];
                values[5] = 2.0 * x[1] - x[0];
                true
            }
        }
    }
    fn eval_h(
        &mut self,
        _x: Option<&[Number]>,
        _n: bool,
        _obj_factor: Number,
        lambda: Option<&[Number]>,
        _nl: bool,
        m: SparsityRequest<'_>,
    ) -> bool {
        match m {
            SparsityRequest::Structure { irow, jcol } => {
                irow[0] = 0;
                jcol[0] = 0;
                irow[1] = 1;
                jcol[1] = 0;
                irow[2] = 1;
                jcol[2] = 1;
                true
            }
            SparsityRequest::Values { values } => {
                let l2 = lambda.and_then(|l| l.get(2).copied()).unwrap_or(0.0);
                values[0] = 0.0;
                values[1] = -l2;
                values[2] = 2.0 * l2;
                true
            }
        }
    }
    fn finalize_solution(&mut self, sol: Solution<'_>, _d: &IpoptData, _c: &IpoptCq) {
        *self.captured.borrow_mut() = Some(Solved { x: sol.x.to_vec() });
    }
}

/// The `.nl`-path options gh#884 measures under.
fn app(always_cd: bool, max_iter: i32) -> IpoptApplication {
    let mut app = IpoptApplication::new();
    {
        let o = app.options_mut();
        let _ = o.set_string_value("sb", "yes", true, false);
        let _ = o.set_integer_value("print_level", 0, true, false);
        let _ = o.set_numeric_value("tol", 1e-8, true, false);
        let _ = o.set_numeric_value("bound_relax_factor", 0.0, true, false);
        let _ = o.set_string_value("honor_original_bounds", "yes", true, false);
        let _ = o.set_integer_value("max_iter", max_iter, true, false);
        if always_cd {
            let _ = o.set_string_value("perturb_always_cd", "yes", true, false);
        }
    }
    app.initialize().expect("initialize");
    app
}

/// Statuses POUNCE offers as "we solved it". A claim of success made
/// under any of these is a claim the user's tooling acts on — Pyomo maps
/// both into its solved family.
fn claims_success(status: ApplicationReturnStatus) -> bool {
    matches!(
        status,
        ApplicationReturnStatus::SolveSucceeded | ApplicationReturnStatus::SolvedToAcceptableLevel
    )
}

/// The invariant every solve on every model owes its caller: **if it
/// claims success, the claim has to be true in the model's own units.**
///
/// Vacuous while a model is failing, which is the point — it is a guard,
/// not a description. It becomes load-bearing the moment anything starts
/// reporting success on the model it is applied to, which is exactly when
/// someone needs catching.
fn a_claimed_success_must_be_real(
    what: &str,
    status: ApplicationReturnStatus,
    unscaled_kkt: Number,
    violation: Number,
) {
    if !claims_success(status) {
        return;
    }
    assert!(
        unscaled_kkt <= 1e-6,
        "{what}: reported {status:?} at an unscaled KKT error of {unscaled_kkt:.3e}. \
         A verdict that passes while the residual it stands for is orders larger \
         is the symptom-only fix gh#884 warns against.",
    );
    assert!(
        violation <= 1e-8,
        "{what}: reported {status:?} at a constraint violation of {violation:.3e}",
    );
}

/// `qpec_small` returns a near-optimal point, and must never claim a
/// false success.
///
/// Two halves, and the split is deliberate.
///
/// The **unconditional** half — the returned point is feasible and lands
/// within `1e-3` of `(1, 1, 0)` with `|f|` under `1e-6` — is true today
/// *and* after any fix, so it never needs revisiting. The tolerances are
/// what they are because the run stops *near* the optimum, not at it:
/// measured `(1.0002321, 1.0001161, 2.67e-15)` with `f = 6.73e-8` on
/// `87402274`. The interesting fact is that it gives up that close.
///
/// The **conditional** half is the guard. gh#884 is open, so today this
/// model exits `RestorationFailed` at an unscaled KKT error of `3.3e11`
/// and the guard is vacuous. That failure is deliberately *not* asserted:
/// a test that pins it would go red on a genuine fix, which is backwards.
/// What the guard catches is the tempting bad fix gh#884 names — relaxing
/// the exit verdict so the model reports success while the residual stays
/// at `1e11`. That fix passes a status assertion and fails this one.
#[test]
fn qpec_small_returns_the_optimum_and_never_claims_a_false_success() {
    let (tnlp, captured) = QpecSmallProdEq::new(false);
    let mut a = app(false, 300);
    let status = a.optimize_tnlp(Rc::new(RefCell::new(tnlp)));
    let sol = captured.borrow().clone().expect("finalize_solution ran");
    let s = a.statistics();
    let violation = true_violation(&sol.x);

    // Forward-compatible: true while gh#884 is open, and after it closes.
    // Deliberately loose in `x` and `f` — the run stops near the optimum,
    // not at it (see the doc comment); tightening these would pin the
    // distance gh#884 leaves on the table.
    assert!(
        violation <= 1e-12,
        "expected a feasible point, got violation {violation:.3e} at {:?}",
        sol.x,
    );
    for (i, want) in [1.0, 1.0, 0.0].iter().enumerate() {
        assert!(
            (sol.x[i] - want).abs() <= 1e-3,
            "x[{i}] = {:.6e} is not at the optimum {want}",
            sol.x[i],
        );
    }
    assert!(
        s.final_objective.abs() <= 1e-6,
        "objective {:.3e} is not at f* = 0",
        s.final_objective,
    );

    a_claimed_success_must_be_real(
        "qpec_small/prod_eq/origin",
        status,
        s.final_unscaled_kkt_error,
        violation,
    );
}

/// Declaring a structurally-zero Hessian entry must not change the solve.
///
/// A permanent contract in its own right: `(2, 1)` is identically zero
/// here, so a caller that declares it — as the MPCC harness does, handing
/// back a dense lower triangle with 6 nonzeros where the structural
/// pattern has 5 — must get the same answer as one that does not.
///
/// It also settles gh#884's stated prerequisite. The issue records that
/// the same model exits differently through the harness's Python path
/// (`Error_In_Step_Computation`, 118 iters) than through `.nl`/CLI
/// (`Solved_To_Acceptable_Level`, 41), and names this sparsity difference
/// as the leading hypothesis, explicitly not established. It is
/// **refuted**: both patterns give bit-identical iterates, the same
/// iteration count and the same status. Whatever separates those two
/// paths is somewhere else.
#[test]
fn a_structurally_zero_hessian_entry_does_not_change_the_solve() {
    let mut seen = Vec::new();
    for dense in [false, true] {
        let (tnlp, captured) = QpecSmallProdEq::new(dense);
        let mut a = app(false, 300);
        let status = a.optimize_tnlp(Rc::new(RefCell::new(tnlp)));
        let sol = captured.borrow().clone().expect("finalize_solution ran");
        let s = a.statistics();
        seen.push((status, s.iteration_count, sol.x.clone()));
    }
    assert_eq!(
        seen[0].0, seen[1].0,
        "status differs between the 5- and 6-nonzero Hessian patterns",
    );
    assert_eq!(
        seen[0].1, seen[1].1,
        "iteration count differs between the 5- and 6-nonzero Hessian patterns",
    );
    assert_eq!(
        seen[0].2, seen[1].2,
        "the returned iterate differs between the 5- and 6-nonzero Hessian \
         patterns — bit-identical is the measured result",
    );
}

/// An honest solve of this model exists, so gh#884 is a POUNCE gap and
/// not a property of the reformulation.
///
/// With dual regularization on from the start, `qpec_small` converges to
/// an **unscaled** KKT error of `~1e-7` — in the model's own units, not
/// by normalising the residual away.
///
/// This is evidence that there is something to fix. It is **not** an
/// argument for turning that option on by default: see
/// `ralph1_must_not_claim_success_where_no_multiplier_certifies_it`, and
/// `dev-notes/mpcc-biactive-dual-divergence.md` for why engaging it only
/// on demand does not work either.
#[test]
fn dual_regularization_reaches_the_optimum_honestly() {
    let (tnlp, captured) = QpecSmallProdEq::new(false);
    let mut a = app(true, 300);
    let status = a.optimize_tnlp(Rc::new(RefCell::new(tnlp)));
    let sol = captured.borrow().clone().expect("finalize_solution ran");
    let s = a.statistics();

    assert!(
        claims_success(status),
        "dual regularization no longer reaches the answer on qpec_small: {status:?}",
    );
    assert!(
        s.final_unscaled_kkt_error <= 1e-6,
        "converged, but not in the model's own units: unscaled KKT {:.3e}",
        s.final_unscaled_kkt_error,
    );
    assert!(
        true_violation(&sol.x) <= 1e-9,
        "converged to an infeasible point: violation {:.3e}",
        true_violation(&sol.x),
    );
}

/// `ralph1`'s origin admits no sign-feasible multiplier, so **failing
/// there is correct** — and a reported success must never sit below `f*`.
///
/// This is gh#884's own acceptance criterion as a contract, and unlike
/// the qpec_small guard it is not vacuous: failing is the *right* answer
/// on this model, today and after any fix. NLP KKT is S-stationarity;
/// the origin is M-stationary but not S-stationary, and the best
/// achievable residual over sign-feasible multipliers there is `0.707`,
/// three orders above any tolerance in play.
///
/// **What this catches.** The obvious way to "fix" gh#884 is to turn
/// `perturb_always_cd` on by default — dual regularization is the one
/// thing measured to reach qpec_small's answer. Do that and this test
/// goes red, because it runs at *default* options: with regularization
/// always on, this model reports plain `Solve_Succeeded` at an objective
/// of `-2.71e-5`, below `f* = 0`, at a point the MPCC does not contain
/// (measured at `max_iter = 3000`; at `300` the cap hides it as
/// `Solved_To_Acceptable_Level` at `-3.81e-5`).
///
/// So the trap is guarded in the direction that stays correct: this test
/// fails when the solver gets *worse*, never when it gets better. It
/// deliberately does not assert that the bad outcome still happens under
/// the non-default option — that measurement is history, and history
/// belongs in `dev-notes/mpcc-biactive-dual-divergence.md`.
///
/// Mutation-checked by applying `a_claimed_success_must_be_real` and the
/// objective bound below to a `perturb_always_cd=yes` run of this same
/// fixture: both fire, on the objective bound, at `-2.71e-5`.
#[test]
fn ralph1_must_not_claim_success_where_no_multiplier_certifies_it() {
    // A generous cap, so a pass here is not the cap hiding an outcome.
    let (tnlp, captured) = Ralph1Direct::new();
    let mut a = app(false, 3000);
    let status = a.optimize_tnlp(Rc::new(RefCell::new(tnlp)));
    let sol = captured.borrow().clone().expect("finalize_solution ran");
    let s = a.statistics();

    assert!(
        !claims_success(status),
        "reported {status:?} at {:?} on a model whose only candidate point \
         admits no sign-feasible multiplier (best residual there is 0.707). \
         If this went red on a `perturb_always_cd` default flip, that is the \
         test working: see dev-notes/mpcc-biactive-dual-divergence.md.",
        sol.x,
    );

    // And whatever it reports, never a success below the true optimum.
    if claims_success(status) {
        assert!(
            s.final_objective >= -1e-9,
            "reported {status:?} at objective {:.6e}, below f* = 0 — a point \
             outside the MPCC",
            s.final_objective,
        );
    }
    a_claimed_success_must_be_real(
        "ralph1/direct/origin",
        status,
        s.final_unscaled_kkt_error,
        true_violation_ralph1(&sol.x),
    );
}

/// Violation of `ralph1`'s own rows and bounds at `x`, in the model's
/// units: `G = y >= 0`, `H = y − x >= 0`, `G·H <= 0`, `x >= 0`.
fn true_violation_ralph1(x: &[Number]) -> Number {
    let g = x[1];
    let h = x[1] - x[0];
    let mut v: Number = 0.0;
    v = v.max((-g).max(0.0));
    v = v.max((-h).max(0.0));
    v = v.max((g * h).max(0.0));
    v.max((-x[0]).max(0.0))
}
