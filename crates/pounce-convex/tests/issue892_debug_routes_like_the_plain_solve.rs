//! gh #892 — attaching the debugger must not change the solve.
//!
//! The conic debug entry point used to be a *parallel implementation*: for
//! symmetric cones it built its own factorization and ran the core loop
//! directly, which never consulted `QpOptions::use_hsde`. Since that flag
//! defaults to `true`, attaching a debugger to a convex QCQP silently
//! substituted the direct IPM for the HSDE embedding — and with it dropped
//! the `σ` cost normalization and the equilibrate-and-verify guards that live
//! on the embedding's side of `solve_qp_core`. On the model below that turned
//! an `Optimal` agreeing with Clarabel to `1.3e-10` into a `NumericalFailure`.
//! Both entry points now enter the same body with an `Option<&mut DebugHook>`,
//! so the routing is the routing and the answer is the answer.
//!
//! **Measured at the CLI**, `solver_selection=socp` with and without
//! `--debug-script` containing only `continue`, over the reported model plus
//! 24 freshly drawn convex QCQPs at `n = 4, 5, 7, 9`: on `d32204e` all 25
//! disagree on status, objective or iteration count — 16 of them turning an
//! `Optimal` into a `Numerical failure` or a `Solved to acceptable level`.
//! After the fix all 25 agree on all three, and the plain column is unmoved.
//! The issue's own four-instance sample understated it; the corpus is
//! uniformly affected, which is what a substituted driver looks like.
//!
//! **Which branch each leg takes** (the rule `CLAUDE.md` states for the
//! sensitivity legs applies verbatim here: a leg is evidence only about the
//! branch its fixture reaches, and the *routing* is exactly what is under
//! test). `solve_qp_core` forks on `use_hsde`, so a leg that only ever runs
//! the default is blind to the direct driver's own divergence — which was
//! real, and is why `leg_qcqp_direct_driver` and `leg_qp_direct_driver` are
//! here and are not duplicates of their defaulted neighbours:
//!
//! | leg | entry point | driver | what the old code did instead |
//! | --- | --- | --- | --- |
//! | `leg_qcqp_hsde_default` | `solve_socp_ipm` | HSDE (`σ` reachable) | direct IPM, no `σ`, no verify |
//! | `leg_qcqp_direct_driver` | `solve_socp_ipm` | direct (`use_hsde=false`) | (same driver, but no box screen / `finite_or_failed`) |
//! | `leg_orthant_cone_verify` | `solve_socp_ipm`, all-`Nonneg` cones | HSDE + `verify_or_repair_optimum` | skipped the gh #414 verify entirely |
//! | `leg_qp_hsde_default` | `solve_qp_ipm` | HSDE | already agreed — the pin against a regression |
//! | `leg_qp_direct_driver` | `solve_qp_ipm` | direct + Ruiz | direct **without** equilibration |
//! | `leg_lp_crossover` | `solve_qp_ipm` | HSDE + LP crossover | skipped crossover |
//!
//! **Mutation table** — reintroduce the historical shape and exactly the
//! named legs go red:
//!
//! | mutation | red |
//! | --- | --- |
//! | `solve_socp_ipm_debug` builds its own `build_factorization` + `run_ipm` for symmetric cones | `leg_qcqp_hsde_default`, `leg_orthant_cone_verify` |
//! | `solve_qp_ipm_debug` re-enters at `solve_qp_ipm_core` with `equilibrate: false` — the historical shape, which is *both* "no Ruiz" and "no crossover" | `leg_qp_direct_driver`, `leg_lp_crossover` |
//! | `solve_qp_core` hands `solve_conic_hsde` a `None` hook instead of `hook` | every leg's `hook.fired > 0` |
//!
//! What these legs are **not** evidence about: the non-symmetric (exp/power)
//! drivers, which route through `solve_nonsym` and already took the hook —
//! `debug.rs::solve_socp_ipm_debug_routes_and_fires` owns that; PSD cones,
//! whose chordal decomposition the debugger now shares with the plain path
//! but which no fixture here reaches; and anything at benchmark scale.

use pounce_common::debug::{DebugAction, DebugHook, DebugState};
use pounce_convex::{
    ConeSpec, QpOptions, QpProblem, QpSolution, QpStatus, Triplet, solve_qp_ipm,
    solve_qp_ipm_debug, solve_socp_ipm, solve_socp_ipm_debug,
};
use pounce_feral::FeralSolverInterface;
use pounce_linsol::SparseSymLinearSolverInterface;

fn backend() -> Box<dyn SparseSymLinearSolverInterface> {
    Box::new(FeralSolverInterface::new())
}

/// The debugger script from the issue — `continue`, and nothing else. A pure
/// no-op on the trajectory, which is the whole premise being tested.
#[derive(Default)]
struct JustContinue {
    fired: usize,
    saw_tau: bool,
}

impl DebugHook for JustContinue {
    fn at_checkpoint(&mut self, st: &mut dyn DebugState) -> DebugAction {
        self.fired += 1;
        if st.block("tau").is_some() {
            self.saw_tau = true;
        }
        DebugAction::Resume
    }
}

/// Assert two solves are the same solve: same verdict, same iterate, same
/// iteration count. The count is the sensitive one — a status and an
/// objective can agree while the trajectory underneath has been replaced,
/// which is how gh #892's two `Optimal` instances hid the same defect its
/// two failing ones exposed.
fn assert_same_solve(leg: &str, plain: &QpSolution, debugged: &QpSolution) {
    assert_eq!(
        plain.status, debugged.status,
        "{leg}: status moved under the debugger"
    );
    assert_eq!(
        plain.iters, debugged.iters,
        "{leg}: iteration count moved under the debugger ({} → {}) — the \
         debugged run is on a different trajectory",
        plain.iters, debugged.iters
    );
    assert!(
        (plain.obj - debugged.obj).abs() <= 1e-12 * (1.0 + plain.obj.abs()),
        "{leg}: objective moved under the debugger ({} → {})",
        plain.obj,
        debugged.obj
    );
    for (i, (a, b)) in plain.x.iter().zip(&debugged.x).enumerate() {
        assert!(
            (a - b).abs() <= 1e-12 * (1.0 + a.abs()),
            "{leg}: x[{i}] moved under the debugger ({a} vs {b})"
        );
    }
}

/// The issue's 5-variable convex QCQP, in the SOC form the CLI extracts:
///
/// ```text
/// min ½xᵀPx + cᵀx   s.t. ‖Fx − g‖ ≤ 2,  −10 ≤ x ≤ 10
/// ```
///
/// Clarabel 0.11.1 at `tol = 1e-12` gives `f* = 0.46851407987426236` with
/// `‖Fx − g‖² = 4` active, so the cone binds and the solve is a real conic
/// one rather than an unconstrained quadratic wearing a cone.
fn issue892_qcqp() -> (QpProblem, Vec<ConeSpec>) {
    #[rustfmt::skip]
    let p = [
        [ 8.081237476191625, -1.3357111717521042, -3.177837792578595,   4.113912068867745,   -0.6167885305951004  ],
        [-1.3357111717521042, 6.166783520955787,   1.6776136382871059, -1.610501780331925,   -0.031514743829699654],
        [-3.177837792578595,  1.6776136382871059, 10.180019811121928,  -6.589542601208363,   -0.860089836489867   ],
        [ 4.113912068867745, -1.610501780331925,  -6.589542601208363,   5.887751599792348,    0.009833639085611079],
        [-0.6167885305951004,-0.031514743829699654,-0.860089836489867,  0.009833639085611079, 3.350517527917375   ],
    ];
    let c = [
        -0.7652887415836972,
        2.020688272883008,
        0.16922834928772218,
        -0.8979265392629115,
        -0.9539822990406178,
    ];
    #[rustfmt::skip]
    let f = [
        [1.8378244304667402, -0.5783508965433001,  1.1887040918656822,  1.42327388117323,    -2.324401356345807  ],
        [1.3509748934963017, -0.20843213237772143,-1.0841285882683698, -0.5438424479925973,   0.5608957690673765 ],
        [0.3739765553621166,  0.42551413722873815,-0.7248960165356201,  1.2790838213635174,   1.502130478896611  ],
        [1.8340638101929774,  1.000462644464517,   1.894314546207076,   2.0939269544234835,   0.7042559943816945 ],
        [0.8811621128331931,  0.5822091166931507,  0.5514244021092147,  0.8632445387294106,  -1.7089819015275898 ],
    ];
    let g = [
        -0.323798368729351,
        0.4878567515797354,
        -2.107331044865737,
        -1.4418405493566442,
        1.5844982856875247,
    ];

    // Lower triangle of P.
    let mut p_lower = Vec::new();
    for (i, row) in p.iter().enumerate() {
        for (j, &v) in row.iter().enumerate().take(i + 1) {
            p_lower.push(Triplet::new(i, j, v));
        }
    }
    // s = h − Gx must lie in the 6-dimensional second-order cone
    // `s₀ ≥ ‖s₁..₅‖`: row 0 is the constant radius 2, rows 1..5 are `Fx − g`.
    let mut gm = Vec::new();
    for (k, row) in f.iter().enumerate() {
        for (i, &v) in row.iter().enumerate() {
            gm.push(Triplet::new(k + 1, i, -v));
        }
    }
    let mut h = vec![2.0];
    h.extend(g.iter().map(|v| -v));

    let prob = QpProblem {
        n: 5,
        p_lower,
        c: c.to_vec(),
        a: vec![],
        b: vec![],
        g: gm,
        h,
        lb: vec![-10.0; 5],
        ub: vec![10.0; 5],
    };
    (prob, vec![ConeSpec::SecondOrder(6)])
}

/// Clarabel's answer on the same SOC formulation, as an outside number.
const ISSUE892_OPTIMUM: f64 = 0.46851407987426236;

/// The reported failure: default options, so `use_hsde` is on and the plain
/// run reaches the embedding. Before the fix the debugged run reached the
/// direct IPM instead and terminated `NumericalFailure`.
#[test]
fn leg_qcqp_hsde_default() {
    let (prob, cones) = issue892_qcqp();
    let opts = QpOptions::default();
    assert!(opts.use_hsde, "this leg is about the HSDE branch");

    let plain = solve_socp_ipm(&prob, &cones, &opts, backend);
    let mut hook = JustContinue::default();
    let debugged = solve_socp_ipm_debug(&prob, &cones, &opts, &mut hook, backend);

    assert_eq!(
        plain.status,
        QpStatus::Optimal,
        "the plain run must solve this — otherwise the leg proves nothing \
         about the debugger (iters={})",
        plain.iters
    );
    assert!(
        (plain.obj - ISSUE892_OPTIMUM).abs() < 1e-8,
        "the plain run must agree with the Clarabel oracle: {} vs {}",
        plain.obj,
        ISSUE892_OPTIMUM
    );
    assert_same_solve("qcqp/hsde", &plain, &debugged);
    assert!(hook.fired > 0, "the hook must actually have been fired");
    assert!(
        hook.saw_tau,
        "the conic solve routes to the HSDE embedding, so the debugger must \
         see `tau` — its absence is the symptom gh #892 was diagnosed by"
    );
}

/// The other side of `solve_qp_core`'s fork. `qp_hsde=no` takes the direct
/// driver on both paths, so the *driver* always agreed here — but the plain
/// path also screens the variable box and gates its exits through
/// `finite_or_failed`, which the debug path did not.
#[test]
fn leg_qcqp_direct_driver() {
    let (prob, cones) = issue892_qcqp();
    let opts = QpOptions {
        use_hsde: false,
        ..QpOptions::default()
    };

    let plain = solve_socp_ipm(&prob, &cones, &opts, backend);
    let mut hook = JustContinue::default();
    let debugged = solve_socp_ipm_debug(&prob, &cones, &opts, &mut hook, backend);

    assert_same_solve("qcqp/direct", &plain, &debugged);
    assert!(hook.fired > 0, "the hook must actually have been fired");
    assert!(
        !hook.saw_tau,
        "the direct driver has no homogenizing scalars to expose"
    );
}

/// A cone program whose cones are *all* nonnegative reaches
/// `verify_or_repair_optimum` (gh #414) on the way out of `solve_socp_ipm`.
/// The debug path used to return before that guard ran, so a repaired verdict
/// was another thing the debugger could not see.
#[test]
fn leg_orthant_cone_verify() {
    // min ½(x0² + x1²) − x0 − x1  s.t. x0 + x1 ≤ 1, x ≥ 0. Optimum (½, ½).
    let prob = QpProblem {
        n: 2,
        p_lower: vec![Triplet::new(0, 0, 1.0), Triplet::new(1, 1, 1.0)],
        c: vec![-1.0, -1.0],
        a: vec![],
        b: vec![],
        g: vec![Triplet::new(0, 0, 1.0), Triplet::new(0, 1, 1.0)],
        h: vec![1.0],
        lb: vec![0.0, 0.0],
        ub: vec![f64::INFINITY; 2],
    };
    let cones = [ConeSpec::Nonneg(1)];
    let opts = QpOptions::default();

    let plain = solve_socp_ipm(&prob, &cones, &opts, backend);
    let mut hook = JustContinue::default();
    let debugged = solve_socp_ipm_debug(&prob, &cones, &opts, &mut hook, backend);

    assert_eq!(plain.status, QpStatus::Optimal, "iters={}", plain.iters);
    assert_same_solve("orthant/verify", &plain, &debugged);
    assert!(hook.fired > 0, "the hook must actually have been fired");
}

/// The LP/QP entry point on its default driver. This one already agreed
/// before the fix — the issue measured it as bit-stable — so the leg is a pin
/// against losing that while unifying the paths, not a new claim.
#[test]
fn leg_qp_hsde_default() {
    let prob = box_qp();
    let opts = QpOptions::default();

    let plain = solve_qp_ipm(&prob, &opts, backend);
    let mut hook = JustContinue::default();
    let debugged = solve_qp_ipm_debug(&prob, &opts, &mut hook, backend);

    assert_eq!(plain.status, QpStatus::Optimal, "iters={}", plain.iters);
    assert_same_solve("qp/hsde", &plain, &debugged);
    assert!(hook.fired > 0, "the hook must actually have been fired");
}

/// `qp_hsde=no` on the QP entry point. The plain path Ruiz-equilibrates the
/// direct driver (`solve_qp_ipm_core`'s first branch); the debug path ran the
/// same driver on *un*-equilibrated data. Same defect class as the conic one,
/// one option away — which is why this leg is not a duplicate of the one
/// above.
#[test]
fn leg_qp_direct_driver() {
    let prob = box_qp();
    let opts = QpOptions {
        use_hsde: false,
        ..QpOptions::default()
    };
    assert!(
        opts.equilibrate,
        "the divergence was Ruiz, so it must be on"
    );

    let plain = solve_qp_ipm(&prob, &opts, backend);
    let mut hook = JustContinue::default();
    let debugged = solve_qp_ipm_debug(&prob, &opts, &mut hook, backend);

    assert_eq!(plain.status, QpStatus::Optimal, "iters={}", plain.iters);
    assert_same_solve("qp/direct", &plain, &debugged);
    assert!(hook.fired > 0, "the hook must actually have been fired");
}

/// A pure LP, where `solve_qp_ipm` layers the active-set crossover on the
/// interior iterate. The debug entry point did not, so the debugged run
/// returned the un-purified point.
#[test]
fn leg_lp_crossover() {
    // max x0 + x1 s.t. x0 + x1 ≤ 1, x ≥ 0. **Dual-degenerate on purpose**:
    // every point of the edge `x0 + x1 = 1` is optimal, so the IPM converges
    // to its analytic center `(½, ½)` and crossover pivots to a vertex. An LP
    // with a unique optimal vertex would leave crossover nothing to do and the
    // leg would stay green under the mutation that skips it.
    let prob = QpProblem {
        n: 2,
        p_lower: vec![],
        c: vec![-1.0, -1.0],
        a: vec![],
        b: vec![],
        g: vec![Triplet::new(0, 0, 1.0), Triplet::new(0, 1, 1.0)],
        h: vec![1.0],
        lb: vec![0.0, 0.0],
        ub: vec![f64::INFINITY; 2],
    };
    // Crossover is opt-in (`QpOptions::crossover` defaults to `false`, CLI
    // `qp_crossover=yes`), so the leg has to ask for it.
    let opts = QpOptions {
        crossover: true,
        ..QpOptions::default()
    };

    let plain = solve_qp_ipm(&prob, &opts, backend);
    let mut hook = JustContinue::default();
    let debugged = solve_qp_ipm_debug(&prob, &opts, &mut hook, backend);

    assert_eq!(plain.status, QpStatus::Optimal, "iters={}", plain.iters);
    // The leg is only evidence about crossover if crossover moved something:
    // assert the plain answer really is a vertex rather than the analytic
    // center the interior iteration converges to.
    let at_vertex = |x: &[f64]| x[0].min(x[1]) < 1e-6;
    assert!(
        at_vertex(&plain.x),
        "crossover must have pivoted to a vertex for this leg to test it: \
         x = {:?}",
        plain.x
    );
    assert_same_solve("lp/crossover", &plain, &debugged);
    assert!(hook.fired > 0, "the hook must actually have been fired");
}

/// An ill-conditioned box QP: curvature spanning six orders, so Ruiz has
/// something to do and the two drivers do not trivially coincide.
fn box_qp() -> QpProblem {
    QpProblem {
        n: 3,
        p_lower: vec![
            Triplet::new(0, 0, 1.0e-3),
            Triplet::new(1, 1, 1.0),
            Triplet::new(2, 2, 1.0e3),
        ],
        c: vec![-1.0, -2.0, -3.0],
        a: vec![],
        b: vec![],
        g: vec![
            Triplet::new(0, 0, 1.0),
            Triplet::new(0, 1, 1.0),
            Triplet::new(0, 2, 1.0),
        ],
        h: vec![2.0],
        lb: vec![-5.0; 3],
        ub: vec![5.0; 3],
    }
}
