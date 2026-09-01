//! Second-order-cone sensitivity: the three faces, and the refusals between
//! them.
//!
//! # Why this file exists
//!
//! `conic_sensitivity_refused.rs` established that a solved SOCP handed to the
//! orthant builder is refused. This file is the other half: `build_conic`
//! *answering* for a `SecondOrder` block, and the evidence that the answer is a
//! derivative rather than a plausible number.
//!
//! A cone's active object is not a set of rows. Its slack sits on a **face**,
//! and for `SOC(k) = { (s₀, s₁) : s₀ ≥ ‖s₁‖ }` there are three of them, which
//! behave differently enough that reading one as another is a wrong answer, not
//! a loose one:
//!
//! | face | rows contributed | curvature | predictor |
//! |---|---|---|---|
//! | `Interior` | none | none | exact (the block is not binding) |
//! | `Apex` | every row of the block | none | exact (the face is a point, hence flat) |
//! | `Boundary` | one, `wᵀG` with `w = (1, −s₁/s₀)` | `(ν/s₀)(Σ gᵣgᵣᵀ − uuᵀ)` | first order (the face is curved) |
//!
//! # The finding this file was written by
//!
//! The boundary curvature term is **not** a refinement. Written without it —
//! which is how the first draft of `build_conic` shipped into this branch, by
//! analogy with the orthant path where the active face is a hyperplane and
//! genuinely carries none — the step converges to the **wrong derivative**:
//! `dx/db` reads `(0.348, 0.652)` where the true answer is `(0.5, 0.5)`, at
//! every `δ` from `1e-2` down. Nothing internal complains, because the step
//! solves exactly the KKT it was handed; the KKT is simply not this problem's.
//! That is `/sens-review` entry 5 — silently wrong while reporting success —
//! and the re-solve oracle below is the only guard in this crate that could
//! have found it, because it is the only one that reads a number the
//! sensitivity layer did not produce.
//!
//! # What this file is NOT evidence about
//!
//! - **Any cone family but `SecondOrder`.** `Psd`, `Exponential` and `Power`
//!   are refused; `conic_sensitivity_refused.rs` owns that.
//! - **Release / fix-relax on a conic build.** Every fixture here is
//!   bound-free, so `release_slots` is empty and the release path is never
//!   entered. `convex_sens_release.rs` owns it, on orthant models. A cone row
//!   cannot be released at all today — the slots are built for variable bounds.
//! - **Mixed partitions beyond the one `[Nonneg, SecondOrder]` case.**
//!   `a_mixed_partition_classifies_each_block_by_its_own_rule` is a single
//!   fixture and is not a sweep.
//! - **Blocks larger than `SOC(3)`, or many cone blocks.** The curvature is
//!   assembled as `d` rank-one outer products, so its density grows with the
//!   block's column support; nothing here measures that at scale.
//! - **Activity classification.** `ConvexActivityReport` reads rows as orthant
//!   rows and is not meaningful on a cone block. `convex_activity.rs` owns the
//!   orthant case; the conic case has no classifier and this file does not
//!   pretend otherwise.
//!
//! # Mutation evidence
//!
//! Every row below was **run** — each mutation applied to
//! `crates/pounce-convex/src/sensitivity.rs`, compiled (a mutation that does
//! not compile emits no failures and reads exactly like a mutation nothing
//! catches), and the suite run with `--no-fail-fast`.
//!
//! | mutation | red here | note |
//! |---|---|---|
//! | `soc_boundary_curvature` returns `Vec::new()` | `the_boundary_step_is_the_analytic_derivative`, `the_boundary_step_matches_a_resolve`, `the_boundary_error_is_second_order`, `the_boundary_face_carries_curvature_the_orthant_case_does_not`, `a_mixed_partition_classifies_each_block_by_its_own_rule` | the original defect: `dx/db` reads `(0.348, 0.652)` against `(0.5, 0.5)` |
//! | `assemble_kkt` ignores its `curvature` argument | the same five | the same defect one layer down, so neither site can drop it quietly |
//! | drop the `− u uᵀ` rank-one correction | the same five | the term is a projection, not a scalar: `Σ gᵣgᵣᵀ` alone is the wrong operator |
//! | boundary normal `w` uses `+sᵣ/s₀` | `the_boundary_step_matches_a_resolve`, `the_boundary_error_is_second_order`, `the_boundary_face_carries_curvature_the_orthant_case_does_not` — **but not** `the_boundary_step_is_the_analytic_derivative` | measured, and worth stating: this fixture's closed-form coordinates survive the sign flip, so the hand-derived assertion alone would have passed a wrong active row. It is the re-solve oracle that catches it. `/sens-review` entry 5 in miniature |
//! | apex branch returns `Interior` | `the_apex_face_pins_the_whole_block`, `the_three_fixtures_take_different_branches`, and `the_apex_case_needs_the_cone_partition` in `conic_sensitivity_refused.rs` | an apex read as interior stops constraining `dx` at all |
//! | `dual_collapsed` hard-coded `false` | `an_apex_with_a_collapsed_dual_is_refused`, `a_boundary_with_a_collapsed_dual_is_refused` | the two weak-conic-activity refusals, and only those |
//! | the near-apex threshold `SOC_APEX_REL` → `0.0` at that one site | `a_boundary_point_too_close_to_the_apex_is_refused` | pins that the thin band is actually reached rather than dead code |
//! | the `gap < −SOC_BOUNDARY_REL` arm disabled | `a_slack_outside_the_cone_is_refused` | otherwise it falls through and builds a normal from `s₀ < ‖s₁‖` |
//! | the interior complementarity arm disabled | `an_interior_block_that_does_not_complement_is_refused` | |
//! | `primal_scale` replaced by a bare `1.0` in `build_conic` | `the_apex_decision_is_relative_to_the_problems_scale` — **and nothing else** | that test was written *because* the first run of this mutation was green across the whole crate: every other fixture here is `O(1)`, where the relative and absolute readings coincide. A scale-convention change with no failing test is exactly `/sens-review` entry 3 |

use pounce_convex::QpOptions;
use pounce_convex::cones::ConeSpec;
use pounce_convex::ipm::solve_socp_ipm;
use pounce_convex::qp::{QpProblem, QpSolution, QpStatus, Triplet};
use pounce_convex::sensitivity::{ConeBlockKind, QpSensitivity, SensError};
use pounce_feral::FeralSolverInterface;
use pounce_linsol::SparseSymLinearSolverInterface;

fn backend() -> Box<dyn SparseSymLinearSolverInterface> {
    Box::new(FeralSolverInterface::new())
}

fn tri(row: usize, col: usize, val: f64) -> Triplet {
    Triplet { row, col, val }
}

/// The base solve's tolerance. The oracle re-solve runs at the same setting:
/// unlike the NLP arm there is no barrier parameter left over at convergence,
/// so "two orders tighter" has nothing to buy — a converged convex QP sits on
/// its exact optimum to within the residual, not `O(√μ)` away from it.
const TOL: f64 = 1e-11;

fn opts() -> QpOptions {
    QpOptions {
        tol: TOL,
        ..Default::default()
    }
}

const SOC3: [ConeSpec; 1] = [ConeSpec::SecondOrder(3)];

// ---------------------------------------------------------------------------
// Fixtures — one per face, each parameterized by the equality right-hand side
// so a perturbation has something to move.
// ---------------------------------------------------------------------------

/// **Boundary.** Variables `(x₀, x₁, t)`; the cone is `s = (t, x₀, x₁) ∈ SOC(3)`,
/// i.e. `t ≥ ‖(x₀, x₁)‖`. Minimizing `½‖x‖² + ½t² − x₀ − 0.2x₁` drives `t` down
/// onto `‖(x₀, x₁)‖`, so the slack lands on the cone boundary away from the
/// apex.
///
/// Its derivative is known in closed form, which is what makes it the fixture
/// the curvature defect was caught on. Eliminating `t = ‖x‖` leaves
/// `x₀² − x₀ + x₁² − 0.2x₁` subject to `x₀ + x₁ = b`, whose stationarity gives
/// `x₀ − x₁ = 0.4` for every `b`. So `dx₀/db = dx₁/db = ½` exactly, and
/// `dt/db` follows from the (curved) cone relation.
fn boundary(b: f64) -> QpProblem {
    QpProblem {
        n: 3,
        p_lower: vec![tri(0, 0, 1.0), tri(1, 1, 1.0), tri(2, 2, 1.0)],
        c: vec![-1.0, -0.2, 0.0],
        a: vec![tri(0, 0, 1.0), tri(0, 1, 1.0)],
        b: vec![b],
        // s = (t, x₀, x₁) = h − Gx with h = 0.
        g: vec![tri(0, 2, -1.0), tri(1, 0, -1.0), tri(2, 1, -1.0)],
        h: vec![0.0, 0.0, 0.0],
        lb: vec![],
        ub: vec![],
    }
}

/// **Apex.** Variables `(x₀, x₁, x₂, t)` with the same cone on `(t, x₀, x₁)`,
/// and a linear cost on `t` heavy enough to crush the cone to its apex — which
/// pins `x₀ = x₁ = t = 0`. The equality `x₀ + x₂ = b` then hands the whole
/// perturbation to `x₂`, so `dx/db = (0, 0, 1, 0)` **exactly**: the apex is a
/// single point, hence a flat face, and the predictor is not an approximation.
fn apex(b: f64) -> QpProblem {
    QpProblem {
        n: 4,
        p_lower: vec![tri(0, 0, 1.0), tri(1, 1, 1.0), tri(2, 2, 1.0)],
        c: vec![0.0, 0.0, -1.0, 5.0],
        a: vec![tri(0, 0, 1.0), tri(0, 2, 1.0)],
        b: vec![b],
        g: vec![tri(0, 3, -1.0), tri(1, 0, -1.0), tri(2, 1, -1.0)],
        h: vec![0.0, 0.0, 0.0],
        lb: vec![],
        ub: vec![],
    }
}

/// **Interior.** [`boundary`] with the cost on `t` reversed, so the optimum
/// pushes `t` well above `‖(x₀, x₁)‖` and the cone is slack. The block must
/// contribute nothing at all — no row, no curvature — and the answer must be
/// the cone-free one.
fn interior(b: f64) -> QpProblem {
    QpProblem {
        c: vec![-1.0, -0.2, -5.0],
        ..boundary(b)
    }
}

fn solve(prob: &QpProblem) -> QpSolution {
    let sol = solve_socp_ipm(prob, &SOC3, &opts(), backend);
    assert_eq!(
        sol.status,
        QpStatus::Optimal,
        "a fixture that does not solve makes every assertion below vacuous"
    );
    sol
}

fn sens_for(prob: &QpProblem, sol: &QpSolution) -> QpSensitivity {
    match QpSensitivity::build_conic(prob, &SOC3, sol, &opts(), 1e-7, backend) {
        Ok(v) => v,
        Err(e) => panic!("build_conic must accept this fixture, got {e:?}"),
    }
}

/// `x*(b + δ) − x*(b)` from an independent re-solve — the only number in this
/// crate's sensitivity guards that the sensitivity layer did not produce.
fn oracle(f: impl Fn(f64) -> QpProblem, b0: f64, delta: f64, base: &QpSolution) -> Vec<f64> {
    let sol = solve(&f(b0 + delta));
    base.x.iter().zip(&sol.x).map(|(a, b)| b - a).collect()
}

fn inf_norm(v: &[f64]) -> f64 {
    v.iter().fold(0.0_f64, |m, x| m.max(x.abs()))
}

// ---------------------------------------------------------------------------
// Preconditions. Without these the tests below assert nothing about cones.
// ---------------------------------------------------------------------------

/// `/sens-review` entry 6, made explicit: the rule under test branches three
/// ways and each fixture must take a different branch. A file whose fixtures
/// all land in one face is green while the other two are broken — which is how
/// gh#756 shipped.
#[test]
fn the_three_fixtures_take_different_branches() {
    let cases = [
        (
            "boundary",
            boundary(1.0) as QpProblem,
            ConeBlockKind::Boundary,
        ),
        ("apex", apex(1.0), ConeBlockKind::Apex),
        ("interior", interior(1.0), ConeBlockKind::Interior),
    ];
    let mut seen = Vec::new();
    for (name, prob, want) in cases {
        let sol = solve(&prob);
        let sens = sens_for(&prob, &sol);
        assert_eq!(
            sens.cone_block_kinds(),
            [(0, want)],
            "the `{name}` fixture must reach the `{want:?}` face"
        );
        seen.push(want);
    }
    seen.sort_by_key(|k| format!("{k:?}"));
    seen.dedup();
    assert_eq!(seen.len(), 3, "the three fixtures must reach three faces");
}

/// The boundary fixture really is on the boundary and really is away from the
/// apex — otherwise `the_boundary_*` tests are measuring the apex path.
#[test]
fn the_boundary_fixture_sits_on_the_cone_boundary_away_from_the_apex() {
    let prob = boundary(1.0);
    let sol = solve(&prob);
    // s = (t, x₀, x₁), since h = 0 and G = −[e₂; e₀; e₁].
    let s = [sol.x[2], sol.x[0], sol.x[1]];
    let tail = (s[1] * s[1] + s[2] * s[2]).sqrt();
    assert!(
        (s[0] - tail).abs() <= 1e-9,
        "the slack must sit on s₀ = ‖s₁‖, got s₀ = {}, ‖s₁‖ = {tail}",
        s[0]
    );
    assert!(
        s[0] > 0.1,
        "and well away from the apex, or this is the apex path: s₀ = {}",
        s[0]
    );
    // The dual is on the boundary of the dual cone, so ν = z₀ > 0 and the
    // curvature term the tests below are about is nonzero.
    assert!(
        sol.z[0] > 0.1,
        "the multiplier ν = z₀ scales the curvature; a collapsed one makes the \
         curvature tests vacuous. got {}",
        sol.z[0]
    );
}

// ---------------------------------------------------------------------------
// The boundary face: the oracle, and the curvature it exists to catch.
// ---------------------------------------------------------------------------

/// The headline. `dx/db` on the two coordinates whose derivative is known in
/// closed form must be `(½, ½)` — not `(0.348, 0.652)`, which is what the same
/// code returns with the cone's curvature omitted from the KKT.
///
/// The tolerance is deliberately tight (`1e-9`, against a defect of `0.15`):
/// this is an exact algebraic identity, not a discretization.
#[test]
fn the_boundary_step_is_the_analytic_derivative() {
    let prob = boundary(1.0);
    let sol = solve(&prob);
    let mut sens = sens_for(&prob, &sol);
    let delta = 1e-3;
    let dx = sens.parametric_step(&[0], &[delta]);
    assert!(
        (dx[0] / delta - 0.5).abs() < 1e-9 && (dx[1] / delta - 0.5).abs() < 1e-9,
        "eliminating t leaves x₀ − x₁ = 0.4 for every b, so dx₀/db = dx₁/db = ½ exactly. \
         got ({}, {}). A value near (0.348, 0.652) is the cone curvature missing from the \
         KKT's (x,x) block.",
        dx[0] / delta,
        dx[1] / delta
    );
}

/// The re-solve oracle: the step against `x*(b + δ) − x*(b)` computed by an
/// independent solve, on every coordinate including the curved one.
#[test]
fn the_boundary_step_matches_a_resolve() {
    let prob = boundary(1.0);
    let sol = solve(&prob);
    let mut sens = sens_for(&prob, &sol);
    let delta = 1e-4;
    let dx = sens.parametric_step(&[0], &[delta]);
    let truth = oracle(boundary, 1.0, delta, &sol);
    let err = inf_norm(
        &dx.iter()
            .zip(&truth)
            .map(|(a, b)| a - b)
            .collect::<Vec<_>>(),
    );
    // The bound is `O(δ²)`, which is what a correct linearization of a curved
    // face earns. A first-order-wrong step lands at `≈ 0.15·δ = 1.5e-5` here,
    // four orders above this; the measured error is `4.5e-10`, three orders
    // below it. The band between is deliberately wide — this test is about
    // ruling out the wrong *order*, not pinning a constant.
    let allowed = 1e2 * delta * delta;
    assert!(
        err < allowed,
        "the step must agree with an independent re-solve to second order.\n  \
         step  {dx:?}\n  truth {truth:?}\n  err   {err:e} (allowed {allowed:e})"
    );
}

/// And it agrees to *second* order, which is the property that separates a
/// correct linearization of a curved face from a wrong one. A first-order-wrong
/// step has an error proportional to `δ`, so its ratio across a decade is 10;
/// a correct one has an error proportional to `δ²`, so the ratio is 100.
#[test]
fn the_boundary_error_is_second_order() {
    let prob = boundary(1.0);
    let sol = solve(&prob);
    let mut sens = sens_for(&prob, &sol);
    let mut errs = Vec::new();
    for delta in [1e-2, 1e-3] {
        let dx = sens.parametric_step(&[0], &[delta]);
        let truth = oracle(boundary, 1.0, delta, &sol);
        errs.push(inf_norm(
            &dx.iter()
                .zip(&truth)
                .map(|(a, b)| a - b)
                .collect::<Vec<_>>(),
        ));
    }
    let ratio = errs[0] / errs[1];
    assert!(
        ratio > 50.0,
        "shrinking δ by 10× must shrink the error by ~100× (second order); a ratio near \
         10 means the step is wrong at first order. errors {errs:?}, ratio {ratio}"
    );
}

/// The face is curved, so the block contributes a nonzero `(x,x)` term — and
/// the sensitivity KKT is therefore *not* the objective's Hessian bordered by
/// the active rows. Pinned by comparison against the same problem posed with
/// the cone replaced by its linearization at the optimum, which is the KKT you
/// get if the curvature is dropped.
#[test]
fn the_boundary_face_carries_curvature_the_orthant_case_does_not() {
    let prob = boundary(1.0);
    let sol = solve(&prob);
    let mut sens = sens_for(&prob, &sol);
    let delta = 1e-3;
    let dx = sens.parametric_step(&[0], &[delta]);

    // The same active row, posed as an ordinary (flat) inequality: w = (1, −ŝ₁)
    // applied to G gives the row (0.919, 0.394, −1) in x-coordinates. Reading
    // the cone that way is exactly the curvature-free KKT.
    let s = [sol.x[2], sol.x[0], sol.x[1]];
    let (w1, w2) = (-s[1] / s[0], -s[2] / s[0]);
    let flat = QpProblem {
        g: vec![tri(0, 2, -1.0), tri(0, 0, -w1), tri(0, 1, -w2)],
        h: vec![0.0],
        ..boundary(1.0)
    };
    let flat_sol = QpSolution {
        z: vec![sol.z[0]],
        ..sol.clone()
    };
    let mut flat_sens = QpSensitivity::build_default(&flat, &flat_sol, backend)
        .expect("the linearized problem is an ordinary orthant QP");
    let flat_dx = flat_sens.parametric_step(&[0], &[delta]);
    assert!(
        inf_norm(
            &dx.iter()
                .zip(&flat_dx)
                .map(|(a, b)| a - b)
                .collect::<Vec<_>>()
        ) > 1e-4 * delta,
        "the conic build must NOT reduce to the flat linearization — if it does, the \
         curvature term is absent. conic {dx:?}, flat {flat_dx:?}"
    );
    // And it is the conic one that is right.
    let truth = oracle(boundary, 1.0, delta, &sol);
    let conic_err = inf_norm(
        &dx.iter()
            .zip(&truth)
            .map(|(a, b)| a - b)
            .collect::<Vec<_>>(),
    );
    let flat_err = inf_norm(
        &flat_dx
            .iter()
            .zip(&truth)
            .map(|(a, b)| a - b)
            .collect::<Vec<_>>(),
    );
    assert!(
        conic_err < flat_err / 100.0,
        "and the curved reading must be the accurate one: conic err {conic_err:e}, \
         flat err {flat_err:e}"
    );
}

// ---------------------------------------------------------------------------
// The apex face: flat, hence exact.
// ---------------------------------------------------------------------------

/// At the apex the whole block is active: `ds = 0`, so every row of `G` for the
/// block enters and the cone coordinates cannot move at all. The perturbation
/// goes entirely to `x₂`, and because the face is a point the predictor is
/// exact — the error is at solver tolerance, not `O(δ²)`.
#[test]
fn the_apex_face_pins_the_whole_block() {
    let prob = apex(1.0);
    let sol = solve(&prob);
    let mut sens = sens_for(&prob, &sol);
    assert_eq!(sens.cone_block_kinds(), [(0, ConeBlockKind::Apex)]);

    for delta in [1e-2, 1e-4] {
        let dx = sens.parametric_step(&[0], &[delta]);
        assert!(
            dx[0].abs() < 1e-12 * delta && dx[1].abs() < 1e-12 * delta && dx[3].abs() < 1e-12,
            "the cone coordinates are pinned at the apex, so their step is zero: {dx:?}"
        );
        assert!(
            (dx[2] / delta - 1.0).abs() < 1e-9,
            "the whole perturbation goes to x₂: dx₂/db = {}",
            dx[2] / delta
        );
        let truth = oracle(apex, 1.0, delta, &sol);
        let err = inf_norm(
            &dx.iter()
                .zip(&truth)
                .map(|(a, b)| a - b)
                .collect::<Vec<_>>(),
        );
        assert!(
            err < 1e-9 * delta.max(1e-6),
            "a flat face makes the predictor exact, not first order: err {err:e} at δ = {delta:e}"
        );
    }
}

// ---------------------------------------------------------------------------
// The interior face: nothing at all.
// ---------------------------------------------------------------------------

/// A slack cone contributes no row and no curvature, so the answer must be the
/// one the problem has with the cone deleted outright. Asserting "no error" is
/// weaker than asserting "the same as the cone-free problem", because a bug
/// that adds a row whose multiplier happens to be zero passes the first.
#[test]
fn an_interior_block_contributes_nothing() {
    let prob = interior(1.0);
    let sol = solve(&prob);
    let mut sens = sens_for(&prob, &sol);
    assert_eq!(sens.cone_block_kinds(), [(0, ConeBlockKind::Interior)]);
    assert_eq!(
        sens.kkt_dim(),
        prob.n + prob.m_eq(),
        "an interior cone block must contribute no active rows at all, so the KKT is the          bare (x, y) system"
    );
    assert!(sens.active_ineq().is_empty());

    let delta = 1e-3;
    let dx = sens.parametric_step(&[0], &[delta]);

    // The same problem with the cone deleted.
    let coneless = QpProblem {
        g: vec![],
        h: vec![],
        ..interior(1.0)
    };
    let coneless_sol = QpSolution {
        z: vec![],
        ..sol.clone()
    };
    let mut coneless_sens = QpSensitivity::build_default(&coneless, &coneless_sol, backend)
        .expect("the cone-free problem is an ordinary equality-constrained QP");
    let coneless_dx = coneless_sens.parametric_step(&[0], &[delta]);
    assert_eq!(
        dx, coneless_dx,
        "a slack cone must leave the step bit-identical to the cone-free problem's"
    );

    let truth = oracle(interior, 1.0, delta, &sol);
    let err = inf_norm(
        &dx.iter()
            .zip(&truth)
            .map(|(a, b)| a - b)
            .collect::<Vec<_>>(),
    );
    assert!(err < 1e-9 * delta, "and it must match a re-solve: {err:e}");
}

// ---------------------------------------------------------------------------
// Mixed partitions.
// ---------------------------------------------------------------------------

/// A partition carrying both an orthant block and a cone block must apply each
/// block's own rule to its own rows — the orthant screen to the first, the face
/// decomposition to the second.
#[test]
fn a_mixed_partition_classifies_each_block_by_its_own_rule() {
    // `boundary` with one extra, strictly inactive orthant row prepended.
    let base = boundary(1.0);
    let mut g = vec![tri(0, 0, 1.0)]; // x₀ ≤ 10, slack
    for t in &base.g {
        g.push(tri(t.row + 1, t.col, t.val));
    }
    let prob = QpProblem {
        g,
        h: vec![10.0, 0.0, 0.0, 0.0],
        ..base
    };
    let cones = [ConeSpec::Nonneg(1), ConeSpec::SecondOrder(3)];
    let sol = solve_socp_ipm(&prob, &cones, &opts(), backend);
    assert_eq!(sol.status, QpStatus::Optimal);

    let mut sens = QpSensitivity::build_conic(&prob, &cones, &sol, &opts(), 1e-7, backend)
        .expect("a Nonneg + SecondOrder partition is fully supported");
    assert_eq!(
        sens.cone_block_kinds(),
        [(1, ConeBlockKind::Boundary)],
        "only the cone block is reported, and by its own index in the partition"
    );
    // The inactive orthant row contributes nothing; the cone boundary
    // contributes one row. So the step is the pure-cone answer.
    let delta = 1e-3;
    let dx = sens.parametric_step(&[0], &[delta]);
    assert!(
        (dx[0] / delta - 0.5).abs() < 1e-9 && (dx[1] / delta - 0.5).abs() < 1e-9,
        "the mixed partition must reach the same derivative as the pure one: {dx:?}"
    );
}

// ---------------------------------------------------------------------------
// Frame: the verdict must not move when the block is rescaled.
// ---------------------------------------------------------------------------

/// Scaling a cone block's rows of `G` by `c > 0` is a modelling choice, not a
/// change of problem: `s → c·s` stays in the same cone, `z → z/c` keeps
/// stationarity, and the face is the same face. So the classification *and* the
/// step must both be unmoved.
///
/// This is the leg that exercises the scale convention at all — every other
/// fixture in the file is `O(1)`, so a `primal_scale` replaced by a bare `1.0`
/// is invisible to them (measured; see the mutation table).
#[test]
fn the_classification_is_unmoved_by_scaling_the_cone_block() {
    let delta = 1e-3;
    let base = boundary(1.0);
    let base_sol = solve(&base);
    let base_dx = sens_for(&base, &base_sol).parametric_step(&[0], &[delta]);

    for c in [1e-3, 1e3] {
        let scaled = QpProblem {
            g: base
                .g
                .iter()
                .map(|t| tri(t.row, t.col, c * t.val))
                .collect(),
            h: base.h.iter().map(|v| c * v).collect(),
            ..boundary(1.0)
        };
        let sol = solve(&scaled);
        let mut sens = sens_for(&scaled, &sol);
        assert_eq!(
            sens.cone_block_kinds(),
            [(0, ConeBlockKind::Boundary)],
            "scaling the block by {c:e} must not change which face it is on"
        );
        let dx = sens.parametric_step(&[0], &[delta]);
        let rel = inf_norm(
            &dx.iter()
                .zip(&base_dx)
                .map(|(a, b)| a - b)
                .collect::<Vec<_>>(),
        ) / inf_norm(&base_dx);
        assert!(
            rel < 1e-6,
            "and it must not change the step: c = {c:e}, base {base_dx:?}, scaled {dx:?}, \
             relative move {rel:e}"
        );
    }
}

/// The apex/boundary decision is relative to the **problem's** primal scale,
/// not to an absolute constant — the same convention the orthant guard in
/// `build` uses, so the two cannot disagree about what "zero" means on one
/// solution.
///
/// This test exists because nothing else in the crate reaches the difference.
/// Replacing `primal_scale` with a bare `1.0` in `build_conic` left the whole
/// suite green (measured; see the mutation table), because every other fixture
/// here is `O(1)` and the two readings coincide there. The fixture below is
/// `O(1e6)`, where they disagree outright: a slack of `1e-3` against data of
/// `1e6` is the apex under the relative rule and a boundary point under the
/// absolute one — a different face, a different active set, and a different
/// answer.
#[test]
fn the_apex_decision_is_relative_to_the_problems_scale() {
    const BIG: f64 = 1e6;
    let s = [1e-3, 6e-4, 8e-4]; // s₀ = ‖s₁‖ exactly, so both readings are legal
    let prob = QpProblem {
        n: 3,
        p_lower: vec![tri(0, 0, 1.0), tri(1, 1, 1.0), tri(2, 2, 1.0)],
        c: vec![0.0, 0.0, 0.0],
        a: vec![],
        b: vec![],
        g: vec![tri(0, 0, 1.0), tri(1, 1, 1.0), tri(2, 2, 1.0)],
        h: vec![BIG + s[0], BIG + s[1], BIG + s[2]],
        lb: vec![],
        ub: vec![],
    };
    let sol = QpSolution {
        status: QpStatus::Optimal,
        x: vec![BIG, BIG, BIG],
        y: vec![],
        z: vec![1.0, -0.6, -0.8],
        z_lb: vec![0.0; 3],
        z_ub: vec![0.0; 3],
        obj: 0.0,
        iters: 0,
        iterates: vec![],
    };
    let sens = QpSensitivity::build_conic(&prob, &SOC3, &sol, &opts(), 1e-7, backend)
        .expect("an apex with a live dual is a supported face");
    assert_eq!(
        sens.cone_block_kinds(),
        [(0, ConeBlockKind::Apex)],
        "a slack of {:e} against problem data of {BIG:e} is at the apex under the \
         problem-relative rule; reading it as a boundary point is what an absolute \
         threshold does",
        s[0]
    );
}

// ---------------------------------------------------------------------------
// The refusals. Each is a point where `dx/db` does not exist or cannot be
// computed from the numbers to hand; answering anyway is the defect.
// ---------------------------------------------------------------------------

/// A hand-built solution, so the degenerate geometry is reachable at all — an
/// interior-point solve will not stop on any of these.
fn hand_built(s: [f64; 3], z: [f64; 3]) -> (QpProblem, QpSolution) {
    // G = I and x = 0, so s = h exactly and the solver's own arithmetic is out
    // of the picture.
    let prob = QpProblem {
        n: 3,
        p_lower: vec![tri(0, 0, 1.0), tri(1, 1, 1.0), tri(2, 2, 1.0)],
        c: vec![0.0, 0.0, 0.0],
        a: vec![],
        b: vec![],
        g: vec![tri(0, 0, 1.0), tri(1, 1, 1.0), tri(2, 2, 1.0)],
        h: s.to_vec(),
        lb: vec![],
        ub: vec![],
    };
    let sol = QpSolution {
        status: QpStatus::Optimal,
        x: vec![0.0, 0.0, 0.0],
        y: vec![],
        z: z.to_vec(),
        z_lb: vec![0.0; 3],
        z_ub: vec![0.0; 3],
        obj: 0.0,
        iters: 0,
        iterates: vec![],
    };
    (prob, sol)
}

fn refusal(s: [f64; 3], z: [f64; 3]) -> SensError {
    let (prob, sol) = hand_built(s, z);
    match QpSensitivity::build_conic(&prob, &SOC3, &sol, &opts(), 1e-7, backend) {
        Err(e) => e,
        Ok(_) => panic!("s = {s:?}, z = {z:?} must be refused, but it was accepted"),
    }
}

fn assert_nonsmooth(e: SensError, needle: &str) {
    match e {
        SensError::NonsmoothConePoint { block, what } => {
            assert_eq!(block, 0, "the refusal must name the offending block");
            assert!(
                what.contains(needle),
                "the message must say which condition fired; wanted {needle:?}, got {what:?}"
            );
        }
        other => panic!("expected NonsmoothConePoint, got {other:?}"),
    }
}

/// Slack *and* multiplier collapsed together at the apex: the conic weakly
/// active case. There is no single `dx/db` — the answer depends on which way
/// the perturbation pushes the block off the apex.
#[test]
fn an_apex_with_a_collapsed_dual_is_refused() {
    assert_nonsmooth(
        refusal([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
        "apex and the dual has collapsed",
    );
}

/// The same thing on the boundary rather than at the apex — the direct analogue
/// of a weakly active orthant row, and the case CLAUDE.md's gh#763 rule is
/// about: the block is genuinely on its face, and the derivative is still
/// two-valued.
#[test]
fn a_boundary_with_a_collapsed_dual_is_refused() {
    assert_nonsmooth(
        refusal([1.0, 0.6, 0.8], [0.0, 0.0, 0.0]),
        "boundary with a collapsed dual",
    );
}

/// On the boundary with `s₀` at round-off: `w = (1, −s₁/s₀)` would be a
/// direction made of noise. This is the thin band between the two relative
/// tests, and this test is the evidence that it is reachable rather than dead
/// code — see `soc_face`' own documentation for why it is thin.
#[test]
fn a_boundary_point_too_close_to_the_apex_is_refused() {
    assert_nonsmooth(
        refusal([6e-9, 1.4e-8, 0.0], [1.0, -0.6, -0.8]),
        "too close to the apex",
    );
}

/// A slack outside the cone by more than the solve's tolerance has no face to
/// linearize against. Without this arm the code falls through and builds a
/// normal from `s₀ < ‖s₁‖`, which is not a boundary point.
#[test]
fn a_slack_outside_the_cone_is_refused() {
    assert_nonsmooth(
        refusal([0.1, 0.6, 0.8], [1.0, -0.6, -0.8]),
        "outside the cone",
    );
}

/// Strictly inside the cone with a live dual: `⟨s, z⟩ ≫ 0`, so this is not a
/// converged optimum whatever its status field says. Reading it as `Interior`
/// would drop a block that is in fact binding.
#[test]
fn an_interior_block_that_does_not_complement_is_refused() {
    assert_nonsmooth(
        refusal([10.0, 0.6, 0.8], [1.0, -0.6, -0.8]),
        "strictly inside the cone",
    );
}
