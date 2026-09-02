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
//! - **Whether a `db` the caller actually asks for is answerable.**
//!   `apex_can_absorb_db` decides whether *every* `db` in `range(A)` is
//!   reachable — `rank([A;B]) == rank(A) + rank(B)`, over the active rows that
//!   **cannot be released** (cone faces and active orthant rows; variable
//!   bounds excluded, see `an_active_bound_does_not_count_as_an_apex_pin`).
//!   That is exact for the question it asks, and it is a *build-time* question:
//!   a build serves every later perturbation, so one unreachable direction
//!   refuses all of them.
//!
//!   What it says nothing about is a `db` outside `range(A)` entirely, on a
//!   build it served — there the perturbed problem is infeasible and no
//!   derivative exists. `ill_conditioned()` is the only thing that tells the
//!   caller, **after the step and never at build time**; the measured
//!   separation is residual `0.5` / `0.333` / `0.8` across the three shapes in
//!   this file against `~1e-13` on a correct step, with no overlap.
//! - **The guard's arithmetic about a bound** is covered twice over, at two
//!   levels, and neither is redundant.
//!   `sensitivity::tests::an_active_bound_is_not_stacked_into_the_apex_rank`
//!   owns the rank arithmetic on constructed rows with no solver;
//!   `a_bound_pinned_apex_is_served_by_fix_relax_and_flagged_without_it` owns
//!   the end-to-end consequence on a real (necessarily degenerate) model, and
//!   is the only test here that shows what the bound-exclusion *buys*.
//!   `an_active_bound_does_not_count_as_an_apex_pin` reaches the line on a
//!   non-degenerate model and says of itself that it does not discriminate.
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
//! | the `apex_can_absorb_db` guard disabled | `an_apex_that_cannot_absorb_db_is_refused` — **and only it** | run after the guard landed. It fails with the defect in the message (`du/db₀ = 0.5`, `|A·dx − db| = 5e-10`) rather than a bare assertion, so a regression reads as what it is |
//! | `apex_can_absorb_db` stacks a unit row per active bound (the rule as first written) | `sensitivity::tests::an_active_bound_is_not_stacked_into_the_apex_rank` — **and only it**, across all 267 tests in the crate | re-review of #889. The mutation changes the signature, so `finish` and both unit call sites move with it; it is still compile-checkable, just not a one-line edit |
//! | the same stacking applied at the **call site** in `finish`, leaving the signature alone | `a_bound_pinned_apex_is_served_by_fix_relax_and_flagged_without_it` — **and only it** | two measurements, and the first one's *conclusion* was wrong. Run before that fixture existed, this mutation left the whole crate green, and the table here concluded "no non-degenerate model can separate the two rules, so an integration fixture cannot carry this line". The counting argument is sound — discrimination forces `[A; B; bounds]` to be dependent, i.e. primal degeneracy — but *must be degenerate* is not *cannot be written*: degenerate is ordinary. The third review of #889 wrote the model, it is four variables long, and the mutation is red on it. The green run measured this crate's corpus, not a possibility |
//! | `apex_can_absorb_db` reverts to the dimension count `n − rank(B) ≥ rank(A)` | `an_equality_inside_the_pinned_coordinates_is_refused`, plus `sensitivity::tests::{the_criterion_is_exact_not_a_dimension_count, apex_can_absorb_db_is_a_dimension_count}` — **and nothing else in the crate** | fourth review of #889. The count is implied by the exact criterion and strictly weaker: it passes a model whose equality lives entirely inside the pinned coordinates, where `A(ker B) = {0}` and no `db` is reachable at all. Served, it returned `dx/db = (⅓,⅓,0,0)` at 33% error, identically at every step size |
//! | `row_rank` uses a global max instead of equilibrating each row | `sensitivity::tests::row_rank_is_not_fooled_by_one_huge_row` — **and only it** | the global scale was harmless while these rows were cone faces; `A` joined them and `A`'s row scaling is the user's. Every other fixture in the crate is unit-scaled, where the two readings coincide — so this test had to be built to make the mutation bite (a modest pivot next to a `1e10` row), not just picked |
//! | `primal_scale` replaced by a bare `1.0` in `build_conic` | `the_apex_decision_is_relative_to_the_problems_scale` — **and nothing else** | that test was written *because* the first run of this mutation was green across the whole crate: every other fixture here is `O(1)`, where the relative and absolute readings coincide. A scale-convention change with no failing test is exactly `/sens-review` entry 3 |

use pounce_convex::QpOptions;
use pounce_convex::cones::ConeSpec;
use pounce_convex::ipm::solve_socp_ipm;
use pounce_convex::qp::{QpProblem, QpSolution, QpStatus, Triplet};
use pounce_convex::sensitivity::{ConeBlockKind, QpSensitivity, SensError};
use pounce_feral::FeralSolverInterface;
use pounce_linsol::SparseSymLinearSolverInterface;
use pounce_sens_core::boundcheck::RefineStop;

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

/// **The hard half of the apex branch**, and the one no fixture reached.
///
/// `apex()` is the *easy* half: its equality is `x₀ + x₂ = b` with `x₂`
/// **outside** the cone block, so the perturbation is absorbed by a variable
/// the apex never pinned. Every apex assertion in this crate was shaped that
/// way — here, in `convex_cone_sensitivity.rs`, and in
/// `conic_sensitivity_refused.rs`. So the branch was never untested; the
/// branch's *other side* was, which is a sharper statement and exactly what
/// `/sens-review` entry 6 is about: "a fixture that always takes one branch
/// says nothing about the other, and it stays green while the other branch is
/// broken."
///
/// The fixture is the parametric distance function, `min t s.t. u = b₀,
/// v = b₁, (t, u, v) ∈ Q₃` — about as ordinary as a parametric SOCP gets. Both
/// equalities pin coordinates the apex also pins, so `ker(B_a)` has dimension
/// zero while `m_eq = 2`: no `dx` satisfies `A·dx = db` at all.
///
/// Before the guard this returned `du/db₀ = 0.5`, where **primal feasibility
/// alone** forces `1` — a least-squares compromise between "`u` must move with
/// `b₀`" and "`u` must stay at the apex". Found by adversarial review of #889,
/// judged against the oracle-free identity `A·dx = db`; that identity needs no
/// cone, no dual and no re-solve, which is why it can convict a step for which
/// no solver oracle was accurate enough to adjudicate.
///
/// Two things make this a refusal rather than a caveat. The problem is
/// **smooth on both sides of the cliff**: at `‖b‖ = 1.12e-8` the true
/// derivative still exists, is still `0.894427`, and the boundary branch finds
/// it — only the classifier changes its mind. And `primal_scale` floors at
/// `1.0`, so `CONE_APEX_REL` is an *absolute* `1e-8` for any model whose data
/// is `O(1)` or smaller: a well-fitting least-squares model, or one in units
/// where the optimal residual is naturally `~1e-9`, lands here.
#[test]
fn an_apex_that_cannot_absorb_db_is_refused() {
    // b = (2, 1)·k, so the exact dt/db₀ is 2/√5 = 0.894427 at every k.
    let model = |k: f64| QpProblem {
        n: 3,
        p_lower: vec![],
        c: vec![1.0, 0.0, 0.0],
        a: vec![tri(0, 1, 1.0), tri(1, 2, 1.0)],
        b: vec![2.0 * k, k],
        // s = (t, u, v) ∈ SOC(3), as h − Gx with h = 0.
        g: vec![tri(0, 0, -1.0), tri(1, 1, -1.0), tri(2, 2, -1.0)],
        h: vec![0.0, 0.0, 0.0],
        lb: vec![],
        ub: vec![],
    };
    let delta = 1e-9;

    // Above the cliff the block is a boundary face and the answer is right.
    // This half must keep working, or the guard is just breaking the feature.
    let prob = model(1e-6);
    let sol = solve(&prob);
    let mut sens = sens_for(&prob, &sol);
    assert_eq!(sens.cone_block_kinds(), [(0, ConeBlockKind::Boundary)]);
    let dx = sens.parametric_step(&[0], &[delta]);
    assert!(
        (dx[0] / delta - 2.0 / 5.0_f64.sqrt()).abs() < 1e-6,
        "above the cliff the derivative exists and must be found: got {}",
        dx[0] / delta
    );
    assert!(
        (dx[1] / delta - 1.0).abs() < 1e-9,
        "A·dx = db is primal feasibility alone, so du/db₀ is exactly 1; got {}",
        dx[1] / delta
    );

    // Below it the block classifies Apex and pins all three coordinates, so
    // `ker(B_a)` is trivial and no step can satisfy `A·dx = db`.
    for k in [1e-9, 1e-16] {
        let prob = model(k);
        let sol = solve(&prob);
        match QpSensitivity::build_conic(&prob, &SOC3, &sol, &opts(), 1e-7, backend) {
            Err(SensError::ActiveSetOverdetermined { block, what }) => {
                assert_eq!(block, 0);
                assert!(
                    what.contains("absorb"),
                    "the refusal must name the reason; got {what:?}"
                );
            }
            Err(other) => panic!("k = {k:e} must be refused as unabsorbable, got {other:?}"),
            Ok(mut sens) => {
                // Report what the accepted step actually does, so a regression
                // reads as the defect rather than as a bare assertion failure.
                let dx = sens.parametric_step(&[0], &[delta]);
                panic!(
                    "k = {k:e} was ACCEPTED at the apex. du/db₀ = {} where primal \
                     feasibility alone forces 1, and |A·dx − db| = {:e}. That is the \
                     least-squares compromise this guard exists to refuse.",
                    dx[1] / delta,
                    (dx[1] - delta).abs().max(dx[2].abs())
                );
            }
        }
    }
}

/// **An active variable bound must not count as a pin here** — the guard's most
/// delicate line, and the one an ordinary fixture cannot convict.
///
/// Raised in re-review of #889. The first version of `apex_can_absorb_db`
/// stacked one unit row per active bound into the rank, on the reasoning that
/// "an active bound pins its coordinate exactly as a face row does". For
/// `parametric_step` that is true. But `release_slots` exists precisely so
/// `parametric_step_bounded` can *open* an active bound — so counting one here
/// refuses the whole build, fix-relax included, for a model fix-relax could
/// serve. A cone face row has no such escape; a bound does. The rank is
/// therefore taken over the un-releasable rows only.
///
/// # What this fixture proves, and what it does not
///
/// It proves the line is **reached with a bound present** and that the build is
/// served there: an apex block, an active lower bound on a variable outside it,
/// and the correct `dx₂/db = 1`. Before this there was no fixture in the crate
/// with both an apex and an active bound — the file header says every fixture
/// here is bound-free, and that was true.
///
/// It does **not** discriminate between the two rules, and saying otherwise
/// would be the exact failure `/sens-review` entry 6 describes. Here
/// `rank(active_rows) = 3` and `n − 3 = 2 ≥ m_eq = 1`; stacking the bound row
/// gives rank 4 and `n − 4 = 1 ≥ 1`. Both rules serve it.
///
/// # Why a discriminating *integration* fixture must be degenerate
///
/// Discrimination needs `rank(B) ≤ n − m_eq` while
/// `rank(B ∪ bounds) ≥ n − m_eq + 1`. Stack `A` on top of that:
/// `m_eq + (n − m_eq + 1) = n + 1 > n`, so `[A; B; bounds]` **must** carry a
/// linear dependency — some equality row lies in the span of the active
/// constraints. That is primal degeneracy by definition, not a fixture that
/// happens to be awkward: it is the only shape in which the two rules can
/// disagree.
///
/// Degenerate is ordinary, not unreachable, and
/// `a_bound_pinned_apex_is_served_by_fix_relax_and_flagged_without_it` below
/// is that model — four variables, contributed by the third review of #889
/// after this file claimed such a fixture could not be written. It is the one
/// that discriminates. `sensitivity::tests::an_active_bound_is_not_stacked_into_the_apex_rank`
/// owns the same question at the level of the rank arithmetic, on constructed
/// rows with no solver, which is cheaper and sharper about *what* is computed.
/// This fixture is the third leg: the line reached on a model that is **not**
/// degenerate, where both rules agree.
///
/// Note also what this is not: it does not exercise the release path itself
/// (`convex_sens_release.rs` owns that, on orthant models). It exercises the
/// guard's treatment of a bound, which is the part that was unreached.
#[test]
fn an_active_bound_does_not_count_as_an_apex_pin() {
    // `apex()`'s geometry — variables (x₀, x₁, x₂, t) with the cone on
    // (t, x₀, x₁) and the equality x₀ + x₂ = 1 — plus a fifth variable `u`
    // carrying an active lower bound, so the guard sees a bound row.
    let prob = QpProblem {
        n: 5,
        p_lower: vec![
            tri(0, 0, 1.0),
            tri(1, 1, 1.0),
            tri(2, 2, 1.0),
            tri(4, 4, 1.0),
        ],
        c: vec![0.0, 0.0, -1.0, 5.0, 0.5],
        a: vec![tri(0, 0, 1.0), tri(0, 2, 1.0)],
        b: vec![1.0],
        g: vec![tri(0, 3, -1.0), tri(1, 0, -1.0), tri(2, 1, -1.0)],
        h: vec![0.0, 0.0, 0.0],
        lb: vec![-1e19, -1e19, -1e19, -1e19, 1.0],
        ub: vec![1e19, 1e19, 1e19, 1e19, 1e19],
    };
    let sol = solve(&prob);
    let mut sens = sens_for(&prob, &sol);
    assert_eq!(
        sens.cone_block_kinds(),
        [(0, ConeBlockKind::Apex)],
        "the fixture must reach the apex branch, or it tests nothing about the guard"
    );
    assert_eq!(
        sens.active_bound_vars(),
        [4],
        "and it must carry an active bound, or the line under test is still unreached"
    );

    // Served, and served correctly: with x₀ pinned at the apex the whole
    // perturbation goes to x₂, so dx₂/db = 1 exactly — primal feasibility
    // alone, the oracle-free identity `A·dx = db`.
    let delta = 1e-6;
    let dx = sens.parametric_step(&[0], &[delta]);
    assert!(
        (dx[2] / delta - 1.0).abs() < 1e-9,
        "x₀ is pinned by the apex, so x₂ must absorb all of db: got dx₂/db = {}",
        dx[2] / delta
    );
}

/// **The bound-exclusion's benefit, as a number** — and the fixture whose
/// absence I wrongly called an impossibility.
///
/// Contributed by the third review of #889. The previous round's mutation
/// table concluded, from the call-site mutation running green, that "no
/// non-degenerate model can separate the two rules, so an integration fixture
/// *cannot* carry this line". The counting argument behind that is right —
/// discrimination forces `[A; B; bounds]` to be linearly dependent, which is
/// primal degeneracy — but the step from *must be degenerate* to *cannot be
/// written* does not follow. Degenerate is ordinary. This model is degenerate
/// in exactly that way (`A = e₀ + e₂` is the `x₀` cone row plus the `x₂` bound
/// row) and is four variables long.
///
/// ```text
///   min ½‖x‖² + push·x₂ + tcost·t
///   s.t.  x₀ + x₂ = 1,  (t, x₀, x₁) ∈ Q₃,  x₂ ≥ 1
/// ```
///
/// `rank(active_rows) = 3` and `4 − 3 = 1 ≥ m_eq`, so it is served. Stack the
/// bound row: rank 4, `4 − 4 = 0 < 1`, and the rule as first written **refused
/// it at build time** — taking `parametric_step_bounded` away with it. That is
/// the `release_slots` argument as a measurement rather than a claim, and it
/// is the first evidence in this crate of what excluding bounds *buys*;
/// `an_active_bound_does_not_count_as_an_apex_pin` only shows the line is
/// reached.
///
/// # Three things it pins at once
///
/// 1. **Fix-relax serves it exactly.** `parametric_step_bounded` reproduces
///    the re-solve — `dx₂/db = 1`, the whole perturbation on the one variable
///    free to take it.
/// 2. **The plain step does not**, and is wrong by a third of the
///    perturbation: it splits `db` between `x₀` and `x₂` and misses primal
///    feasibility, which needs no cone, no dual and no oracle to judge.
/// 3. **`ill_conditioned()` catches that**, which is the fallback the whole
///    bound-exclusion rests on — and it is the **second** clause that fires,
///    *after* a step. At build time `kkt_cond_estimate()` is `3.0e10`,
///    comfortably under the `1e14` threshold, so a caller who checks
///    `ill_conditioned()` straight after `build_conic` gets `false` and then
///    takes the wrong step. The assembled KKT really is well conditioned; what
///    is wrong is that it holds a row the perturbation forces off, and only
///    the residual sees that.
///
/// Assertion 3 is the one to keep: the day someone widens
/// `STEP_UNRELIABLE_RESIDUAL` past `1/3`, this goes red rather than silent.
#[test]
fn a_bound_pinned_apex_is_served_by_fix_relax_and_flagged_without_it() {
    // Swept over the shape's free parameters, because a single point cannot
    // tell a structural result from a coincidence of one objective.
    for (push, tcost) in [(0.5, 1.0), (1.0, 5.0), (2.0, 5.0)] {
        let prob = QpProblem {
            n: 4,
            p_lower: vec![
                tri(0, 0, 1.0),
                tri(1, 1, 1.0),
                tri(2, 2, 1.0),
                tri(3, 3, 1.0),
            ],
            c: vec![0.0, 0.0, push, tcost],
            a: vec![tri(0, 0, 1.0), tri(0, 2, 1.0)],
            b: vec![1.0],
            g: vec![tri(0, 3, -1.0), tri(1, 0, -1.0), tri(2, 1, -1.0)],
            h: vec![0.0, 0.0, 0.0],
            lb: vec![-1e19, -1e19, 1.0, -1e19],
            ub: vec![1e19, 1e19, 1e19, 1e19],
        };
        let sol = solve(&prob);
        let mut sens = sens_for(&prob, &sol);

        // The shape must actually be the discriminating one, or the rest of
        // this test is about some other model.
        assert_eq!(sens.cone_block_kinds(), [(0, ConeBlockKind::Apex)]);
        assert_eq!(sens.active_bound_vars(), [2]);
        assert!(
            sol.z_lb[2] > 1.0,
            "the bound must be genuinely active, not marginal: z_lb₂ = {}",
            sol.z_lb[2]
        );

        // The build-time reading is clean — this is the trap, not an aside.
        assert!(
            sens.kkt_cond_estimate() < 1e14,
            "the regularized KKT is well conditioned here; got {:e}",
            sens.kkt_cond_estimate()
        );
        assert!(
            !sens.ill_conditioned(),
            "and so `ill_conditioned()` is false until a step is taken"
        );

        let delta = 1e-6;

        // (2) the plain step misses primal feasibility by a third.
        let dx = sens.parametric_step(&[0], &[delta]);
        let feas = ((dx[0] + dx[2]) - delta).abs() / delta;
        assert!(
            (feas - 1.0 / 3.0).abs() < 1e-9,
            "the plain step must miss `A·dx = db` by a third; got {feas:e}"
        );

        // (3) …and the residual clause catches it, well clear of threshold.
        assert!(
            sens.ill_conditioned(),
            "the fallback the bound-exclusion rests on must fire after the step"
        );
        let resid = sens
            .last_step_residual()
            .expect("a step was taken, so there is a residual");
        assert!(
            resid > 1e-2,
            "and not marginally: {resid} against STEP_UNRELIABLE_RESIDUAL = 1e-6"
        );

        // (1) fix-relax reproduces the re-solve exactly.
        let (dxb, pinned, stop) = sens
            .parametric_step_bounded(&[0], &[delta], 1e-9, 20)
            .expect("fix-relax must serve the model the old rule refused");
        assert_eq!(stop, RefineStop::Settled);
        assert!(
            !pinned.is_empty(),
            "the bound must be pinned by the refinement"
        );
        assert!(
            (dxb[2] / delta - 1.0).abs() < 1e-9,
            "x₂ is the only variable free to absorb db: got dx₂/db = {}",
            dxb[2] / delta
        );
        assert!(
            (dxb[0] / delta).abs() < 1e-9,
            "and x₀ is pinned at the apex: got dx₀/db = {}",
            dxb[0] / delta
        );
    }
}

/// **The equality side is `rank(A)`, not `A`'s row count** — the third
/// direction of coarseness, on a solved model.
///
/// Raised in the third review of #889, where I first recorded it as a known
/// limitation rather than fixing it, on the grounds that no redundant-equality
/// fixture existed. That is a reason to write one, not a reason to ship the
/// coarser rule, and this is it: [`apex`] with its single equality written
/// twice, the second copy scaled by 2.
///
/// `rank(active_rows) = 3` and `n − 3 = 1`, so the row count `m_eq = 2`
/// refuses and `rank(A) = 1` serves. Serving is right: a redundant equality
/// does not shrink the space a step must reach. The reachable perturbations
/// are `range(A)`, of dimension 1 — and a `db` outside it (perturbing `b₀`
/// alone here) makes the *perturbed problem infeasible*, which is a different
/// failure from the derivative being unrepresentable and not one this guard
/// exists to report.
///
/// So the step is taken along `db = (δ, 2δ) ∈ range(A)`, and the answer is the
/// unperturbed model's: `x₀` is pinned at the apex, so `x₂` takes all of it.
///
/// Mutation: compare against `eq_rows.len()`. This goes red as
/// `ActiveSetOverdetermined`, which is what it did before the fix.
#[test]
fn an_apex_with_a_redundant_equality_is_served() {
    let prob = QpProblem {
        a: vec![
            tri(0, 0, 1.0),
            tri(0, 2, 1.0),
            tri(1, 0, 2.0),
            tri(1, 2, 2.0),
        ],
        b: vec![1.0, 2.0],
        ..apex(1.0)
    };
    let sol = solve(&prob);
    let mut sens = sens_for(&prob, &sol);
    assert_eq!(
        sens.cone_block_kinds(),
        [(0, ConeBlockKind::Apex)],
        "the fixture must reach the apex branch, or the guard is never consulted"
    );

    let delta = 1e-6;
    let dx = sens.parametric_step(&[0, 1], &[delta, 2.0 * delta]);
    assert!(
        (dx[2] / delta - 1.0).abs() < 1e-9,
        "x₀ is pinned at the apex, so x₂ absorbs db: got dx₂/db₀ = {}",
        dx[2] / delta
    );
    assert!(
        (dx[0] / delta).abs() < 1e-9,
        "and x₀ does not move: got dx₀/db₀ = {}",
        dx[0] / delta
    );
    assert!(
        !sens.ill_conditioned(),
        "a served build whose step is exact must not be flagged"
    );

    // **The other branch**, and the one the doc's argument rests on. Serving
    // this build is justified by "a `db` outside `range(A)` makes the
    // *perturbed problem* infeasible, which is not this guard's to report" —
    // but whether the caller can *tell* is reportable, and was unmeasured
    // until the fourth review of #889 asked (`/sens-review` entry 6).
    //
    // Perturbing `b₀` alone gives `db = (δ, 0) ∉ range(A) = span{(1,2)}`. What
    // comes back is a least-squares answer to an unanswerable question — and
    // `ill_conditioned()` fires on it, residual `0.8` against a `1e-6`
    // threshold. That is the sentence the doc was missing, and it is what makes
    // serving the build defensible rather than merely permitted.
    let dx_off = sens.parametric_step(&[0], &[delta]);
    assert!(
        (dx_off[2] / delta - 1.0).abs() > 0.1,
        "an infeasible db cannot be answered correctly; got dx₂/db₀ = {}",
        dx_off[2] / delta
    );
    assert!(
        sens.ill_conditioned(),
        "…and the caller must be able to tell: ill_conditioned() has to fire"
    );
    let resid = sens
        .last_step_residual()
        .expect("a step was taken, so there is a residual");
    assert!(
        resid > 1e-2,
        "and not marginally: {resid} against STEP_UNRELIABLE_RESIDUAL = 1e-6"
    );
}

/// **An equality living inside the coordinates the apex pins is refused** —
/// the case the *dimension count* let through with a wrong answer.
///
/// Found by the fourth review of #889, and it is the guard's own motivating
/// defect one model past the guard. Take [`apex`] and move its equality onto
/// `(x₀, x₁)`, both of which the apex pins:
///
/// ```text
///   min ½‖x‖² − x₂ + 5t   s.t.  x₀ + x₁ = b,  (t, x₀, x₁) ∈ Q₃
/// ```
///
/// `n = 4`, `rank(A) = 1`, `rank(B) = 3`, so `n − rank(B) = 1 ≥ 1` and the
/// dimension count **passes**. But `ker(B) = span(e₂)` and `A e₂ = 0`, so
/// `A(ker B) = {0}`: *no* nonzero `db` is reachable at all. Measured before the
/// fix, the build was served and returned
///
/// ```text
///   dx/db = (⅓, ⅓, 0, 0)      |A·dx − db| / δ = 0.333
/// ```
///
/// identically at `δ = 1e-4, 1e-5, 1e-6` — a least-squares compromise reported
/// as a derivative, 33% off, at every step size. This is not the "subtler
/// dependency" a coarse rule may fairly leave to a residual flag: the equality
/// lives *entirely inside* the pinned coordinates.
///
/// The exact criterion `rank([A;B]) == rank(A) + rank(B)` refuses it —
/// `3 ≠ 1 + 3` — and reproduces every other verdict in the crate, which is why
/// it replaced the count rather than supplementing it.
///
/// Mutation: `n.saturating_sub(rank_b) >= rank_a`. This test goes red as
/// "was accepted", with the wrong derivative in the message.
#[test]
fn an_equality_inside_the_pinned_coordinates_is_refused() {
    let prob = QpProblem {
        a: vec![tri(0, 0, 1.0), tri(0, 1, 1.0)],
        b: vec![0.0],
        ..apex(0.0)
    };
    let sol = solve(&prob);
    match QpSensitivity::build_conic(&prob, &SOC3, &sol, &opts(), 1e-7, backend) {
        Err(SensError::ActiveSetOverdetermined { block, what }) => {
            assert_eq!(block, 0);
            assert!(
                what.contains("absorb"),
                "the refusal must name the reason; got {what:?}"
            );
        }
        Err(other) => panic!("must be refused as unabsorbable, got {other:?}"),
        Ok(mut sens) => {
            // Report the defect rather than a bare assertion, so a regression
            // reads as what it is.
            let delta = 1e-4;
            let dx = sens.parametric_step(&[0], &[delta]);
            panic!(
                "ACCEPTED. dx/db = ({}, {}, {}, {}) and |A·dx − db|/δ = {:e}, where \
                 A(ker B) = {{0}} means no nonzero db is reachable at all. That is \
                 the least-squares compromise the exact criterion exists to refuse.",
                dx[0] / delta,
                dx[1] / delta,
                dx[2] / delta,
                dx[3] / delta,
                ((dx[0] + dx[1]) - delta).abs() / delta
            );
        }
    }
}

/// **`reduced_hessian` on a curved face** — the method round 5 of #889 found
/// returning a plausible wrong number with `Ok`.
///
/// Two defects, one call. `active_ineq` is **provenance** on a conic build (the
/// `G` rows a block contributed); the active object is the face's own row, a
/// combination of them. `reduced_hessian` indexed raw `G` with the provenance
/// index — gh#450's shape, and the `active_rows` field doc had claimed this
/// loop read `active_rows` since the row was introduced. Separately it
/// projected bare `P` where a curved face makes the second-order object the
/// Hessian of the **Lagrangian**.
///
/// Measured on `boundary(1.0)` with `P = diag(1, 1, 9)`, chosen so the third
/// coordinate's curvature cannot hide:
///
/// ```text
///   wrong G row, no curvature   1.0000     <- what shipped, with Ok
///   correct face row, no curv   1.0506
///   correct face row + curv     9.9368     <- correct
/// ```
///
/// The middle line is why the index fix alone is not enough to call this
/// closed, and the spread is why it is not a rounding matter.
///
/// # Why the curvature belongs, by parity rather than by taste
///
/// The NLP arm computes `H_R = B K⁻¹ Bᵀ` off the converged KKT, whose `(x,x)`
/// block is the Lagrangian Hessian — so it has always included constraint
/// curvature, and this method's own doc says it mirrors that. On the orthant
/// path `curvature` is empty and this is exactly `P`, which is the classical
/// definition: a linear constraint has no Hessian.
///
/// The expected value is not taken from the implementation. The test builds
/// the face row and the curvature from the *solution* — `w = (1, −ŝ₁)`,
/// `(ν/s₀)(Σ gᵣgᵣᵀ − uuᵀ)` — forms `Z` by hand, and evaluates `Zᵀ(P+curv)Z`.
/// Both routes agree to every digit.
///
/// Mutations: read `active_ineq` + raw `G` again (→ 1.0000), or drop the
/// curvature term (→ 1.0506). Each reddens this test and nothing else.
#[test]
fn the_reduced_hessian_reads_the_face_row_and_its_curvature() {
    let prob = QpProblem {
        p_lower: vec![tri(0, 0, 1.0), tri(1, 1, 1.0), tri(2, 2, 9.0)],
        ..boundary(1.0)
    };
    let sol = solve(&prob);
    let sens = sens_for(&prob, &sol);
    assert_eq!(
        sens.cone_block_kinds(),
        [(0, ConeBlockKind::Boundary)],
        "the fixture must reach a curved face, or it tests nothing"
    );

    let rh = sens
        .reduced_hessian(1e-9)
        .expect("a converged boundary build must produce a reduced Hessian");
    assert_eq!(rh.eigenvalues.len(), 1, "n − rank(B) = 3 − 2 = 1");

    // ---- the independent hand projection ----
    let (t, x0, x1) = (sol.x[2], sol.x[0], sol.x[1]);
    let nrm = (x0 * x0 + x1 * x1).sqrt();
    let (h0, h1) = (x0 / nrm, x1 / nrm); // ŝ₁
    // face row = wᵀG with w = (1, −ŝ₁); G's rows are (0,0,−1), (−1,0,0), (0,−1,0)
    let face = [h0, h1, -1.0];
    // Z spans null([A; face]) with A = (1,1,0): v = (1, −1, c), c = f₀ − f₁
    let v = [1.0, -1.0, face[0] - face[1]];
    let vn = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    let z = [v[0] / vn, v[1] / vn, v[2] / vn];
    // curvature = (ν/s₀)(Σ_{r≥1} gᵣgᵣᵀ − uuᵀ), u = Σ ŝᵣ gᵣ
    let u = [-h0, -h1, 0.0];
    let mut curv = [[0.0f64; 3]; 3];
    curv[0][0] = 1.0;
    curv[1][1] = 1.0;
    for a in 0..3 {
        for b in 0..3 {
            curv[a][b] -= u[a] * u[b];
            curv[a][b] *= 1.0; // keep the loop shape obvious
        }
    }
    let scale = sol.z[0] / t;
    for a in 0..3 {
        for b in 0..3 {
            curv[a][b] *= scale;
        }
    }
    let pdiag = [1.0, 1.0, 9.0];
    let mut expected = 0.0;
    let mut without_curvature = 0.0;
    for a in 0..3 {
        without_curvature += z[a] * pdiag[a] * z[a];
        for b in 0..3 {
            let h = if a == b {
                pdiag[a] + curv[a][b]
            } else {
                curv[a][b]
            };
            expected += z[a] * h * z[b];
        }
    }

    assert!(
        (rh.eigenvalues[0] - expected).abs() < 1e-9,
        "the reduced Hessian must be Zᵀ(P + face curvature)Z: got {}, hand-computed {expected}",
        rh.eigenvalues[0]
    );

    // And the two defects this replaced are far away, so a regression to
    // either is a failure rather than a rounding difference.
    assert!(
        (rh.eigenvalues[0] - 1.0).abs() > 1.0,
        "1.0 is the value for the WRONG G row with no curvature"
    );
    assert!(
        (rh.eigenvalues[0] - without_curvature).abs() > 1.0,
        "and {without_curvature} is the right row with the curvature dropped"
    );
}

/// **What `activity()` says about a cone row, measured** — so the caveat on
/// that accessor is a number rather than a warning.
///
/// `classify_all` is the orthant rule: it reads each `G` row as an inequality
/// with its own slack and multiplier. A cone block complements as a *block*
/// inner product, so row-wise `sᵢ·zᵢ` is generally nonzero and `z` generally
/// carries negative entries — a cone row's entry is a reading of a quantity
/// that does not exist, not a wrong reading of a real one.
///
/// This pins the actual output on the boundary fixture so the doc cannot drift
/// from it, and so that anyone who later makes `activity()` cone-aware has to
/// come here and say so. Round 5 of #889.
#[test]
fn activity_reads_cone_rows_with_the_orthant_rule_and_says_so() {
    let prob = boundary(1.0);
    let sol = solve(&prob);
    let sens = sens_for(&prob, &sol);
    assert_eq!(sens.cone_block_kinds(), [(0, ConeBlockKind::Boundary)]);

    // The dual's tail is negative — legal for a cone, impossible for an
    // orthant row, which is precisely why the orthant rule cannot read it.
    assert!(
        sol.z[1] < 0.0 && sol.z[2] < 0.0,
        "the fixture must have a negative cone tail, or it shows nothing: z = {:?}",
        sol.z
    );

    let rep = sens.activity();
    assert_eq!(
        rep.row_status.len(),
        3,
        "one entry per G row, cone rows included — there is no gap to notice"
    );
    // The binding block's tail rows read as *inactive*. That is the
    // plausible-and-wrong answer the accessor's doc warns about.
    assert_ne!(
        rep.row_status[1], rep.row_status[0],
        "the block is one object, yet its rows get different orthant verdicts"
    );
}

/// **The weak-activity screens run on a conic build's orthant rows.**
///
/// Both screens were `Vec::new()` on every conic build until round 5 of #889,
/// while their docs promised a "deliberately conservative" screen. A `Nonneg`
/// block inside a mixed partition is an ordinary orthant block and its rows are
/// ordinary rows.
///
/// A *cone* block still never appears, and that is by construction:
/// `cone_block_face` refuses a collapsed dual at the apex or on the boundary —
/// the conic analogue of weak activity — so no built conic sensitivity carries
/// one. The refusal is the report.
///
/// # The fixture has to be non-empty, or it proves nothing
///
/// The first version of this test asserted the screens were *empty* on the
/// strictly complementary `boundary` fixture. That is true with the screens
/// wired and equally true with them stubbed to `Vec::new()` — the mutation ran
/// **green**, which is the exact failure this file's header warns about, made
/// on the test written to close it.
///
/// So the fixture is gh#219's weakly active row carried as a `Nonneg` block
/// beside a strictly interior second-order block: `min ½‖x‖² − 5t` subject to
/// `x₀ + x₁ = 1`, `x₀ − 2x₁ ≤ −½`, `(t, u, v) ∈ Q₃`. The equality-only optimum
/// `(½, ½)` hits the inequality exactly, so its slack and its multiplier
/// vanish together, and the screen must name row 0.
#[test]
fn the_weak_screens_are_wired_on_a_conic_build() {
    let prob = QpProblem {
        n: 5,
        p_lower: (0..5).map(|j| tri(j, j, 1.0)).collect(),
        c: vec![0.0, 0.0, -5.0, 0.0, 0.0],
        a: vec![tri(0, 0, 1.0), tri(0, 1, 1.0)],
        b: vec![1.0],
        g: vec![
            // row 0: the orthant row, weakly active at (½, ½)
            tri(0, 0, 1.0),
            tri(0, 1, -2.0),
            // rows 1..3: s = (t, u, v), driven strictly interior by the −5t cost
            tri(1, 2, -1.0),
            tri(2, 3, -1.0),
            tri(3, 4, -1.0),
        ],
        h: vec![-0.5, 0.0, 0.0, 0.0],
        lb: vec![],
        ub: vec![],
    };
    let cones = [ConeSpec::Nonneg(1), ConeSpec::SecondOrder(3)];
    let sol = solve_socp_ipm(&prob, &cones, &opts(), backend);
    assert_eq!(sol.status, QpStatus::Optimal);

    let sens = match QpSensitivity::build_conic(&prob, &cones, &sol, &opts(), 1e-7, backend) {
        Ok(v) => v,
        Err(e) => panic!("build_conic must accept this fixture, got {e:?}"),
    };
    assert_eq!(
        sens.cone_block_kinds(),
        [(1, ConeBlockKind::Interior)],
        "the cone must be interior so it contributes nothing and the orthant \
         row is the only thing the screen could name"
    );

    assert_eq!(
        sens.weakly_active_ineq(),
        [0],
        "the orthant row of a mixed partition must still be screened; this \
         returned empty on every conic build before round 5 of #889"
    );
}

/// **`build_conic` and `build` admit the same statuses**, which is what makes
/// the all-`Nonneg` delegation honest.
///
/// `build_conic` refuses non-`Optimal` and then, for an all-`Nonneg` partition,
/// hands straight to `build` with the comment "so the two entry points cannot
/// answer differently on the same input". That comment was false: the check
/// read `!= Optimal` while `build` admits `OptimalInaccurate` (gh#880), so the
/// same solution with the same partition was served by one and refused by the
/// other. Round 5 of #889.
///
/// This drives both entry points with an `OptimalInaccurate` status on an
/// all-orthant model and asserts they agree — which is the property the comment
/// claims, tested rather than asserted.
#[test]
fn both_entry_points_admit_the_same_statuses() {
    let prob = QpProblem {
        n: 2,
        p_lower: vec![tri(0, 0, 1.0), tri(1, 1, 1.0)],
        c: vec![-1.0, -1.0],
        a: vec![],
        b: vec![],
        g: vec![tri(0, 0, 1.0), tri(1, 1, 1.0)],
        h: vec![0.5, 0.5],
        lb: vec![],
        ub: vec![],
    };
    let mut sol = solve_socp_ipm(&prob, &[ConeSpec::Nonneg(2)], &opts(), backend);
    assert_eq!(sol.status, QpStatus::Optimal);

    // Relabel exactly as the sigma cascade does: the numbers are unchanged,
    // only the certification verdict is.
    sol.status = QpStatus::OptimalInaccurate;

    let plain = QpSensitivity::build(&prob, &sol, &opts(), 1e-7, backend);
    let conic =
        QpSensitivity::build_conic(&prob, &[ConeSpec::Nonneg(2)], &sol, &opts(), 1e-7, backend);
    assert_eq!(
        plain.is_ok(),
        conic.is_ok(),
        "the entry points must agree on OptimalInaccurate; plain = {:?}, conic = {:?}",
        plain.err(),
        conic.err()
    );
    assert!(
        plain.is_ok(),
        "and gh#880's decision is that they both serve it"
    );
}

/// The guard must not fire on an apex that *can* absorb `db`, or it would
/// refuse the flat, exact case the apex branch exists to serve.
///
/// `apex()`'s geometry stated as the property that matters: `m_eq = 1`, and
/// `x₂` sits outside the block, so `ker(B_a)` still has room.
#[test]
fn an_apex_that_can_absorb_db_is_still_served() {
    let prob = apex(1.0);
    let sol = solve(&prob);
    assert_eq!(
        sens_for(&prob, &sol).cone_block_kinds(),
        [(0, ConeBlockKind::Apex)],
        "the absorbable apex must still classify and build"
    );
}

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
