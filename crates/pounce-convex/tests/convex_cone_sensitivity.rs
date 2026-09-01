//! The remaining cone families: PSD at constant rank, exponential and power at
//! a facet interior.
//!
//! # Why this file exists
//!
//! `convex_soc_sensitivity.rs` established the shape — classify the *face* the
//! slack sits on, contribute that face's rows, and add its curvature to the
//! KKT's `(x,x)` block. This file carries the same shape to the three families
//! that were refused outright, and each brings a piece of geometry the
//! second-order cone does not have:
//!
//! | family | face | rows | curvature |
//! |---|---|---|---|
//! | `Psd(n)` at rank `r` | the constant-rank manifold | `q(q+1)/2`, `q = n−r` | `2 Σ_{l,k} (λ_k/a_l) c_lk c_lkᵀ` |
//! | `Exponential` | `φ = y·log(z/y) − x = 0`, `y, z > 0` | 1, `∇φᵀG` | `−ν Gᵀ∇²φ G`, rank one |
//! | `Power(α)` | `φ = y^α z^{1−α} − |x| = 0`, `y, z > 0` | 1, `∇φᵀG` | `−ν Gᵀ∇²φ G`, rank one |
//!
//! The PSD case is the one that is not just "another `φ`": its face has
//! codimension `q(q+1)/2`, so a `Psd(3)` block at rank 1 contributes **three**
//! rows, and its curvature comes from the Schur complement `C − BᵀA⁻¹B` rather
//! than from a single Hessian.
//!
//! # The oracle here is coarser than the second-order one, and why
//!
//! `convex_soc_sensitivity.rs` asserts *second-order* agreement with a re-solve
//! — shrink `δ` tenfold, watch the error fall a hundredfold. **That assertion
//! is not available on this file's fixtures**, and pretending otherwise would
//! be a test that fails for a reason unrelated to its name.
//!
//! The reason is the oracle, not the step. Measured on these fixtures, the base
//! solve itself lands `~3e-7` from the exact optimum on the PSD path and
//! `~1e-5` on the non-symmetric one (exponential and power route to the HSDE
//! driver, `crate::hsde_nonsym`, whose achieved accuracy is well short of the
//! symmetric IPM's). Divide that by `δ` and it swamps the `O(δ)` content the
//! second-order test needs to see. So the assertion here is a **central
//! difference at a fixed `δ = 1e-2`**, with a relative tolerance set off
//! measured populations rather than a round number:
//!
//! | | correct code | curvature dropped |
//! |---|---|---|
//! | `Psd(2)` rank 1 | `1.7e-4` | `3.3e-1` |
//! | `Psd(3)` rank 1 | `1.6e-4` | `3.3e-1` |
//! | `Exponential` | `1.8e-3` | `9.2e-2` |
//! | `Power(0.6)` | `2.6e-3` | `2.5e-1` |
//!
//! `ORACLE_REL = 1e-2` sits ~4× above the worst correct value and ~9× below the
//! best defective one. That band is narrower than this repo usually likes, and
//! it is narrow because of the solver underneath, not because of the rule under
//! test. Both columns are recorded here so the next reader can see the margin
//! rather than infer one.
//!
//! # What this file is NOT evidence about
//!
//! - **Second-order behaviour of any step here.** See above. The oracle's own
//!   noise is the floor; `the_curvature_is_not_optional` is what stands in for
//!   it, and it works by contrast rather than by convergence rate.
//! - **PSD blocks above `n = 3`, or rank deficiency above `q = 2`.** The
//!   curvature is `r·q` rank-one updates and the rows are `q(q+1)/2`; nothing
//!   here measures either at scale.
//! - **Chordal decomposition.** `crate::cones::chordal` splits a large PSD
//!   block before it reaches the solver; the partition `build_conic` is handed
//!   is whatever the caller passes, and this file does not exercise that path.
//! - **Release / fix-relax on a conic build.** Every fixture is bound-free, so
//!   `release_slots` is empty. `convex_sens_release.rs` owns that, on orthant
//!   models.
//! - **`Power(α)` away from `α ∈ {0.3, 0.6}`.** The `α → 0` and `α → 1` limits
//!   degenerate the facet and are untested.
//!
//! # Mutation evidence
//!
//! Each row was **run** — the mutation applied to
//! `crates/pounce-convex/src/sensitivity.rs`, compile-checked (a mutation that
//! does not build emits no failures and reads exactly like one nothing
//! catches), and the crate's suite run with `--no-fail-fast`.
//!
//! | mutation | red here | note |
//! |---|---|---|
//! | PSD curvature scale `2λ_k/a_l` → `0` | `the_psd_step_matches_a_resolve`, `the_psd3_step_matches_a_resolve` | the curvature is load-bearing on this family too |
//! | `λ_k / a_l` → `λ_k · a_l` | the same two | **and this is the row to read.** On the first draft of these fixtures it was **green across the whole crate**, because their surviving eigenvalue `a_l` was exactly `1.0` and the two expressions coincide there. The objective now puts it at `3`, and `the_psd_fixtures_have_a_nonunit_curvature_scale` pins that. A fixture uniform in the dimension the rule acts on reports nothing however sharp the assertion is |
//! | PSD tangent rows built from `range` instead of `kernel` | `the_psd3_fixture_has_a_two_dimensional_kernel`, both step tests | the face is `Vᵀ dX V = 0` for `V` spanning the *kernel*; the range is the orthogonal complement and gives a different manifold |
//! | `svec_sym_outer` drops the `√2` off-diagonal scaling | both step tests | the isometry is what makes `⟨out, w⟩ = uᵀ smat(w) v`; without it every off-diagonal contribution is off by `√2` |
//! | strict-complementarity check disabled | `a_psd_block_without_strict_complementarity_is_refused` | |
//! | PSD negative-eigenvalue guard disabled | `a_psd_slack_outside_the_cone_is_refused` | |
//! | `exp_face`'s `hess_factors` emptied | `the_exponential_step_matches_a_resolve` | |
//! | `power_face`'s `hess_factors` emptied | `the_power_step_matches_a_resolve` | |
//! | exponential `∇φ`'s `y` component drops its `− 1` | `the_exponential_step_matches_a_resolve`, `an_exponential_dual_off_the_normal_ray_is_refused`, `the_fixtures_take_every_branch`, `every_cone_family_reaches_a_face` | a wrong normal is caught twice over: the step moves *and* the dual stops being parallel to it, which is the dual-ray guard doing exactly its job on a bug in our own arithmetic rather than in the input |
//! | power `∇φ` swaps `α` and `1 − α` | `the_power_step_matches_a_resolve`, `the_fixtures_take_every_branch`, `every_cone_family_reaches_a_face` | same shape |
//! | the `y = 0` / `z = 0` guards disabled on both non-symmetric cones | `the_exponential_cones_degenerate_ray_is_refused`, `the_power_cones_degenerate_face_is_refused` | |
//! | the dual-ray check disabled | `an_exponential_dual_off_the_normal_ray_is_refused` | and only that: the converged fixtures pass it with three orders to spare, which is the margin `FACET_DUAL_REL` was set from |

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

/// The band the header's table justifies.
const ORACLE_REL: f64 = 1e-2;
/// Oracle step size. Larger is *better* here: the re-solve's own error is the
/// noise floor, so shrinking `δ` makes the central difference worse, not
/// better — the opposite of the usual situation.
const DELTA: f64 = 1e-2;

/// The symmetric driver reaches this; the PSD path runs on it.
fn opts() -> QpOptions {
    QpOptions {
        tol: 1e-11,
        ..Default::default()
    }
}

/// The non-symmetric HSDE driver does not. At `1e-11` two of the four
/// exponential/power fixtures return `OptimalInaccurate`; `1e-9` is what it
/// actually delivers, and asking for more would make the fixtures flaky rather
/// than the answers better.
fn ns_opts() -> QpOptions {
    QpOptions {
        tol: 1e-9,
        ..Default::default()
    }
}

fn opts_for(cones: &[ConeSpec]) -> QpOptions {
    if cones
        .iter()
        .any(|c| matches!(c, ConeSpec::Exponential | ConeSpec::Power(_)))
    {
        ns_opts()
    } else {
        opts()
    }
}

// ---------------------------------------------------------------------------
// Fixtures. Every one is the same skeleton: the cone block is the first few
// variables (`G = −I`, `h = 0`, so `s = x` and the geometry is readable
// straight off the solution), one spare variable `t`, one equality tying `t`
// to the block so a perturbation has somewhere to go, and a strictly convex
// objective so the KKT is nonsingular without crossover.
// ---------------------------------------------------------------------------

/// `n` cone rows on the first `n` variables, plus a spare `t`.
fn skeleton(dim: usize, c: Vec<f64>, eq_cols: &[usize], b: f64) -> QpProblem {
    let n = dim + 1;
    QpProblem {
        n,
        p_lower: (0..n).map(|j| tri(j, j, 1.0)).collect(),
        c,
        a: eq_cols.iter().map(|&j| tri(0, j, 1.0)).collect(),
        b: vec![b],
        g: (0..dim).map(|j| tri(j, j, -1.0)).collect(),
        h: vec![0.0; dim],
        lb: vec![],
        ub: vec![],
    }
}

/// `Psd(2)`: `S = smat(x₀, x₁, x₂)`. The objective pulls `S` toward
/// `diag(5, −1)`, which is not PSD, so the solve lands on the rank-1 face.
///
/// The `5` is load-bearing. It puts `S`'s surviving eigenvalue at **3**, and
/// the curvature divides by exactly that: `2 Σ (λ_k / a_l) c c ᵀ`. A first
/// draft used `1`, which sends `a_l` to `1.0` — and then `λ/a` and `λ·a` are
/// the same number, so a mutation swapping them was **green across the whole
/// crate**. A fixture that is uniform in the dimension the rule acts on
/// reports nothing however sharp the assertion is;
/// `the_psd_fixtures_have_a_nonunit_curvature_scale` is what keeps that from
/// coming back.
fn psd2(b: f64) -> QpProblem {
    skeleton(3, vec![-5.0, 0.0, 1.0, 0.0], &[0, 1, 3], b)
}

/// `Psd(3)` at rank 1 — kernel dimension 2, so the face has codimension 3 and
/// the block contributes **three** rows rather than one. This is the fixture
/// that distinguishes the PSD case from "another smooth facet".
fn psd3(b: f64) -> QpProblem {
    skeleton(6, vec![-5.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0], &[0, 1, 6], b)
}

/// `Psd(2)` with the objective pulling toward `diag(1, 1)`, which *is* PSD:
/// the cone is slack and must contribute nothing.
fn psd2_interior(b: f64) -> QpProblem {
    skeleton(3, vec![-1.0, 0.0, -1.0, 0.0], &[0, 1, 3], b)
}

/// `Exponential` on `(x₀, x₁, x₂)`, pulled outward until `φ = 0`.
fn exp_cone(b: f64) -> QpProblem {
    skeleton(3, vec![-1.0, -1.0, -1.0, 0.0], &[0, 1, 3], b)
}

/// `Power(0.6)`. The pull on `x₀` has to be stronger than the exponential
/// cone's to reach the facet — at `c₀ = −1` this fixture solves strictly
/// inside, which `a_slack_power_block_contributes_nothing` uses.
fn power_cone(b: f64) -> QpProblem {
    skeleton(3, vec![-4.0, -1.0, -1.0, 0.0], &[0, 1, 3], b)
}

/// [`power_cone`] with the pull weak enough that the cone stays slack.
fn power_interior(b: f64) -> QpProblem {
    skeleton(3, vec![-1.0, -1.0, -1.0, 0.0], &[0, 1, 3], b)
}

fn solve(prob: &QpProblem, cones: &[ConeSpec]) -> QpSolution {
    let sol = solve_socp_ipm(prob, cones, &opts_for(cones), backend);
    assert_eq!(
        sol.status,
        QpStatus::Optimal,
        "a fixture that does not solve makes every assertion below vacuous"
    );
    sol
}

fn sens_for(prob: &QpProblem, cones: &[ConeSpec], sol: &QpSolution) -> QpSensitivity {
    match QpSensitivity::build_conic(prob, cones, sol, &opts_for(cones), 1e-7, backend) {
        Ok(v) => v,
        Err(e) => panic!("build_conic must accept this fixture, got {e:?}"),
    }
}

/// `(x*(b+δ) − x*(b−δ)) / 2δ` from two independent re-solves — the outside
/// number. Central rather than one-sided because the `O(δ)` truncation of a
/// forward difference is the same size as the effect being measured at the
/// `δ` this file has to use.
fn central_difference(
    f: impl Fn(f64) -> QpProblem,
    cones: &[ConeSpec],
    b0: f64,
    n: usize,
) -> Vec<f64> {
    let up = solve(&f(b0 + DELTA), cones);
    let down = solve(&f(b0 - DELTA), cones);
    (0..n)
        .map(|j| (up.x[j] - down.x[j]) / (2.0 * DELTA))
        .collect()
}

fn relative_gap(step: &[f64], truth: &[f64]) -> f64 {
    let scale = truth.iter().fold(1e-12_f64, |m, x| m.max(x.abs()));
    step.iter()
        .zip(truth)
        .map(|(a, b)| (a / DELTA - b).abs())
        .fold(0.0_f64, f64::max)
        / scale
}

/// One family's oracle check, and the only assertion shape this file's solvers
/// support.
fn check_against_resolve(
    name: &str,
    f: impl Fn(f64) -> QpProblem,
    cones: &[ConeSpec],
    expect: ConeBlockKind,
) {
    let prob = f(1.0);
    let sol = solve(&prob, cones);
    let mut sens = sens_for(&prob, cones, &sol);
    assert_eq!(
        sens.cone_block_kinds(),
        [(0, expect)],
        "the `{name}` fixture must reach the `{expect:?}` face"
    );
    let step = sens.parametric_step(&[0], &[DELTA]);
    let truth = central_difference(f, cones, 1.0, prob.n);
    let gap = relative_gap(&step, &truth);
    assert!(
        gap < ORACLE_REL,
        "`{name}`: the step must agree with an independent central difference.\n  \
         step/δ {:?}\n  truth  {truth:?}\n  relative gap {gap:e} (allowed {ORACLE_REL:e})",
        step.iter().map(|v| v / DELTA).collect::<Vec<_>>()
    );
}

// ---------------------------------------------------------------------------
// Preconditions.
// ---------------------------------------------------------------------------

/// The replacement for the retired `build_conic_refuses_every_unsupported_family`.
/// Every family in `ConeSpec` now reaches a face rather than an error, and the
/// exhaustive `match` in `cone_block_face` is what keeps that true as families
/// are added: a new variant is a compile error, not a runtime refusal.
#[test]
fn every_cone_family_reaches_a_face() {
    let cases: Vec<(&str, Vec<ConeSpec>, QpProblem)> = vec![
        ("Psd(2)", vec![ConeSpec::Psd(2)], psd2(1.0)),
        ("Psd(3)", vec![ConeSpec::Psd(3)], psd3(1.0)),
        ("Exponential", vec![ConeSpec::Exponential], exp_cone(1.0)),
        ("Power(0.6)", vec![ConeSpec::Power(0.6)], power_cone(1.0)),
        (
            "SecondOrder(3)",
            vec![ConeSpec::SecondOrder(3)],
            soc_fixture(1.0),
        ),
        ("Nonneg(3)", vec![ConeSpec::Nonneg(3)], power_interior(1.0)),
    ];
    for (name, cones, prob) in cases {
        let sol = solve(&prob, &cones);
        assert!(
            QpSensitivity::build_conic(&prob, &cones, &sol, &opts_for(&cones), 1e-7, backend)
                .is_ok(),
            "every ConeSpec family must classify rather than refuse; {name} did not"
        );
    }
}

/// A second-order fixture, so the family list above is complete rather than
/// "everything except the one another file owns".
fn soc_fixture(b: f64) -> QpProblem {
    QpProblem {
        c: vec![-1.0, -0.2, 0.0, 0.0],
        ..skeleton(3, vec![], &[0, 1, 3], b)
    }
}

/// `/sens-review` entry 6. Each family's rule branches three ways, and a file
/// whose fixtures all take one branch stays green while the others are broken.
#[test]
fn the_fixtures_take_every_branch() {
    let cases: Vec<(&str, Vec<ConeSpec>, QpProblem, ConeBlockKind)> = vec![
        (
            "psd boundary",
            vec![ConeSpec::Psd(2)],
            psd2(1.0),
            ConeBlockKind::Boundary,
        ),
        (
            "psd interior",
            vec![ConeSpec::Psd(2)],
            psd2_interior(1.0),
            ConeBlockKind::Interior,
        ),
        (
            "exp boundary",
            vec![ConeSpec::Exponential],
            exp_cone(1.0),
            ConeBlockKind::Boundary,
        ),
        (
            "power boundary",
            vec![ConeSpec::Power(0.6)],
            power_cone(1.0),
            ConeBlockKind::Boundary,
        ),
        (
            "power interior",
            vec![ConeSpec::Power(0.6)],
            power_interior(1.0),
            ConeBlockKind::Interior,
        ),
    ];
    for (name, cones, prob, want) in cases {
        let sol = solve(&prob, &cones);
        let sens = sens_for(&prob, &cones, &sol);
        assert_eq!(
            sens.cone_block_kinds(),
            [(0, want)],
            "`{name}` must reach the `{want:?}` face"
        );
    }
    // The apex is hand-built — no interior-point solve stops on it.
    let (prob, sol) = hand_built(&[0.0, 0.0, 0.0], &[1.0, 0.0, 1.0]);
    let sens = QpSensitivity::build_conic(&prob, &[ConeSpec::Psd(2)], &sol, &opts(), 1e-7, backend)
        .expect("S = 0 with a live dual is the apex, a supported (flat) face");
    assert_eq!(sens.cone_block_kinds(), [(0, ConeBlockKind::Apex)]);
}

/// The curvature scale `a_l` — the surviving eigenvalue of `S` — must not be
/// `1`, or the rule's division by it is untested: `λ/a` and `λ·a` agree there,
/// and a mutation swapping them passes. Measured, not assumed.
#[test]
fn the_psd_fixtures_have_a_nonunit_curvature_scale() {
    for (name, cones, prob) in [
        ("Psd(2)", vec![ConeSpec::Psd(2)], psd2(1.0)),
        ("Psd(3)", vec![ConeSpec::Psd(3)], psd3(1.0)),
    ] {
        let sol = solve(&prob, &cones);
        // S = smat(x₀..), and x₀ is its surviving eigenvalue here because the
        // fixture's optimum is diagonal.
        let a = sol.x[0];
        assert!(
            (a - 1.0).abs() > 0.5,
            "`{name}`'s curvature scale is {a}, too close to 1 for the division by it \
             to be under test"
        );
    }
}

/// The `Psd(3)` fixture really is rank-deficient by **two**, or its extra rows
/// are not being exercised and it is a duplicate of the `Psd(2)` one.
#[test]
fn the_psd3_fixture_has_a_two_dimensional_kernel() {
    let prob = psd3(1.0);
    let cones = [ConeSpec::Psd(3)];
    let sol = solve(&prob, &cones);
    let sens = sens_for(&prob, &cones, &sol);
    // n + m_eq + rows; the face of a rank-1 3×3 block has codimension
    // q(q+1)/2 = 3 with q = 2.
    assert_eq!(
        sens.kkt_dim(),
        prob.n + prob.m_eq() + 3,
        "a rank-1 Psd(3) block must contribute three rows, not one"
    );
}

// ---------------------------------------------------------------------------
// The oracle, per family.
// ---------------------------------------------------------------------------

#[test]
fn the_psd_step_matches_a_resolve() {
    check_against_resolve("Psd(2)", psd2, &[ConeSpec::Psd(2)], ConeBlockKind::Boundary);
}

#[test]
fn the_psd3_step_matches_a_resolve() {
    check_against_resolve("Psd(3)", psd3, &[ConeSpec::Psd(3)], ConeBlockKind::Boundary);
}

#[test]
fn the_exponential_step_matches_a_resolve() {
    check_against_resolve(
        "Exponential",
        exp_cone,
        &[ConeSpec::Exponential],
        ConeBlockKind::Boundary,
    );
}

#[test]
fn the_power_step_matches_a_resolve() {
    check_against_resolve(
        "Power(0.6)",
        power_cone,
        &[ConeSpec::Power(0.6)],
        ConeBlockKind::Boundary,
    );
}

/// A slack block contributes nothing at all — asserted as "identical to the
/// cone-free problem", which is stronger than "no error": a spurious row whose
/// multiplier happens to be zero passes the weaker test.
#[test]
fn a_slack_power_block_contributes_nothing() {
    let prob = power_interior(1.0);
    let cones = [ConeSpec::Power(0.6)];
    let sol = solve(&prob, &cones);
    let mut sens = sens_for(&prob, &cones, &sol);
    assert_eq!(
        sens.kkt_dim(),
        prob.n + prob.m_eq(),
        "an interior cone block must contribute no rows at all"
    );

    let coneless = QpProblem {
        g: vec![],
        h: vec![],
        ..power_interior(1.0)
    };
    let coneless_sol = QpSolution {
        z: vec![],
        ..sol.clone()
    };
    let mut coneless_sens = QpSensitivity::build_default(&coneless, &coneless_sol, backend)
        .expect("the cone-free problem is an ordinary equality-constrained QP");
    assert_eq!(
        sens.parametric_step(&[0], &[DELTA]),
        coneless_sens.parametric_step(&[0], &[DELTA]),
        "a slack cone must leave the step bit-identical to the cone-free problem's"
    );
}

// ---------------------------------------------------------------------------
// The geometry the code leans on.
// ---------------------------------------------------------------------------

/// `power_face` has no guard for the `|x| = 0` kink, and this is why: on the
/// facet with `y, z > 0`, `|x| = y^α z^{1−α} > 0` identically. A point with
/// `x = 0` and `y, z > 0` is strictly *inside* the cone, never on its boundary,
/// so a kink guard there would be unreachable code that reads like coverage.
///
/// Asserted as geometry rather than as a branch, because a branch that cannot
/// be taken cannot be tested — which is the whole point.
#[test]
fn the_power_cones_x_kink_is_not_on_the_facet() {
    let alpha = 0.6;
    for (y, z) in [(1.0, 1.0), (1e-3, 5.0), (7.0, 2e-4)] {
        let g: f64 = f64::powf(y, alpha) * f64::powf(z, 1.0 - alpha);
        assert!(
            g > 0.0,
            "with y, z > 0 the facet value is strictly positive, so |x| = g > 0 there"
        );
    }
    // And the classification agrees: x = 0 with y, z > 0 is interior.
    let (prob, sol) = hand_built_cone(&[0.0, 1.0, 1.0], &[0.0, 0.0, 0.0]);
    let sens = QpSensitivity::build_conic(
        &prob,
        &[ConeSpec::Power(alpha)],
        &sol,
        &opts(),
        1e-7,
        backend,
    )
    .expect("x = 0 with y, z > 0 is strictly inside the power cone");
    assert_eq!(sens.cone_block_kinds(), [(0, ConeBlockKind::Interior)]);
}

// ---------------------------------------------------------------------------
// The refusals. Hand-built, because no interior-point solve stops on any of
// these.
// ---------------------------------------------------------------------------

/// `G = I`, `x = 0`, `h = s`, so `s = h` exactly and the solver's arithmetic is
/// out of the picture. `dim` is taken from `s`.
fn hand_built_cone(s: &[f64], z: &[f64]) -> (QpProblem, QpSolution) {
    let d = s.len();
    let prob = QpProblem {
        n: d,
        p_lower: (0..d).map(|j| tri(j, j, 1.0)).collect(),
        c: vec![0.0; d],
        a: vec![],
        b: vec![],
        g: (0..d).map(|j| tri(j, j, 1.0)).collect(),
        h: s.to_vec(),
        lb: vec![],
        ub: vec![],
    };
    let sol = QpSolution {
        status: QpStatus::Optimal,
        x: vec![0.0; d],
        y: vec![],
        z: z.to_vec(),
        z_lb: vec![0.0; d],
        z_ub: vec![0.0; d],
        obj: 0.0,
        iters: 0,
        iterates: vec![],
    };
    (prob, sol)
}

fn hand_built(s: &[f64], z: &[f64]) -> (QpProblem, QpSolution) {
    hand_built_cone(s, z)
}

fn refusal(spec: ConeSpec, s: &[f64], z: &[f64]) -> SensError {
    let (prob, sol) = hand_built_cone(s, z);
    match QpSensitivity::build_conic(&prob, &[spec], &sol, &opts(), 1e-7, backend) {
        Err(e) => e,
        Ok(_) => panic!("{spec:?} at s = {s:?}, z = {z:?} must be refused, but it was accepted"),
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

/// `rank Z = n − rank S` is what makes `ker S` the whole normal direction.
/// Where it fails, a direction exists along which slack and multiplier vanish
/// together — the PSD version of a weakly active row — and `dx/db` is
/// two-valued. `S = diag(1,0,0)` (rank 1, `q = 2`) against `Z = diag(0,0,1)`
/// (rank 1, not 2) is that case, with a live dual so it is not caught by the
/// collapse test upstream.
#[test]
fn a_psd_block_without_strict_complementarity_is_refused() {
    // svec order for n = 3: (0,0),(1,0),(2,0),(1,1),(2,1),(2,2)
    let s = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let z = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0];
    assert_nonsmooth(refusal(ConeSpec::Psd(3), &s, &z), "strict complementarity");
}

/// A slack matrix with a negative eigenvalue is not in the cone at all, so
/// there is no constant-rank manifold through it.
#[test]
fn a_psd_slack_outside_the_cone_is_refused() {
    let s = [1.0, 0.0, -1.0]; // S = diag(1, −1)
    let z = [0.0, 0.0, 1.0];
    assert_nonsmooth(refusal(ConeSpec::Psd(2), &s, &z), "negative eigenvalue");
}

/// The exponential cone's other boundary piece — the ray `{(x, 0, z)}` — has no
/// tangent plane, so it has no normal to linearize against.
#[test]
fn the_exponential_cones_degenerate_ray_is_refused() {
    assert_nonsmooth(
        refusal(ConeSpec::Exponential, &[-1.0, 0.0, 1.0], &[-1.0, 0.0, 1.0]),
        "y = 0",
    );
}

#[test]
fn an_exponential_slack_outside_the_cone_is_refused() {
    assert_nonsmooth(
        refusal(ConeSpec::Exponential, &[5.0, 1.0, 1.0], &[-1.0, 0.0, 1.0]),
        "outside the cone",
    );
}

/// On a facet's interior the normal cone **is** the ray `ℝ₊∇φ`, so `z = ν∇φ` is
/// the optimality condition rather than an approximation. A `z` off that ray
/// means the solution is not the one it claims to be, and building `ν` from it
/// would answer for a face the solution is not on.
///
/// The fixture sits exactly on `φ = 0` (`y = 1`, `z = e`, `x = 1`) with a dual
/// tilted off the ray by one unit in the `y` component — `O(1)` relative,
/// against the `2.8e-8`–`3.4e-5` a converged solve produces.
#[test]
fn an_exponential_dual_off_the_normal_ray_is_refused() {
    let e = std::f64::consts::E;
    assert_nonsmooth(
        refusal(ConeSpec::Exponential, &[1.0, 1.0, e], &[-1.0, 1.0, 1.0 / e]),
        "not on the ray normal to this face",
    );
}

/// The power cone's degenerate faces, where `g = y^α z^{1−α} = 0` and the two
/// smooth sheets `x = ±g` meet.
#[test]
fn the_power_cones_degenerate_face_is_refused() {
    assert_nonsmooth(
        refusal(ConeSpec::Power(0.6), &[0.0, 0.0, 1.0], &[1.0, 1.0, 1.0]),
        "y = 0 or z = 0",
    );
}

#[test]
fn a_power_slack_outside_the_cone_is_refused() {
    assert_nonsmooth(
        refusal(ConeSpec::Power(0.6), &[5.0, 1.0, 1.0], &[-1.0, 1.0, 1.0]),
        "outside the cone",
    );
}

/// The apex, for a family other than the second-order cone, so the generic
/// branch in `cone_block_face` is shown to be generic. `S = 0` with a live dual
/// pins the whole block: `q = n`, the face is a point, and every row enters.
#[test]
fn a_psd_apex_pins_the_whole_block() {
    let (prob, sol) = hand_built(&[0.0, 0.0, 0.0], &[1.0, 0.0, 1.0]);
    let mut sens =
        QpSensitivity::build_conic(&prob, &[ConeSpec::Psd(2)], &sol, &opts(), 1e-7, backend)
            .expect("S = 0 with a live dual is the apex");
    assert_eq!(sens.cone_block_kinds(), [(0, ConeBlockKind::Apex)]);
    assert_eq!(
        sens.kkt_dim(),
        prob.n + prob.m_eq() + prob.m_ineq(),
        "at the apex every row of the block is active"
    );
    // Nothing can move: `ds = 0` and `s = −Gx` here, so `dx = 0`.
    let dx = sens.step_from_db(&[]);
    assert!(dx.iter().all(|v| v.abs() < 1e-12), "{dx:?}");
}

/// The apex with a collapsed dual is weakly active in the conic sense, on every
/// family — the branch is in `cone_block_face`, above the per-family split.
#[test]
fn an_apex_with_a_collapsed_dual_is_refused_on_every_family() {
    for spec in [
        ConeSpec::Psd(2),
        ConeSpec::SecondOrder(3),
        ConeSpec::Exponential,
        ConeSpec::Power(0.6),
    ] {
        assert_nonsmooth(
            refusal(spec, &[0.0, 0.0, 0.0], &[0.0, 0.0, 0.0]),
            "apex and the dual has collapsed",
        );
    }
}
