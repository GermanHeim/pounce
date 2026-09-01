//! The convex sensitivity builder must refuse a solution it cannot read.
//!
//! # Why this file exists
//!
//! `solve_socp_ipm` and `solve_qp_ipm` return the **same** [`QpSolution`]
//! type, and the cone partition travels beside it as a separate
//! `&[ConeSpec]` argument that `QpSensitivity::build` never sees. Before this
//! guard, handing `build` a solved SOCP was accepted: every cone row was read
//! as a nonnegative-orthant row, an active-set KKT was assembled from rows
//! that are not the active object, and `parametric_step` returned a plausible
//! number that is not a derivative. Nothing warned. That is the
//! silently-wrong-while-reporting-success class (`/sens-review` entry 5), on
//! the one path in the crate where two solvers share a result type.
//!
//! The guard is `check_orthant_complementarity`, which needs no cone
//! information: an orthant row complements *per row* (`sᵢ ≥ 0`, `zᵢ ≥ 0`,
//! `sᵢzᵢ ≈ μ`), while a conic block complements only as a block inner product
//! `⟨s, z⟩ = 0`. `QpSensitivity::build_conic` is the explicit entry point and
//! refuses every non-`Nonneg` family outright.
//!
//! # What this file is NOT evidence about
//!
//! - **The apex-with-zero-tail case.** A second-order cone at its apex with
//!   `z_{1:} = 0` is row-wise indistinguishable from a degenerate orthant
//!   block, which is a legitimate input, so `build` accepts it — and answers
//!   with the wrong active object. Only `build_conic` is told, and
//!   `the_apex_case_needs_the_cone_partition` pins that difference: `build`
//!   is silent, `build_conic` classifies the block as `SocBlockKind::Apex`.
//!   The classification's *numerics* are `convex_soc_sensitivity.rs`'s.
//! - **Correctness of any conic `dx/db`.** Nothing here computes one; the
//!   whole point is that none is computed yet.
//! - **The orthant step's numerics.** Owned by the `mod tests` in
//!   `src/sensitivity.rs` and by `python/tests/test_qp_sensitivity.py`. This
//!   file only asserts the guard does not fire on those inputs.
//! - **The guard, from `an_orthant_lp_still_builds`.** That fixture has an
//!   empty `G`, so `m_ineq == 0` and the guard's loop body never executes —
//!   measured, not assumed (see the mutation table). It is here for `build`'s
//!   `P = 0` branch, which nothing else in the crate covered; the guard's
//!   false-positive evidence is `an_orthant_qp_still_builds`, which has a real
//!   inequality row.
//!
//! # Mutation evidence
//!
//! These pass on `main` with the guard in place, so the evidence that they
//! guard anything is what happens when it is removed. Each row below was
//! **run**, not predicted:
//!
//! | mutation | red here | note |
//! |---|---|---|
//! | delete the `check_orthant_complementarity(..)?` call in `build` | `a_solved_socp_is_refused` alone | fails at the `Ok(_)` arm — i.e. the solved SOCP is *accepted*, which is the original defect reproduced |
//! | `ORTHANT_GUARD_REL` `1e-4` → `1e-30` | `an_orthant_qp_still_builds`, `build_conic_on_an_all_nonneg_partition_matches_build` | the false-positive direction; `an_orthant_lp_still_builds` stays green because it has no inequality rows |
//! | `cone_family` returns `"Nonneg"` for every family | `build_conic_refuses_every_unsupported_family` | the refusal must name the family, not merely occur |
//! | `build_conic`'s `_ =>` arm falls through to the orthant path | `build_conic_refuses_every_unsupported_family` | reproduces the original defect for the three unimplemented families |
//! | `soc_block_rows`'s apex branch returns `Interior` | `the_apex_case_needs_the_cone_partition` | an apex read as interior contributes no rows, so the block silently stops constraining `dx` |

use pounce_convex::cones::ConeSpec;
use pounce_convex::ipm::{solve_qp_ipm, solve_socp_ipm};
use pounce_convex::qp::{QpProblem, QpStatus, Triplet};
use pounce_convex::sensitivity::{QpSensitivity, SensError, SocBlockKind};
use pounce_convex::{QpOptions, QpSolution};
use pounce_feral::FeralSolverInterface;
use pounce_linsol::SparseSymLinearSolverInterface;

fn backend() -> Box<dyn SparseSymLinearSolverInterface> {
    Box::new(FeralSolverInterface::new())
}

fn tri(row: usize, col: usize, val: f64) -> Triplet {
    Triplet { row, col, val }
}

/// `min ½‖x‖² s.t. ‖(x₀, x₁)‖ ≤ x₂, x₂ = 1` posed as a second-order cone.
///
/// The `G`/`h` block is `s = (x₂, x₀, x₁) ∈ SOC(3)`, written as `h − Gx` with
/// `h = 0` and `G = −[e₂; e₀; e₁]`. The optimum sits at the cone's apex-facing
/// interior or its boundary depending on the objective; what matters here is
/// only that the block is a genuine `SecondOrder(3)`.
fn soc_problem() -> (QpProblem, Vec<ConeSpec>) {
    let prob = QpProblem {
        n: 3,
        p_lower: vec![tri(0, 0, 2.0), tri(1, 1, 2.0), tri(2, 2, 2.0)],
        c: vec![-1.0, -2.0, 0.0],
        a: vec![tri(0, 2, 1.0)],
        b: vec![1.0],
        // s = (x2, x0, x1) must lie in SOC(3):  s = h - Gx with h = 0.
        g: vec![tri(0, 2, -1.0), tri(1, 0, -1.0), tri(2, 1, -1.0)],
        h: vec![0.0, 0.0, 0.0],
        lb: vec![],
        ub: vec![],
    };
    (prob, vec![ConeSpec::SecondOrder(3)])
}

/// A plain convex QP with one active inequality and one active bound — the
/// input the builder is *for*. `min ½‖x‖² s.t. x₀ + x₁ = 1, x₀ ≥ 0.2`.
fn orthant_qp() -> QpProblem {
    QpProblem {
        n: 2,
        p_lower: vec![tri(0, 0, 1.0), tri(1, 1, 1.0)],
        c: vec![0.0, 0.0],
        a: vec![tri(0, 0, 1.0), tri(0, 1, 1.0)],
        b: vec![1.0],
        g: vec![tri(0, 0, -1.0)],
        h: vec![-0.2],
        lb: vec![],
        ub: vec![],
    }
}

/// [`orthant_qp`] widened to `k` inequality rows, all inactive, so a cone
/// partition of a given total dimension can be posed against it.
/// `min ½‖x‖² s.t. x₀ + x₁ = 1, −x₀ ≤ 0.2 + j` for `j` in `0..k`.
fn orthant_qp_with_ineq_rows(k: usize) -> QpProblem {
    QpProblem {
        n: 2,
        p_lower: vec![tri(0, 0, 1.0), tri(1, 1, 1.0)],
        c: vec![0.0, 0.0],
        a: vec![tri(0, 0, 1.0), tri(0, 1, 1.0)],
        b: vec![1.0],
        g: (0..k).map(|j| tri(j, 0, -1.0)).collect(),
        h: (0..k).map(|j| 0.2 + j as f64).collect(),
        lb: vec![],
        ub: vec![],
    }
}

// ---------------------------------------------------------------------------
// Preconditions — without these the refusals below are vacuous.
// ---------------------------------------------------------------------------

/// The SOC fixture really does solve as a conic program, and really does have
/// a block that is not an orthant. If this stops holding, the refusal tests
/// below are asserting nothing about cones.
#[test]
fn the_soc_fixture_solves_as_a_cone() {
    let (prob, cones) = soc_problem();
    let sol = solve_socp_ipm(&prob, &cones, &QpOptions::default(), backend);
    assert_eq!(
        sol.status,
        QpStatus::Optimal,
        "the SOC fixture must solve, or the refusal tests are vacuous"
    );
    assert!(
        matches!(cones[0], ConeSpec::SecondOrder(_)),
        "the fixture's block must not be an orthant"
    );
}

// ---------------------------------------------------------------------------
// The refusals.
// ---------------------------------------------------------------------------

/// The headline: a solved SOCP handed to the orthant builder is refused, not
/// answered. This is the path that was silently wrong.
#[test]
fn a_solved_socp_is_refused() {
    let (prob, cones) = soc_problem();
    let sol = solve_socp_ipm(&prob, &cones, &QpOptions::default(), backend);
    assert_eq!(sol.status, QpStatus::Optimal);

    match QpSensitivity::build_default(&prob, &sol, backend) {
        Err(SensError::NotOrthantComplementary { row, what }) => {
            assert!(
                row < prob.h.len(),
                "the refusal must name a real inequality row, got {row}"
            );
            assert!(!what.is_empty(), "the refusal must say which test failed");
        }
        Err(other) => panic!(
            "a solved SOCP must be refused as non-orthant, not as {other:?} — a different \
             refusal would mask the defect this guard exists for"
        ),
        Ok(_) => panic!(
            "a solved SOCP was ACCEPTED by the orthant builder: every cone row is being read \
             as an orthant row and `parametric_step` will return a number that is not a \
             derivative. This is the defect the guard exists for."
        ),
    }
}

/// `build_conic` refuses every family it has not implemented, naming the block
/// and the family so the message is diagnosable.
#[test]
fn build_conic_refuses_every_unsupported_family() {
    // Four inequality rows so `[Nonneg(1), spec]` covers the block exactly —
    // every family below has `dim() == 3`. A short partition would be refused
    // as a `ConePartitionMismatch` instead, which is a different test.
    let prob = orthant_qp_with_ineq_rows(4);
    let sol = solve_qp_ipm(&prob, &QpOptions::default(), backend);
    assert_eq!(sol.status, QpStatus::Optimal);

    // `SecondOrder` is deliberately absent: it is implemented, and
    // `convex_soc_sensitivity.rs` owns it. The three below are the families
    // the crate still has no tangent/normal decomposition for.
    for (spec, family) in [
        (ConeSpec::Exponential, "Exponential"),
        (ConeSpec::Power(0.5), "Power"),
        (ConeSpec::Psd(2), "Psd"),
    ] {
        assert_eq!(
            spec.dim(),
            3,
            "the fixture sizing assumes dim 3 for {family}"
        );
        // Block 0 is a Nonneg the builder would accept; the unsupported family
        // is block 1, so this also pins that the *index* is reported, not 0.
        let cones = [ConeSpec::Nonneg(1), spec];
        match QpSensitivity::build_conic(&prob, &cones, &sol, &QpOptions::default(), 1e-7, backend)
        {
            Err(SensError::UnsupportedCone { block, family: got }) => {
                assert_eq!(block, 1, "the refusal must name the offending block");
                assert_eq!(got, family, "the refusal must name the cone family");
            }
            Err(other) => panic!("{family} must be refused as UnsupportedCone, got {other:?}"),
            Ok(_) => panic!("{family} must be refused by build_conic, but it was accepted"),
        }
    }
}

/// A partition that does not cover the inequality block is a caller error, and
/// the short direction is the dangerous one: `[Nonneg(2)]` on a problem whose
/// remaining rows are really a cone would otherwise pass the family check and
/// reach the orthant path.
#[test]
fn build_conic_refuses_a_partition_that_does_not_cover_the_rows() {
    let prob = orthant_qp_with_ineq_rows(4);
    let sol = solve_qp_ipm(&prob, &QpOptions::default(), backend);
    assert_eq!(sol.status, QpStatus::Optimal);

    for cones in [
        vec![ConeSpec::Nonneg(2)],                      // short
        vec![ConeSpec::Nonneg(4), ConeSpec::Nonneg(1)], // long
    ] {
        let covered_expected: usize = cones.iter().map(ConeSpec::dim).sum();
        match QpSensitivity::build_conic(&prob, &cones, &sol, &QpOptions::default(), 1e-7, backend)
        {
            Err(SensError::ConePartitionMismatch { covered, m_ineq }) => {
                assert_eq!(covered, covered_expected);
                assert_eq!(m_ineq, prob.m_ineq());
            }
            Err(other) => panic!(
                "a partition covering {covered_expected} of {} rows must be refused as a \
                 ConePartitionMismatch, got {other:?}",
                prob.m_ineq()
            ),
            Ok(_) => panic!(
                "a partition covering {covered_expected} of {} rows was accepted",
                prob.m_ineq()
            ),
        }
    }
}

// ---------------------------------------------------------------------------
// The guard must not fire on the inputs it is on the path of.
// ---------------------------------------------------------------------------

/// The guard sits in front of every convex sensitivity build, so a false
/// positive here would break the feature outright. A QP with an active
/// inequality and an active bound must still build.
#[test]
fn an_orthant_qp_still_builds() {
    let prob = orthant_qp();
    let sol = solve_qp_ipm(&prob, &QpOptions::default(), backend);
    assert_eq!(sol.status, QpStatus::Optimal);

    let Ok(mut sens) = QpSensitivity::build_default(&prob, &sol, backend) else {
        panic!(
            "an ordinary convex QP must still build a sensitivity — the guard is on the \
                path of every convex sensitivity build, so a false positive here breaks the \
                feature outright"
        );
    };
    // And the step still works — the guard is not just letting `build` through.
    let dx = sens.parametric_step(&[0], &[1.0]);
    assert_eq!(dx.len(), prob.n);
    assert!(
        dx.iter().all(|v| v.is_finite()),
        "the step must stay finite: {dx:?}"
    );
}

/// The LP branch (`P = 0`) — untested anywhere in the crate before this file,
/// and the branch where the KKT's `(x,x)` block is only the regularization.
/// `min −x₀ − 2x₁ s.t. x₀ + x₁ = 2, x ≥ 0` has the vertex `x = (0, 2)`.
#[test]
fn an_orthant_lp_still_builds() {
    let prob = QpProblem {
        n: 2,
        p_lower: vec![],
        c: vec![-1.0, -2.0],
        a: vec![tri(0, 0, 1.0), tri(0, 1, 1.0)],
        b: vec![2.0],
        g: vec![],
        h: vec![],
        lb: vec![0.0, 0.0],
        ub: vec![f64::INFINITY, f64::INFINITY],
    };
    let sol = solve_qp_ipm(&prob, &QpOptions::default(), backend);
    assert_eq!(sol.status, QpStatus::Optimal);

    let Ok(mut sens) = QpSensitivity::build_default(&prob, &sol, backend) else {
        panic!("a pure LP (P = 0) must still build a sensitivity");
    };
    let dx = sens.parametric_step(&[0], &[1.0]);
    // At this nondegenerate vertex x₁ absorbs the whole budget change.
    assert!(
        (dx[0] - 0.0).abs() < 1e-6 && (dx[1] - 1.0).abs() < 1e-6,
        "dx/db at the vertex (0, 2) must be (0, 1), got {dx:?}"
    );
}

/// An all-`Nonneg` partition *is* the orthant problem, so `build_conic` must
/// behave exactly as `build` on it — same answer, not merely "no error".
#[test]
fn build_conic_on_an_all_nonneg_partition_matches_build() {
    let prob = orthant_qp();
    let sol = solve_qp_ipm(&prob, &QpOptions::default(), backend);
    assert_eq!(sol.status, QpStatus::Optimal);

    let Ok(mut via_build) = QpSensitivity::build_default(&prob, &sol, backend) else {
        panic!("the orthant QP must build through `build`");
    };
    let cones = [ConeSpec::Nonneg(prob.h.len())];
    let Ok(mut via_conic) =
        QpSensitivity::build_conic(&prob, &cones, &sol, &QpOptions::default(), 1e-7, backend)
    else {
        panic!("build_conic on an all-Nonneg partition is the orthant case and must build");
    };

    assert_eq!(via_build.kkt_dim(), via_conic.kkt_dim());
    let a = via_build.parametric_step(&[0], &[1.0]);
    let b = via_conic.parametric_step(&[0], &[1.0]);
    assert_eq!(
        a, b,
        "the two entry points must agree exactly on an orthant"
    );
}

// ---------------------------------------------------------------------------
// The limit, pinned rather than hidden.
// ---------------------------------------------------------------------------

/// `check_orthant_complementarity` reads rows, not cones, so a second-order
/// cone at its apex with a zero dual tail (`s = 0`, `z = (z₀, 0, …)`) passes
/// every row-wise test — it is indistinguishable from a degenerate orthant
/// block, which is a legitimate input the guard must not reject.
///
/// This is not a defect in the guard; it is the reason `build_conic` takes the
/// partition. The test states the boundary so a future reader does not mistake
/// the guard for complete cone detection.
#[test]
fn the_apex_case_needs_the_cone_partition() {
    // Hand-built: s = 0 (apex) with a dual supported only on the first row.
    // Row-wise this satisfies s ≥ 0, z ≥ 0 and s·z = 0 on every row.
    let prob = QpProblem {
        n: 3,
        p_lower: vec![tri(0, 0, 1.0), tri(1, 1, 1.0), tri(2, 2, 1.0)],
        c: vec![0.0, 0.0, 0.0],
        a: vec![],
        b: vec![],
        g: vec![tri(0, 0, 1.0), tri(1, 1, 1.0), tri(2, 2, 1.0)],
        h: vec![0.0, 0.0, 0.0],
        lb: vec![],
        ub: vec![],
    };
    let apex = QpSolution {
        status: QpStatus::Optimal,
        x: vec![0.0, 0.0, 0.0],
        y: vec![],
        z: vec![1.0, 0.0, 0.0],
        z_lb: vec![0.0; 3],
        z_ub: vec![0.0; 3],
        obj: 0.0,
        iters: 0,
        iterates: vec![],
    };
    assert!(
        QpSensitivity::build_default(&prob, &apex, backend).is_ok(),
        "the row-wise guard cannot distinguish a SOC apex with a zero dual tail from \
         orthant degeneracy — build_conic is what carries that information"
    );

    // And build_conic, which is told, classifies it as an apex rather than
    // reading three orthant rows off it. The two entry points must therefore
    // disagree about what is active here — which is the whole point of the
    // partition travelling with the solution.
    let cones = [ConeSpec::SecondOrder(3)];
    let Ok(conic) =
        QpSensitivity::build_conic(&prob, &cones, &apex, &QpOptions::default(), 1e-7, backend)
    else {
        panic!("an apex with an interior dual is a supported (flat) face and must build");
    };
    assert_eq!(
        conic.cone_block_kinds(),
        [(0, SocBlockKind::Apex)],
        "the block must be classified as an apex, not as an interior or a boundary"
    );
}
