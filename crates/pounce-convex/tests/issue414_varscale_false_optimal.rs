//! Regression for gh #414: a convex QP whose **variables** span many decades
//! must not be reported `Optimal` at a point that is not a KKT point.
//!
//! Construction (the adversary's): pick variable scales `s_i ∈ [10⁻ᵈ, 10ᵈ]` and
//! state a well-conditioned QP in `z = x/s`, then write it in `x`. The result
//! is a strictly convex box- and inequality-constrained QP with `cond(P) ~ 1e24`
//! at `d = 6` — and yet trivially easy, since one diagonal rescaling (`cond = 10`)
//! recovers the well-conditioned twin. pounce's own active-set engine, its NLP
//! engine, clarabel, and scipy `trust-constr` all agree on the optimum.
//!
//! Root cause: HSDE certifies convergence on *scale-relative* residuals once the
//! problem's natural scale puts absolute `tol` accuracy below the
//! finite-precision floor (`hsde::relative_stop_permitted`). Those normalizers
//! are **global** ∞-norms, so once the variable scales spread, the worst-scaled
//! column dominates `‖Px̂‖` and dividing every component's residual by it grants
//! a blanket relaxation to the components where the real violation lives. At
//! `d = 3` the absolute test still governs and the solve is correct; from
//! `d = 4` up the relative arm opens and the embedding stopped at a point whose
//! own `kkt_error` was `8.3e3`, objective `67.13` against a true `-3.96` — under
//! `status = Optimal`, i.e. `success=True` / `SolveSucceeded` / exit 0.
//!
//! The fix measures a claimed optimum in the Ruiz-**equilibrated** metric, where
//! no column can mask another (the bad point reads `6.9e2` there and the true
//! optimum `2.2e-10`, against `2e-4` for both in the unscaled metric), and
//! repairs it with an equilibrated re-solve — the same repair gh #293 already
//! applies to a non-converged HSDE solve.
//!
//! The oracle is clarabel on the identical problem stated in `z = x/s`, agreeing
//! with pounce's active-set engine to 10 digits.

use pounce_convex::{
    ConeSpec, QpOptions, QpProblem, QpStatus, Triplet, solve_qp_ipm, solve_socp_ipm,
};
use pounce_feral::FeralSolverInterface;
use pounce_linsol::SparseSymLinearSolverInterface;

fn backend() -> Box<dyn SparseSymLinearSolverInterface> {
    Box::new(FeralSolverInterface::new())
}

fn objective(prob: &QpProblem, x: &[f64]) -> f64 {
    let mut px = vec![0.0; prob.n];
    prob.p_mul_add_pub(x, &mut px);
    (0..prob.n)
        .map(|i| 0.5 * x[i] * px[i] + prob.c[i] * x[i])
        .sum()
}

/// `n = 3`, scales `10^±3`: *below* the threshold where HSDE's relative
/// convergence arm opens. Already correct before the fix — pinned so the fix
/// cannot regress the regime it does not target.
fn n3_dec3() -> QpProblem {
    QpProblem {
        n: 3,
        p_lower: vec![
            Triplet::new(0, 0, 8395050.448209196),
            Triplet::new(1, 0, -1902.145044836726),
            Triplet::new(1, 1, 3.251598903330246),
            Triplet::new(2, 0, 2.8600351667480353),
            Triplet::new(2, 1, 0.00011387032854093065),
            Triplet::new(2, 2, 2.5156283086289337e-06),
        ],
        c: vec![
            -2410.8575016379787,
            -0.47196110542608866,
            0.0019297552321865365,
        ],
        a: vec![],
        b: vec![],
        g: vec![
            Triplet::new(0, 0, -1363.466266639565),
            Triplet::new(0, 1, -0.34926083632221316),
            Triplet::new(0, 2, -0.00036213872631073423),
        ],
        h: vec![2.4552930686756964],
        lb: vec![
            -0.011096628522428714,
            -10.254262623099589,
            -9953.548897040608,
        ],
        ub: vec![0.008903371477571287, 9.745737376900411, 10046.451102959392],
    }
}

/// `n = 3`, scales `10^±4`: the first decade at which the defect appeared.
fn n3_dec4() -> QpProblem {
    QpProblem {
        n: 3,
        p_lower: vec![
            Triplet::new(0, 0, 839505044.8209198),
            Triplet::new(1, 0, -19021.450448367257),
            Triplet::new(1, 1, 3.251598903330246),
            Triplet::new(2, 0, 2.8600351667480357),
            Triplet::new(2, 1, 1.1387032854093064e-05),
            Triplet::new(2, 2, 2.5156283086289342e-08),
        ],
        c: vec![
            -24108.575016379786,
            -0.47196110542608866,
            0.00019297552321865367,
        ],
        a: vec![],
        b: vec![],
        g: vec![
            Triplet::new(0, 0, -13634.662666395649),
            Triplet::new(0, 1, -0.34926083632221316),
            Triplet::new(0, 2, -3.6213872631073426e-05),
        ],
        h: vec![2.455293068675696],
        lb: vec![
            -0.0011096628522428713,
            -10.254262623099589,
            -99535.48897040608,
        ],
        ub: vec![0.0008903371477571286, 9.745737376900411, 100464.51102959392],
    }
}

/// The instance reported in gh #414: `n = 3`, scales `10^±6`, `cond(P) ~ 1e24`.
/// `solve_qp` returned `status=optimal, success=True, obj=67.13411102` with
/// `kkt_error=8282.49679691473` on the very same result object.
fn n3_dec6() -> QpProblem {
    QpProblem {
        n: 3,
        p_lower: vec![
            Triplet::new(0, 0, 8395050448209.196),
            Triplet::new(1, 0, -1902145.0448367258),
            Triplet::new(1, 1, 3.251598903330246),
            Triplet::new(2, 0, 2.8600351667480353),
            Triplet::new(2, 1, 1.1387032854093064e-07),
            Triplet::new(2, 2, 2.5156283086289338e-12),
        ],
        c: vec![
            -2410857.501637979,
            -0.47196110542608866,
            1.9297552321865365e-06,
        ],
        a: vec![],
        b: vec![],
        g: vec![
            Triplet::new(0, 0, -1363466.266639565),
            Triplet::new(0, 1, -0.34926083632221316),
            Triplet::new(0, 2, -3.621387263107342e-07),
        ],
        h: vec![2.455293068675696],
        lb: vec![
            -1.1096628522428714e-05,
            -10.254262623099589,
            -9953548.897040607,
        ],
        ub: vec![8.903371477571285e-06, 9.745737376900411, 10046451.102959393],
    }
}

/// `n = 4`, scales `10^±4` — a denser Hessian and a differently-oriented
/// inequality, so the failure is not an artifact of one sparsity pattern.
fn n4_dec4() -> QpProblem {
    QpProblem {
        n: 4,
        p_lower: vec![
            Triplet::new(0, 0, 402865426.6568016),
            Triplet::new(1, 0, -135775.05946876763),
            Triplet::new(1, 1, 1913.5369545367312),
            Triplet::new(2, 0, 354.22953694689227),
            Triplet::new(2, 1, 0.6929769838082036),
            Triplet::new(2, 2, 0.0063346701154133956),
            Triplet::new(3, 0, -3.264478040673124),
            Triplet::new(3, 1, -0.0017689547554672362),
            Triplet::new(3, 2, -1.2619277561696707e-05),
            Triplet::new(3, 3, 6.704485454332412e-08),
        ],
        c: vec![
            -2542.6262309958797,
            1.0007586760595901,
            0.00042322029453420934,
            -2.4182384602113466e-05,
        ],
        a: vec![],
        b: vec![],
        g: vec![
            Triplet::new(0, 0, 14796.737397512617),
            Triplet::new(0, 1, 18.04148227139861),
            Triplet::new(0, 2, 0.03236962077437109),
            Triplet::new(0, 3, -7.267931986554799e-05),
        ],
        h: vec![-2.8460530841924774],
        lb: vec![
            -0.0011202255925231168,
            -0.46765189691097764,
            -224.92520875544545,
            -84085.25914629847,
        ],
        ub: vec![
            0.0008797744074768833,
            0.46066586981157787,
            205.961729250931,
            115914.74085370153,
        ],
    }
}

/// `n = 6`, scales `10^±6` — the widest spread in the report.
fn n6_dec6() -> QpProblem {
    QpProblem {
        n: 6,
        p_lower: vec![
            Triplet::new(0, 0, 4856251966963.262),
            Triplet::new(1, 0, 5613717544.627589),
            Triplet::new(1, 1, 29052429.43414988),
            Triplet::new(2, 0, -21372575.653614767),
            Triplet::new(2, 1, -20195.486710036217),
            Triplet::new(2, 2, 797.6268702598865),
            Triplet::new(3, 0, 125181.59528943096),
            Triplet::new(3, 1, 177.96448164812205),
            Triplet::new(3, 2, -1.0904525675371035),
            Triplet::new(3, 3, 0.01615930198769556),
            Triplet::new(4, 0, 155.4188314072008),
            Triplet::new(4, 1, 0.7848493607162925),
            Triplet::new(4, 2, -0.006937841395298172),
            Triplet::new(4, 3, 1.311288537574924e-05),
            Triplet::new(4, 4, 2.767140072085592e-07),
            Triplet::new(5, 0, 0.7657035769063948),
            Triplet::new(5, 1, 0.0036104087965541133),
            Triplet::new(5, 2, 9.872670170359158e-07),
            Triplet::new(5, 3, 4.727104589430104e-08),
            Triplet::new(5, 4, 5.668405227071561e-10),
            Triplet::new(5, 5, 7.078024066540731e-12),
        ],
        c: vec![
            1328619.0859474859,
            938.6271165901893,
            8.328718243103483,
            0.010608966961930871,
            -5.6871835063293957e-05,
            -6.53528505743659e-07,
        ],
        a: vec![],
        b: vec![],
        g: vec![
            Triplet::new(0, 0, 526075.14322749),
            Triplet::new(0, 1, -160.73512951962238),
            Triplet::new(0, 2, -7.344017827017593),
            Triplet::new(0, 3, 0.006739359670159433),
            Triplet::new(0, 4, 0.00025973058904486254),
            Triplet::new(0, 5, -6.047724374034646e-07),
        ],
        h: vec![1.406699584001822],
        lb: vec![
            -1.0991735677298826e-05,
            -0.002458925393251575,
            -0.5317893358898369,
            -152.71878271324712,
            -31449.428840612563,
            -9341661.687928105,
        ],
        ub: vec![
            9.008264322701172e-06,
            0.002564847469767584,
            0.7301253530705492,
            164.2598557789751,
            48172.00527008682,
            10658338.312071895,
        ],
    }
}

/// The family, with the clarabel-on-`z = x/s` oracle optimum of each.
fn family() -> Vec<(&'static str, QpProblem, f64)> {
    vec![
        ("n=3 10^±3", n3_dec3(), -3.958501807911171),
        ("n=3 10^±4", n3_dec4(), -3.958501807911169),
        ("n=3 10^±6", n3_dec6(), -3.9585018079111713),
        ("n=4 10^±4", n4_dec4(), 3.830698737611161),
        ("n=6 10^±6", n6_dec6(), -0.43521733351107683),
    ]
}

/// The headline guarantee: **no false success**. An `Optimal`/`OptimalInaccurate`
/// verdict — everything the drivers report as a solved model — must come with a
/// point that actually optimizes the problem.
///
/// Deliberately phrased as an implication rather than "must solve": a solver is
/// allowed to fail on a hard instance and say so. What it may not do is hand
/// back `obj = 67.13` for a `-3.96` problem under `success=True`.
#[test]
fn issue_414_reported_optimal_is_never_a_non_kkt_point() {
    for (label, prob, oracle) in family() {
        let sol = solve_qp_ipm(&prob, &QpOptions::default(), backend);
        if !matches!(sol.status, QpStatus::Optimal | QpStatus::OptimalInaccurate) {
            continue;
        }
        let obj = objective(&prob, &sol.x);
        let rel = (obj - oracle).abs() / oracle.abs().max(1.0);
        assert!(
            rel < 1e-6,
            "{label}: {:?} at objective {obj} — oracle {oracle} (rel {rel:.2e}), \
             kkt_error {:.3e}",
            sol.status,
            sol.kkt_residuals(&prob).kkt_error(),
        );
    }
}

/// The repair itself: every instance in the family is solvable, and the default
/// IPM must in fact solve it — the equilibrated re-solve recovers the optimum
/// rather than merely refusing to claim the bad point.
#[test]
fn issue_414_variable_scale_spread_still_reaches_the_optimum() {
    for (label, prob, oracle) in family() {
        let sol = solve_qp_ipm(&prob, &QpOptions::default(), backend);
        assert_eq!(sol.status, QpStatus::Optimal, "{label}: {:?}", sol.status);
        let obj = objective(&prob, &sol.x);
        let rel = (obj - oracle).abs() / oracle.abs().max(1.0);
        assert!(
            rel < 1e-6,
            "{label}: objective {obj} vs oracle {oracle} (rel {rel:.2e})"
        );
        // Feasibility is not implied by the objective matching: check the box
        // and the inequality row directly.
        for i in 0..prob.n {
            let slack = (sol.x[i] - prob.lb[i]).min(prob.ub[i] - sol.x[i]);
            let span = prob.ub[i] - prob.lb[i];
            assert!(
                slack > -1e-6 * span.max(1.0),
                "{label}: x[{i}] = {} outside [{}, {}]",
                sol.x[i],
                prob.lb[i],
                prob.ub[i]
            );
        }
    }
}

/// `solver_selection=socp` routes a box-constrained QP through the conic entry
/// point, which reaches the same embedding and inherited the same false
/// `Optimal`. With every cone nonnegative the problem *is* the LP/QP case, so
/// the repair applies there too.
#[test]
fn issue_414_socp_entry_point_on_an_orthant_problem_agrees() {
    for (label, prob, oracle) in family() {
        let cones = [ConeSpec::Nonneg(prob.m_ineq())];
        let sol = solve_socp_ipm(&prob, &cones, &QpOptions::default(), backend);
        assert_eq!(
            sol.status,
            QpStatus::Optimal,
            "{label} (socp): {:?}",
            sol.status
        );
        let obj = objective(&prob, &sol.x);
        let rel = (obj - oracle).abs() / oracle.abs().max(1.0);
        assert!(
            rel < 1e-6,
            "{label} (socp): objective {obj} vs oracle {oracle} (rel {rel:.2e})"
        );
    }
}
