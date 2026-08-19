//! gh #588 (Q9) — Gondzio multiple centrality correctors on the **direct**
//! driver.
//!
//! The scheme has lived in the HSDE loop since the NETLIB GEN degenerate-face
//! work; the direct driver (`run_ipm`, the warm-start / factor-reuse /
//! `qp_hsde=no` path) had none. Q0(b) of the quadratic-structure series traced
//! two stalling QCQPs and found one of them — `qcqp1000-2nc` — accepting every
//! step on the first trial but travelling almost nowhere, which is
//! fraction-to-boundary limiting caused by poor centrality and exactly what
//! correctors address.
//!
//! These tests pin the three things that must hold regardless of how much the
//! scheme buys on any particular instance:
//!
//! 1. `gondzio_max_corr = 0` is a real off switch, on both drivers.
//! 2. Turning correctors on does not change *what* is solved — same status,
//!    same optimal value to tolerance — only how the solver gets there.
//! 3. Over families built to be centrality-hostile, the correctors do not
//!    *cost* iterations in aggregate.
//!
//! (3) is a floor, not a claim of a win, and the numbers say why. At the
//! shipping commit the correctors do fire on both families — roughly one
//! acceptance per solve out of seven to nine attempts, mean step gain ~2e-2 —
//! and the totals come out **exactly level**: 146 iterations either way on the
//! degenerate corners, 84 either way on the spread LPs, with individual
//! instances moving in both directions and cancelling. So these tests pin that
//! the acceptance rule is not passing correctors that hurt; they are not
//! evidence that the scheme pays. The evidence that it pays is `lp_afiro`
//! (NETLIB `afiro`) in
//! `pounce-cli/tests/issue_588_gondzio_correctors.rs` — 10 -> 9 iterations
//! on the direct driver, 15 -> 13 on HSDE.

use pounce_convex::{
    QpOptions, QpProblem, QpStatus, QpWarmStart, Triplet, solve_qp_ipm, solve_qp_ipm_warm,
};
use pounce_feral::FeralSolverInterface;
use pounce_linsol::SparseSymLinearSolverInterface;

fn backend() -> Box<dyn SparseSymLinearSolverInterface> {
    Box::new(FeralSolverInterface::new())
}

/// The direct driver, which is what this phase changed. HSDE self-starts and
/// ignores the factor-reuse plumbing; `use_hsde: false` is the route the
/// warm-start entry points and `qp_hsde=no` take.
fn direct(corr: usize) -> QpOptions {
    QpOptions {
        use_hsde: false,
        gondzio_max_corr: corr,
        ..QpOptions::default()
    }
}

/// A corner where more constraints hold with equality than the dimension
/// needs — nested partial sums `Σ_{i≤k} xᵢ ≤ cap_k`, all tight at the
/// optimum. Strict complementarity fails, so the complementarity products
/// spread out and the blocking component stops the step well short of the
/// boundary: the regime correctors exist for.
fn degenerate_corner(theta: f64, n: usize) -> QpProblem {
    let mut g = Vec::new();
    let mut h = Vec::new();
    for k in 0..n {
        for i in 0..=k {
            g.push(Triplet::new(k, i, 1.0));
        }
        h.push(0.1 * (k + 1) as f64 + 0.01 * theta);
    }
    QpProblem {
        n,
        p_lower: (0..n).map(|i| Triplet::new(i, i, 1.0)).collect(),
        c: (0..n)
            .map(|i| -1.0 - 0.05 * i as f64 - 0.02 * theta)
            .collect(),
        a: vec![],
        b: vec![],
        g,
        h,
        lb: vec![0.0; n],
        ub: vec![],
    }
}

/// A badly-spread LP: the cost coefficients span six orders of magnitude, so
/// the products `sᵢzᵢ` at the initial point are nowhere near each other and
/// the centrality band has real work to do.
fn spread_lp(theta: f64, n: usize) -> QpProblem {
    QpProblem {
        n,
        p_lower: vec![],
        c: (0..n)
            .map(|i| -(10.0_f64).powi((i % 7) as i32 - 3) * (1.0 + 0.1 * theta))
            .collect(),
        a: (0..n).map(|i| Triplet::new(0, i, 1.0)).collect(),
        b: vec![1.0],
        g: (0..n)
            .map(|i| Triplet::new(i, i, 1.0 + 0.01 * i as f64))
            .collect(),
        h: (0..n).map(|i| 0.5 + 0.01 * (i as f64 + theta)).collect(),
        lb: vec![0.0; n],
        ub: vec![],
    }
}

/// A box-bounded LP with a dense-ish `G`, in the shape the Python QP host
/// actually solves. Deterministic: a plain LCG stands in for the RNG so the
/// family is reproducible without a dependency.
fn boxed_lp(seed: u64, n: usize, m: usize) -> QpProblem {
    let mut st = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    let mut next = || {
        st = st
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((st >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
    };
    let c: Vec<f64> = (0..n).map(|_| next()).collect();
    let mut g = Vec::with_capacity(n * m);
    for r in 0..m {
        for j in 0..n {
            g.push(Triplet::new(r, j, next()));
        }
    }
    let h: Vec<f64> = (0..m).map(|_| next().abs() + 1.0).collect();
    QpProblem {
        n,
        p_lower: vec![],
        c,
        a: vec![],
        b: vec![],
        g,
        h,
        lb: vec![-10.0; n],
        ub: vec![10.0; n],
    }
}

/// Solve `prob` both ways and return `(iters_off, iters_on)`, having checked
/// that both reached the same answer.
fn both_ways(prob: &QpProblem, what: &str) -> (usize, usize) {
    let off = solve_qp_ipm(prob, &direct(0), backend);
    let on = solve_qp_ipm(
        prob,
        &direct(QpOptions::default().gondzio_max_corr),
        backend,
    );
    assert_eq!(off.status, on.status, "{what}: the verdict changed");
    if off.status == QpStatus::Optimal {
        let scale = off.obj.abs().max(1.0);
        assert!(
            (off.obj - on.obj).abs() <= 1e-6 * scale,
            "{what}: objective moved {} -> {}",
            off.obj,
            on.obj
        );
    }
    (off.iters, on.iters)
}

/// `gondzio_max_corr = 0` must actually disable the loop rather than merely
/// shrink it, on both drivers. If it did not, the escape hatch the option doc
/// promises would be a lie and there would be no way to bisect a corrector
/// regression in the field.
#[test]
fn zero_correctors_still_solves_on_both_drivers() {
    let prob = degenerate_corner(0.0, 12);
    for use_hsde in [false, true] {
        let opts = QpOptions {
            use_hsde,
            gondzio_max_corr: 0,
            ..QpOptions::default()
        };
        let sol = solve_qp_ipm(&prob, &opts, backend);
        assert_eq!(
            sol.status,
            QpStatus::Optimal,
            "use_hsde={use_hsde}: correctors off must not cost the solve"
        );
    }
}

/// The default is the value the HSDE driver has always hard-coded, so the
/// option's introduction cannot have moved that driver.
#[test]
fn the_default_is_the_historical_hsde_setting() {
    assert_eq!(QpOptions::default().gondzio_max_corr, 3);
}

/// Correctors change the trajectory, never the answer.
#[test]
fn correctors_do_not_move_the_optimum() {
    for n in [4usize, 9, 15] {
        for k in 0..4 {
            let theta = k as f64;
            both_ways(
                &degenerate_corner(theta, n),
                &format!("corner n={n} θ={theta}"),
            );
            both_ways(&spread_lp(theta, n), &format!("spread n={n} θ={theta}"));
        }
    }
}

/// The scheme must pay for its back-solves on the family it targets.
///
/// Asserted in aggregate, not per instance: a corrector is accepted on the
/// step *length*, and lengthening one iteration's step can hand the next one a
/// different (occasionally worse) starting point. What must not happen is the
/// total going up — that would mean the acceptance test is passing on
/// correctors that do not help.
#[test]
fn correctors_do_not_cost_iterations_on_the_degenerate_family() {
    let (mut off_total, mut on_total) = (0usize, 0usize);
    for n in [6usize, 10, 14, 20] {
        for k in 0..5 {
            let (off, on) = both_ways(&degenerate_corner(k as f64, n), "corner");
            off_total += off;
            on_total += on;
        }
    }
    assert!(
        on_total <= off_total,
        "correctors cost iterations: {off_total} -> {on_total}"
    );
}

/// Same, on the badly-spread LP family. Kept separate from the degenerate one
/// because the two stall for different reasons and a change that helps one can
/// leave the other alone — collapsing them into one total would hide that.
#[test]
fn correctors_do_not_cost_iterations_on_the_spread_family() {
    let (mut off_total, mut on_total) = (0usize, 0usize);
    for n in [7usize, 13, 21] {
        for k in 0..5 {
            let (off, on) = both_ways(&spread_lp(k as f64, n), "spread");
            off_total += off;
            on_total += on;
        }
    }
    assert!(
        on_total <= off_total,
        "correctors cost iterations: {off_total} -> {on_total}"
    );
}

/// gh #588 (Q9a). The two families above solve **cold**, and cold is the case
/// the correctors were written for — a step blocked short, where lengthening
/// it *is* the progress. Warm is where the scheme can hurt, and nothing above
/// covers it.
///
/// It matters because warm is not a corner: the `.nl` CLI route runs HSDE, but
/// `pounce-py`'s `solve_qp` calls straight into this driver, so a warm-started
/// Python solve is the *default* path through the corrected code. The fixture
/// sweep cannot see that route at all.
///
/// The failure this pins is real and was measured, not imagined: with the
/// correctors ungated, 37 of 80 warm-started LPs each lost exactly one
/// iteration and none gained one (366 -> 403), because the band is symmetric
/// and pulls products that the affine step drove *below* `BETA_LO·μ` back up
/// to it — capping μ's descent at exactly 1/`BETA_LO` per iteration and
/// spending the superlinear tail to buy α = 1. See `correctors::ALPHA_MAX`.
#[test]
fn correctors_do_not_cost_iterations_on_a_warm_start() {
    let (mut off_total, mut on_total) = (0usize, 0usize);
    for n in [8usize, 14, 22] {
        for k in 0..8 {
            let m = (n / 2).max(2);
            let prob = boxed_lp(k as u64, n, m);
            // Solve once, perturb the linear term, re-solve from the previous
            // point — the sequential-QP shape `qp_tau_max` exists for.
            let base = solve_qp_ipm(&prob, &direct(0), backend);
            if base.status != QpStatus::Optimal {
                continue;
            }
            let warm = QpWarmStart::from_solution(&base);
            let mut next = boxed_lp(k as u64, n, m);
            for c in next.c.iter_mut() {
                *c *= 1.05;
            }
            let off = solve_qp_ipm_warm(&next, &direct(0), &warm, backend);
            let on = solve_qp_ipm_warm(
                &next,
                &direct(QpOptions::default().gondzio_max_corr),
                &warm,
                backend,
            );
            assert_eq!(off.status, on.status, "warm: the verdict changed");
            if off.status == QpStatus::Optimal {
                let scale = off.obj.abs().max(1.0);
                assert!(
                    (off.obj - on.obj).abs() <= 1e-6 * scale,
                    "warm: objective moved {} -> {}",
                    off.obj,
                    on.obj
                );
            }
            off_total += off.iters;
            on_total += on.iters;
        }
    }
    // Ungated this family runs 105 -> 117: every instance that regresses does
    // so systematically, in the same direction, for the same reason. Gated it
    // is exactly 105, and the 230-instance battery it stands in for is exactly
    // neutral (967 -> 967). See `correctors::ALPHA_MAX`.
    assert!(
        on_total <= off_total,
        "correctors cost iterations on warm starts: {off_total} -> {on_total}"
    );
}
