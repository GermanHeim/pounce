//! Regression for gh #414 **reopened**: the cost normalization `σ` must not buy
//! an `Optimal` verdict at a point whose KKT error, in the caller's own
//! coordinates, is orders above `tol`.
//!
//! The original report was fixed by measuring a claimed optimum in the Ruiz-
//! equilibrated metric and repairing it (`issue414_varscale_false_optimal.rs`).
//! The nightly adversary run then reproduced the *title* symptom — `optimal` /
//! `success=True` at a point whose own reported `kkt_error` is far above `tol` —
//! on a far simpler instance, and through a different door.
//!
//! # The door
//!
//! Before handing a QP to the embedding, `solve_qp_core` divides the objective
//! data by `σ = 2^⌈log₂ max(‖P‖∞, ‖c‖∞)⌉` (gh #286: an `O(1)` objective keeps the
//! homogeneous `τ` off the certificate boundary). The embedding's *absolute*
//! stopping test then runs in the `σ` metric, so what it certifies is
//! `‖r‖ ≤ tol` on the scaled data — i.e. `‖r‖ ≤ σ·tol` in the caller's.
//!
//! That is a *relative* test wearing an absolute one's clothes, and it never
//! passes through the gate that decides whether a relative test is admissible
//! (`hsde::relative_stop_permitted`). Worse, the quantity it is implicitly
//! relative *to* is the wrong one: `σ` is sized by the objective **coefficient**
//! magnitude, while a stationarity residual has to be small against the
//! **gradient** scale `‖Px*‖∞ ∨ ‖c‖∞`. The two differ by `‖x*‖`, and the gap is
//! unbounded below — every problem whose optimum is small pays it.
//!
//! `min (x₀ − 1)² + (10⁴x₁ − 1)²` is the whole story in two variables:
//! `‖P‖∞ = 2e8` so `σ = 2²⁸ ≈ 2.7e8`, but `x* = (1, 1e-4)` makes the gradient
//! scale only `2e4`. The embedding stopped after 3 iterations at
//! `‖Px+c‖∞ = 2.499` — `tol`-accurate against `2e8`, and `1.2e-4` relative
//! against the `2e4` that governs — reporting `Optimal`, `x` wrong by `2.5e-4`
//! relative.
//!
//! # The fix, and why the corpus never saw it
//!
//! `normalized_optimum_is_genuine` (gh #324's re-check of a `σ`-path `Optimal`)
//! already measured the right ratio. It was cut at a flat `1e-3`, calibrated
//! against gh #324's *cold-start* failure, which is `O(1)`. This family lands
//! between the two — genuinely wrong, and five orders inside the cut — so the
//! guard waved it through. The cut is now reached only where a relative test is
//! admissible at all, and the un-normalized re-solve gh #324 already wired up
//! behind that guard recovers every instance below in **one iteration**.
//!
//! The objective is why an objective-parity corpus is blind here: it is
//! second-order in the `x` error at the optimum, so at `x` wrong by `2.5e-4` the
//! objective is right to ~8 digits. Only a KKT-residual or `x`-error check sees
//! it — the `benchmarks`/`sweep-fixtures` blind spot in CLAUDE.md, one level
//! down.
//!
//! # Oracles
//!
//! Every fixture here is **diagonal and separable**, so the exact minimizer is
//! the closed form `x*ᵢ = −cᵢ/Pᵢᵢ` — an oracle independent of any solver, which
//! is what the assertions below use. The issue additionally reports clarabel at
//! `tol=1e-12` agreeing with that closed form to `1.4e-16` relative.

use pounce_convex::{QpOptions, QpProblem, QpStatus, Triplet, solve_qp_ipm, solve_socp_ipm};
use pounce_feral::FeralSolverInterface;
use pounce_linsol::SparseSymLinearSolverInterface;

fn backend() -> Box<dyn SparseSymLinearSolverInterface> {
    Box::new(FeralSolverInterface::new())
}

/// `min Σᵢ (sᵢxᵢ − 1)²`, written as `½xᵀPx + cᵀx` with `P = diag(2sᵢ²)` and
/// `cᵢ = −2sᵢ`. Separable and diagonal ⇒ `x*ᵢ = 1/sᵢ` exactly, and
/// `‖P‖∞ = 2·max sᵢ²` while the gradient scale at `x*` is only `2·max sᵢ` —
/// the `σ`-to-gradient gap this file is about, dialled by `span`.
fn separable(scales: &[f64]) -> (QpProblem, Vec<f64>) {
    let p_lower = scales
        .iter()
        .enumerate()
        .map(|(i, s)| Triplet::new(i, i, 2.0 * s * s))
        .collect();
    (
        QpProblem {
            n: scales.len(),
            p_lower,
            c: scales.iter().map(|s| -2.0 * s).collect(),
            a: vec![],
            b: vec![],
            g: vec![],
            h: vec![],
            lb: vec![],
            ub: vec![],
        },
        scales.iter().map(|s| 1.0 / s).collect(),
    )
}

fn max_rel_x_err(x: &[f64], exact: &[f64]) -> f64 {
    x.iter()
        .zip(exact)
        .map(|(v, e)| ((v - e) / e).abs())
        .fold(0.0_f64, f64::max)
}

/// The instance in the reopened report, asserted on all three of the things it
/// contradicted: the status, the point, and the solver's own `kkt_error`.
///
/// `P = diag(2, 2e8)`, `c = (−2, −2e4)`, no constraints. Before the fix:
/// `Optimal`, `kkt_error = 2.499`, `x` off by `2.47e-4` relative, 3 iterations.
#[test]
fn issue_414_two_variable_diagonal_qp_is_not_false_optimal() {
    let (prob, exact) = separable(&[1.0, 1e4]);
    let opts = QpOptions::default();
    let sol = solve_qp_ipm(&prob, &opts, backend);

    assert_eq!(sol.status, QpStatus::Optimal, "the instance is trivial");
    let kkt = sol.kkt_residuals(&prob).kkt_error();
    // The heart of the report: a result carrying this `kkt_error` must not be
    // labelled `Optimal`. Here it is genuinely small, so it is.
    assert!(
        kkt <= opts.tol,
        "Optimal at kkt_error {kkt:.4e} > tol {:.0e} (the issue reports 2.499)",
        opts.tol
    );
    let rel = max_rel_x_err(&sol.x, &exact);
    assert!(
        rel < 1e-9,
        "x = {:?} off by {rel:.2e} from the closed form {exact:?} \
         (the issue reports 2.47e-4)",
        sol.x
    );
}

/// The band, swept. gh #414's comment reports the damage peaking around
/// `cond(P) ∈ [1e7.4, 1e8.6]` and — the detail that makes a spot check useless —
/// **not** being monotone in `cond(P)`: at `span = 4.4` and `5.0` the returned
/// `x` was already accurate to `1e-10` while `span = 4.0` and `4.3` were wrong
/// by `1e-3`. A fixture at one span says nothing about its neighbour, so the
/// sweep is the test.
///
/// `span = 0.0` and `3.5` sit below the `σ` gate (`σ = 1`, no normalization at
/// all) and were always correct; they are here so the fix cannot be credited
/// for a regime it does not touch, and cannot regress it either.
#[test]
fn issue_414_the_whole_conditioning_band_is_solved_not_just_its_ends() {
    for span in [0.0, 3.5, 3.7, 4.0, 4.3, 4.4, 5.0, 6.0] {
        let scales: Vec<f64> = (0..4).map(|k| 10f64.powf(span * k as f64 / 3.0)).collect();
        let (prob, exact) = separable(&scales);
        let opts = QpOptions::default();
        let sol = solve_qp_ipm(&prob, &opts, backend);

        assert_eq!(sol.status, QpStatus::Optimal, "span {span}");
        let kkt = sol.kkt_residuals(&prob).kkt_error();
        assert!(
            kkt <= opts.tol,
            "span {span}: Optimal at kkt_error {kkt:.4e} > tol {:.0e}",
            opts.tol
        );
        let rel = max_rel_x_err(&sol.x, &exact);
        assert!(rel < 1e-9, "span {span}: x off by {rel:.2e}");
    }
}

/// The defect is independent of `n` (the issue reports identical numbers for
/// `n = 2..10` at `span = 4`), because it is a property of the ratio between
/// the largest coefficient and the gradient at the optimum, not of dimension.
#[test]
fn issue_414_is_independent_of_dimension() {
    for n in 2..=10 {
        let scales: Vec<f64> = (0..n)
            .map(|k| 10f64.powf(4.0 * k as f64 / (n - 1) as f64))
            .collect();
        let (prob, exact) = separable(&scales);
        let opts = QpOptions::default();
        let sol = solve_qp_ipm(&prob, &opts, backend);

        assert_eq!(sol.status, QpStatus::Optimal, "n = {n}");
        let kkt = sol.kkt_residuals(&prob).kkt_error();
        assert!(kkt <= opts.tol, "n = {n}: Optimal at kkt_error {kkt:.4e}");
        assert!(
            max_rel_x_err(&sol.x, &exact) < 1e-9,
            "n = {n}: x off by {:.2e}",
            max_rel_x_err(&sol.x, &exact)
        );
    }
}

/// Tightening `tol` must actually buy accuracy. The issue's sharpest evidence
/// that this was not a tolerance-philosophy disagreement is that it did not:
/// `tol=1e-10` gave `rel x err 2.6e-6` and `tol=1e-12` gave `1.7e-6` — *worse*
/// than the default's `2.5e-4`-scale failure only in the sense that the caller
/// had asked for more and been told they got it. Measured under the fix:
/// `5.0e-11`, `5.0e-11`, `1.3e-13`.
///
/// The residual assertion has a floor and the floor is not a hedge. At
/// `tol = 1e-12` the gradient scale is `2e4`, so one ulp of the quantity being
/// differenced is `ε·2e4 = 4.4e-12` and no arithmetic can put `‖Px+c‖∞` under
/// `1e-12`. That is the regime the relative arm legitimately exists for — the
/// solve stops at `3.6e-12`, inside the floor, with `x` correct to 13 digits.
/// Asserting `kkt ≤ tol` flat would fail on correct code, which is the mistake
/// this whole issue is about, run in the other direction.
#[test]
fn issue_414_a_tighter_tol_is_honoured_not_absorbed() {
    let (prob, exact) = separable(&[1.0, 1e4]);
    // max(‖Px*‖∞, ‖c‖∞) — the scale the stationarity residual is differenced at.
    let gscale = 2e4_f64;
    for tol in [1e-8, 1e-10, 1e-11, 1e-12, 1e-14] {
        let opts = QpOptions {
            tol,
            ..QpOptions::default()
        };
        let sol = solve_qp_ipm(&prob, &opts, backend);
        assert_eq!(sol.status, QpStatus::Optimal, "tol {tol:.0e}");

        let kkt = sol.kkt_residuals(&prob).kkt_error();
        let floor = f64::EPSILON * gscale;
        assert!(
            kkt <= tol.max(floor),
            "tol {tol:.0e}: Optimal at kkt_error {kkt:.4e}, above both tol and              the {floor:.2e} finite-precision floor"
        );
        // The point itself, against the closed form — the assertion the issue's
        // numbers are about, and the one no residual convention can soften.
        let rel = max_rel_x_err(&sol.x, &exact);
        assert!(
            rel < 1e-9,
            "tol {tol:.0e}: x off by {rel:.2e} (the issue reports 2.6e-6 at              1e-10 and 1.7e-6 at 1e-12)"
        );
    }
}

/// `socp` and `auto` inherit the defect because they route to the same engine —
/// stated in the original report and still true of this door, so the entry
/// point reached through `solve_socp_ipm` is pinned on the same instance.
///
/// **The cone machinery is not the subject, and no cone-carrying instance in
/// this family reaches the defect.** That is a measurement, not an omission:
/// adding *any* inequality row to the two-variable instance — `x ≥ -1`,
/// `x ≥ -1e6`, a single row on either variable, `Σx ≤ 1e6` — gives the
/// embedding a real trajectory (12‥21 iterations instead of 3) and it converges
/// to `x` correct at `1e-12`‥`1e-14` on the parent commit, defect and all. The
/// same holds sweeping `span` 3.7‥6.0 at `n = 2` and `n = 4`. What this door
/// therefore pins is that `solve_socp_ipm` reaches the same `solve_qp_core`
/// verdict check, on the one geometry that exhibits the failure — an empty cone
/// list, matching the QP fixture above.
///
/// If a future change makes a cone-carrying instance reach it, that instance
/// belongs here and this comment is the record that none did.
#[test]
fn issue_414_the_socp_entry_point_is_covered_too() {
    let (prob, exact) = separable(&[1.0, 1e4]);
    let opts = QpOptions::default();
    let cones: [pounce_convex::ConeSpec; 0] = [];
    let sol = solve_socp_ipm(&prob, &cones, &opts, backend);

    assert_eq!(sol.status, QpStatus::Optimal);
    let kkt = sol.kkt_residuals_conic(&prob, &cones).kkt_error();
    assert!(
        kkt <= opts.tol,
        "socp: Optimal at kkt_error {kkt:.4e} > tol {:.0e}",
        opts.tol
    );
    let rel = max_rel_x_err(&sol.x, &exact);
    assert!(rel < 1e-9, "socp: x off by {rel:.2e}");
}
