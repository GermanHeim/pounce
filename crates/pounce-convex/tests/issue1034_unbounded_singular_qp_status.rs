//! gh #1034 — a convex QP that is unbounded along `null(P)`, with `P` exactly
//! singular, must come back `DualInfeasible` from the active-set driver, not
//! `NumericalFailure`, `IterationLimit`, or `Optimal`.
//!
//! The issue reported the unconstrained case: integer `P = RᵀR` of rank
//! `n − 1`, `c` with a component in `null(P)`, `qp-active-set` exiting
//! `Error_In_Step_Computation` (solve_result_num 500) where `qp-ipm`, `nlp` and
//! `auto` report `Diverging_Iterates` (300). The same QP with rows or bounds
//! that leave the ray open was worse, measured on 114 instances per family
//! before the fix:
//!
//! | constraints | misreported | as |
//! |---|---|---|
//! | none | 23 | `NumericalFailure` |
//! | one equality row | 1 | `NumericalFailure` |
//! | two inequality rows | 4 | `NumericalFailure` |
//! | one-sided bounds | 54 | `IterationLimit`, `NumericalFailure`, **`Optimal` ×3** |
//! | equality + bounds | 33 | `IterationLimit`, `NumericalFailure` |
//! | equality + inequality + bounds | 33 | `NumericalFailure`, `IterationLimit` |
//!
//! `IterationLimit` tells a modelling layer to raise a limit that can never
//! help; `Optimal` at an objective of `−7.5e14` is a wrong answer about the
//! model.
//!
//! The smallest eigenvalue of such a `P` factors as round-off (`±1e-15`), not
//! zero, so no inertia shift fires and the engine's recession certificates —
//! which key on the shift — stay silent. The engine runs off along the ray
//! regardless: in 127 of the 129 misreports the returned `x` pointed along it to
//! cosine 1.000. So the driver takes the certificate from the iterate's
//! direction (projected onto `null(P)` when the engine stopped at a
//! `δ`-proximal point), and every unboundedness claim it reports — the engine's
//! own included — must pass the IPM's curvature floor, `uᵀPu/‖u‖² ≤ 1e-20`,
//! evaluated in compensated arithmetic.
//!
//! That floor is the other half of the fix and the bounded controls below pin
//! it. An engine-side alternative — letting `pounce-qp` certify its unshifted
//! solves — was measured and rejected: the engine's zero-curvature test is
//! `‖Hd‖ ≤ 1e-10·‖H‖`, and on `P + εI` with `ε ∈ {1e-10, 1e-12}` — strictly
//! convex, bounded — 28 of 228 unconstrained draws came back
//! `DualInfeasible`. The controls are bounded by construction (a blocked ray,
//! `c ∈ range(P)`, `P + εI`, gh #293's mixed scales), and none may read as
//! unbounded.
//!
//! Oracle: closed form. `d` is the integer cofactor vector of `R`, so `Rd = 0`
//! and `Pd = 0` exactly; each family's constraints are built so `d` (signed to
//! descend) stays a feasible recession direction. Every unbounded assertion is
//! also cross-checked against POUNCE's IPM on identical data.

use pounce_convex::{
    ActiveSetOverrides, NEG_INF, POS_INF, QpOptions, QpProblem, QpStatus, Triplet,
    solve_qp_active_set, solve_qp_ipm,
};
use pounce_feral::FeralSolverInterface;
use pounce_linsol::SparseSymLinearSolverInterface;

fn backend() -> Box<dyn SparseSymLinearSolverInterface> {
    Box::new(FeralSolverInterface::new())
}

fn active_set(prob: &QpProblem) -> QpStatus {
    let mut mk = backend;
    solve_qp_active_set(
        prob,
        &QpOptions::default(),
        &ActiveSetOverrides::default(),
        &mut mk,
    )
    .status
}

fn ipm(prob: &QpProblem) -> QpStatus {
    let mut mk = backend;
    solve_qp_ipm(prob, &QpOptions::default(), &mut mk).status
}

/// `min ½xᵀPx + cᵀx`, no rows, no bounds, `P` dense row-major.
fn unconstrained(p: &[Vec<f64>], c: &[f64]) -> QpProblem {
    let n = c.len();
    let mut p_lower = vec![];
    for (i, row) in p.iter().enumerate() {
        for (j, &v) in row.iter().enumerate().take(i + 1) {
            if v != 0.0 {
                p_lower.push(Triplet::new(i, j, v));
            }
        }
    }
    QpProblem {
        n,
        p_lower,
        c: c.to_vec(),
        a: vec![],
        b: vec![],
        g: vec![],
        h: vec![],
        lb: vec![],
        ub: vec![],
    }
}

/// The issue's matrix: `P = RᵀR`, rank 3, `null(P) = span{(0,-2,-1,1)}`.
fn issue_p() -> Vec<Vec<f64>> {
    vec![
        vec![1., 1., -1., 1.],
        vec![1., 2., -2., 2.],
        vec![-1., -2., 3., -1.],
        vec![1., 2., -1., 3.],
    ]
}

#[test]
fn issue_qp_is_dual_infeasible() {
    // c·d = 4 for d = (0,-2,-1,1): f(x − t·d) = f(x) − 4t.
    let prob = unconstrained(&issue_p(), &[-1., -1., -1., 1.]);
    assert_eq!(ipm(&prob), QpStatus::DualInfeasible);
    assert_eq!(active_set(&prob), QpStatus::DualInfeasible);
}

#[test]
fn same_singular_p_with_c_in_range_stays_optimal() {
    // c = P·(1,0,0,0) lies in range(P): bounded, minimum on the line −e₀ + t·d.
    let prob = unconstrained(&issue_p(), &[1., 1., -1., 1.]);
    assert_eq!(ipm(&prob), QpStatus::Optimal);
    assert_eq!(active_set(&prob), QpStatus::Optimal);
}

/// The wrong-answer class: before the fix this returned `Optimal` with
/// `x ≈ (1.5e15, 2.8, −1.5e15, −3.0e15)` and objective `−7.5e14`.
#[test]
fn box_runaway_is_not_reported_optimal() {
    let r = [[-1., 0., 1., -1.], [1., 0., 1., 0.], [-1., 1., -1., 0.]];
    let p: Vec<Vec<f64>> = (0..4)
        .map(|i| {
            (0..4)
                .map(|j| r.iter().map(|row| row[i] * row[j]).sum())
                .collect()
        })
        .collect();
    // d = (1, 0, −1, −2), c·d = −1: descend along +d, which the one-sided
    // bounds below leave open; x₁ is boxed.
    let mut prob = unconstrained(&p, &[0., -1., -1., 1.]);
    prob.lb = vec![-2.0, -2.0, NEG_INF, NEG_INF];
    prob.ub = vec![POS_INF, 3.0, 3.0, 3.0];
    assert_eq!(ipm(&prob), QpStatus::DualInfeasible);
    assert_eq!(active_set(&prob), QpStatus::DualInfeasible);
}

/// `P + 1e-10·I` on the issue's matrix: strictly convex, minimizer near
/// `x ≈ 4e10·d/‖d‖²`. The engine's `‖Hd‖ ≤ 1e-10·‖H‖` test calls that
/// direction flat; the driver must not report the claim.
#[test]
fn near_singular_strictly_convex_is_not_unbounded() {
    for eps in [1e-6, 1e-8, 1e-10, 1e-12] {
        let mut p = issue_p();
        for (i, row) in p.iter_mut().enumerate() {
            row[i] += eps;
        }
        let prob = unconstrained(&p, &[-1., -1., -1., 1.]);
        assert_ne!(active_set(&prob), QpStatus::DualInfeasible, "eps = {eps:e}");
    }
}

/// gh #293's mixed-scale Hessians — bounded, with tiny but real curvature.
#[test]
fn mixed_scale_bounded_is_not_unbounded() {
    for (a, b) in [(1e6, 1e-12), (1.0, 1e-10), (1.0, 1e-12), (1e3, 1e-14)] {
        for cc in [-1.0, -100.0] {
            let p = vec![vec![a, 0.0], vec![0.0, b]];
            let prob = unconstrained(&p, &[0.0, cc]);
            assert_ne!(
                active_set(&prob),
                QpStatus::DualInfeasible,
                "P = diag({a:e}, {b:e}), c₁ = {cc}"
            );
        }
    }
}

/// Tiny deterministic generator so the sweep is reproducible without a dep.
struct Lcg(u64);
impl Lcg {
    fn tri(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 33) % 3) as f64 - 1.0
    }
}

fn det(m: &[Vec<f64>]) -> f64 {
    match m.len() {
        0 => 1.0,
        1 => m[0][0],
        _ => (0..m.len())
            .filter(|&j| m[0][j] != 0.0)
            .map(|j| {
                let minor: Vec<Vec<f64>> = m[1..]
                    .iter()
                    .map(|r| {
                        r.iter()
                            .enumerate()
                            .filter(|&(k, _)| k != j)
                            .map(|(_, &v)| v)
                            .collect()
                    })
                    .collect();
                let sign = if j % 2 == 0 { 1.0 } else { -1.0 };
                sign * m[0][j] * det(&minor)
            })
            .sum(),
    }
}

/// The cofactor vector of an `(n−1)×n` integer `R`: `R·d = 0` exactly.
fn null_vector(r: &[Vec<f64>], n: usize) -> Vec<f64> {
    (0..n)
        .map(|j| {
            let minor: Vec<Vec<f64>> = r
                .iter()
                .map(|row| {
                    row.iter()
                        .enumerate()
                        .filter(|&(k, _)| k != j)
                        .map(|(_, &v)| v)
                        .collect()
                })
                .collect();
            if j % 2 == 0 {
                det(&minor)
            } else {
                -det(&minor)
            }
        })
        .collect()
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// A nonzero integer row `a` with `a·v = 0` (`orthogonal`) or `a·v ≤ 0`.
fn row_along(rng: &mut Lcg, v: &[f64], orthogonal: bool) -> Vec<f64> {
    loop {
        let a: Vec<f64> = (0..v.len()).map(|_| rng.tri()).collect();
        let (av, vv) = (dot(&a, v), dot(v, v));
        let row: Vec<f64> = if orthogonal {
            a.iter().zip(v).map(|(ai, vi)| vv * ai - av * vi).collect()
        } else if av > 0.0 {
            a.iter().map(|x| -x).collect()
        } else {
            a
        };
        if row.iter().any(|&x| x != 0.0) {
            return row;
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum Family {
    Free,
    Eq,
    Ineq,
    /// One-sided bounds in the ray's direction, a box where it does not move.
    Box,
    EqBox,
    EqIneqBox,
}

/// One draw of the issue's family, with `fam`'s constraints added so that the
/// descent direction along `null(P)` stays a feasible recession direction.
/// `None` when the draw has no such direction (`R` rank-deficient, or `c ⟂ d`).
fn unbounded_draw(rng: &mut Lcg, n: usize, fam: Family) -> Option<QpProblem> {
    let r: Vec<Vec<f64>> = (0..n - 1)
        .map(|_| (0..n).map(|_| rng.tri()).collect())
        .collect();
    let c: Vec<f64> = (0..n).map(|_| rng.tri()).collect();
    let d = null_vector(&r, n);
    let cd = dot(&c, &d);
    if cd == 0.0 {
        return None;
    }
    let v: Vec<f64> = d.iter().map(|x| -cd.signum() * x).collect();
    let p: Vec<Vec<f64>> = (0..n)
        .map(|i| {
            (0..n)
                .map(|j| r.iter().map(|row| row[i] * row[j]).sum())
                .collect()
        })
        .collect();
    let mut prob = unconstrained(&p, &c);
    if matches!(fam, Family::Eq | Family::EqBox | Family::EqIneqBox) {
        let a = row_along(rng, &v, true);
        prob.a = a
            .iter()
            .enumerate()
            .filter(|&(_, &x)| x != 0.0)
            .map(|(j, &x)| Triplet::new(0, j, x))
            .collect();
        prob.b = vec![dot(&a, &vec![0.5; n])];
    }
    if matches!(fam, Family::Ineq | Family::EqIneqBox) {
        for k in 0..2 {
            let g = row_along(rng, &v, false);
            prob.g.extend(
                g.iter()
                    .enumerate()
                    .filter(|&(_, &x)| x != 0.0)
                    .map(|(j, &x)| Triplet::new(k, j, x)),
            );
            prob.h.push(5.0);
        }
    }
    if matches!(fam, Family::Box | Family::EqBox | Family::EqIneqBox) {
        prob.lb = vec![NEG_INF; n];
        prob.ub = vec![POS_INF; n];
        for i in 0..n {
            if v[i] == 0.0 {
                (prob.lb[i], prob.ub[i]) = (-2.0, 3.0);
            } else if v[i] > 0.0 {
                prob.lb[i] = -2.0;
            } else {
                prob.ub[i] = 3.0;
            }
        }
    }
    Some(prob)
}

/// Every family, `n = 3..6`: the active-set verdict is `DualInfeasible`, and
/// the IPM agrees, on every draw. Counted so the sweep cannot pass vacuously.
#[test]
fn every_open_ray_family_is_dual_infeasible() {
    let mut rng = Lcg(77);
    for fam in [
        Family::Free,
        Family::Eq,
        Family::Ineq,
        Family::Box,
        Family::EqBox,
        Family::EqIneqBox,
    ] {
        let mut drawn = 0;
        for n in 3..=6 {
            for _ in 0..25 {
                let Some(prob) = unbounded_draw(&mut rng, n, fam) else {
                    continue;
                };
                drawn += 1;
                assert_eq!(ipm(&prob), QpStatus::DualInfeasible, "{fam:?} n = {n}");
                assert_eq!(
                    active_set(&prob),
                    QpStatus::DualInfeasible,
                    "{fam:?} n = {n}, P = {:?}, c = {:?}",
                    prob.p_lower,
                    prob.c
                );
            }
        }
        assert!(drawn >= 60, "{fam:?}: only {drawn} draws");
    }
}

/// The same draws made bounded — every coordinate the ray moves boxed, or a
/// row `vᵀx ≤ 7` across it, or `P + εI` — must never read as unbounded.
#[test]
fn bounded_controls_are_never_dual_infeasible() {
    let mut rng = Lcg(4242);
    let mut checked = 0;
    for n in 3..=6 {
        for _ in 0..25 {
            let Some(base) = unbounded_draw(&mut rng, n, Family::Box) else {
                continue;
            };
            // Recover the open direction from the one-sided bounds just built.
            let dir: Vec<f64> = (0..n)
                .map(|i| match (base.lb[i] > NEG_INF, base.ub[i] < POS_INF) {
                    (true, false) => 1.0,
                    (false, true) => -1.0,
                    _ => 0.0,
                })
                .collect();

            let mut blocked = base.clone();
            for i in 0..n {
                if dir[i] > 0.0 {
                    blocked.ub[i] = 4.0;
                } else if dir[i] < 0.0 {
                    blocked.lb[i] = -4.0;
                }
            }

            let mut row = unconstrained(&[], &[]);
            row.n = n;
            row.p_lower = base.p_lower.clone();
            row.c = base.c.clone();
            row.lb = base.lb.clone();
            row.ub = base.ub.clone();
            for i in 0..n {
                if dir[i] != 0.0 {
                    (row.lb[i], row.ub[i]) = (NEG_INF, POS_INF);
                }
            }
            row.g = (0..n)
                .filter(|&i| dir[i] != 0.0)
                .map(|i| Triplet::new(0, i, dir[i]))
                .collect();
            row.h = vec![7.0];

            let mut controls = vec![("blocked box", blocked), ("blocked row", row)];
            // Without rows or bounds too: that is where the rejected
            // engine-side fix misread 28 of 228 `P + εI` draws as unbounded.
            let free = unbounded_draw(&mut rng, n, Family::Free);
            for eps in [1e-8, 1e-10, 1e-12] {
                for (what, open) in [
                    ("P + eps I, open bounds", Some(&base)),
                    ("P + eps I, free", free.as_ref()),
                ] {
                    let Some(open) = open else { continue };
                    let mut shifted = open.clone();
                    shifted
                        .p_lower
                        .extend((0..n).map(|i| Triplet::new(i, i, eps)));
                    controls.push((what, shifted));
                }
            }
            for (what, prob) in controls {
                checked += 1;
                assert_ne!(
                    active_set(&prob),
                    QpStatus::DualInfeasible,
                    "{what}, n = {n}, P = {:?}, c = {:?}",
                    prob.p_lower,
                    prob.c
                );
            }
        }
    }
    assert!(checked >= 300, "only {checked} controls");
}
