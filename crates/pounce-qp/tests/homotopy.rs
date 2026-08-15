//! §4.2 parametric homotopy — does it reach the same answer as the
//! conventional phase-1/phase-2 path?
//!
//! The homotopy is the algorithm this crate is named for and was previously
//! unimplemented (`solve_parametric` was a stub). It is off by default while
//! being evaluated, so these tests drive it explicitly via
//! `QpOptions::use_homotopy` and assert it agrees with the conventional path on
//! problems with closed-form optima.
//!
//! Scope note, so a future reader is not misled by green tests: the homotopy
//! currently starts from the **box-only relaxation**, which means it cannot
//! start when that relaxation is unbounded (`H` singular in a box-unbounded
//! direction — most LP-like instances), and it has no anti-cycling for
//! *coincident* events, so it can stall at a degenerate parameter value. Both
//! cases return `Ok(None)` internally and fall back to the conventional path, so
//! they are invisible in results but real. See `crates/pounce-qp/src/homotopy.rs`.

use pounce_common::types::{NLP_LOWER_BOUND_INF, NLP_UPPER_BOUND_INF};
use pounce_linalg::triplet::{GenTMatrix, GenTMatrixSpace, SymTMatrix, SymTMatrixSpace};
use pounce_qp::{
    HessianInertia, ParametricActiveSetSolver, QpOptions, QpProblem, QpSolver, QpStatus,
};
use std::rc::Rc;

fn new_solver() -> ParametricActiveSetSolver {
    ParametricActiveSetSolver::new(Box::new(pounce_feral::FeralSolverInterface::new()))
}

struct Case {
    n: usize,
    m: usize,
    h: (Vec<i32>, Vec<i32>, Vec<f64>),
    g: Vec<f64>,
    a: (Vec<i32>, Vec<i32>, Vec<f64>),
    bl: Vec<f64>,
    bu: Vec<f64>,
    xl: Vec<f64>,
    xu: Vec<f64>,
}

fn solve(case: &Case, homotopy: bool) -> (QpStatus, Vec<f64>, f64) {
    let h_space = SymTMatrixSpace::new(case.n as i32, case.h.0.clone(), case.h.1.clone());
    let mut h = SymTMatrix::new(Rc::clone(&h_space));
    h.set_values(&case.h.2);

    let a_space = GenTMatrixSpace::new(
        case.m as i32,
        case.n as i32,
        case.a.0.clone(),
        case.a.1.clone(),
    );
    let mut a = GenTMatrix::new(Rc::clone(&a_space));
    a.set_values(&case.a.2);

    let qp = QpProblem {
        n: case.n,
        m: case.m,
        h: &h,
        g: &case.g,
        a: &a,
        bl: &case.bl,
        bu: &case.bu,
        xl: &case.xl,
        xu: &case.xu,
        hessian_inertia: HessianInertia::Psd,
    };
    let opts = QpOptions {
        use_homotopy: homotopy,
        ..QpOptions::default()
    };
    let sol = new_solver().solve(&qp, None, &opts).expect("qp solve");
    (sol.status, sol.x.clone(), sol.obj)
}

/// `min (x₀−3)² + (x₁−2)²  s.t.  x₀ + x₁ ≤ 4,  x ≥ 0`, in `½xᵀHx + gᵀx` form:
/// `H = 2I`, `g = (−6, −4)` (constant 13 dropped).
///
/// `H` is positive definite so the box relaxation is bounded and the homotopy
/// can start. The unconstrained optimum `(3, 2)` violates the row, so it binds:
/// `x* = (2.5, 1.5)`, and `obj = 6.25 + 2.25 − 15 − 6 = −12.5`.
fn projection_case() -> Case {
    Case {
        n: 2,
        m: 1,
        h: (vec![1, 2], vec![1, 2], vec![2.0, 2.0]),
        g: vec![-6.0, -4.0],
        a: (vec![1, 1], vec![1, 2], vec![1.0, 1.0]),
        bl: vec![NLP_LOWER_BOUND_INF],
        bu: vec![4.0],
        xl: vec![0.0, 0.0],
        xu: vec![NLP_UPPER_BOUND_INF, NLP_UPPER_BOUND_INF],
    }
}

/// Two rows, both binding at the optimum, so the path must add more than one
/// constraint on the way to `t = 1`.
///
/// `min ½(x₀² + x₁²) − 4x₀ − 4x₁  s.t.  x₀ + x₁ ≤ 4,  x₀ ≤ 1.5,  x ≥ 0`.
/// Unconstrained optimum is `(4, 4)`. With `x₀ ≤ 1.5` binding, minimizing over
/// `x₁` subject to `x₀ + x₁ ≤ 4` gives `x₁ = 2.5`, so `x* = (1.5, 2.5)` and
/// `obj = ½(2.25 + 6.25) − 6 − 10 = 4.25 − 16 = −11.75`.
fn two_active_case() -> Case {
    Case {
        n: 2,
        m: 2,
        h: (vec![1, 2], vec![1, 2], vec![1.0, 1.0]),
        g: vec![-4.0, -4.0],
        a: (vec![1, 1, 2], vec![1, 2, 1], vec![1.0, 1.0, 1.0]),
        bl: vec![NLP_LOWER_BOUND_INF, NLP_LOWER_BOUND_INF],
        bu: vec![4.0, 1.5],
        xl: vec![0.0, 0.0],
        xu: vec![NLP_UPPER_BOUND_INF, NLP_UPPER_BOUND_INF],
    }
}

#[test]
fn homotopy_solves_projection_qp() {
    let (status, x, obj) = solve(&projection_case(), true);
    assert_eq!(status, QpStatus::Optimal, "status");
    assert!((x[0] - 2.5).abs() < 1e-8, "x0 = {}", x[0]);
    assert!((x[1] - 1.5).abs() < 1e-8, "x1 = {}", x[1]);
    assert!((obj + 12.5).abs() < 1e-8, "obj = {obj}");
}

#[test]
fn homotopy_solves_two_active_rows() {
    let (status, x, obj) = solve(&two_active_case(), true);
    assert_eq!(status, QpStatus::Optimal, "status");
    assert!((x[0] - 1.5).abs() < 1e-8, "x0 = {}", x[0]);
    assert!((x[1] - 2.5).abs() < 1e-8, "x1 = {}", x[1]);
    assert!((obj + 11.75).abs() < 1e-8, "obj = {obj}");
}

/// The homotopy is a *path*, not a different answer: it must agree with the
/// conventional phase-1/phase-2 path everywhere it runs. Any disagreement means
/// one of the two is wrong, which is the property worth pinning.
#[test]
fn homotopy_agrees_with_conventional_path() {
    for (name, case) in [
        ("projection", projection_case()),
        ("two_active", two_active_case()),
    ] {
        let (cs, cx, cobj) = solve(&case, false);
        let (hs, hx, hobj) = solve(&case, true);
        assert_eq!(hs, cs, "{name}: status differs");
        for i in 0..case.n {
            assert!(
                (hx[i] - cx[i]).abs() < 1e-7,
                "{name}: x[{i}] homotopy {} vs conventional {}",
                hx[i],
                cx[i]
            );
        }
        assert!(
            (hobj - cobj).abs() < 1e-7,
            "{name}: obj homotopy {hobj} vs conventional {cobj}"
        );
    }
}

// ---------------------------------------------------------------------------
// Parametric (warm) solves — `QpSolver::solve_parametric`.
//
// This was a stub that discarded both prior arguments and cold-solved, despite
// the crate advertising "true parametric warm starting". It now traces the
// homotopy from the previous problem to the new one, starting from the previous
// solution's working set.
// ---------------------------------------------------------------------------

/// Solve `case` from cold, then re-solve a `g`-perturbed version parametrically
/// from that solution, and return both the warm result and the cold result for
/// the *same* perturbed problem.
fn parametric_vs_cold(
    case: &Case,
    dg: &[f64],
) -> ((QpStatus, Vec<f64>, f64), (QpStatus, Vec<f64>, f64)) {
    let h_space = SymTMatrixSpace::new(case.n as i32, case.h.0.clone(), case.h.1.clone());
    let mut h = SymTMatrix::new(Rc::clone(&h_space));
    h.set_values(&case.h.2);
    let a_space = GenTMatrixSpace::new(
        case.m as i32,
        case.n as i32,
        case.a.0.clone(),
        case.a.1.clone(),
    );
    let mut a = GenTMatrix::new(Rc::clone(&a_space));
    a.set_values(&case.a.2);

    let g_new: Vec<f64> = case.g.iter().zip(dg).map(|(a, b)| a + b).collect();
    let opts = QpOptions {
        use_homotopy: true,
        ..QpOptions::default()
    };
    // A closure returning `QpProblem<'_>` cannot tie the borrow of `g` to the
    // returned value's lifetime, so build them explicitly.
    macro_rules! mk {
        ($g:expr) => {
            QpProblem {
                n: case.n,
                m: case.m,
                h: &h,
                g: $g,
                a: &a,
                bl: &case.bl,
                bu: &case.bu,
                xl: &case.xl,
                xu: &case.xu,
                hessian_inertia: HessianInertia::Psd,
            }
        };
    }

    let mut s = new_solver();
    let prev = s.solve(&mk!(&case.g), None, &opts).expect("cold prev");
    let warm = s
        .solve_parametric(&mk!(&case.g), &prev, &mk!(&g_new), &opts)
        .expect("parametric");
    let cold = new_solver()
        .solve(&mk!(&g_new), None, &opts)
        .expect("cold new");
    (
        (warm.status, warm.x.clone(), warm.obj),
        (cold.status, cold.x.clone(), cold.obj),
    )
}

/// A warm parametric solve must land on the same answer as a cold solve of the
/// same target. Warm starting is a route, not a different problem.
#[test]
fn parametric_matches_cold_solve() {
    for (name, case, dg) in [
        ("projection", projection_case(), vec![0.5, -0.25]),
        ("two_active", two_active_case(), vec![-1.0, 0.75]),
    ] {
        let ((ws, wx, wobj), (cs, cx, cobj)) = parametric_vs_cold(&case, &dg);
        assert_eq!(ws, cs, "{name}: status warm {ws:?} vs cold {cs:?}");
        for i in 0..case.n {
            assert!(
                (wx[i] - cx[i]).abs() < 1e-7,
                "{name}: x[{i}] warm {} vs cold {}",
                wx[i],
                cx[i]
            );
        }
        assert!(
            (wobj - cobj).abs() < 1e-7,
            "{name}: obj warm {wobj} vs cold {cobj}"
        );
    }
}

/// Re-solving an **unchanged** QP parametrically must be nearly free: the path
/// has zero length, so no constraint can reach a bound and no multiplier can
/// reach zero along it. This is the property that makes warm starting worth
/// having, and the one a stub silently fails while still returning the right
/// answer — so asserting the answer alone would not catch a regression here.
#[test]
fn parametric_on_unchanged_qp_is_free() {
    for (name, case) in [
        ("projection", projection_case()),
        ("two_active", two_active_case()),
    ] {
        let ((ws, wx, wobj), (_, cx, cobj)) = parametric_vs_cold(&case, &vec![0.0; case.n]);
        assert_eq!(ws, QpStatus::Optimal, "{name}: status");
        for i in 0..case.n {
            assert!(
                (wx[i] - cx[i]).abs() < 1e-9,
                "{name}: x[{i}] moved on an unchanged re-solve"
            );
        }
        assert!(
            (wobj - cobj).abs() < 1e-9,
            "{name}: obj moved on an unchanged re-solve"
        );
    }
}

// ---------------------------------------------------------------------------
// The *ineligible* parametric path — `qp_prev` and `qp_new` differ in a way
// the homotopy cannot trace, so `solve_parametric` never reaches the path.
//
// That branch used to cold-solve, discarding the working set the caller had
// just handed over. It now re-uses it through `solve_with_working_set` (gh
// #602). The two problems below differ in `H`, which the guard rejects on
// because `H` is not interpolated along the path.
// ---------------------------------------------------------------------------

/// `n = 12`, `m = 8`, diagonal PD `H`, every row one-sided and several binding
/// at the optimum.
///
/// Deliberately larger than the two closed-form cases above: they are small
/// enough that a cold solve reaches the optimum in 0-2 working-set changes, so
/// they cannot tell a re-used working set from a discarded one. The whole point
/// of the assertion below is the change *count*.
fn warm_hint_case() -> Case {
    const N: usize = 12;
    const M: usize = 8;
    let mut hi = Vec::new();
    let mut hj = Vec::new();
    let mut hv = Vec::new();
    for i in 0..N {
        hi.push((i + 1) as i32);
        hj.push((i + 1) as i32);
        hv.push(1.0 + (i as f64) / (N as f64));
    }
    let mut ai = Vec::new();
    let mut aj = Vec::new();
    let mut av = Vec::new();
    for i in 0..M {
        for k in 0..3 {
            ai.push((i + 1) as i32);
            aj.push(((i * 5 + k * 3) % N + 1) as i32);
            av.push(0.5 + 0.25 * ((i + k) % 4) as f64);
        }
    }
    Case {
        n: N,
        m: M,
        h: (hi, hj, hv),
        g: (0..N).map(|j| -2.0 - 0.1 * (j % 5) as f64).collect(),
        a: (ai, aj, av),
        bl: vec![NLP_LOWER_BOUND_INF; M],
        bu: (0..M).map(|i| 1.0 + 0.1 * (i % 3) as f64).collect(),
        xl: vec![0.0; N],
        xu: vec![NLP_UPPER_BOUND_INF; N],
    }
}

/// A parametric call the guard *declines* must still land on the cold answer —
/// and must get there on the previous working set rather than from scratch.
///
/// Both halves matter. The answer is what a caller sees, but it was already
/// correct when this branch cold-solved, so asserting it alone cannot detect a
/// regression back to that. The change count is the property that actually
/// moved, and it is exactly the kind of silent trajectory regression that
/// `CLAUDE.md` records shipping in gh #544 — right answer, more iterations,
/// nothing asserting the difference.
#[test]
fn ineligible_parametric_reuses_the_working_set() {
    let case = warm_hint_case();
    let h_space = SymTMatrixSpace::new(case.n as i32, case.h.0.clone(), case.h.1.clone());
    let mut h = SymTMatrix::new(Rc::clone(&h_space));
    h.set_values(&case.h.2);
    // A 35% Hessian perturbation: `same_h` is false, so the path is declined.
    let mut h_new = SymTMatrix::new(Rc::clone(&h_space));
    let hv_new: Vec<f64> = case.h.2.iter().map(|v| v * 1.35).collect();
    h_new.set_values(&hv_new);

    let a_space = GenTMatrixSpace::new(
        case.m as i32,
        case.n as i32,
        case.a.0.clone(),
        case.a.1.clone(),
    );
    let mut a = GenTMatrix::new(Rc::clone(&a_space));
    a.set_values(&case.a.2);

    let opts = QpOptions {
        use_homotopy: true,
        ..QpOptions::default()
    };
    macro_rules! mk {
        ($h:expr) => {
            QpProblem {
                n: case.n,
                m: case.m,
                h: $h,
                g: &case.g,
                a: &a,
                bl: &case.bl,
                bu: &case.bu,
                xl: &case.xl,
                xu: &case.xu,
                hessian_inertia: HessianInertia::Psd,
            }
        };
    }

    let mut s = new_solver();
    let prev = s.solve(&mk!(&h), None, &opts).expect("previous solve");
    assert_eq!(prev.status, QpStatus::Optimal, "previous solve");

    let warm = s
        .solve_parametric(&mk!(&h), &prev, &mk!(&h_new), &opts)
        .expect("ineligible parametric solve");
    let cold = new_solver()
        .solve(&mk!(&h_new), None, &opts)
        .expect("cold solve");

    assert_eq!(warm.status, QpStatus::Optimal, "declined parametric status");
    for i in 0..case.n {
        assert!(
            (warm.x[i] - cold.x[i]).abs() < 1e-7,
            "x[{i}]: declined parametric {} vs cold {}",
            warm.x[i],
            cold.x[i]
        );
    }
    assert!(
        (warm.obj - cold.obj).abs() < 1e-7,
        "obj: declined parametric {} vs cold {}",
        warm.obj,
        cold.obj
    );
    assert!(
        warm.stats.n_working_set_changes < cold.stats.n_working_set_changes,
        "declined parametric took {} working-set changes against cold's {} — the \
         previous working set is being discarded again (gh #602)",
        warm.stats.n_working_set_changes,
        cold.stats.n_working_set_changes
    );
}
