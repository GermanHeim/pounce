//! gh#1026 — two node LPs from the same discopt OA-master branch-and-bound
//! as gh#1023 (67 columns, `P = None`, bounds tightened by branching) still
//! failed after the gh#1023 fix: n1 hit `IterationLimit` at 200 with `kappa`
//! underflowed to `1e-183`, n2 ended `OptimalInaccurate` at 75. 0.12.0
//! returned `optimal` in about 24 iterations at objectives `4e-8` and `1e-7`
//! relative from the optimum (HiGHS agrees with the values pinned below to
//! the last printed digit).
//!
//! The rows repeat one another up to sign, and only a class's net multiplier
//! is determined; the interior point drifts along the dual ray. n1 carries
//! `z ≈ 1.2e8` on both rows of a consistent opposed pair (net `2e5`) and
//! `9e8` on both bounds of a column branched to `lb = ub = 0`. From iteration
//! 29 every residual sat at round-off, but the true-slack stop read that
//! column's `1.6e-17` drift times `9e8` as complementarity `1.4e-8` against
//! `tol = 1e-8`, for 170 iterations. The verdict already exempted a pinned
//! column; the loop, which sees it as two bound rows, did not.
//!
//! Both the loop and the verdict now read rows that repeat one another up to
//! round-off at their summed multiplier, and an equality among them --
//! opposed rows whose interval is a point or empty, a pinned column, gh#1023's
//! conflicting duplicates -- not at all: a point off the equality is a primal
//! residual, and is read as one (`qp::ParallelRows`). Every class these LPs
//! carry coincides to `1.4e-14` or better; their 61 and 63 ranges of real width are
//! read row by row, as before.

use pounce_convex::{QpOptions, QpProblem, QpStatus, Triplet, solve_qp_ipm};
use pounce_feral::FeralSolverInterface;
use pounce_linsol::SparseSymLinearSolverInterface;

fn backend() -> Box<dyn SparseSymLinearSolverInterface> {
    Box::new(FeralSolverInterface::new())
}

/// Parses a `tests/data/issue1026_node_lp_*.txt` file: `n`, the vectors `c
/// lb ub h b`, then one `G`/`A` triplet per line.
fn load(text: &str) -> QpProblem {
    let mut prob = QpProblem {
        n: 0,
        p_lower: vec![],
        c: vec![],
        a: vec![],
        b: vec![],
        g: vec![],
        h: vec![],
        lb: vec![],
        ub: vec![],
    };
    let nums = |it: std::str::SplitWhitespace| -> Vec<f64> {
        it.map(|t| t.parse().expect("float")).collect()
    };
    for line in text.lines().filter(|l| !l.starts_with('#')) {
        let mut it = line.split_whitespace();
        match it.next() {
            Some("n") => prob.n = it.next().unwrap().parse().unwrap(),
            Some("c") => prob.c = nums(it),
            Some("lb") => prob.lb = nums(it),
            Some("ub") => prob.ub = nums(it),
            Some("h") => prob.h = nums(it),
            Some("b") => prob.b = nums(it),
            Some(m @ ("G" | "A")) => {
                let row = it.next().unwrap().parse().unwrap();
                let col = it.next().unwrap().parse().unwrap();
                let val = it.next().unwrap().parse().unwrap();
                let t = Triplet::new(row, col, val);
                if m == "G" {
                    prob.g.push(t)
                } else {
                    prob.a.push(t)
                }
            }
            _ => {}
        }
    }
    prob
}

fn check(name: &str, text: &str, m_ineq: usize, obj: f64, was: &str) {
    let prob = load(text);
    assert_eq!((prob.n, prob.h.len(), prob.b.len()), (67, m_ineq, 21));
    let sol = solve_qp_ipm(&prob, &QpOptions::default(), backend);
    assert_eq!(
        sol.status,
        QpStatus::Optimal,
        "{name}: iters={} obj={} (gh#1026: main {was})",
        sol.iters,
        sol.obj
    );
    // 0.12.0 took ~24; the fix lands at 35 (n1) and 32 (n2).
    assert!(sol.iters <= 60, "{name}: iters={}", sol.iters);
    // The verdict agrees with the loop: no downgrade to OptimalInaccurate.
    let kkt = sol.kkt_residuals_above_floor(&prob).kkt_error();
    assert!(kkt <= 1e-8, "{name}: kkt_error={kkt:e}");
    assert!(
        (sol.obj - obj).abs() < 1e-9 * obj,
        "{name}: obj={} want {obj}",
        sol.obj
    );
}

#[test]
fn node_lp_with_a_pinned_column_and_split_multipliers_solves_to_optimal() {
    check(
        "n1",
        include_str!("data/issue1026_node_lp_n1.txt"),
        397,
        531_147_387.197_053_4,
        "hit IterationLimit at 200",
    );
}

#[test]
fn node_lp_ending_optimal_inaccurate_solves_to_optimal() {
    check(
        "n2",
        include_str!("data/issue1026_node_lp_n2.txt"),
        405,
        325_424_179.474_240_3,
        "ended OptimalInaccurate at 75",
    );
}
