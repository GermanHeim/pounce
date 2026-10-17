//! gh#1029 — four more node LPs from the discopt fac2 OA-master
//! branch-and-bound of gh#1023 / gh#1026 (67 columns, `P = None`, 21
//! equality rows, bounds tightened by branching). On `2903d5f` the first
//! HSDE solve of every one ran to `IterationLimit` at 200 with `kappa`
//! underflowed (to `1e-169` on m1); the equilibrated retry then either failed
//! too or returned a point the verdict read at `kkt ≈ 1e-5`, so the solves
//! ended `IterationLimit` or `OptimalInaccurate`. The objectives pinned below
//! are HiGHS's to the last printed digit.
//!
//! Every residual was at round-off long before the cap, and the only gate
//! still shut was the true-slack complementarity, at `1.2e-8`–`4.8e-8`
//! against `tol = 1e-8`. The columns reach `2e9` against `c = e₆₇`, so the
//! multipliers are `1e6`–`1.5e8`, and the products being read were those
//! multipliers times slacks of `3e-16` (a bound row whose other terms are all
//! zero) or `1.2e-14` (a row whose `h` is itself `-1.4e-14` of caller
//! round-off) — products a tenth of one ulp of the objective. The loop and
//! the verdict now both take one ulp of the objective's terms off every
//! product (`hsde::true_slack_complementarity`).
//!
//! 0.12.0 reported `optimal` in 24–26 iterations, but at absolute KKT
//! errors of 24 to 888 and objectives 7 to 700 off. m1 needs 85: from
//! iteration ~30 it holds bound rows violated by `3e-9` at multipliers near
//! `8e7`, a product of `0.27` that is a real complementarity error, and the
//! remaining iterations drive those rows to `3e-15`.

use pounce_convex::{QpOptions, QpProblem, QpStatus, Triplet, solve_qp_ipm};
use pounce_feral::FeralSolverInterface;
use pounce_linsol::SparseSymLinearSolverInterface;

fn backend() -> Box<dyn SparseSymLinearSolverInterface> {
    Box::new(FeralSolverInterface::new())
}

/// Parses a `tests/data/issue1029_node_lp_*.txt` file: `n`, the vectors `c
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

fn check(name: &str, text: &str, m_ineq: usize, obj: f64, max_iters: usize) {
    let prob = load(text);
    assert_eq!((prob.n, prob.h.len(), prob.b.len()), (67, m_ineq, 21));
    let sol = solve_qp_ipm(&prob, &QpOptions::default(), backend);
    assert_eq!(
        sol.status,
        QpStatus::Optimal,
        "{name}: iters={} obj={}",
        sol.iters,
        sol.obj
    );
    // The fix lands at 85 (m1), 36, 38 and 32; `2903d5f` hit 200.
    assert!(sol.iters <= max_iters, "{name}: iters={}", sol.iters);
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
fn m1_bound_rows_at_round_off_with_large_multipliers_solve_to_optimal() {
    check(
        "m1",
        include_str!("data/issue1029_node_lp_m1.txt"),
        405,
        1_951_951_066.708_164_5,
        100,
    );
}

#[test]
fn m2_solves_to_optimal() {
    check(
        "m2",
        include_str!("data/issue1029_node_lp_m2.txt"),
        405,
        407_561_546.032_298_1,
        60,
    );
}

#[test]
fn m3_solves_to_optimal() {
    check(
        "m3",
        include_str!("data/issue1029_node_lp_m3.txt"),
        405,
        407_561_545.402_979_1,
        60,
    );
}

#[test]
fn m4_solves_to_optimal() {
    check(
        "m4",
        include_str!("data/issue1029_node_lp_m4.txt"),
        397,
        405_564_183.487_622_1,
        60,
    );
}
