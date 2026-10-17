//! gh#1023 — a small LP from discopt's outer-approximation master hit the
//! iteration limit at every node.
//!
//! The LP (67 columns, 405 inequality rows, 21 equality rows, `P = None`)
//! repeats its cuts: one cut appears several times, and an equality is
//! written as two opposed rows, with right-hand sides that differ by
//! round-off (`±4.9e-15`, `3.3e-15`, `0`, ...). In exact arithmetic no point
//! satisfies every row of such a class. The best point leaves some rows a
//! slack of up to the conflict width (`9e-15` here), while the paired
//! multipliers grow toward `7e7` without cancelling. The true-slack stop
//! from gh#1000 (I.D) read `slack·z ≈ 3.5e-7` as a complementarity error
//! above `tol = 1e-8` that could never shrink, and the loop ran to
//! `max_iter`. 0.12.0 solved the LP in 23 iterations; main hit
//! `IterationLimit` at 200.
//!
//! Both the loop and the verdict now excuse, per row, the slack that the
//! data's own conflict forces (`qp::parallel_row_conflict`). Rows with no
//! conflicting duplicate are read as before.

use pounce_convex::{QpOptions, QpProblem, QpStatus, Triplet, solve_qp_ipm};
use pounce_feral::FeralSolverInterface;
use pounce_linsol::SparseSymLinearSolverInterface;

fn backend() -> Box<dyn SparseSymLinearSolverInterface> {
    Box::new(FeralSolverInterface::new())
}

/// Parses `tests/data/issue1023_oa_master_lp.txt`: `n`, the vectors `c lb
/// ub h b`, then one `G`/`A` triplet per line.
fn load() -> QpProblem {
    let text = include_str!("data/issue1023_oa_master_lp.txt");
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
    assert_eq!((prob.n, prob.h.len(), prob.b.len()), (67, 405, 21));
    prob
}

#[test]
fn duplicated_cuts_with_round_off_conflict_solve_to_optimal() {
    let prob = load();
    let sol = solve_qp_ipm(&prob, &QpOptions::default(), backend);
    assert_eq!(
        sol.status,
        QpStatus::Optimal,
        "iters={} obj={} (gh#1023: main hit IterationLimit at 200)",
        sol.iters,
        sol.obj
    );
    // 0.12.0 took 23; the fix lands at 33. Anything near the cap is the bug.
    assert!(sol.iters <= 60, "iters={}", sol.iters);
    // The verdict agrees with the loop: no downgrade to OptimalInaccurate.
    let kkt = sol.kkt_residuals_above_floor(&prob).kkt_error();
    assert!(kkt <= 1e-8, "kkt_error={kkt:e}");
    assert!(
        (sol.obj - 255_369_249.184_858).abs() < 1e-6 * 255_369_249.0,
        "obj={}",
        sol.obj
    );
}

/// The same LP through `method="active-set"`. Main hit `IterationLimit` after
/// 12004 iterations and ~270 s; four scale defects stood between the engine
/// and the optimum, each hiding the next:
///
/// 1. the simplex phase-1 seed stalled on a `3.2e-9` round-off residual of the
///    basis solve (entries of `1` beside `1.4e7`), which no pricing move can
///    remove — refinement now runs before the stall is read as infeasibility;
/// 2. the seed's feasibility check, and
/// 3. `pounce-qp`'s warm-start pre-check, were absolute `1e-9` against rows
///    whose right-hand sides reach `6e8`, an ulp of `1.2e-7`;
/// 4. at a vertex, a `1.9e-8` drift-correction step left every `a·p` below the
///    absolute rate threshold, and the empty candidate list certified a
///    recession ray on a bounded LP;
///
/// and the verdict then read the ulp-level primal residual as inaccuracy.
#[test]
fn duplicated_cuts_solve_to_optimal_on_the_active_set_engine() {
    let prob = load();
    let sol = pounce_convex::solve_qp_active_set(
        &prob,
        &QpOptions::default(),
        &Default::default(),
        &mut backend,
    );
    assert_eq!(
        sol.status,
        QpStatus::Optimal,
        "iters={} obj={} (gh#1023: active-set hit IterationLimit at 12004)",
        sol.iters,
        sol.obj
    );
    // The seeded phase-2 takes 36; the cold fallbacks spend thousands.
    assert!(sol.iters <= 200, "iters={}", sol.iters);
    assert!(
        (sol.obj - 255_369_249.184_858).abs() < 1e-6 * 255_369_249.0,
        "obj={}",
        sol.obj
    );
}
