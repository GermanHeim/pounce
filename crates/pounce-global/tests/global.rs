//! End-to-end spatial branch-and-bound on classic nonconvex problems.

use pounce_feral::FeralSolverInterface;
use pounce_global::{
    expr::var, solve_global, BranchRule, GlobalOptions, GlobalProblem, GlobalStatus,
};
use pounce_linsol::SparseSymLinearSolverInterface;

fn backend() -> Box<dyn SparseSymLinearSolverInterface> {
    Box::new(FeralSolverInterface::new())
}

#[test]
fn unconstrained_quartic_two_global_minima() {
    // f(x) = x⁴ − 3x² on [−2, 2]: global minimum −9/4 at x = ±√(3/2).
    let f = var(0).powi(4) - 3.0 * var(0).powi(2);
    let prob = GlobalProblem::new(vec![-2.0], vec![2.0], &f);
    let sol = solve_global(&prob, &GlobalOptions::default(), backend);
    assert_eq!(sol.status, GlobalStatus::Optimal, "{sol:?}");
    assert!(
        (sol.objective + 2.25).abs() < 1e-3,
        "obj = {}",
        sol.objective
    );
    assert!(
        (sol.x[0].abs() - 1.224_744_9).abs() < 1e-2,
        "x = {}",
        sol.x[0]
    );
    // Certified bound brackets the optimum.
    assert!(sol.lower_bound <= sol.objective + 1e-6);
}

#[test]
fn bilinear_box_min() {
    // f(x, y) = x·y on [−1, 1]²: global minimum −1 at (1, −1) or (−1, 1).
    let f = var(0) * var(1);
    let prob = GlobalProblem::new(vec![-1.0, -1.0], vec![1.0, 1.0], &f);
    let sol = solve_global(&prob, &GlobalOptions::default(), backend);
    assert_eq!(sol.status, GlobalStatus::Optimal, "{sol:?}");
    assert!(
        (sol.objective + 1.0).abs() < 1e-3,
        "obj = {}",
        sol.objective
    );
    assert!((sol.x[0] * sol.x[1] + 1.0).abs() < 1e-2, "x = {:?}", sol.x);
}

#[test]
fn nonconvex_equality_constraint() {
    // min x² + y²  s.t.  x·y = 1,  (x, y) ∈ [0.1, 10]².  Optimum 2 at (1, 1).
    let obj = var(0).powi(2) + var(1).powi(2);
    let g = var(0) * var(1);
    let prob = GlobalProblem::new(vec![0.1, 0.1], vec![10.0, 10.0], &obj).equality(&g, 1.0);
    let sol = solve_global(&prob, &GlobalOptions::default(), backend);
    assert_eq!(sol.status, GlobalStatus::Optimal, "{sol:?}");
    assert!(
        (sol.objective - 2.0).abs() < 1e-2,
        "obj = {}",
        sol.objective
    );
    assert!(
        (sol.x[0] - 1.0).abs() < 5e-2 && (sol.x[1] - 1.0).abs() < 5e-2,
        "x = {:?}",
        sol.x
    );
}

#[test]
fn nonconvex_inequality_feasible_region() {
    // min x + y  s.t.  x·y ≥ 4,  (x, y) ∈ [1, 5]².  Optimum 4 at (2, 2).
    let obj = var(0) + var(1);
    let g = var(0) * var(1);
    let prob = GlobalProblem::new(vec![1.0, 1.0], vec![5.0, 5.0], &obj).ge(&g, 4.0);
    let sol = solve_global(&prob, &GlobalOptions::default(), backend);
    assert_eq!(sol.status, GlobalStatus::Optimal, "{sol:?}");
    assert!(
        (sol.objective - 4.0).abs() < 1e-2,
        "obj = {}",
        sol.objective
    );
}

#[test]
fn infeasible_is_detected() {
    // x·y ≥ 100 is unreachable on [0, 1]² (max product 1).
    let obj = var(0) + var(1);
    let g = var(0) * var(1);
    let prob = GlobalProblem::new(vec![0.0, 0.0], vec![1.0, 1.0], &obj).ge(&g, 100.0);
    let sol = solve_global(&prob, &GlobalOptions::default(), backend);
    assert_eq!(sol.status, GlobalStatus::Infeasible, "{sol:?}");
}

#[test]
fn six_hump_camel_global_minimum() {
    // f(x,y) = (4 − 2.1x² + x⁴/3)x² + xy + (−4 + 4y²)y²
    //        = 4x² − 2.1x⁴ + x⁶/3 + xy − 4y² + 4y⁴.
    // Six local minima; two global ones at (±0.0898, ∓0.7126), value ≈ −1.0316.
    let x = var(0);
    let y = var(1);
    let f = 4.0 * x.clone().powi(2) - 2.1 * x.clone().powi(4)
        + (1.0 / 3.0) * x.clone().powi(6)
        + x.clone() * y.clone()
        - 4.0 * y.clone().powi(2)
        + 4.0 * y.powi(4);
    let prob = GlobalProblem::new(vec![-2.0, -1.5], vec![2.0, 1.5], &f);
    let opts = GlobalOptions {
        abs_gap: 1e-4,
        rel_gap: 1e-4,
        max_nodes: 200_000,
        ..GlobalOptions::default()
    };
    let sol = solve_global(&prob, &opts, backend);
    eprintln!(
        "camel: status={:?} obj={} lb={} nodes={} x={:?}",
        sol.status, sol.objective, sol.lower_bound, sol.nodes, sol.x
    );
    assert_eq!(sol.status, GlobalStatus::Optimal, "{sol:?}");
    assert!(
        (sol.objective - (-1.031_628_5)).abs() < 1e-2,
        "obj = {}",
        sol.objective
    );
    // One of the two global minimizers.
    assert!(
        sol.x[0].abs() < 0.2 && sol.x[1].abs() > 0.5,
        "x = {:?}",
        sol.x
    );
}

#[test]
fn local_nlp_upper_bounds_toggle() {
    // min x + y  s.t.  x·y ≥ 4 on [1, 5]² (optimum 4 at (2, 2)). Solve with the
    // local NLP polish on (default) and off — both must certify the global
    // optimum, exercising the tape→TNLP bridge against the relaxation-only path.
    let obj = var(0) + var(1);
    let g = var(0) * var(1);
    let prob = GlobalProblem::new(vec![1.0, 1.0], vec![5.0, 5.0], &obj).ge(&g, 4.0);

    let with_nlp = solve_global(&prob, &GlobalOptions::default(), backend);
    assert_eq!(with_nlp.status, GlobalStatus::Optimal, "{with_nlp:?}");
    assert!(
        (with_nlp.objective - 4.0).abs() < 1e-3,
        "obj = {}",
        with_nlp.objective
    );
    // The NLP polish lands essentially on the true minimizer (2, 2).
    assert!(
        (with_nlp.x[0] - 2.0).abs() < 1e-2 && (with_nlp.x[1] - 2.0).abs() < 1e-2,
        "x = {:?}",
        with_nlp.x
    );

    let no_nlp_opts = GlobalOptions {
        local_solve_iters: 0,
        ..GlobalOptions::default()
    };
    let without = solve_global(&prob, &no_nlp_opts, backend);
    assert_eq!(without.status, GlobalStatus::Optimal, "{without:?}");
    assert!(
        (without.objective - 4.0).abs() < 1e-2,
        "obj = {}",
        without.objective
    );
}

#[test]
fn odd_power_straddling_zero() {
    // f(x) = x³ − 3x on [−2, 2]: critical points x = ±1, endpoints ±2.
    // Global minimum −2 (attained at x = 1 and x = −2). The cube term straddles
    // zero, so this needs the single-inflection envelope (previously box-only).
    let f = var(0).powi(3) - 3.0 * var(0);
    let prob = GlobalProblem::new(vec![-2.0], vec![2.0], &f);
    let sol = solve_global(&prob, &GlobalOptions::default(), backend);
    assert_eq!(sol.status, GlobalStatus::Optimal, "{sol:?}");
    assert!(
        (sol.objective + 2.0).abs() < 1e-3,
        "obj = {}",
        sol.objective
    );
}

#[test]
fn sine_global_minimum() {
    // min sin(x) on [0, 6]: global minimum −1 at x = 3π/2 ≈ 4.712. The root box
    // is wider than π, so the multi-inflection sloped trig relaxation engages
    // (rather than the bare box bound) and the optimum certifies near the root.
    let f = var(0).sin();
    let prob = GlobalProblem::new(vec![0.0], vec![6.0], &f);
    let opts = GlobalOptions {
        max_nodes: 50_000,
        ..GlobalOptions::default()
    };
    let sol = solve_global(&prob, &opts, backend);
    assert_eq!(sol.status, GlobalStatus::Optimal, "{sol:?}");
    assert!(
        (sol.objective + 1.0).abs() < 1e-3,
        "obj = {}",
        sol.objective
    );
    assert!(
        (sol.x[0] - 1.5 * std::f64::consts::PI).abs() < 1e-2,
        "x = {}",
        sol.x[0]
    );
}

#[test]
fn sandwich_cuts_toggle() {
    // x⁴ − 3x² on [−2, 2] (global min −2.25). Solve with cutting-plane rounds
    // on (default) and off — both must certify the global optimum, exercising
    // the validity of the sandwich tangent cuts.
    let f = var(0).powi(4) - 3.0 * var(0).powi(2);
    let prob = GlobalProblem::new(vec![-2.0], vec![2.0], &f);

    let on = solve_global(&prob, &GlobalOptions::default(), backend);
    let off = solve_global(
        &prob,
        &GlobalOptions {
            sandwich_rounds: 0,
            ..GlobalOptions::default()
        },
        backend,
    );
    for sol in [&on, &off] {
        assert_eq!(sol.status, GlobalStatus::Optimal, "{sol:?}");
        assert!(
            (sol.objective + 2.25).abs() < 1e-3,
            "obj = {}",
            sol.objective
        );
    }
}

#[test]
fn obbt_reduces_nodes() {
    // min x + y s.t. x·y ≥ 4 on [1, 5]² (optimum 4 at (2, 2)). OBBT with the
    // incumbent cutoff tightens the box aggressively; both settings certify the
    // optimum and OBBT visits no more nodes.
    let obj = var(0) + var(1);
    let g = var(0) * var(1);
    let prob = GlobalProblem::new(vec![1.0, 1.0], vec![5.0, 5.0], &obj).ge(&g, 4.0);

    let with_obbt = solve_global(&prob, &GlobalOptions::default(), backend);
    let without = solve_global(
        &prob,
        &GlobalOptions {
            obbt_passes: 0,
            ..GlobalOptions::default()
        },
        backend,
    );
    for sol in [&with_obbt, &without] {
        assert_eq!(sol.status, GlobalStatus::Optimal, "{sol:?}");
        assert!(
            (sol.objective - 4.0).abs() < 1e-2,
            "obj = {}",
            sol.objective
        );
    }
    assert!(
        with_obbt.nodes <= without.nodes,
        "OBBT nodes {} should be ≤ {} without",
        with_obbt.nodes,
        without.nodes
    );
}

#[test]
fn obbt_max_depth_certifies_same_optimum() {
    // Gating OBBT to shallow nodes only forgoes *tightening*, never soundness:
    // deeper nodes still get FBBT + the relaxation bound. So a small
    // `obbt_max_depth` must certify the SAME optimum as the unlimited default,
    // even if it visits more nodes. Same nonconvex problem as `obbt_reduces_nodes`
    // (min x + y s.t. x·y ≥ 4 on [1, 5]², optimum 4 at (2, 2)).
    let obj = var(0) + var(1);
    let g = var(0) * var(1);
    let prob = GlobalProblem::new(vec![1.0, 1.0], vec![5.0, 5.0], &obj).ge(&g, 4.0);

    let unlimited = solve_global(&prob, &GlobalOptions::default(), backend);
    // Root + first branch level only; everything deeper relies on FBBT.
    let shallow = solve_global(
        &prob,
        &GlobalOptions {
            obbt_max_depth: 1,
            ..GlobalOptions::default()
        },
        backend,
    );
    // Depth 0 = OBBT at the root only — the most aggressive gate.
    let root_only = solve_global(
        &prob,
        &GlobalOptions {
            obbt_max_depth: 0,
            ..GlobalOptions::default()
        },
        backend,
    );
    for sol in [&unlimited, &shallow, &root_only] {
        assert_eq!(sol.status, GlobalStatus::Optimal, "{sol:?}");
        assert!(
            (sol.objective - 4.0).abs() < 1e-2,
            "obj = {}",
            sol.objective
        );
    }
    // Gating OBBT can only ever cost nodes (less tightening), never save them.
    assert!(
        shallow.nodes >= unlimited.nodes && root_only.nodes >= shallow.nodes,
        "node counts: unlimited {}, depth≤1 {}, root-only {}",
        unlimited.nodes,
        shallow.nodes,
        root_only.nodes
    );
}

#[test]
fn obbt_interval_certifies_same_optimum() {
    // Throttling OBBT to every k-th node only forgoes *tightening* on the
    // skipped nodes (they still get FBBT + the relaxation bound), so a large
    // `obbt_interval` (≈ root-only OBBT) must certify the SAME optimum as the
    // default. Same nonconvex problem as `obbt_reduces_nodes`.
    let obj = var(0) + var(1);
    let g = var(0) * var(1);
    let prob = GlobalProblem::new(vec![1.0, 1.0], vec![5.0, 5.0], &obj).ge(&g, 4.0);

    let every = solve_global(&prob, &GlobalOptions::default(), backend);
    // node_seq 0 (root) still runs OBBT; everything else relies on FBBT.
    let sparse = solve_global(
        &prob,
        &GlobalOptions {
            obbt_interval: 1000,
            ..GlobalOptions::default()
        },
        backend,
    );
    for sol in [&every, &sparse] {
        assert_eq!(sol.status, GlobalStatus::Optimal, "{sol:?}");
        assert!(
            (sol.objective - 4.0).abs() < 1e-2,
            "obj = {}",
            sol.objective
        );
    }
    // Throttling OBBT can only cost nodes (less tightening), never save them.
    assert!(
        sparse.nodes >= every.nodes,
        "interval=1000 nodes {} should be ≥ every-node {}",
        sparse.nodes,
        every.nodes
    );
}

#[test]
fn obbt_max_vars_certifies_same_optimum() {
    // Tightening only the widest-box variable each pass (`obbt_max_vars = 1`) is
    // a strict subset of the full sweep, so the optimum must be unchanged and the
    // run must still complete. Same nonconvex problem as `obbt_reduces_nodes`.
    let obj = var(0) + var(1);
    let g = var(0) * var(1);
    let prob = GlobalProblem::new(vec![1.0, 1.0], vec![5.0, 5.0], &obj).ge(&g, 4.0);

    let full = solve_global(&prob, &GlobalOptions::default(), backend);
    let budgeted = solve_global(
        &prob,
        &GlobalOptions {
            obbt_max_vars: 1,
            ..GlobalOptions::default()
        },
        backend,
    );
    for sol in [&full, &budgeted] {
        assert_eq!(sol.status, GlobalStatus::Optimal, "{sol:?}");
        assert!(
            (sol.objective - 4.0).abs() < 1e-2,
            "obj = {}",
            sol.objective
        );
    }
    // A partial sweep tightens less, so it can only cost nodes, never save them.
    assert!(
        budgeted.nodes >= full.nodes,
        "max_vars=1 nodes {} should be ≥ full-sweep {}",
        budgeted.nodes,
        full.nodes
    );
}

#[test]
fn alphabb_cuts_toggle() {
    // f(x, y) = x·y on [−1, 1]² (global min −1). The objective is nonconvex
    // (indefinite Hessian), so αBB applies a positive spectral shift. Solve with
    // αBB cuts on (default) and off — both certify the optimum, exercising the
    // interval-Hessian / spectral-shift path and the validity of its cuts.
    let f = var(0) * var(1);
    let prob = GlobalProblem::new(vec![-1.0, -1.0], vec![1.0, 1.0], &f);

    let on = solve_global(&prob, &GlobalOptions::default(), backend);
    let off = solve_global(
        &prob,
        &GlobalOptions {
            alphabb_cuts: 0,
            ..GlobalOptions::default()
        },
        backend,
    );
    for sol in [&on, &off] {
        assert_eq!(sol.status, GlobalStatus::Optimal, "{sol:?}");
        assert!(
            (sol.objective + 1.0).abs() < 1e-3,
            "obj = {}",
            sol.objective
        );
    }
}

#[test]
fn rlt_affine_constraint_toggle() {
    // min x·y  s.t.  x + y = 4 (affine),  (x, y) ∈ [0, 4]². On the segment
    // xy = x(4−x) ∈ [0, 4], so the global minimum is 0 (at a segment end).
    // The affine equality drives RLT (linear constraint × bound factors); both
    // RLT on (default) and off must certify the optimum.
    let obj = var(0) * var(1);
    let g = var(0) + var(1);
    let prob = GlobalProblem::new(vec![0.0, 0.0], vec![4.0, 4.0], &obj).equality(&g, 4.0);

    let on = solve_global(&prob, &GlobalOptions::default(), backend);
    let off = solve_global(
        &prob,
        &GlobalOptions {
            rlt: false,
            ..GlobalOptions::default()
        },
        backend,
    );
    for sol in [&on, &off] {
        assert_eq!(sol.status, GlobalStatus::Optimal, "{sol:?}");
        assert!(sol.objective.abs() < 1e-2, "obj = {}", sol.objective);
    }
}

#[test]
fn trilinear_product_toggle() {
    // min x·y·z on [−1, 1]³: global minimum −1 (odd number of −1 factors). The
    // 3-way product triggers the multi-grouping relaxation; both it and the
    // single recursive grouping must certify the optimum.
    let f = var(0) * var(1) * var(2);
    let prob = GlobalProblem::new(vec![-1.0, -1.0, -1.0], vec![1.0, 1.0, 1.0], &f);

    let on = solve_global(&prob, &GlobalOptions::default(), backend);
    let off = solve_global(
        &prob,
        &GlobalOptions {
            multilinear: false,
            ..GlobalOptions::default()
        },
        backend,
    );
    for sol in [&on, &off] {
        assert_eq!(sol.status, GlobalStatus::Optimal, "{sol:?}");
        assert!(
            (sol.objective + 1.0).abs() < 1e-2,
            "obj = {}",
            sol.objective
        );
    }
}

#[test]
fn most_violation_branching_reduces_nodes() {
    // Six-hump camel. Branching on the most-violated variable (default) should
    // certify the global optimum in no more nodes than widest-variable bisection.
    let x = var(0);
    let y = var(1);
    let f = 4.0 * x.clone().powi(2) - 2.1 * x.clone().powi(4)
        + (1.0 / 3.0) * x.clone().powi(6)
        + x.clone() * y.clone()
        - 4.0 * y.clone().powi(2)
        + 4.0 * y.powi(4);
    let prob = GlobalProblem::new(vec![-2.0, -1.5], vec![2.0, 1.5], &f);
    let base = GlobalOptions {
        abs_gap: 1e-4,
        rel_gap: 1e-4,
        max_nodes: 200_000,
        ..GlobalOptions::default()
    };

    let most = solve_global(
        &prob,
        &GlobalOptions {
            branching: BranchRule::MostViolation,
            ..base.clone()
        },
        backend,
    );
    let widest = solve_global(
        &prob,
        &GlobalOptions {
            branching: BranchRule::Widest,
            ..base.clone()
        },
        backend,
    );
    for sol in [&most, &widest] {
        assert_eq!(sol.status, GlobalStatus::Optimal, "{sol:?}");
        assert!(
            (sol.objective + 1.031_628).abs() < 1e-2,
            "obj = {}",
            sol.objective
        );
    }
    assert!(
        most.nodes <= widest.nodes,
        "most-violation {} should be ≤ widest {}",
        most.nodes,
        widest.nodes
    );
}

#[test]
fn reliability_branching_certifies_optimum() {
    // Reliability branching (pseudocosts + strong branching) must certify the
    // global optimum like any rule. min x + y s.t. x·y ≥ 4 on [1, 5]² → 4 at
    // (2, 2). Exercises the strong-branching probes and pseudocost updates.
    let obj = var(0) + var(1);
    let g = var(0) * var(1);
    let prob = GlobalProblem::new(vec![1.0, 1.0], vec![5.0, 5.0], &obj).ge(&g, 4.0);
    let sol = solve_global(
        &prob,
        &GlobalOptions {
            branching: BranchRule::Reliability,
            ..GlobalOptions::default()
        },
        backend,
    );
    assert_eq!(sol.status, GlobalStatus::Optimal, "{sol:?}");
    assert!(
        (sol.objective - 4.0).abs() < 1e-2,
        "obj = {}",
        sol.objective
    );
}

#[test]
fn parallel_obbt_matches_serial() {
    // Parallelizing OBBT's per-variable solves is deterministic: it must explore
    // exactly the same nodes and return the same optimum as the serial sweep.
    let obj = var(0) + var(1);
    let g = var(0) * var(1);
    let prob = GlobalProblem::new(vec![1.0, 1.0], vec![5.0, 5.0], &obj).ge(&g, 4.0);
    let serial = solve_global(&prob, &GlobalOptions::default(), backend);
    let parallel = solve_global(
        &prob,
        &GlobalOptions {
            parallel: true,
            ..GlobalOptions::default()
        },
        backend,
    );
    assert_eq!(serial.status, parallel.status);
    assert_eq!(
        serial.nodes, parallel.nodes,
        "parallel must explore the same nodes"
    );
    assert!((serial.objective - parallel.objective).abs() < 1e-9);
}

#[test]
fn parallel_node_pool_certifies_optimum() {
    // The parallel node pool explores nodes in a non-deterministic order, but
    // must still certify the same global optimum. Six-hump camel on 4 workers.
    let x = var(0);
    let y = var(1);
    let f = 4.0 * x.clone().powi(2) - 2.1 * x.clone().powi(4)
        + (1.0 / 3.0) * x.clone().powi(6)
        + x.clone() * y.clone()
        - 4.0 * y.clone().powi(2)
        + 4.0 * y.powi(4);
    let prob = GlobalProblem::new(vec![-2.0, -1.5], vec![2.0, 1.5], &f);
    let opts = GlobalOptions {
        abs_gap: 1e-4,
        rel_gap: 1e-4,
        max_nodes: 200_000,
        threads: 4,
        ..GlobalOptions::default()
    };
    let sol = solve_global(&prob, &opts, backend);
    assert_eq!(sol.status, GlobalStatus::Optimal, "{sol:?}");
    assert!(
        (sol.objective + 1.031_628).abs() < 1e-2,
        "obj = {}",
        sol.objective
    );
    assert!(
        sol.x[0].abs() < 0.2 && sol.x[1].abs() > 0.5,
        "x = {:?}",
        sol.x
    );
}

#[test]
fn exp_log_atoms() {
    // min eˣ − x on [−2, 2]: convex, optimum 1 at x = 0 (exercises the exp
    // envelope through the global path).
    let f = var(0).exp() - var(0);
    let prob = GlobalProblem::new(vec![-2.0], vec![2.0], &f);
    let sol = solve_global(&prob, &GlobalOptions::default(), backend);
    assert_eq!(sol.status, GlobalStatus::Optimal, "{sol:?}");
    assert!(
        (sol.objective - 1.0).abs() < 1e-3,
        "obj = {}",
        sol.objective
    );
    assert!(sol.x[0].abs() < 1e-2, "x = {}", sol.x[0]);
}

#[test]
fn mixed_equality_and_inequality_three_vars() {
    // min ‖x‖²  s.t.  x₀+x₁+x₂ = 3  and  x₀−x₁ ≥ 0.75,  on [0,3]³.
    //
    // The unconstrained-on-the-equality minimizer (1,1,1) violates the
    // inequality (x₀−x₁ = 0 < 0.75), so it binds. With both active the convex
    // problem has the unique KKT (= global) point (1.375, 0.625, 1.0), value
    // 1.375² + 0.625² + 1.0² = 3.28125. Exercises a general linear equality and
    // a variable-coupling inequality together in 3-D.
    let obj = var(0).powi(2) + var(1).powi(2) + var(2).powi(2);
    let prob = GlobalProblem::new(vec![0.0, 0.0, 0.0], vec![3.0, 3.0, 3.0], &obj)
        .equality(&(var(0) + var(1) + var(2)), 3.0)
        .ge(&(var(0) - var(1)), 0.75);
    let sol = solve_global(&prob, &GlobalOptions::default(), backend);
    assert_eq!(sol.status, GlobalStatus::Optimal, "{sol:?}");
    assert!(
        (sol.objective - 3.281_25).abs() < 1e-3,
        "obj = {}",
        sol.objective
    );
    assert!(
        (sol.x[0] - 1.375).abs() < 1e-2
            && (sol.x[1] - 0.625).abs() < 1e-2
            && (sol.x[2] - 1.0).abs() < 1e-2,
        "x = {:?}",
        sol.x
    );
    // Constraints are honored at the returned incumbent.
    assert!(prob.max_violation(&sol.x) < 1e-5, "violation {:?}", sol.x);
    assert!(sol.lower_bound <= sol.objective + 1e-6);
}

#[test]
fn ratio_term_objective() {
    // min x / y  on [1, 2]²: the ratio is increasing in x, decreasing in y, so
    // the optimum sits at the corner (1, 2) with value 0.5. End-to-end exercise
    // of the `Div` op and its bilinear (w·y = x) relaxation / Ratio branch term.
    let f = var(0) / var(1);
    let prob = GlobalProblem::new(vec![1.0, 1.0], vec![2.0, 2.0], &f);
    let sol = solve_global(&prob, &GlobalOptions::default(), backend);
    assert_eq!(sol.status, GlobalStatus::Optimal, "{sol:?}");
    assert!(
        (sol.objective - 0.5).abs() < 1e-3,
        "obj = {}",
        sol.objective
    );
    assert!(
        (sol.x[0] - 1.0).abs() < 1e-2 && (sol.x[1] - 2.0).abs() < 1e-2,
        "x = {:?}",
        sol.x
    );
    assert!(sol.lower_bound <= sol.objective + 1e-6);
}

#[test]
fn sos_and_bnb_agree_on_polynomial() {
    // The SOS/Lasserre path and spatial branch-and-bound must certify the same
    // global minimum of a shared polynomial: p(x, y) = x⁴ − 3x² + y², coercive
    // with global minimum −9/4 at (±√(3/2), 0). SOS minimizes over ℝⁿ; the
    // minimizers lie inside [−2, 2]², so sBB over that box agrees.
    use pounce_convex::{sos_minimize, PolyProblem, Polynomial, QpStatus};

    let poly = Polynomial::new(
        2,
        vec![
            (vec![4, 0], 1.0),  // x⁴
            (vec![2, 0], -3.0), // −3x²
            (vec![0, 2], 1.0),  // y²
        ],
    );
    let sos = sos_minimize(&PolyProblem::new(poly), None, backend);
    assert_eq!(sos.status, QpStatus::Optimal, "SOS status {:?}", sos.status);

    let f = var(0).powi(4) - 3.0 * var(0).powi(2) + var(1).powi(2);
    let prob = GlobalProblem::new(vec![-2.0, -2.0], vec![2.0, 2.0], &f);
    let bb = solve_global(&prob, &GlobalOptions::default(), backend);
    assert_eq!(bb.status, GlobalStatus::Optimal, "{bb:?}");

    // Both reach −2.25, and the two certificates agree to solver tolerance.
    assert!(
        (sos.lower_bound + 2.25).abs() < 1e-4,
        "SOS = {}",
        sos.lower_bound
    );
    assert!((bb.objective + 2.25).abs() < 1e-3, "B&B = {}", bb.objective);
    assert!(
        (sos.lower_bound - bb.objective).abs() < 2e-3,
        "SOS {} vs B&B {}",
        sos.lower_bound,
        bb.objective
    );
}

// --- Limit-status honesty (PR70 item C) -----------------------------------
//
// When a budget is exhausted the search must report `NodeLimit` / `TimeLimit`,
// NOT a false `Optimal`, and the returned `[lower_bound, objective]` must still
// be a *valid bracket* on the true global optimum (lower ≤ upper).

/// Node-budget honesty: a multi-node nonconvex problem solved under a tiny node
/// cap must report `NodeLimit` (never `Optimal`) and return a valid bracket.
#[test]
fn node_limit_reports_status_and_valid_bracket() {
    // Six-hump camel needs many nodes to certify; cap at 1 so the gap cannot
    // close. (The full solve is covered by `six_hump_camel_global_minimum`.)
    let x = var(0);
    let y = var(1);
    let f = 4.0 * x.clone().powi(2) - 2.1 * x.clone().powi(4)
        + (1.0 / 3.0) * x.clone().powi(6)
        + x.clone() * y.clone()
        - 4.0 * y.clone().powi(2)
        + 4.0 * y.powi(4);
    let prob = GlobalProblem::new(vec![-2.0, -1.5], vec![2.0, 1.5], &f);
    let opts = GlobalOptions {
        max_nodes: 1,
        ..GlobalOptions::default()
    };
    let sol = solve_global(&prob, &opts, backend);
    assert_eq!(
        sol.status,
        GlobalStatus::NodeLimit,
        "1-node cap must report NodeLimit, got {sol:?}"
    );
    assert_ne!(
        sol.status,
        GlobalStatus::Optimal,
        "must not claim Optimal when the node budget was exhausted"
    );
    // The bracket is still valid: certified lower bound ≤ incumbent objective.
    assert!(
        sol.lower_bound <= sol.objective + 1e-9,
        "invalid bracket: lb={} > obj={}",
        sol.lower_bound,
        sol.objective
    );
    // The gap genuinely did not close (this is the whole point of the status).
    assert!(
        sol.gap() > opts.abs_gap,
        "gap {} should still exceed abs_gap {}",
        sol.gap(),
        opts.abs_gap
    );
}

/// Time-budget honesty: with a zero wall-clock budget the search must stop at
/// the first node boundary reporting `TimeLimit` (never `Optimal`), with a
/// valid bracket. (Time is checked once per node, so a one-node problem could
/// finish first; six-hump camel does not close in a single node.)
#[test]
fn time_limit_reports_status_and_valid_bracket() {
    let x = var(0);
    let y = var(1);
    let f = 4.0 * x.clone().powi(2) - 2.1 * x.clone().powi(4)
        + (1.0 / 3.0) * x.clone().powi(6)
        + x.clone() * y.clone()
        - 4.0 * y.clone().powi(2)
        + 4.0 * y.powi(4);
    let prob = GlobalProblem::new(vec![-2.0, -1.5], vec![2.0, 1.5], &f);
    let opts = GlobalOptions {
        max_cpu_time: 0.0,
        // Keep the node cap high so the *time* limit is what stops the search.
        max_nodes: 200_000,
        ..GlobalOptions::default()
    };
    let sol = solve_global(&prob, &opts, backend);
    assert_eq!(
        sol.status,
        GlobalStatus::TimeLimit,
        "zero time budget must report TimeLimit, got {sol:?}"
    );
    assert_ne!(
        sol.status,
        GlobalStatus::Optimal,
        "must not claim Optimal when the time budget was exhausted"
    );
    assert!(
        sol.lower_bound <= sol.objective + 1e-9,
        "invalid bracket: lb={} > obj={}",
        sol.lower_bound,
        sol.objective
    );
}

// --- Global-bound soundness (PR70 item E) ---------------------------------
//
// The defining correctness property of spatial branch-and-bound is that the
// certified `lower_bound` is a valid *global* lower bound: at every stage of the
// search it must never exceed the true global optimum `f*`. If any relaxation
// (αBB / RLT / OBBT / McCormick / multilinear / trig / envelope) ever produced a
// bound *above* the truth in a box containing the optimum, that box could be
// fathomed and the optimum lost — the worst kind of silent wrong answer.
//
// The earlier per-test `lb ≤ objective` checks only bracket the *incumbent*; an
// invalid relaxation can satisfy `lb ≤ objective` while still reporting a lower
// bound above `f*`. These tests check the strong invariant `lb ≤ f*` directly by
// stopping the search early at a range of node caps on problems whose global
// optimum is known in closed form.

/// A nonconvex problem with a known closed-form global optimum.
struct GlobalCase {
    prob: GlobalProblem,
    fstar: f64,
    name: &'static str,
}

fn known_optima_cases() -> Vec<GlobalCase> {
    let x = || var(0);
    let y = || var(1);
    let camel = 4.0 * x().powi(2) - 2.1 * x().powi(4) + (1.0 / 3.0) * x().powi(6) + x() * y()
        - 4.0 * y().powi(2)
        + 4.0 * y().powi(4);
    vec![
        GlobalCase {
            prob: GlobalProblem::new(vec![-2.0], vec![2.0], &(x().powi(4) - 3.0 * x().powi(2))),
            fstar: -2.25,
            name: "quartic x^4-3x^2",
        },
        GlobalCase {
            // bilinear → McCormick envelope
            prob: GlobalProblem::new(vec![-1.0, -1.0], vec![1.0, 1.0], &(x() * y())),
            fstar: -1.0,
            name: "bilinear xy",
        },
        GlobalCase {
            // indefinite Hessian → αBB spectral shift
            prob: GlobalProblem::new(vec![-2.0, -1.5], vec![2.0, 1.5], &camel),
            fstar: -1.031_628_5,
            name: "six-hump camel",
        },
        GlobalCase {
            // nonconvex inequality x·y ≥ 4
            prob: GlobalProblem::new(vec![1.0, 1.0], vec![5.0, 5.0], &(x() + y()))
                .ge(&(x() * y()), 4.0),
            fstar: 4.0,
            name: "x+y s.t. xy>=4",
        },
        GlobalCase {
            // trilinear product → multilinear relaxation
            prob: GlobalProblem::new(
                vec![-1.0, -1.0, -1.0],
                vec![1.0, 1.0, 1.0],
                &(var(0) * var(1) * var(2)),
            ),
            fstar: -1.0,
            name: "trilinear xyz",
        },
    ]
}

/// Core soundness: the certified lower bound is a valid GLOBAL bound — it never
/// exceeds the true global optimum, at any stage of a partially-explored search.
#[test]
fn certified_lower_bound_never_exceeds_true_global() {
    for case in known_optima_cases() {
        for &cap in &[1usize, 3, 10, 50, 500] {
            let opts = GlobalOptions {
                max_nodes: cap,
                ..GlobalOptions::default()
            };
            let sol = solve_global(&case.prob, &opts, backend);
            // The invariant: lb is a valid lower bound on the true global optimum.
            assert!(
                sol.lower_bound <= case.fstar + 1e-6,
                "{}: lower bound {} EXCEEDS true global optimum {} at {}-node cap \
                 (status {:?}) — invalid relaxation would prune the optimum",
                case.name,
                sol.lower_bound,
                case.fstar,
                cap,
                sol.status,
            );
            // The bracket is always valid too: lb ≤ incumbent.
            assert!(
                sol.lower_bound <= sol.objective + 1e-6,
                "{}: invalid bracket lb={} > obj={}",
                case.name,
                sol.lower_bound,
                sol.objective,
            );
            // If it *claims* Optimal, the incumbent must really be the global opt.
            if sol.status == GlobalStatus::Optimal {
                assert!(
                    (sol.objective - case.fstar).abs() < 1e-2,
                    "{}: claimed Optimal but obj {} != f* {}",
                    case.name,
                    sol.objective,
                    case.fstar,
                );
            }
        }
    }
}

/// Time-limit soundness (the Phase-0 live-node-drop guard): stopping on the
/// wall clock — including *mid-node*, while OBBT is still sweeping — must never
/// push the certified `lower_bound` above the true global optimum. A timed-out
/// node is *live*, not pruned; if it were silently dropped (folded into the
/// "pruned" path) the frontier minimum could rise above `f*` and certify a
/// bound that was never proven. Sweep a range of tiny budgets so the deadline
/// lands at different points in different nodes.
#[test]
fn time_limit_never_corrupts_global_lower_bound() {
    for case in known_optima_cases() {
        for &limit in &[0.0_f64, 0.001, 0.005, 0.02, 0.1] {
            let opts = GlobalOptions {
                max_cpu_time: limit,
                max_nodes: 200_000, // the *time* limit must be what stops it
                ..GlobalOptions::default()
            };
            let sol = solve_global(&case.prob, &opts, backend);
            assert!(
                sol.lower_bound <= case.fstar + 1e-6,
                "{}: time-limited lower bound {} EXCEEDS true global optimum {} \
                 at {}s budget (status {:?}) — a dropped timed-out node corrupted \
                 the global bound",
                case.name,
                sol.lower_bound,
                case.fstar,
                limit,
                sol.status,
            );
            assert!(
                sol.lower_bound <= sol.objective + 1e-6,
                "{}: invalid bracket lb={} > obj={} at {}s budget",
                case.name,
                sol.lower_bound,
                sol.objective,
                limit,
            );
            assert!(
                matches!(sol.status, GlobalStatus::TimeLimit | GlobalStatus::Optimal),
                "{}: unexpected status {:?} under a {}s budget",
                case.name,
                sol.status,
                limit,
            );
        }
    }
}

/// Fine-grained (mid-node) enforcement: a problem whose *single* node is
/// expensive — many-variable OBBT runs dozens of LP solves per pass — must not
/// overrun a small wall-clock budget by the cost of a whole node. With the
/// deadline polled only at node boundaries (the old behaviour) this root node
/// would run to completion, blowing past the limit; with mid-node polling it
/// bails promptly. Asserts wall time stays a small multiple of the budget, not
/// minutes, and the reported bracket is still valid.
#[test]
fn time_limit_enforced_within_a_single_expensive_node() {
    // ~28 coupled nonconvex variables ⇒ a large bilinear relaxation and many
    // OBBT LP solves per pass, so one node is far more than 0.1 s of work.
    let n = 28usize;
    let mut f = var(0).powi(4) - 3.0 * var(0).powi(2);
    for i in 1..n {
        f = f + (var(i).powi(4) - 3.0 * var(i).powi(2)) + var(i - 1) * var(i);
    }
    let prob = GlobalProblem::new(vec![-2.0; n], vec![2.0; n], &f);
    let opts = GlobalOptions {
        max_cpu_time: 0.1,
        max_nodes: 200_000,
        ..GlobalOptions::default()
    };
    let t = std::time::Instant::now();
    let sol = solve_global(&prob, &opts, backend);
    let elapsed = t.elapsed().as_secs_f64();
    assert!(
        elapsed < 5.0,
        "mid-node time enforcement failed: a 0.1s budget ran {elapsed:.2}s \
         (status {:?}) — the deadline is only being checked at node boundaries",
        sol.status,
    );
    assert_eq!(
        sol.status,
        GlobalStatus::TimeLimit,
        "a 0.1s budget on a 28-var nonconvex problem must report TimeLimit, got {:?}",
        sol.status,
    );
    assert!(
        sol.lower_bound <= sol.objective + 1e-9,
        "invalid bracket: lb={} > obj={}",
        sol.lower_bound,
        sol.objective,
    );
}

/// Per-relaxation validity: with each cut/relaxation family toggled on in
/// isolation (others off), a partially-explored search must still produce a
/// valid global lower bound (`lb ≤ f*`). This isolates the validity of each
/// outer-approximation generator — a bug in any one of them would surface as a
/// bound above the truth on the matching nonconvex structure.
#[test]
fn each_relaxation_yields_valid_global_lower_bound() {
    let base_off = GlobalOptions {
        // Strip every optional relaxation/cut so each can be re-enabled alone.
        alphabb_cuts: 0,
        rlt: false,
        multilinear: false,
        obbt_passes: 0,
        sandwich_rounds: 0,
        max_nodes: 200, // partial search: bounds must be valid mid-flight
        ..GlobalOptions::default()
    };
    // (label, options with exactly one family enabled)
    let configs = vec![
        ("all-off (box/interval only)", base_off.clone()),
        (
            "alphabb",
            GlobalOptions {
                alphabb_cuts: GlobalOptions::default().alphabb_cuts,
                ..base_off.clone()
            },
        ),
        (
            "rlt",
            GlobalOptions {
                rlt: true,
                ..base_off.clone()
            },
        ),
        (
            "multilinear",
            GlobalOptions {
                multilinear: true,
                ..base_off.clone()
            },
        ),
        (
            "obbt",
            GlobalOptions {
                obbt_passes: GlobalOptions::default().obbt_passes,
                ..base_off.clone()
            },
        ),
        (
            "sandwich",
            GlobalOptions {
                sandwich_rounds: GlobalOptions::default().sandwich_rounds,
                ..base_off.clone()
            },
        ),
    ];
    for case in known_optima_cases() {
        for (label, opts) in &configs {
            let sol = solve_global(&case.prob, opts, backend);
            assert!(
                sol.lower_bound <= case.fstar + 1e-6,
                "[{label}] {}: lower bound {} EXCEEDS true global optimum {} \
                 (status {:?}) — relaxation family produced an invalid bound",
                case.name,
                sol.lower_bound,
                case.fstar,
                sol.status,
            );
        }
    }
}

/// Serial == parallel on a *constrained* nonconvex problem: the parallel node
/// pool explores in a non-deterministic order but must certify the same global
/// optimum (and honor the constraint) as the serial sweep. Complements
/// `parallel_obbt_matches_serial` (unconstrained, exact node-count match) and
/// `parallel_node_pool_certifies_optimum` (unconstrained camel).
#[test]
fn parallel_matches_serial_constrained() {
    let obj = var(0).powi(2) + var(1).powi(2);
    let g = var(0) * var(1);
    // min x²+y² s.t. xy=1 on [0.1,10]² → 2 at (1,1).
    let prob = GlobalProblem::new(vec![0.1, 0.1], vec![10.0, 10.0], &obj).equality(&g, 1.0);

    let serial = solve_global(&prob, &GlobalOptions::default(), backend);
    let parallel = solve_global(
        &prob,
        &GlobalOptions {
            parallel: true,
            threads: 4,
            ..GlobalOptions::default()
        },
        backend,
    );
    assert_eq!(serial.status, GlobalStatus::Optimal, "{serial:?}");
    assert_eq!(parallel.status, GlobalStatus::Optimal, "{parallel:?}");
    assert!(
        (serial.objective - parallel.objective).abs() < 1e-2,
        "serial obj {} vs parallel obj {}",
        serial.objective,
        parallel.objective,
    );
    // Both land on the true optimum and honor the equality.
    for sol in [&serial, &parallel] {
        assert!(
            (sol.objective - 2.0).abs() < 1e-2,
            "obj = {}",
            sol.objective
        );
        assert!(
            prob.max_violation(&sol.x) < 1e-4,
            "violation at {:?}",
            sol.x
        );
        assert!(sol.lower_bound <= sol.objective + 1e-6);
    }
}

/// Phase 6.5 — the warm-started simplex OBBT engine (`obbt_lp = Simplex`) must
/// certify the **same** global optimum as the interior-point default on a
/// spread of nonconvex problems, with a sound bracketing lower bound. The two
/// LP engines agree only to tolerance (different algorithms), so node counts may
/// differ, but the certified value must not — this is the 0-WRONG gate that
/// justifies wiring simplex into OBBT at all.
///
/// PARKED: the simplex engine is gated behind the off-by-default `simplex-obbt`
/// feature (it is unsound on badly-scaled relaxation LPs). This test only runs
/// when that feature is enabled, since otherwise `ObbtLp::Simplex` transparently
/// falls back to the IPM sweep and the comparison would be vacuous.
#[cfg(feature = "simplex-obbt")]
#[test]
fn simplex_obbt_matches_ipm_certified_optimum() {
    use pounce_global::ObbtLp;
    let simplex_opts = || GlobalOptions {
        obbt_lp: ObbtLp::Simplex,
        ..GlobalOptions::default()
    };

    // (builder, expected optimum) over distinct relaxation shapes: a univariate
    // quartic, a bilinear box, an equality-constrained ratio, and the six-hump
    // camel (two coupled quartics).
    let quartic = || {
        let f = var(0).powi(4) - 3.0 * var(0).powi(2);
        (GlobalProblem::new(vec![-2.0], vec![2.0], &f), -2.25_f64)
    };
    let bilinear = || {
        let f = var(0) * var(1);
        (
            GlobalProblem::new(vec![-1.0, -1.0], vec![1.0, 1.0], &f),
            -1.0_f64,
        )
    };
    let ratio = || {
        let obj = var(0).powi(2) + var(1).powi(2);
        let g = var(0) * var(1);
        (
            GlobalProblem::new(vec![0.1, 0.1], vec![10.0, 10.0], &obj).equality(&g, 1.0),
            2.0_f64,
        )
    };

    for mk in [
        &quartic as &dyn Fn() -> (GlobalProblem, f64),
        &bilinear,
        &ratio,
    ] {
        let (prob, expected) = mk();
        let ipm = solve_global(&prob, &GlobalOptions::default(), backend);
        let (prob2, _) = mk();
        let spx = solve_global(&prob2, &simplex_opts(), backend);

        assert_eq!(ipm.status, GlobalStatus::Optimal, "ipm {ipm:?}");
        assert_eq!(spx.status, GlobalStatus::Optimal, "simplex {spx:?}");
        // Both certify the known optimum...
        assert!(
            (spx.objective - expected).abs() < 1e-2,
            "simplex obj {} vs expected {expected}",
            spx.objective
        );
        // ...and agree with the IPM run to tolerance.
        assert!(
            (spx.objective - ipm.objective).abs() < 1e-3,
            "simplex {} vs ipm {}",
            spx.objective,
            ipm.objective
        );
        // The certified lower bound is sound (never above the true optimum).
        assert!(
            spx.lower_bound <= spx.objective + 1e-6,
            "lb {} > obj {}",
            spx.lower_bound,
            spx.objective
        );
    }
}
