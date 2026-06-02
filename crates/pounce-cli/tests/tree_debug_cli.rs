//! The CLI tree debugger (`TreeDebugger`) drives `solve_global_debug`
//! through a script without a TTY, and the debugged solve still reaches the
//! global optimum.

use pounce_cli::cli::DebugMode;
use pounce_cli::debug_repl::SolverDebugger;
use pounce_cli::tree_debug::TreeDebugger;
use pounce_feral::FeralSolverInterface;
use pounce_global::{
    expr::var, solve_global_debug, solve_global_debug_into, GlobalOptions, GlobalProblem,
    GlobalStatus,
};
use pounce_linsol::SparseSymLinearSolverInterface;

fn backend() -> Box<dyn SparseSymLinearSolverInterface> {
    Box::new(FeralSolverInterface::new())
}

/// Write a script file the `TreeDebugger` can replay non-interactively.
fn script(name: &str, body: &str) -> std::path::PathBuf {
    let p = std::env::temp_dir().join(format!("pounce-treedbg-{name}.txt"));
    std::fs::write(&p, body).unwrap();
    p
}

#[test]
fn scripted_tree_debugger_runs_to_the_global_optimum() {
    // f(x) = x⁴ − 3x² on [−2, 2]: global min −9/4. Nonconvex ⇒ it branches.
    let f = var(0).powi(4) - 3.0 * var(0).powi(2);
    let prob = GlobalProblem::new(vec![-2.0], vec![2.0], &f);

    // Inspect a couple of things, set a breakpoint, then run to the end.
    let path = script(
        "optimum",
        "node\nbounds\nbreak incumbent\ncontinue\ngap\ncontinue\n",
    );
    let mut dbg = TreeDebugger::new(DebugMode::Repl).with_script(&path);

    let sol = solve_global_debug(&prob, &GlobalOptions::default(), &mut dbg, backend);
    assert_eq!(sol.status, GlobalStatus::Optimal, "{sol:?}");
    assert!((sol.objective + 2.25).abs() < 1e-3, "obj={}", sol.objective);
}

#[test]
fn scripted_tree_debugger_quit_halts_early() {
    let f = var(0).powi(4) - 3.0 * var(0).powi(2);
    let prob = GlobalProblem::new(vec![-2.0], vec![2.0], &f);

    // `quit` at the very first pause stops the search.
    let path = script("quit", "node\nquit\n");
    let mut dbg = TreeDebugger::new(DebugMode::Repl).with_script(&path);

    let sol = solve_global_debug(&prob, &GlobalOptions::default(), &mut dbg, backend);
    // Stopped at the first node — far short of a full solve.
    assert!(
        sol.nodes <= 1,
        "quit should stop early, got {} nodes",
        sol.nodes
    );
}

/// Scripted step-into: a single `--debug-script`-style queue drives both the
/// tree REPL and the interior-point sub-solve. `into` (tree) is followed by
/// `print mu` / `continue` (interior-point), which the shared queue routes to
/// the sub-solve debugger. The queue is fully consumed by the two REPLs.
#[test]
fn scripted_step_into_drives_the_relaxation_subsolve() {
    let f = var(0).powi(4) - 3.0 * var(0).powi(2);
    let prob = GlobalProblem::new(vec![-2.0], vec![2.0], &f);

    let path = script("into", "into\nprint mu\ncontinue\n");
    let dbg = TreeDebugger::new(DebugMode::Repl).with_script(&path);
    // The interior-point sub-solve shares the tree debugger's command queue.
    let queue = dbg.shared_script().expect("scripted");
    let mut subsolve =
        SolverDebugger::quiet(DebugMode::Repl, None).with_shared_script(queue.clone());
    let mut dbg = dbg;

    let sol = solve_global_debug_into(
        &prob,
        &GlobalOptions::default(),
        &mut dbg,
        &mut subsolve,
        backend,
    );

    assert_eq!(sol.status, GlobalStatus::Optimal, "{sol:?}");
    assert!((sol.objective + 2.25).abs() < 1e-3, "obj={}", sol.objective);
    // Both REPLs drew from the one queue: `into` (tree) + `print mu` /
    // `continue` (interior-point) — all consumed.
    assert!(
        queue.borrow().is_empty(),
        "shared script not fully consumed: {:?}",
        queue.borrow()
    );
}
