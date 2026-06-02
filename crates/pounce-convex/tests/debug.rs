//! The convex IPM honors an attached `DebugHook`: it fires the shared
//! checkpoints, exposes the iterate through the `DebugState` surface, and
//! the attached hook does not change the solve result.

use pounce_common::debug::{Checkpoint, DebugAction, DebugHook, DebugState};
use pounce_convex::{solve_qp_ipm, solve_qp_ipm_debug, QpOptions, QpProblem, QpStatus, Triplet};
use pounce_feral::FeralSolverInterface;
use pounce_linsol::SparseSymLinearSolverInterface;

fn backend() -> Box<dyn SparseSymLinearSolverInterface> {
    Box::new(FeralSolverInterface::new())
}

/// min ½(x0² + x1²) s.t. x0 + x1 ≥ 2  (i.e. −x0 − x1 ≤ −2). Optimum (1, 1),
/// f* = 1, the inequality active with z ≈ 1 — a nonempty cone, so the IPM
/// takes several predictor-corrector iterations.
fn active_ineq_qp() -> QpProblem {
    QpProblem {
        n: 2,
        p_lower: vec![Triplet::new(0, 0, 1.0), Triplet::new(1, 1, 1.0)],
        c: vec![0.0, 0.0],
        a: vec![],
        b: vec![],
        g: vec![Triplet::new(0, 0, -1.0), Triplet::new(0, 1, -1.0)],
        h: vec![-2.0],
        lb: vec![],
        ub: vec![],
    }
}

/// Records what the debugger sees at each checkpoint, and resumes.
#[derive(Default)]
struct Recorder {
    checkpoints: Vec<Checkpoint>,
    max_mu: f64,
    saw_nonempty_z: bool,
    saw_tau: bool,
    x_dim_at_iter_start: Option<usize>,
    terminal_status: Option<String>,
}

impl DebugHook for Recorder {
    fn at_checkpoint(&mut self, st: &mut dyn DebugState) -> DebugAction {
        self.checkpoints.push(st.checkpoint());
        self.max_mu = self.max_mu.max(st.mu());
        if let Some(z) = st.block("z") {
            if !z.is_empty() {
                self.saw_nonempty_z = true;
            }
        }
        if st.block("tau").is_some() {
            self.saw_tau = true;
        }
        if st.checkpoint() == Checkpoint::IterStart {
            self.x_dim_at_iter_start = st.block("x").map(|v| v.len());
        }
        if st.checkpoint() == Checkpoint::Terminated {
            self.terminal_status = st.status().map(str::to_owned);
        }
        DebugAction::Resume
    }
}

#[test]
fn convex_ipm_fires_checkpoints_and_exposes_state() {
    let prob = active_ineq_qp();
    let opts = QpOptions::default();
    let mut rec = Recorder::default();
    let sol = solve_qp_ipm_debug(&prob, &opts, &mut rec, backend);

    // The solve still reaches the known optimum.
    assert_eq!(sol.status, QpStatus::Optimal, "iters={}", sol.iters);
    assert!((sol.x[0] - 1.0).abs() < 1e-6, "x0={}", sol.x[0]);
    assert!((sol.x[1] - 1.0).abs() < 1e-6, "x1={}", sol.x[1]);

    // Every checkpoint kind fired at least once.
    let fired = |c| rec.checkpoints.contains(&c);
    assert!(fired(Checkpoint::IterStart), "no IterStart");
    assert!(
        fired(Checkpoint::AfterSearchDirection),
        "no AfterSearchDirection"
    );
    assert!(fired(Checkpoint::AfterStep), "no AfterStep");
    assert!(fired(Checkpoint::Terminated), "no Terminated");

    // State surfaced correctly: nonempty cone, μ moved, x has the right
    // dimension, and the terminal checkpoint carried the status.
    assert!(
        rec.saw_nonempty_z,
        "z block should be nonempty (one cone row)"
    );
    assert!(rec.max_mu > 0.0, "mu should be positive on a coned solve");
    assert_eq!(rec.x_dim_at_iter_start, Some(2), "x dim");
    assert_eq!(rec.terminal_status.as_deref(), Some("Optimal"));
}

#[test]
fn attaching_a_hook_does_not_change_the_result() {
    let prob = active_ineq_qp();
    let opts = QpOptions::default();

    let plain = solve_qp_ipm(&prob, &opts, backend);
    let mut rec = Recorder::default();
    let debugged = solve_qp_ipm_debug(&prob, &opts, &mut rec, backend);

    assert_eq!(plain.status, debugged.status);
    assert_eq!(plain.iters, debugged.iters, "iteration count must match");
    for (a, b) in plain.x.iter().zip(&debugged.x) {
        assert!((a - b).abs() < 1e-12, "x differs: {a} vs {b}");
    }
    assert!((plain.obj - debugged.obj).abs() < 1e-12, "obj differs");
}

/// The HSDE driver (`use_hsde`) is debuggable through the same entry: it
/// fires the checkpoints, exposes the homogenizing τ/κ as blocks, and the
/// hook does not change the recovered solution.
#[test]
fn hsde_driver_is_debuggable_and_exposes_tau_kappa() {
    let prob = active_ineq_qp();
    let opts = QpOptions {
        use_hsde: true,
        ..QpOptions::default()
    };

    let mut rec = Recorder::default();
    let sol = solve_qp_ipm_debug(&prob, &opts, &mut rec, backend);

    assert_eq!(sol.status, QpStatus::Optimal, "iters={}", sol.iters);
    assert!((sol.x[0] - 1.0).abs() < 1e-5, "x0={}", sol.x[0]);
    assert!((sol.x[1] - 1.0).abs() < 1e-5, "x1={}", sol.x[1]);

    assert!(
        rec.checkpoints.contains(&Checkpoint::IterStart),
        "IterStart"
    );
    assert!(
        rec.checkpoints.contains(&Checkpoint::AfterStep),
        "AfterStep"
    );
    assert!(
        rec.checkpoints.contains(&Checkpoint::Terminated),
        "Terminated"
    );
    assert!(rec.saw_tau, "HSDE must expose the `tau` block");
    assert_eq!(rec.terminal_status.as_deref(), Some("Optimal"));

    // The attached hook leaves the HSDE result untouched.
    let plain = {
        let o = QpOptions {
            use_hsde: true,
            ..QpOptions::default()
        };
        solve_qp_ipm(&prob, &o, backend)
    };
    assert_eq!(plain.status, sol.status);
    for (a, b) in plain.x.iter().zip(&sol.x) {
        assert!((a - b).abs() < 1e-10, "x differs: {a} vs {b}");
    }
}

/// A hook that requests `Stop` at the first checkpoint halts the solve
/// short of convergence (the debugger `quit` path).
#[test]
fn stop_action_halts_the_solve() {
    struct StopNow;
    impl DebugHook for StopNow {
        fn at_checkpoint(&mut self, _st: &mut dyn DebugState) -> DebugAction {
            DebugAction::Stop
        }
    }
    let prob = active_ineq_qp();
    let opts = QpOptions::default();
    let mut hook = StopNow;
    let sol = solve_qp_ipm_debug(&prob, &opts, &mut hook, backend);
    // Stopped at iteration 0 before convergence — not Optimal.
    assert_ne!(sol.status, QpStatus::Optimal);
}
