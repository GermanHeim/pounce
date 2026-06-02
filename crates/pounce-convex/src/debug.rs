//! Debugger glue for the convex interior-point method.
//!
//! [`ConvexDebugState`] adapts one iteration of [`crate::ipm::run_ipm`]'s
//! primal-dual state to the shared [`DebugState`] surface, so the CLI's
//! `SolverDebugger` (a [`DebugHook`]) can step, inspect, and break on a
//! convex LP / QP / conic solve exactly as it does on the NLP filter-IPM.
//!
//! Block names follow the QP standard form: `x` (variables), `s` (cone
//! slacks), `y` (equality multipliers), `z` (inequality / cone
//! multipliers); their search-direction counterparts are addressed by the
//! same names. The convex IPM has no backtracking line search, no
//! restoration phase, and a μ that is *derived* from `⟨s, z⟩` rather than
//! a free knob — so [`ls_count`](DebugState::ls_count) reports "n/a" and
//! the mutation / rewind methods fall back to the trait's "unsupported"
//! defaults (read-only debugging, for now).

use pounce_common::debug::{Checkpoint, DebugAction, DebugHook, DebugState};
use pounce_common::types::Number;

/// A read-only view of one convex-IPM iteration for the debugger.
///
/// Holds borrowed slices into the live iterate (`x`/`s`/`y`/`z`) and the
/// current search direction (`dx`/`dy`/`dz`/`ds`); cheap to build and
/// dropped before the loop mutates the iterate again.
pub(crate) struct ConvexDebugState<'a> {
    pub cp: Checkpoint,
    pub iter: i32,
    pub mu: f64,
    /// Max-norm primal infeasibility (max over equality / cone residuals).
    pub pinf: f64,
    /// Max-norm dual (stationarity) infeasibility.
    pub dinf: f64,
    /// `max(pinf, dinf, mu)` — the scalar convergence test.
    pub res: f64,
    pub obj: f64,
    pub alpha: (f64, f64),
    pub x: &'a [f64],
    pub s: &'a [f64],
    pub y: &'a [f64],
    pub z: &'a [f64],
    pub dx: &'a [f64],
    pub dy: &'a [f64],
    pub dz: &'a [f64],
    pub ds: &'a [f64],
    /// HSDE homogenizing variable τ (the iterate is the homogeneous
    /// `(x, s, y, z, τ, κ)`; the recovered solution is `x/τ`). `None` for
    /// the direct (non-homogeneous) driver.
    pub tau: Option<f64>,
    /// HSDE homogenizing variable κ. `None` for the direct driver.
    pub kappa: Option<f64>,
    pub status: Option<&'a str>,
}

impl DebugState for ConvexDebugState<'_> {
    fn checkpoint(&self) -> Checkpoint {
        self.cp
    }
    fn iter(&self) -> i32 {
        self.iter
    }
    fn mu(&self) -> Number {
        self.mu
    }
    fn objective(&self) -> Number {
        self.obj
    }
    fn inf_pr(&self) -> Number {
        self.pinf
    }
    fn inf_du(&self) -> Number {
        self.dinf
    }
    fn complementarity(&self) -> Number {
        // For a symmetric cone μ = ⟨s, z⟩ / degree is exactly the average
        // complementarity, so it doubles as the central-path gauge.
        self.mu
    }
    fn alpha(&self) -> (Number, Number) {
        self.alpha
    }
    fn block_dims(&self) -> Vec<(&'static str, usize)> {
        let mut v = vec![
            ("x", self.x.len()),
            ("s", self.s.len()),
            ("y", self.y.len()),
            ("z", self.z.len()),
        ];
        // The homogenizing scalars are addressable as 1-element blocks on
        // the HSDE driver (`print tau` / `print kappa`).
        if self.tau.is_some() {
            v.push(("tau", 1));
        }
        if self.kappa.is_some() {
            v.push(("kappa", 1));
        }
        v
    }
    fn block(&self, name: &str) -> Option<Vec<Number>> {
        match name {
            "x" => Some(self.x.to_vec()),
            "s" => Some(self.s.to_vec()),
            "y" => Some(self.y.to_vec()),
            "z" => Some(self.z.to_vec()),
            "tau" => self.tau.map(|t| vec![t]),
            "kappa" => self.kappa.map(|k| vec![k]),
            _ => None,
        }
    }
    fn delta_block(&self, name: &str) -> Option<Vec<Number>> {
        match name {
            "x" => Some(self.dx.to_vec()),
            "s" => Some(self.ds.to_vec()),
            "y" => Some(self.dy.to_vec()),
            "z" => Some(self.dz.to_vec()),
            _ => None,
        }
    }
    fn status(&self) -> Option<&str> {
        self.status
    }
    /// The convex IPM's scalar convergence error `max(pinf, dinf, μ)`, so
    /// `break if err<…` works the same as on the NLP path.
    fn nlp_error(&self) -> Number {
        self.res
    }
}

/// Fire a checkpoint at `state` if a hook is attached. A no-op (and
/// always [`DebugAction::Resume`]) when `hook` is `None`, so the
/// hook-free solve path pays nothing.
pub(crate) fn fire(
    hook: &mut Option<&mut dyn DebugHook>,
    state: &mut dyn DebugState,
) -> DebugAction {
    match hook.as_mut() {
        Some(h) => h.at_checkpoint(state),
        None => DebugAction::Resume,
    }
}
