//! Postsolve frame stack for the auxiliary-equality preprocessing
//! pass.
//!
//! PR 1 lands [`ReductionFrame`] / [`ReductionStack`] as empty
//! placeholders so [`crate::PresolveState`] can carry the stack
//! through the existing `ensure_init` machinery. The real
//! `var_map / row_map / fixed_values / multiplier_recovery` payload —
//! plus the dense-LU stationarity solve that recovers multipliers on
//! postsolve — lands in PR 7. ripopt anchor:
//! `src/reduction_frame.rs:101-231`.

/// One layer of the postsolve stack. PR 7 fills in the variable map,
/// row map, fixed values, and multiplier-recovery payload.
#[derive(Debug, Default, Clone)]
pub struct ReductionFrame {
    _private: (),
}

/// LIFO stack of `ReductionFrame`s; the top of the stack is the most
/// recently applied reduction. `finalize_solution` walks it from top
/// to bottom.
#[derive(Debug, Default, Clone)]
pub struct ReductionStack {
    frames: Vec<ReductionFrame>,
}

impl ReductionStack {
    /// True when no reduction has been pushed (the no-op fast path).
    pub fn is_empty(&self) -> bool {
        self.frames.is_empty()
    }

    /// Number of layers currently on the stack.
    pub fn len(&self) -> usize {
        self.frames.len()
    }
}
