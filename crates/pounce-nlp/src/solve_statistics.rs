//! Per-solve counters and timers.
//!
//! Mirrors `Interfaces/IpSolveStatistics.{hpp,cpp}`. Values are
//! populated by `IpoptApplication` after a successful solve. This is
//! a Phase-3 skeleton — the cumulative timer bookkeeping is wired up
//! in Phase 7 once `IpoptAlg` is producing iterations.

use pounce_common::types::{Index, Number};

#[derive(Debug, Default, Clone)]
pub struct SolveStatistics {
    pub iteration_count: Index,
    pub total_cpu_time_secs: Number,
    pub total_sys_time_secs: Number,
    pub total_wallclock_time_secs: Number,
    pub num_obj_evals: Index,
    pub num_constr_evals: Index,
    pub num_obj_grad_evals: Index,
    pub num_constr_jac_evals: Index,
    pub num_hess_evals: Index,
    pub final_objective: Number,
    pub final_scaled_objective: Number,
    pub final_dual_inf: Number,
    pub final_constr_viol: Number,
    pub final_compl: Number,
    pub final_kkt_error: Number,
}

impl SolveStatistics {
    pub fn new() -> Self {
        Self::default()
    }
}
