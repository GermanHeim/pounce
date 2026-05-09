//! Timing statistics — port of
//! `Algorithm/IpTimingStatistics.{hpp,cpp}`. Aggregates wall-clock
//! durations of each named subsystem (function eval, linear solver
//! analyze/factor/solve, line search, etc.) so they can be reported
//! at the end of `optimize()`.

use pounce_common::timing::TimedTask;

#[derive(Debug, Default)]
pub struct TimingStatistics {
    pub overall_alg: TimedTask,
    pub print_problem_statistics: TimedTask,
    pub initialize_iterates: TimedTask,
    pub update_hessian: TimedTask,
    pub output_iteration: TimedTask,
    pub update_barrier_parameter: TimedTask,
    pub compute_search_direction: TimedTask,
    pub compute_acceptable_trial_point: TimedTask,
    pub accept_trial_point: TimedTask,
    pub check_convergence: TimedTask,

    pub linear_system_factorization: TimedTask,
    pub linear_system_back_solve: TimedTask,
    pub linear_system_structure_converter: TimedTask,
    pub linear_system_structure_converter_init: TimedTask,
    pub quality_function_search: TimedTask,
    pub total_callback_time: TimedTask,
    pub total_function_evaluation_time: TimedTask,
    pub eval_obj: TimedTask,
    pub eval_grad_obj: TimedTask,
    pub eval_constr: TimedTask,
    pub eval_constr_jac: TimedTask,
    pub eval_lag_hess: TimedTask,
}

impl TimingStatistics {
    pub fn new() -> Self {
        Self::default()
    }

    /// Reset all counters. Mirrors upstream `ResetTimes()`.
    pub fn reset(&mut self) {
        *self = Self::default();
    }
}
