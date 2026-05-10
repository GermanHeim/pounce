//! Optimal-error convergence check — port of
//! `Algorithm/IpOptErrorConvCheck.{hpp,cpp}`.
//!
//! Tolerance state machine over `(nlp_err, iter_count)`. The caller
//! (the main loop) pulls `nlp_err` from
//! `IpoptCalculatedQuantities::curr_nlp_error()` and `iter_count`
//! from `IpoptData::iter_count` before each check. CPU/wall time
//! gates land alongside `TimingStatistics` wiring; until then the
//! relevant fields are stored but not consulted.

use crate::conv_check::r#trait::{ConvCheck, ConvergenceStatus};
use pounce_common::types::{Index, Number};

pub struct OptErrorConvCheck {
    pub tol: Number,
    pub acceptable_tol: Number,
    pub acceptable_iter: Index,
    pub max_iter: Index,
    pub max_cpu_time: Number,
    pub max_wall_time: Number,
    pub acceptable_count: Index,
}

impl Default for OptErrorConvCheck {
    fn default() -> Self {
        // Defaults from `IpOptErrorConvCheck.cpp:RegisterOptions`.
        Self {
            tol: 1e-8,
            acceptable_tol: 1e-6,
            acceptable_iter: 15,
            max_iter: 3000,
            max_cpu_time: 1e6,
            max_wall_time: 1e6,
            acceptable_count: 0,
        }
    }
}

impl OptErrorConvCheck {
    pub fn new() -> Self {
        Self::default()
    }
}

impl ConvCheck for OptErrorConvCheck {
    fn check_convergence(&mut self, nlp_err: Number, iter_count: Index) -> ConvergenceStatus {
        if nlp_err <= self.tol {
            return ConvergenceStatus::Converged;
        }
        if nlp_err <= self.acceptable_tol {
            self.acceptable_count += 1;
            if self.acceptable_count >= self.acceptable_iter {
                return ConvergenceStatus::Converged;
            }
        } else {
            self.acceptable_count = 0;
        }
        if iter_count >= self.max_iter {
            return ConvergenceStatus::MaxIterExceeded;
        }
        ConvergenceStatus::Continue
    }

    fn tol_or_default(&self) -> Number {
        self.tol
    }

    fn current_is_acceptable(&self, nlp_err: Number) -> bool {
        // Mirrors upstream
        // `OptimalityErrorConvergenceCheck::CurrentIsAcceptable`
        // (`IpOptErrorConvCheck.cpp`): the iterate is "acceptable" iff
        // the NLP optimality error is at or below the looser
        // acceptable tolerance. Upstream cross-checks the unscaled
        // primal/dual residuals against `acceptable_constr_viol_tol`
        // and friends; we approximate with the scalar nlp_err since
        // pounce's `curr_nlp_error` already aggregates those.
        nlp_err.is_finite() && nlp_err <= self.acceptable_tol
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn converges_at_tol() {
        let mut c = OptErrorConvCheck::new();
        assert_eq!(c.check_convergence(1e-9, 0), ConvergenceStatus::Converged);
    }

    #[test]
    fn acceptable_iter_count_threshold() {
        let mut c = OptErrorConvCheck {
            acceptable_iter: 3,
            ..Default::default()
        };
        // nlp_err between tol (1e-8) and acceptable (1e-6).
        assert_eq!(c.check_convergence(1e-7, 0), ConvergenceStatus::Continue);
        assert_eq!(c.check_convergence(1e-7, 1), ConvergenceStatus::Continue);
        assert_eq!(c.check_convergence(1e-7, 2), ConvergenceStatus::Converged);
    }

    #[test]
    fn streak_resets_when_above_acceptable() {
        let mut c = OptErrorConvCheck {
            acceptable_iter: 3,
            ..Default::default()
        };
        assert_eq!(c.check_convergence(1e-7, 0), ConvergenceStatus::Continue);
        // Above acceptable resets the counter.
        assert_eq!(c.check_convergence(1e-3, 1), ConvergenceStatus::Continue);
        assert_eq!(c.check_convergence(1e-7, 2), ConvergenceStatus::Continue);
        assert_eq!(c.check_convergence(1e-7, 3), ConvergenceStatus::Continue);
        assert_eq!(c.check_convergence(1e-7, 4), ConvergenceStatus::Converged);
    }

    #[test]
    fn max_iter_exceeded() {
        let mut c = OptErrorConvCheck {
            max_iter: 5,
            ..Default::default()
        };
        assert_eq!(
            c.check_convergence(1.0, 5),
            ConvergenceStatus::MaxIterExceeded
        );
    }
}
