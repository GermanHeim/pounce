//! Per-task timing accumulator.
//!
//! Mirrors `Common/IpTimedTask.hpp` (`Common/IpDebug.{hpp,cpp}` is
//! omitted — debug tracing is replaced by the journalist).

use crate::types::Number;
use crate::utils::{cpu_time, sys_time, wallclock_time};
use std::cell::Cell;

/// Equivalent to `Ipopt::TimedTask`. Use [`TimedTask::start`] /
/// [`TimedTask::end`] around a section to accumulate cpu/system/wall
/// time. [`TimedTask::end_if_started`] is the exception-safe variant.
#[derive(Debug)]
pub struct TimedTask {
    enabled: Cell<bool>,
    start_called: Cell<bool>,
    end_called: Cell<bool>,
    start_cpu: Cell<Number>,
    start_sys: Cell<Number>,
    start_wall: Cell<Number>,
    total_cpu: Cell<Number>,
    total_sys: Cell<Number>,
    total_wall: Cell<Number>,
}

impl Default for TimedTask {
    fn default() -> Self {
        Self {
            enabled: Cell::new(true),
            start_called: Cell::new(false),
            end_called: Cell::new(true),
            start_cpu: Cell::new(0.0),
            start_sys: Cell::new(0.0),
            start_wall: Cell::new(0.0),
            total_cpu: Cell::new(0.0),
            total_sys: Cell::new(0.0),
            total_wall: Cell::new(0.0),
        }
    }
}

impl TimedTask {
    pub fn new() -> Self { Self::default() }

    pub fn enable(&self) { self.enabled.set(true); }
    pub fn disable(&self) { self.enabled.set(false); }
    pub fn is_enabled(&self) -> bool { self.enabled.get() }
    pub fn is_started(&self) -> bool { self.start_called.get() }

    pub fn reset(&self) {
        self.total_cpu.set(0.0);
        self.total_sys.set(0.0);
        self.total_wall.set(0.0);
        self.start_called.set(false);
        self.end_called.set(true);
    }

    pub fn start(&self) {
        if !self.enabled.get() { return; }
        self.end_called.set(false);
        self.start_called.set(true);
        self.start_cpu.set(cpu_time());
        self.start_sys.set(sys_time());
        self.start_wall.set(wallclock_time());
    }

    pub fn end(&self) {
        if !self.enabled.get() { return; }
        self.end_called.set(true);
        self.start_called.set(false);
        self.total_cpu.set(self.total_cpu.get() + cpu_time() - self.start_cpu.get());
        self.total_sys.set(self.total_sys.get() + sys_time() - self.start_sys.get());
        self.total_wall.set(self.total_wall.get() + wallclock_time() - self.start_wall.get());
    }

    pub fn end_if_started(&self) {
        if !self.enabled.get() { return; }
        if self.start_called.get() {
            self.end();
        }
    }

    pub fn total_cpu_time(&self) -> Number { self.total_cpu.get() }
    pub fn total_sys_time(&self) -> Number { self.total_sys.get() }
    pub fn total_wallclock_time(&self) -> Number { self.total_wall.get() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn start_end_accumulates_nonneg() {
        let t = TimedTask::new();
        t.start();
        for _ in 0..1000 { std::hint::black_box(0u64); }
        t.end();
        assert!(t.total_wallclock_time() >= 0.0);
    }

    #[test]
    fn disabled_is_noop() {
        let t = TimedTask::new();
        t.disable();
        t.start();
        t.end();
        assert_eq!(t.total_wallclock_time(), 0.0);
    }

    #[test]
    fn end_if_started_handles_unstarted() {
        let t = TimedTask::new();
        t.end_if_started();
        assert_eq!(t.total_wallclock_time(), 0.0);
    }
}
