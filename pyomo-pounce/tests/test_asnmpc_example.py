"""End-to-end checks for the active-set-aware NMPC tutorial."""

import math
from types import SimpleNamespace

import numpy as np
import pytest
from pyomo.common.errors import ApplicationError

# `pounce.examples.asnmpc_cstr` imports `pyomo_cvp`, which is an optional
# extra (`pyomo-cvp==0.7.2` in python/pyproject.toml, installed by CI). Absent
# it, this module used to raise ModuleNotFoundError during *collection*, which
# fails the whole suite rather than reporting one unavailable environment.
pytest.importorskip(
    "pyomo_cvp",
    reason="pounce.examples.asnmpc_cstr needs the optional pyomo-cvp extra "
    "(pip install pyomo-cvp==0.7.2)",
)

import pounce.examples.asnmpc_cstr as asnmpc
from pounce.examples.asnmpc_cstr import (
    BASE_STATE,
    POLICIES,
    CstrConfig,
    directional_degeneracy_experiment,
    make_campaigns,
    propagate_plant,
    run_closed_loop,
    solve_controller,
    trajectory,
)


@pytest.fixture(scope="module")
def smoke_config():
    """Small but structurally identical controller for CI."""
    return CstrConfig(horizon_intervals=8, collocation_points=2, deadline_s=1.0)


@pytest.fixture(scope="module")
def event_config():
    """Small discretization with several deterministic active-set events."""
    return CstrConfig(horizon_intervals=20, collocation_points=2, deadline_s=2.0)


def test_controller_solution_exposes_state_and_held_control_grids(smoke_config):
    solved = solve_controller(BASE_STATE, smoke_config)
    predicted = trajectory(solved.model)

    assert solved.latency_s > 0.0
    assert len(predicted.time_min) > smoke_config.horizon_intervals
    assert predicted.time_min.shape == predicted.coolant.shape
    assert np.isfinite(predicted.concentration).all()
    assert np.isfinite(predicted.temperature).all()
    assert predicted.first_control == predicted.control_at(0.0)


@pytest.mark.parametrize("policy", POLICIES)
def test_every_policy_completes_the_same_nominal_closed_loop(policy, smoke_config):
    nominal = make_campaigns(steps=6, seed=19)[0]
    result = run_closed_loop(policy, nominal, smoke_config)

    assert len(result.samples) == nominal.steps
    assert result.policy == policy
    assert result.campaign == nominal.name
    assert result.metrics.solver_failures == 0
    assert result.metrics.solver_recoveries == 0
    assert result.metrics.maximum_temperature_violation == 0.0
    assert math.isfinite(result.metrics.iae)
    assert math.isfinite(result.metrics.economic_stage_cost)


def test_guard_accepts_nominal_updates_and_falls_back_under_stress(smoke_config):
    nominal, _, stress = make_campaigns(steps=6, seed=19)

    accepted = run_closed_loop("guarded_path", nominal, smoke_config)
    guarded = run_closed_loop("guarded_path", stress, smoke_config)

    assert accepted.metrics.fallback_full_solves == 0
    assert accepted.metrics.fallback_fraction == 0.0
    assert 0 < guarded.metrics.fallback_full_solves <= stress.steps
    assert guarded.metrics.fallback_fraction == pytest.approx(
        guarded.metrics.fallback_full_solves / stress.steps
    )
    rejected = [sample for sample in guarded.samples if not sample.diagnostics.accepted]
    assert len(rejected) == guarded.metrics.fallback_full_solves
    assert all(sample.diagnostics.reasons for sample in rejected)
    assert all(
        sample.diagnostics.full_solve_latency_s is not None for sample in rejected
    )


def test_switching_campaign_records_active_set_events(event_config):
    switching = make_campaigns(steps=12, seed=19)[1]

    result = run_closed_loop("path", switching, event_config)
    event_rows = result.event_timeline()

    assert result.metrics.active_set_changes > 0, (
        "the deterministic switching campaign no longer crosses an active-set bound"
    )
    assert sum(row["path_length"] for row in event_rows) == (
        result.metrics.active_set_changes
    )
    assert any(row["active_set_changes"] for row in event_rows), (
        "the event timeline no longer identifies the switching sample"
    )


def test_directional_degeneracy_experiment_reaches_the_control_kink(event_config):
    result = directional_degeneracy_experiment(event_config)

    assert result.breakpoint_variable.startswith("v1["), (
        "the deterministic probe no longer reaches a coolant-control kink"
    )
    assert result.breakpoint_bound == "lower", (
        "the deterministic coolant kink is no longer on its lower bound"
    )
    assert 0.0 < result.breakpoint_fraction < 0.1, (
        "the configured perturbation no longer reaches its first kink early"
    )
    assert result.steps[0].directional_events, (
        "the forward directional probe no longer records a path event"
    )
    assert all(step.guard_reasons for step in result.steps), (
        "the guard no longer rejects both ambiguous directional probes"
    )
    assert any(
        not np.allclose(
            step.one_sided_probe_control,
            step.directional_probe_control,
            rtol=1.0e-5,
            atol=1.0e-7,
        )
        for step in result.steps
    )


def test_plant_mismatch_is_independent_of_the_controller_model(smoke_config):
    nominal = propagate_plant(BASE_STATE, (0.4, 0.5), smoke_config)
    mismatched = propagate_plant(
        BASE_STATE,
        (0.4, 0.5),
        smoke_config,
        feed_temperature_shift=0.02,
        reaction_rate_scale=1.08,
        heat_transfer_scale=0.92,
    )

    assert not np.allclose(nominal, mismatched, rtol=0.0, atol=1.0e-6)


@pytest.mark.parametrize("failure_type", (RuntimeError, ApplicationError))
def test_failed_warm_start_is_retried_cold_and_counted(
    monkeypatch, smoke_config, failure_type
):
    warm_start = object()
    recovered_model = object()
    attempted_warm_starts = []

    def fake_solve(initial_state, config=None, warm_start=None):
        attempted_warm_starts.append(warm_start)
        if warm_start is not None:
            raise failure_type("injected warm-start failure")
        return asnmpc.SolveRecord(model=recovered_model, latency_s=0.01)

    monkeypatch.setattr(asnmpc, "solve_controller", fake_solve)
    solved = asnmpc._solve_with_recovery(
        BASE_STATE, smoke_config, warm_start=warm_start
    )

    assert attempted_warm_starts == [warm_start, None]
    assert solved.model is recovered_model
    assert solved.solver_failures == 1
    assert solved.solver_recoveries == 1
    assert solved.latency_s > 0.0


def test_guard_uses_corrected_feasibility_and_an_absolute_progress_floor(
    smoke_config,
):
    """The guard must judge the applied corrector point, not its predictor."""
    corrected = SimpleNamespace(temperature=np.asarray([0.5]))
    report = SimpleNamespace(
        violation=1.0,
        corrector={
            "feasibility": 0.0,
            "stationarity": 0.0,
            "complementarity": 0.0,
            "initial_residual": 1.0e-12,
            "residual": 9.0e-13,
        },
        activity={},
    )

    reasons = asnmpc._guard_reasons(
        report,
        corrected,
        BASE_STATE,
        BASE_STATE,
        (),
        smoke_config,
    )

    assert "primal_feasibility" not in reasons
    assert "corrector_no_progress" not in reasons


def test_update_latency_excludes_the_replayed_event_ledger(monkeypatch, smoke_config):
    """The path ledger is collected after the timed report/estimate pair."""
    calls = []
    ticks = iter((10.0, 12.5))
    background = SimpleNamespace(zc0=object(), zt0=object())
    report = object()
    values = object()
    corrected = object()

    def fake_clock():
        calls.append("clock")
        return next(ticks)

    def fake_report(*args, **kwargs):
        calls.append("report")
        return report

    def fake_estimate(*args, **kwargs):
        calls.append("estimate")
        return values

    def fake_events(*args, **kwargs):
        calls.append("events")
        return ()

    def fake_trajectory(*args, **kwargs):
        calls.append("trajectory")
        return corrected

    monkeypatch.setattr(asnmpc.time, "perf_counter", fake_clock)
    monkeypatch.setattr(asnmpc, "sens_solution_report", fake_report)
    monkeypatch.setattr(asnmpc, "sens_solution", fake_estimate)
    monkeypatch.setattr(asnmpc, "sens_active_set_changes", fake_events)
    monkeypatch.setattr(asnmpc, "trajectory", fake_trajectory)

    got = asnmpc._corrected_update(
        background,
        BASE_STATE,
        "path",
        smoke_config,
    )

    assert calls == ["clock", "report", "estimate", "clock", "events", "trajectory"]
    assert got == (corrected, report, (), 2.5, values)


def test_accepted_correction_populates_the_next_warm_start(monkeypatch, smoke_config):
    campaign = asnmpc.Campaign(
        name="warm_start_probe",
        feed_temperature_shift=(0.0, 0.0),
        measurement_noise=((0.01, 0.005), (0.0, 0.0)),
    )
    corrected_trajectories = []
    warm_start_sources = []
    real_correction = asnmpc._corrected_update
    real_solve = asnmpc._solve_with_recovery

    def capture_correction(*args, **kwargs):
        result = real_correction(*args, **kwargs)
        corrected_trajectories.append(result[0])
        return result

    def capture_warm_start(initial_state, config, warm_start=None):
        if warm_start is not None:
            warm_start_sources.append(trajectory(warm_start))
        return real_solve(initial_state, config, warm_start)

    monkeypatch.setattr(asnmpc, "_corrected_update", capture_correction)
    monkeypatch.setattr(asnmpc, "_solve_with_recovery", capture_warm_start)
    run_closed_loop("path", campaign, smoke_config)

    expected = corrected_trajectories[0]
    supplied = warm_start_sources[0]
    assert supplied.state_at(smoke_config.sample_time_min) == pytest.approx(
        expected.state_at(smoke_config.sample_time_min)
    )
    assert supplied.control_at(smoke_config.sample_time_min) == pytest.approx(
        expected.control_at(smoke_config.sample_time_min)
    )


def test_recovery_counts_reach_the_sample_and_summary(monkeypatch, smoke_config):
    nominal = make_campaigns(steps=6, seed=19)[0]
    real_solve = asnmpc.solve_controller
    injected = False

    def fail_one_warm_start(initial_state, config=None, warm_start=None):
        nonlocal injected
        if warm_start is not None and not injected:
            injected = True
            raise RuntimeError("injected warm-start failure")
        return real_solve(initial_state, config, warm_start)

    monkeypatch.setattr(asnmpc, "solve_controller", fail_one_warm_start)
    result = run_closed_loop("full_resolve", nominal, smoke_config)

    assert injected
    assert sum(sample.solver_failures for sample in result.samples) == 1
    assert sum(sample.solver_recoveries for sample in result.samples) == 1
    assert result.metrics.solver_failures == 1
    assert result.metrics.solver_recoveries == 1
