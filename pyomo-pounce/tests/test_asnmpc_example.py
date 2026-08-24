"""End-to-end checks for the active-set-aware NMPC tutorial."""

import math

import numpy as np
import pytest

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


def test_switching_campaign_records_active_set_events():
    config = CstrConfig(deadline_s=2.0)
    switching = make_campaigns(steps=12, seed=19)[1]

    result = run_closed_loop("path", switching, config)
    event_rows = result.event_timeline()

    assert result.metrics.active_set_changes > 0
    assert sum(row["path_length"] for row in event_rows) == (
        result.metrics.active_set_changes
    )
    assert any(row["active_set_changes"] for row in event_rows)


def test_directional_degeneracy_experiment_reaches_the_control_kink():
    result = directional_degeneracy_experiment()

    assert result.breakpoint_variable.startswith("v1[")
    assert result.breakpoint_bound == "lower"
    assert 0.0 < result.breakpoint_fraction < 0.1
    assert result.steps[0].directional_events
    assert all(step.guard_reasons for step in result.steps)
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


def test_failed_warm_start_is_retried_cold_and_counted(monkeypatch, smoke_config):
    warm_start = object()
    recovered_model = object()
    attempted_warm_starts = []

    def fake_solve(initial_state, config=None, warm_start=None):
        attempted_warm_starts.append(warm_start)
        if warm_start is not None:
            raise RuntimeError("injected warm-start failure")
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
