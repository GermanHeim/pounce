"""End-to-end checks for the active-set-aware NMPC tutorial."""

import math

import numpy as np
import pytest

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
    assert result.metrics.maximum_temperature_violation == 0.0
    assert math.isfinite(result.metrics.iae)
    assert math.isfinite(result.metrics.economic_stage_cost)


def test_guard_accepts_nominal_updates_and_falls_back_under_stress(smoke_config):
    nominal, _, stress = make_campaigns(steps=6, seed=19)

    accepted = run_closed_loop("guarded_path", nominal, smoke_config)
    guarded = run_closed_loop("guarded_path", stress, smoke_config)

    assert accepted.metrics.fallback_full_solves == 0
    assert 0 < guarded.metrics.fallback_full_solves <= stress.steps
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
