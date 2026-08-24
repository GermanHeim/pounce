"""Physical and optimization checks for the catalyst-pellet tutorial."""

from dataclasses import replace

import numpy as np
import pytest

pytest.importorskip("jax")

from pounce.catalyst_pellet import (  # noqa: E402
    PelletConfig,
    analytical_effectiveness,
    egg_shell_activity,
    fit_effective_parameters,
    implicit_observable_jacobian,
    koschany_rate,
    refine_solution,
    solve_design,
    solve_first_order_sphere,
    solve_forward,
    solve_nested_design,
    uncertainty_scenarios,
    validate_uncertainty,
)


def test_published_rate_and_analytical_sphere_regression():
    config = PelletConfig()
    partial_pressures = config.pressure_bar * np.asarray(config.bulk_mole_fractions)

    # Independent numeric pin of the published Koschany parameter table at
    # its 555 K reference state (zero products, 1 bar CO2, 4 bar H2).
    np.testing.assert_allclose(
        koschany_rate(partial_pressures, 555.0),
        9.084226002938914e-5,
        rtol=2e-13,
    )

    for thiele_modulus in (0.0, 0.1, 1.0, 10.0):
        numerical, radius, concentration, balance = solve_first_order_sphere(
            thiele_modulus, nodes=160
        )
        np.testing.assert_allclose(
            numerical,
            analytical_effectiveness(thiele_modulus),
            rtol=6e-5,
            atol=2e-8,
        )
        assert balance < 2e-8
        assert np.all(np.diff(radius) > 0.0)
        assert np.all(np.diff(concentration) >= -1e-12)


def test_uniform_pellet_closes_balances_films_and_mesh():
    config = PelletConfig(nodes=8, zones=4)
    activity = np.full(config.zones, config.activity_inventory)
    coarse = solve_forward(activity, config)

    assert coarse.success, coarse.message
    assert coarse.max_scaled_residual < 1e-10
    assert np.max(coarse.species_balance_relative) < 1e-10
    assert coarse.energy_balance_relative < 1e-10
    assert np.all(coarse.concentrations_mol_m3 > 0.0)
    assert config.bulk_temperature_k < coarse.max_temperature_k
    assert coarse.max_temperature_k < config.temperature_limit_k
    # The first cell is a finite volume around r=0; its inner face has exactly
    # zero area, so no singular center equation or numerical 1/r is present.
    assert coarse.radius_m[0] > 0.0

    refined = refine_solution(coarse, config, nodes=12)
    assert refined.success, refined.message
    assert refined.max_scaled_residual < 1e-9
    assert abs(refined.production_mol_s / coarse.production_mol_s - 1.0) < 8e-4
    assert abs(refined.max_temperature_k - coarse.max_temperature_k) < 0.1

    slow_film = replace(
        config,
        mass_transfer_coefficients_m_s=(0.008, 0.008, 0.008, 0.008),
    )
    film_limited = solve_forward(activity, slow_film)
    assert film_limited.success, film_limited.message
    assert film_limited.production_mol_s < coarse.production_mol_s


def test_implicit_design_gradient_matches_perturb_and_resolve():
    config = PelletConfig(nodes=6, zones=3)
    activity = np.full(config.zones, config.activity_inventory)
    solution = solve_forward(activity, config)
    exact = implicit_observable_jacobian(solution, config)
    epsilon = 1e-5
    finite_difference = []
    for index in range(config.zones):
        plus = activity.copy()
        minus = activity.copy()
        plus[index] += epsilon
        minus[index] -= epsilon
        plus_solution = solve_forward(plus, config, initial_state=solution.state_scaled)
        minus_solution = solve_forward(
            minus, config, initial_state=solution.state_scaled
        )
        finite_difference.append(
            (
                np.array(
                    [
                        plus_solution.production_mol_s,
                        plus_solution.max_temperature_k,
                    ]
                )
                - np.array(
                    [
                        minus_solution.production_mol_s,
                        minus_solution.max_temperature_k,
                    ]
                )
            )
            / (2.0 * epsilon)
        )
    np.testing.assert_allclose(
        exact, np.stack(finite_difference, axis=1), rtol=2e-6, atol=2e-10
    )


def test_simultaneous_and_nested_design_agree_and_refine():
    config = PelletConfig(nodes=6, zones=3)
    uniform_activity = np.full(config.zones, config.activity_inventory)
    uniform = solve_forward(uniform_activity, config)
    egg_shell = solve_forward(egg_shell_activity(config), config)
    simultaneous = solve_design(config)
    nested = solve_nested_design(config)

    assert simultaneous.success, simultaneous.status
    assert nested.success, nested.status
    assert simultaneous.max_constraint_violation < 1e-8
    np.testing.assert_allclose(
        simultaneous.activity, nested.activity, rtol=8e-4, atol=3e-6
    )
    np.testing.assert_allclose(
        simultaneous.nominal.production_mol_s,
        nested.nominal.production_mol_s,
        rtol=2e-6,
    )
    assert np.min(simultaneous.activity) >= -1e-8
    assert np.max(simultaneous.activity) <= config.activity_upper + 1e-8
    assert abs(np.mean(simultaneous.activity) - config.activity_inventory) < 1e-9
    assert simultaneous.nominal.production_mol_s > uniform.production_mol_s
    assert egg_shell.production_mol_s > simultaneous.nominal.production_mol_s
    assert np.sum(np.diff(simultaneous.activity) ** 2) < np.sum(
        np.diff(egg_shell.activity) ** 2
    )

    second_start = solve_design(config, initial_activity=egg_shell_activity(config))
    assert second_start.success
    np.testing.assert_allclose(
        simultaneous.activity, second_start.activity, rtol=1e-5, atol=1e-7
    )
    refined = refine_solution(simultaneous.nominal, config, nodes=9)
    assert refined.success, refined.message
    assert (
        abs(refined.production_mol_s / simultaneous.nominal.production_mol_s - 1.0)
        < 0.015
    )
    assert refined.max_temperature_k < config.temperature_limit_k


def test_covariance_drives_robust_design_and_sampled_resolves():
    config = PelletConfig(nodes=6, zones=3)
    fit = fit_effective_parameters(config=config)
    assert fit.success, fit.message
    assert fit.cov_source == "reduced_hessian"
    assert np.all(np.linalg.eigvalsh(fit.pcov) > 0.0)
    assert np.all(np.abs(fit.popt) < 3.0 * fit.perr)

    scenarios = uncertainty_scenarios(fit.popt, fit.pcov)
    nominal = solve_design(config, scenarios=(scenarios[0],))
    robust = solve_design(
        config,
        scenarios=scenarios,
        robust=True,
        initial_activity=nominal.activity,
    )
    assert robust.success, robust.status
    assert robust.guaranteed_production_mol_s is not None
    worst_case = min(solution.production_mol_s for solution in robust.solutions)
    np.testing.assert_allclose(
        robust.guaranteed_production_mol_s, worst_case, rtol=2e-8
    )
    assert all(
        solution.max_temperature_k < config.temperature_limit_k
        for solution in robust.solutions
    )
    assert not np.allclose(robust.activity, nominal.activity, atol=1e-4)

    validation = validate_uncertainty(
        robust.activity,
        fit.popt,
        fit.pcov,
        config,
        samples=8,
        seed=103,
    )
    ratio = validation.sampled_standard_deviation / validation.delta_standard_deviation
    assert np.all((ratio > 0.4) & (ratio < 2.5))
    assert validation.sampled_maximum[1] < config.temperature_limit_k
