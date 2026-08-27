"""Physical and optimization checks for the catalyst-pellet tutorial."""

from dataclasses import replace

import numpy as np
import pytest

pytest.importorskip("jax")

from pounce.examples.catalyst_pellet import (  # noqa: E402
    PelletConfig,
    Scenario,
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


def test_activity_zones_require_an_equal_volume_partition():
    invalid = PelletConfig(nodes=8, zones=3)
    message = "nodes must be divisible by zones"
    with pytest.raises(ValueError, match=message):
        egg_shell_activity(invalid)
    with pytest.raises(ValueError, match=message):
        solve_forward(np.full(3, invalid.activity_inventory), invalid)
    with pytest.raises(ValueError, match=message):
        solve_design(invalid)
    with pytest.raises(ValueError, match=message):
        solve_nested_design(invalid)

    valid = PelletConfig(nodes=8, zones=4)
    solution = solve_forward(np.full(4, valid.activity_inventory), valid)
    with pytest.raises(ValueError, match=message):
        refine_solution(solution, valid, nodes=10)


def test_default_design_configuration_converges():
    """The public no-argument design route must have a convergent seed."""

    result = solve_design()

    assert result.success, result.status
    # POUNCE reports the maximum scaled constraint violation [-].
    assert result.max_constraint_violation < 1.0e-8


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

    scenario_exact = implicit_observable_jacobian(
        solution, config, with_respect_to="scenario"
    )
    scenario_finite_difference = []
    for index in range(2):
        displacement = np.zeros(2)
        displacement[index] = epsilon
        plus_scenario = Scenario(*displacement, label=f"parameter-{index}+")
        minus_scenario = Scenario(*(-displacement), label=f"parameter-{index}-")
        plus_solution = solve_forward(
            activity,
            config,
            scenario=plus_scenario,
            initial_state=solution.state_scaled,
        )
        minus_solution = solve_forward(
            activity,
            config,
            scenario=minus_scenario,
            initial_state=solution.state_scaled,
        )
        scenario_finite_difference.append(
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
        scenario_exact,
        np.stack(scenario_finite_difference, axis=1),
        rtol=2e-6,
        atol=2e-10,
    )

    # The fitted transport parameter is specifically the CO2 effective
    # diffusivity; applying a scenario perturbation must leave the H2, CH4,
    # and H2O diffusivities unchanged.
    diffusivities = np.asarray(config.effective_diffusivities_m2_s)
    perturbed_config = replace(
        config,
        effective_diffusivities_m2_s=tuple(
            diffusivities * np.array([np.exp(epsilon), 1.0, 1.0, 1.0])
        ),
    )
    scenario_solution = solve_forward(
        activity,
        config,
        scenario=Scenario(log_diffusivity_scale=epsilon),
        initial_state=solution.state_scaled,
    )
    explicit_solution = solve_forward(
        activity,
        perturbed_config,
        initial_state=solution.state_scaled,
    )
    np.testing.assert_allclose(
        [scenario_solution.production_mol_s, scenario_solution.max_temperature_k],
        [explicit_solution.production_mol_s, explicit_solution.max_temperature_k],
        rtol=2e-10,
        atol=2e-14,
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


def test_thermal_ceiling_is_a_constraint_not_a_failed_state_solve():
    """gh#787: the design ceiling and the root-solve bracket are separate.

    The nested route can only respond to ``temperature_limit_k`` if a candidate
    that violates it still converges, so that the outer optimizer is handed a
    negative margin rather than an exception.  The configuration below puts the
    ceiling at 570 K, below the 572.5 K peak of the uniform pellet, which makes
    the constraint bind at the nominal eight-cell operating point.
    """

    config = replace(PelletConfig(nodes=8, zones=4), temperature_limit_k=570.0)
    uniform = np.full(config.zones, config.activity_inventory)

    # Converged *and* thermally infeasible. The root solve reaches a genuine
    # root -- residual and energy closure at solver tolerance -- whose peak
    # temperature nonetheless violates the design ceiling.
    hot = solve_forward(uniform, config)
    assert hot.success, hot.message
    assert hot.max_scaled_residual < 1e-10
    assert hot.energy_balance_relative < 1e-10
    assert hot.max_temperature_k > config.temperature_limit_k
    assert hot.thermal_margin_k < 0.0
    assert not hot.thermally_feasible

    # The pre-fix formulation, recovered by collapsing the numerical bracket
    # onto the design ceiling: the identical candidate now fails the bounded
    # least-squares solve, its peak temperature pinned at the bound instead of
    # sitting at a root. That is the state this issue is about -- a failed
    # state solve where the honest answer is "converged, but too hot".
    coupled = replace(config, state_temperature_ceiling_k=config.temperature_limit_k)
    pinned = solve_forward(uniform, coupled)
    assert not pinned.success
    assert pinned.max_scaled_residual > 1e-4
    assert pinned.energy_balance_relative > 1e-3
    assert pinned.max_temperature_k == pytest.approx(
        config.temperature_limit_k, abs=1e-9
    )

    # With the two separated, SLSQP sees the constraint and moves off it. Both
    # routes drive the ceiling active and land on the same design, even though
    # the nested route enforces it as an explicit inequality on the converged
    # inner solve and the simultaneous route as a state variable bound.
    nested = solve_nested_design(config)
    simultaneous = solve_design(config)

    assert nested.success, nested.status
    assert simultaneous.success, simultaneous.status
    assert nested.max_constraint_violation < 1e-7
    assert simultaneous.max_constraint_violation < 1e-8
    assert nested.nominal.thermal_margin_k == pytest.approx(0.0, abs=1e-6)
    assert simultaneous.nominal.thermal_margin_k == pytest.approx(0.0, abs=1e-6)
    np.testing.assert_allclose(
        nested.activity, simultaneous.activity, rtol=1e-5, atol=1e-7
    )
    np.testing.assert_allclose(
        nested.nominal.production_mol_s,
        simultaneous.nominal.production_mol_s,
        rtol=1e-6,
    )
    # The ceiling is load-bearing here: honouring it costs production against
    # the equal-inventory uniform pellet that violates it.
    assert nested.nominal.production_mol_s < hot.production_mol_s


def test_root_solve_bracket_must_contain_the_design_ceiling():
    with pytest.raises(ValueError, match="state_temperature_ceiling_k"):
        PelletConfig(temperature_limit_k=613.0, state_temperature_ceiling_k=600.0)
    with pytest.raises(ValueError, match="state_temperature_floor_k"):
        PelletConfig(temperature_limit_k=613.0, state_temperature_floor_k=620.0)


def test_eight_cell_routes_agree_when_the_thermal_ceiling_is_slack():
    """The nominal validated mesh is unaffected by the gh#787 separation."""

    config = PelletConfig(nodes=8, zones=4)
    nested = solve_nested_design(config)
    simultaneous = solve_design(config)

    assert nested.success, nested.status
    assert simultaneous.success, simultaneous.status
    # Both reported designs sit near 579 K, well under the 613 K ceiling, so
    # the thermal constraint is inactive and the two routes are comparing only
    # the balances and the inventory.
    assert nested.nominal.thermal_margin_k > 30.0
    assert simultaneous.nominal.thermal_margin_k > 30.0
    assert nested.nominal.thermally_feasible
    assert simultaneous.nominal.thermally_feasible
    np.testing.assert_allclose(
        nested.activity, simultaneous.activity, rtol=8e-4, atol=3e-6
    )
    np.testing.assert_allclose(
        nested.nominal.production_mol_s,
        simultaneous.nominal.production_mol_s,
        rtol=2e-6,
    )
    np.testing.assert_allclose(
        nested.nominal.max_temperature_k,
        simultaneous.nominal.max_temperature_k,
        rtol=2e-8,
    )


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
