"""Regression tests for the differentiable phase-envelope example."""

import numpy as np
import pytest

from pounce.examples.phase_envelope import (
    DEITERS_BELL_METHANE_PROPANE,
    NATURAL_GAS,
    composition_from_coordinates,
    design_composition_for_extremum,
    design_parameters,
    diagnose_phase_point,
    diagnose_phase_trace,
    phase_boundary_with_sensitivity,
    reparameterize_trace,
    refine_fold,
    simplex_coordinates,
    solve_low_pressure_anchor,
    trace_envelope,
    vapor_pressure,
)


def test_simplex_log_ratio_coordinates_round_trip():
    """The unconstrained design coordinates must preserve the simplex."""

    mixture = DEITERS_BELL_METHANE_PROPANE
    eta = simplex_coordinates(mixture.composition)
    recovered = np.asarray(composition_from_coordinates(eta))

    np.testing.assert_allclose(recovered, mixture.composition, atol=1e-14)
    assert np.all(recovered > 0.0)
    assert np.isclose(recovered.sum(), 1.0, atol=1e-14)


def test_n_butane_vapor_pressure_matches_literature():
    """Pure-component roots stay on distinct, admissible cubic branches."""

    expected_bar = np.array([1.03, 2.58, 9.42])  # [bar]
    actual_bar = np.array(
        [
            vapor_pressure(NATURAL_GAS, 3, temperature_k) / 1e5
            for temperature_k in (273.15, 300.0, 350.0)  # [K]
        ]
    )  # [bar]

    np.testing.assert_allclose(actual_bar, expected_bar, rtol=0.01, atol=0.02)
    with pytest.raises(ValueError, match="two-phase saturation root"):
        vapor_pressure(NATURAL_GAS, 3, 450.0)  # [K], above n-butane's Tc


def test_fixed_pressure_sensitivity_matches_fresh_phase_boundary_solves():
    """Implicit design derivatives agree with central, fully re-solved roots."""

    mixture = DEITERS_BELL_METHANE_PROPANE
    pressure_pa = 5e5  # [Pa]
    kij_pair = (0, 1)
    point = phase_boundary_with_sensitivity(
        mixture,
        beta=0.0,
        pressure_pa=pressure_pa,
        kij_pair=kij_pair,
    )
    q0 = design_parameters(mixture, kij_pair)
    central = np.empty_like(q0)

    for column, step in enumerate((1e-4, 1e-5)):
        temperatures = []
        for sign in (-1.0, 1.0):
            design = q0.copy()
            design[column] += sign * step
            state = solve_low_pressure_anchor(
                mixture,
                beta=0.0,
                pressure_pa=pressure_pa,
                design=design,
                kij_pair=kij_pair,
            )
            temperatures.append(float(np.exp(np.asarray(state)[-1])))  # [K]
        central[column] = (temperatures[1] - temperatures[0]) / (2.0 * step)

    assert point.max_residual < 1e-10
    np.testing.assert_allclose(
        point.temperature_design_jacobian,
        central,
        rtol=2e-4,
        atol=2e-4,
    )

    # A caller-provided state is a warm start, not permission to differentiate
    # an arbitrary non-root.  Perturb the temperature enough to give a large
    # equilibrium residual and require the public helper to solve it back.
    perturbed_seed = point.state.copy()
    perturbed_seed[-1] += 0.02
    recovered = phase_boundary_with_sensitivity(
        mixture,
        beta=0.0,
        pressure_pa=pressure_pa,
        kij_pair=kij_pair,
        initial_state=perturbed_seed,
    )
    assert recovered.max_residual < 1e-10
    assert not np.array_equal(recovered.state, perturbed_seed)
    np.testing.assert_allclose(recovered.state, point.state, rtol=1e-9, atol=1e-9)


def test_published_binary_fold_and_inverse_design_regression():
    """Reproduce a published maxcondentherm and solve an inverse design."""

    mixture = DEITERS_BELL_METHANE_PROPANE
    trace = trace_envelope(
        mixture,
        beta=0.0,
        mode="temperature",
        ds=0.1,
        n_steps=145,
    )
    fold = refine_fold(
        trace,
        mixture,
        beta=0.0,
        mode="temperature",
        kij_pair=(0, 1),
    )
    pressure_fold = refine_fold(
        reparameterize_trace(
            trace,
            mixture,
            from_mode="temperature",
            to_mode="pressure",
        ),
        mixture,
        beta=0.0,
        mode="pressure",
        kij_pair=(0, 1),
    )
    diagnostics = diagnose_phase_point(
        fold.state,
        fold.log_parameter,
        mixture,
        beta=0.0,
        mode="temperature",
    )

    # Deiters & Bell (AIChE J. 65, 2019, e16730) report 282.53 K for
    # the methane/propane x_methane=0.8 maxcondentherm with this PR model.
    assert abs(fold.temperature_k - 282.53) < 0.02  # [K]
    assert fold.max_residual < 1e-9
    assert diagnostics.max_residual < 1e-9
    assert diagnostics.branch_distance > 0.5
    assert diagnostics.roots_are_admissible
    assert np.isclose(diagnostics.liquid_sum, 1.0, atol=1e-10)
    assert np.isclose(diagnostics.vapor_sum, 1.0, atol=1e-10)

    # The augmented folds should be insensitive to the continuation sampling
    # once each run brackets the same two extrema.
    fine_trace = trace_envelope(
        mixture,
        beta=0.0,
        mode="temperature",
        ds=0.08,
        n_steps=180,
    )
    fine_temperature_fold = refine_fold(
        fine_trace,
        mixture,
        beta=0.0,
        mode="temperature",
        kij_pair=(0, 1),
    )
    fine_pressure_fold = refine_fold(
        reparameterize_trace(
            fine_trace,
            mixture,
            from_mode="temperature",
            to_mode="pressure",
        ),
        mixture,
        beta=0.0,
        mode="pressure",
        kij_pair=(0, 1),
    )
    np.testing.assert_allclose(
        [
            pressure_fold.pressure_pa,
            pressure_fold.temperature_k,
            fold.temperature_k,
            fold.pressure_pa,
        ],
        [
            fine_pressure_fold.pressure_pa,
            fine_pressure_fold.temperature_k,
            fine_temperature_fold.temperature_k,
            fine_temperature_fold.pressure_pa,
        ],
        rtol=1e-8,
        atol=1e-6,
    )

    design = design_composition_for_extremum(
        fold,
        mixture,
        beta=0.0,
        target=280.0,  # [K]
        kij_pair=(0, 1),
    )
    assert abs(design.achieved - design.target) < 1e-7  # [K]
    assert design.max_residual < 1e-9
    assert np.isclose(design.mixture.composition.sum(), 1.0, atol=1e-12)
    assert np.all(design.mixture.composition > 0.0)

    verification_trace = trace_envelope(
        design.mixture,
        beta=0.0,
        mode="temperature",
        ds=0.1,
        n_steps=170,
    )
    verification_fold = refine_fold(
        verification_trace,
        design.mixture,
        beta=0.0,
        mode="temperature",
        kij_pair=(0, 1),
    )
    verification = diagnose_phase_point(
        verification_fold.state,
        verification_fold.log_parameter,
        design.mixture,
        beta=0.0,
        mode="temperature",
    )
    trace_diagnostics = diagnose_phase_trace(
        verification_trace,
        design.mixture,
        beta=0.0,
        mode="temperature",
    )
    trace_temperatures_k = np.exp(verification_trace.theta)  # [K]
    traced_fold_index = int(np.argmax(trace_temperatures_k))  # [-]
    assert abs(verification_fold.temperature_k - design.target) < 1e-6  # [K]
    assert verification_fold.max_residual < 1e-9
    assert verification.max_residual < 1e-9
    assert verification.branch_distance > 0.5
    assert verification.roots_are_admissible
    assert np.isclose(verification.liquid_sum, 1.0, atol=1e-10)
    assert np.isclose(verification.vapor_sum, 1.0, atol=1e-10)
    assert len(trace_diagnostics) == len(trace_temperatures_k) > 2
    assert 0 < traced_fold_index < len(trace_temperatures_k) - 1
    assert trace_temperatures_k[-1] < trace_temperatures_k[traced_fold_index]
    assert max(point.max_residual for point in trace_diagnostics) < 1e-8
    assert all(point.roots_are_admissible for point in trace_diagnostics)
    assert all(
        np.isclose(point.liquid_sum, 1.0, atol=1e-9) for point in trace_diagnostics
    )
    assert all(
        np.isclose(point.vapor_sum, 1.0, atol=1e-9) for point in trace_diagnostics
    )
