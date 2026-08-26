"""Uncertainty-aware inverse design of a nonisothermal catalyst pellet.

This module is the reusable numerical kernel behind notebook 37.  It models
one spherical CO2-methanation pellet with four species and energy balances,
the Koschany--Schlereth--Hinrichsen kinetic law, radially distributed
catalyst activity, external films, and conservative finite volumes.

The scope is deliberately one pellet at a prescribed bulk state.  It is not a
reactor model and does not claim to reproduce the reactor-coupled optimum of
Zimmermann, Bremer, and Sundmacher.  Their open-access study supplies the
particle size, operating point, thermal properties, kinetic parameter table,
and the qualitative structured-particle comparison used here.  Assumptions
not fixed by that study (effective diffusivities and film coefficients) are
explicit fields of :class:`PelletConfig`; no hidden tuning constants are used.

Units
-----
Concentrations are mol m^-3, temperature is K, diffusivity is m^2 s^-1,
thermal conductivity is W m^-1 K^-1, mass-transfer coefficients are m s^-1,
heat-transfer coefficients are W m^-2 K^-1, and the specific kinetic rate is
mol CO2 (g_cat s)^-1.  ``activity`` is dimensionless on the Koschany catalyst
basis.  The internal optimizer scales concentrations by the bulk total molar
concentration and temperature by 100 K; public results are dimensional.

References
----------
* F. Koschany, D. Schlereth, O. Hinrichsen, Applied Catalysis B 181
  (2016) 504--516, doi:10.1016/j.apcatb.2015.07.026.
* R. T. Zimmermann, J. Bremer, K. Sundmacher, Chemical Engineering
  Journal 387 (2020) 123704, doi:10.1016/j.cej.2019.123704.
* R. Baratti, H. Wu, M. Morbidelli, A. Varma, Chemical Engineering
  Science 48 (1993) 1869--1881, doi:10.1016/0009-2509(93)80357-V.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Sequence

import numpy as np
from scipy.optimize import least_squares


GAS_CONSTANT = 8.31446261815324  # [J mol^-1 K^-1]
SPECIES = ("CO2", "H2", "CH4", "H2O")
STOICHIOMETRY = np.array([-1.0, -4.0, 1.0, 2.0])  # [mol_i mol_CO2^-1]
TEMPERATURE_SCALE_K = 100.0


@dataclass(frozen=True)
class KoschanyKinetics:
    """Published CO2-methanation kinetic constants.

    The reference-specific rate coefficient is on a gram-of-catalyst basis.
    Adsorption enthalpies use the sign convention in the published anchored
    correlation ``K(T) = K_ref exp[dH/R (1/T_ref - 1/T)]``.
    """

    reference_temperature_k: float = 555.0  # [K]
    rate_ref_mol_bar_g_s: float = 3.46e-4  # [mol bar^-1 g_cat^-1 s^-1]
    activation_energy_j_mol: float = 77_500.0  # [J mol^-1]
    k_oh_ref_bar_mhalf: float = 0.5  # [bar^-0.5]
    dh_oh_j_mol: float = 22_400.0  # [J mol^-1]
    k_h2_ref_bar_mhalf: float = 0.44  # [bar^-0.5]
    dh_h2_j_mol: float = -6_200.0  # [J mol^-1]
    k_mix_ref_bar_mhalf: float = 0.88  # [bar^-0.5]
    dh_mix_j_mol: float = -10_000.0  # [J mol^-1]
    equilibrium_prefactor_bar_m2: float = 137.0  # [bar^-2]
    equilibrium_temperature_exponent: float = -3.998  # [-]
    equilibrium_energy_j_mol: float = 158_700.0  # [J mol^-1]


@dataclass(frozen=True)
class PelletConfig:
    """Physical and numerical contract for the single-pellet study.

    ``radius_m``, ``pressure_bar``, ``solid_density_kg_m3``, and
    ``thermal_conductivity_w_m_k`` come from Zimmermann et al.  The bulk
    temperature is the Koschany reference temperature, inside the reported
    453--613 K kinetic range.  The upper temperature bound is therefore the
    613 K validity ceiling, which is stricter than the 725 K design limit used
    in the reactor paper.  Effective diffusivities and film coefficients are
    explicit tutorial assumptions representative of a small porous pellet;
    they must be replaced when a particular support is being modeled.
    ``nodes`` must be divisible by ``zones`` so each activity coefficient owns
    the same physical volume on every solve and refinement mesh.
    """

    radius_m: float = 1.25e-3  # [m], 2.5 mm diameter
    pressure_bar: float = 5.0  # [bar]
    bulk_temperature_k: float = 555.0  # [K]
    bulk_mole_fractions: tuple[float, float, float, float] = (0.2, 0.8, 0.0, 0.0)
    effective_diffusivities_m2_s: tuple[float, float, float, float] = (
        1.0e-6,
        2.8e-6,
        1.2e-6,
        1.1e-6,
    )
    mass_transfer_coefficients_m_s: tuple[float, float, float, float] = (
        0.08,
        0.14,
        0.09,
        0.09,
    )
    thermal_conductivity_w_m_k: float = 2.5  # [W m^-1 K^-1]
    heat_transfer_coefficient_w_m2_k: float = 250.0  # [W m^-2 K^-1]
    solid_density_kg_m3: float = 4500.0  # [kg m^-3]
    pellet_porosity: float = 0.35  # [-], explicit tutorial assumption
    reaction_enthalpy_j_mol: float = -164_000.0  # [J mol_CO2^-1]
    temperature_limit_k: float = 613.0  # [K], kinetic validity ceiling
    activity_inventory: float = 0.16  # volume-average activity [-]
    activity_upper: float = 1.0  # Koschany catalyst basis [-]
    regularization_weight: float = 0.5  # objective weight [-]
    nodes: int = 8
    zones: int = 4
    kinetics: KoschanyKinetics = KoschanyKinetics()

    @property
    def bulk_total_concentration_mol_m3(self) -> float:
        """Ideal-gas bulk concentration [mol m^-3]."""

        return self.pressure_bar * 1.0e5 / (GAS_CONSTANT * self.bulk_temperature_k)

    @property
    def bulk_concentrations_mol_m3(self) -> np.ndarray:
        """Bulk species concentrations in ``SPECIES`` order [mol m^-3]."""

        return (
            np.asarray(self.bulk_mole_fractions, dtype=float)
            * self.bulk_total_concentration_mol_m3
        )

    @property
    def catalyst_density_g_m3(self) -> float:
        """Catalyst mass per pellet volume [g_cat m^-3]."""

        return 1000.0 * self.solid_density_kg_m3 * (1.0 - self.pellet_porosity)


@dataclass(frozen=True)
class Scenario:
    """Log-scale kinetic and CO2-diffusivity perturbations, both dimensionless."""

    log_rate_scale: float = 0.0
    log_diffusivity_scale: float = 0.0
    label: str = "nominal"


@dataclass
class PelletSolution:
    """Dimensional forward solution and conservative diagnostics."""

    radius_m: np.ndarray
    concentrations_mol_m3: np.ndarray
    temperature_k: np.ndarray
    activity: np.ndarray
    cell_activity: np.ndarray
    production_mol_s: float
    surface_flux_mol_s: float
    max_scaled_residual: float
    species_balance_relative: np.ndarray
    energy_balance_relative: float
    success: bool
    message: str
    state_scaled: np.ndarray
    scenario: Scenario

    @property
    def max_temperature_k(self) -> float:
        """Maximum cell-center temperature [K]."""

        return float(np.max(self.temperature_k))

    @property
    def effectiveness(self) -> float:
        """Effective/intrinsic production ratio for this activity inventory."""

        intrinsic = getattr(self, "_intrinsic_production_mol_s", np.nan)
        return float(self.production_mol_s / intrinsic)


@dataclass
class DesignResult:
    """Simultaneous POUNCE design result over one or more scenarios."""

    activity: np.ndarray
    solutions: tuple[PelletSolution, ...]
    objective: float
    success: bool
    status: str
    iterations: int
    max_constraint_violation: float
    guaranteed_production_mol_s: float | None
    raw_info: dict

    @property
    def nominal(self) -> PelletSolution:
        """Nominal scenario solution (the first scenario by contract)."""

        return self.solutions[0]


@dataclass(frozen=True)
class CalibrationData:
    """Synthetic intrinsic/apparent-rate observations used for covariance."""

    conditions: np.ndarray
    log_rates: np.ndarray
    sigma: np.ndarray
    true_parameters: np.ndarray


@dataclass(frozen=True)
class UncertaintyValidation:
    """Delta-method and sampled re-solve comparison for two observables."""

    observable_names: tuple[str, str]
    nominal: np.ndarray
    delta_standard_deviation: np.ndarray
    sampled_standard_deviation: np.ndarray
    sampled_minimum: np.ndarray
    sampled_maximum: np.ndarray
    samples: np.ndarray


def _kinetic_rate_backend(partial_pressures_bar, temperature_k, activity, kinetics, xp):
    """Koschany rate law on an ``numpy`` or ``jax.numpy`` backend."""

    p = xp.maximum(xp.asarray(partial_pressures_bar), 1.0e-12)
    temperature_k = xp.maximum(xp.asarray(temperature_k), 250.0)
    inv_delta = 1.0 / kinetics.reference_temperature_k - 1.0 / temperature_k
    k = kinetics.rate_ref_mol_bar_g_s * xp.exp(
        kinetics.activation_energy_j_mol / GAS_CONSTANT * inv_delta
    )
    k_oh = kinetics.k_oh_ref_bar_mhalf * xp.exp(
        kinetics.dh_oh_j_mol / GAS_CONSTANT * inv_delta
    )
    k_h2 = kinetics.k_h2_ref_bar_mhalf * xp.exp(
        kinetics.dh_h2_j_mol / GAS_CONSTANT * inv_delta
    )
    k_mix = kinetics.k_mix_ref_bar_mhalf * xp.exp(
        kinetics.dh_mix_j_mol / GAS_CONSTANT * inv_delta
    )
    k_eq = (
        kinetics.equilibrium_prefactor_bar_m2
        * temperature_k**kinetics.equilibrium_temperature_exponent
        * xp.exp(kinetics.equilibrium_energy_j_mol / (GAS_CONSTANT * temperature_k))
    )
    p_co2, p_h2, p_ch4, p_h2o = p
    driving_force = 1.0 - (p_ch4 * p_h2o**2) / (k_eq * p_co2 * p_h2**4)
    denominator = (
        1.0
        + k_oh * p_h2o / xp.sqrt(p_h2)
        + k_h2 * xp.sqrt(p_h2)
        + k_mix * xp.sqrt(p_co2)
    ) ** 2
    return activity * k * xp.sqrt(p_h2 * p_co2) * driving_force / denominator


def koschany_rate(partial_pressures_bar, temperature_k, activity=1.0):
    """Specific CO2 consumption rate [mol g_cat^-1 s^-1]."""

    return np.asarray(
        _kinetic_rate_backend(
            partial_pressures_bar,
            temperature_k,
            activity,
            KoschanyKinetics(),
            np,
        )
    )


def analytical_effectiveness(thiele_modulus):
    """First-order spherical effectiveness factor ``3/phi*(coth(phi)-1/phi)``.

    The series around zero avoids cancellation and returns exactly one at
    ``phi=0``.  Inputs and outputs are dimensionless.
    """

    phi = np.asarray(thiele_modulus, dtype=float)
    small = np.abs(phi) < 1.0e-4
    safe = np.where(small, 1.0, phi)
    regular = 3.0 / safe * (1.0 / np.tanh(safe) - 1.0 / safe)
    series = 1.0 - phi**2 / 15.0 + 2.0 * phi**4 / 315.0
    out = np.where(small, series, regular)
    return float(out) if out.ndim == 0 else out


def _analytical_effectiveness_backend(phi, xp):
    safe = xp.where(xp.abs(phi) < 1.0e-4, 1.0, phi)
    regular = 3.0 / safe * (1.0 / xp.tanh(safe) - 1.0 / safe)
    series = 1.0 - phi**2 / 15.0 + 2.0 * phi**4 / 315.0
    return xp.where(xp.abs(phi) < 1.0e-4, series, regular)


def _radial_geometry(nodes: int, radius_m: float):
    """Equal-volume spherical finite-volume geometry, omitting common 4*pi."""

    if nodes < 2:
        raise ValueError("nodes must be at least 2")
    faces = radius_m * np.cbrt(np.linspace(0.0, 1.0, nodes + 1))
    centers = np.cbrt(0.5 * (faces[:-1] ** 3 + faces[1:] ** 3))
    volumes = (faces[1:] ** 3 - faces[:-1] ** 3) / 3.0
    face_areas = faces**2
    return centers, faces, volumes, face_areas


def solve_first_order_sphere(thiele_modulus: float, nodes: int = 160):
    """Conservative finite-volume solution of the textbook isothermal sphere.

    The surface concentration is one and the center face has zero area, so the
    discrete center condition is exactly zero flux without evaluating ``1/r``.
    Returns ``(effectiveness, radius_centers, concentration, balance_error)``.
    """

    phi = float(thiele_modulus)
    centers, faces, volumes, areas = _radial_geometry(nodes, 1.0)
    matrix = np.zeros((nodes, nodes), dtype=float)
    rhs = np.zeros(nodes, dtype=float)
    for j in range(nodes):
        if j > 0:
            conductance = areas[j] / (centers[j] - centers[j - 1])
            matrix[j, j - 1] += conductance
            matrix[j, j] -= conductance
        if j < nodes - 1:
            conductance = areas[j + 1] / (centers[j + 1] - centers[j])
            matrix[j, j + 1] += conductance
            matrix[j, j] -= conductance
        else:
            conductance = areas[-1] / (faces[-1] - centers[-1])
            matrix[j, j] -= conductance
            rhs[j] -= conductance
        matrix[j, j] -= phi**2 * volumes[j]
    concentration = np.linalg.solve(matrix, rhs)
    rate = phi**2 * float(np.dot(volumes, concentration))
    effectiveness = rate / (phi**2 / 3.0) if phi != 0.0 else 1.0
    surface_flux = (1.0 - concentration[-1]) / (1.0 - centers[-1])
    balance_error = (
        0.0 if phi == 0.0 else abs(surface_flux - rate) / max(abs(rate), 1.0e-15)
    )
    return effectiveness, centers, concentration, balance_error


def _zone_index(nodes: int, zones: int) -> np.ndarray:
    if zones < 1 or zones > nodes:
        raise ValueError("zones must satisfy 1 <= zones <= nodes")
    if nodes % zones != 0:
        raise ValueError(
            "nodes must be divisible by zones to preserve equal-volume activity zones"
        )
    return np.repeat(np.arange(zones), nodes // zones)


def _cell_activity_backend(activity, nodes: int, zones: int, xp):
    index = xp.asarray(_zone_index(nodes, zones), dtype=int)
    return activity[index]


def _effective_diffusivities_backend(config, log_diffusivity_scale, xp):
    """Species diffusivities with the fitted CO2 multiplier applied."""

    selector = xp.asarray([1.0, 0.0, 0.0, 0.0])
    scale = 1.0 + selector * (xp.exp(log_diffusivity_scale) - 1.0)
    return xp.asarray(config.effective_diffusivities_m2_s) * scale


def _state_bounds(config: PelletConfig, nodes: int):
    lower = np.empty((5, nodes), dtype=float)
    upper = np.empty((5, nodes), dtype=float)
    lower[:4] = 1.0e-12
    upper[:4] = 2.0
    lower[4] = (400.0 - config.bulk_temperature_k) / TEMPERATURE_SCALE_K
    upper[4] = (
        config.temperature_limit_k - config.bulk_temperature_k
    ) / TEMPERATURE_SCALE_K
    return lower.ravel(), upper.ravel()


def _initial_state(config: PelletConfig, nodes: int):
    state = np.zeros((5, nodes), dtype=float)
    bulk_scaled = (
        config.bulk_concentrations_mol_m3 / config.bulk_total_concentration_mol_m3
    )
    state[:4] = np.maximum(bulk_scaled[:, None], 1.0e-10)
    state[4] = 0.0
    return state.ravel()


def _inventory_weights(nodes: int, zones: int) -> np.ndarray:
    index = _zone_index(nodes, zones)
    return np.bincount(index, minlength=zones).astype(float) / nodes


def _residual_backend(
    state_scaled,
    activity,
    log_rate_scale,
    log_diffusivity_scale,
    config: PelletConfig,
    nodes: int,
    zones: int,
    xp,
):
    """Scaled conservative cell balances on an arbitrary array backend."""

    state = xp.reshape(state_scaled, (5, nodes))
    concentration_scale = config.bulk_total_concentration_mol_m3
    concentrations = state[:4] * concentration_scale
    temperature = config.bulk_temperature_k + TEMPERATURE_SCALE_K * state[4]
    centers_np, faces_np, volumes_np, areas_np = _radial_geometry(
        nodes, config.radius_m
    )
    centers = xp.asarray(centers_np)
    faces = xp.asarray(faces_np)
    volumes = xp.asarray(volumes_np)
    areas = xp.asarray(areas_np)
    diffusivities = _effective_diffusivities_backend(
        config, log_diffusivity_scale, xp
    )
    mass_transfer = xp.asarray(config.mass_transfer_coefficients_m_s)
    bulk_concentrations = xp.asarray(config.bulk_concentrations_mol_m3)

    partial_pressures = concentrations * GAS_CONSTANT * temperature[None, :] / 1.0e5
    cell_activity = _cell_activity_backend(activity, nodes, zones, xp)
    specific_rate = _kinetic_rate_backend(
        partial_pressures,
        temperature,
        cell_activity * xp.exp(log_rate_scale),
        config.kinetics,
        xp,
    )
    volumetric_rate = specific_rate * config.catalyst_density_g_m3

    species_rows = []
    for i in range(4):
        values = concentrations[i]
        terms = []
        for j in range(nodes):
            balance = xp.asarray(0.0)
            if j > 0:
                conductance = (
                    diffusivities[i] * areas[j] / (centers[j] - centers[j - 1])
                )
                balance = balance + conductance * (values[j - 1] - values[j])
            if j < nodes - 1:
                conductance = (
                    diffusivities[i] * areas[j + 1] / (centers[j + 1] - centers[j])
                )
                balance = balance + conductance * (values[j + 1] - values[j])
            else:
                half_resistance = (faces[-1] - centers[-1]) / diffusivities[i]
                film_resistance = 1.0 / mass_transfer[i]
                overall = 1.0 / (half_resistance + film_resistance)
                balance = balance + areas[-1] * overall * (
                    bulk_concentrations[i] - values[j]
                )
            balance = balance + STOICHIOMETRY[i] * volumetric_rate[j] * volumes[j]
            scale = diffusivities[i] * concentration_scale * config.radius_m / nodes
            terms.append(balance / scale)
        species_rows.append(xp.stack(terms))

    heat_terms = []
    conductivity = config.thermal_conductivity_w_m_k
    for j in range(nodes):
        balance = xp.asarray(0.0)
        if j > 0:
            conductance = conductivity * areas[j] / (centers[j] - centers[j - 1])
            balance = balance + conductance * (temperature[j - 1] - temperature[j])
        if j < nodes - 1:
            conductance = conductivity * areas[j + 1] / (centers[j + 1] - centers[j])
            balance = balance + conductance * (temperature[j + 1] - temperature[j])
        else:
            half_resistance = (faces[-1] - centers[-1]) / conductivity
            film_resistance = 1.0 / config.heat_transfer_coefficient_w_m2_k
            overall = 1.0 / (half_resistance + film_resistance)
            balance = balance + areas[-1] * overall * (
                config.bulk_temperature_k - temperature[j]
            )
        balance = balance + (
            -config.reaction_enthalpy_j_mol * volumetric_rate[j] * volumes[j]
        )
        scale = conductivity * TEMPERATURE_SCALE_K * config.radius_m / nodes
        heat_terms.append(balance / scale)
    return xp.concatenate((*species_rows, xp.stack(heat_terms)))


@lru_cache(maxsize=16)
def _compiled_model(config: PelletConfig, nodes: int, zones: int):
    import jax
    import jax.numpy as jnp

    # POUNCE and this stiff reaction--diffusion model require float64.  JAX
    # otherwise silently truncates the exact derivatives to float32, which is
    # not accurate enough to certify the finite-volume balances.
    jax.config.update("jax_enable_x64", True)

    def residual(state, activity, scenario_parameters):
        return _residual_backend(
            state,
            activity,
            scenario_parameters[0],
            scenario_parameters[1],
            config,
            nodes,
            zones,
            jnp,
        )

    residual_jit = jax.jit(residual)
    jac_state_jit = jax.jit(jax.jacfwd(residual, argnums=0))
    jac_activity_jit = jax.jit(jax.jacfwd(residual, argnums=1))
    jac_scenario_jit = jax.jit(jax.jacfwd(residual, argnums=2))
    return residual_jit, jac_state_jit, jac_activity_jit, jac_scenario_jit


def _activity_from_cells(cell_activity: np.ndarray, nodes: int, zones: int):
    index = _zone_index(nodes, zones)
    return np.array([np.mean(cell_activity[index == k]) for k in range(zones)])


def egg_shell_activity(config: PelletConfig, zones: int | None = None) -> np.ndarray:
    """Bounded outer-first step profile at the configured inventory."""

    zones = config.zones if zones is None else int(zones)
    weights = _inventory_weights(config.nodes, zones)
    remaining = config.activity_inventory
    activity = np.zeros(zones)
    for k in range(zones - 1, -1, -1):
        value = min(config.activity_upper, remaining / weights[k])
        activity[k] = value
        remaining -= value * weights[k]
    if abs(remaining) > 1.0e-12:
        raise ValueError("activity inventory cannot be represented within bounds")
    return activity


def _solution_from_state(
    state_scaled,
    activity,
    config: PelletConfig,
    nodes: int,
    zones: int,
    scenario: Scenario,
    *,
    success: bool,
    message: str,
):
    state = np.asarray(state_scaled, dtype=float).reshape(5, nodes)
    concentrations = state[:4] * config.bulk_total_concentration_mol_m3
    temperature = config.bulk_temperature_k + TEMPERATURE_SCALE_K * state[4]
    centers, faces, volumes, areas = _radial_geometry(nodes, config.radius_m)
    index = _zone_index(nodes, zones)
    cell_activity = np.asarray(activity, dtype=float)[index]
    partial_pressures = concentrations * GAS_CONSTANT * temperature[None, :] / 1.0e5
    specific_rate = _kinetic_rate_backend(
        partial_pressures,
        temperature,
        cell_activity * np.exp(scenario.log_rate_scale),
        config.kinetics,
        np,
    )
    rate = np.asarray(specific_rate) * config.catalyst_density_g_m3
    production = 4.0 * np.pi * float(np.dot(rate, volumes))

    diffusivities = _effective_diffusivities_backend(
        config, scenario.log_diffusivity_scale, np
    )
    bulk = config.bulk_concentrations_mol_m3
    surface_fluxes = np.empty(4)
    for i in range(4):
        overall = 1.0 / (
            (faces[-1] - centers[-1]) / diffusivities[i]
            + 1.0 / config.mass_transfer_coefficients_m_s[i]
        )
        # Positive is transfer from bulk into the pellet.
        surface_fluxes[i] = (
            4.0 * np.pi * areas[-1] * overall * (bulk[i] - concentrations[i, -1])
        )
    source_totals = STOICHIOMETRY * production
    species_denominator = np.maximum(
        np.maximum(np.abs(surface_fluxes), np.abs(source_totals)), 1.0e-15
    )
    species_relative = np.abs(surface_fluxes + source_totals) / species_denominator

    heat_overall = 1.0 / (
        (faces[-1] - centers[-1]) / config.thermal_conductivity_w_m_k
        + 1.0 / config.heat_transfer_coefficient_w_m2_k
    )
    outward_heat = (
        4.0
        * np.pi
        * areas[-1]
        * heat_overall
        * (temperature[-1] - config.bulk_temperature_k)
    )
    generated_heat = -config.reaction_enthalpy_j_mol * production
    energy_relative = abs(outward_heat - generated_heat) / max(
        abs(generated_heat), 1.0e-12
    )

    residual = np.asarray(
        _residual_backend(
            state_scaled,
            activity,
            scenario.log_rate_scale,
            scenario.log_diffusivity_scale,
            config,
            nodes,
            zones,
            np,
        )
    )
    bulk_pressure = np.asarray(config.bulk_mole_fractions) * config.pressure_bar
    bulk_specific = float(
        _kinetic_rate_backend(
            bulk_pressure,
            config.bulk_temperature_k,
            1.0,
            config.kinetics,
            np,
        )
    )
    intrinsic = (
        4.0
        * np.pi
        * bulk_specific
        * config.catalyst_density_g_m3
        * float(np.dot(cell_activity, volumes))
        * np.exp(scenario.log_rate_scale)
    )
    solution = PelletSolution(
        radius_m=centers,
        concentrations_mol_m3=concentrations,
        temperature_k=temperature,
        activity=np.asarray(activity, dtype=float),
        cell_activity=cell_activity,
        production_mol_s=production,
        surface_flux_mol_s=float(surface_fluxes[0]),
        max_scaled_residual=float(np.max(np.abs(residual))),
        species_balance_relative=species_relative,
        energy_balance_relative=float(energy_relative),
        success=bool(success),
        message=str(message),
        state_scaled=np.asarray(state_scaled, dtype=float),
        scenario=scenario,
    )
    solution._intrinsic_production_mol_s = intrinsic
    return solution


def solve_forward(
    activity: Sequence[float],
    config: PelletConfig = PelletConfig(),
    *,
    nodes: int | None = None,
    scenario: Scenario = Scenario(),
    initial_state: np.ndarray | None = None,
) -> PelletSolution:
    """Solve a fixed-activity pellet with an exact JAX residual Jacobian.

    This is the nested route used for independent design checks, perturb-and-
    resolve gradients, mesh refinement, and uncertainty sampling.  The primary
    design route is :func:`solve_design`, which puts these same state balances
    and the activity variables into one simultaneous POUNCE NLP.
    """

    activity = np.asarray(activity, dtype=float)
    zones = activity.size
    nodes = config.nodes if nodes is None else int(nodes)
    if np.any(activity < -1.0e-9) or np.any(activity > config.activity_upper + 1.0e-9):
        raise ValueError("activity is outside configured bounds")
    activity = np.clip(activity, 0.0, config.activity_upper)
    compiled = _compiled_model(config, nodes, zones)
    residual_jit, jac_state_jit = compiled[:2]
    parameters = np.array(
        [scenario.log_rate_scale, scenario.log_diffusivity_scale], dtype=float
    )
    x0 = (
        _initial_state(config, nodes)
        if initial_state is None
        else np.asarray(initial_state, dtype=float)
    )
    lower, upper = _state_bounds(config, nodes)

    def fun(state):
        return np.asarray(residual_jit(state, activity, parameters), dtype=float)

    def jac(state):
        return np.asarray(jac_state_jit(state, activity, parameters), dtype=float)

    fit = least_squares(
        fun,
        np.clip(x0, lower + 1.0e-12, upper - 1.0e-12),
        jac=jac,
        bounds=(lower, upper),
        xtol=1.0e-11,
        ftol=1.0e-11,
        gtol=1.0e-11,
        max_nfev=250,
    )
    max_residual = float(np.max(np.abs(fun(fit.x))))
    success = bool(fit.success and max_residual < 2.0e-7)
    message = f"{fit.message}; max scaled residual={max_residual:.3e}"
    return _solution_from_state(
        fit.x,
        activity,
        config,
        nodes,
        zones,
        scenario,
        success=success,
        message=message,
    )


def refine_solution(
    solution: PelletSolution,
    config: PelletConfig = PelletConfig(),
    *,
    nodes: int,
) -> PelletSolution:
    """Re-solve a fixed design on a new mesh from an interpolated state.

    The activity basis is kept fixed while all species and temperature states
    are independently re-discretized.  This is the refinement route used for
    reported designs and avoids asking a stiff nonisothermal solve to jump
    directly from the bulk-state initial guess to the hot solution branch.
    """

    nodes = int(nodes)
    new_radius, _, _, _ = _radial_geometry(nodes, config.radius_m)
    old_state = solution.state_scaled.reshape(5, solution.radius_m.size)
    seed = np.stack(
        [np.interp(new_radius, solution.radius_m, row) for row in old_state]
    ).ravel()
    return solve_forward(
        solution.activity,
        config,
        nodes=nodes,
        scenario=solution.scenario,
        initial_state=seed,
    )


def _observables_backend(
    state_scaled,
    activity,
    scenario_parameters,
    config,
    nodes,
    zones,
    xp,
):
    state = xp.reshape(state_scaled, (5, nodes))
    concentrations = state[:4] * config.bulk_total_concentration_mol_m3
    temperature = config.bulk_temperature_k + TEMPERATURE_SCALE_K * state[4]
    _, _, volumes_np, _ = _radial_geometry(nodes, config.radius_m)
    volumes = xp.asarray(volumes_np)
    cell_activity = _cell_activity_backend(activity, nodes, zones, xp)
    partial_pressures = concentrations * GAS_CONSTANT * temperature[None, :] / 1.0e5
    rate = (
        _kinetic_rate_backend(
            partial_pressures,
            temperature,
            cell_activity * xp.exp(scenario_parameters[0]),
            config.kinetics,
            xp,
        )
        * config.catalyst_density_g_m3
    )
    production = 4.0 * xp.pi * xp.dot(rate, volumes)
    return xp.stack((production, xp.max(temperature)))


def implicit_observable_jacobian(
    solution: PelletSolution,
    config: PelletConfig = PelletConfig(),
    *,
    with_respect_to: str = "activity",
) -> np.ndarray:
    """Exact fixed-mesh Jacobian of ``(production, max_temperature)``.

    The finite-volume state satisfies ``R(s, q)=0``.  This routine applies
    ``ds/dq = -R_s^-1 R_q`` using JAX derivatives of the same residual used by
    the solver, then differentiates the dimensional observables through that
    state.  ``with_respect_to`` is ``"activity"`` or ``"scenario"`` (the two
    log-scale uncertainty parameters).
    """

    import jax
    import jax.numpy as jnp

    nodes = solution.radius_m.size
    zones = solution.activity.size
    residual_jit, jac_state_jit, jac_activity_jit, jac_scenario_jit = _compiled_model(
        config, nodes, zones
    )
    state = jnp.asarray(solution.state_scaled)
    activity = jnp.asarray(solution.activity)
    parameters = jnp.asarray(
        [
            solution.scenario.log_rate_scale,
            solution.scenario.log_diffusivity_scale,
        ]
    )
    jac_state = np.asarray(jac_state_jit(state, activity, parameters))
    if with_respect_to == "activity":
        jac_parameter = np.asarray(jac_activity_jit(state, activity, parameters))
        argument = 1
    elif with_respect_to == "scenario":
        jac_parameter = np.asarray(jac_scenario_jit(state, activity, parameters))
        argument = 2
    else:
        raise ValueError("with_respect_to must be 'activity' or 'scenario'")
    state_sensitivity = -np.linalg.solve(jac_state, jac_parameter)

    def observables(s, a, p):
        return _observables_backend(s, a, p, config, nodes, zones, jnp)

    obs_state = np.asarray(
        jax.jacfwd(observables, argnums=0)(state, activity, parameters)
    )
    obs_direct = np.asarray(
        jax.jacfwd(observables, argnums=argument)(state, activity, parameters)
    )
    return obs_direct + obs_state @ state_sensitivity


def _full_dense_pattern(rows: int, columns: int):
    row = np.repeat(np.arange(rows, dtype=np.int64), columns)
    col = np.tile(np.arange(columns, dtype=np.int64), rows)
    return row, col


def solve_design(
    config: PelletConfig = PelletConfig(),
    *,
    nodes: int | None = None,
    zones: int | None = None,
    initial_activity: Sequence[float] | None = None,
    scenarios: Sequence[Scenario] = (Scenario(),),
    robust: bool = False,
    options: dict | None = None,
) -> DesignResult:
    """Simultaneously optimize states and a bounded activity profile.

    All scenarios share one activity profile and independently satisfy the
    four species balances, energy balance, concentration bounds, and thermal
    limit.  The objective maximizes mean production and penalizes adjacent
    activity jumps.  With ``robust=True``, an epigraph variable maximizes the
    worst-case production over the supplied scenarios; otherwise the mean
    scenario production is maximized.
    """

    import jax.numpy as jnp

    from pounce.jax import from_jax

    nodes = config.nodes if nodes is None else int(nodes)
    zones = config.zones if zones is None else int(zones)
    scenarios = tuple(scenarios)
    if not scenarios:
        raise ValueError("at least one scenario is required")
    weights = _inventory_weights(nodes, zones)
    if initial_activity is None:
        activity0 = np.full(zones, config.activity_inventory)
    else:
        activity0 = np.asarray(initial_activity, dtype=float)
        if activity0.shape != (zones,):
            raise ValueError(f"initial_activity must have shape ({zones},)")
    # Project the seed onto the inventory hyperplane without changing its
    # qualitative shape, then keep it strictly inside the activity box.
    activity0 = activity0 + (
        config.activity_inventory - float(np.dot(weights, activity0))
    )
    activity0 = np.clip(activity0, 1.0e-6, config.activity_upper - 1.0e-6)

    state_seeds = []
    previous = None
    for scenario in scenarios:
        forward = solve_forward(
            activity0,
            config,
            nodes=nodes,
            scenario=scenario,
            initial_state=previous,
        )
        if not forward.success:
            raise RuntimeError(
                f"design seed failed for {scenario.label}: {forward.message}"
            )
        state_seeds.append(forward.state_scaled)
        previous = forward.state_scaled
    state_size = 5 * nodes
    scenario_parameters = tuple(
        jnp.asarray([s.log_rate_scale, s.log_diffusivity_scale]) for s in scenarios
    )
    seed_productions = [
        _solution_from_state(
            state,
            activity0,
            config,
            nodes,
            zones,
            scenario,
            success=True,
            message="design seed",
        ).production_mol_s
        for state, scenario in zip(state_seeds, scenarios)
    ]
    uniform_activity = np.full(zones, config.activity_inventory)
    if np.allclose(activity0, uniform_activity, rtol=0.0, atol=1.0e-14):
        reference_productions = seed_productions
    else:
        reference_productions = []
        for scenario in scenarios:
            uniform = solve_forward(
                uniform_activity,
                config,
                nodes=nodes,
                scenario=scenario,
            )
            if not uniform.success:
                raise RuntimeError(
                    f"uniform reference failed for {scenario.label}: {uniform.message}"
                )
            reference_productions.append(uniform.production_mol_s)
    production_reference = max(float(np.mean(reference_productions)), 1.0e-12)

    def split(x):
        states = [
            x[k * state_size : (k + 1) * state_size] for k in range(len(scenarios))
        ]
        activity_start = len(scenarios) * state_size
        activity = x[activity_start : activity_start + zones]
        guarantee = x[-1] if robust else None
        return states, activity, guarantee

    def objective(x):
        states, activity, guarantee = split(x)
        production = []
        for state, parameter in zip(states, scenario_parameters):
            production.append(
                _observables_backend(
                    state,
                    activity,
                    parameter,
                    config,
                    nodes,
                    zones,
                    jnp,
                )[0]
            )
        roughness = jnp.sum((activity[1:] - activity[:-1]) ** 2)
        performance = (
            guarantee
            if robust
            else jnp.mean(jnp.stack(production)) / production_reference
        )
        return -performance + config.regularization_weight * roughness

    def constraints(x):
        states, activity, guarantee = split(x)
        residuals = []
        production = []
        for state, parameter in zip(states, scenario_parameters):
            residuals.append(
                _residual_backend(
                    state,
                    activity,
                    parameter[0],
                    parameter[1],
                    config,
                    nodes,
                    zones,
                    jnp,
                )
            )
            if robust:
                production.append(
                    _observables_backend(
                        state,
                        activity,
                        parameter,
                        config,
                        nodes,
                        zones,
                        jnp,
                    )[0]
                )
        inventory = jnp.dot(jnp.asarray(weights), activity) - config.activity_inventory
        values = [*residuals, jnp.asarray([inventory])]
        if robust:
            values.append(jnp.stack(production) / production_reference - guarantee)
        return jnp.concatenate(values)

    x0_parts = [*state_seeds, activity0]
    if robust:
        x0_parts.append(
            np.asarray([0.99 * min(seed_productions) / production_reference])
        )
    x0 = np.concatenate(x0_parts)
    state_lower, state_upper = _state_bounds(config, nodes)
    lb_parts = [*([state_lower] * len(scenarios)), np.zeros(zones)]
    ub_parts = [
        *([state_upper] * len(scenarios)),
        np.full(zones, config.activity_upper),
    ]
    if robust:
        lb_parts.append(np.asarray([0.0]))
        ub_parts.append(np.asarray([10.0]))
    lb = np.concatenate(lb_parts)
    ub = np.concatenate(ub_parts)
    n_variables = x0.size
    equality_constraints = len(scenarios) * state_size + 1
    n_constraints = equality_constraints + (len(scenarios) if robust else 0)
    cl = np.zeros(n_constraints)
    cu = np.zeros(n_constraints)
    if robust:
        cu[equality_constraints:] = np.inf
    jac_pattern = _full_dense_pattern(n_constraints, n_variables)
    hess_pattern = np.tril_indices(n_variables)
    problem = from_jax(
        objective,
        constraints,
        n=n_variables,
        m=n_constraints,
        lb=lb,
        ub=ub,
        cl=cl,
        cu=cu,
        jac_pattern=jac_pattern,
        hess_pattern=hess_pattern,
    )
    solver_options = {
        "print_level": 0,
        "sb": "yes",
        "tol": 1.0e-7,
        "acceptable_tol": 1.0e-6,
        "max_iter": 500,
        "hessian_approximation": "limited-memory",
        "bound_relax_factor": 0.0,
    }
    if options:
        solver_options.update(options)
    for name, value in solver_options.items():
        problem.add_option(name, value)
    x, info = problem.solve(x0=x0)
    x = np.asarray(x, dtype=float)
    states, activity, guarantee = split(x)
    activity = np.asarray(activity, dtype=float)
    status = str(info.get("status_msg", "unknown"))
    success = status in {"Solve_Succeeded", "Solved_To_Acceptable_Level"}
    solutions = tuple(
        _solution_from_state(
            np.asarray(state),
            activity,
            config,
            nodes,
            zones,
            scenario,
            success=success,
            message=f"simultaneous POUNCE solve: {info.get('status_msg', '')}",
        )
        for state, scenario in zip(states, scenarios)
    )
    constraint_values = np.asarray(constraints(jnp.asarray(x)), dtype=float)
    equality_violation = float(np.max(np.abs(constraint_values[:equality_constraints])))
    inequality_violation = (
        float(np.max(np.maximum(-constraint_values[equality_constraints:], 0.0)))
        if robust
        else 0.0
    )
    return DesignResult(
        activity=activity,
        solutions=solutions,
        objective=float(info.get("obj_val", objective(jnp.asarray(x)))),
        success=success,
        status=status,
        iterations=int(info.get("iter_count", info.get("iterations", -1))),
        max_constraint_violation=max(equality_violation, inequality_violation),
        guaranteed_production_mol_s=(
            float(guarantee) * production_reference if robust else None
        ),
        raw_info=info,
    )


def solve_nested_design(
    config: PelletConfig = PelletConfig(),
    *,
    nodes: int | None = None,
    zones: int | None = None,
    initial_activity: Sequence[float] | None = None,
    options: dict | None = None,
) -> DesignResult:
    """Optimize activity outside the fixed-mesh state solve.

    This independent prototype uses SLSQP for the small outer problem and the
    exact implicit derivative from :func:`implicit_observable_jacobian`.  It is
    retained as a cross-check; the simultaneous POUNCE transcription remains
    the primary route because it exposes every balance and bound directly.
    """

    from scipy.optimize import minimize

    nodes = config.nodes if nodes is None else int(nodes)
    zones = config.zones if zones is None else int(zones)
    weights = _inventory_weights(nodes, zones)
    if initial_activity is None:
        activity0 = np.full(zones, config.activity_inventory)
    else:
        activity0 = np.asarray(initial_activity, dtype=float)
        if activity0.shape != (zones,):
            raise ValueError(f"initial_activity must have shape ({zones},)")
    activity0 = activity0 + (
        config.activity_inventory - float(np.dot(weights, activity0))
    )
    activity0 = np.clip(activity0, 0.0, config.activity_upper)
    reference = solve_forward(
        np.full(zones, config.activity_inventory), config, nodes=nodes
    )
    if not reference.success:
        raise RuntimeError(reference.message)
    production_reference = max(reference.production_mol_s, 1.0e-12)
    cached_activity = None
    cached_solution = None
    cached_jacobian = None

    def evaluate(activity):
        nonlocal cached_activity, cached_solution, cached_jacobian
        candidate = np.asarray(activity, dtype=float)
        if cached_activity is None or not np.array_equal(candidate, cached_activity):
            seed = None if cached_solution is None else cached_solution.state_scaled
            solution = solve_forward(candidate, config, nodes=nodes, initial_state=seed)
            if not solution.success:
                raise RuntimeError(solution.message)
            cached_activity = candidate.copy()
            cached_solution = solution
            cached_jacobian = implicit_observable_jacobian(solution, config)
        return cached_solution, cached_jacobian

    def objective(activity):
        solution, _ = evaluate(activity)
        roughness = float(np.sum(np.diff(activity) ** 2))
        return -solution.production_mol_s / production_reference + (
            config.regularization_weight * roughness
        )

    def objective_jacobian(activity):
        _, jacobian = evaluate(activity)
        gradient = -jacobian[0] / production_reference
        difference = np.diff(activity)
        gradient[:-1] -= 2.0 * config.regularization_weight * difference
        gradient[1:] += 2.0 * config.regularization_weight * difference
        return gradient

    def thermal_margin(activity):
        solution, _ = evaluate(activity)
        return config.temperature_limit_k - solution.max_temperature_k

    def thermal_jacobian(activity):
        _, jacobian = evaluate(activity)
        return -jacobian[1]

    solver_options = {"ftol": 1.0e-10, "maxiter": 100, "disp": False}
    if options:
        solver_options.update(options)
    fit = minimize(
        objective,
        activity0,
        method="SLSQP",
        jac=objective_jacobian,
        bounds=[(0.0, config.activity_upper)] * zones,
        constraints=(
            {
                "type": "eq",
                "fun": lambda activity: (
                    float(np.dot(weights, activity)) - config.activity_inventory
                ),
                "jac": lambda activity: weights,
            },
            {"type": "ineq", "fun": thermal_margin, "jac": thermal_jacobian},
        ),
        options=solver_options,
    )
    solution, _ = evaluate(fit.x)
    violation = max(
        abs(float(np.dot(weights, fit.x)) - config.activity_inventory),
        max(solution.max_temperature_k - config.temperature_limit_k, 0.0),
        max(-float(np.min(fit.x)), 0.0),
        max(float(np.max(fit.x)) - config.activity_upper, 0.0),
    )
    return DesignResult(
        activity=np.asarray(fit.x),
        solutions=(solution,),
        objective=float(fit.fun),
        success=bool(fit.success and solution.success and violation < 1.0e-7),
        status=f"nested SLSQP: {fit.message}",
        iterations=int(fit.nit),
        max_constraint_violation=float(violation),
        guaranteed_production_mol_s=None,
        raw_info={"method": "nested-slsqp", "optimize_result": fit},
    )


def _calibration_model_backend(conditions, parameters, config, xp):
    conditions = xp.asarray(conditions)
    kind = conditions[:, 0]
    temperature = conditions[:, 1]
    radius = conditions[:, 2]
    log_rate_scale, log_diffusivity_scale = parameters
    pressure = xp.asarray(config.bulk_mole_fractions)[:, None] * config.pressure_bar
    pressure = xp.broadcast_to(pressure, (4, temperature.size))
    intrinsic_specific = _kinetic_rate_backend(
        pressure,
        temperature,
        xp.exp(log_rate_scale),
        config.kinetics,
        xp,
    )
    bulk_co2 = (
        config.bulk_mole_fractions[0]
        * config.pressure_bar
        * 1.0e5
        / (GAS_CONSTANT * temperature)
    )
    first_order_rate = intrinsic_specific * config.catalyst_density_g_m3 / bulk_co2
    diffusivity = config.effective_diffusivities_m2_s[0] * xp.exp(log_diffusivity_scale)
    phi = radius * xp.sqrt(xp.maximum(first_order_rate / diffusivity, 0.0))
    eta = _analytical_effectiveness_backend(phi, xp)
    apparent = intrinsic_specific * xp.where(kind < 0.5, 1.0, eta)
    return xp.log(xp.maximum(apparent, 1.0e-30))


def make_calibration_data(
    config: PelletConfig = PelletConfig(),
    *,
    seed: int = 37,
    noise_standard_deviation: float = 0.012,
) -> CalibrationData:
    """Create a deterministic synthetic intrinsic/pellet calibration set.

    Intrinsic measurements identify the activity multiplier; apparent-rate
    measurements at two radii also identify effective diffusivity.  Noise is
    Gaussian in log-rate space, so ``sigma`` is a relative-error scale.
    """

    temperatures = np.array([500.0, 530.0, 555.0, 580.0, 605.0])
    intrinsic = np.column_stack(
        (np.zeros_like(temperatures), temperatures, np.zeros_like(temperatures))
    )
    pellet_small = np.column_stack(
        (
            np.ones_like(temperatures),
            temperatures,
            np.full_like(temperatures, 0.7e-3),
        )
    )
    pellet_large = np.column_stack(
        (
            np.ones_like(temperatures),
            temperatures,
            np.full_like(temperatures, config.radius_m),
        )
    )
    conditions = np.vstack((intrinsic, pellet_small, pellet_large))
    truth = np.array([0.0, 0.0])
    clean = np.asarray(_calibration_model_backend(conditions, truth, config, np))
    rng = np.random.default_rng(seed)
    sigma = np.full(clean.shape, noise_standard_deviation)
    observed = clean + sigma * rng.standard_normal(clean.size)
    return CalibrationData(conditions, observed, sigma, truth)


def fit_effective_parameters(
    data: CalibrationData | None = None,
    config: PelletConfig = PelletConfig(),
):
    """Fit log rate/diffusivity scales and return POUNCE covariance output."""

    import jax
    import jax.numpy as jnp

    from .._curve_fit import curve_fit

    data = make_calibration_data(config) if data is None else data

    def model(conditions, log_rate_scale, log_diffusivity_scale):
        parameters = np.array([log_rate_scale, log_diffusivity_scale])
        return np.asarray(
            _calibration_model_backend(conditions, parameters, config, np)
        )

    def jacobian(conditions, log_rate_scale, log_diffusivity_scale):
        conditions_jax = jnp.asarray(conditions)
        parameters = jnp.asarray([log_rate_scale, log_diffusivity_scale])
        derivative = jax.jacfwd(
            lambda p: _calibration_model_backend(conditions_jax, p, config, jnp)
        )(parameters)
        return np.asarray(derivative)

    return curve_fit(
        model,
        data.conditions,
        data.log_rates,
        p0=np.array([0.0, 0.0]),
        sigma=data.sigma,
        absolute_sigma=True,
        bounds=([-1.0, -1.0], [1.0, 1.0]),
        jac=jacobian,
        options={"print_level": 0, "tol": 1.0e-9},
    )


def uncertainty_scenarios(
    mean: Sequence[float],
    covariance: np.ndarray,
    *,
    standard_deviations: float = 1.645,
) -> tuple[Scenario, ...]:
    """Nominal plus both directions of each covariance principal axis."""

    mean = np.asarray(mean, dtype=float)
    covariance = np.asarray(covariance, dtype=float)
    if mean.shape != (2,) or covariance.shape != (2, 2):
        raise ValueError("mean and covariance must describe two parameters")
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    scenarios = [Scenario(mean[0], mean[1], "nominal")]
    for index in range(2):
        displacement = (
            standard_deviations
            * np.sqrt(max(eigenvalues[index], 0.0))
            * eigenvectors[:, index]
        )
        for sign, suffix in ((1.0, "+"), (-1.0, "-")):
            point = mean + sign * displacement
            scenarios.append(
                Scenario(point[0], point[1], f"principal-{index + 1}{suffix}")
            )
    return tuple(scenarios)


def validate_uncertainty(
    activity: Sequence[float],
    mean: Sequence[float],
    covariance: np.ndarray,
    config: PelletConfig = PelletConfig(),
    *,
    nodes: int | None = None,
    samples: int = 24,
    seed: int = 103,
) -> UncertaintyValidation:
    """Compare delta-method predictions with sampled full forward re-solves."""

    mean = np.asarray(mean, dtype=float)
    covariance = np.asarray(covariance, dtype=float)
    nominal_scenario = Scenario(mean[0], mean[1], "fitted nominal")
    nominal_solution = solve_forward(
        activity, config, nodes=nodes, scenario=nominal_scenario
    )
    if not nominal_solution.success:
        raise RuntimeError(nominal_solution.message)
    jacobian = implicit_observable_jacobian(
        nominal_solution, config, with_respect_to="scenario"
    )
    delta_covariance = jacobian @ covariance @ jacobian.T
    delta_standard_deviation = np.sqrt(np.clip(np.diag(delta_covariance), 0.0, None))

    rng = np.random.default_rng(seed)
    parameter_samples = rng.multivariate_normal(mean, covariance, size=samples)
    observed = []
    previous = nominal_solution.state_scaled
    for index, parameter in enumerate(parameter_samples):
        scenario = Scenario(parameter[0], parameter[1], f"sample-{index}")
        solution = solve_forward(
            activity,
            config,
            nodes=nodes,
            scenario=scenario,
            initial_state=previous,
        )
        if not solution.success:
            raise RuntimeError(f"uncertainty sample {index} failed: {solution.message}")
        observed.append([solution.production_mol_s, solution.max_temperature_k])
        previous = solution.state_scaled
    observed = np.asarray(observed)
    nominal = np.array(
        [nominal_solution.production_mol_s, nominal_solution.max_temperature_k]
    )
    return UncertaintyValidation(
        observable_names=("production_mol_s", "max_temperature_k"),
        nominal=nominal,
        delta_standard_deviation=delta_standard_deviation,
        sampled_standard_deviation=np.std(observed, axis=0, ddof=1),
        sampled_minimum=np.min(observed, axis=0),
        sampled_maximum=np.max(observed, axis=0),
        samples=observed,
    )
