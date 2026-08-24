"""Differentiable Peng--Robinson phase-envelope example.

This module contains the inspectable thermodynamic model used by notebook 34.
It deliberately stops short of being a property package: the scope is a
two-phase Peng--Robinson isopleth with classical one-fluid mixing rules.  The
model exists to demonstrate pseudo-arclength continuation, augmented fold
refinement, implicit sensitivities, and inverse design with POUNCE.

All temperatures are in K, pressures in Pa, and compositions, acentric
factors, equilibrium ratios, and binary interaction parameters are
dimensionless unless a docstring says otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import brentq

from pounce.jax import JaxProblem, PathFollower, PathTrace

R = 8.314462618  # [J mol^-1 K^-1]
Mode = Literal["pressure", "temperature"]


@dataclass(frozen=True)
class PengRobinsonMixture:
    """Data for one Peng--Robinson mixture.

    Parameters
    ----------
    names
        Component names, in the same order as every array.
    critical_temperature
        Pure-component critical temperatures [K].
    critical_pressure
        Pure-component critical pressures [Pa].
    acentric_factor
        Pure-component acentric factors [-].
    composition
        Composition of the phase held fixed by the isopleth [-].  ``beta=1``
        holds the vapor composition and ``beta=0`` the liquid composition.
    binary_interaction
        Symmetric Peng--Robinson ``k_ij`` matrix [-].
    source
        Human-readable provenance for the constants and composition.
    """

    names: tuple[str, ...]
    critical_temperature: np.ndarray
    critical_pressure: np.ndarray
    acentric_factor: np.ndarray
    composition: np.ndarray
    binary_interaction: np.ndarray
    source: str = ""

    def __post_init__(self) -> None:
        n = len(self.names)
        tc = np.asarray(self.critical_temperature, dtype=float)
        pc = np.asarray(self.critical_pressure, dtype=float)
        omega = np.asarray(self.acentric_factor, dtype=float)
        z = np.asarray(self.composition, dtype=float)
        kij = np.asarray(self.binary_interaction, dtype=float)
        for label, array in (
            ("critical_temperature", tc),
            ("critical_pressure", pc),
            ("acentric_factor", omega),
            ("composition", z),
        ):
            if array.shape != (n,):
                raise ValueError(f"{label} must have shape ({n},), got {array.shape}")
        if kij.shape != (n, n):
            raise ValueError(
                f"binary_interaction must have shape ({n}, {n}), got {kij.shape}"
            )
        if np.any(tc <= 0.0) or np.any(pc <= 0.0):
            raise ValueError("critical temperatures and pressures must be positive")
        if np.any(z <= 0.0) or not np.isclose(z.sum(), 1.0, atol=1e-12):
            raise ValueError("composition must be strictly positive and sum to one")
        if not np.allclose(kij, kij.T, atol=1e-14):
            raise ValueError("binary_interaction must be symmetric")
        if not np.allclose(np.diag(kij), 0.0, atol=1e-14):
            raise ValueError("binary_interaction diagonal must be zero")
        object.__setattr__(self, "critical_temperature", tc)
        object.__setattr__(self, "critical_pressure", pc)
        object.__setattr__(self, "acentric_factor", omega)
        object.__setattr__(self, "composition", z)
        object.__setattr__(self, "binary_interaction", kij)

    @property
    def n_components(self) -> int:
        """Number of mixture components [-]."""

        return len(self.names)

    def with_composition(self, composition) -> "PengRobinsonMixture":
        """Return a copy with a new normalized composition [-]."""

        return replace(self, composition=np.asarray(composition, dtype=float))


def _kij_matrix(n: int, pair: tuple[int, int], value) -> jax.Array:
    """Return an otherwise-zero symmetric ``k_ij`` matrix [-]."""

    i, j = pair
    if i == j or not (0 <= i < n and 0 <= j < n):
        raise ValueError(f"kij_pair must name two distinct indices in [0, {n})")
    out = jnp.zeros((n, n), dtype=jnp.float64)
    return out.at[i, j].set(value).at[j, i].set(value)


def simplex_coordinates(composition) -> np.ndarray:
    """Map a positive composition to ``N-1`` log-ratio coordinates [-].

    The last component is the reference, so ``eta_i = log(z_i/z_N)``.
    This is an unconstrained coordinate system on the interior of the
    composition simplex.
    """

    z = np.asarray(composition, dtype=float)
    if z.ndim != 1 or z.size < 2 or np.any(z <= 0.0):
        raise ValueError("composition must be a positive one-dimensional vector")
    z = z / z.sum()
    return np.log(z[:-1] / z[-1])


def composition_from_coordinates(eta) -> jax.Array:
    """Map ``N-1`` log-ratio coordinates to a composition simplex [-]."""

    logits = jnp.concatenate(
        [jnp.asarray(eta, dtype=jnp.float64), jnp.zeros(1, dtype=jnp.float64)]
    )
    return jax.nn.softmax(logits)


def design_parameters(
    mixture: PengRobinsonMixture, kij_pair: tuple[int, int]
) -> np.ndarray:
    """Return ``[eta_1, ..., eta_(N-1), k_ij]`` for a mixture [-]."""

    i, j = kij_pair
    return np.concatenate(
        [
            simplex_coordinates(mixture.composition),
            np.array([mixture.binary_interaction[i, j]], dtype=float),
        ]
    )


def mixture_from_design(
    mixture: PengRobinsonMixture,
    parameters,
    kij_pair: tuple[int, int] | None,
) -> PengRobinsonMixture:
    """Return a concrete mixture represented by differentiable design [-]."""

    composition, kij = _decode_design(mixture, parameters, kij_pair)
    return replace(
        mixture,
        composition=np.asarray(composition, dtype=float),
        binary_interaction=np.asarray(kij, dtype=float),
    )


def _decode_design(
    mixture: PengRobinsonMixture,
    parameters,
    kij_pair: tuple[int, int] | None,
) -> tuple[jax.Array, jax.Array]:
    """Decode a differentiable composition and ``k_ij`` matrix [-]."""

    n = mixture.n_components
    if parameters is None:
        return (
            jnp.asarray(mixture.composition, dtype=jnp.float64),
            jnp.asarray(mixture.binary_interaction, dtype=jnp.float64),
        )
    q = jnp.asarray(parameters, dtype=jnp.float64)
    expected = n - 1 + (1 if kij_pair is not None else 0)
    if q.shape != (expected,):
        raise ValueError(f"design parameter vector must have shape ({expected},)")
    z = composition_from_coordinates(q[: n - 1])
    if kij_pair is None:
        kij = jnp.asarray(mixture.binary_interaction, dtype=jnp.float64)
    else:
        base = jnp.asarray(mixture.binary_interaction, dtype=jnp.float64)
        i, j = kij_pair
        kij = base.at[i, j].set(q[-1]).at[j, i].set(q[-1])
    return z, kij


def cubic_real_root(c2, c1, c0, *, largest: bool):
    """Return the largest or smallest real root of a monic cubic.

    The inactive Cardano branch is sanitized before ``sqrt``/``arccos`` so
    JAX derivatives remain finite in both the one-real-root and three-real-root
    regions.
    """

    p = c1 - c2**2 / 3.0
    q = 2.0 * c2**3 / 27.0 - c2 * c1 / 3.0 + c0
    discriminant = (q / 2.0) ** 2 + (p / 3.0) ** 3
    three = discriminant <= 0.0

    d_safe = jnp.where(three, 1.0, discriminant)
    sqrt_d = jnp.sqrt(d_safe)
    one = jnp.cbrt(-q / 2.0 + sqrt_d) + jnp.cbrt(-q / 2.0 - sqrt_d)

    p_safe = jnp.where(three, jnp.minimum(p, -1e-300), -1.0)
    radius = 2.0 * jnp.sqrt(-p_safe / 3.0)
    argument = jnp.clip((3.0 * q) / (2.0 * p_safe) * jnp.sqrt(-3.0 / p_safe), -1.0, 1.0)
    phi = jnp.arccos(argument) / 3.0
    roots = jnp.stack(
        [
            radius * jnp.cos(phi),
            radius * jnp.cos(phi - 2.0 * jnp.pi / 3.0),
            radius * jnp.cos(phi - 4.0 * jnp.pi / 3.0),
        ]
    )
    three_root = jnp.max(roots) if largest else jnp.min(roots)
    return jnp.where(three, three_root, one) - c2 / 3.0


def _component_ab(mixture: PengRobinsonMixture, temperature_k):
    """Return pure-component Peng--Robinson ``a_i`` and ``b_i`` [SI]."""

    tc = jnp.asarray(mixture.critical_temperature, dtype=jnp.float64)  # [K]
    pc = jnp.asarray(mixture.critical_pressure, dtype=jnp.float64)  # [Pa]
    omega = jnp.asarray(mixture.acentric_factor, dtype=jnp.float64)  # [-]
    kappa = 0.37464 + 1.54226 * omega - 0.26992 * omega**2  # [-]
    alpha = (1.0 + kappa * (1.0 - jnp.sqrt(temperature_k / tc))) ** 2  # [-]
    ai = 0.45724 * R**2 * tc**2 * alpha / pc  # [Pa m^6 mol^-2]
    bi = 0.07780 * R * tc / pc  # [m^3 mol^-1]
    return ai, bi


def compressibility(
    composition,
    temperature_k,
    pressure_pa,
    mixture: PengRobinsonMixture,
    *,
    largest: bool,
    binary_interaction=None,
):
    """Peng--Robinson compressibility factor ``Z`` [-]."""

    w = jnp.asarray(composition, dtype=jnp.float64)  # [-]
    kij = (
        jnp.asarray(mixture.binary_interaction, dtype=jnp.float64)
        if binary_interaction is None
        else jnp.asarray(binary_interaction, dtype=jnp.float64)
    )
    ai, bi = _component_ab(mixture, temperature_k)
    aij = jnp.sqrt(jnp.outer(ai, ai)) * (1.0 - kij)  # [Pa m^6 mol^-2]
    a_mix = w @ aij @ w  # [Pa m^6 mol^-2]
    b_mix = w @ bi  # [m^3 mol^-1]
    a_red = a_mix * pressure_pa / (R * temperature_k) ** 2  # [-]
    b_red = b_mix * pressure_pa / (R * temperature_k)  # [-]
    return cubic_real_root(
        -(1.0 - b_red),
        a_red - 2.0 * b_red - 3.0 * b_red**2,
        -(a_red * b_red - b_red**2 - b_red**3),
        largest=largest,
    )


def log_fugacity_coefficients(
    composition,
    temperature_k,
    pressure_pa,
    mixture: PengRobinsonMixture,
    *,
    largest: bool,
    binary_interaction=None,
):
    """Peng--Robinson log fugacity coefficients ``ln(phi_i)`` [-]."""

    w = jnp.asarray(composition, dtype=jnp.float64)  # [-]
    kij = (
        jnp.asarray(mixture.binary_interaction, dtype=jnp.float64)
        if binary_interaction is None
        else jnp.asarray(binary_interaction, dtype=jnp.float64)
    )
    ai, bi = _component_ab(mixture, temperature_k)
    aij = jnp.sqrt(jnp.outer(ai, ai)) * (1.0 - kij)  # [Pa m^6 mol^-2]
    a_mix = w @ aij @ w  # [Pa m^6 mol^-2]
    b_mix = w @ bi  # [m^3 mol^-1]
    a_red = a_mix * pressure_pa / (R * temperature_k) ** 2  # [-]
    b_red = b_mix * pressure_pa / (R * temperature_k)  # [-]
    z_factor = cubic_real_root(
        -(1.0 - b_red),
        a_red - 2.0 * b_red - 3.0 * b_red**2,
        -(a_red * b_red - b_red**2 - b_red**3),
        largest=largest,
    )
    sqrt_two = jnp.sqrt(2.0)
    return (
        (bi / b_mix) * (z_factor - 1.0)
        - jnp.log(z_factor - b_red)
        - a_red
        / (2.0 * sqrt_two * b_red)
        * (2.0 * (aij @ w) / a_mix - bi / b_mix)
        * jnp.log(
            (z_factor + (1.0 + sqrt_two) * b_red)
            / (z_factor + (1.0 - sqrt_two) * b_red)
        )
    )


def envelope_residual(
    state,
    log_pressure,
    beta: float,
    mixture: PengRobinsonMixture,
    *,
    design=None,
    kij_pair: tuple[int, int] | None = None,
):
    """Michelsen isopleth residual ``F(ln K, ln T; ln P)`` [-].

    ``state = [ln K_1, ..., ln K_N, ln T]`` and ``log_pressure = ln(P/Pa)``.
    ``beta=1`` fixes the vapor composition (dew isopleth at the start), while
    ``beta=0`` fixes the liquid composition (bubble isopleth at the start).
    """

    n = mixture.n_components
    log_k = state[:n]  # [-]
    temperature_k = jnp.exp(state[n])  # [K]
    pressure_pa = jnp.exp(log_pressure)  # [Pa]
    k_values = jnp.exp(log_k)  # [-]
    fixed_composition, kij = _decode_design(mixture, design, kij_pair)
    liquid = fixed_composition / (1.0 - beta + beta * k_values)  # [-]
    vapor = k_values * liquid  # [-]
    equilibrium = (
        log_k
        + log_fugacity_coefficients(
            vapor,
            temperature_k,
            pressure_pa,
            mixture,
            largest=True,
            binary_interaction=kij,
        )
        - log_fugacity_coefficients(
            liquid,
            temperature_k,
            pressure_pa,
            mixture,
            largest=False,
            binary_interaction=kij,
        )
    )
    return jnp.concatenate([equilibrium, jnp.asarray([jnp.sum(vapor - liquid)])])


def wilson_log_k(temperature_k, pressure_pa, mixture: PengRobinsonMixture) -> jax.Array:
    """Wilson equilibrium-ratio estimate ``ln K_i`` [-]."""

    tc = jnp.asarray(mixture.critical_temperature, dtype=jnp.float64)  # [K]
    pc = jnp.asarray(mixture.critical_pressure, dtype=jnp.float64)  # [Pa]
    omega = jnp.asarray(mixture.acentric_factor, dtype=jnp.float64)  # [-]
    return jnp.log(pc / pressure_pa) + 5.373 * (1.0 + omega) * (
        1.0 - tc / temperature_k
    )


def wilson_start(
    pressure_pa: float,
    beta: float,
    mixture: PengRobinsonMixture,
    *,
    composition=None,
) -> tuple[jax.Array, float]:
    """Return a Wilson seed ``[ln K, ln T]`` at pressure [Pa]."""

    z = np.asarray(
        mixture.composition if composition is None else composition, dtype=float
    )  # [-]

    def closure(temperature_k: float) -> float:
        k_values = np.exp(
            np.asarray(wilson_log_k(temperature_k, pressure_pa, mixture))
        )  # [-]
        if beta == 1.0:
            return float(np.sum(z / k_values) - 1.0)
        if beta == 0.0:
            return float(np.sum(z * k_values) - 1.0)
        raise ValueError("wilson_start currently supports beta=0 or beta=1")

    temperature_k = brentq(closure, 80.0, 800.0)  # [K]
    state = jnp.concatenate(
        [
            wilson_log_k(temperature_k, pressure_pa, mixture),
            jnp.asarray([jnp.log(temperature_k)]),
        ]
    )
    return state, float(temperature_k)


def make_envelope_problem(
    mixture: PengRobinsonMixture,
    beta: float,
    *,
    mode: Mode = "pressure",
    design=None,
    kij_pair: tuple[int, int] | None = None,
) -> JaxProblem:
    """Build the square feasibility NLP for one envelope parameterization."""

    n = mixture.n_components
    if mode == "pressure":

        def residual(state, parameter):
            return envelope_residual(
                state,
                jnp.reshape(parameter, ()),
                beta,
                mixture,
                design=design,
                kij_pair=kij_pair,
            )

    elif mode == "temperature":

        def residual(state, parameter):
            log_temperature = jnp.reshape(parameter, ())  # [ln K]
            pressure_state = state[n]  # [ln Pa]
            pressure_parameterized_state = jnp.concatenate(
                [state[:n], jnp.asarray([log_temperature])]
            )
            return envelope_residual(
                pressure_parameterized_state,
                pressure_state,
                beta,
                mixture,
                design=design,
                kij_pair=kij_pair,
            )

    else:
        raise ValueError("mode must be 'pressure' or 'temperature'")

    return JaxProblem(
        f=lambda state, parameter: (
            0.0 * jnp.sum(state) * (1.0 + 0.0 * jnp.sum(parameter))
        ),
        g=residual,
        n=n + 1,
        m=n + 1,
        p_example=jnp.zeros(1),
        cl=jnp.zeros(n + 1),
        cu=jnp.zeros(n + 1),
        options={"tol": 1e-10, "print_level": 0, "sb": "yes"},
    )


def solve_low_pressure_anchor(
    mixture: PengRobinsonMixture,
    *,
    beta: float,
    pressure_pa: float = 1e5,
    design=None,
    kij_pair: tuple[int, int] | None = None,
) -> jax.Array:
    """Solve a low-pressure isopleth anchor ``[ln K, ln T]`` [mixed]."""

    if design is None:
        composition = mixture.composition
    else:
        composition = np.asarray(_decode_design(mixture, design, kij_pair)[0])
    guess, _ = wilson_start(pressure_pa, beta, mixture, composition=composition)
    problem = make_envelope_problem(
        mixture, beta, mode="pressure", design=design, kij_pair=kij_pair
    )
    return problem.solve(jnp.asarray([np.log(pressure_pa)]), guess)


@dataclass(frozen=True)
class PhaseBoundaryPoint:
    """A fixed-pressure phase-boundary point and its design sensitivity."""

    state: np.ndarray
    pressure_pa: float
    design: np.ndarray
    state_design_jacobian: np.ndarray
    max_residual: float

    @property
    def temperature_k(self) -> float:
        """Boundary temperature [K]."""

        return float(np.exp(self.state[-1]))

    @property
    def temperature_design_jacobian(self) -> np.ndarray:
        """Derivative of boundary temperature [K] with design [-]."""

        return self.temperature_k * self.state_design_jacobian[-1]


def phase_boundary_with_sensitivity(
    mixture: PengRobinsonMixture,
    *,
    beta: float,
    pressure_pa: float,
    kij_pair: tuple[int, int],
    initial_state=None,
) -> PhaseBoundaryPoint:
    """Solve one fixed-pressure boundary point and differentiate it.

    The design vector contains ``N-1`` simplex coordinates followed by the
    selected ``k_ij`` [-].  ``pressure_pa`` is fixed [Pa], and the returned
    temperature sensitivity is in K per design coordinate.
    """

    q0 = design_parameters(mixture, kij_pair)
    if initial_state is None:
        initial_state = solve_low_pressure_anchor(
            mixture,
            beta=beta,
            pressure_pa=pressure_pa,
            design=q0,
            kij_pair=kij_pair,
        )

    def residual(state, design):
        return envelope_residual(
            state,
            jnp.log(pressure_pa),
            beta,
            mixture,
            design=design,
            kij_pair=kij_pair,
        )

    state = jnp.asarray(initial_state, dtype=jnp.float64)
    design = jnp.asarray(q0, dtype=jnp.float64)
    state_jacobian = jax.jacobian(residual, argnums=0)(state, design)
    design_jacobian = jax.jacobian(residual, argnums=1)(state, design)
    # F(x(q), q) = 0 implies dx/dq = -F_x^{-1} F_q.  The state came
    # from POUNCE's converged fixed-pressure solve immediately above.
    jacobian = -jnp.linalg.solve(state_jacobian, design_jacobian)
    residual_value = residual(state, design)
    return PhaseBoundaryPoint(
        state=np.asarray(state, dtype=float),
        pressure_pa=float(pressure_pa),
        design=q0,
        state_design_jacobian=np.asarray(jacobian, dtype=float),
        max_residual=float(jnp.max(jnp.abs(residual_value))),
    )


def trace_envelope(
    mixture: PengRobinsonMixture,
    *,
    beta: float,
    mode: Mode = "pressure",
    pressure_pa: float = 1e5,
    ds: float = 0.05,
    n_steps: int = 400,
    direction: float = 1.0,
    design=None,
    kij_pair: tuple[int, int] | None = None,
) -> PathTrace:
    """Trace an isopleth with pseudo-arclength continuation.

    ``pressure_pa`` [Pa] selects a low-pressure anchor.  The returned state is
    ``[ln K, ln T]`` in pressure mode and ``[ln K, ln P]`` in temperature mode.
    """

    n = mixture.n_components
    pressure_state = solve_low_pressure_anchor(
        mixture,
        beta=beta,
        pressure_pa=pressure_pa,
        design=design,
        kij_pair=kij_pair,
    )
    if mode == "pressure":
        state0 = pressure_state
        parameter0 = np.log(pressure_pa)  # [ln Pa]
    elif mode == "temperature":
        state0 = jnp.concatenate(
            [pressure_state[:n], jnp.asarray([np.log(pressure_pa)])]
        )
        parameter0 = float(pressure_state[n])  # [ln K]
    else:
        raise ValueError("mode must be 'pressure' or 'temperature'")
    problem = make_envelope_problem(
        mixture, beta, mode=mode, design=design, kij_pair=kij_pair
    )
    return PathFollower(problem).trace_arclength(
        state0,
        parameter0,
        ds=ds,
        n_steps=n_steps,
        direction=direction,
    )


def physical_coordinates(
    trace: PathTrace, mixture: PengRobinsonMixture, *, mode: Mode
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(temperature [K], pressure [Pa])`` along a trace."""

    n = mixture.n_components
    if mode == "pressure":
        return np.exp(trace.x[:, n]), np.exp(trace.theta)
    if mode == "temperature":
        return np.exp(trace.theta), np.exp(trace.x[:, n])
    raise ValueError("mode must be 'pressure' or 'temperature'")


def branch_distance(trace: PathTrace, mixture: PengRobinsonMixture) -> np.ndarray:
    """Return ``max_i |ln K_i|`` [-] at each traced point.

    A value near zero is the trivial ``K_i=1`` branch, except in the physical
    critical neighborhood where the phases genuinely become identical.
    """

    return np.max(np.abs(trace.x[:, : mixture.n_components]), axis=1)


@dataclass(frozen=True)
class PhasePointDiagnostics:
    """Physical and algebraic checks for one phase-envelope point."""

    max_residual: float
    branch_distance: float
    liquid_sum: float
    vapor_sum: float
    liquid_z_minus_b: float
    vapor_z_minus_b: float

    @property
    def roots_are_admissible(self) -> bool:
        """Whether both selected cubic roots satisfy ``Z > B``."""

        return self.liquid_z_minus_b > 0.0 and self.vapor_z_minus_b > 0.0


def diagnose_phase_point(
    state,
    parameter,
    mixture: PengRobinsonMixture,
    *,
    beta: float,
    mode: Mode,
    design=None,
    kij_pair: tuple[int, int] | None = None,
) -> PhasePointDiagnostics:
    """Evaluate branch, normalization, residual, and cubic-root guards.

    ``parameter`` is ``ln(P/Pa)`` in pressure mode and ``ln(T/K)`` in
    temperature mode.  This is a local root-admissibility check, not a global
    tangent-plane-distance phase-stability calculation.
    """

    n = mixture.n_components
    state = jnp.asarray(state, dtype=jnp.float64)
    if mode == "pressure":
        pressure_state = state
        log_pressure = jnp.asarray(parameter, dtype=jnp.float64)
    elif mode == "temperature":
        pressure_state = jnp.concatenate(
            [state[:n], jnp.asarray([parameter], dtype=jnp.float64)]
        )
        log_pressure = state[n]
    else:
        raise ValueError("mode must be 'pressure' or 'temperature'")

    log_k = pressure_state[:n]  # [-]
    temperature_k = jnp.exp(pressure_state[n])  # [K]
    pressure_pa = jnp.exp(log_pressure)  # [Pa]
    k_values = jnp.exp(log_k)  # [-]
    fixed_composition, kij = _decode_design(mixture, design, kij_pair)
    liquid = fixed_composition / (1.0 - beta + beta * k_values)  # [-]
    vapor = k_values * liquid  # [-]
    residual = envelope_residual(
        pressure_state,
        log_pressure,
        beta,
        mixture,
        design=design,
        kij_pair=kij_pair,
    )

    _, bi = _component_ab(mixture, temperature_k)

    def z_minus_b(composition, *, largest):
        b_mix = composition @ bi  # [m^3 mol^-1]
        b_red = b_mix * pressure_pa / (R * temperature_k)  # [-]
        z_factor = compressibility(
            composition,
            temperature_k,
            pressure_pa,
            mixture,
            largest=largest,
            binary_interaction=kij,
        )
        return z_factor - b_red

    return PhasePointDiagnostics(
        max_residual=float(jnp.max(jnp.abs(residual))),
        branch_distance=float(jnp.max(jnp.abs(log_k))),
        liquid_sum=float(jnp.sum(liquid)),
        vapor_sum=float(jnp.sum(vapor)),
        liquid_z_minus_b=float(z_minus_b(liquid, largest=False)),
        vapor_z_minus_b=float(z_minus_b(vapor, largest=True)),
    )


def _fold_equations(
    fold_state,
    design,
    mixture: PengRobinsonMixture,
    beta: float,
    mode: Mode,
    kij_pair: tuple[int, int] | None,
):
    """Augmented simple-fold equations ``[F, F_x v, v^T v - 1]`` [-]."""

    n_state = mixture.n_components + 1
    state = fold_state[:n_state]
    parameter = fold_state[n_state]
    null_vector = fold_state[n_state + 1 :]

    def residual_at(current_state):
        if mode == "pressure":
            return envelope_residual(
                current_state,
                parameter,
                beta,
                mixture,
                design=design,
                kij_pair=kij_pair,
            )
        if mode == "temperature":
            pressure_parameterized_state = jnp.concatenate(
                [current_state[: mixture.n_components], jnp.asarray([parameter])]
            )
            return envelope_residual(
                pressure_parameterized_state,
                current_state[mixture.n_components],
                beta,
                mixture,
                design=design,
                kij_pair=kij_pair,
            )
        raise ValueError("mode must be 'pressure' or 'temperature'")

    residual = residual_at(state)
    state_jacobian = jax.jacobian(residual_at)(state)
    return jnp.concatenate(
        [
            residual,
            state_jacobian @ null_vector,
            jnp.asarray([jnp.dot(null_vector, null_vector) - 1.0]),
        ]
    )


@dataclass(frozen=True)
class RefinedFold:
    """A refined phase-envelope extremum and its design sensitivity."""

    mode: Mode
    state: np.ndarray
    log_parameter: float
    null_vector: np.ndarray
    design: np.ndarray
    state_design_jacobian: np.ndarray
    max_residual: float

    @property
    def parameter(self) -> float:
        """Extremal pressure [Pa] or temperature [K]."""

        return float(np.exp(self.log_parameter))

    @property
    def parameter_design_jacobian(self) -> np.ndarray:
        """Derivative of pressure [Pa] or temperature [K] with design [-]."""

        return self.parameter * self.state_design_jacobian[len(self.state)]

    @property
    def temperature_k(self) -> float:
        """Temperature at the refined fold [K]."""

        if self.mode == "temperature":
            return self.parameter
        return float(np.exp(self.state[-1]))

    @property
    def pressure_pa(self) -> float:
        """Pressure at the refined fold [Pa]."""

        if self.mode == "pressure":
            return self.parameter
        return float(np.exp(self.state[-1]))

    @property
    def temperature_design_jacobian(self) -> np.ndarray:
        """Derivative of fold temperature [K] with design [-]."""

        if self.mode == "temperature":
            return self.parameter_design_jacobian
        return self.temperature_k * self.state_design_jacobian[len(self.state) - 1]

    @property
    def pressure_design_jacobian(self) -> np.ndarray:
        """Derivative of fold pressure [Pa] with design [-]."""

        if self.mode == "pressure":
            return self.parameter_design_jacobian
        return self.pressure_pa * self.state_design_jacobian[len(self.state) - 1]


def _fold_initial_guess(
    trace: PathTrace,
    mixture: PengRobinsonMixture,
    beta: float,
    mode: Mode,
    design,
    kij_pair,
) -> np.ndarray:
    """Build an augmented fold seed from a traced sign-change bracket."""

    theta = np.asarray(trace.theta, dtype=float)
    increments = np.diff(theta)
    reversals = np.flatnonzero(increments[:-1] * increments[1:] <= 0.0) + 1
    if reversals.size == 0:
        raise ValueError("trace does not bracket a turning point")
    index = int(reversals[np.argmax(theta[reversals])])
    state = jnp.asarray(trace.x[index], dtype=jnp.float64)
    parameter = jnp.asarray(theta[index], dtype=jnp.float64)

    def residual_at(current_state):
        n = mixture.n_components
        if mode == "pressure":
            return envelope_residual(
                current_state,
                parameter,
                beta,
                mixture,
                design=design,
                kij_pair=kij_pair,
            )
        pressure_state = current_state[n]
        pressure_parameterized_state = jnp.concatenate(
            [current_state[:n], jnp.asarray([parameter])]
        )
        return envelope_residual(
            pressure_parameterized_state,
            pressure_state,
            beta,
            mixture,
            design=design,
            kij_pair=kij_pair,
        )

    state_jacobian = np.asarray(jax.jacobian(residual_at)(state), dtype=float)
    _, _, right = np.linalg.svd(state_jacobian)
    null_vector = right[-1]
    return np.concatenate([np.asarray(state), [float(parameter)], null_vector])


def refine_fold(
    trace: PathTrace,
    mixture: PengRobinsonMixture,
    *,
    beta: float,
    mode: Mode,
    kij_pair: tuple[int, int],
) -> RefinedFold:
    """Refine a traced extremum and differentiate it with POUNCE.

    The fold is solved from ``F=0``, ``F_x v=0``, and ``||v||=1``.  The
    POUNCE implicit solve returns derivatives with respect to ``N-1`` simplex
    coordinates followed by the selected binary interaction parameter [-].
    """

    n_state = mixture.n_components + 1
    n_fold = 2 * n_state + 1
    q0 = design_parameters(mixture, kij_pair)
    w0 = _fold_initial_guess(trace, mixture, beta, mode, q0, kij_pair)
    problem = JaxProblem(
        f=lambda fold_state, design: (
            0.0 * jnp.sum(fold_state) * (1.0 + 0.0 * jnp.sum(design))
        ),
        g=lambda fold_state, design: _fold_equations(
            fold_state, design, mixture, beta, mode, kij_pair
        ),
        n=n_fold,
        m=n_fold,
        p_example=jnp.asarray(q0),
        cl=jnp.zeros(n_fold),
        cu=jnp.zeros(n_fold),
        options={"tol": 1e-11, "print_level": 0, "sb": "yes", "max_iter": 500},
        factor_reuse=False,
    )
    solution, _, jacobian = problem.solve_with_jacobian(jnp.asarray(q0), w0)
    solution_np = np.asarray(solution, dtype=float)
    residual = np.asarray(
        _fold_equations(solution, jnp.asarray(q0), mixture, beta, mode, kij_pair)
    )
    return RefinedFold(
        mode=mode,
        state=solution_np[:n_state],
        log_parameter=float(solution_np[n_state]),
        null_vector=solution_np[n_state + 1 :],
        design=q0,
        state_design_jacobian=np.asarray(jacobian, dtype=float),
        max_residual=float(np.max(np.abs(residual))),
    )


@dataclass(frozen=True)
class InverseDesignResult:
    """Composition chosen to hit a prescribed envelope extremum."""

    mixture: PengRobinsonMixture
    target: float
    achieved: float
    log_ratio_change: np.ndarray
    objective: float
    max_residual: float


def design_composition_for_extremum(
    fold: RefinedFold,
    mixture: PengRobinsonMixture,
    *,
    beta: float,
    target: float,
    kij_pair: tuple[int, int],
) -> InverseDesignResult:
    """Minimally change composition so a refined extremum hits ``target``.

    ``target`` is pressure [Pa] for a pressure fold and temperature [K] for a
    temperature fold.  Binary interaction parameters remain fixed.  The
    decision uses unconstrained log-ratio composition coordinates, so every
    iterate is strictly inside the composition simplex.
    """

    n = mixture.n_components
    n_state = n + 1
    n_fold = 2 * n_state + 1
    eta0 = simplex_coordinates(mixture.composition)
    kij_value = mixture.binary_interaction[kij_pair]
    x0 = np.concatenate([eta0, fold.state, [fold.log_parameter], fold.null_vector])

    def design_vector(decision):
        return jnp.concatenate(
            [decision[: n - 1], jnp.asarray([kij_value], dtype=jnp.float64)]
        )

    def constraints(decision, unused):
        fold_state = decision[n - 1 :]
        augmented = _fold_equations(
            fold_state,
            design_vector(decision),
            mixture,
            beta,
            fold.mode,
            kij_pair,
        )
        target_residual = fold_state[n_state] - jnp.log(target)
        return jnp.concatenate([augmented, jnp.asarray([target_residual])])

    problem = JaxProblem(
        f=lambda decision, unused: (
            0.5
            * jnp.sum((decision[: n - 1] - jnp.asarray(eta0)) ** 2)
            * (1.0 + 0.0 * jnp.sum(unused))
        ),
        g=constraints,
        n=(n - 1) + n_fold,
        m=n_fold + 1,
        p_example=jnp.zeros(0),
        cl=jnp.zeros(n_fold + 1),
        cu=jnp.zeros(n_fold + 1),
        options={"tol": 1e-10, "print_level": 0, "sb": "yes", "max_iter": 1000},
        factor_reuse=False,
    )
    solution = np.asarray(problem.solve(jnp.zeros(0), x0), dtype=float)
    eta = solution[: n - 1]
    composition = np.asarray(composition_from_coordinates(eta), dtype=float)
    residual = np.asarray(constraints(jnp.asarray(solution), jnp.zeros(0)))
    achieved = float(np.exp(solution[(n - 1) + n_state]))
    return InverseDesignResult(
        mixture=mixture.with_composition(composition),
        target=float(target),
        achieved=achieved,
        log_ratio_change=eta - eta0,
        objective=float(0.5 * np.sum((eta - eta0) ** 2)),
        max_residual=float(np.max(np.abs(residual))),
    )


def vapor_pressure(
    mixture: PengRobinsonMixture,
    component: int,
    temperature_k: float,
    *,
    bracket_pa: tuple[float, float] = (1e3, 3.5e6),
) -> float:
    """Return pure-component saturation pressure [Pa] by fugacity equality."""

    composition = np.zeros(mixture.n_components)
    composition[component] = 1.0

    def residual(log_pressure: float) -> float:
        pressure_pa = np.exp(log_pressure)
        vapor = log_fugacity_coefficients(
            composition,
            temperature_k,
            pressure_pa,
            mixture,
            largest=True,
        )[component]
        liquid = log_fugacity_coefficients(
            composition,
            temperature_k,
            pressure_pa,
            mixture,
            largest=False,
        )[component]
        return float(vapor - liquid)

    lo, hi = np.log(bracket_pa)
    return float(np.exp(brentq(residual, lo, hi)))


NATURAL_GAS = PengRobinsonMixture(
    names=("methane", "ethane", "propane", "n-butane"),
    critical_temperature=np.array([190.6, 305.3, 369.8, 425.1]),  # [K]
    critical_pressure=np.array([45.99, 48.72, 42.48, 37.96]) * 1e5,  # [Pa]
    acentric_factor=np.array([0.012, 0.100, 0.152, 0.200]),  # [-]
    composition=np.array([0.80, 0.10, 0.06, 0.04]),  # [-]
    binary_interaction=np.zeros((4, 4)),  # [-]
    source="Notebook 34 natural-gas demonstration; all k_ij = 0.",
)


_DEITERS_BELL_KIJ = _kij_matrix(2, (0, 1), 0.042823)
DEITERS_BELL_METHANE_PROPANE = PengRobinsonMixture(
    names=("methane", "propane"),
    critical_temperature=np.array([190.555, 369.825]),  # [K]
    critical_pressure=np.array([4.595, 4.248]) * 1e6,  # [Pa]
    acentric_factor=np.array([0.0, 0.15308]),  # [-]
    composition=np.array([0.80, 0.20]),  # [-], methane-rich liquid isopleth
    binary_interaction=np.asarray(_DEITERS_BELL_KIJ),  # [-]
    source=(
        "Deiters & Bell, AIChE J. 65 (2019) e16730, Table 1 and Figure 2; "
        "doi:10.1002/aic.16730."
    ),
)


__all__ = [
    "DEITERS_BELL_METHANE_PROPANE",
    "NATURAL_GAS",
    "InverseDesignResult",
    "PhaseBoundaryPoint",
    "PhasePointDiagnostics",
    "PengRobinsonMixture",
    "RefinedFold",
    "branch_distance",
    "composition_from_coordinates",
    "compressibility",
    "design_composition_for_extremum",
    "design_parameters",
    "diagnose_phase_point",
    "envelope_residual",
    "log_fugacity_coefficients",
    "make_envelope_problem",
    "mixture_from_design",
    "physical_coordinates",
    "phase_boundary_with_sensitivity",
    "refine_fold",
    "simplex_coordinates",
    "solve_low_pressure_anchor",
    "trace_envelope",
    "vapor_pressure",
    "wilson_log_k",
    "wilson_start",
]
