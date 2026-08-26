"""Closed-loop advanced-step NMPC for the Hicks--Ray CSTR.

This example keeps the nonlinear-programming model from notebook 36 and adds
the controller around it: predicted-state background solves, measurement
updates from the held KKT factorization, guarded fallbacks, horizon shifts,
and an independently integrated plant.

The state and control variables are dimensionless. Time is measured in
minutes. zc is concentration, zt is temperature, v1 is the coolant-valve
fraction, and v2 is the residence-time-valve fraction.
"""

from __future__ import annotations

import platform
import subprocess
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, Sequence

import numpy as np
import pyomo.environ as pyo
from pyomo.common.errors import ApplicationError
from pyomo.core.base.var import VarData
from pyomo.dae import ContinuousSet, DerivativeVar
from pyomo.opt import TerminationCondition
from pyomo_cvp import declare_profile
from pyomo_pounce import (
    active_set_changes,
    declare_sens_param,
    estimate,
    estimate_report,
)

import pyomo_pounce  # noqa: F401 -- registers SolverFactory("pounce")


Policy = Literal[
    "full_resolve",
    "no_correction",
    "clamped_linear",
    "fix_relax",
    "path",
    "guarded_path",
]

POLICIES: tuple[Policy, ...] = (
    "full_resolve",
    "no_correction",
    "clamped_linear",
    "fix_relax",
    "path",
    "guarded_path",
)

MODEL_REVISION = "hicks-ray-collocation-asnmpc-v1"

# Dimensionless steady state and manipulated-variable targets.
ZC_SS = 0.6416
ZT_SS = 0.5387
V1_SS = 0.57828
V2_SS = 0.49989
V1_MIN = 1.0 / 6.0
BASE_STATE = (ZC_SS - 0.12, ZT_SS - 0.05)


@dataclass(frozen=True)
class CstrConfig:
    """Numerical and safety settings for the controller.

    Parameters
    ----------
    horizon_intervals
        Number of finite elements [-]. Use 100 for the paper-scale model and
        a smaller value for deterministic CI smokes.
    sample_time_min
        Controller sample time [min].
    collocation_points
        Radau collocation points per finite element [-].
    temperature_limit
        Controller and plant safety limit for dimensionless temperature [-].
    solver_tol
        POUNCE NLP optimality tolerance [-].
    feasibility_tol
        Maximum corrected-point primal violation accepted by the guard [-].
    stationarity_tol
        Maximum corrected-point stationarity residual accepted by the guard
        [-].
    complementarity_tol
        Maximum corrected-point complementarity residual accepted by the
        guard [-].
    measurement_trust_radius
        Largest scaled prediction-to-measurement displacement accepted by the
        guard [-].
    deadline_s
        Online update deadline [s]. Background preparation is excluded.
    predictor_iter
        Maximum active-set path breakpoints or fix/relax passes [-].
    corrector_iter
        Newton back-solves applied to the corrected KKT point [-].
    plant_substeps
        RK4 substeps per controller sample [-].
    """

    horizon_intervals: int = 100
    sample_time_min: float = 1.0
    collocation_points: int = 3
    temperature_limit: float = 0.70
    solver_tol: float = 1.0e-8
    feasibility_tol: float = 2.0e-6
    stationarity_tol: float = 2.0e-4
    complementarity_tol: float = 2.0e-4
    measurement_trust_radius: float = 1.0
    concentration_scale: float = 0.04
    temperature_scale: float = 0.02
    deadline_s: float = 0.050
    predictor_iter: int = 16
    corrector_iter: int = 6
    degeneracy_iter: int = 16
    plant_substeps: int = 20
    reject_ambiguous_activity: bool = True

    def __post_init__(self) -> None:
        """Validate user-facing configuration at the boundary."""
        if self.horizon_intervals < 2:
            raise ValueError("horizon_intervals must be at least 2")
        if self.sample_time_min <= 0:
            raise ValueError("sample_time_min must be positive")
        if self.collocation_points < 1:
            raise ValueError("collocation_points must be positive")
        if self.temperature_limit <= ZT_SS:
            raise ValueError("temperature_limit must exceed the target")
        positive = {
            "solver_tol": self.solver_tol,
            "feasibility_tol": self.feasibility_tol,
            "stationarity_tol": self.stationarity_tol,
            "complementarity_tol": self.complementarity_tol,
            "measurement_trust_radius": self.measurement_trust_radius,
            "concentration_scale": self.concentration_scale,
            "temperature_scale": self.temperature_scale,
            "deadline_s": self.deadline_s,
        }
        for name, value in positive.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        integer_positive = {
            "predictor_iter": self.predictor_iter,
            "corrector_iter": self.corrector_iter,
            "degeneracy_iter": self.degeneracy_iter,
            "plant_substeps": self.plant_substeps,
        }
        for name, value in integer_positive.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")


@dataclass(frozen=True)
class Campaign:
    """Deterministic disturbance and measurement-error sequence.

    All state, parameter-scale, bias, and noise entries are dimensionless.
    feed_temperature_shift changes the plant feed-temperature parameter at
    each controller sample.
    """

    name: str
    feed_temperature_shift: tuple[float, ...]
    measurement_noise: tuple[tuple[float, float], ...]
    measurement_bias: tuple[float, float] = (0.0, 0.0)
    reaction_rate_scale: float = 1.0
    heat_transfer_scale: float = 1.0

    @property
    def steps(self) -> int:
        """Number of controller samples [-]."""
        return len(self.feed_temperature_shift)

    def __post_init__(self) -> None:
        """Require one deterministic noise pair per disturbance sample."""
        if not self.name:
            raise ValueError("campaign name must not be empty")
        if not self.feed_temperature_shift:
            raise ValueError("campaign must contain at least one sample")
        if len(self.measurement_noise) != len(self.feed_temperature_shift):
            raise ValueError("measurement_noise must match campaign length")
        if self.reaction_rate_scale <= 0 or self.heat_transfer_scale <= 0:
            raise ValueError("plant parameter scales must be positive")


@dataclass(frozen=True)
class SolveRecord:
    """One converged controller NLP and its wall-clock latency [s].

    A warm-start failure is retried once from a cold start. The counters make
    that recovery visible instead of quietly reporting only the successful
    attempt.
    """

    model: pyo.ConcreteModel
    latency_s: float
    solver_failures: int = 0
    solver_recoveries: int = 0


@dataclass(frozen=True)
class Trajectory:
    """Finite-horizon state/control trajectory on the collocation grid."""

    time_min: np.ndarray
    concentration: np.ndarray
    temperature: np.ndarray
    coolant: np.ndarray
    residence: np.ndarray

    @property
    def first_control(self) -> tuple[float, float]:
        """First manipulated-variable move, both dimensionless."""
        return float(self.coolant[0]), float(self.residence[0])

    def state_at(self, time_min: float) -> tuple[float, float]:
        """State at the collocation point nearest time_min."""
        idx = int(np.abs(self.time_min - time_min).argmin())
        return float(self.concentration[idx]), float(self.temperature[idx])

    def control_at(self, time_min: float) -> tuple[float, float]:
        """Held manipulated-variable pair at the requested time."""
        idx = int(np.abs(self.time_min - time_min).argmin())
        return float(self.coolant[idx]), float(self.residence[idx])


@dataclass(frozen=True)
class CorrectionDiagnostics:
    """Online update diagnostics used by the acceptance guard."""

    accepted: bool
    reasons: tuple[str, ...]
    update_latency_s: float
    full_solve_latency_s: float | None
    primal_violation: float
    stationarity: float
    feasibility: float
    complementarity: float
    residual: float
    initial_residual: float
    path_events: tuple[str, ...]
    corrector_iterations: int
    refine_stop: str | None
    ambiguous_activity: tuple[str, ...]


@dataclass(frozen=True)
class SampleRecord:
    """One predict/measure/correct/validate/apply/shift cycle."""

    sample: int
    predicted_state: tuple[float, float]
    measured_state: tuple[float, float]
    plant_state_before: tuple[float, float]
    applied_control: tuple[float, float]
    plant_state_after: tuple[float, float]
    background_latency_s: float
    solver_failures: int
    solver_recoveries: int
    diagnostics: CorrectionDiagnostics

    def event_row(self) -> dict[str, object]:
        """Machine-readable event-timeline row."""
        return {
            "sample": self.sample,
            "predicted_state": self.predicted_state,
            "measured_state": self.measured_state,
            "applied_control": self.applied_control,
            "path_length": len(self.diagnostics.path_events),
            "active_set_changes": self.diagnostics.path_events,
            "corrector_residual": self.diagnostics.residual,
            "accepted": self.diagnostics.accepted,
            "guard_reasons": self.diagnostics.reasons,
            "full_resolve": self.diagnostics.full_solve_latency_s is not None,
            "solver_failures": self.solver_failures,
            "solver_recoveries": self.solver_recoveries,
        }


@dataclass(frozen=True)
class ClosedLoopMetrics:
    """Closed-loop quality, safety, fallback, and latency metrics."""

    iae: float
    ise: float
    economic_stage_cost: float
    total_control_movement: float
    maximum_temperature_violation: float
    active_set_changes: int
    fallback_full_solves: int
    fallback_fraction: float
    update_latency_median_s: float
    update_latency_p95_s: float
    full_solve_latency_median_s: float | None
    background_latency_median_s: float | None
    deadline_s: float
    deadline_misses: int
    solver_failures: int
    solver_recoveries: int


@dataclass(frozen=True)
class BenchmarkStamp:
    """Provenance carried beside timing and iteration results."""

    pounce_commit: str
    model_revision: str
    solver_tolerance: float
    platform: str
    python: str
    warmup_excluded: bool


@dataclass
class ClosedLoopResult:
    """Complete result for one policy/scenario pair."""

    policy: Policy
    campaign: str
    samples: list[SampleRecord]
    metrics: ClosedLoopMetrics
    stamp: BenchmarkStamp

    def event_timeline(self) -> list[dict[str, object]]:
        """Return every sample as a serializable event row."""
        return [sample.event_row() for sample in self.samples]

    def summary(self) -> dict[str, object]:
        """Return a flat, table-friendly summary."""
        return {
            "policy": self.policy,
            "campaign": self.campaign,
            **asdict(self.metrics),
            **asdict(self.stamp),
        }


@dataclass(frozen=True)
class DegeneracyStep:
    """One directional update about an active-set breakpoint."""

    direction: str
    target_state: tuple[float, float]
    probe_time_min: float
    one_sided_probe_control: tuple[float, float]
    directional_probe_control: tuple[float, float]
    directional_events: tuple[str, ...]
    guard_reasons: tuple[str, ...]


@dataclass(frozen=True)
class DegeneracyExperiment:
    """Two-sided measurement updates from one degenerate held solve."""

    breakpoint_fraction: float
    breakpoint_variable: str
    breakpoint_bound: str
    steps: tuple[DegeneracyStep, DegeneracyStep]


def _reaction_rate(zc: object, zt: object, k0: object, ea: object) -> object:
    """Dimensionless reaction rate [state fraction/min]."""
    return k0 * zc * pyo.exp(-ea / zt)


def build_controller(
    initial_state: Sequence[float],
    config: CstrConfig | None = None,
) -> pyo.ConcreteModel:
    """Build the collocated finite-horizon CSTR controller model.

    Parameters
    ----------
    initial_state
        Measured or predicted (zc, zt) state, both dimensionless.
    config
        Horizon, solver, and safety settings.

    Returns
    -------
    pyomo.environ.ConcreteModel
        Model with both initial-state parameters declared for held-factor
        sensitivity.
    """
    cfg = config or CstrConfig()
    if len(initial_state) != 2:
        raise ValueError("initial_state must contain (zc, zt)")
    zc0, zt0 = (float(initial_state[0]), float(initial_state[1]))
    if not (0.0 <= zc0 <= 1.0 and 0.0 < zt0 <= cfg.temperature_limit):
        raise ValueError("initial_state lies outside the controller domain")

    horizon_min = cfg.horizon_intervals * cfg.sample_time_min
    m = pyo.ConcreteModel()
    m.t = ContinuousSet(initialize=pyo.RangeSet(0.0, horizon_min, cfg.sample_time_min))

    # Hicks--Ray dimensionless parameters; time constants use minutes.
    m.u1sf = pyo.Param(initialize=600.0)
    m.u2sf = pyo.Param(initialize=40.0)
    m.k0 = pyo.Param(initialize=300.0)
    m.ea = pyo.Param(initialize=5.0)
    m.a0 = pyo.Param(initialize=1.95e-4)
    m.ztcw = pyo.Param(initialize=0.38)
    m.ztf = pyo.Param(initialize=0.395)

    m.zc0 = pyo.Param(initialize=zc0, mutable=True)
    m.zt0 = pyo.Param(initialize=zt0, mutable=True)
    m.zc = pyo.Var(m.t, bounds=(0.0, 1.0), initialize=ZC_SS)
    m.zt = pyo.Var(m.t, bounds=(0.0, cfg.temperature_limit), initialize=ZT_SS)
    m.dzc = DerivativeVar(m.zc, wrt=m.t)
    m.dzt = DerivativeVar(m.zt, wrt=m.t)
    m.v1 = pyo.Var(m.t, bounds=(V1_MIN, 1.0), initialize=V1_SS)
    m.v2 = pyo.Var(m.t, bounds=(0.025, 1.0), initialize=V2_SS)
    declare_profile(m.v1, m.v2, wrt=m.t, profile="piecewise_constant")

    @m.Constraint(m.t)
    def zc_ode(model: pyo.ConcreteModel, t: float) -> object:
        """Dimensionless concentration balance [state fraction/min]."""
        residence = model.u2sf * model.v2[t]  # [min]
        reaction = _reaction_rate(model.zc[t], model.zt[t], model.k0, model.ea)
        return model.dzc[t] == (1.0 - model.zc[t]) / residence - reaction

    @m.Constraint(m.t)
    def zt_ode(model: pyo.ConcreteModel, t: float) -> object:
        """Dimensionless energy balance [temperature fraction/min]."""
        residence = model.u2sf * model.v2[t]  # [min]
        reaction = _reaction_rate(model.zc[t], model.zt[t], model.k0, model.ea)
        cooling = model.a0 * model.u1sf * model.v1[t] * (model.zt[t] - model.ztcw)
        return model.dzt[t] == (
            (model.ztf - model.zt[t]) / residence + reaction - cooling
        )

    @m.Constraint()
    def zc_init(model: pyo.ConcreteModel) -> object:
        """Pin initial concentration to the sensitivity parameter [-]."""
        return model.zc[0.0] == model.zc0

    @m.Constraint()
    def zt_init(model: pyo.ConcreteModel) -> object:
        """Pin initial temperature to the sensitivity parameter [-]."""
        return model.zt[0.0] == model.zt0

    grid = sorted(m.t)

    @m.Objective()
    def objective(model: pyo.ConcreteModel) -> object:
        """Dimensionless regulation and control-movement stage cost."""
        stage = sum(
            10.0 * (model.zc[t] - ZC_SS) ** 2
            + 2.0 * (model.zt[t] - ZT_SS) ** 2
            + (model.v1[t] - V1_SS) ** 2
            + 0.5 * (model.v2[t] - V2_SS) ** 2
            for t in grid[:-1]
        )
        terminal = 1000.0 * (
            10.0 * (model.zc[grid[-1]] - ZC_SS) ** 2
            + 2.0 * (model.zt[grid[-1]] - ZT_SS) ** 2
        )
        return stage + terminal

    pyo.TransformationFactory("dae.collocation").apply_to(
        m,
        wrt=m.t,
        nfe=cfg.horizon_intervals,
        ncp=cfg.collocation_points,
        scheme="LAGRANGE-RADAU",
    )
    pyo.TransformationFactory("cvp.parameterize").apply_to(m)
    declare_sens_param(m.zc0)
    declare_sens_param(m.zt0)
    return m


def _nearest_index(indices: Sequence[float], requested: float) -> float:
    """Return the grid index nearest a requested time [min]."""
    return min(indices, key=lambda value: abs(float(value) - requested))


def shift_warm_start(
    previous: pyo.ConcreteModel,
    target: pyo.ConcreteModel,
    sample_time_min: float,
) -> None:
    """Shift state, derivative, and control profiles by one sample.

    The final available point is held constant. This is a primal warm start;
    the new solve builds its own KKT factorization and multipliers.
    """
    if sample_time_min <= 0:
        raise ValueError("sample_time_min must be positive")
    for component_name in ("zc", "zt", "dzc", "dzt", "v1", "v2"):
        source_component = previous.find_component(component_name)
        target_component = target.find_component(component_name)
        if source_component is None or target_component is None:
            raise ValueError(f"missing warm-start component {component_name}")
        source_indices = sorted(source_component)
        source_end = float(source_indices[-1])
        for target_index in sorted(target_component):
            requested = min(float(target_index) + sample_time_min, source_end)
            source_index = _nearest_index(source_indices, requested)
            value = pyo.value(source_component[source_index], exception=False)
            if value is not None and np.isfinite(float(value)):
                target_component[target_index].set_value(
                    float(value), skip_validation=True
                )


def solve_controller(
    initial_state: Sequence[float],
    config: CstrConfig | None = None,
    warm_start: pyo.ConcreteModel | None = None,
) -> SolveRecord:
    """Build and solve one background or fallback controller NLP."""
    cfg = config or CstrConfig()
    model = build_controller(initial_state, cfg)
    if warm_start is not None:
        shift_warm_start(warm_start, model, cfg.sample_time_min)
        model.zc[0.0].set_value(float(initial_state[0]), skip_validation=True)
        model.zt[0.0].set_value(float(initial_state[1]), skip_validation=True)

    started = time.perf_counter()
    result = pyo.SolverFactory("pounce").solve(
        model,
        tee=False,
        options={
            "tol": cfg.solver_tol,
            "bound_relax_factor": 0.0,
            "print_level": 0,
        },
    )
    latency_s = time.perf_counter() - started
    condition = result.solver.termination_condition
    if condition != TerminationCondition.optimal:
        raise RuntimeError(f"POUNCE controller solve ended with {condition}")
    return SolveRecord(model=model, latency_s=latency_s)


def _solve_with_recovery(
    initial_state: Sequence[float],
    config: CstrConfig,
    warm_start: pyo.ConcreteModel | None = None,
) -> SolveRecord:
    """Retry one failed warm-started NLP from a cold controller model.

    An initial cold-start failure remains fatal because there is no previously
    verified solve to recover from. The returned latency includes both the
    failed warm attempt and the successful retry.
    """
    started = time.perf_counter()
    try:
        solved = solve_controller(initial_state, config, warm_start)
    except (RuntimeError, ApplicationError):
        if warm_start is None:
            raise
        solved = solve_controller(initial_state, config)
        return SolveRecord(
            model=solved.model,
            latency_s=time.perf_counter() - started,
            solver_failures=1,
            solver_recoveries=1,
        )
    return SolveRecord(
        model=solved.model,
        latency_s=time.perf_counter() - started,
    )


def _component_value(
    component: VarData,
    values: pyo.ComponentMap | None,
) -> float:
    """Read one model value, optionally from a sensitivity estimate."""
    if values is None:
        value = pyo.value(component, exception=False)
    else:
        value = values[component]
    if value is None or not np.isfinite(float(value)):
        raise RuntimeError(f"non-finite trajectory value for {component.name}")
    return float(value)


def _profile_value(
    component: pyo.Var,
    requested: float,
    values: pyo.ComponentMap | None,
) -> float:
    """Read the piecewise-constant value held at a collocation point."""
    indices = sorted(float(index) for index in component)
    held_index = max(index for index in indices if index <= requested + 1.0e-12)
    return _component_value(component[held_index], values)


def trajectory(
    model: pyo.ConcreteModel,
    values: pyo.ComponentMap | None = None,
) -> Trajectory:
    """Extract a model or corrected estimate on the collocation grid."""
    grid = sorted(model.t)
    return Trajectory(
        time_min=np.asarray(grid, dtype=float),
        concentration=np.asarray(
            [_component_value(model.zc[t], values) for t in grid],
            dtype=float,
        ),
        temperature=np.asarray(
            [_component_value(model.zt[t], values) for t in grid],
            dtype=float,
        ),
        coolant=np.asarray(
            [_profile_value(model.v1, float(t), values) for t in grid],
            dtype=float,
        ),
        residence=np.asarray(
            [_profile_value(model.v2, float(t), values) for t in grid],
            dtype=float,
        ),
    )


def _event_label(event: object) -> str:
    """Stable compact label for one active-set path event."""
    variable = getattr(event.var, "name", str(event.var))
    return f"{variable}:{event.bound}:{event.action}@{float(event.fraction):.6f}"


def _corrected_update(
    background: pyo.ConcreteModel,
    measured_state: Sequence[float],
    policy: Policy,
    config: CstrConfig,
) -> tuple[
    Trajectory,
    object | None,
    tuple[str, ...],
    float,
    pyo.ComponentMap | None,
]:
    """Apply one held-factor update and collect its diagnostics."""
    perturbation = [
        (background.zc0, float(measured_state[0])),
        (background.zt0, float(measured_state[1])),
    ]
    if policy == "no_correction":
        return trajectory(background), None, (), 0.0, None

    mode = {
        "clamped_linear": "linear",
        "fix_relax": "fix_relax",
        "path": "path",
        "guarded_path": "path",
    }[policy]
    corrector_iter = 0 if policy == "clamped_linear" else config.corrector_iter
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        started = time.perf_counter()
        report = estimate_report(
            background,
            perturbation,
            mode=mode,
            predictor_iter=config.predictor_iter,
            degeneracy="directional",
            degeneracy_iter=config.degeneracy_iter,
            corrector_iter=corrector_iter,
        )
        values = estimate(
            background,
            perturbation,
            mode=mode,
            clamp=True,
            predictor_iter=config.predictor_iter,
            degeneracy="directional",
            degeneracy_iter=config.degeneracy_iter,
            corrector_iter=corrector_iter,
        )
        latency_s = time.perf_counter() - started

        # The public report and estimate APIs each replay the predictor and
        # corrector, so the timed region deliberately includes both.  The
        # event ledger is a third replay used for diagnostics and the
        # illustrative path-budget check.  Keep it outside the published
        # update timer and document that an end-to-end guard must add its cost.
        events = ()
        if policy in ("path", "guarded_path"):
            events = tuple(
                _event_label(event)
                for event in active_set_changes(
                    background,
                    perturbation,
                    predictor_iter=config.predictor_iter,
                    degeneracy="directional",
                    degeneracy_iter=config.degeneracy_iter,
                )
            )
    return trajectory(background, values), report, events, latency_s, values


def _load_estimate(values: pyo.ComponentMap) -> None:
    """Materialize an accepted estimate as the next primal warm start."""
    for variable, value in values.items():
        variable.set_value(float(value), skip_validation=True)


def _ambiguous_activity(
    report: object,
    *,
    decision_only: bool = False,
) -> tuple[str, ...]:
    """Names whose bound activity is not uniquely differentiable.

    The full report is retained for diagnosis.  The online guard acts on
    manipulated-variable ambiguity: interior collocation states can be in the
    classifier's low-multiplier ambiguity band without representing a control
    active-set decision.
    """
    return tuple(
        name
        for name, status in report.activity.items()
        if status in ("ambiguous", "weakly_active", "unidentified")
        and (not decision_only or name.startswith(("v1[", "v2[")))
    )


def _guard_reasons(
    report: object,
    corrected: Trajectory,
    predicted_state: Sequence[float],
    measured_state: Sequence[float],
    events: Sequence[str],
    config: CstrConfig,
) -> tuple[str, ...]:
    """Evaluate the full-point, trust-region, and safety guard."""
    reasons: list[str] = []
    displacement = max(
        abs(float(measured_state[0]) - float(predicted_state[0]))
        / config.concentration_scale,
        abs(float(measured_state[1]) - float(predicted_state[1]))
        / config.temperature_scale,
    )
    if displacement > config.measurement_trust_radius:
        reasons.append("measurement_displacement")
    corrector = report.corrector
    if corrector is None:
        if float(report.violation) > config.feasibility_tol:
            reasons.append("primal_feasibility")
        reasons.append("missing_full_point_residual")
    else:
        if float(corrector["feasibility"]) > config.feasibility_tol:
            reasons.append("corrector_feasibility")
        if float(corrector["stationarity"]) > config.stationarity_tol:
            reasons.append("stationarity")
        if float(corrector["complementarity"]) > config.complementarity_tol:
            reasons.append("complementarity")
        initial = float(corrector["initial_residual"])
        final = float(corrector["residual"])
        residual_floor = max(
            config.feasibility_tol,
            config.stationarity_tol,
            config.complementarity_tol,
        )
        if final > 0.5 * initial and final > residual_floor:
            reasons.append("corrector_no_progress")

    if len(events) >= config.predictor_iter:
        reasons.append("path_budget")
    if float(corrected.temperature.max()) > (
        config.temperature_limit + config.feasibility_tol
    ):
        reasons.append("predicted_temperature")
    if config.reject_ambiguous_activity and _ambiguous_activity(
        report, decision_only=True
    ):
        reasons.append("ambiguous_activity")
    return tuple(dict.fromkeys(reasons))


def _empty_diagnostics(
    *,
    update_latency_s: float,
    full_solve_latency_s: float | None,
) -> CorrectionDiagnostics:
    """Diagnostics for a policy that did not take a sensitivity step."""
    return CorrectionDiagnostics(
        accepted=True,
        reasons=(),
        update_latency_s=update_latency_s,
        full_solve_latency_s=full_solve_latency_s,
        primal_violation=0.0,
        stationarity=0.0,
        feasibility=0.0,
        complementarity=0.0,
        residual=0.0,
        initial_residual=0.0,
        path_events=(),
        corrector_iterations=0,
        refine_stop=None,
        ambiguous_activity=(),
    )


def _diagnostics(
    *,
    report: object,
    accepted: bool,
    reasons: tuple[str, ...],
    update_latency_s: float,
    full_solve_latency_s: float | None,
    events: tuple[str, ...],
) -> CorrectionDiagnostics:
    """Translate an EstimateReport into a durable event record."""
    corrector = report.corrector or {}
    return CorrectionDiagnostics(
        accepted=accepted,
        reasons=reasons,
        update_latency_s=update_latency_s,
        full_solve_latency_s=full_solve_latency_s,
        primal_violation=float(report.violation),
        stationarity=float(corrector.get("stationarity", 0.0)),
        feasibility=float(corrector.get("feasibility", 0.0)),
        complementarity=float(corrector.get("complementarity", 0.0)),
        residual=float(corrector.get("residual", 0.0)),
        initial_residual=float(corrector.get("initial_residual", 0.0)),
        path_events=events,
        corrector_iterations=int(corrector.get("iterations", 0)),
        refine_stop=report.refine_stop,
        ambiguous_activity=_ambiguous_activity(report),
    )


def _plant_rhs(
    state: np.ndarray,
    control: Sequence[float],
    *,
    feed_temperature_shift: float,
    reaction_rate_scale: float,
    heat_transfer_scale: float,
) -> np.ndarray:
    """Independent plant right-hand side [state fraction/min]."""
    zc, zt = (float(state[0]), float(state[1]))  # [-]
    v1, v2 = (float(control[0]), float(control[1]))  # [-]
    residence_min = 40.0 * v2  # [min]
    reaction_per_min = 300.0 * reaction_rate_scale * zc * np.exp(-5.0 / zt)
    cooling_per_min = 1.95e-4 * heat_transfer_scale * 600.0 * v1 * (zt - 0.38)
    return np.asarray(
        [
            (1.0 - zc) / residence_min - reaction_per_min,
            (0.395 + feed_temperature_shift - zt) / residence_min
            + reaction_per_min
            - cooling_per_min,
        ],
        dtype=float,
    )


def propagate_plant(
    state: Sequence[float],
    control: Sequence[float],
    config: CstrConfig,
    *,
    feed_temperature_shift: float = 0.0,
    reaction_rate_scale: float = 1.0,
    heat_transfer_scale: float = 1.0,
) -> tuple[float, float]:
    """Advance the plant independently with fixed-step RK4.

    Controller and plant share the nominal Hicks--Ray equations, but the
    integration implementation and optional parameter mismatches are separate.
    """
    x = np.asarray(state, dtype=float)
    dt_min = config.sample_time_min / config.plant_substeps
    rhs_kwargs = {
        "feed_temperature_shift": float(feed_temperature_shift),
        "reaction_rate_scale": float(reaction_rate_scale),
        "heat_transfer_scale": float(heat_transfer_scale),
    }
    for _ in range(config.plant_substeps):
        k1 = _plant_rhs(x, control, **rhs_kwargs)
        k2 = _plant_rhs(x + 0.5 * dt_min * k1, control, **rhs_kwargs)
        k3 = _plant_rhs(x + 0.5 * dt_min * k2, control, **rhs_kwargs)
        k4 = _plant_rhs(x + dt_min * k3, control, **rhs_kwargs)
        x = x + dt_min * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
    return float(x[0]), float(x[1])


def make_campaigns(steps: int = 30, seed: int = 19) -> tuple[Campaign, ...]:
    """Create deterministic nominal, switching, and stress campaigns."""
    if steps < 6:
        raise ValueError("steps must be at least 6 to stage all campaigns")
    rng = np.random.default_rng(seed)

    nominal_noise = rng.normal(loc=0.0, scale=(2.0e-4, 1.0e-4), size=(steps, 2))

    switching_feed = np.zeros(steps, dtype=float)
    switching_feed[steps // 3 : 2 * steps // 3] = 0.080
    switching_feed[2 * steps // 3 :] = -0.030
    switching_noise = rng.normal(loc=0.0, scale=(4.0e-4, 2.0e-4), size=(steps, 2))

    stress_feed = np.zeros(steps, dtype=float)
    stress_feed[steps // 3 :] = 0.025
    stress_noise = rng.normal(loc=0.0, scale=(8.0e-4, 4.0e-4), size=(steps, 2))

    def noise_tuple(values: np.ndarray) -> tuple[tuple[float, float], ...]:
        """Convert an N-by-2 noise array to immutable state pairs [-]."""
        return tuple((float(row[0]), float(row[1])) for row in values)

    return (
        Campaign(
            name="nominal",
            feed_temperature_shift=tuple(0.0 for _ in range(steps)),
            measurement_noise=noise_tuple(nominal_noise),
        ),
        Campaign(
            name="constraint_switching",
            feed_temperature_shift=tuple(float(v) for v in switching_feed),
            measurement_noise=noise_tuple(switching_noise),
        ),
        Campaign(
            name="stress_model_mismatch",
            feed_temperature_shift=tuple(float(v) for v in stress_feed),
            measurement_noise=noise_tuple(stress_noise),
            # The concentration bias is 4.5 times the local trust scale.  This
            # campaign deliberately forces an out-of-validity fallback; it is
            # not evidence that the guard detects subtle model mismatch.
            measurement_bias=(0.180, 0.016),
            reaction_rate_scale=1.08,
            heat_transfer_scale=0.92,
        ),
    )


def benchmark_stamp(
    config: CstrConfig,
    *,
    warmup_excluded: bool,
    repository: str | Path | None = None,
) -> BenchmarkStamp:
    """Capture the source and platform identity beside measured timings."""
    workdir = Path(repository) if repository is not None else Path.cwd()
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=workdir,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unavailable"
    return BenchmarkStamp(
        pounce_commit=commit,
        model_revision=MODEL_REVISION,
        solver_tolerance=config.solver_tol,
        platform=platform.platform(),
        python=platform.python_version(),
        warmup_excluded=warmup_excluded,
    )


def _measure_state(
    plant_state: Sequence[float],
    campaign: Campaign,
    sample: int,
    config: CstrConfig,
) -> tuple[float, float]:
    """Apply deterministic sensor bias/noise and admissible-domain clipping.

    Temperature is capped at the controller's own upper bound because the
    direct-transcription model pins the measured initial state to a variable
    with that bound and otherwise rejects the model before a solve.  A plant
    interlock must handle a real measurement above that limit; this example
    does not treat clipping as a safety decision.
    """
    noise = campaign.measurement_noise[sample]
    measured = (
        float(plant_state[0]) + campaign.measurement_bias[0] + noise[0],
        float(plant_state[1]) + campaign.measurement_bias[1] + noise[1],
    )
    return (
        float(np.clip(measured[0], 0.0, 1.0)),
        float(np.clip(measured[1], 1.0e-6, config.temperature_limit)),
    )


def _closed_loop_metrics(
    samples: Sequence[SampleRecord],
    config: CstrConfig,
    fallback_count: int,
) -> ClosedLoopMetrics:
    """Aggregate one policy/scenario timeline."""
    if not samples:
        raise ValueError("cannot summarize an empty closed loop")
    state_error = np.asarray(
        [
            (
                sample.plant_state_after[0] - ZC_SS,
                sample.plant_state_after[1] - ZT_SS,
            )
            for sample in samples
        ],
        dtype=float,
    )
    controls = np.asarray([sample.applied_control for sample in samples], dtype=float)
    previous_controls = np.vstack(
        [np.asarray((V1_SS, V2_SS), dtype=float), controls[:-1]]
    )
    update_latencies = np.asarray(
        [sample.diagnostics.update_latency_s for sample in samples],
        dtype=float,
    )
    full_latencies = np.asarray(
        [
            sample.diagnostics.full_solve_latency_s
            for sample in samples
            if sample.diagnostics.full_solve_latency_s is not None
        ],
        dtype=float,
    )
    background_latencies = np.asarray(
        [
            sample.background_latency_s
            for sample in samples
            if sample.background_latency_s > 0.0
        ],
        dtype=float,
    )
    absolute_error = np.abs(state_error)
    squared_error = state_error**2
    stage_cost = (
        10.0 * squared_error[:, 0]
        + 2.0 * squared_error[:, 1]
        + (controls[:, 0] - V1_SS) ** 2
        + 0.5 * (controls[:, 1] - V2_SS) ** 2
    )
    return ClosedLoopMetrics(
        iae=float(config.sample_time_min * absolute_error.sum()),
        ise=float(config.sample_time_min * squared_error.sum()),
        economic_stage_cost=float(config.sample_time_min * stage_cost.sum()),
        total_control_movement=float(np.abs(controls - previous_controls).sum()),
        maximum_temperature_violation=float(
            max(
                0.0,
                max(
                    sample.plant_state_after[1] - config.temperature_limit
                    for sample in samples
                ),
            )
        ),
        active_set_changes=sum(
            len(sample.diagnostics.path_events) for sample in samples
        ),
        fallback_full_solves=int(fallback_count),
        fallback_fraction=float(fallback_count / len(samples)),
        update_latency_median_s=float(np.median(update_latencies)),
        update_latency_p95_s=float(np.percentile(update_latencies, 95)),
        full_solve_latency_median_s=(
            float(np.median(full_latencies)) if full_latencies.size else None
        ),
        background_latency_median_s=(
            float(np.median(background_latencies))
            if background_latencies.size
            else None
        ),
        deadline_s=config.deadline_s,
        deadline_misses=int((update_latencies > config.deadline_s).sum()),
        solver_failures=sum(sample.solver_failures for sample in samples),
        solver_recoveries=sum(sample.solver_recoveries for sample in samples),
    )


def run_closed_loop(
    policy: Policy,
    campaign: Campaign,
    config: CstrConfig | None = None,
    *,
    initial_state: Sequence[float] = BASE_STATE,
    stamp: BenchmarkStamp | None = None,
) -> ClosedLoopResult:
    """Run one policy through a deterministic closed-loop campaign.

    For sensitivity policies, every sample follows the same visible ordering:
    use the predicted-state background solve, measure, correct, validate, apply,
    propagate the independent plant, shift, and prepare the next background
    solve. A guarded rejection triggers a fresh solve at the measurement and
    uses that solution to reset the next warm start and factorization.
    """
    if policy not in POLICIES:
        raise ValueError(f"unknown policy {policy!r}")
    cfg = config or CstrConfig()
    plant_state = (float(initial_state[0]), float(initial_state[1]))
    predicted_state = plant_state
    background: SolveRecord | None = None
    previous_model: pyo.ConcreteModel | None = None
    if policy != "full_resolve":
        background = _solve_with_recovery(predicted_state, cfg)

    samples: list[SampleRecord] = []
    fallback_count = 0
    for sample in range(campaign.steps):
        measured_state = _measure_state(plant_state, campaign, sample, cfg)
        background_latency_s = 0.0 if background is None else background.latency_s
        solver_failures = 0 if background is None else background.solver_failures
        solver_recoveries = 0 if background is None else background.solver_recoveries

        if policy == "full_resolve":
            full = _solve_with_recovery(measured_state, cfg, previous_model)
            solver_failures += full.solver_failures
            solver_recoveries += full.solver_recoveries
            chosen = trajectory(full.model)
            diagnostics = _empty_diagnostics(
                update_latency_s=full.latency_s,
                full_solve_latency_s=full.latency_s,
            )
            chosen_model = full.model
        else:
            assert background is not None
            (
                chosen,
                report,
                events,
                update_latency_s,
                corrected_values,
            ) = _corrected_update(background.model, measured_state, policy, cfg)
            if policy == "no_correction":
                diagnostics = _empty_diagnostics(
                    update_latency_s=update_latency_s,
                    full_solve_latency_s=None,
                )
                chosen_model = background.model
            else:
                reasons: tuple[str, ...] = ()
                if policy == "guarded_path":
                    reasons = _guard_reasons(
                        report,
                        chosen,
                        predicted_state,
                        measured_state,
                        events,
                        cfg,
                    )
                if reasons:
                    fallback = _solve_with_recovery(
                        measured_state, cfg, background.model
                    )
                    solver_failures += fallback.solver_failures
                    solver_recoveries += fallback.solver_recoveries
                    chosen = trajectory(fallback.model)
                    chosen_model = fallback.model
                    fallback_count += 1
                    diagnostics = _diagnostics(
                        report=report,
                        accepted=False,
                        reasons=reasons,
                        update_latency_s=update_latency_s + fallback.latency_s,
                        full_solve_latency_s=fallback.latency_s,
                        events=events,
                    )
                else:
                    assert corrected_values is not None
                    _load_estimate(corrected_values)
                    chosen_model = background.model
                    diagnostics = _diagnostics(
                        report=report,
                        accepted=True,
                        reasons=(),
                        update_latency_s=update_latency_s,
                        full_solve_latency_s=None,
                        events=events,
                    )

        applied_control = chosen.first_control
        next_plant_state = propagate_plant(
            plant_state,
            applied_control,
            cfg,
            feed_temperature_shift=campaign.feed_temperature_shift[sample],
            reaction_rate_scale=campaign.reaction_rate_scale,
            heat_transfer_scale=campaign.heat_transfer_scale,
        )
        next_predicted_state = chosen.state_at(cfg.sample_time_min)
        samples.append(
            SampleRecord(
                sample=sample,
                predicted_state=predicted_state,
                measured_state=measured_state,
                plant_state_before=plant_state,
                applied_control=applied_control,
                plant_state_after=next_plant_state,
                background_latency_s=background_latency_s,
                solver_failures=solver_failures,
                solver_recoveries=solver_recoveries,
                diagnostics=diagnostics,
            )
        )

        plant_state = next_plant_state
        predicted_state = next_predicted_state
        previous_model = chosen_model
        if policy != "full_resolve" and sample + 1 < campaign.steps:
            background = _solve_with_recovery(
                predicted_state, cfg, warm_start=chosen_model
            )

    return ClosedLoopResult(
        policy=policy,
        campaign=campaign.name,
        samples=samples,
        metrics=_closed_loop_metrics(samples, cfg, fallback_count),
        stamp=stamp or benchmark_stamp(cfg, warmup_excluded=False),
    )


def run_campaigns(
    config: CstrConfig | None = None,
    *,
    steps: int = 30,
    seed: int = 19,
    policies: Sequence[Policy] = POLICIES,
    warmup: bool = True,
    repository: str | Path | None = None,
) -> list[ClosedLoopResult]:
    """Run every selected policy on the same three deterministic campaigns."""
    cfg = config or CstrConfig()
    for policy in policies:
        if policy not in POLICIES:
            raise ValueError(f"unknown policy {policy!r}")
    campaigns = make_campaigns(steps=steps, seed=seed)
    if warmup:
        warm = solve_controller(BASE_STATE, cfg)
        _corrected_update(
            warm.model,
            (BASE_STATE[0] + 1.0e-4, BASE_STATE[1] + 1.0e-4),
            "path",
            cfg,
        )
    stamp = benchmark_stamp(cfg, warmup_excluded=warmup, repository=repository)
    return [
        run_closed_loop(policy, campaign, cfg, stamp=stamp)
        for campaign in campaigns
        for policy in policies
    ]


def directional_degeneracy_experiment(
    config: CstrConfig | None = None,
) -> DegeneracyExperiment:
    """Take two directional updates from the first active-set breakpoint.

    The held point is solved at the first breakpoint along the original
    notebook's upset-to-steady-state path. Equal-magnitude perturbations in
    opposite directions then demonstrate that the derivative is directional,
    not a unique sensitivity at the kink. The same acceptance guard sees the
    weak or ambiguous activity and requests a fresh solve.
    """
    cfg = config or CstrConfig()
    base = solve_controller(BASE_STATE, cfg)
    full_state = (ZC_SS, ZT_SS)
    full_perturbation = [
        (base.model.zc0, full_state[0]),
        (base.model.zt0, full_state[1]),
    ]
    record = active_set_changes(
        base.model,
        full_perturbation,
        predictor_iter=cfg.predictor_iter,
        degeneracy="directional",
        degeneracy_iter=cfg.degeneracy_iter,
    )
    if not record:
        raise RuntimeError("the configured horizon produced no active-set breakpoint")
    first = record[0]
    fraction = float(first.fraction)
    probe_time_min = float(first.var.index())
    breakpoint_state = (
        BASE_STATE[0] + fraction * (ZC_SS - BASE_STATE[0]),
        BASE_STATE[1] + fraction * (ZT_SS - BASE_STATE[1]),
    )
    held = solve_controller(breakpoint_state, cfg, base.model)
    direction = np.asarray((ZC_SS - BASE_STATE[0], ZT_SS - BASE_STATE[1]), dtype=float)

    steps: list[DegeneracyStep] = []
    for label, sign in (("toward_setpoint", 1.0), ("toward_upset", -1.0)):
        target = np.asarray(breakpoint_state) + sign * 0.02 * direction
        perturbation = [
            (held.model.zc0, float(target[0])),
            (held.model.zt0, float(target[1])),
        ]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            one_sided = estimate(
                held.model,
                perturbation,
                mode="path",
                degeneracy="one_sided",
                predictor_iter=cfg.predictor_iter,
                corrector_iter=0,
            )
            directional = estimate(
                held.model,
                perturbation,
                mode="path",
                degeneracy="directional",
                degeneracy_iter=cfg.degeneracy_iter,
                predictor_iter=cfg.predictor_iter,
                corrector_iter=0,
            )
            report = estimate_report(
                held.model,
                perturbation,
                mode="path",
                degeneracy="directional",
                degeneracy_iter=cfg.degeneracy_iter,
                predictor_iter=cfg.predictor_iter,
                corrector_iter=cfg.corrector_iter,
            )
            events = tuple(
                _event_label(event)
                for event in active_set_changes(
                    held.model,
                    perturbation,
                    predictor_iter=cfg.predictor_iter,
                    degeneracy="directional",
                    degeneracy_iter=cfg.degeneracy_iter,
                )
            )
        one_sided_trajectory = trajectory(held.model, one_sided)
        directional_trajectory = trajectory(held.model, directional)
        reasons = _guard_reasons(
            report,
            directional_trajectory,
            breakpoint_state,
            target,
            events,
            cfg,
        )
        steps.append(
            DegeneracyStep(
                direction=label,
                target_state=(float(target[0]), float(target[1])),
                probe_time_min=probe_time_min,
                one_sided_probe_control=one_sided_trajectory.control_at(probe_time_min),
                directional_probe_control=directional_trajectory.control_at(
                    probe_time_min
                ),
                directional_events=events,
                guard_reasons=reasons,
            )
        )

    variable = getattr(first.var, "name", str(first.var))
    return DegeneracyExperiment(
        breakpoint_fraction=fraction,
        breakpoint_variable=variable,
        breakpoint_bound=first.bound,
        steps=(steps[0], steps[1]),
    )


__all__ = [
    "BASE_STATE",
    "MODEL_REVISION",
    "POLICIES",
    "V1_MIN",
    "V1_SS",
    "V2_SS",
    "ZC_SS",
    "ZT_SS",
    "BenchmarkStamp",
    "Campaign",
    "ClosedLoopMetrics",
    "ClosedLoopResult",
    "CorrectionDiagnostics",
    "CstrConfig",
    "DegeneracyExperiment",
    "DegeneracyStep",
    "Policy",
    "SampleRecord",
    "SolveRecord",
    "Trajectory",
    "benchmark_stamp",
    "build_controller",
    "directional_degeneracy_experiment",
    "make_campaigns",
    "propagate_plant",
    "run_campaigns",
    "run_closed_loop",
    "shift_warm_start",
    "solve_controller",
    "trajectory",
]
