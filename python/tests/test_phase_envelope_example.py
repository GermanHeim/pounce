"""Regression tests for the differentiable phase-envelope example."""

import numpy as np
import pytest

from pounce.examples.phase_envelope import (
    DEITERS_BELL_METHANE_PROPANE,
    NATURAL_GAS,
    _fold_equations,
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
from pounce.jax import PathTrace


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


def test_fold_equations_ignore_the_mixture_composition():
    """Pin the invariant ``_fold_problem``'s cache key rests on (gh#788).

    ``refine_fold`` always supplies a design vector, and ``_decode_design``
    reads the composition from *that*, ignoring ``mixture.composition``.
    ``_fold_problem`` therefore leaves the composition out of its cache key,
    so the inverse design's verification fold reuses the compiled graph
    instead of paying a second ~2-minute XLA compile of the augmented
    system's third derivative.

    That invariant lives in a different function, so it is checked here
    rather than trusted.  This test is unmarked and cheap: it evaluates the
    fold equations, never their Lagrangian Hessian, so it runs on every
    pull request and goes red the moment the residual starts reading the
    mixture's own composition.
    """

    mixture = DEITERS_BELL_METHANE_PROPANE
    kij_pair = (0, 1)
    design = design_parameters(mixture, kij_pair)
    n_fold = 2 * (mixture.n_components + 1) + 1
    fold_state = np.linspace(0.1, 0.9, n_fold)

    other = mixture.with_composition(np.array([0.35, 0.65]))
    assert not np.allclose(other.composition, mixture.composition)

    baseline = np.asarray(
        _fold_equations(fold_state, design, mixture, 0.0, "temperature", kij_pair)
    )
    shifted = np.asarray(
        _fold_equations(fold_state, design, other, 0.0, "temperature", kij_pair)
    )
    np.testing.assert_array_equal(baseline, shifted)


def test_reparameterize_trace_swaps_the_parameter_and_its_state_slot():
    """Reparameterization rearranges converged coordinates, and only those.

    This ran only inside the slow published-value test until gh#788 dropped
    the pressure-mode fold it fed.  The function is pure array
    rearrangement, so a synthetic trace covers it exactly and costs no
    continuation run.
    """

    n = DEITERS_BELL_METHANE_PROPANE.n_components
    theta = np.array([0.0, 1.0, 2.0, 1.5, 0.5])  # [ln K], with one reversal
    state = np.column_stack(
        [np.arange(5.0), np.arange(5.0) + 10.0, np.array([7.0, 8.0, 9.0, 8.5, 7.5])]
    )  # [ln K_1, ln K_2, ln P]
    trace = PathTrace(
        s=np.arange(5.0),
        theta=theta,
        x=state,
        lam=np.zeros((5, n + 1)),
        n_steps=5,
        n_correctors=5,
        n_accepts=5,
        turning_points=[2.0],
    )

    swapped = reparameterize_trace(
        trace, DEITERS_BELL_METHANE_PROPANE, from_mode="temperature", to_mode="pressure"
    )

    # The old state slot n becomes the parameter; the old parameter takes it.
    np.testing.assert_array_equal(swapped.theta, state[:, n])
    np.testing.assert_array_equal(swapped.x[:, :n], state[:, :n])
    np.testing.assert_array_equal(swapped.x[:, n], theta)
    # Turning points are re-detected in the new parameter, not carried over.
    assert swapped.turning_points == [9.0]
    assert swapped.active_set_changes == []

    # Same mode is a copy, not a rearrangement.
    same = reparameterize_trace(
        trace, DEITERS_BELL_METHANE_PROPANE, from_mode="pressure", to_mode="pressure"
    )
    np.testing.assert_array_equal(same.theta, theta)
    np.testing.assert_array_equal(same.x, state)

    with pytest.raises(ValueError, match="from_mode"):
        reparameterize_trace(
            trace, DEITERS_BELL_METHANE_PROPANE, from_mode="bogus", to_mode="pressure"
        )


def _sampled_extremum(s, theta):
    """Locate a sampled extremum of ``theta(s)`` by parabolic interpolation.

    The traced points bracket the fold to the sampling resolution, so the
    plain sample maximum carries an O(ds^2) error and is not a
    step-size-independent quantity.  Fitting a parabola through the peak
    sample and its two neighbours removes the leading term, which is what
    makes a coarse and a fine trace comparable without the augmented
    ``refine_fold`` solve and its third-derivative XLA compile.

    Returns ``(s_at_extremum, theta_at_extremum)``.  The stencil is exact
    for a quadratic, so a step-size sweep over one recovers both to
    machine precision.
    """

    s = np.asarray(s, dtype=float)
    theta = np.asarray(theta, dtype=float)
    peak = int(np.argmax(theta))
    if not 0 < peak < len(theta) - 1:
        raise ValueError("the extremum is not bracketed by the trace")
    step = s[peak] - s[peak - 1]
    left, middle, right = theta[peak - 1], theta[peak], theta[peak + 1]
    curvature = left - 2.0 * middle + right
    offset = 0.5 * (left - right) / curvature  # [-], in units of ``step``
    return s[peak] + offset * step, middle - 0.25 * (left - right) * offset


def test_parabolic_extremum_is_exact_on_a_quadratic():
    """The stencil the step-size legs rest on is itself step-size blind.

    If it were not, the coarse/fine comparison below would be measuring
    the stencil rather than the continuation, and would still pass.  It is
    mutation-checked: degrading the stencil to the plain sample maximum --
    which is what the two traces would agree on for a reason that has
    nothing to do with convergence, since halving ``ds`` keeps every
    coarse sample -- turns this test red.
    """

    def quadratic(s):
        return 3.5 - 0.75 * (s - 1.234) ** 2

    for step in (0.1, 0.05, 0.017):
        grid = np.arange(0.0, 3.0, step)
        s_star, theta_star = _sampled_extremum(grid, quadratic(grid))
        assert abs(s_star - 1.234) < 1e-12
        assert abs(theta_star - 3.5) < 1e-12

    # An extremum at the end of the sampled window is not bracketed, and
    # saying so beats fitting a parabola to a one-sided stencil.
    rising = np.arange(0.0, 1.0, 0.1)
    with pytest.raises(ValueError, match="not bracketed"):
        _sampled_extremum(rising, rising)


@pytest.mark.slow
def test_traced_fold_is_invariant_to_the_continuation_step():
    """Assert step-size invariance on the trace alone (gh#798).

    gh#788 dropped the ds=0.10/ds=0.08 pair from the published-value test
    below, and with it the only automated assertion that the
    phase-envelope example's extremum does not depend on how finely the
    envelope was traced.  That pair cost a full ~196 s XLA compile of the
    augmented fold system's Lagrangian Hessian, because the coarse
    pressure-mode fold existed only as one side of it.  This restores the
    claim without that compile: both legs stop at the *trace*, and neither
    calls ``refine_fold``.

    What it costs is two envelope traces, ~50 s, essentially all of it two
    JIT compilations of the residual and its Jacobian plus two
    low-pressure anchor solves.  A trace is length-blind -- 145 steps
    measured 24.5 s against 24.0 s for 5, about 0.003 s a step after the
    compile -- so ~50 s is the floor for any two-trace comparison, and it
    carries the ``slow`` marker the rest of this claim's evidence carries.
    It is a statement about the *example*, which only the example, the JAX
    frontend, or the solver can move, and that is the category the
    marker's CI selection rule is built around.

    The fine leg halves ``ds`` rather than taking the historical 0.08, so
    every coarse point has an exact arclength twin -- ``trace_arclength``
    stamps ``s = arange(K) * ds`` -- and the comparison carries no
    interpolation error of its own.  Two things are then checked, because
    two traces can agree pointwise while disagreeing about the extremum
    they bracket, and can agree about the extremum while having taken
    different paths to it:

    1. the state at matched arclength, which drifts only through the
       O(ds^2) difference between a chord of the curve and its arc; and
    2. the interpolated maxcondentherm, which is the quantity the dropped
       assertion was actually about.

    Both legs are mutation-checked together by biasing the predictor to
    ``ds * (1 + 0.02 * ds)``, which leaves every point converged on the
    same curve with the same extremum -- status ``ok``.  Leg 1 moves to
    1.5e-3 [ln K] and 1.2e-2 [mixed], 3x and 2x their thresholds and ~19x
    and ~13x the clean values.  In leg 2 the extremum's *arclength* moves
    to 1.3e-2, 6x its threshold, while its *temperature* stays at 4.9e-4 K
    and green: the fitted vertex value is blind to a uniform rescaling of
    the abscissa, so the temperature alone would have missed this, which
    is why the arclength is asserted beside it.

    Agreeing with each other is not enough, since two traces can agree on
    the wrong branch, so both are also required to bracket the published
    Deiters--Bell extremum.  That is a branch guard here, not a
    published-value check: the value itself is refined and asserted by
    ``test_published_binary_fold_and_inverse_design_regression``.
    """

    mixture = DEITERS_BELL_METHANE_PROPANE
    coarse = trace_envelope(mixture, beta=0.0, mode="temperature", ds=0.1, n_steps=145)
    fine = trace_envelope(mixture, beta=0.0, mode="temperature", ds=0.05, n_steps=290)

    assert coarse.status == "ok"
    assert fine.status == "ok"

    # Every other fine point sits at a coarse point's arclength exactly.
    twin = slice(None, 2 * len(coarse.s), 2)
    np.testing.assert_allclose(fine.s[twin], coarse.s, rtol=0.0, atol=1e-12)

    # Measured on the reference build: 7.7e-5 [ln K] and 9.1e-4 [mixed].
    # The drift grows along the path, as accumulated chord-versus-arc slip
    # does, and is largest approaching the fold.
    assert np.max(np.abs(fine.theta[twin] - coarse.theta)) < 5e-4  # [ln K]
    assert np.max(np.abs(fine.x[twin] - coarse.x)) < 5e-3  # [mixed]

    coarse_s, coarse_theta = _sampled_extremum(coarse.s, coarse.theta)
    fine_s, fine_theta = _sampled_extremum(fine.s, fine.theta)
    coarse_fold_k = float(np.exp(coarse_theta))  # [K]
    fine_fold_k = float(np.exp(fine_theta))  # [K]

    # Measured: 7.5e-4 K apart, at 13.21945 and 13.21910 arclength.
    assert abs(coarse_fold_k - fine_fold_k) < 5e-3  # [K]
    assert abs(coarse_s - fine_s) < 2e-3  # [-]

    # Deiters & Bell (AIChE J. 65, 2019, e16730), as below.  Both legs must
    # bracket *that* extremum; agreement on some other branch is not
    # step-size invariance of this example.
    assert abs(coarse_fold_k - 282.53) < 0.02  # [K]
    assert abs(fine_fold_k - 282.53) < 0.02  # [K]
    assert min(coarse_fold_k, fine_fold_k) > float(np.exp(coarse.theta[0]))  # [K]


@pytest.mark.slow
def test_published_binary_fold_and_inverse_design_regression():
    """Reproduce a published maxcondentherm and solve an inverse design.

    Marked slow because it is the most expensive test in the suite by an
    order of magnitude, nearly all of it XLA compiling the Lagrangian
    Hessian of the augmented fold equations, which is a third derivative
    of the Peng--Robinson residual.  Two such compiles remain -- the
    temperature fold and the inverse-design NLP -- and both are load
    bearing: they are the published benchmark and the design solve.

    gh#788 cut the other two.  ``_fold_problem``'s cache is keyed
    blind to composition, so the verification retrace's fold reuses the
    temperature graph instead of recompiling it.  And the ds=0.10/ds=0.08
    step-size-convergence pair is gone, taking the pressure-mode fold with
    it -- that fold existed only as one side of that comparison.  Notebook
    34 cell 38 still publishes the coarse/fine agreement (8.41e-12), but a
    notebook is not executed by any workflow, so gh#798 put the step-size
    claim back under an assertion in
    ``test_traced_fold_is_invariant_to_the_continuation_step`` above, at
    the trace level and for ~50 s rather than ~196 s.  The properties the
    #777 review established all survive here: the fold is still reached by
    a real continuation from a real low-pressure anchor, the published
    Deiters--Bell value is still checked, and the inverse design is still
    verified by a fresh trace rather than by re-reading the NLP's own
    answer.

    What it asserts is a *published-value* and *inverse-design* claim about
    the example, not a claim about the solver's arithmetic, and it can only
    move when the example, the JAX frontend, or the solver itself moves.
    So it is deselected on pull requests that touch none of those, and runs
    in full on every push to `main`.  The unmarked tests in this file still
    cover the cubic root branches, the simplex coordinates, the implicit
    derivatives, trace reparameterization, the extremum stencil the
    step-size test rests on, and the composition invariant the fold cache
    depends on, on every PR.
    """

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
