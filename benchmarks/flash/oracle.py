"""The independent flash: Michelsen stability, then Rachford--Rice.

This module is the reason the fixture can claim to be *validated* rather
than merely converged. gh#776 asks Gate 1 to "validate each regime
against an independent flash/stability calculation", and that means
something only if the second calculation does not reuse the first one's
reasoning.

What is shared, and what is not
-------------------------------

Shared: `thermo.ln_phi`, i.e. `phase_envelope.log_fugacity_coefficients`
and the cubic under it, and the phase-label root convention this module
documents below. Everything above that is a different algorithm reaching
the same answer by a different route:

===========================  ==========================  =========================
                             the MPCC (`spec` + solver)  this oracle
===========================  ==========================  =========================
phase detection              a complementarity pair      Michelsen TPD, multistart
the two-phase solve          an interior-point NLP on    successive substitution +
                             2nc+1 variables             Rachford--Rice, then a
                                                         Newton polish on (lnK,beta)
the stability certificate    one stationary point        every stationary point the
                                                         multistart reaches
===========================  ==========================  =========================

The last row is an asymmetry, and it is the right way round. The MPCC's
single-phase branch asserts that the *one* trial phase its solve landed
on has ``sum Y <= 1``; Michelsen's test is a statement about the global
minimum of the tangent-plane distance, which no single stationary point
establishes. Agreement therefore means the MPCC found the stationary
point that matters, and a disagreement is a finding about the MPCC
rather than about the oracle.

Why the cubic root is chosen by phase label
-------------------------------------------

The first draft of this module chose each root by comparing reduced
Gibbs energies, on the reasoning that an energy criterion is more
principled than a label and would make the two calculations independent
in one more dimension. Measured, it is simply wrong here, in two ways
that a green two-phase region would have hidden:

* **It collapses the trial phase onto the trivial solution wherever the
  feed's cubic has one real root.** At 230 K and at 350 K on the pinned
  path the feed has a single root, so "the other root" does not exist to
  compare against, and every start of the tangent-plane iteration
  converged to ``Y = z``. The oracle reported "stable, no incipient
  phase", which reads as a well-posed answer and is not one: the
  incipient phase is there, and iterating with the *opposite label* to
  the feed finds it at ``sum Y = 0.394`` (230 K) and ``0.622`` (350 K).
* **It makes the two-phase system symmetric.** With both phases at
  whichever root is lower in Gibbs energy, ``(x, y, beta)`` and
  ``(y, x, 1-beta)`` are both solutions, and at 310 K the Newton polish
  returned the mirror one -- ``beta = 0.339`` where the answer is
  ``0.661``. Labelling the roots breaks the symmetry, because the liquid
  is *defined* as the small root and the vapor as the large one.

So both calculations use the label convention, which is the ordinary one
and the one the model in `spec.py` uses. That costs a dimension of
independence, and the honest replacement is to check the convention
rather than to vary it: `thermo.root_diagnostics` compares the
label-selected root against the lower-Gibbs one at every point where two
roots exist, and `validate.py` records the verdict. The guard gh#776
asks for is that check, not the choice.

How the regime is decided
--------------------------

Not by a heuristic on ``Z``, and not by which root is lower. The
question "which single phase is this?" is asked as two tangent-plane
tests, one per phase label, and they are exactly the two slacks of the
MPCC's two complementarity pairs:

* **stable as a liquid**: feed at the liquid root, trial phase at the
  vapor root, ``sum Y_V <= 1``. This is the MPCC's ``1 - Sy >= 0``.
* **stable as a vapor**: feed at the vapor root, trial phase at the
  liquid root, ``sum Y_L <= 1``. This is the MPCC's ``1 - Sx >= 0``.

Both unstable is the two-phase regime; exactly one stable names the
phase; both stable would be a metastable ambiguity and is reported as
one rather than resolved by a tie-break. Because these are the same two
numbers the MPCC reports as ``Sy`` and ``Sx``, the regime comparison
between oracle and solver is numeric rather than a label match.
"""

from __future__ import annotations

import dataclasses
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import brentq, root

from pounce.examples.phase_envelope import PengRobinsonMixture

from . import thermo

#: Successive substitution stops here; the Newton polish takes it the
#: rest of the way. SS alone converges linearly with a ratio that
#: approaches one near a switch point, and an oracle only as tight as
#: the thing it judges is not an oracle.
SS_TOL = 1e-11
SS_MAX_ITER = 1000

#: A trial phase whose ``sum Y`` exceeds one by more than this is a
#: genuine instability rather than a stationary point on the boundary.
#: At a switch point the true value *is* one, so this is the width of
#: the band in which the oracle reports the switch itself.
STABILITY_EPS = 1e-9

#: A trial phase this close to the feed composition is the trivial
#: stationary point ``Y = z``, which is always present and says nothing.
TRIVIAL_W = 1e-7


@dataclasses.dataclass
class TrialPhase:
    """One tangent-plane test: is the feed stable against this phase label?"""

    #: ``"vapor"`` tests a vapor-like trial phase against a liquid feed;
    #: ``"liquid"`` the other way round.
    trial_label: str
    #: ``sum_i Y_i`` at the most unstable stationary point found. This is
    #: the MPCC's ``Sy`` (vapor trial) or ``Sx`` (liquid trial).
    sum_y: float
    #: Tangent-plane distance there; negative means unstable.
    tpd: float
    stable: bool
    composition: Optional[np.ndarray]
    #: ``(start label, sum Y, tpd)`` for every start that converged, so
    #: a record can show the verdict came from more than one.
    stationary_points: List[Tuple[str, float, float]] = dataclasses.field(
        default_factory=list
    )


def _starts(z, temperature_k, pressure_pa, mixture) -> List[Tuple[str, np.ndarray]]:
    """Michelsen's recommended starts: vapor-like, liquid-like, and pure.

    The pure-component starts are what make this a multistart rather
    than a two-sided guess, and they are the ones that find the
    incipient phase at the ends of the path where the Wilson estimate is
    poor. Wilson's correlation appears here and nowhere else in this
    module: it supplies *starts*, never values, so nothing the oracle
    reports depends on its accuracy.
    """
    z = np.asarray(z, dtype=float)
    nc = z.size
    k = thermo.wilson_k(temperature_k, pressure_pa, mixture)
    out = [
        ("wilson_vapor", k * z),
        ("wilson_liquid", z / k),
        # Michelsen's cube-root pair. Not decoration: at 266 K on the
        # pinned path the four starts above it reach only the trivial
        # stationary point for the liquid trial phase, while the true
        # one sits at ``sum Y = 4.6``. A multistart that misses a
        # stationary point reports a weaker stability claim than it
        # could, and the whole value of this module is that its claim is
        # the stronger of the two.
        ("wilson_vapor_cbrt", z * np.cbrt(k)),
        ("wilson_liquid_cbrt", z / np.cbrt(k)),
    ]
    for i in range(nc):
        w = np.full(nc, 0.01 / max(nc - 1, 1))
        w[i] = 0.99
        out.append((f"pure_{mixture.names[i]}", w))
    return out


def tangent_plane(
    z,
    temperature_k: float,
    pressure_pa: float,
    mixture: PengRobinsonMixture,
    *,
    trial_is_vapor: bool,
) -> TrialPhase:
    """Michelsen's test on feed ``z`` against a trial phase of one label.

    Iterates ``ln Y_i = d_i - ln phi_i(Y/sum Y)`` from every start, with
    ``d_i = ln z_i + ln phi_i(z)`` the feed's tangent plane. The feed
    takes the root *opposite* the trial phase's, which is what makes
    this the test for "is the feed, as that phase, stable?".
    """
    z = np.asarray(z, dtype=float)
    d = np.log(z) + np.asarray(
        thermo.ln_phi(z, temperature_k, pressure_pa, mixture, largest=not trial_is_vapor)
    )

    best: Optional[Tuple[float, float, np.ndarray]] = None
    points: List[Tuple[str, float, float]] = []
    for label, y0 in _starts(z, temperature_k, pressure_pa, mixture):
        y = np.array(y0, dtype=float)
        converged = False
        for _ in range(SS_MAX_ITER):
            w = y / np.sum(y)
            lnp = np.asarray(
                thermo.ln_phi(w, temperature_k, pressure_pa, mixture, largest=trial_is_vapor)
            )
            y_new = np.exp(d - lnp)
            if not np.all(np.isfinite(y_new)) or np.any(y_new <= 0.0):
                break
            if np.max(np.abs(np.log(y_new / y))) < SS_TOL:
                y = y_new
                converged = True
                break
            y = y_new
        if not converged:
            continue
        s = float(np.sum(y))
        w = y / s
        if np.max(np.abs(np.log(w / z))) < TRIVIAL_W:
            points.append((f"{label} (trivial)", s, 0.0))
            continue
        lnp = np.asarray(
            thermo.ln_phi(w, temperature_k, pressure_pa, mixture, largest=trial_is_vapor)
        )
        tpd = float(1.0 + np.sum(y * (np.log(y) + lnp - d - 1.0)))
        points.append((label, s, tpd))
        if best is None or s > best[0]:
            best = (s, tpd, w)

    if best is None:
        # Only the trivial stationary point. The feed is stable against
        # this phase label and there is no incipient phase to report --
        # a different claim from "sum Y = 0", so the sum is reported as
        # the trivial point's own value of 1 and the composition as None.
        return TrialPhase(
            trial_label="vapor" if trial_is_vapor else "liquid",
            sum_y=1.0,
            tpd=0.0,
            stable=True,
            composition=None,
            stationary_points=points,
        )
    s, tpd, w = best
    return TrialPhase(
        trial_label="vapor" if trial_is_vapor else "liquid",
        sum_y=s,
        tpd=tpd,
        stable=bool(s <= 1.0 + STABILITY_EPS),
        composition=w,
        stationary_points=points,
    )


# --------------------------------------------------------------------
# the two-phase solve
# --------------------------------------------------------------------


def rachford_rice(z, k) -> float:
    """The vapor-fraction root of ``sum_i z_i (K_i-1)/(1 + b(K_i-1))``.

    Monotone decreasing in ``b``, so the root is unique where one
    exists, and the endpoints decide the single-phase cases without a
    bracket search.
    """
    z = np.asarray(z, dtype=float)
    k = np.asarray(k, dtype=float)

    def f(b):
        return float(np.sum(z * (k - 1.0) / (1.0 + b * (k - 1.0))))

    if f(0.0) <= 0.0:
        return 0.0
    if f(1.0) >= 0.0:
        return 1.0
    return float(brentq(f, 0.0, 1.0, xtol=1e-15, rtol=8.9e-16, maxiter=200))


@dataclasses.dataclass
class FlashResult:
    """The oracle's answer at one ``(z, T, P)``."""

    regime: str
    beta: float
    x: np.ndarray
    y: np.ndarray
    k: np.ndarray
    #: Directly comparable with the MPCC's ``Sx`` and ``Sy``: in a
    #: single-phase regime these are the tangent-plane tests' ``sum Y``,
    #: and in the two-phase regime both are one.
    sum_x: float
    sum_y: float
    #: Both tests, at every point, whichever regime won. Reporting the
    #: losing one is not padding: "stable as a vapor" and "unstable as a
    #: liquid" are separate facts, and a point where both tests come
    #: back stable is a metastable ambiguity that a single verdict would
    #: hide.
    vapor_trial: TrialPhase
    liquid_trial: TrialPhase
    converged: bool
    residual: float
    trivial: bool = False
    note: str = ""

    @property
    def ambiguous(self) -> bool:
        """Both labels stable *and* both incipient phases real.

        A metastable point, where the tangent-plane test alone does not
        name the phase. Distinct from `no_incipient_phase` below, which
        is the far more common reason for two "stable" verdicts and is
        not an ambiguity at all -- one of the tests simply had nothing
        to test against. Conflating the two is what made the first draft
        of this module flag five ordinary points on the pinned path.
        """
        return (
            self.regime != "two_phase"
            and self.vapor_trial.stable
            and self.liquid_trial.stable
            and self.vapor_trial.composition is not None
            and self.liquid_trial.composition is not None
        )

    @property
    def no_incipient_phase(self) -> bool:
        """Neither label has a non-trivial stationary point.

        A formulation boundary rather than a numerical one, and worth a
        field of its own because of what it does to the MPCC in
        `spec.py`. With no incipient phase, ``x = y = z`` is the only
        solution of the isofugacity rows, so ``Sx = Sy = 1``, *both*
        pair slacks vanish, and ``beta`` is left completely
        undetermined -- a one-parameter family of solutions rather than
        a degenerate point. The pinned path does not reach it (every
        point has an incipient phase for at least one label), and a
        record that ever reports it is reporting that the fixture has
        left the region where it means anything.
        """
        return self.vapor_trial.composition is None and self.liquid_trial.composition is None


def _two_phase(z, temperature_k, pressure_pa, mixture, k0) -> Tuple[np.ndarray, float, float]:
    """Successive substitution, then a Newton polish on ``(ln K, beta)``.

    The liquid is always the small cubic root and the vapor the large
    one, which is what keeps ``(x, y, beta)`` and ``(y, x, 1-beta)``
    from both being solutions. Returns ``(K, beta, residual)``.
    """
    z = np.asarray(z, dtype=float)
    k = np.array(k0, dtype=float)
    for _ in range(SS_MAX_ITER):
        beta = rachford_rice(z, k)
        x = z / (1.0 + beta * (k - 1.0))
        y = k * x
        xn, yn = x / np.sum(x), y / np.sum(y)
        lx = np.asarray(thermo.ln_phi(xn, temperature_k, pressure_pa, mixture, largest=False))
        ly = np.asarray(thermo.ln_phi(yn, temperature_k, pressure_pa, mixture, largest=True))
        k_new = np.exp(lx - ly)
        if np.max(np.abs(np.log(k_new / k))) < SS_TOL:
            k = k_new
            break
        k = k_new

    def residual(u):
        kk = np.exp(u[:-1])
        b = u[-1]
        xx = z / (1.0 + b * (kk - 1.0))
        yy = kk * xx
        lx = np.asarray(
            thermo.ln_phi(xx / np.sum(xx), temperature_k, pressure_pa, mixture, largest=False)
        )
        ly = np.asarray(
            thermo.ln_phi(yy / np.sum(yy), temperature_k, pressure_pa, mixture, largest=True)
        )
        return np.concatenate([u[:-1] - (lx - ly), [np.sum(yy - xx)]])

    beta = rachford_rice(z, k)
    u0 = np.concatenate([np.log(k), [beta]])
    before = float(np.max(np.abs(residual(u0))))
    sol = root(residual, u0, method="hybr", tol=1e-14)
    # The polish is accepted only if it helps and stays in the box. A
    # Newton step that leaves [0,1] has left the model, and silently
    # keeping it is how an oracle starts reporting the mirror solution.
    if sol.success and 0.0 <= sol.x[-1] <= 1.0:
        after = float(np.max(np.abs(residual(sol.x))))
        if after <= before:
            return np.exp(sol.x[:-1]), float(sol.x[-1]), after
    return k, float(beta), before


def flash(
    temperature_k: float,
    pressure_pa: float,
    mixture: PengRobinsonMixture,
    z=None,
) -> FlashResult:
    """Both stability tests first, then a two-phase solve only if warranted.

    This order is the whole point. A flash that goes straight to
    successive substitution from a Wilson start collapses onto the
    trivial solution wherever the feed is comfortably single-phase and
    then reports a converged two-phase answer with ``K_i = 1``. The
    stability tests are what make "single phase" a *result* rather than
    a failure to converge.
    """
    z = np.asarray(mixture.composition if z is None else z, dtype=float)
    vap = tangent_plane(z, temperature_k, pressure_pa, mixture, trial_is_vapor=True)
    liq = tangent_plane(z, temperature_k, pressure_pa, mixture, trial_is_vapor=False)

    if vap.stable or liq.stable:
        # Which single phase? Not a heuristic on ``Z``, and not the
        # tie-break between two stability numbers that the first draft
        # used. The feed is whichever phase the *incipient* one is not:
        # a non-trivial vapor-like trial phase can only condense out of
        # a liquid.
        #
        # This matters at the ends of the path, where the feed's cubic
        # has a single real root and one of the two tests therefore has
        # nothing to say -- it finds only the trivial stationary point
        # and reports ``sum Y = 1``, stable. Reading that as a second
        # stability verdict is what made the first draft call 230 K a
        # vapor. It is not a verdict; it is the absence of one.
        has_vap, has_liq = vap.composition is not None, liq.composition is not None
        if has_vap != has_liq:
            as_liquid = has_vap
        else:
            as_liquid = vap.stable and (not liq.stable or vap.sum_y <= liq.sum_y)
        trial = vap if as_liquid else liq
        amount = trial.sum_y
        w = trial.composition
        trial_vec = z.copy() if w is None else w * amount
        if as_liquid:
            regime, beta, x, y = "liquid", 0.0, z.copy(), trial_vec
        else:
            regime, beta, x, y = "vapor", 1.0, trial_vec, z.copy()
        k = (y / np.sum(y)) / (x / np.sum(x))
        return FlashResult(
            regime=regime,
            beta=beta,
            x=x,
            y=y,
            k=k,
            sum_x=float(np.sum(x)),
            sum_y=float(np.sum(y)),
            vapor_trial=vap,
            liquid_trial=liq,
            converged=True,
            residual=0.0,
            trivial=bool(w is None),
            note=(
                "single phase; no non-trivial incipient phase"
                if w is None
                else "single phase; incipient phase from the tangent-plane test"
            ),
        )

    # Both labels unstable: two phases. Seed from the incipient phase
    # the vapor test found, which near a switch point is a far better
    # start than Wilson and is free -- the test has already done it.
    if vap.composition is not None:
        k0 = vap.composition / z
        if np.max(np.abs(np.log(k0))) < 1e-3:
            k0 = thermo.wilson_k(temperature_k, pressure_pa, mixture)
    else:  # pragma: no cover - unstable with no trial phase cannot happen
        k0 = thermo.wilson_k(temperature_k, pressure_pa, mixture)

    k, beta, res = _two_phase(z, temperature_k, pressure_pa, mixture, k0)
    x = z / (1.0 + beta * (k - 1.0))
    y = k * x
    trivial = thermo.is_trivial(k)
    return FlashResult(
        regime="two_phase",
        beta=float(beta),
        x=x,
        y=y,
        k=k,
        sum_x=float(np.sum(x)),
        sum_y=float(np.sum(y)),
        vapor_trial=vap,
        liquid_trial=liq,
        converged=bool(res < 1e-9 and not trivial),
        residual=res,
        trivial=trivial,
        note="trivial solution" if trivial else "",
    )


# --------------------------------------------------------------------
# the switch points
# --------------------------------------------------------------------


def switch_temperature(
    lo_k: float,
    hi_k: float,
    pressure_pa: float,
    mixture: PengRobinsonMixture,
    z=None,
    *,
    trial_is_vapor: bool,
    xtol: float = 1e-10,
) -> float:
    """Bisect one stability boundary between ``lo_k`` and ``hi_k``.

    The function bisected is ``sum Y - 1`` from the tangent-plane test,
    which is continuous across the boundary and changes sign there.
    ``beta`` is not -- it is pinned at 0 or 1 on one side -- so
    bisecting on ``beta`` would find nothing, which is why the switch
    points are located from the stability test rather than from the
    flash.
    """
    z = np.asarray(mixture.composition if z is None else z, dtype=float)

    def f(t):
        return (
            tangent_plane(z, t, pressure_pa, mixture, trial_is_vapor=trial_is_vapor).sum_y - 1.0
        )

    f_lo, f_hi = f(lo_k), f(hi_k)
    if f_lo * f_hi > 0.0:
        raise ValueError(
            f"bracket [{lo_k}, {hi_k}] does not straddle the "
            f"{'bubble' if trial_is_vapor else 'dew'} boundary "
            f"(sum Y - 1 = {f_lo:.3e} and {f_hi:.3e})"
        )
    return float(brentq(f, lo_k, hi_k, xtol=xtol, rtol=8.9e-16, maxiter=200))


def bubble_and_dew(case, xtol: float = 1e-10) -> Dict[str, float]:
    """The two switch temperatures on the case's own path, located.

    Brackets come from the path itself by finding the regime changes, so
    this cannot return a number for a path that does not cross a
    boundary -- it raises instead.
    """
    z = case.z
    temps = np.asarray(case.temperatures_k, dtype=float)
    regimes = [flash(float(t), case.pressure_pa, case.mixture, z).regime for t in temps]
    out: Dict[str, float] = {}
    for i in range(len(temps) - 1):
        a, b = regimes[i], regimes[i + 1]
        if a == "liquid" and b == "two_phase":
            out["bubble_k"] = switch_temperature(
                float(temps[i]),
                float(temps[i + 1]),
                case.pressure_pa,
                case.mixture,
                z,
                trial_is_vapor=True,
                xtol=xtol,
            )
        elif a == "two_phase" and b == "vapor":
            out["dew_k"] = switch_temperature(
                float(temps[i]),
                float(temps[i + 1]),
                case.pressure_pa,
                case.mixture,
                z,
                trial_is_vapor=False,
                xtol=xtol,
            )
    return out
