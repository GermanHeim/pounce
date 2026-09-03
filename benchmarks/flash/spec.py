"""The Gate 1 flash as an MPCC, in the source model's own units.

Nothing in this module imports pounce's solver. It defines the flash,
its complementarity pairs, and the source-level residuals a record
reports -- deliberately separate from anything a solve returns, for the
reason `mpcc/runner.py` states and this harness inherits: an MPCC
lowering's NLP residuals are residuals of a *different problem*, and a
converged NLP is not a solved flash.

The model
---------

For ``nc`` components at fixed ``(T, P)`` with feed ``z``, the unknowns
are the vapor fraction ``beta`` and the two phases' mole numbers per
unit feed, ``x`` (liquid) and ``y`` (vapor)::

    n = 2 nc + 1        variables   [beta, x_1..x_nc, y_1..y_nc]

    z_i = (1 - beta) x_i + beta y_i                           (balance)
    ln x_i + ln phi_i^L(x/Sx)
        = ln y_i + ln phi_i^V(y/Sy)                       (isofugacity)

    0 <= beta      _|_  1 - Sy >= 0                            (pair V)
    0 <= 1 - beta  _|_  1 - Sx >= 0                            (pair L)

with ``Sx = sum_i x_i``, ``Sy = sum_i y_i``.

Why *these* pairs, and not ``L _|_ V``
--------------------------------------

gh#776 states the guardrail directly: the pair is "nonnegative phase
amount complementary to nonnegative stability/equilibrium slack", and
**not** liquid flow complementary to vapor flow, because the two phases
coexist on a two-phase tray and ``L _|_ V`` would encode the wrong
physics. What is written above is that guardrail with the slack
identified:

* ``beta`` is the vapor amount; ``1 - Sy`` is the vapor's stability
  slack. Where no vapor is present the vapor variables are not junk --
  they are Michelsen's *trial phase*. Setting ``beta = 0`` makes the
  balance give ``x = z``, hence ``Sx = 1``, and the isofugacity rows
  collapse to ``ln y_i + ln phi_i(y/Sy) = ln z_i + ln phi_i(z)``, which
  is exactly the stationarity condition of the tangent-plane distance.
  ``Sy <= 1`` is then exactly ``TPD >= 0``: the liquid is stable against
  the vapor-like trial phase. The complementarity is not an encoding
  trick, it *is* the stability test.
* ``1 - beta`` is the liquid amount and ``1 - Sx`` its slack, by the
  mirror argument at ``beta = 1``.
* In the two-phase regime both amounts are positive, so both slacks
  vanish, ``Sx = Sy = 1``, and the isofugacity rows are the ordinary
  ones.

The branch labels are carried on the `Pair` objects in the model's own
words, so a flipped sign convention is a review finding rather than a
silent one.

What is biactive, and why that is the point
-------------------------------------------

At the bubble point ``beta = 0`` **and** ``Sy = 1`` together: pair V is
biactive. At the dew point pair L is. The two regime switches on the
temperature path are therefore exactly the two biactive points of this
MPCC, which is what makes the fixture a phase-change test rather than a
decorated NLP -- the degeneracy is not incidental to the physics, it is
the physics.

Where the normalization goes
----------------------------

``phi_i`` is a function of a *composition*, so it is evaluated at
``x/Sx`` and ``y/Sy``. The logarithm outside it is **not** normalized:
the row is ``ln x_i + ln phi_i(x/Sx)``, not ``ln(x_i/Sx) + ln
phi_i(x/Sx)``. That asymmetry is not a typo, it is Michelsen's tangent
plane. At ``beta = 0`` the balance gives ``x = z`` and ``Sx = 1``, and
the row collapses to

    ln y_i + ln phi_i^V(y/Sy) = ln z_i + ln phi_i^L(z),

which is exactly the stationarity condition for the trial phase's mole
*numbers* ``Y``. Normalizing the log term as well adds ``ln(Sy/Sx)`` to
every row.

This is worth stating at length because the first implementation of this
model got it wrong in exactly that way, and the shape of the error is
the shape this whole harness exists to catch: ``ln(Sy/Sx)`` **vanishes
identically in the two-phase region**, where ``Sx = Sy = 1``, so the
model solved, converged, and agreed with the oracle at every two-phase
temperature. It was wrong only in the single-phase regimes -- the ones
the fixture exists to reach -- where it reported
``Infeasible_Problem_Detected`` at 250 K and where evaluating the
oracle's own answer in the model's rows left a residual of exactly
``ln(0.6657) = -0.4069``. A corpus that stopped at the two-phase region
would have shipped it.

Scale
-----

``nc = 2``: five variables, four rows, two pairs. gh#776 gates the tray
and column work behind this fixture and asks for a *fast* regression, so
the size is the smallest one that still has both regime switches in it.
The corresponding limit is stated in the README and repeated in the
report: nothing measured here is evidence about a model with more than
one flash in it.
"""

from __future__ import annotations

import dataclasses
from typing import Dict, List, Tuple

import numpy as np

from pounce.examples.phase_envelope import PengRobinsonMixture

from . import thermo

#: Solver tolerance every route in this harness is pinned to, and the
#: corner tolerance derived from it. Both carry the meaning gh#794's
#: report established: `G*H` is quadratically flat at the corner, so a
#: solve converged to `tol` pins each side only to `sqrt(tol)`, and a
#: membership threshold at solver tolerance misreads a converged MPCC
#: point. `sqrt(tol)` is the "complementarity-accuracy floor" the Gate 0
#: report names, and it is numerical resolution, not phase physics.
SOLVE_TOL = 1e-8
CORNER_TOL = SOLVE_TOL**0.5

#: Floor on the membership tolerance, as in `mpcc/spec.py`.
ACTIVE_TOL = 1e-6

#: Lower bound on a phase mole number, and the reason it is not zero:
#: the isofugacity rows carry ``ln x_i`` and ``ln y_i``. See
#: `FlashCase.lb`.
MOLE_FLOOR = 1e-12

#: Regime labels. `two_phase` is the only one with both phases present;
#: `bubble` and `dew` are the biactive switch points, which are named
#: rather than folded into a neighbour because the whole fixture is
#: about them.
REGIMES = ("liquid", "bubble", "two_phase", "dew", "vapor", "undetermined")


#: The pinned mixture. Constants are the ethane and n-butane rows of
#: `phase_envelope.NATURAL_GAS`, which carries them for notebook 34;
#: taking them from the existing in-repo table rather than retyping a
#: handbook keeps one source of truth for the numbers. `k_ij = 0` is the
#: classical one-fluid choice that table already documents, and it is a
#: modelling choice, not a measurement: nothing here is a claim about
#: the real ethane/n-butane system beyond what Peng--Robinson with zero
#: binary interaction gives.
ETHANE_N_BUTANE = PengRobinsonMixture(
    names=("ethane", "n-butane"),
    critical_temperature=np.array([305.3, 425.1]),  # [K]
    critical_pressure=np.array([48.72, 37.96]) * 1e5,  # [Pa]
    acentric_factor=np.array([0.100, 0.200]),  # [-]
    composition=np.array([0.50, 0.50]),  # [-] feed
    binary_interaction=np.zeros((2, 2)),  # [-]
    source=(
        "Ethane and n-butane rows of phase_envelope.NATURAL_GAS "
        "(notebook 34 natural-gas demonstration); k_ij = 0."
    ),
)


@dataclasses.dataclass(frozen=True)
class Pair:
    """One complementarity pair ``0 <= G _|_ H >= 0``, with its physics."""

    name: str
    #: Which branch means what, in the model's own words. Read by no
    #: code; read by every reviewer, which is the point.
    branch_G_zero: str
    branch_H_zero: str
    units: str = "mol per mol feed"


PAIRS: Tuple[Pair, ...] = (
    Pair(
        name="vapor",
        branch_G_zero="beta = 0: no vapor present; y is Michelsen's trial phase",
        branch_H_zero="Sy = 1: the vapor is a real phase in equilibrium",
    ),
    Pair(
        name="liquid",
        branch_G_zero="beta = 1: no liquid present; x is Michelsen's trial phase",
        branch_H_zero="Sx = 1: the liquid is a real phase in equilibrium",
    ),
)


@dataclasses.dataclass(frozen=True)
class FlashCase:
    """One flash: a mixture, a feed, a pressure, and the temperature path."""

    name: str
    mixture: PengRobinsonMixture
    pressure_pa: float
    #: Temperature path, low to high. The harness walks it in both
    #: directions; `path.py` owns that, not this.
    temperatures_k: np.ndarray
    provenance: str = ""

    @property
    def nc(self) -> int:
        return self.mixture.n_components

    @property
    def n(self) -> int:
        return 2 * self.nc + 1

    @property
    def z(self) -> np.ndarray:
        return np.asarray(self.mixture.composition, dtype=float)

    # -- unpacking -------------------------------------------------

    def unpack(self, v) -> Tuple[float, np.ndarray, np.ndarray]:
        """``[beta, x, y]`` -> ``(beta, x, y)``."""
        v = np.asarray(v, dtype=float)
        nc = self.nc
        return float(v[0]), v[1 : 1 + nc], v[1 + nc : 1 + 2 * nc]

    def pack(self, beta: float, x, y) -> np.ndarray:
        return np.concatenate([[float(beta)], np.asarray(x, float), np.asarray(y, float)])

    @property
    def variable_names(self) -> List[str]:
        return (
            ["beta"]
            + [f"x_{c}" for c in self.mixture.names]
            + [f"y_{c}" for c in self.mixture.names]
        )

    # -- bounds ----------------------------------------------------

    @property
    def lb(self) -> np.ndarray:
        """``beta in [0,1]``; phase mole numbers nonnegative.

        The lower bound on ``x`` and ``y`` is a small positive number,
        not zero, and that is a modelling decision worth stating: the
        isofugacity rows carry ``ln x_i`` and ``ln y_i``, which are not
        defined at zero, and a component's amount in a *present* phase
        is bounded away from zero at any finite ``K_i``. It bounds what
        the fixture is evidence about -- it says nothing about a
        vanishing *component*, only about a vanishing *phase*, which is
        the transition gh#776 asks for.
        """
        return np.concatenate([[0.0], np.full(2 * self.nc, MOLE_FLOOR)])

    @property
    def ub(self) -> np.ndarray:
        return np.concatenate([[1.0], np.full(2 * self.nc, 1.0 / MOLE_FLOOR)])

    # -- the pairs, evaluated --------------------------------------

    def pair_values(self, v) -> Tuple[np.ndarray, np.ndarray]:
        """``(G, H)`` for the two pairs, in path order ``(vapor, liquid)``."""
        beta, x, y = self.unpack(v)
        g = np.array([beta, 1.0 - beta])
        h = np.array([1.0 - float(np.sum(y)), 1.0 - float(np.sum(x))])
        return g, h

    def pair_activity(self, v) -> List[Tuple[bool, bool]]:
        """Which side of each pair counts as zero, on the `sqrt(tol)` rule.

        The threshold is ``max(ACTIVE_TOL, CORNER_TOL)`` rather than
        ``ACTIVE_TOL``, for the reason `mpcc/spec.pair_activity`
        measured: a converged solve is entitled to leave the pair
        ``sqrt(tol)`` from the corner, and a fixed ``1e-6`` reads such a
        point as lying on neither branch. Every quantity here is a mole
        fraction or a phase fraction, so all of them live at O(1) and
        the relative-scale term that function needs does not arise.
        """
        g, h = self.pair_values(v)
        thresh = max(ACTIVE_TOL, CORNER_TOL)
        return [
            (bool(abs(gi) <= thresh), bool(abs(hi) <= thresh)) for gi, hi in zip(g, h)
        ]

    def regime(self, v) -> str:
        """The phase regime at ``v``, from the pair activity alone.

        Deliberately *not* from ``beta`` alone: ``beta = 0`` with the
        vapor slack also at zero is the bubble point, which is a
        different state from a subcooled liquid and is the state the
        fixture exists to reach.
        """
        (beta_zero, sy_one), (beta_one, sx_one) = self.pair_activity(v)
        if beta_zero and sy_one:
            return "bubble"
        if beta_one and sx_one:
            return "dew"
        if beta_zero:
            return "liquid"
        if beta_one:
            return "vapor"
        if sx_one and sy_one:
            return "two_phase"
        return "undetermined"

    # -- source residuals ------------------------------------------

    def balance_residual(self, v) -> np.ndarray:
        """``(1-beta) x_i + beta y_i - z_i``. [mol per mol feed]"""
        beta, x, y = self.unpack(v)
        return (1.0 - beta) * x + beta * y - self.z

    def isofugacity_residual(self, v, temperature_k: float) -> np.ndarray:
        """``ln x_i + ln phi_i^L(x/Sx) - ln y_i - ln phi_i^V(y/Sy)``. [-]

        Dimensionless by construction, so the tolerance it is judged
        against is a pure number and does not move with the pressure.
        """
        beta, x, y = self.unpack(v)
        sx, sy = float(np.sum(x)), float(np.sum(y))
        xn, yn = x / sx, y / sy
        lx = np.asarray(
            thermo.ln_phi(xn, temperature_k, self.pressure_pa, self.mixture, largest=False)
        )
        ly = np.asarray(
            thermo.ln_phi(yn, temperature_k, self.pressure_pa, self.mixture, largest=True)
        )
        return np.log(x) + lx - np.log(y) - ly

    def k_values(self, v) -> np.ndarray:
        """``K_i = (y_i/Sy) / (x_i/Sx)``, from the point itself. [-]"""
        _, x, y = self.unpack(v)
        return (y / np.sum(y)) / (x / np.sum(x))

    def source_feasibility(self, v, temperature_k: float) -> Dict[str, float]:
        """Original-space feasibility, with complementarity kept apart.

        Five quantities, never merged into one headline number, and the
        complementarity product is not among the ones that may be
        compared against ``tol``: it is a source quantity whose floor is
        ``sqrt(tol)`` (the Gate 0 report's accuracy floor), and judging
        it at ``tol`` would report every correct answer as a failure.
        """
        v = np.asarray(v, dtype=float)
        bal = float(np.max(np.abs(self.balance_residual(v))))
        iso = float(np.max(np.abs(self.isofugacity_residual(v, temperature_k))))
        bnd = float(np.max(np.maximum(np.maximum(self.lb - v, v - self.ub), 0.0)))
        g, h = self.pair_values(v)
        sign = float(np.max(np.maximum(np.maximum(-g, -h), 0.0)))
        prod = np.abs(g * h)
        return {
            "balance_viol": bal,
            "isofugacity_viol": iso,
            "bound_viol": bnd,
            "sign_viol": sign,
            "compl_max": float(prod.max()),
            "compl_sum": float(prod.sum()),
        }


#: The pinned Gate 1 case.
#:
#: 10 bar is chosen so that a single temperature path crosses all three
#: regimes with both switches interior to it and far from the mixture
#: critical point: liquid below ~267 K, two-phase, vapor above ~327 K.
#: The path is deliberately not symmetric about the two-phase region --
#: it carries more points near the switches, because that is where a
#: route either follows the branch or does not.
def _path() -> np.ndarray:
    coarse = np.arange(230.0, 361.0, 10.0)
    near_bubble = np.arange(262.0, 273.0, 1.0)
    near_dew = np.arange(322.0, 333.0, 1.0)
    return np.unique(np.concatenate([coarse, near_bubble, near_dew]))


GATE1_FLASH = FlashCase(
    name="ethane_n_butane_10bar",
    mixture=ETHANE_N_BUTANE,
    pressure_pa=10.0e5,
    temperatures_k=_path(),
    provenance=(
        "gh#776 Gate 1. Peng--Robinson, classical mixing, k_ij = 0; constants "
        "from phase_envelope.NATURAL_GAS. The bubble and dew temperatures are "
        "not asserted from a handbook -- they are located by the independent "
        "oracle in oracle.py and recorded in the manifest."
    ),
)
