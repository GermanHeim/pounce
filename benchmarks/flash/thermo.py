"""Peng--Robinson thermodynamics for the Gate 1 flash, and the guards.

Everything here is a thin layer over
`pounce.examples.phase_envelope.log_fugacity_coefficients`, which landed
with gh#777 and is validated there against the published Deiters--Bell
methane/propane envelope. **Reusing it rather than reimplementing it is
deliberate**: this harness and the oracle in `oracle.py` are supposed to
be independent calculations of the same flash, and the honest way to say
that is to name the one primitive they share and check everything above
it. A second copy of the cubic would make them look more independent
than they are while adding a place for the two to disagree for a reason
that has nothing to do with the MPCC.

The three guards
----------------

gh#776 asks the formulation to document "how metastable, trivial
``K_i = 1``, wrong cubic-root, and supercritical states are excluded".
Three of those are checkable at a point and are computed here, as
measurements rather than assumptions:

``root_is_gibbs_optimal``
    The model picks a cubic root by phase *label* -- ``largest=True``
    for the vapor, ``largest=False`` for the liquid -- which is the
    ordinary convention and is wrong exactly when the labelled root is
    not the lower-Gibbs one. `root_diagnostics` computes both roots and
    the reduced Gibbs difference between them, so a record can say the
    label agreed with the energy rather than assume it. Where the cubic
    has one real root the question does not arise and the field says so;
    a caller that reads ``False`` from a missing key would get the guard
    backwards, so it is ``None``.

``trivial_solution``
    ``K_i = 1`` for every ``i`` is a solution of the isofugacity
    equations at any composition, and it is the classic false answer a
    flash converges to. It is detected by ``max_i |ln K_i|`` falling
    under a threshold, never by "the solve succeeded".

``supercritical``
    Recorded, not excluded. At the pinned condition
    (`spec.ETHANE_N_BUTANE`, 10 bar) the mixture is nowhere near its
    critical point, but ethane is above *its own* ``Tc`` over the top
    third of the path, which is ordinary and is not a supercritical
    *mixture* state. The field records the pure-component reduced
    temperatures so a reader can see which is which instead of
    inferring it from the mixture label.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from pounce.examples.phase_envelope import (
    PengRobinsonMixture,
    compressibility,
    log_fugacity_coefficients,
)

#: Below this, ``max_i |ln K_i|`` is the trivial solution rather than a
#: narrow two-phase region. It is loose on purpose: a genuine flash at
#: the pinned condition runs ``|ln K|`` of order 1, and the trivial
#: attractor arrives at 1e-8 or below, so anything in between is a
#: measurement that should be looked at rather than classified.
TRIVIAL_LN_K = 1e-4

#: Two cubic roots closer than this are one root numerically, and the
#: root-selection question does not arise.
ONE_ROOT_GAP = 1e-9


def ln_phi(composition, temperature_k, pressure_pa, mixture, *, largest: bool):
    """``ln phi_i`` for a *normalized* composition. [-]"""
    return log_fugacity_coefficients(
        composition, temperature_k, pressure_pa, mixture, largest=largest
    )


def reduced_gibbs(composition, temperature_k, pressure_pa, mixture, *, largest: bool) -> float:
    """``sum_i w_i (ln w_i + ln phi_i)``, the reduced molar Gibbs energy. [-]

    Only differences of this at fixed composition are used, so the
    ``w ln w`` term cancels and could be dropped; it is kept because a
    quantity named "Gibbs" that is missing its entropy term is a trap
    for whoever reads it next.
    """
    w = np.asarray(composition, dtype=float)
    lnp = np.asarray(ln_phi(w, temperature_k, pressure_pa, mixture, largest=largest))
    with np.errstate(divide="ignore", invalid="ignore"):
        entropy = np.where(w > 0.0, w * np.log(np.where(w > 0.0, w, 1.0)), 0.0)
    return float(np.sum(entropy + w * lnp))


def root_diagnostics(
    composition, temperature_k, pressure_pa, mixture, *, largest: bool
) -> Dict[str, object]:
    """Did the label-selected cubic root agree with the lower-Gibbs one?

    Returns ``z_selected``, ``z_other``, the signed reduced-Gibbs
    difference ``selected - other``, and ``root_is_gibbs_optimal``
    (``None`` where the cubic has a single real root).
    """
    w = np.asarray(composition, dtype=float)
    z_hi = float(compressibility(w, temperature_k, pressure_pa, mixture, largest=True))
    z_lo = float(compressibility(w, temperature_k, pressure_pa, mixture, largest=False))
    z_sel, z_oth = (z_hi, z_lo) if largest else (z_lo, z_hi)
    if abs(z_hi - z_lo) <= ONE_ROOT_GAP:
        return {
            "z_selected": z_sel,
            "z_other": z_oth,
            "one_real_root": True,
            "gibbs_gap": 0.0,
            "root_is_gibbs_optimal": None,
        }
    g_sel = reduced_gibbs(w, temperature_k, pressure_pa, mixture, largest=largest)
    g_oth = reduced_gibbs(w, temperature_k, pressure_pa, mixture, largest=not largest)
    return {
        "z_selected": z_sel,
        "z_other": z_oth,
        "one_real_root": False,
        "gibbs_gap": float(g_sel - g_oth),
        "root_is_gibbs_optimal": bool(g_sel <= g_oth + 1e-12),
    }


def wilson_k(temperature_k, pressure_pa, mixture: PengRobinsonMixture) -> np.ndarray:
    """Wilson correlation ``K_i``. [-] The standard cold start."""
    return np.asarray(
        mixture.critical_pressure
        / pressure_pa
        * np.exp(
            5.373
            * (1.0 + mixture.acentric_factor)
            * (1.0 - mixture.critical_temperature / temperature_k)
        ),
        dtype=float,
    )


def is_trivial(k_values, tol: float = TRIVIAL_LN_K) -> bool:
    """``K_i = 1`` for every component: the classic false flash answer."""
    k = np.asarray(k_values, dtype=float)
    return bool(np.max(np.abs(np.log(k))) < tol)


def reduced_temperatures(temperature_k, mixture: PengRobinsonMixture) -> Dict[str, float]:
    """``T/Tc_i`` per pure component, for the supercritical record."""
    return {
        name: float(temperature_k / tc)
        for name, tc in zip(mixture.names, mixture.critical_temperature)
    }


def supercritical_components(temperature_k, mixture: PengRobinsonMixture) -> Optional[list]:
    """Which pure components are above their own ``Tc`` at this ``T``."""
    return [
        name
        for name, tc in zip(mixture.names, mixture.critical_temperature)
        if temperature_k > tc
    ]
