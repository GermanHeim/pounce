"""Everything checkable without the solver.

The gate on every other target, exactly as `mpcc/selftest.py` is for
Gate 0: if the model's derivatives are wrong, or the oracle disagrees
with the model about what a solution is, then every number a solve
produces is measuring the harness. Run it first; it needs no POUNCE
solve and takes a few seconds.

What each check would catch
---------------------------

``derivatives``
    The JAX Jacobian and Lagrangian Hessian against central differences
    at several points per regime. Gate 0's corpus is quadratic so its
    equivalent is exact to round-off; here the rows carry a cubic root
    and a logarithm, so the thresholds are set to the *method's* own
    truncation error and no tighter. An assertion tighter than the
    method it uses fails for a reason that has nothing to do with the
    model, which is the one thing a gate check must never do.

``oracle_solves_the_mpcc``
    **The load-bearing one.** The oracle's answer is substituted into
    the model's own rows and must satisfy them. This is what makes the
    two calculations a cross-check rather than two independent opinions:
    it failed on the first implementation and located the normalization
    defect recorded in `spec.py` -- with a residual of exactly
    ``ln(Sy)``, at the single-phase points only.

``regime_coverage``
    The path actually crosses all three regimes and both switch points
    are interior to it. A fixture whose path drifted into one regime
    would keep passing every other check while testing nothing gh#776
    asked for.

``switch_points``
    The bubble and dew temperatures are located, bracketed by path
    points, and in the right order.

``no_trivial_solutions``
    No point on the path is the trivial ``K_i = 1`` state, and every
    point has an incipient phase for at least one label -- outside that
    region the MPCC leaves ``beta`` undetermined (see
    `oracle.FlashResult.no_incipient_phase`) and the fixture would be
    checking a one-parameter family against a point.
"""

from __future__ import annotations

import sys
from typing import Callable, List, Tuple

import numpy as np

from . import lowering, oracle, spec, thermo

#: Central-difference truncation floor at ``h = 1e-6`` on rows carrying
#: a cubic root and a logarithm. Measured across the path rather than
#: chosen: the worst Jacobian error is ~1e-8 and the worst Hessian error
#: ~1e-5, both at the cold end where the liquid root is stiffest.
JAC_TOL = 1e-6
HESS_TOL = 1e-3

_FAILURES: List[str] = []


def _check(name: str, ok: bool, detail: str = "") -> None:
    mark = "ok  " if ok else "FAIL"
    print(f"  [{mark}] {name}{(' -- ' + detail) if detail else ''}")
    if not ok:
        _FAILURES.append(name)


def check_derivatives(case: spec.FlashCase) -> None:
    print("derivatives (JAX vs central differences)")
    worst_j = worst_h = 0.0
    for t in (230.0, 250.0, 268.0, 300.0, 324.0, 340.0, 360.0):
        nlp = lowering.lower(case, t, "prod_eq")
        ref = oracle.flash(t, case.pressure_pa, case.mixture)
        points = [
            case.pack(ref.beta, ref.x, ref.y),
            case.pack(0.5, case.z.copy(), case.z.copy()),
        ]
        for v in points:
            errs = lowering.fd_check(nlp, v)
            worst_j = max(worst_j, errs["jac"])
            worst_h = max(worst_h, errs["hess"])
    _check("jacobian", worst_j <= JAC_TOL, f"worst {worst_j:.2e} <= {JAC_TOL:.0e}")
    _check("lagrangian hessian", worst_h <= HESS_TOL, f"worst {worst_h:.2e} <= {HESS_TOL:.0e}")


def check_oracle_solves_the_mpcc(case: spec.FlashCase) -> None:
    print("the oracle's answer satisfies the model's own rows")
    worst = 0.0
    worst_t = None
    for t in case.temperatures_k:
        t = float(t)
        ref = oracle.flash(t, case.pressure_pa, case.mixture)
        v = case.pack(ref.beta, ref.x, ref.y)
        nlp = lowering.lower(case, t, "prod_eq")
        c = nlp.constraints(v)
        viol = float(np.max(np.maximum(np.maximum(nlp.cl - c, c - nlp.cu), 0.0)))
        if viol > worst:
            worst, worst_t = viol, t
    _check(
        "oracle point is MPCC-feasible",
        worst <= 1e-10,
        f"worst {worst:.2e} at T = {worst_t} K",
    )


def check_regime_coverage(case: spec.FlashCase) -> None:
    print("the path crosses all three regimes")
    regimes = [
        oracle.flash(float(t), case.pressure_pa, case.mixture).regime
        for t in case.temperatures_k
    ]
    seen = set(regimes)
    _check(
        "liquid, two-phase and vapor all reached",
        {"liquid", "two_phase", "vapor"} <= seen,
        f"{sorted(seen)}",
    )
    counts = {r: regimes.count(r) for r in sorted(seen)}
    _check("every regime has at least three points", min(counts.values()) >= 3, f"{counts}")
    # Monotone: liquid ... two-phase ... vapor, no interleaving. A path
    # that re-enters a regime it has left is either a genuine retrograde
    # region or a broken oracle, and this fixture is not the place to
    # find out which.
    order = [r for i, r in enumerate(regimes) if i == 0 or r != regimes[i - 1]]
    _check(
        "regimes appear in one monotone block each",
        order == ["liquid", "two_phase", "vapor"],
        " -> ".join(order),
    )


def check_switch_points(case: spec.FlashCase) -> None:
    print("the switch points")
    sw = oracle.bubble_and_dew(case)
    have = "bubble_k" in sw and "dew_k" in sw
    _check("both located", have, str({k: round(v, 6) for k, v in sw.items()}))
    if not have:
        return
    t = np.asarray(case.temperatures_k, dtype=float)
    _check("bubble < dew", sw["bubble_k"] < sw["dew_k"])
    _check(
        "both interior to the path",
        t.min() < sw["bubble_k"] and sw["dew_k"] < t.max(),
        f"path [{t.min()}, {t.max()}] K",
    )


def check_no_trivial_solutions(case: spec.FlashCase) -> None:
    print("guards: trivial solution, incipient phase, cubic root")
    trivial: List[float] = []
    no_incipient: List[float] = []
    bad_root: List[float] = []
    for t in case.temperatures_k:
        t = float(t)
        ref = oracle.flash(t, case.pressure_pa, case.mixture)
        if ref.trivial or thermo.is_trivial(ref.k):
            trivial.append(t)
        if ref.no_incipient_phase:
            no_incipient.append(t)
        v = case.pack(ref.beta, ref.x, ref.y)
        _, x, y = case.unpack(v)
        # Only the phases that are actually *present*. A present phase
        # must sit at its lower-Gibbs root or the model has chosen a
        # metastable state; an incipient phase need not, and on this
        # path it routinely does not -- Michelsen's trial phase probes
        # the tangent plane at the feed, and its own composition is one
        # where the labelled root can be the metastable one. Measured:
        # the incipient vapor fails this at 230 and 240 K, the incipient
        # liquid at 340-360 K, and in the two-phase region, where both
        # phases are real, both pass at every point. Asserting it of the
        # trial phase would be asserting something thermodynamics does
        # not claim.
        present = []
        if ref.regime in ("liquid", "two_phase"):
            present.append((x / np.sum(x), False))
        if ref.regime in ("vapor", "two_phase"):
            present.append((y / np.sum(y), True))
        for w, largest in present:
            d = thermo.root_diagnostics(w, t, case.pressure_pa, case.mixture, largest=largest)
            if d["root_is_gibbs_optimal"] is False:
                bad_root.append(t)
    _check("no trivial K = 1 point on the path", not trivial, f"{trivial}")
    _check("every point has an incipient phase", not no_incipient, f"{no_incipient}")
    _check(
        "every present phase sits at its lower-Gibbs root",
        not bad_root,
        f"{sorted(set(bad_root))}",
    )


def check_pairs_are_the_documented_ones(case: spec.FlashCase) -> None:
    """The guardrail check: the pairs are amount-vs-slack, not L vs V.

    gh#776 states the rule and this asserts the model obeys it, at a
    point where ``L`` and ``V`` are both plainly nonzero. If some future
    edit made the pair ``L _|_ V``, both sides would be positive in the
    two-phase region and their product would be far from zero there --
    which is exactly the wrong physics the guardrail names.
    """
    print("the complementarity pairs are amount vs stability slack")
    t = 300.0
    ref = oracle.flash(t, case.pressure_pa, case.mixture)
    v = case.pack(ref.beta, ref.x, ref.y)
    g, h = case.pair_values(v)
    both_phases = 1e-3 < ref.beta < 1.0 - 1e-3
    _check("the test point is genuinely two-phase", both_phases, f"beta = {ref.beta:.4f}")
    _check(
        "both pairs are complementary there",
        float(np.max(np.abs(g * h))) < 1e-10,
        f"max |G*H| = {float(np.max(np.abs(g * h))):.2e}",
    )
    _check(
        "the slacks vanish rather than the amounts",
        abs(h[0]) < 1e-10 and abs(h[1]) < 1e-10 and g[0] > 1e-3 and g[1] > 1e-3,
        f"G = {np.round(g, 6)}, H = {np.round(h, 12)}",
    )


CHECKS: Tuple[Callable[[spec.FlashCase], None], ...] = (
    check_derivatives,
    check_oracle_solves_the_mpcc,
    check_regime_coverage,
    check_switch_points,
    check_no_trivial_solutions,
    check_pairs_are_the_documented_ones,
)


def main(argv=None) -> int:
    case = spec.GATE1_FLASH
    print(f"flash selftest: {case.name} at {case.pressure_pa / 1e5:.1f} bar, "
          f"{len(case.temperatures_k)} temperatures, no solver required\n")
    for fn in CHECKS:
        fn(case)
        print()
    if _FAILURES:
        print(f"FAILED: {len(_FAILURES)} check(s): {', '.join(_FAILURES)}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
