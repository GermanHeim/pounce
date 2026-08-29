"""Everything checkable without a solver.

    python -m mpcc.selftest

Nine checks. Six of them exist because the corresponding mistake would
be invisible in a result file: a wrong derivative, a wrong expected
optimum, a classifier that cannot tell the classes apart, a rescaling
that is not actually equivalent, a lowering whose feasible set is not
the MPCC's, and a manifest that has drifted from the code. A benchmark
that can be wrong in any of those ways cannot attribute a failure to
anything.

Check 5 is the one worth reading. The classifier's whole value is
discrimination -- it has to return **M and refuse S** on `ralph1` and
`scholtes4`, and **C and refuse M** at `ctrap`'s origin -- so it is
checked against those three answers by name rather than against "it
returned something". A classifier that always returned "W" would pass a
residual check and fail this one.
"""

from __future__ import annotations

from typing import List

import numpy as np

from . import cases as C
from . import manifest as M
from .lowering import fd_check, lower
from .oracle import enumerate_branches
from .spec import CLASSES, SCALINGS
from .stationarity import classify
from .validate import CLASS_VALIDATORS, validate

_FD_TOL = 1e-7


def _check_classes() -> List[str]:
    present = {C.make(n).klass for n in C.REGISTRY}
    missing = [k for k in CLASSES if k not in present]
    return [f"benchmark class {k!r} has no case (gh#794 requires it)" for k in missing]


def _check_smoke_covers_classes() -> List[str]:
    present = {C.make(n).klass for n in C.SMOKE}
    return [
        f"smoke subset misses benchmark class {k!r}"
        for k in CLASSES
        if k not in present
    ]


def _check_derivatives() -> List[str]:
    fails = []
    rng = np.random.default_rng(794)
    for name in C.REGISTRY:
        case = C.make(name)
        pts = [np.asarray(v, float) for v in case.starts.values()]
        pts += [rng.normal(size=case.n) for _ in range(3)]
        if case.expected.x is not None:
            pts.append(np.asarray(case.expected.x, float))
        for lw, tau in (("prod_ineq", None), ("prod_eq", None), ("scholtes", 1e-3)):
            nlp = lower(case, lw, tau)
            for x in pts:
                err = fd_check(nlp, x)
                for k, v in err.items():
                    if v > _FD_TOL:
                        fails.append(f"{name}/{lw}: {k} finite-difference error {v:.2e}")
    return fails


def _check_oracle() -> List[str]:
    fails = []
    for name in C.REGISTRY:
        case = C.make(name)
        orc = enumerate_branches(case)
        exp = case.expected
        if orc["feasible"] != exp.feasible:
            fails.append(
                f"{name}: oracle feasible={orc['feasible']} but expected "
                f"feasible={exp.feasible}"
            )
            continue
        if not exp.feasible:
            continue
        if exp.obj is not None and abs(orc["obj"] - exp.obj) > 1e-6:
            fails.append(
                f"{name}: oracle f*={orc['obj']!r} disagrees with the "
                f"hand-derived f*={exp.obj!r}"
            )
        if exp.x is not None:
            got = np.asarray(orc["x"], float)
            if abs(case.objective.value(got) - exp.obj) > 1e-6:
                fails.append(f"{name}: oracle x has the wrong objective")
    return fails


def _check_expected_points() -> List[str]:
    fails = []
    for name in C.REGISTRY:
        case = C.make(name)
        exp = case.expected
        if exp.x is None:
            continue
        x = np.asarray(exp.x, float)
        s = case.source_feasibility(x)
        worst = max(s["row_viol"], s["bound_viol"], s["sign_viol"], s["compl_max"])
        if worst > 1e-12:
            fails.append(f"{name}: expected x is not source-feasible ({worst:.2e})")
        if exp.obj is not None and abs(case.objective.value(x) - exp.obj) > 1e-12:
            fails.append(f"{name}: f(expected x) != expected obj")
        got = classify(case, x)
        if got["klass"] != exp.stationarity:
            fails.append(
                f"{name}: classifier says {got['klass']}, manifest says "
                f"{exp.stationarity}"
            )
        if got["mpcc_licq"] != exp.mpcc_licq:
            fails.append(
                f"{name}: MPCC-LICQ {got['mpcc_licq']} vs expected {exp.mpcc_licq}"
            )
        if got["n_biactive"] != exp.n_biactive:
            fails.append(
                f"{name}: {got['n_biactive']} biactive pairs vs expected "
                f"{exp.n_biactive}"
            )
    return fails


def _check_classifier_discriminates() -> List[str]:
    """The classifier must return the *right* class, not merely a class.

    Each row is a point whose class was derived by hand in the case's
    docstring, together with the classes that must be **refused**. A
    classifier that collapsed to "W everywhere" satisfies every
    stationarity residual in the harness and fails here.
    """
    fails = []
    rows = [
        ("ralph1", np.zeros(2), "M", ("S",)),
        ("scholtes4", np.zeros(3), "M", ("S",)),
        ("ctrap", np.zeros(2), "C", ("S", "M")),
        ("ctrap", np.array([0.5, 0.0]), "S", ()),
        ("regular_strict", np.array([0.0, 2.0]), "S", ()),
        ("biactive_positive", np.zeros(2), "S", ()),
        ("qpec_small", np.array([1.0, 1.0, 0.0]), "S", ()),
    ]
    for name, x, want, refuse in rows:
        case = C.make(name)
        got = classify(case, x)
        if got["klass"] != want:
            fails.append(
                f"{name} at {x.tolist()}: classifier says {got['klass']}, "
                f"derivation says {want} (residuals {got['residuals']})"
            )
        tol = got["resid_tol"]
        for bad in refuse:
            if got["residuals"].get(bad, np.inf) <= tol:
                fails.append(
                    f"{name} at {x.tolist()}: classifier admits {bad}-stationarity "
                    f"with residual {got['residuals'][bad]:.2e} <= {tol:.2e}, but the "
                    "derivation shows no such multiplier exists"
                )
    return fails


def _check_rescaling() -> List[str]:
    """``rescale`` must produce an algebraically equivalent MPCC.

    Checked pointwise: the objective, every row value, every pair value
    and the whole feasibility triple have to agree at corresponding
    points. A scaling leg built on a rescaling that is not an
    equivalence measures the rescaling, not the solver.
    """
    fails = []
    rng = np.random.default_rng(4794)
    for name in C.REGISTRY:
        case = C.make(name)
        for sname, fn in SCALINGS.items():
            d = fn(case.n)
            sc = case.rescale(d)
            for _ in range(4):
                x = rng.normal(size=case.n) * 2.0
                xt = x / d
                if abs(case.objective.value(x) - sc.objective.value(xt)) > 1e-9 * max(
                    1.0, abs(case.objective.value(x))
                ):
                    fails.append(f"{name}/{sname}: objective not preserved")
                a = case.source_feasibility(x)
                b = sc.source_feasibility(xt)
                # Row values, pair signs and the complementarity products
                # are invariants of the rescaling and must match exactly.
                for k in ("row_viol", "sign_viol", "compl_max", "compl_min", "compl_sum"):
                    if abs(a[k] - b[k]) > 1e-8 * max(1.0, abs(a[k])):
                        fails.append(f"{name}/{sname}: {k} not preserved")
                # A bound violation is scale-dependent by construction
                # (`lb - x` becomes `(lb - x)/d`), so only the verdict
                # carries across.
                if (a["bound_viol"] > 1e-12) != (b["bound_viol"] > 1e-12):
                    fails.append(f"{name}/{sname}: bound feasibility verdict flipped")
            if sc.expected.obj != case.expected.obj:
                fails.append(f"{name}/{sname}: expected objective changed under rescaling")
    return fails


def _check_lowering_feasible_sets() -> List[str]:
    """``prod_ineq`` and ``prod_eq`` must admit exactly the MPCC's points.

    Sampled, not proved: points are drawn on and off each branch and the
    two feasibility verdicts compared. The row-order contract in
    `lowering` is checked at the same time, because a mis-ordered row
    block would put a bound on the wrong function and show up here.
    """
    fails = []
    rng = np.random.default_rng(79400)
    for name in C.REGISTRY:
        case = C.make(name)
        for lw in ("prod_ineq", "prod_eq"):
            nlp = lower(case, lw)
            if nlp.m != len(case.rows) + 3 * case.q:
                fails.append(f"{name}/{lw}: {nlp.m} rows, expected {len(case.rows) + 3 * case.q}")
            for _ in range(200):
                x = rng.normal(size=case.n) * 1.5
                if rng.random() < 0.6 and case.q:  # push onto a branch
                    i = int(rng.integers(case.q))
                    p = case.pairs[i]
                    f = p.G if rng.random() < 0.5 else p.H
                    if np.any(f.a != 0):
                        j = int(np.argmax(np.abs(f.a)))
                        x[j] -= f.value(x) / f.a[j]
                c = nlp.constraints(x)
                nlp_ok = bool(
                    np.all(c >= nlp.cl - 1e-9) and np.all(c <= nlp.cu + 1e-9)
                ) and bool(np.all(x >= case.lb - 1e-9) and np.all(x <= case.ub + 1e-9))
                s = case.source_feasibility(x)
                src_ok = max(s["row_viol"], s["bound_viol"], s["sign_viol"], s["compl_max"]) <= 1e-9
                if nlp_ok != src_ok:
                    fails.append(
                        f"{name}/{lw}: feasibility disagrees at x={x.tolist()} "
                        f"(nlp {nlp_ok}, source {src_ok})"
                    )
                    break
    return fails


def _check_manifest() -> List[str]:
    import os

    if not os.path.exists(M.MANIFEST_PATH):
        return ["manifest.json missing; run `python -m mpcc.run --write-manifest`"]
    committed = M.load()
    fresh = M.build(with_oracle=False)
    fails = []
    for key in (
        "model_data_revision",
        "smoke_subset",
        "tau_schedule",
        "restart_ladder",
        "classes_required",
        "routes",
        "controls",
        "base_options",
    ):
        if committed.get(key) != fresh.get(key):
            fails.append(f"manifest.json stale on {key!r}")
    cm = {c["name"]: c for c in committed["cases"]}
    fm = {c["name"]: c for c in fresh["cases"]}
    if set(cm) != set(fm):
        fails.append("manifest.json case list drifted from cases.py")
    else:
        for n in cm:
            if cm[n]["expected"] != fm[n]["expected"]:
                fails.append(f"manifest.json expected block stale for {n!r}")
    if fails:
        fails.append("regenerate with `python -m mpcc.run --write-manifest`")
    return fails


def _check_class_validators() -> List[str]:
    """gh#794: each benchmark class has a source-level validation function.

    Checked two ways -- every class in `CLASSES` has an entry, and the
    entry actually runs at each case's expected point and passes there.
    A validator that no expected point satisfies is describing a
    different class than the one it is registered under.
    """
    fails = [
        f"benchmark class {k!r} has no source-level validation function"
        for k in CLASSES
        if k not in CLASS_VALIDATORS
    ]
    for name in C.REGISTRY:
        case = C.make(name)
        x = case.expected.x
        if x is None:
            continue
        out = validate(case, np.asarray(x, float))
        for k, v in out.items():
            if k.endswith("_ok") and v is False:
                fails.append(
                    f"{name}: class validator for {case.klass!r} fails at the "
                    f"case's own expected solution ({k})"
                )
    return fails


def _check_validators_are_scale_invariant() -> List[str]:
    """A validator's verdict must not depend on the units it is read in.

    Both scaling legs describe the same MPCC, so a `_ok` key that flips
    between them is measuring the coordinates rather than the model.
    This is not hypothetical: the selector's one-hot test compared
    ``x`` against 1 and failed on every `skew` cell because the solution
    ``(0, 1)`` is ``(0, 1e-3)`` there -- 72 spurious failures in a full
    run, all of them looking like route behaviour. The fix is to read
    the pair values, which `rescale` leaves alone; this check is what
    keeps the next validator honest.
    """
    fails = []
    rng = np.random.default_rng(7942)
    for name in C.REGISTRY:
        case = C.make(name)
        pts = [np.asarray(v, float) for v in case.starts.values()]
        if case.expected.x is not None:
            pts.append(np.asarray(case.expected.x, float))
        pts += [rng.normal(size=case.n) for _ in range(4)]
        for sname, fn in SCALINGS.items():
            if sname == "unit":
                continue
            d = fn(case.n)
            sc = case.rescale(d)
            for x in pts:
                a = dict(validate(case, x))
                for v in case.validators:
                    a.update(v(case, x))
                b = dict(validate(sc, x / d))
                for v in sc.validators:
                    b.update(v(sc, x / d))
                for k in a:
                    if not k.endswith("_ok"):
                        continue
                    if a[k] != b.get(k):
                        fails.append(
                            f"{name}/{sname}: validator key {k!r} is "
                            f"{a[k]} unscaled and {b.get(k)} rescaled at the "
                            "same point"
                        )
    return sorted(set(fails))


CHECKS = (
    ("benchmark classes covered", _check_classes),
    ("each class has a source-level validator", _check_class_validators),
    ("validators are scale invariant", _check_validators_are_scale_invariant),
    ("smoke subset covers every class", _check_smoke_covers_classes),
    ("derivatives against finite differences", _check_derivatives),
    ("branch-enumeration oracle vs hand-derived optima", _check_oracle),
    ("expected points are feasible and correctly classified", _check_expected_points),
    ("classifier discriminates S / M / C", _check_classifier_discriminates),
    ("rescaling is an equivalence", _check_rescaling),
    ("lowering feasible sets equal the MPCC's", _check_lowering_feasible_sets),
    ("manifest is current", _check_manifest),
)


def main(argv=None) -> int:
    failed = 0
    for label, fn in CHECKS:
        fails = fn()
        status = "ok" if not fails else f"FAIL ({len(fails)})"
        print(f"{label:52s} {status}")
        for f in fails:
            print(f"    - {f}")
        failed += len(fails)
    print()
    print("selftest: PASSED" if not failed else f"selftest: {failed} FAILURES")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
